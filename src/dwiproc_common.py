#!/usr/bin/env python3
"""
dwiproc_common.py
=================

Helpers shared by the three pipeline scripts (``dwi_preproc.py``,
``dwi_tract.py`` and ``dwi_launcher.py``) of *compneuro-dwiproc-dipy*, the DIPY
re-implementation of the lab's MRtrix3/FSL/ANTs pipeline ``compneuro-dwiproc``.

Contents (in order):

    1. Paths and constants (FreeSurfer label groups, template folder)
    2. Logging and timing
    3. Participant / BIDS discovery
    4. JSON sidecar metadata (phase encoding, readout time, partial Fourier)
    5. Image and JSON writers
    6. Gradient tables: FSL <-> voxel convention and b-vector rotation
    7. Anatomy loader: FreeSurfer ``Recon`` folder or the original
       ``Preproc/BET`` + ``Preproc/ProbTissue`` layout
    8. Small image utilities (largest connected component, voxel sizes)

Nothing here generates or imputes data: every function either returns what it
finds or raises an explicit error that the launcher logs before moving on to
the next participant.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------------------
# 1. Paths and constants
# ---------------------------------------------------------------------------

# Repository root = parent of the ``src`` folder that contains this file.
REPO_DIR = Path(__file__).resolve().parents[1]
# Templates and parcellations (copied from compneuro-dwiproc/brain_templates).
TEMPLATES_DIR = REPO_DIR / "brain_templates"

# FreeSurfer aseg / aparc+aseg label groups used to build the tissue maps that
# MRtrix's ``5ttgen`` would have produced. The grouping follows the logic of
# MRtrix ``5ttgen freesurfer`` (cerebellar cortex counts as cortical GM; the
# deep nuclei, hippocampus, amygdala and ventral DC as sub-cortical GM).
FS_CORTICAL_GM = [3, 42, 8, 47]            # + every label 1000-2999 (aparc ctx-*)
FS_SUBCORTICAL_GM = [9, 10, 11, 12, 13, 17, 18, 19, 20, 26, 27, 28,
                     48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60]
FS_WM = [2, 41, 7, 46, 16, 77, 78, 79, 85, 192, 251, 252, 253, 254, 255]
FS_CSF = [4, 5, 14, 15, 24, 31, 43, 44, 63, 72]

# ---------------------------------------------------------------------------
# 2. Logging and timing
# ---------------------------------------------------------------------------


def get_logger(log_file: Path | None = None, name: str = "dwiproc") -> logging.Logger:
    """Return a logger that writes to stdout and (optionally) to ``log_file``.

    Calling it again with another ``log_file`` adds that file as an extra
    handler, so the launcher can keep one global log plus one log per
    participant (the per-participant handler is removed with
    :func:`remove_file_handler` when that participant is finished).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        already = any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
                      for h in logger.handlers)
        if not already:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    return logger


def remove_file_handler(logger: logging.Logger, log_file: Path) -> None:
    """Detach (and close) the file handler that writes to ``log_file``."""
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == Path(log_file).resolve():
            logger.removeHandler(h)
            h.close()


@contextmanager
def timed(logger: logging.Logger, label: str, timings: dict | None = None):
    """Context manager that logs the start/end of a step and its duration.

    ``timings`` (optional dict) receives ``{label: seconds}`` so the QC JSON
    can report how long every step took.
    """
    logger.info(f"**{label}...**")
    t0 = time.time()
    yield
    dt = time.time() - t0
    if timings is not None:
        timings[label] = round(dt, 1)
    logger.info(f"   done: {label} ({dt / 60:.1f} min)")


# ---------------------------------------------------------------------------
# 3. Participant / BIDS discovery
# ---------------------------------------------------------------------------


def normalise_label(label: str) -> str:
    """'6175' or 'sub-6175' -> 'sub-6175'."""
    label = label.strip()
    return label if label.startswith("sub-") else f"sub-{label}"


def list_participants(bids_dir: Path, labels: list[str] | None = None) -> list[str]:
    """Participants to process, in this order of preference:

    1. ``labels`` given on the command line;
    2. first column of ``<bids_dir>/participants.tsv`` (as the original
       launcher did);
    3. ``bids_dir`` itself, if it is a single-subject folder (e.g.
       ``~/Desktop/sub-6175``);
    4. every ``sub-*`` folder in ``bids_dir``.
    """
    bids_dir = Path(bids_dir)
    if labels:
        return [normalise_label(x) for x in labels]
    tsv = bids_dir / "participants.tsv"
    if tsv.is_file():
        rows = [ln.split("\t")[0].strip() for ln in tsv.read_text().splitlines()[1:]]
        return [normalise_label(r) for r in rows if r]
    if bids_dir.name.startswith("sub-") and (bids_dir / "dwi").is_dir():
        return [bids_dir.name]
    return sorted(p.name for p in bids_dir.glob("sub-*") if p.is_dir())


def subject_dir(bids_dir: Path, participant: str) -> Path:
    """Locate the participant's raw-data folder.

    Accepts both a normal BIDS root (``<bids_dir>/sub-X``) and a single
    subject folder passed directly as ``bids_dir`` (``.../sub-X``).
    """
    bids_dir = Path(bids_dir)
    if (bids_dir / participant).is_dir():
        return bids_dir / participant
    if bids_dir.name == participant:
        return bids_dir
    raise FileNotFoundError(f"No folder for {participant} in {bids_dir}")


def _strip_nii(p: Path) -> str:
    name = p.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return p.stem


def find_dwi_runs(bids_dir: Path, participant: str) -> list[dict]:
    """Return every DWI run of a participant with its companion files.

    Looks in ``sub-X/dwi`` and ``sub-X/ses-*/dwi``. A run is only returned if
    the ``.bval`` and ``.bvec`` files exist (the ``.json`` sidecar is optional
    but recommended). Each item is a dict with keys
    ``nii, bval, bvec, json, session, stem``.
    """
    sdir = subject_dir(bids_dir, participant)
    candidates = sorted(sdir.glob("dwi/*_dwi.nii*")) + sorted(sdir.glob("ses-*/dwi/*_dwi.nii*"))
    runs = []
    for nii in candidates:
        stem = _strip_nii(nii)
        bval = nii.with_name(stem + ".bval")
        bvec = nii.with_name(stem + ".bvec")
        js = nii.with_name(stem + ".json")
        if not (bval.is_file() and bvec.is_file()):
            # Never invent gradients: skip and let the caller report it.
            continue
        ses = nii.parent.parent.name if nii.parent.parent.name.startswith("ses-") else None
        runs.append(dict(nii=nii, bval=bval, bvec=bvec,
                         json=js if js.is_file() else None, session=ses, stem=stem))
    return runs


def _intended_for_matches(intended, dwi_rel: str) -> bool:
    """True if a BIDS IntendedFor entry points at ``dwi_rel`` (subject-relative path).

    Accepts the subject-relative form ('ses-01/dwi/x.nii.gz') and the BIDS URI
    form ('bids::sub-X/ses-01/dwi/x.nii.gz').
    """
    if isinstance(intended, str):
        intended = [intended]
    for item in intended or []:
        item = item.split("::", 1)[-1]
        if item == dwi_rel or item.endswith("/" + dwi_rel):
            return True
    return False


def find_fieldmaps(bids_dir: Path, participant: str, run: dict) -> tuple[list[dict], str]:
    """Spin-echo EPI fieldmaps for a DWI run (topup inputs).

    1. ``fmap/*_epi.nii.gz`` (same session, or subject level) whose JSON
       ``IntendedFor`` lists this DWI file -- the BIDS way, and how the CAINT
       bidsify script writes them;
    2. otherwise ``fmap/*acq-dwi*_epi`` / ``*acq-diff*_epi`` files of the same
       session (reported as a fallback in the log/QC).

    Returns (list of dicts with keys nii, json, meta; how they were found).
    Nothing is returned if no file qualifies.
    """
    sdir = subject_dir(bids_dir, participant)
    dwi_rel = str(Path(run["nii"]).relative_to(sdir))
    fmap_dirs = [sdir / run["session"] / "fmap"] if run["session"] else []
    fmap_dirs.append(sdir / "fmap")
    cands = []
    for d in fmap_dirs:
        for nii in sorted(d.glob("*_epi.nii*")):
            js = nii.with_name(_strip_nii(nii) + ".json")
            cands.append(dict(nii=nii, json=js if js.is_file() else None, meta=read_json(js)))
    hits = [c for c in cands if _intended_for_matches(c["meta"].get("IntendedFor"), dwi_rel)]
    if hits:
        return hits, "IntendedFor"
    hits = [c for c in cands if ("acq-dwi" in c["nii"].name or "acq-diff" in c["nii"].name)
            and (run["session"] is None or f"_{run['session']}_" in c["nii"].name)]
    return hits, ("filename (acq-dwi/acq-diff; no IntendedFor match)" if hits else "none")


def find_t1w(bids_dir: Path, participant: str, session: str | None) -> Path | None:
    """Raw T1w image of the participant (same session first), for Synb0-DisCo."""
    sdir = subject_dir(bids_dir, participant)
    dirs = ([sdir / session / "anat"] if session else []) + [sdir / "anat"] + \
        sorted(sdir.glob("ses-*/anat"))
    for d in dirs:
        hits = sorted(d.glob("*_T1w.nii*"))
        if hits:
            return hits[0]
    return None


def run_output_dir(output_dir: Path, participant: str, run: dict, n_runs_in_session: int) -> Path:
    """Output folder of one DWI run.

    ``<output>/sub-X`` (as in the original ``Preproc/Dwiprep/sub-X``), plus a
    ``ses-Y`` level when the data have sessions, plus a run-specific level only
    if one session has several DWI runs.
    """
    out = Path(output_dir) / participant
    if run["session"]:
        out = out / run["session"]
    if n_runs_in_session > 1:
        out = out / run["stem"]
    return out


# ---------------------------------------------------------------------------
# 4. JSON sidecar metadata
# ---------------------------------------------------------------------------


def read_json(path: Path | None) -> dict:
    if path is None or not Path(path).is_file():
        return {}
    return json.loads(Path(path).read_text())


def phase_encoding(meta: dict) -> dict:
    """Phase-encoding axis/sign and readout time from a BIDS sidecar.

    Mirrors the logic of the original ``dwi_preproc.sh``:
    ``TotalReadoutTime`` (else ``EstimatedTotalReadoutTime``) and
    ``PhaseEncodingDirection`` (else ``PhaseEncodingAxis``).
    Returns ``{'pe_dir': 'j-', 'axis': 1, 'sign': -1, 'readout_time': 0.0417}``
    with ``None`` for anything that is missing.
    """
    pe = meta.get("PhaseEncodingDirection") or meta.get("PhaseEncodingAxis")
    rt = meta.get("TotalReadoutTime", meta.get("EstimatedTotalReadoutTime"))
    axis = sign = None
    if pe:
        axis = {"i": 0, "j": 1, "k": 2}[pe[0]]
        sign = -1 if pe.endswith("-") else 1
    return dict(pe_dir=pe, axis=axis, sign=sign, readout_time=rt)


def slice_axis(meta: dict) -> int:
    """Voxel axis along which slices were acquired (for Gibbs removal).

    Uses ``SliceEncodingDirection`` when present; otherwise assumes the third
    voxel axis, which is the usual case for 2D EPI DWI saved by dcm2niix.
    """
    sed = meta.get("SliceEncodingDirection")
    if sed:
        return {"i": 0, "j": 1, "k": 2}[sed[0]]
    return 2


# ---------------------------------------------------------------------------
# 5. Writers
# ---------------------------------------------------------------------------


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray, dtype=np.float32) -> Path:
    """Save ``data`` as NIfTI-1 (gzip) with the given affine and dtype.

    Both qform and sform are set to the affine (code 1, scanner), so FSL,
    MRtrix, nilearn and DIPY all read the same geometry.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data).astype(dtype, copy=False), affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header.set_xyzt_units("mm", "sec")
    nib.save(img, str(path))
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def write_json(path: Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def software_versions() -> dict:
    import dipy
    import scipy
    return dict(python=sys.version.split()[0], dipy=dipy.__version__,
                nibabel=nib.__version__, numpy=np.__version__, scipy=scipy.__version__)


# ---------------------------------------------------------------------------
# 6. Gradient tables
# ---------------------------------------------------------------------------
#
# FSL-format b-vectors (what dcm2niix writes and what the original pipeline
# exported as rotated.bvec) are expressed in the *voxel* axes of the image,
# except that FSL flips the first axis when the image affine has a positive
# determinant (neurological storage). DIPY uses the b-vectors exactly as given
# and interprets them in voxel axes. To be correct for every orientation we
# convert FSL -> voxel convention on input and voxel -> FSL on output; for the
# usual dcm2niix output (negative determinant, e.g. LAS) the two are identical.


def fsl_flip(affine: np.ndarray) -> np.ndarray:
    """Diagonal matrix that converts FSL b-vectors to voxel-axis b-vectors."""
    flip = np.eye(3)
    if np.linalg.det(affine[:3, :3]) > 0:
        flip[0, 0] = -1
    return flip


def load_fsl_gradients(bval_file: Path, bvec_file: Path, affine: np.ndarray):
    """Read FSL bval/bvec files; return (bvals, bvecs_voxel) with bvecs as (N, 3)."""
    from dipy.io.gradients import read_bvals_bvecs
    bvals, bvecs = read_bvals_bvecs(str(bval_file), str(bvec_file))
    bvecs = bvecs @ fsl_flip(affine).T
    return bvals.astype(float), bvecs.astype(float)


def write_fsl_gradients(bvals: np.ndarray, bvecs_voxel: np.ndarray, affine: np.ndarray,
                        bval_file: Path, bvec_file: Path) -> None:
    """Write bvals (1 row) and bvecs (3 rows) in FSL convention."""
    bvecs = np.asarray(bvecs_voxel) @ fsl_flip(affine).T
    np.savetxt(bval_file, np.asarray(bvals)[None, :], fmt="%g", delimiter=" ")
    np.savetxt(bvec_file, bvecs.T, fmt="%.6f", delimiter=" ")


def make_gtab(bvals: np.ndarray, bvecs: np.ndarray, b0_threshold: float):
    from dipy.core.gradients import gradient_table
    return gradient_table(bvals, bvecs=bvecs, b0_threshold=b0_threshold)


def direction_cosines(affine: np.ndarray) -> np.ndarray:
    """3x3 matrix whose columns are the unit world directions of the voxel axes."""
    m = affine[:3, :3]
    return m / np.linalg.norm(m, axis=0)


def rotate_bvecs(bvecs_voxel: np.ndarray, reg_affines: np.ndarray, img_affine: np.ndarray,
                 b0_mask: np.ndarray) -> np.ndarray:
    """Rotate voxel-axis b-vectors with the registration affines.

    ``reg_affines`` has shape (4, 4, N): for volume ``n`` it is the DIPY
    ``AffineMap.affine`` returned by ``affine_registration`` with the reference
    b0 as *static* image, i.e. it maps reference-space world coordinates to
    the world coordinates of the moving (acquired) volume.

    The diffusion gradient applied during acquisition is fixed relative to the
    acquired volume. Bringing that volume into reference space rotates the
    tissue by R^-1, so the gradient direction in reference space is
    ``R^-1 g`` (Leemans & Jones 2009; same rule as DIPY's
    ``reorient_bvecs``). R is the rotation part (polar decomposition) of the
    world-space affine; because b-vectors are stored in voxel axes we conjugate
    with the image direction cosines: ``R_vox = M^-1 R_world M``.
    """
    from scipy.linalg import polar
    m = direction_cosines(img_affine)
    m_inv = np.linalg.inv(m)
    out = np.array(bvecs_voxel, dtype=float, copy=True)
    for n in range(out.shape[0]):
        if b0_mask[n] or not np.any(out[n]):
            continue
        r_world, _ = polar(reg_affines[:3, :3, n])
        r_vox = m_inv @ r_world @ m
        g = np.linalg.inv(r_vox) @ out[n]
        out[n] = g / np.linalg.norm(g)
    return out


# ---------------------------------------------------------------------------
# 7. Anatomy loader
# ---------------------------------------------------------------------------


class Anatomy:
    """T1-weighted brain + tissue probability maps in the T1 (world) space.

    Attributes
    ----------
    t1_brain : (X, Y, Z) float array, skull-stripped T1
    brain_mask : (X, Y, Z) bool array
    cgm, sgm, wm, csf : (X, Y, Z) float arrays in [0, 1]
        Cortical GM, sub-cortical GM, WM and CSF fractions (the four non-
        pathological tissue types of MRtrix's 5TT format).
    affine : (4, 4) voxel -> world (scanner RAS mm) matrix
    source : description of where they came from (for the QC JSON)
    """

    def __init__(self, t1_brain, brain_mask, cgm, sgm, wm, csf, affine, source):
        self.t1_brain = t1_brain.astype(np.float32)
        self.brain_mask = brain_mask.astype(bool)
        self.cgm, self.sgm, self.wm, self.csf = (x.astype(np.float32) for x in (cgm, sgm, wm, csf))
        self.affine = affine
        self.source = source


def _load(path: Path) -> nib.spatialimages.SpatialImage:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing anatomical input: {path}")
    return nib.load(str(path))


def load_anatomy(participant: str, source: str, fs_dir: Path | None = None,
                 preproc_dir: Path | None = None) -> Anatomy:
    """Load the anatomical inputs.

    ``source='freesurfer'`` (default of the DIPY pipeline): ``<fs_dir>/sub-X/mri``
    must contain ``brainmask.mgz`` (skull-stripped T1) and ``aparc+aseg.mgz``.
    Works with any FreeSurfer-style ``SUBJECTS_DIR``, e.g. DeepPrep's
    ``Recon`` folder. Tissue maps are the binary label groups defined at the
    top of this file (partial volumes appear when they are resampled to the
    coarser DWI grid).

    ``source='probtissue'``: the layout described in the original
    compneuro-dwiproc README::

        <preproc_dir>/BET/sub-X_T1w_brain.nii.gz
        <preproc_dir>/ProbTissue/sub-X_T1w_brain_{corticalGM,subcorticalGM,WM,CSF}.nii.gz
    """
    if source == "freesurfer":
        if fs_dir is None:
            raise ValueError("--fs-dir is required with --anat-source freesurfer")
        mri = Path(fs_dir) / participant / "mri"
        t1_img = _load(mri / "brainmask.mgz")
        seg_img = _load(mri / "aparc+aseg.mgz")
        t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
        seg = np.asarray(seg_img.dataobj).astype(np.int32)
        if not np.allclose(t1_img.affine, seg_img.affine, atol=1e-3):
            raise ValueError("brainmask.mgz and aparc+aseg.mgz have different geometry")
        cortex = np.isin(seg, FS_CORTICAL_GM) | ((seg >= 1000) & (seg < 3000))
        cgm = cortex.astype(np.float32)
        sgm = np.isin(seg, FS_SUBCORTICAL_GM).astype(np.float32)
        wm = np.isin(seg, FS_WM).astype(np.float32)
        csf = np.isin(seg, FS_CSF).astype(np.float32)
        mask = (seg > 0) | (t1 > 0)
        t1 = np.where(mask, t1, 0)
        return Anatomy(t1, mask, cgm, sgm, wm, csf, t1_img.affine,
                       source=f"freesurfer:{mri}")

    if source == "probtissue":
        if preproc_dir is None:
            raise ValueError("--preproc-dir is required with --anat-source probtissue")
        pdir = Path(preproc_dir)
        t1_img = _load(pdir / "BET" / f"{participant}_T1w_brain.nii.gz")
        t1 = np.asarray(t1_img.dataobj, dtype=np.float32)
        maps = {}
        for key, suffix in [("cgm", "corticalGM"), ("sgm", "subcorticalGM"),
                            ("wm", "WM"), ("csf", "CSF")]:
            img = _load(pdir / "ProbTissue" / f"{participant}_T1w_brain_{suffix}.nii.gz")
            if img.shape[:3] != t1_img.shape[:3] or not np.allclose(img.affine, t1_img.affine, atol=1e-3):
                raise ValueError(f"ProbTissue {suffix} map is not on the BET T1 grid")
            maps[key] = np.clip(np.asarray(img.dataobj, dtype=np.float32), 0, 1)
        mask = t1 > 0
        return Anatomy(t1, mask, maps["cgm"], maps["sgm"], maps["wm"], maps["csf"],
                       t1_img.affine, source=f"probtissue:{pdir}")

    raise ValueError(f"Unknown anatomy source: {source}")


# ---------------------------------------------------------------------------
# 8. Command-line options shared by all scripts
# ---------------------------------------------------------------------------


def add_common_arguments(parser) -> None:
    """Options used by both stages (inputs, outputs, anatomy, CPU)."""
    g = parser.add_argument_group("inputs / outputs")
    g.add_argument("--bids-dir", required=True, type=Path,
                   help="BIDS root (with sub-*/[ses-*/]dwi) or a single subject folder "
                        "(e.g. ~/Desktop/sub-6175).")
    g.add_argument("--output-dir", required=True, type=Path,
                   help="Output root; results go to <output-dir>/sub-X[/ses-Y]. "
                        "Equivalent of Preproc/Dwiprep in the original pipeline.")
    g.add_argument("--participant-label", nargs="+", default=None,
                   help="Participants to process (with or without 'sub-'). Default: "
                        "participants.tsv, else every sub-* folder.")
    a = parser.add_argument_group("anatomy")
    a.add_argument("--anat-source", choices=["freesurfer", "probtissue"], default="freesurfer",
                   help="'freesurfer': brainmask.mgz + aparc+aseg.mgz from --fs-dir/sub-X/mri "
                        "(e.g. DeepPrep's Recon). 'probtissue': the original layout "
                        "<preproc-dir>/BET and <preproc-dir>/ProbTissue. Default: freesurfer.")
    a.add_argument("--fs-dir", type=Path, default=None,
                   help="FreeSurfer SUBJECTS_DIR (folder that contains sub-X/mri).")
    a.add_argument("--preproc-dir", type=Path, default=None,
                   help="Folder with BET/ and ProbTissue/ (only for --anat-source probtissue).")
    c = parser.add_argument_group("general")
    c.add_argument("--n-jobs", type=int, default=4,
                   help="CPU processes/threads to use (default 4, like -nthreads 4 in the original).")
    c.add_argument("--b0-threshold", type=float, default=50,
                   help="b-values <= this are treated as b0 (default 50 s/mm2).")
    c.add_argument("--overwrite", action="store_true",
                   help="Recompute steps whose outputs already exist.")


# ---------------------------------------------------------------------------
# 9. Small image utilities
# ---------------------------------------------------------------------------


def voxel_sizes(affine: np.ndarray) -> np.ndarray:
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Largest 26-connected component of a binary mask (empty stays empty)."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def resolve_template(name_or_path: str) -> Path:
    """Accept a path or a file name/stem inside ``brain_templates``."""
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p
    for cand in (TEMPLATES_DIR / name_or_path, TEMPLATES_DIR / f"{name_or_path}.nii.gz"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Template/parcellation not found: {name_or_path} "
                            f"(looked in {TEMPLATES_DIR})")


def parcellation_name(path: Path) -> str:
    """'.../craddock_2128_2mm.nii.gz' -> 'craddock_2128_2mm' (as basename ... .nii.gz)."""
    return _strip_nii(Path(path))
