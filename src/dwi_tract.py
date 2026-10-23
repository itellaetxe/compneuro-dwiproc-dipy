#!/usr/bin/env python3
"""
dwi_tract.py
============

Registration, tissue maps, fibre orientations, tractography and structural
connectivity with DIPY 1.12 -- the DIPY counterpart of
``compneuro-dwiproc/src/dwi_tract.sh`` (FSL epi_reg, ANTs, MRtrix3).

Step-by-step correspondence with the original script
----------------------------------------------------

====  ==========================================  =============================================
Step  Original                                    This script (DIPY 1.12)
====  ==========================================  =============================================
1     epi_reg (BBR) b0 -> T1, convert_xfm,        Rigid (6 dof) b0 -> T1 registration with mutual
      flirt -applyxfm: T1 into DWI space          information; T1 resampled onto the DWI grid
                                                  (``anat_dwispace.nii.gz``)
2     antsRegistrationSyN.sh MNI -> anat_dwispace Affine (MI) + SyN (cross-correlation) of the
      antsApplyTransforms -n NearestNeighbor      MNI template to ``anat_dwispace``; parcellation
                                                  warped with nearest-neighbour interpolation
3     5ttgen fsl (FAST + FIRST), 5tt2gmwmi        Tissue maps from FreeSurfer aparc+aseg (or the
                                                  original ProbTissue maps) resampled to the DWI
                                                  grid as partial-volume fractions (``act_5tt``);
                                                  GM/WM interface seeding map (``GMWMI``)
4     dwi2response tax                            ``recursive_response`` (Tax et al. 2014, same
                                                  algorithm) in white matter
5     dwi2fod csd                                 ``ConstrainedSphericalDeconvModel`` (lmax 8)
6     tckgen iFOD2 -act -backtrack -seed_gmwmi    Probabilistic tracking on the CSD FODs with
      -angle 45 -maxlength 200 -select 3M         DIPY's ACT criterion (stop at GM > 0.5, reject at
                                                  CSF > 0.5), GM/WM-interface seeding, max angle 45,
                                                  max length 200 mm, run until N streamlines are
                                                  accepted (= -select). ``--prob-algorithm pft``
                                                  adds particle filtering (DIPY's analogue of
                                                  -backtrack).
7     tcksift2                                    No DIPY equivalent: matrices are raw streamline
                                                  counts (no SIFT2 weights).
8     tck2connectome -zero_diagonal -symmetric    Endpoint-based count matrix with the same
      -assignment_radial_search 5                 5 mm radial search, zero diagonal, symmetric
9     tckgen FACT on the tensor eigenvector       Deterministic tracking on the tensor principal
      + tck2connectome                            direction (FA cut-off 0.1) with the same
                                                  stopping/seeding/connectome settings
====  ==========================================  =============================================

Main outputs (``<output-dir>/sub-X[/ses-Y]``, same names as the original)::

    anat_dwispace.nii.gz          T1 brain in DWI space
    <parc>_subSpace.nii.gz        parcellation in DWI space
    act_5tt.nii.gz                5 volumes: cGM, sGM, WM, CSF, pathological(=0), MRtrix order
    GMWMI.nii.gz                  seeding map
    dwi_response.txt, dwi_fod.nii.gz (SH coefficients, DIPY 'descoteaux07' basis)
    <parc>_prob.csv / <parc>_det.csv   connectivity matrices (streamline counts)
    streamlines_prob.trk / streamlines_det.trk   (compressed; --no-save-tractograms to skip)
    dwireg/                       transforms and QC images
    qc_data/tract_<parc>_desc.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dwiproc_common as dc  # noqa: E402
from dwi_preproc import PREPROC_DONE  # noqa: E402


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def add_tract_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("tractography (dwi_tract.py)")
    g.add_argument("--parcellation", default=None,
                   help="Parcellation in MNI152 2 mm space: a path, or a file name in "
                        "brain_templates/ (e.g. Desikan_genetics_2mm).")
    g.add_argument("--template", default="MNI152_T1_2mm_brain",
                   help="Skull-stripped MNI template on the parcellation grid "
                        "(default brain_templates/MNI152_T1_2mm_brain).")
    g.add_argument("--n-streamlines", type=int, default=3_000_000,
                   help="Accepted streamlines per tractogram (MRtrix -select). Default 3000000, "
                        "as in the original pipeline.")
    g.add_argument("--max-seeds-factor", type=float, default=1000,
                   help="Give up after n-streamlines x this many seeds (MRtrix default 1000).")
    g.add_argument("--max-angle", type=float, default=45, help="Max angle per step (deg), 45.")
    g.add_argument("--max-length", type=float, default=200, help="Max length (mm), 200.")
    g.add_argument("--min-length", type=float, default=None,
                   help="Min length (mm). Default 2 x voxel size (MRtrix default with ACT).")
    g.add_argument("--prob-step", type=float, default=None,
                   help="Probabilistic step (mm). Default 0.5 x voxel size (iFOD2 default).")
    g.add_argument("--det-step", type=float, default=None,
                   help="Deterministic step (mm). Default 0.1 x voxel size (FACT default).")
    g.add_argument("--stopping", choices=["act", "cmc"], default="act",
                   help="Anatomical stopping criterion. 'act' (default): stop when the "
                        "interpolated GM fraction > 0.5 (valid end) or CSF > 0.5 (rejected), "
                        "like MRtrix ACT. 'cmc': DIPY's continuous map criterion (probabilistic "
                        "stopping proportional to the partial volumes; gives many very short "
                        "streamlines at DWI resolution).")
    g.add_argument("--prob-algorithm", choices=["prob", "pft"], default="prob",
                   help="'prob' (default) = probabilistic tracking; 'pft' = particle-filtering "
                        "tractography (closest to -backtrack, slower).")
    g.add_argument("--fod-threshold", type=float, default=0.1,
                   help="PMF threshold for probabilistic tracking (like tckgen -cutoff 0.1).")
    g.add_argument("--fa-threshold", type=float, default=0.1,
                   help="FA cut-off for deterministic tracking (FACT -cutoff 0.1).")
    g.add_argument("--sh-order", type=int, default=8, help="CSD lmax (default 8).")
    g.add_argument("--response", choices=["tax", "ssst"], default="tax",
                   help="'tax' = recursive calibration (dwi2response tax, default; falls back "
                        "to 'ssst' with a warning if it fails); 'ssst' = single-fibre response "
                        "from WM voxels with FA > 0.7.")
    g.add_argument("--radial-search", type=float, default=5,
                   help="tck2connectome -assignment_radial_search distance (mm), 5.")
    g.add_argument("--no-save-tractograms", action="store_true",
                   help="Do not write the .trk files (only the matrices).")
    g.add_argument("--skip-det", action="store_true", help="Skip the deterministic tractogram.")
    g.add_argument("--skip-prob", action="store_true", help="Skip the probabilistic tractogram.")
    g.add_argument("--tracking-mem-gb", type=float, default=1.0,
                   help="Memory for DIPY's tracking buffers per batch of seeds (GB, default 1). "
                        "Larger = fewer, bigger batches.")
    g.add_argument("--random-seed", type=int, default=42, help="Seed for reproducible seeding.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    dc.add_common_arguments(p)
    add_tract_arguments(p)
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_preproc(out_dir: Path, b0_threshold: float) -> dict:
    """Read the outputs of dwi_preproc.py from ``out_dir``."""
    img = nib.load(str(out_dir / "dwi_clean.nii.gz"))
    affine = img.affine
    bvals, bvecs = dc.load_fsl_gradients(out_dir / "rotated.bval", out_dir / "rotated.bvec", affine)
    return dict(
        img=img, affine=affine, data=np.asarray(img.dataobj, dtype=np.float32),
        gtab=dc.make_gtab(bvals, bvecs, b0_threshold),
        b0=np.asarray(nib.load(str(out_dir / "dwi_bzero.nii.gz")).dataobj, dtype=np.float32),
        mask=np.asarray(nib.load(str(out_dir / "dwi_mask.nii.gz")).dataobj) > 0,
        fa=np.asarray(nib.load(str(out_dir / "dwi_FA.nii.gz")).dataobj, dtype=np.float32),
        dirs=np.asarray(nib.load(str(out_dir / "dwi_directions.nii.gz")).dataobj, dtype=np.float32),
    )


def corr_in(a, b, mask) -> float:
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def gmwmi_t1(anat: dc.Anatomy) -> np.ndarray:
    """Grey/white-matter interface in T1 space (as a 0/1 map).

    MRtrix ``5tt2gmwmi`` marks voxels where GM and WM partial volumes meet.
    With (near-)binary tissue maps the interface is the one-voxel layer on
    each side of the GM/WM boundary (6-connectivity); after resampling to the
    coarser DWI grid it becomes a continuous seeding weight.
    """
    gm = (anat.cgm + anat.sgm) >= 0.5
    wm = anat.wm >= 0.5
    st = ndimage.generate_binary_structure(3, 1)
    return ((wm & ndimage.binary_dilation(gm, st)) |
            (gm & ndimage.binary_dilation(wm, st))).astype(np.float32)


def radial_label_lookup(parc: np.ndarray, affine: np.ndarray, radius: float) -> np.ndarray:
    """Label volume used for endpoint assignment with a radial search.

    ``tck2connectome -assignment_radial_search R``: an endpoint in an
    unlabelled voxel is assigned to the nearest labelled voxel within R mm.
    Here every unlabelled voxel gets the label of its nearest labelled voxel
    (Euclidean distance in mm, anisotropic voxels respected) if that voxel is
    within R mm, else 0. Endpoint lookup is then a single array access.
    """
    dist, inds = ndimage.distance_transform_edt(parc == 0, sampling=dc.voxel_sizes(affine),
                                                return_indices=True)
    out = parc[tuple(inds)]
    out[dist > radius] = 0
    return out


class SeedSampler:
    """Random seeds inside voxels, with probability proportional to a weight map
    (like ``tckgen -seed_gmwmi`` / ``-seed_image`` with a non-binary image)."""

    def __init__(self, weights: np.ndarray, affine: np.ndarray, rng: np.random.Generator):
        self.idx = np.argwhere(weights > 0)
        w = weights[weights > 0].astype(float)
        self.p = w / w.sum()
        self.affine = affine
        self.rng = rng

    def sample(self, n: int) -> np.ndarray:
        """``n`` seed points in world (scanner RAS mm) coordinates."""
        pick = self.rng.choice(len(self.idx), size=n, p=self.p)
        ijk = self.idx[pick] + self.rng.uniform(-0.5, 0.5, size=(n, 3))
        return ijk @ self.affine[:3, :3].T + self.affine[:3, 3]


def track_and_count(kind, tracker_fn, tracker_kwargs, sampler, n_target, max_seeds,
                    lookup, n_nodes, affine, min_len, keep_streamlines, max_batch, logger):
    """Run the tracker in batches until ``n_target`` streamlines are accepted.

    Equivalent of ``tckgen -select N`` followed by ``tck2connectome``:

    * seeds are drawn in batches from ``sampler``; only streamlines that end
      in a valid way for the anatomical criterion are kept (``return_all=False``)
      and those shorter than ``min_len`` are discarded;
    * the two endpoints of each accepted streamline are looked up in the
      radial-search label volume and the pair (i, j) is counted;
    * batches continue until ``n_target`` streamlines are accepted, or until
      ``max_seeds`` seeds were used (then a warning is logged).

    Only endpoints are kept in memory; streamlines are compressed (0.1 mm
    tolerance) if they are to be saved.

    ``max_batch`` bounds the seeds per tracker call: DIPY pre-allocates, for
    every seed, a buffer of max_len/step points in both directions, so the
    batch size sets the peak memory (see ``--tracking-mem-gb``).
    """
    from dipy.tracking.streamlinespeed import compress_streamlines, length
    from nibabel.streamlines import ArraySequence

    inv = np.linalg.inv(affine)
    counts = np.zeros((n_nodes + 1, n_nodes + 1), dtype=np.int64)
    kept, n_seeds, n_assigned = 0, 0, 0
    lengths = []
    saved = ArraySequence()          # contiguous float32 storage (low memory overhead)
    batch = min(max(20_000, n_target // 10), max_batch)
    t0 = time.time()
    while kept < n_target and n_seeds < max_seeds:
        seeds = sampler.sample(batch)
        n_seeds += batch
        sls = [s for s in tracker_fn(seeds, **tracker_kwargs) if len(s) > 1]
        if sls:
            lens = np.asarray(length(sls))
            ok = lens >= min_len
            sls = [s for s, k in zip(sls, ok) if k]
            lens = lens[ok]
        if not sls:
            logger.warning(f"   {kind}: no streamline accepted in a batch of {batch} seeds")
            continue
        sls = sls[: n_target - kept]
        lens = lens[: len(sls)]
        ends = np.array([[s[0], s[-1]] for s in sls])                     # (n, 2, 3) world
        vox = np.rint(ends @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        vox = np.clip(vox, 0, np.array(lookup.shape) - 1)
        lab = lookup[vox[..., 0], vox[..., 1], vox[..., 2]]               # (n, 2)
        np.add.at(counts, (lab[:, 0], lab[:, 1]), 1)
        n_assigned += int(np.sum((lab[:, 0] > 0) & (lab[:, 1] > 0)))
        lengths.append(lens)
        if keep_streamlines:
            saved.extend(ArraySequence([c.astype(np.float32)
                                        for c in compress_streamlines(sls, tol_error=0.1)]))
        kept += len(sls)
        rate = kept / n_seeds
        logger.info(f"   {kind}: {kept}/{n_target} streamlines from {n_seeds} seeds "
                    f"({100 * rate:.1f}% accepted, {(time.time() - t0) / 60:.1f} min)")
        # next batch sized from the observed acceptance rate (10 % margin)
        batch = int(min(max_batch, max(1_000, 1.1 * (n_target - kept) / max(rate, 1e-3))))
    if kept < n_target:
        logger.warning(f"   {kind}: only {kept} of {n_target} streamlines after {n_seeds} seeds")

    # symmetric, zero diagonal, node 0 (= unassigned) dropped
    counts = counts + counts.T
    np.fill_diagonal(counts, 0)
    mat = counts[1:, 1:]
    all_len = np.concatenate(lengths) if lengths else np.array([])
    stats = dict(accepted=kept, seeds=n_seeds, acceptance_rate=kept / max(n_seeds, 1),
                 assigned_both_ends=n_assigned,
                 fraction_assigned=n_assigned / max(kept, 1),
                 mean_length_mm=float(all_len.mean()) if all_len.size else None,
                 median_length_mm=float(np.median(all_len)) if all_len.size else None,
                 minutes=round((time.time() - t0) / 60, 1))
    return mat, saved, stats


def save_trk(path: Path, streamlines, ref_img) -> None:
    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram
    sft = StatefulTractogram(streamlines, ref_img, Space.RASMM)
    save_tractogram(sft, str(path), bbox_valid_check=False)


# ---------------------------------------------------------------------------
# Main per-run function
# ---------------------------------------------------------------------------


def run_tract(out_dir: Path, participant: str, args, logger) -> dict:
    from dipy.align import affine_registration, syn_registration, write_mapping
    from dipy.align.imaffine import AffineMap
    from dipy.core.sphere import Sphere
    from dipy.data import default_sphere, get_sphere
    from dipy.direction.peaks import PeaksAndMetrics, peaks_from_model
    from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                       recursive_response, response_from_mask_ssst)
    from dipy.tracking.stopping_criterion import ActStoppingCriterion, CmcStoppingCriterion
    from dipy.tracking.tracker import (deterministic_tracking, pft_tracking,
                                       probabilistic_tracking)

    out_dir = Path(out_dir)
    if not args.parcellation:
        raise ValueError("--parcellation is required for the tractography stage")
    if not (out_dir / PREPROC_DONE).is_file():
        raise FileNotFoundError(f"Preprocessing not finished in {out_dir}")
    reg_dir = out_dir / "dwireg"
    qc_dir = out_dir / "qc_data"
    reg_dir.mkdir(parents=True, exist_ok=True)
    parc_path = dc.resolve_template(args.parcellation)
    parc_name = dc.parcellation_name(parc_path)
    tmpl_path = dc.resolve_template(args.template)
    timings: dict = {}
    qc: dict = dict(participant=participant, parcellation=str(parc_path), template=str(tmpl_path),
                    software=dc.software_versions(), parameters={
                        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})

    # Reuse earlier results only if they are newer than the preprocessed data
    # (a re-run of the preprocessing invalidates them).
    clean_mtime = (out_dir / "dwi_clean.nii.gz").stat().st_mtime

    def reuse(path: Path) -> bool:
        return path.is_file() and not args.overwrite and path.stat().st_mtime >= clean_mtime

    pp = load_preproc(out_dir, args.b0_threshold)
    D, shape3 = pp["affine"], pp["data"].shape[:3]
    vox = dc.voxel_sizes(D)

    # ------------------------------------------------------------------ 1
    # Rigid b0 -> T1 (replaces epi_reg + convert_xfm + flirt -applyxfm)
    anat = dc.load_anatomy(participant, args.anat_source, args.fs_dir, args.preproc_dir)
    qc["anatomy_source"] = anat.source
    with dc.timed(logger, "Registering b0 to the T1 (rigid, MI)", timings):
        b0_in_t1, rigid = affine_registration(
            pp["b0"], anat.t1_brain, moving_affine=D, static_affine=anat.affine,
            pipeline=["center_of_mass", "translation", "rigid"], metric="MI",
            level_iters=[1000, 500, 100], static_mask=anat.brain_mask.astype(np.int32))
        np.savetxt(reg_dir / "epi2anat_affine.txt", rigid, fmt="%.8f")
        dc.save_nifti(reg_dir / "b0_anatspace.nii.gz", b0_in_t1, anat.affine)
        # AffineMap with the T1 as domain and the DWI as codomain:
        # transform_inverse() brings T1-space images onto the DWI grid.
        amap = AffineMap(rigid, domain_grid_shape=anat.t1_brain.shape, domain_grid2world=anat.affine,
                         codomain_grid_shape=shape3, codomain_grid2world=D)
        anat_dwi = amap.transform_inverse(anat.t1_brain, interpolation="linear")
        dc.save_nifti(out_dir / "anat_dwispace.nii.gz", anat_dwi, D)

    # ------------------------------------------------------------------ 3
    # Tissue fractions on the DWI grid (5ttgen) and GM/WM interface (5tt2gmwmi)
    with dc.timed(logger, "Tissue maps and GM/WM interface in DWI space", timings):
        pv = {k: np.clip(amap.transform_inverse(getattr(anat, k), interpolation="linear"), 0, 1)
              for k in ("cgm", "sgm", "wm", "csf")}
        total = sum(pv.values())
        scale = np.where(total > 1, 1 / np.maximum(total, 1e-6), 1)
        pv = {k: v * scale for k, v in pv.items()}
        five_tt = np.stack([pv["cgm"], pv["sgm"], pv["wm"], pv["csf"], np.zeros(shape3)], axis=-1)
        dc.save_nifti(out_dir / "act_5tt.nii.gz", five_tt, D)
        gmwmi = np.clip(amap.transform_inverse(gmwmi_t1(anat), interpolation="linear"), 0, 1)
        gmwmi[~pp["mask"]] = 0
        dc.save_nifti(out_dir / "GMWMI.nii.gz", gmwmi, D)
        wm_pv = np.where(pp["mask"], pv["wm"], 0)
        gm_pv = np.where(pp["mask"], pv["cgm"] + pv["sgm"], 0)
        csf_pv = np.where(pp["mask"], pv["csf"], 0)
        # Outside the tissue maps / brain mask = "background". As in DIPY's ACT
        # example it is added to the include map, so a streamline that leaves
        # the brain (e.g. through the brainstem) ends validly instead of
        # continuing (MRtrix ACT also accepts inferior brainstem exits).
        background = np.clip(1 - wm_pv - gm_pv - csf_pv, 0, 1)
        # QC of the b0->T1 alignment: FA should be high in WM, low in GM
        wm_core, gm_core = wm_pv > 0.9, gm_pv > 0.9
        # (independent of the registration cost: better DWI-T1 alignment gives
        # a higher AUC and a higher FA-vs-WM-fraction correlation)
        fa_wm, fa_gm = np.sort(pp["fa"][wm_core]), np.sort(pp["fa"][gm_core])
        auc = (float(np.searchsorted(fa_gm, fa_wm).sum()) / (fa_wm.size * fa_gm.size)
               if fa_wm.size and fa_gm.size else None)
        in_brain = pp["mask"] & (pp["fa"] > 0)
        qc["alignment_qc"] = dict(
            b0_T1_corr_dwispace=corr_in(pp["b0"], anat_dwi, pp["mask"] & (anat_dwi > 0)),
            median_FA_in_WM=float(np.median(fa_wm)) if fa_wm.size else None,
            median_FA_in_GM=float(np.median(fa_gm)) if fa_gm.size else None,
            auc_FA_WM_vs_GM=auc,
            corr_FA_vs_WM_fraction=corr_in(pp["fa"], wm_pv, in_brain))
        logger.info(f"   alignment QC: median FA WM {qc['alignment_qc']['median_FA_in_WM']:.2f} / "
                    f"GM {qc['alignment_qc']['median_FA_in_GM']:.2f}, AUC {auc:.3f}, "
                    f"corr(FA, WM) {qc['alignment_qc']['corr_FA_vs_WM_fraction']:.3f}")

    # ------------------------------------------------------------------ 2
    # MNI template -> anat_dwispace (affine + SyN) and parcellation warp
    parc_out = out_dir / f"{parc_name}_subSpace.nii.gz"
    parc_img = nib.load(str(parc_path))
    parc_mni = np.rint(np.squeeze(np.asarray(parc_img.dataobj, dtype=np.float64))).astype(np.int32)
    n_nodes = int(parc_mni.max())
    warp_file = reg_dir / "standard2dwi_warp.nii.gz"
    if reuse(parc_out):
        logger.info(f"   reusing {parc_out.name}")
        parc_dwi = np.asarray(nib.load(str(parc_out)).dataobj).astype(np.int32)
    else:
        with dc.timed(logger, "Registering the MNI template to anat_dwispace (affine + SyN)", timings):
            tmpl_img = nib.load(str(tmpl_path))
            tmpl = np.asarray(tmpl_img.dataobj, dtype=np.float64)
            static_mask = (anat_dwi > 0).astype(np.int32)
            _, aff = affine_registration(
                tmpl, anat_dwi, moving_affine=tmpl_img.affine, static_affine=D,
                pipeline=["center_of_mass", "translation", "rigid", "affine"], metric="MI",
                level_iters=[1000, 500, 100], static_mask=static_mask)
            np.savetxt(reg_dir / "standard2dwi_affine.txt", aff, fmt="%.8f")
            tmpl_warped, mapping = syn_registration(
                tmpl, anat_dwi, moving_affine=tmpl_img.affine, static_affine=D,
                prealign=aff, metric="CC", level_iters=[100, 50, 25])
            write_mapping(mapping, str(warp_file))
            dc.save_nifti(reg_dir / "template_dwispace.nii.gz", tmpl_warped, D)
            qc["template_registration_corr"] = corr_in(tmpl_warped, anat_dwi, anat_dwi > 0)
            logger.info(f"   template vs T1 correlation in DWI space: "
                        f"{qc['template_registration_corr']:.3f}")
            if not np.allclose(parc_img.affine, tmpl_img.affine, atol=1e-3) or \
                    parc_mni.shape != tmpl.shape:
                raise ValueError("Parcellation and template are not on the same grid")
            parc_dwi = np.rint(mapping.transform(parc_mni.astype(np.float64),
                                                 interpolation="nearest")).astype(np.int32)
            dc.save_nifti(parc_out, parc_dwi, D, dtype=np.int32)
    present = np.unique(parc_dwi[parc_dwi > 0])
    qc["parcellation"] = dict(n_nodes=n_nodes, nodes_present_in_dwi=int(present.size),
                              missing_nodes=sorted(set(range(1, n_nodes + 1)) - set(present.tolist())))
    lookup = radial_label_lookup(parc_dwi, D, args.radial_search)

    # ------------------------------------------------------------------ 4-5
    # Response function (Tax) and CSD fibre orientation distributions
    fod_file = out_dir / "dwi_fod.nii.gz"
    gtab, data, mask = pp["gtab"], pp["data"], pp["mask"]
    # lmax must not need more SH coefficients than there are DW directions
    # ((l+1)(l+2)/2 <= N), otherwise CSD is under-determined and the FODs are
    # wrong. MRtrix lowers lmax automatically in the same way.
    n_dirs = int((~gtab.b0s_mask).sum())
    sh_order = args.sh_order
    while sh_order > 2 and (sh_order + 1) * (sh_order + 2) // 2 > n_dirs:
        sh_order -= 2
    if sh_order != args.sh_order:
        logger.warning(f"   lmax lowered from {args.sh_order} to {sh_order}: only {n_dirs} "
                       "diffusion directions")
    qc["sh_order"] = sh_order
    if not args.skip_prob:
        if reuse(fod_file):
            logger.info(f"   reusing {fod_file.name}")
            sh = np.asarray(nib.load(str(fod_file)).dataobj, dtype=np.float64)
        else:
            with dc.timed(logger, f"Response function ({args.response})", timings):
                wm_mask = mask & (wm_pv > 0.5)
                method = args.response
                if method == "tax":
                    try:
                        response = recursive_response(
                            gtab, data, mask=wm_mask, sh_order_max=sh_order,
                            peak_thr=0.01, init_fa=0.08, init_trace=0.0021, iter=8,
                            convergence=0.001, parallel=args.n_jobs > 1,
                            num_processes=args.n_jobs)
                        resp_txt = np.concatenate([[response.S0], np.ravel(response.dwi_response)])
                        qc["response"] = dict(method="tax", S0=float(response.S0),
                                              sh_m0=np.ravel(response.dwi_response).tolist())
                    except Exception as e:      # reported, never silent
                        logger.warning(f"   Tax response failed ({e!r}); using the single-fibre "
                                       f"(FA > 0.7 in WM) response instead")
                        method = "ssst"
                        qc["response_fallback"] = f"tax failed: {e!r}"
                if method == "ssst":
                    sf_mask = wm_mask & (pp["fa"] > 0.7)
                    if sf_mask.sum() < 100:
                        raise ValueError(f"only {int(sf_mask.sum())} WM voxels with FA > 0.7 "
                                         "for the response function")
                    response, ratio = response_from_mask_ssst(gtab, data, sf_mask)
                    resp_txt = np.concatenate([response[0], [response[1]]])
                    qc["response"] = dict(method="ssst", evals=list(response[0]),
                                          S0=float(response[1]), ratio=float(ratio),
                                          n_voxels=int(sf_mask.sum()))
                np.savetxt(out_dir / "dwi_response.txt", resp_txt[None, :], fmt="%.6g")
            with dc.timed(logger, f"CSD fit (lmax {sh_order})", timings):
                csd = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=sh_order)
                pam = peaks_from_model(csd, data, default_sphere, relative_peak_threshold=0.5,
                                       min_separation_angle=25, mask=mask, return_sh=True,
                                       npeaks=3, parallel=args.n_jobs > 1,
                                       num_processes=args.n_jobs)
                sh = pam.shm_coeff
                dc.save_nifti(fod_file, sh, D)
                dc.save_nifti(out_dir / "dwi_fod_peaks.nii.gz",
                              pam.peak_dirs.reshape(shape3 + (-1,)), D)

    # ------------------------------------------------------------------ 6-9
    # Tractography + connectomes
    rng = np.random.default_rng(args.random_seed)
    sampler = SeedSampler(gmwmi, D, rng)
    min_len = args.min_length or 2 * float(vox.mean())
    max_seeds = int(args.n_streamlines * args.max_seeds_factor)
    qc["tracking"] = {}

    def run_one(kind, tracker_fn, kwargs, step):
        if args.stopping == "act":
            sc = ActStoppingCriterion(np.ascontiguousarray(gm_pv + background, dtype=float),
                                      np.ascontiguousarray(csf_pv, dtype=float))
        else:
            sc = CmcStoppingCriterion.from_pve(wm_pv, gm_pv + background, csf_pv, step_size=step,
                                               average_voxel_size=float(vox.mean()))
        kwargs = dict(kwargs, sc=sc, affine=D, step_size=step, max_angle=args.max_angle,
                      min_len=min_len, max_len=args.max_length, nbr_threads=args.n_jobs,
                      random_seed=args.random_seed, return_all=False)
        with dc.timed(logger, f"{kind} tractography ({args.n_streamlines} streamlines)", timings):
            # seeds per call so that DIPY's per-seed buffers (2 directions x
            # max_len/step points x 3 coords x 8 bytes) fit in --tracking-mem-gb
            per_seed = 2 * (args.max_length / step + 2) * 3 * 8
            max_batch = int(max(1_000, args.tracking_mem_gb * 1e9 / per_seed))
            mat, sls, stats = track_and_count(
                kind, tracker_fn, kwargs, sampler, args.n_streamlines, max_seeds, lookup,
                n_nodes, D, min_len, not args.no_save_tractograms, max_batch, logger)
            if not args.no_save_tractograms:
                save_trk(out_dir / f"streamlines_{kind}.trk", sls, pp["img"])
            stats["step_mm"] = step
            qc["tracking"][kind] = stats
        return mat

    prob_csv = out_dir / f"{parc_name}_prob.csv"
    if not args.skip_prob and not reuse(prob_csv):
        step = args.prob_step or 0.5 * float(vox.min())
        tracker = pft_tracking if args.prob_algorithm == "pft" else probabilistic_tracking
        mat = run_one("prob", tracker, dict(sh=sh, sphere=default_sphere,
                                            pmf_threshold=args.fod_threshold), step)
        np.savetxt(prob_csv, mat, fmt="%d", delimiter=",")

    det_csv = out_dir / f"{parc_name}_det.csv"
    if not args.skip_det and not reuse(det_csv):
        # Tensor principal direction as a single "peak" per voxel (FACT input).
        # dwi_directions is FA-modulated and in world axes -> back to voxel axes.
        fa = np.linalg.norm(pp["dirs"], axis=-1)
        v1 = pp["dirs"] / np.maximum(fa, 1e-12)[..., None]
        v1 = v1 @ np.linalg.inv(dc.direction_cosines(D)).T
        sphere = get_sphere(name="repulsion724")
        verts = sphere.vertices
        idx = np.zeros(shape3, dtype=np.int32)
        inside = np.argwhere(mask & (fa > 0))
        for k in range(0, len(inside), 50_000):      # closest vertex, sign-agnostic, in chunks
            ijk = inside[k:k + 50_000]
            vec = v1[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
            idx[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = np.argmax(np.abs(vec @ verts.T), axis=1)
        pam_t = PeaksAndMetrics()
        pam_t.peak_indices = idx[..., None].astype(np.int32)
        pam_t.peak_values = np.where(mask, fa, 0)[..., None].astype(float)
        pam_t.peak_dirs = verts[idx][..., None, :]
        pam_t.odf_vertices = verts
        pam_t.sphere = sphere
        step = args.det_step or 0.1 * float(vox.min())
        mat = run_one("det", deterministic_tracking,
                      dict(pam=pam_t, sphere=Sphere(xyz=verts),
                           pmf_threshold=args.fa_threshold), step)
        np.savetxt(det_csv, mat, fmt="%d", delimiter=",")

    # Merge with the QC of an earlier, interrupted run (steps that were reused
    # keep their earlier entries).
    qc_file = qc_dir / f"tract_{parc_name}_desc.json"
    old = dc.read_json(qc_file) if qc_file.is_file() and not args.overwrite else {}
    for key in ("tracking", "timings_s", "response"):
        if key in old:
            merged = dict(old[key])
            merged.update(qc.get(key, {}) if key != "timings_s" else timings)
            qc[key] = merged
    for key in ("template_registration_corr",):
        if key in old and key not in qc:
            qc[key] = old[key]
    qc.setdefault("timings_s", timings)
    dc.write_json(qc_file, qc)
    return qc


def main(argv=None):
    args = build_parser().parse_args(argv)
    logger = dc.get_logger(args.output_dir / "log" / f"Dwitract_{time.strftime('%Y%m%d_%H%M')}.txt")
    for participant in dc.list_participants(args.bids_dir, args.participant_label):
        runs = dc.find_dwi_runs(args.bids_dir, participant)
        for run in runs:
            n_same = sum(r["session"] == run["session"] for r in runs)
            out = dc.run_output_dir(args.output_dir, participant, run, n_same)
            run_tract(out, participant, args, logger)


if __name__ == "__main__":
    main()
