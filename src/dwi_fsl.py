#!/usr/bin/env python3
"""
dwi_fsl.py
==========

Susceptibility, eddy-current and motion correction with FSL (topup + eddy),
optionally with a synthetic undistorted b0 from Synb0-DisCo -- the step of
the original pipeline that DIPY cannot do (``dwifslpreproc`` in
compneuro-dwiproc).

Routes (chosen in ``dwi_preproc.py`` with ``--correction``)
-----------------------------------------------------------

``topup``   reverse phase-encoding spin-echo fieldmaps exist (``fmap/*_epi``):
            topup on [first DWI b0, fieldmaps] -> eddy with the topup field.
            Putting the DWI b0 first makes topup's reference space the DWI
            space (as ``dwifslpreproc -align_seepi`` in the original).
``synb0``   no fieldmaps: Synb0-DisCo synthesises an undistorted b0 from the
            T1w (container), then topup on [distorted b0, synthetic b0] with
            the synthetic b0 given a readout time of 0 (infinite bandwidth),
            then eddy with that field (Schilling et al. 2019, 2020).
``eddy``    eddy only (motion + eddy currents, no susceptibility correction).

How the tools are run
---------------------

* FSL natively (``$FSLDIR``, or ``--fsl-dir``), e.g. on the Mac; or inside a
  container (``--fsl-container``: a Docker image name or an Apptainer
  ``.sif``), e.g. on the DIPC cluster with the compneuro-dwiproc image
  (FSL 6.0.7.6).
* Synb0-DisCo always runs in its container (``--synb0-container``):
  Docker on the Mac (default image ``leonyichencai/synb0-disco:v3.1``),
  Apptainer on the cluster (a ``.sif`` built from that image).
* ``--container-engine docker|apptainer|singularity`` (default: docker on
  macOS, otherwise apptainer if installed).

All intermediate files are kept in ``<run dir>/fsl_work`` (one log per
command in ``fsl_work/logs``); finished steps are reused when the pipeline is
re-run, unless ``--overwrite``.
"""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dwiproc_common as dc  # noqa: E402

SYNB0_DOCKER_IMAGE = "leonyichencai/synb0-disco:v3.1"

# topup configuration used by Synb0-DisCo for [distorted b0, synthetic b0]
# (values from Synb0-DisCo's synb0.cnf: the b02b0.cnf schedule without the
# last, unsmoothed levels, which are not meaningful for a synthetic image).
SYNB0_TOPUP_CONFIG = """\
# topup schedule used by Synb0-DisCo (MASILab/Synb0-DISCO, src/synb0.cnf)
--warpres=20,16,14,12,10,6,4
--subsamp=2,2,2,2,2,1,1
--fwhm=8,6,4,3,3,2,1
--miter=5,5,5,5,5,15,15
--lambda=0.005,0.001,0.0001,0.000015,0.000005,0.0000005,0.00000005
--ssqlambda=1
--regmod=bending_energy
--estmov=1,1,1,1,1,0,0
--minmet=0,0,0,0,0,1,1
--splineorder=3
--numprec=double
--interp=spline
--scale=1
"""


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def default_engine() -> str:
    if platform.system() == "Darwin":
        return "docker"
    for eng in ("apptainer", "singularity"):
        if shutil.which(eng):
            return eng
    return "docker"


def default_fs_license() -> str | None:
    for cand in (os.environ.get("FS_LICENSE"),
                 os.path.join(os.environ.get("FREESURFER_HOME", "/nonexistent"), "license.txt"),
                 os.path.expanduser("~/fs_license.txt")):
        if cand and os.path.isfile(cand):
            return cand
    return None


def add_fsl_arguments(parser) -> None:
    g = parser.add_argument_group("distortion / eddy-current correction (FSL, Synb0-DisCo)")
    g.add_argument("--correction", choices=["auto", "topup", "synb0", "eddy", "dipy"],
                   default="auto",
                   help="'auto' (default): topup+eddy when reverse-PE fieldmaps exist, otherwise "
                        "Synb0-DisCo+topup+eddy. 'topup', 'synb0': force a route. 'eddy': eddy "
                        "without susceptibility correction. 'dipy': the DIPY-only route "
                        "(--motion / --sdc).")
    g.add_argument("--fsl-dir", type=Path, default=None,
                   help="FSL installation for native runs (default: $FSLDIR, else ~/fsl).")
    g.add_argument("--fsl-container", default=None,
                   help="Run FSL inside this container instead: a Docker image name or an "
                        "Apptainer/Singularity .sif (e.g. the compneuro-dwiproc image).")
    g.add_argument("--synb0-container", default=None,
                   help=f"Synb0-DisCo container: Docker image (default {SYNB0_DOCKER_IMAGE}) or "
                        "an Apptainer/Singularity .sif (required with apptainer).")
    g.add_argument("--container-engine", choices=["docker", "apptainer", "singularity"],
                   default=None,
                   help="Container engine (default: docker on macOS, else apptainer if found).")
    g.add_argument("--fs-license", type=Path, default=None,
                   help="FreeSurfer license (needed by Synb0-DisCo). Default: $FS_LICENSE, "
                        "$FREESURFER_HOME/license.txt or ~/fs_license.txt.")
    g.add_argument("--synb0-t1", choices=["raw", "stripped"], default="raw",
                   help="T1 given to Synb0-DisCo: 'raw' BIDS T1w (default; Synb0 strips it) or "
                        "'stripped' = the skull-stripped T1 of the anatomy source "
                        "(FreeSurfer brainmask / BET), passed with --stripped.")
    g.add_argument("--topup-config", default=None,
                   help="topup --config (default: b02b0.cnf for fieldmaps; Synb0-DisCo's "
                        "schedule for synthetic b0).")
    g.add_argument("--no-align-seepi", action="store_true",
                   help="Do not put the first DWI b0 at the start of the topup input "
                        "(default: do, like dwifslpreproc -align_seepi).")
    g.add_argument("--eddy-binary", default=None,
                   help="eddy executable (default: eddy_cpu with --n-jobs threads; with "
                        "--container-gpu, FSL's 'eddy' wrapper, which runs eddy_cuda).")
    g.add_argument("--container-gpu", action="store_true",
                   help="Give the FSL container GPU access (--nv / --gpus all) and run eddy "
                        "on the GPU.")
    g.add_argument("--eddy-s2v", action="store_true",
                   help="Slice-to-volume (intra-volume) motion correction: --mporder=6 with the "
                        "slice timing from the DWI JSON (recommended for multiband data; slow "
                        "on CPU).")
    g.add_argument("--eddy-extra", default="",
                   help="Extra eddy options as one string; write it with '=', e.g. "
                        "--eddy-extra=\"--slm=linear --estimate_move_by_susceptibility\".")


# ---------------------------------------------------------------------------
# Running external programs
# ---------------------------------------------------------------------------


class ToolError(RuntimeError):
    pass


def _run(cmd: list[str], log_file: Path, logger, cwd: Path | None = None, env=None) -> None:
    """Run a command, write its output to ``log_file``, raise ToolError on failure."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"   $ {shlex.join(cmd)}")
    with open(log_file, "w") as fh:
        fh.write(shlex.join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=cwd, env=env)
    if proc.returncode != 0:
        tail = "".join(log_file.read_text().splitlines(keepends=True)[-15:])
        raise ToolError(f"{cmd[0]} failed (exit {proc.returncode}); see {log_file}\n{tail}")


class FSL:
    """Calls FSL commands either natively or inside a container.

    In container mode every path is bound at the same location inside the
    container, so commands can use the host paths unchanged.
    """

    def __init__(self, args, work_dir: Path, logger):
        self.logger = logger
        self.work_dir = Path(work_dir)
        self.container = args.fsl_container
        self.engine = args.container_engine or default_engine()
        self.gpu = args.container_gpu
        self.env = os.environ.copy()
        self.env["FSLOUTPUTTYPE"] = "NIFTI_GZ"
        if self.container is None:
            fsl_dir = args.fsl_dir or (Path(os.environ["FSLDIR"]) if os.environ.get("FSLDIR") else None)
            if fsl_dir is None and (Path.home() / "fsl" / "bin").is_dir():
                fsl_dir = Path.home() / "fsl"          # default location of the FSL installer
            if fsl_dir is None or not (Path(fsl_dir) / "bin").is_dir():
                raise ToolError("FSL not found: set $FSLDIR, or use --fsl-dir or --fsl-container")
            self.env["FSLDIR"] = str(fsl_dir)
            self.env["PATH"] = f"{fsl_dir}/bin:{self.env.get('PATH', '')}"
            self.description = f"native FSL in {fsl_dir}"
            version = Path(fsl_dir) / "etc" / "fslversion"
            if version.is_file():
                self.description += f" (version {version.read_text().strip()})"
        else:
            self.description = f"FSL in {self.engine} container {self.container}"

    def _wrap(self, cmd: list[str], binds: list[Path]) -> list[str]:
        if self.container is None:
            exe = shutil.which(cmd[0], path=self.env["PATH"])
            if exe is None:
                raise ToolError(f"FSL command not found: {cmd[0]} ({self.description})")
            return [exe] + cmd[1:]
        binds = sorted({str(Path(b).resolve()) for b in binds})
        if self.engine == "docker":
            out = ["docker", "run", "--rm", "--entrypoint", "", "-e", "FSLOUTPUTTYPE=NIFTI_GZ",
                   "-w", str(self.work_dir.resolve())]
            if hasattr(os, "getuid"):
                out += ["--user", f"{os.getuid()}:{os.getgid()}"]
            if self.gpu:
                out += ["--gpus", "all"]
            for b in binds:
                out += ["-v", f"{b}:{b}"]
            return out + [self.container] + cmd
        out = [self.engine, "exec", "--cleanenv", "--env", "FSLOUTPUTTYPE=NIFTI_GZ",
               "--pwd", str(self.work_dir.resolve())]
        if self.gpu:
            out.append("--nv")
        for b in binds:
            out += ["-B", f"{b}:{b}"]
        return out + [self.container] + cmd

    def __call__(self, cmd: list[str], log_name: str, binds: list[Path] | None = None) -> None:
        full = self._wrap(cmd, [self.work_dir] + list(binds or []))
        _run(full, self.work_dir / "logs" / f"{log_name}.log", self.logger,
             cwd=self.work_dir, env=self.env)

    def available(self, name: str) -> bool:
        if self.container is None:
            return shutil.which(name, path=self.env["PATH"]) is not None
        return True        # cannot cheaply check inside a container; failures are reported


def run_synb0(b0_file: Path, t1_file: Path, acqparams: Path, stripped: bool, out_dir: Path,
              args, logger) -> Path:
    """Run Synb0-DisCo (--notopup) and return the synthetic undistorted b0 (b0_u)."""
    engine = args.container_engine or default_engine()
    image = args.synb0_container
    if image is None:
        if engine != "docker":
            raise ToolError("--synb0-container (a .sif file) is required with "
                            f"--container-engine {engine}")
        image = SYNB0_DOCKER_IMAGE
    lic = args.fs_license or default_fs_license()
    if lic is None or not Path(lic).is_file():
        raise ToolError("FreeSurfer license not found (needed by Synb0-DisCo): use --fs-license")
    inp, outp = out_dir / "INPUTS", out_dir / "OUTPUTS"
    inp.mkdir(parents=True, exist_ok=True)
    outp.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(b0_file, inp / "b0.nii.gz")
    shutil.copyfile(t1_file, inp / "T1.nii.gz")
    shutil.copyfile(acqparams, inp / "acqparams.txt")
    flags = ["--notopup"] + (["--stripped"] if stripped else [])
    lic = str(Path(lic).resolve())
    if engine == "docker":
        cmd = ["docker", "run", "--rm",
               "-v", f"{inp.resolve()}:/INPUTS/", "-v", f"{outp.resolve()}:/OUTPUTS/",
               "-v", f"{lic}:/extra/freesurfer/license.txt"]
        if hasattr(os, "getuid"):
            cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        if platform.machine() in ("arm64", "aarch64"):
            cmd += ["--platform", "linux/amd64"]          # the image is x86-64 only
        cmd += [image] + flags
    else:
        cmd = [engine, "run", "-e", "-B", f"{inp.resolve()}:/INPUTS", "-B",
               f"{outp.resolve()}:/OUTPUTS", "-B", f"{lic}:/extra/freesurfer/license.txt",
               image] + flags
    _run(cmd, out_dir / "synb0.log", logger)
    b0_u = outp / "b0_u.nii.gz"
    if not b0_u.is_file():
        raise ToolError(f"Synb0-DisCo finished without {b0_u}; see {out_dir / 'synb0.log'}")
    return b0_u


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def acq_line(pe_dir: str, readout: float, affine: np.ndarray) -> str:
    """One topup/eddy acqparams line, e.g. 'j-' -> '0 -1 0 0.0959'.

    FSL's vector is in voxel axes, but with the first axis flipped when the
    image is stored with a positive determinant (FSL's radiological view).
    """
    axis = {"i": 0, "j": 1, "k": 2}[pe_dir[0]]
    vec = [0, 0, 0]
    vec[axis] = -1 if pe_dir.endswith("-") else 1
    if axis == 0 and np.linalg.det(affine[:3, :3]) > 0:
        vec[0] = -vec[0]
    return f"{vec[0]} {vec[1]} {vec[2]} {readout:.6g}"


def pad_axes(shape) -> list[int]:
    """Axes with an odd size: topup's default 2x subsampling needs even sizes."""
    return [ax for ax in range(3) if shape[ax] % 2 == 1]


def pad(arr: np.ndarray, axes: list[int]) -> np.ndarray:
    """Append a copy of the last slice along each axis in ``axes`` (as
    dwifslpreproc does). The affine does not change (slices are added at the end)."""
    for ax in axes:
        last = np.take(arr, [arr.shape[ax] - 1], axis=ax)
        arr = np.concatenate([arr, last], axis=ax)
    return arr


def crop(arr: np.ndarray, shape3) -> np.ndarray:
    return arr[: shape3[0], : shape3[1], : shape3[2], ...]


def _save(path: Path, data, affine, dtype=np.float32) -> Path:
    return dc.save_nifti(path, data, affine, dtype=dtype)


def _done(path: Path, args) -> bool:
    return path.is_file() and not args.overwrite


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def choose_route(args, fieldmaps: list[dict]) -> str:
    if args.correction != "auto":
        return args.correction
    return "topup" if fieldmaps else "synb0"


def correct(data, bvals, bvecs_vox, affine, meta, run, participant, route, fieldmaps,
            fmap_how, out_dir, args, logger, timings=None):
    """Run the FSL route on ``data`` (denoised DWI, 4D float32, original grid).

    Returns (corrected data, rotated b-vectors in voxel convention, qc dict).
    """
    work = Path(out_dir) / "fsl_work"
    work.mkdir(parents=True, exist_ok=True)
    fsl = FSL(args, work, logger)
    shape3 = data.shape[:3]
    pax = pad_axes(shape3)
    pe = dc.phase_encoding(meta)
    if pe["pe_dir"] is None or pe["readout_time"] is None:
        raise ToolError("PhaseEncodingDirection and TotalReadoutTime are needed in the DWI JSON")
    b0_mask = bvals <= args.b0_threshold
    if not b0_mask.any():
        raise ToolError("no b0 volume in the DWI series")
    dwi_line = acq_line(pe["pe_dir"], pe["readout_time"], affine)
    if not 0.01 <= pe["readout_time"] <= 0.2:
        raise ToolError(f"TotalReadoutTime {pe['readout_time']} s is outside the 0.01-0.2 s "
                        "range accepted by eddy; check the DWI JSON")
    qc = dict(route=route, fsl=fsl.description, padded_axes=pax, dwi_acqp=dwi_line)
    logger.info(f"   correction route: {route} ({fsl.description})"
                + (f"; padding odd axes {pax} by one slice" if pax else ""))

    eddy_base = work / "eddy"
    eddy_out = work / "eddy.nii.gz"
    rot_bvecs = work / "eddy.eddy_rotated_bvecs"
    topup_base = work / "topup"
    use_topup = route in ("topup", "synb0")

    if args.eddy_s2v:
        if "SliceTiming" not in meta:
            raise ToolError("--eddy-s2v needs SliceTiming in the DWI JSON")
        if dc.slice_axis(meta) in pax:
            raise ToolError("--eddy-s2v is not possible when the slice axis has an odd "
                            "number of slices (it is padded for topup)")
        dc.write_json(work / "dwi.json", meta)

    if not (_done(eddy_out, args) and rot_bvecs.is_file()):
        # ---- eddy inputs (padded DWI, FSL gradients, acqp/index) ----------
        _save(work / "dwi_in.nii.gz", pad(data, pax), affine)
        dc.write_fsl_gradients(bvals, bvecs_vox, affine, work / "bvals", work / "bvecs")
        np.savetxt(work / "index.txt", np.ones((1, data.shape[-1]), dtype=int), fmt="%d")
        acq_lines = [dwi_line]

        # ---- topup inputs ------------------------------------------------
        if route == "topup":
            if not fieldmaps:
                raise ToolError("--correction topup but no fieldmaps were found")
            vols, lines = [], []
            if not args.no_align_seepi:
                vols.append(data[..., np.where(b0_mask)[0][0]])
                lines.append(dwi_line)
            for fm in fieldmaps:
                img = nib.load(str(fm["nii"]))
                if img.shape[:3] != shape3 or not np.allclose(img.affine, affine, atol=1e-3):
                    raise ToolError(f"fieldmap {fm['nii'].name} is not on the DWI voxel grid "
                                    f"({img.shape[:3]} vs {shape3}); resample it first")
                fpe = dc.phase_encoding(fm["meta"])
                if fpe["pe_dir"] is None or fpe["readout_time"] is None:
                    raise ToolError(f"{fm['nii'].name}: PhaseEncodingDirection/TotalReadoutTime missing")
                arr = np.asarray(img.dataobj, dtype=np.float32)
                arr = arr[..., None] if arr.ndim == 3 else arr
                for k in range(arr.shape[-1]):
                    vols.append(arr[..., k])
                    lines.append(acq_line(fpe["pe_dir"], fpe["readout_time"], affine))
            pes = set()
            for ln in lines:
                vec = [int(float(v)) for v in ln.split()[:3]]
                ax = int(np.flatnonzero(vec)[0])
                pes.add((ax, vec[ax]))
            if len({a for a, _ in pes}) != 1 or len({sg for _, sg in pes}) < 2:
                raise ToolError("topup needs opposite phase encodings along one axis; "
                                f"acqparams would be {lines}")
            topup_lines = lines
            te_dwi = meta.get("EchoTime")
            te_fm = {fm["meta"]["EchoTime"] for fm in fieldmaps if fm["meta"].get("EchoTime")}
            if not args.no_align_seepi and te_dwi and any(
                    abs(t - te_dwi) / te_dwi > 0.1 for t in te_fm):
                logger.warning(f"   DWI b0 (TE {te_dwi}) and fieldmaps (TE {sorted(te_fm)}) have "
                               "different contrast; the DWI b0 is still added to topup to align "
                               "the field with the DWI (as dwifslpreproc -align_seepi). Use "
                               "--no-align-seepi to estimate the field from the fieldmaps only.")
            qc["fieldmaps"] = [str(fm["nii"]) for fm in fieldmaps]
            qc["fieldmaps_found_by"] = fmap_how
            qc["topup_includes_dwi_b0"] = not args.no_align_seepi
            config = args.topup_config or "b02b0.cnf"
        elif route == "synb0":
            synb0_dir = work / "synb0"
            b0u_file = synb0_dir / "OUTPUTS" / "b0_u.nii.gz"
            b0_mean = data[..., b0_mask].mean(axis=-1)
            syn_acq = synb0_dir / "acqparams.txt"
            synb0_dir.mkdir(parents=True, exist_ok=True)
            syn_acq.write_text(f"{dwi_line}\n{' '.join(dwi_line.split()[:3])} 0\n")
            if _done(b0u_file, args):
                logger.info(f"   reusing {b0u_file}")
            else:
                if args.synb0_t1 == "stripped":
                    anat = dc.load_anatomy(participant, args.anat_source, args.fs_dir,
                                           args.preproc_dir)
                    t1_file = _save(synb0_dir / "T1_stripped.nii.gz", anat.t1_brain, anat.affine)
                else:
                    t1_file = dc.find_t1w(args.bids_dir, participant, run["session"])
                    if t1_file is None:
                        raise ToolError("no T1w image found for Synb0-DisCo")
                qc["synb0_t1"] = str(t1_file)
                b0_file = _save(synb0_dir / "b0_mean.nii.gz", b0_mean, affine)
                with dc.timed(logger, "Synb0-DisCo (synthetic undistorted b0)", timings):
                    run_synb0(b0_file, t1_file, syn_acq, args.synb0_t1 == "stripped",
                              synb0_dir, args, logger)
            b0_u_img = nib.load(str(b0u_file))
            if b0_u_img.shape[:3] != shape3:
                raise ToolError(f"Synb0 b0_u has shape {b0_u_img.shape} (DWI {shape3})")
            b0_u = np.asarray(b0_u_img.dataobj, dtype=np.float32)
            # Synb0-DisCo smooths the distorted b0 (fslmaths -s 1.15, sigma in mm)
            b0_s = ndimage.gaussian_filter(b0_mean, 1.15 / dc.voxel_sizes(affine))
            vols = [b0_s, b0_u]
            topup_lines = [dwi_line, f"{' '.join(dwi_line.split()[:3])} 0"]
            config = args.topup_config or str(work / "synb0_topup.cnf")
            if args.topup_config is None:
                (work / "synb0_topup.cnf").write_text(SYNB0_TOPUP_CONFIG)
        # ---- topup ---------------------------------------------------------
        if use_topup:
            if _done(work / "topup_fieldcoef.nii.gz", args):
                logger.info("   reusing topup results")
            else:
                _save(work / "topup_in.nii.gz", pad(np.stack(vols, axis=-1), pax), affine)
                (work / "topup_acqparams.txt").write_text("\n".join(topup_lines) + "\n")
                with dc.timed(logger, f"topup ({len(vols)} volumes)", timings):
                    fsl(["topup", f"--imain={work / 'topup_in.nii.gz'}",
                         f"--datain={work / 'topup_acqparams.txt'}", f"--config={config}",
                         f"--out={topup_base}", f"--iout={work / 'topup_iout.nii.gz'}",
                         f"--fout={work / 'topup_field_hz.nii.gz'}", "--verbose"], "topup")
            # eddy gets topup's acqparams lines, minus any with a readout time
            # eddy rejects (it only accepts 0.01-0.2 s): the Synb0 synthetic b0
            # has readout 0 for topup, but no DWI volume uses that line. The
            # DWI volumes point (index) to the line describing the DWI.
            acq_lines = [ln for ln in (work / "topup_acqparams.txt").read_text().split("\n")
                         if ln.strip() and 0.01 <= float(ln.split()[3]) <= 0.2]
            norm = [" ".join(f"{float(v):g}" for v in ln.split()) for ln in acq_lines]
            key = " ".join(f"{float(v):g}" for v in dwi_line.split())
            if key in norm:
                idx = norm.index(key) + 1
            else:
                acq_lines.append(dwi_line)
                idx = len(acq_lines)
            np.savetxt(work / "index.txt", np.full((1, data.shape[-1]), idx, dtype=int), fmt="%d")
        (work / "acqparams.txt").write_text("\n".join(acq_lines) + "\n")

        # ---- brain mask for eddy (BET on the corrected mean b0) --------------
        if use_topup:
            ref = np.asarray(nib.load(str(work / "topup_iout.nii.gz")).dataobj,
                             dtype=np.float32).mean(axis=-1)
        else:
            ref = pad(data[..., b0_mask].mean(axis=-1), pax)
        _save(work / "b0_ref.nii.gz", ref, affine)
        fsl(["bet", str(work / "b0_ref.nii.gz"), str(work / "b0_brain"), "-m", "-f", "0.2"], "bet")

        # ---- eddy ------------------------------------------------------------
        # with a GPU, FSL's 'eddy' wrapper picks the eddy_cudaX.Y matching the driver
        eddy_bin = args.eddy_binary or ("eddy" if args.container_gpu else "eddy_cpu")
        cmd = [eddy_bin, f"--imain={work / 'dwi_in.nii.gz'}", f"--mask={work / 'b0_brain_mask.nii.gz'}",
               f"--acqp={work / 'acqparams.txt'}", f"--index={work / 'index.txt'}",
               f"--bvecs={work / 'bvecs'}", f"--bvals={work / 'bvals'}",
               f"--out={eddy_base}", "--repol", "--ol_nstd=4", "--cnr_maps", "--verbose"]
        if use_topup:
            cmd.append(f"--topup={topup_base}")
        if Path(eddy_bin).name == "eddy_cpu":
            cmd.append(f"--nthr={args.n_jobs}")          # CPU threads (eddy_cpu only)
        if args.eddy_s2v:
            cmd += ["--mporder=6", "--s2v_niter=5", "--s2v_lambda=1", "--s2v_interp=trilinear",
                    f"--json={work / 'dwi.json'}"]
        cmd += shlex.split(args.eddy_extra)
        with dc.timed(logger, f"eddy ({eddy_bin})", timings):
            fsl(cmd, "eddy")
        qc["eddy_command"] = shlex.join(cmd)
    else:
        logger.info("   reusing eddy results")

    # ---- results back to the original grid ------------------------------------
    out = np.asarray(nib.load(str(eddy_out)).dataobj, dtype=np.float32)
    out = np.clip(crop(out, shape3), 0, None)
    _, new_bvecs = dc.load_fsl_gradients(work / "bvals", rot_bvecs, affine)

    # ---- QC -------------------------------------------------------------------
    eqc = {}
    mrms = work / "eddy.eddy_movement_rms"
    if mrms.is_file():
        mv = np.loadtxt(mrms, ndmin=2)
        eqc.update(mean_rms_abs_mm=float(mv[:, 0].mean()), max_rms_abs_mm=float(mv[:, 0].max()),
                   mean_rms_rel_mm=float(mv[:, 1].mean()), max_rms_rel_mm=float(mv[:, 1].max()))
    omap = work / "eddy.eddy_outlier_map"
    if omap.is_file():
        om = np.loadtxt(omap, skiprows=1, ndmin=2)
        eqc.update(outlier_slices=int(om.sum()),
                   outlier_slices_percent=float(100 * om.mean()),
                   volumes_with_outliers=int((om.sum(axis=1) > 0).sum()))
    if use_topup and (work / "topup_field_hz.nii.gz").is_file():
        fld = crop(np.asarray(nib.load(str(work / "topup_field_hz.nii.gz")).dataobj), shape3)
        brain = crop(np.asarray(nib.load(str(work / "b0_brain_mask.nii.gz")).dataobj) > 0, shape3)
        disp = fld * pe["readout_time"] * float(dc.voxel_sizes(affine)[pe["axis"]])
        _save(out_dir / "qc_data" / "topup_field_hz.nii.gz", fld, affine)
        _save(out_dir / "qc_data" / "topup_displacement_pe_mm.nii.gz", disp, affine)
        if brain.any():
            eqc["field_hz_p1_p99_in_brain"] = [float(np.percentile(fld[brain], 1)),
                                               float(np.percentile(fld[brain], 99))]
            eqc["pe_displacement_mm_p1_p99_in_brain"] = [float(np.percentile(disp[brain], 1)),
                                                         float(np.percentile(disp[brain], 99))]
    qdir = out_dir / "qc_data" / "eddy"
    qdir.mkdir(parents=True, exist_ok=True)
    for f in work.glob("eddy.eddy_*"):
        if f.is_file() and not f.name.endswith((".nii", ".nii.gz")):
            shutil.copyfile(f, qdir / f.name)
    cnr = work / "eddy.eddy_cnr_maps.nii.gz"
    if cnr.is_file():
        _save(qdir / "eddy_cnr_maps.nii.gz",
              crop(np.asarray(nib.load(str(cnr)).dataobj, dtype=np.float32), shape3), affine)
    qc["eddy"] = eqc
    if eqc:
        logger.info(f"   eddy: mean movement {eqc.get('mean_rms_abs_mm', float('nan')):.2f} mm, "
                    f"outlier slices {eqc.get('outlier_slices', 'n/a')}"
                    + (f"; susceptibility displacement p1/p99 "
                       f"{eqc['pe_displacement_mm_p1_p99_in_brain'][0]:.1f}/"
                       f"{eqc['pe_displacement_mm_p1_p99_in_brain'][1]:.1f} mm"
                       if "pe_displacement_mm_p1_p99_in_brain" in eqc else ""))

    # ---- eddy_quad report (optional) -----------------------------------------
    quad_dir = out_dir / "qc_data" / "eddy_quad"
    if args.overwrite and quad_dir.exists():
        shutil.rmtree(quad_dir, ignore_errors=True)
    if not quad_dir.exists() and fsl.available("eddy_quad"):
        qcmd = ["eddy_quad", str(eddy_base), "-idx", str(work / "index.txt"),
                "-par", str(work / "acqparams.txt"), "-m", str(work / "b0_brain_mask.nii.gz"),
                "-b", str(work / "bvals"), "-g", str(work / "bvecs"), "-o", str(quad_dir)]
        if use_topup:
            qcmd += ["-f", str(work / "topup_field_hz.nii.gz")]
        if args.eddy_s2v:
            qcmd += ["-j", str(work / "dwi.json")]
        try:
            fsl(qcmd, "eddy_quad", binds=[out_dir / "qc_data"])
        except ToolError as e:
            logger.warning(f"   eddy_quad failed (QC report only): {str(e).splitlines()[0]}")
    return out, new_bvecs, qc
