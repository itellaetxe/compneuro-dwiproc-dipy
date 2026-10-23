#!/usr/bin/env python3
"""
dwi_preproc.py
==============

DWI preprocessing with DIPY 1.12 -- the DIPY counterpart of
``compneuro-dwiproc/src/dwi_preproc.sh`` (MRtrix3 + FSL).

Step-by-step correspondence with the original script
----------------------------------------------------

=====  ======================================  ===========================================
Step   Original (MRtrix3 / FSL)                This script (DIPY 1.12)
=====  ======================================  ===========================================
1      dwidenoise (MP-PCA)                     ``dipy.denoise.localpca.mppca`` (MP-PCA, same
                                               method) or Patch2Self (``--denoise``)
2      (not done)                              Gibbs-ringing removal
                                               ``dipy.denoise.gibbs.gibbs_removal`` (same
                                               method as mrdegibbs; skipped automatically for
                                               partial-Fourier data, see ``--gibbs``)
3-4    dwifslpreproc: FSL topup (when a       Same FSL tools, called from Python (dwi_fsl.py):
       reverse-PE fieldmap exists) + eddy      * fieldmaps found -> topup on [first DWI b0,
       (--repol --ol_nstd=4 --cnr_maps)          fieldmaps] + eddy (--repol --ol_nstd=4
                                                 --cnr_maps; optional slice-to-volume)
                                               * no fieldmaps -> Synb0-DisCo synthetic
                                                 undistorted b0 + topup + eddy
                                               * --correction dipy: the DIPY-only route
                                                 (per-volume affine registration + b-vector
                                                 rotation; optional SyN-SDC, --sdc syn)
5      dwiextract -bzero + fslmaths -Tmean     mean of the b0 volumes
6      dwi2mask                                ``dipy.segment.mask.median_otsu``
7      dwi2tensor + tensor2metric              ``dipy.reconst.dti.TensorModel`` (WLS)
=====  ======================================  ===========================================

Outputs (same names as the original, in ``<output-dir>/sub-X[/ses-Y]``)::

    dwi_den.nii.gz                denoised DWI
    dwi_degibbs.nii.gz            after Gibbs removal (only if it was applied)
    dwi_clean.nii.gz              fully preprocessed DWI
    rotated.bval / rotated.bvec   gradients after motion correction (FSL format)
    dwi_bzero.nii.gz              mean b0
    dwi_mask.nii.gz               brain mask
    dwi_tensor.nii.gz             tensor, 6 volumes in DIPY order (Dxx,Dxy,Dyy,Dxz,Dyz,Dzz),
                                  in the b-vector (voxel-axis) frame
    dwi_FA / dwi_MD / dwi_AD / dwi_RD .nii.gz
    dwi_directions.nii.gz         FA-modulated principal eigenvector in scanner (world) axes,
                                  like ``tensor2metric -vector``
    fsl_work/                     topup/eddy/Synb0 intermediate files and command logs
    qc_data/                      noise map, eddy QC (movement, outliers, CNR, eddy_quad),
                                  topup field (Hz) and displacement (mm), QC json

Usage (one participant; normally called through dwi_launcher.py)::

    python src/dwi_preproc.py --bids-dir ~/Desktop/sub-6175 \
        --output-dir ~/Desktop/sub-6175/derivatives/dwiproc_dipy \
        --fs-dir ~/Desktop/sub-6175/derivatives/Recon
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dwiproc_common as dc  # noqa: E402
import dwi_fsl  # noqa: E402

PREPROC_DONE = "qc_data/preproc_desc.json"   # written last: marks a finished run


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def add_preproc_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("preprocessing (dwi_preproc.py)")
    g.add_argument("--denoise", choices=["mppca", "patch2self", "none"], default="mppca",
                   help="Denoising: 'mppca' = Marchenko-Pastur PCA, the method of MRtrix "
                        "dwidenoise (default); 'patch2self' = self-supervised regression; 'none'.")
    g.add_argument("--mppca-patch-radius", type=int, default=2,
                   help="MP-PCA patch radius in voxels (2 -> 5x5x5, as dwidenoise's default).")
    g.add_argument("--denoise-chunk-slices", type=int, default=12,
                   help="Process MP-PCA in slabs of this many slices to bound memory "
                        "(identical result; 0 = whole volume at once). Default 12.")
    g.add_argument("--gibbs", choices=["auto", "on", "off"], default="auto",
                   help="Gibbs-ringing removal. 'auto' (default) applies it unless the JSON "
                        "sidecar reports partial Fourier < 1, where the method is not valid.")
    g.add_argument("--motion", choices=["affine", "rigid", "none"], default="affine",
                   help="Motion/eddy-current correction: 'affine' (12 dof per volume, default; "
                        "captures linear eddy-current distortions), 'rigid' (motion only), 'none'.")
    g.add_argument("--motion-level-iters", type=int, nargs=3, default=[1000, 500, 100],
                   help="Optimizer iterations per resolution level (coarse..fine).")
    g.add_argument("--sdc", choices=["none", "syn"], default="none",
                   help="Susceptibility distortion correction. 'none' (default). 'syn' = "
                        "fieldmap-less: non-linear b0->T1 registration restricted to the "
                        "phase-encoding axis (needs the anatomy options and the JSON sidecar).")
    g.add_argument("--sdc-level-iters", type=int, nargs=3, default=[100, 50, 25],
                   help="SyN iterations per level for --sdc syn (coarse..fine).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    dc.add_common_arguments(p)
    add_preproc_arguments(p)
    dwi_fsl.add_fsl_arguments(p)
    return p


def preproc_status(out_dir: Path, run: dict, participant: str, args) -> tuple[bool, str]:
    """Is the preprocessing in ``out_dir`` finished and done with the correction
    route that would be chosen now? Returns (up_to_date, explanation).

    A run preprocessed with another route (e.g. the DIPY-only route of an earlier
    version, before fieldmap/Synb0 support) is redone; finished intermediate steps
    such as denoising are reused.
    """
    done = Path(out_dir) / PREPROC_DONE
    if not done.is_file():
        return False, "not preprocessed yet"
    if args.overwrite:
        return False, "--overwrite"
    old = dc.read_json(done).get("correction", {}).get("route", "dipy")
    fieldmaps, _ = dc.find_fieldmaps(args.bids_dir, participant, run)
    new = dwi_fsl.choose_route(args, fieldmaps)
    if old != new:
        return False, f"preprocessed with correction route '{old}', now '{new}'"
    return True, f"already preprocessed (route '{old}')"


# ---------------------------------------------------------------------------
# Step 1 helper: MP-PCA in slabs (same result, bounded memory)
# ---------------------------------------------------------------------------


def mppca_chunked(data, mask, patch_radius, chunk, logger):
    """MP-PCA denoising processed in slabs along the 3rd axis.

    DIPY's ``mppca`` denoises every patch (radius r) and averages the
    overlapping patch estimates, so the output at a voxel depends only on data
    within 2r voxels of it. Each slab is therefore padded with 2r slices on
    both sides and only its core is kept: the result is identical to a single
    call (checked numerically in testing) but peak memory is set by the slab
    size, not the whole series. ``chunk <= 0`` processes the volume in one go.
    """
    from dipy.denoise.localpca import mppca
    nz = data.shape[2]
    if chunk <= 0 or chunk >= nz:
        with np.errstate(invalid="ignore", divide="ignore"):
            den, sigma = mppca(data, mask=mask, patch_radius=patch_radius, pca_method="svd",
                               return_sigma=True, out_dtype=np.float32, suppress_warning=True)
        return np.nan_to_num(den), np.nan_to_num(sigma)
    pad = 2 * patch_radius
    den = np.zeros(data.shape, dtype=np.float32)
    sigma = np.zeros(data.shape[:3], dtype=np.float32)
    for z0 in range(0, nz, chunk):
        z1 = min(z0 + chunk, nz)
        a, b = max(0, z0 - pad), min(nz, z1 + pad)
        if not mask[:, :, z0:z1].any():
            continue
        if b - a < 2 * patch_radius + 1:          # slab too thin for one patch
            a, b = max(0, b - (2 * patch_radius + 1)), min(nz, a + 2 * patch_radius + 1)
        with np.errstate(invalid="ignore", divide="ignore"):   # 0/0 outside the mask
            d, s = mppca(np.ascontiguousarray(data[:, :, a:b]), mask=mask[:, :, a:b],
                         patch_radius=patch_radius, pca_method="svd", return_sigma=True,
                         out_dtype=np.float32, suppress_warning=True)
        den[:, :, z0:z1] = d[:, :, z0 - a:z1 - a]
        sigma[:, :, z0:z1] = s[:, :, z0 - a:z1 - a]
        logger.info(f"     MP-PCA slices {z0}-{z1 - 1} of {nz}")
    return np.nan_to_num(den), np.nan_to_num(sigma)


# ---------------------------------------------------------------------------
# Step 3 helper: register one volume (runs in a worker process)
# ---------------------------------------------------------------------------


def _register_volume(args):
    """Rigid, then (optionally) affine registration of one volume to the reference b0.

    Returns (index, resampled volume, 4x4 affine, used_rigid_fallback). The
    affine is DIPY's ``AffineMap.affine``: it maps reference-space world
    coordinates to the world coordinates of the moving volume.

    Safeguard: eddy-current distortions are small (scalings of a few %). If
    the 12-dof solution scales/shears any axis by more than ``max_stretch``
    (default 10 %), it is considered a failed fit and the rigid solution is
    kept (reported in motion_parameters.tsv and in the log).
    """
    from dipy.align import affine_registration
    from dipy.align.imaffine import AffineMap
    from scipy.linalg import polar
    idx, moving, static, img_affine, static_mask, pipeline, level_iters, max_stretch = args
    common = dict(moving_affine=img_affine, static_affine=img_affine, metric="MI",
                  level_iters=list(level_iters), sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1],
                  static_mask=static_mask)
    _, aff = affine_registration(moving, static, pipeline=["translation", "rigid"],
                                 starting_affine=np.eye(4), **common)
    fallback = False
    if "affine" in pipeline:
        _, aff_a = affine_registration(moving, static, pipeline=["affine"],
                                       starting_affine=aff, **common)
        _, stretch = polar(aff_a[:3, :3])
        if np.max(np.abs(np.linalg.eigvalsh(stretch) - 1)) > max_stretch:
            fallback = True
        else:
            aff = aff_a
    amap = AffineMap(aff, domain_grid_shape=static.shape, domain_grid2world=img_affine,
                     codomain_grid_shape=moving.shape, codomain_grid2world=img_affine)
    return idx, amap.transform(moving).astype(np.float32), aff, fallback


def _rms_displacement(aff_a: np.ndarray, aff_b: np.ndarray, points: np.ndarray) -> float:
    """RMS distance (mm) between two affine transforms over a set of world points."""
    pa = points @ aff_a[:3, :3].T + aff_a[:3, 3]
    pb = points @ aff_b[:3, :3].T + aff_b[:3, 3]
    return float(np.sqrt(np.mean(np.sum((pa - pb) ** 2, axis=1))))


def motion_correct(data, gtab, img_affine, mode, level_iters, n_jobs, logger, max_stretch=0.1):
    """Step 3: per-volume registration to the (mean) b0, and b-vector rotation.

    1. Reference: if there are several b0 volumes, each is rigidly registered
       to the first and the registered b0s are averaged (as DIPY's
       ``register_dwi_series``); with one b0, that b0 is the reference.
    2. Every other volume is registered to the reference ('affine': 12 dof,
       which also absorbs linear eddy-current distortions; 'rigid': 6 dof),
       using mutual information inside a dilated brain mask of the reference.
    3. The rotation part of each transform is applied to its b-vector.
    """
    from dipy.segment.mask import median_otsu

    n_vol = data.shape[-1]
    b0_idx = np.where(gtab.b0s_mask)[0]
    if b0_idx.size == 0:
        raise ValueError("No b0 volume found (check --b0-threshold)")
    affines = np.repeat(np.eye(4)[..., None], n_vol, axis=-1)
    out = np.empty_like(data)
    pipeline = ["translation", "rigid"] + (["affine"] if mode == "affine" else [])

    # --- 1. reference b0 ----------------------------------------------------
    ref = data[..., b0_idx[0]].astype(np.float32)
    _, ref_mask = median_otsu(ref, median_radius=4, numpass=4)
    ref_mask = ndimage.binary_dilation(ref_mask, iterations=2)
    reg_mask = ref_mask.astype(np.int32)          # DIPY's registration needs a numeric mask
    jobs = [(int(i), data[..., i], ref, img_affine, reg_mask, ["translation", "rigid"],
             level_iters, max_stretch) for i in b0_idx[1:]]
    out[..., b0_idx[0]] = ref
    if jobs:
        logger.info(f"   registering {len(jobs)} extra b0 volume(s) to the first b0 (rigid)")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for i, vol, aff, _ in ex.map(_register_volume, jobs):
                out[..., i] = vol
                affines[..., i] = aff
        ref = out[..., b0_idx].mean(axis=-1)

    # --- 2. every diffusion-weighted volume -> reference ---------------------
    dw_idx = np.where(~gtab.b0s_mask)[0]
    jobs = [(int(i), data[..., i], ref, img_affine, reg_mask, pipeline, level_iters, max_stretch)
            for i in dw_idx]
    fallbacks = np.zeros(n_vol, dtype=bool)
    logger.info(f"   registering {len(jobs)} DW volumes to the b0 reference "
                f"({'+'.join(pipeline)}, {n_jobs} processes)")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for k, (i, vol, aff, fb) in enumerate(ex.map(_register_volume, jobs), start=1):
            out[..., i] = vol
            affines[..., i] = aff
            fallbacks[i] = fb
            if k % 10 == 0 or k == len(jobs):
                logger.info(f"     {k}/{len(jobs)} volumes ({(time.time() - t0) / 60:.1f} min)")

    if fallbacks.any():
        logger.warning(f"   affine fit rejected (>{100 * max_stretch:.0f}% scaling) for volumes "
                       f"{np.where(fallbacks)[0].tolist()}: rigid solution kept")

    # --- 3. b-vector rotation -----------------------------------------------
    new_bvecs = dc.rotate_bvecs(gtab.bvecs, affines, img_affine, gtab.b0s_mask)

    # --- QC: displacement of brain voxels (absolute and volume-to-volume) ----
    ijk = np.argwhere(ref_mask)[:: max(1, int(ref_mask.sum() // 20000))]
    pts = ijk @ img_affine[:3, :3].T + img_affine[:3, 3]
    from scipy.linalg import polar
    from scipy.spatial.transform import Rotation
    rows = []
    for i in range(n_vol):
        a = affines[..., i]
        rot, stretch = polar(a[:3, :3])
        angles = Rotation.from_matrix(rot).as_euler("xyz", degrees=True)
        rows.append(dict(
            volume=i, bval=float(gtab.bvals[i]),
            tx=a[0, 3], ty=a[1, 3], tz=a[2, 3], rx_deg=angles[0], ry_deg=angles[1], rz_deg=angles[2],
            scale_x=stretch[0, 0], scale_y=stretch[1, 1], scale_z=stretch[2, 2],
            rms_abs_mm=_rms_displacement(a, np.eye(4), pts),
            rms_rel_mm=_rms_displacement(a, affines[..., i - 1], pts) if i > 0 else 0.0,
            rigid_fallback=int(fallbacks[i]),
        ))
    return out, new_bvecs, affines, ref_mask, rows


# ---------------------------------------------------------------------------
# Step 4 helper: fieldmap-less susceptibility distortion correction
# ---------------------------------------------------------------------------


def syn_sdc(data, gtab, img_affine, pe_axis, anat, level_iters, logger, qc_dir):
    """Fieldmap-less SDC (approximation of topup when no fieldmap exists).

    1. Rigid registration of the mean b0 to the skull-stripped T1 (mutual
       information) and resampling of the T1 onto the DWI grid.
    2. T1 contrast inverted inside the brain (CSF bright, WM dark), so that it
       resembles the T2-weighted b0 -- the trick used by fMRIPrep's SyN-SDC.
    3. Symmetric diffeomorphic (SyN) registration, cross-correlation metric,
       of the b0 (moving) to the inverted T1 (static), on the DWI grid.
    4. Susceptibility only displaces voxels along the phase-encoding axis, so
       the displacement field is projected onto that axis (DIPY's SyN cannot
       restrict the optimisation itself; this projection is the approximation).
    5. The projected warp is applied to every volume. No Jacobian intensity
       modulation is applied.

    Returns the corrected data and a QC dict (b0-T1 correlation before/after).
    """
    from dipy.align import affine_registration, syn_registration
    from dipy.align.imaffine import AffineMap

    b0 = data[..., gtab.b0s_mask].mean(axis=-1)

    # 1. rigid b0 -> T1, then T1 onto the DWI grid
    _, rigid = affine_registration(
        b0, anat.t1_brain, moving_affine=img_affine, static_affine=anat.affine,
        pipeline=["center_of_mass", "translation", "rigid"], metric="MI",
        level_iters=[1000, 500, 100], static_mask=anat.brain_mask.astype(np.int32))
    amap = AffineMap(rigid, domain_grid_shape=anat.t1_brain.shape, domain_grid2world=anat.affine,
                     codomain_grid_shape=b0.shape, codomain_grid2world=img_affine)
    t1_dwi = amap.transform_inverse(anat.t1_brain, interpolation="linear")
    mask_dwi = amap.transform_inverse(anat.brain_mask.astype(np.float32), interpolation="linear") > 0.5

    # 2. inverted, normalised T1 and normalised b0 (both 0 outside the brain)
    hi = np.percentile(t1_dwi[mask_dwi], 99)
    t1_inv = np.where(mask_dwi, np.clip(hi - t1_dwi, 0, None), 0).astype(np.float64)
    t1_inv /= max(np.percentile(t1_inv[mask_dwi], 99), 1e-6)
    b0n = np.where(mask_dwi, b0, 0).astype(np.float64)
    b0n /= max(np.percentile(b0n[mask_dwi], 99), 1e-6)

    # 3. SyN on the DWI grid (static = inverted T1, moving = b0)
    _, mapping = syn_registration(b0n, t1_inv, moving_affine=img_affine, static_affine=img_affine,
                                  metric="CC", level_iters=list(level_iters))

    # 4. keep only the component along the phase-encoding axis. DIPY stores
    #    displacements in world mm; syn_registration returns an inverted map
    #    (is_inverse=True), whose transform() uses the 'backward' field, so
    #    both fields are projected and the one actually applied is saved.
    u = dc.direction_cosines(img_affine)[:, pe_axis]
    for name in ("forward", "backward"):
        field = getattr(mapping, name)
        proj = np.tensordot(field, u, axes=([3], [0]))
        setattr(mapping, name, np.ascontiguousarray((proj[..., None] * u).astype(field.dtype)))
    applied = mapping.backward if mapping.is_inverse else mapping.forward
    disp_mm = np.tensordot(applied, u, axes=([3], [0]))
    dc.save_nifti(qc_dir / "sdc_displacement_pe_mm.nii.gz", disp_mm, img_affine)

    # 5. apply to every volume
    out = np.empty_like(data)
    for i in range(data.shape[-1]):
        out[..., i] = mapping.transform(data[..., i].astype(np.float64), interpolation="linear")

    def corr(a, b):
        return float(np.corrcoef(a[mask_dwi], b[mask_dwi])[0, 1])
    b0_after = out[..., gtab.b0s_mask].mean(axis=-1)
    qc = dict(b0_invT1_corr_before=corr(b0, t1_inv), b0_invT1_corr_after=corr(b0_after, t1_inv),
              pe_displacement_mm_p1_p99=[float(np.percentile(disp_mm[mask_dwi], 1)),
                                         float(np.percentile(disp_mm[mask_dwi], 99))])
    dc.save_nifti(qc_dir / "sdc_t1inv_dwispace.nii.gz", t1_inv, img_affine)
    dc.save_nifti(qc_dir / "sdc_b0_before.nii.gz", b0, img_affine)
    dc.save_nifti(qc_dir / "sdc_b0_after.nii.gz", b0_after, img_affine)
    logger.info(f"   b0 vs inverted-T1 correlation: {qc['b0_invT1_corr_before']:.3f} -> "
                f"{qc['b0_invT1_corr_after']:.3f}; PE displacement p1/p99 = "
                f"{qc['pe_displacement_mm_p1_p99'][0]:.1f}/{qc['pe_displacement_mm_p1_p99'][1]:.1f} mm")
    return out, qc


# ---------------------------------------------------------------------------
# Main per-run function
# ---------------------------------------------------------------------------


def run_preproc(run: dict, out_dir: Path, participant: str, args, logger) -> dict:
    """Preprocess one DWI run. Returns the QC dict (also written to qc_data/)."""
    from dipy.reconst.dti import TensorModel
    from dipy.segment.mask import median_otsu

    out_dir = Path(out_dir)
    qc_dir = out_dir / "qc_data"
    qc_dir.mkdir(parents=True, exist_ok=True)
    timings: dict = {}
    qc: dict = dict(participant=participant, input=str(run["nii"]),
                    software=dc.software_versions(), parameters={
                        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})

    # ------------------------------------------------------------------ 0
    # Inputs: image, gradients (FSL -> voxel convention), JSON metadata
    img = nib.load(str(run["nii"]))
    affine = img.affine
    data = np.asarray(img.dataobj, dtype=np.float32)
    bvals, bvecs = dc.load_fsl_gradients(run["bval"], run["bvec"], affine)
    if data.ndim != 4 or data.shape[-1] != bvals.size:
        raise ValueError(f"{run['nii'].name}: {data.shape[-1]} volumes but {bvals.size} b-values")
    gtab = dc.make_gtab(bvals, bvecs, args.b0_threshold)
    meta = dc.read_json(run["json"])
    pe = dc.phase_encoding(meta)
    qc["acquisition"] = dict(shape=data.shape, voxel_size_mm=dc.voxel_sizes(affine),
                             n_b0=int(gtab.b0s_mask.sum()),
                             shells=sorted({int(round(b, -2)) for b in bvals[~gtab.b0s_mask]}),
                             phase_encoding=pe, partial_fourier=meta.get("PartialFourier"))
    logger.info(f"   {run['nii'].name}: {data.shape}, {int(gtab.b0s_mask.sum())} b0, "
                f"shells {qc['acquisition']['shells']}, PE {pe['pe_dir']}, "
                f"readout {pe['readout_time']}")

    # Intermediate results are reused when they already exist (unless
    # --overwrite), so an interrupted run resumes at the step where it stopped.
    def reuse(path: Path) -> bool:
        return path.is_file() and not args.overwrite

    # ------------------------------------------------------------------ 1
    # Denoising (dwidenoise equivalent)
    den_file = out_dir / "dwi_den.nii.gz"
    if reuse(den_file):
        logger.info(f"   reusing {den_file.name}")
        data = np.asarray(nib.load(str(den_file)).dataobj, dtype=np.float32)
    else:
        with dc.timed(logger, f"Denoising ({args.denoise})", timings):
            if args.denoise == "mppca":
                # A generously dilated head mask (from the raw mean b0) restricts
                # the patch-wise PCA to the head (~2-3x faster; the background is
                # irrelevant downstream). Outside it the output is 0.
                raw_b0 = data[..., gtab.b0s_mask].mean(axis=-1)
                _, head = median_otsu(raw_b0, median_radius=4, numpass=4)
                head = ndimage.binary_dilation(head, iterations=4)
                den, sigma = mppca_chunked(data, head, args.mppca_patch_radius,
                                           args.denoise_chunk_slices, logger)
                dc.save_nifti(qc_dir / "noise_sigma.nii.gz", sigma, affine)
                residual = np.where(head[..., None], data - den, 0)
                dc.save_nifti(qc_dir / "denoise_residuals_rms.nii.gz",
                              np.sqrt(np.mean(residual ** 2, axis=-1)), affine)
                qc["denoise"] = dict(median_sigma_in_head=float(np.median(sigma[head])))
                del residual
                data = den
            elif args.denoise == "patch2self":
                try:
                    import sklearn.linear_model  # noqa: F401  (needed by Patch2Self)
                except ImportError as e:
                    raise ImportError("--denoise patch2self needs scikit-learn "
                                      "(pip install scikit-learn)") from e
                from dipy.denoise.patch2self import patch2self
                data = patch2self(data, bvals, model="ols", b0_threshold=args.b0_threshold,
                                  shift_intensity=True, clip_negative_vals=False,
                                  out_dtype=np.float32)
            data = np.clip(data, 0, None).astype(np.float32)
            dc.save_nifti(den_file, data, affine)

    # ------------------------------------------------------------------ 2
    # Gibbs ringing removal (mrdegibbs equivalent; not in the original pipeline)
    pf = meta.get("PartialFourier")
    do_gibbs = args.gibbs == "on" or (args.gibbs == "auto" and not (pf is not None and pf < 1))
    qc["gibbs_applied"] = bool(do_gibbs)
    gibbs_file = out_dir / "dwi_degibbs.nii.gz"
    if do_gibbs and reuse(gibbs_file):
        logger.info(f"   reusing {gibbs_file.name}")
        data = np.asarray(nib.load(str(gibbs_file)).dataobj, dtype=np.float32)
    elif do_gibbs:
        with dc.timed(logger, "Gibbs ringing removal", timings):
            from dipy.denoise.gibbs import gibbs_removal
            data = gibbs_removal(data, slice_axis=dc.slice_axis(meta), n_points=3,
                                 inplace=False, num_processes=args.n_jobs).astype(np.float32)
            data = np.clip(data, 0, None)
            dc.save_nifti(gibbs_file, data, affine)
    else:
        logger.info(f"   Gibbs removal skipped (--gibbs {args.gibbs}, PartialFourier={pf})")

    # ------------------------------------------------------------------ 3-4
    # Susceptibility, eddy-current and motion correction.
    # FSL routes (default): topup+eddy with reverse-PE fieldmaps, otherwise
    # Synb0-DisCo+topup+eddy (see dwi_fsl.py). 'dipy': DIPY-only route below.
    fieldmaps, fmap_how = dc.find_fieldmaps(args.bids_dir, participant, run)
    route = dwi_fsl.choose_route(args, fieldmaps)
    qc["correction"] = dict(route=route, fieldmaps=[str(f["nii"]) for f in fieldmaps],
                            fieldmaps_found_by=fmap_how)
    logger.info(f"   fieldmaps: {len(fieldmaps)} found ({fmap_how}) -> correction route '{route}'")
    if route != "dipy":
        if args.sdc != "none" or args.motion != "affine":
            logger.warning("   --motion/--sdc only apply to --correction dipy; ignored")
        with dc.timed(logger, f"FSL correction ({route})", timings):
            data, bvecs_rot, cqc = dwi_fsl.correct(
                data, bvals, gtab.bvecs, affine, meta, run, participant, route, fieldmaps,
                fmap_how, out_dir, args, logger, timings)
        qc["correction"].update(cqc)
        qc["motion"] = cqc.get("eddy", {})
        gtab = dc.make_gtab(bvals, bvecs_rot, args.b0_threshold)
    else:
        # ------------------------------------------------------------------ 3
        # Motion + eddy-current correction (replaces FSL eddy)
        mc_file = out_dir / "dwi_mc.nii.gz"
        mc_aff = qc_dir / "motion_affines.npy"
        if args.motion != "none" and reuse(mc_file) and mc_aff.is_file():
            logger.info(f"   reusing {mc_file.name} and {mc_aff.name}")
            data = np.asarray(nib.load(str(mc_file)).dataobj, dtype=np.float32)
            reg_affines = np.load(mc_aff)
            qc["motion"] = dc.read_json(qc_dir / "motion_summary.json")
            gtab = dc.make_gtab(bvals, dc.rotate_bvecs(gtab.bvecs, reg_affines, affine, gtab.b0s_mask),
                                args.b0_threshold)
        elif args.motion != "none":
            with dc.timed(logger, f"Motion/eddy correction ({args.motion})", timings):
                data, bvecs_rot, reg_affines, _, motion_rows = motion_correct(
                    data, gtab, affine, args.motion, args.motion_level_iters, args.n_jobs, logger)
                cols = list(motion_rows[0].keys())
                with open(qc_dir / "motion_parameters.tsv", "w") as fh:
                    fh.write("\t".join(cols) + "\n")
                    for r in motion_rows:
                        fh.write("\t".join(f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])
                                           for c in cols) + "\n")
                rel = np.array([r["rms_rel_mm"] for r in motion_rows])
                ab = np.array([r["rms_abs_mm"] for r in motion_rows])
                qc["motion"] = dict(mean_rms_abs_mm=ab.mean(), max_rms_abs_mm=ab.max(),
                                    mean_rms_rel_mm=rel.mean(), max_rms_rel_mm=rel.max())
                dc.write_json(qc_dir / "motion_summary.json", qc["motion"])
                logger.info(f"   displacement: mean abs {ab.mean():.2f} mm (max {ab.max():.2f}), "
                            f"mean vol-to-vol {rel.mean():.2f} mm (max {rel.max():.2f})")
                gtab = dc.make_gtab(bvals, bvecs_rot, args.b0_threshold)
                data = np.clip(data, 0, None).astype(np.float32)
                dc.save_nifti(mc_file, data, affine)
                np.save(mc_aff, reg_affines)          # saved last: marks the step as complete

        # ------------------------------------------------------------------ 4
        # Susceptibility distortion correction (optional, approximate)
        qc["sdc"] = dict(method=args.sdc)
        if args.sdc == "syn":
            if pe["axis"] is None:
                raise ValueError("--sdc syn needs PhaseEncodingDirection/Axis in the JSON sidecar")
            with dc.timed(logger, "Fieldmap-less SDC (SyN, PE-restricted)", timings):
                anat = dc.load_anatomy(participant, args.anat_source, args.fs_dir, args.preproc_dir)
                data, sdc_qc = syn_sdc(data, gtab, affine, pe["axis"], anat,
                                       args.sdc_level_iters, logger, qc_dir)
                qc["sdc"].update(sdc_qc)
        else:
            logger.info("   No susceptibility distortion correction (--sdc none)")

    data = np.clip(data, 0, None).astype(np.float32)
    dc.save_nifti(out_dir / "dwi_clean.nii.gz", data, affine)
    dc.write_fsl_gradients(gtab.bvals, gtab.bvecs, affine,
                           out_dir / "rotated.bval", out_dir / "rotated.bvec")

    # ------------------------------------------------------------------ 5-6
    # Mean b0 and brain mask (dwiextract/fslmaths -Tmean and dwi2mask)
    with dc.timed(logger, "Mean b0 and brain mask", timings):
        bzero = data[..., gtab.b0s_mask].mean(axis=-1)
        dc.save_nifti(out_dir / "dwi_bzero.nii.gz", bzero, affine)
        # Mask from the mean diffusion-weighted image (scalp, eyes and fat are
        # dark at b>0, unlike the b0). On sub-6175 this gave a Dice of 0.96 with
        # the FreeSurfer brain mask (b0-based median_otsu: 0.93).
        mean_dwi = data[..., ~gtab.b0s_mask].mean(axis=-1) if (~gtab.b0s_mask).any() else bzero
        _, mask = median_otsu(mean_dwi, median_radius=2, numpass=1)
        mask = ndimage.binary_fill_holes(dc.largest_component(mask))
        dc.save_nifti(out_dir / "dwi_mask.nii.gz", mask, affine, dtype=np.uint8)
        qc["mask_volume_ml"] = float(mask.sum() * np.prod(dc.voxel_sizes(affine)) / 1000)

    # ------------------------------------------------------------------ 7
    # Tensor fit and scalar maps (dwi2tensor + tensor2metric)
    with dc.timed(logger, "Tensor fitting", timings):
        tfit = TensorModel(gtab, fit_method="WLS").fit(data, mask=mask)
        fa = np.clip(np.nan_to_num(tfit.fa), 0, 1)
        maps = dict(FA=fa, MD=tfit.md, AD=tfit.ad, RD=tfit.rd)
        for k, v in maps.items():
            dc.save_nifti(out_dir / f"dwi_{k}.nii.gz", np.where(mask, np.nan_to_num(v), 0), affine)
        dc.save_nifti(out_dir / "dwi_tensor.nii.gz",
                      np.where(mask[..., None], tfit.lower_triangular(), 0), affine)
        v1_world = tfit.evecs[..., :, 0] @ dc.direction_cosines(affine).T
        dc.save_nifti(out_dir / "dwi_directions.nii.gz",
                      np.where(mask[..., None], v1_world * fa[..., None], 0), affine)
        wm_like = mask & (fa > 0.4)
        qc["dti"] = dict(median_FA_in_mask=float(np.median(fa[mask])),
                         median_MD_in_FA_gt_0p4=float(np.median(tfit.md[wm_like])) if wm_like.any() else None)

    qc["timings_s"] = timings
    dc.write_json(out_dir / PREPROC_DONE, qc)
    return qc


def main(argv=None):
    args = build_parser().parse_args(argv)
    logger = dc.get_logger(args.output_dir / "log" / f"Dwipreproc_{time.strftime('%Y%m%d_%H%M')}.txt")
    for participant in dc.list_participants(args.bids_dir, args.participant_label):
        runs = dc.find_dwi_runs(args.bids_dir, participant)
        if not runs:
            logger.warning(f"{participant}: no DWI run with bval/bvec found - skipped")
            continue
        for run in runs:
            n_same = sum(r["session"] == run["session"] for r in runs)
            out = dc.run_output_dir(args.output_dir, participant, run, n_same)
            ok, why = preproc_status(out, run, participant, args)
            if ok:
                logger.info(f"{participant}: DWI data already cleaned ({out}): {why}")
                continue
            logger.info(f"{participant}: preprocessing ({why})")
            run_preproc(run, out, participant, args, logger)


if __name__ == "__main__":
    main()
