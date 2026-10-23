# compneuro-dwiproc-dipy

DIPY re-implementation of the lab's DWI pipeline
[compneuro-dwiproc](../compneuro-dwiproc) (MRtrix3 + FSL + ANTs): denoising,
motion/eddy-current correction, brain mask, tensor metrics, registration of an
MNI parcellation to diffusion space, CSD, probabilistic and deterministic
tractography and structural connectivity matrices — with **DIPY 1.12.0**
for everything except susceptibility/eddy-current/motion correction, which uses
**FSL topup + eddy** (as the original's `dwifslpreproc`), with
**Synb0-DisCo** providing the topup input when no reverse phase-encoding
fieldmap was acquired.

Written in October 2026 and tested on the single-subject dataset `sub-6175`
(Siemens Trio, 64 directions at b = 1000 s/mm², one b0, no reverse
phase-encoding scan, DeepPrep/FreeSurfer `Recon` available).

## Requirements

* Python ≥ 3.10 with `dipy==1.12.0`, `nibabel`, `numpy`, `scipy`
  (`pip install -r requirements.txt`). On the lab Mac these are in
  `~/venvs/neurovenv`. `--denoise patch2self` also needs `scikit-learn`
  (installed with nilearn).
* Anatomy, one of:
  * **FreeSurfer** (default): a `SUBJECTS_DIR` with `sub-X/mri/brainmask.mgz`
    and `sub-X/mri/aparc+aseg.mgz` — e.g. DeepPrep's `Recon` folder.
  * **Original layout** (`--anat-source probtissue`): `Preproc/BET/sub-X_T1w_brain.nii.gz`
    and `Preproc/ProbTissue/sub-X_T1w_brain_{corticalGM,subcorticalGM,WM,CSF}.nii.gz`,
    as described in the compneuro-dwiproc README.
* Raw data in BIDS: `sub-X/[ses-Y/]dwi/*_dwi.nii.gz` with `.bval`, `.bvec`
  and `.json` (PhaseEncodingDirection and TotalReadoutTime are needed for
  topup/eddy). `--bids-dir` can be a BIDS root or a single subject folder.
  Reverse-PE spin-echo fieldmaps are taken from `fmap/*_epi.nii.gz` whose
  JSON `IntendedFor` lists the DWI (as the CAINT bidsify script writes them);
  without an `IntendedFor` match, `fmap/*acq-dwi*_epi` of the same session.
* **FSL** ≥ 6.0.7 (topup, eddy_cpu, bet, eddy_quad): natively (`$FSLDIR`, or
  `~/fsl`; the Mac has 6.0.7.23) or in a container (`--fsl-container`).
* **Synb0-DisCo** (only for data without fieldmaps): the
  `leonyichencai/synb0-disco:v3.1` image, run with Docker on the Mac (it is an
  x86-64 image, emulated on Apple Silicon; give Docker ≥ 13 GB RAM) or with
  Apptainer on the cluster, plus a FreeSurfer license (`--fs-license`; default
  `$FS_LICENSE` or `~/fs_license.txt`).

`brain_templates/` is a copy of the original repository's templates and
parcellations (MNI152 2 mm).

## Usage

```bash
cd ~/projects/compneuro-dwiproc-dipy
caffeinate -dimsu ~/venvs/neurovenv/bin/python src/dwi_launcher.py \
    --bids-dir ~/Desktop/sub-6175 \
    --output-dir ~/Desktop/sub-6175/derivatives/dwiproc_dipy \
    --fs-dir ~/Desktop/sub-6175/derivatives/Recon \
    --parcellation Desikan_genetics_2mm \
    --n-jobs 8
```

(`--parcellation` takes a file in `brain_templates/` or any path to an MNI152
2 mm label image, e.g. `craddock_2128_2mm`.) The same command is in
`run_test_sub-6175.sh`.

Useful options (all scripts have `--help`):

| Option | Default | Meaning |
|---|---|---|
| `--stages preproc tract` | both | run only part of the pipeline |
| `--participant-label` | all | participants to run |
| `--denoise mppca\|patch2self\|none` | `mppca` | MP-PCA = dwidenoise's method |
| `--gibbs auto\|on\|off` | `auto` | Gibbs removal; `auto` skips partial-Fourier data |
| `--correction auto\|topup\|synb0\|eddy\|dipy` | `auto` | `auto`: topup+eddy with fieldmaps, else Synb0-DisCo+topup+eddy; `eddy`: no susceptibility correction; `dipy`: the earlier DIPY-only route |
| `--eddy-s2v` | off | eddy slice-to-volume correction (multiband data; slow on CPU) |
| `--eddy-extra="..."` | — | extra eddy options, e.g. `--eddy-extra="--slm=linear"` |
| `--fsl-container`, `--synb0-container`, `--container-engine` | native FSL, Docker | where FSL and Synb0-DisCo run |
| `--container-gpu` | off | GPU eddy (`eddy_cuda`) inside the FSL container |
| `--motion`, `--sdc` | `affine`, `none` | only for `--correction dipy` |
| `--n-streamlines` | 3000000 | accepted streamlines per tractogram (`-select`) |
| `--stopping act\|cmc` | `act` | anatomical stopping criterion |
| `--prob-algorithm prob\|pft` | `prob` | `pft` ≈ MRtrix `-backtrack` (slower) |
| `--no-save-tractograms` | off | only write the matrices |
| `--overwrite` | off | recompute everything |

Each stage can also be run on its own: `src/dwi_preproc.py`, `src/dwi_tract.py`.
Intermediate results are reused, so an interrupted run resumes at the step
where it stopped (use `--overwrite` after changing parameters). A run that was
preprocessed with another correction route (e.g. with the earlier DIPY-only
version) is preprocessed again automatically, reusing the denoised data, and
its tractography is recomputed.

**Cluster (DIPC):** `slurm/dwi_launcher_slurm.sh` runs one participant per
array task. DIPY runs from a Python virtual environment; FSL runs in the lab's
`compneurobilbaolab/compneuro-dwiproc:1.0.0` image (FSL 6.0.7.6) and
Synb0-DisCo in its own image, both pulled once as Apptainer `.sif` files. The
header of the script has the exact `apptainer pull` and `sbatch` commands.
Note that the Mac has FSL 6.0.7.23 and the image 6.0.7.6: process all
participants of one study in the same place (or with the same FSL version).
`slurm/submit_caint68_preproc.sh` submits the preprocessing of the new CAINT_68
participants (sub-ID438/439/440 by default): it checks the repository
(checksums), the Python/DIPY environment, the images and the BIDS data on the
login node before submitting, and `DRY_RUN=1` only prints the `sbatch` command.
A subset of participants can be run with any launcher call by exporting
`DWIPROC_PARTICIPANTS_FILE=<list>` (header line + one label per line).

## What each step does, and how it differs from the original

### Preprocessing (`dwi_preproc.py`)

| # | Original | DIPY version | Equivalent? |
|---|---|---|---|
| 1 | `dwidenoise` | `mppca` (Marchenko–Pastur PCA, 5×5×5 patches) | Same method. Run in slabs (identical result, checked) and only inside a dilated head mask. |
| 2 | — | `gibbs_removal` | Not in the original. Skipped automatically when the JSON reports partial Fourier < 1 (the method assumes full k-space; sub-6175 has 6/8). |
| 3–4 | `dwifslpreproc`: `topup` (only with a reverse-PE pair) + `eddy --repol --ol_nstd=4 --cnr_maps` | **With fieldmaps** (`--correction topup`): topup on [first DWI b0, AP/PA fieldmaps] (`b02b0.cnf`), then eddy with the same options plus `--topup`. **Without fieldmaps** (`synb0`): Synb0-DisCo makes an undistorted b0 from the T1w; topup on [distorted b0, synthetic b0 with readout time 0] (Synb0-DisCo's schedule), then eddy. | Same FSL tools as the original. Putting the DWI b0 first in topup puts the field in DWI space (like `-align_seepi`); `--no-align-seepi` uses the fieldmaps only. Odd image sizes are padded by one slice for topup and cropped back (as `dwifslpreproc`). Without fieldmaps the original did no susceptibility correction; Synb0-DisCo is new. Slice-to-volume correction with `--eddy-s2v`. |
| 3–4 (old) | — | `--correction dipy`: per-volume affine registration to the b0 + b-vector rotation; optional SyN-SDC (`--sdc syn`) | The earlier DIPY-only route, kept for machines without FSL. No outlier replacement, no non-linear eddy model, no field map. |
| 5–6 | `dwiextract -bzero`, `dwi2mask` | mean b0; `median_otsu` on the mean DWI | Mask Dice 0.96 with the FreeSurfer brain mask on sub-6175. |
| 7 | `dwi2tensor`, `tensor2metric` | `TensorModel` (WLS) | Same model. FA/MD/AD/RD comparable; the 6-component tensor is in DIPY order (Dxx, Dxy, Dyy, Dxz, Dyz, Dzz). `dwi_directions` = FA-modulated V1 in scanner axes, like `tensor2metric -vector`. |

B-vectors: FSL convention on disk (`rotated.bvec` = eddy's rotated b-vectors);
internally they are converted to voxel axes (identical for the usual
negative-determinant images written by dcm2niix).

### Tractography and connectomes (`dwi_tract.py`)

| # | Original | DIPY version | Equivalent? |
|---|---|---|---|
| 1 | `epi_reg` (BBR) + `flirt` | rigid b0→T1, mutual information; T1 resampled to DWI space (`anat_dwispace`) | Different cost function (MI, not BBR). QC: median FA in WM vs GM is logged. |
| 2 | `antsRegistrationSyN.sh` MNI→`anat_dwispace`, NN warp | affine (MI) + SyN (CC) MNI→`anat_dwispace`, NN warp | Same design; DIPY's SyN. |
| 3 | `5ttgen fsl`, `5tt2gmwmi` | tissue maps from `aparc+aseg` (as `5ttgen freesurfer`) or ProbTissue, resampled to the DWI grid → partial volumes; GM/WM interface layer as seeding weights | Different segmentation source (FreeSurfer instead of FAST/FIRST). |
| 4 | `dwi2response tax` | `recursive_response` | Same algorithm (Tax et al. 2014), run in WM voxels. If it fails, a single-fibre response (WM, FA > 0.7) is used with a warning in the log and QC JSON. |
| 5 | `dwi2fod csd` | `ConstrainedSphericalDeconvModel`, lmax 8 | Same method; SH basis is DIPY's (`descoteaux07`), not MRtrix's. lmax is lowered automatically when there are too few directions (as MRtrix does), e.g. 6 for 32 directions. |
| 6 | `tckgen iFOD2 -act -backtrack -seed_gmwmi -angle 45 -maxlength 200 -select 3M` | probabilistic tracking on the FODs, ACT criterion (stop at GM > 0.5, reject at CSF > 0.5), GMWMI seeding, 45°, 200 mm, until 3M accepted | DIPY's probabilistic tracker is not iFOD2 (first-order, not second-order integration). `--prob-algorithm pft` gives the closest analogue of `-backtrack`. |
| 7 | `tcksift2` | — | **No DIPY equivalent.** Matrices are raw streamline counts. |
| 8 | `tck2connectome -zero_diagonal -symmetric -assignment_radial_search 5` | endpoint labels with the same 5 mm radial search | Same rules; CSV, comma-separated, nodes = label values 1…max. |
| 9 | `tckgen FACT` on the tensor eigenvector | deterministic tracking on the tensor V1 (FA cut-off 0.1), same seeding/stopping | Equivalent design (V1 discretised to 724 directions). |

About speed: DIPY's command-line tools (`dipy_track`, …) call exactly the
same Python/Cython functions, so they are not faster. What makes tracking fast
is DIPY's multithreaded tracker (`nbr_threads`), which this pipeline uses
directly with `--n-jobs` threads.

## Run time and memory

topup takes minutes; Synb0-DisCo roughly 20–60 min (longer under emulation on
Apple Silicon); eddy on CPU from tens of minutes (sub-6175) to a few hours
(CAINT_68: 99 volumes at 1.5 mm), and much longer with `--eddy-s2v` — use the
cluster or a GPU for large batches. The DIPY steps were measured on sub-6175
(122×122×77, 65 volumes) in a 4-core, 4 GB Linux VM; the Mac is faster.

| Step | Time |
|---|---|
| MP-PCA denoising | ~6.5 min |
| DIPY-only motion correction (`--correction dipy`, 64 volumes, 4 processes) | ~1.4 min |
| Mask + tensor | < 0.5 min |
| b0→T1 + MNI→DWI registration (affine + SyN) | ~0.5 min |
| Tax response + CSD fit | ~1 min |
| Probabilistic tracking, 300 000 streamlines (3 threads) | ~1 min |
| Deterministic tracking, 300 000 streamlines (3 threads) | ~1.2 min |

So the default 3 million streamlines per tractogram take roughly 10 minutes
each with 3 threads (less with `--n-jobs 8`). Peak memory was 2.4 GB; the
`.trk` files are ~0.6 GB (prob) and ~0.2 GB (det) per million streamlines.

## Outputs (`<output-dir>/sub-X[/ses-Y]/`)

Same names as the original where there is an equivalent: `dwi_den`,
`dwi_clean`, `rotated.bval/.bvec`, `dwi_bzero`, `dwi_mask`, `dwi_tensor`,
`dwi_FA/MD/AD/RD`, `dwi_directions`, `anat_dwispace`,
`<parc>_subSpace`, `dwi_response.txt`, `dwi_fod`, `<parc>_prob.csv`,
`<parc>_det.csv`. Additional: `fsl_work/` (topup, eddy and Synb0-DisCo files
and one log per command in `fsl_work/logs/`), `dwi_mc` (DIPY route only),
`act_5tt` (cGM, sGM, WM, CSF, pathological), `GMWMI`, `dwi_fod_peaks`,
`streamlines_prob.trk` / `streamlines_det.trk` (compressed), `dwireg/`
(transforms, QC images) and `qc_data/`: noise map; `topup_field_hz` and
`topup_displacement_pe_mm` (the estimated distortion); `eddy/` (movement RMS,
outlier map/report, rotated b-vectors, CNR maps); `eddy_quad/` (FSL's QC
report); `preproc_desc.json` (route, fieldmaps used, eddy command, movement,
outlier and field summaries) and `tract_<parc>_desc.json` (parameters,
software versions, timings, acceptance rates, lengths and registration QC).

## Known limitations

* Synb0-DisCo is a learned approximation of the undistorted b0 (validated for
  topup in Schilling et al. 2019, 2020); check `qc_data/topup_*` and the
  corrected b0 against the T1. Its image is x86-64 only.
* `--eddy-s2v` cannot be used when the slice axis has an odd number of slices
  (it is padded for topup).
* With fieldmaps, the DWI b0 (TE 89 ms on CAINT_68) and the spin-echo
  fieldmaps (TE 54 ms) have different contrast; it is logged as a warning and
  handled as in `dwifslpreproc -align_seepi` (or use `--no-align-seepi`).
* The `--correction dipy` route has no outlier replacement, non-linear eddy
  model or field map; `--sdc syn` is an approximation.
* No SIFT2: connectivity = streamline counts.
* Tissue maps are resampled to the DWI resolution before tracking (MRtrix
  ACT uses the 1 mm 5TT image). DIPY's continuous-map criterion (`--stopping cmc`)
  stops many streamlines within a few mm at this resolution (median length
  8 mm on sub-6175 vs 28 mm with `act`), so `act` is the default.

## Validation done (October 2026)

* MP-PCA in slabs gives exactly the same result as one call (max difference 0).
* B-vector rotation: on simulated data with known head rotations, the rotated
  b-vectors gave the smallest tensor-direction error (0.68° vs 0.81° unrotated
  and 1.06° rotated the wrong way).
* Synthetic phantom (left–right fibres, FreeSurfer-like and ProbTissue-like
  anatomy): both tractograms run left–right and the strongest connections are
  the left–right parcel pairs; all options (`patch2self`, Gibbs, `--sdc syn`,
  `pft`, `cmc`, `rigid`, both anatomy sources, resume/skip logic) ran.
* FSL/Synb0 routes (22 Oct 2026): command construction, file formats
  (acqparams, index, padding/cropping, eddy outputs), native/Docker/Apptainer
  modes, fieldmap discovery and the resume/re-run logic were tested on
  synthetic data with stand-in programs that check the inputs. Real topup,
  eddy and Synb0-DisCo runs have to be checked on the Mac (sub-6175: Synb0
  route; CAINT_68 sub-ID438: fieldmap route).
* sub-6175 (DIPY-only route): brain mask Dice 0.96 with FreeSurfer's mask; median FA 0.47 in WM
  vs 0.15 in cortex; CSD peak vs tensor direction median 3.8° (FA > 0.6);
  all 82 Desikan nodes present; 94 % of streamlines assigned at both ends;
  median length ~35 mm for both tractograms.
