#!/bin/bash
###########################################################
#                                                         #
#     compneuro-dwiproc-dipy: SLURM array launcher        #
#                                                         #
###########################################################
#
# One array task = one participant (line N+1 of participants.tsv), as in the
# original dwi_launcher_slurm.sh. DIPY runs from a Python virtual environment
# (first argument). FSL (topup/eddy) and Synb0-DisCo run in Apptainer images.
#
# One-time setup on the login node (paths are examples):
#
#   module load Apptainer
#   cd /scratch/<user>/apptainer_images
#   # FSL 6.0.7.6 (+ MRtrix, ANTs) from the lab's original pipeline image
#   apptainer pull compneuro-dwiproc_1.0.0.sif docker://compneurobilbaolab/compneuro-dwiproc:1.0.0
#   # Synb0-DisCo (only needed for data without reverse-PE fieldmaps)
#   apptainer pull synb0-disco_v3.1.sif docker://leonyichencai/synb0-disco:v3.1
#   # Python environment with DIPY
#   python3 -m venv /scratch/<user>/venvs/dipy && \
#       /scratch/<user>/venvs/dipy/bin/pip install -r <repo>/requirements.txt
#
# Submit (login node) from the repository folder (or export DWIPROC_REPO=<repo>),
# e.g. for the 21 participants of a study:
#
#   cd <repo> && mkdir -p slurm_logs
#   sbatch --array=1-21 slurm/dwi_launcher_slurm.sh \
#       /scratch/<user>/venvs/dipy/bin/python \
#       /scratch/<user>/project/bids \
#       /scratch/<user>/project/derivatives/dwiproc_dipy \
#       /scratch/<user>/project/freesurfer \
#       Desikan_genetics_2mm \
#       --container-engine apptainer \
#       --fsl-container /scratch/<user>/apptainer_images/compneuro-dwiproc_1.0.0.sif \
#       --synb0-container /scratch/<user>/apptainer_images/synb0-disco_v3.1.sif \
#       --fs-license /scratch/<user>/apptainer_images/freesurfer_license.txt
#
# To run only some participants, put them in a file (header line + one label
# per line) and export DWIPROC_PARTICIPANTS_FILE=<file> before sbatch; see
# slurm/submit_caint68_preproc.sh.
#
# Everything after the 5th argument is passed to src/dwi_launcher.py (e.g.
# --eddy-s2v, --correction ...). /scratch/... paths in ANY argument are turned
# into the /dipc/... paths seen by the compute nodes.
#
# GPU eddy (much faster, also with --eddy-s2v): submit to a GPU partition with
# e.g. "-p <gpu partition> --gres=gpu:1" on the sbatch line and add
# --container-gpu to the pipeline options.
#
# Logs: slurm_logs/dwiprep_<jobid>_<task>.out/.err in the submission folder;
# pipeline logs in <output dir>/log and <output dir>/sub-X[/ses-Y]/log.txt.

#SBATCH -J dwiprep_dipy
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8                 # cores for one participant (eddy_cpu --nthr, tracking threads)
#SBATCH -p general
#SBATCH --qos=regular
#SBATCH --mem=32G
#SBATCH -t 1-00:00:00        # eddy on CPU can take hours (more with --eddy-s2v)
#SBATCH -o slurm_logs/dwiprep_%A_%a.out
#SBATCH -e slurm_logs/dwiprep_%A_%a.err

set -euo pipefail

# Apptainer for the FSL / Synb0 containers (module name as on DIPC)
if command -v module > /dev/null 2>&1; then
    module load Apptainer || echo "WARNING: could not load the Apptainer module"
fi

# DIPC: files under /scratch on the login node are under /dipc on compute
# nodes. Convert a leading /scratch/ (no-op elsewhere), also inside --opt=/scratch/...
to_compute_path() {
    case "$1" in
        /scratch/*) echo "/dipc/${1#/scratch/}" ;;
        --*=/scratch/*) echo "${1%%=*}=/dipc/${1#*=/scratch/}" ;;
        *) echo "$1" ;;
    esac
}

python_bin=$(to_compute_path "$1")
bids_dir=$(to_compute_path "$2")
output_dir=$(to_compute_path "$3")
fs_dir=$(to_compute_path "$4")
parcellation=$(to_compute_path "$5")
shift 5
extra=()
for arg in "$@"; do
    extra+=("$(to_compute_path "$arg")")
done

# Repository location: sbatch runs a copy of this script from a spool folder,
# so use $DWIPROC_REPO if set, else the submission folder, else this file's folder.
repo_dir=""
for cand in "${DWIPROC_REPO:-}" "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; do
    cand=$(to_compute_path "${cand}")
    if [ -n "${cand}" ] && [ -f "${cand}/src/dwi_launcher.py" ]; then repo_dir=${cand}; break; fi
done
if [ -z "${repo_dir}" ]; then
    echo "Cannot find the repository: submit from its folder or export DWIPROC_REPO=<repo path>"
    exit 1
fi
# Participant list: participants.tsv of the BIDS folder, or the file in
# $DWIPROC_PARTICIPANTS_FILE (same format: header line + one label per line),
# e.g. to run a subset such as new participants.
participants_file=$(to_compute_path "${DWIPROC_PARTICIPANTS_FILE:-${bids_dir}/participants.tsv}")

# Line SLURM_ARRAY_TASK_ID + 1 (skip the header) -> participant label
list_pointer=$((SLURM_ARRAY_TASK_ID + 1))
participant=$(sed -n "${list_pointer}p" "${participants_file}" | tr -d '\r\n' | awk '{print $1}')
if [ -z "${participant}" ]; then
    echo "No participant on line ${list_pointer} of ${participants_file}"; exit 1
fi
echo "Participant: ${participant} (task ${SLURM_ARRAY_TASK_ID})"

# The pipeline controls parallelism with --n-jobs; avoid BLAS oversubscription
# inside its worker processes. (OMP_NUM_THREADS is left alone: DIPY's
# multithreaded tracker uses OpenMP.)
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

"${python_bin}" "${repo_dir}/src/dwi_launcher.py" \
    --bids-dir "${bids_dir}" \
    --output-dir "${output_dir}" \
    --fs-dir "${fs_dir}" \
    --parcellation "${parcellation}" \
    --participant-label "${participant}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-8}" \
    ${extra[@]+"${extra[@]}"}
