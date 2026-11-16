#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=lab3.2_ridge_bert
#SBATCH --cpus-per-task=5
# Ridge uses streaming Y (see lab3.2_ridge_bert.py); 20GB is enough on typical nodes.
#SBATCH --mem=20GB
#SBATCH --time=08:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
#SBATCH -o lab3.2_slurm-ridge-%j.log
#SBATCH -e lab3.2_slurm-ridge-%j.log

set -e
CODE_DIR="${HOME}/lab-3-group-15/code"
# Empty = use Hugging Face dataset YOUR_HF_USER/stat214-lab3.2 (huggingface_hub cache).
# Set to a local dir (e.g. ${HOME}/bert_features) to skip HF download.
BERT_FEATURES_DIR="${BERT_FEATURES_DIR:-}"
VARIANT="${VARIANT:-pt}"
SUBJECT="${SUBJECT:-subject2}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/results}"
HF_REPO="${HF_REPO:-YOUR_HF_USER/stat214-lab3.2}"
# Per-chunk RidgeCV on kept voxels; lower if RSS still spikes (e.g. 2048).
VOXEL_CHUNK="${VOXEL_CHUNK:-4000}"
MASK_COL_CHUNK="${MASK_COL_CHUNK:-4096}"

mkdir -p "${OUTPUT_DIR}"

module load anaconda3
CONDA_ENV="${CONDA_ENV:-stat214}"
source activate "${CONDA_ENV}"

cd "${CODE_DIR}" || { echo "ERROR: CODE_DIR not found: ${CODE_DIR}"; exit 1; }
export PYTHONUNBUFFERED=1

echo "=== $(date)  JOB=${SLURM_JOB_ID}  Ridge on BERT features ==="
if [ -n "${BERT_FEATURES_DIR}" ]; then
  echo "    BERT_FEATURES_DIR=${BERT_FEATURES_DIR}  (local pickles)"
else
  echo "    BERT_FEATURES_DIR=(unset)  -> HF dataset ${HF_REPO}"
fi
echo "    VARIANT=${VARIANT}  SUBJECT=${SUBJECT}"
echo "    OUTPUT_DIR=${OUTPUT_DIR}"
echo "    HF_HOME=${HF_HOME:-<default cache>}"
echo "    VOXEL_CHUNK=${VOXEL_CHUNK}  MASK_COL_CHUNK=${MASK_COL_CHUNK}"
nvidia-smi || true

RIDGE_ARGS=(
  -u lab3.2_ridge_bert.py
  --variant "${VARIANT}"
  --subject "${SUBJECT}"
  --output_dir "${OUTPUT_DIR}"
)
if [ -n "${BERT_FEATURES_DIR}" ]; then
  RIDGE_ARGS+=(--bert_features_dir "${BERT_FEATURES_DIR}")
fi
if [ -n "${HF_REPO}" ] && [ -z "${BERT_FEATURES_DIR}" ]; then
  RIDGE_ARGS+=(--hf-repo "${HF_REPO}")
fi
if [ -n "${MAX_VOXELS:-}" ]; then
  RIDGE_ARGS+=(--max_voxels "${MAX_VOXELS}")
fi
RIDGE_ARGS+=(--voxel_chunk "${VOXEL_CHUNK}" --mask_col_chunk "${MASK_COL_CHUNK}")

python "${RIDGE_ARGS[@]}"
echo "=== $(date)  done  -> ${OUTPUT_DIR}/ridge_bert_${VARIANT}_${SUBJECT}.pkl ==="
