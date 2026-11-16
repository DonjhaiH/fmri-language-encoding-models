#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=lab3.2_bert_ex_pt
#SBATCH --cpus-per-task=5
#SBATCH --mem=20GB
#SBATCH --time=04:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
#SBATCH -o lab3.2_slurm-pt-%j.log
#SBATCH -e lab3.2_slurm-pt-%j.log

set -e
CODE_DIR="${HOME}/lab-3-group-15/code"
OUT_DIR="${HOME}/bert_features"

mkdir -p "${OUT_DIR}"

module load anaconda3
CONDA_ENV="${CONDA_ENV:-stat214}"
source activate "${CONDA_ENV}"

cd "${CODE_DIR}" || { echo "ERROR: CODE_DIR not found: ${CODE_DIR}"; exit 1; }
export PYTHONUNBUFFERED=1
# BERT Hub cache (~1 GB) under $HOME; override with export HF_HOME=... before sbatch if needed.
export HF_HOME="${HOME}/hf_cache"
mkdir -p "${HF_HOME}"

echo "=== $(date)  JOB=${SLURM_JOB_ID}  extract BERT (pretrained) ==="
echo "    OUT_DIR=${OUT_DIR}"
nvidia-smi
python -u lab3.2_extract_bert_features.py \
  --variant pt \
  --output_dir "${OUT_DIR}"
echo "=== $(date)  done  -> ${OUT_DIR}/X_tr_bert_pt.pkl  X_tst_bert_pt.pkl ==="
