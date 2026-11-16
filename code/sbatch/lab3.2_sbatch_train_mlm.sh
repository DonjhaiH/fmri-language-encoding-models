#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=lab3.2_bert_mlm
#SBATCH --cpus-per-task=5
#SBATCH --mem=20GB
#SBATCH --time=08:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1

# Slurm -o/-e are relative to the directory from which you run sbatch (use ~/.../code).
#SBATCH -o lab3.2_slurm-mlm-%j.log
#SBATCH -e lab3.2_slurm-mlm-%j.log

set -e
CODE_DIR="${HOME}/lab-3-group-15/code"
OUT_DIR="${HOME}/bert_mlm_out"

mkdir -p "${OUT_DIR}"

module load anaconda3
# Use 'source activate' in batch shells. Override: export CONDA_ENV=214 if you use the course env name "214".
CONDA_ENV="${CONDA_ENV:-stat214}"
source activate "${CONDA_ENV}"

cd "${CODE_DIR}" || { echo "ERROR: CODE_DIR not found: ${CODE_DIR}"; exit 1; }
export PYTHONUNBUFFERED=1
export HF_HOME="${HOME}/hf_cache"
mkdir -p "${HF_HOME}"

echo "=== $(date)  JOB=${SLURM_JOB_ID}  train BERT MLM ==="
echo "    CODE_DIR=${CODE_DIR}"
echo "    OUT_DIR=${OUT_DIR}"
nvidia-smi
python -u lab3.2_train_bert_mlm.py \
  --output_dir "${OUT_DIR}" \
  --epochs 3 \
  --batch_size 16 \
  --save_steps 500 \
  --fp16
echo "=== $(date)  done  final model -> ${OUT_DIR}/final ==="
