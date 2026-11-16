#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=lab3.2_bert_lora
#SBATCH --cpus-per-task=5
#SBATCH --mem=20GB
#SBATCH --time=08:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
#SBATCH -o lab3.2_slurm-lora-train-%j.log
#SBATCH -e lab3.2_slurm-lora-train-%j.log

set -e
CODE_DIR="${HOME}/lab-3-group-15/code"
OUT_DIR="${HOME}/bert_lora_out"

mkdir -p "${OUT_DIR}"

module load anaconda3
CONDA_ENV="${CONDA_ENV:-stat214}"
source activate "${CONDA_ENV}"

cd "${CODE_DIR}" || { echo "ERROR: CODE_DIR not found: ${CODE_DIR}"; exit 1; }
export PYTHONUNBUFFERED=1
export HF_HOME="${HOME}/hf_cache"
mkdir -p "${HF_HOME}"

echo "=== $(date)  JOB=${SLURM_JOB_ID}  train LoRA (base=bert-base-uncased) ==="
echo "    CODE_DIR=${CODE_DIR}"
echo "    OUT_DIR=${OUT_DIR}"
nvidia-smi
python -u lab3.2_lora_bert.py \
  --output_dir "${OUT_DIR}" \
  --epochs 5 \
  --batch_size 32 \
  --save_steps 500 \
  --fp16
echo "=== $(date)  done  LoRA adapter -> ${OUT_DIR} ==="
