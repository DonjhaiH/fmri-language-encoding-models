#!/bin/bash
#SBATCH --account=YOUR_ALLOCATION
#SBATCH --job-name=lab3.2_bert_ex_lora
#SBATCH --cpus-per-task=5
#SBATCH --mem=20GB
#SBATCH --time=04:00:00
#SBATCH --partition=GPU-shared
#SBATCH --gpus=1
#SBATCH -o lab3.2_slurm-lora-extract-%j.log
#SBATCH -e lab3.2_slurm-lora-extract-%j.log

set -e
# Path to the saved LoRA adapter directory (output of lab3.2_sbatch_lora.sh)
CODE_DIR="${HOME}/lab-3-group-15/code"
LORA_PATH="${HOME}/bert_lora_out"
OUT_DIR="${HOME}/bert_features"

# Fail early if LoRA training hasn't been run yet
if [[ ! -f "${LORA_PATH}/adapter_config.json" ]]; then
  echo "ERROR: LoRA adapter not found: ${LORA_PATH}/adapter_config.json"
  echo "       Run lab3.2_sbatch_lora.sh first and wait for it to complete."
  exit 1
fi

mkdir -p "${OUT_DIR}"

module load anaconda3
CONDA_ENV="${CONDA_ENV:-stat214}"
source activate "${CONDA_ENV}"

cd "${CODE_DIR}" || { echo "ERROR: CODE_DIR not found: ${CODE_DIR}"; exit 1; }
export PYTHONUNBUFFERED=1
export HF_HOME="${HOME}/hf_cache"
mkdir -p "${HF_HOME}"

echo "=== $(date)  JOB=${SLURM_JOB_ID}  extract BERT (LoRA) ==="
echo "    LORA_PATH=${LORA_PATH}"
echo "    OUT_DIR=${OUT_DIR}"
nvidia-smi
python -u lab3.2_extract_bert_features.py \
  --variant lora \
  --lora_path "${LORA_PATH}" \
  --output_dir "${OUT_DIR}"
echo "=== $(date)  done  -> ${OUT_DIR}/X_tr_bert_lora.pkl  X_tst_bert_lora.pkl ==="
