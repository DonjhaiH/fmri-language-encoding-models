#!/bin/bash
#SBATCH --job-name=ridge_w2v_s3
#SBATCH --output=/scratch/path/to/home/logs/ridge_w2v_s3_%j.out
#SBATCH --error=/scratch/path/to/home/logs/ridge_w2v_s3_%j.err
#SBATCH --partition=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=8:00:00

source ~/.bashrc
conda activate env_214
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# make this where ridge_utils is
export PYTHONPATH=/path/to/fmri-language-encoding-models:$PYTHONPATH

python -u /path/to/fmri-language-encoding-models/code/lab3.1_fit_ridge.py \
    --bold_dir /path/to/data \
    --out_dir  /scratch/path/to/home/results \
    --subject  subject3 \
    --embedding w2v
