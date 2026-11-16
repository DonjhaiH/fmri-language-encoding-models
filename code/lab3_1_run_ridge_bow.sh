#!/bin/bash
#SBATCH --job-name=ridge_bow_s2
#SBATCH --output=/scratch/path/to/home/logs/ridge_bow_s2_%j.out
#SBATCH --error=/scratch/path/to/home/logs/ridge_bow_s2_%j.err
#SBATCH --partition=high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=8:00:00

source ~/.bashrc
conda activate stat214
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python -u /path/to/fmri-language-encoding-models/code/fit_ridge_bow.py \
    --x_tr    /scratch/path/to/home/train_test_pkl/X_tr_bow.pkl \
    --x_tst   /scratch/path/to/home/train_test_pkl/X_tst_bow.pkl \
    --bold_dir /path/to/data \
    --out_dir  /scratch/path/to/home/results \
    --subject  subject2