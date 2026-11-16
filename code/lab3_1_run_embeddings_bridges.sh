#!/bin/bash

#SBATCH --account=YOUR_ALLOCATION       # don't change this
#SBATCH --job-name=lab3_embeddings
#SBATCH --partition=RM-shared
#SBATCH --cpus-per-task=32          # GPU-shared allows max 5 cpus per GPU
#SBATCH --time 002:10:00
#SBATCH -o embed.out
#SBATCH -e embed.err

echo "Job ID: $SLURM_JOB_ID"
echo "Running in directory: $(pwd)"
echo "Time: $(date)"

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
module load anaconda3
conda activate env_214

python lab-3-group-15/code/generate_embeddings.py
