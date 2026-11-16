#!/bin/bash

#SBATCH --job-name=lab3_embeddings
#SBATCH --partition=high
#SBATCH --cpus-per-task=8
#SBATCH --time=02:10:00
#SBATCH --output=/path/to/home/embed.out
#SBATCH --error=/path/to/home/embed.err

echo "Job ID: $SLURM_JOB_ID"
echo "Running in directory: $(pwd)"
echo "Time: $(date)"

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate env_214

cd /path/to/home
python lab-3-group-15/code/generate_embeddings.py
