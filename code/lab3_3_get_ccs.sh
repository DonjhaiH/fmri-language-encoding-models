#!/bin/bash

#SBATCH --job-name=voxel_ccs
#SBATCH --partition=high
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/scratch/path/to/home/logs/voxel_ccs_%j.out
#SBATCH --error=/scratch/path/to/home/logs/voxel_ccs_%j.err

# set up environment
source ~/.bashrc
conda activate env_214

# run code in correct directory
cd ~/lab-3-group-15
python code/get_voxel_ccs.py