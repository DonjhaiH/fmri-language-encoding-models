#!/bin/bash
#SBATCH --job-name=shap_fmri
#SBATCH --partition=jsteinhardt
#SBATCH --nodelist=sunstone
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/path/to/home/logs/shap_%j.out
#SBATCH --error=/scratch/path/to/home/logs/shap_%j.err


cd /path/to/fmri-language-encoding-models
/scratch/path/to/home/conda/envs/env_214/bin/python code/shap_lime.py
``