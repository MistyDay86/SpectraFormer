#!/bin/bash
#SBATCH --job-name=spectraformer
#SBATCH --partition=gpuv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Activate environment
source .venv/bin/activate

# Run training with defaults (multi-gpu on 4 GPUs)
python train_script.py
