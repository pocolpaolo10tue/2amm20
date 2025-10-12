#!/bin/bash
#SBATCH --job-name=run-experiments
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=3:00
#SBATCH --partition=gpu_mig
#SBATCH --reservation=terv92681

# Load CUDA module (adjust version to match your system)
module load 2023
module load CUDA/12.1.1

source venv/bin/activate  # activate your virtual environment

pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
python main.py