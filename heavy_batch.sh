#!/bin/bash
#SBATCH --job-name=run-experiments
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00

module load 2023
module load foss/2023a
module load CUDA/12.1.1

which nvcc
nvcc --version

source venv/bin/activate
python main.py
deactivate