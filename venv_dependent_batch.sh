#!/bin/bash
#SBATCH --job-name=run-experiments
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00
#SBATCH --exclusive

source venv/bin/activate
python main.py
deactivate