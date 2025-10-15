#!/bin/bash
#SBATCH --job-name=run-experiments
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_a100
#SBATCH --time=00:30:00
#SBATCH --exclusive

module load foss/2023a
module load Python/3.9.16-foss-2023a
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

which nvidia-smi
nvidia-smi


source venv/bin/activate
python main.py
deactivate