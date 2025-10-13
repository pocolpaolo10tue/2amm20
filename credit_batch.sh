#!/bin/bash
#SBATCH --job-name=run-experiments
#SBATCH --output=%x_%j.out
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=02:00:00
#SBATCH --reservation=terv92681


# Load CUDA module (adjust version to match your system)
deactivate
rm -rf venv

module purge
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

python --version
python -m venv venv

source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python main.py