#!/bin/bash
#SBATCH --job-name=run-experiments
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=15:00
#SBATCH --partition=gpu_mig
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