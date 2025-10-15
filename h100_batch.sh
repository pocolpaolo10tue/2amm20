#!/bin/bash
#SBATCH --job-name=run-experiments
#SBATCH --output=%x_%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_h100
#SBATCH --time=00:30:00

module load 2023
module load foss/2023a
module load CUDA/12.1.1

rm -rf venv
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -c "import llama_cpp; print('llama_cpp loaded')"

python main.py
deactivate
