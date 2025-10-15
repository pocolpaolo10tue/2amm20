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

# Remove broken venv (optional)
rm -rf venv

# Recreate venv on this node
python -m venv venv
source venv/bin/activate

# Install dependencies
export CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=80;90"
pip install --upgrade pip
pip install llama-cpp-python
pip install -r requirements.txt   # if you have other dependencies

# Test
python -c "import llama_cpp; print('llama_cpp loaded')"

# Run main script
python main.py
deactivate
