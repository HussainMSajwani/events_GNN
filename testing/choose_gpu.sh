#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=52
#SBATCH --job-name=gnn
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuin0009
#SBATCH --output=gpu.out
#SBATCH --error=gpu.err


module purge

python choose_gpu.py
