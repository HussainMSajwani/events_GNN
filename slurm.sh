#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=52
#SBATCH --job-name=gnn
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuin0009
#SBATCH --output=results/%a/gnn.out
#SBATCH --error=results/%a/gnn.err
#SBATCH --array=2-2


module purge
module load miniconda/3
conda activate gnn_113
which python
python run.py 2
