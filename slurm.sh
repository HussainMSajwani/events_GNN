#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=gnn
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuin0009
#SBATCH --output=results/%a/gnn.out
#SBATCH --error=results/%a/gnn.err
#SBATCH --array=1-4


module purge

python run.py $SLURM_ARRAY_TASK_ID
