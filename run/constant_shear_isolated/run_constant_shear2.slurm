#!/bin/bash

#SBATCH --partition=roma
#SBATCH --job-name=test
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.txt

#SBATCH --nodes=5
#SBATCH --ntasks=600
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-08:00:00

# Optional: remove if not using GPUs
#SBATCH --gpus=0

source /sdf/home/x/xiangchl/setupIm.sh
mpirun -n 600 python /sdf/scratch/users/x/xiangchl/isolated/constant_shear.py --mode 0 --shear 0.02 --kappa 0.0 --rot 0 --root constant_shear_g1
