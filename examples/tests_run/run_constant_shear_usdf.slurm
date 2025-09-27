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
export PSCRATCH="$SCRATCH"
# change the script path
script=/sdf/home/x/xiangchl/code/xlens/examples/tests_run/constant_shear.py

for mode in 1 0; do
    mpirun -n 600 python "$script" \
      --mode "$mode" \
      --rot 0 \
      --shear 0.02 \
      --kappa 0.0 \
      --target g1 \
      --start 0 \
      --end 50 \
      --layout grid
done
