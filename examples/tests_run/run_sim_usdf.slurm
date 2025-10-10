#!/bin/bash

#SBATCH --partition=roma
#SBATCH --job-name=test
#SBATCH -o %u/log/shear_%x_%j.out
#SBATCH -e %u/log/shear_%x_%j.err

#SBATCH --nodes=5
#SBATCH --ntasks=600
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-10:00:00

# Optional: remove if not using GPUs
#SBATCH --gpus=0


mkdir -p "$HOME/log"

source /sdf/home/x/xiangchl/setupIm.sh

# change the script path
script=/sdf/home/x/xiangchl/code/xlens/examples/tests_run/sim.py

for mode in 40 0; do
    mpirun -n 600 python "$script" \
      --mode "$mode" \
      --rot 0 \
      --shear 0.02 \
      --kappa 0.0 \
      --target g1 \
      --start 0 \
      --end 50 \
      --layout random
done
