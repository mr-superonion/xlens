#!/bin/bash
#SBATCH --job-name=shear
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -o %u/log/shear_%x_%j.out
#SBATCH -e %u/log/shear_%x_%j.err

mkdir -p "$HOME/log"
source /global/u2/x/xiangchl/ana/setup_descwl_shear_sim.sh

# change the script path
script=/pscratch/sd/x/xiangchl/code/image/xlens/examples/tests_run/sim.py

for mode in 40 0; do
    srun --mpi=pmix -n 1024 python "$script" \
      --mode "$mode" \
      --rot 0 \
      --shear 0.02 \
      --kappa 0.0 \
      --target g1 \
      --start 0 \
      --end 30 \
      --layout random
done
