#!/usr/bin/env bash

#SBATCH --job-name=shear-summary
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=256
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -o %u/log/shear_summary_%x_%j.out
#SBATCH -e %u/log/shear_summary_%x_%j.err

mkdir -p "$HOME/log"
source /global/u2/x/xiangchl/ana/setup_descwl_shear_sim.sh

script=/pscratch/sd/x/xiangchl/code/image/xlens/examples/tests_run/summary.py

# Configure the group range (each group spans 100 sim_seeds)
GROUP_START=0
GROUP_END=3000

LAYOUT=random
TARGET=g1
SHEAR=0.02
EMAX=0.30
FLUX_MINS="20,40,60"

srun --mpi=pmix -n 1024 python "$script" \
  --layout "$LAYOUT" \
  --target "$TARGET" \
  --shear "$SHEAR" \
  --emax "$EMAX" \
  --flux-mins "$FLUX_MINS" \
  --group-start "$GROUP_START" \
  --group-end "$GROUP_END"

srun --mpi=pmix -n 1 python "$script" \
  --summary \
  --layout "$LAYOUT" \
  --target "$TARGET" \
  --shear "$SHEAR" \
  --emax "$EMAX" \
  --flux-mins "$FLUX_MINS" \
  --group-start "$GROUP_START" \
  --group-end "$GROUP_END"
