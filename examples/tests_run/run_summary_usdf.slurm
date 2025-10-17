#!/usr/bin/env bash

#SBATCH --partition=roma
#SBATCH --job-name=shear-summary
#SBATCH -o %u/log/shear_summary_%x_%j.out
#SBATCH -e %u/log/shear_summary_%x_%j.err

#SBATCH --nodes=5
#SBATCH --ntasks=600
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-06:00:00

mkdir -p "$HOME/log"

source /sdf/home/x/xiangchl/setupIm.sh

script=/sdf/home/x/xiangchl/code/xlens/examples/tests_run/summary.py

GROUP_START=0
GROUP_END=3000

LAYOUT=random
TARGET=g1
SHEAR=0.02
EMAX=0.30
FLUX_MINS="20,40,60"

mpirun -n 600 python "$script" \
  --layout "$LAYOUT" \
  --target "$TARGET" \
  --shear "$SHEAR" \
  --emax "$EMAX" \
  --flux-mins "$FLUX_MINS" \
  --group-start "$GROUP_START" \
  --group-end "$GROUP_END"

python "$script" \
  --summary \
  --layout "$LAYOUT" \
  --target "$TARGET" \
  --shear "$SHEAR" \
  --emax "$EMAX" \
  --flux-mins "$FLUX_MINS" \
  --group-start "$GROUP_START" \
  --group-end "$GROUP_END"
