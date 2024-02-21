#!/usr/bin/env bash
#PBS -l nodes=hpcs005:ppn=28+hpcs006:ppn=28+hpcs007:ppn=28+hpcs008:ppn=28+hpcs009:ppn=28+hpcs010:ppn=28+hpcs011:ppn=28+hpcs012:ppn=28+hpcs013:ppn=28+hpcs014:ppn=28
#PBS -l walltime=48:00:00
#PBS -N test
#PBS -q large
#PBS -o xsub-out-test1
#PBS -e xsub-err-test1

source /work/xiangchong.li/setupIm.sh
source impt_config
cd /work/xiangchong.li/work/image_tests/lsst/test/

export min_id=0
export max_id=5000

export config_file=./config.ini
export OMP_NUM_THREADS=1

# Simulate images and store to disk
mpirun -np 280 --bind-to core xlens_sim.py simulate_image --config ./$config_file --min_id $min_id --max_id $max_id --mpi &&
# Measure galaxy and store catalog to disk
mpirun -np 280 --bind-to core xlens_sim.py measure_fpfs --config ./$config_file --min_id $min_id --max_id $max_id --mpi &&
# Derive shear
mpirun -np 280 --bind-to core xlens_sim.py summary_fpfs --config ./$config_file --min_id $min_id --max_id $max_id --mpi
