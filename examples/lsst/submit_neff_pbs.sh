#!/usr/bin/env bash
#PBS -l nodes=hpcs005:ppn=28+hpcs006:ppn=28+hpcs007:ppn=28+hpcs008:ppn=28+hpcs009:ppn=28+hpcs010:ppn=28+hpcs011:ppn=28+hpcs012:ppn=28+hpcs013:ppn=28+hpcs014:ppn=28
#PBS -l walltime=48:00:00
#PBS -N neff
#PBS -q large
#PBS -o outcome/xsub-out-neff11
#PBS -e outcome/xsub-err-neff11

source /work/xiangchong.li/setupIm.sh
source impt_config
export min_id=0
export max_id=500
cd /work/xiangchong.li/work/image_tests/lsst/neff/

export config_file=config_neff.ini

# Measure galaxy and store catalog to disk
mpirun -np 280 --bind-to core xlens_sim.py measure_fpfs --config ./$config_file --min_id $min_id --max_id $max_id --mpi &&
# Measure effective galaxy number density
mpirun -np 280 --bind-to core xlens_sim.py neff_fpfs --config ./$config_file --min_id $min_id --max_id $max_id --mpi
