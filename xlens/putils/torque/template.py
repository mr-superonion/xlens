template_string = '''#!/bin/bash

#PBS -S /bin/bash
#PBS -N ${jobname}
#PBS -m n
#PBS -q mini50
#PBS -l nodes=${node_name}:ppn=${tasks_per_node}
#PBS -l mem=240gb
#PBS -l walltime=${walltime}
#PBS -o ${base_dir}/submit/logs/${jobname}.submit.stdout
#PBS -e ${base_dir}/submit/logs/${jobname}.submit.stderr

${worker_init}

export JOBNAME="${jobname}"

cd ${base_dir}

pipetask run -b ./ -j ${tasks_per_node} -i ${input_collection} -o ${output_collection} -p ${config_file_name} -d "skymap='${skymap_name}' AND tract in (${tract_list}) AND band in ('g', 'r', 'i', 'z', 'y')" --register-dataset-types --skip-existing --clobber-outputs

'''
