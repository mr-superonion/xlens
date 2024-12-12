import os
from . import torque
import subprocess
import numpy as np
from astropy.table import Table


def chunk_list(lst, num_chunks):
    lst = lst.astype("str")
    # Calculate the base size of each chunk
    chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks

    chunks = []
    start = 0

    for i in range(num_chunks):
        # Distribute the remainder elements across the first 'remainder' chunks
        end = min(start + chunk_size + (1 if i < remainder else 0), len(lst))
        chunks.append(",".join(list(lst[start:end])))
        start = end

    return chunks


butler_setup_script = '''
butler create .
butler register-skymap ./ --config-file skymap.py
butler register-instrument ./ lsst.obs.subaru.HyperSuprimeCam
'''

sky_map_script = '''
config.name = "hsc"
config.skyMap = "rings"

# Configuration for RingsSkyMap
config.skyMap["rings"].numRings = 120
config.skyMap["rings"].projection = "TAN"
config.skyMap["rings"].tractOverlap = 1.0 / 60
config.skyMap["rings"].pixelScale = 0.168  # arcsec/pixel
'''


def run(
    *,
    server_name,
    num_nodes,
    tasks_per_node,
    input_collection,
    output_collection,
    skymap_name,
    config_file_name,
    walltime,
    worker_init,
    tract_file_name,
):
    config = {}
    if server_name == "gw":
        node_list = torque.gw_node_list
        func = torque.submit_job
    else:
        raise ValueError("server name is wrong")

    if num_nodes > 20:
        raise ValueError("Cannot use more than 10 nodes")

    if not os.path.isfile(tract_file_name):
        raise ValueError("Cannot find the file for tract list")

    tracts_all = chunk_list(
        np.unique(Table.read(tract_file_name)["tract"]),
        num_chunks=num_nodes,
    )

    root_dir = os.getcwd()
    config.update({"tasks_per_node": tasks_per_node})
    config.update({"input_collection": input_collection})
    config.update({"output_collection": output_collection})
    config.update({"skymap_name": skymap_name})
    config_file_name = os.path.join(root_dir, config_file_name)
    config.update({"config_file_name": config_file_name})
    config.update({"walltime": walltime})
    config.update({"worker_init": worker_init})

    for inode, node_name in enumerate(node_list):
        if inode >= num_nodes:
            break
        if not os.path.isdir(f"group{inode}"):
            os.makedirs(f"group{inode}", exist_ok=True)
        os.chdir(f"group{inode}")
        base_dir = os.getcwd()

        tract_list = tracts_all[inode]
        config.update({"base_dir": base_dir})
        config.update({"tract_list": tract_list})
        config.update({"jobname": node_name})
        config.update({"node_name": node_name})

        if not os.path.isfile("butler.yaml"):
            with open("butler_setup.sh", "w") as f:
                f.write(butler_setup_script)

            with open("skymap.py", "w") as f:
                f.write(sky_map_script)

            result = subprocess.run(
                ["bash", "butler_setup.sh"],
                capture_output=True,
                text=True,
                check=True
            )
            assert result.returncode == 0

        if not os.path.isdir("submit/sh/"):
            os.makedirs("submit/sh/", exist_ok=True)

        if not os.path.isdir("submit/logs/"):
            os.makedirs("submit/logs/", exist_ok=True)
        func(config)
        os.chdir(root_dir)
    return
