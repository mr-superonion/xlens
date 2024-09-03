#!/usr/bin/env python
import os
import sys
from argparse import ArgumentParser

import numpy as np
import schwimmbad

from xlens.processor.fpfs_fastsim import FpfsSimConfig, FpfsSimTask
from xlens.simulator.fastsim import SimFShearConfig, SimFShearTask

# from memory_profiler import profile

task_list = [
    "simulate_image",
    "measure_fpfs",
]


def get_processor_count(pool, args):
    if isinstance(pool, schwimmbad.MPIPool):
        # MPIPool
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_size() - 1
    elif isinstance(pool, schwimmbad.MultiPool):
        # MultiPool
        return args.n_cores
    else:
        # SerialPool
        return 1


# @profile
def run(pool, cmd_args, taskname, min_id, max_id, ncores):
    input_list = list(range(min_id, max_id))
    nn = len(input_list)
    if taskname.lower() == "simulate_image":
        config = SimFShearConfig()
        config.load(cmd_args.config)
        worker = SimFShearTask(config=config)
        for _ in pool.map(worker.run, input_list):
            pass
    elif taskname.lower() == "measure_fpfs":
        config = FpfsSimConfig()
        config.load(cmd_args.config)
        worker = FpfsSimTask(config=config)
        res_list = pool.map(worker.run, input_list)
        result = np.swapaxes(np.stack(res_list), 0, 1)
        print(result.shape)
        r = np.average(result[2])
        print(
            np.average(result[0]) / r - 1, np.std(result[0]) / r / np.sqrt(nn)
        )
        print(np.average(result[1]) / r, np.std(result[1]) / r / np.sqrt(nn))
        # print(result[0] / r - 1)
        # print(result[1] / r)
        # print(result[2] / r)

    elif taskname.lower() == "summary_fpfs":
        import fitsio

        from xlens.simulation.summary import SummarySimFpfs

        worker = SummarySimFpfs(
            cmd_args.config,
            min_id=min_id,
            max_id=max_id,
            ncores=ncores,
        )
        if not os.path.isfile(worker.ofname):
            olist = pool.map(worker.run, np.arange(ncores))
            fitsio.write(worker.ofname, np.vstack(list(olist)))
        worker.display_result()
    else:
        raise ValueError(
            "taskname cannot be set to %s, we only support %s"
            % (taskname, task_list)
        )
    pool.close()
    sys.exit(0)
    return


# @profile
def main():
    parser = ArgumentParser(description="simulate blended images")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name fo the task to run.",
    )
    parser.add_argument(
        "--min_id",
        default=0,
        type=int,
        help="minimum simulation id number, e.g. 0",
    )
    parser.add_argument(
        "--max_id",
        default=5000,
        type=int,
        help="maximum simulation id number, e.g. 4000",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="configuration file name",
    )
    #
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--ncores",
        dest="n_cores",
        default=1,
        type=int,
        help="Number of processes (uses multiprocessing).",
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )
    cmd_args = parser.parse_args()
    min_id = cmd_args.min_id
    max_id = cmd_args.max_id
    pool = schwimmbad.choose_pool(mpi=cmd_args.mpi, processes=cmd_args.n_cores)
    ncores = get_processor_count(pool, cmd_args)
    taskname = cmd_args.task_name
    run(
        pool=pool,
        cmd_args=cmd_args,
        taskname=taskname,
        min_id=min_id,
        max_id=max_id,
        ncores=ncores,
    )
    return


if __name__ == "__main__":
    main()
