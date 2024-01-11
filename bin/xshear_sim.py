#!/usr/bin/env python
import os
from argparse import ArgumentParser

import fitsio
import numpy as np
import schwimmbad

task_list = ["simulate_image", "measure_dm", "measure_fpfs", "summary_fpfs"]


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


if __name__ == "__main__":
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

    if taskname.lower() == "simulate_image":
        from xshear.simulation.simulator import SimulateImage

        input_list = list(range(min_id, max_id))
        worker = SimulateImage(cmd_args.config)
        for r in pool.map(worker.run, input_list):
            pass
    elif taskname.lower() == "measure_dm":
        from xshear.simulation.measure import ProcessSimDM

        worker = ProcessSimDM(cmd_args.config)
        input_list = worker.get_sim_fname(min_id=min_id, max_id=max_id)
        for r in pool.map(worker.run, input_list):
            pass
    elif taskname.lower() == "measure_fpfs":
        from xshear.simulation.measure import ProcessSimFPFS

        worker = ProcessSimFPFS(cmd_args.config)
        input_list = worker.get_sim_fname(min_id=min_id, max_id=max_id)
        for r in pool.map(worker.run, input_list):
            pass
    elif taskname.lower() == "summary_fpfs":
        from xshear.simulation.summary import SummarySimFPFS

        worker = SummarySimFPFS(
            cmd_args.config,
            min_id=min_id,
            max_id=max_id,
            ncores=ncores,
        )
        if not os.path.isfile(worker.ofname):
            olist = pool.map(worker.run, np.arange(ncores))
            fitsio.write(worker.ofname, np.vstack(list(olist)))
        worker.display_result()
    elif taskname.lower() == "neff_fpfs":
        from xshear.simulation.neff import NeffSimFPFS

        worker = NeffSimFPFS(
            cmd_args.config,
            min_id=min_id,
            max_id=max_id,
            ncores=ncores,
        )
        outcome = np.vstack(list(pool.map(worker.run, np.arange(ncores))))
        std = np.std(outcome[:, 0]) / np.average(outcome[:, 1])
        print("std: %s" % std)
    else:
        raise ValueError("taskname cannot be set to %s, we only support %s" % (taskname, task_list))
    pool.close()
