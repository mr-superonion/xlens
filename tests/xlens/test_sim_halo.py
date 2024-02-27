import os

import fitsio
import numpy as np

from xlens.simulation.simulator.base import SimulateImageHalo

this_dir = os.path.dirname(os.path.abspath(__file__))


def test_halo():
    config_fname = os.path.join(this_dir, "./config1_halo.ini")
    worker1 = SimulateImageHalo(config_fname)
    worker1.run(0)
    return


if __name__ == "__main__":
    test_halo()
