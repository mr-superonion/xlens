import os

import astropy.table as astable
import numpy as np

from xlens.simulation.simulator.base import SimulateImageHalo

this_dir = os.path.dirname(os.path.abspath(__file__))


def test_halo():
    # Prepare a halo
    src_halo = astable.Table()
    src_halo["index"] = np.array([1, 2])
    src_halo["mass"] = np.array([4e14, 8e14])
    src_halo["conc"] = np.array([6.0, 4.0])
    src_halo["z_lens"] = np.array([0.2, 0.52])
    config_fname = os.path.join(this_dir, "./config1_halo.ini")
    worker1 = SimulateImageHalo(config_fname)
    worker1.run(ifield=0, src_halo=src_halo[0])
    worker1.clear_all()
    return


if __name__ == "__main__":
    test_halo()
