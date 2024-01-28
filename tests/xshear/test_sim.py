import os

import fitsio
import numpy as np

from xshear.simulation.loader import MakeDMExposure
from xshear.simulation.measure import ProcessSimDM, ProcessSimFPFS, utils
from xshear.simulation.neff import NeffSimFPFS
from xshear.simulation.simulator import SimulateImage
from xshear.simulation.summary import SummarySimFPFS

this_dir = os.path.dirname(os.path.abspath(__file__))


def test_lsst():
    config_fname = os.path.join(this_dir, "./config1.ini")
    worker1 = SimulateImage(config_fname)
    worker1.run(0)

    # File name
    worker2 = MakeDMExposure(config_fname)
    fname_list = worker2.get_sim_fnames(min_id=0, max_id=1)
    assert len(fname_list) == 3

    # pixel scale
    exposure = worker2.run(fname_list[0])
    pixel_scale = exposure.getWcs().getPixelScale().asArcseconds()
    np.testing.assert_almost_equal(pixel_scale, 0.2)

    # Variance
    masked_image = exposure.getMaskedImage()
    variance = np.average(masked_image.variance.array)
    np.testing.assert_almost_equal(variance, 0.0478065, decimal=5)

    # PSF
    psf_array = utils.get_psf_array(exposure, ngrid=64)
    _name = os.path.join(this_dir, "psf_lsst.fits")
    psf_target = fitsio.read(_name)
    np.testing.assert_allclose(psf_array, psf_target, atol=1e-5, rtol=1e-3)

    # FPFS measurement
    worker3 = ProcessSimFPFS(config_fname)
    input_list = worker3.get_sim_fnames(min_id=0, max_id=1)
    for _ in input_list:
        worker3.run(_)

    worker4 = ProcessSimDM(config_fname)
    for _ in input_list:
        worker4.run(_)

    worker5 = SummarySimFPFS(
        config_fname,
        min_id=0,
        max_id=1,
        ncores=1,
    )
    olist = worker5.run(0)

    worker6 = NeffSimFPFS(
        config_fname,
        min_id=0,
        max_id=1,
        ncores=1,
    )
    worker6.run(0)
    worker6.clear_all()
    return
