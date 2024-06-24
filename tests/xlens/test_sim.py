import os

import fitsio
import numpy as np

from xlens.simulation.measure import ProcessSimAnacal, ProcessSimDM, utils
from xlens.simulation.neff import NeffSimFpfs
from xlens.simulation.simulator.base import SimulateImage
from xlens.simulation.simulator.loader import MakeDMExposure
from xlens.simulation.summary import SummarySimAnacal

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
    np.testing.assert_almost_equal(variance, 0.354025, decimal=5)

    # PSF
    psf_array = utils.get_psf_array(exposure, ngrid=64)
    _name = os.path.join(this_dir, "psf_lsst.fits")
    psf_target = fitsio.read(_name)
    np.testing.assert_allclose(psf_array, psf_target, atol=1e-5, rtol=1e-3)

    # Measurement
    worker3 = ProcessSimAnacal(config_fname)
    input_list = worker3.get_sim_fnames(min_id=0, max_id=1)
    for _ in input_list:
        worker3.run(_)

    worker4 = ProcessSimDM(config_fname)
    input_list = worker4.get_sim_fnames(min_id=0, max_id=1)
    for _ in input_list:
        worker4.run(_)

    worker5 = SummarySimAnacal(
        config_fname,
        min_id=0,
        max_id=1,
        ncores=1,
    )
    olist = worker5.run(0)
    del olist

    worker6 = NeffSimFpfs(
        config_fname,
        min_id=0,
        max_id=1,
        ncores=1,
    )
    # worker6.run(0)
    worker6.clear_all()
    return


def test_hsc():
    config_fname = os.path.join(this_dir, "./config2.ini")
    worker1 = SimulateImage(config_fname)
    worker1.run(0)

    # File name
    worker2 = MakeDMExposure(config_fname)
    fname_list = worker2.get_sim_fnames(min_id=0, max_id=1)
    assert len(fname_list) == 3

    # pixel scale
    exposure = worker2.run(fname_list[0])
    pixel_scale = exposure.getWcs().getPixelScale().asArcseconds()
    np.testing.assert_almost_equal(pixel_scale, 0.168)

    # # Variance
    # masked_image = exposure.getMaskedImage()
    # variance = np.average(masked_image.variance.array)
    # np.testing.assert_almost_equal(variance, 0.0478065, decimal=5)

    # PSF
    psf_array = utils.get_psf_array(exposure, ngrid=64)
    _name = os.path.join(this_dir, "psf_hsc.fits")
    del psf_array, _name

    # Measurement
    worker3 = ProcessSimAnacal(config_fname)
    input_list = worker3.get_sim_fnames(min_id=0, max_id=1)
    for _ in input_list:
        worker3.run(_)

    worker4 = ProcessSimDM(config_fname)
    for _ in input_list:
        worker4.run(_)

    worker5 = SummarySimAnacal(
        config_fname,
        min_id=0,
        max_id=1,
        ncores=1,
    )
    olist = worker5.run(0)
    del olist

    worker6 = NeffSimFpfs(
        config_fname,
        min_id=0,
        max_id=1,
        ncores=1,
    )
    # worker6.run(0)
    worker6.clear_all()
    return
