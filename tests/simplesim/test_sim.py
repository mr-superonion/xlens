import os

import fitsio
import numpy as np

from xlens.processor.fpfs_simplesim import (
    FpfsSimpleSimConfig,
    FpfsSimpleSimTask,
)
from xlens.processor.utils import get_psf_array
from xlens.simulator.simplesim import SimpleSimShearConfig, SimpleSimShearTask

this_dir = os.path.dirname(os.path.abspath(__file__))


def test_lsst():
    bands = ["r"]
    config = SimpleSimShearConfig()
    config.survey_name = "LSST"
    config.bands = bands
    config.psf_fwhm = 0.8
    config.psf_e1 = 0.0
    config.psf_e2 = 0.0

    task = SimpleSimShearTask(config=config)
    task.run(ifield=0)
    exposure = task.get_dm_exposure(ifield=0, mode=0, rotId=0, band_list=bands)

    pixel_scale = exposure.getWcs().getPixelScale().asArcseconds()
    np.testing.assert_almost_equal(pixel_scale, 0.2)

    # Variance
    masked_image = exposure.getMaskedImage()
    variance = np.average(masked_image.variance.array)
    expect_var = config.noise_stds[bands[0]] ** 2.0
    np.testing.assert_almost_equal(variance, expect_var, decimal=5)

    lsst_bbox = exposure.getBBox()
    lsst_psf = exposure.getPsf()
    psf_array = get_psf_array(
        lsst_psf=lsst_psf,
        lsst_bbox=lsst_bbox,
        npix=64,
    )
    psf_target = fitsio.read(os.path.join(this_dir, "psf_lsst.fits"))
    np.testing.assert_allclose(psf_array, psf_target, atol=1e-5, rtol=1e-3)

    for ind in task.get_sim_id_list(min_id=0, max_id=1):
        ifield, mode, rotId = tuple(ind)
        fname = task.get_image_name(
            ifield=ifield,
            mode=mode,
            rotId=rotId,
            band=bands[0],
        )
        assert os.path.isfile(fname)

        fname = task.get_truth_src_name(ifield=ifield, mode=mode, rotId=rotId)
        assert os.path.isfile(fname)

    config2 = FpfsSimpleSimConfig()
    config2.coadd_bands = bands
    task2 = FpfsSimpleSimTask(config=config2)
    task2.run(ifield=0)

    task.clear()
    return


if __name__ == "__main__":
    test_lsst()
