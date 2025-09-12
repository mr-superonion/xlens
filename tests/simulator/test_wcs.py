import xlens
import numpy as np
import lsst.geom as geom
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig


def test_wcs():
    # Set up the configuration
    config = RingsSkyMapConfig()
    config.numRings = 120
    config.projection = "TAN"
    config.tractOverlap = 1.0 / 60  # degrees
    config.pixelScale = 0.168  # arcsec/pixel
    skymap = RingsSkyMap(config)
    tract_info = skymap[16012]

    wcs_galsim = xlens.simulator.wcs.make_galsim_tanwcs(tract_info)
    wcs_dm = xlens.simulator.wcs.make_dm_wcs(wcs_galsim)
    wcs_0 = tract_info.getWcs()

    np.testing.assert_almost_equal(
        wcs_dm.getCdMatrix(),
        wcs_0.getCdMatrix(),
    )

    np.testing.assert_almost_equal(
        wcs_dm.getPixelOrigin().x,
        wcs_0.getPixelOrigin().x,
    )

    np.testing.assert_almost_equal(
        wcs_dm.getPixelOrigin().y,
        wcs_0.getPixelOrigin().y,
    )

    np.testing.assert_almost_equal(
        wcs_dm.getSkyOrigin().getRa().asDegrees(),
        wcs_0.getSkyOrigin().getRa().asDegrees(),
    )

    np.testing.assert_almost_equal(
        wcs_dm.getSkyOrigin().getDec().asDegrees(),
        wcs_0.getSkyOrigin().getDec().asDegrees(),
    )
    J_target = np.array([
        [-8.14327128e-07, -9.05027013e-09], [-9.06489594e-09, 8.14380525e-07]
    ])

    lin = wcs_dm.linearizePixelToSky(
        geom.Point2D(32165.0, 19905.0),
        geom.radians,
    )
    J = np.array(lin.getLinear().getMatrix(), dtype=np.float64)
    np.testing.assert_almost_equal(J, J_target)
    return
