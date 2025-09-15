import xlens
import numpy as np
import lsst.geom as geom
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig

config = RingsSkyMapConfig()
config.numRings = 120
config.projection = "TAN"
config.tractOverlap = 1.0 / 60  # degrees
config.pixelScale = 0.168  # arcsec/pixel (HSC)
skymap = RingsSkyMap(config)


def test_wcs():
    # Set up the configuration
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


def test_layout():
    # Set up the configuration
    tract_info = skymap[16012]
    wcs = tract_info.getWcs()
    bbox = tract_info.getBBox()
    layout = xlens.simulator.layout.Layout(
        layout_name="random",
        wcs=wcs,
        boundary_box=bbox,
    )

    width = float(bbox.getWidth())
    height = float(bbox.getHeight())
    # Square dimension with 20″ padding on each side
    pad_pix = 20.0 / layout._pixscale_arcsec
    dim_pix = max(width, height) + 2.0 * pad_pix
    dim_pix = int(np.ceil(dim_pix))
    # print(dim_pix)

    np.testing.assert_almost_equal(
        layout._pixscale_arcsec, 0.168
    )
    assert layout._name == "random"
    assert layout._dim_pixels == dim_pix

    patch_info = tract_info[0]
    bbox = patch_info.getOuterBBox()
    layout = xlens.simulator.layout.Layout(
        layout_name="random",
        wcs=wcs,
        boundary_box=bbox,
    )

    width = float(bbox.getWidth())
    height = float(bbox.getHeight())
    # Square dimension with 20″ padding on each side
    pad_pix = 20.0 / layout._pixscale_arcsec
    dim_pix = max(width, height) + 2.0 * pad_pix
    dim_pix = int(np.ceil(dim_pix))
    # print(dim_pix)

    np.testing.assert_almost_equal(
        layout._pixscale_arcsec, 0.168
    )
    assert layout._name == "random"
    assert layout._dim_pixels == dim_pix
    return


def test_galaxies():
    # Set up the configuration
    tract_info = skymap[16012]
    wcs = tract_info.getWcs()
    bbox = tract_info.getBBox()
    layout = xlens.simulator.layout.Layout(
        layout_name="random",
        wcs=wcs,
        boundary_box=bbox,
        sep_arcsec=None,
        pad_arcsec=10.0,
    )

    rng = np.random.RandomState(0)
    catalog = xlens.simulator.galaxies.CatSim2017Catalog(
        rng=rng,
        layout=layout,
    )
    arr = catalog.to_array()
    catalog2 = xlens.simulator.galaxies.CatSim2017Catalog.from_array(table=arr)
    np.testing.assert_almost_equal(
        catalog.angles,
        catalog2.angles,
    )
    np.testing.assert_almost_equal(
        catalog.indices,
        catalog2.indices,
    )
    np.testing.assert_almost_equal(
        catalog.shifts_array["dx"],
        catalog2.shifts_array["dx"],
    )
    np.testing.assert_almost_equal(
        catalog.shifts_array["dy"],
        catalog2.shifts_array["dy"],
    )
    return
