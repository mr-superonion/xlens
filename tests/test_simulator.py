import galsim
import lsst.geom as geom
import numpy as np
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig

import xlens

config = RingsSkyMapConfig()
config.numRings = 120
config.projection = "TAN"
config.tractOverlap = 1.0 / 60  # degrees
config.pixelScale = 0.168  # arcsec/pixel (HSC)
skymap = RingsSkyMap(config)

config = RingsSkyMapConfig()
config.patchInnerDimensions = [501, 501]
config.tractOverlap = 0.0
config.patchBorder = 0        # pixels
config.numRings = 7000
config.pixelScale = 0.2
config.projection = "TAN"
# build it
skymap0 = RingsSkyMap(config=config)


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

    skypos = wcs_galsim.toWorld(galsim.PositionD(x=0, y=0))
    skypos2 = wcs_dm.pixelToSky(geom.Point2D(0, 0))
    np.testing.assert_almost_equal(
        skypos2.getRa().asDegrees(), skypos.ra.deg, decimal=5,
    )
    np.testing.assert_almost_equal(
        skypos2.getDec().asDegrees(), skypos.dec.deg, decimal=5,
    )

    skypos3 = wcs_0.pixelToSky(geom.Point2D(0, 0))
    np.testing.assert_almost_equal(
        skypos3.getRa().asDegrees(), skypos.ra.deg, decimal=5,
    )
    np.testing.assert_almost_equal(
        skypos3.getDec().asDegrees(), skypos.dec.deg, decimal=5,
    )

    J = wcs_galsim.local(
        galsim.PositionD(35530.338170137315, 19914.25926115947)
    ).getMatrix() / 3600.0 / 180 * np.pi
    J_target = np.array(
        [[-8.14242694e-07, -1.11992124e-08], [-1.12168502e-08, 8.14324975e-07]]
    )
    np.testing.assert_almost_equal(J, J_target, decimal=5)
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


def test_galaxies_init():
    # Set up the configuration
    tract_info = skymap[16012]
    rng = np.random.RandomState(0)
    catalog = xlens.simulator.galaxies.CatSim2017Catalog(
        rng=rng,
        tract_info=tract_info,
        layout_name="random",
    )
    arr = catalog.data
    catalog2 = xlens.simulator.galaxies.CatSim2017Catalog.from_array(
        table=arr,
        tract_info=tract_info,
    )
    np.testing.assert_almost_equal(
        catalog.data["angles"],
        catalog2.data["angles"],
    )
    np.testing.assert_almost_equal(
        catalog.data["indices"],
        catalog2.data["indices"],
    )
    np.testing.assert_almost_equal(
        catalog.data["dx"],
        catalog2.data["dx"],
    )
    np.testing.assert_almost_equal(
        catalog.data["dy"],
        catalog2.data["dy"],
    )
    return


def test_galaxies_draw():
    from xlens.simulator.perturbation import ShearHalo, ShearRedshift

    # Set up the configuration
    slist = [
        ShearRedshift(
            mode=0,
            g_dist="g1",
            shear_value=0.02,
            z_bounds=[0.0, 20.0],
            kappa_value=0.0,
        ),
        ShearHalo(
            mass=1e15,
            conc=4,
            z_lens=0.1,
            no_kappa=False,
        )
    ]
    for ii, tract_id in enumerate([0, 1200]):
        tract_info = skymap0[tract_id]
        rng = np.random.RandomState(0)
        catalog = xlens.simulator.galaxies.CatSim2017Catalog(
            rng=rng,
            tract_info=tract_info,
            layout_name="random",
        )
        psf_fwhm = 0.8
        psf_galsim = galsim.Moffat(fwhm=psf_fwhm, beta=2.5)
        gal_data1 = catalog.draw(
            patch_id=0,
            psf_obj=psf_galsim,
            mag_zero=30,
            band="i"
        )
        catalog.rotate(np.pi / 2.0)
        gal_data2 = catalog.draw(
            patch_id=0,
            psf_obj=psf_galsim,
            mag_zero=30,
            band="i"
        )
        assert np.average(np.abs(gal_data1 - np.rot90(gal_data2))) < 1e-4
        shear_obj = slist[ii]
        catalog.lens(shear_obj)
        catalog.draw(
            patch_id=0,
            psf_obj=psf_galsim,
            mag_zero=30,
            band="i"
        )
    return


def test_sim_task():
    from xlens.simulator.catalog import (
        CatalogShearTask,
        CatalogShearTaskConfig,
    )
    config = CatalogShearTaskConfig()
    cattask = CatalogShearTask(config=config)
    catalog = cattask.run(
        tract_info=skymap0[0],
        seed=0,
    ).truthCatalog

    from xlens.simulator.sim import MultibandSimConfig, MultibandSimTask
    config = MultibandSimConfig()
    simtask = MultibandSimTask(config=config)
    out = simtask.run(
        tract_info=skymap0[0],
        patch_id=0,
        band="i",
        seed=0,
        truthCatalog=catalog,
    )
    assert np.sum(out.simExposure.image.array) > 0.0
    return
