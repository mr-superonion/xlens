"""Unit tests for :class:`xlens.simulator.galaxies.ClusterSceneCatalog`.

A scene is a table of objects rendered once each at their own positions,
with a bulge + disk or single-Sersic morphology and per-band magnitudes.  The tests build
a small synthetic scene around the centre of a one-patch tract and check
the truth-catalog construction, the ``ra``/``dec`` <-> ``dx``/``dy``
round trip, rotation and halo lensing, the rendering onto an exposure
(flux, position, orientation), the consistency with the pipeline drawing
loop of ``MultibandSimTask``, the DP2 object-table conversion, and the
FITS loading from ``catsim_dir``.
"""

import os
import tempfile

import fitsio
import galsim
import lsst.afw.image as afwImage
import lsst.geom as geom
import numpy as np
import numpy.lib.recfunctions as rfn
import pytest
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig

from xlens.simulator.defaults import mag_zero_defaults, psf_fwhm_defaults
from xlens.simulator.galaxies import (
    AB_MAG_ZERO_NJY,
    ClusterSceneCatalog,
    get_catalog_class,
)
from xlens.simulator.perturbation.halo import ShearHalo
from xlens.simulator.sim import MultibandSimConfig, MultibandSimTask

# one-patch tract: 501 x 501 pixels at 0.2 arcsec
_config = RingsSkyMapConfig()
_config.patchInnerDimensions = [501, 501]
_config.tractOverlap = 0.0
_config.patchBorder = 0
_config.numRings = 7000
_config.pixelScale = 0.2
_config.projection = "TAN"
skymap0 = RingsSkyMap(config=_config)
# a tract at moderate declination: on the polar tract 0 the tangent frame
# of galsim's TanWCS is rotated relative to the LSST pixel axes, which would
# turn the orientation checks below into tests of that frame
TRACT_INFO = skymap0[skymap0.findTract(geom.SpherePoint(10.2 * geom.degrees, -44.1 * geom.degrees)).getId()]
PIXEL_SCALE = 0.2
MAG_ZERO = 30.0
Z_CLUSTER = 0.3

# dx, dy [arcsec] from the tract centre, magnitude, n, a, b, theta [deg],
# redshift, point source
SCENE_ROWS = [
    # bright member elongated along +x
    (-20.0, -15.0, 20.0, 1.0, 1.0, 0.5, 0.0, Z_CLUSTER, False),
    # the same galaxy turned to +y
    (20.0, -15.0, 20.0, 1.0, 1.0, 0.5, 90.0, Z_CLUSTER, False),
    # round de Vaucouleurs background galaxy
    (0.0, 20.0, 21.0, 4.0, 0.8, 0.8, 30.0, 1.0, False),
    # star
    (15.0, 15.0, 19.0, 1.0, 0.3, 0.3, 0.0, 0.0, True),
    # far outside the tract: kept in the truth catalog, never drawn
    (500.0, 0.0, 18.0, 1.0, 1.0, 1.0, 0.0, Z_CLUSTER, False),
]


# positions sit a little off the pixel grid: the scene is given in ra/dec, and an object
# exactly on a pixel boundary rounds its stamp to either side at the 1e-9 level of the
# sky <-> pixel round trip, depending on which WCS (DM or GalSim) does it
OFF_GRID = 0.013  # arcsec


def _scene_table():
    rows = np.array(SCENE_ROWS, dtype=object)
    table = {
        "dx": rows[:, 0].astype(float) + OFF_GRID,
        "dy": rows[:, 1].astype(float) + OFF_GRID,
        "lsst_i": rows[:, 2].astype(float),
        "lsst_r": rows[:, 2].astype(float) + 0.5,
        "sersic_n": rows[:, 3].astype(float),
        "r50_major": rows[:, 4].astype(float),
        "r50_minor": rows[:, 5].astype(float),
        "theta": rows[:, 6].astype(float),
        "redshift": rows[:, 7].astype(float),
        "is_point_source": rows[:, 8].astype(bool),
    }
    return _structured(table)


def _structured(columns):
    """Structured array from a mapping of equal-length columns."""
    columns = {name: np.asarray(values) for name, values in columns.items()}
    num = len(next(iter(columns.values())))
    out = np.empty(num, dtype=[(name, values.dtype) for name, values in columns.items()])
    for name, values in columns.items():
        out[name] = values
    return out


def _radec_scene(scene, tract_info=TRACT_INFO):
    """The scene with its ``dx``/``dy`` layout (arcsec from the tract centre, along the
    pixel axes) turned into the ``ra``/``dec`` columns a scene file carries."""
    columns = {name: np.asarray(scene[name]) for name in (scene.dtype.names if isinstance(scene, np.ndarray) else scene)}
    if "dx" in columns and "ra" not in columns:
        centre = tract_info.getBBox().getCenter()
        ra, dec = tract_info.getWcs().pixelToSkyArray(
            x=centre.getX() + columns.pop("dx") / PIXEL_SCALE,
            y=centre.getY() + columns.pop("dy") / PIXEL_SCALE,
            degrees=True,
        )
        columns = {"ra": np.asarray(ra), "dec": np.asarray(dec), **columns}
    return _structured(columns)


def _make(scene, **kwargs):
    """Write ``scene`` as the catalog FITS file of a fresh ``catsim_dir`` and read it back,
    the way the pipeline builds the catalog."""
    with tempfile.TemporaryDirectory() as catsim_dir:
        ClusterSceneCatalog.write_scene(_radec_scene(scene, kwargs.get("tract_info", TRACT_INFO)), catsim_dir)
        return ClusterSceneCatalog(catsim_dir=catsim_dir, **kwargs)


def _flux(mag):
    return 10 ** ((MAG_ZERO - mag) / 2.5)


def _psf():
    return galsim.Moffat(fwhm=0.8, beta=2.5)


def _blank_exposure():
    exposure = afwImage.ExposureF(TRACT_INFO.getBBox())
    exposure.setWcs(TRACT_INFO.getWcs())
    return exposure


def _moments(array, cx, cy, half=15):
    """Second moments of a cut-out around (cx, cy) in array indices."""
    cx, cy = int(round(cx)), int(round(cy))
    sub = array[cy - half : cy + half + 1, cx - half : cx + half + 1]
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    total = sub.sum()
    return (sub * xx**2).sum() / total, (sub * yy**2).sum() / total


def test_construct_from_offsets():
    catalog = _make(_scene_table(), tract_info=TRACT_INFO)
    data = catalog.data
    assert len(catalog) == len(SCENE_ROWS)
    for col in (
        "indices",
        "redshift",
        "angles",
        "dx",
        "dy",
        "ra",
        "dec",
        "prelensed_ra",
        "prelensed_dec",
        "has_finite_shear",
        "hlr",
        "lsst_i",
        "lsst_r",
        "sersic_n",
        "r50_major",
        "r50_minor",
        "theta",
        "is_point_source",
    ):
        assert col in data.dtype.names, col
    assert not catalog.lensed
    np.testing.assert_allclose(data["dx"], [r[0] + OFF_GRID for r in SCENE_ROWS], atol=1e-8)
    np.testing.assert_allclose(data["dy"], [r[1] + OFF_GRID for r in SCENE_ROWS], atol=1e-8)
    # the catalog layout keeps each galaxy's own orientation: no extra rotation
    np.testing.assert_array_equal(data["angles"], 0.0)
    np.testing.assert_allclose(data["theta"], [r[6] for r in SCENE_ROWS])
    np.testing.assert_allclose(data["hlr"], np.sqrt([r[4] * r[5] for r in SCENE_ROWS]))
    np.testing.assert_array_equal(data["indices"], np.arange(len(SCENE_ROWS)))
    assert data["is_point_source"].tolist() == [r[8] for r in SCENE_ROWS]

    # ra/dec are the sky positions of x_center + dx / pixel_scale
    wcs = TRACT_INFO.getWcs()
    ra, dec = wcs.pixelToSkyArray(
        x=catalog.x_center + data["dx"] / PIXEL_SCALE,
        y=catalog.y_center + data["dy"] / PIXEL_SCALE,
        degrees=True,
    )
    np.testing.assert_allclose(data["ra"], ra, rtol=0, atol=1e-10)
    np.testing.assert_allclose(data["dec"], dec, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(data["ra"], data["prelensed_ra"])

    # magnitude column lookup used by matchPipe
    assert ClusterSceneCatalog.magnitude_columns("hsc", "i") == ("lsst_i",)

    # the truth catalog rebuilds the catalog without the input table
    catalog2 = ClusterSceneCatalog.from_array(truthCatalog=data, tract_info=TRACT_INFO)
    for col in ("dx", "dy", "angles", "lsst_i", "r50_major"):
        np.testing.assert_array_equal(catalog.data[col], catalog2.data[col])
    assert catalog2.lensed


def test_construct_from_radec_round_trip():
    reference = _make(_scene_table(), tract_info=TRACT_INFO)
    table = _scene_table()
    columns = {name: table[name] for name in table.dtype.names if name not in ("dx", "dy")}
    columns["ra"] = reference.data["ra"]
    columns["dec"] = reference.data["dec"]
    catalog = _make(columns, tract_info=TRACT_INFO)
    np.testing.assert_allclose(catalog.data["dx"], reference.data["dx"], atol=1e-6)
    np.testing.assert_allclose(catalog.data["dy"], reference.data["dy"], atol=1e-6)
    np.testing.assert_allclose(catalog.data["ra"], reference.data["ra"], atol=1e-10)


def test_missing_columns_raise():
    table = _scene_table()
    columns = {name: table[name] for name in table.dtype.names if name != "sersic_n"}
    with pytest.raises(ValueError, match="sersic_n"):
        _make(columns, tract_info=TRACT_INFO)
    columns = {name: table[name] for name in table.dtype.names if name not in ("dx", "dy")}
    with pytest.raises(ValueError, match="ra, dec"):
        _make(columns, tract_info=TRACT_INFO)
    with pytest.raises(ValueError, match="euclid_"):
        _make(_scene_table(), tract_info=TRACT_INFO, survey_name_list=["euclid"])


def test_selection_and_pixel_center():
    catalog = _make(
        _scene_table(),
        tract_info=TRACT_INFO,
        select_observable=["lsst_i"],
        select_upper_limit=[20.5],
        force_pixel_center=True,
    )
    assert len(catalog) == 4
    assert np.all(catalog.data["lsst_i"] <= 20.5)
    # ``indices`` are rows of the unfiltered input
    np.testing.assert_array_equal(catalog.data["indices"], [0, 1, 3, 4])
    dx = catalog.data["dx"] / PIXEL_SCALE
    np.testing.assert_allclose(dx, np.floor(dx) + 0.5)


def test_rotate_and_lens():
    catalog = _make(_scene_table(), tract_info=TRACT_INFO)
    dx0, dy0 = catalog.data["dx"].copy(), catalog.data["dy"].copy()
    angles0 = catalog.data["angles"].copy()
    catalog.rotate(np.pi / 2)
    np.testing.assert_allclose(catalog.data["dx"], -dy0, atol=1e-9)
    np.testing.assert_allclose(catalog.data["dy"], dx0, atol=1e-9)
    np.testing.assert_allclose(catalog.data["angles"], angles0 + np.pi / 2)

    halo = ShearHalo(mass=5e14, conc=4.0, z_lens=Z_CLUSTER)
    catalog.lens(shear_obj=halo)
    assert catalog.lensed
    member = catalog.data["redshift"] <= Z_CLUSTER
    # members and the star sit at or in front of the lens: untouched
    assert np.all(catalog.data["gamma1"][member] == 0.0)
    assert np.all(catalog.data["kappa"][member] == 0.0)
    # the background galaxy is sheared
    background = ~member
    assert background.sum() == 1
    assert np.hypot(catalog.data["gamma1"][background], catalog.data["gamma2"][background]) > 1e-3
    assert np.all(catalog.data["has_finite_shear"])
    with pytest.raises(ValueError):
        catalog.rotate(0.1)


def test_draw_on_exposure():
    catalog = _make(_scene_table(), tract_info=TRACT_INFO)
    exposure = _blank_exposure()
    truth = catalog.draw_on_image(exposure, band="i", mag_zero=MAG_ZERO, psf_obj=_psf())
    array = exposure.getMaskedImage().image.array
    assert truth["drawn"].tolist() == [True, True, True, True, False]

    # every drawn object contributes its flux (stamps are well inside)
    expected = sum(_flux(r[2]) for r in SCENE_ROWS[:4])
    np.testing.assert_allclose(array.sum(), expected, rtol=2e-2)

    # the star is the brightest pixel, at its pixel position
    bbox = exposure.getBBox()
    iy, ix = np.unravel_index(np.argmax(array), array.shape)
    star = truth[3]
    assert abs(ix + bbox.getMinX() - star["image_x"]) <= 1
    assert abs(iy + bbox.getMinY() - star["image_y"]) <= 1
    np.testing.assert_allclose(star["image_x"], catalog.x_center + (15.0 + OFF_GRID) / PIXEL_SCALE)

    # orientation: theta = 0 is elongated along x, theta = 90 along y
    ixx, iyy = _moments(array, truth[0]["image_x"] - bbox.getMinX(), truth[0]["image_y"] - bbox.getMinY())
    assert ixx > 1.3 * iyy
    ixx, iyy = _moments(array, truth[1]["image_x"] - bbox.getMinX(), truth[1]["image_y"] - bbox.getMinY())
    assert iyy > 1.3 * ixx

    # drawing adds to what is already in the image
    before = array.copy()
    catalog.draw_on_image(exposure, band="i", mag_zero=MAG_ZERO, psf_obj=_psf())
    np.testing.assert_allclose(array, 2 * before, rtol=1e-6)

    # a point source with the r band, without point sources
    exposure = _blank_exposure()
    catalog.draw_on_image(exposure, band="r", mag_zero=MAG_ZERO, psf_obj=_psf(), include_point_source=False)
    expected = sum(_flux(r[2] + 0.5) for r in SCENE_ROWS[:3])
    np.testing.assert_allclose(exposure.getMaskedImage().image.array.sum(), expected, rtol=2e-2)


def test_draw_on_galsim_and_numpy_images():
    from xlens.wcs import tanwcs_dm2galsim

    catalog = _make(_scene_table(), tract_info=TRACT_INFO)
    bbox = TRACT_INFO.getBBox()
    wcs_gs = tanwcs_dm2galsim(TRACT_INFO.getWcs())
    gs_image = galsim.ImageF(
        bbox.getWidth(), bbox.getHeight(), xmin=bbox.getMinX(), ymin=bbox.getMinY(), wcs=wcs_gs
    )
    catalog.draw_on_image(gs_image, band="i", mag_zero=MAG_ZERO, psf_obj=_psf())

    array = np.zeros((bbox.getHeight(), bbox.getWidth()), dtype=np.float32)
    catalog.draw_on_image(array, band="i", mag_zero=MAG_ZERO, psf_obj=_psf(), wcs=TRACT_INFO.getWcs())
    # the tract bbox starts at (0, 0), so both agree pixel for pixel
    assert bbox.getMinX() == 0 and bbox.getMinY() == 0
    np.testing.assert_allclose(gs_image.array, array, rtol=1e-5, atol=1e-6)

    with pytest.raises(ValueError, match="wcs"):
        catalog.draw_on_image(array, band="i", mag_zero=MAG_ZERO, psf_obj=_psf())
    with pytest.raises(TypeError):
        catalog.draw_on_image(
            array.astype(np.int32), band="i", mag_zero=MAG_ZERO, psf_obj=_psf(), wcs=TRACT_INFO.getWcs()
        )


def test_matches_pipeline_drawing_loop():
    """``draw_on_image`` and ``MultibandSimTask`` render the same pixels."""
    catalog = _make(_scene_table(), tract_info=TRACT_INFO)
    config = MultibandSimConfig()
    config.galaxy_type = "cluster_scene"
    config.survey_name = "lsst"
    config.validate()
    assert get_catalog_class("cluster_scene") is ClusterSceneCatalog
    task = MultibandSimTask(config=config)
    psf = _psf()
    bbox = TRACT_INFO.getBBox()
    pipeline = task.draw_catalog(
        galaxy_catalog=catalog,
        wcs=TRACT_INFO.getWcs(),
        bbox_outer=bbox,
        psf_obj=psf,
        mag_zero=MAG_ZERO,
        band="i",
    )
    image = afwImage.ImageF(bbox)
    catalog.draw_on_image(
        image,
        band="i",
        mag_zero=MAG_ZERO,
        psf_obj=psf,
        wcs=TRACT_INFO.getWcs(),
        use_field_distortion=config.use_field_distortion,
        nn_trunc=None if config.truncate_stamp_size <= 0 else config.truncate_stamp_size,
    )
    np.testing.assert_allclose(image.array, pipeline, rtol=1e-5, atol=1e-6)


def test_sim_task_run_from_truth_catalog():
    """The full sim task renders a scene truth catalog, PSF from the exposure."""
    catalog = _make(_scene_table(), tract_info=TRACT_INFO)
    halo = ShearHalo(mass=5e14, conc=4.0, z_lens=Z_CLUSTER)
    catalog.lens(shear_obj=halo)

    config = MultibandSimConfig()
    config.galaxy_type = "cluster_scene"
    config.survey_name = "lsst"
    config.draw_image_noise = False
    task = MultibandSimTask(config=config)
    exposure = task.run(
        tract_info=TRACT_INFO, patch_id=0, band="i", seed=3, truthCatalog=catalog.data
    ).simExposure
    sim_array = exposure.getMaskedImage().image.array.copy()
    assert sim_array.sum() > 0

    # re-render with draw_on_image using the exposure's own WCS and PSF
    reference = afwImage.ExposureF(exposure.getBBox())
    reference.setWcs(exposure.getWcs())
    reference.setPsf(exposure.getPsf())
    mag_zero = mag_zero_defaults["lsst"]
    catalog.draw_on_image(
        reference,
        band="i",
        mag_zero=mag_zero,
        use_field_distortion=config.use_field_distortion,
        nn_trunc=None if config.truncate_stamp_size <= 0 else config.truncate_stamp_size,
    )
    # the pixel-sampled exposure PSF differs slightly from the analytic Moffat
    ref_array = reference.getMaskedImage().image.array
    np.testing.assert_allclose(ref_array.sum(), sim_array.sum(), rtol=1e-2)
    assert np.unravel_index(np.argmax(ref_array), ref_array.shape) == np.unravel_index(
        np.argmax(sim_array), sim_array.shape
    )
    # and exactly with the analytic PSF the task used
    reference = afwImage.ExposureF(exposure.getBBox())
    reference.setWcs(exposure.getWcs())
    catalog.draw_on_image(
        reference,
        band="i",
        mag_zero=mag_zero,
        psf_obj=galsim.Moffat(fwhm=psf_fwhm_defaults["i"]["lsst"], beta=2.5),
        use_field_distortion=config.use_field_distortion,
        nn_trunc=None if config.truncate_stamp_size <= 0 else config.truncate_stamp_size,
    )
    np.testing.assert_allclose(reference.getMaskedImage().image.array, sim_array, rtol=1e-5, atol=1e-5)


def test_dp2_objects_to_scene(tmp_path):
    dx = np.array([-10.0, 10.0, 0.0])
    dy = np.array([0.0, 0.0, 12.0])
    reference = _make(
        {
            "dx": dx,
            "dy": dy,
            "redshift": np.zeros(3),
            "sersic_n": np.ones(3),
            "r50_major": np.ones(3),
            "r50_minor": np.ones(3),
            "theta": np.zeros(3),
            "lsst_i": np.zeros(3),
        },
        tract_info=TRACT_INFO,
    )
    flux_i = np.array([1e4, 2e3, 5e2])  # nJy
    objects = _structured(
        {
            "objectId": np.array([11, 22, 33], dtype=np.int64),
            "coord_ra": reference.data["ra"],
            "coord_dec": reference.data["dec"],
            "sersic_index": [1.5, 4.2, 9.0],
            "sersic_reff_major": [0.7, 1.1, 0.2],
            "sersic_reff_minor": [0.35, 1.0, 0.2],
            "sersic_theta": [10.0, -40.0, 0.0],
            "refExtendedness": [1.0, 0.9, 0.1],
            "i_cModelFlux": flux_i,
            "i_psfFlux": flux_i * 0.5,
            "r_cModelFlux": flux_i * 2,
            "r_psfFlux": flux_i,
            "bpz_z_best": [0.31, np.nan, 0.9],
        }
    )
    scene = ClusterSceneCatalog.dp2_objects_to_scene(
        objects, bands=("r", "i"), default_redshift=Z_CLUSTER, morphology="sersic"
    )
    # a FITS file of the object rows converts to the same scene
    fitsio.write(os.path.join(tmp_path, "objects.fits"), objects)
    from_file = ClusterSceneCatalog.dp2_objects_to_scene(
        os.path.join(tmp_path, "objects.fits"), bands=("r", "i"), default_redshift=Z_CLUSTER, morphology="sersic"
    )
    for name in scene.dtype.names:
        np.testing.assert_array_equal(from_file[name], scene[name])
    catalog = _make(scene, tract_info=TRACT_INFO)
    data = catalog.data
    assert len(catalog) == 3
    np.testing.assert_array_equal(data["objectId"], [11, 22, 33])
    np.testing.assert_allclose(data["dx"], dx, atol=1e-6)
    np.testing.assert_allclose(data["dy"], dy, atol=1e-6)
    assert data["is_point_source"].tolist() == [False, False, True]
    np.testing.assert_allclose(data["redshift"], [0.31, Z_CLUSTER, 0.0])
    np.testing.assert_array_equal(data["angles"], 0.0)
    # DP2 angles (East towards North) are negated into the pixel frame
    np.testing.assert_allclose(data["theta"], [-10.0, 40.0, 0.0])
    # galaxies use the cModel flux, the star its PSF flux
    expected_i = -2.5 * np.log10([1e4, 2e3, 2.5e2]) + AB_MAG_ZERO_NJY
    np.testing.assert_allclose(data["lsst_i"], expected_i)
    expected_r = -2.5 * np.log10([2e4, 4e3, 5e2]) + AB_MAG_ZERO_NJY
    np.testing.assert_allclose(data["lsst_r"], expected_r)
    np.testing.assert_allclose(data["hlr"], np.sqrt([0.7 * 0.35, 1.1, 0.04]))

    # rendering: the Sersic index above GalSim's range is clipped, the
    # star is a point source, total flux is conserved
    exposure = _blank_exposure()
    truth = catalog.draw_on_image(exposure, band="i", mag_zero=MAG_ZERO, psf_obj=_psf())
    assert truth["drawn"].all()
    expected = np.sum(10 ** ((MAG_ZERO - expected_i) / 2.5))
    np.testing.assert_allclose(exposure.getMaskedImage().image.array.sum(), expected, rtol=2e-2)

    with pytest.raises(ValueError, match="sersic_index"):
        ClusterSceneCatalog.dp2_objects_to_scene(
            rfn.drop_fields(objects, "sersic_index", usemask=False), bands=("i",), morphology="sersic"
        )


def test_load_from_catsim_dir(tmp_path):
    table = _radec_scene(_scene_table())
    fname = ClusterSceneCatalog.write_scene(table, str(tmp_path))
    assert fname == os.path.join(tmp_path, ClusterSceneCatalog.catalog_filename)

    catalog = ClusterSceneCatalog(
        rng=np.random.RandomState(0),
        tract_info=TRACT_INFO,
        layout_name="random",
        catsim_dir=str(tmp_path),
        select_observable=["lsst_i"],
        select_upper_limit=[20.5],
    )
    assert len(catalog) == 4
    np.testing.assert_array_equal(catalog.data["indices"], [0, 1, 3, 4])
    assert catalog.data["is_point_source"].dtype == np.bool_

    # the pipeline builds it from the galaxy type and $CATSIM_DIR-style directory
    catalog = get_catalog_class("cluster_scene")(tract_info=TRACT_INFO, catsim_dir=str(tmp_path))
    assert len(catalog) == len(SCENE_ROWS)
    np.testing.assert_allclose(catalog.data["lsst_i"], table["lsst_i"])

    with pytest.raises(FileNotFoundError, match="cluster_scene.fits"):
        ClusterSceneCatalog(tract_info=TRACT_INFO, catsim_dir=str(tmp_path / "empty"))


# --- bulge + disk morphology ---------------------------------------------------

# dx, dy [arcsec], i mag, bulge fraction (i, r), disk (a, b, theta), bulge (a, b, theta)
BULGE_DISK_ROWS = [
    # pure disk along +x
    (-20.0, -15.0, 20.0, 0.0, 0.2, 1.0, 0.5, 0.0, 0.4, 0.4, 0.0),
    # pure bulge along +y
    (20.0, -15.0, 20.0, 1.0, 1.0, 0.6, 0.6, 0.0, 1.2, 0.5, 90.0),
    # half and half, disk at 30 degrees, bulge at 60 degrees
    (0.0, 20.0, 21.0, 0.5, 0.5, 0.8, 0.4, 30.0, 0.5, 0.4, 60.0),
]


def _bulge_disk_table():
    rows = np.array(BULGE_DISK_ROWS, dtype=float)
    return {
        "dx": rows[:, 0],
        "dy": rows[:, 1],
        "redshift": np.full(len(rows), Z_CLUSTER),
        "lsst_i": rows[:, 2],
        "lsst_r": rows[:, 2] + 0.5,
        "lsst_i_bulge_frac": rows[:, 3],
        "lsst_r_bulge_frac": rows[:, 4],
        "disk_r50_major": rows[:, 5],
        "disk_r50_minor": rows[:, 6],
        "disk_theta": rows[:, 7],
        "bulge_r50_major": rows[:, 8],
        "bulge_r50_minor": rows[:, 9],
        "bulge_theta": rows[:, 10],
    }


def test_bulge_disk_construction():
    table = _bulge_disk_table()
    catalog = _make(table, tract_info=TRACT_INFO)
    data = catalog.data
    assert len(catalog) == 3
    for col in ("bulge_r50_major", "disk_theta", "lsst_i_bulge_frac", "lsst_r_bulge_frac"):
        assert col in data.dtype.names
    # no extra rotation; the disk sets the half-light radius
    np.testing.assert_array_equal(data["angles"], 0.0)
    np.testing.assert_allclose(data["hlr"], np.sqrt(table["disk_r50_major"] * table["disk_r50_minor"]))
    assert ClusterSceneCatalog.magnitude_columns("lsst", "i") == ("lsst_i",)

    # incomplete bulge/disk columns and a missing bulge fraction are reported
    columns = {name: values for name, values in table.items() if name != "bulge_theta"}
    with pytest.raises(ValueError, match="bulge_theta"):
        _make(columns, tract_info=TRACT_INFO)
    columns = {name: values for name, values in table.items() if not name.endswith("_bulge_frac")}
    with pytest.raises(ValueError, match="bulge_frac"):
        _make(columns, tract_info=TRACT_INFO)
    # a band-independent bulge fraction is accepted instead
    columns["bulge_frac"] = table["lsst_i_bulge_frac"]
    catalog2 = _make(columns, tract_info=TRACT_INFO)
    exposure = _blank_exposure()
    catalog.draw_on_image(exposure, band="i", mag_zero=MAG_ZERO, psf_obj=_psf())
    exposure2 = _blank_exposure()
    catalog2.draw_on_image(exposure2, band="i", mag_zero=MAG_ZERO, psf_obj=_psf())
    np.testing.assert_allclose(
        exposure.getMaskedImage().image.array, exposure2.getMaskedImage().image.array, rtol=1e-6, atol=1e-6
    )


def test_bulge_disk_render():
    catalog = _make(_bulge_disk_table(), tract_info=TRACT_INFO)
    exposure = _blank_exposure()
    truth = catalog.draw_on_image(exposure, band="i", mag_zero=MAG_ZERO, psf_obj=_psf())
    array = exposure.getMaskedImage().image.array
    bbox = exposure.getBBox()
    assert truth["drawn"].all()
    expected = sum(_flux(r[2]) for r in BULGE_DISK_ROWS)
    np.testing.assert_allclose(array.sum(), expected, rtol=2e-2)

    # the pure disk is elongated along x, the pure bulge along y
    ixx, iyy = _moments(array, truth[0]["image_x"] - bbox.getMinX(), truth[0]["image_y"] - bbox.getMinY())
    assert ixx > 1.3 * iyy
    ixx, iyy = _moments(array, truth[1]["image_x"] - bbox.getMinX(), truth[1]["image_y"] - bbox.getMinY())
    assert iyy > 1.2 * ixx

    # the r band splits the light differently but keeps the total
    exposure = _blank_exposure()
    catalog.draw_on_image(exposure, band="r", mag_zero=MAG_ZERO, psf_obj=_psf())
    expected = sum(_flux(r[2] + 0.5) for r in BULGE_DISK_ROWS)
    np.testing.assert_allclose(exposure.getMaskedImage().image.array.sum(), expected, rtol=2e-2)

    # isotropic rendering rounds both components; forced profiles keep the flux
    exposure = _blank_exposure()
    catalog.draw_on_image(exposure, band="i", mag_zero=MAG_ZERO, psf_obj=_psf(), force_isotropic=True)
    array = exposure.getMaskedImage().image.array
    ixx, iyy = _moments(array, truth[0]["image_x"] - bbox.getMinX(), truth[0]["image_y"] - bbox.getMinY())
    np.testing.assert_allclose(ixx, iyy, rtol=0.05)
    exposure = _blank_exposure()
    catalog.draw_on_image(exposure, band="i", mag_zero=MAG_ZERO, psf_obj=_psf(), force_galaxy_profile=1)
    expected = sum(_flux(r[2]) for r in BULGE_DISK_ROWS)
    np.testing.assert_allclose(exposure.getMaskedImage().image.array.sum(), expected, rtol=2e-2)

    # the mixed object is a sum of the two components
    obj = catalog.get_obj(ind=2, mag_zero=MAG_ZERO, band="i", survey_name="lsst")
    np.testing.assert_allclose(obj.flux, _flux(21.0))
    # the pipeline drawing loop renders the same pixels
    config = MultibandSimConfig()
    config.galaxy_type = "cluster_scene"
    config.survey_name = "lsst"
    task = MultibandSimTask(config=config)
    pipeline = task.draw_catalog(
        galaxy_catalog=catalog,
        wcs=TRACT_INFO.getWcs(),
        bbox_outer=bbox,
        psf_obj=_psf(),
        mag_zero=MAG_ZERO,
        band="i",
    )
    reference = afwImage.ImageF(bbox)
    catalog.draw_on_image(
        reference,
        band="i",
        mag_zero=MAG_ZERO,
        psf_obj=_psf(),
        wcs=TRACT_INFO.getWcs(),
        use_field_distortion=config.use_field_distortion,
        nn_trunc=None if config.truncate_stamp_size <= 0 else config.truncate_stamp_size,
    )
    np.testing.assert_allclose(reference.array, pipeline, rtol=1e-5, atol=1e-6)


def test_dp2_objects_to_scene_bulge_disk():
    reference = _make(_bulge_disk_table(), tract_info=TRACT_INFO)
    # fluxes (nJy) reproducing the magnitudes of the hand-built scene
    mag_i = np.array([r[2] for r in BULGE_DISK_ROWS])
    flux_i = 10 ** ((AB_MAG_ZERO_NJY - mag_i) / 2.5)
    flux_r = 10 ** ((AB_MAG_ZERO_NJY - (mag_i + 0.5)) / 2.5)
    objects = _structured(
        {
            "objectId": np.array([1, 2, 3], dtype=np.int64),
            "coord_ra": reference.data["ra"],
            "coord_dec": reference.data["dec"],
            "refExtendedness": [1.0, 1.0, 1.0],
            "i_cModel_dev_reff_major": [r[8] for r in BULGE_DISK_ROWS],
            "i_cModel_dev_reff_minor": [r[9] for r in BULGE_DISK_ROWS],
            # DP2 angles run from East towards North: the negative of the scene's
            "i_cModel_dev_theta": [-r[10] for r in BULGE_DISK_ROWS],
            "i_cModel_exp_reff_major": [r[5] for r in BULGE_DISK_ROWS],
            "i_cModel_exp_reff_minor": [r[6] for r in BULGE_DISK_ROWS],
            "i_cModel_exp_theta": [-r[7] for r in BULGE_DISK_ROWS],
            "r_cModel_dev_reff_major": [9.0, 9.0, 9.0],  # another band's shapes: unused
            "i_cModel_fracDev": [r[3] for r in BULGE_DISK_ROWS],
            "r_cModel_fracDev": [r[4] for r in BULGE_DISK_ROWS],
            "i_cModelFlux": flux_i,
            "r_cModelFlux": flux_r,
            "i_psfFlux": flux_i * 0.5,
            "r_psfFlux": flux_r * 0.5,
            "bpz_z_best": [Z_CLUSTER, Z_CLUSTER, 0.9],
        }
    )
    catalog = _make(ClusterSceneCatalog.dp2_objects_to_scene(objects, bands=("r", "i")), tract_info=TRACT_INFO)
    data = catalog.data
    assert "sersic_n" not in data.dtype.names
    np.testing.assert_allclose(data["bulge_r50_major"], [0.4, 1.2, 0.5])
    np.testing.assert_allclose(data["disk_theta"], [0.0, 0.0, 30.0])
    np.testing.assert_allclose(data["bulge_theta"], [0.0, 90.0, 60.0])
    np.testing.assert_allclose(data["lsst_i_bulge_frac"], [0.0, 1.0, 0.5])
    np.testing.assert_allclose(data["lsst_r_bulge_frac"], [0.2, 1.0, 0.5])
    np.testing.assert_array_equal(data["angles"], 0.0)
    assert not data["is_point_source"].any()
    np.testing.assert_allclose(data["redshift"], [Z_CLUSTER, Z_CLUSTER, 0.9])
    np.testing.assert_allclose(data["lsst_i"], mag_i)
    np.testing.assert_allclose(data["lsst_r"], mag_i + 0.5)

    # the converted scene renders exactly as the hand-built bulge/disk scene
    for band in ("i", "r"):
        exposure = _blank_exposure()
        catalog.draw_on_image(exposure, band=band, mag_zero=MAG_ZERO, psf_obj=_psf())
        handmade = _blank_exposure()
        reference.draw_on_image(handmade, band=band, mag_zero=MAG_ZERO, psf_obj=_psf())
        np.testing.assert_allclose(
            exposure.getMaskedImage().image.array,
            handmade.getMaskedImage().image.array,
            rtol=1e-5,
            atol=1e-4,
        )

    # a missing bulge fraction renders as a pure disk
    no_frac_objects = objects.copy()
    no_frac_objects["i_cModel_fracDev"] = np.nan
    no_frac = _make(ClusterSceneCatalog.dp2_objects_to_scene(no_frac_objects, bands=("i",)), tract_info=TRACT_INFO)
    assert np.isnan(no_frac.data["lsst_i_bulge_frac"]).all()
    disk_only = {name: values for name, values in _bulge_disk_table().items() if name != "lsst_r_bulge_frac"}
    disk_only["lsst_i_bulge_frac"] = np.zeros(3)
    disk_only = _make(disk_only, tract_info=TRACT_INFO)
    exposure = _blank_exposure()
    no_frac.draw_on_image(exposure, band="i", mag_zero=MAG_ZERO, psf_obj=_psf())
    handmade = _blank_exposure()
    disk_only.draw_on_image(handmade, band="i", mag_zero=MAG_ZERO, psf_obj=_psf())
    np.testing.assert_allclose(
        exposure.getMaskedImage().image.array, handmade.getMaskedImage().image.array, rtol=1e-5, atol=1e-4
    )

    with pytest.raises(ValueError, match="i_cModel_fracDev"):
        ClusterSceneCatalog.dp2_objects_to_scene(
            rfn.drop_fields(objects, "i_cModel_fracDev", usemask=False), bands=("i",)
        )
    with pytest.raises(ValueError, match="morphology"):
        ClusterSceneCatalog.dp2_objects_to_scene(objects, bands=("i",), morphology="bad")


def _dp2_position_angle(phi_deg, q, cd):
    """DP2's ``theta`` for a pixel-frame ellipse at ``phi_deg`` from +x, computed the way
    ``pipe_tasks`` ``PositionAngleFromMoments`` does: moments mapped to the sky with the
    CD matrix (u = +RA, v = +Dec), then ``0.5 * atan2(2 uv, uu - vv)``."""
    phi = np.radians(phi_deg)
    rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    moments = rot @ np.diag([1.0, q**2]) @ rot.T
    sky = cd @ moments @ cd.T
    return np.degrees(0.5 * np.arctan2(2 * sky[0, 1], sky[0, 0] - sky[1, 1]))


def test_dp2_position_angle_convention():
    """An object at a known pixel angle, described by DP2's own angle convention,
    is drawn back at that pixel angle (not mirrored)."""
    cd = TRACT_INFO.getWcs().getCdMatrix()
    assert cd[0, 0] < 0  # standard East-left tract
    centre = TRACT_INFO.getBBox().getCenter()
    ra, dec = TRACT_INFO.getWcs().pixelToSkyArray(
        x=np.array([centre.getX() + 0.3]), y=np.array([centre.getY() + 0.3]), degrees=True
    )
    for phi in (30.0, -60.0):
        theta_dp2 = _dp2_position_angle(phi, 0.4, cd)
        assert abs(abs(theta_dp2) - abs(phi)) < 1e-6 and np.sign(theta_dp2) == -np.sign(phi)
        objects = _structured(
            {
                "coord_ra": ra,
                "coord_dec": dec,
                "sersic_index": [1.0],
                "sersic_reff_major": [1.5],
                "sersic_reff_minor": [0.6],
                "sersic_theta": [theta_dp2],
                "i_cModelFlux": [1e5],
            }
        )
        catalog = _make(
            ClusterSceneCatalog.dp2_objects_to_scene(objects, bands=("i",), morphology="sersic"),
            tract_info=TRACT_INFO,
        )
        exposure = _blank_exposure()
        truth = catalog.draw_on_image(exposure, band="i", mag_zero=MAG_ZERO, psf_obj=galsim.Gaussian(fwhm=0.3))
        array = exposure.getMaskedImage().image.array
        bbox = exposure.getBBox()
        cx = int(round(truth[0]["image_x"] - bbox.getMinX()))
        cy = int(round(truth[0]["image_y"] - bbox.getMinY()))
        half = 30
        sub = array[cy - half : cy + half + 1, cx - half : cx + half + 1]
        yy, xx = np.mgrid[-half : half + 1, -half : half + 1].astype(float)
        xx -= (sub * xx).sum() / sub.sum()
        yy -= (sub * yy).sum() / sub.sum()
        ixx, iyy, ixy = ((sub * m).sum() / sub.sum() for m in (xx**2, yy**2, xx * yy))
        drawn = np.degrees(0.5 * np.arctan2(2 * ixy, ixx - iyy))
        assert abs(drawn - phi) < 1.0, (phi, theta_dp2, drawn)
