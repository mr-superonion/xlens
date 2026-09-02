"""Unit tests for galaxies.py covering CatSim2017, Flagship2025 and Diffsky.

Each test is parametrized across the three galaxy catalog classes so a
single function exercises construction, selection filtering, density,
half-light radius, galaxy generation, force_isotropic rendering, and
invalid-selection error handling for all three catalogs.
"""

import os

import fitsio
import galsim
import numpy as np
import pytest
from lsst.skymap.ringsSkyMap import RingsSkyMap, RingsSkyMapConfig

import xlens

# Small skymap matching the layout used in test_simulator.py.
_config = RingsSkyMapConfig()
_config.patchInnerDimensions = [101, 101]
_config.tractOverlap = 0.0
_config.patchBorder = 0
_config.numRings = 7000
_config.pixelScale = 0.2
_config.projection = "TAN"
skymap0 = RingsSkyMap(config=_config)


# --- per-catalog spec --------------------------------------------------------

def _read_catsim():
    return fitsio.read(os.path.join(os.environ["CATSIM_DIR"], "OneDegSq.fits"))


def _read_flagship():
    return fitsio.read(
        os.path.join(os.environ["CATSIM_DIR"], "flagship_cosmos.fits")
    )


def _read_diffsky():
    return fitsio.read(
        os.path.join(os.environ["CATSIM_DIR"], "diffsky2026.fits")
    )


# CatSim2017's ``_generate_galaxy`` doesn't take ``survey_name`` but
# absorbs unknown kwargs via ``**kwargs``; Flagship/Diffsky do need it.
# Passing it unconditionally below is therefore safe for all three.
SPECS = {
    "catsim2017": dict(
        cls=lambda: xlens.simulator.galaxies.CatSim2017Catalog,
        read=_read_catsim,
        mag_col="i_ab",          # column used for selection cuts
        mag_min=20.0,
        mag_max=25.0,
        extra_cols=("i_ab",),
    ),
    "flagship2025": dict(
        cls=lambda: xlens.simulator.galaxies.Flagship2025Catalog,
        read=_read_flagship,
        mag_col="lsst_i",
        mag_min=20.0,
        mag_max=24.0,
        extra_cols=("lsst_i", "disk_r50"),
    ),
    "diffsky": dict(
        cls=lambda: xlens.simulator.galaxies.DiffskyCatalog,
        read=_read_diffsky,
        mag_col="lsst_i_disk",
        mag_min=18.0,
        mag_max=25.0,
        extra_cols=(
            "lsst_i_disk", "lsst_i_bulge", "r50_disk_as", "r50_bulge_as",
            "ellipticity_disk", "ellipticity_bulge", "psi_disk", "psi_bulge",
        ),
    ),
}

ALL_CATALOGS = list(SPECS.keys())


def _bright_finite_idx(cat, spec):
    """Pick a moderately bright, finite catalog row (handles NaNs in
    Diffsky)."""
    mag = cat[spec["mag_col"]]
    mask = np.isfinite(mag) & (mag > 17.0) & (mag < 24.0)
    if "lsst_i_bulge" in cat.dtype.names:
        mask &= np.isfinite(cat["lsst_i_bulge"])
    return int(np.flatnonzero(mask)[0])


# --- parametrized tests ------------------------------------------------------


@pytest.mark.parametrize("name", ALL_CATALOGS)
def test_init_and_from_array(name):
    spec = SPECS[name]
    Cls = spec["cls"]()
    rng = np.random.RandomState(42)
    catalog = Cls(
        rng=rng,
        tract_info=skymap0[0],
        layout_name="random",
    )
    arr = catalog.data
    assert len(arr) > 0
    for col in ("indices", "redshift", "angles", "dx", "dy", "ra", "dec",
                "hlr"):
        assert col in arr.dtype.names
    for col in spec["extra_cols"]:
        assert col in arr.dtype.names, f"{name} missing {col}"

    catalog2 = Cls.from_array(
        truthCatalog=arr,
        tract_info=skymap0[0],
    )
    np.testing.assert_array_equal(
        catalog.data["indices"],
        catalog2.data["indices"]
    )
    np.testing.assert_array_almost_equal(
        catalog.data["dx"],
        catalog2.data["dx"]
    )
    np.testing.assert_array_almost_equal(
        catalog.data["dy"],
        catalog2.data["dy"]
    )


@pytest.mark.parametrize("name", ALL_CATALOGS)
def test_selection(name):
    spec = SPECS[name]
    Cls = spec["cls"]()
    rng = np.random.RandomState(0)
    catalog = Cls(
        rng=rng,
        tract_info=skymap0[0],
        layout_name="random",
        select_observable=[spec["mag_col"]],
        select_lower_limit=[spec["mag_min"]],
        select_upper_limit=[spec["mag_max"]],
    )
    arr = catalog.data
    assert len(arr) > 0
    truth = spec["read"]()
    selected = truth[arr["indices"]][spec["mag_col"]]
    assert selected.min() > spec["mag_min"]
    assert selected.max() <= spec["mag_max"]


@pytest.mark.parametrize("name", ALL_CATALOGS)
def test_density_and_hlr(name):
    spec = SPECS[name]
    Cls = spec["cls"]()
    cat = spec["read"]()
    inst = Cls.__new__(Cls)
    density = inst._compute_density(cat)
    assert 10 < density < 1000
    hlr = inst._half_light_radius(cat[:5])
    assert hlr.shape == (5,)
    assert np.all(hlr > 0)


def test_radec_box_area_handles_the_ra_wrap():
    """A field straddling RA = 0 must not read as a 360-deg-wide one.

    ``OneDegSq.fits`` is exactly this case, and a naive
    ``ra.max() - ra.min()`` puts its density 360x too low.
    """
    Base = xlens.simulator.galaxies.BaseGalaxyCatalog
    rng = np.random.RandomState(42)
    dec = rng.uniform(-0.5, 0.5, 20000)
    # 1 deg of RA centred on 0, i.e. 359.5..360 and 0..0.5
    ra = np.mod(rng.uniform(-0.5, 0.5, 20000), 360.0)
    assert ra.max() - ra.min() > 359.0, "test data must straddle RA=0"

    area = Base._radec_box_area_arcmin2(ra, dec)
    assert area == pytest.approx(3600.0, rel=2e-3)

    # the same field away from the meridian gives the same area
    shifted = Base._radec_box_area_arcmin2(np.mod(ra + 180.0, 360.0), dec)
    assert shifted == pytest.approx(area, rel=1e-6)


def test_radec_box_area_is_the_exact_solid_angle_near_the_pole():
    """``cos(mean dec)`` is a small-box approximation; this is not."""
    Base = xlens.simulator.galaxies.BaseGalaxyCatalog
    ra = np.array([100.0, 110.0])
    dec = np.array([83.5, 85.5])
    exact = 10.0 * (np.sin(np.radians(85.5)) - np.sin(np.radians(83.5)))
    exact *= (180.0 / np.pi) * 3600.0
    assert Base._radec_box_area_arcmin2(ra, dec) == pytest.approx(exact)


def test_radec_box_area_refuses_a_field_wider_than_180_deg():
    """The wrap fix assumes a small field; say so rather than guess.

    At more than 180 deg wide, a wrapped span is no longer proof that
    the field straddles RA = 0, and there is no way to tell which of
    the two readings was meant.
    """
    Base = xlens.simulator.galaxies.BaseGalaxyCatalog
    ra = np.linspace(0.0, 300.0, 500)
    with pytest.raises(ValueError, match="exceeds 180"):
        Base._radec_box_area_arcmin2(ra, np.zeros_like(ra))


def test_footprint_area_rejects_a_degenerate_input():
    Base = xlens.simulator.galaxies.BaseGalaxyCatalog
    with pytest.raises(ValueError, match="fewer than 2"):
        Base._radec_box_area_arcmin2(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="degenerate footprint"):
        Base._radec_box_area_arcmin2(np.zeros(5), np.zeros(5))


@pytest.mark.parametrize("name", ALL_CATALOGS)
def test_generate_galaxy(name):
    spec = SPECS[name]
    Cls = spec["cls"]()
    cat = spec["read"]()
    inst = Cls.__new__(Cls)
    entry = cat[_bright_finite_idx(cat, spec)]
    gal = inst._generate_galaxy(
        entry=entry,
        mag_zero=30.0,
        band="i",
        survey_name="lsst",
    )
    assert isinstance(gal, galsim.GSObject)
    assert gal.flux > 0.0


@pytest.mark.parametrize("name", ALL_CATALOGS)
def test_force_isotropic(name):
    """force_isotropic produces a round galaxy: rendered image matches its
    90-degree rotation to well within the peak intensity."""
    spec = SPECS[name]
    Cls = spec["cls"]()
    cat = spec["read"]()
    inst = Cls.__new__(Cls)
    entry = cat[_bright_finite_idx(cat, spec)]
    gal = inst._generate_galaxy(
        entry=entry,
        mag_zero=30.0,
        band="i",
        force_isotropic=True,
        survey_name="lsst",
    )
    img = gal.drawImage(scale=0.2, nx=64, ny=64).array
    peak = float(np.max(np.abs(img)))
    assert peak > 0
    assert np.max(np.abs(img - np.rot90(img))) / peak < 1e-3


@pytest.mark.parametrize("name", ALL_CATALOGS)
def test_bad_selection_raises(name):
    """Selecting on a non-existent column should raise ValueError."""
    Cls = SPECS[name]["cls"]()
    rng = np.random.RandomState(0)
    with pytest.raises(ValueError):
        Cls(
            rng=rng,
            tract_info=skymap0[0],
            layout_name="random",
            select_observable=["not_a_column"],
            select_lower_limit=[0.0],
        )
