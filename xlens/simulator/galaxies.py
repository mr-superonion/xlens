# This file is part of xlens.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Galaxy catalog classes for building GalSim objects from input truth tables.

Provides an abstract :class:`BaseGalaxyCatalog` and concrete implementations
for CatSim 2017, OpenUniverse 2024 Rubin-Roman, and Euclid Flagship 2025
catalogs, and :class:`ClusterSceneCatalog`, which renders an object table
(for instance a real DP2 cluster field) at its own sky positions.
"""

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Iterable

import fitsio
import galsim
import lsst
import numpy as np
from numpy.lib import recfunctions as rfn

from .defaults import SIM_INCLUSION_PADDING
from .layout import Layout


# ``force_galaxy_profile`` codes shared by every catalog implementation
FORCE_GALAXY_PROFILE_NONE = 0
FORCE_GALAXY_PROFILE_GAUSSIAN = 1
FORCE_GALAXY_PROFILE_EXPONENTIAL = 2

# Default upper bound (arcsec) on the bulge half-light radius used for
# rendering, shared by every catalog implementation.  Input catalogs carry
# a tail of implausibly large bulges (flagship reaches 16.6", 99.9th
# percentile 3.8"), and a Sersic bulge that large makes GalSim size the
# stamp at several thousand pixels; the FFT of that one stamp then sets
# the peak memory of an entire run.  Override per catalog class with the
# ``max_bulge_hlr_arcsec`` attribute.
MAX_BULGE_HLR_ARCSEC = 3.0

# Placement and lensing columns every truth catalog carries, ahead of the
# input-catalog property columns merged in by the constructors.
PLACEMENT_DTYPE = [
    ("indices", "i8"),
    ("redshift", "f8"),
    ("angles", "f8"),
    ("gamma1", "f8"),
    ("gamma2", "f8"),
    ("kappa", "f8"),
    ("dx", "f8"),
    ("dy", "f8"),
    ("ra", "f8"),
    ("dec", "f8"),  # post-lensed ra, dec
    ("prelensed_ra", "f8"),
    ("prelensed_dec", "f8"),
    ("has_finite_shear", "bool"),
    ("hlr", "f8"),
]


def _survey_prefix(survey_name: str) -> str:
    """Column prefix for *survey_name*; hsc reuses the LSST photometry."""
    return "lsst" if survey_name == "hsc" else survey_name


def _galsim_round_sersic(n, sersic_prec):
    """Round a Sersic index to the nearest multiple of *sersic_prec*."""
    return float(int(n / sersic_prec + 0.5)) * sersic_prec


def _forced_profile(force_galaxy_profile, *, flux, half_light_radius):
    """Return the radial profile requested by *force_galaxy_profile*.

    Parameters
    ----------
    force_galaxy_profile : int
        1 for Gaussian, 2 for Exponential.
    flux : float
        Total flux of the component.
    half_light_radius : float
        Half-light radius (arcsec) of the component.
    """
    if force_galaxy_profile == FORCE_GALAXY_PROFILE_GAUSSIAN:
        return galsim.Gaussian(flux=flux, half_light_radius=half_light_radius)
    if force_galaxy_profile == FORCE_GALAXY_PROFILE_EXPONENTIAL:
        return galsim.Exponential(flux=flux, half_light_radius=half_light_radius)
    raise ValueError(
        "force_galaxy_profile must be 1 (gaussian) or 2 (exponential), "
        f"not {force_galaxy_profile}"
    )


def get_catalog(fname, columns=None):
    """Read a FITS catalog.

    Row numbers are tracked separately (see
    :meth:`BaseGalaxyCatalog._apply_selection`) rather than materialised
    as a column: adding a field to a structured array copies the whole
    array, which for the larger inputs costs hundreds of MiB.

    Parameters
    ----------
    fname : str
        Path to a FITS file readable by ``fitsio``.
    columns : list of str or None, optional
        Subset of columns to read.  ``None`` reads every column.

    Returns
    -------
    numpy.ndarray
        Structured array of the requested columns.
    """
    return fitsio.read(fname, columns=columns)


class BaseGalaxyCatalog(ABC):
    """
    Abstract base class for galaxy catalogs used to build GalSim objects.

    Subclasses must implement:
      - _read_catalog(...)
      - _generate_galaxy(entry, mag_zero, band, **kwargs)
    Optionally override:
      - _probabilities_for_sampling(cat) -> Optional[np.ndarray]
    """

    # Sky-position columns, used to measure the footprint the density is
    # computed over.  All three catalogs fill an RA/Dec box, so naming
    # the columns is the only thing that varies between them.
    radec_columns: ClassVar[tuple[str, str]] = ("ra", "dec")

    def __init__(
        self,
        *,
        rng: np.random.RandomState,
        tract_info: lsst.skymap.tractInfo.ExplicitTractInfo,
        layout_name: str,
        sep_arcsec: float | None = None,
        indice_group_id: int | None = None,
        select_observable: list[str] | str | None = None,
        select_lower_limit: Iterable[float] | None = None,
        select_upper_limit: Iterable[float] | None = None,
        extend_ratio: float = 1.08,
        force_pixel_center: bool = False,
        catsim_dir: str | None = None,
        survey_name_list: Iterable[str] | None = None,
    ):
        """Construct a galaxy catalog from scratch with a spatial layout.

        Parameters
        ----------
        rng : numpy.random.RandomState
            Random number generator (old NumPy API).
        tract_info : lsst.skymap.tractInfo.ExplicitTractInfo
            Tract information providing WCS and bounding box.
        layout_name : {'grid', 'hex', 'random', 'random_disk'}
            Pattern used to place galaxies.
        sep_arcsec : float or None, optional
            Spacing for grid/hex layouts.
        indice_group_id : int or None, optional
            When non-negative, select a deterministic block of catalog
            rows instead of random sampling.
        select_observable, select_lower_limit, select_upper_limit
            Optional filtering criteria forwarded to ``_read_catalog``.
        extend_ratio : float, optional
            Padding factor for the layout bounding box.
        force_pixel_center : bool, optional
            Snap galaxy centres to pixel centres.
        catsim_dir : str or None, optional
            Directory for input catalog files.  Falls back to the
            ``CATSIM_DIR`` environment variable when *None*.
        survey_name_list : iterable of str or None, optional
            Surveys whose ``{survey}_{band}`` photometry columns are read
            from the input catalog; the columns of every entry are
            collected, so one catalog can feed simulations of several
            surveys.  Must cover the ``survey_name`` of every simulation
            task that later renders this catalog, otherwise the magnitudes
            it needs will not have been read.  Defaults to ``["lsst"]``.
        """
        self.catsim_dir = catsim_dir or os.environ.get("CATSIM_DIR", ".")
        if survey_name_list is None:
            survey_name_list = ["lsst"]
        self.survey_name_list = tuple(
            str(name).lower() for name in survey_name_list
        )
        self.prepare_tract_info(tract_info)
        wcs = tract_info.getWcs()
        ps = float(wcs.getPixelScale().asArcseconds())
        self.pixel_scale = ps
        bbox = tract_info.getBBox()
        layout = Layout(
            layout_name=layout_name,
            wcs=wcs,
            boundary_box=bbox,
            sep_arcsec=sep_arcsec,
            extend_ratio=extend_ratio,
        )
        # ``input_row_ids`` are the row numbers in the unfiltered input
        # file, carried alongside the (possibly cut) catalog instead of
        # as a column to avoid copying the whole array.
        input_catalog, input_row_ids = self._read_catalog(
            select_observable=select_observable,
            select_lower_limit=select_lower_limit,
            select_upper_limit=select_upper_limit,
        )

        # density drives how many objects the layout will place
        density = self._compute_density(input_catalog)
        # positions to place galaxies
        shifts_array = layout.get_shifts(rng=rng, density=density)

        if force_pixel_center:
            inv_pixel_scale = 1.0 / ps
            shifts_array["dx"] = (np.round(shifts_array["dx"] * inv_pixel_scale) + 0.5) * ps
            shifts_array["dy"] = (np.round(shifts_array["dy"] * inv_pixel_scale) + 0.5) * ps

        # choose which catalog rows populate those positions
        num = len(shifts_array)
        catalog_size = len(input_catalog)
        if (indice_group_id is None) or (indice_group_id < 0):
            probs = self._probabilities_for_sampling(input_catalog)
            integers = np.arange(0, catalog_size, dtype=int)
            idx = rng.choice(integers, size=num, p=probs)
        else:
            indice_min = indice_group_id * num
            indice_max = min(indice_min + num, catalog_size)
            if indice_min >= catalog_size:
                raise ValueError("indice_min too large")
            idx = np.arange(indice_min, indice_max, dtype=int) % catalog_size
            num = indice_max - indice_min
            shifts_array = shifts_array[0:num]
        # random orientation for each placed galaxy
        angles = rng.uniform(low=0.0, high=2.0 * np.pi, size=num)
        # rows of the input galaxy catalog that populate the placed objects
        selected = input_catalog[idx]

        placement = np.zeros(num, dtype=PLACEMENT_DTYPE)
        placement["dx"] = shifts_array["dx"]
        placement["dy"] = shifts_array["dy"]
        placement["angles"] = angles
        image_x = self.x_center + placement["dx"] / ps
        image_y = self.y_center + placement["dy"] / ps
        wcs = tract_info.getWcs()
        ra, dec = wcs.pixelToSkyArray(
            x=image_x,
            y=image_y,
            degrees=True,
        )
        placement["ra"] = ra
        placement["dec"] = dec
        placement["prelensed_ra"] = ra
        placement["prelensed_dec"] = dec
        placement["has_finite_shear"] = np.ones(num, dtype=bool)
        placement["indices"] = input_row_ids[idx]
        placement["redshift"] = selected["redshift"]
        placement["hlr"] = self._build_hlr_array(selected)

        # Merge the selected input-catalog rows into ``data`` so the truth
        # catalog is self-contained: ``from_array`` rebuilds the catalog
        # directly from this array, with no need to re-read the input
        # galaxy catalog from disk.  ``selected[extra]`` is a multi-field
        # view (no copy); ``merge_arrays`` does a single allocation.
        # Placement columns win over identically named input columns.
        extra = [name for name in selected.dtype.names if name not in placement.dtype.names]
        self.data = np.asarray(
            rfn.merge_arrays(
                [placement, selected[extra]],
                flatten=True,
                usemask=False,
            )
        )
        self.dtype = self.data.dtype
        self.lensed = False
        return

    def set_z_source(self, redshift):
        """Override all galaxy redshifts with a fixed value."""
        self.data["redshift"][:] = redshift
        return

    def prepare_tract_info(self, tract_info):
        """Store tract info and compute the pixel-centre coordinates."""
        self.tract_info = tract_info
        bbox = tract_info.getBBox()  # lsst.geom.Box2I
        center_pix = bbox.getCenter()
        self.x_center = center_pix.getX()
        self.y_center = center_pix.getY()
        return

    # ---------- required subclass hooks ----------

    # Basename of the catalog file under ``catsim_dir``. Subclasses MUST
    # set this; ``_read_catalog`` resolves it against ``self.catsim_dir``.
    catalog_filename: ClassVar[str]

    # Columns required from the input catalog.  ``None`` reads every
    # column; subclasses set this so that large inputs do not pull in
    # tens of unused columns.  Selection observables are added on top.
    required_columns: ClassVar[tuple[str, ...] | None] = None

    # Fallback surveys used when the catalog is rebuilt by ``from_array``,
    # which never re-reads the input file.
    survey_name_list: tuple[str, ...] = ("lsst",)

    # Bulge half-light radii are clipped to this value (arcsec) before
    # rendering; see :data:`MAX_BULGE_HLR_ARCSEC`.
    max_bulge_hlr_arcsec: ClassVar[float] = MAX_BULGE_HLR_ARCSEC

    @classmethod
    def magnitude_columns(cls, survey_name: str, band: str) -> tuple[str, ...]:
        """Columns holding the ``(survey_name, band)`` magnitude.

        Returning more than one column means the catalog stores the
        photometry per component (disk, bulge, ...); the total magnitude
        is then the sum of the component fluxes.  These are columns of
        the *input* catalog, which the truth catalog carries over, so
        consumers such as ``matchPipe`` can read the magnitude from the
        truth catalog without re-opening the input file.

        Parameters
        ----------
        survey_name : str
            Survey whose photometry is wanted (``lsst``, ``hsc``, ...).
        band : str
            Band name, e.g. ``i`` for LSST or ``vis`` for Euclid.
        """
        raise NotImplementedError

    def _required_columns(self) -> tuple[str, ...] | None:
        """Columns this catalog needs, possibly survey-dependent.

        Subclasses whose photometry columns are survey-prefixed override
        this to select only the bands of ``self.survey_name_list``.
        """
        return self.required_columns

    def _catalog_columns(self, select_observable) -> list[str] | None:
        """Columns to read, or ``None`` to read the whole catalog."""
        required = self._required_columns()
        if required is None:
            return None
        cols = list(required)
        if select_observable is not None:
            for name in np.atleast_1d(select_observable):
                if str(name) not in cols:
                    cols.append(str(name))
        return cols

    def _load_catalog_file(self, fname: str, columns=None) -> Any:
        """Load the raw catalog from ``fname``.

        Default implementation reads a FITS file via :func:`get_catalog`.
        Subclasses with a non-FITS on-disk format (e.g. Parquet) should
        override this.
        """
        return get_catalog(fname, columns=columns)

    @staticmethod
    def _apply_selection(
        cat,
        *,
        select_observable,
        select_lower_limit,
        select_upper_limit,
    ):
        """Apply per-column lower / upper bound cuts to a structured array.

        Shared by all subclasses; called from :meth:`_read_catalog`.

        Returns
        -------
        tuple
            ``(cat, row_ids)`` where ``row_ids`` are the row numbers of
            the surviving entries in the *unfiltered* input file.  These
            become the ``indices`` column of the truth catalog, which
            downstream code uses to index back into the input catalog.
        """
        if select_observable is None:
            return cat, np.arange(len(cat), dtype=np.int64)
        select_observable = np.atleast_1d(select_observable)
        if not set(select_observable) < set(cat.dtype.names):
            raise ValueError("Selection observables not in the catalog columns")
        mask = np.ones(len(cat), dtype=bool)
        if select_lower_limit is not None:
            select_lower_limit = np.atleast_1d(select_lower_limit)
            if len(select_observable) != len(select_lower_limit):
                raise ValueError(
                    "select_lower_limit length "
                    f"({len(select_lower_limit)}) must match "
                    f"select_observable ({len(select_observable)})"
                )
            for nn, ll in zip(select_observable, select_lower_limit):
                mask = mask & (cat[nn] > ll)
        if select_upper_limit is not None:
            select_upper_limit = np.atleast_1d(select_upper_limit)
            if len(select_observable) != len(select_upper_limit):
                raise ValueError(
                    "select_upper_limit length "
                    f"({len(select_upper_limit)}) must match "
                    f"select_observable ({len(select_observable)})"
                )
            for nn, ul in zip(select_observable, select_upper_limit):
                mask = mask & (cat[nn] <= ul)
        return cat[mask], np.flatnonzero(mask).astype(np.int64)

    def _read_catalog(
        self,
        *,
        select_observable=None,
        select_lower_limit=None,
        select_upper_limit=None,
    ) -> Any:
        """Load the input galaxy catalog and apply optional selection cuts.

        Subclasses customise this by setting ``catalog_filename``,
        ``required_columns`` and, if needed, overriding
        :meth:`_load_catalog_file`.

        Returns
        -------
        tuple
            ``(cat, row_ids)``; see :meth:`_apply_selection`.
        """
        fname = os.path.join(self.catsim_dir, self.catalog_filename)
        if not os.path.isfile(fname):
            raise FileNotFoundError(
                f"Cannot find '{self.catalog_filename}' under "
                f"{self.catsim_dir}. "
                "Please download it and place it under $CATSIM_DIR."
            )
        cat = self._load_catalog_file(
            fname, columns=self._catalog_columns(select_observable)
        )
        return self._apply_selection(
            cat,
            select_observable=select_observable,
            select_lower_limit=select_lower_limit,
            select_upper_limit=select_upper_limit,
        )

    def _compute_density(self, cat: Any) -> float:
        """Return object surface density in objects / arcmin^2.

        ``cat`` is the catalog *after* the ``select_*`` cuts, so a cut
        thins the simulated field rather than resampling the same field
        from a smaller pool.  The denominator is the footprint of the
        input file, which the cuts in practice do not change: every
        current use of ``select_observable`` is photometric.  A cut on
        ``ra``/``dec`` would shrink the measured box along with the
        count, which is also right.
        """
        ra_col, dec_col = self.radec_columns
        return len(cat) / self._radec_box_area_arcmin2(
            cat[ra_col], cat[dec_col]
        )

    @staticmethod
    def _radec_box_area_arcmin2(ra, dec) -> float:
        """Solid angle (arcmin^2) of the RA/Dec bounding box of the input.

        Two details make this safe on the real input files, and both are
        wrong in the naive ``(ra.max() - ra.min()) * cos(dec) *
        (dec.max() - dec.min())`` form:

        * **RA wrap.**  ``OneDegSq.fits`` is a 1 deg^2 field centred on
          RA = 0, so its RA values are 0..0.5 and 359.5..360 with
          nothing between: ``max - min`` reads 360 instead of 1 and the
          density comes out 360x too low.  These catalogs are all small
          regions so a span of very nearly 360 deg cannot be a real extent
          and means the field straddles RA = 0.  It is then split at 180,
          which separates the two clusters of a field this narrow exactly,
          and the high one is re-expressed as negative RA so that
          ``max - min`` works again.
        * **Dec convergence.**  The solid angle of a lon/lat box is
          exactly ``dRA * (sin dec_max - sin dec_min)``; the
          ``cos(mean dec)`` version is a small-box approximation that
          degrades towards the poles.

        This is a *bounding box*, so it is the footprint only for a
        catalog that fills one.  All three input catalogs do -- Diffsky
        is cut to a box for exactly this reason, see
        :class:`DiffskyCatalog`, whose parent cone its own bounding box
        would overestimate by 22%.  A catalog of some other shape needs
        more than a new ``radec_columns``: it has to override
        :meth:`_compute_density` outright.
        """
        ra = np.asarray(ra, dtype=float)
        dec = np.asarray(dec, dtype=float)
        if ra.size < 2:
            raise ValueError(
                "cannot measure a footprint from fewer than 2 objects"
            )
        ra = np.mod(ra, 360.0)
        if ra.max() - ra.min() > 359.5:
            ra = np.where(ra >= 180.0, ra - 360.0, ra)
        ra_extent = ra.max() - ra.min()
        if ra_extent > 180.0:
            raise ValueError(
                f"RA extent {ra_extent} deg exceeds 180: this estimator "
                "assumes a field small enough that an unwrapped span can "
                "only mean it straddles RA = 0"
            )
        dec_extent = np.sin(np.radians(dec.max())) - np.sin(
            np.radians(dec.min())
        )
        # dRA[deg] * dsin(dec) * (180/pi) converts the steradian
        # expression to deg^2; * 3600 to arcmin^2
        area = ra_extent * dec_extent * (180.0 / np.pi) * 3600.0
        if not area > 0.0:
            raise ValueError(
                f"degenerate footprint: RA extent {ra_extent} deg, "
                f"Dec extent {dec_extent} in sin(dec)"
            )
        return float(area)

    @abstractmethod
    def _generate_galaxy(
        self, *, entry: Any, mag_zero: float, band: str, **kwargs
    ) -> galsim.GSObject:
        """Build and return a GalSim GSObject from one catalog entry."""

    @abstractmethod
    def _half_light_radius(self, catalog) -> np.ndarray:
        """Return galaxy half-light radii (arcsec) for the given entries."""

    def _build_hlr_array(self, catalog) -> np.ndarray:
        hlr = self._half_light_radius(catalog)
        return np.asarray(hlr, dtype=float)

    def _probabilities_for_sampling(self, cat: Any) -> np.ndarray | None:
        """Optional per-row sampling probabilities. Default: None (uniform)."""
        return None

    def __len__(self) -> int:
        return len(self.data)

    @classmethod
    def from_array(
        cls,
        *,
        truthCatalog: np.ndarray,
        tract_info: lsst.skymap.tractInfo.ExplicitTractInfo,
        catsim_dir: str | None = None,
    ) -> "BaseGalaxyCatalog":
        """
        Build a catalog directly from a truth-catalog structured array.

        ``truthCatalog`` is the self-contained array produced by
        :class:`~xlens.simulator.catalog.CatalogTask`
        (``galaxy_catalog.data``).
        It carries the galaxy placement and shear columns together with the
        input galaxy-property columns merged in by ``__init__``, so the input
        galaxy catalog never has to be re-read from disk here.

        Parameters
        ----------
        truthCatalog : np.ndarray
            Truth-catalog structured array (``galaxy_catalog.data``).  Must
            contain at least the ``dx``, ``dy``, ``indices`` and ``angles``
            columns, along with the per-galaxy property columns consumed by
            ``_generate_galaxy``.
        tract_info : lsst.skymap.tractInfo.ExplicitTractInfo
            Tract information providing the WCS and bounding box.
        catsim_dir : str or None
            Directory containing input galaxy catalogs.  Retained for
            interface compatibility; unused now that the catalog is rebuilt
            directly from ``truthCatalog``.
        """
        if truthCatalog.dtype.names is None:
            raise TypeError("truthCatalog must be a structured array with named fields")
        # Create instance without running __init__
        self = cls.__new__(cls)
        self.catsim_dir = catsim_dir or os.environ.get("CATSIM_DIR", ".")
        self.prepare_tract_info(tract_info)
        wcs = tract_info.getWcs()
        self.pixel_scale = float(wcs.getPixelScale().asArcseconds())

        # Validate required placement columns. Draw time uses only
        # (ra, dec) via wcs.skyToPixel (see sim.py:400-405), so dx/dy
        # are not required on reload; angles and indices are still
        # consumed by get_obj / _generate_galaxy.
        for col in ["ra", "dec", "indices", "angles"]:
            if col not in list(truthCatalog.dtype.names):
                raise ValueError(f"Missing required column '{col}' in truthCatalog array")
        # The truth catalog is self-contained (placement + shear + galaxy
        # property columns), so use it directly instead of re-reading the
        # input galaxy catalog from disk.
        self.data = np.array(truthCatalog)
        self.dtype = self.data.dtype
        self.lensed = True
        return self

    def rotate(self, theta):
        """Rotate the catalog rigidly around the tract centre.

        Applies a 2D rotation by angle ``theta`` (radians, counter-clockwise)
        to every galaxy's tangent-plane offset ``(dx, dy)``, adds the same
        angle to each galaxy's intrinsic position-angle column
        (``angles``), and recomputes the sky positions ``(ra, dec)`` and
        ``(prelensed_ra, prelensed_dec)`` from the new pixel positions via
        the tract WCS.

        Typical use is the noise-cancellation trick for shear bias tests:
        rendering the same catalog twice, with the second realisation
        rotated by 90 degrees, lets the average of the two images cancel
        the intrinsic shape noise.

        Rotating a catalog after lensing has been applied is not supported
        and raises ``ValueError``: the lensing operation breaks the
        rotation symmetry of the underlying galaxy positions, and rotating
        the lensed catalog would no longer be equivalent to lensing a
        rotated catalog.

        Parameters
        ----------
        theta : float
            Rotation angle in radians (counter-clockwise).

        Returns
        -------
        None
            The catalog is rotated in place.

        Raises
        ------
        ValueError
            If the catalog has already been lensed (``self.lensed`` is
            ``True``).
        """
        if self.lensed:
            raise ValueError("Cannot rotate a lensed catalog")

        c, s = np.cos(theta), np.sin(theta)
        x = c * self.data["dx"] - s * self.data["dy"]
        y = s * self.data["dx"] + c * self.data["dy"]
        self.data["dx"] = x
        self.data["dy"] = y
        self.data["angles"] = self.data["angles"] + theta
        ps = self.pixel_scale
        image_x = self.x_center + self.data["dx"] / ps
        image_y = self.y_center + self.data["dy"] / ps
        wcs = self.tract_info.getWcs()
        ra, dec = wcs.pixelToSkyArray(
            x=image_x,
            y=image_y,
            degrees=True,
        )
        self.data["ra"] = ra
        self.data["dec"] = dec
        self.data["prelensed_ra"] = ra
        self.data["prelensed_dec"] = dec
        return

    def lens(self, *, shear_obj, apply_position_shifts: bool = True):
        """Apply lensing distortions from ``shear_obj`` to every galaxy.

        Parameters
        ----------
        shear_obj
            Object with a ``distort_galaxy(src)`` method that returns a dict
            with keys ``dx, dy, gamma1, gamma2, kappa, has_finite_shear``.
        apply_position_shifts : bool, optional
            If *True*, update image positions to the lensed coordinates;
            otherwise keep pre-lensing positions.
        """
        if self.lensed:
            raise ValueError("Cannot lens a lensed catalog")
        ps = self.pixel_scale
        prelensed_x = self.x_center + self.data["dx"] / ps
        prelensed_y = self.y_center + self.data["dy"] / ps
        wcs = self.tract_info.getWcs()

        # Snapshot pre-lens tangent-plane positions so we can restore
        # them if the caller opts out of position shifts, keeping
        # dx/dy consistent with the ra/dec we write below.
        dx0 = self.data["dx"].copy()
        dy0 = self.data["dy"].copy()

        for row in self.data:
            res = shear_obj.distort_galaxy(row)
            for key in (
                "dx",
                "dy",
                "gamma1",
                "gamma2",
                "kappa",
                "has_finite_shear",
            ):
                row[key] = res[key]
        if apply_position_shifts:
            image_x = self.x_center + self.data["dx"] / ps
            image_y = self.y_center + self.data["dy"] / ps
        else:
            self.data["dx"] = dx0
            self.data["dy"] = dy0
            image_x = prelensed_x
            image_y = prelensed_y

        ra, dec = wcs.pixelToSkyArray(
            x=image_x,
            y=image_y,
            degrees=True,
        )
        self.data["ra"] = ra
        self.data["dec"] = dec
        self.lensed = True
        return

    def get_obj(
        self,
        *,
        ind,
        mag_zero: float,
        band: str,
        force_isotropic: bool = False,
        force_galaxy_profile: int = FORCE_GALAXY_PROFILE_NONE,
        include_point_source: bool = True,
        survey_name: str = "",
    ) -> dict[str, list]:
        """Build a lensed, rotated GalSim object for galaxy at index *ind*.

        Parameters
        ----------
        ind : int
            Index into ``self.data``.
        mag_zero : float
            Zeropoint magnitude for flux conversion.
        band : str
            Photometric band label.
        force_isotropic : bool, optional
            Force all galaxies to have circular isophotes.
        force_galaxy_profile : int, optional
            If greater than zero, override the bulge and disk radial profiles
            with a single fixed profile: 1 for Gaussian, 2 for Exponential.
            The half-light radii, fluxes and ellipticities of the components
            are kept.  Zero (the default) keeps the catalog's native profiles.
        include_point_source : bool, optional
            Include AGN or point-source components.
        survey_name : str, optional
            Survey name used to select magnitude columns.

        Returns
        -------
        galsim.GSObject
            Lensed galaxy object ready for PSF convolution.
        """
        src = self.data[ind]
        # ``data`` carries the merged input-catalog property columns, so the
        # galaxy is rendered directly from it without an input-catalog lookup.
        gal = self._generate_galaxy(
            entry=src,
            mag_zero=mag_zero,
            band=band,
            include_point_source=include_point_source,
            force_isotropic=force_isotropic,
            force_galaxy_profile=force_galaxy_profile,
            survey_name=survey_name,
        )
        gal = gal.rotate(src["angles"] * galsim.radians)
        gamma1, gamma2, kappa = src["gamma1"], src["gamma2"], src["kappa"]
        g1 = gamma1 / (1 - kappa)
        g2 = gamma2 / (1 - kappa)
        mu = 1.0 / ((1 - kappa) ** 2 - gamma1**2 - gamma2**2)
        gal = gal.lens(g1=g1, g2=g2, mu=mu)
        return gal


# --------------------------------------------
# Concrete implementation: CatSim2017 catalog
# --------------------------------------------
class CatSim2017Catalog(BaseGalaxyCatalog):
    """Galaxy catalog from CatSim 2017 (``OneDegSq.fits``).

    Each galaxy is a single Sersic profile with half-light radius,
    Sersic index, axis ratio, and position angle read from the
    input FITS file.
    """

    catalog_filename = "OneDegSq.fits"

    # ``prob`` drives sampling, ``a_*``/``b_*``/``pa_*``/``fluxnorm_*``
    # the profile, ``*_ab`` the photometry.  ``ra``/``dec`` are read only
    # to measure the footprint -- the galaxies are placed by the layout,
    # not at their input-file positions -- and never reach the truth
    # catalog, whose own ``ra``/``dec`` columns take precedence in the
    # merge.  That footprint is the 1 deg^2 of the name, centred on
    # RA = 0, so it is the wrap handling in
    # ``_radec_box_area_arcmin2`` that keeps it from reading 360 deg^2.
    # ``galtileid`` is unused.
    required_columns: ClassVar[tuple[str, ...] | None] = (
        "ra",
        "dec",
        "prob",
        "redshift",
        "a_d",
        "b_d",
        "a_b",
        "b_b",
        "pa_disk",
        "pa_bulge",
        "fluxnorm_disk",
        "fluxnorm_bulge",
        "fluxnorm_agn",
        "u_ab",
        "g_ab",
        "r_ab",
        "i_ab",
        "z_ab",
        "y_ab",
    )

    @classmethod
    def magnitude_columns(cls, survey_name: str, band: str) -> tuple[str, ...]:
        """``*_ab`` photometry, shared by the surveys this catalog covers."""
        if survey_name not in ("lsst", "hsc", "des"):
            raise ValueError(
                f"catsim2017 has no {survey_name!r} photometry; supported "
                "surveys are ['lsst', 'hsc', 'des']"
            )
        return (f"{band}_ab",)

    def _probabilities_for_sampling(self, cat):
        if "prob" in cat.dtype.names and cat.size > 0:
            p = cat["prob"].astype(float)
            p_sum = np.sum(p)
            if p_sum > 0:
                return p / p_sum
        return None

    def _half_light_radius(self, catalog) -> np.ndarray:
        return np.sqrt(np.maximum(catalog["a_d"], 1e-9) * np.maximum(catalog["b_d"], 1e-9))

    def _generate_galaxy(
        self,
        *,
        entry,
        mag_zero,
        band,
        include_point_source=True,
        force_isotropic=False,
        force_galaxy_profile=FORCE_GALAXY_PROFILE_NONE,
        **kwargs,
    ) -> galsim.GSObject:
        """Build a GalSim galaxy from a CatSim 2017 catalog row."""
        dd = entry.copy()
        if not include_point_source:
            dd["fluxnorm_agn"] = 0.0
        ab_magnitude = dd[band + "_ab"]
        total_flux = 10 ** ((mag_zero - ab_magnitude) / 2.5)

        # split flux among components
        total_fluxnorm = dd["fluxnorm_disk"] + dd["fluxnorm_bulge"] + dd["fluxnorm_agn"]
        # guard against zero to avoid NaNs
        if total_fluxnorm <= 0:
            return galsim.Gaussian(flux=total_flux, sigma=1e-4)

        disk_flux = dd["fluxnorm_disk"] / total_fluxnorm * total_flux
        bulge_flux = dd["fluxnorm_bulge"] / total_fluxnorm * total_flux
        agn_flux = dd["fluxnorm_agn"] / total_fluxnorm * total_flux

        components = []

        # Disk
        if disk_flux > 0:
            a_d, b_d = dd["a_d"], dd["b_d"]
            hlr_d = np.sqrt(a_d * b_d)
            if force_isotropic:
                q_d = 1.0
            else:
                q_d = (b_d / a_d) if a_d > 0 else 1.0
            beta_d = np.radians(dd["pa_disk"])
            if force_galaxy_profile > FORCE_GALAXY_PROFILE_NONE:
                disk = _forced_profile(
                    force_galaxy_profile, flux=disk_flux, half_light_radius=hlr_d
                )
            else:
                disk = galsim.Exponential(flux=disk_flux, half_light_radius=hlr_d)
            disk = disk.shear(q=q_d, beta=beta_d * galsim.radians)
            components.append(disk)

        # Bulge
        if bulge_flux > 0:
            a_b, b_b = dd["a_b"], dd["b_b"]
            hlr_b = min(np.sqrt(a_b * b_b), self.max_bulge_hlr_arcsec)
            if force_isotropic:
                q_b = 1.0
            else:
                q_b = (b_b / a_b) if a_b > 0 else 1.0
            beta_b = np.radians(dd["pa_bulge"])
            if force_galaxy_profile > FORCE_GALAXY_PROFILE_NONE:
                bulge = _forced_profile(
                    force_galaxy_profile, flux=bulge_flux, half_light_radius=hlr_b
                )
            else:
                bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=hlr_b)
            bulge = bulge.shear(q=q_b, beta=beta_b * galsim.radians)
            components.append(bulge)

        # AGN (nearly point-like)
        if agn_flux > 0:
            components.append(galsim.Gaussian(flux=agn_flux, sigma=1e-4))

        if not components:
            # fallback if all fluxes zero
            return galsim.Gaussian(flux=total_flux, sigma=1e-4)

        return galsim.Add(components)


# ---------------------------------------------------------
# Concrete implementation: Euclid Flagship 2025 (COSMOS)
# ---------------------------------------------------------
class Flagship2025Catalog(BaseGalaxyCatalog):
    """
    Catalog of galaxies from the Euclid Flagship 2025 simulation
    (COSMOS field extraction, flagship_cosmos.fits).

    The axis ratios (disk_axis_ratio, bulge_axis_ratio) are stored as
    minor/major ratio, which maps directly to GalSim's ``q`` parameter.
    """

    catalog_filename = "flagship_cosmos.fits"

    radec_columns: ClassVar[tuple[str, str]] = ("ra_gal", "dec_gal")

    # Survey-independent columns: ``ra_gal``/``dec_gal`` set the density
    # footprint -- a filled 1.4 x 1.4 deg box in COSMOS -- and the
    # disk/bulge columns the profile.  Photometry is
    # survey-prefixed and added by ``_required_columns`` below, so a run
    # never loads the bands of a survey it is not simulating.  Dropping
    # those plus the unused ``decam_*``, ``disk_nsersic`` and halo columns
    # keeps most of this 3.7M-row catalog out of memory.
    required_columns: ClassVar[tuple[str, ...] | None] = (
        "ra_gal",
        "dec_gal",
        "redshift",
        "pa",
        "bulge_fraction",
        "disk_r50",
        "disk_axis_ratio",
        "bulge_r50",
        "bulge_nsersic",
        "bulge_axis_ratio",
    )

    # Bands carried by this catalog for each survey prefix.  ``hsc`` reuses
    # the LSST photometry, matching ``_generate_galaxy``.
    survey_bands: ClassVar[dict[str, tuple[str, ...]]] = {
        "lsst": ("u", "g", "r", "i", "z", "y"),
        "euclid": ("vis", "nisp_y", "nisp_j", "nisp_h"),
    }

    @classmethod
    def magnitude_columns(cls, survey_name: str, band: str) -> tuple[str, ...]:
        """``{survey}_{band}``; ``hsc`` reuses the LSST photometry."""
        sname = _survey_prefix(survey_name)
        bands = cls.survey_bands.get(sname)
        if bands is None:
            raise ValueError(
                f"flagship2025 has no {survey_name!r} photometry; supported "
                f"surveys are {sorted(cls.survey_bands) + ['hsc']}"
            )
        if band not in bands:
            raise ValueError(
                f"flagship2025 has no {band!r} band for survey "
                f"{survey_name!r}; available bands are {list(bands)}"
            )
        return (f"{sname}_{band}",)

    def _required_columns(self) -> tuple[str, ...] | None:
        """Collect ``{survey}_{band}`` magnitudes of every listed survey."""
        assert self.required_columns is not None
        cols = list(self.required_columns)
        for survey in self.survey_name_list:
            sname = _survey_prefix(survey)
            bands = self.survey_bands.get(sname)
            if bands is None:
                # unknown survey: fall back to reading every column rather
                # than silently dropping the magnitudes the renderer needs
                return None
            for band in bands:
                name = f"{sname}_{band}"
                if name not in cols:
                    cols.append(name)
        return tuple(cols)

    def _half_light_radius(self, catalog) -> np.ndarray:
        return catalog["disk_r50"]

    def _generate_galaxy(
        self,
        *,
        entry,
        mag_zero,
        band,
        survey_name,
        force_isotropic=False,
        force_galaxy_profile=FORCE_GALAXY_PROFILE_NONE,
        **kwargs,
    ) -> galsim.GSObject:
        """Build a GalSim galaxy from a Flagship 2025 catalog row."""
        sname = _survey_prefix(survey_name)

        mag = entry[f"{sname}_{band}"]
        flux = 10 ** ((mag_zero - mag) / 2.5)

        bulge_frac = entry["bulge_fraction"]
        bulge_flux = flux * bulge_frac
        disk_flux = flux * (1.0 - bulge_frac)

        # Position angle (degrees) shared by disk and bulge
        pa = float(entry["pa"]) * galsim.degrees

        components = []

        # Disk (nsersic is always 1.0 in this catalog)
        if disk_flux > 0:
            # axis ratio is minor/major (b/a); clamp to valid range
            q_d_cat = min(max(float(entry["disk_axis_ratio"]), 0.0), 1.0)
            # disk_r50 is the SEMI-MAJOR half-light radius: Flagship assigns
            # it from magnitude before applying inclination (Miller+13 /
            # GALFIT convention).  GalSim's .shear(q=) is area-preserving, so
            # half_light_radius is the circularized radius and the drawn
            # semi-major axis is hlr/sqrt(q).  Pass r50*sqrt(q) so the drawn
            # semi-major axis equals r50 -- the same as CatSim's sqrt(a*b).
            # Circularize with the catalog q even when forcing isotropy, so
            # force_isotropic rounds the shape without changing the size.
            disk_hlr = max(float(entry["disk_r50"]) * np.sqrt(q_d_cat), 1e-4)
            q_d = 1.0 if force_isotropic else q_d_cat
            if force_galaxy_profile > FORCE_GALAXY_PROFILE_NONE:
                disk = _forced_profile(
                    force_galaxy_profile, flux=disk_flux, half_light_radius=disk_hlr
                )
            else:
                disk = galsim.Exponential(
                    flux=disk_flux,
                    half_light_radius=disk_hlr,
                )
            disk = disk.shear(q=q_d, beta=pa)
            components.append(disk)

        # Bulge
        if bulge_flux > 0:
            q_b_cat = min(max(float(entry["bulge_axis_ratio"]), 0.0), 1.0)
            # bulge_r50 is the semi-major half-light radius too (calibrated
            # to CANDELS GALFIT r_e); circularize as for the disk.  The
            # memory cap applies to the drawn (circularized) radius, which
            # is what sets the stamp / FFT size.
            bulge_hlr = min(
                max(float(entry["bulge_r50"]) * np.sqrt(q_b_cat), 1e-4),
                self.max_bulge_hlr_arcsec,
            )
            bulge_n = float(entry["bulge_nsersic"])
            bulge_n = _galsim_round_sersic(bulge_n, 0.1)
            q_b = 1.0 if force_isotropic else q_b_cat
            if force_galaxy_profile > FORCE_GALAXY_PROFILE_NONE:
                bulge = _forced_profile(
                    force_galaxy_profile, flux=bulge_flux, half_light_radius=bulge_hlr
                )
            else:
                bulge = galsim.Sersic(
                    n=bulge_n,
                    flux=bulge_flux,
                    half_light_radius=bulge_hlr,
                )
            bulge = bulge.shear(q=q_b, beta=pa)
            components.append(bulge)

        if not components:
            return galsim.Gaussian(flux=flux, sigma=1e-4)

        return galsim.Add(components)


# ---------------------------------------------------------
# Concrete implementation: Diffsky Simulation
# ---------------------------------------------------------
class DiffskyCatalog(BaseGalaxyCatalog):
    """DiffSky input galaxies (``Diffsky``).

    Galaxies are decomposed into bulge + disk components, each with
    its own Sersic index, half-light radius, and axis ratio.  Read
    from Diffsky mock catalog.
    """

    catalog_filename = "diffsky2026.fits"

    @classmethod
    def magnitude_columns(cls, survey_name: str, band: str) -> tuple[str, ...]:
        """Disk and bulge are stored separately in this catalog."""
        sname = _survey_prefix(survey_name)
        if sname != "lsst":
            raise ValueError(
                f"diffsky has no {survey_name!r} photometry; supported "
                "surveys are ['lsst', 'hsc']"
            )
        return (f"{sname}_{band}_disk", f"{sname}_{band}_bulge")

    def _half_light_radius(self, catalog) -> np.ndarray:
        return catalog["r50_disk_as"]

    def _generate_galaxy(
        self,
        *,
        entry,
        mag_zero,
        band,
        survey_name,
        force_isotropic=False,
        force_galaxy_profile=FORCE_GALAXY_PROFILE_NONE,
        **kwargs,
    ) -> galsim.GSObject:
        """Build a GalSim galaxy from a Diffsky catalog row."""
        sname = _survey_prefix(survey_name)

        bulge_hlr = min(float(entry["r50_bulge_as"]), self.max_bulge_hlr_arcsec)
        disk_hlr = entry["r50_disk_as"]

        # shear-ellipticity components
        if force_isotropic:
            disk_e1, disk_e2 = 0.0, 0.0
            bulge_e1, bulge_e2 = 0.0, 0.0
        else:
            # ellipticity = 1 - q in diffsky catalog
            disk_e = entry["ellipticity_disk"] / (2 - entry["ellipticity_disk"])
            disk_e1 = disk_e * np.cos(2 * entry["psi_disk"])
            disk_e2 = disk_e * np.sin(2 * entry["psi_disk"])

            bulge_e = entry["ellipticity_bulge"] / (2 - entry["ellipticity_bulge"])
            bulge_e1 = bulge_e * np.cos(2 * entry["psi_bulge"])
            bulge_e2 = bulge_e * np.sin(2 * entry["psi_bulge"])

        disk_mag = entry[f"{sname}_{band}_disk"]
        disk_flux = 10 ** ((mag_zero - disk_mag) / 2.5)
        if force_galaxy_profile > FORCE_GALAXY_PROFILE_NONE:
            disk = _forced_profile(
                force_galaxy_profile, flux=disk_flux, half_light_radius=disk_hlr
            )
        else:
            disk = galsim.Exponential(
                flux=disk_flux,
                half_light_radius=disk_hlr,
            )
        disk = disk.shear(g1=disk_e1, g2=disk_e2)

        bulge_mag = entry[f"{sname}_{band}_bulge"]
        bulge_flux = 10 ** ((mag_zero - bulge_mag) / 2.5)
        if force_galaxy_profile > FORCE_GALAXY_PROFILE_NONE:
            bulge = _forced_profile(
                force_galaxy_profile, flux=bulge_flux, half_light_radius=bulge_hlr
            )
        else:
            bulge = galsim.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr)
        bulge = bulge.shear(g1=bulge_e1, g2=bulge_e2)

        gal = (bulge + disk).withFlux(disk_flux + bulge_flux)
        return gal


# ---------------------------------------------------------
# Concrete implementation: object-table scene (e.g. a DP2 cluster field)
# ---------------------------------------------------------

# AB magnitude of 1 nJy; the Rubin object tables report fluxes in nJy.
AB_MAG_ZERO_NJY = 31.4


def _column_to_numpy(column) -> np.ndarray | None:
    """Numeric or boolean NumPy array for a pandas column; ``None`` for text.

    Handles NumPy-backed, nullable and Arrow-backed dtypes alike: integers
    with missing values fall back to float with NaN, and booleans with
    missing values become ``False``.
    """
    kind = getattr(getattr(column, "dtype", None), "kind", "O")
    if kind == "b":
        try:
            return column.to_numpy(dtype=bool)
        except (TypeError, ValueError):
            return column.fillna(False).to_numpy(dtype=bool)
    if kind in "iu":
        try:
            return column.to_numpy(dtype=np.int64)
        except (TypeError, ValueError):
            return column.to_numpy(dtype=float, na_value=np.nan)
    if kind == "f":
        return column.to_numpy(dtype=float, na_value=np.nan)
    if kind in "USO":
        return None
    try:
        return column.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        return None


def _columns_to_structured_array(columns: dict[str, np.ndarray]) -> np.ndarray:
    """Pack a mapping of equal-length arrays into a structured array."""
    if not columns:
        raise ValueError("scene has no usable (numeric or boolean) columns")
    num = len(next(iter(columns.values())))
    for name, values in columns.items():
        if len(values) != num:
            raise ValueError(f"scene column {name!r} has {len(values)} rows, expected {num}")
    out = np.empty(num, dtype=[(name, values.dtype) for name, values in columns.items()])
    for name, values in columns.items():
        out[name] = values
    return out


def scene_to_structured_array(scene) -> np.ndarray:
    """Coerce a scene table into a NumPy structured array.

    Accepts a structured array, an ``astropy.table.Table``, a
    ``pandas.DataFrame`` (Arrow-backed columns included, e.g. the result
    of an ``lsdb`` cone search), a mapping of column name to array, or the
    path of a FITS or Parquet file.  Text columns are dropped: the truth
    catalog only carries numbers.
    """
    if isinstance(scene, np.ndarray):
        if scene.dtype.names is None:
            raise TypeError("a scene array must be a structured array with named fields")
        return scene
    if isinstance(scene, (str, os.PathLike)):
        fname = os.fspath(scene)
        if fname.lower().endswith((".parq", ".parquet")):
            import pyarrow.parquet as pq

            return scene_to_structured_array(pq.read_table(fname).to_pandas())
        return get_catalog(fname)
    if hasattr(scene, "as_array"):  # astropy Table
        arr = scene.as_array()
        if isinstance(arr, np.ma.MaskedArray):
            arr = arr.filled()
        return np.asarray(arr)
    if hasattr(scene, "columns") and hasattr(scene, "to_numpy"):  # pandas DataFrame
        columns = {}
        for name in scene.columns:
            values = _column_to_numpy(scene[name])
            if values is not None:
                columns[str(name)] = values
        return _columns_to_structured_array(columns)
    if isinstance(scene, Mapping):
        return _columns_to_structured_array({str(name): np.asarray(values) for name, values in scene.items()})
    raise TypeError(f"cannot build a scene from an object of type {type(scene).__name__}")


class ClusterSceneCatalog(BaseGalaxyCatalog):
    """Render every object of an input table at its own sky position.

    The other catalogs sample galaxies from a static file and place them
    with a :class:`~xlens.simulator.layout.Layout`.  This one draws a
    *scene*: each row of the input table is rendered exactly once, at its
    own position, with its own single-Sersic morphology and photometry.
    It was written to re-simulate real cluster fields from the Rubin DP2
    object table (see :meth:`from_dp2_objects`), but any table with the
    columns below works, for instance a model cluster whose members were
    drawn from an NFW profile.

    The truth catalog it builds has the same columns as the other
    catalogs, so the scene goes through ``CatalogTask`` (rotation,
    lensing) and ``MultibandSimTask`` unchanged with
    ``galaxy_type = "cluster_scene"``; :meth:`draw_on_image` renders it
    onto an existing image outside the pipeline.

    Input columns
    -------------
    position
        ``ra``, ``dec`` in degrees, **or** ``dx``, ``dy`` in arcsec on the
        tangent plane relative to the centre of the tract bounding box
        (``+dx`` along the pixel ``+x`` axis, ``+dy`` along ``+y``).
        ``ra``/``dec`` win when both are present.
    ``redshift``
        Used by the lensing perturbation; objects at or below the lens
        redshift are not lensed, so cluster members should carry the
        cluster redshift and stars ``0``.
    ``sersic_n``, ``r50_major``, ``r50_minor``, ``theta``
        Sersic index, semi-major and semi-minor half-light radii in
        arcsec, and position angle in degrees counter-clockwise from the
        pixel ``+x`` axis.  The position angle becomes the ``angles``
        column of the truth catalog, so :meth:`rotate` turns the
        orientation together with the position.
    ``{survey}_{band}``
        AB magnitudes, e.g. ``lsst_i``; ``hsc`` reuses the ``lsst``
        columns.  A non-finite magnitude renders nothing.
    ``is_point_source`` (optional)
        Rows flagged ``True`` are rendered as point sources with the same
        magnitude columns.

    Every other column of the table is carried into the truth catalog.
    """

    # Read from ``catsim_dir`` when no ``scene`` is given (the pipeline
    # fallback path); FITS or Parquet.
    catalog_filename: ClassVar[str] = "cluster_scene.fits"
    required_columns: ClassVar[tuple[str, ...] | None] = None

    scene_columns: ClassVar[tuple[str, ...]] = (
        "redshift",
        "sersic_n",
        "r50_major",
        "r50_minor",
        "theta",
    )
    # GalSim's Sersic profile is defined for 0.3 <= n <= 6.2; indices are
    # clipped to this range and rounded to 0.1 so that GalSim can reuse
    # its Sersic look-up tables across objects.
    sersic_n_bounds: ClassVar[tuple[float, float]] = (0.3, 6.2)
    # Whole-object half-light radius cap (arcsec), for the same memory
    # reason as :data:`MAX_BULGE_HLR_ARCSEC`: the scene is a single Sersic
    # per object, and a high-n profile with a large radius makes GalSim
    # size its stamp at thousands of pixels.
    max_hlr_arcsec: ClassVar[float] = MAX_BULGE_HLR_ARCSEC
    min_axis_ratio: ClassVar[float] = 0.05

    def __init__(
        self,
        *,
        tract_info,
        scene=None,
        rng: np.random.RandomState | None = None,
        layout_name: str = "scene",
        sep_arcsec: float | None = None,
        indice_group_id: int | None = None,
        select_observable: list[str] | str | None = None,
        select_lower_limit: Iterable[float] | None = None,
        select_upper_limit: Iterable[float] | None = None,
        extend_ratio: float = 1.08,
        force_pixel_center: bool = False,
        catsim_dir: str | None = None,
        survey_name_list: Iterable[str] | None = None,
    ):
        """Build the truth catalog of a scene.

        Parameters
        ----------
        tract_info : lsst.skymap.tractInfo.ExplicitTractInfo
            Provides the WCS and bounding box that define the pixel frame.
        scene : structured array, Table, DataFrame, mapping or path, optional
            The object table (see the class docstring for the columns).
            When *None*, ``catalog_filename`` is read from ``catsim_dir``,
            which is how the pipeline builds the catalog.
        rng, layout_name, sep_arcsec, indice_group_id, extend_ratio
            Accepted so the class can be constructed like the layout-based
            catalogs; a scene has no random placement, so they are unused.
        select_observable, select_lower_limit, select_upper_limit
            Optional per-column cuts, as for the other catalogs.
        force_pixel_center : bool, optional
            Snap object centres to pixel centres.
        catsim_dir : str or None, optional
            Directory holding ``catalog_filename``; defaults to
            ``$CATSIM_DIR``.
        survey_name_list : iterable of str or None, optional
            Surveys whose ``{survey}_{band}`` magnitudes the table must
            carry.  Defaults to ``["lsst"]``.
        """
        self.catsim_dir = catsim_dir or os.environ.get("CATSIM_DIR", ".")
        if survey_name_list is None:
            survey_name_list = ["lsst"]
        self.survey_name_list = tuple(str(name).lower() for name in survey_name_list)
        self.prepare_tract_info(tract_info)
        wcs = tract_info.getWcs()
        ps = float(wcs.getPixelScale().asArcseconds())
        self.pixel_scale = ps

        if scene is None:
            table, row_ids = self._read_catalog(
                select_observable=select_observable,
                select_lower_limit=select_lower_limit,
                select_upper_limit=select_upper_limit,
            )
        else:
            table, row_ids = self._apply_selection(
                scene_to_structured_array(scene),
                select_observable=select_observable,
                select_lower_limit=select_lower_limit,
                select_upper_limit=select_upper_limit,
            )
        names = tuple(table.dtype.names)
        self._check_scene_columns(names)
        num = len(table)

        # tangent-plane offsets (arcsec) from the tract centre, the frame
        # ``rotate`` and ``lens`` work in
        if "ra" in names and "dec" in names:
            pix_x, pix_y = wcs.skyToPixelArray(
                np.asarray(table["ra"], dtype=float),
                np.asarray(table["dec"], dtype=float),
                degrees=True,
            )
            dx = (np.asarray(pix_x) - self.x_center) * ps
            dy = (np.asarray(pix_y) - self.y_center) * ps
        else:
            dx = np.asarray(table["dx"], dtype=float)
            dy = np.asarray(table["dy"], dtype=float)
        if force_pixel_center:
            inv_pixel_scale = 1.0 / ps
            dx = (np.round(dx * inv_pixel_scale) + 0.5) * ps
            dy = (np.round(dy * inv_pixel_scale) + 0.5) * ps
        ra, dec = wcs.pixelToSkyArray(
            x=self.x_center + dx / ps,
            y=self.y_center + dy / ps,
            degrees=True,
        )

        placement = np.zeros(num, dtype=PLACEMENT_DTYPE)
        placement["dx"] = dx
        placement["dy"] = dy
        # the position angle is the intrinsic orientation: ``_generate_galaxy``
        # builds every profile along +x and ``get_obj`` rotates it by this
        placement["angles"] = np.radians(np.nan_to_num(np.asarray(table["theta"], dtype=float), nan=0.0))
        placement["ra"] = ra
        placement["dec"] = dec
        placement["prelensed_ra"] = ra
        placement["prelensed_dec"] = dec
        placement["has_finite_shear"] = np.ones(num, dtype=bool)
        placement["indices"] = row_ids
        placement["redshift"] = np.asarray(table["redshift"], dtype=float)
        placement["hlr"] = self._build_hlr_array(table)

        extra = [name for name in names if name not in placement.dtype.names]
        if extra:
            self.data = np.asarray(
                rfn.merge_arrays(
                    [placement, table[extra]],
                    flatten=True,
                    usemask=False,
                )
            )
        else:
            self.data = placement
        self.dtype = self.data.dtype
        self.lensed = False
        return

    def _check_scene_columns(self, names: tuple[str, ...]) -> None:
        missing = [name for name in self.scene_columns if name not in names]
        if not ({"ra", "dec"} <= set(names) or {"dx", "dy"} <= set(names)):
            missing.append("ra/dec or dx/dy")
        for survey in self.survey_name_list:
            prefix = _survey_prefix(survey) + "_"
            if not any(name.startswith(prefix) for name in names):
                missing.append(f"{prefix}{{band}} magnitudes")
        if missing:
            raise ValueError("scene table is missing columns: " + ", ".join(missing))

    @classmethod
    def magnitude_columns(cls, survey_name: str, band: str) -> tuple[str, ...]:
        """``{survey}_{band}``; ``hsc`` reuses the LSST photometry."""
        return (f"{_survey_prefix(survey_name)}_{band}",)

    def _load_catalog_file(self, fname: str, columns=None) -> Any:
        """FITS through :func:`get_catalog`, Parquet through pyarrow."""
        if fname.lower().endswith((".parq", ".parquet")):
            import pyarrow.parquet as pq

            return scene_to_structured_array(pq.read_table(fname, columns=columns).to_pandas())
        return get_catalog(fname, columns=columns)

    def _half_light_radius(self, catalog) -> np.ndarray:
        """Circularised radius ``sqrt(a * b)`` of the Sersic fit (arcsec)."""
        return np.sqrt(
            np.maximum(np.asarray(catalog["r50_major"], dtype=float), 1e-9)
            * np.maximum(np.asarray(catalog["r50_minor"], dtype=float), 1e-9)
        )

    def _generate_galaxy(
        self,
        *,
        entry,
        mag_zero,
        band,
        survey_name="lsst",
        include_point_source=True,
        force_isotropic=False,
        force_galaxy_profile=FORCE_GALAXY_PROFILE_NONE,
        **kwargs,
    ) -> galsim.GSObject:
        """Build a GalSim object from one scene row.

        The profile is built with its major axis along ``+x``; the
        position angle is applied by :meth:`get_obj` through the
        ``angles`` column.
        """
        sname = _survey_prefix(survey_name or "lsst")
        mag = float(entry[f"{sname}_{band}"])
        flux = 10 ** ((mag_zero - mag) / 2.5) if np.isfinite(mag) else 0.0

        names = entry.dtype.names
        if "is_point_source" in names and bool(entry["is_point_source"]):
            if not include_point_source:
                flux = 0.0
            # same nearly-point-like convention as the CatSim AGN component
            return galsim.Gaussian(flux=flux, sigma=1e-4)

        a = float(entry["r50_major"])
        b = float(entry["r50_minor"])
        if not (np.isfinite(a) and np.isfinite(b)):
            a = b = 1e-4
        a = max(a, 1e-4)
        b = max(b, 1e-4)
        # GalSim's ``shear(q=)`` preserves area, so ``half_light_radius`` is
        # the circularised radius and the drawn semi-major axis is
        # hlr / sqrt(q) = a, as for CatSim's sqrt(a * b).
        hlr = min(np.sqrt(a * b), self.max_hlr_arcsec)
        q_cat = min(max(b / a, self.min_axis_ratio), 1.0)
        q = 1.0 if force_isotropic else q_cat

        if force_galaxy_profile > FORCE_GALAXY_PROFILE_NONE:
            gal = _forced_profile(force_galaxy_profile, flux=flux, half_light_radius=hlr)
        else:
            n = float(entry["sersic_n"])
            if not np.isfinite(n):
                n = 1.0
            n_min, n_max = self.sersic_n_bounds
            n = _galsim_round_sersic(min(max(n, n_min), n_max), 0.1)
            gal = galsim.Sersic(n=n, flux=flux, half_light_radius=hlr)
        return gal.shear(q=q, beta=0.0 * galsim.radians)

    # ---------- building a scene from the Rubin object table ----------

    dp2_columns: ClassVar[tuple[str, ...]] = (
        "coord_ra",
        "coord_dec",
        "sersic_index",
        "sersic_reff_major",
        "sersic_reff_minor",
        "sersic_theta",
    )

    @classmethod
    def from_dp2_objects(
        cls,
        objects,
        *,
        tract_info,
        bands: Iterable[str] = ("u", "g", "r", "i", "z", "y"),
        survey_name: str = "lsst",
        flux_column: str = "cModelFlux",
        point_source_flux_column: str = "psfFlux",
        extendedness_column: str = "refExtendedness",
        extendedness_threshold: float = 0.5,
        redshift_column: str | None = "bpz_z_best",
        default_redshift: float = 0.0,
        **kwargs,
    ) -> "ClusterSceneCatalog":
        """Build a scene from rows of the Rubin DP2 ``Object`` table.

        ``objects`` is anything :func:`scene_to_structured_array` accepts,
        typically the ``pandas.DataFrame`` of an ``lsdb`` cone search
        around a cluster, or a Parquet file of it.  Column mapping:

        * ``coord_ra``, ``coord_dec`` -> ``ra``, ``dec``
        * ``sersic_index`` -> ``sersic_n``; ``sersic_reff_major``,
          ``sersic_reff_minor`` (arcsec) -> ``r50_major``, ``r50_minor``;
          ``sersic_theta`` (deg) -> ``theta``
        * ``{band}_{flux_column}`` (nJy) -> ``{survey_name}_{band}`` AB
          magnitude; point sources use ``{band}_{point_source_flux_column}``
          instead when that column exists.
        * ``{extendedness_column} < extendedness_threshold`` ->
          ``is_point_source`` (every row is a galaxy when the column is
          absent).
        * ``redshift_column`` -> ``redshift``; missing or non-finite
          values get ``default_redshift``, point sources ``0``.
        * ``objectId`` is carried through when present.

        Remaining keyword arguments go to the constructor (selection cuts,
        ``force_pixel_center``, ...).
        """
        table = scene_to_structured_array(objects)
        names = tuple(table.dtype.names)
        bands = tuple(bands)
        needed = list(cls.dp2_columns) + [f"{band}_{flux_column}" for band in bands]
        missing = [name for name in needed if name not in names]
        if missing:
            raise ValueError("DP2 object table is missing columns: " + ", ".join(missing))
        num = len(table)
        columns: dict[str, np.ndarray] = {
            "ra": np.asarray(table["coord_ra"], dtype=float),
            "dec": np.asarray(table["coord_dec"], dtype=float),
            "sersic_n": np.asarray(table["sersic_index"], dtype=float),
            "r50_major": np.asarray(table["sersic_reff_major"], dtype=float),
            "r50_minor": np.asarray(table["sersic_reff_minor"], dtype=float),
            "theta": np.asarray(table["sersic_theta"], dtype=float),
        }
        if extendedness_column in names:
            ext = np.asarray(table[extendedness_column], dtype=float)
            point = np.isfinite(ext) & (ext < extendedness_threshold)
        else:
            point = np.zeros(num, dtype=bool)
        columns["is_point_source"] = point

        redshift = np.full(num, float(default_redshift))
        if redshift_column is not None and redshift_column in names:
            z_in = np.asarray(table[redshift_column], dtype=float)
            good = np.isfinite(z_in)
            redshift[good] = z_in[good]
        redshift[point] = 0.0
        columns["redshift"] = redshift

        prefix = _survey_prefix(survey_name)
        with np.errstate(divide="ignore", invalid="ignore"):
            for band in bands:
                flux = np.asarray(table[f"{band}_{flux_column}"], dtype=float)
                point_flux_column = f"{band}_{point_source_flux_column}"
                if point.any() and point_flux_column in names:
                    flux = np.where(point, np.asarray(table[point_flux_column], dtype=float), flux)
                columns[f"{prefix}_{band}"] = np.where(
                    flux > 0, -2.5 * np.log10(flux) + AB_MAG_ZERO_NJY, np.nan
                )
        if "objectId" in names:
            columns["objectId"] = np.asarray(table["objectId"])
        return cls(
            scene=_columns_to_structured_array(columns),
            tract_info=tract_info,
            survey_name_list=[survey_name],
            **kwargs,
        )

    # ---------- rendering onto an existing image ----------

    def draw_on_image(
        self,
        image,
        *,
        band: str,
        mag_zero: float,
        psf_obj: galsim.GSObject | None = None,
        wcs=None,
        survey_name: str = "lsst",
        draw_method: str = "auto",
        nn_trunc: int | None = None,
        use_field_distortion: bool = True,
        force_isotropic: bool = False,
        force_galaxy_profile: int = FORCE_GALAXY_PROFILE_NONE,
        include_point_source: bool = True,
        inclusion_padding: float = SIM_INCLUSION_PADDING,
    ) -> np.ndarray:
        """Render the scene onto ``image`` in place.

        This is the drawing loop of
        :meth:`~xlens.simulator.sim.MultibandSimTask.draw_catalog` applied
        to an image the caller already has, so a cluster can be added to
        a simulated exposure, or drawn onto a blank one, without running
        the pipeline.

        Parameters
        ----------
        image
            ``lsst.afw.image.Exposure``, ``MaskedImage`` or ``Image`` (its
            bounding box sets the pixel coordinates; an ``Exposure`` also
            supplies the WCS and PSF when ``wcs``/``psf_obj`` are not
            given), a ``galsim.Image`` (its ``xmin``/``ymin`` and ``wcs``
            are used), or a C-contiguous float32/float64 2-D NumPy array
            whose first pixel is ``(0, 0)``.  Pixels are added to it.
        band : str
            Photometric band.
        mag_zero : float
            Magnitude zero point of the image.
        psf_obj : galsim.GSObject or None, optional
            PSF to convolve with.  Defaults to the PSF of an ``Exposure``,
            evaluated at the image centre.
        wcs : lsst.afw.geom.SkyWcs or galsim.BaseWCS or None, optional
            Sky-to-pixel mapping; required unless ``image`` carries one.
        survey_name : str, optional
            Survey whose magnitude columns are used.
        draw_method, nn_trunc, use_field_distortion, force_isotropic,
        force_galaxy_profile, include_point_source
            As the same-named ``MultibandSimTask`` configuration fields.
        inclusion_padding : float, optional
            Objects centred more than this many pixels outside the image
            are skipped.

        Returns
        -------
        numpy.ndarray
            Structured array with one row per scene object: ``index`` into
            :attr:`data`, ``ra``, ``dec``, the image position ``image_x``,
            ``image_y`` and whether the object was ``drawn``.
        """
        from ..wcs import tanwcs_dm2galsim

        target = _image_target(image, wcs, self.pixel_scale)
        wcs_dm, wcs_gs = target["wcs_dm"], target["wcs_gs"]
        if wcs_dm is not None:
            wcs_gs = tanwcs_dm2galsim(wcs_dm)
            pix_x, pix_y = wcs_dm.skyToPixelArray(self.data["ra"], self.data["dec"], degrees=True)
        elif wcs_gs is not None:
            pix_x, pix_y = wcs_gs.radecToxy(self.data["ra"], self.data["dec"], units=galsim.degrees)
        else:
            raise ValueError("draw_on_image needs a wcs: pass one or use an image that carries one")
        pix_x = np.asarray(pix_x, dtype=float)
        pix_y = np.asarray(pix_y, dtype=float)

        if psf_obj is None:
            psf_obj = target["psf_obj"]
        if psf_obj is None:
            raise ValueError("draw_on_image needs psf_obj unless image is an Exposure with a PSF")

        gs_image = galsim.Image(target["array"], xmin=target["xmin"], ymin=target["ymin"], wcs=wcs_gs)
        xmin, xmax = gs_image.bounds.xmin, gs_image.bounds.xmax
        ymin, ymax = gs_image.bounds.ymin, gs_image.bounds.ymax

        truth = np.zeros(
            len(self.data),
            dtype=[
                ("index", "i8"),
                ("ra", "f8"),
                ("dec", "f8"),
                ("image_x", "f8"),
                ("image_y", "f8"),
                ("drawn", "bool"),
            ],
        )
        truth["index"] = np.arange(len(self.data))
        truth["ra"] = self.data["ra"]
        truth["dec"] = self.data["dec"]
        truth["image_x"] = pix_x
        truth["image_y"] = pix_y

        for i, src in enumerate(self.data):
            ix, iy = pix_x[i], pix_y[i]
            if not (
                ((xmin - inclusion_padding) < ix < (xmax + inclusion_padding))
                and ((ymin - inclusion_padding) < iy < (ymax + inclusion_padding))
                and src["has_finite_shear"]
            ):
                continue
            image_pos = galsim.PositionD(x=ix, y=iy)
            gal_obj = self.get_obj(
                ind=i,
                mag_zero=mag_zero,
                band=band,
                force_isotropic=force_isotropic,
                force_galaxy_profile=force_galaxy_profile,
                include_point_source=include_point_source,
                survey_name=survey_name,
            )
            convolved_object = galsim.Convolve([gal_obj, psf_obj])
            if use_field_distortion:
                stamp = convolved_object.drawImage(
                    center=image_pos,
                    wcs=wcs_gs.local(image_pos=image_pos),
                    method=draw_method,
                    nx=nn_trunc,
                    ny=nn_trunc,
                )
            else:
                stamp = convolved_object.drawImage(
                    center=image_pos,
                    wcs=None,
                    method=draw_method,
                    scale=self.pixel_scale,
                    nx=nn_trunc,
                    ny=nn_trunc,
                )
            bounds = stamp.bounds & gs_image.bounds
            if bounds.isDefined():
                gs_image[bounds] += stamp[bounds]
                truth["drawn"][i] = True
        return truth


def _image_target(image, wcs, pixel_scale: float) -> dict[str, Any]:
    """Resolve the pixel array, origin, WCS and PSF of ``image``.

    See :meth:`ClusterSceneCatalog.draw_on_image` for the accepted types.
    The returned ``array`` is a view onto the image's pixels, so drawing
    into it modifies ``image``.  ``pixel_scale`` (arcsec) sizes the PSF of
    an ``Exposure`` that carries no WCS of its own.
    """
    target: dict[str, Any] = {
        "xmin": 0,
        "ymin": 0,
        "wcs_dm": None,
        "wcs_gs": None,
        "psf_obj": None,
    }
    if isinstance(image, galsim.Image):
        target["array"] = image.array
        target["xmin"], target["ymin"] = image.xmin, image.ymin
        if wcs is None:
            wcs = image.wcs
    elif isinstance(image, np.ndarray):
        if image.ndim != 2:
            raise TypeError("image array must be 2-D")
        target["array"] = image
    elif hasattr(image, "getBBox"):
        if hasattr(image, "getMaskedImage"):  # Exposure
            target["array"] = image.getMaskedImage().getImage().getArray()
            if wcs is None and hasattr(image, "getWcs"):
                wcs = image.getWcs()
            psf = image.getPsf() if hasattr(image, "getPsf") else None
            if psf is not None:
                import lsst.geom as geom

                kernel = psf.computeKernelImage(geom.Point2D(image.getBBox().getCenter())).getArray()
                image_wcs = image.getWcs()
                scale = (
                    float(image_wcs.getPixelScale().asArcseconds())
                    if image_wcs is not None
                    else pixel_scale
                )
                target["psf_obj"] = galsim.InterpolatedImage(
                    galsim.Image(np.array(kernel, dtype=float)), scale=scale, flux=1.0
                )
        elif hasattr(image, "getImage"):  # MaskedImage
            target["array"] = image.getImage().getArray()
        else:  # Image
            target["array"] = image.getArray()
        bbox = image.getBBox()
        target["xmin"], target["ymin"] = bbox.getMinX(), bbox.getMinY()
    else:
        raise TypeError(f"cannot draw on an object of type {type(image).__name__}")

    array = target["array"]
    if array.dtype not in (np.float32, np.float64) or not array.flags["C_CONTIGUOUS"]:
        raise TypeError("the image pixels must be a C-contiguous float32 or float64 array")
    if not array.flags["WRITEABLE"]:
        raise TypeError("the image pixels are read-only")

    if wcs is not None:
        if hasattr(wcs, "skyToPixelArray"):
            target["wcs_dm"] = wcs
        elif isinstance(wcs, galsim.BaseWCS):
            target["wcs_gs"] = wcs
        else:
            raise TypeError(f"wcs must be an lsst SkyWcs or a galsim WCS, not {type(wcs).__name__}")
    return target


# ---------------------------------------------------------
# galaxy_type registry
# ---------------------------------------------------------
GALAXY_CATALOG_CLASSES: dict[str, type[BaseGalaxyCatalog]] = {
    "catsim2017": CatSim2017Catalog,
    "flagship2025": Flagship2025Catalog,
    "diffsky": DiffskyCatalog,
    "cluster_scene": ClusterSceneCatalog,
}


def get_catalog_class(galaxy_type: str) -> type[BaseGalaxyCatalog]:
    """Return the catalog class implementing *galaxy_type*."""
    try:
        return GALAXY_CATALOG_CLASSES[galaxy_type]
    except KeyError:
        raise ValueError(
            f"invalid galaxy_type {galaxy_type!r}; expected one of "
            f"{sorted(GALAXY_CATALOG_CLASSES)}"
        ) from None
