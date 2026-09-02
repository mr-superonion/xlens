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
catalogs.
"""

import os
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable

import fitsio
import galsim
import lsst
import numpy as np
from numpy.lib import recfunctions as rfn

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

        placement_dtype = [
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
        placement = np.zeros(num, dtype=placement_dtype)
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
            disk_hlr = max(float(entry["disk_r50"]), 1e-4)
            if force_isotropic:
                q_d = 1.0
            else:
                q_d = float(entry["disk_axis_ratio"])
                # axis ratio is minor/major; clamp to valid range
                q_d = min(max(q_d, 0.00), 1.0)
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
            bulge_hlr = min(
                max(float(entry["bulge_r50"]), 1e-4),
                self.max_bulge_hlr_arcsec,
            )
            bulge_n = float(entry["bulge_nsersic"])
            bulge_n = _galsim_round_sersic(bulge_n, 0.1)
            if force_isotropic:
                q_b = 1.0
            else:
                q_b = float(entry["bulge_axis_ratio"])
                q_b = min(max(q_b, 0.00), 1.0)
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
# galaxy_type registry
# ---------------------------------------------------------
GALAXY_CATALOG_CLASSES: dict[str, type[BaseGalaxyCatalog]] = {
    "catsim2017": CatSim2017Catalog,
    "flagship2025": Flagship2025Catalog,
    "diffsky": DiffskyCatalog,
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
