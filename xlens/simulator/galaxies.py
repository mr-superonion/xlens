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
import pyarrow.parquet as pq
from numpy.lib import recfunctions as rfn

from . import mog
from .layout import Layout


def _galsim_round_sersic(n, sersic_prec):
    """Round a Sersic index to the nearest multiple of *sersic_prec*."""
    return float(int(n / sersic_prec + 0.5)) * sersic_prec


def get_catalog(fname):
    """Read a FITS or Parquet catalog and append an ``indices`` column.

    Parameters
    ----------
    fname : str
        Path to a FITS or Parquet file readable by ``fitsio``.

    Returns
    -------
    numpy.ndarray
        Structured array with an additional ``indices`` column (``int32``).
    """
    cat = fitsio.read(fname)
    idx = np.arange(len(cat), dtype=np.int32)
    cat = rfn.append_fields(
        cat,
        "indices",
        idx,
        dtypes=[np.int32],
        usemask=False,
    )
    return cat


class BaseGalaxyCatalog(ABC):
    """
    Abstract base class for galaxy catalogs used to build GalSim objects.

    Subclasses must implement:
      - _read_catalog(...)
      - _compute_density(cat)
      - _generate_galaxy(entry, mag_zero, band, **kwargs)
    Optionally override:
      - _probabilities_for_sampling(cat) -> Optional[np.ndarray]
    """

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
        """
        self.catsim_dir = catsim_dir or os.environ.get("CATSIM_DIR", ".")
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
        input_catalog = self._read_catalog(
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
        placement["indices"] = selected["indices"]
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

    def _load_catalog_file(self, fname: str) -> Any:
        """Load the raw catalog from ``fname``.

        Default implementation reads a FITS file via :func:`get_catalog`,
        which also appends an ``indices`` column.  Subclasses with a
        non-FITS on-disk format (e.g. Parquet) should override this.
        """
        return get_catalog(fname)

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
        """
        if select_observable is None:
            return cat
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
        return cat[mask]

    def _read_catalog(
        self,
        *,
        select_observable=None,
        select_lower_limit=None,
        select_upper_limit=None,
    ) -> Any:
        """Load the input galaxy catalog and apply optional selection cuts.

        Subclasses customise this by setting ``catalog_filename`` and,
        if needed, overriding :meth:`_load_catalog_file`.
        """
        fname = os.path.join(self.catsim_dir, self.catalog_filename)
        if not os.path.isfile(fname):
            raise FileNotFoundError(
                f"Cannot find '{self.catalog_filename}' under "
                f"{self.catsim_dir}. "
                "Please download it and place it under $CATSIM_DIR."
            )
        cat = self._load_catalog_file(fname)
        return self._apply_selection(
            cat,
            select_observable=select_observable,
            select_lower_limit=select_lower_limit,
            select_upper_limit=select_upper_limit,
        )

    @abstractmethod
    def _compute_density(self, cat: Any) -> float:
        """Return object surface density in objects / arcmin^2."""

    @abstractmethod
    def _generate_galaxy(
        self, *, entry: Any, mag_zero: float, band: str, use_mog=False, **kwargs
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
        use_mog: bool = False,
        force_isotropic: bool = False,
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
        use_mog : bool, optional
            Use Mixture-of-Gaussians profiles instead of native GalSim.
        force_isotropic : bool, optional
            Force all galaxies to have circular isophotes.
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
            use_mog=use_mog,
            include_point_source=include_point_source,
            force_isotropic=force_isotropic,
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

    def _compute_density(self, cat) -> float:
        """Return density in objects/arcmin^2 for a 1-deg^2 catalog."""
        return cat.size / (60.0 * 60.0)

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
        use_mog=False,
        include_point_source=True,
        force_isotropic=False,
        **kwargs,
    ) -> galsim.GSObject:
        """Build a GalSim galaxy from a CatSim 2017 catalog row."""
        if use_mog:
            _simulator = mog
        else:
            _simulator = galsim
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
            disk = _simulator.Exponential(flux=disk_flux, half_light_radius=hlr_d).shear(
                q=q_d, beta=beta_d * galsim.radians
            )
            components.append(disk)

        # Bulge
        if bulge_flux > 0:
            a_b, b_b = dd["a_b"], dd["b_b"]
            hlr_b = np.sqrt(a_b * b_b)
            if force_isotropic:
                q_b = 1.0
            else:
                q_b = (b_b / a_b) if a_b > 0 else 1.0
            beta_b = np.radians(dd["pa_bulge"])
            bulge = _simulator.DeVaucouleurs(flux=bulge_flux, half_light_radius=hlr_b).shear(
                q=q_b, beta=beta_b * galsim.radians
            )
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

    def _compute_density(self, cat) -> float:
        """Return density (objects/arcmin^2) from the sky footprint."""
        ra = cat["ra_gal"]
        dec = cat["dec_gal"]
        ra_range = ra.max() - ra.min()
        dec_range = dec.max() - dec.min()
        cos_dec = np.cos(np.radians(np.mean(dec)))
        area_deg2 = ra_range * cos_dec * dec_range
        area_arcmin2 = area_deg2 * 3600.0
        return len(cat) / area_arcmin2

    def _half_light_radius(self, catalog) -> np.ndarray:
        return catalog["disk_r50"]

    def _generate_galaxy(
        self,
        *,
        entry,
        mag_zero,
        band,
        survey_name,
        use_mog=False,
        force_isotropic=False,
        **kwargs,
    ) -> galsim.GSObject:
        """Build a GalSim galaxy from a Flagship 2025 catalog row."""
        if use_mog:
            raise NotImplementedError("Flagship2025Catalog does not support the MoG renderer")
        sname = survey_name
        if sname == "hsc":
            sname = "lsst"

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
            disk = galsim.Exponential(
                flux=disk_flux,
                half_light_radius=disk_hlr,
            ).shear(q=q_d, beta=pa)
            components.append(disk)

        # Bulge
        if bulge_flux > 0:
            bulge_hlr = max(float(entry["bulge_r50"]), 1e-4)
            bulge_n = float(entry["bulge_nsersic"])
            bulge_n = _galsim_round_sersic(bulge_n, 0.1)
            if force_isotropic:
                q_b = 1.0
            else:
                q_b = float(entry["bulge_axis_ratio"])
                q_b = min(max(q_b, 0.00), 1.0)
            bulge = galsim.Sersic(
                n=bulge_n,
                flux=bulge_flux,
                half_light_radius=bulge_hlr,
            ).shear(q=q_b, beta=pa)
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

    # diffsky_arr.parquet from "hltds_cosmos_260215_04_07_2026"
    catalog_filename = "diffsky_arr.parquet"

    def _load_catalog_file(self, fname: str):
        return pq.read_table(fname).to_pandas().to_records(index=False)

    def _compute_density(self, cat) -> float:
        """Return density in objects/arcmin^2 for a cone with a 1 deg radius"""
        area_tot_arcmin = 2 * np.pi * (1 - np.cos(np.radians(1))) * (180 * 60 / np.pi) ** 2
        return len(cat) / area_tot_arcmin

    def _half_light_radius(self, catalog) -> np.ndarray:
        return catalog["r50_disk_as"]

    def _generate_galaxy(
        self,
        *,
        entry,
        mag_zero,
        band,
        survey_name,
        use_mog=False,
        force_isotropic=False,
        **kwargs,
    ) -> galsim.GSObject:
        """Build a GalSim galaxy from a Diffsky catalog row."""
        if use_mog:
            _simulator = mog
        else:
            _simulator = galsim
        if survey_name == "hsc":
            sname = "lsst"
        else:
            sname = survey_name

        bulge_hlr = entry["r50_bulge_as"]
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
        disk = _simulator.Exponential(
            flux=disk_flux,
            half_light_radius=disk_hlr,
        ).shear(g1=disk_e1, g2=disk_e2)

        bulge_mag = entry[f"{sname}_{band}_bulge"]
        bulge_flux = 10 ** ((mag_zero - bulge_mag) / 2.5)
        bulge = _simulator.DeVaucouleurs(flux=bulge_flux, half_light_radius=bulge_hlr).shear(
            g1=bulge_e1, g2=bulge_e2
        )

        gal = (bulge + disk).withFlux(disk_flux + bulge_flux)
        return gal
