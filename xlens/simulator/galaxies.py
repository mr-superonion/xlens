import os
from abc import ABC, abstractmethod
from typing import Any, Iterable

import fitsio
import galsim
import lsst
import numpy as np
from numpy.lib import recfunctions as rfn

from . import mog
from .layout import Layout


def get_catalog(fname):
    cat = fitsio.read(fname)
    idx = np.arange(len(cat), dtype=np.int32)
    cat = rfn.append_fields(
        cat, "indices", idx, dtypes=[np.int32], usemask=False,
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
        self.input_catalog = self._read_catalog(
            select_observable=select_observable,
            select_lower_limit=select_lower_limit,
            select_upper_limit=select_upper_limit,
        )

        # density drives how many objects the layout will place
        density = self._compute_density(self.input_catalog)
        # positions to place galaxies
        shifts_array = layout.get_shifts(
            rng=rng, density=density
        )

        if force_pixel_center:
            inv_pixel_scale = 1.0 / ps
            shifts_array["dx"] = (
                (np.round(shifts_array["dx"] * inv_pixel_scale) + 0.5) * ps
            )
            shifts_array["dy"] = (
                (np.round(shifts_array["dy"] * inv_pixel_scale) + 0.5) * ps
            )

        # choose which catalog rows populate those positions
        num = len(shifts_array)
        catalog_size = len(self.input_catalog)
        if (indice_group_id is None) or (indice_group_id < 0):
            probs = self._probabilities_for_sampling(self.input_catalog)
            integers = np.arange(0, catalog_size, dtype=int)
            idx = rng.choice(integers, size=num, p=probs)
        else:
            indice_min = indice_group_id * num
            indice_max = min(indice_min + num, catalog_size)
            if indice_min >= catalog_size:
                raise ValueError("indice_min too large")
            idx = (
                np.arange(indice_min, indice_max, dtype=int) % catalog_size
            )
            num = indice_max - indice_min
            shifts_array = shifts_array[0:num]
        # random orientation for each placed galaxy
        angles = rng.uniform(low=0.0, high=2.0*np.pi, size=num)
        self.dtype = [
            ("indices", "i8"),
            ("redshift", "f8"),
            ("angles", "f8"),
            ("gamma1", "f8"), ("gamma2", "f8"), ("kappa", "f8"),
            ("dx", "f8"), ("dy", "f8"),
            ("ra", "f8"), ("dec", "f8"),       # post-lensed ra, dec
            ("image_x", "f8"), ("image_y", "f8"),
            ("prelensed_image_x", "f8"), ("prelensed_image_y", "f8"),
            ("has_finite_shear", "bool"),
            ("hlr", "f8"),
        ]
        self.data = np.zeros(num, dtype=self.dtype)
        self.data["dx"] = shifts_array["dx"]
        self.data["dy"] = shifts_array["dy"]
        self.data["angles"] = angles
        self.lensed = False
        self.data["prelensed_image_x"] = self.x_center + self.data["dx"] / ps
        self.data["prelensed_image_y"] = self.y_center + self.data["dy"] / ps
        self.data["image_x"] = self.x_center + self.data["dx"] / ps
        self.data["image_y"] = self.y_center + self.data["dy"] / ps
        self.data["has_finite_shear"] = np.ones(num, dtype=bool)
        self.data["indices"] = self.input_catalog["indices"][idx]
        self.data["redshift"] = self.input_catalog["redshift"][idx]
        self.data["hlr"] = self._build_hlr_array(idx)
        return

    def set_z_source(self, redshift):
        self.data["redshift"][:] = redshift
        return

    def prepare_tract_info(self, tract_info):
        self.tract_info = tract_info
        bbox = tract_info.getBBox()   # lsst.geom.Box2I
        center_pix = bbox.getCenter()
        self.x_center = center_pix.getX()
        self.y_center = center_pix.getY()
        return

    # ---------- required subclass hooks ----------

    @abstractmethod
    def _read_catalog(
        self,
        *,
        select_observable,
        select_lower_limit,
        select_upper_limit,
    ) -> Any:
        """Return the catalog object (array / table) with any requested
        filtering applied."""

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

    def _build_hlr_array(self, indices: np.ndarray) -> np.ndarray:
        catalog = self.input_catalog[indices]
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
        table: np.ndarray,
        tract_info: lsst.skymap.tractInfo.ExplicitTractInfo,
        select_observable: list[str] | str | None = None,
        select_lower_limit: Iterable[float] | None = None,
        select_upper_limit: Iterable[float] | None = None,
        catsim_dir: str | None = None,
    ) -> "BaseGalaxyCatalog":
        """
        Build a catalog directly from a table structured array.

        Parameters
        ----------
        table : np.ndarray
            Structured array with columns 'dx', 'dy', 'indices', 'angles'.
        catsim_dir : str or None
            Directory containing input galaxy catalogs.  Falls back to the
            ``CATSIM_DIR`` environment variable when *None*.
        select_observable, select_lower_limit, select_upper_limit
            Passed to _read_catalog(...) so subclasses can load/filter
            input_catalog.
        """
        assert table.dtype.names is not None
        # Create instance without running __init__
        self = cls.__new__(cls)
        self.catsim_dir = catsim_dir or os.environ.get("CATSIM_DIR", ".")
        self.prepare_tract_info(tract_info)
        wcs = tract_info.getWcs()
        self.pixel_scale = float(wcs.getPixelScale().asArcseconds())

        # Load catalog (subclass hook)
        self.input_catalog = self._read_catalog(
            select_observable=select_observable,
            select_lower_limit=select_lower_limit,
            select_upper_limit=select_upper_limit,
        )

        # Validate fields
        for col in ["dx", "dy", "indices", "angles"]:
            if col not in list(table.dtype.names):
                raise ValueError(
                    f"Missing required column '{col}' in table array"
                )
        input_size = len(self.input_catalog)
        if (
            (table["indices"] < 0).any() or
            (table["indices"] >= input_size).any()
        ):
            raise IndexError(
                "Indices in table array out of range for input_catalog"
            )
        self.dtype = [
            ("indices", "i8"),
            ("redshift", "f8"),
            ("angles", "f8"),
            ("gamma1", "f8"), ("gamma2", "f8"), ("kappa", "f8"),
            ("dx", "f8"), ("dy", "f8"),
            ("ra", "f8"), ("dec", "f8"),
            ("image_x", "f8"), ("image_y", "f8"),
            ("prelensed_image_x", "f8"), ("prelensed_image_y", "f8"),
            ("has_finite_shear", "bool"),
            ("hlr", "f8"),
        ]
        self.data = np.zeros(len(table), dtype=self.dtype)
        assert self.data.dtype.names is not None
        for name in table.dtype.names:
            if name in list(self.data.dtype.names):
                self.data[name] = table[name]
        self.lensed = True
        return self

    def rotate(self, theta, degrees=False):
        """Rotate by an angle theta (radians)."""
        if self.lensed:
            raise ValueError("Cannot rotate a lensed catalog")
        if degrees:
            theta = theta / np.pi * 180.0

        c, s = np.cos(theta), np.sin(theta)
        x = c * self.data["dx"] - s * self.data["dy"]
        y = s * self.data["dx"] + c * self.data["dy"]
        self.data["dx"] = x
        self.data["dy"] = y
        self.data["angles"] = self.data["angles"] + theta
        ps = self.pixel_scale
        self.data["prelensed_image_x"] = self.x_center + self.data["dx"] / ps
        self.data["prelensed_image_y"] = self.y_center + self.data["dy"] / ps
        self.data["image_x"] = self.x_center + self.data["dx"] / ps
        self.data["image_y"] = self.y_center + self.data["dy"] / ps
        wcs = self.tract_info.getWcs()
        ra, dec = wcs.pixelToSkyArray(
            x=self.data["image_x"],
            y=self.data["image_y"],
            degrees=True,
        )
        self.data["ra"] = ra
        self.data["dec"] = dec
        return

    def lens(self, *, shear_obj, apply_position_shifts: bool = True):
        if self.lensed:
            raise ValueError("Cannot lens a lensed catalog")
        ps = self.pixel_scale
        self.data["prelensed_image_x"] = self.x_center + self.data["dx"] / ps
        self.data["prelensed_image_y"] = self.y_center + self.data["dy"] / ps
        num = len(self.data)
        for _ in range(num):
            src = self.data[_]
            distort_res = shear_obj.distort_galaxy(src)
            self.data[_]["dx"] = distort_res["dx"]
            self.data[_]["dy"] = distort_res["dy"]
            self.data[_]["gamma1"] = distort_res["gamma1"]
            self.data[_]["gamma2"] = distort_res["gamma2"]
            self.data[_]["kappa"] = distort_res["kappa"]
            self.data[_]["has_finite_shear"] = distort_res["has_finite_shear"]
        if apply_position_shifts:
            self.data["image_x"] = self.x_center + self.data["dx"] / ps
            self.data["image_y"] = self.y_center + self.data["dy"] / ps
        else:
            self.data["image_x"] = self.data["prelensed_image_x"]
            self.data["image_y"] = self.data["prelensed_image_y"]

        wcs = self.tract_info.getWcs()
        ra, dec = wcs.pixelToSkyArray(
            x=self.data["image_x"],
            y=self.data["image_y"],
            degrees=True,
        )
        self.data["ra"] = ra
        self.data["dec"] = dec
        self.lensed = True
        return

    def get_obj(
        self, *,
        ind, mag_zero: float, band: str,
        use_mog: bool = False,
        force_isotropic: bool = False,
        include_point_source: bool = True,
        survey_name: str = "",
    ) -> dict[str, list]:
        """
        Returns
        -------
        """
        src = self.data[ind]
        entry = self.input_catalog[src["indices"]]
        gal = self._generate_galaxy(
            entry=entry, mag_zero=mag_zero, band=band, use_mog=use_mog,
            include_point_source=include_point_source,
            force_isotropic=force_isotropic,
            survey_name=survey_name,
        )
        gal = gal.rotate(
            src["angles"] * galsim.radians
        )
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
    """
    Catalog of galaxies from catsim2017 (OneDegSq.fits)
    """

    def _read_catalog(
        self,
        *,
        select_observable=None,
        select_lower_limit=None,
        select_upper_limit=None,
    ):
        """
        Read the catalog from the cache, but update the position angles each
        time

        Parameters
        ----------
        select_observable: list[str] | str
            A list of observables to apply selection
        select_lower_limit: list[float] | ndarray[float]
            lower limits of the slection cuts
        select_upper_limit: list[float] | ndarray[float]
            upper limits of the slection cuts

        Returns
        -------
        array with fields
        """
        fname = os.path.join(
            self.catsim_dir,
            "OneDegSq.fits",
        )

        # not thread safe
        cat = get_catalog(fname)
        if select_observable is not None:
            select_observable = np.atleast_1d(select_observable)
            if not set(select_observable) < set(cat.dtype.names):
                raise ValueError(
                    "Selection observables not in the catalog columns"
                )
            mask = np.ones(len(cat)).astype(bool)
            if select_lower_limit is not None:
                select_lower_limit = np.atleast_1d(select_lower_limit)
                assert len(select_observable) == len(select_lower_limit)
                for nn, ll in zip(select_observable, select_lower_limit):
                    mask = mask & (cat[nn] > ll)
            if select_upper_limit is not None:
                select_upper_limit = np.atleast_1d(select_upper_limit)
                assert len(select_observable) == len(select_upper_limit)
                for nn, ul in zip(select_observable, select_upper_limit):
                    mask = mask & (cat[nn] <= ul)
            cat = cat[mask]
        return cat

    def _compute_density(self, cat) -> float:
        # One square degree file; convert to arcmin^2
        return cat.size / (60.0 * 60.0)

    def _probabilities_for_sampling(self, cat):
        if "prob" in cat.dtype.names and cat.size > 0:
            p = cat["prob"].astype(float)
            p_sum = np.sum(p)
            if p_sum > 0:
                return p / p_sum
        return None

    def _half_light_radius(self, catalog) -> np.ndarray:
        return np.sqrt(
            np.maximum(catalog["a_d"], 1e-9) * np.maximum(catalog["b_d"], 1e-9)
        )

    def _generate_galaxy(
        self, *, entry, mag_zero, band, use_mog=False,
        include_point_source=True,
        force_isotropic=False,
        **kwargs,
    ) -> galsim.GSObject:
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
        total_fluxnorm = (
            dd["fluxnorm_disk"] + dd["fluxnorm_bulge"]
            + dd["fluxnorm_agn"]
        )
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
            disk = _simulator.Exponential(
                flux=disk_flux, half_light_radius=hlr_d
            ).shear(
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
            bulge = _simulator.DeVaucouleurs(
                flux=bulge_flux, half_light_radius=hlr_b
            ).shear(q=q_b, beta=beta_b * galsim.radians)
            components.append(bulge)

        # AGN (nearly point-like)
        if agn_flux > 0:
            components.append(galsim.Gaussian(flux=agn_flux, sigma=1e-4))

        if not components:
            # fallback if all fluxes zero
            return galsim.Gaussian(flux=total_flux, sigma=1e-4)

        return galsim.Add(components)


# ---------------------------------------------------------
# Concrete implementation: OpenUniverse 2024 Rubin–Roman
# ---------------------------------------------------------
class OpenUniverse2024RubinRomanCatalog(BaseGalaxyCatalog):
    """
    DiffSky-based Rubin–Roman input galaxies (nside=32 tile)
    """

    def _read_catalog(
        self,
        *,
        select_observable=None,
        select_lower_limit=None,
        select_upper_limit=None,
    ):
        """
        Read the catalog from the cache, but update the position angles each
        time

        Parameters
        ----------
        select_observable: list[str] | str
            A list of observables to apply selection
        select_lower_limit: list[float] | ndarray[float]
            lower limits of the slection cuts
        select_upper_limit: list[float] | ndarray[float]
            upper limits of the slection cuts

        Returns
        -------
        array with fields
        """
        # galaxy catalog
        fname = os.path.join(
            self.catsim_dir,
            "rubinroman_nside32_10307.parquet",
        )
        if not os.path.isfile(fname):
            raise FileNotFoundError(
                "Cannot find 'rubinroman_nside32_10307.parquet'",
                "Please download it and place it under $CATSIM_DIR",
            )

        cat = get_catalog(fname)
        if select_observable is not None:
            select_observable = np.atleast_1d(select_observable)
            if not set(select_observable) < set(cat.dtype.names):
                raise ValueError(
                    "Selection observables not in the catalog columns"
                )
            mask = np.ones(len(cat)).astype(bool)
            if select_lower_limit is not None:
                select_lower_limit = np.atleast_1d(select_lower_limit)
                assert len(select_observable) == len(select_lower_limit)
                for nn, ll in zip(select_observable, select_lower_limit):
                    mask = mask & (cat[nn] > ll)
            if select_upper_limit is not None:
                select_upper_limit = np.atleast_1d(select_upper_limit)
                assert len(select_observable) == len(select_upper_limit)
                for nn, ul in zip(select_observable, select_upper_limit):
                    mask = mask & (cat[nn] <= ul)
            cat = cat[mask]
        return cat

    def _compute_density(self, cat) -> float:
        # area of one nside=32 HEALPix pixel, in arcmin^2
        area_tot_arcmin = (
            60.0**2 * (180.0 / np.pi) ** 2 * 4.0 * np.pi / (12.0 * 32.0**2)
        )
        return len(cat) / area_tot_arcmin

    def _half_light_radius(self, catalog) -> np.ndarray:
        return catalog["diskHalfLightRadiusArcsec"]

    def _generate_galaxy(
        self, *, entry, mag_zero, band, survey_name,
        use_mog=False,
        force_isotropic=False,
        **kwargs,
    ) -> galsim.GSObject:
        """
        entry is a row of the columnar table (supports dict-like access).
        """
        if use_mog:
            _simulator = mog
        else:
            _simulator = galsim
        if survey_name == "hsc":
            sname = "lsst"
        else:
            sname = survey_name

        bulge_hlr = entry["spheroidHalfLightRadiusArcsec"]
        disk_hlr = entry["diskHalfLightRadiusArcsec"]

        # shear-ellipticity components
        if force_isotropic:
            disk_e1, disk_e2 = 0.0, 0.0
            bulge_e1, bulge_e2 = 0.0, 0.0
        else:
            disk_e1, disk_e2 = (
                entry["diskEllipticity1"], entry["diskEllipticity2"]
            )
            bulge_e1, bulge_e2 = (
                entry["spheroidEllipticity1"], entry["spheroidEllipticity2"]
            )

        mag = entry[f"{sname}_mag_{band}"]
        flux = 10 ** ((mag_zero - mag) / 2.5)
        bulge_frac = entry[f"{sname}_bulgefrac_{band}"]

        bulge = _simulator.DeVaucouleurs(
            flux=flux * bulge_frac, half_light_radius=bulge_hlr
        ).shear(g1=bulge_e1, g2=bulge_e2)
        disk = _simulator.Exponential(
            flux=flux * (1.0 - bulge_frac), half_light_radius=disk_hlr,
        ).shear(g1=disk_e1, g2=disk_e2)

        gal = (bulge + disk).withFlux(flux)
        return gal


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

    def _read_catalog(
        self,
        *,
        select_observable=None,
        select_lower_limit=None,
        select_upper_limit=None,
    ):
        fname = os.path.join(
            self.catsim_dir,
            "flagship_cosmos.fits",
        )
        if not os.path.isfile(fname):
            raise FileNotFoundError(
                "Cannot find 'flagship_cosmos.fits'",
                "Please download it and place it under $CATSIM_DIR",
            )
        cat = get_catalog(fname)
        if select_observable is not None:
            select_observable = np.atleast_1d(select_observable)
            if not set(select_observable) < set(cat.dtype.names):
                raise ValueError(
                    "Selection observables not in the catalog columns"
                )
            mask = np.ones(len(cat)).astype(bool)
            if select_lower_limit is not None:
                select_lower_limit = np.atleast_1d(select_lower_limit)
                assert len(select_observable) == len(select_lower_limit)
                for nn, ll in zip(select_observable, select_lower_limit):
                    mask = mask & (cat[nn] > ll)
            if select_upper_limit is not None:
                select_upper_limit = np.atleast_1d(select_upper_limit)
                assert len(select_observable) == len(select_upper_limit)
                for nn, ul in zip(select_observable, select_upper_limit):
                    mask = mask & (cat[nn] <= ul)
            cat = cat[mask]
        return cat

    def _compute_density(self, cat) -> float:
        """The catalog are a box on sky
        """
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
        self, *, entry, mag_zero, band, survey_name,
        use_mog=False,
        force_isotropic=False,
        **kwargs,
    ) -> galsim.GSObject:
        assert not use_mog
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
                flux=disk_flux, half_light_radius=disk_hlr,
            ).shear(q=q_d, beta=pa)
            components.append(disk)

        # Bulge
        if bulge_flux > 0:
            bulge_hlr = max(float(entry["bulge_r50"]), 1e-4)
            bulge_n = float(entry["bulge_nsersic"])
            if force_isotropic:
                q_b = 1.0
            else:
                q_b = float(entry["bulge_axis_ratio"])
                q_b = min(max(q_b, 0.00), 1.0)
            bulge = galsim.Sersic(
                n=bulge_n, flux=bulge_flux, half_light_radius=bulge_hlr,
            ).shear(q=q_b, beta=pa)
            components.append(bulge)

        if not components:
            return galsim.Gaussian(flux=flux, sigma=1e-4)

        return galsim.Add(components)
