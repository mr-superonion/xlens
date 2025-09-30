import functools
import os
from abc import ABC, abstractmethod
from typing import Any, Iterable

import galsim
import lsst
import numpy as np
from astropy.table import Table

from .layout import Layout
from .wcs import make_galsim_tanwcs

SIM_INCLUSION_PADDING = 200  # pixels


@functools.lru_cache(maxsize=8)
def cached_catalog_read(fname):
    return Table.read(fname).as_array()


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
        indice_id: int | None = None,
        select_observable: list[str] | str | None = None,
        select_lower_limit: Iterable[float] | None = None,
        select_upper_limit: Iterable[float] | None = None,
        use_field_distortion: bool = False,
        extend_ratio: float = 1.08,
    ):
        self.prepare_tract_info(tract_info, use_field_distortion)
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

        # choose which catalog rows populate those positions
        num = len(shifts_array)
        probs = self._probabilities_for_sampling(self.input_catalog)
        catalog_size = len(self.input_catalog)
        if indice_id is None:
            integers = np.arange(0, catalog_size, dtype=int)
            indices = rng.choice(integers, size=num, p=probs)
        else:
            indice_min = indice_id * num
            indice_max = indice_min + num
            if indice_min >= catalog_size:
                raise ValueError("indice_min too large")
            indices = (
                np.arange(indice_min, indice_max, dtype=int) % catalog_size
            )
        # random orientation for each placed galaxy
        angles = rng.uniform(low=0.0, high=360.0, size=num)
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
        self.data["indices"] = indices
        self.data["angles"] = angles
        self.lensed = False
        self.data["prelensed_image_x"] = self.x_center + self.data["dx"] / ps
        self.data["prelensed_image_y"] = self.y_center + self.data["dy"] / ps
        self.data["image_x"] = self.x_center + self.data["dx"] / ps
        self.data["image_y"] = self.y_center + self.data["dy"] / ps
        self.data["has_finite_shear"] = np.ones(num, dtype=bool)
        self.data["redshift"] = self.input_catalog["redshift"][indices]
        self.data["hlr"] = self._build_hlr_array(indices)
        return

    def set_z_source(self, redshift):
        self.data["redshift"][:] = redshift
        return

    def prepare_tract_info(self, tract_info, use_field_distortion):
        self.tract_info = tract_info
        bbox = tract_info.getBBox()   # lsst.geom.Box2I
        center_pix = bbox.getCenter()
        self.x_center = center_pix.getX()
        self.y_center = center_pix.getY()
        self.use_field_distortion = use_field_distortion
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
        self, *, entry: Any, mag_zero: float, band: str, **kwargs
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
        use_field_distortion: bool = False,
    ) -> "BaseGalaxyCatalog":
        """
        Build a catalog directly from a table structured array.

        Parameters
        ----------
        table : np.ndarray
            Structured array with columns 'dx', 'dy', 'indices', 'angles'.
        select_observable, select_lower_limit, select_upper_limit
            Passed to _read_catalog(...) so subclasses can load/filter
            input_catalog.
        """
        # Create instance without running __init__
        self = cls.__new__(cls)
        self.prepare_tract_info(tract_info, use_field_distortion)
        wcs = tract_info.getWcs()
        self.pixel_scale = float(wcs.getPixelScale().asArcseconds())

        # Load catalog (subclass hook)
        self.input_catalog = self._read_catalog(
            select_observable=select_observable,
            select_lower_limit=select_lower_limit,
            select_upper_limit=select_upper_limit,
        )

        # Validate fields
        for col in ("dx", "dy", "indices", "angles"):
            if col not in table.dtype.names:
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
        for name in table.dtype.names:
            if name in self.data.dtype.names:
                self.data[name] = table[name]
        if "hlr" not in table.dtype.names and len(table) > 0:
            indices = self.data["indices"].astype(int)
            self.data["hlr"] = self._build_hlr_array(indices)
        self.lensed = False
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

    def lens(self, shear_obj):
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
        self.lensed = True
        return

    def get_obj(self, *, ind, mag_zero: float, band: str) -> dict[str, list]:
        """
        Returns
        -------
        """
        src = self.data[ind]
        entry = self.input_catalog[src["indices"]]
        gal = self._generate_galaxy(
            entry=entry, mag_zero=mag_zero, band=band,
        ).rotate(
            src["angles"] * galsim.radians
        )
        gamma1, gamma2, kappa = src["gamma1"], src["gamma2"], src["kappa"]
        g1 = gamma1 / (1 - kappa)
        g2 = gamma2 / (1 - kappa)
        mu = 1.0 / ((1 - kappa) ** 2 - gamma1**2 - gamma2**2)
        gal = gal.lens(g1=g1, g2=g2, mu=mu)
        return gal

    def draw(
        self, *, patch_id,
        psf_obj, mag_zero, band,
        draw_method="auto",
        nn_trunc=None,
    ):
        patch_info = self.tract_info[patch_id]
        outer_bbox = patch_info.getOuterBBox()
        xmin = outer_bbox.getMinX()
        ymin = outer_bbox.getMinY()
        xmax = outer_bbox.getMaxX()
        ymax = outer_bbox.getMaxY()
        width = outer_bbox.getWidth()
        height = outer_bbox.getHeight()
        wcs_gs = make_galsim_tanwcs(self.tract_info)
        image = galsim.ImageF(width, height, xmin=xmin, ymin=ymin, wcs=wcs_gs)
        for i, src in enumerate(self.data):
            if (
                (xmin - SIM_INCLUSION_PADDING) <
                src["image_x"] < (xmax + SIM_INCLUSION_PADDING)
            ) and (
                (ymin - SIM_INCLUSION_PADDING)
                < src["image_y"] < (ymax + SIM_INCLUSION_PADDING)
            ) and src["has_finite_shear"]:
                image_pos = galsim.PositionD(
                    x=src["image_x"], y=src["image_y"]
                )
                gal_obj = self.get_obj(
                    ind=i, mag_zero=mag_zero, band=band
                )
                convolved_object = galsim.Convolve([gal_obj, psf_obj])
                if self.use_field_distortion:
                    local_wcs = wcs_gs.local(image_pos=image_pos)
                    stamp = convolved_object.drawImage(
                        center=image_pos, wcs=local_wcs, method=draw_method,
                        nx=nn_trunc, ny=nn_trunc,
                    )
                else:
                    stamp = convolved_object.drawImage(
                        center=image_pos, wcs=None, method=draw_method,
                        scale=self.pixel_scale,
                        nx=nn_trunc, ny=nn_trunc,
                    )
                b = stamp.bounds & image.bounds
                if b.isDefined():
                    image[b] += stamp[b]
        return image.array


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
            os.environ.get("CATSIM_DIR", "."),
            "OneDegSq.fits",
        )

        # not thread safe
        cat = cached_catalog_read(fname)
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
        a_d = np.maximum(np.asarray(catalog["a_d"], dtype=float), 0.0)
        b_d = np.maximum(np.asarray(catalog["b_d"], dtype=float), 0.0)
        return np.sqrt(a_d * b_d)

    def _generate_galaxy(
        self, *, entry, mag_zero, band, **kwargs,
    ) -> galsim.GSObject:
        ab_magnitude = entry[band + "_ab"]
        total_flux = 10 ** ((mag_zero - ab_magnitude) / 2.5)

        # split flux among components
        total_fluxnorm = (
            entry["fluxnorm_disk"] + entry["fluxnorm_bulge"]
            + entry["fluxnorm_agn"]
        )
        # guard against zero to avoid NaNs
        if total_fluxnorm <= 0:
            return galsim.Gaussian(flux=total_flux, sigma=1e-8)

        disk_flux = entry["fluxnorm_disk"] / total_fluxnorm * total_flux
        bulge_flux = entry["fluxnorm_bulge"] / total_fluxnorm * total_flux
        agn_flux = entry["fluxnorm_agn"] / total_fluxnorm * total_flux

        components = []

        # Disk
        if disk_flux > 0:
            a_d, b_d = entry["a_d"], entry["b_d"]
            hlr_d = np.sqrt(max(a_d, 0.0) * max(b_d, 0.0))
            q_d = (b_d / a_d) if a_d > 0 else 1.0
            beta_d = np.radians(entry["pa_disk"])
            disk = galsim.Exponential(
                flux=disk_flux, half_light_radius=hlr_d
            ).shear(
                q=q_d, beta=beta_d * galsim.radians
            )
            components.append(disk)

        # Bulge
        if bulge_flux > 0:
            a_b, b_b = entry["a_b"], entry["b_b"]
            hlr_b = np.sqrt(max(a_b, 0.0) * max(b_b, 0.0))
            q_b = (b_b / a_b) if a_b > 0 else 1.0
            beta_b = np.radians(entry["pa_bulge"])
            bulge = galsim.DeVaucouleurs(
                flux=bulge_flux, half_light_radius=hlr_b
            ).shear(q=q_b, beta=beta_b * galsim.radians)
            components.append(bulge)

        # AGN (nearly point-like)
        if agn_flux > 0:
            components.append(galsim.Gaussian(flux=agn_flux, sigma=1e-8))

        if not components:
            # fallback if all fluxes zero
            return galsim.Gaussian(flux=total_flux, sigma=1e-8)

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
            os.environ.get("CATSIM_DIR", "."),
            "rubinroman_nside32_10307.parquet",
        )
        if not os.path.isfile(fname):
            raise FileNotFoundError(
                "Cannot find 'rubinroman_nside32_10307.parquet'",
                "Please donwload it from and place it under $CATSIM_DIR",
            )

        cat = cached_catalog_read(fname)
        if select_observable is not None:
            select_observable = np.atleast_1d(select_observable)
            if not set(select_observable) < set(cat.column_names):
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
        disk_major = np.asarray(
            catalog["diskHalfLightRadiusArcsec"], dtype=float
        )
        disk_e1 = np.asarray(catalog["diskEllipticity1"], dtype=float)
        disk_e2 = np.asarray(catalog["diskEllipticity2"], dtype=float)
        ellipticity = np.hypot(disk_e1, disk_e2)
        axis_ratio = (1.0 - ellipticity) / (1.0 + ellipticity)
        axis_ratio = np.clip(axis_ratio, 0.0, None)
        disk_minor = disk_major * axis_ratio
        return np.sqrt(np.maximum(disk_major, 0.0) * np.maximum(disk_minor, 0.0))

    def _generate_galaxy(
        self, *, entry, mag_zero, band, survey_name, **kwargs,
    ) -> galsim.GSObject:
        """
        entry is a row of the columnar table (supports dict-like access).
        """
        if survey_name == "hsc":
            sname = "lsst"
        else:
            sname = survey_name

        bulge_hlr = entry["spheroidHalfLightRadiusArcsec"]
        disk_hlr = entry["diskHalfLightRadiusArcsec"]

        # shear-ellipticity components
        disk_e1, disk_e2 = entry["diskEllipticity1"], entry["diskEllipticity2"]
        bulge_e1, bulge_e2 = (
            entry["spheroidEllipticity1"], entry["spheroidEllipticity2"]
        )

        mag = entry[f"{sname}_mag_{band}"]
        flux = 10 ** ((mag_zero - mag) / 2.5)
        bulge_frac = entry[f"{sname}_bulgefrac_{band}"]

        bulge = galsim.Sersic(
            4, half_light_radius=bulge_hlr, flux=flux * bulge_frac
        ).shear(g1=bulge_e1, g2=bulge_e2)
        disk = galsim.Sersic(
            1, half_light_radius=disk_hlr, flux=flux * (1.0 - bulge_frac)
        ).shear(g1=disk_e1, g2=disk_e2)

        gal = (bulge + disk).withFlux(flux)
        return gal
