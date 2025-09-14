import functools
import os
from abc import ABC, abstractmethod
from typing import Any, Iterable

import galsim
import numpy as np
from astropy.table import Table
from descwl_shear_sims.layout import Layout


@functools.lru_cache(maxsize=8)
def cached_catalog_read(fname):
    return Table.read(fname).as_array()


class BaseGalaxyCatalog(ABC):
    """
    Abstract base class for galaxy catalogs used to build GalSim objects.

    Subclasses must implement:
      - _read_catalog(...)
      - _compute_density(cat)
      - _generate_galaxy(survey, entry)
    Optionally override:
      - _probabilities_for_sampling(cat) -> Optional[np.ndarray]
      - _get_redshift(index, entry) -> float
    """

    def __init__(
        self,
        *,
        rng: np.random.RandomState,
        layout: Layout | str = "random",
        coadd_dim: int | None = None,
        buff: int | None = None,
        pixel_scale: float = 0.2,
        select_observable: list[str] | str | None = None,
        select_lower_limit: Iterable[float] | None = None,
        select_upper_limit: Iterable[float] | None = None,
        sep: float | None = None,
        indice_id: int | None = None,
        simple_coadd_bbox: bool = False,
    ):
        self.gal_type = "wldeblend"
        self.rng = rng

        # subclass: read & (optionally) filter the catalog
        self.input_catalog = self._read_catalog(
            select_observable=select_observable,
            select_lower_limit=select_lower_limit,
            select_upper_limit=select_upper_limit,
        )

        # density drives how many objects the layout will place
        density = self._compute_density(self.input_catalog)

        # layout construction
        if isinstance(layout, str):
            self.layout = Layout(
                layout_name=layout,
                coadd_dim=coadd_dim,
                buff=0 if buff is None else buff,
                pixel_scale=pixel_scale,
                simple_coadd_bbox=simple_coadd_bbox,
            )
        else:
            assert isinstance(layout, Layout)
            self.layout = layout

        # positions to place galaxies
        self.shifts_array = self.layout.get_shifts(
            rng=rng, density=density, sep=sep
        )

        # choose which catalog rows populate those positions
        num = len(self)
        probs = self._probabilities_for_sampling(self.input_catalog)
        if indice_id is None:
            self.indices = self._draw_indices(
                num, len(self.input_catalog), probs,
            )
        else:
            self.indices = self._block_indices(
                num, len(self.input_catalog), indice_id,
            )

        # random orientation for each placed galaxy
        self.angles = self.rng.uniform(low=0.0, high=360.0, size=num)

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
    def _generate_galaxy(self, *, survey: Any, entry: Any) -> galsim.GSObject:
        """Build and return a GalSim GSObject from one catalog entry."""


    def _probabilities_for_sampling(self, cat: Any) -> np.ndarray | None:
        """Optional per-row sampling probabilities. Default: None (uniform)."""
        return None

    def _get_redshift(self, index: int, entry: Any) -> float:
        """Default redshift accessor tries two common patterns."""
        # try both column-access styles
        try:
            return entry["redshift"]
        except Exception:
            return self.input_catalog["redshift"][index]

    def __len__(self) -> int:
        return len(self.shifts_array)

    def _draw_indices(
        self, num: int, catalog_size: int, probs: np.ndarray | None
    ) -> np.ndarray:
        integers = np.arange(0, catalog_size, dtype=int)
        return self.rng.choice(integers, size=num, p=probs)

    def _block_indices(
        self, num: int, catalog_size: int, indice_id: int
    ) -> np.ndarray:
        indice_min = indice_id * num
        indice_max = indice_min + num
        if indice_min >= catalog_size:
            raise ValueError("indice_min too large")
        return (np.arange(indice_min, indice_max, dtype=int) % catalog_size)

    def get_objlist(self, *, survey: Any) -> dict[str, list]:
        """
        Returns
        -------
        {
          "objlist": [GSObject, ...],
          "shifts":  [galsim.PositionD, ...],
          "redshifts": [float, ...],
          "indexes": [int, ...],
        }
        """
        sarray = self.shifts_array
        objlist, shifts, redshifts, indexes = [], [], [], []

        for i in range(len(self)):
            idx = self.indices[i]
            entry = self.input_catalog[idx]
            gal = self._generate_galaxy(survey=survey, entry=entry).rotate(
                self.angles[i] * galsim.degrees
            )

            objlist.append(gal)
            shifts.append(galsim.PositionD(sarray["dx"][i], sarray["dy"][i]))
            redshifts.append(self._get_redshift(idx, entry))
            indexes.append(idx)

        return {
            "objlist": objlist,
            "shifts": shifts,
            "redshifts": redshifts,
            "indexes": indexes,
        }


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

    def _generate_galaxy(self, *, survey, entry) -> galsim.GSObject:
        band = survey.filter_band
        ab_magnitude = entry[band + "_ab"]
        total_flux = survey.get_flux(ab_magnitude)

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
            disk = galsim.Exponential(flux=disk_flux, half_light_radius=hlr_d).shear(
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

    def _generate_galaxy(self, *, survey, entry) -> galsim.GSObject:
        """
        entry is a row of the columnar table (supports dict-like access).
        """
        band = survey.filter_band
        sname = survey.descwl_survey.survey_name.lower()
        if sname == "hsc":
            sname = "lsst"

        bulge_hlr = entry["spheroidHalfLightRadiusArcsec"]
        disk_hlr = entry["diskHalfLightRadiusArcsec"]

        # shear-ellipticity components
        disk_e1, disk_e2 = entry["diskEllipticity1"], entry["diskEllipticity2"]
        bulge_e1, bulge_e2 = (
            entry["spheroidEllipticity1"], entry["spheroidEllipticity2"]
        )

        mag = entry[f"{sname}_mag_{band}"]
        flux = survey.get_flux(mag)
        bulge_frac = entry[f"{sname}_bulgefrac_{band}"]

        bulge = galsim.Sersic(
            4, half_light_radius=bulge_hlr, flux=flux * bulge_frac
        ).shear(g1=bulge_e1, g2=bulge_e2)
        disk = galsim.Sersic(
            1, half_light_radius=disk_hlr, flux=flux * (1.0 - bulge_frac)
        ).shear(g1=disk_e1, g2=disk_e2)

        gal = (bulge + disk).withFlux(flux)
        return gal
