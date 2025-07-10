import os

import descwl
import galsim
import numpy as np
from descwl_shear_sims.cache_tools import cached_catalog_read
from descwl_shear_sims.constants import SCALE
from descwl_shear_sims.layout import Layout


class CatSim2017Catalog(object):
    """
    Catalog of galaxies from catsim2017

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    layout: str|Layout, optional
    coadd_dim: int, optional
        Dimensions of the coadd
    buff: int, optional
        Buffer region with no objects, on all sides of image.  Ingored
        for layout 'grid'.  Default 0.
    pixel_scale: float, optional
        pixel scale
    select_observable: list[str] | str
        A list of observables to apply selection
    select_lower_limit: list | ndarray
        lower limits of the slection cuts
    select_upper_limit: list | ndarray
        upper limits of the slection cuts
    sep: float
        Separation of galaxies in arcsec
    indice_id: None | int
        galaxy index to use, use galaxies in the range between indice_id * num
        and (indice_id + 1) * num
    simple_coadd_bbox: optional, bool. Default: False
        Whether to force the center of coadd boundary box (which is the default
        center single exposure) at the world_origin
    """

    def __init__(
        self,
        *,
        rng,
        layout: Layout | str = "random",
        coadd_dim=None,
        buff=None,
        pixel_scale=SCALE,
        select_observable=None,
        select_lower_limit=None,
        select_upper_limit=None,
        sep=None,
        indice_id=None,
        simple_coadd_bbox=False,
    ):
        self.gal_type = "wldeblend"
        self.rng = rng

        self.input_catalog = read_catsim2017_cat(
            select_observable=select_observable,
            select_lower_limit=select_lower_limit,
            select_upper_limit=select_upper_limit,
        )
        ngal = len(self.input_catalog)
        if "prob" in self.input_catalog.dtype.names and ngal > 0:
            # noramlize probabilities to sum = 1
            probabilities = self.input_catalog["prob"] / np.sum(
                self.input_catalog["prob"]
            )
        else:
            probabilities = None

        # one square degree catalog, convert to arcmin
        density = self.input_catalog.size / (60 * 60)
        if buff is None:
            buff = 0
        if isinstance(layout, str):
            self.layout = Layout(
                layout,
                coadd_dim,
                buff,
                pixel_scale,
                simple_coadd_bbox=simple_coadd_bbox,
            )
        else:
            assert isinstance(layout, Layout)
            self.layout = layout
        self.shifts_array = self.layout.get_shifts(
            rng=rng,
            density=density,
            sep=sep,
        )

        # randomly sample from the catalog
        num = len(self)

        if indice_id is None:
            integers = np.arange(0, self.input_catalog.size, dtype=int)
            self.indices = self.rng.choice(integers, size=num, p=probabilities)
        else:
            indice_min = indice_id * num
            indice_max = indice_min + num
            if indice_min >= self.input_catalog.size:
                raise ValueError("indice_min too large")
            self.indices = (
                np.arange(
                    indice_min,
                    indice_max,
                    dtype=int,
                )
                % self.input_catalog.size
            )
        # do a random rotation for each galaxy
        self.angles = self.rng.uniform(low=0, high=360, size=num)

    def __len__(self):
        return len(self.shifts_array)

    def get_objlist(self, *, survey):
        """
        get a list of galsim objects, position shifts, redshifts and indexes

        Parameters
        ----------
        survey: WLDeblendSurvey
            The survey object

        Returns
        -------
        [galsim objects], [shifts], [redshifts], [indexes]
        """

        builder = descwl.model.GalaxyBuilder(
            survey=survey.descwl_survey,
            no_disk=False,
            no_bulge=False,
            no_agn=False,
            verbose_model=False,
        )

        band = survey.filter_band

        sarray = self.shifts_array
        indexes = []
        objlist = []
        shifts = []
        redshifts = []
        for i in range(len(self)):
            objlist.append(self._get_galaxy(builder, band, i))
            shifts.append(galsim.PositionD(sarray["dx"][i], sarray["dy"][i]))
            index = self.indices[i]
            indexes.append(index)
            redshifts.append(self.input_catalog[index]["redshift"])

        return {
            "objlist": objlist,
            "shifts": shifts,
            "redshifts": redshifts,
            "indexes": indexes,
        }

    def _get_galaxy(self, builder, band, i):
        """
        Get a galaxy

        Parameters
        ----------
        builder: descwl.model.GalaxyBuilder
            Builder for this object
        band: string
            Band string, e.g. 'r'
        i: int
            Index of object

        Returns
        -------
        galsim.GSObject
        """
        index = self.indices[i]

        angle = self.angles[i]

        galaxy = builder.from_catalog(
            self.input_catalog[index],
            0,
            0,
            band,
        ).model.rotate(
            angle * galsim.degrees,
        )

        return galaxy


def read_catsim2017_cat(
    select_observable=None,
    select_lower_limit=None,
    select_upper_limit=None,
):
    """
    Read the catalog from the cache, but update the position angles each time

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
            raise ValueError("Selection observables not in the catalog columns")
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
