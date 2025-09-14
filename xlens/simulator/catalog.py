#!/usr/bin/env python
#
# simple example with ring test (rotating intrinsic galaxies)
# Copyright 20230916 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
from typing import Any

import lsst
import numpy as np
from lsst.pex.config import Config, Field, FieldValidationError
from lsst.pipe.base import Struct

from ..utils.random import gal_seed_base
from lsst.pipe.base import Task
from .galaxies.catsim import CatSim2017Catalog
from .galaxies.skyCatalog import OpenUniverse2024RubinRomanCatalog


class PrepareGalaxyConfig(Config):
    survey_name = Field[str](
        doc="Name of the survey",
        default="LSST",
    )
    layout = Field[str](
        doc="Layout of the galaxy distribution (random, grid, hex)",
        default="random",
    )
    galaxy_type = Field[str](
        doc="galaxy type",
        default="catsim2017",
    )
    extend_ratio = Field[float](
        doc="The ratio to extend for the size of simulated image",
        default=1.06,
    )
    sep = Field[float](
        doc="separation distance (arcsec) between galaxies (for grid and hex)",
        default=11.0,
    )
    order_truth_catalog = Field[bool](
        doc="Whether to keep the order in truth catalog",
        default=False,
    )
    include_stars = Field[bool](
        doc="whether to include stars in the simulation",
        default=False,
    )
    galId = Field[int](
        doc="random seed index for galaxy, 0 <= galId < 10",
        default=0,
    )
    use_real_psf = Field[bool](
        doc="whether to use real PSF",
        default=False,
    )
    force_pixel_center = Field[bool](
        doc="whether to force galaxies at pixel center",
        default=False,
    )

    def validate(self):
        super().validate()
        if self.galId >= gal_seed_base or self.galId < 0:
            raise FieldValidationError(
                self.__class__.galId,
                self,
                "We require 0 <= galId < %d" % (gal_seed_base),
            )
        if self.layout not in ["grid", "hex", "random"]:
            raise FieldValidationError(
                self.__class__.layout,
                self,
                "We require layout in ['grid', 'hex', 'random']",
            )
        if self.sep <= 0:
            raise FieldValidationError(
                self.__class__.sep,
                self,
                "We require sep > 0.0 arcsec",
            )
        if self.galaxy_type not in ["catsim2017", "RomanRubin2024"]:
            raise FieldValidationError(
                self.__class__.galaxy_type,
                self,
                "We require galaxy_type in ['catsim2017', 'RomanRubin2024']",
            )

    def setDefaults(self):
        super().setDefaults()
        self.survey_name = self.survey_name.lower()


class PrepareGalaxyTask(Task):
    _DefaultName = "PrepareGalaxyTask"
    ConfigClass = PrepareGalaxyConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, PrepareGalaxyConfig)
        pass

    def prepare_galaxy_catalog(
        self,
        *,
        rng: np.random.RandomState,
        dim: int,
        pixel_scale: float,
        sep: float = 11.0,
        indice_id=None,
        **kwargs,
    ):
        assert isinstance(self.config, PrepareGalaxyConfig)
        # prepare galaxy catalog
        coadd_dim = dim - 10
        # galaxy catalog;
        if self.config.galaxy_type == "catsim2017":
            GalClass = CatSim2017Catalog
        elif self.config.galaxy_type == "RomanRubin2024":
            GalClass = OpenUniverse2024RubinRomanCatalog
        else:
            raise ValueError("invalid galaxy_type")
        galaxy_catalog = GalClass(
            rng=rng,
            coadd_dim=coadd_dim,
            buff=0.0,
            pixel_scale=pixel_scale,
            layout=self.config.layout,
            simple_coadd_bbox=True,
            sep=sep,
            indice_id=indice_id,
        )
        if self.config.force_pixel_center:
            galaxy_catalog.shifts_array["dx"] = (
                np.floor(galaxy_catalog.shifts_array["dx"] / pixel_scale)
                * pixel_scale
                + pixel_scale / 2.0
            )
            galaxy_catalog.shifts_array["dy"] = (
                np.floor(galaxy_catalog.shifts_array["dy"] / pixel_scale)
                * pixel_scale
                + pixel_scale / 2.0
            )
        return galaxy_catalog


    def run(
        self,
        *,
        band: str,
        seed: int,
        boundaryBox,
        wcs: lsst.afw.geom.SkyWcs,
        patch: int = 0,
        **kwargs,
    ):
        assert isinstance(self.config, PrepareGalaxyConfig)
        if self.config.order_truth_catalog and self.config.layout in [
            "grid",
            "hex",
        ]:
            indice_id = patch
        else:
            indice_id = None

        # Prepare the random number generator and basic parameters
        galaxy_seed = seed * gal_seed_base + self.config.galId
        rng = np.random.RandomState(galaxy_seed)

        # Get the pixel scale in arcseconds per pixel
        pixel_scale = wcs.getPixelScale().asArcseconds()
        width = boundaryBox.getWidth()
        height = boundaryBox.getHeight()

        dim = int(max(width, height) * self.config.extend_ratio)
        galaxy_catalog = self.prepare_galaxy_catalog(
            rng=rng,
            dim=dim,
            pixel_scale=pixel_scale,
            sep=self.config.sep,
            indice_id=indice_id,
        )
        return


