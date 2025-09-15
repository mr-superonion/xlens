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
import galsim
import numpy as np
from lsst.pex.config import Config, Field, FieldValidationError
from lsst.pipe.base import Struct

from ..utils.random import num_rot

from lsst.pipe.base import Task
from .galaxies import CatSim2017Catalog, OpenUniverse2024RubinRomanCatalog
from .layout import Layout


def _rotate_pos(pos, theta):
    """Rotates coordinates by an angle theta

    Args:
        pos (PositionD):a galsim position
        theta (float):  rotation angle [rads]
    Returns:
        x2 (ndarray):   rotated coordiantes [x]
        y2 (ndarray):   rotated coordiantes [y]
    """
    x = pos.x
    y = pos.y
    cost = np.cos(theta)
    sint = np.sin(theta)
    x2 = cost * x - sint * y
    y2 = sint * x + cost * y
    return galsim.PositionD(x=x2, y=y2)


class PrepareGalaxyConfig(Config):
    galaxy_type = Field[str](
        doc="galaxy type",
        default="catsim2017",
    )
    layout = Field[str](
        doc="Layout of the galaxy distribution (random, grid, hex)",
        default="random",
    )
    sep_arcsec = Field[float](
        doc="separation distance (arcsec) between galaxies (for grid and hex)",
        default=11.0,
    )
    pad_arcsec = Field[float](
        doc="padding distance (arcsec)",
        default=20.0,
    )
    order_truth_catalog = Field[bool](
        doc="Whether to keep the order in truth catalog",
        default=False,
    )
    include_stars = Field[bool](
        doc="whether to include stars in the simulation",
        default=False,
    )

    def validate(self):
        super().validate()
        if self.layout not in ["grid", "hex", "random"]:
            raise FieldValidationError(
                self.__class__.layout,
                self,
                "We require layout in ['grid', 'hex', 'random']",
            )
        if self.sep_arcsec <= 0:
            raise FieldValidationError(
                self.__class__.sep_arcsec,
                self,
                "We require sep_arcsec > 0.0 arcsec",
            )
        if self.galaxy_type not in ["catsim2017", "RomanRubin2024"]:
            raise FieldValidationError(
                self.__class__.galaxy_type,
                self,
                "We require galaxy_type in ['catsim2017', 'RomanRubin2024']",
            )

    def setDefaults(self):
        super().setDefaults()


class PrepareGalaxyTask(Task):
    _DefaultName = "PrepareGalaxyTask"
    ConfigClass = PrepareGalaxyConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, PrepareGalaxyConfig)
        pass

    def run(
        self,
        *,
        rng: np.random.RandomState,
        wcs: lsst.afw.geom.SkyWcs,
        boundary_box: lsst.geom.Box2I,
        patch: int = 0,
        **kwargs,
    ):

        rotate_list = [np.pi / num_rot * i for i in range(num_rot)]
        assert isinstance(self.config, PrepareGalaxyConfig)
        if self.config.order_truth_catalog and self.config.layout in [
            "grid",
            "hex",
        ]:
            indice_id = patch
        else:
            indice_id = None

        if self.config.galaxy_type == "catsim2017":
            GalClass = CatSim2017Catalog
        elif self.config.galaxy_type == "RomanRubin2024":
            GalClass = OpenUniverse2024RubinRomanCatalog
        else:
            raise ValueError("invalid galaxy_type")
        layout = Layout(
            layout_name=self.config.layout,
            wcs=wcs,
            boundary_box=boundary_box,
            pad_arcsec=self.config.pad_arcsec,
            sep_arcsec=self.config.sep_arcsec,
        )
        theta0 = rotate_list[self.config.rotId]
        galaxy_catalog = GalClass(
            rng=rng,
            layout=layout,
            indice_id=indice_id,
        )
        array = galaxy_catalog.to_array()
        return Struct(
            outputCatalog=array
        )
