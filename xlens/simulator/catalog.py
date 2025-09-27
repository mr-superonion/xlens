#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Simple example with ring test (rotating intrinsic galaxies)
# Copyright 2023-2025 Xiangchong Li.
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

import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import Field, FieldValidationError, ListField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.tasks.coaddBase import makeSkyInfo
from lsst.skymap import BaseSkyMap

from ..utils.random import (
    gal_seed_base,
    num_rot,
)
from .galaxies import CatSim2017Catalog, OpenUniverse2024RubinRomanCatalog
from .perturbation import ShearHalo, ShearRedshift


class CatalogConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch", "band"),
    defaultTemplates={
        "coaddName": "deep",
        "simCoaddName": "sim",
        "mode": 0,
        "rotId": 0,
    },
):
    skymap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    truthCatalog = cT.Output(
        doc="Output truth catalog",
        name="{simCoaddName}_{mode}_rot{rotId}_Coadd_truthCatalog",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class CatalogConfig(
    PipelineTaskConfig,
    pipelineConnections=CatalogConnections,
):
    galaxy_type = Field[str](
        doc="galaxy type",
        default="catsim2017",
    )
    layout = Field[str](
        doc="layout type",
        default="random",
    )
    galId = Field[int](
        doc="random seed index for galaxy, 0 <= galId < 10",
        default=0,
    )
    rotId = Field[int](
        doc="number of rotations",
        default=0,
    )
    sep_arcsec = Field[float](
        doc="Spacing (arcsec) for 'grid'/'hex' layout",
        default=12.0,
    )
    extend_ratio = Field[float](
        doc="ratio of padded coverage length of galaxy catalog",
        default=1.08,
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()
        if self.galId >= gal_seed_base or self.galId < 0:
            raise FieldValidationError(
                self.__class__.galId,
                self,
                "We require 0 <= galId < %d" % (gal_seed_base),
            )
        if self.rotId >= num_rot:
            raise FieldValidationError(
                self.__class__.rotId,
                self,
                f"rotId needs to be smaller than {num_rot}",
            )
        if self.galaxy_type not in ["catsim2017", "RomanRubin2024"]:
            raise FieldValidationError(
                self.__class__.galaxy_type,
                self,
                "We require galaxy_type in ['catsim2017', 'RomanRubin2024']",
            )

    def setDefaults(self):
        super().setDefaults()


class CatalogTask(PipelineTask):
    _DefaultName = "CatalogTask"
    ConfigClass = CatalogConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, CatalogConfig)
        self.rotate_list = [np.pi / num_rot * i for i in range(num_rot)]
        pass

    def get_perturbation_object(self, **kwargs: Any) -> object:
        raise NotImplementedError(
            "'get_perturbation_object' must be implemented by subclasses."
        )

    def prepare_galaxy_catalog(
        self,
        *,
        seed,
        tract_info,
    ):
        assert isinstance(self.config, CatalogConfig)
        if self.config.galaxy_type == "catsim2017":
            GalClass = CatSim2017Catalog
        elif self.config.galaxy_type == "RomanRubin2024":
            GalClass = OpenUniverse2024RubinRomanCatalog
        else:
            raise ValueError("invalid galaxy_type")

        rng = np.random.RandomState(seed)
        galaxy_catalog = GalClass(
            rng=rng,
            tract_info=tract_info,
            layout_name=self.config.layout,
            sep_arcsec=self.config.sep_arcsec,
            extend_ratio=self.config.extend_ratio,
        )
        return galaxy_catalog


    def run(
        self,
        *,
        tract_info,
        seed: int,
        **kwargs,
    ):
        assert isinstance(self.config, CatalogConfig)
        galaxy_seed = seed * gal_seed_base + self.config.galId
        galaxy_catalog = self.prepare_galaxy_catalog(
            seed=galaxy_seed,
            tract_info=tract_info,
        )
        theta0 = self.rotate_list[self.config.rotId]
        galaxy_catalog.rotate(theta0)
        shear_obj = self.get_perturbation_object()
        galaxy_catalog.lens(shear_obj)
        return Struct(truthCatalog=galaxy_catalog.data)

    def runQuantum(self, butlerQC, inputRefs, outputRefs) -> None:
        assert butlerQC.quantum.dataId is not None
        inputs = butlerQC.get(inputRefs)

        # band name
        assert butlerQC.quantum.dataId is not None
        patch_id = butlerQC.quantum.dataId["patch"]
        inputs["patch_id"] = patch_id

        # Get unique integer ID for IdFactory and RNG seeds; only the latter
        # should really be used as the IDs all come from the input catalog.
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        seed = idGenerator.catalog_id
        inputs["seed"] = seed

        skymap = butlerQC.get(inputRefs.skymap)
        sky_info = makeSkyInfo(
            skymap,
            tractId=butlerQC.quantum.dataId["tract"],
            patchId=butlerQC.quantum.dataId["patch"],
        )
        tract_info = sky_info.tractInfo
        inputs["tract_info"] = tract_info
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)
        return


class CatalogShearTaskConfig(
    CatalogConfig,
    pipelineConnections=CatalogConnections,
):
    z_bounds = ListField[float](
        doc="boundary list of the redshift",
        default=[-0.01, 20.0],
    )
    mode = Field[int](
        doc=(
            "Ternary-encoded shear assignment per z-bin.\n"
            "Each digit in base-3 is one bin \n"
            "(lowest-z is least significant digit):\n"
            "  0 -> -test_value,  1 -> +test_value,  2 -> 0.0\n"
            "Example: z_bounds=[0.,0.5,1.0,1.5,2.0] (4 bins). \n"
            "mode=7 -> '0021' (ternary)\n"
            "=> (-g, -g, 0, +g) for bins: \n"
            " [0,0.5), [0.5,1.0), [1.0,1.5), [1.5,2.0)."
        ),
        default=0,
    )
    test_target = Field[str](
        doc="the shear component to test",
        default="g1",
    )
    test_value = Field[float](
        doc="absolute value of the shear",
        default=0.02,
    )
    kappa_value = Field[float](
        doc="kappa value to use, 0. means no kappa",
        default=0.,
    )

    def validate(self):
        super().validate()
        n_zbins = len(self.z_bounds) - 1
        mode_max = 3 ** (n_zbins)
        if self.mode >= mode_max:
            raise FieldValidationError(
                self.__class__.mode,
                self,
                "mode needs to be smaller than %d" % mode_max,
            )

        if self.test_target not in ["g1", "g2"]:
            raise FieldValidationError(
                self.__class__.test_target,
                self,
                "test target can only be 'g1' or 'g2'",
            )

        if self.test_value < 0.0 or self.test_value > 0.50:
            raise FieldValidationError(
                self.__class__.test_value,
                self,
                "test_value should be in [0.00, 0.30]",
            )

    def setDefaults(self):
        super().setDefaults()


class CatalogShearTask(CatalogTask):
    _DefaultName = "CatalogShearTask"
    ConfigClass = CatalogShearTaskConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def get_perturbation_object(self, **kwargs: Any):
        assert isinstance(self.config, CatalogShearTaskConfig)
        return ShearRedshift(
            mode=self.config.mode,
            g_dist=self.config.test_target,
            shear_value=self.config.test_value,
            z_bounds=self.config.z_bounds,
            kappa_value=self.config.kappa_value,
        )


class CatalogHaloTaskConfig(
    CatalogConfig,
    pipelineConnections=CatalogConnections,
):
    mass = Field[float](
        doc="halo mass",
        default=5e14,
    )
    conc = Field[float](
        doc="halo concertration",
        default=1.0,
    )
    z_lens = Field[float](
        doc="halo redshift",
        default=1.0,
    )
    z_source = Field[float](
        doc="Fixed redshift for all galaxies. If None, use catalog values.",
        default=None,
        optional=True,
    )
    no_kappa = Field[bool](
        doc="whether to exclude kappa field",
        default=False,
    )

    def validate(self):
        super().validate()
        if self.mass < 1e8:
            raise FieldValidationError(
                self.__class__.mass,
                self,
                "halo mass too small",
            )
        if self.z_lens < 0 or self.z_lens > 5.0:
            raise FieldValidationError(
                self.__class__.z_lens,
                self,
                "halo redshift is wrong",
            )
        if self.z_source is not None and self.z_lens > self.z_source:
            raise FieldValidationError(
                self.__class__.z_lens,
                self,
                "halo redshift is larger than source redshift",
            )

    def setDefaults(self):
        super().setDefaults()


class CatalogHaloTask(CatalogTask):
    _DefaultName = "CatalogHaloTask"
    ConfigClass = CatalogHaloTaskConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, CatalogHaloTaskConfig)

    def prepare_galaxy_catalog(
        self,
        *,
        seed,
        tract_info,
    ):
        assert isinstance(self.config, CatalogHaloTaskConfig)
        galaxy_catalog = super().prepare_galaxy_catalog(
            seed=seed,
            tract_info=tract_info,
        )
        # for fix source redshift
        if self.config.z_source is not None:
            galaxy_catalog.set_z_source(self.config.z_source)
        return galaxy_catalog

    def get_perturbation_object(self, **kwargs: Any):
        assert isinstance(self.config, CatalogHaloTaskConfig)
        return ShearHalo(
            mass=self.config.mass,
            conc=self.config.conc,
            z_lens=self.config.z_lens,
            no_kappa=self.config.no_kappa,
        )
