# This file is part of pipe_tasks.
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

__all__ = [
    "NeffMultibandPipeConnections",
    "NeffMultibandPipeConfig",
    "NeffMultibandPipe",
    "NeffSummaryMultibandPipeConnections",
    "NeffSummaryMultibandPipeConfig",
    "NeffSummaryMultibandPipe",
]

import logging
from typing import Any

from lsst.skymap import BaseSkyMap
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.pex.config import Field, FieldValidationError, ListField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.utils.logging import LsstLogAdapter


class NeffMultibandPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={
        "coaddName": "deep",
        "dataType": "",
        "version": "",
    },
):
    skyMap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    src = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_0_rot0_Coadd_anacal_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )
    summary = cT.Output(
        doc="Summary statistics",
        name="{coaddName}Coadd_anacal_neff_flux_{dataType}{version}",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "patch"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class NeffMultibandPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=NeffMultibandPipeConnections,
):
    do_correct_selection_bias = Field[bool](
        doc="Whether correct for selection bias",
        default=True,
    )
    shear_name = Field[str](
        doc="the shear component to test",
        default="g1",
    )
    shear_value = Field[float](
        doc="absolute value of the shear",
        default=0.02,
    )
    shape_name = Field[str](
        doc="ellipticity column name",
        default="fpfs_e1",
    )
    dshape_name = Field[str](
        doc="ellipticity's shear response column name",
        default="fpfs_de1",
    )
    weight_name = Field[str](
        doc="weight column name",
        default="fpfs_w",
    )
    dweight_name = Field[str](
        doc="weight's shear response column name",
        default="fpfs_dw",
    )
    flux_name = Field[str](
        doc="flux column name",
        default="fpfs_m",
    )
    dflux_name = Field[str](
        doc="flux's shear response column name",
        default="fpfs_dm",
    )
    flux_cuts = ListField[float](
        doc="lower limit of flux",
        default=[6.0, 12.0, 18.0, 24.0, 30.0],
    )
    bound = Field[int](
        doc="Sources to be removed if too close to boundary",
        default=40,
    )

    def validate(self):
        super().validate()
        if len(self.connections.dataType) == 0:
            raise ValueError("connections.dataType missing")

        if self.shear_name not in ["g1", "g2"]:
            raise FieldValidationError(
                self.__class__.shear_name,
                self,
                "shear_name can only be 'g1' or 'g2'",
            )

        if self.shear_value < 0.0 or self.shear_value > 0.10:
            raise FieldValidationError(
                self.__class__.shear_value,
                self,
                "shear_value should be in [0., 0.10]",
            )


class NeffMultibandPipe(PipelineTask):
    _DefaultName = "FpfsNeffTask"
    ConfigClass = NeffMultibandPipeConfig

    def __init__(
        self,
        *,
        config: NeffMultibandPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, NeffMultibandPipeConfig)

        self.sname = self.config.shear_name
        self.svalue = self.config.shear_value
        self.ename = self.config.shape_name
        ins = "_dg" + self.ename[-1]
        self.egname = self.config.dshape_name + ins
        self.wname = self.config.weight_name
        self.wgname = self.config.dweight_name + ins
        self.fname = self.config.flux_name
        self.fgname = self.config.dflux_name + ins
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, NeffMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        inputs = butlerQC.get(inputRefs)
        tract = butlerQC.quantum.dataId["tract"]
        patch = butlerQC.quantum.dataId["patch"]
        skyMap = inputs["skyMap"]
        patch_info = skyMap[tract][patch]
        outputs = self.run(inputs["src"], patch_info)
        butlerQC.put(outputs, outputRefs)
        return

    def measure_shear_flux_cut(self, src, flux_min):
        assert isinstance(self.config, NeffMultibandPipeConfig)
        en = self.ename
        egn = self.egname
        wname = self.wname
        wgname = self.wgname
        fname = self.fname
        fgname = self.fgname
        msk = (~np.isnan(src[fgname])) & (~np.isnan(src[egn]))
        src = src[msk]
        tmp = src[src[fname] > flux_min]
        ngal = len(msk)
        ell = np.sum(tmp[en] * tmp[wname])
        res = np.sum(tmp[egn] * tmp[wname] + tmp[en] * tmp[wgname])

        if self.config.do_correct_selection_bias:
            dg = 0.02
            # selection
            tmp = src[(src[fname] + dg * src[fgname]) > flux_min]
            ellp = np.sum(tmp[en] * tmp[wname])
            del tmp

            # selection
            tmp = src[(src[fname] - dg * src[fgname]) > flux_min]
            ellm = np.sum(tmp[en] * tmp[wname])
            res_sel = (ellp - ellm) / 2.0 / dg
            del tmp
        else:
            res_sel = 0.0
        return ell, (res + res_sel), ngal

    def run(self, src, patch_info):
        assert isinstance(self.config, NeffMultibandPipeConfig)
        bbox = patch_info.getOuterBBox()
        pixel_scale = (
            patch_info.getWcs().getPixelScale().asDegrees() * 60  # arcmin
        )
        area = (
            (bbox.getHeight() - 2.0 * self.config.bound) *
            (bbox.getWidth() - 2.0 * self.config.bound) * pixel_scale**2.0
        )  # arcmin

        ncuts = len(self.config.flux_cuts)
        data_type = [
            ("up", "f8"),
            ("down", "f8"),
            ("ngal", "f8"),
            ("area", "f8"),
        ]
        summary = np.zeros(ncuts, dtype=data_type)
        for ic, flux_min in enumerate(self.config.flux_cuts):
            ell, res, ngal = self.measure_shear_flux_cut(src, flux_min)
            summary["up"][ic] = ell
            summary["down"][ic] = res
            summary["ngal"][ic] = ngal
            summary["area"][ic] = area
        return Struct(summary=summary)


class NeffSummaryMultibandPipeConnections(
    PipelineTaskConnections,
    dimensions=(),
    defaultTemplates={
        "coaddName": "deep",
        "dataType": "",
        "version": "",
    },
):
    summary_list = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}Coadd_anacal_neff_flux_{dataType}{version}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class NeffSummaryMultibandPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=NeffSummaryMultibandPipeConnections,
):

    def validate(self):
        super().validate()
        if len(self.connections.dataType) == 0:
            raise ValueError("connections.dataType missing")


class NeffSummaryMultibandPipe(PipelineTask):
    _DefaultName = "FpfsNeffSummaryTask"
    ConfigClass = NeffSummaryMultibandPipeConfig

    def __init__(
        self,
        *,
        config: NeffSummaryMultibandPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, NeffSummaryMultibandPipeConfig)
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, NeffSummaryMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        inputs = butlerQC.get(inputRefs)
        self.run(**inputs)
        return

    def run(self, *, summary_list):
        assert isinstance(self.config, NeffSummaryMultibandPipeConfig)
        up = []
        down = []
        ngal = []
        for res in summary_list:
            res = res.get()
            up.append(np.array(res["up"]))
            down.append(np.array(res["down"]))
            ngal.append(np.array(res["ngal"]))
            area = res["area"][0]
        up = np.vstack(up)
        down = np.vstack(down)
        ngal = np.vstack(ngal)
        std = np.std(up / down, axis=0)
        neff = (0.26 / std) ** 2.0 / area
        print("neff: ", neff)
        return
