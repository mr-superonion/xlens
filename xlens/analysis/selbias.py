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
    "SelBiasMultibandPipeConnections",
    "SelBiasMultibandPipeConfig",
    "SelBiasMultibandPipe",
    "SelBiasSummaryMultibandPipeConnections",
    "SelBiasSummaryMultibandPipeConfig",
    "SelBiasSummaryMultibandPipe",
]

import logging
from typing import Any

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


class SelBiasMultibandPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={
        "coaddName": "deep",
        "dataType": "",
        "version": "",
    },
):
    src00 = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_0_rot0_Coadd_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    src01 = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_0_rot1_Coadd_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    src10 = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_1_rot0_Coadd_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    src11 = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_1_rot1_Coadd_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    summary = cT.Output(
        doc="Summary statistics",
        name="{coaddName}Coadd_anacal_selbias_flux_{dataType}{version}",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "patch"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class SelBiasMultibandPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=SelBiasMultibandPipeConnections,
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
        default="fpfs_m00",
    )
    dflux_name = Field[str](
        doc="flux's shear response column name",
        default="fpfs_dm00",
    )
    flux_cuts = ListField[float](
        doc="lower limit of flux",
        default=[6.0, 12.0, 18.0, 24.0, 30.0],
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
                "shear_value should be in [0.00, 0.10]",
            )


class SelBiasMultibandPipe(PipelineTask):
    _DefaultName = "FpfsSelBiasTask"
    ConfigClass = SelBiasMultibandPipeConfig

    def __init__(
        self,
        *,
        config: SelBiasMultibandPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, SelBiasMultibandPipeConfig)

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
        assert isinstance(self.config, SelBiasMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)
        return

    def measure_shear_flux_cut(self, src, flux_min):
        assert isinstance(self.config, SelBiasMultibandPipeConfig)
        en = self.ename
        egn = self.egname
        if en[-1] == "1":
            en2 = en.replace("e1", "e2")
            egn2 = egn.replace("e1", "e2").replace("g1", "g2")
        else:
            en2 = en.replace("e2", "e1")
            egn2 = egn.replace("e2", "e1").replace("g2", "g1")
        if "fpfs" in en:
            emax = 0.3
        else:
            emax = 2.0
        wname = self.wname
        wgname = self.wgname
        fname = self.fname
        fgname = self.fgname
        msk = (~np.isnan(src[fgname])) & (~np.isnan(src[egn]))
        src = src[msk]

        # selection
        esq = src[en] ** 2 + src[en2] ** 2
        msk = (src[fname] > flux_min) & (esq < emax ** 2.0)
        tmp = src[msk]
        ell = np.sum(tmp[en] * tmp[wname])
        res = np.sum(tmp[egn] * tmp[wname] + tmp[en] * tmp[wgname])
        del msk, tmp, esq

        if self.config.do_correct_selection_bias:
            dg = 0.02
            # selection
            esq = (
                src[en] ** 2 + src[en2] ** 2
                + 2.0 * dg * (src[en] * src[egn] + src[en2] * src[egn2])
            )
            msk = (
                ((src[fname] + dg * src[fgname]) > flux_min) &
                (esq < emax)
            )
            tmp = src[msk]
            ellp = np.sum(tmp[en] * tmp[wname])
            del tmp, esq, msk

            # selection
            esq = (
                src[en] ** 2 + src[en2] ** 2
                - 2.0 * dg * (src[en] * src[egn] + src[en2] * src[egn2])
            )
            msk = (
                ((src[fname] - dg * src[fgname]) > flux_min) &
                (esq < emax)
            )
            tmp = src[msk]
            ellm = np.sum(tmp[en] * tmp[wname])
            res_sel = (ellp - ellm) / 2.0 / dg
            del tmp, esq, msk
        else:
            res_sel = 0.0
        return ell, (res + res_sel)

    def run(self, *, src00, src10, src01=None, src11=None, **kwargs):
        assert isinstance(self.config, SelBiasMultibandPipeConfig)
        ncuts = len(self.config.flux_cuts)
        em = np.zeros(ncuts)
        ep = np.zeros(ncuts)
        rm = np.zeros(ncuts)
        rp = np.zeros(ncuts)

        for ic, flux_min in enumerate(self.config.flux_cuts):
            ell00, res00 = self.measure_shear_flux_cut(src00, flux_min)
            ell10, res10 = self.measure_shear_flux_cut(src10, flux_min)
            if (src01 is None) or (src11 is None):
                ell01, res01, ell11, res11 = (0.0, 0.0, 0.0, 0.0)
            else:
                ell01, res01 = self.measure_shear_flux_cut(src01, flux_min)
                ell11, res11 = self.measure_shear_flux_cut(src11, flux_min)
            em[ic] = ell00 + ell01
            ep[ic] = ell10 + ell11
            rm[ic] = res00 + res01
            rp[ic] = res10 + res11

        data_type = [
            ("up1", "f8"),
            ("up2", "f8"),
            ("down", "f8"),
        ]
        summary = np.zeros(ncuts, dtype=data_type)
        summary["up1"] = (ep - em) / 2.0
        summary["up2"] = (em + ep) / 2.0
        summary["down"] = (rm + rp) / 2.0
        return Struct(summary=summary)


class SelBiasSummaryMultibandPipeConnections(
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
        name="{coaddName}Coadd_anacal_selbias_flux_{dataType}{version}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class SelBiasSummaryMultibandPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=SelBiasSummaryMultibandPipeConnections,
):

    estimate_multiplicative_bias = Field[bool](
        doc="Whether estimate multiplicative bias",
        default=True,
    )
    shear_value = Field[float](
        doc="absolute value of the shear",
        default=0.02,
    )

    def validate(self):
        super().validate()
        if len(self.connections.dataType) == 0:
            raise ValueError("connections.dataType missing")

        if self.shear_value < 0.0 or self.shear_value > 0.10:
            raise FieldValidationError(
                self.__class__.shear_value,
                self,
                "shear_value should be in [0.00, 0.10]",
            )


class SelBiasSummaryMultibandPipe(PipelineTask):
    _DefaultName = "FpfsSelBiasSummaryTask"
    ConfigClass = SelBiasSummaryMultibandPipeConfig

    def __init__(
        self,
        *,
        config: SelBiasSummaryMultibandPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, SelBiasSummaryMultibandPipeConfig)
        self.svalue = self.config.shear_value
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, SelBiasSummaryMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        inputs = butlerQC.get(inputRefs)
        self.run(**inputs)
        return

    def run(self, *, summary_list, **kwargs):
        assert isinstance(self.config, SelBiasSummaryMultibandPipeConfig)
        up1 = []
        up2 = []
        down = []
        for res in summary_list:
            res = res.get()
            up1.append(np.array(res["up1"]))
            up2.append(np.array(res["up2"]))
            down.append(np.array(res["down"]))

        nsim = len(up1)
        denom = np.average(down, axis=0)
        tmp = np.vstack(up1) + np.vstack(up2)
        print(
            "Positive shear:",
            np.average(tmp, axis=0) / denom,
            "+-",
            np.std(tmp, axis=0) / denom / np.sqrt(nsim),
        )
        tmp = np.vstack(up2) - np.vstack(up1)
        print(
            "Negative shear:",
            np.average(tmp, axis=0) / denom,
            "+-",
            np.std(tmp, axis=0) / denom / np.sqrt(nsim),
        )
        if self.config.estimate_multiplicative_bias:
            print(
                "Multiplicative bias:",
                np.average(up1, axis=0) / denom / self.svalue - 1,
                "+-",
                np.std(up1, axis=0) / denom / np.sqrt(nsim) / self.svalue,
            )
        else:
            print(
                "We do not estimate multiplicative bias:",
            )
        print(
            "Additive bias:",
            np.average(up2, axis=0) / denom,
            "+-",
            np.std(up2, axis=0) / denom / np.sqrt(nsim),
        )
        return
