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
    "SelBiasRfMultibandPipeConnections",
    "SelBiasRfMultibandPipeConfig",
    "SelBiasRfMultibandPipe",
    "SelBiasRfSummaryMultibandPipeConnections",
    "SelBiasRfSummaryMultibandPipeConfig",
    "SelBiasRfSummaryMultibandPipe",
]

import pickle
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


class SelBiasRfMultibandPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={
        "coaddName": "deep",
        "dataType": "",
    },
):
    src00 = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_0_rot0_Coadd_anacal_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    src01 = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_0_rot1_Coadd_anacal_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    src10 = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_1_rot0_Coadd_anacal_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    src11 = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}_1_rot1_Coadd_anacal_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    summary = cT.Output(
        doc="Summary statistics",
        name="{coaddName}Coadd_anacal_selbias_ranforest_{dataType}",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "patch"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class SelBiasRfMultibandPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=SelBiasRfMultibandPipeConnections,
):
    do_correct_selection_bias = Field[bool](
        doc="Whether correct for selection bias",
        default=True,
    )
    shape_name = Field[str](
        doc="ellipticity column name",
        default="e1",
    )
    shear_name = Field[str](
        doc="the shear component to test",
        default="g1",
    )
    shear_value = Field[float](
        doc="absolute value of the shear",
        default=0.02,
    )
    thresholds = ListField[float](
        doc="upper limit of score",
        default=[0.04, 0.08, 0.12, 0.16, 0.20],
    )
    mag_zero = Field[float](
        doc="calibration magnitude zero point",
        default=30.0,
    )
    model_name = Field[str](
        doc="random forest modle pickle file name",
        default="simple_sim_RF.pkl",
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


def name_add_d(ename, nchars):
    return ename[:-nchars] + "d" + ename[-nchars:]


class SelBiasRfMultibandPipe(PipelineTask):
    _DefaultName = "FpfsSelBiasRfTask"
    ConfigClass = SelBiasRfMultibandPipeConfig

    def __init__(
        self,
        *,
        config: SelBiasRfMultibandPipeConfig | None = None,
        log: LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, SelBiasRfMultibandPipeConfig)

        self.sname = self.config.shear_name
        self.svalue = self.config.shear_value
        self.ename = self.config.shape_name
        ins = "_dg" + self.ename[-1]
        self.egname = name_add_d(self.ename, 2) + ins
        self.wname = "fpfs_w"
        self.wgname = name_add_d(self.wname, 1) + ins
        with open(self.config.model_name, "rb") as f:
            self.clf = pickle.load(f)
        return

    @staticmethod
    def measure_distorted_photomoetry(*, src, dg, mag_zero):
        phot = []
        for band in "grizy":
            phot.append(
                mag_zero
                - np.log10(
                    src[f"{band}_fpfs1_m00"]
                    + dg * src[f"{band}_fpfs1_dm00_dg1"]
                )
                * 2.5
            )

        phot = np.vstack(phot).T
        return phot

    def measure_shear(self, *, src, dg, threshold):
        assert isinstance(self.config, SelBiasRfMultibandPipeConfig)
        en = self.ename
        egn = self.egname
        wname = self.wname
        wgname = self.wgname
        phot = self.measure_distorted_photomoetry(
            src=src,
            dg=dg,
            mag_zero=self.config.mag_zero,
        )
        scores = self.clf.predict_proba(phot)[:, 1]
        mask = scores < threshold
        tmp = src[mask]
        ell = np.sum(tmp[en] * tmp[wname])
        res = np.sum(tmp[egn] * tmp[wname] + tmp[en] * tmp[wgname])
        return {
            "ellipticity": ell,
            "response": res,
        }

    def measure_shear_ranforest(self, src, threshold):
        assert isinstance(self.config, SelBiasRfMultibandPipeConfig)
        summary = self.measure_shear(src=src, dg=0.00, threshold=threshold)
        ell = summary["ellipticity"]
        res = summary["response"]

        if self.config.do_correct_selection_bias:
            dg = 0.01
            ellp = self.measure_shear(
                src=src,
                dg=dg,
                threshold=threshold,
            )["ellipticity"]

            ellm = self.measure_shear(
                src=src,
                dg=-dg,
                threshold=threshold,
            )["ellipticity"]

            res_sel = (ellp - ellm) / 2.0 / dg
        else:
            res_sel = 0.0
        return ell, (res + res_sel)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, SelBiasRfMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)
        return

    def run(self, src00, src01, src10, src11):
        assert isinstance(self.config, SelBiasRfMultibandPipeConfig)
        ncuts = len(self.config.thresholds)
        em = np.zeros(ncuts)
        ep = np.zeros(ncuts)
        rm = np.zeros(ncuts)
        rp = np.zeros(ncuts)

        for ic, threshold in enumerate(self.config.thresholds):
            ell00, res00 = self.measure_shear_ranforest(src00, threshold)
            ell10, res10 = self.measure_shear_ranforest(src10, threshold)
            ell01, res01 = self.measure_shear_ranforest(src01, threshold)
            ell11, res11 = self.measure_shear_ranforest(src11, threshold)

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


class SelBiasRfSummaryMultibandPipeConnections(
    PipelineTaskConnections,
    dimensions=(),
    defaultTemplates={
        "coaddName": "deep",
        "dataType": "",
    },
):
    summary_list = cT.Input(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}Coadd_anacal_selbias_ranforest_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class SelBiasRfSummaryMultibandPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=SelBiasRfSummaryMultibandPipeConnections,
):

    shape_name = Field[str](
        doc="ellipticity column name",
        default="e1",
    )
    shear_name = Field[str](
        doc="the shear component to test",
        default="g1",
    )
    shear_value = Field[float](
        doc="absolute value of the shear",
        default=0.02,
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


class SelBiasRfSummaryMultibandPipe(PipelineTask):
    _DefaultName = "FpfsSelBiasRfSummaryTask"
    ConfigClass = SelBiasRfSummaryMultibandPipeConfig

    def __init__(
        self,
        *,
        config: SelBiasRfSummaryMultibandPipeConfig | None = None,
        log: LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, SelBiasRfSummaryMultibandPipeConfig)
        self.ename = self.config.shape_name
        self.sname = self.config.shear_name
        self.svalue = self.config.shear_value
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, SelBiasRfSummaryMultibandPipeConfig)
        # Retrieve the filename of the input exposure
        inputs = butlerQC.get(inputRefs)
        self.run(**inputs)
        return

    def run(self, summary_list):
        assert isinstance(self.config, SelBiasRfSummaryMultibandPipeConfig)
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
        if self.sname[-1] == self.ename[-1]:
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
