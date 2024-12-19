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

__all__ = ["FpfsSimpleSimConfig", "FpfsSimpleSimTask"]

import os
from typing import Any

import fitsio
import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import numpy as np
from lsst.meas.algorithms import SourceDetectionTask
from lsst.pex.config import (
    Config,
    ConfigurableField,
    Field,
    FieldValidationError,
    ListField,
)
from lsst.pipe.base import Task
from numpy.typing import NDArray

from ..simulator.simplesim import SimpleSimShearTask
from .fpfs import FpfsMeasurementTask


class FpfsSimpleSimConfig(Config):
    do_dm_detection = Field[bool](
        doc="whether to do detection",
        default=False,
    )
    coadd_bands = ListField[str](
        doc="Image bands to coadd",
        default=["r"],
    )

    detection = ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect Sources Task",
    )
    detection = ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect Sources Task",
    )
    simulator = ConfigurableField(
        target=SimpleSimShearTask,
        doc="Image simulation task",
    )
    fpfs = ConfigurableField(
        target=FpfsMeasurementTask,
        doc="Fpfs Source Measurement Task",
    )

    def validate(self):
        super().validate()
        if not set(self.coadd_bands).issubset(self.simulator.noise_stds.keys()):
            raise FieldValidationError(
                self.__class__.coadd_bands, self, "band list is not suported"
            )
        if not os.path.isdir(self.simulator.root_dir):
            raise FileNotFoundError("Cannot find root_dir")
        if (
            self.simulator.draw_image_noise
            != self.fpfs.do_noise_bias_correction
        ):
            raise ValueError(
                "simulator.draw_image noise not equal"
                "fpfs.do_noise_bias_correction"
            )

    def setDefaults(self):
        super().setDefaults()
        # input truth directory
        src_dir = os.path.join(self.simulator.root_dir, "fpfs_src")
        if not os.path.isdir(src_dir):
            os.makedirs(src_dir, exist_ok=True)


class FpfsSimpleSimTask(Task):
    _DefaultName = "FpfsTask"
    ConfigClass = FpfsSimpleSimConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, FpfsSimpleSimConfig)
        if self.config.do_dm_detection:
            self.schema = afwTable.SourceTable.makeMinimalSchema()
            self.algMetadata = dafBase.PropertyList()
            self.makeSubtask("detection", schema=self.schema)
            self.makeSubtask("deblend", schema=self.schema)
        self.makeSubtask("fpfs")
        self.makeSubtask("simulator")
        return

    def get_fpfs_src_name(self, *, ifield: int, mode: int, rotId: int) -> str:
        assert isinstance(self.config, FpfsSimpleSimConfig)
        outdir = os.path.join(self.config.simulator.root_dir, "fpfs_src")
        return "%s/src-%05d_%s-%d_rot%d.fits" % (
            outdir,
            ifield,
            self.config.simulator.test_target,
            mode,
            rotId,
        )

    def write_fpfs_src(
        self, *, ifield: int, mode: int, rotId: int, data: NDArray
    ) -> None:
        name = self.get_fpfs_src_name(ifield=ifield, mode=mode, rotId=rotId)
        fitsio.write(name, data)

    def process_data(self, *, ifield: int, mode: int, rotId: int) -> NDArray:
        assert isinstance(self.config, FpfsSimpleSimConfig)
        data = self.prepare_data(ifield=ifield, mode=mode, rotId=rotId)
        result = self.fpfs.run(**data)
        return result

    def run(self, ifield: int) -> NDArray:
        assert isinstance(self.config, FpfsSimpleSimConfig)
        nmodes = len(self.config.simulator.mode_list)
        up = np.zeros(nmodes)
        down = np.zeros(nmodes)
        out = np.zeros(3)
        for mode in self.config.simulator.mode_list:
            for rotId in range(self.config.simulator.nrot):
                name = self.get_fpfs_src_name(
                    ifield=ifield,
                    mode=mode,
                    rotId=rotId,
                )
                if os.path.isfile(name):
                    res = fitsio.read(name)
                else:
                    res = self.process_data(
                        ifield=ifield, mode=mode, rotId=rotId
                    )
                    self.write_fpfs_src(
                        ifield=ifield,
                        mode=mode,
                        rotId=rotId,
                        data=res,
                    )
                up[mode] = up[mode] + np.sum(res["fpfs_e1"] * res["fpfs_w"])
                down[mode] = down[mode] + np.sum(
                    res["fpfs_e1"] * res["fpfs_dw_dg1"]
                    + res["fpfs_de1_dg1"] * res["fpfs_w"]
                )
        out[0] = (up[1] - up[0]) / 2.0 / self.config.simulator.test_value
        out[1] = (up[1] + up[0]) / 2.0
        out[2] = (down[1] + down[0]) / 2.0
        return out

    def prepare_data(self, ifield: int, mode: int, rotId: int):
        assert isinstance(self.config, FpfsSimpleSimConfig)
        exposure = self.simulator.get_dm_exposure(
            ifield=ifield,
            mode=mode,
            rotId=rotId,
            band_list=self.config.coadd_bands,
        )
        # Seed for additional noise layer for noise bias correction
        # This seed is different from the seed used for image simulation
        # since we set band = None
        seed = self.simulator.get_random_seed(
            ifield=ifield,
            rotId=rotId,
            band=None,
        )
        noise_corr = self.simulator.get_noise_corr()
        if self.config.do_dm_detection:
            detection = None
        else:
            detection = None
        return self.fpfs.prepare_data(
            exposure=exposure,
            seed=seed,
            noise_corr=noise_corr,
            detection=detection,
        )
