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
    "FpfsForcePipeConfig",
    "FpfsForcePipe",
    "FpfsForcePipeConnections",
]

import logging
from typing import Any

import lsst.pipe.base.connectionTypes as cT
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.meas.deblender import SourceDeblendTask
from lsst.pex.config import ConfigurableField, Field
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.utils.logging import LsstLogAdapter
from numpy.lib import recfunctions as rfn

from ..processor.fpfs import FpfsMeasurementTask


class FpfsForcePipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={
        "coaddName": "deep",
    },
):
    joint_catalog = cT.Input(
        doc="Source catalog with joint detection and measurement",
        name="{coaddName}Coadd_anacal_joint",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=False,
    )
    exposure = cT.Input(
        doc="Input coadd image",
        name="{coaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        deferLoad=True,
    )
    noise_corr = cT.Input(
        doc="noise correlation function",
        name="{coaddName}Coadd_systematics_noisecorr",
        storageClass="ImageF",
        dimensions=("skymap", "tract", "patch", "band"),
        minimum=0,
        multiple=True,
        deferLoad=True,
    )
    catalog = cT.Output(
        doc="Source catalog with all the measurement generated in this task",
        name="{coaddName}Coadd_anacal_force",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class FpfsForcePipeConfig(
    PipelineTaskConfig,
    pipelineConnections=FpfsForcePipeConnections,
):
    deblend = ConfigurableField(
        target=SourceDeblendTask,
        doc="Deblending Task",
    )
    fpfs = ConfigurableField(
        target=FpfsMeasurementTask,
        doc="Fpfs Source Measurement Task",
    )
    psf_cache = Field[int](
        doc="Size of PSF cache",
        default=100,
    )
    id_generator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()

    def setDefaults(self):
        super().setDefaults()
        self.fpfs.do_compute_detect_weight = False


class FpfsForcePipe(PipelineTask):
    _DefaultName = "FpfsForcePipe"
    ConfigClass = FpfsForcePipeConfig

    def __init__(
        self,
        *,
        config: FpfsForcePipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, FpfsForcePipeConfig)
        self.makeSubtask("fpfs")
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, FpfsForcePipeConfig)
        inputs = butlerQC.get(inputRefs)
        exposure_handles = inputs["exposure"]
        exposure_handles_dict = {
            handle.dataId["band"]: handle for handle in exposure_handles
        }
        correlation_handles = inputs["noise_corr"]
        if len(correlation_handles) == 0:
            correlation_handles_dict = None
        else:
            correlation_handles_dict = {
                handle.dataId["band"]: handle for handle in correlation_handles
            }

        outputs = self.run(
            joint_catalog=inputs["joint_catalog"],
            exposure_handles_dict=exposure_handles_dict,
            correlation_handles_dict=correlation_handles_dict,
        )
        butlerQC.put(outputs, outputRefs)
        return

    def run(
        self,
        *,
        joint_catalog,
        exposure_handles_dict: dict,
        correlation_handles_dict: dict | None,
    ):
        assert isinstance(self.config, FpfsForcePipeConfig)
        det_names = ["y", "x", "fpfs_w", "fpfs_dw_dg1", "fpfs_dw_dg2"]
        det = (
            joint_catalog[det_names].to_pandas(
                index=False
            ).to_records(index=False)
        )
        catalog = [det]
        for band in exposure_handles_dict.keys():
            handle = exposure_handles_dict[band]
            exposure = handle.get()
            exposure.getPsf().setCacheCapacity(self.config.psf_cache)
            if correlation_handles_dict is not None:
                handle = correlation_handles_dict[band]
                noise_corr = handle.get()
            else:
                noise_corr = None

            id_generator = self.config.id_generator.apply(handle.dataId)
            seed = id_generator.catalog_id
            data = self.fpfs.prepare_data(
                exposure=exposure,
                seed=seed,
                noise_corr=noise_corr,
                detection=joint_catalog,
                band=band,
            )
            cat = self.fpfs.run(**data)
            catalog.append(cat)
            del exposure
        catalog = rfn.merge_arrays(catalog, flatten=True)
        return Struct(catalog=catalog)
