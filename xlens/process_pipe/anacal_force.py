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
    "AnacalForcePipeConfig",
    "AnacalForcePipe",
    "AnacalForcePipeConnections",
]

import logging
from typing import Any

import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.afw.image import ExposureF
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import (
    ConfigurableField,
    Field,
    FieldValidationError,
)
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

from ..processor.anacal import AnacalTask
from ..processor.fpfs import FpfsMeasurementTask
import anacal


class AnacalForcePipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={
        "coaddName": "deep",
    },
):
    skyMap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    input_catalog = cT.Input(
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
        name="deepCoadd_systematics_noisecorr",
        storageClass="ImageF",
        dimensions=("skymap", "tract"),
        minimum=0,
        multiple=True,
        deferLoad=True,
    )
    catalog = cT.Output(
        doc="Source catalog with joint detection and measurement",
        name="{coaddName}Coadd_anacal_force",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class AnacalForcePipeConfig(
    PipelineTaskConfig,
    pipelineConnections=AnacalForcePipeConnections,
):
    anacal = ConfigurableField(
        target=AnacalTask,
        doc="AnaCal Task Force",
    )
    fpfs = ConfigurableField(
        target=FpfsMeasurementTask,
        doc="Fpfs Source Measurement Task",
    )
    do_fpfs = Field[bool](
        doc="Whether to drun fpfs task",
        default=False,
    )
    psfCache = Field[int](
        doc="Size of PSF cache",
        default=100,
    )
    size = Field[float](
        doc="Size of Gaussian for measurement [arcsec]",
        default=-1,
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()
        if self.do_fpfs:
            if self.fpfs.sigma_shapelets1 < 0.0:
                raise FieldValidationError(
                    self.fpfs.fields["sigma_shapelets1"],
                    self,
                    "sigma_shapelets1 in a wrong range",
                )

    def setDefaults(self):
        super().setDefaults()
        self.anacal.force_size = True
        self.anacal.force_center = True
        self.fpfs.do_compute_detect_weight = False


class AnacalForcePipe(PipelineTask):
    _DefaultName = "AnacalForcePipe"
    ConfigClass = AnacalForcePipeConfig

    def __init__(
        self,
        *,
        config: AnacalForcePipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, AnacalForcePipeConfig)
        self.makeSubtask("anacal")
        if self.config.do_fpfs:
            self.makeSubtask("fpfs")
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, AnacalForcePipeConfig)
        inputs = butlerQC.get(inputRefs)
        tract = int(butlerQC.quantum.dataId["tract"])
        patch = int(butlerQC.quantum.dataId["patch"])
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
        skyMap = inputs["skyMap"]
        detection = inputs["input_catalog"].as_array()
        if self.config.size > 0:
            detection["a1"] = self.config.size
            detection["da1_dg1"] = 0.0
            detection["da1_dg2"] = 0.0
            detection["a2"] = self.config.size
            detection["da2_dg1"] = 0.0
            detection["da2_dg2"] = 0.0
        outputs = self.run(
            detection=detection,
            exposure_handles_dict=exposure_handles_dict,
            correlation_handles_dict=correlation_handles_dict,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
        )
        butlerQC.put(outputs, outputRefs)
        return

    def run_one_band(
        self,
        *,
        exposure: ExposureF,
        detection: NDArray,
        band: str,
        seed: int,
        noise_corr: NDArray | None = None,
        skyMap=None,
        tract: int = 0,
        patch: int = 0,
        mask_array: NDArray | None = None,
        **kwargs,
    ) -> np.ndarray:
        assert isinstance(self.config, AnacalForcePipeConfig)
        data = self.anacal.prepare_data(
            exposure=exposure,
            seed=seed,
            noise_corr=noise_corr,
            detection=detection,
            band=band,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            mask_array=mask_array,
        )
        colnames = [
            "flux_gauss0",
            "dflux_gauss0_dg1",
            "dflux_gauss0_dg2",
            "flux_gauss2",
            "dflux_gauss2_dg1",
            "dflux_gauss2_dg2",
            "flux_gauss4",
            "dflux_gauss4_dg1",
            "dflux_gauss4_dg2",
        ]
        cat = rfn.repack_fields(
            self.anacal.run(**data)[colnames]
        )
        out = [cat]
        if self.config.do_fpfs:
            out.append(
                self.fpfs.run(**data)
            )
        return rfn.merge_arrays(out, flatten=True)

    def run(
        self,
        *,
        detection,
        exposure_handles_dict: dict,
        correlation_handles_dict: dict | None,
        skyMap,
        tract: int,
        patch: int,
        seed_offset: int = 0,
        mask_array: NDArray | None = None,
        **kwargs,
    ):
        assert isinstance(self.config, AnacalForcePipeConfig)
        catalog = []
        for band in exposure_handles_dict.keys():
            handle = exposure_handles_dict[band]
            exposure = handle.get()
            exposure.getPsf().setCacheCapacity(self.config.psfCache)
            if correlation_handles_dict is not None:
                noise_corr = correlation_handles_dict[band].get().getArray()
                variance = np.amax(noise_corr)
                noise_corr = noise_corr / variance
                ny, nx = noise_corr.shape
                assert noise_corr[ny // 2, nx // 2] == 1
                self.log.debug("With correlation, variance:", variance)
            else:
                noise_corr = None
            idGenerator = self.config.idGenerator.apply(handle.dataId)
            seed = idGenerator.catalog_id + seed_offset
            res = self.run_one_band(
                exposure=exposure,
                detection=detection,
                band=band,
                seed=seed,
                noise_corr=noise_corr,
                skyMap=skyMap,
                tract=tract,
                patch=patch,
                mask_array=mask_array,
            )
            colnames = res.dtype.names
            map_dict = {name: f"{band}_" + name for name in colnames}
            res = rfn.rename_fields(res, map_dict)
            catalog.append(res)
        catalog = rfn.merge_arrays(catalog, flatten=True)
        return Struct(catalog=catalog)
