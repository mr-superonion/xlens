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
    "MeasureCoaddsPipeConfig",
    "MeasureCoaddsPipe",
    "MeasureCoaddsPipeConnections",
]

import logging
from typing import Any

import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.afw.image import MaskX
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import ConfigurableField, Field, FieldValidationError
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

band_order = "ugrizy"


class MeasureCoaddsPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={"coaddName": "deep"},
):
    skyMap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    exposure = cT.Input(
        doc="Input coadd image",
        name="{coaddName}_coadd",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        deferLoad=True,
    )
    noiseCorrArray = cT.Input(
        doc="Stacked noise correlation array (6 x npix x npix).",
        name="deep_coadd_systematics_noisecorr_6bands",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
        multiple=False,
        minimum=0,
    )
    mask = cT.Input(
        doc="Combined mask from bad pixels and bright stars across all bands.",
        name="deep_coadd_systematics_mask",
        storageClass="Mask",
        dimensions=("skymap", "tract", "patch"),
        minimum=0,
        multiple=False,
    )
    output_catalog = cT.Output(
        doc="anacal catalog",
        name="{coaddName}_coadd_anacal_catalog",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class MeasureCoaddsPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=MeasureCoaddsPipeConnections,
):
    anacal = ConfigurableField(
        target=AnacalTask,
        doc="AnaCal Task for detection stage (i-band)",
    )
    fpfs = ConfigurableField(
        target=FpfsMeasurementTask,
        doc="Fpfs Source Measurement Task",
    )
    psfCache = Field[int](
        doc="Size of PSF cache",
        default=100,
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def validate(self):
        super().validate()
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


class MeasureCoaddsPipe(PipelineTask):
    _DefaultName = "MeasureCoaddsPipe"
    ConfigClass = MeasureCoaddsPipeConfig

    def __init__(
        self,
        *,
        config: MeasureCoaddsPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log,
            initInputs=initInputs, **kwargs,
        )
        assert isinstance(self.config, MeasureCoaddsPipeConfig)

        self.makeSubtask("anacal")
        self.makeSubtask("fpfs")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, MeasureCoaddsPipeConfig)

        inputs = butlerQC.get(inputRefs)
        tract = int(butlerQC.quantum.dataId["tract"])
        patch = int(butlerQC.quantum.dataId["patch"])

        exposure_handles = inputs["exposure"]
        exposure_handles_dict = {h.dataId["band"]: h for h in exposure_handles}

        outputs = self.run(
            exposure_handles_dict=exposure_handles_dict,
            corr_array=inputs.get("noiseCorrArray", None),
            skyMap=inputs["skyMap"],
            tract=tract,
            patch=patch,
            mask=inputs.get("mask", None),
        )
        butlerQC.put(outputs, outputRefs)

    def _load_noise_corr(
        self, corr_array: np.ndarray | None, band: str
    ) -> NDArray | None:
        if corr_array is None:
            return None
        if band not in band_order:
            return None

        iband = band_order.index(band)
        noise_corr = corr_array[iband]
        variance = float(np.amax(noise_corr))
        if variance <= 0:
            return None
        noise_corr = noise_corr / variance
        ny, nx = noise_corr.shape
        if not np.isclose(noise_corr[ny // 2, nx // 2], 1.0):
            raise RuntimeError(
                "Noise correlation is not normalized to 1 at the center pixel."
            )

        self.log.debug(
            "With correlation (band=%s), variance=%s", band, variance,
        )
        return noise_corr

    def _detect(
        self,
        *,
        exposure_handles_dict: dict,
        corr_array: np.ndarray | None,
        skyMap,
        tract: int,
        patch: int,
        mask_array: NDArray | None = None,
    ) -> np.ndarray:
        assert isinstance(self.config, MeasureCoaddsPipeConfig)
        band = "i"
        if band not in exposure_handles_dict:
            raise KeyError(
                f"band '{band}' not in {exposure_handles_dict.keys()}"
            )

        handle = exposure_handles_dict[band]
        exposure = handle.get()
        exposure.getPsf().setCacheCapacity(self.config.psfCache)

        noise_corr = self._load_noise_corr(corr_array, band)
        idGenerator = self.config.idGenerator.apply(handle.dataId)
        data = self.anacal.prepare_data(
            exposure=exposure,
            band=band,
            seed=idGenerator.catalog_id,
            noise_corr=noise_corr,
            detection=None,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            mask_array=mask_array,
        )
        return self.anacal.run(**data)

    def _force(
        self,
        *,
        detection: NDArray,
        exposure_handles_dict: dict,
        corr_array: np.ndarray | None,
        skyMap,
        tract: int,
        patch: int,
        mask_array: NDArray | None = None,
    ) -> np.ndarray:
        assert isinstance(self.config, MeasureCoaddsPipeConfig)
        """Loop over bands and run forced measurement. Returns band-prefixed
        merged structured array.
        """
        per_band = []
        for band, handle in exposure_handles_dict.items():
            exposure = handle.get()
            exposure.getPsf().setCacheCapacity(self.config.psfCache)

            noise_corr = self._load_noise_corr(corr_array, band)
            idGenerator = self.config.idGenerator.apply(handle.dataId)
            data = self.anacal.prepare_data(
                exposure=exposure,
                seed=idGenerator.catalog_id,
                noise_corr=noise_corr,
                detection=detection,
                band=band,
                skyMap=skyMap,
                tract=tract,
                patch=patch,
                mask_array=mask_array,
            )

            colnames = [
                "flux_gauss0", "dflux_gauss0_dg1", "dflux_gauss0_dg2",
                "flux_gauss2", "dflux_gauss2_dg1", "dflux_gauss2_dg2",
                "flux_gauss4", "dflux_gauss4_dg1", "dflux_gauss4_dg2",
                'flux_gauss0_err', 'flux_gauss2_err', 'flux_gauss4_err',
            ]
            out = []
            out.append(
                rfn.repack_fields(
                    self.anacal.run(**data)[colnames]
                )
            )
            out.append(self.fpfs.run(**data))
            res = rfn.merge_arrays(out, flatten=True)
            # Prefix all per-band columns: g_flux_gauss0, r_flux_gauss0, ...
            map_dict = {name: f"{band}_{name}" for name in colnames}
            res = rfn.rename_fields(res, map_dict)
            per_band.append(res)

        return rfn.merge_arrays(per_band, flatten=True)

    def run(
        self,
        *,
        exposure_handles_dict: dict,
        corr_array: np.ndarray | None,
        skyMap,
        tract: int,
        patch: int,
        mask: MaskX | None = None,
        **kwargs,
    ):
        if mask is not None:
            mask_array = mask.getArray()
        else:
            mask_array = None
        det_cat = self._detect(
            exposure_handles_dict=exposure_handles_dict,
            corr_array=corr_array,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            mask_array=mask_array,
        )
        force_cat = self._force(
            detection=det_cat,
            exposure_handles_dict=exposure_handles_dict,
            corr_array=corr_array,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            mask_array=mask_array,
        )
        final = rfn.merge_arrays([det_cat, force_cat], flatten=True)
        return Struct(output_catalog=final)
