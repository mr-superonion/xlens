# This file is part of xlens.
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

import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.afw.image import MaskX
from lsst.daf.butler import DataCoordinate
from lsst.meas.base import SingleFrameMeasurementTask, SkyMapIdGeneratorConfig
from lsst.pex.config import (
    ConfigurableField,
    Field,
    FieldValidationError,
    ListField,
)
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.tasks.coaddBase import makeSkyInfo
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

from ..catalog.utils import add_magnitude_columns
from ..simulator.sim import MultibandSimTask
from ..utils.catalog import set_isPrimary
from ..utils.columns import (
    select_band_gauss_fluxes,
    select_detection_columns,
)
from ..utils.constants import MAG_ZERO_AB
from ..utils.handle import SimulatedExposureHandle
from ..utils.image import (
    broadcast_psf_hsm_moments,
    build_psf_hsm_context,
    default_psf_hsm_plugin_config,
    measure_psf_hsm_moments,
)
from .anacal import AnacalTask
from .fpfs import FpfsMeasurementTask

band_order = "ugrizy"


class MeasureCoaddsPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={"inputName": "deep_coadd", "outName": "deep_coadd", "catName": "cat"},
):
    skyMap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    exposure = cT.Input(
        doc="Input coadd image (one per band).",
        name="{inputName}",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        deferLoad=True,
        minimum=0,
    )
    truthCatalog = cT.Input(
        doc="Truth catalog used to drive image simulation.",
        name="{catName}_truthCatalog",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
        minimum=0,
    )
    psfArray = cT.Input(
        doc="Stacked PSF image array (6 x npix x npix).",
        name="{inputName}_systematics_psfcentered_6bands",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
        multiple=False,
    )
    noiseCorrArray = cT.Input(
        doc="Stacked noise correlation array (6 x npix x npix).",
        name="{inputName}_systematics_noisecorr_6bands",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
        multiple=False,
    )
    mask = cT.Input(
        doc="Combined mask from bad pixels and bright stars across all bands.",
        name="{inputName}_systematics_mask",
        storageClass="Mask",
        dimensions=("skymap", "tract", "patch"),
        multiple=False,
    )
    anacalCatalog = cT.Output(
        doc="anacal catalog",
        name="{outName}_anacal_catalog",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config is None:
            return

        # Drop inputs that don't apply to the chosen mode.
        if config.use_sim:
            self.inputs.discard("exposure")
        else:
            self.inputs.discard("truthCatalog")
            self.inputs.discard("psfArray")


class MeasureCoaddsPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=MeasureCoaddsPipeConnections,
):
    anacal = ConfigurableField(
        target=AnacalTask,
        doc="AnaCal Task for the detection stage (see detection_bands)",
    )
    fpfs = ConfigurableField(
        target=FpfsMeasurementTask,
        doc="Fpfs Source Measurement Task",
    )
    simulator = ConfigurableField(
        target=MultibandSimTask,
        doc="Simulation task used to generate per-band exposures.",
    )
    use_sim = Field[bool](
        doc=(
            "If True, run image simulation per band instead of reading "
            "real coadd exposures. Requires ``truthCatalog`` input."
        ),
        default=False,
    )
    do_measure_flux_gauss = Field[bool](
        doc=(
            "If True, also run AnaCal forced measurement during the "
            "force stage to extract per-band Gaussian fluxes and merge "
            "them into the output catalog."
        ),
        default=False,
    )
    sim_bands = ListField[str](
        doc="PHYSICAL bands to simulate when ``use_sim`` is True.",
        default=["u", "g", "r", "i", "z", "y"],
    )
    detection_bands = ListField[str](
        doc=(
            "PHYSICAL bands combined to form the detection image. AnaCal "
            "removes each band's own PSF first and then averages the bands "
            "with inverse-variance weights, so the bands need not be "
            "PSF-matched. Forced measurement in the second stage is "
            "unaffected and still runs band by band. A single entry "
            "reproduces the previous single-band behaviour exactly."
        ),
        default=["i"],
    )
    survey = Field[str](
        doc=(
            "Survey name for the survey-prefixed output columns "
            "``{survey}_{band}_...`` and the survey-aware noise seed."
        ),
        default="lsst",
    )
    psfCache = Field[int](
        doc="Size of PSF cache",
        default=100,
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()
    doPsfHsmMoments = Field[bool](
        doc=(
            "If True, run lsst.meas.extensions.shapeHSM.HsmPsfMomentsPlugin "
            "+ HigherOrderMomentsPSFPlugin once per (exposure, band) at "
            "the exposure bbox centre (PSF model sampled on the pixel "
            "grid via computeKernelImage — no subpixel offset) and "
            "broadcast those PSF moments to every source. Adds "
            "{band}_ext_shapeHSM_HsmPsfMoments_{xx,yy,xy,flag,...} and "
            "{band}_ext_shapeHSM_HigherOrderMomentsPSF_{pq,flag} columns "
            "to the per-source anacal catalog."
        ),
        default=False,
    )
    psfHsmMeasurement = ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc=(
            "DRP-style single-frame measurement subtask used to evaluate "
            "the per-band PSF model HSM moments at the exposure centre. "
            "Only used when doPsfHsmMoments is True."
        ),
    )

    def validate(self):
        super().validate()
        if self.fpfs.sigma_shapelets1 < 0.0:
            raise FieldValidationError(
                self.fpfs.__class__.sigma_shapelets1,
                self,
                "sigma_shapelets1 in a wrong range",
            )
        if len(self.detection_bands) == 0:
            raise FieldValidationError(
                self.__class__.detection_bands,
                self,
                "detection_bands must name at least one band.",
            )
        if len(set(self.detection_bands)) != len(self.detection_bands):
            raise FieldValidationError(
                self.__class__.detection_bands,
                self,
                f"detection_bands has duplicates: {list(self.detection_bands)}",
            )
        if self.use_sim:
            missing = [b for b in self.detection_bands if b not in self.sim_bands]
            if missing:
                raise FieldValidationError(
                    self.__class__.sim_bands,
                    self,
                    f"sim_bands must include every detection band; "
                    f"missing {missing}.",
                )

    def setDefaults(self):
        super().setDefaults()
        self.anacal.force_size = True
        self.anacal.force_center = True
        self.fpfs.do_compute_detect_weight = False
        # Shared DRP-equivalent plugin wiring; same setup as
        # MeasureCellCoaddsPipe so column names match across both
        # flavours.
        default_psf_hsm_plugin_config(self.psfHsmMeasurement)


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
            config=config,
            log=log,
            initInputs=initInputs,
            **kwargs,
        )
        assert isinstance(self.config, MeasureCoaddsPipeConfig)

        self.makeSubtask("anacal")
        self.makeSubtask("fpfs")
        if self.config.use_sim:
            self.makeSubtask("simulator")
        if self.config.doPsfHsmMoments:
            schema = afwTable.SourceTable.makeMinimalSchema()
            self.makeSubtask(
                "psfHsmMeasurement",
                schema=schema,
                algMetadata=dafBase.PropertyList(),
            )
            self._psfHsmCtx = build_psf_hsm_context(
                schema, self.config.psfHsmMeasurement,
            )

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, MeasureCoaddsPipeConfig)

        inputs = butlerQC.get(inputRefs)
        tract = int(butlerQC.quantum.dataId["tract"])
        patch = int(butlerQC.quantum.dataId["patch"])

        seed: int | None = None
        if self.config.use_sim:
            truthCatalog = inputs.get("truthCatalog", None)
            if truthCatalog is None:
                raise RuntimeError("use_sim=True requires a truthCatalog input.")
            # ``butlerQC.quantum.dataId`` may not carry dimension records,
            # but every input ref's dataId does. ``psfArray`` is a
            # required ``(skymap, tract, patch)`` input under use_sim, so
            # use it to seed the IdGenerator (matches the patch-level
            # quantum dimensions). The same seed is forwarded to ``run``
            # because ``SimulatedExposureHandle`` carries no dataId.
            seed = self.config.idGenerator.apply(inputRefs.psfArray.dataId).catalog_id
            exposure_handles_dict = self._build_simulated_handles(
                quantum_data_id=butlerQC.quantum.dataId,
                truthCatalog=truthCatalog,
                skyMap=inputs["skyMap"],
                tract=tract,
                patch=patch,
                psf_array=inputs.get("psfArray", None),
                corr_array=inputs.get("noiseCorrArray", None),
                mask=inputs.get("mask", None),
                seed=seed,
            )
        else:
            exposure_handles = inputs.get("exposure", None)
            if not exposure_handles:
                raise RuntimeError("use_sim=False requires the 'exposure' input.")
            exposure_handles_dict = {h.dataId["band"]: h for h in exposure_handles}

        outputs = self.run(
            exposure_handles_dict=exposure_handles_dict,
            corr_array=inputs.get("noiseCorrArray", None),
            skyMap=inputs["skyMap"],
            tract=tract,
            patch=patch,
            mask=inputs.get("mask", None),
            seed=seed,
        )
        butlerQC.put(outputs, outputRefs)

    def _build_simulated_handles(
        self,
        *,
        quantum_data_id: DataCoordinate,
        truthCatalog,
        skyMap,
        tract: int,
        patch: int,
        psf_array: NDArray | None,
        corr_array: NDArray | None,
        mask: MaskX | None,
        seed: int | None = None,
    ) -> dict:
        assert isinstance(self.config, MeasureCoaddsPipeConfig)
        if seed is None:
            band_data_id = DataCoordinate.standardize(
                quantum_data_id,
                band="i",
            )
            seed = self.config.idGenerator.apply(band_data_id).catalog_id

        sky_info = makeSkyInfo(skyMap, tractId=tract, patchId=patch)
        tract_info = sky_info.tractInfo

        handles: dict = {}
        for band in self.config.sim_bands:
            handles[band] = SimulatedExposureHandle(
                simulator=self.simulator,
                tract_info=tract_info,
                patch=patch,
                band=band,
                seed=seed,
                truthCatalog=truthCatalog,
                psf_array=psf_array,
                corr_array=corr_array,
                mask=mask,
            )
        return handles

    def _load_noise_corr(self, corr_array: np.ndarray | None, band: str) -> NDArray | None:
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
            raise RuntimeError("Noise correlation is not normalized to 1 at the center pixel.")

        self.log.debug(
            "With correlation (band=%s), variance=%s",
            band,
            variance,
        )
        return noise_corr

    def _detect(
        self,
        *,
        exposure_handles_dict: dict,
        seed: int,
        corr_array: np.ndarray | None,
        skyMap,
        tract: int,
        patch: int,
        mask_array: NDArray | None = None,
    ) -> np.ndarray:
        assert isinstance(self.config, MeasureCoaddsPipeConfig)
        bands = list(self.config.detection_bands)
        missing = [b for b in bands if b not in exposure_handles_dict]
        if missing:
            raise KeyError(
                f"detection band(s) {missing} not in "
                f"{list(exposure_handles_dict.keys())}"
            )

        exposures = {}
        noise_corrs = {}
        for band in bands:
            exposure = exposure_handles_dict[band].get()
            exposure.getPsf().setCacheCapacity(self.config.psfCache)
            exposures[band] = exposure
            noise_corrs[band] = self._load_noise_corr(corr_array, band)

        if len(bands) == 1:
            band = bands[0]
            data = self.anacal.prepare_data(
                exposure=exposures[band],
                band=band,
                survey=self.config.survey,
                seed=seed,
                noise_corr=noise_corrs[band],
                detection=None,
                skyMap=skyMap,
                tract=tract,
                patch=patch,
                mask_array=mask_array,
            )
        else:
            self.log.info("Detecting on the coadd of bands %s", bands)
            data = self.anacal.prepare_data_multiband(
                exposures=exposures,
                bands=bands,
                survey=self.config.survey,
                seed=seed,
                noise_corrs=noise_corrs,
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
        seed: int,
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
            data = self.anacal.prepare_data(
                exposure=exposure,
                seed=seed,
                noise_corr=noise_corr,
                detection=detection,
                band=band,
                survey=self.config.survey,
                skyMap=skyMap,
                tract=tract,
                patch=patch,
                mask_array=mask_array,
            )
            cat = self.fpfs.run(**data)
            if self.config.do_measure_flux_gauss:
                gauss_cat = select_band_gauss_fluxes(
                    self.anacal.run(**data),
                    band,
                    survey=self.config.survey,
                )
                cat = np.asarray(rfn.merge_arrays([cat, gauss_cat], flatten=True))
            if self.config.doPsfHsmMoments:
                # One PSF model HSM measurement per (exposure, band) at
                # the exposure bbox centre; broadcast across all
                # sources in this band. HsmPsfMomentsPlugin uses
                # computeKernelImage (no subpixel offset) by default.
                psf_moments = measure_psf_hsm_moments(
                    self._psfHsmCtx, self.psfHsmMeasurement, exposure,
                )
                psf_block = broadcast_psf_hsm_moments(
                    psf_moments, band, n=len(cat),
                    survey=self.config.survey,
                )
                cat = np.asarray(rfn.merge_arrays([cat, psf_block], flatten=True))
            per_band.append(cat)

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
        detection: NDArray | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        """Run detection (or use external catalog) then forced measurement.

        Parameters
        ----------
        exposure_handles_dict : dict
            Mapping of band name to deferred exposure handle.  Handles
            may be real butler handles or
            :class:`SimulatedExposureHandle` instances.
        corr_array : np.ndarray or None
            Stacked noise correlation array.
        skyMap : BaseSkyMap
            Sky map used for processing.
        tract, patch : int
            Tract and patch identifiers.
        mask : MaskX or None
            Combined bad-pixel / bright-star mask.
        detection : NDArray or None
            External detection catalog.  When provided the internal
            detection step is skipped and this catalog is used directly
            for forced measurement.
        """
        # Seed is band-independent under the default
        # SkyMapIdGeneratorConfig (n_bands=0), so any handle in the dict
        # gives the same catalog_id.  Use the first one to avoid hard-
        # coding "i".
        if seed is None:
            first_handle = next(iter(exposure_handles_dict.values()))
            idGenerator = self.config.idGenerator.apply(first_handle.dataId)
            seed = idGenerator.catalog_id
        if mask is not None:
            mask_array = mask.getArray()
        else:
            mask_array = None
        if detection is not None:
            det_cat = detection
        else:
            det_cat = self._detect(
                exposure_handles_dict=exposure_handles_dict,
                seed=seed,
                corr_array=corr_array,
                skyMap=skyMap,
                tract=tract,
                patch=patch,
                mask_array=mask_array,
            )
        force_cat = self._force(
            detection=det_cat,
            exposure_handles_dict=exposure_handles_dict,
            seed=seed,
            corr_array=corr_array,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            mask_array=mask_array,
        )
        final = rfn.merge_arrays(
            [select_detection_columns(det_cat), force_cat],
            flatten=True,
        )
        # Per-band AB magnitude + shear response for each published flux
        # family (fluxes are on the fixed MAG_ZERO_AB zeropoint here).
        final = np.asarray(add_magnitude_columns(final, MAG_ZERO_AB))
        object_ids = np.int64(seed) * np.int64(1_000_000) + np.arange(len(final), dtype=np.int64)
        final = rfn.append_fields(
            final,
            "object_id",
            object_ids,
            usemask=False,
        )
        if skyMap is not None:
            # Use skymap's patchInfo for is_primary (not exposure bbox)
            tractInfo = skyMap[tract]
            patchInfo = tractInfo[patch]
            pixel_scale = float(tractInfo.getWcs().getPixelScale().asArcseconds())
            set_isPrimary(final, skyMap, tractInfo, patchInfo, pixel_scale)
        return Struct(anacalCatalog=final)
