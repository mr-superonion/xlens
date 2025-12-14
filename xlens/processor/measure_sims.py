"""Simulate multi-band exposures and measure sources efficiently.

This task orchestrates the simulation of coadd exposures for multiple
bands using :class:`~xlens.simulator.sim.MultibandSimTask` followed by
source detection and forced measurements mirroring the logic in
``measure_coadds``.  Exposures are processed one band at a time to keep
memory usage low.
"""

__all__ = [
    "MeasureSimsConfig",
    "MeasureSimsConnections",
    "MeasureSimsTask",
]

import logging
from typing import Any

import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.afw.image import ExposureF
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import ConfigurableField, Field, FieldValidationError
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.tasks.coaddBase import makeSkyInfo
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

from ..simulator.sim import MultibandSimTask
from .anacal import AnacalTask
from .fpfs import FpfsMeasurementTask


class MeasureSimsConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={
        "coaddName": "deep",
        "simCoaddName": "sim",
        "mode": 0,
        "rotId": 0,
    },
):
    """Butler connections for :class:`MeasureSimsTask`."""

    skyMap = cT.Input(
        doc="SkyMap to use in processing",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    truthCatalog = cT.Input(
        doc="Truth catalog for simulation",
        name="{simCoaddName}_{mode}_rot{rotId}_coadd_truthCatalog",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract"),
    )
    psfImage = cT.Input(
        doc="PSF image for simulation (optional)",
        name="{coaddName}_coadd_systematics_psfcentered",
        storageClass="ImageF",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        minimum=0,
        deferLoad=True,
    )
    noiseCorrImage = cT.Input(
        doc="Noise correlation image for simulation and measurement (optional)",
        name="{coaddName}_coadd_systematics_noisecorr",
        storageClass="ImageF",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        minimum=0,
        deferLoad=True,
    )
    systematicsMask = cT.Input(
        doc="Systematics mask for coadd exposures used during simulation (optional)",
        name="{coaddName}_coadd_systematics_mask",
        storageClass="Mask",
        dimensions=("skymap", "tract", "patch", "band"),
        multiple=True,
        minimum=0,
        deferLoad=True,
    )
    output_catalog = cT.Output(
        doc="Measurement catalog on simulated exposures",
        name="{simCoaddName}_coadd_anacal_catalog",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class MeasureSimsConfig(
    PipelineTaskConfig,
    pipelineConnections=MeasureSimsConnections,
):
    anacal = ConfigurableField(
        target=AnacalTask,
        doc="AnaCal Task for detection stage (i-band)",
    )
    fpfs = ConfigurableField(
        target=FpfsMeasurementTask,
        doc="Fpfs Source Measurement Task",
    )
    simulator = ConfigurableField(
        target=MultibandSimTask,
        doc="Simulation task used to generate exposures band-by-band",
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


class MeasureSimsTask(PipelineTask):
    """Simulate coadds for multiple bands and run measurements sequentially."""

    _DefaultName = "MeasureSimsTask"
    ConfigClass = MeasureSimsConfig

    def __init__(
        self,
        *,
        config: MeasureSimsConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(config=config, log=log, initInputs=initInputs, **kwargs)
        assert isinstance(self.config, MeasureSimsConfig)

        self.makeSubtask("anacal")
        self.makeSubtask("fpfs")
        self.makeSubtask("simulator")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, MeasureSimsConfig)

        inputs = butlerQC.get(inputRefs)
        tract = int(butlerQC.quantum.dataId["tract"])
        patch = int(butlerQC.quantum.dataId["patch"])

        mask_handles = inputs.get("systematicsMask", [])
        mask_handles_dict = {h.dataId["band"]: h for h in mask_handles}

        corr_handles = inputs.get("noiseCorrImage", [])
        if len(corr_handles) == 0:
            corr_handles_dict = None
        else:
            corr_handles_dict = {h.dataId["band"]: h for h in corr_handles}

        psf_handles = inputs.get("psfImage", [])
        psf_handles_dict = {h.dataId["band"]: h for h in psf_handles}

        outputs = self.run(
            truthCatalog=inputs["truthCatalog"],
            skyMap=inputs["skyMap"],
            tract=tract,
            patch=patch,
            mask_handles_dict=mask_handles_dict,
            correlation_handles_dict=corr_handles_dict,
            psf_handles_dict=psf_handles_dict,
        )
        butlerQC.put(outputs, outputRefs)

    def _load_noise_corr(
        self, correlation_handles_dict: dict | None, band: str
    ) -> NDArray | None:
        if correlation_handles_dict is None:
            return None
        if band not in correlation_handles_dict:
            return None

        noise_corr = correlation_handles_dict[band].get().getArray()
        variance = float(np.amax(noise_corr))
        if variance > 0:
            noise_corr = noise_corr / variance

        ny, nx = noise_corr.shape
        if not np.isclose(noise_corr[ny // 2, nx // 2], 1.0):
            raise RuntimeError(
                "Noise correlation is not normalized to 1 at the center pixel."
            )

        self.log.debug("With correlation (band=%s), variance=%s", band, variance)
        return noise_corr

    def _simulate_band(
        self,
        *,
        band: str,
        tract_info,
        patch: int,
        truthCatalog,
        seed: int,
        psf_handles_dict: dict,
        correlation_handles_dict: dict | None,
        mask_handles_dict: dict,
    ) -> ExposureF:
        kwargs: dict[str, Any] = {
            "tract_info": tract_info,
            "patch_id": patch,
            "band": band,
            "seed": seed,
            "truthCatalog": truthCatalog,
        }

        if band in psf_handles_dict:
            kwargs["psfImage"] = psf_handles_dict[band].get()
        if correlation_handles_dict is not None and band in correlation_handles_dict:
            kwargs["noiseCorrImage"] = correlation_handles_dict[band].get()
        if band in mask_handles_dict:
            kwargs["mask"] = mask_handles_dict[band].get()

        sim_output = self.simulator.run(**kwargs)
        exposure = sim_output.simExposure
        exposure.getPsf().setCacheCapacity(self.config.psfCache)
        return exposure

    def _detect(
        self,
        *,
        exposure: ExposureF,
        band: str,
        noise_corr: NDArray | None,
        skyMap,
        tract: int,
        patch: int,
        seed: int,
        mask_array: NDArray | None,
    ) -> np.ndarray:
        data = self.anacal.prepare_data(
            exposure=exposure,
            band=band,
            seed=seed,
            noise_corr=noise_corr,
            detection=None,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            mask_array=mask_array,
        )
        return self.anacal.run(**data)

    def _measure_band(
        self,
        *,
        exposure: ExposureF,
        band: str,
        detection: NDArray,
        noise_corr: NDArray | None,
        skyMap,
        tract: int,
        patch: int,
        seed: int,
        mask_array: NDArray | None,
    ) -> NDArray:
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
            "flux_gauss0_err",
            "flux_gauss2_err",
            "flux_gauss4_err",
        ]
        out = [rfn.repack_fields(self.anacal.run(**data)[colnames])]
        out.append(self.fpfs.run(**data))
        res = rfn.merge_arrays(out, flatten=True)
        map_dict = {name: f"{band}_{name}" for name in colnames}
        return rfn.rename_fields(res, map_dict)

    def run(
        self,
        *,
        truthCatalog,
        skyMap,
        tract: int,
        patch: int,
        exposure_handles_dict: dict,
        correlation_handles_dict: dict | None,
        psf_handles_dict: dict,
        seed_offset: int = 0,
        mask_handles_dict: dict | None = None,
        **kwargs,
    ):
        sky_info = makeSkyInfo(
            skyMap,
            tractId=tract,
            patchId=patch,
        )
        tract_info = sky_info.tractInfo

        mask_handles_dict = mask_handles_dict or {}

        bands = ["i", "u", "g", "r", "z", "y"]
        id_data_id = dict(tract=tract, patch=patch)

        # Detection on i-band
        detect_band = "i"
        idGenerator = self.config.idGenerator.apply({**id_data_id, "band": detect_band})
        seed = idGenerator.catalog_id + seed_offset
        i_mask = mask_handles_dict.get(detect_band)

        i_exposure = self._simulate_band(
            band=detect_band,
            tract_info=tract_info,
            patch=patch,
            truthCatalog=truthCatalog,
            seed=seed,
            psf_handles_dict=psf_handles_dict,
            correlation_handles_dict=correlation_handles_dict,
            mask_handles_dict=mask_handles_dict,
        )
        i_noise_corr = self._load_noise_corr(correlation_handles_dict, detect_band)
        det_cat = self._detect(
            exposure=i_exposure,
            band=detect_band,
            noise_corr=i_noise_corr,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            seed=seed,
            mask_array=i_mask.get().getArray() if i_mask is not None else None,
        )
        force_outputs = [
            self._measure_band(
                exposure=i_exposure,
                band=detect_band,
                detection=det_cat,
                noise_corr=i_noise_corr,
                skyMap=skyMap,
                tract=tract,
                patch=patch,
                seed=seed,
                mask_array=i_mask.get().getArray() if i_mask is not None else None,
            )
        ]
        del i_exposure

        # Forced measurements for remaining bands
        for band in bands:
            if band == detect_band:
                continue
            idGenerator = self.config.idGenerator.apply({**id_data_id, "band": band})
            seed = idGenerator.catalog_id + seed_offset
            exposure = self._simulate_band(
                band=band,
                tract_info=tract_info,
                patch=patch,
                truthCatalog=truthCatalog,
                seed=seed,
                psf_handles_dict=psf_handles_dict,
                correlation_handles_dict=correlation_handles_dict,
                mask_handles_dict=mask_handles_dict,
            )
            noise_corr = self._load_noise_corr(correlation_handles_dict, band)
            band_mask = mask_handles_dict.get(band)
            force_outputs.append(
                self._measure_band(
                    exposure=exposure,
                    band=band,
                    detection=det_cat,
                    noise_corr=noise_corr,
                    skyMap=skyMap,
                    tract=tract,
                    patch=patch,
                    seed=seed,
                    mask_array=band_mask.get().getArray() if band_mask is not None else None,
                )
            )
            del exposure

        force_cat = rfn.merge_arrays(force_outputs, flatten=True)
        final = rfn.merge_arrays([det_cat, force_cat], flatten=True)
        return Struct(output_catalog=final)
