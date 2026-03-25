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
    "MeasureCellCoaddsPipeConfig",
    "MeasureCellCoaddsPipe",
    "MeasureCellCoaddsPipeConnections",
]

import logging
from itertools import product
from typing import Any

import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.cell_coadds import MultipleCellCoadd
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import ConfigurableField, Field, FieldValidationError
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap, Index2D
from lsst.utils.logging import LsstLogAdapter
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

from ..processor.anacal import AnacalTask
from ..processor.fpfs import FpfsMeasurementTask
from ..utils.image import resize_array, truncate_square

band_order = "ugrizy"


class MeasureCellCoaddsPipeConnections(
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
    cellCoadd = cT.Input(
        doc="Input cell-based coadd image",
        name="{coaddName}_coadd_cell_predetection",
        storageClass="MultipleCellCoadd",
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
    output_catalog = cT.Output(
        doc="anacal catalog",
        name="{coaddName}_cell_coadd_anacal_catalog",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class MeasureCellCoaddsPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=MeasureCellCoaddsPipeConnections,
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
                self.fpfs.__class__.sigma_shapelets1,
                self,
                "sigma_shapelets1 in a wrong range",
            )

    def setDefaults(self):
        super().setDefaults()
        self.anacal.force_size = True
        self.anacal.force_center = True
        self.fpfs.do_compute_detect_weight = False


class MeasureCellCoaddsPipe(PipelineTask):
    _DefaultName = "MeasureCellCoaddsPipe"
    ConfigClass = MeasureCellCoaddsPipeConfig

    def __init__(
        self,
        *,
        config: MeasureCellCoaddsPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log,
            initInputs=initInputs, **kwargs,
        )
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)

        self.makeSubtask("anacal")
        self.makeSubtask("fpfs")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)

        inputs = butlerQC.get(inputRefs)
        tract = int(butlerQC.quantum.dataId["tract"])
        patch = int(butlerQC.quantum.dataId["patch"])

        coadd_handles = inputs["cellCoadd"]
        coadd_handles_dict = {h.dataId["band"]: h for h in coadd_handles}

        outputs = self.run(
            coadd_handles_dict=coadd_handles_dict,
            corr_array=inputs.get("noiseCorrArray", None),
            skyMap=inputs["skyMap"],
            tract=tract,
            patch=patch,
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
                "Noise correlation is not normalized to 1 "
                "at the center pixel."
            )

        self.log.debug(
            "With correlation (band=%s), variance=%s", band, variance,
        )
        return noise_corr

    def _compute_cell_psf(self, exposure) -> np.ndarray:
        """Compute PSF at the center of a stitched cell exposure."""
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        npix = self.config.anacal.npix
        bbox = exposure.getBBox()
        xc = (bbox.getMinX() + bbox.getMaxX()) / 2.0
        yc = (bbox.getMinY() + bbox.getMaxY()) / 2.0
        from lsst.geom import Point2D
        psf_img = exposure.getPsf().computeImage(Point2D(xc, yc)).getArray()
        psf_array = np.asarray(
            resize_array(psf_img, (npix, npix)), dtype=np.float64,
        )
        psf_array /= np.sum(psf_array)
        psf_rcut = npix // 2 - 2
        truncate_square(psf_array, psf_rcut)
        return psf_array

    def _detect_cell(
        self,
        *,
        exposure,
        corr_array: np.ndarray | None,
        skyMap,
        tract: int,
        patch: int,
        seed: int,
        band: str = "i",
    ) -> np.ndarray:
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        exposure.getPsf().setCacheCapacity(self.config.psfCache)
        noise_corr = self._load_noise_corr(corr_array, band)
        psf_array = self._compute_cell_psf(exposure)
        data = self.anacal.prepare_data(
            exposure=exposure,
            band=band,
            seed=seed,
            noise_corr=noise_corr,
            detection=None,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            psf_array=psf_array,
        )
        return self.anacal.run(**data)

    def _force_cell(
        self,
        *,
        detection: NDArray,
        exposures_dict: dict[str, Any],
        corr_array: np.ndarray | None,
        skyMap,
        tract: int,
        patch: int,
        seed: int,
    ) -> np.ndarray:
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        per_band = []
        for band, exposure in exposures_dict.items():
            exposure.getPsf().setCacheCapacity(self.config.psfCache)

            noise_corr = self._load_noise_corr(corr_array, band)
            psf_array = self._compute_cell_psf(exposure)
            data = self.anacal.prepare_data(
                exposure=exposure,
                seed=seed,
                noise_corr=noise_corr,
                detection=detection,
                band=band,
                skyMap=skyMap,
                tract=tract,
                patch=patch,
                psf_array=psf_array,
            )

            colnames = [
                "flux_gauss0", "dflux_gauss0_dg1", "dflux_gauss0_dg2",
                "flux_gauss2", "dflux_gauss2_dg1", "dflux_gauss2_dg2",
                "flux_gauss4", "dflux_gauss4_dg1", "dflux_gauss4_dg2",
                "flux_gauss0_err", "flux_gauss2_err", "flux_gauss4_err",
            ]
            out = []
            out.append(
                rfn.repack_fields(
                    self.anacal.run(**data)[colnames]
                )
            )
            out.append(self.fpfs.run(**data))
            res = rfn.merge_arrays(out, flatten=True)
            map_dict = {name: f"{band}_{name}" for name in colnames}
            res = rfn.rename_fields(res, map_dict)
            per_band.append(res)

        return rfn.merge_arrays(per_band, flatten=True)

    def run(
        self,
        *,
        coadd_handles_dict: dict[str, Any],
        corr_array: np.ndarray | None,
        skyMap,
        tract: int,
        patch: int,
        detection: NDArray | None = None,
        **kwargs,
    ):
        """Run detection and forced measurement on cell-based coadds.

        Parameters
        ----------
        coadd_handles_dict : dict
            Mapping of band name to deferred MultipleCellCoadd handle.
        corr_array : np.ndarray or None
            Stacked noise correlation array.
        skyMap : BaseSkyMap
            Sky map used for processing.
        tract, patch : int
            Tract and patch identifiers.
        detection : NDArray or None
            External detection catalog. When provided the internal
            detection step is skipped and this catalog is used directly
            for forced measurement.
        """
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)

        # Load all MultipleCellCoadd objects (deferred)
        coadds_by_band: dict[str, MultipleCellCoadd] = {
            band: handle.get()
            for band, handle in coadd_handles_dict.items()
        }

        # Use i-band grid to define cells
        det_band = "i"
        if det_band not in coadds_by_band:
            raise KeyError(
                f"band '{det_band}' not in {coadds_by_band.keys()}"
            )
        grid = coadds_by_band[det_band].grid
        nx_cells, ny_cells = grid.shape

        idGenerator = self.config.idGenerator.apply(
            coadd_handles_dict[det_band].dataId
        )
        seed = idGenerator.catalog_id

        cell_results = []
        for ix, iy in product(range(nx_cells), range(ny_cells)):
            cell_id = Index2D(ix, iy)
            bbox = grid.bbox_of(cell_id)
            self.log.debug("Processing cell (%d, %d)", ix, iy)

            # Stitch cell exposures per band
            cell_exposures: dict[str, Any] = {}
            for band, patch_coadd in coadds_by_band.items():
                stitched = patch_coadd.stitch(bbox)
                cell_exposures[band] = stitched.asExposure()

            try:
                # Detection on i-band cell
                if detection is not None:
                    det_cat = detection
                else:
                    det_cat = self._detect_cell(
                        exposure=cell_exposures[det_band],
                        corr_array=corr_array,
                        skyMap=skyMap,
                        tract=tract,
                        patch=patch,
                        seed=seed,
                        band=det_band,
                    )
                if len(det_cat) == 0:
                    continue

                # Forced measurement on all bands
                force_cat = self._force_cell(
                    detection=det_cat,
                    exposures_dict=cell_exposures,
                    corr_array=corr_array,
                    skyMap=skyMap,
                    tract=tract,
                    patch=patch,
                    seed=seed,
                )
                final = rfn.merge_arrays(
                    [det_cat, force_cat], flatten=True,
                )
                cell_results.append(final)
            except Exception as e:
                self.log.error(
                    "Failed to process cell (%d, %d): %s", ix, iy, e,
                )
                continue

        if not cell_results:
            raise RuntimeError("No objects found in any cell")

        output = np.concatenate(cell_results)
        return Struct(output_catalog=output)
