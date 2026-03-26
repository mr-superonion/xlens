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
from typing import Any

import anacal
import lsst.afw.image as afwImage
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
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

from ..processor.anacal import AnacalTask
from ..processor.fpfs import FpfsMeasurementTask
from ..utils.catalog import set_isPrimary
from ..utils.image import resize_array, truncate_square


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
    """Detect and measure sources on cell-based coadds.

    Each SingleCellCoadd has a 250x250 outer region and a 150x150 inner
    region with 50px padding on all sides. Detection and measurement are
    performed on the full outer region so that objects near inner-region
    boundaries have complete pixel context. The anacal block inner region
    (pad=50) keeps only objects whose centers fall within the 150x150
    inner region, preventing double-counting across neighboring cells.

    The noise realization stored in each cell coadd is passed directly
    to anacal for noise bias correction. The noise image is rotated by
    90 degrees inside ``prepare_data`` to decorrelate noise from shear,
    matching the LSST metadetection convention.
    """

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
            skyMap=inputs["skyMap"],
            tract=tract,
            patch=patch,
        )
        butlerQC.put(outputs, outputRefs)

    def _compute_cell_psf(self, psf_image_array: NDArray) -> np.ndarray:
        """Compute PSF array from a SingleCellCoadd's psf_image."""
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        npix = self.config.anacal.npix
        psf_array = np.asarray(
            resize_array(psf_image_array, (npix, npix)), dtype=np.float64,
        )
        psf_array /= np.sum(psf_array)
        psf_rcut = npix // 2 - 2
        truncate_square(psf_array, psf_rcut)
        return psf_array

    @staticmethod
    def _build_cell_exposure(cell, photoCalib) -> afwImage.ExposureF:
        """Build an ExposureF from a SingleCellCoadd's outer region."""
        mi = cell.outer.asMaskedImage()
        exp = afwImage.ExposureF(mi)
        exp.setWcs(cell.wcs)
        exp.setPhotoCalib(photoCalib)
        return exp

    @staticmethod
    def _get_cell_noise(cell) -> NDArray | None:
        """Extract the noise array from a SingleCellCoadd's outer region."""
        noise_reals = cell.outer.noise_realizations
        if len(noise_reals) > 0:
            return np.asarray(noise_reals[0].array, dtype=np.float64)
        return None

    @staticmethod
    def _build_single_block(exposure, psf_array, pixel_scale):
        """Build a single anacal block covering the full 250x250 cell.

        The block inner region is set with pad=50, matching the cell's
        inner/outer structure (250x250 outer, 150x150 inner, 50px on
        each side). Anacal only keeps detections whose centers fall
        within the block inner region [50, 200) x [50, 200).
        """
        bbox = exposure.getBBox()
        width = bbox.getWidth()
        height = bbox.getHeight()
        pad = 50
        bb = anacal.geometry.block(
            int(width // 2),   # xcen
            int(height // 2),  # ycen
            0, 0,              # xmin, ymin
            width, height,     # xmax, ymax
            pad, pad,          # xmin_in, ymin_in
            width - pad,       # xmax_in
            height - pad,      # ymax_in
            pixel_scale,
            0,                 # index
        )
        bb.psf_array = psf_array.copy()
        return [bb]

    def _detect_cell(
        self,
        *,
        exposure,
        psf_array: NDArray,
        noise_array: NDArray | None,
        skyMap,
        tract: int,
        patch: int,
        seed: int,
        band: str = "i",
    ) -> np.ndarray:
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        pixel_scale = float(exposure.getWcs().getPixelScale().asArcseconds())
        blocks = self._build_single_block(exposure, psf_array, pixel_scale)
        data = self.anacal.prepare_data(
            exposure=exposure,
            band=band,
            seed=seed,
            detection=None,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            psf_array=psf_array,
            noise_array=noise_array,
            blocks=blocks,
        )
        return self.anacal.run(**data)

    def _force_cell(
        self,
        *,
        detection: NDArray,
        exposures_dict: dict[str, Any],
        psf_arrays_dict: dict[str, NDArray],
        noise_arrays_dict: dict[str, NDArray | None],
        skyMap,
        tract: int,
        patch: int,
        seed: int,
    ) -> np.ndarray:
        assert isinstance(self.config, MeasureCellCoaddsPipeConfig)
        per_band = []
        for band, exposure in exposures_dict.items():
            psf_array = psf_arrays_dict[band]
            noise_array = noise_arrays_dict.get(band, None)
            pixel_scale = float(
                exposure.getWcs().getPixelScale().asArcseconds()
            )
            blocks = self._build_single_block(
                exposure, psf_array, pixel_scale,
            )
            data = self.anacal.prepare_data(
                exposure=exposure,
                seed=seed,
                detection=detection,
                band=band,
                skyMap=skyMap,
                tract=tract,
                patch=patch,
                psf_array=psf_array,
                noise_array=noise_array,
                blocks=blocks,
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
        skyMap,
        tract: int,
        patch: int,
        detection: NDArray | None = None,
        **kwargs,
    ):
        """Run detection and forced measurement on cell-based coadds.

        Iterates over individual SingleCellCoadd objects. For each cell,
        measurement is performed on the full 250x250 outer region so
        objects near inner-region boundaries have complete pixel data.
        The anacal block inner region (pad=50) retains only objects in
        the 150x150 inner region, ensuring no double-counting.

        Parameters
        ----------
        coadd_handles_dict : dict
            Mapping of band name to deferred MultipleCellCoadd handle.
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

        # Use i-band to define cells
        det_band = "i"
        if det_band not in coadds_by_band:
            raise KeyError(
                f"band '{det_band}' not in {coadds_by_band.keys()}"
            )
        det_coadd = coadds_by_band[det_band]

        idGenerator = self.config.idGenerator.apply(
            coadd_handles_dict[det_band].dataId
        )
        seed = idGenerator.catalog_id

        # Get photoCalib from the stitched coadd (spatially constant)
        photoCalib = det_coadd.stitch().asExposure().getPhotoCalib()

        cell_results = []
        for cell_id, det_cell in det_coadd.cells.items():
            ix, iy = int(cell_id.x), int(cell_id.y)
            self.log.debug("Processing cell (%d, %d)", ix, iy)

            # Build outer exposures, PSF arrays, and noise per band
            cell_exposures: dict[str, Any] = {}
            cell_psf_arrays: dict[str, NDArray] = {}
            cell_noise_arrays: dict[str, NDArray | None] = {}
            for band, band_coadd in coadds_by_band.items():
                cell = band_coadd.cells[cell_id]
                cell_exposures[band] = self._build_cell_exposure(
                    cell, photoCalib,
                )
                cell_psf_arrays[band] = self._compute_cell_psf(
                    cell.psf_image.array,
                )
                cell_noise_arrays[band] = self._get_cell_noise(cell)

            try:
                # Detection on i-band outer region
                if detection is not None:
                    det_cat = detection
                else:
                    det_cat = self._detect_cell(
                        exposure=cell_exposures[det_band],
                        psf_array=cell_psf_arrays[det_band],
                        noise_array=cell_noise_arrays.get(det_band, None),
                        skyMap=skyMap,
                        tract=tract,
                        patch=patch,
                        seed=seed,
                        band=det_band,
                    )

                if len(det_cat) == 0:
                    continue

                # Forced measurement on all bands (outer region)
                force_cat = self._force_cell(
                    detection=det_cat,
                    exposures_dict=cell_exposures,
                    psf_arrays_dict=cell_psf_arrays,
                    noise_arrays_dict=cell_noise_arrays,
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
        if skyMap is not None:
            tractInfo = skyMap[tract]
            patchInfo = tractInfo[patch]
            pixel_scale = float(
                tractInfo.getWcs().getPixelScale().asArcseconds()
            )
            set_isPrimary(output, skyMap, tractInfo, patchInfo, pixel_scale)
        return Struct(output_catalog=output)
