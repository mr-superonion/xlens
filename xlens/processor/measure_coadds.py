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

import anacal
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.afw.image import MaskX
from lsst.daf.butler import DataCoordinate
from lsst.pex.config import (
    ConfigurableField,
    Field,
    FieldValidationError,
    ListField,
)
from lsst.pipe.base import (
    NoWorkFound,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.tasks.coaddBase import makeSkyInfo
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

from ..simulator.sim import MultibandSimTask
from ..utils.columns import select_detection_columns
from ..utils.handle import SimulatedExposureHandle
from ..utils.image import make_object_psf, rle_table_to_mask
from .measure_base import AnacalMeasureTaskBase, MeasureBandsConfigBase

band_order = "ugrizy"


class MeasureCoaddsPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={
        "inputName": "deep_coadd",
        "outName": "deep_coadd",
        "catName": "cat"
    },
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
    nImage = cT.Input(
        doc=(
            "Per-band coadd visit-count map (nImage). Optional: DP1 "
            "registers it as deep_coadd_n_image, DP2 does not persist "
            "it at all, so the task runs without it. Used for the "
            "config.n_image_min coverage cut."
        ),
        name="{inputName}_n_image",
        storageClass="ImageU",
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
        doc=(
            "Combined anacal bitmask from BuildSystematicsTask, "
            "run-length encoded with a value column (decode with "
            "xlens.utils.image.rle_table_to_mask). Bit 0 = masked "
            "(bad pixels / bright stars; cut), bit 1 = discontinuity "
            "(INEXACT_PSF union; stamped per source as "
            "n_mask_discontinuity, never cut)."
        ),
        name="{inputName}_systematics_mask_rle",
        storageClass="ArrowAstropy",
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
    MeasureBandsConfigBase,
    PipelineTaskConfig,
    pipelineConnections=MeasureCoaddsPipeConnections,
):
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
    sim_bands = ListField[str](
        doc="PHYSICAL bands to simulate when ``use_sim`` is True.",
        default=["u", "g", "r", "i", "z", "y"],
    )
    psfCache = Field[int](
        doc="Size of PSF cache",
        default=100,
    )

    def validate(self):
        super().validate()
        if self.use_sim:
            missing = [
                b for b in self.detection_bands if b not in self.sim_bands
            ]
            if missing:
                raise FieldValidationError(
                    self.__class__.sim_bands,
                    self,
                    f"sim_bands must include every detection band; "
                    f"missing {missing}.",
                )


class MeasureCoaddsPipe(AnacalMeasureTaskBase):
    """Detect and measure sources on patch coadds, cell by cell.

    The patch is covered by overlapping AnaCal cells (250x250 outer
    region, 80px overlap) whose inner regions tile the patch exactly.
    Like :class:`MeasureCellCoaddsPipe` loops over cells, this task loops
    over the cells in Python -- one ``anacal`` call per cell for
    detection + shape measurement, then per band one ``fpfs`` call per
    cell on that cell's own detections.  Detection scans only the inner
    regions, so the per-cell results are the same sources, in the same
    order, as a single whole-patch call.

    The cell loops run through a thread pool of ``num_workers`` threads
    (default 1, a plain serial loop).  Each cell is a complete,
    independent work unit, so this is also the dispatch point for future
    threaded or GPU AnaCal backends.
    """

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

        self._make_measure_subtasks()
        if self.config.use_sim:
            self.makeSubtask("simulator")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, MeasureCoaddsPipeConfig)

        inputs = butlerQC.get(inputRefs)
        tract = int(butlerQC.quantum.dataId["tract"])
        patch = int(butlerQC.quantum.dataId["patch"])

        # The combined bitmask arrives run-length encoded; decode to
        # pixels once, preserving the bit values (0..3).
        if inputs.get("mask", None) is not None:
            arr = rle_table_to_mask(inputs["mask"])
            msk = MaskX(width=arr.shape[1], height=arr.shape[0])
            msk.getArray()[:, :] = arr.astype(
                msk.getArray().dtype, copy=False
            )
            inputs["mask"] = msk

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
            seed = self._seed_from_handle(inputRefs.psfArray)
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
            n_image_handles=self.n_image_handles_dict(inputs),
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

    def _load_noise_corr(
        self, corr_array: np.ndarray | None, band: str
    ) -> NDArray | None:
        if corr_array is None:
            return None
        if band not in band_order:
            return None
        noise_corr = self._normalize_noise_corr(
            corr_array[band_order.index(band)]
        )
        if noise_corr is not None:
            self.log.debug("With correlation (band=%s)", band)
        return noise_corr

    def _cache_bands_multiband(
        self, *, bands: list, exposures: dict, data: dict, cells: list
    ) -> dict:
        """Per-band force-measurement inputs recovered from the detection
        preparation, so ``_force`` can skip ``prepare_data`` for the
        detection bands.

        The multiband stacks were filled from the same per-band
        ``prepare_data`` calls (same seed, same code) that ``_force``
        would repeat, so the band slices here are value-identical to
        what ``_force`` would rebuild, and the per-band cells reuse the
        PSF stamps already evaluated for detection.  The slices are
        views, so the only extra memory is keeping the detection stacks
        (and exposures) alive through the force stage.
        """
        acfg = self.anacal.config
        gal = data["gal_array"]
        noise = data["noise_array"]
        psf_stack = data["psf_array"]
        by_index = {bb.index: bb for bb in cells}
        band_cache: dict = {}
        for ib, band in enumerate(bands):
            exposure = exposures[band]
            bdata = {k: v for k, v in data.items() if k != "detection"}
            bdata["gal_array"] = gal[ib]
            bdata["noise_array"] = None if noise is None else noise[ib]
            bdata["psf_array"] = np.ascontiguousarray(psf_stack[ib])
            bdata["noise_variance"] = float(data["noise_variance"][ib])
            if self.config.survey is not None:
                bdata["base_column_name"] = f"{self.config.survey}_{band}_"
            else:
                bdata["base_column_name"] = band + "_"
            if acfg.psf_model_type == "object":
                bdata["psf_object"] = make_object_psf(
                    exposure.getPsf(),
                    npix=acfg.npix,
                    lsst_bbox=exposure.getBBox(),
                )
            else:
                bdata["psf_object"] = None
            # Same cell geometry as utils.image.get_cells; the PSF
            # stamp is this band's slice of the stack computed for
            # detection, so no LSST PSF evaluations are repeated.
            geo = anacal.geometry.get_cell_list(
                img_ny=gal.shape[1],
                img_nx=gal.shape[2],
                cell_nx=250,
                cell_ny=250,
                cell_overlap=80,
                scale=data["pixel_scale"],
            )
            bcells = []
            for gg in geo:
                src = by_index.get(gg.index)
                if src is None:
                    continue
                gg.psf_array = np.ascontiguousarray(
                    np.asarray(src.psf_array)[ib]
                )
                bcells.append(gg)
            band_cache[band] = {
                "exposure": exposure,
                "data": bdata,
                "cells": bcells,
            }
        return band_cache

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
    ) -> tuple[dict, dict]:
        """Detect on the coadd of ``config.detection_bands``, per cell.

        Returns ``(det_cats, band_cache)``: ``det_cats`` maps each cell
        index with non-empty detections to that cell's anacal catalog
        (in cell order, which matches the row order of the previous
        whole-patch call); ``band_cache`` carries the detection bands'
        prepared per-band data so ``_force`` does not prepare them again.
        """
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
                num_workers=self.config.num_workers,
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
                num_workers=self.config.num_workers,
            )
        cells = data.pop("cells")

        if len(bands) == 1:
            # The single-band prepare_data call is identical to the one
            # _force would make for this band, so hand it over as-is.
            band = bands[0]
            band_cache = {
                band: {
                    "exposure": exposures[band],
                    "data": {
                        k: v for k, v in data.items() if k != "detection"
                    },
                    "cells": cells,
                }
            }
        else:
            band_cache = self._cache_bands_multiband(
                bands=bands, exposures=exposures, data=data, cells=cells,
            )

        def _detect_one(cell):
            try:
                return self._run_anacal(**data, cells=[cell])
            except Exception as e:
                self.log.error(
                    "Detection failed tract=%d patch=%d cell=%d: %s",
                    tract,
                    patch,
                    cell.index,
                    e,
                )
                return None

        det_cats: dict = {}
        for cell, cat in zip(cells, self._map_parallel(_detect_one, cells)):
            if cat is not None and len(cat) > 0:
                det_cats[cell.index] = cat

        if not det_cats:
            # Same edge-of-tract semantics as MeasureCellCoaddsPipe: no
            # detections anywhere means the quantum is SKIPPED, not FAILED.
            raise NoWorkFound(
                f"No objects detected in any cell "
                f"(tract={tract}, patch={patch}); skipping this patch."
            )
        return det_cats, band_cache

    def _force(
        self,
        *,
        detection_dict: dict,
        exposure_handles_dict: dict,
        seed: int,
        corr_array: np.ndarray | None,
        skyMap,
        tract: int,
        patch: int,
        mask_array: NDArray | None = None,
        band_cache: dict | None = None,
        n_image_handles: dict | None = None,
    ) -> dict:
        """Force-measure each detection group across all bands.

        ``detection_dict`` maps a cell index to that cell's detection
        catalog -- or the single key ``None`` to an external whole-patch
        catalog when the detection step was skipped.  Returns a dict with
        the same keys mapping to the band-merged forced catalog for
        groups where every band succeeded.
        """
        assert isinstance(self.config, MeasureCoaddsPipeConfig)
        active = list(detection_dict.keys())
        force_parts: dict[Any, list] = {key: [] for key in active}
        bands = list(exposure_handles_dict.keys())

        for band in bands:
            cached = (band_cache or {}).get(band)
            if cached is not None:
                # The detection stage already prepared this band with the
                # same seed and code path; reuse it instead of loading and
                # preparing the exposure a second time.
                self.log.debug("Measuring band %s (reusing detection prep)",
                               band)
                exposure = cached["exposure"]
                data = dict(cached["data"])
                cell_map = {bb.index: bb for bb in cached["cells"]}
            else:
                self.log.debug("Measuring band %s", band)
                exposure = exposure_handles_dict[band].get()
                exposure.getPsf().setCacheCapacity(self.config.psfCache)
                noise_corr = self._load_noise_corr(corr_array, band)
                data = self.anacal.prepare_data(
                    exposure=exposure,
                    seed=seed,
                    noise_corr=noise_corr,
                    detection=None,
                    band=band,
                    survey=self.config.survey,
                    skyMap=skyMap,
                    tract=tract,
                    patch=patch,
                    mask_array=mask_array,
                    num_workers=self.config.num_workers,
                )
                data.pop("detection")
                cell_map = {bb.index: bb for bb in data.pop("cells")}

            # PSF HSM moments per cell, measured SERIALLY here rather
            # than inside _force_one: the plugins hold the GIL, so in the
            # thread pool they would serialise the AnaCal work they are
            # meant to overlap. They depend only on the cell's PSF stamp,
            # not on the sources, so one pass over the cells is enough --
            # the previous code re-measured the SAME patch-centre PSF
            # once per cell and threw the spatial variation away.
            psf_hsm = self._psf_hsm_moments_per_cell(
                list(cell_map.values()), band,
            )

            # Per-band coverage map, sampled per source below. Optional:
            # a repo without nImage simply gets no column.
            n_image_band = None
            handle = (n_image_handles or {}).get(band)
            if handle is not None:
                n_image_band = np.asarray(handle.get().array)
            begin_x = int(data.get("begin_x", 0))
            begin_y = int(data.get("begin_y", 0))
            pixel_scale = float(
                exposure.getWcs().getPixelScale().asArcseconds()
            )

            def _force_one(key):
                det = detection_dict[key]
                try:
                    # fpfs is position-driven; the cell only steers the
                    # gauss-flux run.  A cell present in the detection
                    # band can be ABSENT here (get_cells drops cells
                    # whose PSF cannot be evaluated at the centre, and
                    # coverage differs band to band), so look it up
                    # inside the guard: that makes the cell drop out of
                    # this band like any other failure instead of
                    # killing the whole patch.
                    cell = cell_map.get(key)
                    if cell is None:
                        raise KeyError(
                            f"cell {key} has no PSF in band {band}"
                        )
                    cells = [cell]
                    cat = self._run_fpfs(**data, detection=det)
                    cat = self._append_gauss_fluxes(
                        cat,
                        data={**data, "detection": det, "cells": cells},
                        band=band,
                    )
                    # Pure lookup + column merge: no HSM call, no GIL.
                    cat = self._attach_psf_hsm_moments(
                        cat, band=band, moments=psf_hsm.get(key),
                    )
                    if n_image_band is not None:
                        # det positions are arcsec in the parent frame;
                        # the nImage shares the exposure's pixel grid.
                        cat = self.attach_n_inputs_column(
                            cat,
                            self.n_inputs_at(
                                n_image_band,
                                np.asarray(det["x1_det"]) / pixel_scale
                                - begin_x,
                                np.asarray(det["x2_det"]) / pixel_scale
                                - begin_y,
                                sigma=self.config.anacal.sigma_arcsec * 1.5,
                                scale=pixel_scale,
                            ),
                            band,
                        )
                    return cat
                except Exception as e:
                    self.log.error(
                        "Measurement failed tract=%d patch=%d cell=%s "
                        "band=%s: %s",
                        tract,
                        patch,
                        key,
                        band,
                        e,
                    )
                    return None

            for key, cat in zip(active, self._map_parallel(_force_one, active)):
                if cat is not None:
                    force_parts[key].append(cat)

        nbands = len(bands)
        return {
            key: rfn.merge_arrays(parts, flatten=True)
            for key, parts in force_parts.items()
            if len(parts) == nbands
        }

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
        n_image_handles: dict | None = None,
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
            Combined anacal bitmask: bit 0 = masked (bad pixels /
            bright stars; zeroed, extended, cut), bit 1 = discontinuity
            (INEXACT_PSF union; never cut, but detection stamps every
            source with ``n_mask_discontinuity``, the
            Gaussian-weighted bit-1 fraction in [0, 1], same kernel as
            ``n_mask_base``).
        detection : NDArray or None
            External detection catalog.  When provided the internal
            detection step is skipped; the catalog is partitioned into
            the same per-cell groups internal detection would produce,
            so forced measurement runs (and threads) identically either
            way.  The output preserves the input row order; rows whose
            cell failed in any band are dropped.
        """
        # Seed is band-independent under the default
        # SkyMapIdGeneratorConfig (n_bands=0), so any handle in the dict
        # gives the same catalog_id.
        if seed is None:
            first_handle = next(iter(exposure_handles_dict.values()))
            seed = self._seed_from_handle(first_handle)
        # The mask already carries the anacal uint8 bit convention:
        # bit 0 = masked (cut), bit 1 = discontinuity (kept but stamped
        # into n_mask_discontinuity). anacal only zeroes / extends /
        # cuts on bit 0; bit 1 rides along for the column.
        if mask is not None:
            mask_array = mask.getArray().astype(np.uint8)
        else:
            mask_array = None
            if not self.config.use_sim:
                # Mask building lives entirely in BuildSystematicsTask now;
                # without its output, real-data measurement runs unmasked
                # (saturated pixels, streaks, bright-star halos included).
                self.log.warning(
                    "No systematics mask for tract=%d patch=%d; measuring "
                    "with NO pixel masking. Run BuildSystematicsTask "
                    "upstream unless this is intentional.",
                    tract,
                    patch,
                )
        # Coverage cut: low-visit pixels are where coadd outlier
        # rejection is weakest, so artefacts survive there.
        mask_array = self.apply_n_image_cut(mask_array, n_image_handles or {})

        order: dict | None = None
        if detection is not None:
            # External catalog: partition it into the SAME per-cell
            # groups internal detection would produce (cell inner
            # regions tile the patch), so _force sees one interface and
            # the cell loop threads either way.  The input row order is
            # restored below after the per-group results are merged.
            exposure = next(iter(exposure_handles_dict.values())).get()
            bbox = exposure.getBBox()
            pixel_scale = float(
                exposure.getWcs().getPixelScale().asArcseconds()
            )
            geo = anacal.geometry.get_cell_list(
                img_ny=bbox.getHeight(),
                img_nx=bbox.getWidth(),
                cell_nx=250,
                cell_ny=250,
                cell_overlap=80,
                scale=pixel_scale,
            )
            x0, y0 = bbox.getBeginX(), bbox.getBeginY()
            regions = [
                (bb.index, x0 + bb.xmin_in, y0 + bb.ymin_in,
                 x0 + bb.xmax_in, y0 + bb.ymax_in)
                for bb in geo
            ]
            det_cats, order = self._partition_external_detection(
                detection, regions, pixel_scale,
            )
            if not det_cats:
                raise NoWorkFound(
                    f"External detection catalog is empty "
                    f"(tract={tract}, patch={patch}); skipping this patch."
                )
            band_cache: dict = {}
        else:
            det_cats, band_cache = self._detect(
                exposure_handles_dict=exposure_handles_dict,
                seed=seed,
                corr_array=corr_array,
                skyMap=skyMap,
                tract=tract,
                patch=patch,
                mask_array=mask_array,
            )
        force_cats = self._force(
            detection_dict=det_cats,
            exposure_handles_dict=exposure_handles_dict,
            seed=seed,
            corr_array=corr_array,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            mask_array=mask_array,
            band_cache=band_cache,
            n_image_handles=n_image_handles,
        )
        cell_results = []
        for key, force_cat in force_cats.items():
            final = rfn.merge_arrays(
                [select_detection_columns(det_cats[key]), force_cat],
                flatten=True,
            )
            cell_results.append(final)

        if not cell_results:
            # Same edge-of-tract path as in `_detect` above, but here
            # forced measurement is what produced the empty result.
            raise NoWorkFound(
                f"No measurements produced in any cell "
                f"(tract={tract}, patch={patch}); skipping this patch."
            )

        final = np.concatenate(cell_results)
        if order is not None:
            # External catalogs are row-aligned to their producer (e.g.
            # a det_id column); undo the cell grouping.
            final = self._restore_input_order(
                final, list(force_cats.keys()), order,
            )
        final = self._finalize_catalog(
            final, seed=seed, skyMap=skyMap, tract=tract, patch=patch,
        )
        return Struct(anacalCatalog=final)
