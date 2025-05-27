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
    "matchPipeConfig",
    "matchPipe",
    "matchPipeConnections",
]

import logging
import os
from typing import Any

import fitsio
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap
from lsst.utils.logging import LsstLogAdapter
from numpy.lib import recfunctions as rfn
from scipy.spatial import KDTree

dm_colnames = [
    "deblend_nChild",
    "deblend_blendedness",
    "deblend_peak_center_x",
    "deblend_peak_center_y",
    "base_Blendedness_raw",
    "base_Blendedness_abs",
    "base_CircularApertureFlux_3_0_instFlux",
    "base_CircularApertureFlux_3_0_instFluxErr",
    "base_CircularApertureFlux_4_5_instFlux",
    "base_CircularApertureFlux_4_5_instFluxErr",
    "base_GaussianFlux_flag",
    "base_GaussianFlux_instFlux",
    "base_GaussianFlux_instFluxErr",
    "base_PsfFlux_instFlux",
    "base_PsfFlux_instFluxErr",
    "base_Variance_value",
    "ext_photometryKron_KronFlux_instFlux",
    "ext_photometryKron_KronFlux_instFluxErr",
    "modelfit_CModel_instFlux",
    "modelfit_CModel_instFluxErr",
    "base_ClassificationExtendedness_value",
    "base_ClassificationSizeExtendedness_value",
    "base_FootprintArea_value",
]


class matchPipeConnections(
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
    anacal_catalog = cT.Input(
        doc="Source catalog with joint detection and measurement",
        name="{coaddName}Coadd_anacal_force",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=False,
    )
    dm_catalog = cT.Input(
        doc="Catalog containing all the single-band measurement information",
        name="{coaddName}Coadd_meas",
        dimensions=("tract", "patch", "band", "skymap"),
        storageClass="SourceCatalog",
        multiple=True,
        deferLoad=True,
    )
    truth_catalog = cT.Input(
        doc="Output truth catalog",
        name="{coaddName}Coadd_truthCatalog",
        dimensions=("skymap", "tract", "patch", "band"),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )
    catalog = cT.Output(
        doc="Source catalog with joint detection and measurement",
        name="{coaddName}Coadd_anacal_match",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class matchPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=matchPipeConnections,
):

    def validate(self):
        super().validate()

    def setDefaults(self):
        super().setDefaults()


class matchPipe(PipelineTask):
    _DefaultName = "matchPipe"
    ConfigClass = matchPipeConfig

    def __init__(
        self,
        *,
        config: matchPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, matchPipeConfig)
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, matchPipeConfig)
        inputs = butlerQC.get(inputRefs)
        tract = butlerQC.quantum.dataId["tract"]
        patch = butlerQC.quantum.dataId["patch"]
        skyMap = inputs["skyMap"]

        dm_handles = inputs["dm_catalog"]
        dm_handles_dict = {
            handle.dataId["band"]: handle for handle in dm_handles
        }
        truth_handles = inputs["truth_catalog"]
        truth_handles_dict = {
            handle.dataId["band"]: handle for handle in truth_handles
        }
        truth_catalog = truth_handles_dict["i"].get().as_array()
        anacal_catalog = inputs["anacal_catalog"].as_array()
        outputs = self.run(
            dm_handles_dict=dm_handles_dict,
            truth_catalog=truth_catalog,
            anacal_catalog=anacal_catalog,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
        )
        butlerQC.put(outputs, outputRefs)
        return

    def merge_dm(self, src: np.ndarray, mrc: np.ndarray, pixel_scale=0.168):
        assert isinstance(self.config, matchPipeConfig)
        # Apply quality mask to DM
        msk = mrc["i_deblend_nChild"] == 0
        mrc = mrc[msk]
        mag_mrc = 27 - 2.5 * np.log10(mrc["i_base_GaussianFlux_instFlux"])
        x_mrc = np.array(mrc["deblend_peak_center_x"])
        y_mrc = np.array(mrc["deblend_peak_center_y"])

        # Magnitude from src
        mag = 27 - 2.5 * np.log10(src["i_flux"])

        # Coordinates
        mrc_coords = np.vstack((x_mrc, y_mrc)).T
        ana_coords = np.vstack(
            (src["x1"] / pixel_scale, src["x2"] / pixel_scale)
        ).T
        ana_tree = KDTree(ana_coords)
        match_dist, match_ndx = ana_tree.query(mrc_coords)
        mag_diffs = mag[match_ndx] - mag_mrc

        # Filter on distance
        mask = match_dist < 6
        ana_idx = match_ndx[mask]
        mrc_idx = np.flatnonzero(mask)
        abs_diffs = np.abs(mag_diffs[mask])

        # Resolve duplicates by lowest magnitude difference
        order = np.lexsort((abs_diffs, ana_idx))
        ana_idx_sorted = ana_idx[order]
        mrc_idx_sorted = mrc_idx[order]
        _, first = np.unique(ana_idx_sorted, return_index=True)

        final_src = src[ana_idx_sorted[first]]
        final_mrc = mrc[mrc_idx_sorted[first]]

        # Combine fields
        combined = rfn.merge_arrays(
            (final_src, final_mrc),
            flatten=True,
            usemask=False,
        )
        return combined

    def merge_truth(self, src: np.ndarray, mrc: np.ndarray, pixel_scale=0.168):
        assert isinstance(self.config, matchPipeConfig)

        cat_ref = fitsio.read(
            os.path.join(os.environ["CATSIM_DIR"], "OneDegSq.fits")
        )
        mag_mrc = cat_ref[mrc["index"]]["i_ab"]
        x_mrc = np.array(mrc["image_x"])
        y_mrc = np.array(mrc["image_y"])

        # Magnitude from src
        mag = 27 - 2.5 * np.log10(src["flux"])

        # Coordinates
        mrc_coords = np.vstack((x_mrc, y_mrc)).T
        ana_coords = np.vstack(
            (src["x1"] / pixel_scale, src["x2"] / pixel_scale)
        ).T
        ana_tree = KDTree(ana_coords)
        match_dist, match_ndx = ana_tree.query(mrc_coords)
        mag_diffs = mag[match_ndx] - mag_mrc

        # Filter on distance
        mask = match_dist < 6
        ana_idx = match_ndx[mask]
        mrc_idx = np.flatnonzero(mask)
        abs_diffs = np.abs(mag_diffs[mask])

        # Resolve duplicates by lowest magnitude difference
        order = np.lexsort((abs_diffs, ana_idx))
        ana_idx_sorted = ana_idx[order]
        mrc_idx_sorted = mrc_idx[order]
        _, first = np.unique(ana_idx_sorted, return_index=True)

        final_src = src[ana_idx_sorted[first]]
        final_mrc = rfn.repack_fields(
            mrc[mrc_idx_sorted[first]][["index", "z"]]
        )
        final_mrc = rfn.rename_fields(final_mrc, {"z": "redshift"})

        # Combine fields
        combined = rfn.merge_arrays(
            (final_src, final_mrc),
            flatten=True,
            usemask=False,
        )
        return combined

    def run(
        self,
        *,
        dm_handles_dict: dict,
        truth_catalog,
        anacal_catalog,
        skyMap,
        tract: int,
        patch: int,
    ):
        assert isinstance(self.config, matchPipeConfig)
        # TODO: Will be removed
        bbox = skyMap[tract][patch].getOuterBBox()
        truth_catalog["image_x"] = bbox.beginX + truth_catalog["image_x"]
        truth_catalog["image_y"] = bbox.beginY + truth_catalog["image_y"]
        # TODO: Will be removed

        dm_catalog = []
        for band in dm_handles_dict.keys():
            handle = dm_handles_dict[band]
            cat = rfn.repack_fields(
                handle.get()
                .asAstropy()
                .to_pandas(index=False)
                .to_records(index=False)[dm_colnames]
            )
            map_dict = {name: f"{band}_" + name for name in dm_colnames}
            dm_catalog.append(rfn.rename_fields(cat, map_dict))
        dm_catalog = rfn.merge_arrays(dm_catalog, flatten=True)
        pixel_scale = (
            skyMap[tract][patch].getWcs().getPixelScale().asDegrees() * 3600
        )
        catalog = self.merge_dm(anacal_catalog, dm_catalog, pixel_scale)
        catalog = self.merge_truth(catalog, truth_catalog, pixel_scale)
        return Struct(catalog=catalog)
