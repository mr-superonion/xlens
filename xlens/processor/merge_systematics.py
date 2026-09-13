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

"""Per-tract average of the per-patch cell-systematics noise correlation.

``BuildCellSystematicsTask`` writes one ``(6, npix, npix)`` noise-correlation
array per patch (``deep_coadd_cell_systematics_noisecorr_6bands``), band slots
ordered by :data:`xlens.processor.systematics_base.band_order` (``ugrizy``).
This task stacks every patch of a tract into one tract-level ``(6, npix, npix)``
array: for each band slot it averages only over the patches that actually have
that band (a non-zero slot), so the unused slots (and any tract with partial
band coverage) stay well defined rather than being diluted by structural zeros.
"""

__all__ = [
    "MergeSystematicsConfig",
    "MergeSystematicsTask",
    "MergeSystematicsConnections",
]

import logging
from typing import Any

import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.pipe.base import (
    NoWorkFound,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.utils.logging import LsstLogAdapter

from .systematics_base import band_order


class MergeSystematicsConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract"),
    defaultTemplates={"coaddName": "deep"},
):
    noiseCorrList = cT.Input(
        doc=(
            "Per-patch stacked noise correlation arrays (6 x npix x npix, "
            "band slots ordered ugrizy) for every patch of the tract."
        ),
        name="{coaddName}_coadd_cell_systematics_noisecorr_6bands",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract", "patch"),
        multiple=True,
        deferLoad=True,
    )
    outputNoiseCorr = cT.Output(
        doc=(
            "Tract-averaged noise correlation array (6 x npix x npix). Each "
            "band slot is the mean over the patches that carry that band; "
            "slots with no contributing patch stay zero."
        ),
        name="{coaddName}_coadd_systematics_noisecorr_6bands_tractavg",
        storageClass="NumpyArray",
        dimensions=("skymap", "tract"),
    )


class MergeSystematicsConfig(
    PipelineTaskConfig,
    pipelineConnections=MergeSystematicsConnections,
):
    pass


class MergeSystematicsTask(PipelineTask):
    """Average the per-patch noise correlation arrays of a tract."""

    _DefaultName = "MergeSystematicsTask"
    ConfigClass = MergeSystematicsConfig

    def __init__(
        self,
        *,
        config: MergeSystematicsConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        tract = int(butlerQC.quantum.dataId["tract"])
        outputs = self.run(tract=tract, **inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, *, noiseCorrList, tract: int) -> Struct:
        """Average per-patch noise correlation arrays over a tract.

        Parameters
        ----------
        noiseCorrList : list
            Deferred handles (or arrays) for each patch's ``(6, npix, npix)``
            noise-correlation array.
        tract : int
            Tract identifier of the quantum.

        Returns
        -------
        Struct
            outputNoiseCorr : np.ndarray (6, npix, npix)
                Per-band mean over the patches that carry that band.
        """
        if not noiseCorrList:
            # No noise-corr inputs for this tract (e.g. systematics run with
            # do_noise_corr_estimation off). Nothing to average -> skip the
            # quantum rather than write an all-zero array that would read as
            # a real estimate.
            raise NoWorkFound(
                f"tract {tract}: no noise-correlation inputs to average"
            )

        accum: np.ndarray | None = None
        count: np.ndarray | None = None  # per-band number of contributing patches
        n_patch = 0
        for handle in noiseCorrList:
            arr = np.asarray(
                handle.get() if hasattr(handle, "get") else handle,
                dtype=np.float64,
            )
            if accum is None:
                accum = np.zeros_like(arr)
                count = np.zeros(arr.shape[0], dtype=np.int64)
            elif arr.shape != accum.shape:
                raise RuntimeError(
                    f"tract {tract}: noise-corr shape mismatch "
                    f"{arr.shape} vs {accum.shape}"
                )
            # A band is "present" on this patch when its slot is not all-zero
            # (BuildCellSystematics leaves absent-band slots exactly zero).
            present = np.any(arr != 0.0, axis=(1, 2))
            accum[present] += arr[present]
            count[present] += 1
            n_patch += 1

        assert accum is not None and count is not None
        mean = np.zeros_like(accum)
        nonzero = count > 0
        mean[nonzero] = accum[nonzero] / count[nonzero].reshape(-1, 1, 1)

        self.log.info(
            "tract %d: averaged noise correlation over %d patches; "
            "per-band patch counts %s (bands %s)",
            tract, n_patch, count.tolist(), list(band_order),
        )
        return Struct(outputNoiseCorr=mean)
