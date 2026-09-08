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
    "photoZPipeConfig",
    "photoZPipe",
    "photoZPipeConnections",
]

import gc
import logging
import pickle
from typing import Any

import astropy.table
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.pex.config import Field, FieldValidationError, ListField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.utils.logging import LsstLogAdapter
from numpy.typing import NDArray

from ..catalog.redshift import flexzboostEstimator

POINT_KEYS = ("zmode", "z025", "z160", "z500", "z840", "z975", "zbest")

# (suffix, comp, dg) -- suffix "0" is the undistorted version
DISTORTIONS = (
    ("0", 1, 0.00),
    ("1p", 1, 0.01),
    ("1m", 1, -0.01),
    ("2p", 2, 0.01),
    ("2m", 2, -0.01),
)


class photoZPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={
        "inputName": "deep_coadd_cell",
    },
):
    anacalCatalog = cT.Input(
        doc="anacal catalog with per-band fluxes and shear derivatives",
        name="{inputName}_anacal_catalog",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=False,
    )
    extinction = cT.Input(
        doc=(
            "Per-object extinction magnitudes with columns a_g, a_r, ... "
            "row-matched to anacalCatalog. Optional."
        ),
        name="extinction",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=False,
        minimum=0,
    )
    redshiftCatalog = cT.Output(
        doc="FlexZBoost point estimates per object and per distortion.",
        name="{inputName}_anacal_fzb_point",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )
    redshiftPdfs = cT.Output(
        doc=(
            "FlexZBoost p(z) grids. Shape (N, nz) when output_distorted_pdfs "
            "is False (undistorted only), else (N, ndist, nz)."
        ),
        name="{inputName}_anacal_fzb_pdfs",
        dimensions=("skymap", "tract", "patch"),
        storageClass="NumpyArray",
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config is not None and not config.output_pdfs:
            self.outputs.remove("redshiftPdfs")


class photoZPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=photoZPipeConnections,
):
    model_path = Field[str](
        doc="Path to the pickled FlexZBoost photo-z model.",
        default="",
    )
    flux_name = Field[str](
        doc="Flux column suffix (e.g. gauss2 -> {band}_fluxgauss2).",
        default="gauss2",
    )
    bands = ListField[str](
        doc="Survey-prefixed band names used as color features, in order.",
        default=["lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"],
    )
    ref_band = Field[str](
        doc="Reference (survey-prefixed) band for the reference magnitude feature.",
        default="lsst_i",
    )
    do_distortions = Field[bool](
        doc=(
            "If True run all five shear distortions (0, 1p, 1m, 2p, 2m); "
            "if False only the undistorted call is made."
        ),
        default=True,
    )
    output_pdfs = Field[bool](
        doc="If True also write the p(z) grid as a second output.",
        default=False,
    )
    output_distorted_pdfs = Field[bool](
        doc=(
            "Only used when output_pdfs is True. If True, save p(z) for all "
            "shear distortions -> shape (N, ndist, nz); if False (default), "
            "save only the undistorted p(z) -> shape (N, nz)."
        ),
        default=False,
    )
    z_max = Field[float](
        doc="Maximum redshift of the p(z) grid (grid runs 0..z_max).",
        default=5.0,
    )
    nzbins = Field[int](
        doc="Number of points on the p(z) redshift grid.",
        default=501,
    )
    include_mag_err = Field[bool](
        doc="If True include color errors as additional features.",
        default=False,
    )

    def validate(self):
        super().validate()
        if not self.model_path:
            raise FieldValidationError(
                self.__class__.model_path,
                self,
                "model_path must be set to a FlexZBoost pickle file.",
            )
        if self.ref_band not in self.bands:
            raise FieldValidationError(
                self.__class__.ref_band,
                self,
                f"ref_band={self.ref_band!r} not in bands={self.bands!r}.",
            )

    def setDefaults(self):
        super().setDefaults()


class photoZPipe(PipelineTask):
    _DefaultName = "photoZPipe"
    ConfigClass = photoZPipeConfig

    def __init__(
        self,
        *,
        config: photoZPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(config=config, log=log, initInputs=initInputs, **kwargs)
        assert isinstance(self.config, photoZPipeConfig)
        with open(self.config.model_path, "rb") as f:
            pz_obj = pickle.load(f)
        self.zobj = flexzboostEstimator(
            pz_obj=pz_obj,
            z_max=self.config.z_max,
            nzbins=self.config.nzbins,
        )
        return

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        assert isinstance(self.config, photoZPipeConfig)
        inputs = butlerQC.get(inputRefs)

        anacal_catalog = inputs["anacalCatalog"].as_array()
        extinction = inputs.get("extinction", None)
        if extinction is not None:
            extinction = extinction.as_array()

        outputs = self.run(
            catalog=anacal_catalog,
            extinction=extinction,
        )
        butlerQC.put(outputs, outputRefs)
        return

    def run(
        self,
        *,
        catalog: NDArray,
        extinction: NDArray | None = None,
        **kwargs,
    ):
        assert isinstance(self.config, photoZPipeConfig)
        cfg = self.config

        if extinction is not None and len(extinction) != len(catalog):
            raise ValueError(
                f"extinction length ({len(extinction)}) does not match " f"catalog length ({len(catalog)})"
            )

        distortions = DISTORTIONS if cfg.do_distortions else DISTORTIONS[:1]
        n = catalog.shape[0]

        common_kwargs = dict(
            flux_name=cfg.flux_name,
            bands=list(cfg.bands),
            ref_band=cfg.ref_band,
            extinction=extinction,
            include_mag_err=cfg.include_mag_err,
        )

        dtype: list = [("object_id", catalog.dtype["object_id"])]
        dtype += [(f"{key}_{suf}", "f4") for suf, _, _ in distortions for key in POINT_KEYS]
        points = np.empty(n, dtype=dtype)
        points["object_id"] = catalog["object_id"]

        # p(z) grids: all distortions only when output_distorted_pdfs is set,
        # otherwise just the undistorted ("0") grid.  "0" is always
        # distortions[0], so its p(z) is filled on the i == 0 iteration.
        pdfs: NDArray | None = None
        if cfg.output_pdfs:
            if cfg.output_distorted_pdfs:
                pdfs = np.empty((n, len(distortions), cfg.nzbins), dtype=np.float32)
            else:
                pdfs = np.empty((n, cfg.nzbins), dtype=np.float32)

        # Empty patch (no detections): the measure step still writes a 0-row
        # anacal catalog for edge/low-coverage patches.  FlexZBoost's predict
        # cannot run on 0 rows (apply_along_axis over an empty axis), so return
        # the already-empty outputs with the full column schema instead.
        if n == 0:
            result = Struct(redshiftCatalog=astropy.table.Table(points))
            if cfg.output_pdfs:
                result.redshiftPdfs = pdfs
            return result

        for i, (suf, comp, dg) in enumerate(distortions):
            self.log.info(
                "FlexZBoost distortion suf=%s comp=%d dg=%+0.2f",
                suf,
                comp,
                dg,
            )
            want_pdf = cfg.output_pdfs and (cfg.output_distorted_pdfs or i == 0)
            res = self.zobj.get_z(
                src=catalog,
                comp=comp,
                dg=dg,
                return_pdfs=want_pdf,
                **common_kwargs,
            )
            for key in POINT_KEYS:
                points[f"{key}_{suf}"] = res[key]
            if want_pdf:
                if cfg.output_distorted_pdfs:
                    pdfs[:, i, :] = res["pdfs"].astype(np.float32)
                else:
                    pdfs[:] = res["pdfs"].astype(np.float32)
            del res
            gc.collect()

        result = Struct(redshiftCatalog=astropy.table.Table(points))
        if cfg.output_pdfs:
            result.redshiftPdfs = pdfs
        return result
