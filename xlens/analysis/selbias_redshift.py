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

"""Selection-bias measurements aggregated in photometric-redshift bins."""
from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Iterable, List

import lsst.pipe.base.connectionTypes as cT
import numpy as np
from astropy.stats import sigma_clipped_stats
from lsst.pex.config import Field, FieldValidationError, ListField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.utils.logging import LsstLogAdapter

from xlens.catalog import measure_shear
from xlens.catalog.redshift import (
    bpzEstimator,
    flexzboostEstimator,
    load_bpz_templates,
)

DEFAULT_BPZ_DATA_PATH = "/gpfs/mnt/gpfs02/astro/workarea/xli6/work/2025-10-15/rail/bpz"

__all__ = [
    "SelBiasRedshiftPipeConnections",
    "SelBiasRedshiftPipeConfig",
    "SelBiasRedshiftPipe",
    "SelBiasRedshiftSummaryPipeConnections",
    "SelBiasRedshiftSummaryPipeConfig",
    "SelBiasRedshiftSummaryPipe",
]


def _bootstrap_m(
    rng: np.random.Generator,
    e_pos: np.ndarray,
    e_neg: np.ndarray,
    r_pos: np.ndarray,
    r_neg: np.ndarray,
    shear_value: float,
    nsamp: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_obj = e_pos.shape[0]
    ncut = e_pos.shape[1]
    ms = np.zeros((nsamp, ncut), dtype=np.float64)
    cs = np.zeros((nsamp, ncut), dtype=np.float64)
    for idx in range(nsamp):
        choices = rng.integers(0, n_obj, size=n_obj, endpoint=False)
        denom = np.sum(r_pos[choices] + r_neg[choices], axis=0)

        num_m = np.sum(e_pos[choices] - e_neg[choices], axis=0)
        gamma = num_m / denom
        ms[idx] = gamma / shear_value - 1.0

        num_c = np.sum(e_pos[choices] + e_neg[choices], axis=0)
        cs[idx] = num_c / denom
    return ms, cs


def _build_redshift_estimator(
    *,
    redshift: str,
    bands: str,
    model_path: str,
    filter_name: str,
    bpz_data_path: str,
):
    if redshift == "flexzboost":
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return flexzboostEstimator(pz_obj=model)
    if redshift == "bpz":
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        flux_templates = load_bpz_templates(
            data_path=bpz_data_path,
            bands=bands,
            filter_name=filter_name,
        )
        zp_errors = [0.02] * len(bands)
        return bpzEstimator(flux_templates, model, zp_errors)
    raise ValueError(f"Unsupported redshift estimator '{redshift}'")


class SelBiasRedshiftPipeConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "patch"),
    defaultTemplates={
        "coaddName": "deep",
        "dataType": "",
        "version": "",
    },
):
    src00 = cT.Input(
        doc="Negative shear catalog (rotation 0).",
        name="{coaddName}_0_rot0_coadd_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    src01 = cT.Input(
        doc="Negative shear catalog (rotation 1).",
        name="{coaddName}_0_rot1_coadd_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
        minimum=0,
    )

    src10 = cT.Input(
        doc="Positive shear catalog (rotation 0).",
        name="{coaddName}_1_rot0_coadd_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
    )

    src11 = cT.Input(
        doc="Positive shear catalog (rotation 1).",
        name="{coaddName}_1_rot1_coadd_{dataType}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
        minimum=0,
    )

    summary = cT.Output(
        doc="Summary statistics per redshift bin.",
        name="{coaddName}_coadd_anacal_selbias_redshift_{dataType}{version}",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "patch"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class SelBiasRedshiftPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=SelBiasRedshiftPipeConnections,
):
    do_correct_selection_bias = Field[bool](
        doc="Whether to apply selection-bias corrections.",
        default=True,
    )
    target = Field[str](
        doc="Shear component to measure ('g1' or 'g2').",
        default="g1",
    )
    shear_value = Field[float](
        doc="Absolute value of the applied shear.",
        default=0.02,
    )
    flux_min = Field[float](
        doc="Flux cut applied to each band before selection.",
        default=30.0,
    )
    zbounds = ListField[float](
        doc="Redshift boundaries defining the bins.",
        default=[0.3, 0.6, 0.9, 1.2, 1.5, 1.8],
    )
    emax = Field[float](
        doc="Maximum ellipticity magnitude.",
        default=0.3,
    )
    dg = Field[float](
        doc="Finite-difference step for selection response.",
        default=0.02,
    )
    z_width95_max = Field[float](
        doc="Maximum allowed 95% width of the redshift PDF.",
        default=2.75,
    )
    mag_zero = Field[float](
        doc="Zero-point magnitude used by the redshift estimator.",
        default=30.0,
    )
    flux_name = Field[str](
        doc="Flux column suffix used for the selection cut.",
        default="gauss2",
    )
    bands = Field[str](
        doc="Ordered list of bands used for flux cuts.",
        default="grizy",
    )
    ref_band = Field[str](
        doc="Reference band for color features.",
        default="i",
    )
    redshift_estimator = Field[str](
        doc="Photometric redshift estimator to use (flexzboost/bpz).",
        default="flexzboost",
    )
    model_path = Field[str](
        doc="Path to the serialized photometric redshift estimator.",
        default="",
    )
    bpz_data_path = Field[str](
        doc="Directory with BPZ flux templates.",
        default=DEFAULT_BPZ_DATA_PATH,
    )
    filter_name = Field[str](
        doc="Observation filter name",
        default="DC2LSST",
    )

    def validate(self):
        super().validate()
        if len(self.connections.dataType) == 0:
            raise ValueError("connections.dataType missing")

        if self.target not in {"g1", "g2"}:
            raise FieldValidationError(
                self.__class__.target,
                self,
                "target must be either 'g1' or 'g2'",
            )

        if self.redshift_estimator not in {"flexzboost", "bpz"}:
            raise FieldValidationError(
                self.__class__.redshift_estimator,
                self,
                "redshift_estimator must be 'flexzboost' or 'bpz'",
            )

        if self.shear_value < 0.0 or self.shear_value > 0.10:
            raise FieldValidationError(
                self.__class__.shear_value,
                self,
                "shear_value should be in [0.00, 0.10]",
            )


class SelBiasRedshiftPipe(PipelineTask):
    _DefaultName = "FpfsSelBiasRedshiftTask"
    ConfigClass = SelBiasRedshiftPipeConfig

    def __init__(
        self,
        *,
        config: SelBiasRedshiftPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, SelBiasRedshiftPipeConfig)
        self._zbounds = list(self.config.zbounds)
        self._ncut = len(self._zbounds) + 1
        self._z_estimator = self._init_z_estimator()

    def _init_z_estimator(self):
        assert isinstance(self.config, SelBiasRedshiftPipeConfig)
        config = self.config
        if config.model_path:
            model_path = config.model_path
        else:
            if config.redshift_estimator == "flexzboost":
                env_name = "FLEXZ_MODEL"
            else:
                env_name = "BPZ_MODEL"
            model_path = os.environ.get(env_name, "")
        if not model_path:
            raise RuntimeError(
                "model_path is not configured and the corresponding environment"
                "variable is not set"
            )

        if config.redshift_estimator == "bpz":
            data_path = config.bpz_data_path or os.environ.get(
                "BPZ_DATA_PATH", ""
            )
            if not data_path:
                data_path = DEFAULT_BPZ_DATA_PATH
        else:
            data_path = ""

        return _build_redshift_estimator(
            redshift=config.redshift_estimator,
            model_path=model_path,
            bands=self.config.bands,
            filter_name=self.config.filter_name,
            bpz_data_path=data_path,
        )

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def _measure_catalog(self, src) -> tuple[np.ndarray, np.ndarray]:
        assert isinstance(self.config, SelBiasRedshiftPipeConfig)
        config = self.config
        out = measure_shear(
            src=src,
            z_estimator=self._z_estimator,
            zbounds=self._zbounds,
            flux_min=config.flux_min,
            emax=config.emax,
            z_width95_max=config.z_width95_max,
            dg=config.dg,
            target=config.target,
            do_correction=config.do_correct_selection_bias,
            mag_zero=config.mag_zero,
            flux_name=config.flux_name,
            bands=config.bands,
            ref_band=config.ref_band,
        )
        ell = np.asarray(out["e"], dtype=np.float64)
        resp = np.asarray(out["r"], dtype=np.float64)
        resp_sel = np.asarray(out["r_sel"], dtype=np.float64)
        return ell, resp + resp_sel

    def _accumulate_pair(
        self, catalogs: Iterable[np.ndarray | None]
    ) -> tuple[np.ndarray, np.ndarray]:
        e_total = np.zeros(self._ncut, dtype=np.float64)
        r_total = np.zeros(self._ncut, dtype=np.float64)
        for src in catalogs:
            if src is None:
                continue
            ell, resp = self._measure_catalog(src)
            e_total += ell
            r_total += resp
        return e_total, r_total

    def run(self, *, src00, src10, src01=None, src11=None, **kwargs):
        e_neg, r_neg = self._accumulate_pair([src00, src01])
        e_pos, r_pos = self._accumulate_pair([src10, src11])

        data_type = [
            ("e_pos", "f8"),
            ("e_neg", "f8"),
            ("r_pos", "f8"),
            ("r_neg", "f8"),
        ]
        summary = np.zeros(self._ncut, dtype=data_type)
        summary["e_pos"] = e_pos
        summary["e_neg"] = e_neg
        summary["r_pos"] = r_pos
        summary["r_neg"] = r_neg
        return Struct(summary=summary)


class SelBiasRedshiftSummaryPipeConnections(
    PipelineTaskConnections,
    dimensions=(),
    defaultTemplates={
        "coaddName": "deep",
        "dataType": "",
        "version": "",
    },
):
    summary_list = cT.Input(
        doc="Summary catalogs per patch/tract.",
        name="{coaddName}_coadd_anacal_selbias_redshift_{dataType}{version}",
        dimensions=("skymap", "tract", "patch"),
        storageClass="ArrowAstropy",
        multiple=True,
        deferLoad=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class SelBiasRedshiftSummaryPipeConfig(
    PipelineTaskConfig,
    pipelineConnections=SelBiasRedshiftSummaryPipeConnections,
):
    shear_value = Field[float](
        doc="Absolute value of the shear used in simulations.",
        default=0.02,
    )
    stamp_dim = Field[int](
        doc="Usable image dimension in pixels for density/area calculation.",
        default=3900,
    )
    pixel_scale = Field[float](
        doc="Pixel scale in arcsec/pixel.",
        default=0.2,
    )
    bootstrap_samples = Field[int](
        doc="Number of bootstrap resamples used for the m/c uncertainties.",
        default=10000,
    )
    zbounds = ListField[float](
        doc="Redshift boundaries used to validate the stacked outputs.",
        default=[0.3, 0.6, 0.9, 1.2, 1.5, 1.8],
    )

    def validate(self):
        super().validate()
        if len(self.connections.dataType) == 0:
            raise ValueError("connections.dataType missing")

        if self.shear_value < 0.0 or self.shear_value > 0.10:
            raise FieldValidationError(
                self.__class__.shear_value,
                self,
                "shear_value should be in [0.00, 0.10]",
            )


class SelBiasRedshiftSummaryPipe(PipelineTask):
    _DefaultName = "FpfsSelBiasRedshiftSummaryTask"
    ConfigClass = SelBiasRedshiftSummaryPipeConfig

    def __init__(
        self,
        *,
        config: SelBiasRedshiftSummaryPipeConfig | None = None,
        log: logging.Logger | LsstLogAdapter | None = None,
        initInputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            config=config, log=log, initInputs=initInputs, **kwargs
        )
        assert isinstance(self.config, SelBiasRedshiftSummaryPipeConfig)
        self._ncut = len(self.config.zbounds) + 1

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        self.run(**inputs)

    @staticmethod
    def _stack(blocks: List[np.ndarray], ncut: int) -> np.ndarray:
        valid = [blk for blk in blocks if blk.size > 0]
        if not valid:
            return np.zeros((0, ncut), dtype=np.float64)
        return np.vstack(valid)

    def run(self, *, summary_list, **kwargs):
        assert isinstance(self.config, SelBiasRedshiftSummaryPipeConfig)
        arrays_e_pos: List[np.ndarray] = []
        arrays_e_neg: List[np.ndarray] = []
        arrays_r_pos: List[np.ndarray] = []
        arrays_r_neg: List[np.ndarray] = []
        for ref in summary_list:
            res = ref.get()
            arrays_e_pos.append(np.asarray(res["e_pos"], dtype=np.float64))
            arrays_e_neg.append(np.asarray(res["e_neg"], dtype=np.float64))
            arrays_r_pos.append(np.asarray(res["r_pos"], dtype=np.float64))
            arrays_r_neg.append(np.asarray(res["r_neg"], dtype=np.float64))

        ncut = self._ncut
        all_e_pos = self._stack(arrays_e_pos, ncut)
        all_e_neg = self._stack(arrays_e_neg, ncut)
        all_r_pos = self._stack(arrays_r_pos, ncut)
        all_r_neg = self._stack(arrays_r_neg, ncut)

        if all_e_pos.size == 0 or all_e_neg.size == 0:
            raise RuntimeError(
                "No valid (+g/-g) pairs found in the summary inputs."
            )

        num = np.sum(all_e_pos - all_e_neg, axis=0)
        denom = np.sum(all_r_pos + all_r_neg, axis=0)
        m = (num / denom) / self.config.shear_value - 1.0

        c = np.sum(all_e_pos + all_e_neg, axis=0) / denom

        area_arcmin2 = (
            self.config.stamp_dim * self.config.stamp_dim
            * (self.config.pixel_scale / 60.0) ** 2
        )

        _, _, clipped_std = sigma_clipped_stats(
            all_e_pos / np.average(all_r_pos, axis=0), sigma=5.0, axis=0
        )
        neff = (0.26 / clipped_std) ** 2.0 / area_arcmin2

        rng = np.random.default_rng(0)
        if self.config.bootstrap_samples > 0:
            ms, cs = _bootstrap_m(
                rng,
                all_e_pos,
                all_e_neg,
                all_r_pos,
                all_r_neg,
                self.config.shear_value,
                nsamp=self.config.bootstrap_samples,
            )
            ord_ms = np.sort(ms, axis=0)
            lo_idx = int(0.1587 * self.config.bootstrap_samples)
            hi_idx = int(0.8413 * self.config.bootstrap_samples)
            sigma_m = (ord_ms[hi_idx] - ord_ms[lo_idx]) / 2.0

            ord_cs = np.sort(cs, axis=0)
            sigma_c = (ord_cs[hi_idx] - ord_cs[lo_idx]) / 2.0
        else:
            sigma_m = np.zeros_like(m)
            sigma_c = np.zeros_like(c)

        zbounds = list(self.config.zbounds)
        print("==============================================")
        print("Photometric redshift summary")
        print(f"Redshift boundaries: {zbounds}")
        print(f"Paired inputs: {all_e_pos.shape[0]}")
        print(f"Area (arcmin^2): {area_arcmin2:.3f}")
        print("m (per redshift bin):", m)
        print("c (per redshift bin):", c)
        print("n_eff (per redshift bin):", neff)
        print("m 1-sigma (bootstrap):", sigma_m)
        print("c 1-sigma (bootstrap):", sigma_c)
        print("==============================================")

        return Struct(
            m=m,
            c=c,
            n_eff=neff,
            sigma_m=sigma_m,
            sigma_c=sigma_c,
        )
