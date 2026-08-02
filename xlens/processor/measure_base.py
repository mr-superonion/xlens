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

"""Shared config and task base classes for the AnaCal measurement tasks.

``MeasureCoaddsPipe``, ``MeasureCellCoaddsPipe`` and ``AnacalDetectPipe``
share their anacal/fpfs subtask wiring, validation, seed derivation, the
per-band extra columns (Gaussian fluxes, PSF HSM moments) and catalog
finalization.  That logic lives here ONCE, so a fix lands in every task
at the same time -- the ``.fields`` validation crash existed in three
copies precisely because this wiring used to be triplicated.
"""

__all__ = [
    "AnacalMeasureConfigBase",
    "MeasureBandsConfigBase",
    "AnacalMeasureTaskBase",
]

import lsst.afw.table as afwTable
import lsst.daf.base as dafBase
import numpy as np
from lsst.meas.base import SingleFrameMeasurementTask, SkyMapIdGeneratorConfig
from lsst.pex.config import (
    Config,
    ConfigurableField,
    Field,
    FieldValidationError,
    ListField,
)
from lsst.pipe.base import PipelineTask
from numpy.lib import recfunctions as rfn
from numpy.typing import NDArray

from ..catalog.utils import add_magnitude_columns
from ..utils.catalog import set_isPrimary
from ..utils.columns import select_band_gauss_fluxes
from ..utils.constants import MAG_ZERO_AB
from ..utils.image import (
    broadcast_psf_hsm_moments,
    build_psf_hsm_context,
    default_psf_hsm_plugin_config,
    measure_psf_hsm_moments,
)
from .anacal import AnacalTask
from .fpfs import FpfsMeasurementTask


class AnacalMeasureConfigBase(Config):
    """Config fields and validation shared by every AnaCal measurement task."""

    anacal = ConfigurableField(
        target=AnacalTask,
        doc="AnaCal Task for the detection stage (see detection_bands)",
    )
    fpfs = ConfigurableField(
        target=FpfsMeasurementTask,
        doc="Fpfs Source Measurement Task",
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def _fpfs_required(self) -> bool:
        """Whether the fpfs subtask will actually run (hook for subclasses)."""
        return True

    def validate(self):
        super().validate()
        if self._fpfs_required() and self.fpfs.sigma_shapelets1 < 0.0:
            raise FieldValidationError(
                self.fpfs.__class__.sigma_shapelets1,
                self,
                "sigma_shapelets1 in a wrong range",
            )


class MeasureBandsConfigBase(AnacalMeasureConfigBase):
    """Adds the fields shared by the two detect-then-force measurement tasks.

    The per-task superset check on ``detection_bands`` (against ``sim_bands``
    or ``bands``) stays in the subclasses: the reference list, the blamed
    field and the message genuinely differ.
    """

    do_measure_flux_gauss = Field[bool](
        doc=(
            "If True, also run AnaCal forced measurement during the "
            "force stage to extract per-band Gaussian fluxes and merge "
            "them into the output catalog."
        ),
        default=False,
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
    doPsfHsmMoments = Field[bool](
        doc=(
            "If True, run lsst.meas.extensions.shapeHSM.HsmPsfMomentsPlugin "
            "+ HigherOrderMomentsPSFPlugin (the same plugins DRP uses) once "
            "per (exposure or cell, band) on the PSF model, and broadcast "
            "the resulting PSF moments to every source. Adds "
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
            "the PSF model HSM moments. Only used when doPsfHsmMoments is "
            "True."
        ),
    )

    def validate(self):
        super().validate()
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

    def setDefaults(self):
        super().setDefaults()
        self.anacal.force_size = True
        self.anacal.force_center = True
        # Shared DRP-equivalent plugin wiring, so column names match across
        # both measurement flavours. HigherOrderMomentsPSF defaults to
        # (min_order=3, max_order=4); to widen, set
        # cfg.psfHsmMeasurement.plugins[
        #     "ext_shapeHSM_HigherOrderMomentsPSF"].max_order = N.
        default_psf_hsm_plugin_config(self.psfHsmMeasurement)


class AnacalMeasureTaskBase(PipelineTask):
    """Base ``PipelineTask`` with the helpers shared by the AnaCal
    measurement tasks.

    Subclasses use a config deriving from :class:`MeasureBandsConfigBase`
    (except :meth:`_seed_from_handle` and :meth:`_normalize_noise_corr`,
    which only need ``idGenerator``).  Not runnable on its own: it defines
    no ``ConfigClass`` or connections.
    """

    def _make_measure_subtasks(self) -> None:
        """anacal + fpfs subtasks, plus the PSF-HSM subtask when enabled."""
        self.makeSubtask("anacal")
        self.makeSubtask("fpfs")
        if self.config.doPsfHsmMoments:
            # Schema is shared across every measurement; the plugins
            # register their fields on construction.
            schema = afwTable.SourceTable.makeMinimalSchema()
            self.makeSubtask(
                "psfHsmMeasurement",
                schema=schema,
                algMetadata=dafBase.PropertyList(),
            )
            self._psfHsmCtx = build_psf_hsm_context(
                schema, self.config.psfHsmMeasurement,
            )

    def _seed_from_handle(self, handle) -> int:
        """Patch-level seed from a butler handle's dataId."""
        return self.config.idGenerator.apply(handle.dataId).catalog_id

    def _normalize_noise_corr(self, noise_corr: NDArray | None) -> NDArray | None:
        """Normalize a noise-correlation image to 1 at its centre pixel.

        Returns ``None`` for missing input or a non-positive peak; raises if
        the maximum is not at the centre pixel (the kernel would then be
        applied off-centre).
        """
        if noise_corr is None:
            return None
        variance = float(np.amax(noise_corr))
        if variance <= 0:
            return None
        noise_corr = noise_corr / variance
        ny, nx = noise_corr.shape
        if not np.isclose(noise_corr[ny // 2, nx // 2], 1.0):
            raise RuntimeError(
                "Noise correlation is not normalized to 1 at the center pixel."
            )
        return noise_corr

    def _append_gauss_fluxes(
        self,
        cat: NDArray,
        *,
        data: dict,
        band: str,
    ) -> NDArray:
        """Merge the per-band Gaussian fluxes into a forced catalog.

        No-op unless ``do_measure_flux_gauss`` is set.  ``data`` is the
        prepared measurement dict for this band, reused for the AnaCal
        forced run that produces the fluxes.
        """
        if not self.config.do_measure_flux_gauss:
            return cat
        gauss_cat = select_band_gauss_fluxes(
            self.anacal.run(**data),
            band,
            survey=self.config.survey,
        )
        return np.asarray(rfn.merge_arrays([cat, gauss_cat], flatten=True))

    def _append_psf_hsm_moments(
        self,
        cat: NDArray,
        *,
        band: str,
        hsm_exposure,
    ) -> NDArray:
        """Merge the broadcast PSF HSM moments into a forced catalog.

        No-op unless ``doPsfHsmMoments`` is set.  ``hsm_exposure`` is what
        the PSF HSM plugins measure on -- the band exposure for patch
        coadds, the synthetic PSF stamp exposure for cell coadds.  One
        measurement per (exposure or cell, band), broadcast across all
        sources of this band.
        """
        if not self.config.doPsfHsmMoments or hsm_exposure is None:
            return cat
        psf_moments = measure_psf_hsm_moments(
            self._psfHsmCtx, self.psfHsmMeasurement, hsm_exposure,
        )
        psf_block = broadcast_psf_hsm_moments(
            psf_moments, band, n=len(cat),
            survey=self.config.survey,
        )
        return np.asarray(rfn.merge_arrays([cat, psf_block], flatten=True))

    def _finalize_catalog(
        self,
        catalog: NDArray,
        *,
        seed: int,
        skyMap,
        tract: int,
        patch: int,
    ) -> NDArray:
        """Magnitude columns, stable object ids, and is_primary flags."""
        # Per-band AB magnitude + shear response for each published flux
        # family (fluxes are on the fixed MAG_ZERO_AB zeropoint here).
        catalog = np.asarray(add_magnitude_columns(catalog, MAG_ZERO_AB))
        # Stable per-object IDs derived from the patch-level seed. Used
        # downstream by ``photoZPipe`` and any object-level joiners.
        object_ids = np.int64(seed) * np.int64(1_000_000) + np.arange(
            len(catalog), dtype=np.int64
        )
        catalog = rfn.append_fields(
            catalog,
            "object_id",
            object_ids,
            usemask=False,
        )
        if skyMap is not None:
            # Use skymap's patchInfo for is_primary: the skymap patch inner
            # bbox defines the non-overlapping tiling (not the exposure or
            # cell-coadd bbox).
            tractInfo = skyMap[tract]
            patchInfo = tractInfo[patch]
            pixel_scale = float(
                tractInfo.getWcs().getPixelScale().asArcseconds()
            )
            set_isPrimary(catalog, skyMap, tractInfo, patchInfo, pixel_scale)
        return catalog
