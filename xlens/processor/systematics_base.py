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

"""Shared config and task base for the systematics-building tasks.

``BuildSystematicsTask`` (patch coadds) and ``BuildCellSystematicsTask``
(cell coadds) differ only in how they get an exposure per band -- one
reads a ``deep_coadd``, the other stitches a ``MultipleCellCoadd``.
Everything downstream of that is the same: the same config knobs, the
same per-band bad-pixel mask, the same cross-band union, the same GAIA
bright-star halos, and the same windowed FFT autocorrelation.  That
logic lives here ONCE so a fix lands in both tasks at the same time.

What deliberately stays in the subclasses is the *window* each task
builds before the autocorrelation: the patch task uses a fixed central
[1000:3000] box and no mean subtraction, the cell task an adaptive box
with mean subtraction.  Only the shared FFT tail is factored out here,
so unifying the two windows remains a separate, visible decision.
"""

__all__ = [
    "band_order",
    "BandMaskPlanesConfig",
    "BuildSystematicsConfigBase",
    "BuildSystematicsTaskBase",
]

import anacal
import numpy as np
from lsst.meas.algorithms import (
    LoadReferenceObjectsConfig,
    ReferenceObjectLoader,
)
from lsst.meas.base import SkyMapIdGeneratorConfig
from lsst.pex.config import (
    ChoiceField,
    Config,
    ConfigDictField,
    ConfigField,
    Field,
    FieldValidationError,
    ListField,
)
from lsst.pipe.base import PipelineTask
from numpy.typing import NDArray

from ..utils.image import badMaskDefault, prepare_mask
from ..utils.mask import (
    GAIA_TABLE_DTYPE,
    STAR_MASK_RADIUS_FUNCS,
    build_gaia_xyr,
    get_gaia_table,
)

# Slot order of every stacked per-band output (noise correlation, PSF).
band_order = "ugrizy"


class BandMaskPlanesConfig(Config):
    """One band's bad-mask-plane list, for ``badMaskPlanesPerBand``."""

    planes = ListField[str](
        doc="Mask planes used to reject bad pixels in this band.",
        default=badMaskDefault,
    )


class BuildSystematicsConfigBase(Config):
    """Config fields shared by both systematics tasks."""

    npix = Field[int](
        doc="Size of noise correlation and PSF stamps (must be odd).",
        default=49,
    )
    badMaskPlanes = ListField[str](
        doc=(
            "Mask planes used to reject bad pixels, for every band that "
            "``badMaskPlanesPerBand`` does not override."
        ),
        default=badMaskDefault,
    )
    badMaskPlanesPerBand = ConfigDictField(
        doc=(
            "Per-band override of ``badMaskPlanes``, keyed by PHYSICAL "
            "band. A band absent from this dict falls back to "
            "``badMaskPlanes``. Note the per-band masks are OR-ed into "
            "one patch mask, so dropping a plane for a single band only "
            "changes the result where NO other band flags that pixel."
        ),
        keytype=str,
        itemtype=BandMaskPlanesConfig,
        default={},
    )
    gaiaPadding = Field[int](
        doc="Padding (pixels) when selecting GAIA sources around the patch.",
        default=300,
    )
    gaiaLoader = ConfigField(
        dtype=LoadReferenceObjectsConfig,
        doc="Reference catalog loader for GAIA",
    )
    starMaskType = ChoiceField[str](
        doc=(
            "Name of the GAIA halo-radius model in "
            "xlens.utils.mask.STAR_MASK_RADIUS_FUNCS. 'default' = "
            "450/200/100 px step for mag <= 11/14/20; 'no_mask' = "
            "flat 10 px for every GAIA star with mag <= 20."
        ),
        allowed={k: k for k in STAR_MASK_RADIUS_FUNCS},
        default="default",
    )
    starMaskMagMax = Field[float](
        doc=(
            "Drop GAIA stars fainter than this g magnitude, on top of "
            "the radius model's own cut. None leaves the model's limit "
            "in force (20 for 'default')."
        ),
        default=None,
        optional=True,
    )
    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def mask_planes(self, band: str) -> list[str]:
        """Bad mask planes for ``band``: its override, else the default."""
        override = self.badMaskPlanesPerBand.get(band)
        if override is None:
            return list(self.badMaskPlanes)
        return list(override.planes)

    def setDefaults(self):
        super().setDefaults()
        self.gaiaLoader.requireProperMotion = False
        self.gaiaLoader.anyFilterMapsToThis = "phot_g_mean"

    def validate(self):
        super().validate()
        if self.npix % 2 == 0:
            raise FieldValidationError(
                self.__class__.npix,
                self,
                "npix should be odd number",
            )
        unknown = [b for b in self.badMaskPlanesPerBand if b not in band_order]
        if unknown:
            raise FieldValidationError(
                self.__class__.badMaskPlanesPerBand,
                self,
                f"unknown band(s) {unknown}; expected any of "
                f"{list(band_order)}",
            )


class BuildSystematicsTaskBase(PipelineTask):
    """Base ``PipelineTask`` with the helpers shared by both systematics
    tasks.

    Subclasses use a config deriving from
    :class:`BuildSystematicsConfigBase`.  Not runnable on its own: it
    defines no ``ConfigClass`` and no connections.
    """

    def _build_mask_band(self, exposure, band: str) -> NDArray:
        """Bad-pixel mask for one band: that band's configured mask
        planes plus the image < -6 sigma negative-outlier guard.  This is
        the ONLY place a mask is built for either shear path; the
        measurement tasks consume the union across bands (plus bright
        stars) as-is.
        """
        return prepare_mask(
            exposure.image.array,
            exposure.mask,
            exposure.variance.array,
            self.config.mask_planes(band),
        )

    @staticmethod
    def _merge_mask(
        global_mask: NDArray | None,
        band_mask: NDArray,
    ) -> NDArray:
        """OR one band's mask into the running cross-band union."""
        if global_mask is None:
            return band_mask.astype(np.int16)
        return (global_mask | band_mask).astype(np.int16)

    def _apply_gaia_mask(
        self,
        *,
        mask_array: NDArray,
        bbox,
        wcs,
        gaia_loader: ReferenceObjectLoader | None,
    ) -> NDArray:
        """Add GAIA bright-star halos to ``mask_array``, in place.

        Returns the wide GAIA table for the patch (empty when there is no
        loader), which callers may emit as a data product.
        """
        if gaia_loader is None or bbox is None or wcs is None:
            return np.empty(0, dtype=GAIA_TABLE_DTYPE)
        gaia = gaia_loader.loadPixelBox(
            bbox=bbox,
            filterName="phot_g_mean",
            wcs=wcs,
            bboxToSpherePadding=self.config.gaiaPadding,
        ).refCat
        gaia_table = get_gaia_table(gaia_catalog=gaia, wcs=wcs)
        gaia_array = build_gaia_xyr(
            gaia_table,
            bbox=bbox,
            star_mask_type=self.config.starMaskType,
            mag_max=self.config.starMaskMagMax,
        )
        if gaia_array is not None:
            self.log.info(
                "Adding bright star mask for %d of %d GAIA sources "
                "(starMaskType=%s, magMax=%s)",
                len(gaia_array),
                len(gaia_table),
                self.config.starMaskType,
                self.config.starMaskMagMax,
            )
            anacal.mask.add_bright_star_mask(
                mask_array=mask_array,
                star_array=gaia_array,
            )
        return gaia_table

    @staticmethod
    def _correlate(
        noise_array: NDArray,
        window_array: NDArray,
        npix: int,
    ) -> NDArray:
        """Windowed noise autocorrelation, cut to ``npix`` x ``npix``.

        Both inputs are the already-masked central cut-out; they are
        zero-padded here so the FFT does not wrap around, correlated,
        and the noise correlation is divided by the window's own
        autocorrelation to undo the window's lag-dependent weighting.
        """
        pad_width = ((10, 10), (10, 10))
        window_array = np.pad(
            window_array, pad_width=pad_width, mode="constant",
            constant_values=0.0,
        )
        noise_array = np.pad(
            noise_array, pad_width=pad_width, mode="constant",
            constant_values=0.0,
        )
        ny, nx = window_array.shape
        npixl = npix // 2
        npixr = npix // 2 + 1

        def _auto(array):
            return np.fft.fftshift(
                np.fft.ifft2(np.abs(np.fft.fft2(array)) ** 2.0)
            ).real[
                ny // 2 - npixl: ny // 2 + npixr,
                nx // 2 - npixl: nx // 2 + npixr,
            ]

        noise_corr = _auto(noise_array)
        window_corr = _auto(window_array)
        good = window_corr > 0
        out = np.zeros_like(window_corr, dtype=np.float32)
        out[good] = noise_corr[good] / window_corr[good]
        return out
