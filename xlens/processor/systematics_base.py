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
    "GALAXY_REFCAT_NAME",
    "lookup_padded_skypix",
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

from ..utils.image import badMaskDefault, prepare_mask, remove_noise_corr_pedestal
from ..utils.mask import (
    GAIA_TABLE_DTYPE,
    STAR_MASK_RADIUS_FUNCS,
    add_galaxy_ellipse_mask,
    build_gaia_xyr,
    get_gaia_table,
)
from ..wcs import pixel_scale_arcsec

# Slot order of every stacked per-band output (noise correlation, PSF).
band_order = "ugrizy"

# HyperLEDA large-galaxy reference catalog (HTM7 shards of the VizieR
# VII/237 extract with D25 > 0.32 arcmin; columns id = PGC, coord_ra,
# coord_dec, d25_arcmin, logr25, pa), registered in the dp2 butler in
# the RUN collection ``refcats/hyperleda_vii237_d25gt0p32``.
GALAXY_REFCAT_NAME = "hyperleda_vii237_d25gt0p32"
# Degrees added around the quantum's region when looking up galaxy
# shards: a galaxy centred in a trixel that does not overlap the patch
# can still reach it (the mask is capped at galaxyMaskRmaxArcsec).
GALAXY_LOOKUP_PAD_DEG = 0.1


def lookup_padded_skypix(datasetType, registry, quantumDataId, collections):
    """Prerequisite lookup for skypix-sharded refcats with a padded region.

    The default lookup returns only the shards overlapping the patch, so
    a large galaxy centred just outside them is never seen.  This grows
    the patch region's bounding circle by ``GALAXY_LOOKUP_PAD_DEG``.
    """
    from lsst.sphgeom import Angle, Circle

    (name,) = tuple(datasetType.dimensions.names)
    pixelization = registry.dimensions[name].pixelization
    circle = quantumDataId.region.getBoundingCircle()
    padded = Circle(
        circle.getCenter(),
        Angle(circle.getOpeningAngle().asRadians()
              + np.deg2rad(GALAXY_LOOKUP_PAD_DEG)),
    )
    refs = []
    for begin, end in pixelization.envelope(padded):
        for index in range(begin, end):
            ref = registry.findDataset(
                datasetType, {name: index}, collections=collections,
            )
            if ref is not None:
                refs.append(ref)
    return refs


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
    do_noise_corr_estimation = Field[bool](
        doc=(
            "Estimate the per-band noise correlation function. Pass 2 "
            "stitches every band's coadd again purely for this, so "
            "turning it off is a large saving. Safe to disable on the "
            "CELL path: MeasureCellCoaddsPipe has no noiseCorr input "
            "at all (the cell coadds carry their own noise "
            "realisations). NOT safe on the patch path, where "
            "MeasureCoaddsPipe consumes it -- the product is written "
            "as zeros when disabled, which fails loudly in "
            "rotate_noise_corr rather than biasing anything quietly."
        ),
        default=True,
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
    doGalaxyMask = Field[bool](
        doc=(
            "OR the HyperLEDA large-galaxy ellipses into bit 0 of the "
            "mask (needs the galaxy refcat, connection galaxyCatalog, in "
            "the input collections; skipped with a warning when absent)."
        ),
        default=False,
    )
    galaxyMaskScale = Field[float](
        doc="Mask semi-major axis in units of the catalog D25/2.",
        default=2.5,
    )
    galaxyMaskD25Min = Field[float](
        doc="Smallest catalog D25 [arcmin] that is masked.",
        default=0.3,
    )
    galaxyMaskRmaxArcsec = Field[float](
        doc="Cap on the mask semi-major axis [arcsec].",
        default=300.0,
    )
    starMaskType = ChoiceField[str](
        doc=(
            "Name of the GAIA halo-radius model in "
            "xlens.utils.mask.STAR_MASK_RADIUS_FUNCS; radii are in "
            "arcsec and divided by the image pixel scale. 'default' = "
            "75.6/33.6/16.8 arcsec step for mag <= 11/14/20 (450/200/100 "
            "HSC px); 'DP2_nosim' = power-law fit to DP2; 'no_mask' = "
            "flat 2 arcsec for every GAIA star with mag <= 20."
        ),
        allowed={k: k for k in STAR_MASK_RADIUS_FUNCS},
        default="default",
    )
    discontinuityMaskPlanes = ListField[str](
        doc=(
            "Mask planes unioned into the DISCONTINUITY mask -- pixels "
            "holding real data that was not built from the visit set "
            "CoaddPsf assumes (chip gaps, clipped artifacts, rejected "
            "inputs), so the PSF model there is wrong by an unknown "
            "amount. Kept separate from badMaskPlanes because it flags "
            "a model error, not a bad pixel: masking it would cost a "
            "quarter of a typical patch. Downstream it becomes the "
            "per-source ``n_mask_discontinuity`` column instead."
        ),
        default=["INEXACT_PSF"],
    )
    discontinuityMaskBands = ListField[str](
        doc=(
            "Bands unioned into the discontinuity mask. Empty (the "
            "default) means every band the task receives; a subset "
            "(e.g. the detection bands) restricts it."
        ),
        default=[],
    )
    starMaskMagMax = Field[float](
        doc=(
            "Drop GAIA stars fainter than this g magnitude, on top of "
            "the radius model's own cut. None leaves the model's limit "
            "in force (20 for 'default'). The default 18 is tighter "
            "than that on purpose: the model gives every star from 14 "
            "to 20 the same 100 px halo, so the 18-20 stars cost more "
            "masked area than any other magnitude bin while being the "
            "least contaminating."
        ),
        default=18.0,
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

    def _plane_union_mask(self, exposure, band: str) -> NDArray:
        """One band's discontinuity-plane union, as a 0/1 uint8 array.

        Unlike :meth:`_build_mask_band` this applies NO -6 sigma image
        guard: the discontinuity mask is a statement about the PSF
        model, not about outlier pixels. Planes absent from the
        exposure are logged and skipped.
        """
        mask = exposure.mask
        planes = list(self.config.discontinuityMaskPlanes)
        avail = set(mask.getMaskPlaneDict())
        missing = [p for p in planes if p not in avail]
        if missing:
            self.log.info(
                "band %s: discontinuity plane(s) %s absent; skipped",
                band, missing,
            )
        use = [p for p in planes if p in avail]
        if not use:
            return np.zeros(mask.array.shape, dtype=np.uint8)
        return (
            (mask.array & mask.getPlaneBitMask(use)) != 0
        ).astype(np.uint8)

    def _discontinuity_band_selected(self, band: str) -> bool:
        """Whether ``band`` contributes to the discontinuity mask."""
        sel = list(self.config.discontinuityMaskBands)
        return not sel or band in sel

    @staticmethod
    def _merge_mask(
        global_mask: NDArray | None,
        band_mask: NDArray,
    ) -> NDArray:
        """OR one band's mask into the running cross-band union."""
        if global_mask is None:
            return band_mask.astype(np.uint8)
        return (global_mask | band_mask).astype(np.uint8)

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
            pixel_scale=pixel_scale_arcsec(wcs),
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

    def _apply_galaxy_mask(self, *, mask_array, bbox, wcs, galaxy_cats):
        """OR the large-galaxy ellipses into bit 0 of ``mask_array``.

        ``galaxy_cats`` are the deferred handles of the galaxy refcat
        shards (connection ``galaxyCatalog``); None or empty skips.
        """
        if not self.config.doGalaxyMask:
            return 0
        if not galaxy_cats or bbox is None or wcs is None:
            self.log.warning(
                "doGalaxyMask is set but no %s shard was found; "
                "no galaxy mask applied", GALAXY_REFCAT_NAME,
            )
            return 0
        cols = {k: [] for k in ("ra", "dec", "d25", "logr25", "pa")}
        for handle in galaxy_cats:
            cat = handle.get() if hasattr(handle, "get") else handle
            if len(cat) == 0:
                continue
            cols["ra"].append(np.rad2deg(cat["coord_ra"]))
            cols["dec"].append(np.rad2deg(cat["coord_dec"]))
            cols["d25"].append(np.asarray(cat["d25_arcmin"]))
            cols["logr25"].append(np.asarray(cat["logr25"]))
            cols["pa"].append(np.asarray(cat["pa"]))
        if not cols["ra"]:
            return 0
        cols = {k: np.concatenate(v) for k, v in cols.items()}
        before = int(np.count_nonzero(mask_array & 1))
        ngal = add_galaxy_ellipse_mask(
            mask_array,
            ra=cols["ra"], dec=cols["dec"], d25_arcmin=cols["d25"],
            logr25=cols["logr25"], pa=cols["pa"], wcs=wcs, bbox=bbox,
            scale=self.config.galaxyMaskScale,
            d25_min=self.config.galaxyMaskD25Min,
            rmax_arcsec=self.config.galaxyMaskRmaxArcsec,
            value=1,
        )
        added = int(np.count_nonzero(mask_array & 1)) - before
        self.log.info(
            "Galaxy mask: %d of %d catalog galaxies reach the patch; "
            "+%.4f of the pixels masked (scale %.2f x D25/2)",
            ngal, len(cols["ra"]), added / mask_array.size,
            self.config.galaxyMaskScale,
        )
        return ngal

    # 5 sigma of the ROBUST noise level. The previous per-pixel form,
    # ``image**2 < 9 * variance``, defeats itself exactly where it
    # matters: the variance plane is inflated on a bright star's wings,
    # so those pixels pass a cut that is relative to their own variance.
    # On GAMA15H 9370 patch 6,4 a single pixel at 3007 counts survived
    # (variance 3.8e6 there) and drove the i-band correlation peak to
    # 3.79 against a tract median of 0.003, imprinting the star's shape
    # as a 0.73 ellipticity. A scalar threshold from
    # ``estimate_noise_variance`` -- median-based, and the same estimator
    # the measurement path already uses -- has no such blind spot.
    NOISE_CLIP_NSIGMA2 = 25.0

    @classmethod
    def _noise_window(cls, noise_array, variance_array, noise_variance):
        """Pixels of the cut-out that count as noise, as a float32 mask."""
        return (
            (noise_array ** 2.0 < cls.NOISE_CLIP_NSIGMA2 * noise_variance)
            & (~np.isnan(variance_array))
        ).astype(np.float32)

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
        The large-lag pedestal (background structure beyond the stamp,
        see ``remove_noise_corr_pedestal``) is subtracted, so the result
        vanishes at the stamp edge and has a positive spectrum.
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
        out, _ = remove_noise_corr_pedestal(out)
        return out
