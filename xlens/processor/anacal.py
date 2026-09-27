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

from typing import Any, Sequence

import anacal
import astropy
import numpy as np
from lsst.afw.geom import SkyWcs
from lsst.afw.image import ExposureF
from lsst.pex.config import Config, Field, FieldValidationError
from lsst.pipe.base import Task
from numpy.typing import NDArray

from .. import utils
from ..utils.constants import FPFS_C0
from ..wcs import pixel_to_sky


class AnacalConfig(Config):
    npix = Field[int](
        doc="number of pixels in stamp",
        default=64,
    )
    bound = Field[int](
        doc="Sources to be removed if too close to boundary [pixel]",
        default=35,
    )
    psf_dilation_max = Field[float](
        doc=(
            "Cap on the ellipticity dilation d of the re-smoothing kernel. "
            "The kernel is not set by hand: for every cell its width is "
            "metadetect's default ``fitgauss`` target, sigma = sqrt(T d / "
            "2), from the adaptive moments (e1, e2, T) of that cell's PSF "
            "(anacal.ngmix.fit_psf_gauss) with d = 1 + 2 (sqrt(1 + |e|) - "
            "1) capped at this value (metadetect: 1.1). See "
            "AnacalTask.get_sigma_arcsec."
        ),
        default=1.1,
    )
    snr_min = Field[float](
        doc="snr min for detection",
        default=5.0,
    )
    num_epochs = Field[int](
        doc=(
            "Maximum number of Gaussian-fit epochs. 0 exports the moment "
            "initialisation (production)."
        ),
        default=0,
    )
    conv_tol = Field[float](
        doc=(
            "Smooth convergence gate of the Gaussian model fit: the step "
            "of an epoch is scaled by a smoothstep of the chi2 decrease "
            "the previous step achieved, relative to the source's own "
            "chi2 scale, exactly 0 below conv_tol (the source stops) and "
            "1 above 10 x conv_tol. Default 0: gate off, every source "
            "takes num_epochs epochs. On real coadds the gate's own "
            "derivative term is heavy-tailed on sources passing through "
            "the ramp, so leave it off for production."
        ),
        default=0.0,
    )
    force_size = Field[bool](
        doc="Whether forcing the size and shape of galaxies",
        default=False,
    )
    force_center = Field[bool](
        doc="Whether forcing the size and shape of galaxies",
        default=True,
    )
    prior_sigma_T = Field[float](
        doc=(
            "Width [arcsec^2] of the Gaussian prior towards 0 on the "
            "intrinsic size T = mxx + myy of the model fit (T = 2 a^2 for "
            "a round source of semi-axis a); 0 disables it."
        ),
        default=0.07,
    )
    prior_sigma_x = Field[float](
        doc=(
            "Width [arcsec] of the Gaussian prior on the fitted centre "
            "towards the detection position; 0 disables it."
        ),
        default=0.05,
    )
    prior_sigma_e = Field[float](
        doc=(
            "Width of the Gaussian prior towards 0 on the intrinsic "
            "ellipticity (e1, e2) = (mxx - myy, 2 mxy) / T of the model; "
            "0 disables it."
        ),
        default=0.3,
    )
    do_noise_bias_correction = Field[bool](
        doc="whether to doulbe the noise for noise bias correction",
        default=True,
    )
    do_fpfs = Field[bool](
        doc="whether to do FPFS measurement",
        default=True,
    )
    noiseId = Field[int](
        doc="Noise realization id",
        default=0,
    )
    rotId = Field[int](
        doc="rotation id",
        default=0,
    )
    psf_model_type = Field[str](
        doc="type of psf model (choose from object, cell, patch)",
        default="patch",
    )

    def validate(self):
        super().validate()
        if not self.psf_dilation_max >= 1.0:
            raise FieldValidationError(
                self.__class__.psf_dilation_max,
                self,
                "psf_dilation_max must be >= 1",
            )
        if self.noiseId < 0:
            raise FieldValidationError(
                self.__class__.noiseId,
                self,
                "We require noiseId >=0",
            )
        if self.rotId >= utils.random.num_rot:
            raise FieldValidationError(
                self.__class__.rotId,
                self,
                "rotId needs to be smaller than 2",
            )

    def setDefaults(self):
        super().setDefaults()


class AnacalTask(Task):
    """Measure Fpfs FPFS observables"""

    _DefaultName = "AnacalTask"
    ConfigClass = AnacalConfig

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        assert isinstance(self.config, AnacalConfig)
        self.config_kwargs = {
            "snr_peak_min": self.config.snr_min,
            "stamp_size": self.config.npix,
            "image_bound": self.config.bound,
            "num_epochs": self.config.num_epochs,
            "force_size": self.config.force_size,
            "force_center": self.config.force_center,
            "conv_tol": self.config.conv_tol,
        }
        return

    def get_sigma_arcsec(
        self,
        psf_array: NDArray,
        pixel_scale: float,
        noise_variance: float | Sequence[float] | None = None,
    ) -> float:
        """Width [arcsec] of the re-smoothing kernel for a PSF.

        metadetect's default ``fitgauss`` reconvolution target (without
        metacal's separate 1 + 2 step shear-step dilation):
        sigma = sqrt(T d / 2), with (e1, e2, T) the adaptive moments of
        the PSF, measured by AnaCal's model fit
        (``anacal.ngmix.fit_psf_gauss``), and d the ellipticity dilation
        capped at ``psf_dilation_max``.  For a (nband, npix, npix) stack
        e1, e2 and T are averaged over the bands with inverse-variance
        weights before d is taken, as metadetect averages its shear
        bands.
        """
        assert isinstance(self.config, AnacalConfig)
        psf = np.asarray(psf_array, dtype=np.float64)
        if psf.ndim == 2:
            psf = psf[None]
        if noise_variance is None:
            wgt = np.ones(len(psf))
        else:
            wgt = 1.0 / np.broadcast_to(
                np.asarray(noise_variance, dtype=np.float64), (len(psf),)
            )
        e1 = e2 = T = 0.0
        for w, p in zip(wgt / np.sum(wgt), psf):
            r = anacal.ngmix.fit_psf_gauss(np.ascontiguousarray(p), pixel_scale)
            if not (r.converged and r.T > 0):
                raise RuntimeError("Gaussian fit to the PSF failed")
            e1 += w * r.e1
            e2 += w * r.e2
            T += w * r.T
        d = anacal.ngmix.ellip_dilation(e1, e2, self.config.psf_dilation_max)
        return float(np.sqrt(T * d / 2.0))

    def make_prior(self):
        """The Gaussian priors of the model fit from the configuration:
        size T = mxx + myy [arcsec^2], centre offset [arcsec], and the
        intrinsic ellipticity (mxx - myy, 2 mxy) / T.  A width of 0 leaves
        that prior off."""
        prior = anacal.ngmix.modelPrior()
        if self.config.prior_sigma_T > 0:
            prior.set_sigma_T(anacal.math.qnumber(self.config.prior_sigma_T))
        if self.config.prior_sigma_x > 0:
            prior.set_sigma_x(anacal.math.qnumber(self.config.prior_sigma_x))
        if self.config.prior_sigma_e > 0:
            prior.set_sigma_e(anacal.math.qnumber(self.config.prior_sigma_e))
        return prior

    def run(
        self,
        *,
        pixel_scale: float,
        mag_zero: float,
        # One value for a plain image, one per band for a (nband, ny, nx)
        # stack -- anacal takes either.
        noise_variance: float | Sequence[float],
        gal_array: NDArray,
        # (npix, npix), or (nband, npix, npix), centred on pixel
        # (npix // 2, npix // 2), 0-based: AnaCal shifts that pixel to the
        # FFT origin, so a PSF centred elsewhere shifts the deconvolved
        # image, and with it every detection and measured position.
        psf_array: NDArray,
        mask_array: NDArray,
        noise_array: NDArray | None,
        begin_x: int = 0,
        begin_y: int = 0,
        wcs: SkyWcs | None = None,
        skyMap=None,
        tractInfo=None,
        patchInfo=None,
        detection: NDArray | None,
        cells,
        n_mask_base_max: float | None = None,
        **kwargs,
    ):
        assert isinstance(self.config, AnacalConfig)

        # These flux-scale thresholds are defined at AnaCal's
        # THRESHOLD_REF_MAG_ZERO, which equals MAG_ZERO_AB. The image is
        # already normalized to MAG_ZERO_AB upstream (prepare_data), so
        # ``mag_zero`` here is MAG_ZERO_AB and the Task's threshold scaling
        # is exactly 1.0 -- the values below are the ones actually used.
        #
        # ``omega_v`` is the single neighbour-difference parameter (AnaCal
        # uses it as both centre and width, so the smooth step vanishes
        # exactly at v = 0, the strict-local-maximum boundary).  This value is
        # the "v.003" configuration, which gave the best selection-response
        # conditioning in the blended-simulation scan (values rounded to
        # three decimals).
        if detection is not None:
            det = detection.copy()
            det["x1"] = det["x1"] - begin_x * pixel_scale
            det["x2"] = det["x2"] - begin_y * pixel_scale
            det["x1_det"] = det["x1_det"] - begin_x * pixel_scale
            det["x2_det"] = det["x2_det"] - begin_y * pixel_scale
        else:
            det = None

        # The re-smoothing kernel follows each cell's PSF
        # (get_sigma_arcsec), so every cell is its own anacal Task.  An
        # external catalog is split by the same ownership rule
        # process_image applies internally.
        owner = None
        if det is not None and len(cells) > 1:
            owner = anacal.task.assign_cell_ids(det, list(cells))
        prior = self.make_prior()
        parts = []
        for cell in cells:
            cell_det = det
            if owner is not None:
                cell_det = det[owner == cell.index]
                if len(cell_det) == 0:
                    continue
            cell_psf = getattr(cell, "psf_array", None)
            if cell_psf is None or np.size(cell_psf) == 0:
                cell_psf = psf_array
            task = anacal.task.Task(
                scale=pixel_scale,
                sigma_arcsec=self.get_sigma_arcsec(
                    cell_psf, pixel_scale, noise_variance
                ),
                omega_f=0.218,
                omega_v=0.011,
                fpfs_c0=FPFS_C0,
                mag_zero=mag_zero,
                prior=prior,
                **self.config_kwargs,
            )
            parts.append(task.process_image(
                gal_array,
                psf_array,
                variance=noise_variance,
                cell_list=[cell],
                detection=cell_det,
                noise_array=noise_array,
                mask_array=mask_array,
                do_fpfs=self.config.do_fpfs,
                n_mask_base_max=n_mask_base_max,
            ))
        if parts:
            catalog = np.concatenate(parts)
        else:
            catalog = anacal.table.make_catalog_empty(
                np.zeros(0), np.zeros(0)
            )
        catalog["x1"] = catalog["x1"] + begin_x * pixel_scale
        catalog["x2"] = catalog["x2"] + begin_y * pixel_scale
        catalog["x1_det"] = catalog["x1_det"] + begin_x * pixel_scale
        catalog["x2_det"] = catalog["x2_det"] + begin_y * pixel_scale
        if wcs is not None:
            # catalog x1/x2 are already global (parent) pixels here, since
            # begin_x/begin_y were added back above -> no XY0 offset needed.
            ra, dec = pixel_to_sky(
                wcs,
                catalog["x1"] / pixel_scale,
                catalog["x2"] / pixel_scale,
            )
            catalog["ra"] = ra
            catalog["dec"] = dec
        return catalog

    def prepare_data(
        self,
        *,
        exposure: ExposureF,
        seed: int,
        band: str | None,
        survey: str | None = None,
        noise_corr: NDArray | None = None,
        skyMap=None,
        tract: int = 0,
        patch: int = 0,
        star_cat: NDArray | None = None,
        psf_array: NDArray | None = None,
        mask_array: NDArray | None = None,
        noise_array: NDArray | None = None,
        detection: astropy.table.Table | None = None,
        cells: list | None = None,
        num_workers: int = 1,
        **kwargs,
    ):
        """Prepares the data from LSST exposure
        Args:
        exposure (ExposureF):   LSST exposure
        seed (int):  random seed
        noise_corr (NDArray):  image noise correlation function (None)
        tractInfo:  tract information
        patchInfo:  patch information

        Returns:
            (dict)
        """
        assert isinstance(self.config, AnacalConfig)
        pixel_scale = float(exposure.wcs.getPixelScale().asArcseconds())
        if cells is None:
            cells = utils.image.get_cells(
                lsst_psf=exposure.getPsf(),
                lsst_bbox=exposure.getBBox(),
                pixel_scale=pixel_scale,
                npix=self.config.npix,
                psf_array=psf_array,
                num_workers=num_workers,
            )
        data = utils.image.prepare_data(
            exposure=exposure,
            seed=seed,
            noiseId=self.config.noiseId,
            rotId=self.config.rotId,
            npix=self.config.npix,
            noise_corr=noise_corr,
            do_noise_bias_correction=self.config.do_noise_bias_correction,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            star_cat=star_cat,
            psf_array=psf_array,
            mask_array=mask_array,
            noise_array=noise_array,
            detection=detection,
            band=band,
            survey=survey,
            cells=cells,
        )
        if band is None:
            data["base_column_name"] = None
        elif survey is not None:
            data["base_column_name"] = f"{survey}_{band}_"
        else:
            data["base_column_name"] = band + "_"
        if self.config.psf_model_type == "object":
            data["psf_object"] = utils.image.make_object_psf(
                exposure.getPsf(),
                npix=self.config.npix,
                lsst_bbox=exposure.getBBox(),
            )
        else:
            data["psf_object"] = None
        return data

    def prepare_data_multiband(
        self,
        *,
        exposures: dict,
        bands: Sequence[str],
        seed: int,
        survey: str | None = None,
        noise_corrs: dict | None = None,
        skyMap=None,
        tract: int = 0,
        patch: int = 0,
        star_cat: NDArray | None = None,
        mask_array: NDArray | None = None,
        detection: astropy.table.Table | None = None,
        cells: list | None = None,
        num_workers: int = 1,
        **kwargs,
    ):
        """Prepare a stack of bands as one anacal detection input.

        Same as :meth:`prepare_data` but for several bands at once: the
        image, PSF and noise arrays gain a leading band axis and
        ``noise_variance`` becomes one value per band.  anacal deconvolves
        each band's own PSF before averaging them, so the bands do not need
        to be PSF-matched here -- only aligned on the same pixel grid.

        Args:
        exposures (dict):  band -> LSST exposure
        bands (Sequence[str]):  bands to coadd, in the order they are stacked
        seed (int):  random seed
        noise_corrs (dict):  band -> noise correlation function (None)
        """
        assert isinstance(self.config, AnacalConfig)
        bands = list(bands)
        missing = [b for b in bands if b not in exposures]
        if missing:
            raise KeyError(f"no exposure for band(s) {missing}")

        exps = [exposures[b] for b in bands]
        pixel_scale = float(exps[0].wcs.getPixelScale().asArcseconds())
        if cells is None:
            cells = utils.image.get_cells_multiband(
                lsst_psfs=[e.getPsf() for e in exps],
                lsst_bbox=exps[0].getBBox(),
                pixel_scale=pixel_scale,
                npix=self.config.npix,
                num_workers=num_workers,
            )
        data = utils.image.prepare_data_multiband(
            bands=bands,
            exposures=exposures,
            seed=seed,
            noiseId=self.config.noiseId,
            rotId=self.config.rotId,
            npix=self.config.npix,
            noise_corrs=noise_corrs,
            do_noise_bias_correction=self.config.do_noise_bias_correction,
            skyMap=skyMap,
            tract=tract,
            patch=patch,
            star_cat=star_cat,
            mask_array=mask_array,
            detection=detection,
            cells=cells,
            survey=survey,
        )
        # The detection image belongs to no single band, so it carries no
        # band prefix, and per-object PSFs -- defined against one
        # exposure -- do not apply.
        data["base_column_name"] = None
        data["psf_object"] = None
        return data
