"""Intrinsic alignment perturbation implemented via BATSim IaTransform."""

from __future__ import annotations

from typing import Any

import galsim

from .base import BasePerturbation
from .utils import _get_shear_res_dict

try:  # pragma: no cover - optional dependency
    import batsim
except ImportError:  # pragma: no cover - optional dependency
    batsim = None


class IaTransformDistort(BasePerturbation):
    """Apply the BATSim intrinsic alignment transform when drawing galaxies."""

    def __init__(
        self,
        *,
        amplitude: float,
        beta: float,
        phi: float,
        clip_radius: float,
        stamp_size: int,
    ) -> None:
        if batsim is None:
            raise ImportError(
                "batsim is required to use IaTransformDistort but is not installed."
            )

        if stamp_size <= 0:
            raise ValueError("stamp_size must be a positive integer")

        self.amplitude = amplitude
        self.beta = beta
        self.phi = phi
        self.clip_radius = clip_radius
        self.stamp_size = int(stamp_size)

    def distort_galaxy(self, src: Any) -> dict[str, Any]:
        # The intrinsic alignment transform modifies galaxy shapes but not
        # their sky positions.  We therefore leave the positional information
        # unchanged and report zero shear/kappa for bookkeeping purposes.
        return _get_shear_res_dict(
            lensed_x=src["dx"],
            lensed_y=src["dy"],
            gamma1=0.0,
            gamma2=0.0,
            kappa=0.0,
            has_finite_shear=True,
        )

    # No need to override ``apply_to_galaxy`` because the transform is applied
    # during the custom drawing step below.

    def draw_stamp(
        self,
        *,
        gal_obj: galsim.GSObject,
        psf_obj: galsim.GSObject,
        image_pos: galsim.PositionD,
        draw_method: str,
        pixel_scale: float,
        local_wcs,
        nn_trunc,
        source_row,
        entry,
    ) -> galsim.Image:
        if local_wcs is not None:
            raise ValueError(
                "IaTransformDistort does not support drawing with a local WCS."
            )

        hlr = float(source_row["hlr"])
        stamp_npix = int(nn_trunc) if nn_trunc is not None else self.stamp_size
        if stamp_npix <= 0:
            raise ValueError("Requested stamp size must be positive")

        transform = batsim.IaTransform(
            scale=pixel_scale,
            hlr=hlr,
            A=self.amplitude,
            beta=self.beta,
            phi=self.phi,
            clip_radius=self.clip_radius,
        )

        gal_img = batsim.simulate_galaxy(
            ngrid=stamp_npix,
            pix_scale=pixel_scale,
            gal_obj=gal_obj,
            transform_obj=transform,
            psf_obj=psf_obj,
            draw_method=draw_method,
        )

        stamp = galsim.ImageF(gal_img, scale=pixel_scale)
        stamp.setCenter(image_pos.x, image_pos.y)
        return stamp

