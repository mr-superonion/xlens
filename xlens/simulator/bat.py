"""Utilities for applying BATSim transforms."""
import numpy as np
from typing import Any
import galsim
try:  # pragma: no cover - optional dependency
    import batsim
except ImportError:  # pragma: no cover - optional dependency
    batsim = None


def draw_ia(
    amplitude,
    beta,
    phi,
    clip_radius,
    stamp_size,
    gal_obj: galsim.GSObject,
    psf_obj: galsim.GSObject,
    image_pos: galsim.PositionD,
    draw_method: str,
    pixel_scale: float,
    entry,
) -> galsim.Image:
    """Draw a postage stamp using the BATSim intrinsic alignment transform."""

    if batsim is None:
        raise ImportError("Cannot import batsim")

    hlr = float(entry["hlr"])
    transform_obj = batsim.IaTransform(
        scale=pixel_scale,
        hlr=hlr,
        A=amplitude,
        beta=beta,
        phi=phi,
        clip_radius=clip_radius,
    )
    x_d = image_pos.x
    y_d = image_pos.y
    x_i = np.round(x_d).astype(int)
    y_i = np.round(y_d).astype(int)

    gal_img = batsim.simulate_galaxy(
        ngrid=stamp_size,
        pix_scale=pixel_scale,
        gal_obj=gal_obj,
        transform_obj=transform_obj,
        psf_obj=psf_obj,
        draw_method=draw_method,
        delta_image_x=(x_d - x_i),
        delta_image_y=(y_d - y_i),
    )
    stamp = galsim.ImageF(gal_img, scale=pixel_scale)
    stamp.setCenter(x_i, y_i)
    return stamp


def draw_flexion(
    stamp_size,
    gal_obj: galsim.GSObject,
    psf_obj: galsim.GSObject,
    image_pos: galsim.PositionD,
    draw_method: str,
    pixel_scale: float,
    entry,
) -> galsim.Image:
    """Draw a postage stamp using the BATSim flexion transform."""
    pass


__all__ = ["draw_ia"]
