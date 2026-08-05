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

"""Utilities for applying BATSim transforms."""

import galsim
import numpy as np

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
    """Draw a postage stamp using the BATSim intrinsic alignment transform.

    Parameters
    ----------
    amplitude : float
        IA amplitude parameter ``A`` passed to ``batsim.IaTransform``.
    beta : float
        IA beta parameter.
    phi : float
        IA orientation angle (radians).
    clip_radius : float
        Clip radius in units of half-light radii.
    stamp_size : int
        Side length (pixels) of the output stamp.
    gal_obj : galsim.GSObject
        Galaxy profile before PSF convolution.
    psf_obj : galsim.GSObject
        PSF profile.
    image_pos : galsim.PositionD
        Absolute pixel position of the galaxy on the coadd.
    draw_method : str
        GalSim draw method (e.g. ``"auto"``, ``"no_pixel"``).
    pixel_scale : float
        Pixel scale in arcseconds.
    entry : numpy structured scalar
        Galaxy catalog row; must contain ``"hlr"`` field.

    Returns
    -------
    galsim.Image
        Rendered postage stamp with bounds set on the coadd grid.
    """

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
    x_i = int(np.round(x_d))
    y_i = int(np.round(y_d))

    # BATSim samples the profile on a grid whose origin is the stamp centre, so
    # the sub-pixel residual of the requested position is applied to the galaxy
    # itself.  ``use_true_center=False`` puts that origin on integer pixel
    # ``stamp_size // 2``, which is where ``setCenter`` below places (x_i, y_i)
    # for an even-sized stamp.
    gal_obj = gal_obj.shift(
        (x_d - x_i) * pixel_scale,
        (y_d - y_i) * pixel_scale,
    )

    gal_img = batsim.simulate_galaxy(
        ngrid=stamp_size,
        pix_scale=pixel_scale,
        gal_obj=gal_obj,
        transform_obj=transform_obj,
        psf_obj=psf_obj,
        draw_method=draw_method,
        use_true_center=False,
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
    """Draw a postage stamp using the BATSim flexion transform.

    .. note::
       Not yet implemented.
    """
    pass


__all__ = ["draw_ia"]
