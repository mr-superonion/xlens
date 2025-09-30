"""Common utilities for perturbation models used in the simulator.

The simulator previously assumed that all perturbations could be described as
simple shear fields applied through :meth:`galsim.GSObject.lens`.  Introducing
new perturbations (e.g. intrinsic alignment transforms) requires a slightly
more flexible interface so that each perturbation can decide how the
``galsim`` galaxy object should be rendered.  This module provides a small base
class with default behaviour matching the original shear-based implementation.
"""

from __future__ import annotations

from typing import Any

import galsim


class BasePerturbation:
    """Base class for perturbations applied during galaxy rendering.

    Subclasses are expected to override :meth:`distort_galaxy` to update the
    source catalog entries (position offsets and shear/kappa values).  Most
    perturbations can use the default :meth:`draw_stamp` implementation which
    applies the stored shear using :meth:`galsim.GSObject.lens` and then draws
    the convolved image.  More complex perturbations can override
    :meth:`draw_stamp` to provide bespoke rendering logic while still relying
    on :meth:`distort_galaxy` for catalog updates.
    """

    def distort_galaxy(self, src: Any) -> dict[str, Any]:  # pragma: no cover
        """Return the distorted position and shear information for a galaxy.

        Parameters
        ----------
        src
            Structured array row containing the original catalog information
            for the galaxy being lensed.

        Returns
        -------
        dict
            Dictionary with keys ``dx``, ``dy``, ``gamma1``, ``gamma2``,
            ``kappa`` and ``has_finite_shear`` describing the updated galaxy
            properties.  See :func:`_get_shear_res_dict` for the exact
            structure.
        """

        raise NotImplementedError

    # NOTE: The ``entry`` argument is unused in the default implementation but
    # is provided so that subclasses have access to the full catalog record if
    # needed for bespoke drawing logic.
    def apply_to_galaxy(
        self, gal_obj: galsim.GSObject, *, source_row, entry
    ) -> galsim.GSObject:
        """Apply the stored shear values to the ``galsim`` object.

        Parameters
        ----------
        gal_obj
            The intrinsic (pre-lensing) ``galsim`` galaxy.
        source_row
            Structured array row describing the distorted galaxy.  The row is
            expected to contain the fields ``gamma1``, ``gamma2`` and
            ``kappa``.
        entry
            Full catalog record for the galaxy.  Included for subclasses that
            may need additional metadata.
        """

        gamma1 = source_row["gamma1"]
        gamma2 = source_row["gamma2"]
        kappa = source_row["kappa"]

        denom = 1.0 - kappa
        g1 = gamma1 / denom
        g2 = gamma2 / denom
        mu = 1.0 / ((1.0 - kappa) ** 2 - gamma1**2 - gamma2**2)
        return gal_obj.lens(g1=g1, g2=g2, mu=mu)

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
        """Draw a postage stamp for the distorted galaxy.

        Subclasses can override this method when they need to control the
        drawing process directly.  The default implementation applies the
        shear parameters stored in ``source_row`` and draws the convolved image
        using GalSim's standard routines.
        """

        distorted = self.apply_to_galaxy(
            gal_obj, source_row=source_row, entry=entry,
        )
        convolved_object = galsim.Convolve([distorted, psf_obj])
        if local_wcs is not None:
            stamp = convolved_object.drawImage(
                center=image_pos,
                wcs=local_wcs,
                method=draw_method,
                nx=nn_trunc,
                ny=nn_trunc,
            )
        else:
            stamp = convolved_object.drawImage(
                center=image_pos,
                wcs=None,
                method=draw_method,
                scale=pixel_scale,
                nx=nn_trunc,
                ny=nn_trunc,
            )
        return stamp


class IdentityPerturbation(BasePerturbation):
    """Fallback perturbation that leaves galaxies unchanged."""

    def distort_galaxy(self, src):
        dtype = getattr(src, "dtype", None)
        names = getattr(dtype, "names", None)
        has_gamma = names is not None and "gamma1" in names
        has_kappa = names is not None and "kappa" in names
        gamma1 = float(src["gamma1"]) if has_gamma else 0.0
        gamma2 = float(src["gamma2"]) if has_gamma else 0.0
        kappa = float(src["kappa"]) if has_kappa else 0.0
        return {
            "dx": src["dx"],
            "dy": src["dy"],
            "gamma1": gamma1,
            "gamma2": gamma2,
            "kappa": kappa,
            "has_finite_shear": True,
        }

