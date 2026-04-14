"""Constant tangential/cross shear perturbation applied per redshift slice."""

import galsim
import numpy as np

from .utils import _get_shear_res_dict, _ternary


class ShearTanCross(object):
    """Constant tan/cross shear in each redshift slice.

    The shear pattern across redshift bins is encoded as a ternary integer
    (``mode``), where each base-3 digit maps to ``-shear_value``,
    ``+shear_value``, or ``0.0``.

    Parameters
    ----------
    mode : int
        Ternary-encoded shear assignment (see class docstring).
    g_dist : {'gt', 'gx'}
        Which shear component receives the test signal.
    shear_value : float
        Absolute shear amplitude per bin.
    kappa_value : float
        Constant convergence applied to all bins.
    """

    def __init__(
        self, mode, g_dist="gt", shear_value=0.02, kappa_value=0.0
    ):
        assert isinstance(mode, int), "mode must be an integer"
        assert isinstance(g_dist, str), "g_dist must be a string"
        self.mode = mode
        self.g_dist = g_dist
        
        if mode == 0:
            self.shear_value = -shear_value
        elif mode == 1:
            self.shear_value = shear_value
        else:
            self.shear_value = 0

        # 0 means no kappa value is provided
        self.kappa = kappa_value
        return

    def distort_galaxy(self, src):
        """This function distorts the galaxy's shape and position
        Parameters
        ---------
        src (np.array):        row of structured array

        Returns
        ---------
            distorted galaxy position and lensing distortions
        """

        theta = np.arctan2(src["dy"], src["dx"])

        if self.g_dist == "gt":
            gamma1 = self.shear_value * np.cos(2.0 * theta)
            gamma2 = self.shear_value * np.sin(2.0 * theta)
        else:
            gamma1 = self.shear_value * np.sin(2.0 * theta)
            gamma2 = -self.shear_value * np.cos(2.0 * theta)

        g1 = gamma1 / (1 - self.kappa)
        g2 = gamma2 / (1 - self.kappa)
        mu = 1.0 / ((1 - self.kappa) ** 2 - gamma1**2 - gamma2**2)

        mat = galsim.Shear(g1=g1, g2=g2).getMatrix() * np.sqrt(mu)
        lensed_x = src["dx"] * mat[0, 0] + src["dy"] * mat[0, 1]
        lensed_y = src["dx"] * mat[1, 0] + src["dy"] * mat[1, 1]
        return _get_shear_res_dict(
            lensed_x=lensed_x,
            lensed_y=lensed_y,
            gamma1=gamma1,
            gamma2=gamma2,
            kappa=self.kappa,
            has_finite_shear=True,
        )
