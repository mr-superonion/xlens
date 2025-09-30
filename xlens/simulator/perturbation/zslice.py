import galsim
import numpy as np

from .base import BasePerturbation
from .utils import _get_shear_res_dict, _ternary


class ShearRedshift(BasePerturbation):
    """
    Constant shear in each redshift slice
    """
    def __init__(
        self, z_bounds, mode, g_dist="g1", shear_value=0.02, kappa_value=0.0
    ):
        assert isinstance(mode, int), "mode must be an integer"
        self.nz_bins = int(len(z_bounds) - 1)
        # nz_bins is the number of redshift bins
        # note that there are three options in each redshift bin
        # 0: g=-0.02; 1: g=0.02; 2: g=0.00
        # for example, number of redshift bins is 4, (z_bounds = [0., 0.5, 1.0,
        # 1.5, 2.0]) if mode = 7 which in ternary is "0021" --- meaning that
        # the shear is (-0.02, -0.02, 0.00, 0.02) in each bin, respectively.
        self.code = _ternary(int(mode), self.nz_bins)
        assert 0 <= int(mode) < 3 ** self.nz_bins, "mode code is too large"
        # maybe we need it to be more flexible in the future
        # but now we keep the linear spacing
        self.z_bounds = z_bounds
        self.g_dist = g_dist
        self.shear_value = shear_value
        self.shear_list = self.determine_shear_list(code=self.code)

        # 0 means no kappa value is provided
        self.kappa = kappa_value
        return

    def determine_shear_list(self, code):
        values = [-self.shear_value, self.shear_value, 0.0]
        shear_list = [values[int(i)] for i in code]
        return shear_list

    def _get_zshear(self, redshift):
        bin_num = np.searchsorted(a=self.z_bounds, v=redshift, side="left") - 1
        nz = len(self.z_bounds) - 1
        if bin_num < nz and bin_num >= 0:
            # if the redshift is within the boundaries of lower and uper limits
            # we add shear
            shear = self.shear_list[bin_num]
        else:
            # if not, we set shear to 0 and leave the galaxy image undistorted
            shear = 0.0
        return shear

    def get_shear(self, redshift):
        shear = self._get_zshear(redshift=redshift)
        if self.g_dist == 'g1':
            gamma1, gamma2 = (shear, 0.)
        elif self.g_dist == 'g2':
            gamma1, gamma2 = (0., shear)
        else:
            raise ValueError("g_dist must be either 'g1' or 'g2'")

        g1 = gamma1 / (1 - self.kappa)
        g2 = gamma2 / (1 - self.kappa)
        mu = 1.0 / ((1 - self.kappa) ** 2 - gamma1**2 - gamma2**2)
        return g1, g2, mu, gamma1, gamma2

    def distort_galaxy(self, src):
        """This function distorts the galaxy's shape and position
        Parameters
        ---------
        src (np.array):        row of structured array

        Returns
        ---------
            distorted galaxy position and lensing distortions
        """
        distortion = self.get_shear(src["redshift"])

        g1, g2, mu, gamma1, gamma2 = distortion
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
