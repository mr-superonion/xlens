import numpy as np
from astropy.cosmology import Planck18
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

from .utils import _get_shear_res_dict


class ShearHalo(object):
    def __init__(
        self,
        *,
        mass,
        conc,
        z_lens,
        halo_profile="NFW",
        cosmo=None,
        no_kappa=False,
        gmax=0.95,
    ):
        """Shear distortion from halo

        Args:
        mass (float):               mass of the halo [M_sun]
        conc (float):               concerntration
        z_lens (float):             lens redshift
        halo_profile (str):         halo profile name
        cosmo (astropy.cosmology):  cosmology object
        no_kappa (bool):            if True, turn off kappa field
        """

        if cosmo is None:
            cosmo = Planck18
        self.cosmo = cosmo
        self.mass = mass
        self.z_lens = z_lens
        self.conc = conc
        self.no_kappa = no_kappa
        self.lens = LensModel(lens_model_list=[halo_profile])
        self.lens_solver = LensEquationSolver(lensModel=self.lens)
        self.gmax = gmax
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
        if src["redshift"] <= self.z_lens:
            # foreground or at lens plane: no lensing (return identity)
            return _get_shear_res_dict(
                lensed_x=src["dx"],
                lensed_y=src["dy"],
                gamma1=0.0,
                gamma2=0.0,
                kappa=0.0,
                has_finite_shear=True,
            )

        lens_cosmo = LensCosmo(
            z_lens=self.z_lens,
            z_source=src["redshift"],
            cosmo=self.cosmo,
        )
        rs_angle, alpha_rs = lens_cosmo.nfw_physical2angle(
            M=self.mass, c=self.conc
        )
        kwargs = [{"Rs": rs_angle, "alpha_Rs": alpha_rs}]

        lensed_x, lensed_y = self.lens_solver.image_position_from_source(
            src["dx"],
            src["dy"],
            kwargs,
            min_distance=0.2,  # finding solutions as close as min_distance
            search_window=50,  # largest distance to find a solution for
        )

        # if lenstronomy cannot find a solution
        # do not shift
        if len(lensed_x) == 0 or len(lensed_y) == 0:
            lensed_x, lensed_y = src["dx"], src["dy"]
        else:
            lensed_x, lensed_y = lensed_x[0], lensed_y[0]

        f_xx, f_xy, f_yx, f_yy = self.lens.hessian(
            lensed_x, lensed_y, kwargs
        )
        gamma1 = 1.0 / 2 * (f_xx - f_yy)
        gamma2 = f_xy
        kappa = 0.0 if self.no_kappa else 0.5 * (f_xx + f_yy)

        if kappa < 0.0 or kappa >= 1.0:
            return _get_shear_res_dict(
                lensed_x=lensed_x,
                lensed_y=lensed_y,
                gamma1=0.0,
                gamma2=0.0,
                kappa=0.0,
                has_finite_shear=False,
            )

        denom = 1.0 - kappa
        g1 = gamma1 / denom
        g2 = gamma2 / denom
        gmag = np.hypot(g1, g2)
        if gmag > self.gmax:
            s = self.gmax / gmag
            # Keep kappa fixed; scale gamma so that g -> s * g
            gamma1 *= s
            gamma2 *= s

        return _get_shear_res_dict(
            lensed_x=lensed_x,
            lensed_y=lensed_y,
            gamma1=gamma1,
            gamma2=gamma2,
            kappa=kappa,
            has_finite_shear=True,
        )
