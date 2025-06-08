import galsim
from astropy.cosmology import Planck18
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from descwl_shear_sims.constants import WORLD_ORIGIN
from descwl_shear_sims.shear import _get_shear_res_dict
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u



class ShearHalo(object):
    def __init__(
        self,
        mass,
        conc,
        z_lens,
        ra_lens=200.0,
        dec_lens=0.0,
        halo_profile="NFW",
        cosmo=None,
        no_kappa=False,
    ):
        """Shear distortion from halo

        Args:
        mass (float):               mass of the halo [M_sun]
        conc (float):               concerntration
        z_lens (float):             lens redshift
        ra_lens (float):            ra of halo position [arcsec]
        dec_lens (float):           dec of halo position [arcsec]
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
        self.pos_lens = galsim.PositionD(ra_lens * 3600., dec_lens * 3600.)
        self.lens_eqn_solver = LensEquationSolver(lensModel=self.lens)
        return

    # def get_offset(ra_arcsec, dec_arcsec, ra0_arcsec, dec0_arcsec):
    #     c0 = SkyCoord(ra=ra0_arcsec * u.arcsec, dec=dec0_arcsec * u.arcsec)
    #     c1 = SkyCoord(ra=ra_arcsec * u.arcsec, dec=dec_arcsec * u.arcsec)
    #     dlon, dlat = c0.spherical_offsets_to(c1)
    #     delta_ra_astro = dlon.to(u.arcsec).value
    #     delta_dec_astro = dlat.to(u.arcsec).value

    #     return delta_ra_astro, delta_dec_astro

    def distort_galaxy(self, gso, shift, redshift):
        """This function distorts the galaxy's shape and position
        Parameters
        ---------
        gso (galsim object):        galsim galaxy
        shift (galsim.PositionD):   position of the galaxy
        redshift (float):           redshift of galaxy

        Returns
        ---------
        gso, shift:
            distorted galaxy object and shift
        """

        
        if redshift > self.z_lens:
            r = shift

            lens_cosmo = LensCosmo(
                z_lens=self.z_lens,
                z_source=redshift,
                cosmo=self.cosmo,
            )
            rs_angle, alpha_rs = lens_cosmo.nfw_physical2angle(
                M=self.mass, c=self.conc
            )
            kwargs = [{"Rs": rs_angle, "alpha_Rs": alpha_rs}]

            lensed_x, lensed_y = self.lens_eqn_solver.image_position_from_source(
                r.x,
                r.y,
                kwargs,
                min_distance=0.2,  # chance of finding solutions as close as min_distance away from each other
                search_window=50,  # largest distance to find a solution for
            )

            # if lenstronomy cannot find a solution
            # do not shift
            if len(lensed_x) == 0 or len(lensed_y) == 0:
                lensed_x, lensed_y = shift.x, shift.y
            else:
                lensed_x, lensed_y = lensed_x[0], lensed_y[0]
                
            f_xx, f_xy, f_yx, f_yy = self.lens.hessian(
                lensed_ra, lensed_dec, kwargs
            )
            gamma1 = 1.0 / 2 * (f_xx - f_yy)
            gamma2 = f_xy
            if self.no_kappa:
                kappa = 0.0
            else:
                kappa = 1.0 / 2 * (f_xx + f_yy)

            g1 = gamma1 / (1 - kappa)
            g2 = gamma2 / (1 - kappa)
            mu = 1.0 / ((1 - kappa) ** 2 - gamma1**2 - gamma2**2)

            if g1**2.0 + g2**2.0 > 0.95:
                return _get_shear_res_dict(gso, shift, gamma1, gamma2, kappa)
                return gso, shift, shift, gamma1, gamma2, kappa

            gso = gso.lens(g1=g1, g2=g2, mu=mu)


            lensed_shift = galsim.PositionD(lensed_x, lensed_y)

            return _get_shear_res_dict(gso, lensed_shift, gamma1, gamma2, kappa)