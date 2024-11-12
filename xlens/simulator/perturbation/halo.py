import galsim
from astropy.cosmology import Planck18
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.LensModel.lens_model import LensModel


class ShearHalo(object):
    def __init__(
        self,
        mass,
        conc,
        z_lens,
        ra_lens=0.0,
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
        self.pos_lens = galsim.PositionD(ra_lens, dec_lens)
        return

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
            r = shift - self.pos_lens

            lens_cosmo = LensCosmo(
                z_lens=self.z_lens,
                z_source=redshift,
                cosmo=self.cosmo,
            )
            rs_angle, alpha_rs = lens_cosmo.nfw_physical2angle(
                M=self.mass, c=self.conc
            )
            kwargs = [{"Rs": rs_angle, "alpha_Rs": alpha_rs}]
            f_xx, f_xy, f_yx, f_yy = self.lens.hessian(r.x, r.y, kwargs)
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
                return gso, shift, shift, gamma1, gamma2, kappa

            dra, ddec = self.lens.alpha(r.x, r.y, kwargs)
            gso = gso.lens(g1=g1, g2=g2, mu=mu)
            lensed_shift = shift + galsim.PositionD(dra, ddec)
        return gso, lensed_shift, shift, gamma1, gamma2, kappa
