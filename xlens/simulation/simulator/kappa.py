import galsim


class ShearKappa(object):
    def __init__(
        self,
        gamma1,
        gamma2,
        kappa,
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
        """
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.kappa = kappa
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
        g1 = self.gamma1 / (1 - self.kappa)
        g2 = self.gamma2 / (1 - self.kappa)
        mu = 1.0 / ((1 - self.kappa) ** 2 - self.gamma1**2 - self.gamma2**2)
        gso = gso.lens(g1=g1, g2=g2, mu=mu)
        shift = shift.lens(g1=g1, g2=g2, mu=mu)
        return gso, shift
