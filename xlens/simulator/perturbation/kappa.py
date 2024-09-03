class ShearKappa(object):
    def __init__(self, mode, g_dist="g1", shear_value=0.02, kappa=0.05):
        """Shear distortion from halo

        Args:
        shear_value (float)     the amplitude of shear
        kappa (float)           kappa
        """
        self.kappa = kappa

        if mode == "0000":
            gv = shear_value * -1.0
        elif mode == "1111":
            gv = shear_value
        else:
            raise ValueError("mode not supported")
        if g_dist == "g1":
            self.gamma1 = gv
            self.gamma2 = 0.0
        elif g_dist == "g2":
            self.gamma1 = 0.0
            self.gamma2 = gv
        else:
            raise ValueError("g_dist not supported")
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
