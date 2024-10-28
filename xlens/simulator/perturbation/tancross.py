import galsim
import numpy as np


class ShearTanCross(object):
    def __init__(self, mode, g_dist="gt", shear_value=0.02):
        """Shear distortion from halo

        Args:
        shear_value (float)     the amplitude of shear
        """

        if mode == 0:
            self.gv = shear_value * -1.0
        elif mode == 1:
            self.gv = shear_value
        else:
            raise ValueError("mode not supported")

        if g_dist not in ["gt", "gx"]:
            raise ValueError("g_dist not supported")

        self.g_dist = g_dist
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
        theta = np.arctan2(shift.y, shift.x)
        if self.g_dist == "gt":
            g1 = self.gv * np.cos(2.0 * theta)
            g2 = self.gv * np.sin(2.0 * theta)
        else:
            g1 = self.gv * np.sin(2.0 * theta)
            g2 = -self.gv * np.cos(2.0 * theta)

        shear = galsim.Shear(g1=g1, g2=g2)
        gso = gso.shear(shear)
        shift = shift.shear(shear)
        return gso, shift
