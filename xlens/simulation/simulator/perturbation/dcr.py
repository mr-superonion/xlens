import galsim


class DcrDistort(object):
    def __init__(self, distort_func):
        """Shear distortion from an astrometry error

        Args:
        distort_func (function): a function to distort galaxy position

        Example:
        >>> def distort_func(x, y):
                # shift by one arcsec in ra and dec, respectively
        ...     return x + 1, y + 1
        ...
        >>> DcrDistort(distort_func)
        """

        self.distort_func = distort_func

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
        dra, ddec = self.distort_func(shift.x, shift.y)

        # TODO: Ideally, this should be a function of color
        shift = shift + galsim.PositionD(dra, ddec)
        return gso, shift
