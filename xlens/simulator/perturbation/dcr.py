# This file is part of xlens.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Differential chromatic refraction (DCR) perturbation model."""

import galsim


class DcrDistort(object):
    """Apply an astrometric position shift to simulate DCR effects."""

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
