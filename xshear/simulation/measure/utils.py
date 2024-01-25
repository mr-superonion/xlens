#
# LSST Data Management System
# Copyright 20082014 LSST Corpoalphan.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
import lsst.geom as afwGeom
import numpy as np


def get_psf_array(exposure, ngrid):
    """This function returns the PSF model at the center of the exposure."""
    bbox = exposure.getBBox()
    width, height = bbox.getWidth(), bbox.getHeight()
    # Calculate the central point
    centerX, centerY = width // 2, height // 2
    centroid = afwGeom.Point2I(centerX, centerY)
    psf_array = np.zeros((ngrid, ngrid))
    data = exposure.getPsf().computeImage(centroid).getArray()
    dx = data.shape[0]
    if ngrid > dx:
        shift = (ngrid - dx + 1) // 2
        psf_array[shift : shift + dx, shift : shift + dx] = data[:, :]
    else:
        shift = -(ngrid - dx) // 2
        psf_array[:, :] = data[shift : shift + ngrid, shift : shift + ngrid]
    return psf_array
