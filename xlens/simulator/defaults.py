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

"""Default survey parameters for LSST and HSC simulations.

Module-level dictionaries provide magnitude zero-points, PSF FWHM values,
and per-band noise variances keyed by survey name.
"""

mag_zero_defaults = {
    "lsst": 30.0,
    "hsc": 27.0,
}

psf_fwhm_defaults = {
    "u": {
        "lsst": 0.8,
    },
    "g": {
        "lsst": 0.8,
        "hsc": 0.798,
    },
    "r": {
        "lsst": 0.8,
        "hsc": 0.749,
    },
    "i": {
        "lsst": 0.8,
        "hsc": 0.617,
    },
    "z": {
        "lsst": 0.8,
        "hsc": 0.697,
    },
    "y": {
        "lsst": 0.8,
        "hsc": 0.688,
    },
}

sys_npix = 49

noise_variance_defaults = {
    "u": {
        "lsst": 0.4517,
    },
    "g": {
        "lsst": 0.099,
        "hsc": 1.4e-3,
    },
    "r": {
        "lsst": 0.138,
        "hsc": 2.9e-3,
    },
    "i": {
        "lsst": 0.354,
        "hsc": 4.7e-3,
    },
    "z": {
        "lsst": 1.334,
        "hsc": 19e-3,
    },
    "y": {
        "lsst": 1.41,
        "hsc": 85e-3,
    },
}
