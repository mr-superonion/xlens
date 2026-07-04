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

mag_zero_defaults = {
    "lsst": 30.0,
    "hsc": 27.0,
    "euclid": 30.0,
}

# Native Euclid MER ``MAGZERO`` for each Euclid band (informational; the
# simulation
# renders at the common mag_zero_defaults["euclid"] = 30.0 instead).
# euclid_mag_zero_native = {
#     "vis": 24.6,
#     "nisp_y": 29.8,
#     "nisp_j": 30.0,
#     "nisp_h": 29.9,
# }

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
    "vis": {
        "euclid": 0.203,
    },
    "nisp_y": {
        "euclid": 0.478,
    },
    "nisp_j": {
        "euclid": 0.506,
    },
    "nisp_h": {
        "euclid": 0.545,
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
    # Euclid bands: 3-sigma-clipped per-pixel sky variance,
    # rescaled to the common ZP=30 (see module docstring).
    "vis": {
        "euclid": 0.135,
    },
    "nisp_y": {
        "euclid": 0.374,
    },
    "nisp_j": {
        "euclid": 0.293,
    },
    "nisp_h": {
        "euclid": 0.314,
    },
}
