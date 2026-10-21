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

import numpy as np

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

# Source photon-noise coefficient c for LSST, so that the per-pixel variance
# is noise_variance_defaults[band]["lsst"] + c * flux, with flux the pixel
# value of the noiseless image at mag_zero_defaults["lsst"] = 30.
# Only LSST is supported.
source_noise_coeff_defaults = {
    "u": {
        "lsst": 1.63e-02,
    },
    "g": {
        "lsst": 2.06e-03,
    },
    "r": {
        "lsst": 1.04e-03,
    },
    "i": {
        "lsst": 1.41e-03,
    },
    "z": {
        "lsst": 2.31e-03,
    },
    "y": {
        "lsst": 4.95e-03,
    },
}


def get_variance_array(image_array, band, survey_name="lsst"):
    """Per-pixel variance map including source photon noise (LSST only).

    Args:
    image_array (ndarray): noiseless image; each pixel value is the flux
        at mag_zero_defaults["lsst"] = 30
    band (str): band name
    survey_name (str): survey name; only "lsst" is supported

    Returns:
    variance_array (ndarray): noise_variance_defaults + c * image_array,
        with the same shape as image_array
    """
    if survey_name != "lsst":
        raise ValueError(
            "source photon noise is only supported for lsst, got %s"
            % survey_name
        )
    image_array = np.asarray(image_array)
    return (
        noise_variance_defaults[band]["lsst"]
        + source_noise_coeff_defaults[band]["lsst"] * image_array
    )


# Objects centred more than this many pixels outside an image are not drawn.
SIM_INCLUSION_PADDING = 200  # pixels
