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

"""Cross-match a measurement catalog to a spec-z catalog and apply the
basic spec-z calibration selection (per-band mag cuts + trace + confidence).

Functions
---------
sky_match(meas, specz, radius_arcsec=0.75)
    Return (idx_into_specz, sepcheck_bool) for the nearest-neighbour
    cross-match.
append_specz_columns(meas, specz, idx)
    Append `redshift`, `confidence`, `type`, `source` from specz[idx] onto
    meas.
spec_selection(meas, sepcheck)
    Boolean mask = matched & per-band mag cuts & trace cut & confidence cut.
"""
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

# Basic golden selection: per-band mag cuts (mag_zero = 31.4) + trace > 0.1
# + spec-z confidence > 0.82.
from .constants import MAG_ZERO_AB as MAG_ZERO
MAG_CUTS = {
    "lsst_u": 27.5, "lsst_g": 26.5, "lsst_r": 26.0,
    "lsst_i": 25.0, "lsst_z": 25.0, "lsst_y": 25.0,
}
REF_BAND = "lsst_i"  # detection/reference band for the trace cut
TRACE_MIN = 0.1  # (lsst_i_fpfs1_m00 + lsst_i_fpfs1_m20) / lsst_i_fpfs1_m00 > TRACE_MIN
CONF_MIN = 0.82


def sky_match(meas, specz, radius_arcsec: float = 0.75):
    """Cross-match meas (LSST measurement) to specz (spec-z catalog)."""
    c1 = SkyCoord(
        ra=np.array(meas["ra"]) * u.degree,
        dec=np.array(meas["dec"]) * u.degree,
    )
    c2 = SkyCoord(
        ra=np.array(specz["RA"]) * u.degree,
        dec=np.array(specz["DEC"]) * u.degree,
    )
    idx, d2d, _ = c1.match_to_catalog_sky(c2)
    sepcheck = np.asarray(d2d < radius_arcsec * u.arcsec)
    return idx, sepcheck


def append_specz_columns(meas, specz, idx) -> None:
    """In-place append of the four spec-z columns from specz[idx] onto meas."""
    meas["redshift"] = np.array(specz["redshift"])[idx]
    meas["confidence"] = np.array(specz["confidence"])[idx]
    meas["type"] = np.array(specz["type"])[idx]
    meas["source"] = np.array(specz["source"])[idx]


def spec_selection(meas, sepcheck) -> np.ndarray:
    """Sky-match + per-band mag cuts + trace + confidence."""
    sel = np.asarray(sepcheck).copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        for b, cut in MAG_CUTS.items():
            sel &= (
                MAG_ZERO - 2.5 * np.log10(np.asarray(meas[f"{b}_flux_gauss2"]))
            ) < cut
        m00 = np.asarray(meas[f"{REF_BAND}_fpfs1_m00"])
        m20 = np.asarray(meas[f"{REF_BAND}_fpfs1_m20"])
        sel &= ((m00 + m20) / m00) > TRACE_MIN
    sel &= np.asarray(meas["confidence"]) > CONF_MIN
    return sel
