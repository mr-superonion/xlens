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

"""Survey-prefixed band-name helpers.

Band names in xlens catalogs are ``{survey}_{band}`` (e.g. ``lsst_g``,
``euclid_vis``, ``hsc_i``) so bands from different surveys are unambiguous in a
merged catalog. The *physical* band (the single-letter/short band label carried
by the butler ``band`` dimension, on-disk coadds, and filter files) is the part
after the first underscore; the *survey* is the part before it. Split on the
FIRST underscore so multi-token physical bands survive (``euclid_nisp_j`` ->
survey ``euclid``, band ``nisp_j``).
"""


def prefixed(survey: str, band: str) -> str:
    """Compose a survey-prefixed band name, e.g.
    ``('lsst','g') -> 'lsst_g'``.
    """
    return f"{survey}_{band}"


def survey_of(name: str) -> str:
    """Survey prefix of a ``{survey}_{band}`` name (before the first ``_``)."""
    return name.split("_", 1)[0]


def physical_band(name: str) -> str:
    """Physical band of a ``{survey}_{band}`` name (after the first ``_``).

    A bare band with no prefix is returned unchanged.
    """
    parts = name.split("_", 1)
    return parts[1] if len(parts) == 2 else parts[0]
