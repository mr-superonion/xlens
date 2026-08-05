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

"""Shared numeric constants for xlens."""

import anacal

# AB magnitude zeropoint for fluxes measured in nanojansky:
#   m_AB = MAG_ZERO_AB - 2.5 * log10(F_nJy)
#   F_nJy = 10**((MAG_ZERO_AB - m)/2.5)
# The measurement stage (xlens.utils.image.prepare_data*) rescales every
# FPFS moment/flux onto this fixed zeropoint, so the output catalog is
# independent of each coadd's native photometric zeropoint and
# downstream consumers convert flux->mag with this single constant
# (no per-catalog mag_zero needed).
MAG_ZERO_AB = 31.4

# Soft-bias regulariser in the FPFS ellipticity denominator,
#   e = m22c / (m00 + FPFS_C0),
# on the same MAG_ZERO_AB flux scale as the moments it is added to, so it is
# used as-is with no per-coadd rescaling.
#
# READ FROM ANACAL rather than duplicated here, so the two repos cannot drift:
# this IS ``anacal.fpfs.FpfsConfig.c0``, which AnaCal defines at its
# ``THRESHOLD_REF_MAG_ZERO`` (= MAG_ZERO_AB).  Every xlens consumer -- the
# Task path (``processor.anacal``), the FPFS path (``processor.fpfs``), the
# per-band combination (``processor.merge``) and the standalone recombination
# (``utils.nxg.calibrate_shapes``) -- must use this one value, otherwise they
# produce different ellipticities for the same catalog.
FPFS_C0 = anacal.fpfs.FpfsConfig().c0
