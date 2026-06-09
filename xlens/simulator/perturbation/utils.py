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

"""Internal helpers shared by perturbation models."""


def _ternary(n: int, n_digits: int) -> str:
    """Convert integer `n` to zero-padded base-3 string with `n_digits`
    length."""
    if n == 0:
        return "0".zfill(n_digits)
    digits = []
    while n:
        n, r = divmod(n, 3)
        digits.append(str(r))
    return "".join(reversed(digits)).zfill(n_digits)


def _get_shear_res_dict(lensed_x, lensed_y, gamma1, gamma2, kappa, has_finite_shear):
    """Build the standard result dict returned by ``distort_galaxy`` methods.

    Parameters
    ----------
    lensed_x, lensed_y : float
        Post-lensing arcsecond positions on the tangent plane.
    gamma1, gamma2 : float
        Shear components at the galaxy position.
    kappa : float
        Convergence at the galaxy position.
    has_finite_shear : bool
        Whether the shear is physically valid (e.g. ``|g| < 1``).
    """
    shear_res_dict = {
        "dx": lensed_x,
        "dy": lensed_y,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "kappa": kappa,
        "has_finite_shear": has_finite_shear,
    }
    return shear_res_dict
