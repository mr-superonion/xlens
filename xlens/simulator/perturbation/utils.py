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
    return ''.join(reversed(digits)).zfill(n_digits)


def _get_shear_res_dict(
    lensed_x, lensed_y, gamma1, gamma2, kappa, has_finite_shear
):
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
