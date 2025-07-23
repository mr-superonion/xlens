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
    gso, lensed_shift, gamma1, gamma2, kappa
):

    assert kappa >= 0, "kappa must be non-negative"

    shear_res_dict = {
        "gso": gso,
        "lensed_shift": lensed_shift,
        "gamma1": gamma1,
        "gamma2": gamma2,
        "kappa": kappa,
    }
    return shear_res_dict
