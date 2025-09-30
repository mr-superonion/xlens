import galsim

# ---- MoG coefficient tables from Hogg & Lang (2012), Table 1 ----
# Each tuple is (a_m, sqrt(v_m));
M_EXP_DEFAULT = 6
M_DEV_DEFAULT = 10

_MOG_COEFFS = {
    # Exponential profile (n=1). Choices: M=4,6,8
    ("exp", 4): [
        (0.09733, 0.12068),
        (1.12804, 0.32730),
        (4.99846, 0.68542),
        (5.63632, 1.28089),
    ],
    ("exp", 6): [
        (0.00735, 0.05072),
        (0.09481, 0.13756),
        (0.63572, 0.28781),
        (2.60077, 0.53195),
        (5.42848, 0.91209),
        (3.16445, 1.50157),
    ],
    ("exp", 8): [
        (0.00077, 0.02394),
        (0.01017, 0.06492),
        (0.07313, 0.13581),
        (0.37184, 0.25095),
        (1.39736, 0.42942),
        (3.56100, 0.69675),
        (4.74338, 1.08885),
        (1.78684, 1.67302),
    ],

    # de Vaucouleurs profile (n=4). Choices: M=6,8,10
    ("dev", 6): [
        (0.01308, 0.00263),
        (0.12425, 0.01202),
        (0.63551, 0.04031),
        (2.22560, 0.12128),
        (5.63989, 0.36229),
        (9.81523, 1.23604),
    ],
    ("dev", 8): [
        (0.00262, 0.00113),
        (0.02500, 0.00475),
        (0.13413, 0.01462),
        (0.51326, 0.03930),
        (1.52005, 0.09926),
        (3.56204, 0.24699),
        (6.44845, 0.63883),
        (8.10105, 1.92560),
    ],
    ("dev", 10): [
        (0.00139, 0.00087),
        (0.00941, 0.00296),
        (0.04441, 0.00792),
        (0.16162, 0.01902),
        (0.48121, 0.04289),
        (1.20357, 0.09351),
        (2.54182, 0.20168),
        (4.46441, 0.44126),
        (6.22820, 1.01833),
        (6.15393, 2.74555),
    ],
}


def _mog_gal(profile: str, M: int, flux: float, hlr: float):
    """
    Build circular Gaussian components for a given profile ('exp' or 'dev'),
    mixture size M, target *total* flux, and half-light radius [arcsec]
    """
    coeffs = _MOG_COEFFS[(profile, M)]
    sum_a = sum(a for a, _rv in coeffs)  # table's total dimensionless flux
    scale = flux / sum_a                 # normalize to requested total flux

    components = []
    for a, rv in coeffs:
        sigma = rv * hlr                  # rv= sqrt(v_m) in HLR units
        # Flux of each circular Gaussian after normalization:
        comp = galsim.Gaussian(flux=a * scale, sigma=sigma)
        components.append(comp)
    return galsim.Add(components)


def Exponential(
    flux: float, half_light_radius: float, M: int = M_EXP_DEFAULT
):
    return _mog_gal("exp", M, flux, half_light_radius)


def DeVaucouleurs(
    flux: float, half_light_radius: float, M: int = M_DEV_DEFAULT
):
    return _mog_gal("dev", M, flux, half_light_radius)

__all__ = ["Exponential", "DeVaucouleurs"]
