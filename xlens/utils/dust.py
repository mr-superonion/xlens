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
"""Galactic dust extinction per band, ``A_band = <A_lambda/A_V>_band * R_V *
E(B-V)``.

Follows ``dp2utils.dust`` (LSSTDESC/dp2utils PR #1, S. Saha and S. Mau):
the extinction curve is averaged over each passband with photon weighting,
``<A_lambda/A_V> = int T A_lambda/A_V lambda dlambda / int T lambda dlambda``,
and ``E(B-V)`` is read from a dust map and rescaled by 0.86 (Schlafly &
Finkbeiner 2011) because both maps are in SFD units.

Extinction curves: any ``R_V``-parametrised curve of ``dust_extinction``
(``F99`` Fitzpatrick 1999, default; ``G23`` Gordon et al. 2023; also
``CCM89``, ``O94``, ``F04``, ``F19``...).  Dust maps: ``csfd`` (corrected
SFD, Chiang 2023, default) or ``sfd`` (Schlegel et al. 1998) through
``dustmaps``.  Passbands: ``lsst`` (DP2 ``standard_passband`` from the
butler when available, else the bundled curve), ``des``, ``euclid``,
``hsc`` from ``xlens/data/filters`` (SVO Filter Profile Service), or
user-supplied ``{band: (wavelength_nm, throughput)}``.

    dust = DustCorrector("lsst")               # F99, R_V = 3.1, CSFD
    Ax = dust(ra, dec)                         # {band: A_band [mag]}
    mag_g_corrected = mag_g - Ax["g"]
    DustCorrector("euclid", model="G23")(ra, dec, "vis")

Heavy imports (``dust_extinction``, ``dustmaps``, the butler) happen on
first use, so importing ``xlens.utils`` does not need them.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "DustCorrector",
    "DUST_MAP_DIR",
    "PASSBANDS",
    "band_AxAv",
    "load_passbands",
]

# DESC data registry dataset 145 (dp2utils DUST_MAP_ID): <dir>/sfd/SFD_dust_4096_{ngp,sgp}.fits
# and <dir>/csfd/csfd_ebv.fits.  Override with the ``dust_map_dir`` argument or
# ``$XLENS_DUST_MAP_DIR``.
DUST_MAP_DIR = "/global/cfs/cdirs/lsst/utilities/data-registry/lsst_desc_working/user/s-sayan/dust_maps"
# Both maps are calibrated in SFD units (Schlafly & Finkbeiner 2011, 0.86 rescaling).
EBV_SCALE = 0.86

FILTER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "filters")
PASSBANDS = {
    "lsst": ("u", "g", "r", "i", "z", "y"),
    "des": ("g", "r", "i", "z", "y"),
    "euclid": ("vis", "y", "j", "h"),
    "hsc": ("g", "r", "i", "z", "y"),
}
BUTLER_CONFIG = "dp2"
BUTLER_COLLECTIONS = ("LSSTCam/runs/DRP/DP2",)


def load_passbands(survey: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Bundled ``{band: (wavelength_nm, throughput)}`` of ``survey`` (SVO curves)."""
    if survey not in PASSBANDS:
        raise ValueError(f"survey must be one of {sorted(PASSBANDS)}, got {survey!r}")
    out = {}
    for band in PASSBANDS[survey]:
        lam, thr = np.loadtxt(os.path.join(FILTER_DIR, f"{survey}_{band}.txt"), unpack=True)
        out[band] = (lam, thr)
    return out


def _butler_passbands(bands, butler_config, butler_collections):
    from lsst.daf.butler import Butler

    butler = Butler(butler_config, collections=list(butler_collections))
    out = {}
    for band in bands:
        table = butler.get("standard_passband", band=band)
        lam = np.asarray(table["wavelength"].to("nm").value, dtype=np.float64)
        thr = np.asarray(table["throughput"], dtype=np.float64)
        out[band] = (lam, thr)
    return out


def _extinction_model(model, Rv):
    from dust_extinction import parameter_averages

    if isinstance(model, str):
        try:
            cls = getattr(parameter_averages, model)
        except AttributeError:
            names = sorted(n for n in dir(parameter_averages) if n[:1].isupper() and n[:2] != "Ba")
            raise ValueError(f"unknown extinction curve {model!r}; dust_extinction offers {names}") from None
        return cls(Rv=Rv)
    return model  # an instantiated dust_extinction model


def band_AxAv(model, wavelength_nm, throughput) -> float:
    """Photon-weighted ``<A_lambda/A_V>`` of one passband; raises if the
    passband leaves the curve's validity range."""
    import astropy.units as u

    lam = np.asarray(wavelength_nm, dtype=np.float64)
    thr = np.asarray(throughput, dtype=np.float64)
    lo, hi = model.x_range
    x = 1.0 / (lam * 1e-3)  # 1/micron
    inside = (x >= lo) & (x <= hi)
    live = thr > 1e-4 * thr.max()
    if not np.all(inside[live]):
        raise ValueError(
            f"passband {lam[live].min():.0f}-{lam[live].max():.0f} nm is outside the "
            f"{type(model).__name__} range {1e3 / hi:.0f}-{1e3 / lo:.0f} nm"
        )
    axav = np.zeros_like(lam)  # zero-throughput tails outside the curve contribute nothing
    axav[inside] = model(lam[inside] * u.nm)
    return float(np.trapezoid(thr * axav * lam, lam) / np.trapezoid(thr * lam, lam))


class DustCorrector:
    """Per-band Galactic extinction ``A_band(ra, dec)`` [mag].

    Parameters
    ----------
    survey : {"lsst", "des", "euclid", "hsc"}
        Which bundled passband set to use (ignored when ``passbands`` is given).
    Rv : float
        Total-to-selective extinction ratio of the curve (3.1).
    model : str or dust_extinction model
        ``"F99"`` (default), ``"G23"``, or any other ``R_V`` curve of
        ``dust_extinction.parameter_averages``.
    dustmap : {"csfd", "sfd"} or a ``dustmaps`` query object
        Map giving ``E(B-V)`` in SFD units.
    dust_map_dir : str, optional
        Directory holding ``csfd/`` and ``sfd/`` (default ``$XLENS_DUST_MAP_DIR``
        or the DESC data-registry copy).
    ebv_scale : float
        Rescaling of the map's ``E(B-V)`` (0.86 for both SFD-calibrated maps).
    passbands : mapping, optional
        ``{band: (wavelength_nm, throughput)}`` overriding the bundled set.
    use_butler : bool, optional
        For ``survey="lsst"``: read the DP2 ``standard_passband`` curves from
        the butler (default: try, fall back to the bundled curves).
    """

    def __init__(
        self,
        survey: str = "lsst",
        Rv: float = 3.1,
        model="F99",
        dustmap="csfd",
        dust_map_dir: str | None = None,
        ebv_scale: float = EBV_SCALE,
        passbands: Mapping[str, tuple] | None = None,
        use_butler: bool | None = None,
        butler_config: str = BUTLER_CONFIG,
        butler_collections: Sequence[str] = BUTLER_COLLECTIONS,
    ):
        self.survey = survey
        self.Rv = float(Rv)
        self.ebv_scale = float(ebv_scale)
        self.model = _extinction_model(model, self.Rv)
        self.dustmap = self._make_dustmap(dustmap, dust_map_dir)
        if passbands is None:
            passbands = self._survey_passbands(survey, use_butler, butler_config, butler_collections)
        self.passbands = {b: (np.asarray(l, dtype=np.float64), np.asarray(t, dtype=np.float64))
                          for b, (l, t) in passbands.items()}
        self.bands = tuple(self.passbands)
        self.band_AxAv = {b: band_AxAv(self.model, *self.passbands[b]) for b in self.bands}

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _make_dustmap(dustmap, dust_map_dir):
        if not isinstance(dustmap, str):
            return dustmap
        root = dust_map_dir or os.environ.get("XLENS_DUST_MAP_DIR") or DUST_MAP_DIR
        if dustmap == "csfd":
            from dustmaps.csfd import CSFDQuery

            return CSFDQuery(map_fname=os.path.join(root, "csfd", "csfd_ebv.fits"),
                             mask_fname=os.path.join(root, "csfd", "mask.fits"))
        if dustmap == "sfd":
            from dustmaps.sfd import SFDQuery

            return SFDQuery(map_dir=os.path.join(root, "sfd"))
        raise ValueError(f"dustmap must be 'csfd', 'sfd' or a dustmaps query, got {dustmap!r}")

    @staticmethod
    def _survey_passbands(survey, use_butler, butler_config, butler_collections):
        if survey == "lsst" and use_butler is not False:
            try:
                return _butler_passbands(PASSBANDS["lsst"], butler_config, butler_collections)
            except Exception:
                if use_butler:
                    raise
        return load_passbands(survey)

    # ------------------------------------------------------------------ queries
    def get_ebv(self, ra, dec):
        """Rescaled ``E(B-V)`` [mag] at ``ra``, ``dec`` [deg]."""
        from astropy.coordinates import SkyCoord

        coords = SkyCoord(np.asarray(ra, dtype=np.float64), np.asarray(dec, dtype=np.float64), unit="deg")
        return np.asarray(self.dustmap(coords), dtype=np.float64) * self.ebv_scale

    def get_Av(self, ra, dec):
        """``A_V = R_V E(B-V)`` [mag]."""
        return self.get_ebv(ra, dec) * self.Rv

    def get_Ax(self, band, ra, dec):
        """Extinction [mag] in one band."""
        return self.get_Av(ra, dec) * self.band_AxAv[band]

    def get_Ax_all(self, ra, dec, return_type="dict"):
        """Extinction [mag] in every band: a dict, or an ``(N, n_band)`` array
        in ``self.bands`` order."""
        Av = self.get_Av(ra, dec)
        Ax = {b: Av * self.band_AxAv[b] for b in self.bands}
        if return_type == "dict":
            return Ax
        if return_type == "array":
            return np.vstack([Ax[b] for b in self.bands]).T
        raise ValueError("return_type must be 'dict' or 'array'")

    def __call__(self, ra, dec, bands="all", return_type="dict"):
        """``dust(ra, dec)`` -> all bands; ``dust(ra, dec, "g")`` -> array;
        ``dust(ra, dec, ["g", "r"])`` -> dict of the subset."""
        if bands == "all":
            return self.get_Ax_all(ra, dec, return_type=return_type)
        if isinstance(bands, str):
            return self.get_Ax(bands, ra, dec)
        if isinstance(bands, (list, tuple)):
            Av = self.get_Av(ra, dec)
            return {b: Av * self.band_AxAv[b] for b in bands}
        raise ValueError("bands must be 'all', a band name or a list of band names")
