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

import os
from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize_scalar

from ..utils.bands import physical_band, survey_of
from ..utils.constants import MAG_ZERO_AB
from .utils import _resolve_cut_name, flux_to_mag

NUM_Z_GRIDS = 501
Z_MIN = 0.0
Z_MAX = 5.0
Z_GRIDS = np.linspace(Z_MIN, Z_MAX, NUM_Z_GRIDS)
PROBS = np.array([0.025, 0.16, 0.5, 0.84, 0.975], dtype=float)
INV1PZ = 1.0 / (1.0 + Z_GRIDS)  # precompute once
GAMMA_RISK = 0.15


def risk(
    zx: float,
    p_norm: np.ndarray,
    z_grids: np.ndarray = Z_GRIDS,
    inv1pz: np.ndarray = INV1PZ,
) -> float:
    # loss = 1 - 1/(1 + (( (zx-z)/(1+z) )/gamma)^2)
    dz = (zx - z_grids) * inv1pz
    t = dz / GAMMA_RISK
    t2 = t * t
    loss_vec = t2 / (1.0 + t2)  # same as 1 - 1/(1+t2)
    return float(simpson(y=p_norm * loss_vec, x=z_grids))


def get_point_estimate(
    p,
    z_grids: np.ndarray = Z_GRIDS,
    inv1pz: np.ndarray = INV1PZ,
):
    total = float(np.sum(p))
    if (not np.isfinite(total)) or total <= 0.0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    # normalized pdf for risk
    p_norm = p / total
    # mode (peak on grid)
    # percentiles (CDF from discrete sum)
    cdf = np.cumsum(p, dtype=float) / total
    zqs = np.interp(PROBS, cdf, z_grids)
    # minimize risk
    res = minimize_scalar(
        lambda zx: risk(zx, p_norm, z_grids, inv1pz),
        bounds=(z_grids[0], z_grids[-1]),
        method="bounded",
    )
    zmode = z_grids[int(np.argmax(p))]
    z025, z160, z500, z840, z975 = zqs
    zbest = res.x
    return (zmode, z025, z160, z500, z840, z975, zbest)


def get_point_estimates_from_pdfs(
    pdfs: np.ndarray,
    z_grids: np.ndarray = Z_GRIDS,
):
    """
    Compute point estimates from PDF samples on a redshift grid.

    ``z_grids`` is the redshift grid the ``pdfs`` columns live on; it must have
    the same length as ``pdfs.shape[1]``. Defaults to the module ``Z_GRIDS``.

    Returns dict of arrays (shape (N,)):
      - zmode : z_grid[argmax p(z)]
      - z025, z160, z500, z840, z975 : CDF percentiles at [0.025, 0.16, 0.50,
        0.84, 0.975]
      - zbest : argmin_zx ∫ p_norm(z) * loss(zx,z) dz   (bounded to [z_grid[0],
        z_grid[-1]])
    """
    if pdfs.ndim != 2:
        raise ValueError(f"pdfs must be 2D (N, M); got {pdfs.shape}")
    if pdfs.shape[1] != len(z_grids):
        raise ValueError(
            f"pdfs has {pdfs.shape[1]} grid points but z_grids has {len(z_grids)}"
        )
    inv1pz = 1.0 / (1.0 + z_grids)

    N = pdfs.shape[0]
    zbest = np.full(N, np.nan, dtype=float)
    zmode = np.full(N, np.nan, dtype=float)
    z025 = np.full(N, np.nan, dtype=float)
    z160 = np.full(N, np.nan, dtype=float)
    z500 = np.full(N, np.nan, dtype=float)
    z840 = np.full(N, np.nan, dtype=float)
    z975 = np.full(N, np.nan, dtype=float)
    for i, p in enumerate(pdfs):
        zmode[i], z025[i], z160[i], z500[i], z840[i], z975[i], zbest[i] = get_point_estimate(
            p, z_grids, inv1pz
        )
    return {
        "zmode": zmode,
        "z025": z025,
        "z160": z160,
        "z500": z500,
        "z840": z840,
        "z975": z975,
        "zbest": zbest,
    }


def get_color(
    src: np.ndarray,
    *,
    bands: tuple[str, ...] = ("lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"),
    ref_band: str = "lsst_i",
    mag_zero: float = MAG_ZERO_AB,
    comp: int = 1,
    dg: float = 0.0,
    flux_name: str = "gauss2",
    include_mag_err: bool = False,
    extinction: np.ndarray | None = None,
) -> np.ndarray:
    """
    Returns
    -------
    np.ndarray
        If include_mag_err=False: shape (N, 1 + (len(bands)-1))
          [ref_mag, (b0-b1), (b1-b2), ...]
        If include_mag_err=True: shape (N, 1 + 2*(len(bands)-1))
          [ref_mag, (b0-b1), err01, (b1-b2), err12, ...]
    """
    fn = _resolve_cut_name(flux_name)
    n = src.shape[0]

    mags: list[np.ndarray] = []
    merrs: list[np.ndarray] | None = [] if include_mag_err else None
    # Compute mag (and optionally mag_err) per band, in the same order as
    # `bands`
    for b in bands:
        flux = src[f"{b}_flux{fn}"] + dg * src[f"{b}_dflux{fn}_dg{comp}"]
        ferr = src[f"{b}_flux{fn}_err"] if merrs is not None else None
        a_ext = None if extinction is None else extinction[f"a_{b}"]
        mag, mag_err = flux_to_mag(flux, mag_zero, flux_err=ferr, a_ext=a_ext)
        mags.append(mag)
        if merrs is not None:
            merrs.append(mag_err)

    nb = len(bands) - 1
    ncols = 1 + (2 * nb if include_mag_err else nb)
    feat = np.empty((n, ncols), dtype=np.float32)
    try:
        ref_idx = bands.index(ref_band)
    except ValueError:
        raise ValueError(f"ref_band={ref_band!r} not found in bands={bands!r}")
    feat[:, 0] = mags[ref_idx]

    j = 1
    if include_mag_err:
        assert merrs is not None
        for i in range(nb):
            np.subtract(mags[i], mags[i + 1], out=feat[:, j])
            j += 1
            feat[:, j] = np.hypot(merrs[i], merrs[i + 1])
            j += 1
    else:
        for i in range(nb):
            np.subtract(mags[i], mags[i + 1], out=feat[:, j])
            j += 1
    return feat


# ------------------------
# Z-Estimator Implementations
# ------------------------


class zEstimator(ABC):
    @abstractmethod
    def get_z(
        self,
        src: np.ndarray,
        *,
        mag_zero: float = MAG_ZERO_AB,
        flux_name: str = "gauss2",
        bands: tuple[str, ...] = ("lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"),
        ref_band: str = "lsst_i",
        comp: int = 1,
        dg: float = 0.0,
        flux_name2: str | None = None,
        flux_name3: str | None = None,
        extinction: np.ndarray | None = None,
        **kwargs,
    ) -> dict:
        """Method to get redshift point estimates"""

    def get_zsel(
        self,
        src: np.ndarray,
        *,
        mag_zero: float = MAG_ZERO_AB,
        flux_name: str = "gauss2",
        bands: tuple[str, ...] = ("lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"),
        ref_band: str = "lsst_i",
        comp: int = 1,
        dg: float = 0.0,
        z_point_name: str = "zmode",
        flux_name2: str | None = None,
        flux_name3: str | None = None,
        extinction: np.ndarray | None = None,
        **kwargs,
    ):
        zout = self.get_z(
            src=src,
            mag_zero=mag_zero,
            flux_name=flux_name,
            bands=bands,
            ref_band=ref_band,
            comp=comp,
            dg=dg,
            flux_name2=flux_name2,
            flux_name3=flux_name3,
            extinction=extinction,
            **kwargs,
        )
        zpoint = zout[z_point_name]
        width95 = zout["z975"] - zout["z025"]
        return zpoint, width95


class flexzboostEstimator(zEstimator):
    """
    Wraps a FlexZBoost-like predictor object with a uniform `get_z` API.
    """

    def __init__(
        self,
        pz_obj,
        z_max: float = Z_MAX,
        nzbins: int = NUM_Z_GRIDS,
    ):
        self.pz_obj = pz_obj
        self.pz_obj.model.models.n_jobs = 1
        self.nzbins = int(nzbins)
        self.z_grids = np.linspace(Z_MIN, float(z_max), self.nzbins)

    def get_z(
        self,
        src: np.ndarray,
        *,
        mag_zero: float = MAG_ZERO_AB,
        flux_name: str = "gauss2",
        bands: tuple[str, ...] = ("lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"),
        ref_band: str = "lsst_i",
        comp: int = 1,
        dg: float = 0.0,
        flux_name2: str | None = None,
        flux_name3: str | None = None,
        include_mag_err: bool = False,
        return_pdfs: bool = False,
        extinction: np.ndarray | None = None,
        **kwargs,
    ) -> dict:
        colors = get_color(
            src,
            mag_zero=mag_zero,
            comp=comp,
            dg=dg,
            flux_name=flux_name,
            bands=bands,
            ref_band=ref_band,
            include_mag_err=include_mag_err,
            extinction=extinction,
        )
        pdfs, _ = self.pz_obj.predict(colors, n_grid=self.nzbins)
        points = get_point_estimates_from_pdfs(pdfs, z_grids=self.z_grids)
        if return_pdfs:
            points["pdfs"] = pdfs
        return points


def load_bpz_templates(
    data_path: str,
    bands: tuple[str, ...] | list[str],
    filtersets: dict[str, str] | None = None,
    spectra_name: str = "cosmossedswdust136.list",
):
    """Load BPZ template fluxes on Z_GRIDS for the given survey-prefixed bands.

    ``filtersets`` maps a survey to its on-disk BPZ filter set (default
    ``{"lsst":"comcam","euclid":"euclid"}``). The AB template files are
    named ``{SED}.{filterset}_{physical_band}.AB`` (e.g.
    ``El_B2004a.comcam_g.AB``, ``El_B2004a.euclid_vis.AB``), so a
    survey-prefixed band ``lsst_g`` resolves to filter id ``comcam_g``
    and ``euclid_vis`` to ``euclid_vis``. This handles single- and
    mixed-survey band lists.
    """
    if filtersets is None:
        filtersets = {"lsst": "comcam", "euclid": "euclid"}
    filters = [
        f"{filtersets[survey_of(b)]}_{physical_band(b)}" for b in bands
    ]
    from desc_bpz.useful_py3 import get_data, get_str, match_resol

    z = Z_GRIDS
    spectra_file = os.path.join(data_path, "SED", spectra_name)
    spectra = [s[:-4] for s in get_str(spectra_file)]
    nt = len(spectra)
    nf = len(filters)
    nz = len(z)
    flux_templates = np.zeros((nz, nt, nf))
    # # Pre-scan AB dir (kept in case you want to validate presence)
    # _ab_file_list = glob.glob(ab_dir + "/*.AB")
    # _ab_file_db = [os.path.split(x)[-1] for x in _ab_file_list]

    for i, s in enumerate(spectra):
        for j, f in enumerate(filters):
            model = f"{s}.{f}.AB"
            model_path = os.path.join(data_path, "AB", model)
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Cannot find template model: {model_path}")
            zo, f_mod_0 = get_data(model_path, (0, 1))
            flux_templates[:, i, j] = match_resol(zo, f_mod_0, z)
    return flux_templates


class bpzEstimator(zEstimator):
    """
    Wraps BPZ template/prior configuration with a uniform `get_z` API.
    """

    def __init__(
        self,
        flux_templates: np.ndarray,
        prior_dict: dict,
        zp_errors,
    ):
        """
        Parameters
        ----------
        flux_templates : array, shape (NZ, NT, NF)
        prior_dict : dict
        zp_errors : zero-point mag errors per band, same order as `bands`
        """
        self.flux_templates = flux_templates
        self.prior_dict = prior_dict
        self.zp_errors = np.array(zp_errors, dtype=float)

    def _measure_one_source(
        self,
        flux: np.ndarray,
        flux_err: np.ndarray,
        mag_0: float,
    ):
        from desc_bpz.bpz_tools_py3 import p_c_z_t
        from desc_bpz.prior_from_dict import prior_function

        nt = self.flux_templates.shape[1]
        pczt = p_c_z_t(flux, flux_err, self.flux_templates)
        L = pczt.likelihood
        P = prior_function(Z_GRIDS, mag_0, self.prior_dict, nt)
        post = L * P
        pdf = post.sum(axis=1)
        return pdf

    def get_z(
        self,
        src: np.ndarray,
        *,
        mag_zero: float = MAG_ZERO_AB,
        flux_name: str = "gauss2",
        bands: tuple[str, ...] = ("lsst_g", "lsst_r", "lsst_i", "lsst_z", "lsst_y"),
        ref_band: str = "lsst_i",
        comp: int = 1,
        dg: float = 0.0,
        return_pdfs: bool = False,
        extinction: np.ndarray | None = None,
        **kwargs,
    ) -> dict:

        fn = _resolve_cut_name(flux_name)
        # Per-band magnitudes from the SAME smooth ``flux_to_mag`` used
        # for the FlexZBoost features, so the flux -> mag -> pseudo-flux
        # path is identical between training and measurement.
        mags = []
        merrs = []
        for b in bands:
            flux = src[f"{b}_flux{fn}"] + dg * src[f"{b}_dflux{fn}_dg{comp}"]
            ferr = src[f"{b}_flux{fn}_err"]
            a_ext = None if extinction is None else extinction[f"a_{b}"]
            mag, mag_err = flux_to_mag(flux, mag_zero, flux_err=ferr, a_ext=a_ext)
            mags.append(mag)
            merrs.append(mag_err)

        mags = np.array(mags).T
        merrs = np.array(merrs).T

        from desc_bpz.bpz_tools_py3 import e_mag2frac

        zp_frac = e_mag2frac(np.array(self.zp_errors))

        # Derive the AB pseudo-flux (zeropoint 0, where the templates live) and
        # its error from the magnitude, the standard BPZ conversion. mag_err is
        # clipped at 70 before e_mag2frac (10**(0.4*70)=1e28) so a smoothly
        # truncated non-detection (huge mag_err) gives a huge but finite error
        # (>>1) -> the band is dropped by ``p_c_z_t``, without float overflow.
        flux = 10.0 ** (-0.4 * mags)
        flux_err = flux * e_mag2frac(np.minimum(merrs, 70.0))
        add_err = (zp_frac * flux) ** 2
        flux_err = np.sqrt(flux_err**2 + add_err)

        mag0 = mags[:, bands.index(ref_band)]
        del mags, merrs

        ng = len(src)

        pdfs = np.stack(
            [self._measure_one_source(flux[i], flux_err[i], mag0[i]) for i in range(ng)],
            dtype=float,
        )
        points = get_point_estimates_from_pdfs(pdfs)
        if return_pdfs:
            points["pdfs"] = pdfs
        return points
