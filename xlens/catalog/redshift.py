import os
from abc import ABC, abstractmethod

import numpy as np

from .utils import _resolve_flux_name

NUM_Z_GRIDS = 501
Z_MAX = 5.0
Z_GRIDS = np.linspace(0.0, Z_MAX, NUM_Z_GRIDS)
PROBS = np.array([0.025, 0.16, 0.5, 0.84, 0.975], dtype=float)


def get_color(
    src: np.ndarray,
    *,
    bands: str = "grizy",
    ref_band: str = "i",
    mag_zero: float = 30.0,
    comp: int = 1,
    dg: float = 0.0,
    flux_name: str = "gauss2",
    include_mag_err: bool = False,
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
    fn = _resolve_flux_name(flux_name)
    A = 2.5 / np.log(10.0)
    n = src.shape[0]

    mags: list[np.ndarray] = []
    merrs: list[np.ndarray] | None = [] if include_mag_err else None
    # Compute mag (and optionally mag_err) per band, in the same order as
    # `bands`
    for b in bands:
        flux_col = f"{b}_flux{fn}"
        dflux_col = f"{b}_dflux{fn}_dg{comp}"
        err_col = f"{flux_col}_err"

        flux_base = src[flux_col]
        dflux = src[dflux_col]
        ferr = src[err_col]

        flux = flux_base + dg * dflux
        mag = np.full(n, mag_zero, dtype=np.float64)
        pos = flux > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            mag[pos] = mag_zero - 2.5 * np.log10(flux[pos])
        mags.append(mag)
        if merrs is not None:
            mag_err = np.full(n, 1.0, dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                mag_err[pos] = A * (ferr[pos] / flux[pos])
            merrs.append(mag_err)

    nb = len(bands) - 1
    ncols = 1 + (2 * nb if include_mag_err else nb)
    feat = np.empty((n, ncols), dtype=np.float64)
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


def get_percentiles_from_pdf(pdfs: np.ndarray):
    n = pdfs.shape[0]
    z025 = np.full(n, np.nan, dtype=float)
    z160 = np.full(n, np.nan, dtype=float)
    z500 = np.full(n, np.nan, dtype=float)
    z840 = np.full(n, np.nan, dtype=float)
    z975 = np.full(n, np.nan, dtype=float)
    for i, p in enumerate(pdfs):
        total = p.sum()
        if not np.isfinite(total) or total <= 0.0:
            continue
        cdf = np.cumsum(p, dtype=float) / total
        zqs = np.interp(PROBS, cdf, Z_GRIDS)
        z025[i] = zqs[0]
        z160[i] = zqs[1]
        z500[i] = zqs[2]
        z840[i] = zqs[3]
        z975[i] = zqs[4]
    return {
        "z025": z025,
        "z160": z160,
        "z500": z500,
        "z840": z840,
        "z975": z975,
    }


# ------------------------
# Z-Estimator Implementations
# ------------------------

class zEstimator(ABC):
    @abstractmethod
    def get_z(
        self,
        src: np.ndarray,
        *,
        mag_zero: float = 30.0,
        flux_name: str = "gauss2",
        bands: str = "grizy",
        ref_band: str = "i",
        comp: int = 1,
        dg: float = 0.0,
        flux_name2: str | None = None,
        flux_name3: str | None = None,
        **kwargs,
    ) -> dict:
        """Method to get redshift point estimates
        """

    def get_zsel(
        self,
        src: np.ndarray,
        *,
        mag_zero: float = 30.0,
        flux_name: str = "gauss2",
        bands: str = "grizy",
        ref_band: str = "i",
        comp: int = 1,
        dg: float = 0.0,
        flux_name2: str | None = None,
        flux_name3: str | None = None,
        **kwargs,
    ):
        zout = self.get_z(
            src=src, mag_zero=mag_zero, flux_name=flux_name,
            bands=bands, ref_band=ref_band, comp=comp, dg=dg,
            flux_name2=flux_name2, flux_name3=flux_name3,
            **kwargs,
        )
        zmode = zout["zmode"]
        width95 = zout["z975"] - zout["z025"]
        return zmode, width95


class flexzboostEstimator(zEstimator):
    """
    Wraps a FlexZBoost-like predictor object with a uniform `get_z` API.
    """

    def __init__(
        self, pz_obj,
    ):
        self.pz_obj = pz_obj
        self.pz_obj.model.models.n_jobs = 1

    def get_z(
        self,
        src: np.ndarray,
        *,
        mag_zero: float = 30.0,
        flux_name: str = "gauss2",
        bands: str = "grizy",
        ref_band: str = "i",
        comp: int = 1,
        dg: float = 0.0,
        flux_name2: str | None = None,
        flux_name3: str | None = None,
        include_mag_err: bool = False,
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
        )
        pdfs, _ = self.pz_obj.predict(colors, n_grid=NUM_Z_GRIDS)
        # Argmax per row, then map to z_grid
        idx = np.argmax(pdfs, axis=1)
        zmode = np.take(Z_GRIDS, idx)
        zps = get_percentiles_from_pdf(pdfs)
        return {"zmode": zmode, **zps}


def load_bpz_templates(
    data_path: str,
    bands: str,
    filter_name: str = "DC2LSST",
    spectra_name: str = "cosmossedswdust136.list",
):
    """Load BPZ template fluxes on Z_GRIDS for provided filter set."""
    filters = [f"{filter_name}_{b}" for b in bands]
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
            assert os.path.isfile(model_path), "Cannot find model"
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

    def _meas_one(
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
        post_z = post.sum(axis=1)

        zmode = Z_GRIDS[np.argmax(post_z)]
        total = post_z.sum()
        if not np.isfinite(total) or total <= 0:
            return (
                float(zmode),
                np.nan, np.nan, np.nan, np.nan, np.nan,
            )
        cdf = np.cumsum(post_z, dtype=float) / total
        zqs = np.interp(PROBS, cdf, Z_GRIDS)
        return (
            float(zmode), float(zqs[0]),
            float(zqs[1]), float(zqs[2]),
            float(zqs[3]), float(zqs[4]),
        )

    def get_z(
        self,
        src: np.ndarray,
        *,
        mag_zero: float = 30.0,
        flux_name: str = "gauss2",
        bands: str = "grizy",
        ref_band: str = "i",
        comp: int = 1,
        dg: float = 0.0,
        **kwargs,
    ) -> dict:
        fn = _resolve_flux_name(flux_name)
        A = 2.5 / np.log(10.0)
        n = src.shape[0]

        mags = []
        merrs = []
        for b in bands:
            flux_col = f"{b}_flux{fn}"
            dflux_col = f"{b}_dflux{fn}_dg{comp}"
            err_col = f"{flux_col}_err"

            flux_base = src[flux_col]
            dflux = src[dflux_col]
            ferr = src[err_col]

            flux = flux_base + dg * dflux

            mag = np.full(n, 30.0, dtype=np.float64)
            mag_err = np.full(n, 1.0, dtype=np.float64)

            pos = flux > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                mag[pos] = mag_zero - 2.5 * np.log10(flux[pos])
                mag_err[pos] = A * (ferr[pos] / flux[pos])

            mags.append(mag)
            merrs.append(mag_err)

        mags = np.array(mags).T
        merrs = np.array(merrs).T

        from desc_bpz.bpz_tools_py3 import e_mag2frac
        zp_frac = e_mag2frac(np.array(self.zp_errors))

        # Convert to pseudo-fluxes and propagate errors
        flux = 10.0**(-0.4 * mags)
        flux_err = flux * (10.0**(0.4 * merrs) - 1.0)
        add_err = ((zp_frac * flux)**2)
        flux_err = np.sqrt(flux_err**2 + add_err)

        m_0_col = bands.index(ref_band)
        mag0 = mags[:, m_0_col]
        # Free some memory
        del mags, merrs

        ng = len(src)

        results = np.array(
            [self._meas_one(flux[i], flux_err[i], mag0[i]) for i in range(ng)],
            dtype=float,
        )
        zmode, z025, z160, z500, z840, z975 = results.T
        return {
            "zmode": zmode,
            "z025": z025,
            "z160": z160,
            "z500": z500,
            "z840": z840,
            "z975": z975,
        }
