import numpy as np

s0 = 0.01  # ground for std

def evar_model(mag, radius, c0, c1, c2, c3, c4, c5):
    logAB = c0 + c1 * mag + c2 * radius + (
        c3 * mag ** 2.0 + c4 * radius ** 2.0
        + c5 * mag * radius
    )
    AB = np.exp(logAB)
    return s0**2 + AB**2


def estd_model_fit(coords, c0, c1, c2, c3, c4, c5):
    mag, radius = coords
    return np.sqrt(
        evar_model(mag, radius, c0, c1, c2, c3, c4, c5)
    )


def w_model(flux, m0, m2, mag_zero, c0, c1, c2, c3, c4, c5):
    mag = mag_zero - 2.5 * np.log10(flux)
    trace = m2 / m0
    radius = np.sqrt(trace / 2.0)
    return 1.0 / evar_model(mag, radius, c0, c1, c2, c3, c4, c5)


def w_model_derivs(flux, m0, m2, mag_zero, c0, c1, c2, c3, c4, c5):
    """
    Compute derivatives of w = 1 / evar_model(mag, radius)
    with respect to flux, m0, and m2.

    Returns
    -------
    dict with keys:
      - "dw_dflux"
      - "dw_dm0"
      - "dw_dm2"
    """
    # shared pieces
    mag = mag_zero - 2.5 * np.log10(flux)
    trace = m2 / m0
    radius = np.sqrt(trace / 2.0)

    logAB = c0 + c1*mag + c2*radius + (
        c3*mag**2 + c4*radius**2 + c5*mag*radius
    )
    AB = np.exp(logAB)
    evar = s0**2 + AB**2

    # derivatives of logAB
    dlogAB_dmag = c1 + 2*c3*mag + c5*radius
    dlogAB_dr   = c2 + 2*c4*radius + c5*mag

    # --- dw/dflux ---
    # mag depends on flux; radius does not
    dw_dflux = 5.0 * AB**2 * dlogAB_dmag / (
        flux * np.log(10.0) * evar**2
    )

    # --- dw/dtrace (used for dm0, dm2) ---
    dw_dtrace = - AB**2 * dlogAB_dr / (2.0 * radius * evar**2)

    # dtrace/dm0 = -trace/m0
    dtrace_dm0 = -trace / m0
    dw_dm0 = dw_dtrace * dtrace_dm0

    # dtrace/dm2 = 1/m0
    dtrace_dm2 = 1.0 / m0
    dw_dm2 = dw_dtrace * dtrace_dm2

    return {
        "dw_dflux": dw_dflux,
        "dw_dm0": dw_dm0,
        "dw_dm2": dw_dm2,
    }


def estimate_mean_in_bins(
    *,
    mag,
    radius,
    obs,
    mag_edges,
    radius_edges,
    min_count: int = 100,
):
    """
    Estimate mean(obs) in 2D bins of (mag, radius).

    Parameters
    ----------
    mag, radius, obs : array_like, shape (N,)
        Input data.
    mag_edges, radius_edges : array_like
        Bin edges along mag and radius.
    min_count : int, optional
        Minimum number of objects required in a bin to keep it.
        Bins with fewer galaxies are dropped from the output.

    Returns
    -------
    x_array, y_array : ndarray, shape (Nbins_kept,)
        Bin centers in mag and radius for bins that pass the min_count cut.
    mean_array : ndarray, shape (Nbins_kept,)
        Mean of obs in each kept bin.
    n_array : ndarray, shape (Nbins_kept,)
        Number of objects in each kept bin.
    """
    n_mag_bins = len(mag_edges) - 1
    n_radius_bins = len(radius_edges) - 1

    # --- bin centers for plotting ---
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    X, Y = np.meshgrid(mag_centers, radius_centers, indexing="ij")

    # --- digitize points into bin indices ---
    mag_idx = np.digitize(mag, mag_edges) - 1
    radius_idx = np.digitize(radius, radius_edges) - 1

    # mask out anything that fell outside the range
    valid = (
        (mag_idx >= 0) & (mag_idx < n_mag_bins) &
        (radius_idx >= 0) & (radius_idx < n_radius_bins)
    )
    mag_idx = mag_idx[valid]
    radius_idx = radius_idx[valid]
    obs = obs[valid]

    obs_mean = np.full((n_mag_bins, n_radius_bins), np.nan, dtype=float)
    ngals = np.full((n_mag_bins, n_radius_bins), np.nan, dtype=int)

    # loop over bins and compute std
    # (fine for modest numbers of bins; easy to read)
    for i in range(n_mag_bins):
        for j in range(n_radius_bins):
            mask = (mag_idx == i) & (radius_idx == j)
            ngals[i, j] = np.sum(mask)
            if ngals[i, j] > min_count:
                obs_mean[i, j] = np.mean(obs[mask])

    # flatten and drop empty bins
    x_array = X.ravel()
    y_array = Y.ravel()
    mean_array = obs_mean.ravel()
    n_array = ngals.ravel()

    good = ~np.isnan(mean_array)
    x_array = x_array[good]
    y_array = y_array[good]
    mean_array = mean_array[good]
    n_array = n_array[good]
    return x_array, y_array, mean_array, n_array


def estimate_std_in_bins(
    *,
    mag: np.ndarray,
    radius: np.ndarray,
    obs: np.ndarray,
    mag_edges: np.ndarray,
    radius_edges: np.ndarray,
    min_count: int = 100,
):
    """
    Estimate std(obs) in 2D bins of (mag, radius).

    Parameters
    ----------
    mag, radius, obs : array-like, shape (N,)
        Input data.
    mag_edges, radius_edges : array-like
        Bin edges along mag and radius.
    min_count : int, optional
        Minimum number of objects required in a bin to report a std.

    Returns
    -------
    x_array, y_array : ndarray, shape (Nbins_kept,)
        Bin centers in mag and radius.
    std_array : ndarray, shape (Nbins_kept,)
        Standard deviation of obs in each kept bin.
    n_array : ndarray, shape (Nbins_kept,)
        Number of objects in each kept bin.
    """
    n_mag_bins = len(mag_edges) - 1
    n_radius_bins = len(radius_edges) - 1

    # --- bin centers for plotting ---
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    X, Y = np.meshgrid(mag_centers, radius_centers, indexing="ij")

    # --- digitize points into bin indices ---
    mag_idx = np.digitize(mag, mag_edges) - 1
    radius_idx = np.digitize(radius, radius_edges) - 1

    # mask out anything that fell outside the range
    valid = (
        (mag_idx >= 0) & (mag_idx < n_mag_bins) &
        (radius_idx >= 0) & (radius_idx < n_radius_bins)
    )
    mag_idx = mag_idx[valid]
    radius_idx = radius_idx[valid]
    obs = obs[valid]

    obs_std = np.full((n_mag_bins, n_radius_bins), np.nan, dtype=float)
    ngals = np.full((n_mag_bins, n_radius_bins), np.nan, dtype=int)

    # loop over bins and compute std
    # (fine for modest numbers of bins; easy to read)
    for i in range(n_mag_bins):
        for j in range(n_radius_bins):
            mask = (mag_idx == i) & (radius_idx == j)
            ngals[i, j] = np.sum(mask)
            if ngals[i, j] > min_count:
                obs_std[i, j] = np.std(obs[mask])

    # flatten and drop empty bins
    x_array = X.ravel()
    y_array = Y.ravel()
    std_array = obs_std.ravel()
    n_array = ngals.ravel()

    good = ~np.isnan(std_array)
    x_array = x_array[good]
    y_array = y_array[good]
    std_array = std_array[good]
    n_array = n_array[good]
    return x_array, y_array, std_array, n_array
