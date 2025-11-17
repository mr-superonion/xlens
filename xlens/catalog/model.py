import numpy as np


def evar_model(mag, radius, c0, c1, c2, c3, c4, c5):
    s0 = 0.01
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


def w_model(flux, trace, mag_zero, c0, c1, c2, c3, c4, c5):
    mag = mag_zero - 2.5 * np.log10(flux)
    radius = np.sqrt(trace / 2.0)
    return 1.0 / evar_model(mag, radius, c0, c1, c2, c3, c4, c5)


def w_model_dflux(flux, trace, mag_zero, c0, c1, c2, c3, c4, c5):
    s0 = 0.01
    mag = mag_zero - 2.5 * np.log10(flux)
    radius = np.sqrt(trace / 2.0)

    logAB = c0 + c1*mag + c2*radius + (
        c3*mag**2 + c4*radius**2 + c5*mag*radius
    )
    AB = np.exp(logAB)
    evar = s0**2 + AB**2

    dlogAB_dmag = c1 + 2*c3*mag + c5*radius

    return 5.0 * AB**2 * dlogAB_dmag / (
        flux * np.log(10.0) * evar**2
    )


def w_model_dtrace(flux, trace, mag_zero, c0, c1, c2, c3, c4, c5):
    s0 = 0.01
    mag = mag_zero - 2.5 * np.log10(flux)
    radius = np.sqrt(trace / 2.0)

    logAB = c0 + c1*mag + c2*radius + (
        c3*mag**2 + c4*radius**2 + c5*mag*radius
    )
    AB = np.exp(logAB)
    evar = s0**2 + AB**2

    dlogAB_dr = c2 + 2*c4*radius + c5*mag

    return - AB**2 * dlogAB_dr / (
        2.0 * radius * evar**2
    )


def estimate_mean_in_bins(
    *,
    mag,
    radius,
    obs,
    mag_edges,
    radius_edges,
    min_count: int = 1,
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
    mag = np.asarray(mag)
    radius = np.asarray(radius)
    obs = np.asarray(obs)

    # --- define numbers of bins ---
    n_mag_bins = len(mag_edges) - 1
    n_radius_bins = len(radius_edges) - 1
    n_bins_total = n_mag_bins * n_radius_bins

    # --- bin centers for plotting ---
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    X, Y = np.meshgrid(mag_centers, radius_centers, indexing="ij")

    # --- digitize points into bin indices ---
    mag_idx = np.digitize(mag, mag_edges) - 1
    radius_idx = np.digitize(radius, radius_edges) - 1

    # --- mask out anything that fell outside the range ---
    valid = (
        (mag_idx >= 0) & (mag_idx < n_mag_bins) &
        (radius_idx >= 0) & (radius_idx < n_radius_bins)
    )
    mag_idx = mag_idx[valid]
    radius_idx = radius_idx[valid]
    obs_valid = obs[valid]

    # --- map 2D (i, j) -> 1D bin index k = i * n_radius_bins + j ---
    k = mag_idx * n_radius_bins + radius_idx

    # --- use bincount to accumulate counts and sums in each 2D bin ---
    counts = np.bincount(k, minlength=n_bins_total).astype(int)
    sums = np.bincount(k, weights=obs_valid, minlength=n_bins_total)

    # --- compute means where we have enough objects ---
    mean_flat = np.full(n_bins_total, np.nan, dtype=float)
    good_bins = counts >= min_count
    mean_flat[good_bins] = sums[good_bins] / counts[good_bins]

    # --- flatten bin centers ---
    x_array = X.ravel()
    y_array = Y.ravel()
    n_array = counts

    # --- keep only bins with mean defined (>= min_count) ---
    keep = good_bins
    x_array = x_array[keep]
    y_array = y_array[keep]
    mean_array = mean_flat[keep]
    n_array = n_array[keep]

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
    mag = np.asarray(mag)
    radius = np.asarray(radius)
    obs = np.asarray(obs)

    # --- bin definitions ---
    n_mag_bins = len(mag_edges) - 1
    n_radius_bins = len(radius_edges) - 1
    n_bins_total = n_mag_bins * n_radius_bins

    # --- bin centers (for plotting) ---
    mag_centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    X, Y = np.meshgrid(mag_centers, radius_centers, indexing="ij")

    # --- digitize into 2D bin indices ---
    mag_idx = np.digitize(mag, mag_edges) - 1
    radius_idx = np.digitize(radius, radius_edges) - 1

    # keep only points inside the bin ranges
    valid = (
        (mag_idx >= 0) & (mag_idx < n_mag_bins) &
        (radius_idx >= 0) & (radius_idx < n_radius_bins)
    )
    mag_idx = mag_idx[valid]
    radius_idx = radius_idx[valid]
    obs_valid = obs[valid]

    # map 2D bin index (i, j) -> 1D index k
    k = mag_idx * n_radius_bins + radius_idx

    # --- accumulate counts, sum, and sum of squares per bin ---
    counts = np.bincount(k, minlength=n_bins_total).astype(int)
    sum_obs = np.bincount(k, weights=obs_valid, minlength=n_bins_total)
    sum_obs2 = np.bincount(k, weights=obs_valid**2, minlength=n_bins_total)

    # --- compute std with ddof=0 (to match np.std default) ---
    std_flat = np.zeros(n_bins_total, dtype=float)
    non_empty = counts > 0
    mean = np.zeros_like(std_flat)
    mean[non_empty] = sum_obs[non_empty] / counts[non_empty]
    var = np.zeros_like(std_flat)
    var[non_empty] = (
        sum_obs2[non_empty] / counts[non_empty] - mean[non_empty]**2
    )
    std_flat[non_empty] = np.sqrt(np.maximum(var[non_empty], 0.0))

    # --- flatten bin centers ---
    x_array = X.ravel()
    y_array = Y.ravel()
    n_array = counts

    # --- keep only bins with enough galaxies ---
    keep = counts >= min_count
    x_array = x_array[keep]
    y_array = y_array[keep]
    std_array = std_flat[keep]
    n_array = n_array[keep]

    return x_array, y_array, std_array, n_array
