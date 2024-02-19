import astropy.io.fits as pyfits
import healpy as hp
import numpy as np
from astropy.table import Table, join


def prepare_simulation_dm_catalog(filename, isim=-1):
    """From LSST pipeline data to HSC catalog data

    Args:
    filename (str):     fits filename to read
    fieldname (str):    HSC field information
    isim (int):         simulation subfield information
    ngalR (int):        number of stamps in each row
    ngrid (int):        number of grids in each stamp

    Returns:
    catalog (ndarray): the prepared catalog for a subfield
    """

    # Read the catalog data
    catalog = Table.read(filename)
    colNames = catalog.colnames

    # Load the header to get proper name of flags
    header = pyfits.getheader(filename, 1)
    n_flag = catalog["flags"].shape[1]
    for i in range(n_flag):
        catalog[header["TFLAG%s" % (i + 1)]] = catalog["flags"][:, i]

    # Then, apply mask for permissive cuts
    mask = (
        (~(catalog["base_SdssCentroid_flag"]))
        & (~catalog["ext_shapeHSM_HsmShapeRegauss_flag"])
        & (catalog["base_ClassificationExtendedness_value"] > 0)
        & (~np.isnan(catalog["modelfit_CModel_instFlux"]))
        & (~np.isnan(catalog["modelfit_CModel_instFluxErr"]))
        & (~np.isnan(catalog["ext_shapeHSM_HsmShapeRegauss_resolution"]))
        & (~np.isnan(catalog["ext_shapeHSM_HsmPsfMoments_xx"]))
        & (~np.isnan(catalog["ext_shapeHSM_HsmPsfMoments_yy"]))
        & (~np.isnan(catalog["ext_shapeHSM_HsmPsfMoments_xy"]))
        & (~np.isnan(catalog["base_Variance_value"]))
        & (~np.isnan(catalog["modelfit_CModel_instFlux"]))
        & (~np.isnan(catalog["modelfit_CModel_instFluxErr"]))
        & (~np.isnan(catalog["ext_shapeHSM_HsmShapeRegauss_resolution"]))
        & (catalog["deblend_nChild"] == 0)
    )
    catalog = catalog[mask]
    if len(catalog) == 0:
        return None
    catalog = catalog[colNames]
    # Add column to make a WL flag for the simulation data:
    catalog["weak_lensing_flag"] = get_wl_cuts(catalog)
    return catalog


def get_pixel_cuts(catalog):
    """Returns pixel cuts"""
    mask = (
        (~catalog["i_deblend_skipped"])
        & (~catalog["i_cmodel_flag_badcentroid"])
        & (~catalog["i_sdsscentroid_flag"])
        & (
            ~catalog["i_detect_isprimary"]
            & (~catalog["i_pixelflags_edge"])
            & (~catalog["i_pixelflags_interpolatedcenter"])
            & (~catalog["i_pixelflags_saturatedcenter"])
            & (~catalog["i_pixelflags_crcenter"])
            & (~catalog["i_pixelflags_bad"])
            & (~catalog["i_pixelflags_suspectcenter"])
            & (~catalog["i_pixelflags_clipped"])
        )
    )
    return mask


def get_snr(catalog):
    """This utility computes the S/N for each object in the catalog, based on
    cmodel_flux. It does not impose any cuts and returns NaNs for invalid S/N
    values.
    """
    if "snr" in catalog.dtype.names:
        return catalog["snr"]
    elif "i_cmodel_fluxsigma" in catalog.dtype.names:  # s18
        snr = catalog["i_cmodel_flux"] / catalog["i_cmodel_fluxsigma"]
    elif "iflux_cmodel" in catalog.dtype.names:  # s15
        snr = catalog["iflux_cmodel"] / catalog["iflux_cmodel_err"]
    elif "i_cmodel_fluxerr" in catalog.dtype.names:  # s19
        snr = catalog["i_cmodel_flux"] / catalog["i_cmodel_fluxerr"]
    elif "modelfit_CModel_instFlux" in catalog.dtype.names:  # pipe 7
        snr = catalog["modelfit_CModel_instFlux"] / catalog["modelfit_CModel_instFluxErr"]
    else:
        snr = _nan_array(len(catalog))
    return snr


def get_snr_apertures(catalog):
    """This utility computes the S/N for each object in the catalog, based on
    aperture_fluxes. It does not impose any cuts and returns NaNs for invalid
    S/N values.
    """
    if "i_apertureflux_10_fluxsigma" in catalog.dtype.names:  # s18
        snr10 = catalog["i_apertureflux_10_flux"] / catalog["i_apertureflux_10_fluxsigma"]
        snr15 = catalog["i_apertureflux_15_flux"] / catalog["i_apertureflux_15_fluxsigma"]
        snr20 = catalog["i_apertureflux_20_flux"] / catalog["i_apertureflux_20_fluxsigma"]
    elif "i_apertureflux_10_fluxerr" in catalog.dtype.names:  # s19
        snr10 = catalog["i_apertureflux_10_flux"] / catalog["i_apertureflux_10_fluxerr"]
        snr15 = catalog["i_apertureflux_15_flux"] / catalog["i_apertureflux_15_fluxerr"]
        snr20 = catalog["i_apertureflux_20_flux"] / catalog["i_apertureflux_20_fluxerr"]
    elif "base_CircularApertureFlux_3_0_instFlux" in catalog.dtype.names:  # pipe 7
        snr10 = (
            catalog["base_CircularApertureFlux_3_0_instFlux"]
            / catalog["base_CircularApertureFlux_3_0_instFluxErr"]
        )
        snr15 = (
            catalog["base_CircularApertureFlux_4_5_instFlux"]
            / catalog["base_CircularApertureFlux_4_5_instFluxErr"]
        )
        snr20 = (
            catalog["base_CircularApertureFlux_6_0_instFlux"]
            / catalog["base_CircularApertureFlux_6_0_instFluxErr"]
        )
    else:
        snr10 = _nan_array(len(catalog))
        snr15 = _nan_array(len(catalog))
        snr20 = _nan_array(len(catalog))
    return snr10, snr15, snr20


def get_snr_localBG(catalog):
    """This utility computes the S/N for each object in the catalog,
    based on local background flux. It does not impose any cuts
    and returns NaNs for invalid S/N values.
    """
    if "i_localbackground_fluxsigma" in catalog.dtype.names:  # s18
        snrloc = catalog["i_localbackground_flux"] / catalog["i_localbackground_fluxsigma"]
    elif "i_localbackground_fluxerr" in catalog.dtype.names:  # s19
        snrloc = catalog["i_localbackground_flux"] / catalog["i_localbackground_fluxerr"]
    elif "base_LocalBackground_instFlux" in catalog.dtype.names:  # pipe 7
        snrloc = catalog["base_LocalBackground_instFlux"] / catalog["base_LocalBackground_instFluxErr"]
    else:
        snrloc = _nan_array(len(catalog))
    return snrloc


def get_photo_z(catalog, method_name):
    """Returns the best photon-z estimation

    Args:
    catalog (ndarray):  input catlog
    method_name:        name of the photo-z method (mizuki, dnn, demp)

    Returns:
    z (ndarray):        photo-z best estimation
    """
    if method_name == "mizuki":
        if "mizuki_photoz_best" in catalog.dtype.names:
            z = catalog["mizuki_photoz_best"]
        elif "mizuki_Z" in catalog.dtype.names:
            z = catalog["mizuki_Z"]
        else:
            raise ValueError("Cannot get photo-z: %s" % method_name)
    elif method_name == "dnn":
        if "dnnz_photoz_best" in catalog.dtype.names:
            z = catalog["dnnz_photoz_best"]
        elif "dnn_Z" in catalog.dtype.names:
            z = catalog["dnn_Z"]
        else:
            raise ValueError("Cannot get photo-z: %s" % method_name)
    elif method_name == "demp":
        if "dempz_photoz_best" in catalog.dtype.names:
            z = catalog["dempz_photoz_best"]
        elif "demp_Z" in catalog.dtype.names:
            z = catalog["demp_Z"]
        else:
            raise ValueError("Cannot get photo-z: %s" % method_name)
    else:
        z = _nan_array(len(catalog))
    return z


def get_imag_A10(catalog):
    """This utility returns the i-band magnitude of the objects in the input
    data or simulation catalog. Does not apply any cuts and returns NaNs for
    invalid values.
    """
    if "magA10" in catalog.dtype.names:
        return catalog["magA10"]
    elif "i_apertureflux_10_mag" in catalog.dtype.names:  # s18 s19
        magnitude = catalog["i_apertureflux_10_mag"]
    elif "base_CircularApertureFlux_3_0_instFlux" in catalog.dtype.names:  # pipe 7
        magnitude = -2.5 * np.log10(catalog["base_CircularApertureFlux_3_0_instFlux"]) + 27.0
    else:
        magnitude = _nan_array(len(catalog))
    return magnitude


def get_imag_A15(catalog):
    """This utility returns the i-band magnitude of the objects in the input
    data or simulation catalog. Does not apply any cuts and returns NaNs for
    invalid values.
    """
    if "magA15" in catalog.dtype.names:
        return catalog["magA15"]
    elif "i_apertureflux_15_mag" in catalog.dtype.names:  # s18 s19
        magnitude = catalog["i_apertureflux_15_mag"]
    elif "base_CircularApertureFlux_4_5_instFlux" in catalog.dtype.names:  # pipe 7
        magnitude = -2.5 * np.log10(catalog["base_CircularApertureFlux_4_5_instFlux"]) + 27.0
    else:
        magnitude = _nan_array(len(catalog))
    return magnitude


def get_imag_A20(catalog):
    """This utility returns the i-band magnitude of the objects in the input
    data or simulation catalog. Does not apply any cuts and returns NaNs for
    invalid values.
    """
    if "magA20" in catalog.dtype.names:
        return catalog["magA20"]
    if "i_apertureflux_20_mag" in catalog.dtype.names:  # s18 s19
        magnitude = catalog["i_apertureflux_20_mag"]
    elif "base_CircularApertureFlux_6_0_instFlux" in catalog.dtype.names:  # pipe 7
        magnitude = -2.5 * np.log10(catalog["base_CircularApertureFlux_6_0_instFlux"]) + 27.0
    else:
        magnitude = _nan_array(len(catalog))
    return magnitude


def get_imag(catalog):
    """This utility returns the i-band magnitude of the objects in the input
    data or simulation catalog. Does not apply any cuts and returns NaNs for
    invalid values.

    Args:
    catalog (ndarray):  input catlog

    Returns:
    mag (ndarray):      iband magnitude
    """
    if "mag" in catalog.dtype.names:
        mag = catalog["mag"]
    elif "i_cmodel_mag" in catalog.dtype.names:  # s18 and s19
        mag = catalog["i_cmodel_mag"]
    elif "imag_cmodel" in catalog.dtype.names:  # s15
        mag = catalog["imag_cmodel"]
    elif "modelfit_CModel_instFlux" in catalog.dtype.names:  # pipe 7
        mag = -2.5 * np.log10(catalog["modelfit_CModel_instFlux"]) + 27.0
    else:
        mag = _nan_array(len(catalog))
    return mag


def get_imag_psf(catalog):
    """Returns the i-band magnitude of the objects in the input data or
    simulation catalog. Does not apply any cuts and returns NaNs for invalid
    values.

    Args:
    catalog (ndarray):     input catalog

    Returns:
    magnitude (ndarray):   PSF magnitude
    """
    if "i_psfflux_mag" in catalog.dtype.names:  # s18 and s19
        magnitude = catalog["i_psfflux_mag"]
    elif "imag_psf" in catalog.dtype.names:  # s15
        magnitude = catalog["imag_psf"]
    elif "base_PsfFlux_instFlux" in catalog.dtype.names:  # pipe 7
        magnitude = -2.5 * np.log10(catalog["base_PsfFlux_instFlux"]) + 27.0
    else:
        magnitude = _nan_array(len(catalog))
    return magnitude


def get_npass(catalog, meas="cmodel"):
    """Returns npass values

    Args:
    catalog (ndarray):  input catalog
    meas (str):         SNR definition used to get npass [default: 'cmodel']

    Returns:
    npass (ndarray):    npass array
    """
    if "npass" in catalog.dtype.names:
        return catalog["npass"]
    elif "g_inputcount_value" in catalog.dtype.names:
        gcountinputs = catalog["g_inputcount_value"]
        rcountinputs = catalog["r_inputcount_value"]
        zcountinputs = catalog["z_inputcount_value"]
        ycountinputs = catalog["y_inputcount_value"]
        if meas == "cmodel":
            pendN = "cmodel_mag"
        elif meas == "aperture":
            pendN = "apertureflux_10_mag"
        else:
            return _nan_array(len(catalog))
        if "forced_g_%ssigma" % pendN in catalog.dtype.names:  # For S18A
            pendN += "sigma"
        if "forced_g_%serr" % pendN in catalog.dtype.names:  # For S19A
            pendN += "err"
        # multi-band detection to remove junk
        g_snr = (2.5 / np.log(10.0)) / catalog["forced_g_%s" % pendN]
        r_snr = (2.5 / np.log(10.0)) / catalog["forced_r_%s" % pendN]
        z_snr = (2.5 / np.log(10.0)) / catalog["forced_z_%s" % pendN]
        y_snr = (2.5 / np.log(10.0)) / catalog["forced_y_%s" % pendN]
    elif "gcountinputs" in catalog.dtype.names:
        gcountinputs = catalog["gcountinputs"]
        rcountinputs = catalog["rcountinputs"]
        zcountinputs = catalog["zcountinputs"]
        ycountinputs = catalog["ycountinputs"]
        # multi-band detection to remove junk
        g_snr = (2.5 / np.log(10.0)) / catalog["gmag_forced_cmodel_err"]
        r_snr = (2.5 / np.log(10.0)) / catalog["rmag_forced_cmodel_err"]
        z_snr = (2.5 / np.log(10.0)) / catalog["zmag_forced_cmodel_err"]
        y_snr = (2.5 / np.log(10.0)) / catalog["ymag_forced_cmodel_err"]
    else:
        out = _nan_array(len(catalog))
        return out

    # Calculate npass
    min_multiband_snr_data = 5.0
    g_mask = (g_snr >= min_multiband_snr_data) & (~np.isnan(g_snr) & (gcountinputs >= 2))
    r_mask = (r_snr >= min_multiband_snr_data) & (~np.isnan(r_snr) & (rcountinputs >= 2))
    z_mask = (z_snr >= min_multiband_snr_data) & (~np.isnan(z_snr) & (zcountinputs >= 2))
    y_mask = (y_snr >= min_multiband_snr_data) & (~np.isnan(y_snr) & (ycountinputs >= 2))
    npass = g_mask.astype(int) + r_mask.astype(int) + z_mask.astype(int) + y_mask.astype(int)
    return npass


def get_abs_ellip(catalog):
    """Returns the norm of galaxy ellipticities.

    Args:
    catalog (ndarray):  input catlog

    Returns:
    absE (ndarray):     norm of galaxy ellipticities
    """
    if "absE" in catalog.dtype.names:
        absE = catalog["absE"]
    elif "i_hsmshaperegauss_e1" in catalog.dtype.names:  # For S18A
        absE = catalog["i_hsmshaperegauss_e1"] ** 2.0 + catalog["i_hsmshaperegauss_e2"] ** 2.0
        absE = np.sqrt(absE)
    elif "ishape_hsm_regauss_e1" in catalog.dtype.names:  # For S16A
        absE = catalog["ishape_hsm_regauss_e1"] ** 2.0 + catalog["ishape_hsm_regauss_e2"] ** 2.0
        absE = np.sqrt(absE)
    elif "ext_shapeHSM_HsmShapeRegauss_e1" in catalog.dtype.names:  # For pipe 7
        absE = (
            catalog["ext_shapeHSM_HsmShapeRegauss_e1"] ** 2.0
            + catalog["ext_shapeHSM_HsmShapeRegauss_e2"] ** 2.0
        )
        absE = np.sqrt(absE)
    else:
        absE = _nan_array(len(catalog))
    return absE


def get_abs_ellip_psf(catalog):
    """Returns the amplitude of ellipticities of PSF

    Args:
    catalog (ndarray):  input catalog

    Returns:
    out (ndarray):      the modulus of galaxy distortions.
    """
    e1, e2 = get_psf_ellip(catalog)
    out = e1**2.0 + e2**2.0
    out = np.sqrt(out)
    return out


def get_FDFC_flag(data, hpfname):
    """Returns the Full Depth Full Color (FDFC) cut

    Args:
    data (ndarray):     input catalog array
    hpfname (str):      healpix fname (s16a_wide2_fdfc.fits,
                        s18a_fdfc_hp_contarea.fits, or
                        s19a_fdfc_hp_contarea_izy-gt-5_trimmed_fd001.fits)
    Returns:
    mask (ndarray):     mask array for FDFC region
    """
    ra, dec = get_radec(data)
    m = hp.read_map(hpfname, nest=True, dtype=bool)

    # Get flag
    mfactor = np.pi / 180.0
    indices_map = np.where(m)[0]
    nside = hp.get_nside(m)
    phi = ra * mfactor
    theta = np.pi / 2.0 - dec * mfactor
    indices_obj = hp.ang2pix(nside, theta, phi, nest=True)
    return np.in1d(indices_obj, indices_map)


def get_radec(catalog):
    """Returns the angular position

    Args:
    catalog (ndarray):  input catalog

    Returns:
    ra (ndarray): ra
    dec (ndarray): dec
    """
    if "ra" in catalog.dtype.names:  # small catalog
        ra = catalog["ra"]
        dec = catalog["dec"]
    elif "i_ra" in catalog.dtype.names:  # s18 & s19
        ra = catalog["i_ra"]
        dec = catalog["i_dec"]
    elif "ira" in catalog.dtype.names:  # s15
        ra = catalog["ira"]
        dec = catalog["idec"]
    elif "coord_ra" in catalog.dtype.names:  # pipe 7
        ra = catalog["coord_ra"]
        dec = catalog["coord_dec"]
    elif "ra_mock" in catalog.dtype.names:  # mock catalog
        ra = catalog["ra_mock"]
        dec = catalog["dec_mock"]
    else:
        ra = _nan_array(len(catalog))
        dec = _nan_array(len(catalog))
    return ra, dec


def get_res(catalog):
    """Returns the resolution

    Args:
    catalog (ndarray):  input catalog

    Returns:
    res (ndarray):      resolution
    """
    if "res" in catalog.dtype.names:
        return catalog["res"]
    elif "i_hsmshaperegauss_resolution" in catalog.dtype.names:  # s18 & s19
        res = catalog["i_hsmshaperegauss_resolution"]
    elif "ishape_hsm_regauss_resolution" in catalog.dtype.names:  # s15
        res = catalog["ishape_hsm_regauss_resolution"]
    elif "ext_shapeHSM_HsmShapeRegauss_resolution" in catalog.dtype.names:  # pipe 7
        res = catalog["ext_shapeHSM_HsmShapeRegauss_resolution"]
    else:
        res = _nan_array(len(catalog))
    return res


def get_sdss_size(catalog, dtype="det"):
    """This utility gets the observed galaxy size from a data or sims catalog
    using the specified size definition from the second moments matrix.

    Args:
    catalog (ndarray):  Simulation or data catalog
    dtype (str):        Type of psf size measurement in ['trace', 'determin']

    Returns:
    size (ndarray):     galaxy size
    """
    if "base_SdssShape_xx" in catalog.dtype.names:  # pipe 7
        gal_mxx = catalog["base_SdssShape_xx"] * 0.168**2.0
        gal_myy = catalog["base_SdssShape_yy"] * 0.168**2.0
        gal_mxy = catalog["base_SdssShape_xy"] * 0.168**2.0
    elif "i_sdssshape_shape11" in catalog.dtype.names:  # s18 & s19
        gal_mxx = catalog["i_sdssshape_shape11"]
        gal_myy = catalog["i_sdssshape_shape22"]
        gal_mxy = catalog["i_sdssshape_shape12"]
    elif "ishape_sdss_ixx" in catalog.dtype.names:  # s15
        gal_mxx = catalog["ishape_sdss_ixx"]
        gal_myy = catalog["ishape_sdss_iyy"]
        gal_mxy = catalog["ishape_sdss_ixy"]
    else:
        gal_mxx = _nan_array(len(catalog))
        gal_myy = _nan_array(len(catalog))
        gal_mxy = _nan_array(len(catalog))

    if dtype == "trace":
        size = np.sqrt(gal_mxx + gal_myy)
    elif dtype == "det":
        size = (gal_mxx * gal_myy - gal_mxy**2) ** (0.25)
    else:
        raise ValueError("Unknown PSF size type: %s" % dtype)
    return size


def get_logb(catalog):
    """Returns the logb"""
    if "logb" in catalog.dtype.names:
        logb = catalog["logb"]
    elif "base_Blendedness_abs" in catalog.dtype.names:  # pipe 7
        logb = np.log10(np.maximum(catalog["base_Blendedness_abs"], 1.0e-6))
    elif "i_blendedness_abs_flux" in catalog.dtype.names:  # s18
        logb = np.log10(np.maximum(catalog["i_blendedness_abs_flux"], 1.0e-6))
    elif "i_blendedness_abs" in catalog.dtype.names:  # s19
        logb = np.log10(np.maximum(catalog["i_blendedness_abs"], 1.0e-6))
    elif "iblendedness_abs_flux" in catalog.dtype.names:  # s15
        logb = np.log10(np.maximum(catalog["iblendedness_abs_flux"], 1.0e-6))
    else:
        logb = _nan_array(len(catalog))
    return logb


def get_logbAll(catalog):
    """Returns the logb"""
    if "base_Blendedness_abs" in catalog.dtype.names:  # pipe 7
        logbA = np.log10(np.maximum(catalog["base_Blendedness_abs"], 1.0e-6))
        logbR = np.log10(np.maximum(catalog["base_Blendedness_raw"], 1.0e-6))
        logbO = np.log10(np.maximum(catalog["base_Blendedness_old"], 1.0e-6))
    elif "i_blendedness_abs_flux" in catalog.dtype.names:  # s18
        logbA = np.log10(np.maximum(catalog["i_blendedness_abs_flux"], 1.0e-6))
        logbR = np.log10(np.maximum(catalog["i_blendedness_raw_flux"], 1.0e-6))
        logbO = np.log10(np.maximum(catalog["i_blendedness_old"], 1.0e-6))
    elif "i_blendedness_abs" in catalog.dtype.names:  # s19
        logbA = np.log10(np.maximum(catalog["i_blendedness_abs"], 1.0e-6))
        logbR = np.log10(np.maximum(catalog["i_blendedness_raw"], 1.0e-6))
        logbO = np.log10(np.maximum(catalog["i_blendedness_old"], 1.0e-6))
    else:
        logbA = _nan_array(len(catalog))
        logbR = _nan_array(len(catalog))
        logbO = _nan_array(len(catalog))
    return logbA, logbR, logbO


def get_sigma_e(catalog):
    """
    This utility returns the hsm_regauss_sigma values for the catalog, without
    imposing any additional flag cuts.
    In the case of GREAT3-like simulations, the noise rescaling factor is
    applied to match the data.
    """
    if "sigma_e" in catalog.dtype.names:
        return catalog["sigma_e"]
    elif "i_hsmshaperegauss_sigma" in catalog.dtype.names:
        sigma_e = catalog["i_hsmshaperegauss_sigma"]
    elif "ishape_hsm_regauss_sigma" in catalog.dtype.names:
        sigma_e = catalog["ishape_hsm_regauss_sigma"]
    elif "ext_shapeHSM_HsmShapeRegauss_sigma" in catalog.dtype.names:
        sigma_e = catalog["ext_shapeHSM_HsmShapeRegauss_sigma"]
    else:
        sigma_e = _nan_array(len(catalog))
    return sigma_e


def get_psf_size(catalog, dtype="fwhm"):
    """This utility gets the PSF size from a data or sims catalog using the
    specified size definition from the second moments matrix.

    Args:
    catalog (ndarray):  Simulation or data catalog
    dtype (str):        Type of psf size measurement in ['trace', 'det',
                        'fwhm'] (default: 'fwhm')
    Returns:
    size (ndarray):     PSF size
    """
    if "base_SdssShape_psf_xx" in catalog.dtype.names:
        psf_mxx = catalog["base_SdssShape_psf_xx"] * 0.168**2.0
        psf_myy = catalog["base_SdssShape_psf_yy"] * 0.168**2.0
        psf_mxy = catalog["base_SdssShape_psf_xy"] * 0.168**2.0
    elif "i_sdssshape_psf_shape11" in catalog.dtype.names:
        psf_mxx = catalog["i_sdssshape_psf_shape11"]
        psf_myy = catalog["i_sdssshape_psf_shape22"]
        psf_mxy = catalog["i_sdssshape_psf_shape12"]
    elif "ishape_sdss_psf_ixx" in catalog.dtype.names:
        psf_mxx = catalog["ishape_sdss_psf_ixx"]
        psf_myy = catalog["ishape_sdss_psf_iyy"]
        psf_mxy = catalog["ishape_sdss_psf_ixy"]
    else:
        psf_mxx = _nan_array(len(catalog))
        psf_myy = _nan_array(len(catalog))
        psf_mxy = _nan_array(len(catalog))

    if dtype == "trace":
        if "traceR" in catalog.dtype.names:
            size = catalog["traceR"]
        else:
            size = np.sqrt(psf_mxx + psf_myy)
    elif dtype == "det":
        if "detR" in catalog.dtype.names:
            size = catalog["detR"]
        else:
            size = (psf_mxx * psf_myy - psf_mxy**2) ** (0.25)
    elif dtype == "fwhm":
        if "fwhm" in catalog.dtype.names:
            size = catalog["fwhm"]
        else:
            size = 2.355 * (psf_mxx * psf_myy - psf_mxy**2) ** (0.25)
    else:
        raise ValueError("Unknown PSF size type: %s" % dtype)
    return size


def get_noi_var(catalog):
    if "noivar" in catalog.dtype.names:  # smallcat
        varNois = catalog["noivar"]
    elif "ivariance" in catalog.dtype.names:  # s18&s19
        varNois = catalog["forced_ivariance"]
    elif "i_variance_value" in catalog.dtype.names:  # s18&s19
        varNois = catalog["i_variance_value"]
    elif "base_Variance_value" in catalog.dtype.names:  # sim
        varNois = catalog["base_Variance_value"]
    else:
        varNois = _nan_array(len(catalog))
    return varNois


def get_gal_ellip(catalog):
    if "e1_regaus" in catalog.dtype.names:  # small catalog
        return catalog["e1_regaus"], catalog["e2_regaus"]
    elif "i_hsmshaperegauss_e1" in catalog.dtype.names:  # catalog
        return catalog["i_hsmshaperegauss_e1"], catalog["i_hsmshaperegauss_e2"]
    elif "ishape_hsm_regauss_e1" in catalog.dtype.names:
        return catalog["ishape_hsm_regauss_e1"], catalog["ishape_hsm_regauss_e2"]
    elif "ext_shapeHSM_HsmShapeRegauss_e1" in catalog.dtype.names:  # S16A
        return (
            catalog["ext_shapeHSM_HsmShapeRegauss_e1"],
            catalog["ext_shapeHSM_HsmShapeRegauss_e2"],
        )
    elif "i_sdssshape_shape11" in catalog.dtype.names:
        mxx = catalog["i_sdssshape_shape11"]
        myy = catalog["i_sdssshape_shape22"]
        mxy = catalog["i_sdssshape_shape12"]
    elif "ishape_sdss_ixx" in catalog.dtype.names:
        mxx = catalog["ishape_sdss_ixx"]
        myy = catalog["ishape_sdss_iyy"]
        mxy = catalog["ishape_sdss_ixy"]
    else:
        raise ValueError("Input catalog does not have required coulmn name")
    return (mxx - myy) / (mxx + myy), 2.0 * mxy / (mxx + myy)


def get_psf_ellip(catalog, return_shear=False):
    """This utility gets the PSF ellipticity (uncalibrated shear) from a data
    or sims catalog.
    """
    if "e1_psf" in catalog.dtype.names:
        return catalog["e1_psf"], catalog["e2_psf"]
    elif "base_SdssShape_psf_xx" in catalog.dtype.names:
        psf_mxx = catalog["base_SdssShape_psf_xx"] * 0.168**2.0
        psf_myy = catalog["base_SdssShape_psf_yy"] * 0.168**2.0
        psf_mxy = catalog["base_SdssShape_psf_xy"] * 0.168**2.0
    elif "i_sdssshape_psf_shape11" in catalog.dtype.names:
        psf_mxx = catalog["i_sdssshape_psf_shape11"]
        psf_myy = catalog["i_sdssshape_psf_shape22"]
        psf_mxy = catalog["i_sdssshape_psf_shape12"]
    elif "ishape_sdss_psf_ixx" in catalog.dtype.names:
        psf_mxx = catalog["ishape_sdss_psf_ixx"]
        psf_myy = catalog["ishape_sdss_psf_iyy"]
        psf_mxy = catalog["ishape_sdss_psf_ixy"]
    else:
        raise ValueError("Input catalog does not have required coulmn name")

    if return_shear:
        return (psf_mxx - psf_myy) / (psf_mxx + psf_myy) / 2.0, psf_mxy / (psf_mxx + psf_myy)
    else:
        return (psf_mxx - psf_myy) / (psf_mxx + psf_myy), 2.0 * psf_mxy / (psf_mxx + psf_myy)


def get_sdss_ellip(catalog, return_shear=False):
    """This utility gets the SDSS ellipticity (uncalibrated shear) from a data
    or sims catalog.
    """
    if "i_sdssshape_shape11" in catalog.dtype.names:  # s19
        psf_mxx = catalog["i_sdssshape_shape11"]
        psf_myy = catalog["i_sdssshape_shape22"]
        psf_mxy = catalog["i_sdssshape_shape12"]
    elif "ishape_sdss_ixx" in catalog.dtype.names:  # s16
        psf_mxx = catalog["ishape_sdss_ixx"]
        psf_myy = catalog["ishape_sdss_iyy"]
        psf_mxy = catalog["ishape_sdss_ixy"]
    elif "base_SdssShape_xx" in catalog.dtype.names:  # pipeline
        psf_mxx = catalog["base_SdssShape_xx"] * 0.168**2.0
        psf_myy = catalog["base_SdssShape_yy"] * 0.168**2.0
        psf_mxy = catalog["base_SdssShape_xy"] * 0.168**2.0
    else:
        raise ValueError("Input catalog does not have required coulmn name")
    if return_shear:
        return (psf_mxx - psf_myy) / (psf_mxx + psf_myy) / 2.0, psf_mxy / (psf_mxx + psf_myy)
    else:
        return (psf_mxx - psf_myy) / (psf_mxx + psf_myy), 2.0 * psf_mxy / (psf_mxx + psf_myy)


def update_wl_cuts(catalog):
    """Update the weak-lensing cuts"""
    catalog["weak_lensing_flag"] = get_wl_cuts(catalog)
    return catalog


def get_wl_cuts(catalog):
    """Returns the weak-lensing cuts"""
    sig_e = get_sigma_e(catalog)
    absE = get_abs_ellip(catalog)
    fwhm = get_psf_size(catalog, "fwhm")
    wlflag = (
        ((get_imag(catalog) - catalog["a_i"]) < 24.5)
        & (absE <= 2.0)
        & (get_res(catalog) >= 0.3)
        & (get_snr(catalog) >= 10.0)
        & (sig_e > 0.0)
        & (sig_e < 0.4)
        & (get_logb(catalog) <= -0.38)
        & (get_imag_A10(catalog) < 25.5)
        & (~np.isnan(fwhm))
    )
    return wlflag


def get_mask_visit_104994(data):
    """In S19A, We found that visit 104994 has tracking errors, but that visit
    contributes to coadds, we remove this region from the catalog level

    Args:
    data (ndarray): input catalog

    Returns:
    mask (ndarray): mask removing the problematic region
    """
    ra, dec = get_radec(data)

    def _calDistanceAngle(a1, d1):
        """Returns the angular distance on sphere
        a1 (ndarray): ra of galaxies
        d1 (ndarray): dec of galaxies
        """
        a2 = 130.43
        d2 = -1.02
        a1_f64 = np.array(a1, dtype=np.float64) * np.pi / 180.0
        d1_f64 = np.array(d1, dtype=np.float64) * np.pi / 180.0
        a2_f64 = np.array(a2, dtype=np.float64) * np.pi / 180.0
        d2_f64 = np.array(d2, dtype=np.float64) * np.pi / 180.0
        return (
            np.arccos(
                np.cos(d1_f64) * np.cos(d2_f64) * np.cos(a1_f64 - a2_f64) + np.sin(d1_f64) * np.sin(d2_f64)
            )
            / np.pi
            * 180.0
        )

    d = _calDistanceAngle(ra, dec)
    mask = (ra > 130.5) & (ra < 131.5) & (dec < -1.5)
    return (d > 0.80) & (~mask)


def get_binarystar_flags(data):
    """Returns the flags for binary stars (|e|>0.8 & logR<1.8-0.1r)

    Args:
    data (ndarray): an hsc-like catalog

    Returns:
    mask (ndarray):  a boolean (True for binary stars)
    """
    absE = get_abs_ellip(data)
    logR = np.log10(get_sdss_size(data))
    rmag = data["forced_r_cmodel_mag"] - data["a_r"]
    mask = absE > 0.8
    a = 1
    b = 10.0
    c = -18.0
    mask = mask & ((a * rmag + b * logR + c) < 0.0)
    return mask


def get_mask_G09_good_seeing(data):
    """Gets the mask for the good-seeing region with large high order PSF shape
    residuals

    Parameters:
    data (ndarray):     HSC shape catalog data

    Returns:
    mm (ndarray):       mask array [if False, in the good-seeing region]
    """
    (ra, dec) = get_radec(data)
    mm = (ra >= 132.5) & (ra <= 140.0) & (dec >= 1.6) & (dec < 5.2)
    mm = ~mm
    return mm


def fix_nan(catalog, key):
    """Fixes NaN entries."""
    x = catalog[key]
    mask = np.isnan(x) | np.isinf(x)
    n_fix = mask.astype(int).sum()
    if n_fix > 0:
        catalog[key][mask] = 0.0
    return


def _nan_array(n):
    """Creates an NaN array."""
    out = np.empty(n)
    out.fill(np.nan)
    return out


def del_colnull(data):
    """Deletes the '_isnull' column from the catalog downloaded from database

    Args:
    data (ndarray):     catalog downloaded from database

    Returns:
    data (ndarray):     catalog after removing '_isnull' column
    """
    colns = data.dtype.names
    colns2 = [cn for cn in colns if "_isnull" not in cn]
    data = data[colns2]
    return data
