import galsim
import numpy as np

SIM_INCLUSION_PADDING = 200


def get_objlist(*, galaxy_catalog, survey, star_catalog=None, noise=None):
    """
    get the objlist and shifts, possibly combining the galaxy catalog
    with a star catalog

    Parameters
    ----------
    galaxy_catalog: catalog
        e.g. WLDeblendGalaxyCatalog
    survey: descwl Survey
        For the appropriate band
    star_catalog: catalog
        e.g. StarCatalog
    noise: float
        Needed for star catalog

    Returns
    -------
    objlist, shifts
        objlist is a list of galsim GSObject with transformations applied.
        Shifts is an array with fields dx and dy for each object
    """
    gal_res = galaxy_catalog.get_objlist(
        survey=survey,
    )

    if star_catalog is not None:
        assert noise is not None
        star_res = star_catalog.get_objlist(
            survey=survey, noise=noise,
        )

    else:
        star_res = {
            "star_objlist": None,
            "star_shifts": None,
            "bright_objlist": None,
            "bright_shifts": None,
            "bright_mags": None,
        }
    gal_res.update(star_res)
    return gal_res

res = get_objlist(
    galaxy_catalog=galaxy_catalog,
    survey=survey,
    star_catalog=star_catalog,
    noise=noise_for_gsparams,
)


def get_bright_info_struct():
    dt = [
        ("ra", "f8"),
        ("dec", "f8"),
        ("radius_pixels", "f4"),
        ("has_bleed", bool),
    ]
    return np.zeros(1, dtype=dt)


def get_truth_info_struct():
    dt = [
        ("index", "i4"),
        ("ra", "f8"),
        ("dec", "f8"),
        ("shift_x", "f8"),
        ("shift_y", "f8"),
        ("lensed_shift_x", "f8"),
        ("lensed_shift_y", "f8"),
        ("z", "f8"),
        ("image_x", "f8"),
        ("image_y", "f8"),
        ("prelensed_image_x", "f8"),
        ("prelensed_image_y", "f8"),
        ("prelensed_ra", "f8"),
        ("prelensed_dec", "f8"),
        ("kappa", "f8"),
        ("gamma1", "f8"),
        ("gamma2", "f8"),]
    return np.zeros(1, dtype=dt)


def _roate_pos(pos, theta):
    """Rotates coordinates by an angle theta

    Args:
        pos (PositionD):a galsim position
        theta (float):  rotation angle [rads]
    Returns:
        x2 (ndarray):   rotated coordiantes [x]
        y2 (ndarray):   rotated coordiantes [y]
    """
    x = pos.x
    y = pos.y
    cost = np.cos(theta)
    sint = np.sin(theta)
    x2 = cost * x - sint * y
    y2 = sint * x + cost * y
    return galsim.PositionD(x=x2, y=y2)


def _draw_objects(
    *,
    image,
    objlist,
    shifts,
    redshifts,
    psf,
    draw_method,
    coadd_bbox_cen_gs_skypos,
    rng,
    shear_obj=None,
    theta0=None,
    indexes=None,
):
    """
    draw objects and return the input galaxy catalog.

    Returns
    -------
        truth_info: structured array
        fields are
        index: index in the input galaxy catalog
        ra, dec: sky position of input galaxies
        z: redshift of input galaxies
        image_x, image_y: image position of input galaxies
    """

    wcs = image.wcs
    kw = {}
    if draw_method == "phot":
        kw["maxN"] = 1_000_000
        kw["rng"] = galsim.BaseDeviate(seed=rng.randint(low=0, high=2**30))

    if redshifts is None:
        # set redshifts to -1 if not sepcified
        redshifts = np.ones(len(objlist)) * -1.0

    if indexes is None:
        # set input galaxy indexes to -1 if not sepcified
        indexes = np.ones(len(objlist)) * -1.0

    truth_info = []

    for obj, shift, z, ind in zip(objlist, shifts, redshifts, indexes):

        if theta0 is not None:
            ang = theta0 * galsim.radians
            # rotation on intrinsic galaxies comes before shear distortion
            obj = obj.rotate(ang)
            shift = _roate_pos(shift, theta0)

        if shear_obj is not None:
            distor_res = shear_obj.distort_galaxy(obj, shift, z)
            obj = distor_res["gso"]
            lensed_shift = distor_res["lensed_shift"]
            gamma1 = distor_res["gamma1"]
            gamma2 = distor_res["gamma2"]
            kappa = distor_res["kappa"]
        else:
            lensed_shift = shift
            gamma1, gamma2, kappa = 0.0, 0.0, 0.0

        # Deproject from u,v onto sphere. Then use wcs to get to image pos.
        world_pos = coadd_bbox_cen_gs_skypos.deproject(
            lensed_shift.x * galsim.arcsec,
            lensed_shift.y * galsim.arcsec,
        )

        image_pos = wcs.toImage(world_pos)

        prelensed_world_pos = coadd_bbox_cen_gs_skypos.deproject(
            shift.x * galsim.arcsec,
            shift.y * galsim.arcsec,
        )
        prelensed_image_pos = wcs.toImage(prelensed_world_pos)

        if (
            (image.bounds.xmin - SIM_INCLUSION_PADDING) <
            image_pos.x < (image.bounds.xmax + SIM_INCLUSION_PADDING)
        ) and (
            (image.bounds.ymin - SIM_INCLUSION_PADDING)
            < image_pos.y < (image.bounds.ymax + SIM_INCLUSION_PADDING)
        ):
            local_wcs = wcs.local(image_pos=image_pos)
            convolved_object = galsim.Convolve(obj, psf)
            stamp = convolved_object.drawImage(
                center=image_pos, wcs=local_wcs, method=draw_method, **kw
            )

            b = stamp.bounds & image.bounds
            if b.isDefined():
                image[b] += stamp[b]

        info = get_truth_info_struct()
        info["index"] = (ind,)
        info["ra"] = world_pos.ra / galsim.degrees
        info["dec"] = world_pos.dec / galsim.degrees
        info["shift_x"] = (shift.x,)
        info["shift_y"] = (shift.y,)
        info["lensed_shift_x"] = (lensed_shift.x,)
        info["lensed_shift_y"] = (lensed_shift.y,)
        info["z"] = (z,)
        info["image_x"] = (image_pos.x - 1,)
        info["image_y"] = (image_pos.y - 1,)
        info["gamma1"] = (gamma1,)
        info["gamma2"] = (gamma2,)
        info["kappa"] = (kappa,)
        info["prelensed_image_x"] = (prelensed_image_pos.x - 1,)
        info["prelensed_image_y"] = (prelensed_image_pos.y - 1,)
        info["prelensed_ra"] = (prelensed_world_pos.ra / galsim.degrees,)
        info["prelensed_dec"] = (prelensed_world_pos.dec / galsim.degrees,)

        truth_info.append(info)
    return truth_info


def make_exp(
    *,
    rng,
    gal_list,
    shifts,
    redshifts,
    dim,
    psf,
    shear_obj,
    coadd_bbox_cen_gs_skypos=None,
    rotate=False,
    draw_method="auto",
    theta0=0.0,
    pixel_scale=0.2,
    calib_mag_zero=30.0,
    indexes=None,
    se_wcs=None,
):
    """
    Make an Signle Exposure (SE) observation

    Parameters
    ----------
    rng: numpy.random.RandomState
        The random number generator
    gal_list: list
        List of GSObj
    shifts: array
        List of PositionD representing offsets
    dim: int
        Dimension of image
    psf: GSObject or PowerSpectrumPSF
        the psf
    coadd_bbox_cen_gs_skypos: galsim.CelestialCoord, optional
        The sky position of the center (origin) of the coadd we
        will make, as a galsim object not stack object
    theta0: float
        rotation angle of intrinsic galaxies and positions [for ring test],
        default 0, in units of radians
    pixel_scale: float
        pixel scale of single exposure in arcsec
    calib_mag_zero: float
        magnitude zero point after calibration
    indexes: list
        list of indexes in the input galaxy catalog, default: None
    se_wcs: galsim WCS
        wcs for single exposure, default: None
    Returns
    -------
    gal_array: array of galaxy image
    truth_info: structured array
        fields are
        index: index in the input catalog
        ra, dec: sky position of input galaxies
        z: redshift of input galaxies
        image_x, image_y: image position of input galaxies

    """
    # dims = [int(dim)] * 2
    # cen = (np.array(dims) + 1) / 2
    # se_origin = galsim.PositionD(x=cen[1], y=cen[0])
    # se_wcs = make_se_wcs(
    #     pixel_scale=pixel_scale,
    #     image_origin=se_origin,
    #     world_origin=coadd_bbox_cen_gs_skypos,
    #     dither=dither,
    #     dither_size=dither_size,
    #     rotate=rotate,
    #     rng=rng,
    # )
    # cen = se_wcs.crpix
    # se_origin = galsim.PositionD(x=cen[1], y=cen[0])
    # pixel_area = se_wcs.pixelArea(se_origin)
    # if not (pixel_area - pixel_scale ** 2.0) < pixel_scale ** 2.0 / 100.0:
    #     raise ValueError("The input se_wcs has wrong pixel scale")


    image = galsim.Image(dim, dim, wcs=se_wcs)
    assert shifts is not None
    truth_info = _draw_objects(
        image=image,
        objlist=gal_list,
        shifts=shifts,
        redshifts=redshifts,
        psf=psf,
        draw_method=draw_method,
        coadd_bbox_cen_gs_skypos=coadd_bbox_cen_gs_skypos,
        rng=rng,
        shear_obj=shear_obj,
        theta0=theta0,
        indexes=indexes,
    )
    return {
        "gal_array": image.array,
        "truth_info": truth_info,
    }
