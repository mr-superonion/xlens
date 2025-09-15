import galsim
import lsst
import numpy as np

SIM_INCLUSION_PADDING = 200  # pixels

TRUTH_DTYPE = np.dtype([
    ("index", "i8"),
    ("z", "f8"),
    ("gamma1", "f8"), ("gamma2", "f8"), ("kappa", "f8"),
    ("image_x", "f8"), ("image_y", "f8"),
    ("ra", "f8"), ("dec", "f8"),
    ("prelensed_image_x", "f8"), ("prelensed_image_y", "f8"),
    ("prelensed_ra", "f8"), ("prelensed_dec", "f8"),
])


def make_exp(
    *,
    wcs: lsst.afw.geom.SkyWcs,
    boundary_box: lsst.geom.Box2I,
    gal_list,
    shifts,
    redshifts,
    indexes,
    psf,
    shear_obj,
    draw_method="auto",
    theta0=0.0,
):
    """
    Make an Single Exposure (SE) observation

    Parameters
    ----------
    wcs : SkyWcs-like
        Used to obtain pixel scale via `getPixelScale().asArcseconds()`.
    boundary_box : lsst.geom.Box2I or Box2D
        outer boundary box for patch
    gal_list: list
        List of GSObj
    shifts: array
        List of PositionD representing offsets
    redshifts: array
        List of redshifts
    indexes: list
        list of indexes in the input galaxy catalog
    psf: GSObject or PowerSpectrumPSF
        the psf
    theta0: float
        rotation angle of intrinsic galaxies and positions [for ring test],
        default 0, in units of radians
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
    assert draw_method != "phot", "do not support photon shooting"

    pixel_scale = float(wcs.getPixelScale().asArcseconds())
    image = galsim.ImageF(
        bounds=galsim.BoundsI(
            boundary_box.beginX,
            boundary_box.endX,
            boundary_box.beginY,
            boundary_box.endY,
        ),
        scale=pixel_scale,
    )
    n = len(indexes)
    truth_info = np.zeros(n, dtype=TRUTH_DTYPE)

    for i, (obj, shift, z, ind) in enumerate(
        zip(gal_list, shifts, redshifts, indexes)
    ):
        if theta0:
            obj = obj.rotate(theta0 * galsim.radians)
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

        image_pos = galsim.PositionD(
            lensed_shift.x / pixel_scale, lensed_shift.y / pixel_scale
        )
        prelensed_image_pos = galsim.PositionD(
            shift.x / pixel_scale, shift.y / pixel_scale
        )
        world_pos = wcs.pixelToSky(image_pos.x, image_pos.y)
        prelensed_world_pos = wcs.pixelToSky(
            prelensed_image_pos.x, prelensed_image_pos.y
        )

        if (
            (image.bounds.xmin - SIM_INCLUSION_PADDING) <
            image_pos.x < (image.bounds.xmax + SIM_INCLUSION_PADDING)
        ) and (
            (image.bounds.ymin - SIM_INCLUSION_PADDING)
            < image_pos.y < (image.bounds.ymax + SIM_INCLUSION_PADDING)
        ):
            convolved_object = galsim.Convolve(obj, psf)
            stamp = convolved_object.drawImage(
                center=image_pos, wcs=None, method=draw_method,
                scale=pixel_scale,
            )

            b = stamp.bounds & image.bounds
            if b.isDefined():
                image[b] += stamp[b]

        truth_info["index"][i] = int(ind)
        truth_info["z"][i] = float(z)
        truth_info["gamma1"][i] = gamma1
        truth_info["gamma2"][i] = gamma2
        truth_info["kappa"][i] = kappa
        truth_info["image_x"][i] = image_pos.x
        truth_info["image_y"][i] = image_pos.y
        truth_info["ra"][i] = world_pos.getRa().asDegrees()
        truth_info["dec"][i] = world_pos.getDec().asDegrees()
        truth_info["prelensed_image_x"][i] = prelensed_image_pos.x
        truth_info["prelensed_image_y"][i] = prelensed_image_pos.y
        truth_info["prelensed_ra"][i] = \
            prelensed_world_pos.getRa().asDegrees()
        truth_info["prelensed_dec"][i] = \
            prelensed_world_pos.getDec().asDegrees()
    return {
        "gal_array": image.array,
        "truth_info": truth_info,
    }
