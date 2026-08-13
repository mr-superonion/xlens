"""Constant (KernelPsf/FixedKernel) PSFs through the native model path.

The simulator attaches a spatially constant ``KernelPsf`` to its
exposures; these tests pin down that such PSFs load as a native 1x1
``GridPsfModel`` -- so cell drawing threads and per-source forced
measurement run in C++ exactly like a real CoaddPsf, with no DM
``computeImage`` fallback and no error from ``--psf-model-type object``.
"""
import anacal
import numpy as np
import pytest

import lsst.afw.math as afwMath
import lsst.geom as lsst_geom
from lsst.afw.image import ImageD
from lsst.meas.algorithms import KernelPsf

from xlens.utils.image import make_object_psf, resize_array
from xlens.utils.image.psf import try_native_coadd_model

NPIX = 64


def _gauss_stamp(n=45, sigma=3.0):
    yy, xx = np.mgrid[0:n, 0:n] - (n - 1) / 2.0
    st = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return st / st.sum()


def _kernel_psf(stamp):
    img = ImageD(stamp.shape[1], stamp.shape[0])
    img.array[:, :] = stamp
    return KernelPsf(afwMath.FixedKernel(img))


@pytest.fixture()
def psf_and_bbox():
    stamp = _gauss_stamp()
    bbox = lsst_geom.Box2I(
        lsst_geom.Point2I(100, 200), lsst_geom.Extent2I(500, 500)
    )
    return _kernel_psf(stamp), stamp, bbox


def test_native_model_loads_and_draws_everywhere(psf_and_bbox):
    psf, stamp, bbox = psf_and_bbox
    model = try_native_coadd_model(psf, bbox)
    assert model is not None
    expect = resize_array(stamp, (NPIX, NPIX))
    # constant: the same stamp at any position, inside the bbox or out
    # (atol: DM renormalizes the kernel on evaluation, ~1 ulp)
    for x, y in [(100.0, 200.0), (350.5, 450.5), (599.0, 699.0), (-50.0, 9000.0)]:
        np.testing.assert_allclose(
            model.draw(x, y, NPIX), expect, rtol=0, atol=1e-14
        )


def test_make_object_psf_accepts_constant_psf(psf_and_bbox):
    psf, stamp, bbox = psf_and_bbox
    obj = make_object_psf(psf, npix=NPIX, lsst_bbox=bbox)
    assert obj.native_model is not None
    # local coordinates, like LsstPsf
    np.testing.assert_allclose(
        obj.draw(10.0, 20.0), resize_array(stamp, (NPIX, NPIX))
    )


def test_spatially_varying_kernel_is_rejected(psf_and_bbox):
    _, _, bbox = psf_and_bbox
    # a LinearCombinationKernel with spatial variation must NOT be
    # frozen at one position
    n = 45
    basis = [
        afwMath.FixedKernel(ImageD(np.ascontiguousarray(_gauss_stamp(n, s))))
        for s in (2.0, 4.0)
    ]
    spatial = afwMath.PolynomialFunction2D(1)
    kernel = afwMath.LinearCombinationKernel(basis, [spatial, spatial])
    kernel.setSpatialParameters([[0.5, 1e-3, 0.0], [0.5, -1e-3, 0.0]])
    assert kernel.isSpatiallyVarying()
    assert try_native_coadd_model(KernelPsf(kernel), bbox) is None


def test_forced_measurement_matches_constant_psf_array(psf_and_bbox):
    """Per-source native drawing == constant psf_array measurement.

    Tolerances, not bit equality: DM renormalizes the kernel on
    evaluation (~1 ulp on the stamp), and the native path FFTs the
    drawn per-source stamp instead of reusing the constant one.
    """
    psf, stamp, bbox = psf_and_bbox
    rng = np.random.RandomState(5)
    scale = 0.168
    ny = nx = 200
    psf_use = resize_array(stamp, (NPIX, NPIX))

    gal = rng.normal(0, 0.1, size=(ny, nx)).astype(np.float32)
    positions = [(60.0, 70.0), (130.0, 120.0)]
    for x, y in positions:
        gal[int(y) - 22:int(y) + 23, int(x) - 22:int(x) + 23] += (
            200.0 * _gauss_stamp(45, 4.0)
        ).astype(np.float32)

    det = np.zeros(len(positions), dtype=[("y", "f8"), ("x", "f8")])
    det["x"] = [p[0] for p in positions]
    det["y"] = [p[1] for p in positions]
    config = anacal.fpfs.FpfsConfig(
        npix=NPIX, sigma_shapelets1=0.52, sigma_shapelets2=-1
    )

    common = dict(
        fpfs_config=config,
        pixel_scale=scale,
        mag_zero=27.0,
        noise_variance=0.01,
        gal_array=gal,
        psf_array=psf_use,
        detection=det,
    )
    cat_const = anacal.fpfs.process_image(**common)
    obj = make_object_psf(psf, npix=NPIX, lsst_bbox=bbox)
    mask_value = np.zeros(len(det), dtype=np.int32)
    cat_native = anacal.fpfs.process_image(
        **common, psf_object=obj, mask_value=mask_value,
    )

    assert (mask_value == 0).all()   # constant model covers everything
    for name in cat_const.dtype.names:
        np.testing.assert_allclose(
            cat_const[name], cat_native[name],
            rtol=1e-9, atol=1e-11, err_msg=name,
        )
