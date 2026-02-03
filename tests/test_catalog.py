import os
from pathlib import Path

import fitsio
import numpy as np

import xlens
import pickle

from xlens.catalog import measure_shear, ShearEstimator


def test_pz_point_estimates():
    # Set up the configuration
    DATA_DIR = Path(__file__).parent / "data"
    pfname = os.path.join(DATA_DIR, "pz_pdfs_test.fits")
    pdfs = fitsio.read(pfname)
    out = xlens.catalog.redshift.get_point_estimates_from_pdfs(pdfs)
    key_target = ['zmode', 'z025', 'z160', 'z500', 'z840', 'z975', 'zbest']
    assert list(out.keys()) == key_target
    zbest_target = np.array([4.28967696, 0.72257506, 1.52362052])
    np.testing.assert_almost_equal(
        out["zbest"],
        zbest_target,
    )
    zmode_target = np.array([0.34, 0.73, 1.53])
    np.testing.assert_almost_equal(
        out["zmode"],
        zmode_target,
    )
    return


def test_pz():
    # Set up the configuration
    DATA_DIR = Path(__file__).parent / "data"
    fname = os.path.join(DATA_DIR, "catalog.fits")
    catalog = fitsio.read(fname)
    model_fname = os.path.join(DATA_DIR, "model_inform_fzboost.pkl")
    with open(model_fname, "rb") as f:
        pz_obj = pickle.load(f)
    fzbobj = xlens.catalog.redshift.flexzboostEstimator(pz_obj)
    out = fzbobj.get_z(
        catalog,
        mag_zero=30.0,
        flux_name="gauss2",
        bands="ugrizy",
        ref_band="i",
    )
    key_target = ['zmode', 'z025', 'z160', 'z500', 'z840', 'z975', 'zbest']
    assert list(out.keys()) == key_target
    zbest_target = np.array([4.28967696, 0.72257506, 1.52362052])
    np.testing.assert_almost_equal(
        out["zbest"],
        zbest_target,
    )
    zmode_target = np.array([0.34, 0.73, 1.53])
    np.testing.assert_almost_equal(
        out["zmode"],
        zmode_target,
    )
    return


def test_measure_shear_consistency():
    DATA_DIR = Path(__file__).parent / "data"
    fname = os.path.join(DATA_DIR, "catalog.fits")
    catalog = fitsio.read(fname)
    model_fname = os.path.join(DATA_DIR, "model_inform_fzboost.pkl")
    with open(model_fname, "rb") as f:
        pz_obj = pickle.load(f)
    fzbobj = xlens.catalog.redshift.flexzboostEstimator(pz_obj)

    emax = 0.3
    zbounds = [0.3, 0.6, 0.9, 1.2]
    dg = 0.02
    target = "g1g2"
    do_correction = True
    z_width95_max = 2.75
    mag_zero = 30.0
    bands = "ugrizy"
    ref_band = "i"
    trace_min = 0.05

    mag_max_list = [27.5, 27.0, 26.0, 24.6, 25.0, 25.5]
    flux_min = {
        b: 10.0 ** ((mag_zero - mmax) / 2.5)
        for b, mmax in zip(bands, mag_max_list)
    }
    mag_max = {b: mmax for b, mmax in zip(bands, mag_max_list)}

    shear_kwargs = {
        "flux_min": flux_min,
        "z_estimator": fzbobj,
        "emax": emax,
        "trace_min": trace_min,
        "zbounds": zbounds,
        "dg": dg,
        "target": target,
        "do_correction": do_correction,
        "z_width95_max": z_width95_max,
        "mag_zero": mag_zero,
        "flux_name": "gauss2",
        "bands": bands,
        "ref_band": ref_band,
    }
    out1 = measure_shear(
        src=catalog,
        **shear_kwargs,
    )
    shear_obj = ShearEstimator(
        mag_max=mag_max,
        emax=emax,
        trace_min=trace_min,
        mag_zero=mag_zero,
        flux_name="gauss2",
        bands=bands,
        ref_band=ref_band,
        z_estimator=fzbobj,
        zbounds=zbounds,
        z_width95_max=z_width95_max,
        dg=dg,
    )
    out2 = shear_obj.measure_shear(catalog, target=target)

    np.testing.assert_almost_equal(
        out1["e1"],
        out2["e1"],
    )
    np.testing.assert_almost_equal(
        out1["e2"],
        out2["e2"],
    )
    np.testing.assert_almost_equal(
        out1["r1"],
        out2["r1"] + out2["r1_det"],
    )
    np.testing.assert_almost_equal(
        out1["r2"],
        out2["r2"] + out2["r2_det"],
    )
    np.testing.assert_almost_equal(
        out1["r1_sel"],
        out2["r1_sel"],
    )
    np.testing.assert_almost_equal(
        out1["r2_sel"],
        out2["r2_sel"],
    )
    return
