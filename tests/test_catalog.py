import os

import fitsio
import numpy as np
import xlens
from pathlib import Path


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
    import pickle
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
