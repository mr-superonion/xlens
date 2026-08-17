"""The single ``n_mask_base_max`` knob: validation and the row cut.

One config field drives the whole mask cut -- the anacal/fpfs subtasks
skip those sources in C++ (so nothing is measured that is about to be
thrown away) and the finalized per-patch catalog drops the rows.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from lsst.pex.config import FieldValidationError

from xlens.processor.measure_base import (
    AnacalMeasureConfigBase,
    AnacalMeasureTaskBase,
)


def _config():
    cfg = AnacalMeasureConfigBase()
    # unrelated required field; the base validate() checks it too
    cfg.fpfs.sigma_shapelets1 = 0.52
    return cfg


def _catalog(n_mask_base, wsel=None, is_primary=None):
    n = len(n_mask_base)
    cat = np.zeros(
        n,
        dtype=[
            ("n_mask_base", "f4"),
            ("wsel", "f8"),
            ("is_primary", "?"),
        ],
    )
    cat["n_mask_base"] = n_mask_base
    cat["wsel"] = 1.0 if wsel is None else wsel
    cat["is_primary"] = True if is_primary is None else is_primary
    return cat


def _select(cat, n_mask_base_max):
    stub = SimpleNamespace(
        config=SimpleNamespace(n_mask_base_max=n_mask_base_max),
        log=logging.getLogger("test_n_mask_base_cut"),
    )
    return AnacalMeasureTaskBase._select_rows(stub, cat)


def test_default_is_one():
    assert _config().n_mask_base_max == 1.0


@pytest.mark.parametrize("value", [0.035, 0.5, 1.0])
def test_valid_range_accepted(value):
    cfg = _config()
    cfg.n_mask_base_max = value
    cfg.validate()


@pytest.mark.parametrize("value", [0.0, -0.1, 1.0 + 1e-6, 35.0])
def test_out_of_range_rejected(value):
    """n_mask_base is a FRACTION: (0, 1] is the whole valid range.

    35 is the pre-rename threshold (a x1000 density); catching it here
    is the point -- carried over silently it would keep everything.
    """
    cfg = _config()
    cfg.n_mask_base_max = value
    with pytest.raises(FieldValidationError, match="must be in"):
        cfg.validate()


def test_default_drops_only_the_unusable_rows():
    # 1.0 is both "footprint entirely masked" and the psf_invalid
    # sentinel; everything below it survives at the default.
    cat = _catalog([0.0, 0.03, 0.5, 0.999, 1.0])
    out = _select(cat, 1.0)
    assert len(out) == 4
    assert out["n_mask_base"].max() < 1.0


def test_lower_threshold_drops_partially_masked():
    cat = _catalog([0.0, 0.03, 0.04, 1.0])
    out = _select(cat, 0.035)
    np.testing.assert_allclose(out["n_mask_base"], [0.0, 0.03], rtol=1e-6)


def test_wsel_and_is_primary_also_cut():
    cat = _catalog(
        [0.0, 0.0, 0.0, 0.0],
        wsel=[1.0, 0.0, 1.0, 1e-9],
        is_primary=[True, True, False, True],
    )
    out = _select(cat, 1.0)
    assert len(out) == 1
    # only the row that is primary AND carries a real selection weight
    assert out["wsel"][0] == 1.0


def test_no_copy_when_nothing_is_dropped():
    cat = _catalog([0.0, 0.1, 0.2])
    out = _select(cat, 1.0)
    # the all-keep fast path returns the input itself, not a copy
    assert out is cat
