import anacal
import numpy as np
from numpy.lib import recfunctions as rfn
from xlens.utils.columns import (
    DETECTION_KEEP_COLUMNS,
    MODEL_COLUMNS,
    merge_structured,
    select_detection_columns,
)


def test_model_columns_kept_with_prefix():
    cat = anacal.table.make_catalog_empty(np.zeros(3), np.zeros(3))
    for k, name in enumerate(cat.dtype.names):
        cat[name] = k + 1
    out = select_detection_columns(cat)
    names = out.dtype.names
    assert set(names) == set(DETECTION_KEEP_COLUMNS) | set(
        MODEL_COLUMNS.values()
    )
    for src, dst in MODEL_COLUMNS.items():
        assert src not in names or src in DETECTION_KEEP_COLUMNS
        np.testing.assert_array_equal(out[dst], cat[src])
    for name in ("model_flux", "model_dmxx_dg1", "model_dmxy_dg2"):
        assert name in names


def test_merge_structured_matches_merge_arrays():
    import pytest

    cat = anacal.table.make_catalog_empty(np.zeros(4), np.zeros(4))
    for k, name in enumerate(cat.dtype.names):
        cat[name] = k + 1
    extra = np.zeros(4, dtype=[("a", "f4"), ("b", "i8"), ("c", "?")])
    extra["a"] = 1.5
    extra["b"] = np.arange(4)
    extra["c"] = [True, False, True, False]
    ref = rfn.merge_arrays([cat, extra], flatten=True)
    out = merge_structured([cat, extra])
    assert out.dtype == ref.dtype  # packed, same names, order, types
    for name in ref.dtype.names:
        np.testing.assert_array_equal(out[name], ref[name])
    with pytest.raises(ValueError):
        merge_structured([cat, np.zeros(4, dtype=[("ra", "f8")])])
    with pytest.raises(ValueError):
        merge_structured([cat, np.zeros(3, dtype=[("z", "f8")])])
