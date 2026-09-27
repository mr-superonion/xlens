import anacal
import numpy as np
from xlens.utils.columns import (
    DETECTION_KEEP_COLUMNS,
    MODEL_COLUMNS,
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
