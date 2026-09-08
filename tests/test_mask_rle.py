"""Round-trip tests for the value-aware mask run-length encoding."""
import numpy as np
import pytest

from xlens.utils.image import (
    mask_to_rle,
    mask_to_rle_table,
    rle_table_to_mask,
    rle_to_mask,
)


def _roundtrip(mask):
    rle = mask_to_rle(mask)
    back = rle_to_mask(rle, mask.shape)
    np.testing.assert_array_equal(back, np.asarray(mask).astype(back.dtype))
    return rle


def test_empty_and_full():
    empty = np.zeros((7, 11), dtype=np.int32)
    assert len(_roundtrip(empty)) == 0
    full = np.ones((7, 11), dtype=np.int32)
    rle = _roundtrip(full)
    # one run per row, spanning the full width, exclusive end
    assert len(rle) == 7
    assert (rle["x_start"] == 0).all()
    assert (rle["x_end"] == 11).all()
    assert (rle["value"] == 1).all()


def test_runs_touching_both_edges():
    m = np.zeros((3, 8), dtype=np.int16)
    m[0, 0:3] = 1        # touches the left edge
    m[1, 5:8] = 1        # touches the right edge (end == nx)
    m[2, 0:8] = 1        # full row
    rle = _roundtrip(m)
    assert len(rle) == 3


def test_multiple_runs_per_row_and_single_pixels():
    m = np.zeros((2, 10), dtype=np.int32)
    m[0, [0, 2, 4, 6, 8]] = 1     # alternating single pixels
    m[1, 1:4] = 1
    m[1, 6:9] = 1
    rle = _roundtrip(m)
    assert len(rle) == 7          # 5 singles + 2 runs


def test_random_masks():
    rng = np.random.RandomState(7)
    for frac in (0.01, 0.3, 0.9):
        m = (rng.uniform(size=(64, 57)) < frac).astype(np.int32)
        _roundtrip(m)


def test_random_bitmasks():
    # the combined anacal mask: values 0..3
    rng = np.random.RandomState(11)
    m = rng.randint(0, 4, size=(64, 57)).astype(np.uint8)
    _roundtrip(m)


def test_disk_morphology():
    # a bright-star halo: the shape these masks actually contain
    yy, xx = np.mgrid[0:200, 0:200]
    m = (((yy - 90) ** 2 + (xx - 110) ** 2) < 60 ** 2).astype(np.int32)
    rle = _roundtrip(m)
    # one run per row the disk crosses
    assert len(rle) == len(np.unique(np.nonzero(m)[0]))


def test_values_preserved_and_runs_split_on_change():
    # adjacent runs of different nonzero values must not merge
    m = np.zeros((2, 10), dtype=np.uint8)
    m[0, 1:4] = 1
    m[0, 4:7] = 3        # touches the value-1 run
    m[1, 0:5] = 2
    rle = _roundtrip(m)
    assert len(rle) == 3
    assert sorted(rle["value"].tolist()) == [1, 2, 3]


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        mask_to_rle(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        mask_to_rle(np.full((4, 4), 300, dtype=np.int32))
    with pytest.raises(ValueError):
        mask_to_rle(np.full((4, 4), -1, dtype=np.int32))


def test_table_roundtrip_needs_the_shape_from_the_caller():
    """The table carries runs only; geometry comes from the caller.

    Shape used to travel in ``meta['MASK_NY'/'MASK_NX']``, but the
    butler's ArrowAstropy read path strips ``meta``, so a mask written
    by one task came back undecodable in the next. ``rle_table_to_mask``
    therefore takes the patch outer bbox shape as an argument.
    """
    rng = np.random.RandomState(3)
    m = (rng.uniform(size=(31, 45)) < 0.2).astype(np.int32) * 2
    tab = mask_to_rle_table(m)
    assert "MASK_NY" not in tab.meta and "MASK_NX" not in tab.meta
    back = rle_table_to_mask(tab, m.shape)
    np.testing.assert_array_equal(back, m)


def test_table_without_value_column_decodes_binary():
    # tables written before the value column existed decode as 0/1
    m = np.zeros((6, 9), dtype=np.uint8)
    m[2, 3:7] = 1
    m[4, 0:2] = 1
    tab = mask_to_rle_table(m)
    tab.remove_column("value")
    back = rle_table_to_mask(tab, m.shape)
    np.testing.assert_array_equal(back, m)


def test_table_empty_mask():
    tab = mask_to_rle_table(np.zeros((5, 9), dtype=np.int32))
    assert len(tab) == 0
    back = rle_table_to_mask(tab, (5, 9))
    assert back.shape == (5, 9) and not back.any()


def test_rejects_non_2d():
    with pytest.raises(ValueError):
        mask_to_rle(np.zeros(10, dtype=np.int32))
