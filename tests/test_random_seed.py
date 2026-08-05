import hashlib

import numpy as np
import pytest

from xlens.utils.random import get_noise_seed


def _historical_seed(galaxy_seed, noiseId, rotId, band, is_sim):
    """The pre-fix encoding: every int as one uint32, strings via sha256."""
    parts = []
    for item in [galaxy_seed, noiseId, rotId, band, int(is_sim)]:
        if isinstance(item, int):
            parts.append(np.uint32(item))
        else:
            h = hashlib.sha256(str(item).encode("utf-8")).digest()
            parts.extend(np.frombuffer(h, dtype=np.uint32))
    seed_data = np.array(parts, dtype=np.uint32)
    return np.random.SeedSequence(seed_data).generate_state(1)[0]


def test_small_seed_unchanged():
    # Values below 2**32 must keep the historical single-uint32 encoding,
    # so every sim seed ever used reproduces the same noise.
    for galaxy_seed in (0, 12345, 2**32 - 1):
        assert get_noise_seed(
            galaxy_seed=galaxy_seed, band="i", is_sim=True
        ) == _historical_seed(galaxy_seed, 0, 0, "i", True)


def test_large_seed_no_overflow():
    # Packed DM catalog_ids exceed 2**32; np.uint32() alone raises
    # OverflowError on numpy >= 2.
    big = 600_000_000_000_000_000  # DP1-scale packed catalog_id
    s = get_noise_seed(galaxy_seed=big, band="i")
    assert 0 <= int(s) < 2**32

    # All 64 bits must matter: packed catalog_ids differ only in the high
    # bits (their low bits are counter padding), so ids 2**32 apart must
    # still get distinct seeds.
    assert s != get_noise_seed(galaxy_seed=big + 2**32, band="i")


@pytest.mark.parametrize(
    "override",
    [
        {"band": "r"},
        {"survey": "hsc"},
        {"noiseId": 3},
        {"rotId": 1},
        {"is_sim": True},
    ],
)
def test_seed_components_are_mixed(override):
    base = dict(galaxy_seed=42, band="i", survey="lsst")
    assert get_noise_seed(**base) != get_noise_seed(**{**base, **override})
