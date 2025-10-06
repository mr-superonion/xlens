import hashlib

import numpy as np

# 90 degree rotation of the scene
num_rot = 2
gal_seed_base = 10

def get_noise_seed(
    *,
    galaxy_seed,
    noiseId=0,
    rotId=0,
    band="i",
    is_sim=False
):
    """Generate a stable pseudo-random seed for noise realisations.

    The function mixes deterministic galaxy identifiers with optional
    meta-data, hashes the values into a uniform byte representation, and
    derives a 32-bit integer seed.  The resulting seed is reproducible for a
    given combination of inputs and has a vanishingly small collision
    probability within typical survey data sets.

    Parameters
    ----------
    galaxy_seed : int
        Base integer identifier that uniquely labels the galaxy.
    noiseId : int, optional
        Identifier describing the desired noise realisation.  Defaults to ``0``.
    rotId : int, optional
        Identifier describing the rotation realisation.  Defaults to ``0``.
    band : str, optional
        Photometric band label (``"g"``, ``"r"``, ``"i"``, ``"z"``, ``"y"``).
        Defaults to ``"i"``.
    is_sim : bool, optional
        Flag that indicates whether the galaxy originates from a simulation
        (``True``) or observations (``False``).  Defaults to ``False``.

    Returns
    -------
    int
        Unsigned 32-bit integer seed suitable for initialising NumPy random
        generators.
    """
    mixed_list = [
        galaxy_seed, noiseId, rotId, band, int(is_sim)
    ]
    parts = []
    for item in mixed_list:
        if isinstance(item, int):
            # Directly store integer as uint32
            parts.append(np.uint32(item))
        else:
            # Convert non-int (e.g., str) into uint32s via hashing
            h = hashlib.sha256(str(item).encode("utf-8")).digest()
            arr = np.frombuffer(h, dtype=np.uint32)
            parts.extend(arr)

    # Combine all parts into one uint32 array
    seed_data = np.array(parts, dtype=np.uint32)
    return np.random.SeedSequence(seed_data).generate_state(1)[0]
