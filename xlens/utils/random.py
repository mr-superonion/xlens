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
    """
    Generate a unique, reproducible 32-bit integer seed based on galaxy
    parameters.

    Args:
        galaxy_seed (int):
            Base seed or unique identifier for the galaxy.
        noiseId (int, optional):
            Noise realization ID. Defaults to 0.
        rotId (int, optional):
            Rotation ID for galaxy orientation. Defaults to 0.
        band (str, optional):
            Photometric band identifier (e.g., "g", "r", "i", "z", "y").
            Defaults to "i".
        is_sim (bool or numpy.bool_, optional):
            Flag indicating whether the galaxy is from a simulation (`True`) or
            real data (`False`). Internally converted to an integer (1 for
            True, 0 for False). Defaults to False.

    Returns:
        int:
            A reproducible 32-bit unsigned integer seed derived from the
            provided parameters. Changing any parameter will produce a
            different seed.

    Notes:
        - This method ensures low collision probability by hashing non-integer
          values and combining them with integer inputs before creating the
          seed.
        - For extremely large-scale uniqueness requirements (billions of
          seeds), consider using 64-bit output for extra safety.
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
