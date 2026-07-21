# This file is part of xlens.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import hashlib

import numpy as np

# 90 degree rotation of the scene
num_rot = 2
gal_seed_base = 10


def get_noise_seed(*, galaxy_seed, noiseId=0, rotId=0, band: None | str = "i", survey: None | str = None, is_sim=False):
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
        Identifier describing the desired noise realisation.  Default ``0``.
    rotId : int, optional
        Identifier describing the rotation realisation.  Defaults to ``0``.
    band : str, optional
        Physical photometric band label (``"g"``, ``"r"``, ``"i"``, ``"z"``,
        ``"y"``).  Defaults to ``"i"``.
    survey : str, optional
        Survey name (``"lsst"``, ``"hsc"``, ``"euclid"``).  Mixed into the seed
        so the same physical band in different surveys (``lsst_g`` vs ``hsc_g``)
        draws INDEPENDENT noise.  ``None`` reproduces the pre-survey seed.
    is_sim : bool, optional
        Flag that indicates whether the galaxy originates from a simulation
        (``True``) or observations (``False``).  Defaults to ``False``.

    Returns
    -------
    int
        Unsigned 32-bit integer seed suitable for initialising NumPy random
        generators.
    """
    # ``survey`` is included only when given, so passing None reproduces the
    # historical (survey-agnostic) seed; galaxy_seed itself is never survey/band
    # dependent (it labels the galaxy, identical across bands/surveys).
    mixed_list = [galaxy_seed, noiseId, rotId, band, int(is_sim)]
    if survey is not None:
        mixed_list.append(survey)
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
