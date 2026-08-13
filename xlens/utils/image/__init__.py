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

"""Image utilities for working with LSST exposures and PSF models.

The implementation lives in six focused submodules; every public name is
re-exported here so existing ``xlens.utils.image.<name>`` imports keep
working:

- :mod:`~xlens.utils.image.masks`   -- bad-mask planes, per-band + union masks
- :mod:`~xlens.utils.image.noise`   -- noise generation, variance, noise plane
- :mod:`~xlens.utils.image.psf`     -- PSF wrappers and PSF-stamp preparation
- :mod:`~xlens.utils.image.prepare` -- exposure / patch-coadd prepare_data
- :mod:`~xlens.utils.image.cells`   -- cell-coadd prepare_data
- :mod:`~xlens.utils.image.hsm`     -- PSF HSM-moment utilities
"""

from . import cells, hsm, masks, noise, prepare, psf  # noqa: F401
from .cells import (  # noqa: F401
    prepare_data_one_cell,
    prepare_data_one_cell_multiband,
)
from .hsm import (  # noqa: F401
    PsfHsmContext,
    broadcast_psf_hsm_moments,
    build_psf_hsm_context,
    default_psf_hsm_plugin_config,
    make_psf_stamp_exposure,
    measure_psf_hsm_moments,
)
from .masks import (  # noqa: F401
    badMaskDefault,
    mask_to_rle,
    mask_to_rle_table,
    prepare_mask,
    rle_table_origin,
    rle_table_to_mask,
    rle_to_mask,
)
from .noise import (  # noqa: F401
    estimate_noise_variance,
    generate_pure_noise,
    prepare_noise_array,
    rotate_noise_corr,
)
from .prepare import (  # noqa: F401
    _stack_bands,
    combine_sim_exposures,
    get_cells,
    get_cells_multiband,
    prepare_data,
    prepare_data_multiband,
    prepare_detection,
)
from .psf import (  # noqa: F401
    GridPsf,
    LsstPsf,
    make_object_psf,
    get_psf_array,
    prepare_psf_array,
    prepare_psf_array_cell,
    resize_array,
    stack_psfs_cells,
    subpixel_shift,
    truncate_square,
)
