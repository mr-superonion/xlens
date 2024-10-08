import xlens.sim_pipe.shear_multiband
assert type(config) is xlens.sim_pipe.shear_multiband.MultibandSimShearPipeConfig, f"config is of type {type(config).__module__}.{type(config).__name__} instead of xlens.sim_pipe.shear_multiband.MultibandSimShearPipeConfig"

import lsst.meas.base._id_generator
import lsst.pipe.base.config
import lsst.skymap.packers
import xlens.simulator.multiband
# Flag to enable/disable saving of log output for a task, enabled by default.
config.saveLogOutput=True

# Name of the survey
config.simulator.survey_name='hsc'

# Layout of the galaxy distribution
config.simulator.layout='random'

# The ratio to extend for the size of simulated image
config.simulator.extend_ratio=1.06

# whether to include pixel masks in the simulation
config.simulator.include_pixel_masks=False

# whether to include stars in the simulation
config.simulator.include_stars=False

# Whether to draw image noise in the simulation
config.simulator.draw_image_noise=False

# number of rotations
config.simulator.nrot=2

# number of rotations
config.simulator.irot=0

# boundary list of the redshift
config.simulator.z_bounds=[-0.01, 20.0]

# number of rotations
config.simulator.mode=1

# the shear component to test
config.simulator.test_target='g1'

# absolute value of the shear
config.simulator.test_value=0.02

# Identifier for a data release or other version to embed in generated IDs. Zero is reserved for IDs with no embedded release identifier.
config.idGenerator.release_id=0

# Number of (contiguous, starting from zero) `release_id` values to reserve space for. One (not zero) is used to reserve no space.
config.idGenerator.n_releases=1

# Mapping from band name to integer to use in the packed ID. The default (None) is to use a hard-coded list of common bands; pipelines that can enumerate the set of bands they are likely to see should override this.
config.idGenerator.packer.bands=None

# Number of bands to reserve space for. If zero, bands are not included in the packed integer at all. If `None`, the size of 'bands' is used.
config.idGenerator.packer.n_bands=0

# Number of tracts, or, more precisely, one greater than the maximum tract ID.Default (None) obtains this value from the skymap dimension record.
config.idGenerator.packer.n_tracts=None

# Number of patches per tract, or, more precisely, one greater than the maximum patch ID.Default (None) obtains this value from the skymap dimension record.
config.idGenerator.packer.n_patches=None

# name for connection skymap
config.connections.skymap='skyMap'

# name for connection outputExposure
config.connections.outputExposure='{outputCoaddName}Coadd_calexp{simType}_{mode}_rot{irot}'

# name for connection outputTruthCatalog
config.connections.outputTruthCatalog='{outputCoaddName}Coadd_truthCatalog{simType}_{mode}_rot{irot}'

# Template parameter used to format corresponding field template parameter
config.connections.inputCoaddName='deep'

# Template parameter used to format corresponding field template parameter
config.connections.outputCoaddName='deep'

# Template parameter used to format corresponding field template parameter
config.connections.psfType='moffat'

# Template parameter used to format corresponding field template parameter
config.connections.simType='_test'

# Template parameter used to format corresponding field template parameter
config.connections.mode='1'

# Template parameter used to format corresponding field template parameter
config.connections.irot='0'

