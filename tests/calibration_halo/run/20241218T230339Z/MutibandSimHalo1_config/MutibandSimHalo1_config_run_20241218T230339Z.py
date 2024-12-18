import xlens.sim_pipe.multibandSim
assert type(config) is xlens.sim_pipe.multibandSim.MultibandSimHaloPipeConfig, f"config is of type {type(config).__module__}.{type(config).__name__} instead of xlens.sim_pipe.multibandSim.MultibandSimHaloPipeConfig"

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

# random seed index, 0 <= galId < 10
config.simulator.galId=0

# number of rotations
config.simulator.rotId=0

# random seed for noise, 0 <= noiseId < 10
config.simulator.noiseId=0

# whether to use real PSF
config.simulator.use_real_psf=False

# halo mass
config.simulator.mass=800000000000000.0

# halo concertration
config.simulator.conc=1.0

# halo redshift
config.simulator.z_lens=0.25

# halo ra [arcsec]
config.simulator.ra_lens=0.0

# halo dec [arcsec]
config.simulator.dec_lens=0.0

# source redshift
config.simulator.z_source=1.0

# whether to exclude kappa field
config.simulator.no_kappa=False

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

# name for connection exposure
config.connections.exposure='{inputCoaddName}Coadd_calexp'

# name for connection noiseCorrImage
config.connections.noiseCorrImage='{inputCoaddName}Coadd_systematics_noisecorr'

# name for connection psfImage
config.connections.psfImage='{inputCoaddName}Coadd_systematics_psfcentered'

# name for connection outputExposure
config.connections.outputExposure='{outputCoaddName}_{mode}_rot{rotId}_Coadd_calexp'

# name for connection outputTruthCatalog
config.connections.outputTruthCatalog='{outputCoaddName}_{mode}_rot{rotId}_Coadd_truthCatalog'

# Template parameter used to format corresponding field template parameter
config.connections.inputCoaddName='deep'

# Template parameter used to format corresponding field template parameter
config.connections.outputCoaddName='sim'

# Template parameter used to format corresponding field template parameter
config.connections.mode='0'

# Template parameter used to format corresponding field template parameter
config.connections.rotId='0'

