import xlens.process_pipe.fpfs_joint
assert type(config) is xlens.process_pipe.fpfs_joint.FpfsJointPipeConfig, f"config is of type {type(config).__module__}.{type(config).__name__} instead of xlens.process_pipe.fpfs_joint.FpfsJointPipeConfig"

import lsst.meas.base._id_generator
import lsst.pipe.base.config
import lsst.skymap.packers
import xlens.processor.fpfs
# Flag to enable/disable saving of log output for a task, enabled by default.
config.saveLogOutput=True

# number of pixels in stamp
config.fpfs.npix=64

# Sources to be removed if too close to boundary
config.fpfs.bound=32

# Shapelet's Gaussian kernel size for detection
config.fpfs.sigma_arcsec=0.52

# Shapelet's Gaussian kernel size for measurement
config.fpfs.sigma_arcsec1=-1.0

# Shapelet's Gaussian kernel size for the second measurement
config.fpfs.sigma_arcsec2=-1.0

# Shapelet's Gaussian kernel size for the second measurement
config.fpfs.snr_min=12.0

# Shapelet's Gaussian kernel size for the second measurement
config.fpfs.r2_min=0.1

# peak detection threshold
config.fpfs.pthres=0.12

# threshold to determine the maximum k in Fourier space
config.fpfs.kmax_thres=1e-12

# whether to use average PSF over the exposure
config.fpfs.use_average_psf=True

# whether to doulbe the noise for noise bias correction
config.fpfs.do_adding_noise=False

# whether to compute detection mode
config.fpfs.do_compute_detect_weight=True

# Mask planes used to reject bad pixels.
config.fpfs.badMaskPlanes=['BAD', 'SAT', 'CR']

# Noise realization id
config.fpfs.noiseId=0

# rotation id
config.fpfs.rotId=0

# Size of PSF cache
config.psfCache=100

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

# name for connection exposure
config.connections.exposure='{coaddName}Coadd_calexp'

# name for connection noise_corr
config.connections.noise_corr='{coaddName}Coadd_systematics_noisecorr'

# name for connection joint_catalog
config.connections.joint_catalog='{coaddName}Coadd_anacal_joint'

# Template parameter used to format corresponding field template parameter
config.connections.coaddName='sim_0_rot1_'

