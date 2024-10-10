import xlens.process_pipe.fpfs_multiband
assert type(config) is xlens.process_pipe.fpfs_multiband.FpfsMultibandPipeConfig, f"config is of type {type(config).__module__}.{type(config).__name__} instead of xlens.process_pipe.fpfs_multiband.FpfsMultibandPipeConfig"

import lsst.meas.algorithms.detection
import lsst.meas.algorithms.subtractBackground
import lsst.meas.base._id_generator
import lsst.meas.deblender.sourceDeblendTask
import lsst.pipe.base.config
import lsst.skymap.packers
import xlens.processor.fpfs
# Flag to enable/disable saving of log output for a task, enabled by default.
config.saveLogOutput=True

# whether to do detection
config.do_dm_detection=False

# detected sources with fewer than the specified number of pixels will be ignored
config.detection.minPixels=1

# Grow pixels as isotropically as possible? If False, use a Manhattan metric instead.
config.detection.isotropicGrow=True

# Grow all footprints at the same time? This allows disconnected footprints to merge.
config.detection.combinedGrow=True

# Grow detections by nSigmaToGrow * [PSF RMS width]; if 0 then do not grow
config.detection.nSigmaToGrow=2.4

# Grow detections to set the image mask bits, but return the original (not-grown) footprints
config.detection.returnOriginalFootprints=False

# Threshold for detecting footprints; exact meaning and units depend on thresholdType.
config.detection.thresholdValue=5.0

# Multiplier on thresholdValue for whether a source is included in the output catalog. For example, thresholdValue=5, includeThresholdMultiplier=10, thresholdType='pixel_stdev' results in a catalog of sources at >50 sigma with the detection mask and footprints including pixels >5 sigma.
config.detection.includeThresholdMultiplier=1.0

# Specifies the meaning of thresholdValue.
config.detection.thresholdType='pixel_stdev'

# Specifies whether to detect positive, or negative sources, or both.
config.detection.thresholdPolarity='positive'

# Fiddle factor to add to the background; debugging only
config.detection.adjustBackground=0.0

# Estimate the background again after final source detection?
config.detection.reEstimateBackground=True

# type of statistic to use for grid points
config.detection.background.statisticsProperty='MEANCLIP'

# behaviour if there are too few points in grid for requested interpolation style
config.detection.background.undersampleStyle='REDUCE_INTERP_ORDER'

# how large a region of the sky should be used for each background point
config.detection.background.binSize=128

# Sky region size to be used for each background point in X direction. If 0, the binSize config is used.
config.detection.background.binSizeX=0

# Sky region size to be used for each background point in Y direction. If 0, the binSize config is used.
config.detection.background.binSizeY=0

# how to interpolate the background values. This maps to an enum; see afw::math::Background
config.detection.background.algorithm='AKIMA_SPLINE'

# Names of mask planes to ignore while estimating the background
config.detection.background.ignoredPixelMask=['BAD', 'EDGE', 'DETECTED', 'DETECTED_NEGATIVE', 'NO_DATA']

# Ignore NaNs when estimating the background
config.detection.background.isNanSafe=False

# Use Approximate (Chebyshev) to model background.
config.detection.background.useApprox=True

# Approximation order in X for background Chebyshev (valid only with useApprox=True)
config.detection.background.approxOrderX=6

# Approximation order in Y for background Chebyshev (valid only with useApprox=True)
config.detection.background.approxOrderY=-1

# Use inverse variance weighting in calculation (valid only with useApprox=True)
config.detection.background.weighting=True

# type of statistic to use for grid points
config.detection.tempLocalBackground.statisticsProperty='MEANCLIP'

# behaviour if there are too few points in grid for requested interpolation style
config.detection.tempLocalBackground.undersampleStyle='REDUCE_INTERP_ORDER'

# how large a region of the sky should be used for each background point
config.detection.tempLocalBackground.binSize=64

# Sky region size to be used for each background point in X direction. If 0, the binSize config is used.
config.detection.tempLocalBackground.binSizeX=0

# Sky region size to be used for each background point in Y direction. If 0, the binSize config is used.
config.detection.tempLocalBackground.binSizeY=0

# how to interpolate the background values. This maps to an enum; see afw::math::Background
config.detection.tempLocalBackground.algorithm='AKIMA_SPLINE'

# Names of mask planes to ignore while estimating the background
config.detection.tempLocalBackground.ignoredPixelMask=['BAD', 'EDGE', 'DETECTED', 'DETECTED_NEGATIVE', 'NO_DATA']

# Ignore NaNs when estimating the background
config.detection.tempLocalBackground.isNanSafe=False

# Use Approximate (Chebyshev) to model background.
config.detection.tempLocalBackground.useApprox=False

# Approximation order in X for background Chebyshev (valid only with useApprox=True)
config.detection.tempLocalBackground.approxOrderX=6

# Approximation order in Y for background Chebyshev (valid only with useApprox=True)
config.detection.tempLocalBackground.approxOrderY=-1

# Use inverse variance weighting in calculation (valid only with useApprox=True)
config.detection.tempLocalBackground.weighting=True

# Enable temporary local background subtraction? (see tempLocalBackground)
config.detection.doTempLocalBackground=True

# type of statistic to use for grid points
config.detection.tempWideBackground.statisticsProperty='MEANCLIP'

# behaviour if there are too few points in grid for requested interpolation style
config.detection.tempWideBackground.undersampleStyle='REDUCE_INTERP_ORDER'

# how large a region of the sky should be used for each background point
config.detection.tempWideBackground.binSize=512

# Sky region size to be used for each background point in X direction. If 0, the binSize config is used.
config.detection.tempWideBackground.binSizeX=0

# Sky region size to be used for each background point in Y direction. If 0, the binSize config is used.
config.detection.tempWideBackground.binSizeY=0

# how to interpolate the background values. This maps to an enum; see afw::math::Background
config.detection.tempWideBackground.algorithm='AKIMA_SPLINE'

# Names of mask planes to ignore while estimating the background
config.detection.tempWideBackground.ignoredPixelMask=['BAD', 'EDGE', 'NO_DATA']

# Ignore NaNs when estimating the background
config.detection.tempWideBackground.isNanSafe=False

# Use Approximate (Chebyshev) to model background.
config.detection.tempWideBackground.useApprox=False

# Approximation order in X for background Chebyshev (valid only with useApprox=True)
config.detection.tempWideBackground.approxOrderX=6

# Approximation order in Y for background Chebyshev (valid only with useApprox=True)
config.detection.tempWideBackground.approxOrderY=-1

# Use inverse variance weighting in calculation (valid only with useApprox=True)
config.detection.tempWideBackground.weighting=True

# Do temporary wide (large-scale) background subtraction before footprint detection?
config.detection.doTempWideBackground=False

# The maximum number of peaks in a Footprint before trying to replace its peaks using the temporary local background
config.detection.nPeaksMaxSimple=1

# Multiple of PSF RMS size to use for convolution kernel bounding box size; note that this is not a half-size. The size will be rounded up to the nearest odd integer
config.detection.nSigmaForKernel=7.0

# Mask planes to ignore when calculating statistics of image (for thresholdType=stdev)
config.detection.statsMask=['BAD', 'SAT', 'EDGE', 'NO_DATA']

# Mask planes to exclude when detecting sources.
config.detection.excludeMaskPlanes=[]

# What to do when a peak to be deblended is close to the edge of the image
config.deblend.edgeHandling='ramp'

# When the deblender should attribute stray flux to point sources
config.deblend.strayFluxToPointSources='necessary'

# Assign stray flux (not claimed by any child in the deblender) to deblend children.
config.deblend.assignStrayFlux=True

# How to split flux among peaks
config.deblend.strayFluxRule='trim'

# When splitting stray flux, clip fractions below this value to zero.
config.deblend.clipStrayFluxFraction=0.001

# Chi-squared per DOF cut for deciding a source is a PSF during deblending (un-shifted PSF model)
config.deblend.psfChisq1=1.5

# Chi-squared per DOF cut for deciding a source is PSF during deblending (shifted PSF model)
config.deblend.psfChisq2=1.5

# Chi-squared per DOF cut for deciding a source is a PSF during deblending (shifted PSF model #2)
config.deblend.psfChisq2b=1.5

# Only deblend the brightest maxNumberOfPeaks peaks in the parent (<= 0: unlimited)
config.deblend.maxNumberOfPeaks=0

# Maximum area for footprints before they are ignored as large; non-positive means no threshold applied. Default value is to prevent excessive memory usage.
config.deblend.maxFootprintArea=10000

# Maximum linear dimension for footprints before they are ignored as large; non-positive means no threshold applied
config.deblend.maxFootprintSize=0

# Minimum axis ratio for footprints before they are ignored as large; non-positive means no threshold applied
config.deblend.minFootprintAxisRatio=0.0

# Mask name for footprints not deblended, or None
config.deblend.notDeblendedMask='NOT_DEBLENDED'

# Footprints smaller in width or height than this value will be ignored; minimum of 2 due to PSF gradient calculation.
config.deblend.tinyFootprintSize=2

# Guarantee that all peaks produce a child source.
config.deblend.propagateAllPeaks=False

# If True, catch exceptions thrown by the deblender, log them, and set a flag on the parent, instead of letting them propagate up.
config.deblend.catchFailures=True

# Mask planes to ignore when performing statistics
config.deblend.maskPlanes=['SAT', 'INTRP', 'NO_DATA']

# Mask planes with the corresponding limit on the fraction of masked pixels. Sources violating this limit will not be deblended. Default rejects sources in vignetted regions.
config.deblend.maskLimits={'NO_DATA': 0.25}

# If true, a least-squares fit of the templates will be done to the full image. The templates will be re-weighted based on this fit.
config.deblend.weightTemplates=False

# Try to remove similar templates?
config.deblend.removeDegenerateTemplates=False

# If the dot product between two templates is larger than this value, we consider them to be describing the same object (i.e. they are degenerate).  If one of the objects has been labeled as a PSF it will be removed, otherwise the template with the lowest value will be removed.
config.deblend.maxTempDotProd=0.5

# Apply a smoothing filter to all of the template images
config.deblend.medianSmoothTemplate=True

# Limit the number of sources deblended for CI to prevent long build times
config.deblend.useCiLimits=False

# Only deblend parent Footprints with a number of peaks in the (inclusive) range indicated.If `useCiLimits==False` then this parameter is ignored.
config.deblend.ciDeblendChildRange=[2, 10]

# Only use the first `ciNumParentsToDeblend` parent footprints with a total peak count within `ciDebledChildRange`. If `useCiLimits==False` then this parameter is ignored.
config.deblend.ciNumParentsToDeblend=10

# The maximum radial number of shapelets used in anacal.fpfs
config.fpfs.norder=4

# number of pixels in stamp
config.fpfs.npix=64

# Sources to be removed if too close to boundary
config.fpfs.bound=32

# Shapelet's Gaussian kernel size for detection
config.fpfs.sigma_arcsec=0.52

# Shapelet's Gaussian kernel size for measurement
config.fpfs.sigma_arcsec2=-1.0

# peak detection threshold
config.fpfs.pthres=0.12

# threshold to determine the maximum k in Fourier space
config.fpfs.kmax_thres=1e-12

# whether to use average PSF over the exposure
config.fpfs.use_average_psf=True

# whether to doulbe the noise for noise bias correction
config.fpfs.do_adding_noise=False

# Mask planes used to reject bad pixels.
config.fpfs.badMaskPlanes=['BAD', 'SAT', 'CR']

# Size of psfCache
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
config.connections.exposure='{inputCoaddName}Coadd_calexp{dataType}'

# name for connection outputCatalog
config.connections.outputCatalog='{outputCoaddName}Coadd_anacal_meas{dataType}'

# Template parameter used to format corresponding field template parameter
config.connections.inputCoaddName='deep'

# Template parameter used to format corresponding field template parameter
config.connections.outputCoaddName='deep'

# Template parameter used to format corresponding field template parameter
config.connections.dataType='_test_1_rot0'

