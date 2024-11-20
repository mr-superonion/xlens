config.skyMap = "discrete"

# Position of M31 (Andromeda)
config.skyMap["discrete"].raList = [10.685]  # degrees
config.skyMap["discrete"].decList = [41.269]  # degrees
config.skyMap["discrete"].radiusList = [5.0]  # degrees
config.skyMap["discrete"].pixelScale = 0.168  # HSC arcsec/pixel
config.skyMap["discrete"].patchInnerDimensions = [500, 500]  # in pixels
config.skyMap["discrete"].projection = "TAN"
config.skyMap["discrete"].tractOverlap = 0
config.name = "hsc_sim"
