# butler register-skymap config: one small discrete tract for the simulation
# consistency check (examples/pipelines/run_sim_noise_consistency.sh).
#
# Same geometry as bin/basic/common.py's build_skymap() except for the patch
# size: 600 px patches instead of 1500 keep the per-quantum rendering and
# measurement to a minute or so.
config.name = "sim_test"
config.skyMap = "discrete"
config.skyMap["discrete"].raList = [0.0]
config.skyMap["discrete"].decList = [0.0]
config.skyMap["discrete"].radiusList = [0.1]
config.skyMap["discrete"].rotation = 0.0
config.skyMap["discrete"].projection = "TAN"
config.skyMap["discrete"].patchInnerDimensions = [600, 600]
config.skyMap["discrete"].patchBorder = 0
config.skyMap["discrete"].pixelScale = 0.2
config.skyMap["discrete"].tractOverlap = 0.0
