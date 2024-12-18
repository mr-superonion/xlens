import xlens.summary_pipe.halo_mcbias_multiband
assert type(config) is xlens.summary_pipe.halo_mcbias_multiband.HaloMcBiasMultibandPipeConfig, f"config is of type {type(config).__module__}.{type(config).__name__} instead of xlens.summary_pipe.halo_mcbias_multiband.HaloMcBiasMultibandPipeConfig"

import lsst.pipe.base.config
# Flag to enable/disable saving of log output for a task, enabled by default.
config.saveLogOutput=True

# ellipticity column name
config.ename='e'

# detection coordinate row name
config.xname='x'

# detection coordinate column name
config.yname='y'

# halo mass
config.mass=800000000000000.0

# halo concertration
config.conc=1.0

# halo redshift
config.z_lens=0.25

# source redshift
config.z_source=1.0

# name for connection skymap
config.connections.skymap='skyMap'

# name for connection src00List
config.connections.src00List='{coaddName}_0_rot0_Coadd_anacal_{dataType}'

# name for connection src01List
config.connections.src01List='{coaddName}_0_rot1_Coadd_anacal_{dataType}'

# name for connection truth00List
config.connections.truth00List='{coaddName}_0_rot0_Coadd_truthCatalog'

# name for connection truth01List
config.connections.truth01List='{coaddName}_0_rot1_Coadd_truthCatalog'

# name for connection outputSummary
config.connections.outputSummary='{coaddName}_halo_mc_{dataType}_summary_stats'

# name for connection summaryPlot
config.connections.summaryPlot='halo_mc_summary_{dataType}_plot'

# Template parameter used to format corresponding field template parameter
config.connections.coaddName='sim'

# Template parameter used to format corresponding field template parameter
config.connections.dataType='joint'

