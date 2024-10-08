import xlens.summary_pipe.mcbias_multiband
assert type(config) is xlens.summary_pipe.mcbias_multiband.McBiasMultibandPipeConfig, f"config is of type {type(config).__module__}.{type(config).__name__} instead of xlens.summary_pipe.mcbias_multiband.McBiasMultibandPipeConfig"

import lsst.pipe.base.config
# Flag to enable/disable saving of log output for a task, enabled by default.
config.saveLogOutput=True

# ellipticity column name
config.shape_name='e1'

# the shear component to test
config.shear_name='g1'

# absolute value of the shear
config.shear_value=0.02

# name for connection src00List
config.connections.src00List='{inputCoaddName}Coadd_anacal_meas{dataType}_0_rot0'

# name for connection src01List
config.connections.src01List='{inputCoaddName}Coadd_anacal_meas{dataType}_0_rot1'

# name for connection src10List
config.connections.src10List='{inputCoaddName}Coadd_anacal_meas{dataType}_1_rot0'

# name for connection src11List
config.connections.src11List='{inputCoaddName}Coadd_anacal_meas{dataType}_1_rot1'

# Template parameter used to format corresponding field template parameter
config.connections.inputCoaddName='deep'

# Template parameter used to format corresponding field template parameter
config.connections.dataType='_test'

