from glob import glob

import astropy.units as u
import numpy as np
import opencosmo as oc
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord

diff_folder = (
    '/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies/'
    'hltds_cosmos_260215_04_07_2026'
)
save_path = '/global/homes/p/pladuca/main'

diffsky_path = glob(f'{diff_folder}/lc_cores*.hdf5')

cat = oc.open(diffsky_path, synth_cores=True)
cat = cat.select([
    'ra', 'dec', 'redshift', 'ellipticity_disk', 'ellipticity_bulge',
    'r50_disk', 'r50_bulge', 'psi_bulge', 'psi_disk',
    'lsst_u_bulge', 'lsst_u_disk', 'lsst_g_bulge', 'lsst_g_disk',
    'lsst_r_bulge', 'lsst_r_disk', 'lsst_i_bulge', 'lsst_i_disk',
    'lsst_z_bulge', 'lsst_z_disk', 'lsst_y_bulge', 'lsst_y_disk',
])

# hard coded for diffsky 4_7 catalog
region = oc.make_cone(SkyCoord(100 * u.deg, 84.5 * u.deg), 1 * u.deg)
cat = cat.bound(region)

cosmology = cat.cosmology
cat = cat.get_data('numpy')

cat['indices'] = np.arange(len(cat['redshift']))
dA = cosmology.angular_diameter_distance(cat['redshift'])
for i in ['r50_bulge', 'r50_disk']:
    theta = (cat[i] * u.kpc / dA) * u.rad
    cat[f'{i}_as'] = theta.to(u.arcsec).value

dtype = [(name, np.array(values).dtype) for name, values in cat.items()]
arr = np.zeros(len(next(iter(cat.values()))), dtype=dtype)
for name, values in cat.items():
    arr[name] = values

table = pa.Table.from_arrays(
    [pa.array(arr[name]) for name in arr.dtype.names],
    names=list(arr.dtype.names)
)
pq.write_table(table, f'{save_path}/diffsky_arr.parquet')
