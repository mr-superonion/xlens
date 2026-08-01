# xlens (Gravitational Lensing from Image Pixels)

[![tests](https://github.com/mr-superonion/xlens/actions/workflows/tests.yml/badge.svg)](https://github.com/mr-superonion/xlens/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mr-superonion/xlens/branch/main/graph/badge.svg)](https://codecov.io/gh/mr-superonion/xlens)
[![conda-forge](https://anaconda.org/conda-forge/xlens/badges/version.svg)](https://anaconda.org/conda-forge/xlens)
[![docs](https://readthedocs.org/projects/xlens/badge/?version=latest)](https://xlens.readthedocs.io/en/latest/index.html)

**xlens** is a weak gravitational lensing analysis framework built on
[AnaCal](https://github.com/mr-superonion/AnaCal) and the
[LSST Science Pipelines](https://pipelines.lsst.io/).
It provides end-to-end tools for simulating galaxy images, measuring
shapes with FPFS shapelets, and correcting for WCS distortions.

### Subpackages

| Subpackage | Description |
|---|---|
| `catalog` | Truth catalog generation with controlled shear signals |
| `simulator` | Galaxy image simulation with realistic WCS, lensing, PSF, and noise |
| `processor` | Shear measurement pipelines calibrated with AnaCal (FPFS, NGMIX) |
| `analysis` | Shear measurement validation and cluster lensing analysis |
| `wcs` | WCS coordinate conversion and shape correction |

## Installation

### From GitHub
```shell
git clone https://github.com/mr-superonion/xlens.git
cd xlens
conda install -c conda-forge --file requirements.txt
pip install .
```

### From conda-forge
```shell
conda install -c conda-forge xlens
```

### Input Galaxy Catalog
Download and set up the DESC DC1 (2017) galaxy catalog:
```shell
wget https://www.cosmo.bnl.gov/www/xiangchong/data/catsim_test.tar.xz
tar xvfJ catsim_test.tar.xz
export CATSIM_DIR=$(realpath catsim_test)
```

## Examples

Example notebooks are in the [`examples/`](examples/) directory:

**Shear measurement**
- [Isolated galaxy simulation (forced positions)](examples/shear/example1_1_isolated_sim_force.ipynb)
- [Multiband simulation with linear-to-e shear estimation](examples/shear/example1_2_isolated_sim_force_multiband_linear.ipynb)
- [Isolated galaxy simulation](examples/shear/example2_isolated_sim.ipynb)
- [FPFS measurement on blended galaxies (noiseless)](examples/shear/example3_fpfs_blended_anacal_noiseless.ipynb)
- [Blended simulation with matching](examples/shear/example4_blended_sim_measure_match.ipynb)
- [Euclid VIS simulation](examples/shear/example5_euclid_vis_sim.ipynb)

**Cluster lensing**
- [Cluster lensing profile](examples/cluster/example1_cluster_lensing.ipynb)

**Random field**
- [Lognormal random field](examples/field/example1_lognormal.ipynb)

**Intrinsic alignments**
- [BATSim intrinsic alignment](examples/batsim/ia.ipynb)

Rendered versions are available in the
[online documentation](https://xlens.readthedocs.io/en/latest/notebooks.html).

## Documentation

Full API reference and rendered notebooks:
https://xlens.readthedocs.io/en/latest/

## Development

Before sending a pull request, please make sure the modified code passes
the pytest and flake8 tests:

```shell
flake8
pytest -vv
```

---

## License and Acknowledgements

This project is distributed under the terms of the GNU General Public License
version 3. Portions of the codebase originate from the Rubin Observatory Legacy
Survey of Space and Time (LSST) Science Pipelines. In accordance with the LSST
license requirements, we acknowledge that this product includes software
developed by the LSST Project (https://www.lsst.org/). Additional copyright
details for bundled LSST-derived software can be found in the accompanying
`COPYRIGHT` file.
