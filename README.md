# xlens (Graviational Lensing from Image Pixels)

[![tests](https://github.com/mr-superonion/xlens/actions/workflows/tests.yml/badge.svg)](https://github.com/mr-superonion/xlens/actions/workflows/tests.yml)
[![conda-forge](https://anaconda.org/conda-forge/xlens/badges/version.svg)](https://anaconda.org/conda-forge/xlens)


## Installation

### Github
Users can clone this repository and install the latest package by
```shell
git clone https://github.com/mr-superonion/xlens.git
cd xlens
# install required softwares
conda install -c conda-forge --file requirements.txt
# install required softwares for unit tests (if necessary)
conda install -c conda-forge --file requirements_test.txt
pip install . --user
```

### Conda-forge
stable verion can be installed from conda-forge
```
conda install -c conda-forge xlens
```

### Input Galaxy Catalog
One can download and setup the input CATSIM2017 galaxy catalog:

```shell
wget https://github.com/mr-superonion/xlens/releases/download/v0.3.0/catsim-v4.tar.gz

tar xvfz catsim-v4.tar.gz
export CATSIM_DIR=$(realpath catsim-v4)
```
## Examples
Examples can be found [here](https://github.com/mr-superonion/xlens/blob/main/examples/).

## Development

Before sending pull request, please make sure that the modified code passed the
pytest and flake8 tests. Run the following commands under the root directory
for the tests:

```shell
flake8
pytest -vv
```
----

## License and Acknowledgements

This project is distributed under the terms of the GNU General Public License
version 3. Portions of the codebase originate from the Rubin Observatory Legacy
Survey of Space and Time (LSST) Science Pipelines. In accordance with the LSST
license requirements, we acknowledge that this product includes software
developed by the LSST Project (https://www.lsst.org/). Additional copyright
details for bundled LSST-derived software can be found in the accompanying
``COPYRIGHT`` file.

