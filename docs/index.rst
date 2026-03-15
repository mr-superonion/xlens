xlens documentation
===================

**xlens** is a weak gravitational lensing analysis framework built on
`AnaCal <https://github.com/mr-superonion/AnaCal>`_ and the
`LSST Science Pipelines <https://pipelines.lsst.io/>`_.
It provides tools for simulating galaxy images, measuring shapes with
FPFS shapelets, and correcting for WCS distortions.

Subpackages
-----------

- **simulator** -- Galaxy image simulation with realistic WCS,
  lensing distortions, PSF convolution, and noise.
- **processor** -- Shape measurement pipelines (FPFS, NGMIX).
- **catalog** -- Truth catalog generation with controlled shear signals.
- **analysis** -- Shear calibration and cluster lensing analysis.
- **wcs** -- WCS coordinate conversion and shapelet moment correction
  between LSST and GalSim conventions.

Getting Started
---------------

Install into a conda environment with the LSST Science Pipelines
(e.g. via `stackvana <https://github.com/conda-forge/stackvana>`_):

.. code-block:: bash

   pip install -e '.[dev]'

Run the test suite:

.. code-block:: bash

   pytest -vv

.. toctree::
   :hidden:

   Home page <self>
   API Reference <autoapi/index>
   Notebooks <notebooks>
