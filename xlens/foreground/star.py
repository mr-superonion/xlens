"""Foreground (bright-star) models for the image simulator.

TODO: the foreground star carries no photon noise.  ``MultibandSimTask``
draws the image noise onto the galaxy exposure and only then clones it and
adds the star, so neither the noise realisation nor the variance plane knows
about the star at all.  See the ``foreground_model`` block of
``MultibandSimTask.run`` in ``xlens/simulator/sim.py``, where the ordering
that causes this is written up.
"""

import galsim
import numpy as np


def bright_star(model, mag_zero, band):
    ## how do we handle the model and parameters? this can either be a galsim model
    ## or a galsim interpolated image
    if model == 'test_moffat':
        star_mag = 13
        star_flux = 10**((star_mag - mag_zero) / -2.5)
        foreground_object = galsim.Moffat(beta=1.6, scale_radius=6.3, flux=star_flux)
    
        return foreground_object
    return None
