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
