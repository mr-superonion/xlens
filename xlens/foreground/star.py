import galsim
import numpy as np


def bright_star(model, mag_zero, band):
    ## how do we handle the model and parameters? this can either be a galsim model
    ## or a galsim interpolated image
    star_mag = 8.5
    star_flux = -2.5 * np.log10(star_mag) + mag_zero
    foreground_object = galsim.Moffat(beta=1.6, scale_radius=15.3, flux=star_flux)
    
    return foreground_object
