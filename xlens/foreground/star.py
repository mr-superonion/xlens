import galsim
import numpy as np
import pickle

def bright_star(model, mag_zero, band, psf, center, pixel_scale, width, height, draw_method, seed):
    ## how do we handle the model and parameters? this can either be a galsim model
    ## or a galsim interpolated image
    if model == 'test_moffat':
        star_mag = 13
        star_flux = 10**((star_mag - mag_zero) / -2.5)
        foreground_obj = galsim.Moffat(beta=1.6, scale_radius=6.3, flux=star_flux)

        convolved_object = galsim.Convolve([foreground_obj, psf])
        
        stamp = convolved_object.drawImage(
            center=center,
            wcs=None,
            method=draw_method,
            scale=pixel_scale,
            nx=width,
            ny=height,
        )
        return stamp.array, np.zeros_like(stamp.array)
    
    elif model == 'dp2_star':
        ### temp
        with open('/hildafs/home/pladuca/main/bright_star_sims/bright_star_cutout/mag_10_300stack.pkl', 'rb') as f:
            stack = pickle.load(f)
        cy = (stack.shape[0] - 1) / 2
        cx = (stack.shape[1] - 1) / 2
        # radial_bins = np.arange(0, min(cx,cy)+1, 10)
        radial_bins = [int(cx * 0.8),int(cx)]
        yy, xx = np.indices(stack.shape)
        radius = np.sqrt((xx - cx)**2 + (yy - cy)**2)
        for i, (r0, r1) in enumerate(zip(radial_bins[:-1], radial_bins[1:])): 
            mask = (radius >= r0) & (radius < r1)
            values = stack[mask]
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            mean_clipped = np.mean(values)
            std_clipped = np.std(values)
        
        star_clip = stack - mean_clipped

        y0 = int(round(cy - (height - 1) / 2))
        y1 = y0 + height
        x0 = int(round(cx - (width - 1) / 2))
        x1 = x0 + width
        star_clip = star_clip[y0:y1, x0:x1]

        noise_array = np.random.RandomState(seed + 1).normal(
            scale=std_clipped,
            size=(height, width))
        
        return star_clip, noise_array

    return None
