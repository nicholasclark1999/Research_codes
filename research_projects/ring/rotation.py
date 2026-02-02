#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:30:28 2024

@author: nclark
"""

import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt

#used for fits file handling
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.table import Table

#Functions python script
import RingNebulaFunctions as rnf


with fits.open('data/cams/miri_color_F1000W_F1130W.fits') as hdul:
    pass
    
    
    
    
    
    
    
    
    
    
    

fits_to_reproject = 'data/cams/miri_color_F1000W_F1130W.fits'

with fits.open(fits_to_reproject) as hdul:
        
    unrotated = hdul[0].data
    
    where_are_NaNs = np.isnan(unrotated) 
    unrotated[where_are_NaNs] = 0
    
    header = hdul[0].header
    
    angle = header['PA_V3']
    
    rotated = rotate(unrotated, angle=-angle, reshape=False)
        
    #replacing data in the currently open fits file
    hdul[0].data = rotated



            
    hdul[0].header['PA_V3'] = 0
    hdul[0].header['PA_APER'] = hdul[0].header['PA_APER'] - angle
    
    hdul[0].header['PC1_1'] = 1
    hdul[0].header['PC1_2'] = 0
    hdul[0].header['PC2_1'] = 0
    hdul[0].header['PC2_2'] = 1

        
    #saving data, replacing any files with the same name for convinient reruns
    hdul.writeto('data/cams/miri_color_F1000W_F1130W_norot.fits', overwrite=True)

#%%
plt.figure()
plt.imshow(unrotated, vmin=0.5, vmax=2)
plt.scatter(863,1423)
plt.show()
#%%
plt.figure()
plt.imshow(rotated, vmin=0.5, vmax=2)
plt.title('rotated')
plt.show()