# -*- coding: utf-8 -*-
"""
Created on Mon May  6 08:58:16 2024

@author: nickj
"""



'''
IMPORTING MODULES
'''

#standard stuff
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import  AutoMinorLocator
import random

#used for fits file handling
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits



####################################################################

'''
FUNCTIONS
'''



def loading_function(file_loc, header_index):
    '''
    This function loads in JWST MIRI and NIRSPEC fits data cubes, and extracts wavelength 
    data from the header and builds the corresponding wavelength array. It takes file_loc2, although it
    is unused by this function and is instead used by an old version of this function, which is now
    loading_function_reproject.
    
    Parameters
    ----------
    file_loc
        TYPE: string
        DESCRIPTION: where the fits file is located.
    header_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index to get wavelength data from in the header.
        
        file_loc2
            TYPE: string
            DESCRIPTION: where the fits file is located for rebinning
        header_index
            TYPE: index (nonzero integer)
            DESCRIPTION: the index to get wavelength data from in the header.

    Returns
    -------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns.
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [k,i,j] k is wavelength index, i and j are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [k,i,j] k is wavelength index, i and j are position index.
    '''
    
    #load in the data
    image_file = get_pkg_data_filename(file_loc)
    
    #header data
    science_header = fits.getheader(image_file, header_index)
    
    #wavelength data from header
    number_wavelengths = science_header["NAXIS3"]
    wavelength_increment = science_header["CDELT3"]
    wavelength_start = science_header["CRVAL3"]
    
    #constructing the ending point using given data
    #subtracting 1 so wavelength array is the right size.
    wavelength_end = wavelength_start + (number_wavelengths - 1)*wavelength_increment

    #making wavelength array, in micrometers
    wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment)
    
    #extracting image data
    image_data = fits.getdata(image_file, ext=1)
    error_data = fits.getdata(image_file, ext=2)
    
    #sometimes wavelength array is 1 element short, this will fix that
    if len(wavelengths) != len(image_data):
        wavelength_end = wavelength_start + number_wavelengths*wavelength_increment
        wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment)

    return wavelengths, image_data, error_data
