# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:46:51 2023

@author: nickj
"""



'''
IMPORTING MODULES
'''

#standard stuff
import numpy as np

#used for fits file handling
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits



####################################



'''
FUNCTIONS
'''



def loading_function(file_loc, file_loc2, header_index):
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



def weighted_mean_finder_simple(data, error_data):
    '''
    This function takes a weighted mean of the
    background-subtracted data, for 3 dimensional arrays.
    The mean is taken over the 1st and 2nd indicies (not the 0th), i.e. the spacial dimensions.
    
    Parameters
    ----------
    data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [k,i,j] k is wavelength index, i and j are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [k,i,j] k is wavelength index, i and j are position index.
    
    Returns
    -------
        weighted_mean
            TYPE: 1d array of floats
            DESCRIPTION: weighted mean of the background-subtracted input data 
                spacial dimensions, as a spectra.
        mean_error
            TYPE: 1d array of floats
            DESCRIPTION: errors corresponding to the background-subtracted weighted mean.
    '''
    
    #replacing nans with 0, as background data has nans on border
    where_are_NaNs = np.isnan(data) 
    data[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(error_data) 
    error_data[where_are_NaNs] = 0
    
    #defining arrays to fill with final values
    weighted_mean = []
    weighted_mean_error = []
    
    #note JWST provides uncertainty (variance), standard deviation**2 = variance
    for i in range(len(data[:,0,0])):
        #defining temporary arrays to assist in computation of weighted mean
        error_list = []
        error_temp_list = []
        mean_list = []
        
        #iterate over both spacial axes
        for j in range(len(data[0,:,0])):
            for k in range(len(data[0,0,:])):
                
                #only include values that have a valid error
                if error_data[i,j,k] != 0:
                    
                    #calculating weighted mean and error in steps
                    temp_error = 1/(error_data[i,j,k])**2
                    mean_list.append(data[i,j,k]/(error_data[i,j,k])**2)
                    
                    #adding values to lists
                    error_temp_list.append(temp_error)
                    error_list.append(error_data[i,j,k])
        
        #turning lists into arrays
        error_list = np.array(error_list)
        error_temp_list = np.array(error_temp_list)
        mean_list = np.array(mean_list)
        
        #summing temp arrays for final values
        error = np.sqrt(1/np.sum(error_temp_list))
        mean = (np.sum(mean_list))*error**2
        
        #adding values to lists
        weighted_mean.append(mean)
        weighted_mean_error.append(error)
    
    #turning lists into arrays
    weighted_mean = np.array(weighted_mean)
    mean_error = np.array(weighted_mean_error)
    
    return weighted_mean, mean_error



####################################



'''
LOADING DATA
'''

#calling MIRI_function
#naming is ordered from smallest to largest wavelength range
wavelengths1, image_data1, error_data1 = loading_function(
    'data/MIRI_MRS/version2_030323/cubes/north/ring_neb_obs2_ch1-short_s3d.fits', 
    'data/nirspec_dec2022/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 1)
wavelengths2, image_data2, error_data2 = loading_function(
    'data/MIRI_MRS/version2_030323/cubes/north/ring_neb_obs2_ch1-medium_s3d.fits', 
    'data/nirspec_dec2022/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 1)
wavelengths3, image_data3, error_data3 = loading_function(
    'data/MIRI_MRS/version2_030323/cubes/north/ring_neb_obs2_ch1-long_s3d.fits', 
    'data/nirspec_dec2022/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 1)



####################################



'''
WEIGHTED MEAN
'''



data1, weighted_mean_error1 = weighted_mean_finder_simple(image_data1, error_data1)
data2, weighted_mean_error2 = weighted_mean_finder_simple(image_data2, error_data2)
data3, weighted_mean_error3 = weighted_mean_finder_simple(image_data3, error_data3)





