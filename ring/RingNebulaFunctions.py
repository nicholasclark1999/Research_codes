
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:06:29 2023

@author: nclark
"""

#note: make 2 separate weighted mean calculators, one that does background subtraction
#and one that doesnt, to accomodate data that is already background subtracted.
#or include a flag or something to say if its background subtracted.

'''
IMPORTING MODULES
'''

#standard stuff
import matplotlib.pyplot as plt
import numpy as np

#used for fits file handling
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

#PCA function
from sklearn.decomposition import PCA

#Import needed scipy libraries for curve_fit
import scipy.optimize

#Import needed sklearn libraries for RANSAC
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

#needed for fringe remover
import pickle 
from astropy.units import Unit

#needed for unit_changer
import astropy.units as u

#needed for els' region function
import regions
from astropy.wcs import wcs
from astropy.stats import sigma_clip



####################################



'''
LOADING DATA
'''



def loading_function(file_loc, header_index):
    '''
    This function loads in JWST MIRI and NIRSPEC fits data cubes, and extracts wavelength 
    data from the header and builds the corresponding wavelength array.
    
    Parameters
    ----------
    file_loc
        TYPE: string
        DESCRIPTION: where the fits file is located.
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
            for [wave, y, x] wave is wavelength index, x and y are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [wave, y, x] wave is wavelength index, x and y are position index.
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



####################################



'''
BACKGROUND SUBTRACTION
'''

def bkg_sub_and_weighted_mean_finder(data, error_data, data_off, error_data_off):
    '''
    This function subtracts the background from the data, and also calculates
    the associated errors. Then, this function takes a weighted mean of the
    background-subtracted data, for 3 dimensional arrays.
    The mean is taken over the 1st and 2nd indicies, i.e. the spacial dimensions.
    
    Parameters
    ----------
    data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [wave, y, x] wave is wavelength index, x and y are position index.
    data_off
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data for background.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    error_data_off
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data for background.
                for [wave, y, x] wave is wavelength index, x and y are position index.
    
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
    
    #first background subtract, propagate errors
    data = data - data_off
    error_data_new = (error_data**2 + error_data_off**2)**0.5
    
    #lists to store weighted mean for each wavelength
    weighted_mean = []
    weighted_mean_error = []
    
    #note JWST provides uncertainty (standard deviation), standard deviation**2 = variance
    for wave in range(len(data[:,0,0])):
        
        #making lists to store values and sum over later
        error_list = []
        error_temp_list = []
        mean_list = []
        
        for y in range(len(data[0,:,0])):
            for x in range(len(data[0,0,:])):
                if error_data[wave, y, x] != 0 and error_data_off[wave, y, x] != 0: #this being 0 means its outside the fov
                    #single component of the weighted mean error
                    temp_error = 1/(error_data_new[wave, y, x])**2
                    
                    #adding single components of weighted mean and weighted mean error to lists, to sum later
                    mean_list.append(data[wave, y, x]/(error_data_new[wave, y, x])**2)
                    error_temp_list.append(temp_error)
                    error_list.append(error_data[wave, y, x])
        
        #turning lists into arrays
        error_list = np.array(error_list)
        error_temp_list = np.array(error_temp_list)
        mean_list = np.array(mean_list)
        
        #summing lists to get error and weighted mean for this wavelength
        error = np.sqrt(1/np.sum(error_temp_list))
        mean = (np.sum(mean_list))*error**2
        
        #adding to list
        weighted_mean.append(mean)
        weighted_mean_error.append(error)
    
    #turning lists into arrays
    weighted_mean = np.array(weighted_mean)
    mean_error = np.array(weighted_mean_error)
    
    return weighted_mean, mean_error



def weighted_mean_finder(data, error_data):
    '''
    This function takes a weighted mean of the (assumed background-subtracted, 
    in the case of JWST cubes) data, for 3 dimensional arrays.
    The mean is taken over the 1st and 2nd indicies (not the 0th), i.e. the spacial dimensions.
    
    Parameters
    ----------
    data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [wave, y, x] wave is wavelength index, x and y are position index.
    
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
    
    #replacing nans with 0, as data has nans on border
    where_are_NaNs = np.isnan(data) 
    data[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(error_data) 
    error_data[where_are_NaNs] = 0
    
    #lists to store weighted mean for each wavelength
    weighted_mean = []
    weighted_mean_error = []
    
    #note JWST provides uncertainty (standard deviation), standard deviation**2 = variance
    for wave in range(len(data[:,0,0])):
        #making lists to store values and sum over later
        error_list = []
        error_temp_list = []
        mean_list = []
        
        for y in range(len(data[0,:,0])):
            for x in range(len(data[0,0,:])):
                if error_data[wave, y, x] != 0:
                    #single component of the weighted mean error
                    temp_error = 1/(error_data[wave, y, x])**2
                    
                    #adding single components of weighted mean and weighted mean error to lists, to sum later
                    mean_list.append(data[wave, y, x]/(error_data[wave, y, x])**2)
                    error_temp_list.append(temp_error)
                    error_list.append(error_data[wave, y, x])
        
        #turning lists into arrays
        error_list = np.array(error_list)
        error_temp_list = np.array(error_temp_list)
        mean_list = np.array(mean_list)
        
        #summing lists to get error and weighted mean for this wavelength
        error = np.sqrt(1/np.sum(error_temp_list))
        mean = (np.sum(mean_list))*error**2
        
        #adding to list
        weighted_mean.append(mean)
        weighted_mean_error.append(error)
    
    #turning lists into arrays
    weighted_mean = np.array(weighted_mean)
    mean_error = np.array(weighted_mean_error)
    
    return weighted_mean, mean_error



def extract_weighted_mean_from_region(fname_cube, data, error_data, fname_region):
    '''
    calculates a weighted mean, of a data cube over the spacial indices, with 
    the pixels to use selected with a .reg file. Any bad pixels should have their
    value set to 0 before using this function (flagged in DQ, for example), 
    and they will be ignored in the weighted mean calculation. This function uses
    nansum instead of sum, so nans do not need to be removed in the data for
    this function to work.
    
    Parameters
    ----------
    fname_cube
        TYPE: string
        DESCRIPTION: the location of the data cube, where data, error and wavelengths
            came from. 
    data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [wave, y, x] wave is wavelength index, x and y are position index.

    Returns
    -------
    combined_spectra
        TYPE: dictionary of spectra
        DESCRIPTION: a dictionary, where the keys are 'region_{region_index}', and the
        values of each key are the weighted mean of the slice. 
    combined_spectra_error
        TYPE: dictionary of spectra
        DESCRIPTION: a dictionary, where the keys are 'region_{region_index}', and the
        values of each key are the weighted mean error of the slice. 
    '''
    
    #jank fix for the code including 0 error pixels when it shouldnt
    
    for i in range(len(error_data[:,0,0])):
        for j in range(len(error_data[0,:,0])):
            for k in range(len(error_data[0,0,:])):
                if error_data[i,j,k] == 0:
                    error_data[i,j,k] = np.nan
    
    #loading in the region file
    reg = regions.Regions.read(fname_region, format='ds9')
    
    #loading in fits data cube
    fits_cube = fits.open(fname_cube)
    
    #extracting wcs info from header ext 1, which corresponds to the main data. 
    #axis 2, which corresponds to wavelength info, is removed.
    w = wcs.WCS(fits_cube[1].header).dropaxis(2)
    
    #define dictionaries to store spectra. Each entry in the dictionary corresponds
    #to 1 region in the .reg file.
    all_spectra = dict()
    all_spectra_error = dict()
    combined_spectra = dict()
    combined_spectra_error = dict()
    
    # loop over regions in .reg file
    for region_index in range(len(reg)):
        #converts region into a 2d array with 1 if inside the region, and 0 if outside
        regmask = reg[region_index].to_pixel(w).to_mask('center').to_image(shape=data.shape[1:])
        
        #adding empty list entries to dictionaries
        all_spectra[f'region_{region_index}'] = []
        all_spectra_error[f'region_{region_index}'] = []
        
        #extracting pixels contained in the regions
        for y in range(data.shape[1]):
            for x in range(data.shape[2]):
                if regmask[y, x] == 1:
                    # pixel is in this region
                    spec = data[:, y, x]
                    spec_error = error_data[:, y, x]
                    
                    #adding values to dictionary lists
                    all_spectra[f'region_{region_index}'].append(spec)
                    all_spectra_error[f'region_{region_index}'].append(spec_error)
        
        #how the pixels are combined depends on the number of pixels in each region
        #at least 1 pixel was found to be in this region:
        if len(all_spectra[f'region_{region_index}']) > 0:
            #currently consists of several appended spectra. 
            #Need to turn into a proper 2d array to easily calculate weighted mean
            all_spectra[f'region_{region_index}'] = np.vstack(all_spectra[f'region_{region_index}'])
            all_spectra_error[f'region_{region_index}'] = np.vstack(all_spectra_error[f'region_{region_index}']) 
            
            #only 1 pixel in region, final value mean is just this pixel value:
            if len(all_spectra[f'region_{region_index}']) == 1:
                combined_spectra[f'region_{region_index}'] = all_spectra[f'region_{region_index}']
                combined_spectra_error[f'region_{region_index}'] = all_spectra_error[f'region_{region_index}']
            
            #more than 1 pixel present, can calculate weighted mean and error:
            else:
                #extracting data from dictionary
                all_data = all_spectra[f'region_{region_index}']
                all_error = all_spectra_error[f'region_{region_index}']
                
                # Calculate weighted mean along position axis (axis 0 after vstack)
                combined_spectra[f'region_{region_index}'] = np.nansum(all_data / all_error**2, axis=0) / np.nansum(1. / all_error**2, axis=0)
                # Uncertainty on the mean
                combined_spectra_error[f'region_{region_index}'] = np.sqrt(1 / np.nansum(1. / all_error**2, axis=0))
                
        #no pixels were found to be in this region:
        else:
            combined_spectra[f'region_{region_index}'] = 'no pixels in this region'
            combined_spectra_error[f'region_{region_index}'] = 'no pixels in this region'
            
    return combined_spectra, combined_spectra_error



def extract_regular_mean_from_region(fname_cube, data, error_data, fname_region):
    '''
    calculates a regular mean, of a data cube over the spacial indices, with 
    the pixels to use selected with a .reg file. Any bad pixels should have their
    value set to 0 before using this function (flagged in DQ, for example), 
    and they will be ignored in the weighted mean calculation. This function uses
    nansum instead of sum, so nans do not need to be removed in the data for
    this function to work.
    
    Parameters
    ----------
    fname_cube
        TYPE: string
        DESCRIPTION: the location of the data cube, where data, error and wavelengths
            came from. 
    data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [wave, y, x] wave is wavelength index, x and y are position index.

    Returns
    -------
    combined_spectra
        TYPE: dictionary of spectra
        DESCRIPTION: a dictionary, where the keys are 'region_{region_index}', and the
        values of each key are the mean of the slice. 
    combined_spectra_error
        TYPE: dictionary of spectra
        DESCRIPTION: a dictionary, where the keys are 'region_{region_index}', and the
        values of each key are the mean error of the slice. 
    '''
    
    #loading in the region file
    reg = regions.Regions.read(fname_region, format='ds9')
    
    #loading in fits data cube
    fits_cube = fits.open(fname_cube)
    
    #extracting wcs info from header ext 1, which corresponds to the main data. 
    #axis 2, which corresponds to wavelength info, is removed.
    w = wcs.WCS(fits_cube[1].header).dropaxis(2)
    
    #define dictionaries to store spectra. Each entry in the dictionary corresponds
    #to 1 region in the .reg file.
    all_spectra = dict()
    all_spectra_error = dict()
    combined_spectra = dict()
    combined_spectra_error = dict()
    
    # loop over regions in .reg file
    for region_index in range(len(reg)):
        #converts region into a 2d array with 1 if inside the region, and 0 if outside
        regmask = reg[region_index].to_pixel(w).to_mask('center').to_image(shape=data.shape[1:])
        
        #adding empty list entries to dictionaries
        all_spectra[f'region_{region_index}'] = []
        all_spectra_error[f'region_{region_index}'] = []
        
        #extracting pixels contained in the regions
        for y in range(data.shape[1]):
            for x in range(data.shape[2]):
                if regmask[y, x] == 1:
                    # pixel is in this region
                    spec = data[:, y, x]
                    spec_error = error_data[:, y, x]
                    
                    #adding values to dictionary lists
                    all_spectra[f'region_{region_index}'].append(spec)
                    all_spectra_error[f'region_{region_index}'].append(spec_error)
        
        #how the pixels are combined depends on the number of pixels in each region
        #at least 1 pixel was found to be in this region:
        if len(all_spectra[f'region_{region_index}']) > 0:
            #currently consists of several appended spectra. 
            #Need to turn into a proper 2d array to easily calculate weighted mean
            all_spectra[f'region_{region_index}'] = np.vstack(all_spectra[f'region_{region_index}'])
            all_spectra_error[f'region_{region_index}'] = np.vstack(all_spectra_error[f'region_{region_index}']) 
            
            #only 1 pixel in region, final value mean is just this pixel value:
            if len(all_spectra[f'region_{region_index}']) == 1:
                combined_spectra[f'region_{region_index}'] = all_spectra[f'region_{region_index}']
                combined_spectra_error[f'region_{region_index}'] = all_spectra_error[f'region_{region_index}']
            
            #more than 1 pixel present, can calculate weighted mean and error:
            else:
                #extracting data from dictionary
                all_data = all_spectra[f'region_{region_index}']
                all_error = all_spectra_error[f'region_{region_index}']
                
                # Calculate regular mean along position axis (axis 0 after vstack)
                combined_spectra[f'region_{region_index}'] = np.mean(all_data, axis=0)
                # Uncertainty on the mean
                combined_spectra_error[f'region_{region_index}'] = np.sqrt(np.nansum(all_error**2, axis=0))/len(all_error)
                
        #no pixels were found to be in this region:
        else:
            combined_spectra[f'region_{region_index}'] = 'no pixels in this region'
            combined_spectra_error[f'region_{region_index}'] = 'no pixels in this region'
            
    return combined_spectra, combined_spectra_error



def extract_weighted_mean_slice_from_region(fname_cube, data, error_data, fname_region):
    '''
    calculates a weighted mean, of a data slice over the spacial indices, with 
    the pixels to use selected with a .reg file. Any bad pixels should have their
    value set to 0 before using this function (flagged in DQ, for example), 
    and they will be ignored in the weighted mean calculation. This function uses
    nansum instead of sum, so nans do not need to be removed in the data for
    this function to work.
    
    Parameters
    ----------
    fname_cube
        TYPE: string
        DESCRIPTION: the location of the data cube, where data, error and wavelengths
            came from. 
    data
        TYPE: 2d array of floats
        DESCRIPTION: position and spectral data.
            for [y, x] x and y are position index.
    error_data
        TYPE: 2d array of floats
        DESCRIPTION: position and spectral error data.
                for [y, x] x and y are position index.

    Returns
    -------
    combined_spectra
        TYPE: dictionary of spectra
        DESCRIPTION: a dictionary, where the keys are 'region_{region_index}', and the
        values of each key are the weighted mean of the slice. 
    combined_spectra_error
        TYPE: dictionary of spectra
        DESCRIPTION: a dictionary, where the keys are 'region_{region_index}', and the
        values of each key are the weighted mean error of the slice. 
    '''
    
    #loading in the region file
    reg = regions.Regions.read(fname_region, format='ds9')
    
    #loading in fits data cube
    fits_cube = fits.open(fname_cube)
    
    #extracting wcs info from header ext 1, which corresponds to the main data. 
    w = wcs.WCS(fits_cube[1].header)
    
    #define dictionaries to store spectra. Each entry in the dictionary corresponds
    #to 1 region in the .reg file.
    all_spectra = dict()
    all_spectra_error = dict()
    combined_spectra = dict()
    combined_spectra_error = dict()
    
    # loop over regions in .reg file
    for region_index in range(len(reg)):
        #converts region into a 2d array with 1 if inside the region, and 0 if outside
        regmask = reg[region_index].to_pixel(w).to_mask('center').to_image(shape=data.shape)
        
        #adding empty list entries to dictionaries
        all_spectra[f'region_{region_index}'] = []
        all_spectra_error[f'region_{region_index}'] = []
        
        #extracting pixels contained in the regions
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                if regmask[y, x] == 1:
                    # pixel is in this region
                    spec = data[y, x]
                    spec_error = error_data[y, x]
                    
                    #adding values to dictionary lists
                    all_spectra[f'region_{region_index}'].append(spec)
                    all_spectra_error[f'region_{region_index}'].append(spec_error)
        
        #how the pixels are combined depends on the number of pixels in each region
        #at least 1 pixel was found to be in this region:
        if len(all_spectra[f'region_{region_index}']) > 0:
            #currently consists of several appended spectra. 
            #Need to turn into a proper 2d array to easily calculate weighted mean
            all_spectra[f'region_{region_index}'] = np.vstack(all_spectra[f'region_{region_index}'])
            all_spectra_error[f'region_{region_index}'] = np.vstack(all_spectra_error[f'region_{region_index}']) 
            
            #only 1 pixel in region, final value mean is just this pixel value:
            if len(all_spectra[f'region_{region_index}']) == 1:
                combined_spectra[f'region_{region_index}'] = all_spectra[f'region_{region_index}']
                combined_spectra_error[f'region_{region_index}'] = all_spectra_error[f'region_{region_index}']
            
            #more than 1 pixel present, can calculate weighted mean and error:
            else:
                #extracting data from dictionary
                all_data = all_spectra[f'region_{region_index}']
                all_error = all_spectra_error[f'region_{region_index}']
                
                # Calculate weighted mean along position axis (axis 0 after vstack)
                combined_spectra[f'region_{region_index}'] = np.nansum(all_data / all_error**2, axis=0) / np.nansum(1. / all_error**2, axis=0)
                # Uncertainty on the mean
                combined_spectra_error[f'region_{region_index}'] = np.sqrt(1 / np.nansum(1. / all_error**2, axis=0))
                
        #no pixels were found to be in this region:
        else:
            combined_spectra[f'region_{region_index}'] = 'no pixels in this region'
            combined_spectra_error[f'region_{region_index}'] = 'no pixels in this region'
            
    return combined_spectra, combined_spectra_error



def extract_regular_mean_slice_from_region(fname_cube, data, error_data, fname_region):
    '''
    calculates a regular mean, of a data slice over the spacial indices, with 
    the pixels to use selected with a .reg file. Any bad pixels should have their
    value set to 0 before using this function (flagged in DQ, for example), 
    and they will be ignored in the weighted mean calculation. This function uses
    nansum instead of sum, so nans do not need to be removed in the data for
    this function to work.
    
    Parameters
    ----------
    fname_cube
        TYPE: string
        DESCRIPTION: the location of the data cube, where data, error and wavelengths
            came from. 
    data
        TYPE: 2d array of floats
        DESCRIPTION: position and spectral data.
            for [y, x] x and y are position index.
    error_data
        TYPE: 2d array of floats
        DESCRIPTION: position and spectral error data.
                for [y, x] x and y are position index.

    Returns
    -------
    combined_spectra
        TYPE: dictionary of spectra
        DESCRIPTION: a dictionary, where the keys are 'region_{region_index}', and the
        values of each key are the weighted mean of the slice. 
    combined_spectra_error
        TYPE: dictionary of spectra
        DESCRIPTION: a dictionary, where the keys are 'region_{region_index}', and the
        values of each key are the weighted mean error of the slice. 
    '''
    
    #loading in the region file
    reg = regions.Regions.read(fname_region, format='ds9')
    
    #loading in fits data cube
    fits_cube = fits.open(fname_cube)
    
    #extracting wcs info from header ext 1, which corresponds to the main data. 
    w = wcs.WCS(fits_cube[1].header)
    
    #define dictionaries to store spectra. Each entry in the dictionary corresponds
    #to 1 region in the .reg file.
    all_spectra = dict()
    all_spectra_error = dict()
    combined_spectra = dict()
    combined_spectra_error = dict()
    
    # loop over regions in .reg file
    for region_index in range(len(reg)):
        #converts region into a 2d array with 1 if inside the region, and 0 if outside
        regmask = reg[region_index].to_pixel(w).to_mask('center').to_image(shape=data.shape)
        
        #adding empty list entries to dictionaries
        all_spectra[f'region_{region_index}'] = []
        all_spectra_error[f'region_{region_index}'] = []
        
        #extracting pixels contained in the regions
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                if regmask[y, x] == 1:
                    # pixel is in this region
                    spec = data[y, x]
                    spec_error = error_data[y, x]
                    
                    #adding values to dictionary lists
                    all_spectra[f'region_{region_index}'].append(spec)
                    all_spectra_error[f'region_{region_index}'].append(spec_error)
        
        #how the pixels are combined depends on the number of pixels in each region
        #at least 1 pixel was found to be in this region:
        if len(all_spectra[f'region_{region_index}']) > 0:
            #currently consists of several appended spectra. 
            #Need to turn into a proper 2d array to easily calculate weighted mean
            all_spectra[f'region_{region_index}'] = np.vstack(all_spectra[f'region_{region_index}'])
            all_spectra_error[f'region_{region_index}'] = np.vstack(all_spectra_error[f'region_{region_index}']) 
            
            #only 1 pixel in region, final value mean is just this pixel value:
            if len(all_spectra[f'region_{region_index}']) == 1:
                combined_spectra[f'region_{region_index}'] = all_spectra[f'region_{region_index}']
                combined_spectra_error[f'region_{region_index}'] = all_spectra_error[f'region_{region_index}']
            
            #more than 1 pixel present, can calculate weighted mean and error:
            else:
                #extracting data from dictionary
                all_data = all_spectra[f'region_{region_index}']
                all_error = all_spectra_error[f'region_{region_index}']
                
                # Calculate regular mean along position axis (axis 0 after vstack)
                combined_spectra[f'region_{region_index}'] = np.mean(all_data, axis=0)
                # Uncertainty on the mean
                combined_spectra_error[f'region_{region_index}'] = np.sqrt(np.nansum(all_error**2, axis=0))/len(all_error)
                
        #no pixels were found to be in this region:
        else:
            combined_spectra[f'region_{region_index}'] = 'no pixels in this region'
            combined_spectra_error[f'region_{region_index}'] = 'no pixels in this region'
            
    return combined_spectra, combined_spectra_error



def extract_pixels_from_region(fname_cube, data, fname_region):
    '''
    Returns an array, which corresponds to what is used when a region
    is used to calculate weighted means above. 1 is inside the region,
    0 is outside. Unlike the other region functions, this one assumes
    there is only 1 shape located inside the .reg file. If there are multiple,
    only the first is returned.
    
    Parameters
    ----------
    fname_cube
        TYPE: string
        DESCRIPTION: the location of the data cube, where data and wavelengths
            came from. 
    data
        TYPE: 2d array of floats
        DESCRIPTION: position and spectral data.
            for [y, x] x and y are position index.

    Returns
    -------
    region_array
        TYPE: 2d array of floats
        DESCRIPTION: array displaying where the region is, in index coordinates
            for [y, x] x and y are position index.
    '''
    
    #loading in the region file
    reg = regions.Regions.read(fname_region, format='ds9')
    
    #loading in fits data cube
    fits_cube = fits.open(fname_cube)
    
    #extracting wcs info from header ext 1, which corresponds to the main data. 
    w = wcs.WCS(fits_cube[1].header).dropaxis(2)
    
    region_array = reg[0].to_pixel(w).to_mask('center').to_image(shape=data.shape[1:])
             
    return region_array





####################################



'''
STITCHING DATA
'''



def flux_aligner_offset(wave_lower, wave_higher, data_lower, data_higher):
    '''
    This function takes in 2 adjacent wavelength and image data arrays, presumably 
    from the same part of the image fov (field of view), so they correspond to the 
    same location in the sky. It then finds which indices in the lower wavelength data 
    overlap with the beginning of the higher wavelength data, and combines the 2 data 
    arrays in the middle of this region. Lastly, in order for there to be a smooth 
    transition, it offsets the second data so that the mean of the overlapping region 
    is the same in the 2 data sets. This offsetting is makes this function unsuitable
    to use for datasets that did not come from JWST, as differences in channels should 
    be corrected with scaling for other telescopes.
    
    Parameters
    ----------
    wave_lower
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the smaller wavelengths.
    wave_higher
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the larger wavelengths.
    data_lower
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_lower.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    data_higher
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_higher.
            for [wave, y, x] wave is wavelength index, x and y are position index.
            
    Returns
    -------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_lower and data_higher joined together as described above.
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_lower and data_higher joined together as described above.
            for [wave, y, x] wave is wavelength index, x and y are position index.
overlap
    TYPE: integer (index) OR tuple (index)
    DESCRIPTION: index of the wavelength value in wave_lower that equals the first element in wave_higher. In the 
    case of the two wavelength arrays having different intervals, overlap is instead a tuple of the regular
    overlap, followed by the starting index in the 2nd array.
    '''
    
    #check if wavelength interval is the same or different
    check_lower = np.round(wave_lower[1] - wave_lower[0], 4)
    check_higher = np.round(wave_higher[1] - wave_higher[0], 4)
    
    if check_lower == check_higher:
        
        #check where the overlap is
        overlap = np.where(np.round(wave_lower, 2) == np.round(wave_higher[0], 2))[0][0]
        
        #find how many entries are overlapped, subtract 1 for index
        overlap_length = len(wave_lower) -1 - overlap
        
        #making a temp array to scale
        data_higher_temp = np.copy(data_higher)
        
        #find the mean of overlap area
        mean_overlap_lower = np.mean(data_lower[overlap:])
        mean_overlap_higher = np.mean(data_higher[:overlap_length])
        
        #amount to offset by
        mean_difference = mean_overlap_lower - mean_overlap_higher
        
        data_higher_temp += mean_difference
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
    
        #making an index to perform the stitching at
        split_index = overlap_length/2
        
        #check if even or odd, do different things depending on which
        if overlap_length % 2 == 0: #even
            lower_index = overlap + split_index
            higher_index = split_index
            
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap + split_index + 0.5
            higher_index = split_index - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        higher_index = int(higher_index)
        
        image_data = np.hstack((data_lower[:lower_index], data_higher_temp[higher_index:]))
        wavelengths = np.hstack((wave_lower[:lower_index], wave_higher[higher_index:]))
        
    else:
        #check where the overlap is for lower
        overlap_lower = np.where(np.round(wave_lower, 2) == np.round(wave_higher[0], 2))[0][0]
        
        #find how many microns the overlap is
        overlap_micron = wave_lower[-1] - wave_lower[overlap_lower]
        
        #find how many entries of wave_a are overlapped, subtract 1 for index
        overlap_length_lower = len(wave_lower) -1 - overlap_lower
        split_index_lower = overlap_length_lower/2
        
        #number of indices in wave_B over the wavelength range
        overlap_length_higher = int(overlap_micron/check_higher)
        split_index_higher = overlap_length_higher/2
        
        #making a temp array to scale
        data_higher_temp = np.copy(data_higher)
        
        #find the mean of overlap area
        mean_overlap_lower = np.mean(data_lower[overlap_lower:])
        mean_overlap_higher = np.mean(data_higher[:overlap_length_higher])
        
        #amount to offset by
        mean_difference = mean_overlap_lower - mean_overlap_higher
        
        data_higher_temp += mean_difference
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
        
        #check if even or odd, do different things depending on which
        if overlap_length_lower % 2 == 0: #even
            lower_index = overlap_lower + split_index_lower
            
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap_lower + split_index_lower + 0.5
            
        if overlap_length_higher % 2 == 0: #even
            higher_index = split_index_higher
            
        else: #odd, so split_index is a number of the form int+0.5
            higher_index = split_index_higher - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        higher_index = int(higher_index)
        
        image_data = np.hstack((data_lower[:lower_index], data_higher_temp[higher_index:]))
        wavelengths = np.hstack((wave_lower[:lower_index], wave_higher[higher_index:]))
        
        #creating overlap tuple
        overlap = (lower_index, higher_index)
    
    return wavelengths, image_data, overlap



def flux_aligner_offset_reverse(wave_lower, wave_higher, data_lower, data_higher):
    '''
    This function takes in 2 adjacent wavelength and image data arrays, presumably 
    from the same part of the image fov (field of view), so they correspond to the 
    same location in the sky. It then finds which indices in the lower wavelength data 
    overlap with the beginning of the higher wavelength data, and combines the 2 data 
    arrays in the middle of this region. Lastly, in order for there to be a smooth 
    transition, it offsets the FIRST data so that the mean of the overlapping region 
    is the same in the 2 data sets. It does the opposite to flux_aligner_offset.
    This offsetting is makes this function unsuitable
    to use for datasets that did not come from JWST, as differences in channels should 
    be corrected with scaling for other telescopes.
    
    Parameters
    ----------
    wave_lower
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the smaller wavelengths.
    wave_higher
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the larger wavelengths.
    data_lower
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_lower.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    data_higher
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_higher.
            for [wave, y, x] wave is wavelength index, x and y are position index.
            
    Returns
    -------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_lower and data_higher joined together as described above.
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_lower and data_higher joined together as described above.
            for [wave, y, x] wave is wavelength index, x and y are position index.
overlap
    TYPE: integer (index) OR tuple (index)
    DESCRIPTION: index of the wavelength value in wave_lower that equals the first element in wave_higher. In the 
    case of the two wavelength arrays having different intervals, overlap is instead a tuple of the regular
    overlap, followed by the starting index in the 2nd array.
    '''
    
    #check if wavelength interval is the same or different
    check_lower = np.round(wave_lower[1] - wave_lower[0], 4)
    check_higher = np.round(wave_higher[1] - wave_higher[0], 4)
    
    if check_lower == check_higher:
        
        #check where the overlap is
        overlap = np.where(np.round(wave_lower, 2) == np.round(wave_higher[0], 2))[0][0]
        
        #find how many entries are overlapped, subtract 1 for index
        overlap_length = len(wave_lower) -1 - overlap
        
        #making a temp array to scale
        data_lower_temp = np.copy(data_lower)
        
        #find the mean of overlap area
        mean_overlap_lower = np.mean(data_lower[overlap:])
        mean_overlap_higher = np.mean(data_higher[:overlap_length])
        
        #amount to offset by
        mean_difference = mean_overlap_higher - mean_overlap_lower
        
        data_lower_temp += mean_difference
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
    
        #making an index to perform the stitching at
        split_index = overlap_length/2
        
        #check if even or odd, do different things depending on which
        if overlap_length % 2 == 0: #even
            lower_index = overlap + split_index
            higher_index = split_index
            
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap + split_index + 0.5
            higher_index = split_index - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        higher_index = int(higher_index)
        
        image_data = np.hstack((data_lower_temp[:lower_index], data_higher[higher_index:]))
        wavelengths = np.hstack((wave_lower[:lower_index], wave_higher[higher_index:]))
        
    else:
        #check where the overlap is for lower
        overlap_lower = np.where(np.round(wave_lower, 2) == np.round(wave_higher[0], 2))[0][0]
        
        #find how many microns the overlap is
        overlap_micron = wave_lower[-1] - wave_lower[overlap_lower]
        
        #find how many entries of wave_a are overlapped, subtract 1 for index
        overlap_length_lower = len(wave_lower) -1 - overlap_lower
        split_index_lower = overlap_length_lower/2
        
        #number of indices in wave_B over the wavelength range
        overlap_length_higher = int(overlap_micron/check_higher)
        split_index_higher = overlap_length_higher/2
        
        #making a temp array to scale
        data_lower_temp = np.copy(data_lower)
        
        #find the mean of overlap area
        mean_overlap_lower = np.mean(data_lower[overlap_lower:])
        mean_overlap_higher = np.mean(data_higher[:overlap_length_higher])
        
        #amount to offset by
        mean_difference = mean_overlap_higher - mean_overlap_lower
        
        data_lower_temp += mean_difference
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
        
        #check if even or odd, do different things depending on which
        if overlap_length_lower % 2 == 0: #even
            lower_index = overlap_lower + split_index_lower
            
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap_lower + split_index_lower + 0.5
            
        if overlap_length_higher % 2 == 0: #even
            higher_index = split_index_higher
            
        else: #odd, so split_index is a number of the form int+0.5
            higher_index = split_index_higher - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        higher_index = int(higher_index)
        
        image_data = np.hstack((data_lower_temp[:lower_index], data_higher[higher_index:]))
        wavelengths = np.hstack((wave_lower[:lower_index], wave_higher[higher_index:]))
        
        #creating overlap tuple
        overlap = (lower_index, higher_index)
    
    return wavelengths, image_data, overlap



def flux_aligner_manual(wave_lower, wave_higher, data_lower, data_higher):
    '''
    This function takes in 2 adjacent wavelength and image data arrays, presumably 
    from the same part of the image fov (field of view), so they correspond to the 
    same location in the sky. It then finds which indices in the lower wavelength data 
    overlap with the beginning of the higher wavelength data, and combines the 2 data 
    arrays in the middle of this region. This function does not apply an offset
    or scaling to combine the 2 channels, as it is made for manual alignment.
    
    Parameters
    ----------
    wave_lower
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the smaller wavelengths.
    wave_higher
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the larger wavelengths.
    data_lower
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_lower.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    data_higher
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_higher.
            for [wave, y, x] wave is wavelength index, x and y are position index.
            
    Returns
    -------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_lower and data_higher joined together as described above.
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_lower and data_higher joined together as described above.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    overlap
        TYPE: tuple of integers (index)
        DESCRIPTION: indices corresponding to where the overlap is in wave_lower and wave_higher.
        
    '''
    
    #check if wavelength interval is the same or different
    check_lower = np.round(wave_lower[1] - wave_higher[0], 4)
    check_higher = np.round(wave_higher[1] - wave_higher[0], 4)
    
    if check_lower == check_higher:
    
        #check where the overlap is
        overlap = np.where(np.round(wave_lower, 2) == np.round(wave_higher[0], 2))[0][0]
        
        #find how many entries are overlapped, subtract 1 for index
        overlap_length = len(wave_lower) -1 - overlap
        
        #making a temp array to scale
        temp = np.copy(data_higher)
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
        
        split_index = overlap_length/2
        
        #check if even or odd, do different things depending on which
        if overlap_length%2 == 0: #even
            lower_index = overlap + split_index
            upper_index = split_index
            #print(lower_index, upper_index)
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap + split_index + 0.5
            upper_index = split_index - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        
        image_data = np.hstack((data_lower[:lower_index], temp[upper_index:]))
        wavelengths = np.hstack((wave_lower[:lower_index], wave_higher[upper_index:]))
        
    else:
        #check where the overlap is, only works for wave_lower
        overlap_lower = np.where(np.round(wave_lower, 2) == np.round(wave_higher[0], 2))[0][0]
        
        #find how many microns the overlap is
        overlap_micron = wave_lower[-1] - wave_lower[overlap_lower]
        
        #find how many entries of wave_lower are overlapped, subtract 1 for index
        overlap_length_lower = len(wave_lower) -1 - overlap_lower
        split_index_lower = overlap_length_lower/2
        
        #number of indices in wave_higher over the wavelength range
        overlap_length_higher = int(overlap_micron/check_higher)
        split_index_higher = overlap_length_higher/2
        
        #making a temp array to scale
        temp = np.copy(data_higher)
        
        #check if even or odd, do different things depending on which
        if overlap_length_lower%2 == 0: #even
            lower_index = overlap_lower + split_index_lower
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap_lower + split_index_lower + 0.5
            
        if overlap_length_higher%2 == 0: #even
            upper_index = split_index_higher
        else: #odd, so split_index is a number of the form int+0.5
            upper_index = split_index_higher - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        
        image_data = np.hstack((data_lower[:lower_index], temp[upper_index:]))
        wavelengths = np.hstack((wave_lower[:lower_index], wave_higher[upper_index:]))
        overlap = (lower_index, upper_index)
    
    return wavelengths, image_data, overlap
    




####################################



'''
REMOVING EMISSION AND ABSORPTION LINES
'''



def emission_line_remover(data, width, limit):
    '''
    A function that removes emission lines from data, using an inputted 
    maximum width and height defintion to find emission lines.
    
    Parameters
    ----------
    data
        TYPE: 1d array of floats
        DESCRIPTION: a spectra.
    width
        TYPE: integer
        DESCRIPTION: max index width of spectral peaks.
    limit
        TYPE: float
        DESCRIPTION: defines the min flux dif between an emission line and continuum.

    Returns
    -------
    new_data 
        TYPE: 1d array of floats
        DESCRIPTION: spectra with emission lines removed.
    '''
    
    #defining data set to perform operations on
    new_data = np.copy(data)
    
    for wave in range(len(new_data)):
        #central regions away from the edges
        if (wave > width) and (wave < len(data) - 1 - width):
            if (new_data[wave] - new_data[wave - width] > limit) and (new_data[wave] - new_data[wave + width] > limit):
                new_data[wave] = (new_data[wave + width] + new_data[wave - width])/2
                
        #left edge case
        elif (wave < width):
            if new_data[wave] - new_data[wave+width] > limit:
                new_data[wave] = new_data[wave+width]
                
        #right edge case
        elif (wave > len(data) - 1 - width):
            if new_data[wave] - new_data[wave - width] > limit:
                new_data[wave] = new_data[wave - width]
    
    return new_data



def absorption_line_remover(data, width, limit):
    '''
    A function that removes absorption lines from data, using an inputted 
    maximum width and height defintion to find absorption lines.
    
    Parameters
    ----------
    data
        TYPE: 1d array
        DESCRIPTION: a spectra.
    width
        TYPE: integer
        DESCRIPTION: max index width of spectral peaks.
    limit
        TYPE: float
        DESCRIPTION: defines the min flux dif between an absorption line and continuum.

    Returns
    -------
    new_data 
        TYPE: 1d array
        DESCRIPTION: spectra with absorption lines removed.
    '''
    
    new_data = np.copy(data)
    
    for wave in range(len(new_data)):
        #central regions away from the edges
        if (wave > width) and (wave < len(data) - 1 - width):
            if (new_data[wave] - new_data[wave - width] < limit) and (new_data[wave] - new_data[wave + width] < limit):
                new_data[wave] = (new_data[wave + width] + new_data[wave - width])/2
                
        #left edge case
        elif (wave < width):
            if new_data[wave] - new_data[wave + width] < limit:
                new_data[wave] = new_data[wave + width]
                
        #right edge case
        elif (wave > len(data) - 1 - width):
            if new_data[wave] - new_data[wave - width] < limit:
                new_data[wave] = new_data[wave - width]
    
    return new_data



####################################



'''
LINE FITTING
'''



def line_fitter(wavelengths, data):
    '''
    A function that fits a 20th order polynomial to input data using RANSAC.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d array
        DESCRIPTION: wavelengths of spectra.
    data
        TYPE: 1d array
        DESCRIPTION: a spectra.

    Returns
    -------
    line_ransac
        TYPE: 1d array
        DESCRIPTION: the line that was fit to the data.
    '''
    
    wavelengths = wavelengths.reshape(-1,1)
    data = data.reshape(-1,1)

    # Init the RANSAC regressor
    ransac = make_pipeline(PolynomialFeatures(20), RANSACRegressor(max_trials=200000, random_state=41))

    # Fit with RANSAC
    ransac.fit(wavelengths, data)

    # Get the fitted data result
    line_ransac = ransac.predict(wavelengths)
    
    return line_ransac



####################################



'''
CONTINUUM FITTING
'''



def linear_continuum_single_channel(wavelengths, image_data, wave_list):
    '''
    Fits a series of linear functions at input indices, to data to serve
    as its continuum.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d array
        DESCRIPTION: wavelengths of spectra, with implicit units of microns.
    data
        TYPE: 1d array
        DESCRIPTION: a spectra to fit the continuum to, with implicit units of MJy/sr.
    wave_list
        TYPE: list of floats
        DESCRIPTION: 4 wavelengths, to put the anchor points between each linear function
            on. In units of microns.

    Returns
    -------
    continuum
        TYPE: 1d array
        DESCRIPTION: a series of linear functions acting as the continuum, with implicit units of MJy/sr.
    '''

    
    def temp_index_generator(wave):
        #defining this within my function so i can make more advanced rounding code
        #without cluttering my function with repetition
        temp_index = np.where(np.round(wavelengths, 3) == wave)[0]
        if len(temp_index) == 0:
            temp_index = np.where(np.round(wavelengths, 2) == wave)[0][0]
        else:
            temp_index = temp_index[0]
            
        return temp_index
        
    temp_index_1 = temp_index_generator(wave_list[0])
    temp_index_2 = temp_index_generator(wave_list[1])
    temp_index_3 = temp_index_generator(wave_list[2])
    temp_index_4 = temp_index_generator(wave_list[3])
    
    #calculating the slope of the line to use
    #also want to have the very end values included so that the shape of the array is consistent
    #preventing the value found from being on a line or something
    pah_slope_beginning = image_data[0]
    pah_slope_ending = image_data[-1]
    
    pah_slope_1 = np.median(image_data[temp_index_1 - 15:15+temp_index_1])
    pah_slope_2 = np.median(image_data[temp_index_2 - 15:15+temp_index_2])
    
    pah_slope = (pah_slope_1 - pah_slope_beginning)/\
        (wavelengths[temp_index_1] - wavelengths[0])

    continuum0 = pah_slope*(wavelengths[:temp_index_1] - wavelengths[0]) + pah_slope_beginning
    
    pah_slope = (pah_slope_2 - pah_slope_1)/\
        (wavelengths[temp_index_2] - wavelengths[temp_index_1])

    #putting it all together
    continuum1 = pah_slope*(wavelengths[temp_index_1:temp_index_2] - wavelengths[temp_index_1]) + pah_slope_1
    
    #going to try using 10 data points for weighted mean instead of 40
    
    #before adding the adjustment list, points 6, 7, and 8 had 100 subtracted from them
    
    pah_slope_3 = np.median(image_data[temp_index_3 - 15:15+temp_index_3])
    pah_slope_4 = np.median(image_data[temp_index_4 - 15:15+temp_index_4])

    
    pah_slope = (pah_slope_3 - pah_slope_2)/\
        (wavelengths[temp_index_3] - wavelengths[temp_index_2])
    continuum2 = pah_slope*(wavelengths[temp_index_2:temp_index_3] - wavelengths[temp_index_2]) + pah_slope_2
    
    pah_slope = (pah_slope_4 - pah_slope_3)/\
        (wavelengths[temp_index_4] - wavelengths[temp_index_3])
    continuum3 = pah_slope*(wavelengths[temp_index_3:temp_index_4] - wavelengths[temp_index_3]) + pah_slope_3
    
    pah_slope = (pah_slope_ending - pah_slope_4)/\
        (wavelengths[-1] - wavelengths[temp_index_4])
    continuum4 = pah_slope*(wavelengths[temp_index_4:] - wavelengths[temp_index_4]) + pah_slope_4
    
    
    #putting it all together
    continuum = np.concatenate((continuum0, continuum1))
    continuum = np.concatenate((continuum, continuum2))
    continuum = np.concatenate((continuum, continuum3))
    continuum = np.concatenate((continuum, continuum4))

    return continuum



####################################



'''
UNITS, ERRORS, INTEGRALS
'''



def unit_changer(wavelengths, data):
    '''
    #takes a cube, assumed to have implicit units of MJy/sr, and adds them using
    astropy units. It then converts these units to W/m^2/micron/sr, ready for
    integration.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d array
        DESCRIPTION: wavelengths of spectra, with implicit units of microns.
    data
        TYPE: 1d array
        DESCRIPTION: a spectra, with implicit units of MJy/sr.

    Returns
    -------
    new_data
        TYPE: 1d array
        DESCRIPTION: a spectra, that has had its implicit units changed to W/m^2/micron/sr.
    '''
    
    #defining array to store new data
    new_data = np.zeros(data.shape)
    new_data_with_units = np.zeros(data.shape)
    
    #defining array of data that has units added
    data_with_units = (data*10**6)*(u.Jy/u.sr)
    
    #converting units, including replacing frequency dependence to wavelength dependence
    new_data_with_units = data_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavelengths*u.micron))
    
    #removing units
    new_data_with_units = new_data_with_units*(u.micron)
    new_data_with_units = new_data_with_units*(u.m)
    new_data_with_units = new_data_with_units*(u.m)
    new_data_with_units = new_data_with_units*(u.sr/u.W)

    #verifying that units are gone by transferring to a new array
    for wave in range(len(data)):
        new_data[wave] = float(new_data_with_units[wave])
    
    return new_data



def pah_feature_integrator(wavelengths, data):
    '''
    A function for integrating PAH features, using Simpson's method. Units are converted
    to SI before integration. It is assumed that any errors on the data are much
    larger than the errors from Simpson's method, so these are not calculated. 
    Note that simpson's method uses evenly spaced wavelengths, and so the wavelength
    array of the data should not contain multiple wavelength intervals.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d array
        DESCRIPTION: wavelengths of spectra, with implicit units of microns.
    data
        TYPE: 1d array
        DESCRIPTION: a spectra, with implicit units of MJy/sr.

    Returns
    -------
    integral
        TYPE: float
        DESCRIPTION: The calculated intensity, in units of W/m^2/sr.
    '''
    
    #converting data to correct units
    integrand = unit_changer(wavelengths, data)

    #applying simpson's method
    
    #summing odd indices excluding first and last entry
    odd_sum = 0
    
    for i in range(1, len(integrand), 2):
        odd_sum += integrand[i] 
    
    #summing even indices, excluding first and last entry
    even_sum = 0    

    for i in range(2, len(integrand), 2):
        even_sum += integrand[i] 
    
    #calculating wavelength interval
    h = wavelengths[1] - wavelengths[0]
    
    #final integral from the even and odd sums, and the edge cases
    integral = (h/3)*(integrand[0] + integrand[-1] + 4*odd_sum + 2*even_sum)
    
    return integral



def pah_feature_integrator_no_units(wavelengths, data):
    '''
    A function for integrating PAH features, using Simpson's method. Units are NOT converted
    to SI before integration. Currently, this is used for calculating 
    Synthetic IFUs. It is assumed that any errors on the data are much
    larger than the errors from Simpson's method, so these are not calculated. 
    Note that simpson's method uses evenly spaced wavelengths, and so the wavelength
    array of the data should not contain multiple wavelength intervals.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d array
        DESCRIPTION: wavelengths of spectra, with implicit units of microns.
    data
        TYPE: 1d array
        DESCRIPTION: a spectra, with implicit units of MJy/sr.

    Returns
    -------
    integral
        TYPE: float
        DESCRIPTION: The calculated intensity, in units of W/m^2/sr.
    '''
    
    #no unit conversion is performed
    integrand = data

    #applying simpson's method
    
    #summing odd indices excluding first and last entry
    odd_sum = 0
    
    for i in range(1, len(integrand), 2):
        odd_sum += integrand[i] 
    
    #summing even indices, excluding first and last entry
    even_sum = 0    

    for i in range(2, len(integrand), 2):
        even_sum += integrand[i] 
    
    #calculating wavelength interval
    h = wavelengths[1] - wavelengths[0]
    
    #final integral from the even and odd sums, and the edge cases
    integral = (h/3)*(integrand[0] + integrand[-1] + 4*odd_sum + 2*even_sum)
    
    return integral



def Calculate_R(wavelength):
    '''
    Calculates the resolution R (wavelength divided by wavelength interval) for MIRI MRS data.
    Currently, this consists of a series of fit quadratics. 
    Credit to Aiden Hembruff for making this function.
    
    For Nirspec data, assume the R is 1000, as per the jdocs for the G---M filters
    
    Note that there is currently a 0.2 micron overlap between nirspec and miri, and
    it is assumed this is miri data. I do not use anything in nirspec above 4.9 
    microns, but it is noted just in case.
    
    Parameters
    ----------
    wavelength
        TYPE: float
        DESCRIPTION: The wavelength for which R will be calculated, in units of microns.

    Returns
    -------
    R
        TYPE: float
        DESCRIPTION: The resolution R, corresponding to the input wavelength.
    '''
    
    #nirspec:
    if wavelength < 4.9:
        coeff = [1000, 0, 0]
    
    #ch1:
    #A:
    elif 4.9 <= wavelength <= 5.74 :
            coeff = [ 8.4645410e+03, -2.4806001e+03,  2.9600000e+02]
    #B:
    elif 5.66 <= wavelength <= 6.63 :
        coeff = [ 1.3785873e+04, -3.8733003e+03,  3.6100000e+02]
    #C:
    elif 6.53 <= wavelength <= 7.65 :
        coeff = [ 9.0737793e+03, -2.0355999e+03,  1.7800000e+02]
        
    #ch2:
    #A:
    elif 7.51 <= wavelength <= 8.76 :
        coeff = [-1.3392804e+04,  3.8513999e+03, -2.1800000e+02]
    #B:
    elif 8.67 <= wavelength <= 10.15 :
        coeff = [-3.0707996e+03,  1.0530000e+03, -4.0000000e+01]
    #C:
    elif 10.01 <= wavelength <= 11.71:
        coeff = [-1.4632270e+04,  3.0245999e+03, -1.2700000e+02]
        
    #ch3:
    #A:
    elif 11.55 <= wavelength <= 13.47:
        coeff = [-6.9051500e+04,  1.1490000e+04, -4.5800000e+02]
    #B:
    elif 13.29 <= wavelength <= 15.52:
        coeff = [ 3.2627500e+03, -1.9200000e+02,  9.0000000e+00]
    #C:
    elif 15.41 <= wavelength <= 18.02:
        coeff = [-1.2368500e+04,  1.4890000e+03, -3.6000000e+01]
            
    #ch4:
    #A:
    elif 17.71 <= wavelength <= 20.94:
        coeff = [-1.1510681e+04,  1.2088000e+03, -2.7000000e+01]
    #B:
    elif 20.69 <= wavelength <= 24.44:
        coeff = [-4.5252500e+03,  5.4800000e+02, -1.2000000e+01]
    #C:
    elif 23.22 <= wavelength <= 28.1:
        coeff = [-4.9578794e+03,  5.5819995e+02, -1.2000000e+01]
                
    R = coeff[0] + (coeff[1]*wavelength) + (coeff[2]*(wavelength**2))
        
    return(R)



def error_finder(wavelengths, data, feature_wavelength, feature_indices, error_index):
    '''
    Calculates the error of a nearby PAH feature, using a part of the spectrum
    lacking in activity, i.e. only dust continuum. Wavelength interval is 
    calculated using the resolution R for a wavelength in the center of the PAH feature in question;
    since R doesnt vary a whole lot over a given 0.1 micron wavelength range, the 
    specific wavelength used for this doesn't matter a whole lot.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d array
        DESCRIPTION: wavelengths of spectra, with implicit units of microns.
    data
        TYPE: 1d array
        DESCRIPTION: a spectra, with implicit units of MJy/sr.
    feature_wavelength
        TYPE: float
        DESCRIPTION: the wavelength in microns, for the resolution R to be calculated.
            Corresponds to the PAH feature in question.
    feature_indices
        TYPE: tuple of indices
        DESCRIPTION: The indices over which the PAH feature in question is integrated,
            provided for calculating the number of points.
    error_index:
        TYPE: index
        DESCRIPTION: The index corresponding to a region where the RMS of the noise is 
            measured, so it contains only dust emission and no features or lines.

    Returns
    -------
    error
        TYPE: float
        DESCRIPTION: The calculated error of the intensity, in units of W/m^2/sr.
    '''
    
    #convert units to SI
    rms_data = unit_changer(wavelengths[error_index-25:error_index+25], data[error_index-25:error_index+25])
    
    #calculating root mean square of region to be used for error calculation
    rms = (np.var(rms_data))**0.5
    
    #determining the resolution of the PAH feature
    resolution = Calculate_R(feature_wavelength)
    
    #calculating the wavelength interval
    delta_wave = feature_wavelength/resolution
    
    #calculating the number of points used in integrating the PAH feature
    num_points = (wavelengths[feature_indices[1]] - wavelengths[feature_indices[0]])/delta_wave
    
    #calculating error
    error = (rms*delta_wave*(num_points)**0.5)
    
    return error





####################################



'''
UNUSED FUNCTIONS
'''



def border_remover(data, amount=0):
    '''
    This function shrinks the fov to remove the areas where there is no data,
    or where there is likely to be bad data on the fov. Note that this function
    is currently unused, as it is better to remove bad data using .reg files.
    
    Parameters
    ----------
    data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    amount
        TYPE: positive integer (index)
        DESCRIPTION: how much to remove from each edge (optional).
    
    Returns
    -------
    shrunk_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, edges removed.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    '''
    
    #find the length of the 2 dimensions to be shrunk
    len_y = len(data[0,:,0])
    len_x = len(data[0,0,:])
    
    #if amount of unspecified, define amount to be removed based off of length
    if amount == 0:
        #will serve as beginning boundary
        amount_y = len_y//5
        amount_x = len_x//5
        
        #ending boundary, such that the same number of pixels are removed from both sides
        end_y = len_y - amount_y
        end_x = len_x - amount_x
        
        shrunk_data = data[:, amount_y:end_y, amount_x:end_x]
    
    #if amount is nonzero, use that instead
    else:
        end_y = len_y - amount
        end_x = len_x - amount
        
        shrunk_data = data[:, amount:end_y, amount:end_x]
    
    return shrunk_data



def regrid(data, error_data, N):
    '''
    This function regrids a data cube, such that its pixel size goes from 1x1 to NxN, where N is specified.
    This is done by taking a weighted mean. Note that if the size of the array is not
    divisible by N, the indices at the end are discarded. 
    This should be ok since edge pixels are usually ignored.
    
    Parameters
    ----------
    data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data cube to be rebinned.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data cube to be rebinned.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    N
        TYPE: positive integer
        DESCRIPTION: the value N such that the number of pixels that go into the new pixel are N^2,
            i.e. before eacch pixel is 1x1, after its NxN per pixel.

    Returns
    -------
    rebinned_data
        TYPE: 3d array of floats
        DESCRIPTION: new data, on a smaller grid size in the positional dimensions.

    rebinned_error_data
        TYPE: 3d array of floats
        DESCRIPTION: new error data, on a smaller grid size in the positional dimensions.
    '''
    
    #defining current size
    size_y = len(data[0,:,0])
    size_x = len(data[0,0,:])
    
    #Figure out if any indices need to be discarded, so that the current size will
    #be divisible by N
    remainder_y = size_y % N
    remainder_x = size_x % N
    
    if remainder_y != 0:
        size_y = size_y - remainder_y
        
    if remainder_x != 0:
        size_x = size_x - remainder_x

    #building new arrays
    size_wavelength = int(len(data[:,0,0]))
    
    rebinned_data = np.zeros((size_wavelength, int(size_y/N), int(size_x/N)))
    rebinned_error_data = np.zeros((size_wavelength, int(size_y/N), int(size_x/N)))
    
    for y in range(0, size_y, N):
        for x in range(0, size_x, N):
            #note that y:y+N will have y+1,...,y+N, with length N, so want to subtract 1 from these to include y
            
            #taking weighted mean over the pixels to be put in 1 bin
            temp_data, temp_error_data = weighted_mean_finder(
                data[:, y:y + N, x:x + N], error_data[:, y:y + N, x:x + N])
            
            #adding new pixel to array. y/N and x/N should always be integers, because the remainder was removed above.
            rebinned_data[:, int(y/N), int(x/N)] = temp_data
            rebinned_error_data[:,int(y/N),int(x/N)] = temp_error_data
            
    return rebinned_data, rebinned_error_data