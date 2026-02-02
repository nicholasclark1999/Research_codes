#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:09:02 2024

@author: nclark
"""

#%%
#standard stuff
import numpy as np

#used for fits file handling
from astropy.io import fits

#needed for els' region function
import regions
from astropy.wcs import wcs



#function for data cubes

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
    new_data
        TYPE: 1d array
        DESCRIPTION: a spectra, that has had its implicit units changed to W/m^2/micron/sr.
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
                
                # Calculate weighted mean along position axis (axis 0 after vstack)
                combined_spectra[f'region_{region_index}'] = np.nansum(all_data / all_error**2, axis=0) / np.nansum(1. / all_error**2, axis=0)
                # Uncertainty on the mean
                combined_spectra_error[f'region_{region_index}'] = np.sqrt(1 / np.nansum(1. / all_error**2, axis=0))
                
        #no pixels were found to be in this region:
        else:
            combined_spectra[f'region_{region_index}'] = 'no pixels in this region'
            combined_spectra_error[f'region_{region_index}'] = 'no pixels in this region'
            
    return combined_spectra, combined_spectra_error



#function for slices

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
            for [y, x] wave is wavelength index, x and y are position index.
    error_data
        TYPE: 2d array of floats
        DESCRIPTION: position and spectral error data.
                for [y, x] wave is wavelength index, x and y are position index.

    Returns
    -------
    combined_spectra
        TYPE: dictionary of spectra
        DESCRIPTION: a dictionary, where the keys are 'region_{region_index}', and the
        values of each key are the weighted of the slice. 
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




