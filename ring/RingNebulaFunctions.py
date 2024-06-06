
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
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_lower and data_higher joined together as described above.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_lower and data_higher joined together as described above.
    overlap
        TYPE: integer (index)
        DESCRIPTION: index of the wavelength value in wave_lower that equals the first element in wave_higher
    '''
    
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
    
    return image_data, wavelengths, overlap



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
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_lower and data_higher joined together as described above.
            for [wave, y, x] wave is wavelength index, x and y are position index.
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_lower and data_higher joined together as described above.
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
    
    return image_data, wavelengths, overlap
    




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
UNITS, ERRORS, INTEGRAL
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



def RMS(data):
    
    return (np.var(data))**0.5



def SNR(intensity, rms, delta_wave, num_points):
    
    return intensity/(rms*2*delta_wave*(num_points/2)**0.5)



def extract_weighted_mean_from_region(fname_cube, wavelengths, data, error_data, fname_region):
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
    wavelengths
        TYPE: 1d array
        DESCRIPTION: wavelengths of spectra, with implicit units of microns.
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
        values of each key are the weighted mean spectrum. 
    combined_spectra_error
        TYPE: dictionary of spectra
        DESCRIPTION: a dictionary, where the keys are 'region_{region_index}', and the
        values of each key are the weighted mean spectrum error. 
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
                
                # Calculate weighted mean along wavelength axis (axis 0)
                combined_spectra[f'region_{region_index}'] = np.nansum(all_data / all_error**2, axis=0) / np.nansum(1. / all_error**2, axis=0)
                # Uncertainty on the mean
                combined_spectra_error[f'region_{region_index}'] = np.sqrt(1 / np.nansum(1. / all_error**2, axis=0))
                
        #no pixels were found to be in this region:
        else:
            combined_spectra[f'region_{region_index}'] = 'no pixels in this region'
            combined_spectra_error[f'region_{region_index}'] = 'no pixels in this region'
            
    return combined_spectra, combined_spectra_error



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
    
    rebinned_data = np.zeros((size_wavelength, int(size_x/N), int(size_y/N)))
    rebinned_error_data = np.zeros((size_wavelength, int(size_x/N), int(size_y/N)))
    
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