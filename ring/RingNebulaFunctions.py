
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

#needed for ryan's reproject function
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_exact
#from jwst import datamodels

#rebinning module
from reproject import reproject_interp, reproject_adaptive



####################################



'''
REPROJECTING DATA
'''

#needed for ryans reproject function
def get_2d_wcs_from_cube(fname):
    '''
    Gets 2D (spatial) WCS from IRS cube.
    For some reason, extracting WCS from cubism cubes doesn't work well
    (the spectral axis messes things up).
​
    '''
    fits_in = fits.open(fname)
    w_in = wcs.WCS(fits_in[1].header, fobj=fits_in, naxis=2)
    # Turn WCS into a header, and then back to WCS again (this cleans up garbage to do with the 3rd axis we don't want anyway)
    w_in = wcs.WCS(w_in.to_header())
    return w_in



#needed for ryans reproject function
def clip_spikes_spec(spec,
                     spec_unc,
                     window_size=20,
                     thresh_factor=5.,
                     snr_thresh=5.):
    
    spec_out = np.zeros(spec.shape)
    spec_unc_out = np.zeros(spec.shape)

    snr = np.log10(np.abs(spec / spec_unc))
    x = snr
    y = np.log10(np.abs(spec))

    good = np.isfinite(x) & np.isfinite(y) & (spec > 0)

    spec_out[good] = spec[good]
    spec_unc_out[good] = spec_unc[good]

    # good = np.isfinite(y)
    if np.sum(good) > 10:
        x = x[good]
        y = y[good]

        medx = np.nanmedian(x)
        medy = np.nanmedian(y) + 1.

        maxx = np.nanmax(x)
        maxy = y[x == maxx]

        if maxy.size > 1:
            maxy = maxy[0]

        slope = (maxy - medy) / (maxx - medx)
        intercept = maxy - (slope * maxx)

        # Find points that are 1 dex above this line
        #print(x.size)
        #print(y.size)
        #print(slope, intercept)
        i_outlier = (y > (slope * x + intercept + 1.))
        #print(f"Found {np.sum(i_outlier)} outliers")

        # Set outliers to 0
        spec_out[good][i_outlier] = 0.
        spec_unc_out[good][i_outlier] = 0.

    return spec_out, spec_unc_out



#needed for ryans reproject function
def clip_spikes_cube(cube, cube_unc):
    cube_out = np.zeros(cube.shape)
    cube_unc_out = np.zeros(cube.shape)
    nw, nx, ny = cube.shape
    #print(cube.shape)
    for ix in range(nx):
        #print(f"row {ix}/{nx}")
        for iy in range(ny):
            # print(f'ix, iy = {ix}, {iy}') this was originally commented out before i got it
            spec = cube[:, ix, iy]
            spec_unc = cube_unc[:, ix, iy]
            res = clip_spikes_spec(spec, spec_unc)
            cube_out[:, ix, iy] = res[0]
            cube_unc_out[:, ix, iy] = res[1]
    return cube_out, cube_unc_out


#ryans reproject function
def reproject_cube(fname_fits, fname,
                     uncertainty=False,
                     clip_spikes=False):
    
    #fname_fits is file to be changed, fname is reference file
    
    input_data = []
    data = np.nansum(fits.open(fname)['SCI'].data, axis=0)
    hdr = fits.open(fname)['SCI'].header
    w = wcs.WCS(hdr).dropaxis(2)
    input_data += [(data, w)]

    wcs_out, shape_out = find_optimal_celestial_wcs(input_data)


    # spectrum_im = datamodels.MultiSpecModel(fname_fits)
    spectrum_im = datamodels.open(fname_fits)
    data = spectrum_im.data
    data_unc = fits.open(fname_fits)['ERR'].data
    data_dq = fits.open(fname_fits)['DQ'].data
    wcs_cube = get_2d_wcs_from_cube(fname_fits)

    if clip_spikes:
        #print("projection_field: clipping spikes")
        data, data_unc = clip_spikes_cube(data, data_unc)

    if uncertainty:
        data = data_unc

    cube = np.zeros([shape_out[0], shape_out[1], data.shape[0]])

    n_planes = data.shape[0]
    for i in range(n_planes):
        #print(f'Reprojecting plane {i} of {n_planes}')
        arr = data[i, :, :]
        if uncertainty:
            arr = arr**2

        dqmask = data_dq[i, :, :] != 0
        # Dilate the mask by 1 pixel
        # print("Binary dilation") this was originally commented out before i got it
        # dqmask = ndimage.binary_dilation(dqmask)
        arr[dqmask] = np.nan
        #print("Reprojection")
        arr, _ = reproject_exact((arr, wcs_cube),
                                 wcs_out,
                                 shape_out,
                                 parallel=False)
        if uncertainty:
            arr = np.sqrt(arr)
        cube[:, :, i] = arr
        #print("Done")
    return cube




####################################



'''
LOADING DATA
'''

#there are a lot of files, so make a function to sort it out

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



def loading_function_reproject(file_loc, file_loc2, header_index):
    '''
    This function loads in JWST MIRI and NIRSPEC fits data cubes, and extracts wavelength 
    data from the header and builds the corresponding wavelength array. It then reprojects the fits files
    to match the header of the fits file located at file_loc2, and saves the data in a separate folder.
    
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
    image_file2 = get_pkg_data_filename(file_loc2)
    
    hdu=fits.open(file_loc) #need to use this to get reproject to work
    hdu2=fits.open(file_loc2)
    
    #header data
    science_header = fits.getheader(image_file, 1)
    science_header2 = fits.getheader(image_file2, 1)
    
    #wavelength data from header
    number_wavelengths = science_header["NAXIS3"]
    wavelength_increment = science_header["CDELT3"]
    wavelength_start = science_header["CRVAL3"]
    
    #constructing the ending point using given data
    #subtracting 1 so wavelength array is the right size.
    wavelength_end = wavelength_start + (number_wavelengths - 1)*wavelength_increment
    
    #extracting image data, error data
    image_data = reproject_cube(file_loc, file_loc2)
    error_data = reproject_cube(file_loc, file_loc2, uncertainty=True)
    
    #changing axes order
    temp = np.swapaxes(image_data, 0, 2)
    image_data = np.swapaxes(temp, 1, 2)
    
    temp = np.swapaxes(error_data, 0, 2)
    error_data = np.swapaxes(temp, 1, 2)
    
    #replacing nans with 0
    where_are_NaNs = np.isnan(image_data) 
    image_data[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(error_data) 
    error_data[where_are_NaNs] = 0
    
    #note that the number of data points is reprojected also, so need to modify wavelength increment to account for this
    
    wavelength_increment = (wavelength_end - wavelength_start)/len(image_data[:,0,0])
    
    #making wavelength array, in micrometers
    wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment)
    
    #sometimes wavelength array is 1 element short, this will fix that
    if len(wavelengths) != len(image_data):
        wavelength_end = wavelength_start + number_wavelengths*wavelength_increment
        wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment)

    return wavelengths, image_data, error_data



#simple old version, can choose header
def loading_function1(file_loc, header_index):
    '''
    This function loads in JWST MIRI and NIRSPEC fits data cubes, and extracts wavelength 
    data from the header and builds the corresponding wavelength array.
    
    Parameters
    ----------
    file_loc
        TYPE: string
        DESCRIPTION: where the fits file is located.

    Returns
    -------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns.
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
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



#same as above but looks at 0 instead of 1 for getheader
def loading_function2(file_loc):
    '''
    This function loads in JWST MIRI and NIRSPEC fits data cubes, and extracts wavelength 
    data from the header and builds the corresponding wavelength array.
    
    Parameters
    ----------
    file_loc
        TYPE: string
        DESCRIPTION: where the fits file is located.

    Returns
    -------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns.
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [k,i,j] k is wavelength index, i and j are position index.
    '''
    
    #load in the data
    image_file = get_pkg_data_filename(file_loc)
    
    #header data
    science_header = fits.getheader(image_file, 0)
    
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
    
    #sometimes wavelength array is 1 element short, this will fix that
    if len(wavelengths) != len(image_data):
        wavelength_end = wavelength_start + number_wavelengths*wavelength_increment
        wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment)

    return wavelengths, image_data



def border_remover(data, amount=0):
    '''
    This function shrinks the fov to remove the areas where there is no data,
    or where there is likely to be bad data on the fov.
    
    Parameters
    ----------
    data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [k,i,j] k is wavelength index, i and j are position index.
    amount
        TYPE: positive integer(index)
        DESCRIPTION: how much to remove from each edge (optional).
    
    Returns
    -------
   shrunk_data
       TYPE: 3d array of floats
       DESCRIPTION: position and spectral data, edges removed.
           for [k,i,j] k is wavelength index, i and j are position index.
    '''
    
    #find the length of the 2 dimensions to be shrunk
    len1 = len(data[0,:,0])
    len2 = len(data[0,0,:])
    
    if amount == 0:
        amount1 = len1//5
        amount2 = len2//5
        
        end1 = len1 - amount1
        end2 = len2 - amount2
        
        shrunk_data = data[:,amount1:end1,amount2:end2]
    
    else:
        end1 = len1 - amount
        end2 = len2 - amount
        
        shrunk_data = data[:,amount:end1,amount:end2]
    
    return shrunk_data
    



####################################



'''
BACKGROUND SUBTRACTION
'''

def weighted_mean_finder(data, error_data, data_off, error_data_off):
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
            for [k,i,j] k is wavelength index, i and j are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [k,i,j] k is wavelength index, i and j are position index.
    data_off
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data for background.
            for [k,i,j] k is wavelength index, i and j are position index.
    error_data_off
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data for background.
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
    
    #first background subtract, propagate errors
    data = data - data_off
    error_data_new = (error_data**2 + error_data_off**2)**0.5
    
    weighted_mean = []
    weighted_mean_error = []
    
    #note JWST provides uncertainty (variance), standard deviation**2 = variance
    for i in range(len(data[:,0,0])):
        error_list = []
        error_temp_list = []
        mean_list = []
        for j in range(len(data[0,:,0])):
            for k in range(len(data[0,0,:])):
                if error_data[i,j,k] != 0 and error_data_off[i,j,k] != 0: #this being 0 means its outside the fov
                    temp_error = 1/(error_data_new[i,j,k])**2
                    mean_list.append(data[i,j,k]/(error_data_new[i,j,k])**2)
                    error_temp_list.append(temp_error)
                    error_list.append(error_data[i,j,k])
        
        error_list = np.array(error_list)
        error_temp_list = np.array(error_temp_list)
        mean_list = np.array(mean_list)
        
        error = np.sqrt(1/np.sum(error_temp_list))
        mean = (np.sum(mean_list))*error**2
        
        weighted_mean.append(mean)
        weighted_mean_error.append(error)
        
    weighted_mean = np.array(weighted_mean)
    mean_error = np.array(weighted_mean_error)
    
    return weighted_mean, mean_error



def weighted_mean_finder_simple(data, error_data):
    '''
    This function takes a weighted mean of the
    background-subtracted data, for 3 dimensional arrays.
    The mean is taken over the 1st and 2nd indicies, i.e. the spacial dimensions.
    
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
    
    
    weighted_mean = []
    weighted_mean_error = []
    
    #note JWST provides uncertainty (variance), standard deviation**2 = variance
    for i in range(len(data[:,0,0])):
        error_list = []
        error_temp_list = []
        mean_list = []
        for j in range(len(data[0,:,0])):
            for k in range(len(data[0,0,:])):
                if error_data[i,j,k] != 0:
                    temp_error = 1/(error_data[i,j,k])**2
                    mean_list.append(data[i,j,k]/(error_data[i,j,k])**2)
                    error_temp_list.append(temp_error)
                    error_list.append(error_data[i,j,k])
        
        error_list = np.array(error_list)
        error_temp_list = np.array(error_temp_list)
        mean_list = np.array(mean_list)
        
        error = np.sqrt(1/np.sum(error_temp_list))
        mean = (np.sum(mean_list))*error**2
        
        weighted_mean.append(mean)
        weighted_mean_error.append(error)
        
    weighted_mean = np.array(weighted_mean)
    mean_error = np.array(weighted_mean_error)
    
    return weighted_mean, mean_error



####################################



'''
STITCHING DATA
'''

#note that this is currently not in use due to the various channels seeming to have
#separate issues



#need a function that can stitch together fluxes, and take into account that there are overlapping wavelength ranges in the data

def flux_aligner(wave_a, wave_b, data_a, data_b):
    '''
    This function takes in 2 adjacent wavelength and image data arrays, presumably 
    from the same part of the image fov (field of view), so they correspond to the 
    same location in the sky. It then finds which indices in the lower wavelength data 
    overlap with the beginning of the higher wavelength data, and combines the 2 data 
    arrays in the middle of this region. Lastly, in order for there to be a smooth 
    transition, it scales the second data so that the mean of the overlapping region 
    is the same in the 2 data sets. This scaling is ok to do since the absolute flux 
    is unimportant to my research, only relative flux.
    
    Parameters
    ----------
    wave_a
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the smaller wavelengths.
        wave_b
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the larger wavelengths.
    data_a
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_a.
            for [k,i,j] k is wavelength index, i and j are position index.
    data_b
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_b.
            for [k,i,j] k is wavelength index, i and j are position index.
            
    Returns
    -------
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_a and data_b joined together as described above.
            for [k,i,j] k is wavelength index, i and j are position index.
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined together as described above.
    overlap
        TYPE: integer (index)
        DESCRIPTION: index of the wavelength value in wave_a that equals the first element in wave_b
    '''
    
    #check where the overlap is
    overlap = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
    
    #find how many entries are overlapped, subtract 1 for index
    overlap_length = len(wave_a) -1 - overlap
    
    #making a temp array to scale
    temp = np.copy(data_b)
    
    #find the mean of overlap area
    mean1 = np.mean(data_a[overlap:])
    mean2 = np.mean(data_b[:overlap_length])
    
    #shift by addition instead of multiplication
    ratio = mean1 - mean2
    
    temp = ratio + temp
    
    
    #ratio of means
    #ratio = mean1/mean2
            
    #scale array
    #temp = ratio*data_b
            
    #combine arrays such that the first half of one is used, and the second half
    #of the other is used. This way data at the end of the wavelength range is avoided
    
    split_index = overlap_length/2
    
    #check if even or odd, do different things depending on which
    if overlap_length%2 == 0: #even
        lower_index = overlap + split_index
        upper_index = split_index
    else: #odd, so split_index is a number of the form int+0.5
        lower_index = overlap + split_index + 0.5
        upper_index = split_index - 0.5
    
    #make sure they are integers
    lower_index = int(lower_index)
    upper_index = int(upper_index)
    
    image_data = np.hstack((data_a[:lower_index], temp[upper_index:]))
    wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
    
    return image_data, wavelengths, overlap



#this one just slaps them together without flux adjustment
def flux_aligner2(wave_a, wave_b, data_a, data_b):
    '''
    This function takes in 2 adjacent wavelength and image data arrays, presumably 
    from the same part of the image fov (field of view), so they correspond to 
    the same location in the sky. It then finds which indices in the lower wavelength 
    data overlap with the beginning of the higher wavelength data, and combines 
    the 2 data arrays in the middle of this region. For this function, no scaling is done.
    
    It needs to work with arrays that may have different intervals, so it is split into 2
    to take a longer running but more careful approach if needed.
    
    Note that in the latter case, the joining is not perfect, and currently a gap
    of ~0.005 microns is present; this is much better than the previous gap of ~0.05 microns,
    as 0.005 microns corresponds to 2-3 indices.
    
    Parameters
    ----------
    wave_a
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the smaller wavelengths.
        wave_b
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the larger wavelengths.
    data_a
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_a.
            for [k,i,j] k is wavelength index, i and j are position index.
    data_b
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_b.
            for [k,i,j] k is wavelength index, i and j are position index.
            
    Returns
    -------
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_a and data_b joined together as described above.
            for [k,i,j] k is wavelength index, i and j are position index.
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined together as described above.
    overlap
        TYPE: integer (index) OR tuple (index)
        DESCRIPTION: index of the wavelength value in wave_a that equals the first element in wave_b. In the 
        case of the two wavelength arrays having different intervals, overlap is instead a tuple of the regular
        overlap, followed by the starting index in the 2nd array.
        
        UPDATE: now returns lower_index and upper_index, as opposed to the indices where wave_a first meets wave_b,
        i.e. it now returns the indices where the connection happens.
        
    '''
    
    #check if wavelength interval is the same or different
    check_a = np.round(wave_a[1] - wave_b[0], 4)
    check_b = np.round(wave_b[1] - wave_b[0], 4)
    
    if check_a == check_b:
    
        #check where the overlap is
        overlap = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many entries are overlapped, subtract 1 for index
        overlap_length = len(wave_a) -1 - overlap
        
        #making a temp array to scale
        temp = np.copy(data_b)
                
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
        
        image_data = np.hstack((data_a[:lower_index], temp[upper_index:]))
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        
    else:
        #check where the overlap is, only works for wave_a
        overlap_a = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many microns the overlap is
        overlap_micron = wave_a[-1] - wave_a[overlap_a]
        
        #find how many entries of wave_a are overlapped, subtract 1 for index
        overlap_length_a = len(wave_a) -1 - overlap_a
        split_index_a = overlap_length_a/2
        
        #number of indices in wave_B over the wavelength range
        overlap_length_b = int(overlap_micron/check_b)
        split_index_b = overlap_length_b/2
        
        #making a temp array to scale
        temp = np.copy(data_b)
        
        #check if even or odd, do different things depending on which
        if overlap_length_a%2 == 0: #even
            lower_index = overlap_a + split_index_a
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap_a + split_index_a + 0.5
            
        if overlap_length_b%2 == 0: #even
            upper_index = split_index_b
        else: #odd, so split_index is a number of the form int+0.5
            upper_index = split_index_b - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        #print(lower_index, upper_index)
        
        image_data = np.hstack((data_a[:lower_index], temp[upper_index:]))
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        #overlap = (overlap_a, overlap_length_b)
        overlap = (lower_index, upper_index)
    
    return image_data, wavelengths, overlap



####################################



'''
REMOVING EMISSION AND ABSORPTION LINES
'''

#removing emission lines from data

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
    
    new_data = np.copy(data)
    
    for i in range(len(new_data)):
        #central regions away from the edges
        if (i > width) and (i < len(data) - 1 - width):
            if new_data[i] - new_data[i-width] > limit and new_data[i] - new_data[i+width] > limit:
                new_data[i] = (new_data[i+width] + new_data[i-width])/2
        #left edge case
        elif (i < width):
            if new_data[i] - new_data[i+width] > limit:
                new_data[i] = new_data[i+width]
        #right edge case
        elif (i > len(data) - 1 - width):
            if new_data[i] - new_data[i-width] > limit:
                new_data[i] = new_data[i-width]
    
    return new_data



#removing absorption lines from data

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
    
    for i in range(len(new_data)):
        #central regions away from the edges
        if (i > width) and (i < len(data) - 1 - width):
            if new_data[i] - new_data[i-width] < limit and new_data[i] - new_data[i+width] < limit:
                new_data[i] = (new_data[i+width] + new_data[i-width])/2
        #left edge case
        elif (i < width):
            if new_data[i] - new_data[i+width] < limit:
                new_data[i] = new_data[i+width]
        #right edge case
        elif (i > len(data) - 1 - width):
            if new_data[i] - new_data[i-width] < limit:
                new_data[i] = new_data[i-width]
    
    return new_data



####################################



'''
LINE FITTING
'''

#making a function that uses ransac to fit lines

def line_fitter(wavelength, data):
    '''
    A function that fits a 20th order polynomial to input data using RANSAC.
    
    Parameters
    ----------
    wavelength
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
    
    wavelength = wavelength.reshape(-1,1)
    data = data.reshape(-1,1)

    # Init the RANSAC regressor
    ransac = make_pipeline(PolynomialFeatures(20), RANSACRegressor(max_trials=200000, random_state=41))

    # Fit with RANSAC
    ransac.fit(wavelength, data)

    # Get the fitted data result
    line_ransac = ransac.predict(wavelength)
    
    return line_ransac



####################################



'''
FRINGE REMOVAL
'''

#Ryan's fringe remover

def fringe_remover(segment, wavelengths, data, A):

    #to remove units and convert back to regular floats for easier manipulation
    unit_um = Unit(s='um')
    
    
    
    #loading in fringe data
    with open(r'data/misc/fringe_correction/data_10lac_corr.pk', 'rb') as file:
        fringe_dict = pickle.load(file)
    
    
    
    #example dictionary calling:
    #fringe_dict['CH2 LONG']['wave'] wavelength
    #fringe_dict['CH2 LONG']['ratio_corr_ext_to_model'] data for correcting fringes    
    
    fringe_wavelengths_temp = fringe_dict[segment]['wave']
    fringe_data = fringe_dict[segment]['ratio_corr_ext_to_model']
    
    fringe_wavelengths = np.zeros(len(fringe_wavelengths_temp))
    
    for i in range(len(fringe_wavelengths)):
        fringe_wavelengths[i] = float(fringe_wavelengths_temp[i]/unit_um)
    
    
    
    #note: way more elements in fringe data than what i have, so need a loop to
    #make proper comparison possible
    
    #figure out where the overlap between miri and fringe begins
    
    #miri starts before fringe
    first_index = np.where(np.round(wavelengths, 3) == np.round(fringe_wavelengths[0], 3))[0][0]
    
    equal_index = np.where(np.round(fringe_wavelengths, 3) == np.round(wavelengths[first_index], 3))[0]
    
    #if multiple are returned, find the index that is closest
    if len(equal_index) > 1:
        comparison = np.zeros(len(equal_index))
        for i in range(len(equal_index)):
            comparison[i] = abs(wavelengths[first_index] - fringe_wavelengths[equal_index[i]])
        true_equal_index_first_fringe = equal_index[np.argmin(comparison)]
        true_equal_index_first_miri = int(np.copy(first_index))
    
    else:
        true_equal_index_first_fringe = int(np.copy(equal_index))
        true_equal_index_first_miri = int(np.copy(first_index))
    
    #figure out which miri element is closest to the end of fringe_wavelengths
    
    #miri ends before fringe
    last_index = np.where(np.round(fringe_wavelengths, 3) == np.round(wavelengths[-1], 3))[0]
    
    #if multiple are returned, find the index that is closest
    if len(last_index) > 1:
        comparison = np.zeros(len(last_index))
        for i in range(len(equal_index)):
            comparison[i] = abs(wavelengths[-1] - fringe_wavelengths[last_index[i]])
        true_equal_index_last_fringe = last_index[np.argmin(comparison)]
        true_equal_index_last_miri = -1
        
    else:
        true_equal_index_last = int(np.copy(last_index))
        true_equal_index_last_miri = -1
    
    
    
    #need to construct an array that contains all the values for fringe correction that
    #are relevant, since there are way more in fringe_correction than miri data at the moment
    
    #various possibilities on what to include in wavelength array
    if (true_equal_index_first_miri == 0) and (true_equal_index_last_miri == -1):
        comparison_wavelengths = np.copy(wavelengths)
        size_miri = len(wavelengths)
        
        #defining trimmed miri flux data for later
        data_trimmed = np.copy(data)
        
    elif true_equal_index_first_miri == 0:
        comparison_wavelengths = np.copy(wavelengths[:true_equal_index_last_miri])
        size_miri = len(wavelengths[:true_equal_index_last_miri])
        
        #defining trimmed miri flux data for later
        data_trimmed = np.copy(data[:true_equal_index_last_miri])
        
    elif true_equal_index_last_miri == -1:
        comparison_wavelengths = np.copy(wavelengths[true_equal_index_first_miri:])
        size_miri = len(wavelengths[true_equal_index_first_miri:])
        
        #defining trimmed miri flux data for later
        data_trimmed = np.copy(data[true_equal_index_first_miri:])
        
    else:
        comparison_wavelengths = np.copy(wavelengths[true_equal_index_first_miri:true_equal_index_last_miri])
        size_miri = len(wavelengths[true_equal_index_first_miri:true_equal_index_last_miri])
    
        #defining trimmed miri flux data for later
        data_trimmed = np.copy(data[true_equal_index_first_miri:true_equal_index_last_miri])
    
    comparison_data = np.zeros(size_miri)
    
    
    #filling in first and last elements
    comparison_data[0] = fringe_data[true_equal_index_first_fringe]
    comparison_data[-1] = fringe_data[true_equal_index_last_fringe]
    
    for i in range(size_miri):
        #if i == 0 or i == size_miri - 1:
            #break #these elements are taken care of before the loop
    
        #converting i to index useable with miri wavelengths
        j = i + true_equal_index_first_miri
        
        #index where fringe and miri data equal
        temp_index = np.where(np.round(fringe_wavelengths, 3) == np.round(wavelengths[j], 3))[0]
        
        #if multiple are returned, find the index that is closest
        if len(temp_index) > 1:
            comparison = np.zeros(len(temp_index))
            for k in range(len(temp_index)):
                comparison[k] = abs(wavelengths[j] - fringe_wavelengths[temp_index[k]])
            true_temp_index = temp_index[np.argmin(comparison)]
            
        else:
            true_temp_index = np.copy(temp_index)
            
        comparison_data[i] = fringe_data[true_temp_index]
    

    
    #correcting fringes in data
    
    #defining a free variable that depends on spectrum, adjusts amplitude of correction
    
    #A = 2.35
    #B = 2.35
    
    #2.35 seems good (for data6 at least)
    
    data_corrected = data_trimmed/(A*(comparison_data - 1) + 1)
    #data_corrected2 = data_trimmed/(B*(comparison_data - 1) + 1)
    
    return comparison_wavelengths, data_corrected



def unit_changer(data, wavelengths):
    
    final_cube = np.zeros(data.shape)
    cube_with_units = (data*10**6)*(u.Jy/u.sr)


    final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavelengths*u.micron))

    final_cube = final_cube*(u.micron)
    final_cube = final_cube*(u.m)
    final_cube = final_cube*(u.m)
    final_cube = final_cube*(u.sr/u.W)

    new_data = np.copy(data)
    for i in range(len(data)):
        new_data[i] = float(final_cube[i])
    
    return new_data


def RMS(data):
    
    return (np.var(data))**0.5



def SNR(intensity, rms, delta_wave, num_points):
    
    return intensity/(rms*2*delta_wave*(num_points/2)**0.5)



#this function is from els and i modified it

def extract_spectra_from_regions_one_pointing(fname_cube, data, error, wave, background, background_error, fname_region, do_sigma_clip=True, use_dq=False):
    reg = regions.Regions.read(fname_region, format='ds9')
    fits_cube = fits.open(fname_cube)
    w = wcs.WCS(fits_cube[1].header).dropaxis(2)
    nw = wave.shape[0]
    cube = data
    #dq = fits_cube['DQ'].data
    cube_unc = error
    all_spectra = dict()
    all_spectra_unc = dict()
    combined_spectra = dict()
    combined_spectra_unc = dict()

    do_coadd = ~do_sigma_clip
    # loop over regions in .reg file
    for i in range(len(reg)):
        regmask = reg[i].to_pixel(w).to_mask('center').to_image(shape=cube.shape[1:])
        if regmask is not None:
            all_spectra[f'region_{i}'] = []
            all_spectra_unc[f'region_{i}'] = []
            
            #print(f"Region {i}")
            spec_comb = np.full(nw, fill_value=0.)
            spec_comb_unc = np.full(nw, fill_value=0.)
            nspax_comb = np.zeros(nw)
            for ix in range(cube.shape[1]):
                for iy in range(cube.shape[2]):
                    if regmask[ix, iy] == 1: #subtract background and propogate error through
                        # spaxel is in this region
                        spec = cube[:, ix, iy] - background[:, ix, iy]
                        spec_unc = (cube_unc[:, ix, iy]**2 + background_error[:, ix, iy]**2)**0.5
                        #spec_dq = dq[:, ix, iy]
                        
                        #good = np.isfinite(spec) & (spec > 0) & np.isfinite(spec_unc) & (spec_unc > 0)
                        #if use_dq:
                        #    # Ignore any fluxes that have DQ != 0
                         #   # See https://jwst-reffiles.stsci.edu/source/data_quality.html
                          #  good &= (spec_dq == 0)
                        
                        #spec[~good] = np.nan
                        #spec_unc[~good] = np.nan
                        
                        #spec_comb[good] += spec[good] 
                        #spec_comb_unc[good] += spec_unc[good]**2
                        all_spectra[f'region_{i}'].append(spec)
                        all_spectra_unc[f'region_{i}'].append(spec_unc)
                    
            if len(all_spectra[f'region_{i}']) > 0:
                all_spectra[f'region_{i}'] = np.vstack(all_spectra[f'region_{i}'])
                all_spectra_unc[f'region_{i}'] = np.vstack(all_spectra_unc[f'region_{i}']) 
                
                if len(all_spectra[f'region_{i}']) == 1:
                    #print("Only one spaxel in this region -- can't do clipping")
                    combined_spectra[f'region_{i}'] = all_spectra[f'region_{i}']
                    # Uncertainty on the mean
                    combined_spectra_unc[f'region_{i}'] = all_spectra_unc[f'region_{i}']

                else:
                    if do_sigma_clip:
                        # Sigma-clipped data
                        data = sigma_clip(all_spectra[f'region_{i}'], axis=0, sigma=3, maxiters=5, cenfunc='median', stdfunc='std')
                        # Uncertainties
                        unc = np.ma.array(all_spectra_unc[f'region_{i}'], mask=data.mask, fill_value=np.nan)
                        # Calculate weighted mean spectrum
                        # combined_spectra[f'region_{i}'] = np.nanmean(data, axis=0)
                        combined_spectra[f'region_{i}'] = np.nansum(data / unc**2, axis=0) / np.nansum(1. / unc**2, axis=0) 
    
                        # Uncertainty on the mean
                        combined_spectra_unc[f'region_{i}'] = np.sqrt(np.nanmean(unc**2, axis=0))
                    else:
                        # No sigma clipping, just coadd everything
                        data = all_spectra[f'region_{i}']
                        unc = all_spectra_unc[f'region_{i}']
                        # Calculate mean spectrum
                        combined_spectra[f'region_{i}'] = np.nansum(data / unc**2, axis=0) / np.nansum(1. / unc**2, axis=0)
                        # Uncertainty on the mean
                        combined_spectra_unc[f'region_{i}'] = np.sqrt(np.nanmean(unc**2, axis=0))
            else:
                combined_spectra[f'region_{i}'] = np.full(nw, fill_value=-1e20)
                combined_spectra_unc[f'region_{i}'] = np.full(nw, fill_value=1e20)    

        if regmask is None:
            combined_spectra[f'region_{i}'] = np.full(nw, fill_value=-1e20)
            combined_spectra_unc[f'region_{i}'] = np.full(nw, fill_value=1e20)    
            #spectra_in_region.append(spec_comb)
            #spectra_unc_in_region.append(spec_comb_unc)
    
    return combined_spectra, combined_spectra_unc


#this version is for data without a background
def extract_spectra_from_regions_one_pointing_no_bkg(fname_cube, data, error, wave, fname_region, do_sigma_clip=True, use_dq=False):
    reg = regions.Regions.read(fname_region, format='ds9')
    fits_cube = fits.open(fname_cube)
    w = wcs.WCS(fits_cube[1].header).dropaxis(2)
    nw = wave.shape[0]
    cube = data
    #dq = fits_cube['DQ'].data
    cube_unc = error
    all_spectra = dict()
    all_spectra_unc = dict()
    combined_spectra = dict()
    combined_spectra_unc = dict()

    do_coadd = ~do_sigma_clip
    # loop over regions in .reg file
    for i in range(len(reg)):
        regmask = reg[i].to_pixel(w).to_mask('center').to_image(shape=cube.shape[1:])
        if regmask is not None:
            all_spectra[f'region_{i}'] = []
            all_spectra_unc[f'region_{i}'] = []
            #print(f"Region {i}")
            spec_comb = np.full(nw, fill_value=0.)
            spec_comb_unc = np.full(nw, fill_value=0.)
            nspax_comb = np.zeros(nw)
            for ix in range(cube.shape[1]):
                for iy in range(cube.shape[2]):
                    if regmask[ix, iy] == 1:
                        # spaxel is in this region
                        spec = cube[:, ix, iy]
                        spec_unc = cube_unc[:, ix, iy]
                        #spec_dq = dq[:, ix, iy]
                        
                        #good = np.isfinite(spec) & (spec > 0) & np.isfinite(spec_unc) & (spec_unc > 0)
                        #if use_dq:
                        #    # Ignore any fluxes that have DQ != 0
                         #   # See https://jwst-reffiles.stsci.edu/source/data_quality.html
                          #  good &= (spec_dq == 0)
                        
                        #spec[~good] = np.nan
                        #spec_unc[~good] = np.nan
                        
                        #spec_comb[good] += spec[good] 
                        #spec_comb_unc[good] += spec_unc[good]**2
                        all_spectra[f'region_{i}'].append(spec)
                        all_spectra_unc[f'region_{i}'].append(spec_unc)
                    
            if len(all_spectra[f'region_{i}']) > 0:
                all_spectra[f'region_{i}'] = np.vstack(all_spectra[f'region_{i}'])
                all_spectra_unc[f'region_{i}'] = np.vstack(all_spectra_unc[f'region_{i}']) 
                
                if len(all_spectra[f'region_{i}']) == 1:
                    #print("Only one spaxel in this region -- can't do clipping")
                    combined_spectra[f'region_{i}'] = all_spectra[f'region_{i}']
                    # Uncertainty on the mean
                    combined_spectra_unc[f'region_{i}'] = all_spectra_unc[f'region_{i}']

                else:
                    if do_sigma_clip:
                        # Sigma-clipped data
                        data = sigma_clip(all_spectra[f'region_{i}'], axis=0, sigma=3, maxiters=5, cenfunc='median', stdfunc='std')
                        # Uncertainties
                        unc = np.ma.array(all_spectra_unc[f'region_{i}'], mask=data.mask, fill_value=np.nan)
                        # Calculate weighted mean spectrum
                        # combined_spectra[f'region_{i}'] = np.nanmean(data, axis=0)
                        combined_spectra[f'region_{i}'] = np.nansum(data / unc**2, axis=0) / np.nansum(1. / unc**2, axis=0) 
    
                        # Uncertainty on the mean
                        combined_spectra_unc[f'region_{i}'] = np.sqrt(np.nanmean(unc**2, axis=0))
                    else:
                        # No sigma clipping, just coadd everything
                        data = all_spectra[f'region_{i}']
                        unc = all_spectra_unc[f'region_{i}']
                        # Calculate mean spectrum
                        #print('weighted mean time')
                        combined_spectra[f'region_{i}'] = np.nansum(data / unc**2, axis=0) / np.nansum(1. / unc**2, axis=0)
                        # Uncertainty on the mean
                        combined_spectra_unc[f'region_{i}'] = np.sqrt(np.nanmean(unc**2, axis=0))
            else:
                combined_spectra[f'region_{i}'] = np.full(nw, fill_value=-1e20)
                combined_spectra_unc[f'region_{i}'] = np.full(nw, fill_value=1e20)    

        if regmask is None:
            combined_spectra[f'region_{i}'] = np.full(nw, fill_value=-1e20)
            combined_spectra_unc[f'region_{i}'] = np.full(nw, fill_value=1e20)    
            #spectra_in_region.append(spec_comb)
            #spectra_unc_in_region.append(spec_comb_unc)
    return combined_spectra, combined_spectra_unc



#this version is for 2d data without a background
def extract_spectra_from_regions_one_pointing_no_bkg_2d(fname_cube, data, error, wave, fname_region, do_sigma_clip=True, use_dq=False):
    reg = regions.Regions.read(fname_region, format='ds9')
    fits_cube = fits.open(fname_cube)
    w = wcs.WCS(fits_cube[1].header) #no axis 2 to remove if 2d
    nw = wave.shape[0]
    cube = data
    #dq = fits_cube['DQ'].data
    cube_unc = error
    all_spectra = dict()
    all_spectra_unc = dict()
    combined_spectra = dict()
    combined_spectra_unc = dict()

    do_coadd = ~do_sigma_clip
    # loop over regions in .reg file
    for i in range(len(reg)):
        regmask = reg[i].to_pixel(w).to_mask('center').to_image(shape=cube.shape[1:])
        if regmask is not None:
            all_spectra[f'region_{i}'] = []
            all_spectra_unc[f'region_{i}'] = []
            #print(f"Region {i}")
            spec_comb = np.full(nw, fill_value=0.)
            spec_comb_unc = np.full(nw, fill_value=0.)
            nspax_comb = np.zeros(nw)
            for ix in range(cube.shape[1]):
                for iy in range(cube.shape[2]):
                    if regmask[ix, iy] == 1:
                        # spaxel is in this region
                        spec = cube[:, ix, iy]
                        spec_unc = cube_unc[:, ix, iy]
                        #spec_dq = dq[:, ix, iy]
                        
                        #good = np.isfinite(spec) & (spec > 0) & np.isfinite(spec_unc) & (spec_unc > 0)
                        #if use_dq:
                        #    # Ignore any fluxes that have DQ != 0
                         #   # See https://jwst-reffiles.stsci.edu/source/data_quality.html
                          #  good &= (spec_dq == 0)
                        
                        #spec[~good] = np.nan
                        #spec_unc[~good] = np.nan
                        
                        #spec_comb[good] += spec[good] 
                        #spec_comb_unc[good] += spec_unc[good]**2
                        all_spectra[f'region_{i}'].append(spec)
                        all_spectra_unc[f'region_{i}'].append(spec_unc)
                    
            if len(all_spectra[f'region_{i}']) > 0:
                all_spectra[f'region_{i}'] = np.vstack(all_spectra[f'region_{i}'])
                all_spectra_unc[f'region_{i}'] = np.vstack(all_spectra_unc[f'region_{i}']) 
                
                if len(all_spectra[f'region_{i}']) == 1:
                    #print("Only one spaxel in this region -- can't do clipping")
                    combined_spectra[f'region_{i}'] = all_spectra[f'region_{i}']
                    # Uncertainty on the mean
                    combined_spectra_unc[f'region_{i}'] = all_spectra_unc[f'region_{i}']

                else:
                    if do_sigma_clip:
                        # Sigma-clipped data
                        data = sigma_clip(all_spectra[f'region_{i}'], axis=0, sigma=3, maxiters=5, cenfunc='median', stdfunc='std')
                        # Uncertainties
                        unc = np.ma.array(all_spectra_unc[f'region_{i}'], mask=data.mask, fill_value=np.nan)
                        # Calculate weighted mean spectrum
                        # combined_spectra[f'region_{i}'] = np.nanmean(data, axis=0)
                        combined_spectra[f'region_{i}'] = np.nansum(data / unc**2, axis=0) / np.nansum(1. / unc**2, axis=0) 
    
                        # Uncertainty on the mean
                        combined_spectra_unc[f'region_{i}'] = np.sqrt(np.nanmean(unc**2, axis=0))
                    else:
                        # No sigma clipping, just coadd everything
                        data = all_spectra[f'region_{i}']
                        unc = all_spectra_unc[f'region_{i}']
                        # Calculate mean spectrum
                        #print('weighted mean time')
                        combined_spectra[f'region_{i}'] = np.nansum(data / unc**2, axis=0) / np.nansum(1. / unc**2, axis=0)
                        # Uncertainty on the mean
                        combined_spectra_unc[f'region_{i}'] = np.sqrt(np.nanmean(unc**2, axis=0))
            else:
                combined_spectra[f'region_{i}'] = np.full(nw, fill_value=-1e20)
                combined_spectra_unc[f'region_{i}'] = np.full(nw, fill_value=1e20)    

        if regmask is None:
            combined_spectra[f'region_{i}'] = np.full(nw, fill_value=-1e20)
            combined_spectra_unc[f'region_{i}'] = np.full(nw, fill_value=1e20)    
            #spectra_in_region.append(spec_comb)
            #spectra_unc_in_region.append(spec_comb_unc)
    return combined_spectra, combined_spectra_unc



#this function is from els and i modified it, meant for 2 region files to make a doughnut shaped flux

def extract_spectra_from_regions_one_pointing2(fname_cube, data, error, wave, background, background_error, fname_region1, fname_region2, do_sigma_clip=True, use_dq=False):
    reg1 = regions.Regions.read(fname_region1, format='ds9')
    reg2 = regions.Regions.read(fname_region2, format='ds9')
    fits_cube = fits.open(fname_cube)
    w = wcs.WCS(fits_cube[1].header).dropaxis(2)
    nw = wave.shape[0]
    cube = data
    #dq = fits_cube['DQ'].data
    cube_unc = error
    all_spectra = dict()
    all_spectra_unc = dict()
    combined_spectra = dict()
    combined_spectra_unc = dict()

    do_coadd = ~do_sigma_clip
    # loop over regions in .reg file
    for i in range(len(reg1)): #reg1 should always have length 1 because each file has 1 region in it
        regmask1 = reg1[i].to_pixel(w).to_mask('center').to_image(shape=cube.shape[1:])
        regmask2 = reg2[i].to_pixel(w).to_mask('center').to_image(shape=cube.shape[1:])
        if regmask1 is not None: 
            all_spectra[f'region_{i}'] = []
            all_spectra_unc[f'region_{i}'] = []
            #print(f"Region {i}")
            spec_comb = np.full(nw, fill_value=0.)
            spec_comb_unc = np.full(nw, fill_value=0.)
            nspax_comb = np.zeros(nw)
            for ix in range(cube.shape[1]):
                for iy in range(cube.shape[2]):
                    if regmask1[ix, iy] == 1 and regmask2[ix, iy] == 0: #should be the area in the rectangle but outside the blob
                        # spaxel is in this region
                        spec = cube[:, ix, iy] - background[:, ix, iy] #subtract background and propogate error through
                        spec_unc = (cube_unc[:, ix, iy]**2 + background_error[:, ix, iy]**2)**0.5
                        #spec_dq = dq[:, ix, iy]
                        
                        #good = np.isfinite(spec) & (spec > 0) & np.isfinite(spec_unc) & (spec_unc > 0)
                        #if use_dq:
                        #    # Ignore any fluxes that have DQ != 0
                         #   # See https://jwst-reffiles.stsci.edu/source/data_quality.html
                          #  good &= (spec_dq == 0)
                        
                        #spec[~good] = np.nan
                        #spec_unc[~good] = np.nan
                        
                        #spec_comb[good] += spec[good] 
                        #spec_comb_unc[good] += spec_unc[good]**2
                        all_spectra[f'region_{i}'].append(spec)
                        all_spectra_unc[f'region_{i}'].append(spec_unc)
                    
            if len(all_spectra[f'region_{i}']) > 0:
                all_spectra[f'region_{i}'] = np.vstack(all_spectra[f'region_{i}'])
                all_spectra_unc[f'region_{i}'] = np.vstack(all_spectra_unc[f'region_{i}']) 
                
                if len(all_spectra[f'region_{i}']) == 1:
                    #print("Only one spaxel in this region -- can't do clipping")
                    combined_spectra[f'region_{i}'] = all_spectra[f'region_{i}']
                    # Uncertainty on the mean
                    combined_spectra_unc[f'region_{i}'] = all_spectra_unc[f'region_{i}']

                else:
                    if do_sigma_clip:
                        # Sigma-clipped data
                        data = sigma_clip(all_spectra[f'region_{i}'], axis=0, sigma=3, maxiters=5, cenfunc='median', stdfunc='std')
                        # Uncertainties
                        unc = np.ma.array(all_spectra_unc[f'region_{i}'], mask=data.mask, fill_value=np.nan)
                        # Calculate weighted mean spectrum
                        # combined_spectra[f'region_{i}'] = np.nanmean(data, axis=0)
                        combined_spectra[f'region_{i}'] = np.nansum(data / unc**2, axis=0) / np.nansum(1. / unc**2, axis=0) 
    
                        # Uncertainty on the mean
                        combined_spectra_unc[f'region_{i}'] = np.sqrt(np.nanmean(unc**2, axis=0))
                    else:
                        # No sigma clipping, just coadd everything
                        data = all_spectra[f'region_{i}']
                        unc = all_spectra_unc[f'region_{i}']
                        # Calculate mean spectrum
                        combined_spectra[f'region_{i}'] = np.nansum(data / unc**2, axis=0) / np.nansum(1. / unc**2, axis=0)
                        # Uncertainty on the mean
                        combined_spectra_unc[f'region_{i}'] = np.sqrt(np.nanmean(unc**2, axis=0))
            else:
                combined_spectra[f'region_{i}'] = np.full(nw, fill_value=-1e20)
                combined_spectra_unc[f'region_{i}'] = np.full(nw, fill_value=1e20)    

        if regmask1 is None:
            combined_spectra[f'region_{i}'] = np.full(nw, fill_value=-1e20)
            combined_spectra_unc[f'region_{i}'] = np.full(nw, fill_value=1e20)    
            #spectra_in_region.append(spec_comb)
            #spectra_unc_in_region.append(spec_comb_unc)
    
    return combined_spectra, combined_spectra_unc


#this version is for data without a background
def extract_spectra_from_regions_one_pointing_no_bkg2(fname_cube, data, error, wave, fname_region1, fname_region2, do_sigma_clip=True, use_dq=False):
    reg1 = regions.Regions.read(fname_region1, format='ds9')
    reg2 = regions.Regions.read(fname_region2, format='ds9')
    fits_cube = fits.open(fname_cube)
    w = wcs.WCS(fits_cube[1].header).dropaxis(2)
    nw = wave.shape[0]
    cube = data
    #dq = fits_cube['DQ'].data
    cube_unc = error
    all_spectra = dict()
    all_spectra_unc = dict()
    combined_spectra = dict()
    combined_spectra_unc = dict()

    do_coadd = ~do_sigma_clip
    # loop over regions in .reg file
    for i in range(len(reg1)):
        regmask1 = reg1[i].to_pixel(w).to_mask('center').to_image(shape=cube.shape[1:])
        regmask2 = reg2[i].to_pixel(w).to_mask('center').to_image(shape=cube.shape[1:])
        if regmask1 is not None:
            all_spectra[f'region_{i}'] = []
            all_spectra_unc[f'region_{i}'] = []
            #print(f"Region {i}")
            spec_comb = np.full(nw, fill_value=0.)
            spec_comb_unc = np.full(nw, fill_value=0.)
            nspax_comb = np.zeros(nw)
            for ix in range(cube.shape[1]):
                for iy in range(cube.shape[2]):
                    if regmask1[ix, iy] == 1 and regmask2[ix, iy] == 0:
                        # spaxel is in this region
                        spec = cube[:, ix, iy]
                        spec_unc = cube_unc[:, ix, iy]
                        #spec_dq = dq[:, ix, iy]
                        
                        #good = np.isfinite(spec) & (spec > 0) & np.isfinite(spec_unc) & (spec_unc > 0)
                        #if use_dq:
                        #    # Ignore any fluxes that have DQ != 0
                         #   # See https://jwst-reffiles.stsci.edu/source/data_quality.html
                          #  good &= (spec_dq == 0)
                        
                        #spec[~good] = np.nan
                        #spec_unc[~good] = np.nan
                        
                        #spec_comb[good] += spec[good] 
                        #spec_comb_unc[good] += spec_unc[good]**2
                        all_spectra[f'region_{i}'].append(spec)
                        all_spectra_unc[f'region_{i}'].append(spec_unc)
                    
            if len(all_spectra[f'region_{i}']) > 0:
                all_spectra[f'region_{i}'] = np.vstack(all_spectra[f'region_{i}'])
                all_spectra_unc[f'region_{i}'] = np.vstack(all_spectra_unc[f'region_{i}']) 
                
                if len(all_spectra[f'region_{i}']) == 1:
                    #print("Only one spaxel in this region -- can't do clipping")
                    combined_spectra[f'region_{i}'] = all_spectra[f'region_{i}']
                    # Uncertainty on the mean
                    combined_spectra_unc[f'region_{i}'] = all_spectra_unc[f'region_{i}']

                else:
                    if do_sigma_clip:
                        # Sigma-clipped data
                        data = sigma_clip(all_spectra[f'region_{i}'], axis=0, sigma=3, maxiters=5, cenfunc='median', stdfunc='std')
                        # Uncertainties
                        unc = np.ma.array(all_spectra_unc[f'region_{i}'], mask=data.mask, fill_value=np.nan)
                        # Calculate weighted mean spectrum
                        # combined_spectra[f'region_{i}'] = np.nanmean(data, axis=0)
                        combined_spectra[f'region_{i}'] = np.nansum(data / unc**2, axis=0) / np.nansum(1. / unc**2, axis=0) 
    
                        # Uncertainty on the mean
                        combined_spectra_unc[f'region_{i}'] = np.sqrt(np.nanmean(unc**2, axis=0))
                    else:
                        # No sigma clipping, just coadd everything
                        data = all_spectra[f'region_{i}']
                        unc = all_spectra_unc[f'region_{i}']
                        # Calculate mean spectrum
                        combined_spectra[f'region_{i}'] = np.nansum(data / unc**2, axis=0) / np.nansum(1. / unc**2, axis=0)
                        # Uncertainty on the mean
                        combined_spectra_unc[f'region_{i}'] = np.sqrt(np.nanmean(unc**2, axis=0))
            else:
                combined_spectra[f'region_{i}'] = np.full(nw, fill_value=-1e20)
                combined_spectra_unc[f'region_{i}'] = np.full(nw, fill_value=1e20)    

        if regmask1 is None:
            combined_spectra[f'region_{i}'] = np.full(nw, fill_value=-1e20)
            combined_spectra_unc[f'region_{i}'] = np.full(nw, fill_value=1e20)    
            #spectra_in_region.append(spec_comb)
            #spectra_unc_in_region.append(spec_comb_unc)
    
    return combined_spectra, combined_spectra_unc



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
            for [k,i,j] k is wavelength index, i and j are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data cube to be rebinned.
            for [k,i,j] k is wavelength index, i and j are position index.
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

    #Figure out if any indices need to be discarded
    
    size_x = len(data[0,:,0])
    size_y = len(data[0,0,:])
    
    remainder_x = size_x % N
    remainder_y = size_y % N
    
    if remainder_x != 0:
        size_x = size_x - remainder_x
        
    if remainder_y != 0:
        size_y = size_y - remainder_y
        
    #building new arrays
    
    size_wavelength = int(len(data[:,0,0]))
    
    rebinned_data = np.zeros((size_wavelength, int(size_x/N), int(size_y/N)))
    
    rebinned_error_data = np.zeros((size_wavelength, int(size_x/N), int(size_y/N)))
    
    print(rebinned_data.shape)
    
    for i in range(0, size_x, N):
        for j in range(0, size_y, N):
            print(i,j)
            #note that i:i+N will have i+1,...,i+N, with length N, so want to subtract 1 from these to include i
            temp_data, temp_error_data = weighted_mean_finder_simple(
                data[:, i:i+N, j:j+N], error_data[:, i:i+N, j:j+N])
            rebinned_data[:,int(i/N),int(j/N)] = temp_data
            rebinned_error_data[:,int(i/N),int(j/N)] = temp_error_data
            
    return rebinned_data, rebinned_error_data
        



