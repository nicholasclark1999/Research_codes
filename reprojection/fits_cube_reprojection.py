#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:14:16 2023

@author: nclark
"""



'''
IMPORTING MODULES
'''



#standard stuff
import matplotlib.pyplot as plt
import numpy as np

#used for fits file handling
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

#needed for Ryan's reprojection function
from jwst import datamodels
from astropy import wcs
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_exact



####################################



'''
TABLE OF CONTENTS
'''



# reproject_slice:                 outputs reprojected slices (data OR error). 
#                                    fits files CAN have modified headers (from JWST default)

# reproject_cube:                  outputs reprojected cubes (data OR error OR dq). 
#                                    fits files CAN have modified headers

# loading_function_reproject_cube: outputs reprojected cubes (data AND error), and corresponding wavelength arrays. 
#                                    fits files CANNOT have modified headers

# fits_reprojection_cube:          saves reprojected cubes (data AND error), without creating output variables.
#                                    fits files CANNOT have modified headers



####################################



'''
HELPER FUNCTIONS
'''



#needed for Ryan's reproject function
def get_2d_wcs_from_data(fits_to_reproject, header_index): 
    '''
    Gets a wcs object from header data.
    Works with fits files containing cubes and slices.
    
    Note from Nicholas: This function seems to have been built to fix a problem with spitzer data. Since this fix
    doesnt do anything to JWST data I am leaving the fix in.
    
    Gets 2D (spatial) WCS from IRS cube.
    For some reason, extracting WCS from cubism cubes doesn't work well
    (the spectral axis messes things up).
​
    Parameters
    ----------
    fits_to_reproject
        TYPE: string
        DESCRIPTION: where the fits file to be reprojected is located.        
    header_index
        TYPE: positive integer
        DESCRIPTION: header that corresponds to image data, for JWST is 1 by default
            
    Returns
    -------
    w_in
        TYPE: WCS object
        DESCRIPTION: contains WCS info about RA and Dec to be used by other functions
    '''
    fits_in = fits.open(fits_to_reproject)
    
    #naxis=2, corresponds to RA and Dec
    w_in = wcs.WCS(fits_in[header_index].header, fobj=fits_in, naxis=2)
    
    # Turn WCS into a header, and then back to WCS again (this cleans up garbage to do with the 3rd axis we don't want anyway)
    w_in = wcs.WCS(w_in.to_header())
    
    return w_in



#needed for ryans reproject function
def clip_spikes_spec(spec, spec_unc):
    '''
    Tidies up data by removing anything that is not finite, and anything less than zero, which are assumed to be bad pixels.
​
    Parameters
    ----------
    spec
        TYPE: 1d array of floats
        DESCRIPTION: a spectra
    spec_unc
        TYPE: 1d array of floats
        DESCRIPTION: the error assiciated with spec

    Returns
    -------
    spec_out
        TYPE: 1d array of floats
        DESCRIPTION: a spectra, artifacts removed.
    spec_unc_out
        TYPE: 1d array of floats
        DESCRIPTION:  the error assiciated with spec, artifacts removed.
    '''
    
    #defining output varaibles
    spec_out = np.zeros(spec.shape)
    spec_unc_out = np.zeros(spec.shape)
    
    #defining variables to serve as exclusion criteria
    snr = np.log10(np.abs(spec / spec_unc))
    x = snr
    y = np.log10(np.abs(spec))
    
    #creating array which has bad pixels identified
    good = np.isfinite(x) & np.isfinite(y) & (spec > 0)
    
    #new array has values of old array for good pixels, 0 for bad pixels
    spec_out[good] = spec[good]
    spec_unc_out[good] = spec_unc[good]

    return spec_out, spec_unc_out



#needed for ryans reproject function
def clip_spikes_cube(cube, cube_unc):
    '''
    A wrapper function for clip_spikes_spec, meant to work on data cubes. 
    To work with a slice, change dimentions from (x,y) to (1,x,y)
    For example: data = data[np.newaxis, :, :]
​
    Parameters
    ----------
    cube
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [k,i,j] k is wavelength index, i and j are position index.
    cube_unc
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [k,i,j] k is wavelength index, i and j are position index.
            
    Returns
    -------
    cube_out
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, artifacts removed.
            for [k,i,j] k is wavelength index, i and j are position index.
    cube_unc_out
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data, artifacts removed.
                for [k,i,j] k is wavelength index, i and j are position index.
    '''
    
    #defining output variables
    cube_out = np.zeros(cube.shape)
    cube_unc_out = np.zeros(cube.shape)
    nw, nx, ny = cube.shape
    for ix in range(nx):
        for iy in range(ny):
            #calling clip_spikes_spec
            spec = cube[:, ix, iy]
            spec_unc = cube_unc[:, ix, iy]
            
            #saving output
            res = clip_spikes_spec(spec, spec_unc)
            cube_out[:, ix, iy] = res[0]
            cube_unc_out[:, ix, iy] = res[1]
            
    return cube_out, cube_unc_out



####################################



'''
REPROJECTION FUNCTIONS
'''



def reproject_slice(fits_to_reproject, fits_reference,
                     header_index, data_index, error_index, dq_index,
                     uncertainty=False,
                     clip_spikes=False,
                     progress_updates=True):
    '''
    This function loads in fits data slices (not necessarily JWST data products). It then reprojects the fits files
    to match the header of the fits file located at fits_reference, and returns the reprojected data OR error slice,
    but not both at the same time.
    
    If you do not have error or dq extentions, set them to None.
    
    Call this function if you are using modified data, and not JWST s3d data cubes whose header and overall shape is unmodified.
    
    Parameters
    ----------
    fits_to_reproject
        TYPE: string
        DESCRIPTION: where the fits file to be reprojected is located.        
    fits_reference
        TYPE: string
        DESCRIPTION: where the fits file with the header to be used as a reference is located
    header_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index to get wavelength data from in the header; usually 1. This header will also
            store info about WCS coordinates, and so it is used for 'fits reference', in the event that 'fits_to_reproject' 
            has its data in different extensions
    data_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index where the primary data in fits_to_reproject is stored (the default for JWST is 1)
    error_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index where the error data in fits_to_reproject is stored (the default for JWST is 2)
    dq_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index where the dq (data quality) data in fits_to_reproject is stored (the default for JWST is 3)
    uncertainty
        TYPE: boolean
        DESCRIPTION: whether or not to treat the data to be reprojected as an error array, is False by default.
    clip_spikes
        TYPE: boolean
        DESCRIPTION: whether or not to run the clip_spikes_cube function. Requires error data, is False by default.
    progress_updates
        TYPE: boolean
        DESCRIPTION: whether or not to include the various print statements built in to the function that indicate progress.
            Is useful for larger cubes and bug-testing, but would be annoying if the cubes are small and the function finishes quickly.
            Is True by default

    Returns
    -------
    cube
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, reprojected.
            for [i,j,k] k is wavelength index, i and j are position index. (check this, ryans code loads the data weird)
    '''
    
    #loading in data
    input_data = []
    
    data = fits.open(fits_reference)[header_index].data
    hdr = fits.open(fits_reference)[header_index].header
    w = wcs.WCS(hdr)
    input_data += [(data, w)]
    
    #getting wcs coords
    wcs_out, shape_out = find_optimal_celestial_wcs(input_data)


    #loading in data
    data = fits.open(fits_to_reproject)[data_index].data
    
    if error_index is not None:
        data_unc = fits.open(fits_to_reproject)[error_index].data
        
    if dq_index is not None:
        data_dq = fits.open(fits_to_reproject)[dq_index].data
        
    wcs_cube = get_2d_wcs_from_data(fits_to_reproject, header_index)

    if clip_spikes == True:
        if progress_updates == True:
            print("projection_field: clipping spikes")
        data, data_unc = clip_spikes_cube(data, data_unc)
    
    #this function is designed to do data or uncertainty, but not both at the same time, for increased malleability
    if uncertainty == True:
        data = abs(data_unc)
    
    #masking bad pixels
    if dq_index is not None:
        #any pixels with 'data quality' (DQ) nonzero have been flagged for some reason (the specific reason depends on the number and telescope)
        #and all of them are removed for simplicity. They are set to nan and not 0 to differentiate them
        #from other steps, where different kinds of bad data are set to 0 (I think that was Ryans intention with nan and not 0 here)
        #also, if it is nan and not 0 then the value wont get reprojected at all, so its clear after that it was a bad pixel.
        dqmask = data_dq != 0
        data[dqmask] = np.nan
        
    if progress_updates == True:
        print("Reprojection")
        
    #note reproject_exact takes a tuple of the data and an astropy wcs object
    reprojected_slice, _ = reproject_exact((data, wcs_cube),
                             wcs_out,
                             shape_out,
                             parallel=False)
        
    if progress_updates == True:
        print("Done")
            
    return reprojected_slice



def reproject_cube(fits_to_reproject, fits_reference,
                     header_index, data_index, error_index, dq_index,
                     reprojection_index,
                     clip_spikes=False,
                     progress_updates=True,
                     wavelength_cube_axis=0):
    '''
    This function loads in fits data cubes (not necessarily JWST data products). It then reprojects the fits files
    to match the header of the fits file located at fits_reference, and returns the reprojected data OR error cube,
    but not both at the same time.
    
    If you do not have error or dq extentions, set them to the same extention as the data.
    
    Call this function if you are using modified data, and not JWST s3d data cubes whose header and overall shape is unmodified.
    
    Parameters
    ----------
    fits_to_reproject
        TYPE: string
        DESCRIPTION: where the fits file to be reprojected is located.        
    fits_reference
        TYPE: string
        DESCRIPTION: where the fits file with the header to be used as a reference is located
    header_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index to get wavelength data from in the header; usually 1. This header will also
            store info about WCS coordinates, and so it is used for 'fits reference', in the event that 'fits_to_reproject' 
            has its data in different extensions
    error_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index where the error data in fits_to_reproject is stored (the default for JWST is 2)
    dq_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index where the dq (data quality) data in fits_to_reproject is stored (the default for JWST is 3)
    reprojection_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index specifying which cube will be reprojected. by default, 1 for data, 2 for error, 3 for DQ.
    clip_spikes
        TYPE: boolean
        DESCRIPTION: whether or not to run the clip_spikes_cube function, is False by default. Only works if reprojecting data
    progress_updates
        TYPE: boolean
        DESCRIPTION: whether or not to include the various print statements built in to the function that indicate progress.
            Is useful for larger cubes and bug-testing, but would be annoying if the cubes are small and the function finishes quickly.
            Is True by default
    wavelength_cube_axis
        TYPE: index (0, 1, 2)
        DESCRIPTION: which axis in the cube corresponds to wavelength, is 0 by default.

    Returns
    -------
    cube
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, reprojected.
            for [i,j,k] k is wavelength index, i and j are position index. (check this, ryans code loads the data weird)
    '''
    
    #loading in data, ryan set it up to have wavelengths be the final index. 
    input_data = []
    
    #this approach assumes that your data is a cube and thats it, so its ideal for finding the size of the 2d array to make
    data = np.nansum(fits.open(fits_reference)[header_index].data, axis=wavelength_cube_axis)
    hdr = fits.open(fits_reference)[header_index].header
    w = wcs.WCS(hdr).dropaxis(2)
    
    #peculiar structure of input_data needed for find_optimal_celestial_wcs
    input_data += [(data, w)]
    
    #getting wcs coords
    wcs_out, shape_out = find_optimal_celestial_wcs(input_data)

    #this function is designed to do data or uncertainty or DQ, but only one, for increased malleability
    #loading in data
    data = fits.open(fits_to_reproject)[reprojection_index].data
    data_unc = fits.open(fits_to_reproject)[error_index].data
    data_dq = fits.open(fits_to_reproject)[dq_index].data
    wcs_cube = get_2d_wcs_from_data(fits_to_reproject, header_index)

    if clip_spikes == True:
        if progress_updates == True:
            print("projection_field: clipping spikes")
        data, data_unc = clip_spikes_cube(data, data_unc)
        
    #creating output variable
    cube = np.zeros([shape_out[0], shape_out[1], data.shape[0]])

    n_planes = data.shape[0]
    for i in range(n_planes):
        if progress_updates == True:
            print(f'Reprojecting plane {i} of {n_planes}')
        arr = data[i, :, :]
        
        #any pixels with 'data quality' (DQ) nonzero have been flagged for some reason (the specific reason depends on the number and telescope)
        #and all of them are removed for simplicity. They are set to nan and not 0 to differentiate them
        #from other steps, where different kinds of bad data are set to 0 (I think that was Ryans intention with nan and not 0 here)
        #also, if it is nan and not 0 then the value wont get reprojected at all, so its clear after that it was a bad pixel. skipped if
        #reprojecting DQ array
        if reprojection_index !=3:
            dqmask = data_dq[i, :, :] != 0
            arr[dqmask] = np.nan
        if progress_updates == True:
            print("Reprojection")
        arr, _ = reproject_exact((arr, wcs_cube),
                                 wcs_out,
                                 shape_out,
                                 parallel=False)

        cube[:, :, i] = arr
        if progress_updates == True:
            print("Done")
            
    return cube



####################################



'''
REPROJECTION FUNCTION WRAPPERS
'''



def loading_function_reproject_cube(fits_to_reproject, fits_reference, 
                               header_index, data_index = 1, error_index = 2, dq_index = 3, is_slice=False):
    '''
    This function loads in JWST MIRI and NIRSPEC fits data cubes, extracts wavelength 
    data from the header, and builds the corresponding wavelength array. It then reprojects the fits files
    to match the header of the fits file located at fits_reference, and returns the reprojected data and error cubes,
    as well as the corresponding wavelength array.
    
    Call this function if you are interested in your reprojected data being immediately available as variables for you to use.
    
    Parameters
    ----------
    fits_to_reproject
        TYPE: string
        DESCRIPTION: where the fits file to be reprojected is located.        
    fits_reference
        TYPE: string
        DESCRIPTION: where the fits file with the header to be used as a reference is located
    header_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index to get wavelength data from in the header; usually 1. This header will also
            store info about WCS coordinates, and so it is used for 'fits reference', in the event that 'fits_to_reproject' 
            has its data in different extensions
    data_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index where the primary data in fits_to_reproject is stored, is 1 by default (the default for JWST)
    error_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index where the error data in fits_to_reproject is stored, is 2 by default (the default for JWST)
    dq_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index where the dq (data quality) data in fits_to_reproject is stored, is 3 by default (the default for JWST)
    is_slice
        TYPE: boolean
        DESCRIPTION: if set to true, data is a single slice lacking wavelength information, instead of a data cube.

    Returns
    -------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns.
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, reprojected.
            for [k,i,j] k is wavelength index, i and j are position index.
    error_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data, reprojected.
                for [k,i,j] k is wavelength index, i and j are position index.
    '''
    
    #load in the data for building wavelength array
    image_file = get_pkg_data_filename(fits_to_reproject)
    
    
    #header data
    science_header = fits.getheader(image_file, header_index)
    
    #wavelength data from header
    number_wavelengths = science_header["NAXIS3"]
    wavelength_increment = science_header["CDELT3"]
    wavelength_start = science_header["CRVAL3"]
    
    #constructing the ending point using given data
    #subtracting 1 so wavelength array is the right size.
    wavelength_end = wavelength_start + (number_wavelengths - 1)*wavelength_increment
    
    #extracting reprojected image data, error data (both 3d cubes)
    image_data = reproject_cube(fits_to_reproject, fits_reference, header_index, data_index, error_index, dq_index, 1)
    error_data = reproject_cube(fits_to_reproject, fits_reference, header_index, data_index, error_index, dq_index, 2)
    dq_data = reproject_cube(fits_to_reproject, fits_reference, header_index, data_index, error_index, dq_index, 3)
    
    #changing axes order, so that they match JWST data cubes default shape
    temp = np.swapaxes(image_data, 0, 2)
    image_data = np.swapaxes(temp, 1, 2)
    
    temp = np.swapaxes(error_data, 0, 2)
    error_data = np.swapaxes(temp, 1, 2)
    
    temp = np.swapaxes(dq_data, 0, 2)
    dq_data = np.swapaxes(temp, 1, 2)
    
    #replacing nans with 0
    where_are_NaNs = np.isnan(image_data) 
    image_data[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(error_data) 
    error_data[where_are_NaNs] = 0
    
    #in case the wavelength separation was reprojected, check new vs old values. 
    #(This is nicholas future proofing the code and saving a headache later for if/when this gets added)
    
    wavelength_increment_new = (wavelength_end - wavelength_start)/len(image_data[:,0,0])
    
    if wavelength_increment_new != wavelength_increment:
    
        #remaking wavelength array, in micrometers
        wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment_new)
    
    #sometimes wavelength array is 1 element short, this will fix that
    if len(wavelengths) != len(image_data):
        wavelength_end = wavelength_start + number_wavelengths*wavelength_increment
        wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment)

    return wavelengths, image_data, error_data, dq_data



'''
REPROJECTION FUNCTION WRAPPER THAT SAVES DATA TO FITS FILE
'''


def fits_reprojection_cube(fits_to_reproject, new_fits_name, fits_reference):
    '''
    Note that this function is tailored to using JWST data products that have not been heavily modified, 
    i.e. default header and extension placement for JWST data products. Notably, it assumes that both 
    the file getting reprojected and the reference file have the same (untampered) layout.
    
    it is also assumed that data is at ext 1, errors at ext 2, and DQ at ext 3 (the default for JWST)
    
    This function is a wrapper for loading_function_reproject, saving the reprojected fits cube in a separate 
    location that is specified by the user. Specifically, the new cube will be a copy of the old cube, but with
    its science data, error data (located at extentions 1 and 2 respectively) reprojected, and its header
    updated to reflect the reprojection. Other extentions will reflect the cube before reprojection (notably DQ), and this
    function will need to be updated to reproject those also if they are needed, but at the moment they are not needed,
    so they are ignored to keep the runtime lower.
    
    Parameters
    ----------
    fits_to_reproject
        TYPE: string
        DESCRIPTION: where the fits file to be reprojected is located.        
    new_fits_name
        TYPE: string
        DESCRIPTION: where the new reprojected fits file is saved.        
    fits_reference
        TYPE: string
        DESCRIPTION: where the fits file with the header to be used as a reference is located
    '''

    with fits.open(fits_to_reproject) as hdul:
        
        #calling reprojection loading function, with header_index = 1
        wavelengths, image_data, error_data, dq_data = loading_function_reproject_cube(
            fits_to_reproject, fits_reference, 1)
        
        #replacing data in the currently open fits file
        hdul[1].data = image_data
        hdul[2].data = error_data
        hdul[3].data = dq_data
        
        #updating science header, new values should match those in the reference file
        
        #opening reference file
        with fits.open(fits_reference) as hdul2:
            
            '''
            CRVAL1,2 = wcs coordinates of fits file
            CRPIX1,2 = pixel coordinate that corresponds to CRVAL1,2
            CDELT1,2 = pixel size
            '''
            
            hdul[1].header['CRVAL1'] = hdul2[1].header['CRVAL1']
            hdul[1].header['CRVAL2'] = hdul2[1].header['CRVAL2']
            hdul[1].header['CRPIX1'] = hdul2[1].header['CRPIX1']
            hdul[1].header['CRPIX2'] = hdul2[1].header['CRPIX2'] 
            hdul[1].header['CDELT1'] = hdul2[1].header['CDELT1']
            hdul[1].header['CDELT2'] = hdul2[1].header['CDELT2']
        
        #saving data, replacing any files with the same name for convinient reruns
        hdul.writeto(new_fits_name, overwrite=True)
