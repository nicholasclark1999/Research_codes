#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:40:43 2024

@author: nclark
"""

'''
IMPORTING MODULES
'''

#standard stuff
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import  AutoMinorLocator
import random

#saving imagaes as PDFs
from PIL import Image  # install by > python3 -m pip install --upgrade Pillow  # ref. https://pillow.readthedocs.io/en/latest/installation.html#basic-installation

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

#barias correlation colour coding code
#import Corr_to_Map_Plotting_Function as corr




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



def single_emission_line_remover(wavelengths, image_data, wave_list):
    '''
    removes a single emission line occupying a specified wavelength range, by replacing the line with
    a linear function. The slope is calculated by using points below the blue end and above the red end.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined together as described above.
    image_data
        TYPE: 1d array of floats
        DESCRIPTION: a spectra, with the line to be removed.
        wave_list
            TYPE: list of floats
            DESCRIPTION: the wavelengths in microns, corresponding to the beginning and ending of the line to be removed

    Returns
    -------
    new_data 
        TYPE: 1d array of floats
        DESCRIPTION: spectra with emission line removed.
    '''
    
    
    def temp_index_generator(wave):
        #defining this within my function so i can make more advanced rounding code
        #without cluttering my function with repetition
        temp_index = np.where(np.round(wavelengths, 3) == wave)[0]
        if len(temp_index) == 0:
            temp_index = np.where(np.round(wavelengths, 2) == np.round(wave, 2))[0][0]
        else:
            temp_index = temp_index[0]
            
        return temp_index
        
    temp_index_1 = temp_index_generator(wave_list[0])
    temp_index_2 = temp_index_generator(wave_list[1])
    
    #calculating the slope of the line to use
    
    pah_slope_1 = np.median(image_data[temp_index_1 - 5:temp_index_1])
    if temp_index_1 < 5:
        pah_slope_1 = np.median(image_data[:temp_index_1])
    pah_slope_2 = np.median(image_data[temp_index_2:5+temp_index_2])
    if int(len(wavelengths)) - temp_index_2 < 5:
        pah_slope_2 = np.median(image_data[temp_index_2:])
    
    pah_slope = (pah_slope_2 - pah_slope_1)/\
        (wavelengths[temp_index_2] - wavelengths[temp_index_1])

    #putting it all together
    no_line = pah_slope*(wavelengths[temp_index_1:temp_index_2] - wavelengths[temp_index_1]) + pah_slope_1
    
    #removing line in input array
    
    new_data = np.copy(image_data)
    
    for i in range(len(no_line)):
        new_data[temp_index_1 + i] = no_line[i]

    return new_data



def emission_line_remover_wrapper(wavelengths, image_data, wave_list):
    '''
    Wrapper function for the emission line remover, allowing it to work over the entire data cube and not
    just a single spectra.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined together as described above.
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: a spectra, indices [k,i,j] where k is spectral index and i,j are position index
        wave_list
            TYPE: list of floats
            DESCRIPTION: the wavelengths in microns, corresponding to the beginning and ending of the line to be removed

    Returns
    -------
    new_data 
        TYPE: 3d array of floats
        DESCRIPTION: spectra with emission line removed.
    '''
    
    new_data = np.copy(image_data)
    for i in range(len(image_data[0,:,0])):
        for j in range(len(image_data[0,0,:])):\
            new_data[:,i,j] = single_emission_line_remover(wavelengths, image_data[:,i,j], wave_list)
            
    return new_data




def correct_units_astropy(cube, wavelengths):
        """
        Corrects the units of cubes by changing them from MJy/sr to W/m^2/um/str (with astropy)
        
        Parameters
        ----------
        cube [str or subtract_fits]: the cube(1d array now) whose units need to be corrected (in MJy/sr)
        
        directory_cube_data [str]: the directory of a .fits spectral cube file (in MJy/sr)
    
        directory_wave [str]: the directory of the file with the wavelengths (in micrometers)
    
        function_wave [function]: the function used to read in the wavelengths
                                  get_data: for .xdr files
                                  get_wavlengths: for .tbl files
                                  get_wave_fits: for .fits files
    
        Returns
        -------
        np.array : an array in W/m^2/sr/um
    
        """
        final_cube = np.zeros(cube.shape)
        cube_with_units = (cube*10**6)*(u.Jy/u.sr)
        print('cube with units')
        for i in range(len(wavelengths)):
            final_cube[i] = cube_with_units[i].to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavelengths[i]*u.micron))
        return(final_cube)
    
    
def omega_linear_continuum(wavelengths, image_data, wave_list, adjustment):
    
    temp_index_1 = np.where(np.round(wavelengths, 2) == wave_list[0])[0][0]
    temp_index_2 = np.where(np.round(wavelengths, 2) == wave_list[1])[0][0]
    temp_index_3 = np.where(np.round(wavelengths, 2) == wave_list[2])[0][0]
    temp_index_4 = np.where(np.round(wavelengths, 2) == wave_list[3])[0][0]
    temp_index_5 = np.where(np.round(wavelengths, 2) == wave_list[4])[0][0]
    temp_index_6 = np.where(np.round(wavelengths, 2) == wave_list[5])[0][0]
    temp_index_7 = np.where(np.round(wavelengths, 2) == wave_list[6])[0][0]
    temp_index_8 = np.where(np.round(wavelengths, 2) == wave_list[7])[0][0]
    temp_index_9 = np.where(np.round(wavelengths, 2) == wave_list[8])[0][0]
    temp_index_10 = np.where(np.round(wavelengths, 2) == wave_list[9])[0][0]
    temp_index_11 = np.where(np.round(wavelengths, 2) == wave_list[10])[0][0]
    temp_index_12 = np.where(np.round(wavelengths, 2) == wave_list[11])[0][0]
    temp_index_13 = np.where(np.round(wavelengths, 2) == wave_list[12])[0][0]
    temp_index_14 = np.where(np.round(wavelengths, 2) == wave_list[13])[0][0]
    temp_index_15 = np.where(np.round(wavelengths, 2) == wave_list[14])[0][0]
    temp_index_16 = np.where(np.round(wavelengths, 2) == wave_list[15])[0][0]
    temp_index_17 = np.where(np.round(wavelengths, 2) == wave_list[16])[0][0]
    temp_index_18 = np.where(np.round(wavelengths, 2) == wave_list[17])[0][0]
    temp_index_19 = np.where(np.round(wavelengths, 2) == wave_list[18])[0][0]
    temp_index_20 = np.where(np.round(wavelengths, 2) == wave_list[19])[0][0]
    temp_index_21 = np.where(np.round(wavelengths, 2) == wave_list[20])[0][0]
    temp_index_22 = np.where(np.round(wavelengths, 2) == wave_list[21])[0][0]
    temp_index_23 = np.where(np.round(wavelengths, 2) == wave_list[22])[0][0]
    temp_index_24 = np.where(np.round(wavelengths, 2) == wave_list[23])[0][0]
    temp_index_25 = np.where(np.round(wavelengths, 2) == wave_list[24])[0][0]
    
    pah_slope_beginning = image_data[0]
    pah_slope_ending = image_data[-1]
    
    #calculating the slope of the line to use
    
    #preventing the value found from being on a line or something
    pah_slope_1 = np.median(image_data[temp_index_1 - 15:15+temp_index_1]) - adjustment[0]
    pah_slope_2 = np.median(image_data[temp_index_2 - 15:15+temp_index_2]) - adjustment[1]
    pah_slope_3 = np.median(image_data[temp_index_3 - 15:15+temp_index_3]) - adjustment[2]
    pah_slope_4 = np.median(image_data[temp_index_4 - 15:15+temp_index_4]) - adjustment[3]
    pah_slope_5 = np.median(image_data[temp_index_5 - 15:15+temp_index_5]) - adjustment[4] 
    pah_slope_6 = np.median(image_data[temp_index_6 - 15:15+temp_index_6]) - adjustment[5]
    pah_slope_7 = np.median(image_data[temp_index_7 - 15:15+temp_index_7]) - adjustment[6]
    pah_slope_8 = np.median(image_data[temp_index_8 - 15:15+temp_index_8]) - adjustment[7]
    pah_slope_9 = np.median(image_data[temp_index_9 - 15:15+temp_index_9]) - adjustment[8]
    pah_slope_10 = np.median(image_data[temp_index_10 - 15:15+temp_index_10]) - adjustment[9] # (was 20 instead of 40 before) shorter because theres a feature to the left and line to the right
    pah_slope_11 = np.median(image_data[temp_index_11 - 15:15+temp_index_11]) - adjustment[10]
    pah_slope_12 = np.median(image_data[temp_index_12 - 15:15+temp_index_12]) - adjustment[11]
    pah_slope_13 = np.median(image_data[temp_index_13 - 15:15+temp_index_13]) - adjustment[12]
    pah_slope_14 = np.median(image_data[temp_index_14 - 15:15+temp_index_14]) - adjustment[13]
    pah_slope_15 = np.median(image_data[temp_index_15 - 15:15+temp_index_15]) - adjustment[14]
    pah_slope_16 = np.median(image_data[temp_index_16 - 15:15+temp_index_16]) - adjustment[15]
    pah_slope_17 = np.median(image_data[temp_index_17 - 15:15+temp_index_17]) - adjustment[16]
    pah_slope_18 = np.median(image_data[temp_index_18 - 15:15+temp_index_18]) - adjustment[17]
    pah_slope_19 = np.median(image_data[temp_index_19 - 15:15+temp_index_19]) - adjustment[18]
    pah_slope_20 = np.median(image_data[temp_index_20 - 15:15+temp_index_20]) - adjustment[19]
    pah_slope_21 = np.median(image_data[temp_index_21 - 15:15+temp_index_21]) - adjustment[20]
    pah_slope_22 = np.median(image_data[temp_index_22 - 15:15+temp_index_22]) - adjustment[21]
    pah_slope_23 = np.median(image_data[temp_index_23 - 15:15+temp_index_23]) - adjustment[22]
    pah_slope_24 = np.median(image_data[temp_index_24 - 15:15+temp_index_24]) - adjustment[23]
    pah_slope_25 = np.median(image_data[temp_index_25 - 15:15+temp_index_25]) - adjustment[24]
    
    pah_slope = (pah_slope_1 - pah_slope_beginning)/\
        (wavelengths[temp_index_1] - wavelengths[0])
    continuum0 = pah_slope*(wavelengths[:temp_index_1] - wavelengths[0]) + pah_slope_beginning
    
    pah_slope = (pah_slope_2 - pah_slope_1)/\
        (wavelengths[temp_index_2] - wavelengths[temp_index_1])
    continuum1 = pah_slope*(wavelengths[temp_index_1:temp_index_2] - wavelengths[temp_index_1]) + pah_slope_1
    
    pah_slope = (pah_slope_3 - pah_slope_2)/\
        (wavelengths[temp_index_3] - wavelengths[temp_index_2])
    continuum2 = pah_slope*(wavelengths[temp_index_2:temp_index_3] - wavelengths[temp_index_2]) + pah_slope_2
    
    pah_slope = (pah_slope_4 - pah_slope_3)/\
        (wavelengths[temp_index_4] - wavelengths[temp_index_3])
    continuum3 = pah_slope*(wavelengths[temp_index_3:temp_index_4] - wavelengths[temp_index_3]) + pah_slope_3
    
    pah_slope = (pah_slope_5 - pah_slope_4)/\
        (wavelengths[temp_index_5] - wavelengths[temp_index_4])
    continuum4 = pah_slope*(wavelengths[temp_index_4:temp_index_5] - wavelengths[temp_index_4]) + pah_slope_4
    
    pah_slope = (pah_slope_6 - pah_slope_5)/\
        (wavelengths[temp_index_6] - wavelengths[temp_index_5])
    continuum5 = pah_slope*(wavelengths[temp_index_5:temp_index_6] - wavelengths[temp_index_5]) + pah_slope_5
    
    pah_slope = (pah_slope_7 - pah_slope_6)/\
        (wavelengths[temp_index_7] - wavelengths[temp_index_6])
    continuum6 = pah_slope*(wavelengths[temp_index_6:temp_index_7] - wavelengths[temp_index_6]) + pah_slope_6
    
    pah_slope = (pah_slope_8 - pah_slope_7)/\
        (wavelengths[temp_index_8] - wavelengths[temp_index_7])
    continuum7 = pah_slope*(wavelengths[temp_index_7:temp_index_8] - wavelengths[temp_index_7]) + pah_slope_7
    
    pah_slope = (pah_slope_9 - pah_slope_8)/\
        (wavelengths[temp_index_9] - wavelengths[temp_index_8])
    continuum8 = pah_slope*(wavelengths[temp_index_8:temp_index_9] - wavelengths[temp_index_8]) + pah_slope_8
    
    pah_slope = (pah_slope_10 - pah_slope_9)/\
        (wavelengths[temp_index_10] - wavelengths[temp_index_9])
    continuum9 = pah_slope*(wavelengths[temp_index_9:temp_index_10] - wavelengths[temp_index_9]) + pah_slope_9
    
    pah_slope = (pah_slope_11 - pah_slope_10)/\
        (wavelengths[temp_index_11] - wavelengths[temp_index_10])
    continuum10 = pah_slope*(wavelengths[temp_index_10:temp_index_11] - wavelengths[temp_index_10]) + pah_slope_10
    
    pah_slope = (pah_slope_12 - pah_slope_11)/\
        (wavelengths[temp_index_12] - wavelengths[temp_index_11])
    continuum11 = pah_slope*(wavelengths[temp_index_11:temp_index_12] - wavelengths[temp_index_11]) + pah_slope_11
    
    pah_slope = (pah_slope_13 - pah_slope_12)/\
        (wavelengths[temp_index_13] - wavelengths[temp_index_12])
    continuum12 = pah_slope*(wavelengths[temp_index_12:temp_index_13] - wavelengths[temp_index_12]) + pah_slope_12
    
    pah_slope = (pah_slope_14 - pah_slope_13)/\
        (wavelengths[temp_index_14] - wavelengths[temp_index_13])
    continuum13 = pah_slope*(wavelengths[temp_index_13:temp_index_14] - wavelengths[temp_index_13]) + pah_slope_13
    
    pah_slope = (pah_slope_15 - pah_slope_14)/\
        (wavelengths[temp_index_15] - wavelengths[temp_index_14])
    continuum14 = pah_slope*(wavelengths[temp_index_14:temp_index_15] - wavelengths[temp_index_14]) + pah_slope_14
    
    pah_slope = (pah_slope_16 - pah_slope_15)/\
        (wavelengths[temp_index_16] - wavelengths[temp_index_15])
    continuum15 = pah_slope*(wavelengths[temp_index_15:temp_index_16] - wavelengths[temp_index_15]) + pah_slope_15
    
    pah_slope = (pah_slope_17 - pah_slope_16)/\
        (wavelengths[temp_index_17] - wavelengths[temp_index_16])
    continuum16 = pah_slope*(wavelengths[temp_index_16:temp_index_17] - wavelengths[temp_index_16]) + pah_slope_16
    
    pah_slope = (pah_slope_18 - pah_slope_17)/\
        (wavelengths[temp_index_18] - wavelengths[temp_index_17])
    continuum17 = pah_slope*(wavelengths[temp_index_17:temp_index_18] - wavelengths[temp_index_17]) + pah_slope_17
    
    pah_slope = (pah_slope_19 - pah_slope_18)/\
        (wavelengths[temp_index_19] - wavelengths[temp_index_18])
    continuum18 = pah_slope*(wavelengths[temp_index_18:temp_index_19] - wavelengths[temp_index_18]) + pah_slope_18
    
    pah_slope = (pah_slope_20 - pah_slope_19)/\
        (wavelengths[temp_index_20] - wavelengths[temp_index_19])
    continuum19 = pah_slope*(wavelengths[temp_index_19:temp_index_20] - wavelengths[temp_index_19]) + pah_slope_19
    
    pah_slope = (pah_slope_21 - pah_slope_20)/\
        (wavelengths[temp_index_21] - wavelengths[temp_index_20])
    continuum20 = pah_slope*(wavelengths[temp_index_20:temp_index_21] - wavelengths[temp_index_20]) + pah_slope_20
    
    pah_slope = (pah_slope_22 - pah_slope_21)/\
        (wavelengths[temp_index_22] - wavelengths[temp_index_21])
    continuum21 = pah_slope*(wavelengths[temp_index_21:temp_index_22] - wavelengths[temp_index_21]) + pah_slope_21
    
    pah_slope = (pah_slope_23 - pah_slope_22)/\
        (wavelengths[temp_index_23] - wavelengths[temp_index_22])
    continuum22 = pah_slope*(wavelengths[temp_index_22:temp_index_23] - wavelengths[temp_index_22]) + pah_slope_22
    
    pah_slope = (pah_slope_24 - pah_slope_23)/\
        (wavelengths[temp_index_24] - wavelengths[temp_index_23])
    continuum23 = pah_slope*(wavelengths[temp_index_23:temp_index_24] - wavelengths[temp_index_23]) + pah_slope_23
    
    pah_slope = (pah_slope_25 - pah_slope_24)/\
        (wavelengths[temp_index_25] - wavelengths[temp_index_24])
    continuum24 = pah_slope*(wavelengths[temp_index_24:temp_index_25] - wavelengths[temp_index_24]) + pah_slope_24
    
    pah_slope = (pah_slope_ending - pah_slope_25)/\
        (wavelengths[-1] - wavelengths[temp_index_25])
    continuum25 = pah_slope*(wavelengths[temp_index_25:] - wavelengths[temp_index_25]) + pah_slope_25
    
    #putting it all together
    continuum = np.concatenate((continuum0, continuum1))
    continuum = np.concatenate((continuum, continuum2))
    continuum = np.concatenate((continuum, continuum3))
    continuum = np.concatenate((continuum, continuum4))
    continuum = np.concatenate((continuum, continuum5))
    continuum = np.concatenate((continuum, continuum6))
    continuum = np.concatenate((continuum, continuum7))
    continuum = np.concatenate((continuum, continuum8))
    continuum = np.concatenate((continuum, continuum9))
    continuum = np.concatenate((continuum, continuum10))
    continuum = np.concatenate((continuum, continuum11))
    continuum = np.concatenate((continuum, continuum12))
    continuum = np.concatenate((continuum, continuum13))
    continuum = np.concatenate((continuum, continuum14))
    continuum = np.concatenate((continuum, continuum15))
    continuum = np.concatenate((continuum, continuum16))
    continuum = np.concatenate((continuum, continuum17))
    continuum = np.concatenate((continuum, continuum18))
    continuum = np.concatenate((continuum, continuum19))
    continuum = np.concatenate((continuum, continuum20))
    continuum = np.concatenate((continuum, continuum21))
    continuum = np.concatenate((continuum, continuum22))
    continuum = np.concatenate((continuum, continuum23))
    continuum = np.concatenate((continuum, continuum24))
    continuum = np.concatenate((continuum, continuum25))

    return continuum
    









def linear_continuum_single_channel(wavelengths, image_data, wave_list):
    
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
    



#why do i have 2 different unit converters and a non function version in my ring nebula data that all seem to do the same thing?

def unit_changer(wavelengths, data):
    
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





def pah_feature_integrator(wavelengths, data):

        
    final_cube = np.zeros(data.shape)
    cube_with_units = (data*10**6)*(u.Jy/u.sr)

    final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies =\
                                    u.spectral_density(wavelengths*u.micron))
        
    final_cube = final_cube*(u.micron)
    final_cube = final_cube*(u.m)
    final_cube = final_cube*(u.m)
    final_cube = final_cube*(u.sr/u.W)
    
    integrand_temp = np.copy(data)
    for i in range(len(data)):
        integrand_temp[i] = float(final_cube[i])



    odd_sum = 0

    for i in range(1, len(integrand_temp), 2):
        odd_sum += integrand_temp[i] 

    even_sum = 0    

    for i in range(2, len(integrand_temp), 2):
        even_sum += integrand_temp[i] 
    
    #NOTE THAT THIS WILL NOT WORK IF WAVELENGTH CONTAINS MULTIPLE H; WILL NEED TO INTEGRATE
    #ALONG WAVELENGTH CHANNELS AND ADD THEM TOGETHER
    
    h = wavelengths[1] - wavelengths[0]
    
    integral = (h/3)*(integrand_temp[0] + integrand_temp[-1] + 4*odd_sum + 2*even_sum)
    
    
    return integral



def pah_feature_integrator_centroid(wavelengths, data, integral):

    
    #first, calculate from the left

    final_cube = np.zeros(data.shape)
    cube_with_units = (data*10**6)*(u.Jy/u.sr)

    final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies =\
                                    u.spectral_density(wavelengths*u.micron))
        
    final_cube = final_cube*(u.micron)
    final_cube = final_cube*(u.m)
    final_cube = final_cube*(u.m)
    final_cube = final_cube*(u.sr/u.W)
    
    integrand_temp = np.copy(data)
    for i in range(len(data)):
        integrand_temp[i] = float(final_cube[i])

    #add them one at a time instead of all at once to find centroid
    
    h = wavelengths[1] - wavelengths[0]
    
    #from the left
    
    i = 1
    integral_left = (h/3)*(integrand_temp[0])
    
    #note that this approach will have a slight innacuracy, in that the end point will be overcounted (multiplied by 2 or 4 instead of 1)
    
    while integral_left <integral/2:
        
        if i%2 == 0: 
            integral_left += 2*(h/3)*(integrand_temp[i])
            
        else:
            integral_left += 4*(h/3)*(integrand_temp[i])
            
        i += 1
        
    #from the right
    
    j = len(integrand_temp) - 1

    integral_right = (h/3)*(integrand_temp[-1])
    
    #note that this approach will have a slight innacuracy, in that the end point will be overcounted (multiplied by 2 or 4 instead of 1)
    
    while integral_right <integral/2:
        
        if j%2 == 0: 
            integral_right += 2*(h/3)*(integrand_temp[j])
            
        else:
            integral_right += 4*(h/3)*(integrand_temp[j])
            
        j -= 1

    return i, j, integral_left, integral_right



def CalculateR(wavelength):
    
    #ch1:
    #A:
    if 4.9 <= wavelength <= 5.74 :
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



def error_finder(wavelengths, data, feature_wavelength, integral, feature_indices, error_index):
    #calculates error of assosiated integrals
    rms_data = unit_changer(wavelengths[error_index-25:error_index+25], data[error_index-25:error_index+25])
    
    rms = (np.var(rms_data))**0.5
    
    resolution = CalculateR(feature_wavelength)
    
    delta_wave = feature_wavelength/resolution
    
    num_points = (wavelengths[feature_indices[1]] - wavelengths[feature_indices[0]])/delta_wave
    
    snr = integral/(rms*delta_wave*(num_points)**0.5)
    
    error = integral/snr
    
    return error



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
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
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

        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        #overlap = (overlap_a, overlap_length_b)
        overlap = (lower_index, upper_index)
    
    return image_data, wavelengths, overlap


def flux_aligner3(wave_a, wave_b, data_a, data_b):
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
    check_a = np.round(wave_a[-1] - wave_a[-2], 4)
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
        
        #apply offset
        distance = 10
        if overlap_length < 20:
            distance = 5
        
        offset1 = data_a[lower_index] - temp[upper_index]
        offset2 = data_a[lower_index - distance] - temp[upper_index - distance]
        offset3 = data_a[lower_index + distance] - temp[upper_index + distance]
        
        offset = (offset1 + offset2 + offset3)/3
        
        #offset = data_a[lower_index, i] - temp[upper_index, i]
        temp = temp + offset
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
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
        
        #apply offset
        distance = 10
        if overlap_length_a < 20 or overlap_length_b < 20:
            distance = 5
        
        offset1 = data_a[lower_index] - temp[upper_index]
        offset2 = data_a[lower_index - distance] - temp[upper_index - distance]
        offset3 = data_a[lower_index + distance] - temp[upper_index + distance]
        
        offset = (offset1 + offset2 + offset3)/3
        
        #offset = data_a[lower_index, i] - temp[upper_index, i]
        temp = temp + offset
        
        '''
        for i in range(len(temp[0])):
            #applying an offset to each spectra
            #mean_dataA = np.mean(data_a[lower_index:,i])
            #mean_dataB = np.mean(temp[:upper_index,i])
            #offset = mean_dataA - mean_dataB
            
            offset1 = data_a[lower_index, i] - temp[upper_index, i]
            offset2 = data_a[lower_index-10, i] - temp[upper_index-10, i]
            offset3 = data_a[lower_index+10, i] - temp[upper_index+10, i]
            
            offset = (offset1 + offset2 + offset3)/3
            
            #offset = data_a[lower_index, i] - temp[upper_index, i]
            temp[:,i] = temp[:,i] + offset
            if i == 7:
                print(i, offset)
            if i == 27:
                print(i, offset)
            '''
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        #overlap = (overlap_a, overlap_length_b)
        overlap = (lower_index, upper_index)
    
    return image_data, wavelengths, overlap


def flux_aligner4(wave_a, wave_b, data_a, data_b):
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
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
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

        #hard coded because the offset is weird around 7.58, the usual strat wont work
            
            #using a wavelength of 7.525 roughly
            
        offset = data_a[1243] - temp[11]
        temp = temp + offset

        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        #overlap = (overlap_a, overlap_length_b)
        overlap = (lower_index, upper_index)
    
    return image_data, wavelengths, overlap





#this version is for data without a background
def extract_spectra_from_regions_one_pointing_no_bkg(fname_cube, data, fname_region, do_sigma_clip=True, use_dq=False):
    
    reg = regions.Regions.read(fname_region, format='ds9')
    fits_cube = fits.open(fname_cube)
    w = wcs.WCS(fits_cube[1].header).dropaxis(2)
    #modifying to use 2d slice instead of 3d array for increased utility
    region_indicator = np.zeros((len(data[:,0]), len(data[0,:])))
    #dq = fits_cube['DQ'].data
    all_spectra = dict()
    all_spectra_unc = dict()

    # loop over regions in .reg file
    for i in range(len(reg)):
        regmask = reg[i].to_pixel(w).to_mask('center').to_image(shape=data.shape[0:])
        if regmask is not None:
            all_spectra[f'region_{i}'] = []
            all_spectra_unc[f'region_{i}'] = []
            #print(f"Region {i}")

            for ix in range(data.shape[0]):
                for iy in range(data.shape[1]):
                    if regmask[ix, iy] == 1:
                        region_indicator[ix, iy] = 1


    return region_indicator




def colormap_values(data, error, snr):
    

    good_data = np.zeros((len(data[:,0]), len(data[0,:]))) - 1 #all empty values will be -1

    
    for i in range(len(data[:,0])):
        for j in range(len(data[0,:])):
            if data[i,j]/error[i,j] > snr:
                good_data[i,j] = data[i,j]

    max_value = np.max(good_data)
    
    good_data = np.abs(good_data) #now all empty values will be 1; since all intergal values are between 0 and 1, will allow for min and max to be extracted
    
    min_value = np.min(good_data)
    
    return max_value, min_value



def colormap_values_normalized(data, error, norm, snr):
    

    good_data = np.zeros((len(data[:,0]), len(data[0,:]))) - 1 #all empty values will be -1

    
    for i in range(len(data[:,0])):
        for j in range(len(data[0,:])):
            if data[i,j]/error[i,j] > snr:
                good_data[i,j] = data[i,j]/norm[i,j] #good data is now normalized

    max_value = np.max(good_data)
    
    good_data = np.abs(good_data) #now all empty values will be 1; since all intergal values are between 0 and 1, will allow for min and max to be extracted
    
    min_value = np.min(good_data)
    
    return max_value, min_value



def colormap_values_for_error(data, error, snr):
    
    #does the same analysis as above but meant for errors with the exact same array meant to 
    #correspond to the array created for the data

    good_data = np.zeros((len(data[:,0]), len(data[0,:]))) - 1 #all empty values will be -1

    
    for i in range(len(data[:,0])):
        for j in range(len(data[0,:])):
            if data[i,j]/error[i,j] > snr:
                good_data[i,j] = error[i,j]

    max_value = np.max(good_data)
    
    good_data = np.abs(good_data) #now all empty values will be 1; since all intergal values are between 0 and 1, will allow for min and max to be extracted
    
    min_value = np.min(good_data)
    
    return max_value, min_value



def colormap_values_for_comparison(data, error, snr):
    

    good_data = np.zeros((len(data[:,0]), len(data[0,:]))) - 1 #all empty values will be -1

    
    for i in range(len(data[:,0])):
        for j in range(len(data[0,:])):
            if data[i,j]/error[i,j] > snr:
                good_data[i,j] = data[i,j]

    return good_data




def error_check_imager(wavelengths, data, pdf_name, lower, upper, ylim_scaler, 
                       data_no_lines=None, continuum=None, comparison_wave_1=None, comparison_data_1=None, 
                       comparison_wave_2=None, comparison_data_2=None, comparison_scale_wave=None,
                       scale_wave=None, scale_data=None, scale_wave_comp=None, scale_data_comp=None, 
                       min_ylim=None, conttype=None, cont_points=None, regrid=None, check_plat=None, check_curve=None,
                       selection_array=None):
    '''
    A function that plots 10 spectra from within a cube using a hard coded seed, for checking that things
    being done to the entire cube are working, such as emission line removal and continuum fitting. Saves these 
    10 figures into a PDF.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array that corresponds to data, with units um
    data
        TYPE: 3d array of floats
        DESCRIPTION: a spectra, indices [k,i,j] where k is spectral index and i,j are position index
    pdf_name
        TYPE: string
        DESCRIPTION: name of the pdf file, the string includes .pdf in it
    data_no_lines
        TYPE: 3d array of floats
        DESCRIPTION: data with emission lines removed, indices [k,i,j] where k is spectral index and i,j are position index.
    continuum
        TYPE: 3d array of floats
        DESCRIPTION: a continuum corresponding to data, indices [k,i,j] where k is spectral index and i,j are position index.
    comparison_wave_1
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array that corresponds to comparison data, with units um.
    comparison_data_1
        TYPE: 1d array of floats, or 3d array of floats
        DESCRIPTION: spectra of another object, for comparison. If 3d array is given, will use the same i and j as data.
    comparison_wave_2
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array that corresponds to second comparison data, with units um.
    comparison_data_2
        TYPE: 1d array of floats
        DESCRIPTION: spectra of a second another object, for comparison.
    comparison_scale_wave
        TYPE: 1d array of floats
        DESCRIPTION: wavelength to be used in scaling for comparison.
    scale_wave
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array to be used in scaling, if comparison_scale_wave is outside the wavelength range of data.
    scale_data
        TYPE: 1d array of floats
        DESCRIPTION: spectra of data to be used in scaling, if comparison_scale_wave is outside the wavelength range of data.
    scale_wave_comp
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array to be used in scaling, if comparison_scale_wave is outside the wavelength range of data.
    scale_data_comp
        TYPE: 1d array of floats
        DESCRIPTION: spectra of data to be used in scaling, if comparison_scale_wave is outside the wavelength range of data.
    min_ylim
        TYPE: float
        DESCRIPTION: optional scaler for the min in ylim.
    conttype
        TYPE: 2d array of floats
        DESCRIPTION: type of continuum used, stored in the title of the plots.
    cont_points
        TYPE: list of floats
        DESCRIPTION: anchor points used in continuum, plotted as vertical lines
    regrid
        TYPE: string
        DESCRIPTION: whether a '2x2' or '3x3' grid should be used, changing the array size. None is for '1x1'.
    check_plat
        TYPE: 2d array of binary
        DESCRIPTION: array to display if 'good' or 'bad' should be displayed for the plateau line difference test.
    check_curve
        TYPE: 2d array of binary
        DESCRIPTION: array to display if 'good' or 'bad' should be displayed for the plateau curvature test.
    selection_array
        TYPE: 2d array of binary
        DESCRIPTION: the array to draw random indices from, needs to be the same size as input cubes

    '''
    
    #loading in regions for labelling
    wavelengths1a, image_data_1a, error_data_1a = loading_function('data/ngc6302_ch1-short_s3d.fits', 1)
    
    region_indicator = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'data/ch1Arectangle.reg', do_sigma_clip=True, use_dq=False)
    
    action_zone = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_action_zone.reg', do_sigma_clip=True, use_dq=False)
    
    baby_action_zone = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_action_zone_baby.reg', do_sigma_clip=True, use_dq=False)
    
    disk_mask_north = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_disk_north.reg', do_sigma_clip=True, use_dq=False)
    
    disk_mask_south = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_disk_south.reg', do_sigma_clip=True, use_dq=False)

    star_mask = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_star.reg', do_sigma_clip=True, use_dq=False)
    
    central_north_blob_mask = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_central_north_blob.reg', do_sigma_clip=True, use_dq=False)

    central_south_blob_mask = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_central_south_blob.reg', do_sigma_clip=True, use_dq=False)
    
    '''
    array_length_x = len(data[0,:,0])
    array_length_y = len(data[0,0,:])
    
    array_length_regrid_x = int((array_length_x-1)/2)
    array_length_regrid_y = int((array_length_y-1)/2)
    
    array_length_regrid2_x = int((array_length_x)/3)
    array_length_regrid2_y = int((array_length_y-1)/3)
    '''
    
    index1 = []
    index2 = []
    
    random.seed(10)
    
    i = 0
    while i < 100:
        #if the edges are sampled, it could be in a region with no data due to my data being a rotated rectangle shape.
        index_i = random.randint(0,len(data[0,:,0])-1)
        index_j = random.randint(0,len(data[0,0,:])-1)
        
        if selection_array is not None:
            if region_indicator[index_i,index_j] == 1 and selection_array[index_i,index_j] == 1:
                #if index_i not in index1 and index_j not in index2:
                i = i+1
                index1.append(index_i)
                index2.append(index_j)
        else:
            if region_indicator[index_i,index_j] == 1 and baby_action_zone[index_i,index_j] == 1:
                #if index_i not in index1 and index_j not in index2:
                    i = i+1
                    index1.append(index_i)
                    index2.append(index_j)

    print('randoms selected')

    #calculating xticks array
    interval = 0.1
    if upper - lower > 1.5:
        interval = 0.2
    if upper - lower > 5.0:
        interval = 0.5
    if upper - lower > 10.0:
        interval = 1.0
        
    xticks_array = np.arange(lower, upper, interval)

    for i in range(len(index1)):
        
        if i%5 == 0:
            print(i)
        
        #title logic
        if disk_mask_north[index1[i], index2[i]] == 1:
            title = ' (disk north)'
        elif disk_mask_south[index1[i], index2[i]] == 1:
            title = ' (disk south)'
        elif central_north_blob_mask[index1[i], index2[i]] == 1:
            title = ' (north central blob)'
        elif central_south_blob_mask[index1[i], index2[i]] == 1:
            title = ' (south central blob)'
        else:
            title = ''
        
        if star_mask[index1[i], index2[i]] == 1:
            star_title =' (star)'
        else:
            star_title = ''
            
        if conttype is not None:
            conttype_title = ' (cont version ' + str(conttype[index1[i], index2[i]]) + ')' 
        else:
            conttype_title = ''
            
        if check_plat is not None:
            if  check_plat[index1[i], index2[i]] == 1:
                plat_check_title = ' plat bad, '
            else:
                plat_check_title = ' plat good, '
        else:
            plat_check_title = ''
            
        if check_curve is not None:
            if  check_curve[index1[i], index2[i]] == 1:
                curve_check_title = ' curve bad, '
            else:
                curve_check_title = ' curve good, '
        else:
            curve_check_title = ''
        
        if regrid == '2x2':
            index1[i] = int(np.round((index1[i]-1)/2))
            index2[i] = int(np.round((index2[i]-1)/2))
            
        if regrid == '3x3':
            index1[i] = int(np.round((index1[i]-0)/3))
            index2[i] = int(np.round((index2[i]-1)/3))
        
        ax = plt.figure(figsize=(16,6)).add_subplot(111)
        plt.title('Number ' + str(i) + ', Index ' + str(index1[i]) + ', ' + str(index2[i]) + '; Data Investigation' +
                  title + star_title + conttype_title + plat_check_title + curve_check_title, fontsize=16)
        plt.plot(wavelengths, data[:,index1[i],index2[i]], label='original')
        
        #optional plots
        if data_no_lines is not None:
            plt.plot(wavelengths, data_no_lines[:,index1[i],index2[i]], alpha=0.5, label='lines removed')
        if continuum is not None:
            plt.plot(wavelengths, continuum[:,index1[i],index2[i]], alpha=0.5, label='continuum')
        if comparison_data_1 is not None and comparison_wave_1 is not None:
            if scale_wave is None or scale_data is None:
                scaling = 1
            else:
                #scale_wave = np.copy(wavelengths)
                #scale_data = np.copy(data)
                #scale_wave_comp = np.copy(comparison_wave_1)
                #scale_data_comp = np.copy(comparison_data_1)
                data_scale = scale_data[np.where(np.round(scale_wave, 2) == comparison_scale_wave)[0][0],index1[i],index2[i]]
                comparison_scale = scale_data_comp[np.where(np.round(scale_wave_comp, 2) == comparison_scale_wave)[0][0]]
                scaling = data_scale/comparison_scale
            if comparison_data_1.ndim == 1:
                plt.plot(comparison_wave_1, scaling*comparison_data_1, alpha=0.5, label='comparison 1, scale = ' + str(scaling))
            elif comparison_data_1.ndim == 3:
                plt.plot(comparison_wave_1, scaling*comparison_data_1[:,index_i,index_j], alpha=0.5, label='comparison 1, scale = ' + str(scaling))
        if comparison_data_2 is not None and comparison_wave_2 is not None:
            if scale_wave is None or scale_data is None:
                scaling = 1
                #scale_wave = np.copy(wavelengths)
                #scale_data = np.copy(data)
                #scale_wave_comp = np.copy(comparison_wave_2)
                #scale_data_comp = np.copy(comparison_data_2)
            else:
                data_scale = scale_data[np.where(np.round(scale_wave, 2) == comparison_scale_wave)[0][0],index1[i],index2[i]]
                comparison_scale = scale_data_comp[np.where(np.round(scale_wave_comp, 2) == comparison_scale_wave)[0][0]]
                scaling = data_scale/comparison_scale
            plt.plot(comparison_wave_2, scaling*comparison_data_2, alpha=0.5, label='comparison 2, scale = ' + str(scaling))
        if cont_points is not None:
            for wave in cont_points:
                plt.plot([wave, wave], [0, 100000], color='black', linestyle='dashed')
        
        #y scaling
        lower_index = np.where(np.round(wavelengths, 2) == lower)[0][0]
        upper_index = np.where(np.round(wavelengths, 2) == upper)[0][0]
        
        if min_ylim is not None:
            ylim_lower = min_ylim
        else:
            ylim_lower=0
            
        plt.ylim((ylim_lower, ylim_scaler*max(data[:, index1[i], index2[i]])))
        
        if data_no_lines is not None:
            plt.ylim((ylim_lower, ylim_scaler*1.5*max(data_no_lines[lower_index:upper_index, index1[i], index2[i]])))
        elif continuum is not None:
            plt.ylim((0.95*min(continuum[lower_index:upper_index, index1[i],index2[i]]), ylim_scaler*1.05*max(continuum[lower_index:upper_index, index1[i], index2[i]])))
        elif comparison_data_1 is not None:
            plt.ylim((ylim_lower*min(scaling*comparison_data_1), ylim_scaler*5*max(scaling*comparison_data_1)))
        plt.xlim((lower, upper))
        ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
        ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
        ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
        ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        plt.xlabel('Wavelength (micron)', fontsize=16)
        plt.ylabel('Flux (MJy/sr)', fontsize=16)
        plt.xticks(xticks_array, fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=11)
        plt.savefig('PDFtime/spectra_checking/temp/'+str(i))
        plt.show()
        plt.close()
    
    #turning saved png images into a pdf
    
    images = [
        Image.open("PDFtime/spectra_checking/temp/" + f)
        for f in ["0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png", 
                  "10.png", "11.png", "12.png", "13.png", "14.png", "15.png", "16.png", "17.png", "18.png", "19.png", 
                  "20.png", "21.png", "22.png", "23.png", "24.png", "25.png", "26.png", "27.png", "28.png", "29.png", 
                  "30.png", "31.png", "32.png", "33.png", "34.png", "35.png", "36.png", "37.png", "38.png", "39.png", 
                  "40.png", "41.png", "42.png", "43.png", "44.png", "45.png", "46.png", "47.png", "48.png", "49.png", 
                  "50.png", "51.png", "52.png", "53.png", "54.png", "55.png", "56.png", "57.png", "58.png", "59.png", 
                  "60.png", "61.png", "62.png", "63.png", "64.png", "65.png", "66.png", "67.png", "68.png", "69.png", 
                  "70.png", "71.png", "72.png", "73.png", "74.png", "75.png", "76.png", "77.png", "78.png", "79.png", 
                  "80.png", "81.png", "82.png", "83.png", "84.png", "85.png", "86.png", "87.png", "88.png", "89.png", 
                  "90.png", "91.png", "92.png", "93.png", "94.png", "95.png", "96.png", "97.png", "98.png", "99.png", ]
    ]
    
    alpha_removed = []
    
    for i in range(len(images)):
        images[i].load()
        background = Image.new("RGB", images[i].size, (255, 255, 255))
        background.paste(images[i], mask=images[i].split()[3]) # 3 is the alpha channel
        alpha_removed.append(background)
    
    alpha_removed[0].save(
        pdf_name, "PDF" ,resolution=1000.0, save_all=True, append_images=alpha_removed[1:]
    )



def spectra_comparison_imager(wavelengths, data, pdf_name, lower, upper, ylim_scaler, comparison_scale_wave, legend, poi_list,
                              hard_code_ylim=None):
    '''
    A function that plots 10 spectra from within a cube using a hard coded seed, for checking that things
    being done to the entire cube are working, such as emission line removal and continuum fitting. Saves these 
    10 figures into a PDF.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array that corresponds to data, with units um
    data
        TYPE: 3d array of floats
        DESCRIPTION: a spectra, indices [k,i,j] where k is spectral index and i,j are position index
    pdf_name
        TYPE: string
        DESCRIPTION: name of the pdf file, the string includes .pdf in it
    comparison_scale_wave
        TYPE: 1d array of floats
        DESCRIPTION: wavelength interval to be used in scaling for comparison, given as the lower and upper index.
    legend
        TYPE: list of strings
        DESCRIPTION: the names to be assigned to each spectra, in order. should be the same length as poi_list.
    poi_list
        TYPE: 2d list
        DESCRIPTION: a list consisting of the x and y indices of points of interest, manually input, itself as a list. 
            i.e. [[x1,y1], [x2,y2], ...]
    '''
    
    points_num = len(poi_list)
    
    index1 = comparison_scale_wave[0]
    index2 = comparison_scale_wave[1]
    
    #calculating xticks array
    interval = 0.1
    if upper - lower > 1.5:
        interval = 0.2
    if upper - lower > 5.0:
        interval = 0.5
    if upper - lower > 10.0:
        interval = 1.0
        
    xticks_array = np.arange(lower, upper, interval)
    
    ax = plt.figure(figsize=(16,6)).add_subplot(111)
    plt.title('Spectra Comparison', fontsize=16)
    #first entry is the reference that others are scaled to
    plt.plot(wavelengths, data[:, poi_list[0][0],poi_list[0][1]], label='Index ' + str(poi_list[0][0]) + ', ' + str(poi_list[0][1]) + ', ' + legend[0])
    plt.plot([lower, upper], [0,0], color='black')
    for i in range(1, points_num):
        #finding scaling
        scale_data = np.copy(data[index1:index2,poi_list[0][0],poi_list[0][1]])
        comp_data = np.copy(data[index1:index2,poi_list[i][0],poi_list[i][1]])
        scaling = np.max(scale_data)/np.max(comp_data)
        
        plt.plot(wavelengths, scaling*data[:,poi_list[i][0],poi_list[i][1]], label='Index ' + str(poi_list[i][0]) + ', ' + str(poi_list[i][1]) + ', ' + legend[i] + ', scale = ' + str(scaling))
        
    #y scaling
    lower_index = np.where(np.round(wavelengths, 2) == lower)[0][0]
    upper_index = np.where(np.round(wavelengths, 2) == upper)[0][0]
    if hard_code_ylim is None:
        plt.ylim((0.95/ylim_scaler*min(data[lower_index:upper_index,poi_list[0][0],poi_list[0][1]]), 1.05*ylim_scaler*max(data[lower_index:upper_index,poi_list[0][0],poi_list[0][1]])))
    else:
        print('hello')
        plt.ylim((0.95/ylim_scaler*min(data[lower_index:upper_index,poi_list[0][0],poi_list[0][1]]), hard_code_ylim))

    plt.xlim((lower, upper))
    ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
    ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
    ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
    ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel('Wavelength (micron)', fontsize=16)
    plt.ylabel('Flux (MJy/sr)', fontsize=16)
    plt.xticks(xticks_array, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=11)
    plt.savefig('PDFtime/' + pdf_name + '.png') 
    plt.show()
    plt.close()
    


def single_feature_imager(intensity, intensity_112, error, feature_wavelength_name, feature_wavelength_save, snr_cutoff, current_reprojection):
    '''
    A function that plots  the intensity of a given PAH feature, alongside its SNR.
    
    Parameters
    ----------
    intensity
        TYPE: 2d array of floats
        DESCRIPTION: intensities of the given PAH feature, in units of W m^-2 sr^-1
        intensity_112
            TYPE: 2d array of floats
            DESCRIPTION: intensities of the 11.2 PAH feature, in units of W m^-2 sr^-1
                It is included for the purpose of making contours of it.
    error
        TYPE: 2d array of floats
        DESCRIPTION: errors corresponding to the intensities of the given PAH feature, in units of W m^-2 sr^-1
    feature_wavelength_name
        TYPE: string
        DESCRIPTION: the wavelength corresponding to the feature, to be used in plot titles.
    feature_wavelength_save
        TYPE: string
        DESCRIPTION: the wavelength corresponding to the feature, in a specific number format, to be used in file saving. 
            Examples: 6.2 feature is 062, 11.2 feature is 112, 5.25 feature is 052, etc.
    snr_cutoff
        TYPE:  float
        DESCRIPTION: the threshold SNR used for setting vmin and vmax in the intensity plot.
        TYPE: string
        DESCRIPTION: which miri channel the reprojection was performed for. Can be 1A or 3C.
    '''
    
    #calculating values to serve as vmax and vmin, based off of an input SNR cutoff value
    max_value, min_value = colormap_values(intensity, error, snr_cutoff)
   
    #intensity plot
    ax = plt.figure(figsize=(10,10)).add_subplot(111)
    plt.title(feature_wavelength_name + ' Feature')
    plt.imshow(intensity, vmin=min_value, vmax=max_value)
    plt.colorbar() 
    
    #adding contours. 11.2 is included always.
    ax.contour(intensity_112, 3, colors='red') #first one is just 0 put the code is being weird
    
    if feature_wavelength_save != '112':
        ax.contour(intensity, 5, colors='black') 
    
    #adding a data boundary box, dependent on which reprojection is used
    if current_reprojection == '1A':
        #data border
        plt.plot([4, 138, 160, 26, 4], [117, 144, 33, 6, 117], color='yellow')
        #central star
        plt.scatter(75, 80, s=180, facecolors='none', edgecolors='purple')
        
    elif current_reprojection == '3C':
        #data border
        plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='yellow')
        #disk
        plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
        #central star
        plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
        #action zone
        plt.plot([33, 54, 65, 76, 76, 70.5, 47, 33, 30.5, 33], [85, 89, 82, 64, 29, 17, 13, 35, 65, 85], color='C9')

    ax.invert_yaxis()
    plt.savefig('PDFtime/single_images/' + feature_wavelength_save + '_intensity.png')
    plt.show()
    plt.close()

    #SNR plot
    ax = plt.figure(figsize=(10,10)).add_subplot(111)
    plt.title(feature_wavelength_name + ' SNR')
    plt.imshow(intensity/error, vmin=0)
    plt.colorbar() 
    
    #adding a data boundary box, dependent on which reprojection is used
    if current_reprojection == '1A':
        #data border
        plt.plot([4, 138, 160, 26, 4], [117, 144, 33, 6, 117], color='yellow')
        #central star
        plt.scatter(75, 80, s=180, facecolors='none', edgecolors='purple')
        
    elif current_reprojection == '3C':
        #data border
        plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='yellow')
        #disk
        plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
        #central star
        plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
        #action zone
        plt.plot([33, 54, 65, 76, 76, 70.5, 47, 33, 30.5, 33], [85, 89, 82, 64, 29, 17, 13, 35, 65, 85], color='C9')
        
    ax.invert_yaxis()
    plt.savefig('PDFtime/single_images/' + feature_wavelength_save + '_SNR.png')
    plt.show()
    plt.close()
    
    #turning saved png images into a pdf
    images = [
        Image.open("PDFtime/single_images/" + f)
        for f in [feature_wavelength_save + '_intensity.png', feature_wavelength_save + '_SNR.png']
    ]
    
    alpha_removed = []
    
    for i in range(len(images)):
        images[i].load()
        background = Image.new("RGB", images[i].size, (255, 255, 255))
        background.paste(images[i], mask=images[i].split()[3]) # 3 is the alpha channel
        alpha_removed.append(background)
    
    alpha_removed[0].save(
        'PDFtime/' + feature_wavelength_save + '.pdf', "PDF" ,resolution=1000.0, save_all=True, append_images=alpha_removed[1:]
    )    



def feature_ratio_imager(intensity_num, intensity_den, intensity_112, error_num, error_den, 
                         feature_name_num, feature_name_den, feature_wavelength_save, snr_cutoff_num, snr_cutoff_den, current_reprojection):
    '''
    A function that plots  the intensity ratio of two given PAH features, alongside the ratio's error and a correlation plot.
    
    Parameters
    ----------
    intensity_num
        TYPE: 2d array of floats
        DESCRIPTION: intensities of the numerator PAH feature, in units of W m^-2 sr^-1
    intensity_den
        TYPE: 2d array of floats
        DESCRIPTION: intensities of the denominator PAH feature, in units of W m^-2 sr^-1
        intensity_112
            TYPE: 2d array of floats
            DESCRIPTION: intensities of the 11.2 PAH feature, in units of W m^-2 sr^-1
                It is included for the purpose of making contours of it.
    error_num
        TYPE: 2d array of floats
        DESCRIPTION: errors corresponding to the intensities of the given PAH feature, in units of W m^-2 sr^-1
    error_den
        TYPE: 2d array of floats
        DESCRIPTION: errors corresponding to the intensities of the given PAH feature, in units of W m^-2 sr^-1
    feature_name_num
        TYPE: string
        DESCRIPTION: the wavelength corresponding to the numerator feature, to be used in plot titles.
    feature_name_den
        TYPE: string
        DESCRIPTION: the wavelength corresponding to the denominator feature, to be used in plot titles.
    feature_wavelength_save
        TYPE: string
        DESCRIPTION: the wavelength corresponding to the feature, in a specific number format, to be used in file saving. 
            Examples: 6.2 feature is 062, 11.2 feature is 112, 5.25 feature is 052, etc.
    snr_cutoff_num
        TYPE:  float
        DESCRIPTION: the numerator intensity threshold SNR used for setting vmin and vmax in the intensity plot.
    snr_cutoff_den
        TYPE:  float
        DESCRIPTION: the denominator threshold SNR used for setting vmin and vmax in the intensity plot.
    current_reprojection
        TYPE: string
        DESCRIPTION: which miri channel the reprojection was performed for. Can be 1A or 3C.
    '''
    
    #making ratios
    
    intensity_ratio = intensity_num/intensity_den

    intensity_ratio_error = (intensity_num/intensity_den)*np.sqrt(
        (error_num/intensity_num)**2 + (error_den/intensity_den)**2)
    
    #use the min cutoff between the two provided cutoffs
    snr_cutoff = min([snr_cutoff_num, snr_cutoff_den])
    
    max_value, min_value = colormap_values(intensity_ratio, intensity_ratio_error, snr_cutoff)
    
    max_value_error, min_value_error = colormap_values_for_error(intensity_ratio, intensity_ratio_error, snr_cutoff)
    
    #intensity ratio plot
    ax = plt.figure(figsize=(10,10)).add_subplot(111)
    plt.title(feature_name_num + '/' + feature_name_den + ' Feature Ratio')
    plt.imshow(intensity_ratio, vmin=min_value, vmax=max_value)
    plt.colorbar() 
    
    #adding contours. 11.2 is included always.
    ax.contour(intensity_112, 3, colors='red') #first one is just 0 put the code is being weird
    
    #adding a data boundary box, dependent on which reprojection is used
    if current_reprojection == '1A':
        #data border
        plt.plot([4, 138, 160, 26, 4], [117, 144, 33, 6, 117], color='yellow')
        #central star
        plt.scatter(75, 80, s=180, facecolors='none', edgecolors='purple')
        
    elif current_reprojection == '3C':
        #data border
        plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='yellow')
        #disk
        plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
        #central star
        plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
        #action zone
        plt.plot([33, 54, 65, 76, 76, 70.5, 47, 33, 30.5, 33], [85, 89, 82, 64, 29, 17, 13, 35, 65, 85], color='C9')

    ax.invert_yaxis()
    plt.savefig('PDFtime/single_images/' + feature_wavelength_save + '_intensity_ratio.png')
    plt.show()
    plt.close()
    
    #ratio error plot
    ax = plt.figure(figsize=(10,10)).add_subplot(111)
    plt.title(feature_name_num + '/' + feature_name_den + ' Ratio Error')
    plt.imshow(intensity_ratio_error, vmin=min_value_error, vmax=max_value_error)
    plt.colorbar() 
    
    #adding a data boundary box, dependent on which reprojection is used
    if current_reprojection == '1A':
        #data border
        plt.plot([4, 138, 160, 26, 4], [117, 144, 33, 6, 117], color='yellow')
        #central star
        plt.scatter(75, 80, s=180, facecolors='none', edgecolors='purple')
        
    elif current_reprojection == '3C':
        #data border
        plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='yellow')
        #disk
        plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
        #central star
        plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
        #action zone
        plt.plot([33, 54, 65, 76, 76, 70.5, 47, 33, 30.5, 33], [85, 89, 82, 64, 29, 17, 13, 35, 65, 85], color='C9')
        
    ax.invert_yaxis()
    plt.savefig('PDFtime/single_images/' + feature_wavelength_save + '_ratio_error.png')
    plt.show()
    plt.close()
    
    #note the colormap_values_for_comparison function is no longer needed, now that
    #the correlation plots are colour coded.
    
    #determining which values to use in the correlation plot based off of snr cutoff 
    values_to_include_num = np.copy(intensity_num) #colormap_values_for_comparison(intensity_num, error_num, snr_cutoff_num)
    values_to_include_den = np.copy(intensity_den) #colormap_values_for_comparison(intensity_den, error_den, snr_cutoff_den)

    #calculating values to serve as vmax and vmin, based off of an input SNR cutoff value
    max_value_num, min_value_num = colormap_values(intensity_num, error_num, snr_cutoff_num)
    max_value_den, min_value_den = colormap_values(intensity_den, error_den, snr_cutoff_den)
    
    #creating an array that indicates where the disk is, to compare data inside and outside the disk.
    values_to_include_num_disk = np.zeros(values_to_include_num.shape)
    values_to_include_den_disk = np.zeros(values_to_include_den.shape)
    
    disk_mask = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', intensity_112, 'butterfly_disk.reg', do_sigma_clip=True, use_dq=False)
    
    for i in range(len(disk_mask[:,0])):
        for j in range(len(disk_mask[0])):
            if disk_mask[i,j] == 1:
                values_to_include_num_disk[i,j] = values_to_include_num[i,j]
                values_to_include_den_disk[i,j] = values_to_include_den[i,j]
                
    #creating an array that indicates where the central star is, to compare data inside and outside the central star.
    values_to_include_num_star = np.zeros(values_to_include_num.shape)
    values_to_include_den_star = np.zeros(values_to_include_den.shape)
    
    star_mask = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', intensity_112, 'butterfly_star.reg', do_sigma_clip=True, use_dq=False)
    
    for i in range(len(star_mask[:,0])):
        for j in range(len(star_mask[0])):
            if star_mask[i,j] == 1:
                values_to_include_num_star[i,j] = values_to_include_num[i,j]
                values_to_include_den_star[i,j] = values_to_include_den[i,j]
                
    #creating an array that indicates where the action zone is, to compare data inside and outside the edges of the data.
    values_to_include_num_action_zone = np.zeros(values_to_include_num.shape)
    values_to_include_den_action_zone = np.zeros(values_to_include_den.shape)
    
    action_zone_mask = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', intensity_112, 'butterfly_action_zone.reg', do_sigma_clip=True, use_dq=False)
    
    for i in range(len(action_zone_mask[:,0])):
        for j in range(len(action_zone_mask[0])):
            if action_zone_mask[i,j] == 1:
                values_to_include_num_action_zone[i,j] = values_to_include_num[i,j]
                values_to_include_den_action_zone[i,j] = values_to_include_den[i,j]
    
    #correlation plot
    ax = plt.figure(figsize=(16,6)).add_subplot(111)
    plt.title(feature_name_num + ' vs ' + feature_name_den + ' Feature Comparison')
    
    #everything, only outskirts dont get overplotted though
    plt.scatter(values_to_include_den, values_to_include_num, color='orange')
    #action zone
    plt.scatter(values_to_include_den_action_zone, values_to_include_num_action_zone, color='C9')
    #disk
    plt.scatter(values_to_include_den_disk, values_to_include_num_disk, color='green')
    #star
    plt.scatter(values_to_include_den_star, values_to_include_num_star, color='purple')

    
    #bootleg way of getting a legend for my scatterplot
    plt.plot([100], [100], color='orange', label='Outskirts')
    plt.plot([100], [100], color='C9', label='Action Zone')
    plt.plot([100], [100], color='green', label='Disk')
    plt.plot([100], [100], color='purple', label='Central Star')
    plt.legend(fontsize=11)
    
    plt.ylim((min_value_num, max_value_num))
    plt.xlim((min_value_den, max_value_den))
    ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
    ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
    ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
    ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel(feature_name_den + ' Feature Intensity (SI units)', fontsize=16)
    plt.ylabel(feature_name_num + '  Feature Intensity (SI units)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('PDFtime/single_images/' + feature_wavelength_save + '_correlation.png')
    plt.show()
    plt.close()
    
    #turning saved png images into a pdf
    images = [
        Image.open("PDFtime/single_images/" + f)
        for f in [feature_wavelength_save + '_intensity_ratio.png', feature_wavelength_save + '_ratio_error.png', feature_wavelength_save + '_correlation.png']
    ]
    
    alpha_removed = []
    
    for i in range(len(images)):
        images[i].load()
        background = Image.new("RGB", images[i].size, (255, 255, 255))
        background.paste(images[i], mask=images[i].split()[3]) # 3 is the alpha channel
        alpha_removed.append(background)
    
    alpha_removed[0].save(
        'PDFtime/' + 'comparison_' + feature_wavelength_save + '.pdf', "PDF" ,resolution=100.0, save_all=True, append_images=alpha_removed[1:]
    )    
    
    
    
def feature_ratio_imager_normalized(intensity_num, intensity_den, intensity_112, error_num, error_den, 
                         feature_name_num, feature_name_den, feature_wavelength_save, snr_cutoff_num, snr_cutoff_den, current_reprojection):
    '''
    A function that plots the intensity ratio of two given PAH features, alongside the ratio's error and a correlation plot.
    Normalized to the 11.2 PAH feature.
    
    Parameters
    ----------
    intensity_num
        TYPE: 2d array of floats
        DESCRIPTION: intensities of the numerator PAH feature, in units of W m^-2 sr^-1
    intensity_den
        TYPE: 2d array of floats
        DESCRIPTION: intensities of the denominator PAH feature, in units of W m^-2 sr^-1
        intensity_112
            TYPE: 2d array of floats
            DESCRIPTION: intensities of the 11.2 PAH feature, in units of W m^-2 sr^-1
                It is included for the purpose of making contours of it.
    error_num
        TYPE: 2d array of floats
        DESCRIPTION: errors corresponding to the intensities of the given PAH feature, in units of W m^-2 sr^-1
    error_den
        TYPE: 2d array of floats
        DESCRIPTION: errors corresponding to the intensities of the given PAH feature, in units of W m^-2 sr^-1
    feature_name_num
        TYPE: string
        DESCRIPTION: the wavelength corresponding to the numerator feature, to be used in plot titles.
    feature_name_den
        TYPE: string
        DESCRIPTION: the wavelength corresponding to the denominator feature, to be used in plot titles.
    feature_wavelength_save
        TYPE: string
        DESCRIPTION: the wavelength corresponding to the feature, in a specific number format, to be used in file saving. 
            Examples: 6.2 feature is 062, 11.2 feature is 112, 5.25 feature is 052, etc.
    snr_cutoff_num
        TYPE:  float
        DESCRIPTION: the numerator intensity threshold SNR used for setting vmin and vmax in the intensity plot.
    snr_cutoff_den
        TYPE:  float
        DESCRIPTION: the denominator threshold SNR used for setting vmin and vmax in the intensity plot.
    current_reprojection
        TYPE: string
        DESCRIPTION: which miri channel the reprojection was performed for. Can be 1A or 3C.
    '''
    
    #making ratios
    
    intensity_ratio = intensity_num/intensity_den

    intensity_ratio_error = (intensity_num/intensity_den)*np.sqrt(
        (error_num/intensity_num)**2 + (error_den/intensity_den)**2)
    
    #use the min cutoff between the two provided cutoffs
    snr_cutoff = min([snr_cutoff_num, snr_cutoff_den])
    
    max_value, min_value = colormap_values(intensity_ratio, intensity_ratio_error, snr_cutoff)
    
    max_value_error, min_value_error = colormap_values_for_error(intensity_ratio, intensity_ratio_error, snr_cutoff)
    
    #intensity ratio plot
    ax = plt.figure(figsize=(10,10)).add_subplot(111)
    plt.title(feature_name_num + '/' + feature_name_den + ' Feature Ratio')
    plt.imshow(intensity_ratio, vmin=min_value, vmax=max_value)
    plt.colorbar() 
    
    #adding contours. 11.2 is included always.
    ax.contour(intensity_112, 3, colors='red') #first one is just 0 put the code is being weird
    
    #adding a data boundary box, dependent on which reprojection is used
    if current_reprojection == '1A':
        #data border
        plt.plot([4, 138, 160, 26, 4], [117, 144, 33, 6, 117], color='yellow')
        #central star
        plt.scatter(75, 80, s=180, facecolors='none', edgecolors='purple')
        
    elif current_reprojection == '3C':
        #data border
        plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='yellow')
        #disk
        plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
        #central star
        plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
        #action zone
        plt.plot([33, 54, 65, 76, 76, 70.5, 47, 33, 30.5, 33], [85, 89, 82, 64, 29, 17, 13, 35, 65, 85], color='C9')

    ax.invert_yaxis()
    plt.savefig('PDFtime/single_images/' + feature_wavelength_save + '_intensity_ratio.png')
    plt.show()
    plt.close()
    
    #ratio error plot
    ax = plt.figure(figsize=(10,10)).add_subplot(111)
    plt.title(feature_name_num + '/' + feature_name_den + ' Ratio Error')
    plt.imshow(intensity_ratio_error, vmin=min_value_error, vmax=max_value_error)
    plt.colorbar() 
    
    #adding a data boundary box, dependent on which reprojection is used
    if current_reprojection == '1A':
        #data border
        plt.plot([4, 138, 160, 26, 4], [117, 144, 33, 6, 117], color='yellow')
        #central star
        plt.scatter(75, 80, s=180, facecolors='none', edgecolors='purple')
        
    elif current_reprojection == '3C':
        #data border
        plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='yellow')
        #disk
        plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
        #central star
        plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
        #action zone
        plt.plot([33, 54, 65, 76, 76, 70.5, 47, 33, 30.5, 33], [85, 89, 82, 64, 29, 17, 13, 35, 65, 85], color='C9')
        
    ax.invert_yaxis()
    plt.savefig('PDFtime/single_images/' + feature_wavelength_save + '_ratio_error.png')
    plt.show()
    plt.close()
    
    #note the colormap_values_for_comparison function is no longer needed, now that
    #the correlation plots are colour coded.
    
    #determining which values to use in the correlation plot based off of snr cutoff
    values_to_include_num = np.copy(intensity_num) #colormap_values_for_comparison(intensity_num, error_num, snr_cutoff_num)
    values_to_include_den = np.copy(intensity_den) #colormap_values_for_comparison(intensity_den, error_den, snr_cutoff_den)

    #calculating values to serve as vmax and vmin, based off of an input SNR cutoff value
    max_value_num, min_value_num = colormap_values_normalized(intensity_num, error_num, intensity_112, snr_cutoff_num)
    max_value_den, min_value_den = colormap_values_normalized(intensity_den, error_den, intensity_112, snr_cutoff_den)
    
    #creating an array that indicates where the disk is, to compare data inside and outside the disk.
    values_to_include_num_disk = np.zeros(values_to_include_num.shape)
    values_to_include_den_disk = np.zeros(values_to_include_den.shape)
    
    disk_mask = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', intensity_112, 'butterfly_disk.reg', do_sigma_clip=True, use_dq=False)
    
    for i in range(len(disk_mask[:,0])):
        for j in range(len(disk_mask[0])):
            if disk_mask[i,j] == 1:
                values_to_include_num_disk[i,j] = values_to_include_num[i,j]
                values_to_include_den_disk[i,j] = values_to_include_den[i,j]

    #creating an array that indicates where the central star is, to compare data inside and outside the central star.
    values_to_include_num_star = np.zeros(values_to_include_num.shape)
    values_to_include_den_star = np.zeros(values_to_include_den.shape)
    
    star_mask = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', intensity_112, 'butterfly_star.reg', do_sigma_clip=True, use_dq=False)
    
    for i in range(len(star_mask[:,0])):
        for j in range(len(star_mask[0])):
            if star_mask[i,j] == 1:
                values_to_include_num_star[i,j] = values_to_include_num[i,j]
                values_to_include_den_star[i,j] = values_to_include_den[i,j]        
                
    #creating an array that indicates where the action zone is, to compare data inside and outside the edges of the data.
    values_to_include_num_action_zone = np.zeros(values_to_include_num.shape)
    values_to_include_den_action_zone = np.zeros(values_to_include_den.shape)
    
    action_zone_mask = extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', intensity_112, 'butterfly_action_zone.reg', do_sigma_clip=True, use_dq=False)
    
    for i in range(len(action_zone_mask[:,0])):
        for j in range(len(action_zone_mask[0])):
            if action_zone_mask[i,j] == 1:
                values_to_include_num_action_zone[i,j] = values_to_include_num[i,j]
                values_to_include_den_action_zone[i,j] = values_to_include_den[i,j]
    
    #correlation plot
    ax = plt.figure(figsize=(16,6)).add_subplot(111)
    plt.title(feature_name_num + ' vs ' + feature_name_den + ' Feature Comparison, Normalized by 11.2 Feature')
    
    #everything, only the outskirts dont get overplotted though
    plt.scatter(values_to_include_den/intensity_112, values_to_include_num/intensity_112, color='orange')
    #action zone
    plt.scatter(values_to_include_den_action_zone/intensity_112, values_to_include_num_action_zone/intensity_112, color='C9')
    #disk
    plt.scatter(values_to_include_den_disk/intensity_112, values_to_include_num_disk/intensity_112, color='green')
    #star
    plt.scatter(values_to_include_den_star/intensity_112, values_to_include_num_star/intensity_112, color='purple')

    
    #bootleg way of getting a legend for my scatterplot
    plt.plot([100], [100], color='orange', label='Outskirts')
    plt.plot([100], [100], color='green', label='Disk')
    plt.plot([100], [100], color='purple', label='Central Star')
    plt.plot([100], [100], color='C9', label='Action Zone')
    plt.legend(fontsize=11)
    
    plt.ylim((min_value_num, max_value_num))
    plt.xlim((min_value_den, max_value_den))
    ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
    ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
    ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
    ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel(feature_name_den + ' Feature Intensity (SI units)', fontsize=16)
    plt.ylabel(feature_name_num + '  Feature Intensity (SI units)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('PDFtime/single_images/' + feature_wavelength_save + '_correlation.png')
    plt.show()
    plt.close()
    
    #turning saved png images into a pdf
    images = [
        Image.open("PDFtime/single_images/" + f)
        for f in [feature_wavelength_save + '_intensity_ratio.png', feature_wavelength_save + '_ratio_error.png', feature_wavelength_save + '_correlation.png']
    ]
    
    alpha_removed = []
    
    for i in range(len(images)):
        images[i].load()
        background = Image.new("RGB", images[i].size, (255, 255, 255))
        background.paste(images[i], mask=images[i].split()[3]) # 3 is the alpha channel
        alpha_removed.append(background)
    
    alpha_removed[0].save(
        'PDFtime/' + 'comparison_' + feature_wavelength_save + '.pdf', "PDF" ,resolution=1000.0, save_all=True, append_images=alpha_removed[1:]
    )    



def feature_centroid(wavelengths, image_data, wavelengths_plot, image_data_plot, pah_intensity, pah_intensity_error, region_indicator, 
                     feature_name, feature_wavelength_save, snr_cutoff, current_reprojection, comparison_scale_wave, poi_list):
    '''
    A function that calculates the centroid and centroid error, from both the right and left, and returns the mean of them (they should be the same, but this will account for 
    some calculation errors in the integral). It then plots the data.
    
    Parameters
    ----------
    wavelengths
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns.
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, units MJy sr^-1.
            for [k,i,j] k is wavelength index, i and j are position index
            This will get integrated, so the input should be line removed and continuum subtracted.
    wavelengths_plot
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, meant for plotting (may include wavelengths the centroid was not calculated over).
    image_data_plot
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, units MJy sr^-1.
            for [k,i,j] k is wavelength index, i and j are position index
            This will get plotted, it is included to make centroid spectra plots with more wavelengths present.
    pah_intensity
        TYPE: 2d array of floats
        DESCRIPTION: intensities of the PAH feature, in units of W m^-2 sr^-1
    pah_intensity_error
        TYPE: 2d array of floats
        DESCRIPTION: the errors of the PAH feature, in units of W m^-2 sr^-1
    region_indicator
        TYPE: 2d array of floats
        DESCRIPTION: a masking array of 1 and 0, indicating an FOV where data is present across all channels.
    feature_name
        TYPE: string
        DESCRIPTION: the wavelength corresponding to the PAH feature, to be used in plot titles.
    feature_wavelength_save
        TYPE: string
        DESCRIPTION: the wavelength corresponding to the feature, in a specific number format, to be used in file saving. 
            Examples: 6.2 feature is 062, 11.2 feature is 112, 5.25 feature is 052, etc.
    snr_cutoff
        TYPE:  float
        DESCRIPTION: the intensity threshold 'SNR' used for setting vmin and vmax in the centroid plot. 
    current_reprojection
        TYPE: string
        DESCRIPTION: which miri channel the reprojection was performed for. Can be 1A or 3C.
    comparison_scale_wave
        TYPE: 1d array of floats
        DESCRIPTION: wavelength to be used in scaling for comparison
    poi_list
        TYPE: 2d list
        DESCRIPTION: a list consisting of the x and y indices of 2 or 3 points of interest, manually input, itself as a list. 
        i.e. [[x1,y1], [x2,y2], ...]
        first point is assigned colour red, 2nd colour blue, 3rd colour green. This ordering is to allow for a redshifted
        and blueshifted point to be compared, with an intuitive colour.
        
    Returns
    -------
    wavelengths_centroid
        TYPE: 2d array of floats
        DESCRIPTION: centroids of a given PAH feature, in units of microns.
        wavelengths_centroid_error
            TYPE: 2d array of floats
            DESCRIPTION: errors of the centroids of a given PAH feature, in units of microns.
    '''

    #calculating centroid
    array_length_x = len(image_data[0,:,0])
    array_length_y = len(image_data[0,0,:])
    
    index_left = np.zeros((array_length_x, array_length_y))
    index_right = np.zeros((array_length_x, array_length_y))
    integral_half_left = np.zeros((array_length_x, array_length_y))
    integral_half_right = np.zeros((array_length_x, array_length_y))
    
    #combined left and right, should be very similar to one another
    index = np.zeros((array_length_x, array_length_y))
    wavelengths_centroid = np.zeros((array_length_x, array_length_y))
    
    
    
    for i in range(array_length_x):
        for j in range(array_length_y):
            if region_indicator[i,j] == 1:
    
                index_left[i,j], index_right[i,j], integral_half_left[i,j], integral_half_right[i,j] =\
                    pah_feature_integrator_centroid(wavelengths, image_data[:,i,j], pah_intensity[i,j])
                
            index[i,j] = int(np.round((index_left[i,j] + index_right[i,j])/2))
            #doesnt seem to convert to int properly, probably because numpy arrays are cringe
            wavelengths_centroid[i,j] = wavelengths[int(index[i,j])]
    
    print(feature_name + ' centroids calculated')



    #centroid error calculation
    
    #new approach: calculate error for lower bound and upper bound, i.e. find wavelength that gives int/2 - error, int/2 + error, these
    #wavelengths are the lower and upper bounds. Then, can average them for a wavelength to plot, which will be
    #centroid_error instead.
    
    index_error_lower_left = np.zeros((array_length_x, array_length_y))
    index_error_lower_right = np.zeros((array_length_x, array_length_y))
    index_error_upper_left = np.zeros((array_length_x, array_length_y))
    index_error_upper_right = np.zeros((array_length_x, array_length_y))
    integral_half_lower_left = np.zeros((array_length_x, array_length_y))
    integral_half_lower_right = np.zeros((array_length_x, array_length_y))
    integral_half_upper_left = np.zeros((array_length_x, array_length_y))
    integral_half_upper_right = np.zeros((array_length_x, array_length_y))
    
    index_lower = np.zeros((array_length_x, array_length_y))
    index_upper = np.zeros((array_length_x, array_length_y))
    wavelengths_centroid_lower = np.zeros((array_length_x, array_length_y))
    wavelengths_centroid_upper = np.zeros((array_length_x, array_length_y))

    
    for i in range(array_length_x):
        for j in range(array_length_y):
            if region_indicator[i,j] == 1:
                #if error is larger than integral there is no feature so the centroid shouldnt be calculated
                if pah_intensity_error[i,j] > pah_intensity[i,j]:
                    wavelengths_centroid_lower[i,j] = 0
                else:
                    index_error_lower_left[i,j], index_error_lower_right[i,j], integral_half_lower_left[i,j], integral_half_lower_right[i,j] =\
                        pah_feature_integrator_centroid(wavelengths, image_data[:,i,j], pah_intensity[i,j] - pah_intensity_error[i,j])
                
                    index_lower[i,j] = int(np.round((index_error_lower_left[i,j] + index_error_lower_right[i,j])/2))
                    #doesnt seem to convert to int properly, probably because numpy arrays are cringe
                    wavelengths_centroid_lower[i,j] = wavelengths[int(index[i,j])]
            
                    #if they are equal it means the error is some value less than an interval h, so set the error
                    #to h as an upper bound, so the new wavelength is the original value minus h
                    if wavelengths_centroid_lower[i,j] == wavelengths_centroid[i,j]:
                        wavelengths_centroid_lower[i,j] -= wavelengths[1] - wavelengths[0]
    
    print(feature_name + ' lower centroids (for error) calculated')



    for i in range(array_length_x):
        for j in range(array_length_y):
            if region_indicator[i,j] == 1:
                #if error is larger than integral there is no feature so the centroid shouldnt be calculated
                if pah_intensity_error[i,j] > pah_intensity[i,j]:
                    wavelengths_centroid_upper[i,j] = 0
                else:
    
                    index_error_upper_left[i,j], index_error_upper_right[i,j], integral_half_upper_left[i,j], integral_half_upper_right[i,j] =\
                        pah_feature_integrator_centroid(wavelengths, image_data[:,i,j], pah_intensity[i,j] + pah_intensity_error[i,j])
                
                    index_upper[i,j] = int(np.round((index_error_upper_left[i,j] + index_error_upper_right[i,j])/2))
                    #doesnt seem to convert to int properly, probably because numpy arrays are cringe
                    wavelengths_centroid_upper[i,j] = wavelengths[ int(index[i,j])]
            
                    #if they are equal it means the error is some value less than an interval h, so set the error
                    #to h as an upper bound, so the new wavelength is the original value plus h
                    if wavelengths_centroid_upper[i,j] == wavelengths_centroid[i,j]:
                        wavelengths_centroid_upper[i,j] += wavelengths[1] - wavelengths[0]
    
    print(feature_name + ' upper centroids (for error) calculated')

    
    wavelengths_centroid_error = np.zeros((array_length_x, array_length_y))
    for i in range(array_length_x):
        for j in range(array_length_y):
            if wavelengths_centroid_upper[i,j] != 0 and wavelengths_centroid_lower[i,j] != 0:
                wavelengths_centroid_error[i,j] = (wavelengths_centroid_upper[i,j] - wavelengths_centroid_lower[i,j])/2



    #using inverse of error so big error = small value and the code can ignore this as usual
    
    good_data = np.zeros((array_length_x, array_length_y)) - 100 #all empty values will be -100
        
    for i in range(array_length_x):
        for j in range(array_length_y):
            if pah_intensity[i,j]/pah_intensity_error[i,j] > snr_cutoff: #pixels where centroid is plotted the same as those where intensity is plotted
                good_data[i,j] = wavelengths_centroid[i,j]
    
    #note for future me: use np.unravel_index(good_data.argmax, good_data.shape) to return a tuple of the max indices
    
    max_value = np.max(good_data)
    
    good_data = np.abs(good_data) #now all empty values will be 100; since all wavelengths are less than 30, will allow for min and max to be extracted.
        
    min_value = np.min(good_data)
        
    max_contour = np.copy(max_value)
    min_contour = np.copy(min_value)

    #braking up poi_list
    red = poi_list[0]
    blue = poi_list[1]
    if len(poi_list) == 3:
        green = poi_list[2]
        
    #note that indices are defined as data[:,x,y], and so to be displayed in imshow
    #they need to be presented as y,x
        
    #centroid plot
    ax = plt.figure(figsize=(10,10)).add_subplot(111)
    plt.title(feature_name + ' Centroids')
    plt.imshow(wavelengths_centroid, vmin=min_contour, vmax=max_contour)
    plt.colorbar() 
    
    if current_reprojection == '1A':
        #data border
        plt.plot([4, 138, 160, 26, 4], [117, 144, 33, 6, 117], color='yellow')
        #central star
        plt.scatter(75, 80, s=180, facecolors='none', edgecolors='purple')
        
    elif current_reprojection == '3C':
        #data border
        plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='yellow')
        #disk
        plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
        #central star
        plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
        #action zone
        plt.plot([33, 54, 65, 76, 76, 70.5, 47, 33, 30.5, 33], [85, 89, 82, 64, 29, 17, 13, 35, 65, 85], color='C9')

    plt.scatter(red[1], red[0], s=20, facecolors='red', edgecolors='red')
    plt.scatter(blue[1], blue[0], s=20, facecolors='blue', edgecolors='blue')
    
    if len(poi_list) == 3:
        plt.scatter(green[1], green[0], s=20, facecolors='green', edgecolors='green')
    
    ax.invert_yaxis()
    
    plt.savefig('PDFtime/centroids/'  + feature_wavelength_save +'_centroid.png')
    plt.show()
    #plt.close()
    

    #centroid error plot
    ax = plt.figure(figsize=(10,10)).add_subplot(111)
    plt.title(feature_name + ' Centroid Error, wavelength interval = ' + str(np.round(wavelengths[1] - wavelengths[0], 4)))
    plt.imshow(wavelengths_centroid_error, vmin=0, vmax=3*np.round(wavelengths[1] - wavelengths[0], 4))
    plt.colorbar() 
    
    if current_reprojection == '1A':
        #data border
        plt.plot([4, 138, 160, 26, 4], [117, 144, 33, 6, 117], color='yellow')
        #central star
        plt.scatter(75, 80, s=180, facecolors='none', edgecolors='purple')
        
    elif current_reprojection == '3C':
        #data border
        plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='yellow')
        #disk
        plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
        #central star
        plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
        #action zone
        plt.plot([33, 54, 65, 76, 76, 70.5, 47, 33, 30.5, 33], [85, 89, 82, 64, 29, 17, 13, 35, 65, 85], color='C9')
        

    ax.invert_yaxis()
    plt.savefig('PDFtime/centroids/'  + feature_wavelength_save +'_centroid_error.png')
    plt.show()
    plt.close()

    #centroid spectra variation plot
    
    index1 = comparison_scale_wave[0]
    index2 = comparison_scale_wave[1]
    
    #finding scaling
    scale_data = np.copy(image_data_plot[index1:index2,blue[0], blue[1]])
    comp_data_red = np.copy(image_data_plot[index1:index2,red[0], red[1]])
    scale_red = np.max(scale_data)/np.max(comp_data_red)
    
    if len(poi_list) == 3:
        comp_data_green = np.copy(image_data_plot[index1:index2,green[0], green[1]])
        scale_green = np.max(scale_data)/np.max(comp_data_green)
    
    lower = np.round(wavelengths_plot[0], 1)
    upper = np.round(wavelengths_plot[-1], 1)
    
    ylim_max = 1.2*np.max(image_data_plot[:, blue[0], blue[1]])
    
    ax = plt.figure(figsize=(16,6)).add_subplot(111)
    plt.title(feature_name + ' Centroid Variation', fontsize=16)
    plt.plot(wavelengths_plot, scale_red*image_data_plot[:, red[0], red[1]], 
             label= str(red[0]) + ', ' + str(red[1]) + ' Spectra, Scale= ' + str(scale_red), color='red')
    plt.plot(wavelengths_plot, image_data_plot[:, blue[0], blue[1]], 
             label= str(blue[0]) + ', ' + str(blue[1]) + ', Scale= 1.0', color='blue')
    if len(poi_list) == 3:
        plt.plot(wavelengths_plot, scale_green*image_data_plot[:, green[0], green[1]], 
                 label= str(green[0]) + ', ' + str(green[1]) + ' Spectra, Scale= ' + str(scale_green), color='green')
    
    plt.plot([wavelengths_centroid[red[0], red[1]], wavelengths_centroid[red[0], red[1]]], [0, 10000], 
             color='red', linestyle='dashed', label= str(red[0]) + ', '  + str(red[1]) + ' Centroid, Wavelength = ' +  str(np.round(wavelengths_centroid[red[0], red[1]], 4)))
    plt.plot([wavelengths_centroid[blue[0], blue[1]], wavelengths_centroid[blue[0], blue[1]]], [0, 10000], 
             color='blue', linestyle='dashed', label=  str(blue[0]) + ', ' + str(blue[1]) + ' Centroid, Wavelength = '  + str(np.round(wavelengths_centroid[blue[0], blue[1]], 4)))
    if len(poi_list) == 3:
        plt.plot([wavelengths_centroid[green[0], green[1]], wavelengths_centroid[green[0], green[1]]], [0, 10000], 
                 color='green', linestyle='dashed', label= str(green[0]) + ', '  + str(green[1]) + ' Centroid, Wavelength = ' +  str(np.round(wavelengths_centroid[green[0], green[1]], 4)))
    plt.plot([lower, upper], [0, 0], color='black')
    plt.xlim((lower, upper))
    plt.ylim((-100, ylim_max))
    ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
    ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
    ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
    ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.xlabel('Wavelength (micron)', fontsize=16)
    plt.ylabel('Flux (MJy/sr)', fontsize=16)
    plt.xticks(np.arange(lower, upper, 0.1), fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=11)
    plt.savefig('PDFtime/centroids/'  + feature_wavelength_save +'_centroid_spectra.png')
    plt.show()
    plt.close()
    

    
    images = [
        Image.open("PDFtime/centroids/" + feature_wavelength_save + f)
        for f in ["_centroid.png", "_centroid_error.png", "_centroid_spectra.png"]
    ]
    
    pdf_path = "PDFtime/centroid_ " + feature_wavelength_save + ".pdf"
    
    alpha_removed = []
    
    for i in range(len(images)):
        images[i].load()
        background = Image.new("RGB", images[i].size, (255, 255, 255))
        background.paste(images[i], mask=images[i].split()[3]) # 3 is the alpha channel
        alpha_removed.append(background)
    
    alpha_removed[0].save(
        pdf_path, "PDF" ,resolution=1000.0, save_all=True, append_images=alpha_removed[1:]
    )

    return wavelengths_centroid, wavelengths_centroid_error












'''
    lower_index
        TYPE: index
        DESCRIPTION: the index that serves as the lower bound for PAH feature integration.
    upper_index
        TYPE: index
        DESCRIPTION: the index that serves as the upper bound for PAH feature integration.
'''

