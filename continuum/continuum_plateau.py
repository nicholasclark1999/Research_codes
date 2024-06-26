#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:10:29 2023

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

from matplotlib.ticker import  AutoMinorLocator

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)


'''
LOADING DATA
'''

#there are a lot of files, so make a function to sort it out

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


####################################


#%%












# Import necessary packages:

from astropy.io import fits
from astropy.io import ascii

import numpy as np

from scipy.interpolate import UnivariateSpline
import statistics

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from for_loading_data import read_in_fits, get_wavelengths, get_data, get_wave_fits

import PlottingStyle
import copy
import pandas as pd

from math import floor, ceil


class Anchor_points_and_splines():
    """
    A class that finds the anchor ponits and splines for the continua.

    """

    def __init__(self, directory_cube, directory_ipac, array_waves = None, directory_waves = None, function_wave = None, length = None):
        """
        The constructor for Anchor_points_and_splines
        
        Parameters
        ----------
        directory_cube [str]: the directory of the spectral cube data (note: i made this a 3d array)

        directory_ipac [str]: the directory of the ipac file with the anchor info

        array_waves [array-like or NoneType]: the array with the wavelengths

        directory_waves [str or NoneType]: the directory of the file with the wavelengths

        function_wave [function or NoneType]: the function used to read in the wavelengths if a directory is given
                                              (e.g. get_data: for .xdr files - see for_loading_data.py for more
                                               functions)

        length [int or NoneType]: How many consecutive continuum points need to be above the data
                                  to be an issue. If None, no continua overshoot correcting occurs.

        """

        # Check that valid wavelength input was given
        if directory_waves is not None and array_waves is not None:
            raise AttributeError("Can accept either an array of wavelengths or a directory of a file with wavelengths, but not both.")
        elif directory_waves is None and array_waves is None:
            raise AttributeError("Must pass either an array of wavelengths or a directory of a file with wavelengths.")
        else:
            pass

        if array_waves is not None: # Array of wavelengths given
            wavelengths = np.array(array_waves)
        else: # Directory of file with wavelengths given
            if function_wave == None:
                raise AttributeError("If a directory of wavelengths is given, a function used to read in the wavelengths must also be passed.")
            elif function_wave == get_wavelengths:
                wavelengths = function_wave(directory_waves, directory_cube)
            else:
                wavelengths = function_wave(directory_waves)
        self.wavelengths = wavelengths

        self.spectral_cube = directory_cube
        anchor_ipac_data = ascii.read(directory_ipac, format = 'ipac')
        low_to_high_inds = np.array(anchor_ipac_data['x0']).argsort()
        # Because UnivariateSpline needs ordered values
        self.Pl_inds = [i for i, x in enumerate(anchor_ipac_data['on_plateau'][low_to_high_inds]) if x == "True"]
        if len(self.Pl_inds) > 0:
            self.Pl_starting_waves = anchor_ipac_data['x0'][low_to_high_inds][self.Pl_inds]
        self.moments = anchor_ipac_data['moment'][low_to_high_inds]
        self.starting_waves = anchor_ipac_data['x0'][low_to_high_inds]
        self.x0_mins = anchor_ipac_data['x0_min'][low_to_high_inds]
        self.x0_maxes = anchor_ipac_data['x0_max'][low_to_high_inds]
        self.bumps = anchor_ipac_data['bumps'][low_to_high_inds]
        self.starting_waves = anchor_ipac_data['x0'][low_to_high_inds]
        self.length = length

    def starting_anchors_x(self):
        """
        Gets the desired wavelengths for the starting anchor points.

        Returns
        -------
        The anchors points and their indices within wavelengths
        """
        indices = []
        for wave in self.starting_waves:
            indices.append(np.abs(np.asarray(self.wavelengths) - wave).argmin())
        anchors = list(self.wavelengths[indices])
        return(anchors, indices)

    def find_min_brightness(self, wave, wave_min, wave_max, pixel):
        """
        Finds the initial x and y values of an anchor point to use when the user wants the lowest brightness.

        Parameters
        ---------

        wave [float]: a wavelength between wave_min and wave_max

        wave_min, wave_max [float or int]: the minimum and maximum wavelength values to look for a minimum brightness
                                           between

        pixel [lst]: a pixel within the data cube (e.g. [10, 15])

        Returns
        -------

        The minimum brightness value and its corresponding wavelength
        """
        wavelengths = self.wavelengths
        wanted_wavelength_inds = np.where(np.logical_and(wavelengths > wave_min, wavelengths < wave_max))[0]
        brightness_list = list(self.spectral_cube[wanted_wavelength_inds, pixel[1], pixel[0]])
        brightness_anchor = min(brightness_list)
        ind_lowest_brightness = wanted_wavelength_inds[brightness_list.index(min(brightness_list))]
        wavelength_to_change = wavelengths[ind_lowest_brightness]
        return(brightness_anchor, wavelength_to_change)

    def new_anchor_points(self, pixel):
        """
        Finds more accurate lists of x and y values for anchor points (based on the 'moment' column of
        the anchor ipac table)

        Parameters
        ---------
        pixel [lst]: a pixel within the data cube (e.g. [10, 15])

        Returns
        -------
        A list of brightness and wavelength values for anchor points

        """
        cube = self.spectral_cube
        moments = self.moments
        wavelength_anchors_data =  self.starting_anchors_x()
        wavelength_anchors = wavelength_anchors_data[0]
        anchor_indices = wavelength_anchors_data[1]
        # To get desired list size - will change elements of list in code
        brightness_anchors = list(range(len(wavelength_anchors)))
        for ind in range(len(anchor_indices)):
            if moments[ind] == 1: # Find the mean between nearby points
                list_for_mean = [cube[anchor_indices[ind]][pixel[1]][pixel[0]],
                                 cube[anchor_indices[ind]-1][pixel[1]][pixel[0]],
                                 cube[anchor_indices[ind]+1][pixel[1]][pixel[0]]]
                brightness_anchors[ind] = statistics.mean(list_for_mean)
            elif moments[ind] == 2: # Find the mean between more nearby points
                list_for_mean = [cube[anchor_indices[ind]][pixel[1]][pixel[0]],
                                 cube[anchor_indices[ind]-1][pixel[1]][pixel[0]],
                                 cube[anchor_indices[ind]+1][pixel[1]][pixel[0]],
                                 cube[anchor_indices[ind]-2][pixel[1]][pixel[0]],
                                 cube[anchor_indices[ind]+2][pixel[1]][pixel[0]]]
                brightness_anchors[ind] = statistics.mean(list_for_mean)
            elif moments[ind] == 3:
             # Find the average of two anchor point means and places the x location between them
             # Currently, the second anchor point for the average will be 0.3 microns behind the first
             # Find indice of anchor point not given:
                 wavelengths = self.wavelengths
                 wave = wavelength_anchors[ind] + 0.3
                 location = np.abs(np.asarray(wavelengths) - wave).argmin()
                 list_for_mean_1 = [cube[anchor_indices[ind]][pixel[1]][pixel[0]],
                                    cube[anchor_indices[ind]-1][pixel[1]][pixel[0]],
                                    cube[anchor_indices[ind]+1][pixel[1]][pixel[0]]]
                 list_for_mean_2 = [cube[location][pixel[1]][pixel[0]],
                                    cube[location-1][pixel[1]][pixel[0]],
                                    cube[location+1][pixel[1]][pixel[0]]]
                 brightness_anchors[ind] = (statistics.mean(list_for_mean_1) + statistics.mean(list_for_mean_2))/2
                 wavelength_anchors[ind] = (wave + wavelength_anchors[ind])/2
            elif moments[ind] == 4:
                pt_inds = np.where(np.logical_and(self.wavelengths > self.x0_mins[ind],
                                                  self.wavelengths < self.x0_maxes[ind]))[0]
                brightness_anchors[ind] = np.average(cube[pt_inds, pixel[1], pixel[0]])
                wavelength_anchors[ind] = (self.x0_mins[ind] + self.x0_maxes[ind])/2
            elif moments[ind] == 0 or self.bumps[ind] == "True":
                # Find the min brightness within a wavelength region
                wave = wavelength_anchors[ind]
                wave_min = self.x0_mins[ind]
                wave_max = self.x0_maxes[ind]
                brightness_anchor, wavelength_to_change = self.find_min_brightness(wave, wave_min,
                                                                                   wave_max, pixel)
                brightness_anchors[ind] = brightness_anchor
                wavelength_anchors[ind] = wavelength_to_change
        return(brightness_anchors, wavelength_anchors)

    def get_anchors_for_all_splines(self, pixel):
        """
        Creates the GS, LS, and plateaus' anchor points' values (brightnesses and wavelengths)

        Parameters
        ---------
        pixel [lst]: a pixel within the data cube (e.g. [10, 15])

        Returns
        -------
        A new list of brightness and wavelength values for anchor points

        """
        if len(self.Pl_inds) > 0:
            plateau_waves = self.Pl_starting_waves
        anchor_info = self.new_anchor_points(pixel)
        brightness_anchors = anchor_info[0]
        new_wavelength_anchors = anchor_info[1]
        if "True" in self.bumps:
        # We need a continuum that includes the bumps (LS) and another that doesn't (GS)
            LS_brightness = brightness_anchors
            LS_wavelengths = new_wavelength_anchors
            no_bump_inds = [i for i, x in enumerate(self.bumps) if x == "False"]
            GS_brightness = [LS_brightness[ind] for ind in no_bump_inds]
            GS_wavelengths = [LS_wavelengths[ind] for ind in no_bump_inds]
        if len(self.Pl_inds) > 0:
        # We need to add a plateau (PL) continuum
            indices = []
            for anchor in plateau_waves:
                indices.append(np.abs(np.asarray(new_wavelength_anchors) - anchor).argmin())
            PL_wavelengths = [new_wavelength_anchors[ind] for ind in indices]
            PL_brightness = [brightness_anchors[ind] for ind in indices]
        if len(self.Pl_inds) > 0 and "True" in self.bumps:
            return(LS_brightness, LS_wavelengths, GS_brightness, GS_wavelengths, PL_brightness,
                   PL_wavelengths)
        elif len(self.Pl_inds) > 0 and "True" not in self.bumps:
            return(GS_brightness, GS_wavelengths, PL_brightness, PL_wavelengths)
        elif len(self.Pl_inds) == 0 and "True" in self.bumps:
            return(LS_brightness, LS_wavelengths, GS_brightness, GS_wavelengths)
        else:
            return(brightness_anchors, new_wavelength_anchors)

    def lower_brightness(self, anchor_waves, anchor_brightness, Cont, pixel):
        """
        Checks if the continuum is higher than the brightness for self.length consecutive points and
        lowers the continuum if it is
        This function likely needs some refinement

        Parameters
        ----------
        anchor_waves [lst]: a list of the anchor's wavelengths

        anchor_brightnesses [lst]: a list of the anchor's brightnesses

        Cont [UnivariateSpline object]: a continuum

        pixel [lst]: a pixel within the data cube (e.g. [10, 15])

        """
        pixel_brightnesses = self.spectral_cube[:, pixel[1], pixel[0]]
        no_cont_brightnesses = pixel_brightnesses - Cont
        below_zero_inds = np.where(no_cont_brightnesses < 0)[0] # Where data is less than the continuum
        # A few points bellow zero in no_cont_brightnesses could be due to noise, but a large range of
        # consecutive values below zero is a problem with the continuum
        consecutive_diff = np.diff(below_zero_inds)
        consecutive_diff_not_1 = np.where(consecutive_diff != 1)[0]
        split_below_zero_inds = np.split(below_zero_inds, consecutive_diff_not_1 + 1)
        # split_below_zero_inds is a list of arrays. The arrays are split based on where
        # the indicies of the points below zero aren't consecutive
        for array in split_below_zero_inds:
        # May be redundant if two arrays with consecutive < 0 values are between the same 2 anchor points
            if len(array) > self.length:
                subtracted_brightness_lower = np.median(no_cont_brightnesses[array])
                # Find the nearest anchor ind to the start of the problem area
                anchor_ind = np.abs(anchor_waves - self.wavelengths[array[0]]).argmin()
                anchor_brightness[anchor_ind] = anchor_brightness[anchor_ind] + subtracted_brightness_lower
                # Recall that subtracted_brightness_min is less than 0
                # Adding it will lower brightness values
        new_Cont = UnivariateSpline(anchor_waves, anchor_brightness, k = 3, s = 0)(self.wavelengths)
        # s = 0 so anchor points can't move
        # k = 3 for cubic
        return(new_Cont, anchor_brightness)

    def get_splines_with_anchors(self, pixel):
        """
        Creates cubic splines (LS, GS, and plateau).  Note that the plateau spline is also
        made using lines on both ends (with the cubic spline in the middle)

        Parameters
        ---------
        pixel [lst]: a pixel within the data cube (e.g. [10, 15])

        Returns
        -------
        A dictionary of continua, as well as the values of the wavelengths and brightnesses used for the
        anchor points of each continuum

        """
        all_wavelengths = self.wavelengths
        anchor_data = self.get_anchors_for_all_splines(pixel)
        spline_and_anchor_dict = {}
        if "True" in self.bumps:
            spline_and_anchor_dict["LS_wave_anchors"] = anchor_data[1]
            spline_and_anchor_dict["LS_brightness_anchors"] = anchor_data[0]
            spline_and_anchor_dict["GS_wave_anchors"] = anchor_data[3]
            spline_and_anchor_dict["GS_brightness_anchors"] = anchor_data[2]
            ContLS = UnivariateSpline(spline_and_anchor_dict["LS_wave_anchors"],
                                      spline_and_anchor_dict["LS_brightness_anchors"],
                                      k = 3, s = 0)(all_wavelengths)
            if self.length != None:
                new_LS = self.lower_brightness(spline_and_anchor_dict["LS_wave_anchors"],
                                               spline_and_anchor_dict["LS_brightness_anchors"],
                                               ContLS, pixel)
                spline_and_anchor_dict["LS_brightness_anchors"] = new_LS[1]
                spline_and_anchor_dict["ContLS"] = new_LS[0]
            else:
                spline_and_anchor_dict["ContLS"] = ContLS
            if len(self.Pl_inds) > 0:
                spline_and_anchor_dict["PL_wave_anchors"] = anchor_data[5]
                spline_and_anchor_dict["PL_brightness_anchors"] = anchor_data[4]
        else:
            spline_and_anchor_dict["GS_wave_anchors"] = anchor_data[1]
            spline_and_anchor_dict["GS_brightness_anchors"] = anchor_data[0]
            if len(self.Pl_inds) > 0:
                spline_and_anchor_dict["PL_wave_anchors"] = anchor_data[3]
                spline_and_anchor_dict["PL_brightness_anchors"] = anchor_data[2]
        ContGS = UnivariateSpline(spline_and_anchor_dict["GS_wave_anchors"],
                                  spline_and_anchor_dict["GS_brightness_anchors"],
                                  k = 3, s = 0)(all_wavelengths)
        if self.length != None:
            new_GS = self.lower_brightness(spline_and_anchor_dict["GS_wave_anchors"],
                                           spline_and_anchor_dict["GS_brightness_anchors"],
                                           ContGS, pixel)
            spline_and_anchor_dict["GS_brightness_anchors"] = new_GS[1]
            spline_and_anchor_dict["ContGS"] = new_GS[0]
        else:
            spline_and_anchor_dict["ContGS"] = ContGS
        if len(self.Pl_inds) > 0: # Easier to do this here since two cases above have PL conts
            ContPL = copy.deepcopy(spline_and_anchor_dict["ContGS"])
            for plateau in range(int(len(self.Pl_inds)/2)): # Each Pl defined by 2 points
                line = UnivariateSpline([spline_and_anchor_dict["PL_wave_anchors"][0 + 2*plateau],
                                         spline_and_anchor_dict["PL_wave_anchors"][1 + 2*plateau]],
                                        [spline_and_anchor_dict["PL_brightness_anchors"][0 + 2*plateau],
                                         spline_and_anchor_dict["PL_brightness_anchors"][1 + 2*plateau]],
                                        k = 1, s = 0)(all_wavelengths) # k = 1 for a line
                indices_contPL = np.where(np.logical_and(all_wavelengths > spline_and_anchor_dict["PL_wave_anchors"][0 + 2*plateau],
                                          all_wavelengths < spline_and_anchor_dict["PL_wave_anchors"][1 + 2*plateau]))[0]
                ContPL[indices_contPL] = line[indices_contPL]
            spline_and_anchor_dict["ContPL"] = ContPL
        return(spline_and_anchor_dict)

    def fake_get_splines_with_anchors(self):
        """
        A function that returns the same output as get_splines_with_anchors, but everything is
        0. Intended for flagged pixels.

        Returns
        -------
        See get_splines_with_anchors

        """
        spline_and_anchor_dict = {}
        splines = np.zeros(len(self.wavelengths))
        bump_inds = [i for i, x in enumerate(self.bumps) if x == "True"]
        Pl_inds = self.Pl_inds
        spline_and_anchor_dict["ContGS"] = splines
        if "True" in self.bumps:
            spline_and_anchor_dict["LS_wave_anchors"] = np.zeros(len(self.bumps))
            spline_and_anchor_dict["LS_brightness_anchors"] = np.zeros(len(self.bumps))
            spline_and_anchor_dict["ContLS"] = splines
            spline_and_anchor_dict["GS_wave_anchors"] = np.zeros(len(self.bumps) - len(bump_inds))
            spline_and_anchor_dict["GS_brightness_anchors"] = np.zeros(len(self.bumps) - len(bump_inds))
        else:
            spline_and_anchor_dict["GS_wave_anchors"] = np.zeros(len(self.bumps))
            spline_and_anchor_dict["GS_brightness_anchors"] = np.zeros(len(self.bumps))
        if len(Pl_inds) > 0:
            spline_and_anchor_dict["PL_wave_anchors"] = np.zeros(len(Pl_inds))
            spline_and_anchor_dict["PL_brightness_anchors"] = np.zeros(len(Pl_inds))
            spline_and_anchor_dict["ContPL"] = splines
        return(spline_and_anchor_dict)

class Continua():
    """
    A class for plotting and making continua files

    """

    def __init__(self, directory_cube, directory_cube_unc, directory_ipac, array_waves = None, directory_waves = None, function_wave = None,
                 flags = None, length = None):
        """
        The constructor for Continua

        Parameters
        ----------
        directory_cube [str]: the directory of the spectral cube data

        directory_cube_unc [str or NoneType]: the directory of the uncertainty cube (or None if no
                                              uncertainties exist)

        directory_ipac [str]: the directory of the ipac file with the anchor info

        array_waves [array-like or NoneType]: the array with the wavelengths

        directory_waves [str or NoneType]: the directory of the file with the wavelengths

        function_wave [function or NoneType]: the function used to read in the wavelengths if a directory is given
                                              (e.g. get_data: for .xdr files - see for_loading_data.py for more
                                               functions)

        flags [str or NoneType]: the directory of the flagged pixel file

        length [int or NoneType]: How many consecutive continuum points need to be above the data
                                  to be an issue. If None, no continua overshoot correcting occurs.

        """

        # Check that valid wavelength input was given
        if directory_waves is not None and array_waves is not None:
            raise AttributeError("Can accept either an array of wavelengths or a directory of a file with wavelengths, but not both.")
        elif directory_waves is None and array_waves is None:
            raise AttributeError("Must pass either an array of wavelengths or a directory of a file with wavelengths.")
        else:
            pass

        if array_waves is not None: # Array of wavelengths given
            wavelengths = np.array(array_waves)
        else: # Directory of file with wavelengths given
            if function_wave == None:
                raise AttributeError("If a directory of wavelengths is given, a function used to read in the wavelengths must also be passed.")
            elif function_wave == get_wavelengths:
                wavelengths = function_wave(directory_waves, directory_cube)
            else:
                wavelengths = function_wave(directory_waves)
        self.wavelengths = wavelengths

        self.spectral_cube = directory_cube
        if directory_cube_unc != None:
            if directory_cube_unc[-3:-1] == 'xd':
                self.spectral_cube_unc = get_data(directory_cube_unc)
            else:
                self.spectral_cube_unc = fits.getdata(directory_cube_unc)
        else:
            self.spectral_cube_unc = None
        self.anchors_and_splines = Anchor_points_and_splines(directory_cube, directory_ipac, array_waves = array_waves,
                                                                directory_waves = directory_waves, function_wave = function_wave,
                                                                length = length)

        self.moments = self.anchors_and_splines.moments
        self.starting_waves = self.anchors_and_splines.starting_waves
        self.Pl_inds = self.anchors_and_splines.Pl_inds
        self.bumps = self.anchors_and_splines.bumps
        if flags != None:
            self.flags = fits.getdata(flags)
        else:
            self.flags = None

    def plot_pixel_continuum(self, pixel, splines_and_anchors, max_y = None, min_y = None,
                             max_x = None, min_x = None):

        """
        Plots the continua, anchor points, and data points for a single pixel

        Parameters
        ----------
        pixel [lst]: a pixel within the data cube (e.g. [10, 15])

        splines_and_anchors [dict]: the spline and anchor dictionary returned by get_splines_with_anchors

        max_y, min_y, max_x, min_x [int, float, str, NoneType]: the desired plt.lim values for the plots
                                                                max_y = 'median'is the only excepted str
                                                                max_y = 'median'is recommended when high
                                                                spectral resolution results in many features
                                                                atop PAH features (like with JWST data)
        Returns
        -------
        The plot of a pixel's continua, anchor points, and data points

        """
        # Get all values for plotting
        data = self.spectral_cube
        data_unc = self.spectral_cube_unc
        wavelengths = self.wavelengths
        all_brightness_data = data[:, pixel[1], pixel[0]]
        if data_unc!= None:
            all_brightness_data_unc = data_unc[:, pixel[1], pixel[0]]
        # Plot values
        fig, ax = plt.subplots(figsize = (12, 8))
        plt.minorticks_on()
        plt.plot(wavelengths, all_brightness_data, label = "Data", color = "k")
        if len(self.Pl_inds) > 0:
            plt.plot(wavelengths, splines_and_anchors["ContPL"], label = "Pl Continuum", linewidth = 1)
            plt.plot(splines_and_anchors["PL_wave_anchors"], splines_and_anchors["PL_brightness_anchors"],
                     label = "Plateau Points", marker = "d", linestyle='None', markersize = 7,
                     color = '#F994EC')
        plt.plot(wavelengths, splines_and_anchors["ContGS"], label = "GS Continuum")
        plt.plot(splines_and_anchors["GS_wave_anchors"], splines_and_anchors["GS_brightness_anchors"],
                 label = "GS Points", marker = "P", linestyle='None', markersize = 5, color = 'm')
        if data_unc!= None:
            plt.fill_between(wavelengths, all_brightness_data - all_brightness_data_unc,
                             all_brightness_data + all_brightness_data_unc, color = "#AAB7B8")
        if "True" in self.bumps:
            plt.plot(wavelengths, splines_and_anchors["ContLS"], label = "LS Continuum",
                     linestyle = "--", linewidth = 1.5)
            plt.plot(splines_and_anchors["LS_wave_anchors"], splines_and_anchors["LS_brightness_anchors"] ,
                     label = "LS Points", marker = ".", linestyle='None', markersize = 5, color = 'c')

        if max_y != None and max_y != 'median':
            # The user input their own max_y
            if min_y == None:
                plt.ylim(0, max_y)
            else:
                plt.ylim(min_y, max_y)
        else:
            # A max_y will be found
            if max_y == None:
                # Matplotlib picks the max_y
                if min_y != None:
                    plt.ylim(min_y)
            if max_y == 'median':
                # We'll find a max_y by grouping points and finding the groups' median
                num_groups = ceil(len(wavelengths)*0.0004)
                # ceil to avoid getting zero with low number of data points
                remainder = len(wavelengths) % num_groups
                # pad array with zeros at the end so that reshape doesn't complain
                padded = np.pad(all_brightness_data, (0, remainder), 'constant')
                max_y = np.nanmax(np.median(padded.reshape(-1, num_groups), axis = 1))
                if min_y != None:
                    plt.ylim(min_y, max_y)
                else:
                    plt.ylim(0, max_y)

        if min_x == None and max_x != None:
                plt.xlim(np.min(self.wavelengths), max_x)
        elif min_x != None and max_x == None:
            plt.xlim(min_x)
        elif min_x != None and max_x != None:
            plt.xlim(min_x, max_x)

        plt.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
        plt.legend(frameon = False)
        plt.xlabel(r'Wavelength ($\mu$m)')
        plt.ylabel(r'Surface Brightness (MJy/sr)')
        plt.title("Pixel: " + str(pixel))

    def make_fits_file(self, array, save_loc):
        """
        Takes a 3 dimensional np array and turns it into a .fits file.

        Parameters
        ----------
        array [np.array]: a 3D np.array

        save_loc [str]: the directory where the saved data is stored (e.g. r"path/file_name.fits")

        Returns
        -------
        A fits file of a cube

        """
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU())
        hdul.append(fits.ImageHDU(data = array))
        hdul.writeto(save_loc, overwrite=True)

    def make_continua(self, x_min_pix = None, x_max_pix = None, y_min_pix = None, y_max_pix = None,
                      fits = True, plot = False, per_pix = None, save_loc_fit_GS = 'continuum_plateau.fits',
                      save_loc_fit_LS = None, save_loc_fit_PL = None, save_loc_plot = None,
                      max_y_plot = None, min_y_plot = None, max_x_plot = None, min_x_plot = None):
        """
        Creats a PDF of plots of the continua and/or the continua fits files

        Parameters
        ----------
        x_min_pix, x_max_pix, y_min_pix, y_max_pix [int or Nonetype]: the range of the pixels to include
                                                                      if None, the shape of the cube
                                                                      will be used (i.e. mins = 0,
                                                                      y_max_pix = cube.shape[1],
                                                                      x_max_pix = cube.shape[2])

        fits [bool]: True or False - whether or not fits files should be created

        plot [bool]: True or False - whether or not a pdf of continua plots should be created

        per_pix [int or NoneType]: the nth pixels to plot (e.g. per_pix = 6 means that every 6th pixel
                                                           is graphed, provided that the pixel's brightnesses
                                                           aren't all 0. Pixels with all 0 values are skipped)
                                   if None and plot = True, all continua will be graphed

        save_loc_fit_GS, save_loc_fit_LS, save_loc_fit_PL [str or NoneType]: the directories to
                                                                                save the contiua fits
                                                                                to (e.g.
                                                                                r"path/file_name.fits")
                                                                                MUST HAVE AT LEAST
                                                                                save_loc_fits_GS
                                                                                IF fits = True

        save_loc_plot [str or NoneType]: the directory where the continua pdf is saved to
                                         (e.g. r"path/file_name.pdf"). MUST HAVE IF plots = True

        max_y_plot, min_y_plot, max_x_plot, min_x_plot [int, float, str, NoneType]: the desired plt.lim values for
                                                                                    the plots
                                                                                    max_y = 'median'is the only
                                                                                    excepted str
                                                                                    max_y = 'median'is recommended
                                                                                    when high spectral resolution
                                                                                    results in many features
                                                                                    atop PAH features (like with
                                                                                    JWST data)

        Returns
        -------
        A PDF of plots of the continua and/or the continua fits files

        """
        data = self.spectral_cube
        if self.flags != None:
            data = np.multiply(data, self.flags[np.newaxis, :])
        if x_min_pix == None:
            x_min_pix = 0
        if y_min_pix == None:
            y_min_pix = 0
        if x_max_pix == None:
            x_max_pix = data.shape[2]
        if y_max_pix == None:
            y_max_pix = data.shape[1]
        if fits:
            ContGS_cube = np.zeros(data.shape)
            if "True" in self.bumps:
                ContLS_cube = np.zeros(data.shape)
            if len(self.Pl_inds) > 0:
                ContPL_cube = np.zeros(data.shape)
        pix_dict = {}
        for x in range(x_min_pix, x_max_pix):
            for y in range(y_min_pix, y_max_pix):
                pixel = [x, y]
                if not np.all(data[:, pixel[1], pixel[0]] == 0):
                    splines_and_anchors = self.anchors_and_splines.get_splines_with_anchors(pixel)
                else:
                    splines_and_anchors = self.anchors_and_splines.fake_get_splines_with_anchors()
                pix_dict[str(pixel)] = splines_and_anchors

        num_of_pixels = (x_max_pix - x_min_pix)*(y_max_pix - y_min_pix)
        pix_count = 0
        if fits:
            for x in range(x_min_pix, x_max_pix):
                for y in range(y_min_pix, y_max_pix):
                    pix_count = pix_count + 1
                    pixel = [x, y]
                    ContGS_cube[:, y, x] = pix_dict[str(pixel)]["ContGS"]
                    if "True" in self.bumps:
                         ContLS_cube[:, y, x] = pix_dict[str(pixel)]["ContLS"]
                    if len(self.Pl_inds) > 0:
                         ContPL_cube[:, y, x] = pix_dict[str(pixel)]["ContPL"]
                    if pix_count % 100 == 0:
                        print("Continua " + str(pix_count*100/num_of_pixels) + "% completed")
            # Make .fits files
            self.make_fits_file(ContGS_cube, save_loc_fit_GS)
            if "True" in self.bumps:
                self.make_fits_file(ContLS_cube, save_loc_fit_LS)
            if len(self.Pl_inds) > 0:
                self.make_fits_file(ContPL_cube, save_loc_fit_PL)
            pix_count = 0
        if plot:
            plot_count = 0
            if per_pix != None:
                num_of_pixels_to_plot = floor((x_max_pix - x_min_pix)*(y_max_pix - y_min_pix)/per_pix)
            else:
                num_of_pixels_to_plot = (x_max_pix - x_min_pix)*(y_max_pix - y_min_pix)
            with PdfPages(save_loc_plot) as pdf:
                for x in range(x_min_pix, x_max_pix):
                    for y in range(y_min_pix, y_max_pix):
                        pix_count = pix_count + 1
                        if per_pix == None or pix_count % per_pix == 0:
                            pixel = [x, y]
                            anchor_info = pix_dict[str(pixel)]
                            pix_all_brightness_data = self.spectral_cube[:, pixel[1], pixel[0]]
                            if not(np.all(pix_all_brightness_data == 0)):
                                fig = self.plot_pixel_continuum(pixel, anchor_info, max_y_plot,
                                                                min_y_plot, max_x_plot, min_x_plot)
                                pdf.savefig(fig)
                                #plt.close(fig) # To close figure after it's saved/save memory
                                plt.clf()
                                plot_count = plot_count + 1
                                if plot_count % 25 == 0:
                                    print("Plots " + str(plot_count*100/num_of_pixels_to_plot) + "% completed")
                pdf.close()

#%%


wavelengths230cs = np.load('Analysis/wavelengths230cs.npy', allow_pickle=True)
image_data_230cs = np.load('Analysis/image_data_230cs.npy', allow_pickle=True)

good_or_bad_diff = np.load('Analysis/good_or_bad_diff.npy', allow_pickle=True)
good_or_bad_curve = np.load('Analysis/good_or_bad_curve.npy', allow_pickle=True)
good_or_bad = np.load('Analysis/good_or_bad.npy', allow_pickle=True)

image_file = get_pkg_data_filename('continuum_dust.fits')
    
continuum_dust = fits.getdata(image_file, ext=1)

#wavelengths1 = np.load('wavelengths1.npy')

#jw01742-o002_t003_miri_ch1-shortmediumlong_s3d.fits' 
cont = Continua(directory_cube=image_data_230cs-continuum_dust, 
                directory_cube_unc=None, directory_ipac = 'anchors_plateau.ipac',
                 array_waves = wavelengths230cs)

cont.make_continua()

#%%

image_file = get_pkg_data_filename('continuum_plateau.fits')
    
continuum = fits.getdata(image_file, ext=1)

#%%

i = 19
j = 49

plt.figure()
plt.plot(wavelengths230cs, (image_data_230cs-continuum_dust)[:,i,j])
plt.plot(wavelengths230cs, continuum[:,i,j])
plt.xlim(5,23)
plt.ylim(0,5000)
plt.show()

#%%

i = 40
j = 31

plt.figure()
plt.plot(wavelengths230cs, (image_data_230cs-continuum_dust)[:,i,j])
plt.plot(wavelengths230cs, continuum[:,i,j])
plt.xlim(5,23)
plt.ylim(0,5000)
plt.show()

#%%
'''
import ButterflyNebulaFunctions as bnf

wavelengths3b, image_data_3b, error_data_3b = bnf.loading_function('data/ngc6302_ch3-medium_s3d.fits', 1)
wavelengths3c, image_data_3c, error_data_3c = bnf.loading_function('data/ngc6302_ch3-long_s3d.fits', 1)
image_data_3b_noline = np.load('Analysis/image_data_3b_noline.npy', allow_pickle=True)
image_data_3c_noline = np.load('Analysis/image_data_3c_noline.npy', allow_pickle=True)

i = 19
j = 49

plt.figure()
plt.plot(wavelengths3b, image_data_3b_noline[:,i,j])
plt.plot(wavelengths3c, image_data_3c_noline[:,i,j])
plt.plot(wavelengths230cs, image_data_230cs[:,i,j])
plt.xlim(15,16)
plt.ylim(0,10000)
plt.show()
'''
#%%

array_length_x = len(image_data_230cs[0,:,0])
array_length_y = len(image_data_230cs[0,0,:])

pah_intensity_164 = np.load('Analysis/pah_intensity_164.npy')
pah_intensity_error_164 = np.load('Analysis/pah_intensity_error_164.npy')

#defining SNR

snr_164 = pah_intensity_164/pah_intensity_error_164

where_are_NaNs = np.isnan(snr_164) 
snr_164[where_are_NaNs] = 0

selection_164 = np.zeros((array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        if snr_164[i,j] > 50:
            selection_164[i,j] = 1

#%%

import ButterflyNebulaFunctions as bnf

bnf.error_check_imager(wavelengths230cs[4000:8200], (image_data_230cs-continuum_dust)[4000:8200], 'PDFtime/spectra_checking/plateau_check_continuum_dust_removed_zoom.pdf', 9.0, 16.0, 1.5, 
                       continuum=continuum[4000:8200], check_plat=good_or_bad_diff, check_curve=good_or_bad_curve, selection_array=selection_164)


bnf.error_check_imager(wavelengths230cs[4000:9000], (image_data_230cs-continuum_dust)[4000:9000], 'PDFtime/spectra_checking/plateau_check_continuum_dust_removed.pdf', 9.0, 18.0, 1.5, 
                       continuum=continuum[4000:9000], check_plat=good_or_bad_diff, check_curve=good_or_bad_curve, selection_array=selection_164)




#%%

