#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:26:13 2023

@author: nclark
"""

'''
IMPORTING MODULES
'''

#standard stuff
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import  AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
import random

#saving imagaes as PDFs
from PIL import Image  # install by > python3 -m pip install --upgrade Pillow  # ref. https://pillow.readthedocs.io/en/latest/installation.html#basic-installation

#used for fits file handling
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.table import Table

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

#for smoothing data
from scipy.signal import lfilter

#importing functions
import ButterflyNebulaFunctions as bnf
    

    
'''
LOADING IN DATA
'''
    


wavelengths1a, image_data_1a, error_data_1a = bnf.loading_function('data/ngc6302_ch1-short_s3d.fits', 1)
wavelengths1b, image_data_1b, error_data_1b = bnf.loading_function('data/ngc6302_ch1-medium_s3d.fits', 1)
wavelengths1c, image_data_1c, error_data_1c = bnf.loading_function('data/ngc6302_ch1-long_s3d.fits', 1)
wavelengths2a, image_data_2a, error_data_2a = bnf.loading_function('data/ngc6302_ch2-short_s3d.fits', 1)
wavelengths2b, image_data_2b, error_data_2b = bnf.loading_function('data/ngc6302_ch2-medium_s3d.fits', 1)
wavelengths2c, image_data_2c, error_data_2c = bnf.loading_function('data/ngc6302_ch2-long_s3d.fits', 1)
wavelengths3a, image_data_3a, error_data_3a = bnf.loading_function('data/ngc6302_ch3-short_s3d.fits', 1)
wavelengths3b, image_data_3b, error_data_3b = bnf.loading_function('data/ngc6302_ch3-medium_s3d.fits', 1)
wavelengths3c, image_data_3c, error_data_3c = bnf.loading_function('data/ngc6302_ch3-long_s3d.fits', 1) #note that this is the original, as psf matching is janky here

#these are loaded in for the purposes of hunting for crystalline silicates, and have wider psfs than the above stuff (they arent matched to one another either)
wavelengths4a, image_data_4a, error_data_4a = bnf.loading_function('data/ngc6302_ch4-short_s3d.fits', 1)
wavelengths4b, image_data_4b, error_data_4b = bnf.loading_function('data/ngc6302_ch4-medium_s3d.fits', 1)
wavelengths4c, image_data_4c, error_data_4c = bnf.loading_function('data/ngc6302_ch4-long_s3d.fits', 1)



'''
LOADING IN ANALYSIS DATA
'''



image_data_1a_noline = np.load('Analysis/image_data_1a_noline.npy', allow_pickle=True)
image_data_1b_noline = np.load('Analysis/image_data_1b_noline.npy', allow_pickle=True)
image_data_1c_noline = np.load('Analysis/image_data_1c_noline.npy', allow_pickle=True)
image_data_2a_noline = np.load('Analysis/image_data_2a_noline.npy', allow_pickle=True)
image_data_2b_noline = np.load('Analysis/image_data_2b_noline.npy', allow_pickle=True)
image_data_2c_noline = np.load('Analysis/image_data_2c_noline.npy', allow_pickle=True)
image_data_3a_noline = np.load('Analysis/image_data_3a_noline.npy', allow_pickle=True)
image_data_3b_noline = np.load('Analysis/image_data_3b_noline.npy', allow_pickle=True)
image_data_3c_noline = np.load('Analysis/image_data_3c_noline.npy', allow_pickle=True)
image_data_4a_noline = np.load('Analysis/image_data_4a_noline.npy', allow_pickle=True)
image_data_4b_noline = np.load('Analysis/image_data_4b_noline.npy', allow_pickle=True)
image_data_4c_noline = np.load('Analysis/image_data_4c_noline.npy', allow_pickle=True)

image_data_112_lines = np.load('Analysis/image_data_112_lines.npy', allow_pickle=True)
image_data_77_lines = np.load('Analysis/image_data_77_lines.npy', allow_pickle=True)
image_data_135_lines = np.load('Analysis/image_data_135_lines.npy', allow_pickle=True)
image_data_57_lines = np.load('Analysis/image_data_57_lines.npy', allow_pickle=True)
image_data_230cs_lines = np.load('Analysis/image_data_230cs_lines.npy', allow_pickle=True)

error_data_112 = np.load('Analysis/error_data_112.npy', allow_pickle=True)
error_data_77 = np.load('Analysis/error_data_77.npy', allow_pickle=True)
error_data_135 = np.load('Analysis/error_data_135.npy', allow_pickle=True)
error_data_57 = np.load('Analysis/error_data_57.npy', allow_pickle=True)
error_data_230cs = np.load('Analysis/error_data_230cs.npy', allow_pickle=True)

wavelengths112 = np.load('Analysis/wavelengths112.npy', allow_pickle=True)
image_data_112 = np.load('Analysis/image_data_112.npy', allow_pickle=True)
wavelengths77 = np.load('Analysis/wavelengths77.npy', allow_pickle=True)
image_data_77 = np.load('Analysis/image_data_77.npy', allow_pickle=True)
wavelengths135 = np.load('Analysis/wavelengths135.npy', allow_pickle=True)
image_data_135 = np.load('Analysis/image_data_135.npy', allow_pickle=True)
wavelengths57 = np.load('Analysis/wavelengths57.npy', allow_pickle=True)
image_data_57 = np.load('Analysis/image_data_57.npy', allow_pickle=True)
wavelengths230cs = np.load('Analysis/wavelengths230cs.npy', allow_pickle=True)
image_data_230cs = np.load('Analysis/image_data_230cs.npy', allow_pickle=True)

current_reprojection = np.load('Analysis/current_reprojection.npy', allow_pickle=True)

image_data_230cs_cont_1 = np.load('Analysis/image_data_230cs_cont_1.npy', allow_pickle=True)
image_data_230cs_cont_2 = np.load('Analysis/image_data_230cs_cont_2.npy', allow_pickle=True)
cont_type_230cs = np.load('Analysis/cont_type_230cs.npy', allow_pickle=True)
image_data_230cs_cont = np.load('Analysis/image_data_230cs_cont.npy', allow_pickle=True)
image_data_113cs_cont = np.load('Analysis/image_data_113cs_cont.npy', allow_pickle=True)
        
image_data_112_cont_1 = np.load('Analysis/image_data_112_cont_1.npy', allow_pickle=True)
image_data_112_cont_2 = np.load('Analysis/image_data_112_cont_2.npy', allow_pickle=True)
cont_type_112 = np.load('Analysis/cont_type_112.npy', allow_pickle=True)
image_data_112_cont = np.load('Analysis/image_data_112_cont.npy', allow_pickle=True)
lower_index_112 = np.load('Analysis/lower_index_112.npy', allow_pickle=True)
upper_index_112 = np.load('Analysis/upper_index_112.npy', allow_pickle=True)
pah_intensity_112 = np.load('Analysis/pah_intensity_112.npy', allow_pickle=True)
error_index_112 = np.load('Analysis/error_index_112.npy', allow_pickle=True)
pah_intensity_error_112 = np.load('Analysis/pah_intensity_error_112.npy', allow_pickle=True)
snr_cutoff_112 = np.load('Analysis/snr_cutoff_112.npy', allow_pickle=True)
        
lower_index_230cs = np.load('Analysis/lower_index_230cs.npy', allow_pickle=True)
upper_index_230cs = np.load('Analysis/upper_index_230cs.npy', allow_pickle=True)
pah_intensity_230cs = np.load('Analysis/pah_intensity_230cs.npy', allow_pickle=True)
error_index_230cs = np.load('Analysis/error_index_230cs.npy', allow_pickle=True)
pah_intensity_error_230cs = np.load('Analysis/pah_intensity_error_230cs.npy', allow_pickle=True)
snr_cutoff_230cs = np.load('Analysis/snr_cutoff_230cs.npy', allow_pickle=True)

image_data_52_cont = np.load('Analysis/image_data_52_cont.npy', allow_pickle=True)
lower_index_52 = np.load('Analysis/lower_index_52.npy', allow_pickle=True)
upper_index_52 = np.load('Analysis/upper_index_52.npy', allow_pickle=True)
pah_intensity_52 = np.load('Analysis/pah_intensity_52.npy', allow_pickle=True)
error_index_52 = np.load('Analysis/error_index_52.npy', allow_pickle=True)
pah_intensity_error_52 = np.load('Analysis/pah_intensity_error_52.npy', allow_pickle=True)
snr_cutoff_52 = np.load('Analysis/snr_cutoff_52.npy', allow_pickle=True)  

image_data_57_cont = np.load('Analysis/image_data_57_cont.npy', allow_pickle=True)
lower_index_57 = np.load('Analysis/lower_index_57.npy', allow_pickle=True)
upper_index_57 = np.load('Analysis/upper_index_57.npy', allow_pickle=True)
pah_intensity_57 = np.load('Analysis/pah_intensity_57.npy', allow_pickle=True)
error_index_57 = np.load('Analysis/error_index_57.npy', allow_pickle=True)
pah_intensity_error_57 = np.load('Analysis/pah_intensity_error_57.npy', allow_pickle=True)
snr_cutoff_57 = np.load('Analysis/snr_cutoff_57.npy', allow_pickle=True)

lower_index_59 = np.load('Analysis/lower_index_59.npy', allow_pickle=True)
upper_index_59 = np.load('Analysis/upper_index_59.npy', allow_pickle=True)
pah_intensity_59 = np.load('Analysis/pah_intensity_59.npy', allow_pickle=True)
error_index_59 = np.load('Analysis/error_index_59.npy', allow_pickle=True)
pah_intensity_error_59 = np.load('Analysis/pah_intensity_error_59.npy', allow_pickle=True)
snr_cutoff_59 = np.load('Analysis/snr_cutoff_59.npy', allow_pickle=True)
        
image_data_1b_cont = np.load('Analysis/image_data_1b_cont.npy', allow_pickle=True)
lower_index_62 = np.load('Analysis/lower_index_62.npy', allow_pickle=True)
upper_index_62 = np.load('Analysis/upper_index_62.npy', allow_pickle=True)
pah_intensity_62 = np.load('Analysis/pah_intensity_62.npy', allow_pickle=True)
error_index_62 = np.load('Analysis/error_index_62.npy', allow_pickle=True)
pah_intensity_error_62 = np.load('Analysis/pah_intensity_error_62.npy', allow_pickle=True)
snr_cutoff_62 = np.load('Analysis/snr_cutoff_62.npy', allow_pickle=True)

pah_intensity_60_and_62 = np.load('Analysis/pah_intensity_60_and_62.npy', allow_pickle=True)
error_index_60_and_62 = np.load('Analysis/error_index_60_and_62.npy', allow_pickle=True)
pah_intensity_error_60_and_62 = np.load('Analysis/pah_intensity_error_60_and_62.npy', allow_pickle=True)
snr_cutoff_60_and_62 = np.load('Analysis/snr_cutoff_60_and_62.npy', allow_pickle=True)

image_data_77_cont = np.load('Analysis/image_data_77_cont.npy', allow_pickle=True)
image_data_77_cont_local = np.load('Analysis/image_data_77_cont_local.npy', allow_pickle=True)
lower_index_77 = np.load('Analysis/lower_index_77.npy', allow_pickle=True)
middle_index_77_1 = np.load('Analysis/middle_index_77_1.npy', allow_pickle=True)
middle_index_77_2 = np.load('Analysis/middle_index_77_2.npy', allow_pickle=True)
upper_index_77 = np.load('Analysis/upper_index_77.npy', allow_pickle=True)
pah_intensity_77 = np.load('Analysis/pah_intensity_77.npy', allow_pickle=True)
error_index_77 = np.load('Analysis/error_index_77.npy', allow_pickle=True)
pah_intensity_error_77 = np.load('Analysis/pah_intensity_error_77.npy', allow_pickle=True)
snr_cutoff_77 = np.load('Analysis/snr_cutoff_77.npy', allow_pickle=True)

lower_index_86 = np.load('Analysis/lower_index_86.npy', allow_pickle=True)
upper_index_86 = np.load('Analysis/upper_index_86.npy', allow_pickle=True)
pah_intensity_86 = np.load('Analysis/pah_intensity_86.npy', allow_pickle=True)
error_index_86 = np.load('Analysis/error_index_86.npy', allow_pickle=True)
pah_intensity_error_86 = np.load('Analysis/pah_intensity_error_86.npy', allow_pickle=True)
snr_cutoff_86 = np.load('Analysis/snr_cutoff_86.npy', allow_pickle=True)

lower_index_86_plat = np.load('Analysis/lower_index_86_plat.npy', allow_pickle=True)
upper_index_86_plat = np.load('Analysis/upper_index_86_plat.npy', allow_pickle=True)
pah_intensity_86_plat = np.load('Analysis/pah_intensity_86_plat.npy', allow_pickle=True)
error_index_86_plat = np.load('Analysis/error_index_86_plat.npy', allow_pickle=True)
pah_intensity_error_86_plat = np.load('Analysis/pah_intensity_error_86_plat.npy', allow_pickle=True)
snr_cutoff_86_plat = np.load('Analysis/snr_cutoff_86_plat.npy', allow_pickle=True)

pah_intensity_86_local = np.load('Analysis/pah_intensity_86_local.npy', allow_pickle=True)
error_index_86_local = np.load('Analysis/error_index_86_local.npy', allow_pickle=True)
pah_intensity_error_86_local = np.load('Analysis/pah_intensity_error_86_local.npy', allow_pickle=True)
snr_cutoff_86_local = np.load('Analysis/snr_cutoff_86_local.npy', allow_pickle=True)

lower_index_110 = np.load('Analysis/lower_index_110.npy', allow_pickle=True)
upper_index_110 = np.load('Analysis/upper_index_110.npy', allow_pickle=True)
pah_intensity_110 = np.load('Analysis/pah_intensity_110.npy', allow_pickle=True)
error_index_110 = np.load('Analysis/error_index_110.npy', allow_pickle=True)
pah_intensity_error_110 = np.load('Analysis/pah_intensity_error_110.npy', allow_pickle=True)
snr_cutoff_110 = np.load('Analysis/snr_cutoff_110.npy', allow_pickle=True)

image_data_3a_cont = np.load('Analysis/image_data_3a_cont.npy', allow_pickle=True)
lower_index_120 = np.load('Analysis/lower_index_120.npy', allow_pickle=True)
upper_index_120 = np.load('Analysis/upper_index_120.npy', allow_pickle=True)
pah_intensity_120 = np.load('Analysis/pah_intensity_120.npy', allow_pickle=True)
error_index_120 = np.load('Analysis/error_index_120.npy', allow_pickle=True)
pah_intensity_error_120 = np.load('Analysis/pah_intensity_error_120.npy', allow_pickle=True)
snr_cutoff_120 = np.load('Analysis/snr_cutoff_120.npy', allow_pickle=True)

image_data_135_cont = np.load('Analysis/image_data_135_cont.npy', allow_pickle=True)
lower_index_135 = np.load('Analysis/lower_index_135.npy', allow_pickle=True)
upper_index_135 = np.load('Analysis/upper_index_135.npy', allow_pickle=True)
pah_intensity_135 = np.load('Analysis/pah_intensity_135.npy', allow_pickle=True)
error_index_135 = np.load('Analysis/error_index_135.npy', allow_pickle=True)
pah_intensity_error_135 = np.load('Analysis/pah_intensity_error_135.npy', allow_pickle=True)
snr_cutoff_135 = np.load('Analysis/snr_cutoff_135.npy', allow_pickle=True)

image_data_3c_cont_1 = np.load('Analysis/image_data_3c_cont_1.npy', allow_pickle=True)
image_data_3c_cont_2 = np.load('Analysis/image_data_3c_cont_2.npy', allow_pickle=True)
image_data_3c_cont_3 = np.load('Analysis/image_data_3c_cont_3.npy', allow_pickle=True)
cont_type_164 = np.load('Analysis/cont_type_164.npy', allow_pickle=True)
image_data_3c_cont = np.load('Analysis/image_data_3c_cont.npy', allow_pickle=True)
lower_index_164 = np.load('Analysis/lower_index_164.npy', allow_pickle=True)
upper_index_164 = np.load('Analysis/upper_index_164.npy', allow_pickle=True)
pah_intensity_164 = np.load('Analysis/pah_intensity_164.npy', allow_pickle=True)
error_index_164 = np.load('Analysis/error_index_164.npy', allow_pickle=True)
pah_intensity_error_164 = np.load('Analysis/pah_intensity_error_164.npy', allow_pickle=True)
snr_cutoff_164 = np.load('Analysis/snr_cutoff_164.npy', allow_pickle=True)
                                                   

print('data loaded')

#%%

#all arrays should have same spacial x and y dimensions, so define variables for this to use in for loops
array_length_x = len(image_data_1a[0,:,0])
array_length_y = len(image_data_1a[0,0,:])



#%%





'''
REGION FILE MASKING ARRAYS
'''



#creating an array that indicates where the Ch1 FOV is, so that comparison is only done between pixels with data.

region_indicator = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'data/ch1Arectangle.reg', do_sigma_clip=True, use_dq=False)



#%%

##################################


'''
TEMPLATE SPECTRA
'''



#templates:
# pah blob: 25, 15                                # north blob: 25, 39 (combine with south blob)

# north disk: 31, 33 (34, 35) (28, 38) (32, 35) now called
# filament: 21, 20
# west blob: 34, 29

# 31, 40: no plateau region (further out in north disk)
# 28, 39: strong plateau

# 21, 9 no 6.0 (SE of south blob) (18, 16)



# maybe distinct regions:
    
# 16, 35: basically 0 6-9 but strongish 11.2 (29, 20)



# 24, 30: has emission at 11.6
# 25, 31: note: enhanced 6-9 emission in west blob, central source region (25, 29) (24, 30)
# above, north disk can be grouped together as 'problem areas'



#resizing data

image_data, error_data = bnf.regrid(image_data_230cs, error_data_230cs, 2)
wavelengths = np.copy(wavelengths230cs)

#%%

array_length_x_2 = len(image_data[0,:,0])
array_length_y_2 = len(image_data[0,0,:])

# 5.25 feature

data_57, error_57 = bnf.regrid(image_data_57, error_data_57, 2)

data_52_cont = np.zeros((len(image_data_57[:,0,0]), array_length_x_2, array_length_y_2))

points52 = [5.15, 5.39, 5.55, 5.81]

for i in range(array_length_x_2):
    for j in range(array_length_y_2):
        data_52_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths57, data_57[:,i,j], points52)

# making continua for larger aperture

no60_data_57 = (data_57[:,9,21] +\
           data_57[:,11,18] +\
           data_57[:,10,20] +\
           data_57[:,9,22] +\
           data_57[:,14,16] +\
           data_57[:,15,15] +\
           data_57[:,16,15] +\
           data_57[:,17,15] +\
           data_57[:,11,19])/9
    
no60_data_52_cont = bnf.linear_continuum_single_channel(
    wavelengths57, no60_data_57, points52)

print('5.25 cont resized')

# 5.25, 5.7, 5.9 features

data_57_cont = np.zeros((len(image_data_57[:,0,0]), array_length_x_2, array_length_y_2))

points57 = [5.39, 5.55, 5.81, 5.94]

for i in range(array_length_x_2):
    for j in range(array_length_y_2):
        data_57_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths57, data_57[:,i,j], points57)

# making continua for larger aperture
    
no60_data_57_cont = bnf.linear_continuum_single_channel(
    wavelengths57, no60_data_57, points52)

print('5.7 cont resized')

# 6.0, 6.2 features

data_1b, error_1b = bnf.regrid(image_data_1b_noline, error_data_1b, 2)

data_1b_cont = np.zeros((len(image_data_1b[:,0,0]), array_length_x_2, array_length_y_2))

points62 = [5.68, 5.945, 6.53, 6.61]

for i in range(array_length_x_2):
    for j in range(array_length_y_2):
        data_1b_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths1b, data_1b[:,i,j], points62)

# making continua for larger aperture

no60_data_1b = (data_1b[:,9,21] +\
           data_1b[:,11,18] +\
           data_1b[:,10,20] +\
           data_1b[:,9,22] +\
           data_1b[:,14,16] +\
           data_1b[:,15,15] +\
           data_1b[:,16,15] +\
           data_1b[:,17,15] +\
           data_1b[:,11,19])/9
    
no60_data_1b_cont = bnf.linear_continuum_single_channel(
    wavelengths1b, no60_data_1b, points62)

print('6.2 cont resized')

# 7.7, 8.6 features

data_77, error_77 = bnf.regrid(image_data_77, error_data_77, 2)

data_77_cont = np.zeros((len(image_data_77[:,0,0]), array_length_x_2, array_length_y_2))

points77 = [6.55, 7.06, 9.08, 9.30] #used to be 11.65 instead of 11.70

for i in range(array_length_x_2):
    for j in range(array_length_y_2):
        data_77_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths77, data_77[:,i,j], points77) #note image_data_112 is built out of things with no lines

# making continua for larger aperture

no60_data_77 = (data_77[:,9,21] +\
           data_77[:,11,18] +\
           data_77[:,10,20] +\
           data_77[:,9,22] +\
           data_77[:,14,16] +\
           data_77[:,15,15] +\
           data_77[:,16,15] +\
           data_77[:,17,15] +\
           data_77[:,11,19])/9
    
no60_data_77_cont = bnf.linear_continuum_single_channel(
    wavelengths77, no60_data_77, points77)

print('7.7 cont resized')

# 11.0, 11.2 features

data_112, error_112 = bnf.regrid(image_data_112, error_data_112, 2)

data_112_cont = np.zeros((len(image_data_112[:,0,0]), array_length_x_2, array_length_y_2))

points112 = [10.61, 10.87, 11.70, 11.79] #used to be 11.65 instead of 11.70

for i in range(array_length_x_2):
    for j in range(array_length_y_2):
        data_112_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths112, data_112[:,i,j], points112) #note image_data_112 is built out of things with no lines

# making continua for larger aperture

no60_data_112 = (data_112[:,9,21] +\
           data_112[:,11,18] +\
           data_112[:,10,20] +\
           data_112[:,9,22] +\
           data_112[:,14,16] +\
           data_112[:,15,15] +\
           data_112[:,16,15] +\
           data_112[:,17,15] +\
           data_112[:,11,19])/9
    
no60_data_112_cont = bnf.linear_continuum_single_channel(
    wavelengths112, no60_data_112, points112)

print('11.2 cont resized')

# 12.0 feature

data_3a, error_3a = bnf.regrid(image_data_3a_noline, error_data_3a, 2)

data_3a_cont = np.zeros((len(image_data_3a[:,0,0]), array_length_x_2, array_length_y_2))

points120 = [11.65, 11.79, 12.25, 13.08]

for i in range(array_length_x_2):
    for j in range(array_length_y_2):
        data_3a_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths3a, data_3a[:,i,j], points120)

# making continua for larger aperture

no60_data_3a = (data_3a[:,9,21] +\
           data_3a[:,11,18] +\
           data_3a[:,10,20] +\
           data_3a[:,9,22] +\
           data_3a[:,14,16] +\
           data_3a[:,15,15] +\
           data_3a[:,16,15] +\
           data_3a[:,17,15] +\
           data_3a[:,11,19])/9
    
no60_data_3a_cont = bnf.linear_continuum_single_channel(
    wavelengths3a, no60_data_3a, points120)

print('12.0 cont resized')

# 13.5 feature

data_135, error_135 = bnf.regrid(image_data_135, error_data_135, 2)

data_135_cont = np.zeros((len(image_data_135[:,0,0]), array_length_x_2, array_length_y_2))

points135 = [13.00, 13.17, 13.78, 14.00] # [13.21, 13.31, 13.83, 14.00]

for i in range(array_length_x_2):
    for j in range(array_length_y_2):
        data_135_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths135, data_135[:,i,j], points135) #note image_data_135 is built out of things with no lines

# making continua for larger aperture

no60_data_135 = (data_135[:,9,21] +\
           data_135[:,11,18] +\
           data_135[:,10,20] +\
           data_135[:,9,22] +\
           data_135[:,14,16] +\
           data_135[:,15,15] +\
           data_135[:,16,15] +\
           data_135[:,17,15] +\
           data_135[:,11,19])/9
    
no60_data_135_cont = bnf.linear_continuum_single_channel(
    wavelengths135, no60_data_135, points135)
      
print('13.5 cont resized')
        
# 16.4 feature

data_3c, error_3c = bnf.regrid(image_data_3c_noline, error_data_3c, 2)

data_3c_cont = np.zeros((len(image_data_3c[:,0,0]), array_length_x_2, array_length_y_2))

points164 = [16.12, 16.27, 16.73, 16.85]

for i in range(array_length_x_2):
    for j in range(array_length_y_2):
        data_3c_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths3c, data_3c[:,i,j], points164)

# making continua for larger aperture

no60_data_3c = (data_3c[:,9,21] +\
           data_3c[:,11,18] +\
           data_3c[:,10,20] +\
           data_3c[:,9,22] +\
           data_3c[:,14,16] +\
           data_3c[:,15,15] +\
           data_3c[:,16,15] +\
           data_3c[:,17,15] +\
           data_3c[:,11,19])/9
    
no60_data_3c_cont = bnf.linear_continuum_single_channel(
    wavelengths3c, no60_data_3c, points164)

print('16.4 cont resized')

#%%

# resizing intensity maps

pah_164, pah_error_164 = bnf.regrid(pah_intensity_164.reshape(1, pah_intensity_164.shape[0], pah_intensity_164.shape[1])
                                    , pah_intensity_error_164.reshape(1, pah_intensity_164.shape[0], pah_intensity_164.shape[1]), 2)
pah_164 = pah_164[0]

pah_62, pah_error_62 = bnf.regrid(pah_intensity_62.reshape(1, pah_intensity_62.shape[0], pah_intensity_62.shape[1])
                                  , pah_intensity_error_62.reshape(1, pah_intensity_62.shape[0], pah_intensity_62.shape[1]), 2)
pah_62 = pah_62[0]

print('intensity maps resized')

#%%

'''
TEMPLATE SPECTRA LOCATION MAP
'''


#cam_header_f335m = fits.getheader('data/cams/jw01558005001_04101_00001_nrcblong_combined_i2d_flipped.fits', ext=1)

#pog1 = wcs.WCS(cam_header_f335m)

#ax = plt.figure('BNF_paper_template_indices', figsize=(10,8)).add_subplot(111, projection=pog1)

ax = plt.figure('BNF_paper_template_indices', figsize=(10,8)).add_subplot(111)

plt.rcParams.update({'font.size': 28})

im = plt.imshow(pah_62, cmap='gnuplot')

# Add the colorbar:
cbar = plt.colorbar(location = "right", fraction=0.05, pad=0.02)
cbar.formatter.set_powerlimits((0, 0))
cbar.ax.yaxis.set_offset_position('left')

#disk
plt.plot([49/2, 86/2, 61/2, 73/2, 69/2, 54/2, 60.5/2, 49/2], 
         [88/2, 95/2, 54/2, 42/2, 17/2, 14/2, 54/2, 88/2], color='green')
#central star
plt.scatter(54/2, 56/2, s=600, facecolors='none', edgecolors='purple')

plt.scatter([25], [15], s=100, facecolors='#dc267f', edgecolors='yellow')
plt.scatter([28], [39], s=100, facecolors='#785ef0', edgecolors='yellow')
plt.scatter([21, 20, 19, 18, 16, 15, 15, 15, 22], [9, 10, 11, 11, 14, 15, 16, 17, 9], s=100, facecolors='#fe6100', edgecolors='yellow')
plt.scatter([21], [20], s=100, facecolors='#648fff', edgecolors='yellow')
plt.scatter([31], [33], s=100, color='black')
plt.scatter([24], [30], s=100, color='gray')



ax.invert_yaxis()

#Customization of axes' appearance:
'''
lon = ax.coords[0]
lat = ax.coords[1]
ax.set_xlabel(r'Right Ascension ($\alpha$)')
ax.set_ylabel('Declination ($\delta$)', labelpad = -1)


lon.set_ticks(number=10)
lat.set_ticks(number=10)
lon.display_minor_ticks(False)
lat.display_minor_ticks(False)
lon.set_ticklabel(exclude_overlapping=True)
ax.tick_params(axis = "y", color = "k", left = True, right = True, direction = "out")
ax.tick_params(axis = "x", color = "k", bottom = True, top = True,  direction = "out")
'''


#plt.xlim((500, 1900)) #1400
#plt.ylim((600, 1700)) #1100




plt.savefig('PDFtime/paper/BNF_paper_template_indices.pdf', bbox_inches='tight')



#%%

'''
TEMPLATE SPECTA FIGURE
'''

#make larger aperture 

#            image_data[:,12,17] +\ this one has nans near 6.2 for some reason

no60_image_data = (image_data[:,9,21] +\

           image_data[:,11,18] +\
           image_data[:,10,20] +\
           image_data[:,9,22] +\
           image_data[:,14,16] +\
           image_data[:,15,15] +\
           image_data[:,16,15] +\
           image_data[:,17,15] +\
           image_data[:,11,19])/9
#calculate scaling

#scaling index

scaling_index = np.where(np.round(wavelengths, 3) == 9.250)[0][0]

pah_blob_scaling = 1000/np.median(image_data[scaling_index-10:scaling_index+10,15,25])
enhanced_plateau_scaling = 1000/np.median(image_data[scaling_index-10:scaling_index+10,39,28])
no60_scaling = 1000/np.median(image_data[scaling_index-10:scaling_index+10,9,21])
enhanced60_scaling = 1000/np.median(image_data[scaling_index-10:scaling_index+10,20,21])

pah_blob_scaling = 1
enhanced_plateau_scaling = 1
no60_scaling = 1
enhanced60_scaling = 1



ax = plt.figure('BNF_paper_template_spectra', figsize=(18,8)).add_subplot(111)

plt.rcParams.update({'font.size': 28})

ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.ylabel('Flux (MJy/sr)', labelpad=60)
plt.xlabel('Wavelength (micron)', labelpad=60)

ax = plt.figure('BNF_paper_template_spectra', figsize=(18,8)).add_subplot(111)


plt.loglog(wavelengths, pah_blob_scaling*image_data[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.loglog(wavelengths, enhanced_plateau_scaling*image_data[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.loglog(wavelengths, no60_image_data, color='#fe6100', label='No 6.0 (21,9)')
plt.loglog(wavelengths, enhanced60_scaling*image_data[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.loglog([6.2, 6.2], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([8.6, 8.6], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([11.2, 11.2], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([13.5, 13.5], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([16.4, 16.4], [0, 10**10], color='black', linestyle='dashed')




plt.ylim(40, 400000)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.d'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%.d'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)


#ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
plt.xlim(5.0, 28.0)
#plt.legend()

axT = ax.secondary_xaxis('top')

#axT.tick_params(axis='x', which='major', labelbottom='off', labeltop='on', top=True, length=10, width=4)
axT.set_xticks([6.2, 8.6, 11.2, 13.5, 16.4])
axT.set_xticks([], minor=True)
axT.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#axT.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
axT.tick_params(axis='x', which='major', length=10, width=4)
#axT.tick_params(axis='x', which='minor', labelbottom='off', labeltop='off', top=False, length=5, width=2)

plt.savefig('PDFtime/paper/BNF_paper_template_spectra.pdf', bbox_inches='tight')
plt.show()



#%%

'''
KEEPING THE ORIGINAL VERSION OF THE FIGURE UNTIL A LAYOUT IS FINALIZED
'''

ax = plt.figure('BNF_paper_template_spectra', figsize=(18,18)).add_subplot(111)

ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.ylabel('Flux (MJy/sr)', fontsize=16, labelpad=60)
plt.xlabel('Wavelength (micron)', fontsize=16, labelpad=30)

ax = plt.figure('BNF_paper_template_spectra', figsize=(18,18)).add_subplot(311)

plt.plot(wavelengths, pah_blob_scaling*image_data[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths, enhanced_plateau_scaling*image_data[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths, no60_scaling*image_data[:,9,21], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths, enhanced60_scaling*image_data[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')
#plt.plot(wavelengths, strong6to9_scaling*image_data[:,29,34], color='#fe6100', label='Strong 6-9 (34,29)')
#plt.plot(wavelengths, test_scaling*image_data[:,y,x], color='green', label='test')

plt.ylim((0,1.2*pah_blob_scaling*image_data[6180,15,25]))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.0, 13.0, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(5, 13.0)
plt.legend()

'''
10-15
'''

ax = plt.figure('BNF_paper_template_spectra', figsize=(18,18)).add_subplot(312)

plt.plot(wavelengths, pah_blob_scaling*image_data[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths, enhanced_plateau_scaling*image_data[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths, no60_scaling*image_data[:,9,21], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths, enhanced60_scaling*image_data[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')
#plt.plot(wavelengths, strong6to9_scaling*image_data[:,29,34], color='#fe6100', label='Strong 6-9 (34,29)')
#plt.plot(wavelengths, test_scaling*image_data[:,y,x], color='green', label='test')

#plt.ylim((0, 2.2*pah_blob_scaling*image_data[9110,15,25]))
plt.ylim((0,2*pah_blob_scaling*image_data[6180,15,25]))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.0, 15.0, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(10.0, 15.0)
plt.legend()

'''
13-19
'''

ax = plt.figure('BNF_paper_template_spectra', figsize=(18,18)).add_subplot(313)

plt.plot(wavelengths, pah_blob_scaling*image_data[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths, enhanced_plateau_scaling*image_data[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths, no60_scaling*image_data[:,9,21], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths, enhanced60_scaling*image_data[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')
#plt.plot(wavelengths, strong6to9_scaling*image_data[:,29,34], color='#fe6100', label='Strong 6-9 (34,29)')
#plt.plot(wavelengths, test_scaling*image_data[:,y,x], color='green', label='test')

plt.ylim((0, 2.2*pah_blob_scaling*image_data[9110,15,25]))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(13.0, 19.0, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(13.0, 19.0)
plt.legend()

#plt.savefig('PDFtime/paper/BNF_paper_template_spectra.pdf', bbox_inches='tight')
plt.show()
plt.close()

#%%

#now the bad spectra

#calculate scaling
north_disk_scaling = 100/np.max(image_data[6100:6300,33,31])
central_blob_scaling = 100/np.max(image_data[6100:6300,30,24])

north_disk_scaling = 1
central_blob_scaling = 1

ax = plt.figure('BNF_paper_template_spectra_bad', figsize=(18,8)).add_subplot(111)

plt.rcParams.update({'font.size': 28})

ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.ylabel('Flux (MJy/sr)', labelpad=60)
plt.xlabel('Wavelength (micron)', labelpad=60)

ax = plt.figure('BNF_paper_template_spectra_bad', figsize=(18,8)).add_subplot(111)


plt.loglog(wavelengths, north_disk_scaling*image_data[:,33,31], color='black', label='North disk (31,33)')
plt.loglog(wavelengths, central_blob_scaling*image_data[:,30,24], color='gray', label='Central blob (24,30)')

plt.loglog([6.2, 6.2], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([8.6, 8.6], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([11.2, 11.2], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([13.5, 13.5], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([16.4, 16.4], [0, 10**10], color='black', linestyle='dashed')




plt.ylim(300, 400000)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.d'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%.d'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)


#ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
plt.xlim(5.0, 28.0)
#plt.legend()

axT = ax.secondary_xaxis('top')

#axT.tick_params(axis='x', which='major', labelbottom='off', labeltop='on', top=True, length=10, width=4)
axT.set_xticks([6.2, 8.6, 11.2, 13.5, 16.4])
axT.set_xticks([], minor=True)
axT.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#axT.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
axT.tick_params(axis='x', which='major', length=10, width=4)
#axT.tick_params(axis='x', which='minor', labelbottom='off', labeltop='off', top=False, length=5, width=2)

plt.savefig('PDFtime/paper/BNF_paper_template_spectra_bad.pdf', bbox_inches='tight')
plt.show()

#%%

'''
KEEPING ORIGINAL FIGURE UNTIL NEW ONE IS DECIDED
'''

ax = plt.figure('BNF_paper_template_spectra_bad', figsize=(18,18)).add_subplot(111)

ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.ylabel('Flux (MJy/sr)', fontsize=16, labelpad=60)
plt.xlabel('Wavelength (micron)', fontsize=16, labelpad=30)

'''
5-13
'''

ax = plt.figure('BNF_paper_template_spectra_bad', figsize=(18,18)).add_subplot(311)

plt.plot(wavelengths, north_disk_scaling*image_data[:,33,31], color='black', linestyle='dashed', label='North disk (31,33)')
plt.plot(wavelengths, central_blob_scaling*image_data[:,30,24], color='black', linestyle='dotted', label='Central blob (24,30)')

plt.ylim((0,2.5*north_disk_scaling*image_data[6180,33,31]))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.0, 13.0, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(5, 13.0)
plt.legend()

'''
10-15
'''

ax = plt.figure('BNF_paper_template_spectra_bad', figsize=(18,18)).add_subplot(312)

plt.plot(wavelengths, north_disk_scaling*image_data[:,33,31], color='black', linestyle='dashed', label='North disk (31,33)')
plt.plot(wavelengths, central_blob_scaling*image_data[:,30,24], color='black', linestyle='dotted', label='Central blob (24,30)')

#plt.ylim((0, 2.2*pah_blob_scaling*image_data[9110,15,25]))
plt.ylim((0,7*north_disk_scaling*image_data[6180,33,31]))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.0, 15.0, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(10.0, 15.0)
plt.legend()

'''
13-19
'''

ax = plt.figure('BNF_paper_template_spectra_bad', figsize=(18,18)).add_subplot(313)

plt.plot(wavelengths, north_disk_scaling*image_data[:,33,31], color='black', linestyle='dashed', label='North disk (31,33)')
plt.plot(wavelengths, central_blob_scaling*image_data[:,30,24], color='black', linestyle='dotted', label='Central blob (24,30)')

plt.ylim((0, 1*north_disk_scaling*image_data[9110,33,31]))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(13.0, 19.0, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(13.0, 19.0)
plt.legend()

#plt.savefig('PDFtime/paper/BNF_paper_template_spectra_bad.pdf', bbox_inches='tight')
plt.show()
plt.close()

#%%

'''
CRYSTALLINE SILICATE IMPACT ON TEMPLATES PLOT
'''


ax = plt.figure('BNF_paper_template_spectra_silicates', figsize=(18,8)).add_subplot(111)

plt.rcParams.update({'font.size': 28})

ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.ylabel('Flux (MJy/sr)', labelpad=60)
plt.xlabel('Wavelength (micron)', labelpad=60)

ax = plt.figure('BNF_paper_template_spectra_silicates', figsize=(18,8)).add_subplot(111)


plt.loglog(wavelengths, north_disk_scaling*image_data[:,33,31], color='black', label='North disk (31,33)')
plt.loglog(wavelengths, enhanced_plateau_scaling*image_data[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')

plt.loglog([6.2, 6.2], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([8.6, 8.6], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([11.2, 11.2], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([13.5, 13.5], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([16.4, 16.4], [0, 10**10], color='black', linestyle='dashed')




plt.ylim(300, 200000)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.d'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%.d'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)


#ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
plt.xlim(5.0, 28.0)
#plt.legend()

axT = ax.secondary_xaxis('top')

#axT.tick_params(axis='x', which='major', labelbottom='off', labeltop='on', top=True, length=10, width=4)
axT.set_xticks([6.2, 8.6, 11.2, 13.5, 16.4])
axT.set_xticks([], minor=True)
axT.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#axT.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
axT.tick_params(axis='x', which='major', length=10, width=4)
#axT.tick_params(axis='x', which='minor', labelbottom='off', labeltop='off', top=False, length=5, width=2)

plt.savefig('PDFtime/paper/BNF_paper_template_spectra_silicates.pdf', bbox_inches='tight')
plt.show()

#%%



'''
TEMPLATE SPECTA INDIVIDUAL FEATURES SCALED TO 11.2 VERSION
'''

#calculate scaling
scaling_lower_index_112 = np.where(np.round(wavelengths112, 2) == 11.24)[0][0]
scaling_upper_index_112 = np.where(np.round(wavelengths112, 2) == 11.30)[0][0]

pah_blob_scaling_112_index = scaling_lower_index_112 + np.argmax((data_112 - data_112_cont)[scaling_lower_index_112:scaling_upper_index_112,15,25])
pah_blob_scaling_112 = 100/np.max((data_112 - data_112_cont)[pah_blob_scaling_112_index-5:pah_blob_scaling_112_index+5,15,25])

enhanced_plateau_scaling_112_index = scaling_lower_index_112 + np.argmax((data_112 - data_112_cont)[scaling_lower_index_112:scaling_upper_index_112,39,28])
enhanced_plateau_scaling_112 = 100/np.max((data_112 - data_112_cont)[enhanced_plateau_scaling_112_index-5:enhanced_plateau_scaling_112_index+5,39,28])

no60_scaling_112_index = scaling_lower_index_112 + np.argmax((no60_data_112 - no60_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
no60_scaling_112 = 100/np.median((no60_data_112 - no60_data_112_cont)[no60_scaling_112_index-5:no60_scaling_112_index+5])

enhanced60_scaling_112_index = scaling_lower_index_112 + np.argmax((data_112 - data_112_cont)[scaling_lower_index_112:scaling_upper_index_112,20,21])
enhanced60_scaling_112 = 100/np.max((data_112 - data_112_cont)[enhanced60_scaling_112_index-5:enhanced60_scaling_112_index+5,20,21])



ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,36)).add_subplot(111)

ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.ylabel('Flux (MJy/sr)', labelpad=90)
plt.xlabel('Wavelength (micron)', labelpad=60)

plt.rcParams.update({'font.size': 24})

'''
5.25
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,36)).add_subplot(421)

plt.title('5.25 feature', fontsize=18)

plt.plot(wavelengths57, pah_blob_scaling_112*(data_57 - data_52_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths57, enhanced_plateau_scaling_112*(data_57 - data_52_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths57, no60_scaling_112*(no60_data_57 - no60_data_52_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths57, enhanced60_scaling_112*(data_57 - data_52_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.1, 5.5, 0.1))
plt.yticks()
plt.xlim(5.1, 5.5)
#plt.legend()
plt.show()

'''
5.7
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,36)).add_subplot(422)

plt.title('5.7 and 5.9 feature', fontsize=18)

plt.plot(wavelengths57, pah_blob_scaling_112*(data_57 - data_57_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths57, enhanced_plateau_scaling_112*(data_57 - data_57_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths57, no60_scaling_112*(no60_data_57 - no60_data_57_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths57, enhanced60_scaling_112*(data_57 - data_57_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.5, 6.0, 0.1))
plt.yticks()
plt.xlim(5.5, 6.0)
#plt.legend()
plt.show()

'''
6.2
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,36)).add_subplot(423)

plt.title('6.0 and 6.2 features', fontsize=18)

plt.plot(wavelengths1b, pah_blob_scaling_112*(data_1b - data_1b_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths1b, enhanced_plateau_scaling_112*(data_1b - data_1b_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths1b, no60_scaling_112*(no60_data_1b - no60_data_1b_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths1b, enhanced60_scaling_112*(data_1b - data_1b_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.8, 6.6, 0.1))
plt.yticks()
plt.xlim(5.8, 6.6)
#plt.legend()
plt.show()

'''
7-9
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,36)).add_subplot(424)

plt.title('7-9 features', fontsize=18)

plt.plot(wavelengths77, pah_blob_scaling_112*(data_77 - data_77_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths77, enhanced_plateau_scaling_112*(data_77 - data_77_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths77, no60_scaling_112*(no60_data_77 - no60_data_77_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths77, enhanced60_scaling_112*(data_77 - data_77_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,200))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(7.0, 9.0, 0.2))
plt.yticks()
plt.xlim(7.0, 9.0)
#plt.legend()
plt.show()

'''
11.2
'''


#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,36)).add_subplot(425)

plt.title('11.0 and 11.2 feature', fontsize=18)

plt.plot(wavelengths112, pah_blob_scaling_112*(data_112 - data_112_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths112, enhanced_plateau_scaling_112*(data_112 - data_112_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths112, no60_scaling_112*(no60_data_112 - no60_data_112_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths112, enhanced60_scaling_112*(data_112 - data_112_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.8, 11.8, 0.2))
plt.yticks()
plt.xlim(10.8, 11.8)
#plt.legend()
plt.show()

'''
12.0
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,36)).add_subplot(426)

plt.title('12.0 feature', fontsize=18)

plt.plot(wavelengths3a, pah_blob_scaling_112*(data_3a - data_3a_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths3a, enhanced_plateau_scaling_112*(data_3a - data_3a_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths3a, no60_scaling_112*(no60_data_3a - no60_data_3a_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths3a, enhanced60_scaling_112*(data_3a - data_3a_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(11.6, 12.6, 0.2))
plt.yticks()
plt.xlim(11.6, 12.6)
#plt.legend()
plt.show()

'''
13.5
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,36)).add_subplot(427)

plt.title('13.5 feature', fontsize=18)

plt.plot(wavelengths135, pah_blob_scaling_112*(data_135 - data_135_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths135, enhanced_plateau_scaling_112*(data_135 - data_135_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths135, no60_scaling_112*(no60_data_135 - no60_data_135_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths135, enhanced60_scaling_112*(data_135 - data_135_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(13.0, 14.0, 0.2))
plt.yticks()
plt.xlim(13.0, 14.0)
#plt.legend()
plt.show()

'''
16.4
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,36)).add_subplot(428)

plt.title('16.4 feature', fontsize=18)

plt.plot(wavelengths3c, pah_blob_scaling_112*(data_3c - data_3c_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths3c, enhanced_plateau_scaling_112*(data_3c - data_3c_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths3c, no60_scaling_112*(no60_data_3c - no60_data_3c_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths3c, pah_blob_scaling_112*(data_3c - data_3c_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(16.1, 17.1, 0.2))
plt.yticks()
plt.xlim(16.1, 17.1)
#plt.legend()
plt.show()

plt.savefig('PDFtime/paper/BNF_paper_template_spectra_features_112_scaled.pdf', bbox_inches='tight')
plt.show()

#%%



'''
TEMPLATE SPECTA INDIVIDUAL FEATURES
'''



ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(111)

ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.ylabel('Flux (MJy/sr)', labelpad=90)
plt.xlabel('Wavelength (micron)', labelpad=60)

plt.rcParams.update({'font.size': 24})

'''
5.25
'''

#calculate scaling
scaling_lower_index_52 = np.where(np.round(wavelengths1a, 2) == 5.22)[0][0]
scaling_upper_index_52 = np.where(np.round(wavelengths1a, 2) == 5.24)[0][0]

pah_blob_scaling_52_index = scaling_lower_index_52 + np.argmax((data_57 - data_52_cont)[scaling_lower_index_52:scaling_upper_index_52,15,25])
pah_blob_scaling_52 = 100/np.median((data_57 - data_52_cont)[pah_blob_scaling_52_index-5:pah_blob_scaling_52_index+5,15,25])

enhanced_plateau_scaling_52_index = scaling_lower_index_52 + np.argmax((data_57 - data_52_cont)[scaling_lower_index_52:scaling_upper_index_52,39,28])
enhanced_plateau_scaling_52 = 100/np.median((data_57 - data_52_cont)[enhanced_plateau_scaling_52_index-5:enhanced_plateau_scaling_52_index+5,39,28])

no60_scaling_52_index = scaling_lower_index_52 + np.argmax((no60_data_57 - no60_data_52_cont)[scaling_lower_index_52:scaling_upper_index_52])
no60_scaling_52 = 100/np.median((no60_data_57 - no60_data_52_cont)[no60_scaling_52_index-5:no60_scaling_52_index+5])

enhanced60_scaling_52_index = scaling_lower_index_52 + np.argmax((data_57 - data_52_cont)[scaling_lower_index_52:scaling_upper_index_52,20,21])
enhanced60_scaling_52 = 100/np.median((data_57 - data_52_cont)[enhanced60_scaling_52_index-5:enhanced60_scaling_52_index+5,20,21])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(421)

plt.title('5.25 feature, scaled to 5.25 feature peak', fontsize=18)

plt.plot(wavelengths57, pah_blob_scaling_52*(data_57 - data_52_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths57, enhanced_plateau_scaling_52*(data_57 - data_52_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths57, pah_blob_scaling_52*(no60_data_57 - no60_data_52_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths57, enhanced60_scaling_52*(data_57 - data_52_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.1, 5.5, 0.1))
plt.yticks()
plt.xlim(5.1, 5.5)
#plt.legend()
plt.show()

'''
5.7
'''

#calculate scaling
scaling_lower_index_57 = np.where(np.round(wavelengths57, 2) == 5.7)[0][0]
scaling_upper_index_57 = np.where(np.round(wavelengths57, 2) == 5.75)[0][0]

pah_blob_scaling_57_index = scaling_lower_index_57 + np.argmax((data_57 - data_57_cont)[scaling_lower_index_57:scaling_upper_index_57,15,25])
pah_blob_scaling_57 = 100/np.median((data_57 - data_57_cont)[pah_blob_scaling_57_index-5:pah_blob_scaling_57_index+5,15,25])

enhanced_plateau_scaling_57_index = scaling_lower_index_57 + np.argmax((data_57 - data_57_cont)[scaling_lower_index_57:scaling_upper_index_57,39,28])
enhanced_plateau_scaling_57 = 100/np.median((data_57 - data_57_cont)[enhanced_plateau_scaling_57_index-5:enhanced_plateau_scaling_57_index+5,39,28])

no60_scaling_57_index = scaling_lower_index_57 + np.argmax((no60_data_57 - no60_data_57_cont)[scaling_lower_index_57:scaling_upper_index_57])
no60_scaling_57 = 100/np.median((no60_data_57 - no60_data_57_cont)[no60_scaling_57_index-5:no60_scaling_57_index+5])

enhanced60_scaling_57_index = scaling_lower_index_57 + np.argmax((data_57 - data_57_cont)[scaling_lower_index_57:scaling_upper_index_57,20,21])
enhanced60_scaling_57 = 100/np.median((data_57 - data_57_cont)[enhanced60_scaling_57_index-5:enhanced60_scaling_57_index+5,20,21])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(422)

plt.title('5.7 and 5.9 feature, scaled to 5.7 feature peak', fontsize=18)

plt.plot(wavelengths57, pah_blob_scaling_57*(data_57 - data_57_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths57, enhanced_plateau_scaling_57*(data_57 - data_57_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths57, no60_scaling_57*(no60_data_57 - no60_data_57_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths57, enhanced60_scaling_57*(data_57 - data_57_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.5, 6.0, 0.1))
plt.yticks()
plt.xlim(5.5, 6.0)
#plt.legend()
plt.show()

'''
6.2
'''

#calculate scaling
scaling_lower_index_62 = np.where(np.round(wavelengths1b, 2) == 6.15)[0][0]
scaling_upper_index_62 = np.where(np.round(wavelengths1b, 2) == 6.25)[0][0]

pah_blob_scaling_62_index = scaling_lower_index_62 + np.argmax((data_1b - data_1b_cont)[scaling_lower_index_62:scaling_upper_index_62,15,25])
pah_blob_scaling_62 = 100/np.max((data_1b - data_1b_cont)[pah_blob_scaling_62_index-5:pah_blob_scaling_62_index+5,15,25])

enhanced_plateau_scaling_62_index = scaling_lower_index_62 + np.argmax((data_1b - data_1b_cont)[scaling_lower_index_62:scaling_upper_index_62,39,28])
enhanced_plateau_scaling_62 = 100/np.max((data_1b - data_1b_cont)[enhanced_plateau_scaling_62_index-5:enhanced_plateau_scaling_62_index+5,39,28])

no60_scaling_62_index = scaling_lower_index_62 + np.argmax((no60_data_1b - no60_data_1b_cont)[scaling_lower_index_62:scaling_upper_index_62])
no60_scaling_62 = 100/np.median((no60_data_1b - no60_data_1b_cont)[no60_scaling_62_index-5:no60_scaling_62_index+5])

enhanced60_scaling_62_index = scaling_lower_index_62 + np.argmax((data_1b - data_1b_cont)[scaling_lower_index_62:scaling_upper_index_62,20,21])
enhanced60_scaling_62 = 100/np.max((data_1b - data_1b_cont)[enhanced60_scaling_62_index-5:enhanced60_scaling_62_index+5,20,21])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(423)

plt.title('6.0 and 6.2 feature, scaled to 6.2 feature peak', fontsize=18)

plt.plot(wavelengths1b, pah_blob_scaling_62*(data_1b - data_1b_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths1b, enhanced_plateau_scaling_62*(data_1b - data_1b_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths1b, no60_scaling_62*(no60_data_1b - no60_data_1b_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths1b, enhanced60_scaling_62*(data_1b - data_1b_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.8, 6.6, 0.1))
plt.yticks()
plt.xlim(5.8, 6.6)
#plt.legend()
plt.show()

'''
7-9
'''

#calculate scaling
scaling_lower_index_77 = np.where(np.round(wavelengths77, 2) == 8.5)[0][0]
scaling_upper_index_77 = np.where(np.round(wavelengths77, 2) == 8.7)[0][0]

pah_blob_scaling_77_index = scaling_lower_index_77 + np.argmax((data_77 - data_77_cont)[scaling_lower_index_77:scaling_upper_index_77,15,25])
pah_blob_scaling_77 = 100/np.max((data_77 - data_77_cont)[pah_blob_scaling_77_index-5:pah_blob_scaling_77_index+5,15,25])

enhanced_plateau_scaling_77_index = scaling_lower_index_77 + np.argmax((data_77 - data_77_cont)[scaling_lower_index_77:scaling_upper_index_77,39,28])
enhanced_plateau_scaling_77 = 100/np.max((data_77 - data_77_cont)[enhanced_plateau_scaling_77_index-5:enhanced_plateau_scaling_77_index+5,39,28])

no60_scaling_77_index = scaling_lower_index_77 + np.argmax((no60_data_77 - no60_data_77_cont)[scaling_lower_index_77:scaling_upper_index_77])
no60_scaling_77 = 100/np.median((no60_data_77 - no60_data_77_cont)[no60_scaling_77_index-5:no60_scaling_77_index+5])

enhanced60_scaling_77_index = scaling_lower_index_77 + np.argmax((data_77 - data_77_cont)[scaling_lower_index_77:scaling_upper_index_77,20,21])
enhanced60_scaling_77 = 100/np.max((data_77 - data_77_cont)[enhanced60_scaling_77_index-5:enhanced60_scaling_77_index+5,20,21])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(424)

plt.title('7-9 features, scaled to 8.6', fontsize=18)

plt.plot(wavelengths77, pah_blob_scaling_77*(data_77 - data_77_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths77, enhanced_plateau_scaling_77*(data_77 - data_77_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths77, no60_scaling_77*(no60_data_77 - no60_data_77_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths77, enhanced60_scaling_77*(data_77 - data_77_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,200))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(7.0, 9.0, 0.2))
plt.yticks()
plt.xlim(7.0, 9.0)
#plt.legend()
plt.show()

'''
11.2
'''

#calculate scaling
scaling_lower_index_112 = np.where(np.round(wavelengths112, 2) == 11.24)[0][0]
scaling_upper_index_112 = np.where(np.round(wavelengths112, 2) == 11.30)[0][0]

pah_blob_scaling_112_index = scaling_lower_index_112 + np.argmax((data_112 - data_112_cont)[scaling_lower_index_112:scaling_upper_index_112,15,25])
pah_blob_scaling_112 = 100/np.max((data_112 - data_112_cont)[pah_blob_scaling_112_index-5:pah_blob_scaling_112_index+5,15,25])

enhanced_plateau_scaling_112_index = scaling_lower_index_112 + np.argmax((data_112 - data_112_cont)[scaling_lower_index_112:scaling_upper_index_112,39,28])
enhanced_plateau_scaling_112 = 100/np.max((data_112 - data_112_cont)[enhanced_plateau_scaling_112_index-5:enhanced_plateau_scaling_112_index+5,39,28])

no60_scaling_112_index = scaling_lower_index_112 + np.argmax((no60_data_112 - no60_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
no60_scaling_112 = 100/np.median((no60_data_112 - no60_data_112_cont)[no60_scaling_112_index-5:no60_scaling_112_index+5])

enhanced60_scaling_112_index = scaling_lower_index_112 + np.argmax((data_112 - data_112_cont)[scaling_lower_index_112:scaling_upper_index_112,20,21])
enhanced60_scaling_112 = 100/np.max((data_112 - data_112_cont)[enhanced60_scaling_112_index-5:enhanced60_scaling_112_index+5,20,21])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(425)

plt.title('11.0 and 11.2 feature, scaled to 11.2 feature peak', fontsize=18)

plt.plot(wavelengths112, pah_blob_scaling_112*(data_112 - data_112_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths112, enhanced_plateau_scaling_112*(data_112 - data_112_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths112, no60_scaling_112*(no60_data_112 - no60_data_112_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths112, enhanced60_scaling_112*(data_112 - data_112_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.8, 11.8, 0.2))
plt.yticks()
plt.xlim(10.8, 11.8)
#plt.legend()
plt.show()

'''
12.0
'''
#calculate scaling
scaling_lower_index_120 = np.where(np.round(wavelengths3a, 2) == 12.0)[0][0]
scaling_upper_index_120 = np.where(np.round(wavelengths3a, 2) == 12.1)[0][0]

pah_blob_scaling_120_index = scaling_lower_index_120 + np.argmax((data_3a - data_3a_cont)[scaling_lower_index_120:scaling_upper_index_120,15,25])
pah_blob_scaling_120 = 100/np.max((data_3a - data_3a_cont)[pah_blob_scaling_120_index-5:pah_blob_scaling_120_index+5,15,25])

enhanced_plateau_scaling_120_index = scaling_lower_index_120 + np.argmax((data_3a - data_3a_cont)[scaling_lower_index_120:scaling_upper_index_120,39,28])
enhanced_plateau_scaling_120 = 100/np.max((data_3a - data_3a_cont)[enhanced_plateau_scaling_120_index-5:enhanced_plateau_scaling_120_index+5,39,28])

no60_scaling_120_index = scaling_lower_index_120 + np.argmax((no60_data_3a - no60_data_3a_cont)[scaling_lower_index_120:scaling_upper_index_120])
no60_scaling_120 = 100/np.median((no60_data_3a - no60_data_3a_cont)[no60_scaling_120_index-5:no60_scaling_120_index+5])

enhanced60_scaling_120_index = scaling_lower_index_120 + np.argmax((data_3a - data_3a_cont)[scaling_lower_index_120:scaling_upper_index_120,20,21])
enhanced60_scaling_120 = 100/np.max((data_3a - data_3a_cont)[enhanced60_scaling_120_index-5:enhanced60_scaling_120_index+5,20,21])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(426)

plt.title('12.0 feature, scaled to 12.0 feature peak', fontsize=18)

plt.plot(wavelengths3a, pah_blob_scaling_120*(data_3a - data_3a_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths3a, enhanced_plateau_scaling_120*(data_3a - data_3a_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
#plt.plot(wavelengths3a, no60_scaling_120*(data_3a - data_3a_cont)[:,9,21], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths3a, pah_blob_scaling_120*(no60_data_3a - no60_data_3a_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths3a, enhanced60_scaling_120*(data_3a - data_3a_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(11.6, 12.6, 0.2))
plt.yticks()
plt.xlim(11.6, 12.6)
#plt.legend()
plt.show()

'''
13.5
'''

#calculate scaling
scaling_lower_index_135 = np.where(np.round(wavelengths135, 2) == 13.61)[0][0]
scaling_upper_index_135 = np.where(np.round(wavelengths135, 2) == 13.65)[0][0]

pah_blob_scaling_135_index = scaling_lower_index_135 + np.argmax((data_135 - data_135_cont)[scaling_lower_index_135:scaling_upper_index_135,15,25])
pah_blob_scaling_135 = 100/np.max((data_135 - data_135_cont)[pah_blob_scaling_135_index-5:pah_blob_scaling_135_index+5,15,25])

enhanced_plateau_scaling_135_index = scaling_lower_index_135 + np.argmax((data_135 - data_135_cont)[scaling_lower_index_135:scaling_upper_index_135,39,28])
enhanced_plateau_scaling_135 = 100/np.max((data_135 - data_135_cont)[enhanced_plateau_scaling_135_index-5:enhanced_plateau_scaling_135_index+5,39,28])

no60_scaling_135_index = scaling_lower_index_135 + np.argmax((no60_data_135 - no60_data_135_cont)[scaling_lower_index_135:scaling_upper_index_135])
no60_scaling_135 = 100/np.median((no60_data_135 - no60_data_135_cont)[no60_scaling_135_index-5:no60_scaling_135_index+5])

enhanced60_scaling_135_index = scaling_lower_index_135 + np.argmax((data_135 - data_135_cont)[scaling_lower_index_135:scaling_upper_index_135,20,21])
enhanced60_scaling_135 = 100/np.max((data_135 - data_135_cont)[enhanced60_scaling_135_index-5:enhanced60_scaling_135_index+5,20,21])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(427)

plt.title('13.5 feature, scaled to 13.5 feature peak', fontsize=18)

plt.plot(wavelengths135, pah_blob_scaling_135*(data_135 - data_135_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths135, enhanced_plateau_scaling_135*(data_135 - data_135_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths135, no60_scaling_135*(no60_data_135 - no60_data_135_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths135, enhanced60_scaling_135*(data_135 - data_135_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

#plt.plot(wavelengths135, 0.2*(data_135 - data_135_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
#plt.plot(wavelengths135, 0.2*(data_135 - data_135_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
#plt.plot(wavelengths135, 0.2*(data_135 - data_135_cont)[:,9,21], color='#fe6100', label='No 6.0 (21,9)')
#plt.plot(wavelengths135, 0.2*(data_135 - data_135_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(13.0, 14.0, 0.2))
plt.yticks()
plt.xlim(13.0, 14.0)
#plt.legend()
plt.show()

'''
16.4
'''

#calculate scaling
scaling_lower_index_164 = np.where(np.round(wavelengths3c, 2) == 16.3)[0][0]
scaling_upper_index_164 = np.where(np.round(wavelengths3c, 2) == 16.5)[0][0]

pah_blob_scaling_164_index = scaling_lower_index_164 + np.argmax((data_3c - data_3c_cont)[scaling_lower_index_164:scaling_upper_index_164,15,25])
pah_blob_scaling_164 = 100/np.max((data_3c - data_3c_cont)[pah_blob_scaling_164_index-5:pah_blob_scaling_164_index+5,15,25])

enhanced_plateau_scaling_164_index = scaling_lower_index_164 + np.argmax((data_3c - data_3c_cont)[scaling_lower_index_164:scaling_upper_index_164,39,28])
enhanced_plateau_scaling_164 = 100/np.max((data_3c - data_3c_cont)[enhanced_plateau_scaling_164_index-5:enhanced_plateau_scaling_164_index+5,39,28])

no60_scaling_164_index = scaling_lower_index_164 + np.argmax((no60_data_3c - no60_data_3c_cont)[scaling_lower_index_164:scaling_upper_index_164])
no60_scaling_164 = 100/np.median((no60_data_3c - no60_data_3c_cont)[no60_scaling_164_index-5:no60_scaling_164_index+5])

enhanced60_scaling_164_index = scaling_lower_index_164 + np.argmax((data_3c - data_3c_cont)[scaling_lower_index_164:scaling_upper_index_164,20,21])
enhanced60_scaling_164 = 100/np.max((data_3c - data_3c_cont)[enhanced60_scaling_164_index-5:enhanced60_scaling_164_index+5,20,21])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(428)

plt.title('16.4 feature, scaled to 16.4 feature peak', fontsize=18)

plt.plot(wavelengths3c, pah_blob_scaling_164*(data_3c - data_3c_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths3c, enhanced_plateau_scaling_164*(data_3c - data_3c_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths3c, no60_scaling_164*(no60_data_3c - no60_data_3c_cont)[:], color='#fe6100', label='No 6.0 (21,9)')
#plt.plot(wavelengths3c, enhanced60_scaling_164*(data_3c - data_3c_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')
plt.plot(wavelengths3c, pah_blob_scaling_164*(data_3c - data_3c_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,120))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(16.1, 17.1, 0.2))
plt.yticks()
plt.xlim(16.1, 17.1)
#plt.legend()
plt.show()

plt.savefig('PDFtime/paper/BNF_paper_template_spectra_features.pdf', bbox_inches='tight')
plt.show()



#%%







#%%



##################################


'''
FIDGETTING AND BUGTESTING
'''



ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,8)).add_subplot(111)



plt.title('16.4 feature, scaled to 16.4 feature peak', fontsize=14)

plt.plot(wavelengths3c, pah_blob_scaling_164*(data_3c - data_3c_cont)[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths3c, enhanced_plateau_scaling_164*(data_3c - data_3c_cont)[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths3c, no60_scaling_164*(data_3c - data_3c_cont)[:,9,21], color='#fe6100', label='No 6.0 (21,9)')
#plt.plot(wavelengths3c, enhanced60_scaling_164*(data_3c - data_3c_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')
plt.plot(wavelengths3c, pah_blob_scaling_164*(data_3c - data_3c_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-20,120))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(16.1, 17.0, 0.1), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(16.1, 17.0)
plt.legend()
plt.show()

#%%

# 15,25   39,28   9,21   20,21

i,j = 20,21

plt.plot(wavelengths135, (data_135)[:,i,j])
plt.plot(wavelengths135, (data_135_cont)[:,i,j])





#%%

# 15,25   39,28   9,21   20,21

plt.plot(wavelengths3a, (data_3a)[:,9,21])
plt.plot(wavelengths3a, (data_3a_cont)[:,9,21])


#%%

# 15,25   39,28   9,21   20,21
plt.figure()
plt.plot(wavelengths135, (data_135)[:,15,25])
plt.plot(wavelengths135, (data_135_cont)[:,15,25])


#%%






x=20
y=21

ax = plt.figure('11.0 intensity, log 10, points', figsize=(8,8)).add_subplot(111)
plt.title('11.0 intensity, log 10')
plt.imshow(pah_164)
plt.scatter(x, y, color='red')

#disk
plt.plot([49/2, 86/2, 61/2, 73/2, 69/2, 54/2, 60.5/2, 49/2], 
         [88/2, 95/2, 54/2, 42/2, 17/2, 14/2, 54/2, 88/2], color='green')
#central star
plt.scatter(54/2, 56/2, s=600, facecolors='none', edgecolors='purple')

plt.colorbar()

ax.invert_yaxis()
plt.show()


#%%



ax = plt.figure('BNF_paper_template_spectra', figsize=(18,9)).add_subplot(111)
plt.plot(wavelengths, pah_blob_scaling*image_data[:,33,31], color='#dc267f', label='North disk (31,33)')
plt.plot(wavelengths, enhanced_plateau_scaling*image_data[:,39,28], color='#785ef0', label='Central blob (24,30)')
plt.plot(wavelengths, no60_scaling*image_data[:,9,21], color='black', label='No 6.0 (21,9)')
plt.plot(wavelengths, enhanced60_scaling*image_data[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')


plt.show()


#%%

region_indicator = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'data/ch1Arectangle.reg', do_sigma_clip=True, use_dq=False)

temp_region_indicator = region_indicator.reshape(1, region_indicator.shape[0], region_indicator.shape[1])

#making region correct size
region_indicator, temp = bnf.regrid(temp_region_indicator, np.ones(temp_region_indicator.shape), 2)

region_indicator = region_indicator[0]
    
#%%

#checking that chosen template spectra are representative

bnf.template_check_imager(wavelengths, image_data, 'PDFtime/spectra_checking/_5to13_template_check.pdf', 5.0, 13.0, pah_62, region_indicator, 7897890)

#%%

bnf.template_check_imager(wavelengths, image_data, 'PDFtime/spectra_checking/_13to19_template_check.pdf', 13.0, 19.0, pah_62, region_indicator, 7897890)

#%%

ax = plt.figure('BNF_paper_template_spectra', figsize=(18,8)).add_subplot(111)

plt.rcParams.update({'font.size': 28})

ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.ylabel('Flux (MJy/sr)', labelpad=60)
plt.xlabel('Wavelength (micron)', labelpad=60)

ax = plt.figure('BNF_paper_template_spectra', figsize=(18,8)).add_subplot(111)



plt.loglog(wavelengths, no60_scaling*image_data[:,9,21], color='#fe6100', label='No 6.0 (21,9)')

x = 24
y = 8

plt.loglog(wavelengths, no60_scaling*image_data[:,y,x], alpha=0.5, label='No 6.0 (21,9)')

plt.loglog([6.2, 6.2], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([8.6, 8.6], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([11.2, 11.2], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([13.5, 13.5], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([16.4, 16.4], [0, 10**10], color='black', linestyle='dashed')

# NOT 14 18 or 13 18 or 23 9 or 23 8 or 24 8
# 16 14, 15 15, 15 16, 15 17, 22 9, 


plt.ylim(40, 20000)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.d'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%.d'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)


#ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
plt.xlim(5.0, 28.0)
#plt.legend()

axT = ax.secondary_xaxis('top')

#axT.tick_params(axis='x', which='major', labelbottom='off', labeltop='on', top=True, length=10, width=4)
axT.set_xticks([6.2, 8.6, 11.2, 13.5, 16.4])
axT.set_xticks([], minor=True)
axT.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#axT.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
axT.tick_params(axis='x', which='major', length=10, width=4)
#axT.tick_params(axis='x', which='minor', labelbottom='off', labeltop='off', top=False, length=5, width=2)


plt.show()

#%%

# image_data[:,9,20] +
# 16 14, 15 15, 15 16, 15 17, 22 9, 

big_pog = (image_data[:,9,21] +\
           image_data[:,11,18] +\
           image_data[:,10,20] +\
           image_data[:,9,22] +\
           image_data[:,14,16] +\
           image_data[:,15,15] +\
           image_data[:,16,15] +\
           image_data[:,17,15] +\
           image_data[:,11,19])/9
    

ax = plt.figure('BNF_paper_template_spectra', figsize=(18,8)).add_subplot(111)

plt.rcParams.update({'font.size': 28})

ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.ylabel('Flux (MJy/sr)', labelpad=60)
plt.xlabel('Wavelength (micron)', labelpad=60)

ax = plt.figure('BNF_paper_template_spectra', figsize=(18,8)).add_subplot(111)



plt.plot(wavelengths, 0.25*image_data[:,15,25], color='#dc267f', label='PAH blob (25,15), 0.25 scaling')
plt.plot(wavelengths, enhanced_plateau_scaling*image_data[:,39,28], color='#785ef0', label='Enhanced plateau (28,39)')
plt.plot(wavelengths, 1*image_data[:,41,30], color='black', label='(30, 41)')

#plt.plot(wavelengths, 0.1*image_data[:,15,25], color='#dc267f', label='PAH blob (25,15)')
#plt.plot(wavelengths, no60_scaling*image_data[:,9,21], color='#fe6100', label='No 6.0 (21,9)')
#plt.plot(wavelengths, 0.9*big_pog, label='No 6.0 (21,9)')

plt.legend()

plt.ylim(0, 5000)

plt.xlim(5.5,11.6)
#plt.ylim(0, 450)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)


#ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.xlim(5.0, 28.0)
#plt.legend()



#axT.tick_params(axis='x', which='major', labelbottom='off', labeltop='on', top=True, length=10, width=4)

axT.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#axT.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
axT.tick_params(axis='x', which='major', length=10, width=4)
#axT.tick_params(axis='x', which='minor', labelbottom='off', labeltop='off', top=False, length=5, width=2)


plt.show()

