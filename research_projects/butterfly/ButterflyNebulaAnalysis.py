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

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

#importing functions
import ButterflyNebulaFunctions as bnf

#fringe removal
#from jwst.residual_fringe.utils import fit_residual_fringes_1d as rf1d

#time
import time

start = time.time()


    
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

wavelengths57 = np.load('Analysis/wavelengths57.npy', allow_pickle=True)
wavelengths77 = np.load('Analysis/wavelengths77.npy', allow_pickle=True)
wavelengths112 = np.load('Analysis/wavelengths112.npy', allow_pickle=True)
wavelengths135 = np.load('Analysis/wavelengths135.npy', allow_pickle=True)
wavelengths230cs = np.load('Analysis/wavelengths230cs.npy', allow_pickle=True)

image_data_57 = np.load('Analysis/image_data_57.npy', allow_pickle=True)
image_data_77 = np.load('Analysis/image_data_77.npy', allow_pickle=True)
image_data_112 = np.load('Analysis/image_data_112.npy', allow_pickle=True)
image_data_135 = np.load('Analysis/image_data_135.npy', allow_pickle=True)
image_data_230cs = np.load('Analysis/image_data_230cs.npy', allow_pickle=True)

image_data_57_noline = np.load('Analysis/image_data_57_noline.npy', allow_pickle=True)
image_data_77_noline = np.load('Analysis/image_data_77_noline.npy', allow_pickle=True)
image_data_112_noline = np.load('Analysis/image_data_112_noline.npy', allow_pickle=True)
image_data_135_noline = np.load('Analysis/image_data_135_noline.npy', allow_pickle=True)
image_data_230cs_noline = np.load('Analysis/image_data_230cs_noline.npy', allow_pickle=True)




'''
MISC HELPER VARIABLES
'''



#all arrays should have same spacial x and y dimensions, so define variables for this to use in for loops
array_length_x = len(image_data_1a[0,:,0])
array_length_y = len(image_data_1a[0,0,:])


#creating an array that indicates where the Ch1 FOV is, so that comparison is only done between pixels with data.
region_indicator = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'data/ch1Arectangle.reg', do_sigma_clip=True, use_dq=False)

#current subchannel data is reprojected to
current_reprojection = '3C'

np.save('Analysis/current_reprojection', current_reprojection)



#%%

'''
INDIVIDUAL PAH FEATURE CONTINUUM SUBTACTION, INTEGRATION AND PLOTS
'''



#first 11.2, then others in ascending wavelength order (11.2 used for comparison in plots)

'''
11.2 feature
'''



#continuum
image_data_112_cont_1 = np.zeros((len(image_data_112_noline[:,0,0]), array_length_x, array_length_y))
points112_1 = [10.61, 10.87, 11.65, 11.79]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_112_cont_1[:,i,j] =\
            bnf.linear_continuum_single_channel(
            wavelengths112, image_data_112_noline[:,i,j], points112_1) #note image_data_112 is built out of things with no lines
        
#bnf.error_check_imager(wavelengths112, image_data_112, 'PDFtime/spectra_checking/112_check_continuum_1.pdf', 10.1, 13.1, 1.5, continuum=image_data_112_cont_1)

np.save('Analysis/image_data_112_cont_1', image_data_112_cont_1)

image_data_112_cont_2 = np.zeros((len(image_data_112_noline[:,0,0]), array_length_x, array_length_y))
points112_2 = [10.61, 10.87, 11.79, 11.89] #last point is a filler for now

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_112_cont_2[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths112, image_data_112_noline[:,i,j], points112_2) #note image_data_112 is built out of things with no lines
        
#bnf.error_check_imager(wavelengths112, image_data_112, 'PDFtime/spectra_checking/112_check_continuum_2.pdf', 10.1, 13.1, 1.5, continuum=image_data_112_cont_2)

np.save('Analysis/image_data_112_cont_2', image_data_112_cont_2)


image_data_112_cont = np.copy(image_data_112_cont_1)

#make an array to keep track of which 11.2 continuum is being used
cont_type_112 = np.ones((array_length_x, array_length_y))

temp_index_1 = np.where(np.round(wavelengths112, 2) == 11.65)[0][0] 
temp_index_2 = np.where(np.round(wavelengths112, 2) == 11.75)[0][0] 

#if continuum decreases significantly (more than 100) between these points, use the cont_2

for i in range(array_length_x):
    for j in range(array_length_y):
        if image_data_112_cont[temp_index_2,i,j] - image_data_112_cont[temp_index_1,i,j] < -25:
            cont_type_112[i,j] += 1
            image_data_112_cont[:,i,j] = image_data_112_cont_2[:,i,j]  

#bnf.error_check_imager(wavelengths112, image_data_112, 'PDFtime/spectra_checking/112_check_continuum.pdf', 10.1, 13.1, 1.5, continuum=image_data_112_cont)

np.save('Analysis/cont_type_112', cont_type_112)
np.save('Analysis/image_data_112_cont', image_data_112_cont)



#integration
pah_intensity_112 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_112 = np.zeros((array_length_x, array_length_y))

#have 11.2 feature go from 11.085 (826) to 11.65 (1250), this seems reasonable but theres arguments for slight adjustment i think
#NOTE: 1250 has the wavelength index of channel 3, use 11.621 (1239) instead, as it is the last wavelength of channel 2
lower_index_112 = 826
upper_index_112 = 1239

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_112[i,j] =\
                bnf.pah_feature_integrator(wavelengths112[lower_index_112:upper_index_112], 
                image_data_112_noline[lower_index_112:upper_index_112,i,j] - image_data_112_cont[lower_index_112:upper_index_112,i,j])
    
print('11.2 feature intensity calculated')
np.save('Analysis/lower_index_112', lower_index_112)
np.save('Analysis/upper_index_112', upper_index_112)
np.save('Analysis/pah_intensity_112', pah_intensity_112)



#error calculation
error_index_112 = 186 #(10.25)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_112[i,j] =\
                bnf.error_finder(wavelengths112, image_data_112_noline[:,i,j] - image_data_112_cont[:,i,j], 11.25, 
                pah_intensity_112[i,j], (lower_index_112, upper_index_112), error_index_112)

print('11.2 feature intensity error calculated')
np.save('Analysis/error_index_112', error_index_112)
np.save('Analysis/pah_intensity_error_112', pah_intensity_error_112)



#plotting
snr_cutoff_112 = 300
bnf.single_feature_imager(pah_intensity_112, pah_intensity_112, pah_intensity_error_112, '11.2', '112', snr_cutoff_112, current_reprojection)

np.save('Analysis/snr_cutoff_112', snr_cutoff_112)
print('11.2 feature done')



'''
5.25 feature
'''



#continuum
image_data_52_cont = np.zeros((len(image_data_57_noline[:,0,0]), array_length_x, array_length_y))
points52 = [5.15, 5.39, 5.55, 5.81] #used to be [5.06, 5.15, 5.39, 5.55]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_52_cont[:,i,j] =\
            bnf.linear_continuum_single_channel(
            wavelengths57, image_data_57_noline[:,i,j], points52)
        
#bnf.error_check_imager(wavelengths57, image_data_57, 'PDFtime/spectra_checking/052_check_continuum.pdf', 5.0, 5.6, 1.5, continuum=image_data_52_cont)

np.save('Analysis/image_data_52_cont', image_data_52_cont)



#integration
pah_intensity_52 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_52 = np.zeros((array_length_x, array_length_y))

#have 5.25 feature go from 5.17 (337) to 5.40 (624), this seems reasonable but theres arguments for slight adjustment i think
#originally went to 16.67 (504)
lower_index_52 = 337
upper_index_52 = 624

#pah integral negative value flag
pah_intensity_52_flag = np.zeros((array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_52[i,j] =\
                bnf.pah_feature_integrator(wavelengths1a[lower_index_52:upper_index_52], 
                image_data_57_noline[lower_index_52:upper_index_52,i,j] - image_data_52_cont[lower_index_52:upper_index_52,i,j])

print('5.25 feature intensity calculated')
np.save('Analysis/lower_index_52', lower_index_52)
np.save('Analysis/upper_index_52', upper_index_52)
np.save('Analysis/pah_intensity_52', pah_intensity_52)



#error calculation
error_index_52 = 650 # (5.42)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_52[i,j] =\
                bnf.error_finder(wavelengths57, image_data_57_noline[:,i,j] - image_data_52_cont[:,i,j], 5.25, 
                pah_intensity_52[i,j], (lower_index_52, upper_index_52), error_index_52)

print('5.25 feature intensity error calculated')
np.save('Analysis/error_index_52', error_index_52)
np.save('Analysis/pah_intensity_error_52', pah_intensity_error_52)



#plotting
snr_cutoff_52 = 20
bnf.single_feature_imager(pah_intensity_52, pah_intensity_112, pah_intensity_error_52, '5.25', '052', snr_cutoff_52, current_reprojection)

np.save('Analysis/snr_cutoff_52', snr_cutoff_52)
print('5.2 feature done')



'''
5.7 feature
'''



#continuum
image_data_57_cont = np.zeros((len(image_data_57_noline[:,0,0]), array_length_x, array_length_y))
points57 = [5.39, 5.55, 5.81, 5.94] #first point is used as an upper bound for 5.25 feature (5.39)

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_57_cont[:,i,j] =\
            bnf.linear_continuum_single_channel(
            wavelengths57, image_data_57_noline[:,i,j], points57) #note image_data_57 is built out of things with no lines

#bnf.error_check_imager(wavelengths57, image_data_57, 'PDFtime/spectra_checking/057_check_continuum.pdf', 5.3, 6.0, 1.5, continuum=image_data_57_cont)

np.save('Analysis/image_data_57_cont', image_data_57_cont)



#integration
pah_intensity_57 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_57 = np.zeros((array_length_x, array_length_y))

#have 5.7 feature go from 5.55 (810) to 5.81 (1130)
lower_index_57 = 810
upper_index_57 = 1130

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_57[i,j] =\
                bnf.pah_feature_integrator(wavelengths57[lower_index_57:upper_index_57], 
                image_data_57_noline[lower_index_57:upper_index_57,i,j] - image_data_57_cont[lower_index_57:upper_index_57,i,j])

print('5.7 feature intensity calculated')
np.save('Analysis/lower_index_57', lower_index_57)
np.save('Analysis/upper_index_57', upper_index_57)
np.save('Analysis/pah_intensity_57', pah_intensity_57)



#error calculation
error_index_57 = np.copy(error_index_52) #(5.42, same as 5.25)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_57[i,j] =\
                bnf.error_finder(wavelengths57, image_data_57_noline[:,i,j] - image_data_57_cont[:,i,j], 5.7, 
                pah_intensity_57[i,j], (lower_index_57, upper_index_57), error_index_57)
            
print('5.7 feature intensity error calculated')
np.save('Analysis/error_index_57', error_index_57)
np.save('Analysis/pah_intensity_error_57', pah_intensity_error_57)



#plotting
snr_cutoff_57 = 20
bnf.single_feature_imager(pah_intensity_57, pah_intensity_112, pah_intensity_error_57, '5.7', '057', snr_cutoff_57, current_reprojection)

np.save('Analysis/snr_cutoff_57', snr_cutoff_57)
print('5.7 feature done')



'''
5.9 feature
'''



#integration
pah_intensity_59 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_59 = np.zeros((array_length_x, array_length_y))

#have 5.9 feature go from 5.81 (1130) to 5.94 (1290)
lower_index_59 = np.copy(upper_index_57)
upper_index_59 = 1290

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_59[i,j] =\
                bnf.pah_feature_integrator(wavelengths57[lower_index_59:upper_index_59], 
                image_data_57_noline[lower_index_59:upper_index_59,i,j] - image_data_57_cont[lower_index_59:upper_index_59,i,j])

print('5.9 feature intensity calculated')
np.save('Analysis/lower_index_59', lower_index_59)
np.save('Analysis/upper_index_59', upper_index_59)
np.save('Analysis/pah_intensity_59', pah_intensity_59)



#error calculation
error_index_59 = np.copy(error_index_57) #(5.42, same as 5.25)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_59[i,j] =\
                bnf.error_finder(wavelengths57, image_data_57_noline[:,i,j] - image_data_57_cont[:,i,j], 5.9, 
                pah_intensity_59[i,j], (lower_index_59, upper_index_59), error_index_59)
            
print('5.9 feature intensity error calculated')
np.save('Analysis/error_index_59', error_index_59)
np.save('Analysis/pah_intensity_error_59', pah_intensity_error_59)



#plotting
snr_cutoff_59 = 20
bnf.single_feature_imager(pah_intensity_59, pah_intensity_112, pah_intensity_error_59, '5.9', '059', snr_cutoff_59, current_reprojection)

np.save('Analysis/snr_cutoff_59', snr_cutoff_59)
print('5.9 feature done')



'''
6.0 feature
'''



#continuum
image_data_62_cont = np.zeros((len(image_data_57_noline[:,0,0]), array_length_x, array_length_y))
points62 = [5.68, 5.945, 6.53, 6.61]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_62_cont[:,i,j] =\
            bnf.linear_continuum_single_channel(
            wavelengths57, image_data_57_noline[:,i,j], points62)
        
#bnf.error_check_imager(wavelengths1b, image_data_1b_noline, 'PDFtime/spectra_checking/062_check_continuum.pdf', 5.7, 6.6, 1.5, continuum=image_data_1b_cont)

np.save('Analysis/image_data_62_cont', image_data_62_cont)



#integration
pah_intensity_60 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_60 = np.zeros((array_length_x, array_length_y))

#have the 6.0 feature go from 5.968 (385) to 6.125 (581) (in combined array it is 1330, 1526)
lower_index_60 = 1330
upper_index_60 = 1526

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_60[i,j] =\
                bnf.pah_feature_integrator(wavelengths57[lower_index_60:upper_index_60], 
                image_data_57_noline[lower_index_60:upper_index_60,i,j] - image_data_62_cont[lower_index_60:upper_index_60,i,j])

print('6.0 feature intensity calculated')
np.save('Analysis/lower_index_60', lower_index_60)
np.save('Analysis/upper_index_60', upper_index_60)
np.save('Analysis/pah_intensity_60', pah_intensity_60)
        


#error calculation
error_index_60 = 2057 # (6.55) # was 1110 for old array

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_60[i,j] =\
                bnf.error_finder(wavelengths57, image_data_57_noline[:,i,j] - image_data_62_cont[:,i,j], 6.0, 
                pah_intensity_60[i,j], (lower_index_60, upper_index_60), error_index_60)
            
print('6.0 feature intensity error calculated')
np.save('Analysis/error_index_60', error_index_60)
np.save('Analysis/pah_intensity_error_60', pah_intensity_error_60)



#plotting
snr_cutoff_60 = 50
bnf.single_feature_imager(pah_intensity_60, pah_intensity_112, pah_intensity_error_60, '6.0', '060', snr_cutoff_60, current_reprojection)

np.save('Analysis/snr_cutoff_60', snr_cutoff_60)
print('6.0 feature done')



'''
6.2 feature
'''



#integration
pah_intensity_62 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_62 = np.zeros((array_length_x, array_length_y))

#have the 6.2 feature go from 6.125 (581) to 6.5 (1050)
#note: upper limit changed from 6.6 (1175) to 6.5, due to there being no feature between 6.5 and 6.6
# with new stitched array, upper index is no longer 1050, and is now 1995
lower_index_62 = np.copy(upper_index_60)
upper_index_62 = 1995

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_62[i,j] =\
                bnf.pah_feature_integrator(wavelengths57[lower_index_62:upper_index_62], 
                image_data_57_noline[lower_index_62:upper_index_62,i,j] - image_data_62_cont[lower_index_62:upper_index_62,i,j])
            
print('6.2 feature intensity calculated')
np.save('Analysis/lower_index_62', lower_index_62)
np.save('Analysis/upper_index_62', upper_index_62)
np.save('Analysis/pah_intensity_62', pah_intensity_62)



#error calculation
error_index_62 = np.copy(error_index_60) # (6.55)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_62[i,j] =\
                bnf.error_finder(wavelengths57, image_data_57_noline[:,i,j] - image_data_62_cont[:,i,j], 6.2, 
                pah_intensity_62[i,j], (lower_index_62, upper_index_62), error_index_62)
            
print('6.2 feature intensity error calculated')
np.save('Analysis/error_index_62', error_index_62)
np.save('Analysis/pah_intensity_error_62', pah_intensity_error_62)



#plotting
snr_cutoff_62 = 200
bnf.single_feature_imager(pah_intensity_62, pah_intensity_112, pah_intensity_error_62, '6.2', '062', snr_cutoff_62, current_reprojection)

np.save('Analysis/snr_cutoff_62', snr_cutoff_62)
print('6.2 feature done')



'''
6.0 and 6.2 feature
'''


#currently in the everything code only



'''
6.9 feature
'''


# currently in everything code only



'''
7.7 feature
'''



#continuum
image_data_77_cont = np.zeros((len(image_data_77_noline[:,0,0]), array_length_x, array_length_y))
points77 = [6.55, 7.06, 9.08, 9.30] #last one is a filler value at the moment

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_77_cont[:,i,j] =\
            bnf.linear_continuum_single_channel(
            wavelengths77, image_data_77_noline[:,i,j], points77) #note image_data_77 is built out of things with no lines

#bnf.error_check_imager(wavelengths77, image_data_77, 'PDFtime/spectra_checking/077_check_continuum.pdf', 6.6, 9.5, 1.5, continuum=image_data_77_cont)

np.save('Analysis/image_data_77_cont', image_data_77_cont)



# currently integration in everything code only



'''
8.6 feature
'''



# currently in everything code only



'''
8.6 plateau feature
'''



# currently in everything code only



'''
8.6 feature local continuum
'''



# currently in everything code only



'''
11.0 feature
'''

#integration
pah_intensity_110 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_110 = np.zeros((array_length_x, array_length_y))

#have 11.0 feature go from 10.96 (730) to 11.085 (826)
lower_index_110 = 730
upper_index_110 = np.copy(lower_index_112) #this ensures smooth transition from 11.0 to 11.2 feature

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_110[i,j] =\
                bnf.pah_feature_integrator(wavelengths112[lower_index_110:upper_index_110], 
                image_data_112_noline[lower_index_110:upper_index_110,i,j] - image_data_112_cont[lower_index_110:upper_index_110,i,j])

print('11.0 feature intensity calculated')
np.save('Analysis/lower_index_110', lower_index_110)
np.save('Analysis/upper_index_110', upper_index_110)
np.save('Analysis/pah_intensity_110', pah_intensity_110)



#error calculation
error_index_110 = np.copy(error_index_112) #uses same error

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_110[i,j] =\
                bnf.error_finder(wavelengths112, image_data_112_noline[:,i,j] - image_data_112_cont[:,i,j], 11.0, 
                pah_intensity_110[i,j], (lower_index_110, upper_index_110),  error_index_110)

print('11.0 feature intensity error calculated')
np.save('Analysis/error_index_110', error_index_110)
np.save('Analysis/pah_intensity_error_110', pah_intensity_error_110)



#plotting
snr_cutoff_110 = 25
#bnf.single_feature_imager(pah_intensity_110, pah_intensity_112, pah_intensity_error_110, '11.0', '110', snr_cutoff_110, current_reprojection)

np.save('Analysis/snr_cutoff_110', snr_cutoff_110)
print('11.0 feature done')



'''
12.0 feature
'''



#continuum
image_data_120_cont = np.zeros((len(image_data_135_noline[:,0,0]), array_length_x, array_length_y))
points120 = [11.65, 11.79, 12.25, 13.08]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_120_cont[:,i,j] =\
            bnf.linear_continuum_single_channel(
            wavelengths135, image_data_135_noline[:,i,j], points120)

#bnf.error_check_imager(wavelengths3a, image_data_3a_noline, 'PDFtime/spectra_checking/120_check_continuum.pdf', 11.6, 12.3, 1.25, continuum=image_data_3a_cont)

np.save('Analysis/image_data_120_cont', image_data_120_cont)



#integration
pah_intensity_120 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_120 = np.zeros((array_length_x, array_length_y))

#have 12.0 feature go from 11.80 (96) to 12.24 (276), this seems reasonable but theres arguments for slight adjustment i think
lower_index_120 = 96
upper_index_120 = 276

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_120[i,j] =\
                bnf.pah_feature_integrator(wavelengths135[lower_index_120:upper_index_120], 
                image_data_135_noline[lower_index_120:upper_index_120,i,j] - image_data_120_cont[lower_index_120:upper_index_120,i,j])

print('12.0 feature intensity calculated')
np.save('Analysis/lower_index_120', lower_index_120)
np.save('Analysis/upper_index_120', upper_index_120)
np.save('Analysis/pah_intensity_120', pah_intensity_120)



#error calculation
error_index_120 = 60 #(11.70)
for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_120[i,j] =\
                bnf.error_finder(wavelengths135, image_data_135_noline[:,i,j] - image_data_120_cont[:,i,j], 12.0, 
                pah_intensity_120[i,j], (lower_index_120, upper_index_120), error_index_120)

print('12.0 feature intensity error calculated')
np.save('Analysis/error_index_120', error_index_120)
np.save('Analysis/pah_intensity_error_120', pah_intensity_error_120)



#plotting
snr_cutoff_120 = 50
#bnf.single_feature_imager(pah_intensity_120, pah_intensity_112, pah_intensity_error_120, '12.0', '120', snr_cutoff_120, current_reprojection)

np.save('Analysis/snr_cutoff_120', snr_cutoff_120)
print('12.0 feature done')



'''
13.5 feature
'''



#continuum
image_data_135_cont = np.zeros((len(image_data_135_noline[:,0,0]), array_length_x, array_length_y))
points135 = [13.21, 13.31, 13.83, 14.00] #last one is a filler value at the moment

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_135_cont[:,i,j] =\
            bnf.linear_continuum_single_channel(
            wavelengths135, image_data_135_noline[:,i,j], points135) #note image_data_135 is built out of things with no lines

#bnf.error_check_imager(wavelengths135, image_data_135, 'PDFtime/spectra_checking/135_check_continuum.pdf', 13.2, 14.0, 1.25, continuum=image_data_135_cont)

np.save('Analysis/image_data_135_cont', image_data_135_cont)



#integration
pah_intensity_135 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_135 = np.zeros((array_length_x, array_length_y))

#have 13.5 feature go from 13.31 (650) to 13.83 (910)

lower_index_135 = 705
upper_index_135 = 910

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_135[i,j] =\
                bnf.pah_feature_integrator(wavelengths135[lower_index_135:upper_index_135], 
                image_data_135_noline[lower_index_135:upper_index_135,i,j] - image_data_135_cont[lower_index_135:upper_index_135,i,j])

print('13.5 feature intensity calculated')
np.save('Analysis/lower_index_135', lower_index_135)
np.save('Analysis/upper_index_135', upper_index_135)
np.save('Analysis/pah_intensity_135', pah_intensity_135)



#error calculation
error_index_135 = 981 #(14.01)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_135[i,j] =\
                bnf.error_finder(wavelengths135, image_data_135_noline[:,i,j] - image_data_135_cont[:,i,j], 13.5, 
                pah_intensity_135[i,j], (lower_index_135, upper_index_135), error_index_135)
            
print('13.5 feature intensity error calculated')
np.save('Analysis/error_index_135', error_index_135)
np.save('Analysis/pah_intensity_error_135', pah_intensity_error_135)



#plotting
snr_cutoff_135 = 50
bnf.single_feature_imager(pah_intensity_135, pah_intensity_112, pah_intensity_error_135, '13.5', '135', snr_cutoff_135, current_reprojection)

np.save('Analysis/snr_cutoff_135', snr_cutoff_135)
print('13.5 feature done')



'''
15.8 feature
'''



# currently in everything code only



'''
16.4 feature
'''

import ButterflyNebulaFunctions as bnf

#continuum
#make 3 continua, as the 16.4 feature seems to be present as a strong version with a red wing, and a weaker version with no red wing.
image_data_3c_cont = np.zeros((len(image_data_3c[:,0,0]), array_length_x, array_length_y))
points164 = [16.12, 16.27, 16.73, 16.85]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_3c_cont[:,i,j] =\
            bnf.linear_continuum_single_channel(
            wavelengths3c, image_data_3c_noline[:,i,j], points164)

#bnf.error_check_imager(wavelengths3c, image_data_3c, 'PDFtime/spectra_checking/164_check_continuum.pdf', 16.1, 16.9, 1, continuum=image_data_3c_cont)

np.save('Analysis/image_data_3c_cont', image_data_3c_cont)



#integration
pah_intensity_164 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_164 = np.zeros((array_length_x, array_length_y))

#have 16.4 feature go from 16.27 (344) to 16.70 (514), this seems reasonable but theres arguments for slight adjustment i think
#originally went to 16.67 (504)
lower_index_164 = 344
upper_index_164 = 514

#pah integral negative value flag
pah_intensity_164_flag = np.zeros((array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_164[i,j] =\
                bnf.pah_feature_integrator(wavelengths3c[lower_index_164:upper_index_164], 
                image_data_3c_noline[lower_index_164:upper_index_164,i,j] - image_data_3c_cont[lower_index_164:upper_index_164,i,j])
            #this one has a lot of integrals that go negative, set them to zero for now, i can properly address this when doing a better
            #continuum fit down the road.
        if pah_intensity_164[i,j] < 0:
            pah_intensity_164[i,j] = 0
            pah_intensity_164_flag[i,j] = 1

print('16.4 feature intensity calculated')
np.save('Analysis/lower_index_164', lower_index_164)
np.save('Analysis/upper_index_164', upper_index_164)
np.save('Analysis/pah_intensity_164', pah_intensity_164)



#error calculation
error_index_164 = 560 # (16.81)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_164[i,j] =\
                bnf.error_finder(wavelengths3c, image_data_3c_noline[:,i,j] - image_data_3c_cont[:,i,j], 16.4, 
                pah_intensity_164[i,j], (lower_index_164, upper_index_164), error_index_164)

print('16.4 feature intensity error calculated')
np.save('Analysis/error_index_164', error_index_164)
np.save('Analysis/pah_intensity_error_164', pah_intensity_error_164)



#plotting
snr_cutoff_164 = 50
bnf.single_feature_imager(pah_intensity_164, pah_intensity_112, pah_intensity_error_164, '16.4', '164', snr_cutoff_164, current_reprojection)

np.save('Analysis/snr_cutoff_164', snr_cutoff_164)
print('16.4 feature done')



'''
CENTROID CALCULATIONS
'''



# currently in everything code only



end = time.time()
length = np.array([end - start])
print(length)
np.savetxt('Analysis/time_analysis', length)
