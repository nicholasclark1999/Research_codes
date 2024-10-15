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

data_57_regridded = np.load('Analysis/data_57_regridded.npy')
data_77_regridded = np.load('Analysis/data_77_regridded.npy')
data_112_regridded = np.load('Analysis/data_112_regridded.npy')
data_135_regridded = np.load('Analysis/data_135_regridded.npy')
data_3c_regridded = np.load('Analysis/data_3c_regridded.npy')
data_230cs_regridded = np.load('Analysis/data_230cs_regridded.npy')

data_57_noline_regridded = np.load('Analysis/data_57_noline_regridded.npy')
data_77_noline_regridded = np.load('Analysis/data_77_noline_regridded.npy')
data_112_noline_regridded = np.load('Analysis/data_112_noline_regridded.npy')
data_135_noline_regridded = np.load('Analysis/data_135_noline_regridded.npy')
data_3c_noline_regridded = np.load('Analysis/data_3c_noline_regridded.npy')
data_230cs_noline_regridded = np.load('Analysis/data_230cs_noline_regridded.npy')

current_reprojection = np.load('Analysis/current_reprojection.npy', allow_pickle=True)
'''
image_data_230cs_cont_1 = np.load('Analysis/image_data_230cs_cont_1.npy', allow_pickle=True)
image_data_230cs_cont_2 = np.load('Analysis/image_data_230cs_cont_2.npy', allow_pickle=True)
cont_type_230cs = np.load('Analysis/cont_type_230cs.npy', allow_pickle=True)
image_data_230cs_cont = np.load('Analysis/image_data_230cs_cont.npy', allow_pickle=True)
image_data_113cs_cont = np.load('Analysis/image_data_113cs_cont.npy', allow_pickle=True)
'''     
image_data_112_cont_1 = np.load('Analysis/image_data_112_cont_1.npy', allow_pickle=True)
image_data_112_cont_2 = np.load('Analysis/image_data_112_cont_2.npy', allow_pickle=True)
cont_type_112 = np.load('Analysis/cont_type_112.npy', allow_pickle=True)
image_data_112_cont = np.load('Analysis/image_data_112_cont.npy', allow_pickle=True)
pah_intensity_112 = np.load('Analysis/pah_intensity_112.npy', allow_pickle=True)
pah_intensity_error_112 = np.load('Analysis/pah_intensity_error_112.npy', allow_pickle=True)
snr_cutoff_112 = np.load('Analysis/snr_cutoff_112.npy', allow_pickle=True)
'''       
pah_intensity_230cs = np.load('Analysis/pah_intensity_230cs.npy', allow_pickle=True)
pah_intensity_error_230cs = np.load('Analysis/pah_intensity_error_230cs.npy', allow_pickle=True)
snr_cutoff_230cs = np.load('Analysis/snr_cutoff_230cs.npy', allow_pickle=True)
'''
image_data_52_cont = np.load('Analysis/image_data_52_cont.npy', allow_pickle=True)
pah_intensity_52 = np.load('Analysis/pah_intensity_52.npy', allow_pickle=True)
pah_intensity_error_52 = np.load('Analysis/pah_intensity_error_52.npy', allow_pickle=True)
snr_cutoff_52 = np.load('Analysis/snr_cutoff_52.npy', allow_pickle=True)  

image_data_57_cont = np.load('Analysis/image_data_57_cont.npy', allow_pickle=True)
pah_intensity_57 = np.load('Analysis/pah_intensity_57.npy', allow_pickle=True)
pah_intensity_error_57 = np.load('Analysis/pah_intensity_error_57.npy', allow_pickle=True)
snr_cutoff_57 = np.load('Analysis/snr_cutoff_57.npy', allow_pickle=True)

pah_intensity_59 = np.load('Analysis/pah_intensity_59.npy', allow_pickle=True)
pah_intensity_error_59 = np.load('Analysis/pah_intensity_error_59.npy', allow_pickle=True)
snr_cutoff_59 = np.load('Analysis/snr_cutoff_59.npy', allow_pickle=True)
        
#image_data_62_cont = np.load('Analysis/image_data_62_cont.npy', allow_pickle=True)
pah_intensity_62 = np.load('Analysis/pah_intensity_62.npy', allow_pickle=True)
pah_intensity_error_62 = np.load('Analysis/pah_intensity_error_62.npy', allow_pickle=True)
snr_cutoff_62 = np.load('Analysis/snr_cutoff_62.npy', allow_pickle=True)
'''
pah_intensity_60_and_62 = np.load('Analysis/pah_intensity_60_and_62.npy', allow_pickle=True)
pah_intensity_error_60_and_62 = np.load('Analysis/pah_intensity_error_60_and_62.npy', allow_pickle=True)
snr_cutoff_60_and_62 = np.load('Analysis/snr_cutoff_60_and_62.npy', allow_pickle=True)
'''
image_data_77_cont = np.load('Analysis/image_data_77_cont.npy', allow_pickle=True)
'''
image_data_77_cont_local = np.load('Analysis/image_data_77_cont_local.npy', allow_pickle=True)

pah_intensity_77 = np.load('Analysis/pah_intensity_77.npy', allow_pickle=True)
pah_intensity_error_77 = np.load('Analysis/pah_intensity_error_77.npy', allow_pickle=True)
snr_cutoff_77 = np.load('Analysis/snr_cutoff_77.npy', allow_pickle=True)

pah_intensity_86 = np.load('Analysis/pah_intensity_86.npy', allow_pickle=True)
pah_intensity_error_86 = np.load('Analysis/pah_intensity_error_86.npy', allow_pickle=True)
snr_cutoff_86 = np.load('Analysis/snr_cutoff_86.npy', allow_pickle=True)

pah_intensity_86_plat = np.load('Analysis/pah_intensity_86_plat.npy', allow_pickle=True)
pah_intensity_error_86_plat = np.load('Analysis/pah_intensity_error_86_plat.npy', allow_pickle=True)
snr_cutoff_86_plat = np.load('Analysis/snr_cutoff_86_plat.npy', allow_pickle=True)

pah_intensity_86_local = np.load('Analysis/pah_intensity_86_local.npy', allow_pickle=True)
pah_intensity_error_86_local = np.load('Analysis/pah_intensity_error_86_local.npy', allow_pickle=True)
snr_cutoff_86_local = np.load('Analysis/snr_cutoff_86_local.npy', allow_pickle=True)
'''
pah_intensity_110 = np.load('Analysis/pah_intensity_110.npy', allow_pickle=True)
pah_intensity_error_110 = np.load('Analysis/pah_intensity_error_110.npy', allow_pickle=True)
snr_cutoff_110 = np.load('Analysis/snr_cutoff_110.npy', allow_pickle=True)

#image_data_120_cont = np.load('Analysis/image_data_120_cont.npy', allow_pickle=True)
pah_intensity_120 = np.load('Analysis/pah_intensity_120.npy', allow_pickle=True)
pah_intensity_error_120 = np.load('Analysis/pah_intensity_error_120.npy', allow_pickle=True)
snr_cutoff_120 = np.load('Analysis/snr_cutoff_120.npy', allow_pickle=True)

image_data_135_cont = np.load('Analysis/image_data_135_cont.npy', allow_pickle=True)
pah_intensity_135 = np.load('Analysis/pah_intensity_135.npy', allow_pickle=True)
pah_intensity_error_135 = np.load('Analysis/pah_intensity_error_135.npy', allow_pickle=True)
snr_cutoff_135 = np.load('Analysis/snr_cutoff_135.npy', allow_pickle=True)

image_data_3c_cont = np.load('Analysis/image_data_3c_cont.npy', allow_pickle=True)
pah_intensity_164 = np.load('Analysis/pah_intensity_164.npy', allow_pickle=True)
pah_intensity_error_164 = np.load('Analysis/pah_intensity_error_164.npy', allow_pickle=True)
snr_cutoff_164 = np.load('Analysis/snr_cutoff_164.npy', allow_pickle=True)                                          

#loading in template PAH spectra, from the Orion bar (JWST)
orion_image_file_miri = np.loadtxt('data/misc/templatesT_MRS_crds1154_add_20231212.dat', skiprows=7)
orion_wavelengths_miri = orion_image_file_miri[:,0]
orion_data_miri = orion_image_file_miri[:,3]

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
PREPARING TEMPLATE SPECTRA
'''



#calculating RMS (for image_data and oops, use rms for 112)
rms_57 = (np.var(data_57_noline_regridded[2057-25:2057+25], axis=0))**0.5 # 6.55 microns
rms_77 = (np.var(data_77_noline_regridded[2470-25:2470+25], axis=0))**0.5 # 9.1 microns
rms_112 = (np.var(data_112_noline_regridded[186-25:186+25], axis=0))**0.5 # 10.25 microns
rms_135 = np.copy(rms_112)
rms_3c = (np.var(data_3c_noline_regridded[560-25:560+25], axis=0))**0.5 # 16.81 microns
rms_230cs = np.copy(rms_112)

#removing very small rms
for i in range(len(rms_57)):
    for j in range(len(rms_57)):
        if rms_57[i,j] < 1:
            rms_57[i,j] = 1
        if rms_77[i,j] < 1:
            rms_77[i,j] = 1
        if rms_112[i,j] < 1:
            rms_112[i,j] = 1
        if rms_135[i,j] < 1:
            rms_135[i,j] = 1
        if rms_3c[i,j] < 1:
            rms_3c[i,j] = 1
        if rms_230cs[i,j] < 1:
            rms_230cs[i,j] = 1

# template spectra points
pah_blob_x_points = [25, 25, 24, 25, 24, 26, 26]
pah_blob_y_points = [15, 16, 15, 14, 14, 15, 14]

enhanced_plateau_x_points = [30, 29, 31, 32, 32, 33] # 28    29, 30
enhanced_plateau_y_points = [41, 41, 41, 41, 40, 40] # 41    42, 42

no60_x_points = [21, 20, 19, 18, 16, 15, 15, 15, 22]
no60_y_points = [9, 10, 11, 11, 14, 15, 16, 17, 9]

enhanced60_x_points = [21, 22, 23, 24, 25, 26, 26, 25, 24, 23] # 22       , 23, 24, 25
enhanced60_y_points = [20, 20, 20, 20, 20, 20, 19, 19, 19, 19] # 19       , 21, 21, 21



# version with lines

pah_blob_data_57 = bnf.weighted_mean_finder_rms_template(data_57_regridded, rms_57, pah_blob_y_points, pah_blob_x_points)
pah_blob_data_77 = bnf.weighted_mean_finder_rms_template(data_77_regridded, rms_77, pah_blob_y_points, pah_blob_x_points)
pah_blob_data_112 = bnf.weighted_mean_finder_rms_template(data_112_regridded, rms_112, pah_blob_y_points, pah_blob_x_points)
pah_blob_data_135 = bnf.weighted_mean_finder_rms_template(data_135_regridded, rms_135, pah_blob_y_points, pah_blob_x_points)
pah_blob_data_3c = bnf.weighted_mean_finder_rms_template(data_3c_regridded, rms_3c, pah_blob_y_points, pah_blob_x_points)
pah_blob_data_230cs = bnf.weighted_mean_finder_rms_template(data_230cs_regridded, rms_230cs, pah_blob_y_points, pah_blob_x_points)

enhanced_plateau_data_57 = bnf.weighted_mean_finder_rms_template(data_57_regridded, rms_57, enhanced_plateau_y_points, enhanced_plateau_x_points)
enhanced_plateau_data_77 = bnf.weighted_mean_finder_rms_template(data_77_regridded, rms_77, enhanced_plateau_y_points, enhanced_plateau_x_points)
enhanced_plateau_data_112 = bnf.weighted_mean_finder_rms_template(data_112_regridded, rms_112, enhanced_plateau_y_points, enhanced_plateau_x_points)
enhanced_plateau_data_135 = bnf.weighted_mean_finder_rms_template(data_135_regridded, rms_135, enhanced_plateau_y_points, enhanced_plateau_x_points)
enhanced_plateau_data_3c = bnf.weighted_mean_finder_rms_template(data_3c_regridded, rms_3c, enhanced_plateau_y_points, enhanced_plateau_x_points)
enhanced_plateau_data_230cs = bnf.weighted_mean_finder_rms_template(data_230cs_regridded, rms_230cs, enhanced_plateau_y_points, enhanced_plateau_x_points)

no60_data_57 = bnf.weighted_mean_finder_rms_template(data_57_regridded, rms_57, no60_y_points, no60_x_points)
no60_data_77 = bnf.weighted_mean_finder_rms_template(data_77_regridded, rms_77, no60_y_points, no60_x_points)
no60_data_112 = bnf.weighted_mean_finder_rms_template(data_112_regridded, rms_112, no60_y_points, no60_x_points)
no60_data_135 = bnf.weighted_mean_finder_rms_template(data_135_regridded, rms_135, no60_y_points, no60_x_points)
no60_data_3c = bnf.weighted_mean_finder_rms_template(data_3c_regridded, rms_3c, no60_y_points, no60_x_points)
no60_data_230cs = bnf.weighted_mean_finder_rms_template(data_230cs_regridded, rms_230cs, no60_y_points, no60_x_points)

enhanced60_data_57 = bnf.weighted_mean_finder_rms_template(data_57_regridded, rms_57, enhanced60_y_points, enhanced60_x_points)
enhanced60_data_77 = bnf.weighted_mean_finder_rms_template(data_77_regridded, rms_77, enhanced60_y_points, enhanced60_x_points)
enhanced60_data_112 = bnf.weighted_mean_finder_rms_template(data_112_regridded, rms_112, enhanced60_y_points, enhanced60_x_points)
enhanced60_data_135 = bnf.weighted_mean_finder_rms_template(data_135_regridded, rms_135, enhanced60_y_points, enhanced60_x_points)
enhanced60_data_3c = bnf.weighted_mean_finder_rms_template(data_3c_regridded, rms_3c, enhanced60_y_points, enhanced60_x_points)
enhanced60_data_230cs = bnf.weighted_mean_finder_rms_template(data_230cs_regridded, rms_230cs, enhanced60_y_points, enhanced60_x_points)



image_data = np.copy(data_230cs_regridded)
wavelengths = np.copy(wavelengths230cs)

pah_blob_data = bnf.weighted_mean_finder_rms_template(image_data, rms_112, pah_blob_y_points, pah_blob_x_points)
enhanced_plateau_data = bnf.weighted_mean_finder_rms_template(image_data, rms_112, enhanced_plateau_y_points, enhanced_plateau_x_points)
no60_data = bnf.weighted_mean_finder_rms_template(image_data, rms_112, no60_y_points, no60_x_points)
enhanced60_data = bnf.weighted_mean_finder_rms_template(image_data, rms_112, enhanced60_y_points, enhanced60_x_points)


'''
wavelengths_oops = wavelengths[5570:7785] # 10.5 - 15.0 
data_oops = image_data[5570:7785]

pah_blob_data_oops = bnf.weighted_mean_finder_rms_template(data_oops, rms_112, pah_blob_y_points, pah_blob_x_points)
enhanced_plateau_data_oops = bnf.weighted_mean_finder_rms_template(data_oops, rms_112, enhanced_plateau_y_points, enhanced_plateau_x_points)
no60_data_oops = bnf.weighted_mean_finder_rms_template(data_oops, rms_112, no60_y_points, no60_x_points)
enhanced60_data_oops = bnf.weighted_mean_finder_rms_template(data_oops, rms_112, enhanced60_y_points, enhanced60_x_points)
'''


# version with no lines

pah_blob_data_57_noline = bnf.weighted_mean_finder_rms_template(data_57_noline_regridded, rms_57, pah_blob_y_points, pah_blob_x_points)
pah_blob_data_77_noline = bnf.weighted_mean_finder_rms_template(data_77_noline_regridded, rms_77, pah_blob_y_points, pah_blob_x_points)
pah_blob_data_112_noline = bnf.weighted_mean_finder_rms_template(data_112_noline_regridded, rms_112, pah_blob_y_points, pah_blob_x_points)
pah_blob_data_135_noline = bnf.weighted_mean_finder_rms_template(data_135_noline_regridded, rms_135, pah_blob_y_points, pah_blob_x_points)
pah_blob_data_3c_noline = bnf.weighted_mean_finder_rms_template(data_3c_noline_regridded, rms_3c, pah_blob_y_points, pah_blob_x_points)
pah_blob_data_230cs_noline = bnf.weighted_mean_finder_rms_template(data_230cs_noline_regridded, rms_230cs, pah_blob_y_points, pah_blob_x_points)

enhanced_plateau_data_57_noline = bnf.weighted_mean_finder_rms_template(data_57_noline_regridded, rms_57, enhanced_plateau_y_points, enhanced_plateau_x_points)
enhanced_plateau_data_77_noline = bnf.weighted_mean_finder_rms_template(data_77_noline_regridded, rms_77, enhanced_plateau_y_points, enhanced_plateau_x_points)
enhanced_plateau_data_112_noline = bnf.weighted_mean_finder_rms_template(data_112_noline_regridded, rms_112, enhanced_plateau_y_points, enhanced_plateau_x_points)
enhanced_plateau_data_135_noline = bnf.weighted_mean_finder_rms_template(data_135_noline_regridded, rms_135, enhanced_plateau_y_points, enhanced_plateau_x_points)
enhanced_plateau_data_3c_noline = bnf.weighted_mean_finder_rms_template(data_3c_noline_regridded, rms_3c, enhanced_plateau_y_points, enhanced_plateau_x_points)
enhanced_plateau_data_230cs_noline = bnf.weighted_mean_finder_rms_template(data_230cs_noline_regridded, rms_230cs, enhanced_plateau_y_points, enhanced_plateau_x_points)

no60_data_57_noline = bnf.weighted_mean_finder_rms_template(data_57_noline_regridded, rms_57, no60_y_points, no60_x_points)
no60_data_77_noline = bnf.weighted_mean_finder_rms_template(data_77_noline_regridded, rms_77, no60_y_points, no60_x_points)
no60_data_112_noline = bnf.weighted_mean_finder_rms_template(data_112_noline_regridded, rms_112, no60_y_points, no60_x_points)
no60_data_135_noline = bnf.weighted_mean_finder_rms_template(data_135_noline_regridded, rms_135, no60_y_points, no60_x_points)
no60_data_3c_noline = bnf.weighted_mean_finder_rms_template(data_3c_noline_regridded, rms_3c, no60_y_points, no60_x_points)
no60_data_230cs_noline = bnf.weighted_mean_finder_rms_template(data_230cs_noline_regridded, rms_230cs, no60_y_points, no60_x_points)

enhanced60_data_57_noline = bnf.weighted_mean_finder_rms_template(data_57_noline_regridded, rms_57, enhanced60_y_points, enhanced60_x_points)
enhanced60_data_77_noline = bnf.weighted_mean_finder_rms_template(data_77_noline_regridded, rms_77, enhanced60_y_points, enhanced60_x_points)
enhanced60_data_112_noline = bnf.weighted_mean_finder_rms_template(data_112_noline_regridded, rms_112, enhanced60_y_points, enhanced60_x_points)
enhanced60_data_135_noline = bnf.weighted_mean_finder_rms_template(data_135_noline_regridded, rms_135, enhanced60_y_points, enhanced60_x_points)
enhanced60_data_3c_noline = bnf.weighted_mean_finder_rms_template(data_3c_noline_regridded, rms_3c, enhanced60_y_points, enhanced60_x_points)
enhanced60_data_230cs_noline = bnf.weighted_mean_finder_rms_template(data_230cs_noline_regridded, rms_230cs, enhanced60_y_points, enhanced60_x_points)



image_data_noline = np.copy(data_230cs_noline_regridded)

pah_blob_data_noline = bnf.weighted_mean_finder_rms_template(image_data_noline, rms_112, pah_blob_y_points, pah_blob_x_points)
enhanced_plateau_data_noline = bnf.weighted_mean_finder_rms_template(image_data_noline, rms_112, enhanced_plateau_y_points, enhanced_plateau_x_points)
no60_data_noline = bnf.weighted_mean_finder_rms_template(image_data_noline, rms_112, no60_y_points, no60_x_points)
enhanced60_data_noline = bnf.weighted_mean_finder_rms_template(image_data_noline, rms_112, enhanced60_y_points, enhanced60_x_points)



array_length_x_2 = len(image_data[0,:,0])
array_length_y_2 = len(image_data[0,0,:])

#%%

# generating array that has only templates highlighted (not reprojected)

x_points = [25, 25, 24, 25, 24, 26, 26, 30, 29, 31, 32, 32, 33, 21, 20, 19, 18, 16, 15, 15, 15, 22, 21, 22, 23, 24, 25, 26, 26, 25, 24, 23]
y_points = [15, 16, 15, 14, 14, 15, 14, 41, 41, 41, 41, 40, 40, 9, 10, 11, 11, 14, 15, 16, 17, 9, 20, 20, 20, 20, 20, 20, 19, 19, 19, 19]



region_array = np.zeros(image_data_1a.shape)
region_array = region_array[0]

for i in range(len(x_points)):
    region_array[2*y_points[i], 2*x_points[i]] = 1
    region_array[2*y_points[i]+1, 2*x_points[i]] = 1
    region_array[2*y_points[i], 2*x_points[i]+1] = 1
    region_array[2*y_points[i]+1, 2*x_points[i]+1] = 1



with fits.open('data/ngc6302_ch1-short_s3d.fits') as hdul:
    hdul[1].data = region_array
    #hdul[1].header["NAXIS3"] = 1
    #print(hdul[1].header)
    #print(hdul[1].header["NAXIS3"])
    hdul.writeto('data/template_spectra.fits', overwrite=True)


#%%

'''
CONTINUUM CODE FOR TEMPLATE SPECTRA
'''

import ButterflyNebulaFunctions as bnf

# making arrays into cubes to work with continuum code

# dust continuum

pah_blob_cube = pah_blob_data_noline[:,np.newaxis,np.newaxis]
enhanced_plateau_cube = enhanced_plateau_data_noline[:,np.newaxis,np.newaxis]
no60_cube = no60_data_noline[:,np.newaxis,np.newaxis]
enhanced60_cube = enhanced60_data_noline[:,np.newaxis,np.newaxis]

cont = bnf.Continua(directory_cube=pah_blob_cube, 
                directory_cube_unc=None, 
                directory_ipac = 'continuum/anchors/anchors_dust.ipac',
                array_waves = wavelengths230cs)
pah_blob_cont = cont.make_continua()

cont = bnf.Continua(directory_cube=enhanced_plateau_cube, 
                directory_cube_unc=None, 
                directory_ipac = 'continuum/anchors/anchors_dust.ipac',
                array_waves = wavelengths230cs)
enhanced_plateau_cont = cont.make_continua()

cont = bnf.Continua(directory_cube=no60_cube, 
                directory_cube_unc=None, 
                directory_ipac = 'continuum/anchors/anchors_dust.ipac',
                array_waves = wavelengths230cs)
no60_cont = cont.make_continua()

cont = bnf.Continua(directory_cube=enhanced60_cube, 
                directory_cube_unc=None, 
                directory_ipac = 'continuum/anchors/anchors_dust.ipac',
                array_waves = wavelengths230cs)
enhanced60_cont = cont.make_continua()
'''
pah_blob_cube = pah_blob_cube[:,0,0]
enhanced_plateau_cube = enhanced_plateau_cube[:,0,0]
no60_cube = no60_cube[:,0,0]
enhanced60_cube = enhanced60_cube[:,0,0]
'''
pah_blob_cont = pah_blob_cont[:,0,0]
enhanced_plateau_cont = enhanced_plateau_cont[:,0,0]
no60_cont = no60_cont[:,0,0]
enhanced60_cont = enhanced60_cont[:,0,0]

np.save('for_charmi/wavelengths', wavelengths230cs)
np.save('for_charmi/pink_data', pah_blob_data)
np.save('for_charmi/pink_cont', pah_blob_cont)
np.save('for_charmi/purple_data', enhanced_plateau_data)
np.save('for_charmi/purple_cont', enhanced_plateau_cont)
np.save('for_charmi/blue_data', enhanced60_data)
np.save('for_charmi/blue_cont', enhanced60_cont)
np.save('for_charmi/orange_data', no60_data)
np.save('for_charmi/orange_cont', no60_cont)

#%%
plt.rcParams.update({'font.size': 14})
plt.plot(wavelengths230cs, no60_data)
plt.plot(wavelengths230cs, no60_cont)
plt.xlim((5, 28))
#plt.ylim((-1000, 10000))

#%%





'''
CONTINUA FOR TEMPLATE SPECTRA
'''



# 5.25 feature
points52 = [5.15, 5.39, 5.55, 5.81]

pah_blob_data_52_cont = bnf.linear_continuum_single_channel(wavelengths57, pah_blob_data_57_noline, points52)
enhanced_plateau_data_52_cont = bnf.linear_continuum_single_channel(wavelengths57, enhanced_plateau_data_57_noline, points52)
no60_data_52_cont = bnf.linear_continuum_single_channel(wavelengths57, no60_data_57_noline, points52)
enhanced60_data_52_cont = bnf.linear_continuum_single_channel(wavelengths57, enhanced60_data_57_noline, points52)

print('5.25 cont resized')

# 5.25, 5.7, 5.9 features
points57 = [5.39, 5.55, 5.81, 5.94]

pah_blob_data_57_cont = bnf.linear_continuum_single_channel(wavelengths57, pah_blob_data_57_noline, points57)
enhanced_plateau_data_57_cont = bnf.linear_continuum_single_channel(wavelengths57, enhanced_plateau_data_57_noline, points57)
no60_data_57_cont = bnf.linear_continuum_single_channel(wavelengths57, no60_data_57_noline, points57)
enhanced60_data_57_cont = bnf.linear_continuum_single_channel(wavelengths57, enhanced60_data_57_noline, points57)

print('5.7 cont resized')

# 6.0, 6.2 features
points62 = [5.68, 5.945, 6.53, 6.61]

pah_blob_data_62_cont = bnf.linear_continuum_single_channel(wavelengths57, pah_blob_data_57_noline, points62)
enhanced_plateau_data_62_cont = bnf.linear_continuum_single_channel(wavelengths57, enhanced_plateau_data_57_noline, points62)
no60_data_62_cont = bnf.linear_continuum_single_channel(wavelengths57, no60_data_57_noline, points62)
enhanced60_data_62_cont = bnf.linear_continuum_single_channel(wavelengths57, enhanced60_data_57_noline, points62)

print('6.2 cont resized')

# 7.7, 8.6 features
points77 = [6.55, 7.06, 9.08, 9.30] #used to be 11.65 instead of 11.70

pah_blob_data_77_cont = bnf.linear_continuum_single_channel(wavelengths77, pah_blob_data_77_noline, points77)
enhanced_plateau_data_77_cont = bnf.linear_continuum_single_channel(wavelengths77, enhanced_plateau_data_77_noline, points77)
no60_data_77_cont = bnf.linear_continuum_single_channel(wavelengths77, no60_data_77_noline, points77)
enhanced60_data_77_cont = bnf.linear_continuum_single_channel(wavelengths77, enhanced60_data_77_noline, points77)

print('7.7 cont resized')

# 11.0, 11.2 features
points112 = [10.61, 10.87, 11.70, 11.79] #used to be 11.65 instead of 11.70

pah_blob_data_112_cont = bnf.linear_continuum_single_channel(wavelengths112, pah_blob_data_112_noline, points112)
enhanced_plateau_data_112_cont = bnf.linear_continuum_single_channel(wavelengths112, enhanced_plateau_data_112_noline, points112)
no60_data_112_cont = bnf.linear_continuum_single_channel(wavelengths112, no60_data_112_noline, points112)
enhanced60_data_112_cont = bnf.linear_continuum_single_channel(wavelengths112, enhanced60_data_112_noline, points112)

print('11.2 cont resized')

# 12.0 feature



points120 = [11.65, 11.79, 12.25, 13.08]

pah_blob_data_120_cont = bnf.linear_continuum_single_channel(wavelengths135, pah_blob_data_135_noline, points120)
enhanced_plateau_data_120_cont = bnf.linear_continuum_single_channel(wavelengths135, enhanced_plateau_data_135_noline, points120)
no60_data_120_cont = bnf.linear_continuum_single_channel(wavelengths135, no60_data_135_noline, points120)
enhanced60_data_120_cont = bnf.linear_continuum_single_channel(wavelengths135, enhanced60_data_135_noline, points120)

print('12.0 cont resized')

# 12.8 feature

points128 = [11.79, 12.25, 13.08, 13.20]

pah_blob_data_128_cont = bnf.linear_continuum_single_channel(wavelengths135, pah_blob_data_135_noline, points128)
enhanced_plateau_data_128_cont = bnf.linear_continuum_single_channel(wavelengths135, enhanced_plateau_data_135_noline, points128)
no60_data_128_cont = bnf.linear_continuum_single_channel(wavelengths135, no60_data_135_noline, points128)
enhanced60_data_128_cont = bnf.linear_continuum_single_channel(wavelengths135, enhanced60_data_135_noline, points128)

# 13.5 feature
points135 = [13.00, 13.17, 13.78, 14.00] # [13.21, 13.31, 13.83, 14.00]

pah_blob_data_135_cont = bnf.linear_continuum_single_channel(wavelengths135, pah_blob_data_135_noline, points135)
enhanced_plateau_data_135_cont = bnf.linear_continuum_single_channel(wavelengths135, enhanced_plateau_data_135_noline, points135)
no60_data_135_cont = bnf.linear_continuum_single_channel(wavelengths135, no60_data_135_noline, points135)
enhanced60_data_135_cont = bnf.linear_continuum_single_channel(wavelengths135, enhanced60_data_135_noline, points135)
      
print('13.5 cont resized')
'''
# oops 

points_oops = [10.61, 10.75, 14.10, 14.50]

pah_blob_data_oops_cont = bnf.linear_continuum_single_channel(wavelengths_oops, pah_blob_data_oops, points_oops)
enhanced_plateau_data_oops_cont = bnf.linear_continuum_single_channel(wavelengths_oops, enhanced_plateau_data_oops, points_oops)
no60_data_oops_cont = bnf.linear_continuum_single_channel(wavelengths_oops, no60_data_oops, points_oops)
enhanced60_data_oops_cont = bnf.linear_continuum_single_channel(wavelengths_oops, enhanced60_data_oops, points_oops)
'''
# 15.8 feature
points158 = [15.40, 15.65, 16.10, 16.15]

pah_blob_data_158_cont = bnf.linear_continuum_single_channel(wavelengths, pah_blob_data_noline, points158)
enhanced_plateau_data_158_cont = bnf.linear_continuum_single_channel(wavelengths, enhanced_plateau_data_noline, points158)
no60_data_158_cont = bnf.linear_continuum_single_channel(wavelengths, no60_data_noline, points158)
enhanced60_data_158_cont = bnf.linear_continuum_single_channel(wavelengths, enhanced60_data_noline, points158)
   
# 16.4 feature
points164 = [16.12, 16.27, 16.73, 16.85]

pah_blob_data_164_cont = bnf.linear_continuum_single_channel(wavelengths3c, pah_blob_data_3c_noline, points164)
enhanced_plateau_data_164_cont = bnf.linear_continuum_single_channel(wavelengths3c, enhanced_plateau_data_3c_noline, points164)
no60_data_164_cont = bnf.linear_continuum_single_channel(wavelengths3c, no60_data_3c_noline, points164)
enhanced60_data_164_cont = bnf.linear_continuum_single_channel(wavelengths3c, enhanced60_data_3c_noline, points164)

# 17.4 feature
points174 = [16.70, 17.20, 17.60, 18.00]

pah_blob_data_174_cont = bnf.linear_continuum_single_channel(wavelengths, pah_blob_data_noline, points174)
enhanced_plateau_data_174_cont = bnf.linear_continuum_single_channel(wavelengths, enhanced_plateau_data_noline, points174)
no60_data_174_cont = bnf.linear_continuum_single_channel(wavelengths, no60_data_noline, points174)
enhanced60_data_174_cont = bnf.linear_continuum_single_channel(wavelengths, enhanced60_data_noline, points174)

print('16.4 cont resized')



#%%

# resizing intensity maps

rms_77_big = (np.var(image_data_77[2470-25:2470+25], axis=0))**0.5 # 9.1 microns
rms_3c_big = (np.var(image_data_3c[560-25:560+25], axis=0))**0.5 # 16.81 microns

pah_164 = bnf.regrid(pah_intensity_164.reshape(1, pah_intensity_164.shape[0], pah_intensity_164.shape[1]), rms_3c_big, 2)
pah_164 = pah_164[0]

pah_62 = bnf.regrid(pah_intensity_62.reshape(1, pah_intensity_62.shape[0], pah_intensity_62.shape[1]), rms_77_big, 2)
pah_62 = pah_62[0]

print('intensity maps resized')







#%%

'''
TEMPLATE SPECTRA LOCATION MAP
'''

# XXX


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

plt.scatter(pah_blob_x_points, pah_blob_y_points, s=100, facecolors='#dc267f', edgecolors='yellow') # pink
plt.scatter([enhanced_plateau_x_points], [enhanced_plateau_y_points], s=100, facecolors='#785ef0', edgecolors='yellow') # purple
plt.scatter(no60_x_points, no60_y_points, s=100, facecolors='#fe6100', edgecolors='yellow') # orange
plt.scatter([enhanced60_x_points], [enhanced60_y_points], s=100, facecolors='#648fff', edgecolors='yellow') # blue
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
'''
no60_image_data = (image_data[:,9,21] +\

           image_data[:,11,18] +\
           image_data[:,10,20] +\
           image_data[:,9,22] +\
           image_data[:,14,16] +\
           image_data[:,15,15] +\
           image_data[:,16,15] +\
           image_data[:,17,15] +\
           image_data[:,11,19])/9
'''
#calculate scaling

#scaling index

scaling_index = np.where(np.round(wavelengths, 3) == 9.250)[0][0]

pah_blob_scaling = 1000/np.median(image_data_noline[scaling_index-10:scaling_index+10,15,25])
enhanced_plateau_scaling = 1000/np.median(image_data_noline[scaling_index-10:scaling_index+10,41,30])
no60_scaling = 1000/np.median(image_data_noline[scaling_index-10:scaling_index+10,9,21])
enhanced60_scaling = 1000/np.median(image_data_noline[scaling_index-10:scaling_index+10,20,21])

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


plt.loglog(wavelengths, pah_blob_scaling*pah_blob_data, color='#dc267f', label='PAH blob (25,15)') # pink
plt.loglog(wavelengths, enhanced_plateau_scaling*enhanced_plateau_data, color='#785ef0', label='Enhanced plateau (30,41)') # purple
plt.loglog(wavelengths, no60_scaling*no60_data, color='#fe6100', label='No 6.0 (21,9)') # orange
plt.loglog(wavelengths, enhanced60_scaling*enhanced60_data, color='#648fff', label='Enhanced 6.0 (21,20)') # blue

plt.loglog(wavelengths, pah_blob_scaling*pah_blob_cont, color='#dc267f', linestyle='dashed')
plt.loglog(wavelengths, enhanced_plateau_scaling*enhanced_plateau_cont, color='#785ef0', linestyle='dashed')
plt.loglog(wavelengths, no60_scaling*no60_cont, color='#fe6100', linestyle='dashed')
plt.loglog(wavelengths, enhanced60_scaling*enhanced60_cont, color='#648fff', linestyle='dashed')

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

pah_blob_scaling = 1
enhanced_plateau_scaling = 1
no60_scaling = 1
enhanced60_scaling = 1
ax = plt.figure('BNF_paper_template_spectra_zoom', figsize=(18,8)).add_subplot(111)
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
ax = plt.figure('BNF_paper_template_spectra_zoom', figsize=(18,8)).add_subplot(111)
# data
plt.loglog(wavelengths, pah_blob_scaling*pah_blob_data, color='#DC267F', label='PAH blob (25,15)') # pink
plt.loglog(wavelengths, enhanced_plateau_scaling*enhanced_plateau_data, color='#785EF0', label='Enhanced plateau (30,41)') # purple
plt.loglog(wavelengths, no60_scaling*no60_data, color='#FE6100', label='No 6.0 (21,9)') # orange
plt.loglog(wavelengths, enhanced60_scaling*enhanced60_data, color='#648FFF', label='Enhanced 6.0 (21,20)') # blue
# continuum
plt.loglog(wavelengths, pah_blob_scaling*pah_blob_cont, color='#DC267F', linestyle='dashed')
plt.loglog(wavelengths, enhanced_plateau_scaling*enhanced_plateau_cont, color='#785EF0', linestyle='dashed')
plt.loglog(wavelengths, no60_scaling*no60_cont, color='#FE6100', linestyle='dashed')
plt.loglog(wavelengths, enhanced60_scaling*enhanced60_cont, color='#648FFF', linestyle='dashed')
# pah feature indicators
plt.loglog([6.2, 6.2], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([8.6, 8.6], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([11.2, 11.2], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([13.5, 13.5], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([16.4, 16.4], [0, 10**10], color='black', linestyle='dashed')
plt.ylim(40, 60000)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.d'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%.d'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
plt.xlim(5.0, 18.0)
axT = ax.secondary_xaxis('top')
axT.set_xticks([6.2, 8.6, 11.2, 13.5, 16.4])
axT.set_xticks([], minor=True)
axT.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axT.tick_params(axis='x', which='major', length=10, width=4)
plt.savefig('PDFtime/paper/BNF_paper_template_spectra_zoom.pdf', bbox_inches='tight')
plt.show()


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
plt.loglog(wavelengths, enhanced_plateau_scaling*image_data[:,41,30], color='#785ef0', label='Enhanced plateau (30,41)')

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
TEMPLATE SPECTA INDIVIDUAL FEATURES SCALED TO 11.2 VERSION FOR NIRSPEC PROPOSAL
'''

#calculate scaling
scaling_lower_index_112 = np.where(np.round(wavelengths112, 2) == 11.24)[0][0]
scaling_upper_index_112 = np.where(np.round(wavelengths112, 2) == 11.30)[0][0]

pah_blob_scaling_112_index = scaling_lower_index_112 + np.argmax((pah_blob_data_112 - pah_blob_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
pah_blob_scaling_112 = 100/np.max((pah_blob_data_112 - pah_blob_data_112_cont)[pah_blob_scaling_112_index-5:pah_blob_scaling_112_index+5])

enhanced_plateau_scaling_112_index = scaling_lower_index_112 + np.argmax((enhanced_plateau_data_112 - enhanced_plateau_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
enhanced_plateau_scaling_112 = 100/np.max((enhanced_plateau_data_112 - enhanced_plateau_data_112_cont)[enhanced_plateau_scaling_112_index-5:enhanced_plateau_scaling_112_index+5])

no60_scaling_112_index = scaling_lower_index_112 + np.argmax((no60_data_112 - no60_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
no60_scaling_112 = 100/np.median((no60_data_112 - no60_data_112_cont)[no60_scaling_112_index-5:no60_scaling_112_index+5])

enhanced60_scaling_112_index = scaling_lower_index_112 + np.argmax((enhanced60_data_112 - enhanced60_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
enhanced60_scaling_112 = 100/np.max((enhanced60_data_112 - enhanced60_data_112_cont)[enhanced60_scaling_112_index-5:enhanced60_scaling_112_index+5])

orion_cont_62 = bnf.linear_continuum_single_channel(orion_wavelengths_miri, orion_data_miri, points62)
orion_cont_112 = bnf.linear_continuum_single_channel(orion_wavelengths_miri, orion_data_miri, points112)

scaling_lower_index_112_orion = np.where(np.round(orion_wavelengths_miri, 2) == 11.18)[0][0]
scaling_upper_index_112_orion = np.where(np.round(orion_wavelengths_miri, 2) == 11.30)[0][0]

orion_scaling_112_index = scaling_lower_index_112_orion + np.argmax((orion_data_miri - orion_cont_112)[scaling_lower_index_112_orion:scaling_upper_index_112_orion])
orion_scaling_112 = 100/np.max((orion_data_miri - orion_cont_112)[orion_scaling_112_index-5:orion_scaling_112_index+5])



ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(9,18)).add_subplot(111)

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
6.2
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(9,18)).add_subplot(211)

#plt.title('6.0 and 6.2 features', fontsize=18)

plt.plot(wavelengths57, pah_blob_scaling_112*(pah_blob_data_57 - pah_blob_data_62_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths57, enhanced_plateau_scaling_112*(enhanced_plateau_data_57 - enhanced_plateau_data_62_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths57, no60_scaling_112*(no60_data_57 - no60_data_62_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths57, enhanced60_scaling_112*(enhanced60_data_57 - enhanced60_data_62_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

plt.plot(orion_wavelengths_miri, orion_scaling_112*(orion_data_miri - orion_cont_62), color='black')

plt.ylim((-10,60))
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
11.2
'''


#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(9,18)).add_subplot(212)

#plt.title('11.0 and 11.2 feature', fontsize=18)

plt.plot(wavelengths112, pah_blob_scaling_112*(pah_blob_data_112 - pah_blob_data_112_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths112, enhanced_plateau_scaling_112*(enhanced_plateau_data_112 - enhanced_plateau_data_112_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths112, no60_scaling_112*(no60_data_112 - no60_data_112_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths112, enhanced60_scaling_112*(enhanced60_data_112 - enhanced60_data_112_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

plt.plot(orion_wavelengths_miri, orion_scaling_112*(orion_data_miri - orion_cont_112), color='black')

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

plt.savefig('PDFtime/paper/template_spectra_features_112_scaled_orion.pdf', bbox_inches='tight')
plt.show()

#%%



'''
TEMPLATE SPECTA INDIVIDUAL FEATURES SCALED TO 11.2 VERSION
'''

#calculate scaling
scaling_lower_index_112 = np.where(np.round(wavelengths112, 2) == 11.24)[0][0]
scaling_upper_index_112 = np.where(np.round(wavelengths112, 2) == 11.30)[0][0]

pah_blob_scaling_112_index = scaling_lower_index_112 + np.argmax((pah_blob_data_112 - pah_blob_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
pah_blob_scaling_112 = 100/np.max((pah_blob_data_112 - pah_blob_data_112_cont)[pah_blob_scaling_112_index-5:pah_blob_scaling_112_index+5])

enhanced_plateau_scaling_112_index = scaling_lower_index_112 + np.argmax((enhanced_plateau_data_112 - enhanced_plateau_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
enhanced_plateau_scaling_112 = 100/np.max((enhanced_plateau_data_112 - enhanced_plateau_data_112_cont)[enhanced_plateau_scaling_112_index-5:enhanced_plateau_scaling_112_index+5])

no60_scaling_112_index = scaling_lower_index_112 + np.argmax((no60_data_112 - no60_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
no60_scaling_112 = 100/np.median((no60_data_112 - no60_data_112_cont)[no60_scaling_112_index-5:no60_scaling_112_index+5])

enhanced60_scaling_112_index = scaling_lower_index_112 + np.argmax((enhanced60_data_112 - enhanced60_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
enhanced60_scaling_112 = 100/np.max((enhanced60_data_112 - enhanced60_data_112_cont)[enhanced60_scaling_112_index-5:enhanced60_scaling_112_index+5])



ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(111)

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
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(621)

plt.title('5.25 feature', fontsize=18)

plt.plot(wavelengths57, pah_blob_scaling_112*(pah_blob_data_57 - pah_blob_data_52_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths57, enhanced_plateau_scaling_112*(enhanced_plateau_data_57 - enhanced_plateau_data_52_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths57, no60_scaling_112*(no60_data_57 - no60_data_52_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths57, enhanced60_scaling_112*(enhanced60_data_57 - enhanced60_data_52_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-5,20))
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
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(622)

plt.title('5.7 and 5.9 feature', fontsize=18)

plt.plot(wavelengths57, pah_blob_scaling_112*(pah_blob_data_57 - pah_blob_data_57_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths57, enhanced_plateau_scaling_112*(enhanced_plateau_data_57 - enhanced_plateau_data_57_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths57, no60_scaling_112*(no60_data_57 - no60_data_57_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths57, enhanced60_scaling_112*(enhanced60_data_57 - enhanced60_data_57_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-5,20))
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
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(623)

plt.title('6.0 and 6.2 features', fontsize=18)

plt.plot(wavelengths57, pah_blob_scaling_112*(pah_blob_data_57 - pah_blob_data_62_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths57, enhanced_plateau_scaling_112*(enhanced_plateau_data_57 - enhanced_plateau_data_62_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths57, no60_scaling_112*(no60_data_57 - no60_data_62_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths57, enhanced60_scaling_112*(enhanced60_data_57 - enhanced60_data_62_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,80))
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
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(624)

plt.title('7-9 features', fontsize=18)

plt.plot(wavelengths77, pah_blob_scaling_112*(pah_blob_data_77 - pah_blob_data_77_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths77, enhanced_plateau_scaling_112*(enhanced_plateau_data_77 - enhanced_plateau_data_77_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths77, no60_scaling_112*(no60_data_77 - no60_data_77_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths77, enhanced60_scaling_112*(enhanced60_data_77 - enhanced60_data_77_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,80))

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
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(625)

plt.title('11.0 and 11.2 feature', fontsize=18)

plt.plot(wavelengths112, pah_blob_scaling_112*(pah_blob_data_112 - pah_blob_data_112_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths112, enhanced_plateau_scaling_112*(enhanced_plateau_data_112 - enhanced_plateau_data_112_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths112, no60_scaling_112*(no60_data_112 - no60_data_112_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths112, enhanced60_scaling_112*(enhanced60_data_112 - enhanced60_data_112_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

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
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(626)

plt.title('12.0 feature', fontsize=18)

plt.plot(wavelengths135, pah_blob_scaling_112*(pah_blob_data_135 - pah_blob_data_120_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths135, enhanced_plateau_scaling_112*(enhanced_plateau_data_135 - enhanced_plateau_data_120_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths135, no60_scaling_112*(no60_data_135 - no60_data_120_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths135, enhanced60_scaling_112*(enhanced60_data_135 - enhanced60_data_120_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,20))
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
12.8
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(627)

plt.title('12.8 feature', fontsize=18)

plt.plot(wavelengths135, pah_blob_scaling_112*(pah_blob_data_135 - pah_blob_data_128_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths135, enhanced_plateau_scaling_112*(enhanced_plateau_data_135 - enhanced_plateau_data_128_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths135, no60_scaling_112*(no60_data_135 - no60_data_128_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths135, enhanced60_scaling_112*(enhanced60_data_135 - enhanced60_data_128_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,60))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(12.2, 13.2, 0.2))
plt.yticks()
plt.xlim(12.2, 13.2)
#plt.legend()
plt.show()

'''
13.5
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(628)

plt.title('13.5 feature', fontsize=18)

plt.plot(wavelengths135, pah_blob_scaling_112*(pah_blob_data_135 - pah_blob_data_135_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths135, enhanced_plateau_scaling_112*(enhanced_plateau_data_135 - enhanced_plateau_data_135_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths135, no60_scaling_112*(no60_data_135 - no60_data_135_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths135, enhanced60_scaling_112*(enhanced60_data_135 - enhanced60_data_135_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,20))

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
OOPS
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(629)

plt.title('OOPS plateau', fontsize=18)

plt.plot(wavelengths, pah_blob_scaling_112*(pah_blob_data - pah_blob_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths, enhanced_plateau_scaling_112*(enhanced_plateau_data - enhanced_plateau_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths, no60_scaling_112*(no60_data - no60_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths, enhanced60_scaling_112*(enhanced60_data - enhanced60_cont), color='#648fff', label='Enhanced 6.0 (21,20)')
plt.plot(wavelengths, 0*pah_blob_data, color='black')

#plt.ylim((-10,100))
plt.ylim((-10,50))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.7, 14.7, 1.0))
plt.yticks()
plt.xlim(10.7, 14.7)
#plt.xlim(13.0, 14.5)
#plt.legend()
plt.show()

'''
15.8
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(6,2,10)

plt.title('15.8 feature', fontsize=18)

plt.plot(wavelengths, pah_blob_scaling_112*(pah_blob_data - pah_blob_data_158_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths, enhanced_plateau_scaling_112*(enhanced_plateau_data - enhanced_plateau_data_158_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths, no60_scaling_112*(no60_data - no60_data_158_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths, pah_blob_scaling_112*(enhanced60_data - enhanced60_data_158_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-2,10))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(15.4, 16.2, 0.2))
plt.yticks()
plt.xlim(15.4, 16.2)
#plt.legend()
plt.show()

'''
16.4
'''
# XXX
#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(6,2,11)

plt.title('16.4 feature', fontsize=18)

plt.plot(wavelengths3c, pah_blob_scaling_112*(pah_blob_data_3c - pah_blob_data_164_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths3c, enhanced_plateau_scaling_112*(enhanced_plateau_data_3c - enhanced_plateau_data_164_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths3c, no60_scaling_112*(no60_data_3c - no60_data_164_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths3c, pah_blob_scaling_112*(enhanced60_data_3c - enhanced60_data_164_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-10,40))
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

'''
17.4
'''

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features_112_scaled', figsize=(18,54)).add_subplot(6,2,12)

plt.title('17.4 feature', fontsize=18)

plt.plot(wavelengths, pah_blob_scaling_112*(pah_blob_data - pah_blob_data_174_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths, enhanced_plateau_scaling_112*(enhanced_plateau_data - enhanced_plateau_data_174_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths, no60_scaling_112*(no60_data - no60_data_174_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths, pah_blob_scaling_112*(enhanced60_data - enhanced60_data_174_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

plt.ylim((-2,10))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(16.8, 17.8, 0.2))
plt.yticks()
plt.xlim(16.8, 17.8)
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

pah_blob_scaling_52_index = scaling_lower_index_52 + np.argmax((pah_blob_data_57_noline - pah_blob_data_52_cont)[scaling_lower_index_52:scaling_upper_index_52])
pah_blob_scaling_52 = 100/np.median((pah_blob_data_57_noline - pah_blob_data_52_cont)[pah_blob_scaling_52_index-5:pah_blob_scaling_52_index+5])

enhanced_plateau_scaling_52_index = scaling_lower_index_52 + np.argmax((enhanced_plateau_data_57_noline - enhanced_plateau_data_52_cont)[scaling_lower_index_52:scaling_upper_index_52])
enhanced_plateau_scaling_52 = 100/np.median((enhanced_plateau_data_57_noline - enhanced_plateau_data_52_cont)[enhanced_plateau_scaling_52_index-5:enhanced_plateau_scaling_52_index+5])

no60_scaling_52_index = scaling_lower_index_52 + np.argmax((no60_data_57_noline - no60_data_52_cont)[scaling_lower_index_52:scaling_upper_index_52])
no60_scaling_52 = 100/np.median((no60_data_57_noline - no60_data_52_cont)[no60_scaling_52_index-5:no60_scaling_52_index+5])

enhanced60_scaling_52_index = scaling_lower_index_52 + np.argmax((enhanced60_data_57_noline - enhanced60_data_52_cont)[scaling_lower_index_52:scaling_upper_index_52])
enhanced60_scaling_52 = 100/np.median((enhanced60_data_57_noline - enhanced60_data_52_cont)[enhanced60_scaling_52_index-5:enhanced60_scaling_52_index+5])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(421)

plt.title('5.25 feature, scaled to 5.25 feature peak', fontsize=18)

plt.plot(wavelengths57, pah_blob_scaling_52*(pah_blob_data_57 - pah_blob_data_52_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths57, enhanced_plateau_scaling_52*(enhanced_plateau_data_57 - enhanced_plateau_data_52_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths57, pah_blob_scaling_52*(no60_data_57 - no60_data_52_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths57, enhanced60_scaling_52*(enhanced60_data_57 - enhanced60_data_52_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

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
5.7 (but using 5.9)
'''

#calculate scaling
scaling_lower_index_57 = np.where(np.round(wavelengths57, 2) == 5.85)[0][0]
scaling_upper_index_57 = np.where(np.round(wavelengths57, 2) == 5.92)[0][0]

pah_blob_scaling_57_index = scaling_lower_index_57 + np.argmax((pah_blob_data_57_noline - pah_blob_data_57_cont)[scaling_lower_index_57:scaling_upper_index_57])
pah_blob_scaling_57 = 100/np.median((pah_blob_data_57_noline - pah_blob_data_57_cont)[pah_blob_scaling_57_index-5:pah_blob_scaling_57_index+5])

enhanced_plateau_scaling_57_index = scaling_lower_index_57 + np.argmax((enhanced_plateau_data_57_noline - enhanced_plateau_data_57_cont)[scaling_lower_index_57:scaling_upper_index_57])
enhanced_plateau_scaling_57 = 100/np.median((enhanced_plateau_data_57_noline - enhanced_plateau_data_57_cont)[enhanced_plateau_scaling_57_index-5:enhanced_plateau_scaling_57_index+5])

no60_scaling_57_index = scaling_lower_index_57 + np.argmax((no60_data_57_noline - no60_data_57_cont)[scaling_lower_index_57:scaling_upper_index_57])
no60_scaling_57 = 100/np.median((no60_data_57_noline - no60_data_57_cont)[no60_scaling_57_index-5:no60_scaling_57_index+5])

enhanced60_scaling_57_index = scaling_lower_index_57 + np.argmax((enhanced60_data_57_noline - enhanced60_data_57_cont)[scaling_lower_index_57:scaling_upper_index_57])
enhanced60_scaling_57 = 100/np.median((enhanced60_data_57_noline - enhanced60_data_57_cont)[enhanced60_scaling_57_index-5:enhanced60_scaling_57_index+5])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(422)

plt.title('5.7 and 5.9 feature, scaled to 5.9 feature peak', fontsize=18)

plt.plot(wavelengths57, pah_blob_scaling_57*(pah_blob_data_57 - pah_blob_data_57_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths57, enhanced_plateau_scaling_57*(enhanced_plateau_data_57 - enhanced_plateau_data_57_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths57, pah_blob_scaling_57*(no60_data_57 - no60_data_57_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths57, enhanced60_scaling_57*(enhanced60_data_57 - enhanced60_data_57_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

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
scaling_lower_index_62 = np.where(np.round(wavelengths57, 2) == 6.15)[0][0]
scaling_upper_index_62 = np.where(np.round(wavelengths57, 2) == 6.25)[0][0]

pah_blob_scaling_62_index = scaling_lower_index_62 + np.argmax((pah_blob_data_57_noline - pah_blob_data_62_cont)[scaling_lower_index_62:scaling_upper_index_62])
pah_blob_scaling_62 = 100/np.max((pah_blob_data_57_noline - pah_blob_data_62_cont)[pah_blob_scaling_62_index-5:pah_blob_scaling_62_index+5])

enhanced_plateau_scaling_62_index = scaling_lower_index_62 + np.argmax((enhanced_plateau_data_57_noline - enhanced_plateau_data_62_cont)[scaling_lower_index_62:scaling_upper_index_62])
enhanced_plateau_scaling_62 = 100/np.max((enhanced_plateau_data_57_noline - enhanced_plateau_data_62_cont)[enhanced_plateau_scaling_62_index-5:enhanced_plateau_scaling_62_index+5])

no60_scaling_62_index = scaling_lower_index_62 + np.argmax((no60_data_57_noline - no60_data_62_cont)[scaling_lower_index_62:scaling_upper_index_62])
no60_scaling_62 = 100/np.median((no60_data_57_noline - no60_data_62_cont)[no60_scaling_62_index-5:no60_scaling_62_index+5])

enhanced60_scaling_62_index = scaling_lower_index_62 + np.argmax((enhanced60_data_57_noline - enhanced60_data_62_cont)[scaling_lower_index_62:scaling_upper_index_62])
enhanced60_scaling_62 = 100/np.max((enhanced60_data_57_noline - enhanced60_data_62_cont)[enhanced60_scaling_62_index-5:enhanced60_scaling_62_index+5])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(423)

plt.title('6.0 and 6.2 feature, scaled to 6.2 feature peak', fontsize=18)

plt.plot(wavelengths57, pah_blob_scaling_62*(pah_blob_data_57 - pah_blob_data_62_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths57, enhanced_plateau_scaling_62*(enhanced_plateau_data_57 - enhanced_plateau_data_62_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths57, no60_scaling_62*(no60_data_57 - no60_data_62_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths57, enhanced60_scaling_62*(enhanced60_data_57 - enhanced60_data_62_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

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

pah_blob_scaling_77_index = scaling_lower_index_77 + np.argmax((pah_blob_data_77_noline - pah_blob_data_77_cont)[scaling_lower_index_77:scaling_upper_index_77])
pah_blob_scaling_77 = 100/np.max((pah_blob_data_77_noline - pah_blob_data_77_cont)[pah_blob_scaling_77_index-5:pah_blob_scaling_77_index+5])

enhanced_plateau_scaling_77_index = scaling_lower_index_77 + np.argmax((enhanced_plateau_data_77_noline - enhanced_plateau_data_77_cont)[scaling_lower_index_77:scaling_upper_index_77])
enhanced_plateau_scaling_77 = 100/np.max((enhanced_plateau_data_77_noline - enhanced_plateau_data_77_cont)[enhanced_plateau_scaling_77_index-5:enhanced_plateau_scaling_77_index+5])

no60_scaling_77_index = scaling_lower_index_77 + np.argmax((no60_data_77_noline - no60_data_77_cont)[scaling_lower_index_77:scaling_upper_index_77])
no60_scaling_77 = 100/np.median((no60_data_77_noline - no60_data_77_cont)[no60_scaling_77_index-5:no60_scaling_77_index+5])

enhanced60_scaling_77_index = scaling_lower_index_77 + np.argmax((enhanced60_data_77_noline - enhanced60_data_77_cont)[scaling_lower_index_77:scaling_upper_index_77])
enhanced60_scaling_77 = 100/np.max((enhanced60_data_77_noline - enhanced60_data_77_cont)[enhanced60_scaling_77_index-5:enhanced60_scaling_77_index+5])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(424)

plt.title('7-9 features, scaled to 8.6', fontsize=18)

plt.plot(wavelengths77, pah_blob_scaling_77*(pah_blob_data_77 - pah_blob_data_77_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths77, enhanced_plateau_scaling_77*(enhanced_plateau_data_77 - enhanced_plateau_data_77_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths77, no60_scaling_77*(no60_data_77 - no60_data_77_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths77, enhanced60_scaling_77*(enhanced60_data_77 - enhanced60_data_77_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

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

pah_blob_scaling_112_index = scaling_lower_index_112 + np.argmax((pah_blob_data_112_noline - pah_blob_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
pah_blob_scaling_112 = 100/np.max((pah_blob_data_112_noline - pah_blob_data_112_cont)[pah_blob_scaling_112_index-5:pah_blob_scaling_112_index+5])

enhanced_plateau_scaling_112_index = scaling_lower_index_112 + np.argmax((enhanced_plateau_data_112_noline - enhanced_plateau_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
enhanced_plateau_scaling_112 = 100/np.max((enhanced_plateau_data_112_noline - enhanced_plateau_data_112_cont)[enhanced_plateau_scaling_112_index-5:enhanced_plateau_scaling_112_index+5])

no60_scaling_112_index = scaling_lower_index_112 + np.argmax((no60_data_112_noline - no60_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
no60_scaling_112 = 100/np.median((no60_data_112_noline - no60_data_112_cont)[no60_scaling_112_index-5:no60_scaling_112_index+5])

enhanced60_scaling_112_index = scaling_lower_index_112 + np.argmax((enhanced60_data_112_noline - enhanced60_data_112_cont)[scaling_lower_index_112:scaling_upper_index_112])
enhanced60_scaling_112 = 100/np.max((enhanced60_data_112_noline - enhanced60_data_112_cont)[enhanced60_scaling_112_index-5:enhanced60_scaling_112_index+5])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(425)

plt.title('11.0 and 11.2 feature, scaled to 11.2 feature peak', fontsize=18)

plt.plot(wavelengths112, pah_blob_scaling_112*(pah_blob_data_112 - pah_blob_data_112_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths112, enhanced_plateau_scaling_112*(enhanced_plateau_data_112 - enhanced_plateau_data_112_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths112, no60_scaling_112*(no60_data_112 - no60_data_112_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths112, enhanced60_scaling_112*(enhanced60_data_112 - enhanced60_data_112_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

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
scaling_lower_index_120 = np.where(np.round(wavelengths135, 2) == 11.9)[0][0]
scaling_upper_index_120 = np.where(np.round(wavelengths135, 2) == 12.0)[0][0]

pah_blob_scaling_120_index = scaling_lower_index_120 + np.argmax((pah_blob_data_135_noline - pah_blob_data_120_cont)[scaling_lower_index_120:scaling_upper_index_120])
pah_blob_scaling_120 = 100/np.median((pah_blob_data_135_noline - pah_blob_data_120_cont)[pah_blob_scaling_120_index-5:pah_blob_scaling_120_index+5])

enhanced_plateau_scaling_120_index = scaling_lower_index_120 + np.argmax((enhanced_plateau_data_135_noline - enhanced_plateau_data_120_cont)[scaling_lower_index_120:scaling_upper_index_120])
enhanced_plateau_scaling_120 = 100/np.median((enhanced_plateau_data_135_noline - enhanced_plateau_data_120_cont)[enhanced_plateau_scaling_120_index-5:enhanced_plateau_scaling_120_index+5])

no60_scaling_120_index = scaling_lower_index_120 + np.argmax((no60_data_135_noline - no60_data_120_cont)[scaling_lower_index_120:scaling_upper_index_120])
no60_scaling_120 = 100/np.median((no60_data_135_noline - no60_data_120_cont)[no60_scaling_120_index-5:no60_scaling_120_index+5])

enhanced60_scaling_120_index = scaling_lower_index_120 + np.argmax((enhanced60_data_135_noline - enhanced60_data_120_cont)[scaling_lower_index_120:scaling_upper_index_120])
enhanced60_scaling_120 = 100/np.median((enhanced60_data_135_noline - enhanced60_data_120_cont)[enhanced60_scaling_120_index-5:enhanced60_scaling_120_index+5])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(426)

plt.title('12.0 feature, scaled to 12.0 feature peak', fontsize=18)

plt.plot(wavelengths135, pah_blob_scaling_120*(pah_blob_data_135 - pah_blob_data_120_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths135, enhanced_plateau_scaling_120*(enhanced_plateau_data_135 - enhanced_plateau_data_120_cont), color='#785ef0', label='Enhanced plateau (30,41)')
#plt.plot(wavelengths3a, no60_scaling_120*(data_3a - data_3a_cont)[:,9,21], color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths135, no60_scaling_120*(no60_data_135 - no60_data_120_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths135, enhanced60_scaling_120*(enhanced60_data_135 - enhanced60_data_120_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

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
scaling_lower_index_135 = np.where(np.round(wavelengths135, 2) == 13.50)[0][0]
scaling_upper_index_135 = np.where(np.round(wavelengths135, 2) == 13.65)[0][0]

pah_blob_scaling_135_index = scaling_lower_index_135 + np.argmax((pah_blob_data_135_noline - pah_blob_data_135_cont)[scaling_lower_index_135:scaling_upper_index_135])
pah_blob_scaling_135 = 100/np.median((pah_blob_data_135_noline - pah_blob_data_135_cont)[pah_blob_scaling_135_index-5:pah_blob_scaling_135_index+5])

enhanced_plateau_scaling_135_index = scaling_lower_index_135 + np.argmax((enhanced_plateau_data_135_noline - enhanced_plateau_data_135_cont)[scaling_lower_index_135:scaling_upper_index_135])
enhanced_plateau_scaling_135 = 100/np.median((enhanced_plateau_data_135_noline - enhanced_plateau_data_135_cont)[enhanced_plateau_scaling_135_index-5:enhanced_plateau_scaling_135_index+5])

no60_scaling_135_index = scaling_lower_index_135 + np.argmax((no60_data_135_noline - no60_data_135_cont)[scaling_lower_index_135:scaling_upper_index_135])
no60_scaling_135 = 100/np.median((no60_data_135_noline - no60_data_135_cont)[no60_scaling_135_index-5:no60_scaling_135_index+5])

enhanced60_scaling_135_index = scaling_lower_index_135 + np.argmax((enhanced60_data_135_noline - enhanced60_data_135_cont)[scaling_lower_index_135:scaling_upper_index_135])
enhanced60_scaling_135 = 100/np.median((enhanced60_data_135_noline - enhanced60_data_135_cont)[enhanced60_scaling_135_index-5:enhanced60_scaling_135_index+5])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(427)

plt.title('13.5 feature, scaled to 13.5 feature peak', fontsize=18)

plt.plot(wavelengths135, pah_blob_scaling_135*(pah_blob_data_135 - pah_blob_data_135_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths135, enhanced_plateau_scaling_135*(enhanced_plateau_data_135 - enhanced_plateau_data_135_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths135, no60_scaling_135*(no60_data_135 - no60_data_135_cont), color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths135, enhanced60_scaling_135*(enhanced60_data_135 - enhanced60_data_135_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

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

pah_blob_scaling_164_index = scaling_lower_index_164 + np.argmax((pah_blob_data_3c_noline - pah_blob_data_164_cont)[scaling_lower_index_164:scaling_upper_index_164])
pah_blob_scaling_164 = 100/np.max((pah_blob_data_3c_noline - pah_blob_data_164_cont)[pah_blob_scaling_164_index-5:pah_blob_scaling_164_index+5])

enhanced_plateau_scaling_164_index = scaling_lower_index_164 + np.argmax((enhanced_plateau_data_3c_noline - enhanced_plateau_data_164_cont)[scaling_lower_index_164:scaling_upper_index_164])
enhanced_plateau_scaling_164 = 100/np.max((enhanced_plateau_data_3c_noline - enhanced_plateau_data_164_cont)[enhanced_plateau_scaling_164_index-5:enhanced_plateau_scaling_164_index+5])

no60_scaling_164_index = scaling_lower_index_164 + np.argmax((no60_data_3c_noline - no60_data_164_cont)[scaling_lower_index_164:scaling_upper_index_164])
no60_scaling_164 = 100/np.median((no60_data_3c_noline - no60_data_164_cont)[no60_scaling_164_index-5:no60_scaling_164_index+5])

enhanced60_scaling_164_index = scaling_lower_index_164 + np.argmax((enhanced60_data_3c_noline - enhanced60_data_164_cont)[scaling_lower_index_164:scaling_upper_index_164])
enhanced60_scaling_164 = 100/np.max((enhanced60_data_3c_noline - enhanced60_data_164_cont)[enhanced60_scaling_164_index-5:enhanced60_scaling_164_index+5])

#making the plot
ax = plt.figure('BNF_paper_template_spectra_features', figsize=(18,36)).add_subplot(428)

plt.title('16.4 feature, scaled to 16.4 feature peak', fontsize=18)

plt.plot(wavelengths3c, pah_blob_scaling_164*(pah_blob_data_3c - pah_blob_data_164_cont), color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths3c, enhanced_plateau_scaling_164*(enhanced_plateau_data_3c - enhanced_plateau_data_164_cont), color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths3c, no60_scaling_164*(no60_data_3c - no60_data_164_cont), color='#fe6100', label='No 6.0 (21,9)')
#plt.plot(wavelengths3c, enhanced60_scaling_164*(data_3c - data_164_cont)[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')
plt.plot(wavelengths3c, pah_blob_scaling_164*(enhanced60_data_3c - enhanced60_data_164_cont), color='#648fff', label='Enhanced 6.0 (21,20)')

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




#%%


'''
TEMPLATE SPECTA FIGURE
'''

# XXX
pah_blob_scaling = 1
enhanced_plateau_scaling = 1
no60_scaling = 1
enhanced60_scaling = 1



ax = plt.figure('BNF_paper_template_spectra', figsize=(18,8)).add_subplot(111)

plt.rcParams.update({'font.size': 14})

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


plt.plot(wavelengths, pah_blob_scaling*pah_blob_data, color='#dc267f', label='PAH blob (25,15)')
plt.plot(wavelengths, enhanced_plateau_scaling*enhanced_plateau_data, color='#785ef0', label='Enhanced plateau (30,41)')
plt.plot(wavelengths, no60_scaling*no60_data, color='#fe6100', label='No 6.0 (21,9)')
plt.plot(wavelengths, enhanced60_scaling*enhanced60_data, color='#648fff', label='Enhanced 6.0 (21,20)')

plt.plot([6.2, 6.2], [0, 10**10], color='black', linestyle='dashed')
plt.plot([8.6, 8.6], [0, 10**10], color='black', linestyle='dashed')
plt.plot([11.2, 11.2], [0, 10**10], color='black', linestyle='dashed')
plt.plot([13.5, 13.5], [0, 10**10], color='black', linestyle='dashed')
plt.plot([16.4, 16.4], [0, 10**10], color='black', linestyle='dashed')




plt.ylim(40, 100000)

ax.tick_params(axis='x', which='major', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)


#ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
plt.xlim(5.0, 20.0)
#plt.legend()



axT.tick_params(axis='x', which='major', length=10, width=4)


plt.show()

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



pah_blob_x_points_old = [25, 25, 24, 25, 24, 26, 26]
pah_blob_y_points_old = [15, 16, 15, 14, 14, 15, 14]

pah_blob_x_points = [26, 26, 25, 25, 25, 24, 24, 24] # 26
pah_blob_y_points = [10, 9, 9, 10, 11, 11, 10, 9] # 11

enhanced_plateau_x_points = [30, 29, 31, 32, 32, 33] # 28    29, 30
enhanced_plateau_y_points = [41, 41, 41, 41, 40, 40] # 41    42, 42

no60_x_points = [21, 20, 19, 18, 16, 15, 15, 15, 22]
no60_y_points = [9, 10, 11, 11, 14, 15, 16, 17, 9]

enhanced60_x_points = [21, 22, 23, 24, 25, 26, 26, 25, 24, 23, 23, 24, 25] # 22
enhanced60_y_points = [20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 21, 21, 21] # 19



x = 21
y = 9

plt.loglog(wavelengths, image_data[:,15,25], color='#648fff', label='Enhanced 6.0 (21,20)')

plt.loglog(wavelengths, 1*image_data[:,y,x], color='red', alpha=0.5, label='No 6.0 (21,9)')

plt.ylim(100, 50000)
plt.xlim(5.0, 28.0)

#plt.ylim(7000, 15000)
#plt.xlim(13.0, 14.0)




plt.loglog([6.2, 6.2], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([8.6, 8.6], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([11.2, 11.2], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([13.5, 13.5], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([16.4, 16.4], [0, 10**10], color='black', linestyle='dashed')

ax.xaxis.set_major_formatter(FormatStrFormatter('%.d'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%.d'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)

axT = ax.secondary_xaxis('top')

axT.set_xticks([6.2, 8.6, 11.2, 13.5, 16.4])
axT.set_xticks([], minor=True)
axT.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axT.tick_params(axis='x', which='major', length=10, width=4)

plt.show()

#%%

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



pah_blob_x_points = [25, 25, 24, 25, 24, 26, 26]
pah_blob_y_points = [15, 16, 15, 14, 14, 15, 14]

enhanced_plateau_x_points = [30, 29, 31, 32, 32, 33] # 28    29, 30
enhanced_plateau_y_points = [41, 41, 41, 41, 40, 40] # 41    42, 42

no60_x_points = [21, 20, 19, 18, 16, 15, 15, 15, 22]
no60_y_points = [9, 10, 11, 11, 14, 15, 16, 17, 9]

enhanced60_x_points = [21, 22, 23, 24, 25, 26, 26, 25, 24, 23, 23, 24, 25] # 22
enhanced60_y_points = [20, 20, 20, 20, 20, 20, 19, 19, 19, 19, 21, 21, 21] # 19



x = 31
y = 10

# check out 38, 32 in more detail (compare to orange)
# check out 35, 21 in more detail (compare to enhanced 60)

#plt.plot(wavelengths135, data_135_135[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')
plt.plot(wavelengths, image_data[:,20,21], color='green', label='Enhanced 6.0 (21,20)')
plt.plot(wavelengths, image_data[:,15,25], color='red', label='Enhanced 6.0 (21,20)')

plt.plot(wavelengths, 3*image_data[:,y,x], color='black', label='Enhanced 6.0 (21,20)')
#plt.plot(wavelengths, enhanced60_image_data, color='#648fff', linestyle='dashed', label='Enhanced 6.0 (21,20)')

#plt.plot(wavelengths, enhanced60_image_data, color='green', linestyle='dashed', label='Enhanced 6.0 (21,20)')
#plt.plot(wavelengths135, enhanced60_data_135_cont, color='purple', linestyle='dashed', label='Enhanced 6.0 (21,20)')


#plt.plot([13.44, 13.44], [0, 10**10], color='black', linestyle='dashed')

#plt.ylim(7000, 13000)
#plt.xlim(5.0, 28.0)

plt.ylim(0, max(image_data[7783,20,21], image_data[7783,15,25], image_data[7783,y,x]))

#plt.ylim(7000, 1.2*data_135[np.where(np.round(wavelengths135, 2) == 13.60)[0][0],y,x])
plt.xlim(5.0, 17.0)

plt.show()

#%%

plt.plot(wavelengths, enhanced60_data)
plt.xlim(10.7, 14.7)
plt.ylim(4000, 18000)

#%%

'''

# data[:,9,20] +
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

'''
#%%
'''
ax = plt.figure('BNF_paper_template_spectra', figsize=(18,8)).add_subplot(111)
plt.rcParams.update({'font.size': 14})

plt.loglog(wavelengths, pah_blob_scaling*image_data[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.loglog(wavelengths, enhanced_plateau_scaling*image_data[:,41,30], color='#785ef0', label='Enhanced plateau (30,41)')
plt.loglog(wavelengths, no60_image_data, color='#fe6100', label='No 6.0 (21,9)')
plt.loglog(wavelengths, enhanced60_scaling*image_data[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')
plt.loglog(wavelengths, north_disk_scaling*image_data[:,33,31], color='black', label='North disk (31,33)')
plt.loglog(wavelengths, central_blob_scaling*image_data[:,30,24], color='gray', label='Central blob (24,30)')


ax.xaxis.set_major_formatter(FormatStrFormatter('%.d'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%.d'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)



plt.loglog([9.35, 9.35], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([9.85, 9.85], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([10.6, 10.6], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([11.6, 11.6], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([13.8, 13.8], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([15.4, 15.4], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([18.0, 18.0], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([20.65, 20.65], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([23.7, 23.7], [0, 10**10], color='black', linestyle='dashed')

#ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
plt.xlim(5.8, 28.0)
plt.ylim(100, 400000)
#plt.legend()

axT = ax.secondary_xaxis('top')

#axT.tick_params(axis='x', which='major', labelbottom='off', labeltop='on', top=True, length=10, width=4)
axT.set_xticks([9.35, 9.85, 10.6, 11.6, 13.8, 15.4, 18.0, 20.65, 23.7])
axT.set_xticks([], minor=True)
axT.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#axT.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
axT.tick_params(axis='x', which='major', length=10, width=4)
#axT.tick_params(axis='x', which='minor', labelbottom='off', labeltop='off', top=False, length=5, width=2)
'''
#%%

plt.plot(wavelengths3b, image_data_3b[:,40, 42])
plt.plot(wavelengths3b, image_data_3b_noline[:,40, 42])

#%%

'''
ax = plt.figure('BNF_paper_template_spectra', figsize=(18,8)).add_subplot(111)
plt.rcParams.update({'font.size': 14})

plt.loglog(wavelengths, pah_blob_scaling*image_data[:,15,25], color='#dc267f', label='PAH blob (25,15)')
plt.loglog(wavelengths, enhanced_plateau_scaling*image_data[:,41,30], color='#785ef0', label='Enhanced plateau (30,41)')
plt.loglog(wavelengths, no60_image_data, color='#fe6100', label='No 6.0 (21,9)')
plt.loglog(wavelengths, enhanced60_scaling*image_data[:,20,21], color='#648fff', label='Enhanced 6.0 (21,20)')
plt.loglog(wavelengths, north_disk_scaling*image_data[:,33,31], color='black', label='North disk (31,33)')
plt.loglog(wavelengths, central_blob_scaling*image_data[:,30,24], color='gray', label='Central blob (24,30)')


ax.xaxis.set_major_formatter(FormatStrFormatter('%.d'))
ax.xaxis.set_minor_formatter(FormatStrFormatter('%.d'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=False, length=10, width=4)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)



plt.loglog([9.35, 9.35], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([9.85, 9.85], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([10.6, 10.6], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([11.6, 11.6], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([13.8, 13.8], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([15.4, 15.4], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([18.0, 18.0], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([20.65, 20.65], [0, 10**10], color='black', linestyle='dashed')
plt.loglog([23.7, 23.7], [0, 10**10], color='black', linestyle='dashed')

#ax.yaxis.set_minor_locator(AutoMinorLocator())
#ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
plt.xlim(9.0, 12.0)
plt.ylim(100, 21000)
#plt.legend()

axT = ax.secondary_xaxis('top')

#axT.tick_params(axis='x', which='major', labelbottom='off', labeltop='on', top=True, length=10, width=4)
axT.set_xticks([9.35, 9.85, 10.6, 11.6, 13.8, 15.4, 18.0, 20.65, 23.7])
axT.set_xticks([], minor=True)
axT.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#axT.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
axT.tick_params(axis='x', which='major', length=10, width=4)
#axT.tick_params(axis='x', which='minor', labelbottom='off', labeltop='off', top=False, length=5, width=2)

'''
