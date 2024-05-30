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




#%%

#all arrays should have same spacial x and y dimensions, so define variables for this to use in for loops
array_length_x = len(image_data_1a[0,:,0])
array_length_y = len(image_data_1a[0,0,:])



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
        
hd100546_lower_index_113 = np.load('Analysis/hd100546_lower_index_113.npy', allow_pickle=True)
hd100546_upper_index_113 = np.load('Analysis/hd100546_upper_index_113.npy', allow_pickle=True)
hd100546_lower_index_230 = np.load('Analysis/hd100546_lower_index_230.npy', allow_pickle=True)
hd100546_upper_index_230 = np.load('Analysis/hd100546_upper_index_230.npy', allow_pickle=True)
hd100546_image_data_113 = np.load('Analysis/hd100546_image_data_113.npy', allow_pickle=True)
hd100546_image_data_230 = np.load('Analysis/hd100546_image_data_230.npy', allow_pickle=True)
hd100546_wavelengths113 = np.load('Analysis/hd100546_wavelengths113.npy', allow_pickle=True)
hd100546_wavelengths230 = np.load('Analysis/hd100546_wavelengths230.npy', allow_pickle=True)
hd100546_image_data_113_cont = np.load('Analysis/hd100546_image_data_113_cont.npy', allow_pickle=True)
hd100546_image_data_230_cont = np.load('Analysis/hd100546_image_data_230_cont.npy', allow_pickle=True)

hale_bopp_image_data_113_cont = np.load('Analysis/hale_bopp_image_data_113_cont.npy', allow_pickle=True)
hale_bopp_image_data_230_cont = np.load('Analysis/hale_bopp_image_data_230_cont.npy', allow_pickle=True)

image_data_230cs_cont_1 = np.load('Analysis/image_data_230cs_cont_1.npy', allow_pickle=True)
image_data_230cs_cont_2 = np.load('Analysis/image_data_230cs_cont_2.npy', allow_pickle=True)
cont_type_230cs = np.load('Analysis/cont_type_230cs.npy', allow_pickle=True)
image_data_230cs_cont = np.load('Analysis/image_data_230cs_cont.npy', allow_pickle=True)
image_data_113cs_cont = np.load('Analysis/image_data_113cs_cont.npy', allow_pickle=True)
        
current_reprojection = np.load('Analysis/current_reprojection.npy', allow_pickle=True)
        
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
        
lower_index_110 = np.load('Analysis/lower_index_110.npy', allow_pickle=True)
upper_index_110 = np.load('Analysis/upper_index_110.npy', allow_pickle=True)
pah_intensity_110 = np.load('Analysis/pah_intensity_110.npy', allow_pickle=True)
error_index_110 = np.load('Analysis/error_index_110.npy', allow_pickle=True)
pah_intensity_error_110 = np.load('Analysis/pah_intensity_error_110.npy', allow_pickle=True)
snr_cutoff_110 = np.load('Analysis/snr_cutoff_110.npy', allow_pickle=True)
        
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
        
image_data_135_cont = np.load('Analysis/image_data_135_cont.npy', allow_pickle=True)
lower_index_135 = np.load('Analysis/lower_index_135.npy', allow_pickle=True)
upper_index_135 = np.load('Analysis/upper_index_135.npy', allow_pickle=True)
pah_intensity_135 = np.load('Analysis/pah_intensity_135.npy', allow_pickle=True)
error_index_135 = np.load('Analysis/error_index_135.npy', allow_pickle=True)
pah_intensity_error_135 = np.load('Analysis/pah_intensity_error_135.npy', allow_pickle=True)
snr_cutoff_135 = np.load('Analysis/snr_cutoff_135.npy', allow_pickle=True)
        
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

lower_index_60 = np.load('Analysis/lower_index_60.npy', allow_pickle=True)
upper_index_60 = np.load('Analysis/upper_index_60.npy', allow_pickle=True)
pah_intensity_60 = np.load('Analysis/pah_intensity_60.npy', allow_pickle=True)
error_index_60 = np.load('Analysis/error_index_60.npy', allow_pickle=True)
pah_intensity_error_60 = np.load('Analysis/pah_intensity_error_60.npy', allow_pickle=True)
snr_cutoff_60 = np.load('Analysis/snr_cutoff_60.npy', allow_pickle=True)

pah_intensity_60_and_62 = np.load('Analysis/pah_intensity_60_and_62.npy', allow_pickle=True)
error_index_60_and_62 = np.load('Analysis/error_index_60_and_62.npy', allow_pickle=True)
pah_intensity_error_60_and_62 = np.load('Analysis/pah_intensity_error_60_and_62.npy', allow_pickle=True)
snr_cutoff_60_and_62 = np.load('Analysis/snr_cutoff_60_and_62.npy', allow_pickle=True)
        
image_data_3a_cont = np.load('Analysis/image_data_3a_cont.npy', allow_pickle=True)
lower_index_120 = np.load('Analysis/lower_index_120.npy', allow_pickle=True)
upper_index_120 = np.load('Analysis/upper_index_120.npy', allow_pickle=True)
pah_intensity_120 = np.load('Analysis/pah_intensity_120.npy', allow_pickle=True)
error_index_120 = np.load('Analysis/error_index_120.npy', allow_pickle=True)
pah_intensity_error_120 = np.load('Analysis/pah_intensity_error_120.npy', allow_pickle=True)
snr_cutoff_120 = np.load('Analysis/snr_cutoff_120.npy', allow_pickle=True)
        
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
                                  
image_data_1a_cont = np.load('Analysis/image_data_1a_cont.npy', allow_pickle=True)
lower_index_52 = np.load('Analysis/lower_index_52.npy', allow_pickle=True)
upper_index_52 = np.load('Analysis/upper_index_52.npy', allow_pickle=True)
pah_intensity_52 = np.load('Analysis/pah_intensity_52.npy', allow_pickle=True)
error_index_52 = np.load('Analysis/error_index_52.npy', allow_pickle=True)
pah_intensity_error_52 = np.load('Analysis/pah_intensity_error_52.npy', allow_pickle=True)
snr_cutoff_52 = np.load('Analysis/snr_cutoff_52.npy', allow_pickle=True)
                                 
image_data_1c_cont = np.load('Analysis/image_data_1c_cont.npy', allow_pickle=True)
lower_index_69 = np.load('Analysis/lower_index_69.npy', allow_pickle=True)
upper_index_69 = np.load('Analysis/upper_index_69.npy', allow_pickle=True)
pah_intensity_69 = np.load('Analysis/pah_intensity_69.npy', allow_pickle=True)
error_index_69 = np.load('Analysis/error_index_69.npy', allow_pickle=True)
pah_intensity_error_69 = np.load('Analysis/pah_intensity_error_69.npy', allow_pickle=True)
snr_cutoff_69 = np.load('Analysis/snr_cutoff_69.npy', allow_pickle=True)
                                 
image_data_158_cont = np.load('Analysis/image_data_158_cont.npy', allow_pickle=True)
lower_index_158 = np.load('Analysis/lower_index_158.npy', allow_pickle=True)
upper_index_158 = np.load('Analysis/upper_index_158.npy', allow_pickle=True)
pah_intensity_158 = np.load('Analysis/pah_intensity_158.npy', allow_pickle=True)
error_index_158 = np.load('Analysis/error_index_158.npy', allow_pickle=True)
pah_intensity_error_158 = np.load('Analysis/pah_intensity_error_158.npy', allow_pickle=True)
snr_cutoff_158 = np.load('Analysis/snr_cutoff_158.npy', allow_pickle=True)
                                  
wavelengths_centroid_62 = np.load('Analysis/wavelengths_centroid_62.npy', allow_pickle=True)
wavelengths_centroid_error_62 = np.load('Analysis/wavelengths_centroid_error_62.npy', allow_pickle=True)
wavelengths_centroid_86_local = np.load('Analysis/wavelengths_centroid_86_local.npy', allow_pickle=True)
wavelengths_centroid_error_86_local = np.load('Analysis/wavelengths_centroid_error_86_local.npy', allow_pickle=True)
wavelengths_centroid_112 = np.load('Analysis/wavelengths_centroid_112.npy', allow_pickle=True)
wavelengths_centroid_error_112 = np.load('Analysis/wavelengths_centroid_error_112.npy', allow_pickle=True)



#%%



'''
REGION FILE MASKING ARRAYS
'''



#creating an array that indicates where the Ch1 FOV is, so that comparison is only done between pixels with data.

region_indicator = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'data/ch1Arectangle.reg', do_sigma_clip=True, use_dq=False)

#%%




#%%


#bounds to update for integrals: 12.0 upper bound now 12.26; 15.8 now 15.65, 16.14

#looks like can do 12.8, current bounds 12.26 to 13.08

#13.5 bounds will likely need to be updated to accomodate some nonsense in the red wing, also its unclear what is continuum and what is 13.5 on the blue wing some of the time

#looks like i can do 14.2, currently bound from 14.21 to 14.77 (seems to be a very broad red wing), will probably need to make multiple continua for the red wing, 14.11 used sometimes

#maybe something around 15.3, would need several continua for it though 








#%%

##################################


'''
FIDGETTING AND BUGTESTING
'''



#creating an array that indicates where the disk is, to compare data inside and outside the disk.

disk_mask = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_disk.reg', do_sigma_clip=True, use_dq=False)

star_mask = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_star.reg', do_sigma_clip=True, use_dq=False)

action_zone_mask = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_action_zone.reg', do_sigma_clip=True, use_dq=False)






central_north_blob_mask = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_central_north_blob.reg', do_sigma_clip=True, use_dq=False)

central_south_blob_mask = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_central_south_blob.reg', do_sigma_clip=True, use_dq=False)


