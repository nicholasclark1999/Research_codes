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



#HD 100546
hd100546_file_loc = 'comparison_data/07200660_sws.fit'
hd100546_image_file = get_pkg_data_filename(hd100546_file_loc)
hd100546_data = fits.getdata(hd100546_image_file, ext=0)
hd100546_image_data = hd100546_data[:,1]
hd100546_wavelengths = hd100546_data[:,0]

#%%

#comet Hale-Bopp
hale_bopp_file_loc_113 = 'comparison_data/ISO1707839139/31500113/swaa31500113.fit'
hale_bopp_image_file_113 = get_pkg_data_filename(hale_bopp_file_loc_113)

with fits.open(hale_bopp_image_file_113) as hdul:
    table_form = Table(hdul[1].data)
    hale_bopp_wavelengths113 = table_form['SWAAWAVE'][:29039]
    hale_bopp_image_data_113 = table_form['SWAAFLUX'][:29039]
    
hale_bopp_file_loc_230 = 'comparison_data/ISO1707839164/31500112/swaa31500112.fit'
hale_bopp_image_file_230 = get_pkg_data_filename(hale_bopp_file_loc_230)

with fits.open(hale_bopp_image_file_230) as hdul:
    table_form = Table(hdul[1].data)
    hale_bopp_wavelengths230 = table_form['SWAAWAVE'][57862:]
    hale_bopp_image_data_230 = table_form['SWAAFLUX'][57862:]
    
#%%

plt.figure()

#plt.plot(hale_bopp_wavelengths113, hale_bopp_image_data_113)
plt.plot(hale_bopp_image_data_230)
#plt.ylim(0, 600)
plt.show()
plt.close()

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


ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(cont_type_112)
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
ax.invert_yaxis()

plt.show()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(central_north_blob_mask)
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
ax.invert_yaxis()

plt.show()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(pah_intensity_62)
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
ax.invert_yaxis()

plt.show()


#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(cont_type_164)
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
ax.invert_yaxis()

plt.show()

#%%

disk_mask_north = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_disk_north.reg', do_sigma_clip=True, use_dq=False)
disk_mask_south = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_disk_south.reg', do_sigma_clip=True, use_dq=False)

ax = plt.figure(figsize=(8,8)).add_subplot(111)
#plt.imshow(disk_mask + disk_mask_north + disk_mask_south)
plt.imshow(pah_intensity_62)
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
ax.invert_yaxis()

plt.show()




#%%


'''
MODIFIED CONTINUA (TO GO IN ANALYSIS)
'''



'''
13.8 feature
'''



#continuum

image_data_3b_cont = np.zeros((len(image_data_3b[:,0,0]), array_length_x, array_length_y))

points138 = [13.56, 13.66, 13.92, 14.02] #first and last indices are filler

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_3b_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths3b, image_data_3b_noline[:,i,j], points138)
        
bnf.error_check_imager(wavelengths3b, image_data_3b_noline, 'PDFtime/spectra_checking/138_check_continuum.pdf', 13.4, 14.5, 1.0, continuum=image_data_3b_cont)

np.save('Analysis/image_data_3b_cont', image_data_3b_cont)

#%%

plt.figure()
plt.plot(wavelengths3b, image_data_3b_noline[:,63,49])
plt.show()
plt.close()

#%%

#integration

pah_intensity_138 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_138 = np.zeros((array_length_x, array_length_y))

#have the 13.8 feature go from 13.65 (125) to 13.86 (207)

lower_index_138 = 125
upper_index_138 = 207

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_138[i,j] = bnf.pah_feature_integrator(wavelengths3b[lower_index_138:upper_index_138], 
                                                            image_data_3b_noline[lower_index_138:upper_index_138,i,j] - image_data_3b_cont[lower_index_138:upper_index_138,i,j])
            
print('13.8 feature intensity calculated')

np.save('Analysis/lower_index_138', lower_index_138)
np.save('Analysis/upper_index_138', upper_index_138)
np.save('Analysis/pah_intensity_138', pah_intensity_138)

#%%

error_index_138 = 265 # (14.00)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_138[i,j] = bnf.error_finder(wavelengths3b, image_data_3b_noline[:,i,j] - image_data_3b_cont[:,i,j], 
                                                           upper_index_138 - lower_index_138, pah_intensity_138[i,j], error_index_138)
            
print('13.8 feature intensity error calculated')

np.save('Analysis/error_index_138', error_index_138)
np.save('Analysis/pah_intensity_error_138', pah_intensity_error_138)

#%%

snr_cutoff_138 = 25

bnf.single_feature_imager(pah_intensity_138, pah_intensity_138, pah_intensity_error_138, '13.8', '138', snr_cutoff_138, current_reprojection)

np.save('Analysis/snr_cutoff_138', snr_cutoff_138)

#%%
'''
ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(pah_intensity_158, vmin=1e-7, vmax=1.2e-6)
plt.colorbar()

#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='yellow')
#disk
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
#central star
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
#action zone
plt.plot([33, 54, 65, 76, 76, 70.5, 47, 33, 30.5, 33], [85, 89, 82, 64, 29, 17, 13, 35, 65, 85], color='C9')


ax.invert_yaxis()
plt.show()
plt.close()
'''
ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('13.8 micron feature (13.65 to 13.86)', fontsize=16)
plt.imshow(pah_intensity_138)
plt.colorbar()
#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

#plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/138_feature_negative.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('13.5 micron feature (13.31 to 13.83), 13.8 removed', fontsize=16)
plt.imshow(pah_intensity_135 - pah_intensity_138)
plt.colorbar()
#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

#plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/135_feature_no_138.png', bbox_inches='tight')
plt.show()
plt.close()



#%%

i = 11
j = 28

plt.figure()
plt.title('index 28, 11')
#plt.plot(wavelengths77, image_data_77[:,i,j])
plt.plot(wavelengths1c, image_data_1c[:,i,j])
plt.plot(wavelengths2a, image_data_2a[:,i,j])
#plt.plot(wavelengths77, image_data_77_lines[:,i,j])
#plt.plot(wavelengths3c, image_data_158_cont[:,i,j])
#plt.plot(wavelengths3c, image_data_3c_cont[:,i,j])
plt.xlim(7.0, 9.0)
plt.ylim(0, 400)
plt.show()
plt.close()



'''
6.2, 6.0 feature
'''

#%%

#continuum, much wider

#combining channels 1A, 1B, and 1C

image_data_62_noline_wide_temp, wavelengths62_wide, overlap1b_wide_temp = bnf.flux_aligner3(wavelengths1a, wavelengths1b, image_data_1a_noline[:,50,50], image_data_1b_noline[:,50,50])
image_data_62_noline_wide_temp_temp, wavelengths62_wide, overlap1b_wide_temp = bnf.flux_aligner3(wavelengths62_wide, wavelengths1c[:600], image_data_62_noline_wide_temp, image_data_1c_noline[:600,50,50])



#using the above to make an array of the correct size to fill
image_data_62_noline_wide_1 = np.zeros((len(image_data_62_noline_wide_temp), array_length_x, array_length_y))

image_data_62_noline_wide = np.zeros((len(image_data_62_noline_wide_temp_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_62_noline_wide_1[:,i,j], wavelengths62_wide_1, overlap1b_noline_wide = bnf.flux_aligner3(wavelengths1a, wavelengths1b, image_data_1a_noline[:,i,j], image_data_1b_noline[:,i,j])
        
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_62_noline_wide[:,i,j], wavelengths62_wide, overlap1b_noline_wide = bnf.flux_aligner3(wavelengths62_wide_1, wavelengths1c[:600], image_data_62_noline_wide_1[:,i,j], image_data_1c_noline[:600,i,j])

print('6.0 and 6.2 features stitching complete')

np.save('Analysis/wavelengths62_wide', wavelengths62_wide)
np.save('Analysis/image_data_62_noline_wide', image_data_62_noline_wide)

#%%

#combining channels 1A, 1B, and 1C (No lines)

image_data_62_wide_temp, wavelengths62_wide, overlap1b_wide_temp = bnf.flux_aligner3(wavelengths1a, wavelengths1b, image_data_1a[:,50,50], image_data_1b[:,50,50])
image_data_62_wide_temp_temp, wavelengths62_wide, overlap1b_wide_temp = bnf.flux_aligner3(wavelengths62_wide, wavelengths1c[:600], image_data_62_wide_temp, image_data_1c[:600,50,50])



#using the above to make an array of the correct size to fill
image_data_62_wide_1 = np.zeros((len(image_data_62_wide_temp), array_length_x, array_length_y))

image_data_62_wide = np.zeros((len(image_data_62_wide_temp_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_62_wide_1[:,i,j], wavelengths62_wide_1, overlap1b_wide = bnf.flux_aligner3(wavelengths1a, wavelengths1b, image_data_1a[:,i,j], image_data_1b[:,i,j])
        
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_62_wide[:,i,j], wavelengths62_wide, overlap1b_wide = bnf.flux_aligner3(wavelengths62_wide_1, wavelengths1c[:600], image_data_62_wide_1[:,i,j], image_data_1c[:600,i,j])

print('6.0 and 6.2 features stitching complete')

np.save('Analysis/wavelengths62_wide', wavelengths62_wide)
np.save('Analysis/image_data_62_wide', image_data_62_wide)

#%%

import ButterflyNebulaFunctions as bnf

image_data_62_cont_wide_1 = np.zeros((len(image_data_62_noline_wide[:,0,0]), array_length_x, array_length_y))

points62_wide_1 = [5.945, 6.59, 6.65, 6.79] #was originally [5.68, 5.945, 6.53, 6.61]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_62_cont_wide_1[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths62_wide, image_data_62_noline_wide[:,i,j], points62_wide_1)
        
bnf.error_check_imager(wavelengths62_wide, image_data_62_noline_wide, 'PDFtime/spectra_checking/062_check_continuum_wide_1.pdf', 5.2, 7.0, 1.5, continuum=image_data_62_cont_wide_1, cont_points=points62_wide_1)

np.save('Analysis/image_data_62_cont_wide_1', image_data_62_cont_wide_1)

#%%

image_data_62_cont_wide_2 = np.zeros((len(image_data_62_noline_wide[:,0,0]), array_length_x, array_length_y))

points62_wide_2 = [5.945, 6.61, 6.79, 6.81] #5.81 is a filler value

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_62_cont_wide_2[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths62_wide, image_data_62_noline_wide[:,i,j], points62_wide_2)
        
bnf.error_check_imager(wavelengths62_wide, image_data_62_noline_wide, 'PDFtime/spectra_checking/062_check_continuum_wide_2.pdf', 5.2, 7.0, 1.5, continuum=image_data_62_cont_wide_2, cont_points=points62_wide_2)

np.save('Analysis/image_data_62_cont_wide_2', image_data_62_cont_wide_2)

#%%

image_data_62_cont_wide = np.copy(image_data_62_cont_wide_1)

#make an array to keep track of which 6.2 continuum is being used
cont_type_62 = np.ones((array_length_x, array_length_y))

#the idea is if if [1:2] is decreasing/increasing, [2:3] should also be increasing/deceasing.
temp_index_1 = np.where(np.round(wavelengths62_wide, 2) == points62_wide_1[1])[0][0] #approx 6.53
temp_index_2 = np.where(np.round(wavelengths62_wide, 2) == points62_wide_1[2])[0][0] #approx 6.61
temp_index_3 = np.where(np.round(wavelengths62_wide, 2) == points62_wide_1[3])[0][0] #approx 6.79

temp_index_4 = int(temp_index_1 + (temp_index_2-temp_index_1)/2)
temp_index_5 = int(temp_index_2 + (temp_index_3-temp_index_2)/2)

'''
for i in range(array_length_x):
    for j in range(array_length_y):
        if (image_data_62_cont_wide_1[temp_index_2,i,j] - image_data_62_cont_wide_1[temp_index_1,i,j] < -10
        and image_data_62_cont_wide_1[temp_index_3,i,j] - image_data_62_cont_wide_1[temp_index_2,i,j] > 10):
            cont_type_62[i,j] += 1
            image_data_62_cont_wide[:,i,j] = image_data_62_cont_wide_2[:,i,j]  
        elif (image_data_62_cont_wide_1[temp_index_2,i,j] - image_data_62_cont_wide_1[temp_index_1,i,j] > 10
        and image_data_62_cont_wide_1[temp_index_3,i,j] - image_data_62_cont_wide_1[temp_index_2,i,j] < -10):
            cont_type_62[i,j] += 1
            image_data_62_cont_wide[:,i,j] = image_data_62_cont_wide_2[:,i,j]
'''

var=20

for i in range(array_length_x):
    for j in range(array_length_y):
        if (np.median(image_data_62_noline_wide[temp_index_1:temp_index_4,i,j]) - np.median(image_data_62_noline_wide[temp_index_4:temp_index_2,i,j]) < var
        and np.median(image_data_62_noline_wide[temp_index_2:temp_index_5,i,j]) - np.median(image_data_62_noline_wide[temp_index_5:temp_index_3,i,j]) > -0):
            cont_type_62[i,j] += 1
            image_data_62_cont_wide[:,i,j] = image_data_62_cont_wide_2[:,i,j]  
        elif (np.median(image_data_62_noline_wide[temp_index_1:temp_index_4,i,j]) - np.median(image_data_62_noline_wide[temp_index_4:temp_index_2,i,j]) > -var
        and np.median(image_data_62_noline_wide[temp_index_2:temp_index_5,i,j]) - np.median(image_data_62_noline_wide[temp_index_5:temp_index_3,i,j]) < 0):
            cont_type_62[i,j] += 1
            image_data_62_cont_wide[:,i,j] = image_data_62_cont_wide_2[:,i,j]

#%% 

i = 24
j = 72
           
plt.figure()
plt.plot(wavelengths62_wide, image_data_62_noline_wide[:,i,j])
plt.plot(wavelengths62_wide, image_data_62_cont_wide_1[:,i,j])
plt.plot(wavelengths62_wide, image_data_62_cont_wide_2[:,i,j])
plt.plot(wavelengths62_wide, image_data_62_cont_wide[:,i,j], alpha=0.5)
plt.show()

#%%
bnf.error_check_imager(wavelengths62_wide, image_data_62_noline_wide, 'PDFtime/spectra_checking/062_check_continuum_wide.pdf', 5.2, 7.0, 1.5, 
                       continuum=image_data_62_cont_wide, conttype=cont_type_62)
#%%

np.save('Analysis/cont_type_62', cont_type_62)
np.save('Analysis/image_data_62_cont_wide', image_data_62_cont_wide)

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('6.2 micron feature continuum types', fontsize=16)
plt.imshow(cont_type_62)
plt.colorbar()
#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

#plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/spectra_checking/062_continuum_type.png', bbox_inches='tight')
plt.show()
#plt.close()

#%%

'''
16.4 feature
'''

#continuum, much wider

#combining channels 1A, 1B, and 1C

image_data_164_noline_wide_temp, wavelengths164_wide, overlap1b_wide_temp = bnf.flux_aligner3(wavelengths3b, wavelengths3c, image_data_3b_noline[:,50,50], image_data_3c_noline[:,50,50])



#using the above to make an array of the correct size to fill
image_data_164_noline_wide = np.zeros((len(image_data_164_noline_wide_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_164_noline_wide[:,i,j], wavelengths164_wide, overlap1b_noline_wide = bnf.flux_aligner3(wavelengths3b, wavelengths3c, image_data_3b_noline[:,i,j], image_data_3c_noline[:,i,j])
      
print('16.4 feature stitching complete')

np.save('Analysis/wavelengths164_wide', wavelengths164_wide)
np.save('Analysis/image_data_164_noline_wide', image_data_164_noline_wide)

#%%

import ButterflyNebulaFunctions as bnf

#continuum

#make 3 continua, as the 16.4 feature seems to be present as a strong version with a red wing, and a weaker version with no red wing.

image_data_164_cont_wide_1 = np.zeros((len(image_data_164_noline_wide[:,0,0]), array_length_x, array_length_y))

points164_1 = [15.45, 16.25, 16.73, 16.85] #originally 16.12, 16.27

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_164_cont_wide_1[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths164_wide, image_data_164_noline_wide[:,i,j], points164_1)
        
bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_wide, 'PDFtime/spectra_checking/164_check_continuum_wide_1.pdf', 14.5, 17.5, 1, continuum=image_data_164_cont_wide_1, cont_points=points164_1)

np.save('Analysis/image_data_164_cont_wide_1', image_data_164_cont_wide_1)



image_data_164_cont_wide_2 = np.zeros((len(image_data_164_noline_wide[:,0,0]), array_length_x, array_length_y))

points164_2 = [15.45, 16.25, 16.63, 16.75]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_164_cont_wide_2[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths164_wide, image_data_164_noline_wide[:,i,j], points164_2)
        
bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_wide, 'PDFtime/spectra_checking/164_check_continuum_wide_2.pdf', 14.5, 17.5, 1, continuum=image_data_164_cont_wide_2, cont_points=points164_2)

np.save('Analysis/image_data_164_cont_wide_2', image_data_164_cont_wide_2)



image_data_164_cont_wide_3 = np.zeros((len(image_data_164_noline_wide[:,0,0]), array_length_x, array_length_y))

points164_3 = [15.45, 16.25, 16.53, 16.65]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_164_cont_wide_3[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths164_wide, image_data_164_noline_wide[:,i,j], points164_3)
        
import ButterflyNebulaFunctions as bnf
bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_wide, 'PDFtime/spectra_checking/164_check_continuum_wide_3.pdf', 14.5, 17.5, 1, continuum=image_data_164_cont_wide_3, cont_points=points164_3)

np.save('Analysis/image_data_164_cont_wide_3', image_data_164_cont_wide_3)


image_data_164_cont_wide = np.copy(image_data_164_cont_wide_1)

#make an array to keep track of which 16.4 continuum is being used
cont_type_164_noline_wide = np.ones((array_length_x, array_length_y))

#note these are named based off of which continuum the point belongs to, and so appear backwards than what would otherwise be expected; [1:2] is expected but this is [2:1]
temp_index_1 = np.where(np.round(wavelengths164_wide, 2) == points164_1[2])[0][0] #approx 16.73
temp_index_2 = np.where(np.round(wavelengths164_wide, 2) == points164_2[2])[0][0] #approx 16.63
temp_index_3 = np.where(np.round(wavelengths164_wide, 2) == points164_3[2])[0][0] #approx 16.53

for i in range(array_length_x):
    for j in range(array_length_y):
        if np.median(image_data_164_noline_wide[temp_index_2:temp_index_1,i,j] - image_data_164_cont_wide[temp_index_2:temp_index_1,i,j]) < 0:
            cont_type_164_noline_wide[i,j] += 1
            image_data_164_cont_wide[:,i,j] = image_data_164_cont_wide_2[:,i,j]
        if np.median(image_data_164_noline_wide[temp_index_3:temp_index_2,i,j] - image_data_164_cont_wide[temp_index_3:temp_index_2,i,j]) < 0:
            cont_type_164_noline_wide[i,j] += 1
            image_data_164_cont_wide[:,i,j] = image_data_164_cont_wide_3[:,i,j]      

bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_wide, 'PDFtime/spectra_checking/164_check_continuum_wide.pdf', 14.5, 17.5, 1, 
                       continuum=image_data_164_cont_wide, conttype=cont_type_164_noline_wide)

np.save('Analysis/cont_type_164_noline_wide', cont_type_164_noline_wide)
np.save('Analysis/image_data_164_cont_wide', image_data_164_cont_wide)

#%%

#attempt at continuum fitting using a ransac regressor



#making a function that uses ransac to fit lines

def line_fitter(wavelength, data, wavelength_plot):
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
    wavelength_plot = wavelength_plot.reshape(-1,1)

    # Init the RANSAC regressor
    ransac = make_pipeline(PolynomialFeatures(20), RANSACRegressor(max_trials=10000, random_state=41))

    # Fit with RANSAC
    ransac.fit(wavelength, data)

    # Get the fitted data result
    line_ransac = ransac.predict(wavelength_plot)[:,0]
    #line_ransac = wavelength_plot
    
    return line_ransac
'''
image_data_164_cont_wide_ransac = np.copy(image_data_164_cont_wide)

temp1 = np.where(np.round(wavelengths164_wide, 2) == 15.8)[0][0]
temp2 = np.where(np.round(wavelengths164_wide, 2) == 15.9)[0][0]
temp3 = np.where(np.round(wavelengths164_wide, 2) == 16.7)[0][0]
temp4 = np.where(np.round(wavelengths164_wide, 2) == 16.9)[0][0]

just_cont_wave_164 = np.hstack((wavelengths164_wide[temp1:temp2], wavelengths164_wide[temp3:temp4]))

just_cont_164 = np.vstack((image_data_164_noline_wide[temp1:temp2], image_data_164_noline_wide[temp3:temp4]))

#%%

#note: ransac uses mad to determine if points are noise or not (inline), so use this to speed things up 

def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        Indices variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    med = np.median(arr)
    return np.median(np.abs(arr - med))

mad_164 = np.zeros((array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        mad_164[i,j] = mad(just_cont_164[:,i,j])

#%%

for i in range(array_length_x):
    for j in range(array_length_y):
        if mad(just_cont_164[:,i,j]) != 0:
            print(i,j)
            image_data_164_cont_wide_ransac[:,i,j] = line_fitter(just_cont_wave_164, just_cont_164[:,i,j], wavelengths164_wide)
        else: 
            image_data_164_cont_wide_ransac[:,i,j] = 0*wavelengths164_wide

#%%
import ButterflyNebulaFunctions as bnf
bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_wide, 'PDFtime/spectra_checking/164_check_continuum_wide_ransac.pdf', 14.5, 17.5, 1, 
                       continuum=image_data_164_cont_wide, conttype=cont_type_164_noline_wide, 
                       comparison_wave_1=wavelengths164_wide, comparison_data_1=image_data_164_cont_wide_ransac)
'''

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(cont_type_112)
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
ax.invert_yaxis()

plt.show()

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(cont_type_62)
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
ax.invert_yaxis()

plt.show()

#%%

'''
5.25 feature
'''

wavelengths52_wide = np.copy(wavelengths62_wide)
image_data_52_noline_wide = np.copy(image_data_62_noline_wide)

np.save('Analysis/wavelengths52_wide', wavelengths52_wide)
np.save('Analysis/image_data_52_wide', image_data_52_noline_wide)

#continuum

image_data_52_cont_wide = np.zeros((len(image_data_52_noline_wide[:,0,0]), array_length_x, array_length_y))

points52 = [5.06, 5.15, 5.39, 5.55]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_52_cont_wide[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths52_wide, image_data_52_noline_wide[:,i,j], points52)

bnf.error_check_imager(wavelengths52_wide, image_data_52_noline_wide, 'PDFtime/spectra_checking/052_check_continuum_wide.pdf', 5.0, 7.0, 1.5, continuum=image_data_52_cont_wide, cont_points=points52)

#%%

'''
5.9 feature
'''

wavelengths59_wide = np.copy(wavelengths62_wide)
image_data_59_noline_wide = np.copy(image_data_62_noline_wide)

np.save('Analysis/wavelengths59_wide', wavelengths59_wide)
np.save('Analysis/image_data_59_wide', image_data_59_noline_wide)

#continuum

image_data_59_cont_wide = np.zeros((len(image_data_59_noline_wide[:,0,0]), array_length_x, array_length_y))

points59 = [5.39, 5.55, 5.81, 5.94]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_59_cont_wide[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths59_wide, image_data_59_noline_wide[:,i,j], points59)

bnf.error_check_imager(wavelengths59_wide, image_data_59_noline_wide, 'PDFtime/spectra_checking/059_check_continuum_wide.pdf', 5.0, 7.0, 1.5, continuum=image_data_59_cont_wide, cont_points=points59)



#%%

'''
16.4 feature, pah shape approach
'''

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(pah_intensity_164)
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
ax.invert_yaxis()

plt.show()


#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(cont_type_164)
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
ax.invert_yaxis()

plt.show()

#%%

#figure out which 16.4 blob looks the most similar to orion
south_164 = image_data_164_noline_wide[:,28,50]
east_164 = image_data_164_noline_wide[:,41,36]
north_164 = image_data_164_noline_wide[:,83,50]

lower = 14.5
upper = 17.5

interval = 0.25

xticks_array = np.arange(lower, upper, interval)

data_scale = south_164[np.where(np.round(wavelengths164_wide, 2) == 16.7)[0][0]]
comparison_scale_1 = east_164[np.where(np.round(wavelengths164_wide, 2) == 16.7)[0][0]]
comparison_scale_2 = north_164[np.where(np.round(wavelengths164_wide, 2) == 16.7)[0][0]]
scaling1 = data_scale/comparison_scale_1
scaling2 = data_scale/comparison_scale_2

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 profiles', fontsize=16)
plt.plot(wavelengths164_wide, south_164, color='black', label='south')
plt.plot(wavelengths164_wide, scaling1*east_164, color='blue', label='east, scale=' + str(scaling1))
plt.plot(wavelengths164_wide, scaling2*north_164, color='purple', label='north, scale=' + str(scaling2))


plt.xlim(lower,upper)
plt.ylim(np.min(south_164[400:1700]), np.max(south_164[400:1700]))
plt.legend(fontsize=14)
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
#plt.savefig('PDFtime/line_investigation/Ne_VI_07652_index6063_bigzoom.png', bbox_inches='tight')
plt.show()
#plt.close()

#%%

plt.figure()
plt.title('south, 28 50')
plt.plot(wavelengths164_wide, south_164, color='black', label='south')
plt.plot(wavelengths164_wide, image_data_164_cont_wide[:,28,50])
plt.xlim(14.5, 17.5)
plt.ylim(5000, 15000)
plt.savefig('PDFtime/orion_comparison/164_south_cont.pdf', bbox_inches='tight')
plt.show()

plt.figure()
plt.title('east, 41 36')
plt.plot(wavelengths164_wide, east_164, color='blue', label='east, scale=' + str(scaling1))
plt.plot(wavelengths164_wide, image_data_164_cont_wide[:,41,36])
plt.xlim(14.5, 17.5)
plt.ylim(5000, 25000)
plt.savefig('PDFtime/orion_comparison/164_east_cont.pdf', bbox_inches='tight')
plt.show()

plt.figure()
plt.title('north, 83 50')
plt.plot(wavelengths164_wide, north_164, color='purple', label='north, scale=' + str(scaling2))
plt.plot(wavelengths164_wide, image_data_164_cont_wide[:,83,50])
plt.xlim(14.5, 17.5)
plt.ylim(3000, 13000)
plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()

#%%

plt.figure()
plt.title('north, 63 60')
plt.plot(wavelengths164_wide, image_data_164_noline_wide[:,63, 60], color='purple', label='north, scale=' + str(scaling2))
plt.plot(wavelengths164_wide, image_data_164_cont_wide[:,63, 60])
plt.xlim(14.5, 17.5)
#plt.ylim(3000, 13000)
#plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()



#%%

#loading in template PAH spectra
pah_image_file = np.loadtxt('comparison_data/barh_stick_csub.fits.dat', skiprows=1)
pah_wavelengths = pah_image_file[:,0]
pah_data = pah_image_file[:,1]

lower = 16.0
upper = 17.0

interval = 0.1

xticks_array = np.arange(lower, upper, interval)

#data_scale = (south_164-image_data_164_cont_wide[:,28,50])[np.where(np.round(wavelengths164_wide, 2) == 16.41)[0][0]]
#comparison_scale_1 = (east_164-image_data_164_cont_wide[:,41,36])[np.where(np.round(wavelengths164_wide, 2) == 16.41)[0][0]]
#comparison_scale_2 = (north_164-image_data_164_cont_wide[:,83,50])[np.where(np.round(wavelengths164_wide, 2) == 16.41)[0][0]]
#comparison_scale_orion = pah_data[np.where(np.round(pah_wavelengths, 2) == 16.41)[0][0]]

data_scale = np.max((south_164-image_data_164_cont_wide[:,28,50])[1000:1500])
comparison_scale_1 = np.max((east_164-image_data_164_cont_wide[:,41,36])[1000:1500])
comparison_scale_2 = np.max((north_164-image_data_164_cont_wide[:,83,50])[1000:1500])
comparison_scale_orion = np.max(pah_data[11600:11650])

scaling1 = data_scale/comparison_scale_1
scaling2 = data_scale/comparison_scale_2
scaling_orion = data_scale/comparison_scale_orion

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 profiles, orion comparison', fontsize=16)
plt.plot(wavelengths164_wide, south_164-image_data_164_cont_wide[:,28,50], color='black', label='south')
plt.plot(wavelengths164_wide, scaling1*(east_164-image_data_164_cont_wide[:,41,36]), color='blue', label='east, scale=' + str(scaling1))
plt.plot(wavelengths164_wide, scaling2*(north_164-image_data_164_cont_wide[:,83,50]), color='purple', label='north, scale=' + str(scaling2))

plt.plot(pah_wavelengths, scaling_orion*(pah_data), color='red', label='orion, scale=' + str(scaling_orion))


plt.xlim(lower,upper)
plt.ylim(np.min((south_164-image_data_164_cont_wide[:,28,50])[1000:1500]), 1.2*np.max((south_164-image_data_164_cont_wide[:,28,50])[1000:1500]))
plt.legend(fontsize=14)
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
plt.savefig('PDFtime/orion_comparison/164_comparison.pdf', bbox_inches='tight')
plt.show()
#plt.close()

#%%

#smooth version

#smoothing data for easier comparison

image_data_230cs_smooth = np.copy(image_data_230cs)

n = 15  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1


south_164_smooth = lfilter(b, a, south_164-image_data_164_cont_wide[:,28,50])
east_164_smooth = lfilter(b, a, east_164-image_data_164_cont_wide[:,41,36])
north_164_smooth = lfilter(b, a, north_164-image_data_164_cont_wide[:,83,50])
orion_smooth = lfilter(b, a, pah_data)


data_scale = np.max(south_164_smooth[1000:1500])
comparison_scale_1 = np.max(east_164_smooth[1000:1500])
comparison_scale_2 = np.max(north_164_smooth[1000:1500])
comparison_scale_orion = np.max(orion_smooth[11600:11650])

scaling1 = data_scale/comparison_scale_1
scaling2 = data_scale/comparison_scale_2
scaling_orion = data_scale/comparison_scale_orion

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 profiles, orion comparison', fontsize=16)
plt.plot(wavelengths164_wide, south_164_smooth, color='black', label='south')
plt.plot(wavelengths164_wide, scaling1*(east_164_smooth), color='blue', label='east, scale=' + str(scaling1))
plt.plot(wavelengths164_wide, scaling2*(north_164_smooth), color='purple', label='north, scale=' + str(scaling2))

plt.plot(pah_wavelengths, scaling_orion*(orion_smooth), color='red', label='orion, scale=' + str(scaling_orion))


plt.xlim(lower,upper)
plt.ylim(np.min((south_164-image_data_164_cont_wide[:,28,50])[1000:1500]), 1.2*np.max((south_164-image_data_164_cont_wide[:,28,50])[1000:1500]))
plt.legend(fontsize=14)
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
plt.savefig('PDFtime/orion_comparison/164_comparison_smooth.pdf', bbox_inches='tight')
plt.show()
#plt.close()

#%%

#attempting to fit the weird silicate feature near 16 microns

#%%

import ButterflyNebulaFunctions as bnf

#continuum

#make 3 continua, as the 16.4 feature seems to be present as a strong version with a red wing, and a weaker version with no red wing.

image_data_164_cont_wide_cs = np.zeros((len(image_data_230cs[:,0,0]), array_length_x, array_length_y))

points164cs = [14.60, 15.41, 16.85, 17.00] #originally 16.12, 16.27

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_164_cont_wide_cs[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths230cs, image_data_230cs[:,i,j], points164cs)
        
bnf.error_check_imager(wavelengths230cs, image_data_230cs, 'PDFtime/spectra_checking/164_check_continuum_wide_cs.pdf', 14.5, 17.5, 1, continuum=image_data_164_cont_wide_cs, cont_points=points164cs)

np.save('Analysis/image_data_164_cont_wide_cs', image_data_164_cont_wide_cs)



#%%
hd100546_image_data = hd100546_data[:,1]
hd100546_wavelengths = hd100546_data[:,0]
plt.figure()
plt.title('63 60')
plt.plot(wavelengths230cs, image_data_230cs[:,63, 60]/4, color='green', label='north, scale=' + str(scaling2))
plt.plot(hd100546_wavelengths, 100*hd100546_image_data, color='blue', label='north, scale=' + str(scaling2))
#plt.plot(wavelengths164_wide, image_data_164_cont_wide[:,63, 60])
plt.xlim(14.5, 25)
plt.ylim(3000, 30000)
#plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()

#%%
hd100546_image_data = hd100546_data[:,1]
hd100546_wavelengths = hd100546_data[:,0]
plt.figure()
plt.title('60, 45 vs HD100546')
plt.plot(wavelengths230cs, image_data_230cs[:,60, 45]/10, color='purple', label='north, scale=' + str(scaling2))
plt.plot(hd100546_wavelengths, 100*hd100546_image_data, color='blue', label='north, scale=' + str(scaling2))
#plt.plot(wavelengths164_wide, image_data_164_cont_wide[:,63, 60])
plt.xlim(14.5, 25)
plt.ylim(3000, 30000)
#plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()

#%%
hd100546_image_data = hd100546_data[:,1]
hd100546_wavelengths = hd100546_data[:,0]
plt.figure()
plt.title('both vs HD100546')
plt.plot(wavelengths230cs, image_data_230cs[:,60, 45]/10, color='purple', label='north, scale=' + str(scaling2))
plt.plot(wavelengths230cs, image_data_230cs[:,63, 60]/4, color='green', label='north, scale=' + str(scaling2))
plt.plot(hd100546_wavelengths, 100*hd100546_image_data, color='blue', label='north, scale=' + str(scaling2))
#plt.plot(wavelengths164_wide, image_data_164_cont_wide[:,63, 60])
plt.xlim(14.5, 25)
plt.ylim(3000, 30000)
#plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()

#%%
hd100546_image_data = hd100546_data[:,1]
hd100546_wavelengths = hd100546_data[:,0]
plt.figure()
plt.title('both continuum subtracted')
plt.plot(wavelengths230cs, (image_data_230cs - image_data_164_cont_wide_cs)[:,60, 45], color='purple', label='north, scale=' + str(scaling2))
plt.plot(wavelengths230cs, (image_data_230cs - image_data_164_cont_wide_cs)[:,63, 60], color='green', label='north, scale=' + str(scaling2))
#plt.plot(hd100546_wavelengths, 100*hd100546_image_data, color='blue', label='north, scale=' + str(scaling2))
#plt.plot(wavelengths164_wide, image_data_164_cont_wide[:,63, 60])
plt.xlim(14.5, 17)
plt.ylim(-8000, 5000)
#plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()


#%%

#regridding 16.4 intensity, 2x2

array_length_regrid_x = int((array_length_x-1)/2)
array_length_regrid_y = int((array_length_y-1)/2)

pah_intensity_regrid_164 = np.zeros((array_length_regrid_x, array_length_regrid_y))
image_data_164_noline_regrid_wide = np.zeros((len(image_data_164_noline_wide[:,0,0]), array_length_regrid_x, array_length_regrid_y))


for i in range(array_length_regrid_x):
    print(2*i, 2*i+1)
    for j in range(array_length_regrid_y):
        pah_intensity_regrid_164[i,j] = (pah_intensity_164[2*i, 2*j] + pah_intensity_164[2*i+1, 2*j] + 
                                         pah_intensity_164[2*i, 2*j+1] + pah_intensity_164[2*i+1, 2*j+1])/4
        image_data_164_noline_regrid_wide[:,i,j] = (image_data_164_noline_wide[:,2*i,2*j] + image_data_164_noline_wide[:,2*i+1,2*j] + 
                                                    image_data_164_noline_wide[:,2*i,2*j+1] + image_data_164_noline_wide[:,2*i+1,2*j+1])/4



#%%

import ButterflyNebulaFunctions as bnf

#continuum

#make 3 continua, as the 16.4 feature seems to be present as a strong version with a red wing, and a weaker version with no red wing.

image_data_164_cont_regrid_wide_1 = np.zeros((len(image_data_164_noline_regrid_wide[:,0,0]), array_length_regrid_x, array_length_regrid_y))

points164_1 = [15.45, 16.25, 16.73, 16.85] #originally 16.12, 16.27

for i in range(array_length_regrid_x):
    for j in range(array_length_regrid_y):
        image_data_164_cont_regrid_wide_1[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths164_wide, image_data_164_noline_regrid_wide[:,i,j], points164_1)
        
#bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_wide, 'PDFtime/spectra_checking/164_check_continuum_regrid_wide_1.pdf', 14.5, 17.5, 1, continuum=image_data_164_cont_regrid_wide_1, cont_points=points164_1)

np.save('Analysis/image_data_164_cont_regrid_wide_1', image_data_164_cont_regrid_wide_1)



image_data_164_cont_regrid_wide_2 = np.zeros((len(image_data_164_noline_regrid_wide[:,0,0]), array_length_regrid_x, array_length_regrid_y))

points164_2 = [15.45, 16.25, 16.63, 16.75]

for i in range(array_length_regrid_x):
    for j in range(array_length_regrid_y):
        image_data_164_cont_regrid_wide_2[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths164_wide, image_data_164_noline_regrid_wide[:,i,j], points164_2)
        
#bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_wide, 'PDFtime/spectra_checking/164_check_continuum_regrid_wide_2.pdf', 14.5, 17.5, 1, continuum=image_data_164_cont_regrid_wide_2, cont_points=points164_2)

np.save('Analysis/image_data_164_cont_regrid_wide_2', image_data_164_cont_regrid_wide_2)



image_data_164_cont_regrid_wide_3 = np.zeros((len(image_data_164_noline_regrid_wide[:,0,0]), array_length_regrid_x, array_length_regrid_y))

points164_3 = [15.45, 16.25, 16.53, 16.65]

for i in range(array_length_regrid_x):
    for j in range(array_length_regrid_y):
        image_data_164_cont_regrid_wide_3[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths164_wide, image_data_164_noline_regrid_wide[:,i,j], points164_3)
        
import ButterflyNebulaFunctions as bnf
#bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_wide, 'PDFtime/spectra_checking/164_check_continuum_regrid_wide_3.pdf', 14.5, 17.5, 1, continuum=image_data_164_cont_regrid_wide_3, cont_points=points164_3)

np.save('Analysis/image_data_164_cont_regrid_wide_3', image_data_164_cont_regrid_wide_3)


image_data_164_cont_regrid_wide = np.copy(image_data_164_cont_regrid_wide_1)

#make an array to keep track of which 16.4 continuum is being used
cont_type_164_noline_regrid_wide = np.ones((array_length_regrid_x, array_length_regrid_y))

#note these are named based off of which continuum the point belongs to, and so appear backwards than what would otherwise be expected; [1:2] is expected but this is [2:1]
temp_index_1 = np.where(np.round(wavelengths164_wide, 2) == points164_1[2])[0][0] #approx 16.73
temp_index_2 = np.where(np.round(wavelengths164_wide, 2) == points164_2[2])[0][0] #approx 16.63
temp_index_3 = np.where(np.round(wavelengths164_wide, 2) == points164_3[2])[0][0] #approx 16.53

for i in range(array_length_regrid_x):
    for j in range(array_length_regrid_y):
        if np.median(image_data_164_noline_regrid_wide[temp_index_2:temp_index_1,i,j] - image_data_164_cont_regrid_wide[temp_index_2:temp_index_1,i,j]) < 0:
            cont_type_164_noline_regrid_wide[i,j] += 1
            image_data_164_cont_regrid_wide[:,i,j] = image_data_164_cont_regrid_wide_2[:,i,j]
        if np.median(image_data_164_noline_regrid_wide[temp_index_3:temp_index_2,i,j] - image_data_164_cont_regrid_wide[temp_index_3:temp_index_2,i,j]) < 0:
            cont_type_164_noline_regrid_wide[i,j] += 1
            image_data_164_cont_regrid_wide[:,i,j] = image_data_164_cont_regrid_wide_3[:,i,j]      

#bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_wide, 'PDFtime/spectra_checking/164_check_continuum_wide.pdf', 14.5, 17.5, 1, 
#                       continuum=image_data_164_cont_wide, conttype=cont_type_164_noline_wide)

np.save('Analysis/cont_type_164_noline_regrid_wide', cont_type_164_noline_regrid_wide)
np.save('Analysis/image_data_164_cont_regrid_wide', image_data_164_cont_regrid_wide)

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('16.4 2x2', fontsize=16)
plt.imshow(pah_intensity_regrid_164)
plt.colorbar()
#data border
#plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
#plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
#plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

#plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
#plt.savefig('PDFtime/spectra_checking/062_continuum_type.png', bbox_inches='tight')
plt.show()
#plt.close()

#north is 41,25, east is 20,18, south is 14,25

#%%

plt.figure()
plt.title('south, 14, 25')
plt.plot(wavelengths164_wide, image_data_164_noline_regrid_wide[:,14,25], color='black', label='south')
plt.plot(wavelengths164_wide, image_data_164_cont_regrid_wide[:,14,25])
plt.xlim(14.5, 17.5)
plt.ylim(5000, 15000)
#plt.savefig('PDFtime/orion_comparison/164_south_cont.pdf', bbox_inches='tight')
plt.show()

plt.figure()
plt.title('east, 20 18')
plt.plot(wavelengths164_wide, image_data_164_noline_regrid_wide[:,20,18], color='blue', label='east, scale=' + str(scaling1))
plt.plot(wavelengths164_wide, image_data_164_cont_regrid_wide[:,20,18])
plt.xlim(14.5, 17.5)
plt.ylim(5000, 25000)
#plt.savefig('PDFtime/orion_comparison/164_east_cont.pdf', bbox_inches='tight')
plt.show()

plt.figure()
plt.title('north, 41 25')
plt.plot(wavelengths164_wide, image_data_164_noline_regrid_wide[:,41,25], color='purple', label='north, scale=' + str(scaling2))
plt.plot(wavelengths164_wide, image_data_164_cont_regrid_wide[:,41,25])
plt.xlim(14.5, 17.5)
plt.ylim(3000, 13000)
#plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()

#%%

#regridding 16.4 intensity , 3x3

array_length_regrid2_x = int((array_length_x)/3)
array_length_regrid2_y = int((array_length_y-1)/3)

pah_intensity_regrid2_164 = np.zeros((array_length_regrid2_x, array_length_regrid2_y))
image_data_164_noline_regrid2_wide = np.zeros((len(image_data_164_noline_wide[:,0,0]), array_length_regrid2_x, array_length_regrid2_y))


for i in range(array_length_regrid2_x):
    print(3*i, 3*(i+1))
    for j in range(array_length_regrid2_y):
        pah_intensity_regrid2_164[i,j] = np.mean(pah_intensity_164[3*i:3*(i+1), 3*j:3*(j+1)])
        image_data_164_noline_regrid2_wide[:,i,j] = np.mean(image_data_164_noline_wide[:,3*i:3*(i+1), 3*j:3*(j+1)], axis=(1,2))




#%%

import ButterflyNebulaFunctions as bnf

#continuum

#make 3 continua, as the 16.4 feature seems to be present as a strong version with a red wing, and a weaker version with no red wing.

image_data_164_cont_regrid2_wide_1 = np.zeros((len(image_data_164_noline_regrid2_wide[:,0,0]), array_length_regrid2_x, array_length_regrid2_y))

points164_1 = [15.45, 16.25, 16.73, 16.85] #originally 16.12, 16.27

for i in range(array_length_regrid2_x):
    for j in range(array_length_regrid2_y):
        image_data_164_cont_regrid2_wide_1[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths164_wide, image_data_164_noline_regrid2_wide[:,i,j], points164_1)
        
#bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_wide, 'PDFtime/spectra_checking/164_check_continuum_regrid2_wide_1.pdf', 14.5, 17.5, 1, continuum=image_data_164_cont_regrid2_wide_1, cont_points=points164_1)

np.save('Analysis/image_data_164_cont_regrid2_wide_1', image_data_164_cont_regrid2_wide_1)



image_data_164_cont_regrid2_wide_2 = np.zeros((len(image_data_164_noline_regrid2_wide[:,0,0]), array_length_regrid2_x, array_length_regrid2_y))

points164_2 = [15.45, 16.25, 16.63, 16.75]

for i in range(array_length_regrid2_x):
    for j in range(array_length_regrid2_y):
        image_data_164_cont_regrid2_wide_2[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths164_wide, image_data_164_noline_regrid2_wide[:,i,j], points164_2)
        
#bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_wide, 'PDFtime/spectra_checking/164_check_continuum_regrid2_wide_2.pdf', 14.5, 17.5, 1, continuum=image_data_164_cont_regrid2_wide_2, cont_points=points164_2)

np.save('Analysis/image_data_164_cont_regrid2_wide_2', image_data_164_cont_regrid2_wide_2)



image_data_164_cont_regrid2_wide_3 = np.zeros((len(image_data_164_noline_regrid2_wide[:,0,0]), array_length_regrid2_x, array_length_regrid2_y))

points164_3 = [15.45, 16.25, 16.53, 16.65]

for i in range(array_length_regrid2_x):
    for j in range(array_length_regrid2_y):
        image_data_164_cont_regrid2_wide_3[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths164_wide, image_data_164_noline_regrid2_wide[:,i,j], points164_3)
        
import ButterflyNebulaFunctions as bnf
#bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_wide, 'PDFtime/spectra_checking/164_check_continuum_regrid2_wide_3.pdf', 14.5, 17.5, 1, continuum=image_data_164_cont_regrid2_wide_3, cont_points=points164_3)

np.save('Analysis/image_data_164_cont_regrid2_wide_3', image_data_164_cont_regrid2_wide_3)


image_data_164_cont_regrid2_wide = np.copy(image_data_164_cont_regrid2_wide_1)

#make an array to keep track of which 16.4 continuum is being used
cont_type_164_noline_regrid2_wide = np.ones((array_length_regrid2_x, array_length_regrid2_y))

#note these are named based off of which continuum the point belongs to, and so appear backwards than what would otherwise be expected; [1:2] is expected but this is [2:1]
temp_index_1 = np.where(np.round(wavelengths164_wide, 2) == points164_1[2])[0][0] #approx 16.73
temp_index_2 = np.where(np.round(wavelengths164_wide, 2) == points164_2[2])[0][0] #approx 16.63
temp_index_3 = np.where(np.round(wavelengths164_wide, 2) == points164_3[2])[0][0] #approx 16.53

for i in range(array_length_regrid2_x):
    for j in range(array_length_regrid2_y):
        if np.median(image_data_164_noline_regrid2_wide[temp_index_2:temp_index_1,i,j] - image_data_164_cont_regrid2_wide[temp_index_2:temp_index_1,i,j]) < 0:
            cont_type_164_noline_regrid2_wide[i,j] += 1
            image_data_164_cont_regrid2_wide[:,i,j] = image_data_164_cont_regrid2_wide_2[:,i,j]
        if np.median(image_data_164_noline_regrid2_wide[temp_index_3:temp_index_2,i,j] - image_data_164_cont_regrid2_wide[temp_index_3:temp_index_2,i,j]) < 0:
            cont_type_164_noline_regrid2_wide[i,j] += 1
            image_data_164_cont_regrid2_wide[:,i,j] = image_data_164_cont_regrid2_wide_3[:,i,j]      



np.save('Analysis/cont_type_164_noline_regrid2_wide', cont_type_164_noline_regrid2_wide)
np.save('Analysis/image_data_164_cont_regrid2_wide', image_data_164_cont_regrid2_wide)

#%%
import ButterflyNebulaFunctions as bnf
bnf.error_check_imager(wavelengths164_wide, image_data_164_noline_regrid2_wide, 'PDFtime/spectra_checking/164_check_continuum_regrid2_wide.pdf', 14.5, 17.5, 1, 
                       continuum=image_data_164_cont_regrid2_wide, conttype=cont_type_164_noline_regrid2_wide, regrid='3x3')


#%%
ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('16.4 3x3', fontsize=16)
plt.imshow(pah_intensity_regrid2_164)
plt.colorbar()
#data border
#plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
#plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
#plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

#plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
#plt.savefig('PDFtime/spectra_checking/062_continuum_type.png', bbox_inches='tight')
plt.show()
#plt.close()

#north is 41,25, east is 20,18, south is 14,25

#%%

plt.figure()
plt.title('south, 9, 16')
plt.plot(wavelengths164_wide, image_data_164_noline_regrid2_wide[:,9,16], color='black', label='south')
plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide[:,9,16])
plt.xlim(14.5, 17.5)
#plt.ylim(5000, 15000)
#plt.savefig('PDFtime/orion_comparison/164_south_cont.pdf', bbox_inches='tight')
plt.show()

plt.figure()
plt.title('east, 13, 12')
plt.plot(wavelengths164_wide, image_data_164_noline_regrid2_wide[:,13,12], color='blue', label='east, scale=' + str(scaling1))
plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide[:,13,12])
plt.xlim(14.5, 17.5)
#plt.ylim(5000, 25000)
#plt.savefig('PDFtime/orion_comparison/164_east_cont.pdf', bbox_inches='tight')
plt.show()

plt.figure()
plt.title('north, 26, 17')
plt.plot(wavelengths164_wide, image_data_164_noline_regrid2_wide[:,26,17], color='purple', label='north, scale=' + str(scaling2))
plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide[:,26,17])
plt.xlim(14.5, 17.5)
#plt.ylim(3000, 13000)
#plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()

#%%

data_scale = (image_data_164_noline_regrid2_wide[:,9,16] - image_data_164_cont_regrid2_wide[:,9,16])[np.where(np.round(wavelengths164_wide, 2) == 16.43)[0][0]]
comparison_scale_1 = (image_data_164_noline_regrid2_wide[:,13,12] - image_data_164_cont_regrid2_wide[:,13,12])[np.where(np.round(wavelengths164_wide, 2) == 16.43)[0][0]]
comparison_scale_2 = (image_data_164_noline_regrid2_wide[:,26,17] - image_data_164_cont_regrid2_wide[:,26,17])[np.where(np.round(wavelengths164_wide, 2) == 16.43)[0][0]]
scaling1 = data_scale/comparison_scale_1
scaling2 = data_scale/comparison_scale_2

plt.figure()
plt.title('16.4 profiles, 3x3')
plt.plot(wavelengths164_wide, (image_data_164_noline_regrid2_wide[:,9,16] - image_data_164_cont_regrid2_wide[:,9,16]), color='black', label='south')
plt.plot(wavelengths164_wide, scaling1*(image_data_164_noline_regrid2_wide[:,13,12] - image_data_164_cont_regrid2_wide[:,13,12]), color='blue', label='east, scale=' + str(scaling1), alpha=0.5)
plt.plot(wavelengths164_wide, scaling2*(image_data_164_noline_regrid2_wide[:,26,17] - image_data_164_cont_regrid2_wide[:,26,17]), color='purple', label='north, scale=' + str(scaling2), alpha=0.5)
plt.xlim(16.0, 17.0)
plt.ylim(0, 1500)
#plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()




#%%

#regridding 6.2 intensity , 3x3

array_length_regrid2_x = int((array_length_x)/3)
array_length_regrid2_y = int((array_length_y-1)/3)

pah_intensity_regrid2_62 = np.zeros((array_length_regrid2_x, array_length_regrid2_y))
image_data_62_noline_regrid2_wide = np.zeros((len(image_data_62_noline_wide[:,0,0]), array_length_regrid2_x, array_length_regrid2_y))


for i in range(array_length_regrid2_x):
    print(3*i, 3*(i+1))
    for j in range(array_length_regrid2_y):
        pah_intensity_regrid2_62[i,j] = np.mean(pah_intensity_62[3*i:3*(i+1), 3*j:3*(j+1)])
        image_data_62_noline_regrid2_wide[:,i,j] = np.mean(image_data_62_noline_wide[:,3*i:3*(i+1), 3*j:3*(j+1)], axis=(1,2))




#%%

import ButterflyNebulaFunctions as bnf

image_data_62_cont_regrid2_wide_1 = np.zeros((len(image_data_62_noline_regrid2_wide[:,0,0]), array_length_regrid2_x, array_length_regrid2_y))

points62_regrid2_wide_1 = [5.945, 6.59, 6.65, 6.79] #was originally [5.68, 5.945, 6.53, 6.61], then [5.945, 6.56, 6.61, 6.79]

for i in range(array_length_regrid2_x):
    for j in range(array_length_regrid2_y):
        image_data_62_cont_regrid2_wide_1[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths62_wide, image_data_62_noline_regrid2_wide[:,i,j], points62_regrid2_wide_1)


bnf.error_check_imager(wavelengths62_wide, image_data_62_noline_regrid2_wide, 'PDFtime/spectra_checking/062_check_continuum_regrid2_wide_1.pdf', 5.2, 7.0, 1.5, 
                       continuum=image_data_62_cont_regrid2_wide_1, cont_points=points62_regrid2_wide_1, regrid='3x3')

np.save('Analysis/image_data_62_cont_regrid2_wide_1', image_data_62_cont_regrid2_wide_1)

#%%
'''
image_data_62_cont_regrid2_wide_2 = np.zeros((len(image_data_62_noline_regrid2_wide[:,0,0]), array_length_regrid2_x, array_length_regrid2_y))

points62_regrid2_wide_2 = [5.945, 6.61, 6.79, 6.81] #5.81 is a filler value

for i in range(array_length_regrid2_x):
    for j in range(array_length_regrid2_y):
        image_data_62_cont_regrid2_wide_2[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths62_wide, image_data_62_noline_regrid2_wide[:,i,j], points62_regrid2_wide_2)
        
bnf.error_check_imager(wavelengths62_wide, image_data_62_noline_regrid2_wide, 'PDFtime/spectra_checking/062_check_continuum_regrid2_wide_2.pdf', 5.2, 7.0, 1.5, continuum=image_data_62_cont_regrid2_wide_2, cont_points=points62_regrid2_wide_2)

np.save('Analysis/image_data_62_cont_regrid2_wide_2', image_data_62_cont_regrid2_wide_2)
'''
#%%

image_data_62_cont_regrid2_wide = np.copy(image_data_62_cont_regrid2_wide_1)

#make an array to keep track of which 6.2 continuum is being used
cont_type_62_noline_regrid2_wide = np.ones((array_length_regrid2_x, array_length_regrid2_y))
'''
#the idea is if if [1:2] is decreasing/increasing, [2:3] should also be increasing/deceasing.
temp_index_1 = np.where(np.round(wavelengths62_wide, 2) == points62_regrid2_wide_1[1])[0][0] #approx 6.53
temp_index_2 = np.where(np.round(wavelengths62_wide, 2) == points62_regrid2_wide_1[2])[0][0] #approx 6.61
temp_index_3 = np.where(np.round(wavelengths62_wide, 2) == points62_regrid2_wide_1[3])[0][0] #approx 6.79

temp_index_4 = int(temp_index_1 + (temp_index_2-temp_index_1)/2)
temp_index_5 = int(temp_index_2 + (temp_index_3-temp_index_2)/2)



var=0

for i in range(array_length_regrid2_x):
    for j in range(array_length_regrid2_y):
        if (np.median(image_data_62_noline_regrid2_wide[temp_index_1:temp_index_4,i,j]) - np.median(image_data_62_noline_regrid2_wide[temp_index_4:temp_index_2,i,j]) < var
        and np.median(image_data_62_noline_regrid2_wide[temp_index_2:temp_index_5,i,j]) - np.median(image_data_62_noline_regrid2_wide[temp_index_5:temp_index_3,i,j]) > -0):
            cont_type_62_noline_regrid2_wide[i,j] += 1
            image_data_62_cont_regrid2_wide[:,i,j] = image_data_62_cont_regrid2_wide_2[:,i,j]  
        elif (np.median(image_data_62_noline_regrid2_wide[temp_index_1:temp_index_4,i,j]) - np.median(image_data_62_noline_regrid2_wide[temp_index_4:temp_index_2,i,j]) > -var
        and np.median(image_data_62_noline_regrid2_wide[temp_index_2:temp_index_5,i,j]) - np.median(image_data_62_noline_regrid2_wide[temp_index_5:temp_index_3,i,j]) < 0):
            cont_type_62_noline_regrid2_wide[i,j] += 1
            image_data_62_cont_regrid2_wide[:,i,j] = image_data_62_cont_regrid2_wide_2[:,i,j]
'''
#%%
import ButterflyNebulaFunctions as bnf
bnf.error_check_imager(wavelengths62_wide, image_data_62_noline_regrid2_wide, 'PDFtime/spectra_checking/62_check_continuum_regrid2_wide.pdf', 5.2, 7.0, 1.5, 
                       continuum=image_data_62_cont_regrid2_wide, conttype=cont_type_62_noline_regrid2_wide, regrid='3x3')


#%%
ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('6.2 3x3', fontsize=16)
plt.imshow(pah_intensity_regrid2_62)
plt.colorbar()
#data border
#plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
#plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
#plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

#plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
#plt.savefig('PDFtime/spectra_checking/062_continuum_type.png', bbox_inches='tight')
plt.show()
#plt.close()

#north is 41,25, east is 20,18, south is 14,25

#%%

plt.figure()
plt.title('south, 10, 16')
plt.plot(wavelengths62_wide, image_data_62_noline_regrid2_wide[:,10,16], color='black', label='south')
plt.plot(wavelengths62_wide, image_data_62_cont_regrid2_wide[:,10,16])
plt.xlim(5.2, 7.0)
#plt.ylim(5000, 15000)
#plt.savefig('PDFtime/orion_comparison/62_south_cont.pdf', bbox_inches='tight')
plt.show()

plt.figure()
plt.title('east, 14, 12')
plt.plot(wavelengths62_wide, image_data_62_noline_regrid2_wide[:,14,12], color='blue', label='east, scale=' + str(scaling1))
plt.plot(wavelengths62_wide, image_data_62_cont_regrid2_wide[:,14,12])
plt.xlim(5.2, 7.0)
#plt.ylim(5000, 25000)
#plt.savefig('PDFtime/orion_comparison/62_east_cont.pdf', bbox_inches='tight')
plt.show()

plt.figure()
plt.title('north, 26, 17')
plt.plot(wavelengths62_wide, image_data_62_noline_regrid2_wide[:,26,17], color='purple', label='north, scale=' + str(scaling2))
plt.plot(wavelengths62_wide, image_data_62_cont_regrid2_wide[:,26,17])
plt.xlim(5.2, 7.0)
#plt.ylim(3000, 13000)
#plt.savefig('PDFtime/orion_comparison/62_north_cont.pdf', bbox_inches='tight')
plt.show()

#%%

data_scale = (image_data_62_noline_regrid2_wide[:,10,16] - image_data_62_cont_regrid2_wide[:,10,16])[np.where(np.round(wavelengths62_wide, 2) == 6.22)[0][0]]
comparison_scale_1 = (image_data_62_noline_regrid2_wide[:,14,12] - image_data_62_cont_regrid2_wide[:,14,12])[np.where(np.round(wavelengths62_wide, 2) == 6.22)[0][0]]
comparison_scale_2 = (image_data_62_noline_regrid2_wide[:,26,17] - image_data_62_cont_regrid2_wide[:,26,17])[np.where(np.round(wavelengths62_wide, 2) == 6.22)[0][0]]
scaling1 = data_scale/comparison_scale_1
scaling2 = data_scale/comparison_scale_2

plt.figure()
plt.title('16.4 profiles, 3x3')
plt.plot(wavelengths62_wide, (image_data_62_noline_regrid2_wide[:,10,16] - image_data_62_cont_regrid2_wide[:,10,16]), color='black', label='south')
plt.plot(wavelengths62_wide, scaling1*(image_data_62_noline_regrid2_wide[:,14,12] - image_data_62_cont_regrid2_wide[:,14,12]), color='blue', label='east, scale=' + str(scaling1), alpha=0.5)
plt.plot(wavelengths62_wide, scaling2*(image_data_62_noline_regrid2_wide[:,26,17] - image_data_62_cont_regrid2_wide[:,26,17]), color='purple', label='north, scale=' + str(scaling2), alpha=0.5)
plt.xlim(5.2, 7.0)
#plt.ylim(0, 1500)
#plt.savefig('PDFtime/orion_comparison/62_north_cont.pdf', bbox_inches='tight')
plt.show()

#%%

'''
12.0 feature
'''



#continuum

image_data_3a_cont = np.zeros((len(image_data_3a[:,0,0]), array_length_x, array_length_y))

points120 = [11.65, 11.79, 12.25, 13.08]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_3a_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths3a, image_data_3a_noline[:,i,j], points120)

bnf.error_check_imager(wavelengths3a, image_data_3a_noline, 'PDFtime/spectra_checking/120_check_continuum.pdf', 11.6, 13.4, 1.20, continuum=image_data_3a_cont, cont_points=points120)

np.save('Analysis/image_data_3a_cont', image_data_3a_cont)


'''
16.4 different method comparison
'''



#%%
ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('16.4 3x3', fontsize=16)
plt.imshow(pah_intensity_regrid2_164)
plt.colorbar()
#data border
#plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
#plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
#plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

#plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
#plt.savefig('PDFtime/spectra_checking/062_continuum_type.png', bbox_inches='tight')
plt.show()
#plt.close()

#north is 41,25, east is 20,18, south is 14,25

#%%

plt.figure()
plt.title('south, 9, 16')
plt.plot(wavelengths164_wide, image_data_164_noline_regrid2_wide[:,9,16], color='black', label='south')
plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide[:,9,16])
#plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide_1[:,9,16])
#plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide_2[:,9,16])
plt.xlim(14.5, 17.5)
plt.ylim(5000, 15000)
#plt.savefig('PDFtime/orion_comparison/164_south_cont.pdf', bbox_inches='tight')
plt.show()

#%%

plt.figure()
plt.title('east, 13, 12')
plt.plot(wavelengths164_wide, image_data_164_noline_regrid2_wide[:,13,12], color='blue', label='east, scale=' + str(scaling1))
#plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide[:,13,12])
#plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide_1[:,13,12])
plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide_2[:,13,12])
plt.xlim(14.5, 17.5)
plt.ylim(5000, 25000)
#plt.savefig('PDFtime/orion_comparison/164_east_cont.pdf', bbox_inches='tight')
plt.show()

#%%

plt.figure()
plt.title('method 3, 18, 12')
plt.plot(wavelengths164_wide, image_data_164_noline_regrid2_wide[:,18,12], color='blue', label='east, scale=' + str(scaling1))
plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide[:,18,12])
#plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide_1[:,13,12])
#plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide_2[:,13,12])
plt.xlim(14.5, 17.5)
plt.ylim(10000, 50000)
#plt.savefig('PDFtime/orion_comparison/164_east_cont.pdf', bbox_inches='tight')
plt.show()

#%%

plt.figure()
plt.title('north, 26, 17')
plt.plot(wavelengths164_wide, image_data_164_noline_regrid2_wide[:,26,17], color='purple', label='north, scale=' + str(scaling2))
plt.plot(wavelengths164_wide, image_data_164_cont_regrid2_wide[:,26,17])
plt.xlim(14.5, 17.5)
plt.ylim(6000, 18000)
#plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(cont_type_164_noline_regrid2_wide)
plt.colorbar()
plt.scatter([16, 12, 12], [9, 13, 18], color='black')
ax.invert_yaxis()
plt.show()

#%%

data_scale = (image_data_164_noline_regrid2_wide[:,9,16] - image_data_164_cont_regrid2_wide[:,9,16])[np.where(np.round(wavelengths164_wide, 2) == 16.42)[0][0]]
comparison_scale_1 = (image_data_164_noline_regrid2_wide[:,13,12] - image_data_164_cont_regrid2_wide[:,13,12])[np.where(np.round(wavelengths164_wide, 2) == 16.42)[0][0]]
comparison_scale_2 = (image_data_164_noline_regrid2_wide[:,13,18] - image_data_164_cont_regrid2_wide[:,13,18])[np.where(np.round(wavelengths164_wide, 2) == 16.42)[0][0]]
scaling1 = data_scale/comparison_scale_1
scaling2 = data_scale/comparison_scale_2

plt.figure()
#plt.title('north, 26, 17')
plt.plot(wavelengths164_wide, (image_data_164_noline_regrid2_wide - image_data_164_cont_regrid2_wide)[:,9,16], color='purple', label='south (method 1)')
plt.plot(wavelengths164_wide, scaling1*(image_data_164_noline_regrid2_wide - image_data_164_cont_regrid2_wide)[:,13,12], color='blue', label='east (method 2), scale=' + str(scaling1))
plt.plot(wavelengths164_wide, scaling2*(image_data_164_noline_regrid2_wide - image_data_164_cont_regrid2_wide)[:,13,18], color='pink', label='method 3, scale=' + str(scaling2))
plt.plot([16.73, 16.73], [-10010, 10000], color='purple')
plt.plot([16.73, 16.63], [-10010, 10000], color='blue')
plt.plot([16.73, 16.53], [-10010, 10000], color='pink')
plt.plot(wavelengths164_wide, 0*image_data_164_cont_regrid2_wide[:,0,0], color='black')
plt.xlim(16.0, 17.0)
plt.ylim(-100, 1500)
plt.legend()
#plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()

#%%

#for smoothing data
from scipy.signal import lfilter

n = 15  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1


smooth = lfilter(b, a, (image_data_164_noline_regrid2_wide[:,13,18] - image_data_164_cont_regrid2_wide[:,13,18]))

data_scale = (image_data_164_noline_regrid2_wide[:,9,16] - image_data_164_cont_regrid2_wide[:,9,16])[np.where(np.round(wavelengths164_wide, 2) == 16.42)[0][0]]
comparison_scale_1 = (image_data_164_noline_regrid2_wide[:,13,12] - image_data_164_cont_regrid2_wide[:,13,12])[np.where(np.round(wavelengths164_wide, 2) == 16.42)[0][0]]
comparison_scale_2 = smooth[np.where(np.round(wavelengths164_wide, 2) == 16.42)[0][0]]
scaling1 = data_scale/comparison_scale_1
scaling2 = data_scale/comparison_scale_2

plt.figure()
#plt.title('north, 26, 17')
plt.plot(wavelengths164_wide, (image_data_164_noline_regrid2_wide - image_data_164_cont_regrid2_wide)[:,9,16], color='purple', label='south (method 1)')
plt.plot(wavelengths164_wide, scaling1*(image_data_164_noline_regrid2_wide - image_data_164_cont_regrid2_wide)[:,13,12], color='blue', label='east (method 2), scale=' + str(scaling1))
plt.plot(wavelengths164_wide, scaling2*smooth, color='green', label='method 3, scale=' + str(scaling2))
plt.plot([16.73, 16.73], [-10010, 10000], color='purple')
plt.plot([16.73, 16.63], [-10010, 10000], color='blue')
plt.plot([16.73, 16.53], [-10010, 10000], color='green')
plt.plot(wavelengths164_wide, 0*image_data_164_cont_regrid2_wide[:,0,0], color='black')
plt.xlim(16.0, 17.0)
plt.ylim(-100, 1500)
plt.legend()
#plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()

#%%

data_scale = (image_data_164_noline_regrid2_wide[:,9,16] - image_data_164_cont_regrid2_wide[:,9,16])[np.where(np.round(wavelengths164_wide, 2) == 16.42)[0][0]]
comparison_scale_1 = (image_data_164_noline_regrid2_wide[:,13,12] - image_data_164_cont_regrid2_wide[:,13,12])[np.where(np.round(wavelengths164_wide, 2) == 16.42)[0][0]]
comparison_scale_2 = (image_data_164_noline_regrid2_wide[:,26,17] - image_data_164_cont_regrid2_wide[:,26,17])[np.where(np.round(wavelengths164_wide, 2) == 16.42)[0][0]]
scaling1 = data_scale/comparison_scale_1
scaling2 = data_scale/comparison_scale_2

plt.figure()
#plt.title('north, 26, 17')
plt.plot(wavelengths164_wide, (image_data_164_noline_regrid2_wide - image_data_164_cont_regrid2_wide)[:,9,16], color='purple', label='south (method 1)')
plt.plot(wavelengths164_wide, scaling1*(image_data_164_noline_regrid2_wide - image_data_164_cont_regrid2_wide)[:,13,12], color='blue', label='east (method 2), scale=' + str(scaling1), alpha=0.5)
plt.plot(wavelengths164_wide, scaling2*(image_data_164_noline_regrid2_wide - image_data_164_cont_regrid2_wide)[:,26,17], color='pink', label='north (method 1), scale=' + str(scaling2), alpha=0.9)
plt.plot([16.73, 16.73], [-10010, 10000], color='purple')
plt.plot([16.73, 16.63], [-10010, 10000], color='blue')
#plt.plot([16.73, 16.53], [-10010, 10000], color='pink')
plt.plot(wavelengths164_wide, 0*image_data_164_cont_regrid2_wide[:,0,0], color='black')
plt.xlim(16.0, 17.0)
plt.ylim(-100, 1500)
plt.legend()
#plt.savefig('PDFtime/orion_comparison/164_north_cont.pdf', bbox_inches='tight')
plt.show()


