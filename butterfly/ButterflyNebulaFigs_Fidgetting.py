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



'''
CRYSTALLINE SILICATE COMPARISON
'''

import ButterflyNebulaFunctions as bnf

#23.0 silicate feature

bnf.error_check_imager(wavelengths230cs, image_data_230cs - image_data_230cs_cont, 'PDFtime/spectra_checking/230cs_comparison.pdf', 14.0, 28.0, 1,
                       comparison_wave_1=hd100546_wavelengths230, comparison_data_1=hd100546_image_data_230 - hd100546_image_data_230_cont, 
                       comparison_wave_2=hale_bopp_wavelengths230, comparison_data_2=hale_bopp_image_data_230 - hale_bopp_image_data_230_cont, comparison_scale_wave=23.60)

#%%

#11.3 silicate feature, uses same scaling as 23.0 

bnf.error_check_imager(wavelengths112, image_data_112 - image_data_113cs_cont, 'PDFtime/spectra_checking/113cs_comparison.pdf', 10.6, 11.8, 1,
                       comparison_wave_1=hd100546_wavelengths113, comparison_data_1=hd100546_image_data_113 - hd100546_image_data_113_cont, 
                       comparison_wave_2=hale_bopp_wavelengths113, comparison_data_2=hale_bopp_image_data_113 - hale_bopp_image_data_113_cont, comparison_scale_wave=23.60, 
                       scale_wave=wavelengths230cs, scale_data=image_data_230cs - image_data_230cs_cont,
                       scale_wave_comp=hd100546_wavelengths230, scale_data_comp=hd100546_image_data_230 - hd100546_image_data_230_cont)

#%%

import ButterflyNebulaFunctions as bnf

#silica 'feature' at 13.8?

hd100546_wavelengths138 = hd100546_wavelengths230[:500]
hd100546_image_data_138 = hd100546_image_data_230[:500]

bnf.error_check_imager(wavelengths230cs, image_data_230cs, 'PDFtime/spectra_checking/138cs_comparison.pdf', 12.5, 14.5, 0.25,
                       comparison_wave_1=hd100546_wavelengths138, comparison_data_1=hd100546_image_data_138, 
                       comparison_scale_wave=13.95, min_ylim=0.75)

#%%

#silica 'feature' at 16.4?

hd100546_wavelengths164 = hd100546_wavelengths230[500:1500]
hd100546_image_data_164 = hd100546_image_data_230[500:1500]

bnf.error_check_imager(wavelengths230cs, image_data_230cs, 'PDFtime/spectra_checking/164cs_comparison.pdf', 14.5, 18.0, 0.25,
                       comparison_wave_1=hd100546_wavelengths164, comparison_data_1=hd100546_image_data_164, 
                       comparison_scale_wave=16.7, min_ylim=0.95)

#%%

lower = 5
upper = 35
interval = 1

xticks_array = np.arange(lower, upper, interval)


ax = plt.figure(figsize=(16,10)).add_subplot(111)

plt.plot(hd100546_wavelengths, hd100546_image_data)
plt.plot(hd100546_wavelengths113, hd100546_image_data_113_cont)
plt.plot(hd100546_wavelengths230, hd100546_image_data_230_cont)

plt.xlim((lower, upper))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.show()
plt.close()



'''
FEATURE RATIO COMPARISON PLOTS
'''



#show imshows of each, and scatterplots of each.

#6.2 vs 11.2
bnf.feature_ratio_imager(pah_intensity_62, pah_intensity_112, pah_intensity_112, pah_intensity_error_62, pah_intensity_error_112, 
                         '6.2', '11.2', '062_112', snr_cutoff_62, snr_cutoff_112, current_reprojection)

#6.0 vs 11.2
bnf.feature_ratio_imager(pah_intensity_60, pah_intensity_112, pah_intensity_112, pah_intensity_error_60, pah_intensity_error_112, 
                         '6.0', '11.2', '060_112', snr_cutoff_60, snr_cutoff_112, current_reprojection)

#12.0 vs 11.2
bnf.feature_ratio_imager(pah_intensity_120, pah_intensity_112, pah_intensity_112, pah_intensity_error_120, pah_intensity_error_112, 
                         '12.0', '11.2', '120_112', snr_cutoff_120, snr_cutoff_112, current_reprojection)

#16.4 vs 11.2
bnf.feature_ratio_imager(pah_intensity_164, pah_intensity_112, pah_intensity_112, pah_intensity_error_164, pah_intensity_error_112, 
                         '16.4', '11.2', '164_112', snr_cutoff_164, snr_cutoff_112, current_reprojection)

#5.25 vs 11.2
bnf.feature_ratio_imager(pah_intensity_52, pah_intensity_112, pah_intensity_112, pah_intensity_error_52, pah_intensity_error_112, 
                         '5.25', '11.2', '052_112', snr_cutoff_52, snr_cutoff_112, current_reprojection)

#5.7 vs 11.2
bnf.feature_ratio_imager(pah_intensity_57, pah_intensity_112, pah_intensity_112, pah_intensity_error_57, pah_intensity_error_112, 
                         '5.7', '11.2', '057_112', snr_cutoff_57, snr_cutoff_112, current_reprojection)

#5.9 vs 11.2
bnf.feature_ratio_imager(pah_intensity_59, pah_intensity_112, pah_intensity_112, pah_intensity_error_59, pah_intensity_error_112, 
                         '5.9', '11.2', '059_112', snr_cutoff_59, snr_cutoff_112, current_reprojection)

#7.7 vs 11.2
bnf.feature_ratio_imager(pah_intensity_77, pah_intensity_112, pah_intensity_112, pah_intensity_error_77, pah_intensity_error_112, 
                         '7.7', '11.2', '077_112', snr_cutoff_77, snr_cutoff_112, current_reprojection)

#8.6 vs 11.2
bnf.feature_ratio_imager(pah_intensity_86, pah_intensity_112, pah_intensity_112, pah_intensity_error_86, pah_intensity_error_112, 
                         '8.6', '11.2', '086_112', snr_cutoff_86, snr_cutoff_112, current_reprojection)

#8.6 vs 11.2 local continuum
bnf.feature_ratio_imager(pah_intensity_86_local, pah_intensity_112, pah_intensity_112, pah_intensity_error_86_local, pah_intensity_error_112, 
                         '8.6 Local Continuum', '11.2', '086_local_112', snr_cutoff_86_local, snr_cutoff_112, current_reprojection)

#11.0 vs 11.2
bnf.feature_ratio_imager(pah_intensity_110, pah_intensity_112, pah_intensity_112, pah_intensity_error_110, pah_intensity_error_112, 
                         '11.0', '11.2', '110_112', snr_cutoff_110, snr_cutoff_112, current_reprojection)

#13.5 vs 11.2
bnf.feature_ratio_imager(pah_intensity_135, pah_intensity_112, pah_intensity_112, pah_intensity_error_135, pah_intensity_error_112, 
                         '13.5', '11.2', '135_112', snr_cutoff_135, snr_cutoff_112, current_reprojection)

#7.7 vs 6.2
bnf.feature_ratio_imager_normalized(pah_intensity_77, pah_intensity_62, pah_intensity_112, pah_intensity_error_77, pah_intensity_error_62, 
                         '7.7', '6.2', '077_062', snr_cutoff_77, snr_cutoff_62, current_reprojection)

#8.6 vs 6.2
bnf.feature_ratio_imager_normalized(pah_intensity_86, pah_intensity_62, pah_intensity_112, pah_intensity_error_86, pah_intensity_error_62, 
                         '8.6', '6.2', '086_062', snr_cutoff_86, snr_cutoff_62, current_reprojection)

#8.6 local continuum vs 6.2
bnf.feature_ratio_imager_normalized(pah_intensity_86_local, pah_intensity_62, pah_intensity_112, pah_intensity_error_86_local, pah_intensity_error_62, 
                         '8.6 Local Continuum', '6.2', '086_local_062', snr_cutoff_86_local, snr_cutoff_62, current_reprojection)

#6.0 vs 6.2
bnf.feature_ratio_imager_normalized(pah_intensity_60, pah_intensity_62, pah_intensity_112, pah_intensity_error_60, pah_intensity_error_62, 
                         '6.0', '6.2', '060_062', snr_cutoff_60, snr_cutoff_62, current_reprojection)

#8.6 vs 8.6 local continuum
bnf.feature_ratio_imager_normalized(pah_intensity_86, pah_intensity_86_local, pah_intensity_112, pah_intensity_error_86, pah_intensity_error_86_local, 
                         '8.6', '8.6 Local Continuum', '086_086_local', snr_cutoff_86, snr_cutoff_86_local, current_reprojection)

#16.4 vs 13.5
bnf.feature_ratio_imager(pah_intensity_164, pah_intensity_135, pah_intensity_112, pah_intensity_error_164, pah_intensity_error_135, 
                         '16.4', '13.5', '164_135', snr_cutoff_164, snr_cutoff_135, current_reprojection)

#combining all of the ratio plots into one giant pdf

images = [
    Image.open("PDFtime/single_images/" + f)
    for f in ["062_112_intensity_ratio.png", "062_112_ratio_error.png", "062_112_correlation.png", 
              "060_112_intensity_ratio.png", "060_112_ratio_error.png", "060_112_correlation.png", 
              "052_112_intensity_ratio.png", "052_112_ratio_error.png", "052_112_correlation.png", 
              "057_112_intensity_ratio.png", "057_112_ratio_error.png", "057_112_correlation.png", 
              "059_112_intensity_ratio.png", "059_112_ratio_error.png", "059_112_correlation.png", 
              "077_112_intensity_ratio.png", "077_112_ratio_error.png", "077_112_correlation.png", 
              "086_112_intensity_ratio.png", "086_112_ratio_error.png", "086_112_correlation.png", 
              "086_local_112_intensity_ratio.png", "086_local_112_ratio_error.png", "086_local_112_correlation.png", 
              "110_112_intensity_ratio.png", "110_112_ratio_error.png", "110_112_correlation.png", 
              "120_112_intensity_ratio.png", "120_112_ratio_error.png", "120_112_correlation.png", 
              "135_112_intensity_ratio.png", "135_112_ratio_error.png", "135_112_correlation.png", 
              "164_112_intensity_ratio.png", "164_112_ratio_error.png", "164_112_correlation.png", 
              "060_062_intensity_ratio.png", "060_062_ratio_error.png", "060_062_correlation.png", 
              "077_062_intensity_ratio.png", "077_062_ratio_error.png", "077_062_correlation.png", 
              "086_062_intensity_ratio.png", "086_062_ratio_error.png", "086_062_correlation.png", 
              "086_local_062_intensity_ratio.png", "086_local_062_ratio_error.png", "086_local_062_correlation.png", 
              "086_086_local_intensity_ratio.png", "086_086_local_ratio_error.png", "086_086_local_correlation.png", 
              "164_135_intensity_ratio.png", "164_135_ratio_error.png", "164_135_correlation.png"
              ]
]

pdf_path = "PDFtime/feature_comparison.pdf"

alpha_removed = []

for i in range(len(images)):
    images[i].load()
    background = Image.new("RGB", images[i].size, (255, 255, 255))
    background.paste(images[i], mask=images[i].split()[3]) # 3 is the alpha channel
    alpha_removed.append(background)

alpha_removed[0].save(
    pdf_path, "PDF" ,resolution=1000.0, save_all=True, append_images=alpha_removed[1:]
)

#%%



'''
SPECTRA COMPARISONS
'''



#spectra with everything

poi_list = [
    [41, 51], 
    [49, 42],
    [66, 63],
    [57, 46], 
    [48, 40]]

legend = [
    '11.2 red',
    '11.2 blue',
    '11.2 green',
    '6.2 red',
    '6.2 blue']

bnf.spectra_comparison_imager(wavelengths112, image_data_112, '112_spectra_comparison_no_bkg_sub', 10.1, 11.9, 1, [lower_index_112, upper_index_112], legend, poi_list)

bnf.spectra_comparison_imager(wavelengths112, image_data_112 - image_data_112_cont, '112_spectra_comparison', 10.1, 11.9, 1, [lower_index_112, upper_index_112], legend, poi_list)

bnf.spectra_comparison_imager(wavelengths1b, image_data_1b_noline, '062_spectra_comparison_no_bkg_sub', 5.7, 6.6, 1.25, [lower_index_62, upper_index_62-300], legend, poi_list)

bnf.spectra_comparison_imager(wavelengths1b, image_data_1b_noline - image_data_1b_cont, '062_spectra_comparison', 5.7, 6.6, 1.25, [lower_index_62, upper_index_62-300], legend, poi_list)
#%%

bnf.spectra_comparison_imager(wavelengths230cs, image_data_230cs, '050130_spectra_comparison_no_bkg_sub', 5.0, 13.1, 1.25, [6070, 6340], legend, poi_list, hard_code_ylim=21000)

#%%



poi_list = [
    [56, 54],
    [41, 51], 
    [49, 42],
    [66, 63]]

legend = [
    'central star',
    '11.2 red',
    '11.2 blue',
    '11.2 green']

bnf.spectra_comparison_imager(wavelengths112, image_data_112, 'star_comparison/112_star_comparison_no_bkg_sub', 10.1, 13.1, 1.25, [lower_index_110, upper_index_110], legend, poi_list)

#bnf.spectra_comparison_imager(wavelengths112, image_data_112 - image_data_112_cont, 'star_comparison/112_star_comparison', 10.1, 13.1, 1, [lower_index_112, upper_index_112], legend, poi_list)

#%%

bnf.spectra_comparison_imager(wavelengths230cs, image_data_230cs, 'star_comparison/star_comparison_no_bkg_sub', 5, 28, 1, [lower_index_230cs, upper_index_230cs], legend, poi_list)

#bnf.spectra_comparison_imager(wavelengths230cs, image_data_230cs - image_data_230cs_cont, 'star_comparison/112_star_comparison', 5, 28, 1, [lower_index_230cs, upper_index_230cs], legend, poi_list)

#%%

bnf.spectra_comparison_imager(wavelengths1b, image_data_1b_noline, 'star_comparison/062_star_comparison_no_bkg_sub', 5.7, 6.6, 1.25, [lower_index_62, upper_index_62-300], legend, poi_list)

#%%

bnf.spectra_comparison_imager(wavelengths230cs, image_data_230cs, 'star_comparison/050130_star_comparison_no_bkg_sub', 5.0, 13.1, 1.25, [6070, 6340], legend, poi_list, hard_code_ylim=30000)

#%%



'''
CUSTOM COMPARISON PLOTS
'''



#%%

#Crystalline Silicate Profiles

lower = np.log10(5)
upper = np.log10(28)

lower = 0.7
upper = 1.5

interval = 0.1

xticks_array = np.arange(lower, upper, interval)
#xticks_array = 10**xticks_array

index1 = np.where(np.round(hd100546_wavelengths, 2) == 22.2)[0][0]
index2 = np.where(np.round(hd100546_wavelengths, 2) == 24.4)[0][0]



ref_data = np.copy(hd100546_image_data[index1:index2])
scale_1 = np.max(ref_data)/np.max(image_data_230cs[lower_index_230cs:upper_index_230cs,49,42])
scale_2 = np.max(ref_data)/np.max(image_data_230cs[lower_index_230cs:upper_index_230cs,66,63])



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('Silicate Profiles', fontsize=16)
plt.plot(np.log10(hd100546_wavelengths), hd100546_image_data, label='HD 100546')
plt.plot(np.log10(wavelengths230cs), scale_1*image_data_230cs[:,49,42], label = 'Index 49 42, 11.2 blue spectra, scale=' + str(scale_1))
plt.plot(np.log10(wavelengths230cs), scale_2*image_data_230cs[:,66,63], label = 'Index 66 63, 11.2 green spectra, scale=' + str(scale_2))
plt.plot([1.05, 1.05], [0, 100], color='black', alpha=0.5)
plt.plot([1.36, 1.36], [200, 300], color='black', alpha=0.5)
#plt.plot(hd100546_wavelengths113, hd100546_image_data_113)
#plt.plot(hd100546_wavelengths230, hd100546_image_data_230_cont)

plt.xlim((lower, upper))
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('log base 10 Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.show()
plt.close()



#%%

#disk investigation

#points of interest

j_indices_north = [61, 62, 63, 64, 65, 66, 67]
i_indices_north = []

index_slope_north = (94-54)/(68-61)

for j in j_indices_north:
    index = index_slope_north*(j-61) + 54
    i_indices_north.append(int(np.round(index)))
    
j_indices_south = [59, 60, 61, 62, 63, 64]
i_indices_south = []

index_slope_south = (43-31)/(64-59)

for j in j_indices_south:
    index = index_slope_south*(j-59) + 31
    i_indices_south.append(int(np.round(index)))

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(wavelengths_centroid_112, vmin=11.26, vmax=11.32)
plt.colorbar()

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.plot([61, 68], [54, 94])
plt.scatter(j_indices_north, i_indices_north, s=5, color='red')
plt.plot([59, 64], [31, 43])
plt.scatter(j_indices_south, i_indices_south, s=5, color='blue')

ax.invert_yaxis()
plt.savefig('PDFtime/disk_investigation/investigation_point_locations_112.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#smoothing data for easier comparison

image_data_230cs_smooth = np.copy(image_data_230cs)

n = 15  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1

for k in range(len(i_indices_north)):
    image_data_230cs_smooth[:, i_indices_north[k], j_indices_north[k]] = lfilter(b, a, image_data_230cs[:, i_indices_north[k], j_indices_north[k]])

for k in range(len(i_indices_south)):
    image_data_230cs_smooth[:, i_indices_south[k], j_indices_south[k]] = lfilter(b, a, image_data_230cs[:, i_indices_south[k], j_indices_south[k]])

#%%

#north disk investigation

lower = 5
upper = 13

interval = 0.5

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 11.0)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 11.6)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 5.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 13.0)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('North Disk Investigation, normalized to 11.2 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices_north[0], j_indices_north[0]], label = str(i_indices_north[0]) + ', ' + str(j_indices_north[0]))

for k in range(1, len(i_indices_north)):

    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices_north[0], j_indices_north[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices_north[k], j_indices_north[k]])

    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices_north[k], j_indices_north[k]], label = str(i_indices_north[k]) + ', '  + str(j_indices_north[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0, 0.6*np.max(image_data_230cs_smooth[index1:index4, i_indices_north[0], j_indices_north[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/disk_investigation/North_Disk_Investigation_112_scaling.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

lower = 5
upper = 13

interval = 0.5

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 9.8)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 9.9)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 5.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 13.0)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('North Disk Investigation, normalized to 9.7 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices_north[0], j_indices_north[0]], label = str(i_indices_north[0]) + ', ' + str(j_indices_north[0]))

for k in range(1, len(i_indices_north)):

    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices_north[0], j_indices_north[0]])
    scale = np.min(ref_data)/np.min(image_data_230cs_smooth[index1:index2, i_indices_north[k], j_indices_north[k]])#uses bottom of dip for scaling

    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices_north[k], j_indices_north[k]], label = str(i_indices_north[k]) + ', '  + str(j_indices_north[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0, 0.5*np.max(image_data_230cs_smooth[index1:index4, i_indices_north[0], j_indices_north[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/disk_investigation/North_Disk_Investigation_097_scaling.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

lower = 5
upper = 13

interval = 0.5

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 9.9)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 10.1)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 5.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 13.0)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('North Disk Investigation, normalized to 10.0 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices_north[0], j_indices_north[0]], label = str(i_indices_north[0]) + ', ' + str(j_indices_north[0]))

for k in range(1, len(i_indices_north)):

    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices_north[0], j_indices_north[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices_north[k], j_indices_north[k]])

    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices_north[k], j_indices_north[k]], label = str(i_indices_north[k]) + ', '  + str(j_indices_north[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0, 0.5*np.max(image_data_230cs_smooth[index1:index4, i_indices_north[0], j_indices_north[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/disk_investigation/North_Disk_Investigation_100_scaling.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

lower = 5
upper = 13

interval = 0.5

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 9.2)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 9.35)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 5.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 13.0)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('North Disk Investigation, normalized to 9.3 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices_north[0], j_indices_north[0]], label = str(i_indices_north[0]) + ', ' + str(j_indices_north[0]))

for k in range(1, len(i_indices_north)):

    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices_north[0], j_indices_north[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices_north[k], j_indices_north[k]])

    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices_north[k], j_indices_north[k]], label = str(i_indices_north[k]) + ', '  + str(j_indices_north[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0, 0.5*np.max(image_data_230cs_smooth[index1:index4, i_indices_north[0], j_indices_north[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/disk_investigation/North_Disk_Investigation_093_scaling.png', bbox_inches='tight')
plt.show()
plt.close()



#%%

#south disk investigation

lower = 5
upper = 13

interval = 0.5

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 11.0)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 11.6)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 5.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 13.0)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('South Disk Investigation, normalized to 11.2 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices_south[0], j_indices_south[0]], label = str(i_indices_south[0]) + ', ' + str(j_indices_south[0]))

for k in range(1, len(i_indices_south)):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices_south[0], j_indices_south[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices_south[k], j_indices_south[k]])

    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices_south[k], j_indices_south[k]], label = str(i_indices_south[k]) + ', '  + str(j_indices_south[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0, 1.05*np.max(image_data_230cs_smooth[index1:index4, i_indices_south[0], j_indices_south[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/disk_investigation/South_Disk_Investigation_112_scaling.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

lower = 5
upper = 13

interval = 0.5

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 9.9)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 10.1)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 5.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 13.0)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('South Disk Investigation, normalized to 10.0 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices_south[0], j_indices_south[0]], label = str(i_indices_south[0]) + ', ' + str(j_indices_south[0]))

for k in range(1, len(i_indices_south)):

    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices_south[0], j_indices_south[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices_south[k], j_indices_south[k]])

    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices_south[k], j_indices_south[k]], label = str(i_indices_south[k]) + ', '  + str(j_indices_south[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0, 0.6*np.max(image_data_230cs_smooth[index1:index4, i_indices_south[0], j_indices_south[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/disk_investigation/South_Disk_Investigation_100_scaling.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

lower = 5
upper = 13

interval = 0.5

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 9.2)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 9.35)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 5.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 13.0)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('South Disk Investigation, normalized to 9.3 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices_south[0], j_indices_south[0]], label = str(i_indices_south[0]) + ', ' + str(j_indices_south[0]))

for k in range(1, len(i_indices_south)):

    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices_south[0], j_indices_south[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices_south[k], j_indices_south[k]])

    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices_south[k], j_indices_south[k]], label = str(i_indices_south[k]) + ', '  + str(j_indices_south[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0, 0.6*np.max(image_data_230cs_smooth[index1:index4, i_indices_south[0], j_indices_south[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/disk_investigation/South_Disk_Investigation_093_scaling.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#comparing the north and south disks

lower = 5
upper = 13

interval = 0.5

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 9.2)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 9.35)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 5.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 13.0)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('Disk Investigation, normalized to 9.3 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices_north[0], j_indices_north[0]], label = 'North, ' + str(i_indices_north[0]) + ', ' + str(j_indices_north[0]))


ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices_north[0], j_indices_north[0]])
scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices_north[1], j_indices_north[1]])

plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices_north[1], j_indices_north[1]], label = 'North, ' + str(i_indices_north[1]) + ', '  + str(j_indices_north[1]) + ', scale='  + str(scale), alpha=0.75)

ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices_north[0], j_indices_north[0]])
scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices_south[0], j_indices_south[0]])

plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices_south[0], j_indices_south[0]], label = 'South, ' + str(i_indices_south[0]) + ', '  + str(j_indices_south[0]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0, 0.6*np.max(image_data_230cs_smooth[index1:index4, i_indices_south[0], j_indices_south[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/disk_investigation/Disk_Investigation_093_scaling.png', bbox_inches='tight')
plt.show()
plt.close()



#%%

#comparing the north and south disks, with misc pixels

i_indices = [54, 60, 31, 56, 41, 49]
j_indices = [61, 62, 59, 54, 51, 42]

legend = [
    'disk center',
    'disk north',
    'disk south',
    'central star',
    '11.2 red',
    '11.2 blue',
    ]

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(wavelengths_centroid_112, vmin=11.26, vmax=11.32)
plt.colorbar()

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')



plt.plot([61, 68], [54, 94])
plt.plot([59, 64], [31, 43])
plt.scatter(j_indices[:4], i_indices[:4], s=8, color='black')

plt.scatter(j_indices[4], i_indices[4], s=8, color='red')
plt.scatter(j_indices[5], i_indices[5], s=8, color='blue')

ax.invert_yaxis()
plt.savefig('PDFtime/disk_investigation/investigation_point_locations_112_extra_points.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

lower = 5
upper = 13

interval = 0.5

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 9.2)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 9.35)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 5.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 13.0)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('North Disk Investigation, normalized to 9.3 microns, extra pixels', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, len(i_indices)):

    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]])

    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0, 0.5*np.max(image_data_230cs_smooth[index1:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/disk_investigation/North_Disk_Investigation_093_scaling_extra_pixels.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#15.8 investigation

i_indices = [81, 28, 63, 58, 54, 60, 31, 56]
j_indices = [51, 49, 49, 67, 61, 62, 59, 54]

legend = [
    '16.4 North blob',
    '16.4 South blob',
    '15.8 bright spot left of disk',
    '15.8 bright spot right of disk',
    'disk center',
    'disk north',
    'disk south',
    'central star'
    ]

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(wavelengths_centroid_112, vmin=11.26, vmax=11.32)
plt.colorbar()

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter(j_indices, i_indices, s=8, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/feature_investigation/158/158_investigation_point_locations_112.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(pah_intensity_158, vmin=1e-7, vmax=1.2e-6)
plt.colorbar()

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter(j_indices, i_indices, s=8, color='red')

ax.invert_yaxis()
plt.savefig('PDFtime/feature_investigation/158/158_investigation_point_locations_158.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

lower = 15.0
upper = 17.0

interval = 0.25

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 15.7)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 15.9)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 15.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 17.0)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('15.8 Investigation, normalized to 15.8 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, len(i_indices)):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(1.0*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]]), 1.0*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/158/158_investigation_158_scaling.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#splitting the figure into 2 for increased readability

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('15.8 Investigation, normalized to 15.8 microns pt. 1', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, 5):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(1.0*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]]), 1.0*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/158/158_investigation_158_scaling_pt1.png', bbox_inches='tight')
plt.show()
plt.close()



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('15.8 Investigation, normalized to 15.8 microns pt. 2', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(4, len(i_indices)):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(1.0*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]]), 1.0*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/158/158_investigation_158_scaling_pt2.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('15.8 Investigation, normalized to 15.8 microns pt. 3', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in [2, 7]:
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(1.0*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]]), 1.0*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/158/158_investigation_158_scaling_pt3.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#13.5 investigation

i_indices = [81, 28, 63, 58, 54, 60, 31, 56]
j_indices = [51, 49, 49, 67, 61, 62, 59, 54]

legend = [
    '16.4 North blob',
    '16.4 South blob',
    '15.8 bright spot left of disk',
    '15.8 bright spot right of disk',
    'disk center',
    'disk north',
    'disk south',
    'central star'
    ]

i_indices = [81, 28, 56, 58, 54, 60, 31, 63]
j_indices = [51, 49, 54, 67, 61, 62, 59, 49]

legend = [
    '16.4 North blob',
    '16.4 South blob',
    'central star', 
    '15.8 bright spot right of disk',
    'disk center',
    'disk north',
    'disk south',
    '15.8 bright spot left of disk',
    ]

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(wavelengths_centroid_112, vmin=11.26, vmax=11.32)
plt.colorbar()

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter(j_indices, i_indices, s=8, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/feature_investigation/135/135_investigation_point_locations_112.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(pah_intensity_135, vmin=1e-7, vmax=3e-6)
plt.colorbar()

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter(j_indices, i_indices, s=8, color='red')

ax.invert_yaxis()
plt.savefig('PDFtime/feature_investigation/135/135_investigation_point_locations_135.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(pah_intensity_135, vmin=1e-7, vmax=3e-6)
plt.colorbar()

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter(j_indices[0], i_indices[0], s=8, color='red')
plt.scatter(j_indices[1], i_indices[1], s=8, color='red')
plt.scatter(j_indices[-1], i_indices[-1], s=8, color='red')

ax.invert_yaxis()
plt.savefig('PDFtime/feature_investigation/135/135_investigation_point_locations_135_cond.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

lower = 13.0
upper = 14.0

interval = 0.1

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 13.4)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 13.6)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 13.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 14.0)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('13.5 Investigation, normalized to 13.5 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, len(i_indices)):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.5*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]]), 1.5*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/135/135_Investigation_135_scaling.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#splitting the figure into 2 for increased readability

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('13.5 Investigation, normalized to 13.5 microns pt. 1', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, 4):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]]), 1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/135/135_Investigation_135_scaling_pt1.png', bbox_inches='tight')
plt.show()
plt.close()



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('13.5 Investigation, normalized to 13.5 microns, pt. 2', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(4, len(i_indices)):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs_smooth[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]]), 1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/135/135_Investigation_135_scaling_pt2.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#scaling to 13.9 continuum

lower = 13.2
upper = 14.2

interval = 0.1

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 13.88)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 13.92)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 13.2)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 14.2)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('13.5 Investigation, normalized to 13.9 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, len(i_indices)):
    
    ref_data = np.copy(image_data_230cs[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.7*np.min(image_data_230cs[index3:index4, i_indices[0], j_indices[0]]), 1.3*np.max(image_data_230cs[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/135/135_Investigation_139_scaling.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#splitting the figure into 2 for increased readability

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('13.5 Investigation, normalized to 13.9 microns pt. 1', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, 4):
    
    ref_data = np.copy(image_data_230cs[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs[index3:index4, i_indices[0], j_indices[0]]), 1.0*np.max(image_data_230cs[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/135/135_Investigation_139_scaling_pt1.png', bbox_inches='tight')
plt.show()
plt.close()



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('13.5 Investigation, normalized to 13.9 microns, pt. 2', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(4, len(i_indices)):
    
    ref_data = np.copy(image_data_230cs[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs[index3:index4, i_indices[0], j_indices[0]]), 1.0*np.max(image_data_230cs[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/135/135_Investigation_139_scaling_pt2.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#scaling to 13.3 continuum

lower = 13.2
upper = 14.2

interval = 0.1

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 13.28)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 13.32)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 13.2)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 14.2)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('13.5 Investigation, normalized to 13.3 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, len(i_indices)):
    
    ref_data = np.copy(image_data_230cs[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.7*np.min(image_data_230cs[index3:index4, i_indices[0], j_indices[0]]), 1.3*np.max(image_data_230cs[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/135/135_Investigation_133_scaling.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#splitting the figure into 2 for increased readability

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('13.5 Investigation, normalized to 13.3 microns pt. 1', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, 4):
    
    ref_data = np.copy(image_data_230cs[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(1.0*np.min(image_data_230cs[index3:index4, i_indices[0], j_indices[0]]), 1.25*np.max(image_data_230cs[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/135/135_Investigation_133_scaling_pt1.png', bbox_inches='tight')
plt.show()
plt.close()



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('13.5 Investigation, normalized to 13.3 microns, pt. 2', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(4, len(i_indices)):
    
    ref_data = np.copy(image_data_230cs[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(1.0*np.min(image_data_230cs[index3:index4, i_indices[0], j_indices[0]]), 1.25*np.max(image_data_230cs[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/135/135_Investigation_133_scaling_pt2.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('13.5 Investigation, normalized to 13.3 microns', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
#plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in [1, 7]:
    
    ref_data = np.copy(image_data_230cs[index1:index2, i_indices[0], j_indices[0]])
    scale = np.max(ref_data)/np.max(image_data_230cs[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.95*np.min(image_data_230cs[index3:index4, i_indices[0], j_indices[0]]), 1.20*np.max(image_data_230cs[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/135/135_Investigation_133_scaling_cond.png', bbox_inches='tight')
plt.show()
plt.close()


#%%
#north disk vs east blob

#scaling to 13.9 continuum

lower = 9.0
upper = 15.0

interval = 0.5

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 13.88)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 13.92)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 9.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 15.0)[0][0]




ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('13.5 Investigation, normalized to 13.9 microns, pt. 3', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_lines[:,i_indices[7], j_indices[7]], label = legend[7] + ', index ' + str(i_indices[7]) + ', ' + str(j_indices[7]))
#plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in [5]:
    
    ref_data = np.copy(image_data_230cs[index1:index2, i_indices[7], j_indices[7]])
    scale = np.max(ref_data)/np.max(image_data_230cs[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs_lines[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(1000, 31000)
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/135/135_Investigation_139_scaling_pt3.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#scaling to 13.9 continuum

lower = 15.0
upper = 25.0

interval = 1.0

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 13.88)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 13.92)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 15.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 25.0)[0][0]



ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('13.5 Investigation, normalized to 13.9 microns, pt. 3', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_lines[:,i_indices[7], j_indices[7]], label = legend[7] + ', index ' + str(i_indices[7]) + ', ' + str(j_indices[7]))
#plt.plot(wavelengths3c, image_data_3c_cont[:,i_indices[0], j_indices[0]], color='black', label = '16.4 continuum '  + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in [5]:
    
    ref_data = np.copy(image_data_230cs[index1:index2, i_indices[7], j_indices[7]])
    scale = np.max(ref_data)/np.max(image_data_230cs[index1:index2, i_indices[k], j_indices[k]])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*image_data_230cs_lines[:, i_indices[k], j_indices[k]], label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(31000, 241000)
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/135/135_Investigation_139_scaling_pt3_verylong.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#entire wavelength range is there 16.4 investigation

i_indices = [78, 25, 42, 58, 63]
j_indices = [51, 51, 36, 68, 48]

legend = [
    'North blob',
    'South blob',
    'East blob',
    'West blob',
    'Near star',
    ]

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(wavelengths_centroid_112, vmin=11.26, vmax=11.32)
plt.colorbar()

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter(j_indices, i_indices, s=8, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/164_investigation_point_locations_112.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(pah_intensity_62, vmin=1e-7, vmax=5e-5)
plt.colorbar()

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter(j_indices, i_indices, s=8, color='red')

ax.invert_yaxis()
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/164_investigation_point_locations_062.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(pah_intensity_164, vmin=1e-7, vmax=2e-6)
plt.colorbar()

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter(j_indices, i_indices, s=8, color='red')

ax.invert_yaxis()
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/164_investigation_point_locations_164.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

import ButterflyNebulaFunctions as bnf

#continuum subtraction for entire wavelength range

#taking into account modified continua at 11.2 and 16.4

#i.e. cont_type_164[i_indices[0], j_indices[0]] and cont_type_112[i_indices[0], j_indices[0]]

#north blob

lower = 5.0
upper = 17.0

interval = 1.0

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 6.1)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 6.3)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 5.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 17.0)[0][0]

giga_list = [5.06, 5.15, 5.39, 5.55, 5.81, 5.94, 6.53, 7.06, 8.40, 9.08, 9.41, 10.65, 11.65, 11.79, 12.26, 13.08, 13.31, 13.83, 14.21, 14.77, 15.03, 15.65, 16.14, 16.27, 16.73]

adjustment = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

giga_cont_0 = bnf.omega_linear_continuum(
    wavelengths230cs, image_data_230cs[:,i_indices[0], j_indices[0]], giga_list, adjustment)

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation continuum, north blob', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]], label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))
plt.plot(wavelengths230cs, giga_cont_0, label = legend[0] + 'continuum')

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]]), 1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/continuum/164_continuum_north_blob.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#south blob

giga_list = [5.06, 5.15, 5.39, 5.55, 5.81, 5.94, 6.53, 7.06, 8.40, 9.08, 9.41, 10.65, 11.65, 11.79, 12.26, 13.08, 13.31, 13.83, 14.21, 14.77, 15.03, 15.65, 16.14, 16.27, 16.73]

adjustment = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

giga_cont_1 = bnf.omega_linear_continuum(
    wavelengths230cs, image_data_230cs[:,i_indices[1], j_indices[1]], giga_list, adjustment)

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation continuum, south blob', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[1], j_indices[1]], label = legend[1] + ', index ' + str(i_indices[1]) + ', ' + str(j_indices[1]))
plt.plot(wavelengths230cs, giga_cont_1, label = legend[1] + 'continuum')

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[1], j_indices[1]]), 1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[1], j_indices[1]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/continuum/164_continuum_south_blob.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#east blob (16.4 is 2)

giga_list = [5.06, 5.15, 5.39, 5.55, 5.81, 5.94, 6.53, 7.06, 8.40, 9.08, 9.41, 10.65, 11.65, 11.79, 12.26, 13.08, 13.31, 13.83, 14.21, 14.77, 15.03, 15.65, 16.14, 16.27, 16.63]

adjustment = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

giga_cont_2 = bnf.omega_linear_continuum(
    wavelengths230cs, image_data_230cs[:,i_indices[2], j_indices[2]], giga_list, adjustment)

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation continuum, east blob', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[2], j_indices[2]], label = legend[2] + ', index ' + str(i_indices[2]) + ', ' + str(j_indices[2]))
plt.plot(wavelengths230cs, giga_cont_2, label = legend[2] + 'continuum')

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[2], j_indices[2]]), 1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[2], j_indices[2]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/continuum/164_continuum_east_blob.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#west blob (16.4 is 3)

giga_list = [5.06, 5.15, 5.39, 5.55, 5.81, 5.94, 6.53, 7.06, 8.40, 9.08, 9.41, 10.65, 11.65, 11.79, 12.26, 13.08, 13.31, 13.83, 14.21, 14.77, 15.03, 15.65, 16.14, 16.27, 16.53]

adjustment = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

giga_cont_3 = bnf.omega_linear_continuum(
    wavelengths230cs, image_data_230cs[:,i_indices[3], j_indices[3]], giga_list, adjustment)

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation continuum, west blob', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[3], j_indices[3]], label = legend[3] + ', index ' + str(i_indices[3]) + ', ' + str(j_indices[3]))
plt.plot(wavelengths230cs, giga_cont_3, label = legend[3] + 'continuum')

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[3], j_indices[3]]), 1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[3], j_indices[3]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/continuum/164_continuum_west_blob.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#near star (16.4 is 3, 11.2 is 2)

giga_list = [5.06, 5.15, 5.39, 5.55, 5.81, 5.94, 6.53, 7.06, 8.40, 9.08, 9.41, 10.65, 11.79, 12.26, 13.08, 13.31, 13.89, 14.11, 14.77, 15.03, 15.65, 16.14, 16.27, 16.53, 17.00] #17.00 is a filler value

adjustment = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

giga_cont_4 = bnf.omega_linear_continuum(
    wavelengths230cs, image_data_230cs[:,i_indices[4], j_indices[4]], giga_list, adjustment)

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation continuum, near star', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[4], j_indices[4]], label = legend[4] + ', index ' + str(i_indices[4]) + ', ' + str(j_indices[4]))
plt.plot(wavelengths230cs, giga_cont_4, label = legend[4] + 'continuum')

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[4], j_indices[4]]), 1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[4], j_indices[4]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/continuum/164_continuum_near_star.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#making a zoomed in version of the continuum near 16.4 for this one

lower = 15.0
upper = 17.0

interval = 0.1

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 6.1)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 6.3)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 15.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 17.0)[0][0]

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation continuum, near star, zoomed in', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs[:,i_indices[4], j_indices[4]], label = legend[4] + ', index ' + str(i_indices[4]) + ', ' + str(j_indices[4]))
plt.plot(wavelengths230cs, giga_cont_4, label = legend[4] + 'continuum')

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[4], j_indices[4]]), 1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[4], j_indices[4]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/continuum/164_continuum_near_star_zoom.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#making a ultra zoomed in version of the continuum near 16.4 for this one

lower = 16.2
upper = 16.6

interval = 0.05

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 6.1)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 6.3)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 16.2)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 16.6)[0][0]

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation continuum, near star, ultra zoomed in', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs[:,i_indices[4], j_indices[4]], label = legend[4] + ', index ' + str(i_indices[4]) + ', ' + str(j_indices[4]))
plt.plot(wavelengths230cs, giga_cont_4, label = legend[4] + 'continuum')

plt.xlim((lower, upper))
plt.ylim(0.95*np.min(image_data_230cs_smooth[index3:index4, i_indices[4], j_indices[4]]), 1.05*np.max(image_data_230cs_smooth[index3:index4, i_indices[4], j_indices[4]])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/continuum/164_continuum_near_star_ultra_zoom.png', bbox_inches='tight')
plt.show()
plt.close()

#%%


#bounds to update for integrals: 12.0 upper bound now 12.26; 15.8 now 15.65, 16.14

#looks like can do 12.8, current bounds 12.26 to 13.08

#13.5 bounds will likely need to be updated to accomodate some nonsense in the red wing, also its unclear what is continuum and what is 13.5 on the blue wing some of the time

#looks like i can do 14.2, currently bound from 14.21 to 14.77 (seems to be a very broad red wing), will probably need to make multiple continua for the red wing, 14.11 used sometimes

#maybe something around 15.3, would need several continua for it though 








#%%

#putting continua into 1 array

giga_cont = np.zeros((len(i_indices), giga_cont_0.shape[0]))

giga_cont[0] = giga_cont_0
giga_cont[1] = giga_cont_1
giga_cont[2] = giga_cont_2
giga_cont[3] = giga_cont_3
giga_cont[4] = giga_cont_4



lower = 5.0
upper = 17.0

interval = 1.0

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 6.1)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 6.3)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 9.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 12.0)[0][0]

#splitting the figure into 2 for increased readability

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation, normalized to 6.2 microns pt. 1', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]] - giga_cont[0], 
         label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, 3):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]] - giga_cont[0, index1:index2])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]] - giga_cont[k, index1:index2])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*(image_data_230cs_smooth[:, i_indices[k], j_indices[k]] - giga_cont[k]), 
             label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4]), 
         1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/164_Investigation_062_scaling_pt1.png', bbox_inches='tight')
plt.show()
plt.close()

print('hello')

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation, normalized to 6.2 microns pt. 2', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]] - giga_cont[0], 
         label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(3, len(i_indices)):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]] - giga_cont[0, index1:index2])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]] - giga_cont[k, index1:index2])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*(image_data_230cs_smooth[:, i_indices[k], j_indices[k]] - giga_cont[k]), 
             label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4]), 
         1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/164_Investigation_062_scaling_pt2.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#further splitting into short, medium and long wavelengths 

#short

lower = 5.0
upper = 10.0

interval = 0.5

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 6.1)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 6.3)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 5.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 7.0)[0][0]

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation, normalized to 6.2 microns short pt. 1', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]] - giga_cont[0], 
         label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, 3):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]] - giga_cont[0, index1:index2])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]] - giga_cont[k, index1:index2])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*(image_data_230cs_smooth[:, i_indices[k], j_indices[k]] - giga_cont[k]), 
             label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4]), 
         1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/164_Investigation_062_scaling_short_pt1.png', bbox_inches='tight')
plt.show()
plt.close()

print('hello')

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation, normalized to 6.2 microns short pt. 2', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]] - giga_cont[0], 
         label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(3, len(i_indices)):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]] - giga_cont[0, index1:index2])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]] - giga_cont[k, index1:index2])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*(image_data_230cs_smooth[:, i_indices[k], j_indices[k]] - giga_cont[k]), 
             label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4]), 
         1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/164_Investigation_062_scaling_short_pt2.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#medium

lower = 10.0
upper = 15.0

interval = 0.5

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 6.1)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 6.3)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 10.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 15.0)[0][0]

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation, normalized to 6.2 microns medium pt. 1', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]] - giga_cont[0], 
         label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, 3):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]] - giga_cont[0, index1:index2])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]] - giga_cont[k, index1:index2])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*(image_data_230cs_smooth[:, i_indices[k], j_indices[k]] - giga_cont[k]), 
             label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4]), 
         1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/164_Investigation_062_scaling_medium_pt1.png', bbox_inches='tight')
plt.show()
plt.close()

print('hello')

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation, normalized to 6.2 microns medium pt. 2', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]] - giga_cont[0], 
         label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(3, len(i_indices)):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]] - giga_cont[0, index1:index2])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]] - giga_cont[k, index1:index2])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*(image_data_230cs_smooth[:, i_indices[k], j_indices[k]] - giga_cont[k]), 
             label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4]), 
         1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/164_Investigation_062_scaling_medium_pt2.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#long

lower = 15.0
upper = 17.0

interval = 0.25

xticks_array = np.arange(lower, upper, interval)

index1 = np.where(np.round(wavelengths230cs, 2) == 6.1)[0][0]
index2 = np.where(np.round(wavelengths230cs, 2) == 6.3)[0][0]

index3 = np.where(np.round(wavelengths230cs, 2) == 15.0)[0][0]
index4 = np.where(np.round(wavelengths230cs, 2) == 17.0)[0][0]

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation, normalized to 6.2 microns long pt. 1', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]] - giga_cont[0], 
         label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(1, 3):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]] - giga_cont[0, index1:index2])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]] - giga_cont[k, index1:index2])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*(image_data_230cs_smooth[:, i_indices[k], j_indices[k]] - giga_cont[k]), 
             label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4]), 
         1.2*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/164_Investigation_062_scaling_long_pt1.png', bbox_inches='tight')
plt.show()
plt.close()

print('hello')

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title('16.4 Investigation, normalized to 6.2 microns long pt. 2', fontsize=16)

plt.plot(wavelengths230cs, image_data_230cs_smooth[:,i_indices[0], j_indices[0]] - giga_cont[0], 
         label = legend[0] + ', index ' + str(i_indices[0]) + ', ' + str(j_indices[0]))

for k in range(3, len(i_indices)):
    
    ref_data = np.copy(image_data_230cs_smooth[index1:index2, i_indices[0], j_indices[0]] - giga_cont[0, index1:index2])
    scale = np.max(ref_data)/np.max(image_data_230cs_smooth[index1:index2, i_indices[k], j_indices[k]] - giga_cont[k, index1:index2])
    print(k, scale)
    plt.plot(wavelengths230cs, scale*(image_data_230cs_smooth[:, i_indices[k], j_indices[k]] - giga_cont[k]), 
             label = legend[k] + ', index ' + str(i_indices[k]) + ', '  + str(j_indices[k]) + ', scale='  + str(scale), alpha=0.75)

plt.xlim((lower, upper))
plt.ylim(0.75*np.min(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4]), 
         2.0*np.max(image_data_230cs_smooth[index3:index4, i_indices[0], j_indices[0]] - giga_cont[0, index3:index4])) #index 1 used instead of 3 to exclude strong line at 7.7 for scaling
plt.legend(fontsize=14)
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (Jy/sr)', fontsize=16)
plt.xticks(xticks_array, fontsize=14)
plt.yticks(fontsize=14)
plt.savefig('PDFtime/feature_investigation/164_entire_wavelength/164_Investigation_062_scaling_long_pt2.png', bbox_inches='tight')
plt.show()
plt.close()



#%%

'''
SATURATED NEON LINE INVESTIGATION 
'''

#loading in lines

line_loc = 'line_lists/kevin_lines/Neon/line_7p652_ch2_short_bkgsub.fits'
with fits.open(line_loc) as hdul:
    Neon6 = hdul[1].data
    
line_loc = 'line_lists/kevin_lines/Neon/line_14p3201_ch3_medium_bkgsub.fits'
with fits.open(line_loc) as hdul:
    Neon5_1 = hdul[1].data
    
line_loc = 'line_lists/kevin_lines/Neon/line_24p3114_ch4_medium_bkgsub.fits'
with fits.open(line_loc) as hdul:
    Neon5_2 = hdul[1].data
    
line_loc = 'line_lists/kevin_lines/Iron/line_5p449_ch1_short_bkgsub.fits'
with fits.open(line_loc) as hdul:
    Iron8 = hdul[1].data
    
line_loc = 'line_lists/kevin_lines/Iron/line_7p812_ch2_short_bkgsub.fits'
with fits.open(line_loc) as hdul:
    Iron7_1 = hdul[1].data
    
line_loc = 'line_lists/kevin_lines/Iron/line_9p523_ch2_medium_bkgsub.fits'
with fits.open(line_loc) as hdul:
    Iron7_2 = hdul[1].data

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('Ne VI, 7.652 microns', fontsize=16)
plt.imshow(np.log10(Neon6), vmin=4, vmax=5)
plt.colorbar()

#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/line_investigation/Ne_VI_07652.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('Ne V, 14.3201 microns', fontsize=16)
plt.imshow(np.log10(Neon5_1), vmin=4, vmax=5)
plt.colorbar()
#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='white')

ax.invert_yaxis()
plt.savefig('PDFtime/line_investigation/Ne_V_143201.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('Ne V, 24.3114 microns', fontsize=16)
plt.imshow(np.log10(Neon5_2), vmin=4, vmax=6)
plt.colorbar()
#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/line_investigation/Ne_V_243114.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('Ne V, 24.3114 microns', fontsize=16)
plt.imshow(np.log10(Neon5_2), vmin=5)
plt.colorbar()
#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/line_investigation/Ne_V_243114_bright_only.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('Ne V, 24.3114 microns', fontsize=16)
plt.imshow(np.log10(Neon5_2), vmin=5.8)
plt.colorbar()
#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/line_investigation/Ne_V_243114_ultrabright.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('Fe VIII, 5.449 microns', fontsize=16)
plt.imshow(np.log10(Iron8), vmin=1.5)
plt.colorbar()
#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/line_investigation/Fe_VIII_05449.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('Fe VII, 7.812 microns', fontsize=16)
plt.imshow(np.log10(Iron7_1), vmin=1.5)
plt.colorbar()
#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/line_investigation/Fe_VII_07812.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('Fe VII, 9.523 microns', fontsize=16)
plt.imshow(np.log10(Iron7_2), vmin=1.5)
plt.colorbar()
#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/line_investigation/Fe_VII_09523.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('Fe VII, 9.523 microns', fontsize=16)
plt.imshow(np.log10(Iron7_2), vmin=2.5)
plt.colorbar()
#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/line_investigation/Fe_VII_09523_bright_only.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title('Fe VII, 9.523 microns', fontsize=16)
plt.imshow(np.log10(Iron7_2), vmin=3.0)
plt.colorbar()
#data border
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter([41, 39, 60], [50, 63, 63], s=5, color='black')

ax.invert_yaxis()
plt.savefig('PDFtime/line_investigation/Fe_VII_09523_ultrabright.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

#plotting a gaussian on the line

def gaussian(x, mean, fwhm, a, b):
    
    var = fwhm/(2*(2*np.log(2))**0.5)
    return a*np.exp(-1*((x - mean)**2)/(2*var**2)) + b

#%%
i = 50
j = 41

gaussian_1 = gaussian(wavelengths77, 7.65, 0.07, 30000, 0)

lower = 7.5
upper = 7.9

interval = 0.025

xticks_array = np.arange(lower, upper, interval)

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title(str(j) + ', ' + str(i), fontsize=16)

plt.plot(wavelengths77, image_data_77_lines[:,i,j], color='black')
#plt.plot(wavelengths77, image_data_77[:,i,j])
plt.plot(wavelengths77, image_data_77_cont[:,i,j] + gaussian_1, alpha=0.75, color='red')

plt.xlim(7.5, 7.9)
plt.ylim(np.min(image_data_77[:,i,j]), 0.8*np.max(image_data_77[:,i,j]))
#plt.legend(fontsize=14)
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
plt.savefig('PDFtime/line_investigation/Ne_VI_07652_index4150.png', bbox_inches='tight')
plt.show()
plt.close()

#%%
i = 63
j = 39

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title(str(j) + ', ' + str(i), fontsize=16)
plt.plot(wavelengths77, image_data_77_lines[:,i,j], color='black')
#plt.plot(wavelengths77, image_data_77[:,i,j])
plt.plot(wavelengths77, image_data_77_cont[:,i,j] + gaussian_1, alpha=0.75, color='red')
plt.xlim(7.5, 7.9)
plt.ylim(np.min(image_data_77[:,i,j]), 0.8*np.max(image_data_77[:,i,j]))
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
plt.savefig('PDFtime/line_investigation/Ne_VI_07652_index3963.png', bbox_inches='tight')
plt.show()
plt.close()

#%%
i = 63
j = 60

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title(str(j) + ', ' + str(i), fontsize=16)
plt.plot(wavelengths77, image_data_77_lines[:,i,j], color='black', label='60, 63')
plt.plot(wavelengths77, image_data_77_lines[:,63,39], color='blue', alpha=0.5, label='39, 63')
plt.plot(wavelengths77, image_data_77_lines[:,50,41], color='purple', alpha=0.5, label='41, 50')
#plt.plot(wavelengths77, image_data_77[:,i,j])
plt.plot(wavelengths77, image_data_77_cont[:,i,j] + gaussian_1, alpha=0.75, color='red')
plt.xlim(7.5, 7.9)
plt.ylim(np.min(image_data_77[:,i,j]), 0.8*np.max(image_data_77[:,i,j]))
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
plt.savefig('PDFtime/line_investigation/Ne_VI_07652_index6063.png', bbox_inches='tight')
plt.show()
plt.close()

#%%
i = 63
j = 60

lower = 7.55
upper = 7.65

interval = 0.01

xticks_array = np.arange(lower, upper, interval)

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title(str(j) + ', ' + str(i), fontsize=16)
plt.plot(wavelengths77, image_data_77_lines[:,i,j], color='black', label='60, 63')
plt.plot(wavelengths77, image_data_77_lines[:,63,39], color='blue', alpha=0.5, label='39, 63')
plt.plot(wavelengths77, image_data_77_lines[:,50,41], color='purple', alpha=0.5, label='41, 50')
#plt.plot(wavelengths77, image_data_77[:,i,j])
plt.plot(wavelengths77, image_data_77_cont[:,i,j] + gaussian_1, alpha=0.75, color='red')
plt.xlim(7.55, 7.65)
plt.ylim(np.min(image_data_77[:,i,j]), 0.8*np.max(image_data_77[:,i,j]))
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
plt.savefig('PDFtime/line_investigation/Ne_VI_07652_index6063_bigzoom.png', bbox_inches='tight')
plt.show()
plt.close()

#%%
i = 25
j = 51

lower = 7.5
upper = 7.9

interval = 0.025

xticks_array = np.arange(lower, upper, interval)

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title(str(j) + ', ' + str(i), fontsize=16)
plt.plot(wavelengths77, image_data_77_lines[:,i,j], color='black', label='51, 25 (16.4 south blob)')
plt.plot(wavelengths77, image_data_77_lines[:,50,41], color='purple', alpha=0.5, label='41, 50')
#plt.plot(wavelengths77, image_data_77[:,i,j])
#plt.plot(wavelengths77, image_data_77_cont[:,i,j] + gaussian_1, alpha=0.75, color='red')
plt.xlim(7.5, 7.9)
plt.ylim(3500, 20000)
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
plt.savefig('PDFtime/line_investigation/Ne_VI_07652_index5125.png', bbox_inches='tight')
plt.show()
plt.close()

#%%
i = 25
j = 51

lower = 7.1
upper = 8.3

interval = 0.05

xticks_array = np.arange(lower, upper, interval)

ax = plt.figure(figsize=(16,10)).add_subplot(111)
plt.title(str(j) + ', ' + str(i), fontsize=16)
plt.plot(wavelengths77, image_data_77_lines[:,i,j], color='black', label='51, 25 (16.4 south blob)')
plt.plot(wavelengths77, image_data_77_lines[:,50,41], color='purple', alpha=0.5, label='41, 50')
plt.plot([7.5, 7.5], [1500, 10000], color='blue')
plt.plot([7.9, 7.9], [1500, 10000], color='blue')
#plt.plot(wavelengths77, image_data_77[:,i,j])
#plt.plot(wavelengths77, image_data_77_cont[:,i,j] + gaussian_1, alpha=0.75, color='red')
plt.xlim(7.1, 8.3)
plt.ylim(1500, 10000)
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
plt.savefig('PDFtime/line_investigation/Ne_VI_07652_index5125_zoomout.png', bbox_inches='tight')
plt.show()
plt.close()




#%%



##################################


'''
FIDGETTING AND BUGTESTING
'''

#%%

#creating an array that indicates where the disk is, to compare data inside and outside the disk.

disk_mask = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_disk.reg', do_sigma_clip=True, use_dq=False)

star_mask = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_star.reg', do_sigma_clip=True, use_dq=False)

action_zone_mask = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'butterfly_action_zone.reg', do_sigma_clip=True, use_dq=False)



#%%




#%%

#disk investigation

#points of interest

j_indices_north = [61, 62, 63, 64, 65, 66, 67]
i_indices_north = []

index_slope_north = (94-54)/(68-61)

for j in j_indices_north:
    index = index_slope_north*(j-61) + 54
    i_indices_north.append(int(np.round(index)))
    
j_indices_south = [59, 60, 61, 62, 63, 64]
i_indices_south = []

index_slope_south = (43-31)/(64-59)

for j in j_indices_south:
    index = index_slope_south*(j-59) + 31
    i_indices_south.append(int(np.round(index)))

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(wavelengths_centroid_112, vmin=11.26, vmax=11.32)

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.plot([61, 68], [54, 94])
plt.scatter(j_indices_north, i_indices_north, s=5, color='red')
plt.plot([59, 64], [31, 43])
plt.scatter(j_indices_south, i_indices_south, s=5, color='blue')

ax.invert_yaxis()
plt.savefig('PDFtime/disk_investigation/investigation_point_locations_112_no_cbar.png', bbox_inches='tight')
plt.show()
plt.close()

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(pah_intensity_135, vmin=1e-7, vmax=3e-6)

plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')

plt.scatter(j_indices[0], i_indices[0], s=8, color='red')
plt.scatter(j_indices[1], i_indices[1], s=8, color='red')
plt.scatter(j_indices[-1], i_indices[-1], s=8, color='red')

ax.invert_yaxis()
plt.savefig('PDFtime/feature_investigation/135/135_investigation_point_locations_135_cond_no_cbar.png', bbox_inches='tight')
plt.show()
plt.close()

#%%








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

'''
HUNTING FOR WEAK FEATURES
'''

#10.1 feature, attributed to PASH

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(cont_type_164)
plt.plot([10, 95, 109, 24, 10], [80, 97, 25, 8, 80], color='black')
plt.plot([49, 86, 61, 73, 69, 54, 60.5, 49], [88, 95, 54, 42, 17, 14, 54, 88], color='green')
plt.scatter(54, 56, s=600, facecolors='none', edgecolors='purple')
ax.invert_yaxis()

plt.show()


#%%

#loading in template PAH spectra, from the Orion bar (JWST)
orion_image_file_miri = np.loadtxt('data/misc/templatesT_MRS_crds1154_add_20231212.dat', skiprows=7)
orion_wavelengths_miri = orion_image_file_miri[:,0]
orion_data_miri = orion_image_file_miri[:,9]

#%%

#combining channels 3A and 3B to get proper continua for the 13.5 feature

image_data_101_temp, wavelengths101, overlap101_temp = bnf.flux_aligner3(wavelengths2b, wavelengths2c, image_data_2b_noline[:,50,50], image_data_2c_noline[:,50,50])



#using the above to make an array of the correct size to fill
image_data_101 = np.zeros((len(image_data_101_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_101[:,i,j], wavelengths101, overlap101 =bnf. flux_aligner3(wavelengths2b, wavelengths2c, image_data_2b_noline[:,i,j], image_data_2c_noline[:,i,j])

#%%

import matplotlib.pyplot as plt


#examples for 10.1 (y, x): 21, 46; 26, 52

i = 21
j = 46

#ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title(str(j) + ', ' + str(i))
plt.plot(wavelengths101, image_data_101[:,i,j])
#plt.plot(wavelengths2b, np.log10(image_data_2b[:,i,j]), alpha=0.5)
#plt.plot(wavelengths2c, np.log10(100+image_data_2c[:,i,j]), alpha=0.5)
plt.xlim(9, 11)
#plt.ylim(2.8,3.2)

plt.show()

#%%

#5x5 grid

i = 26
j = 52

#ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title(str(j) + ', ' + str(i))
plt.plot(wavelengths101, np.mean(image_data_101[:,i-2:i+2,j-2:j+2],axis=(1,2)))
#plt.plot(wavelengths2b, np.log10(image_data_2b[:,i,j]), alpha=0.5)
#plt.plot(wavelengths2c, np.log10(100+image_data_2c[:,i,j]), alpha=0.5)
plt.xlim(9, 11)
#plt.ylim(2.8,3.2)

plt.show()

#%%

#5x5 grid

i = 26
j = 52

#ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.title(str(j) + ', ' + str(i))
plt.plot(wavelengths4c, np.log10(np.mean(image_data_4c_noline[:,i-2:i+2,j-2:j+2],axis=(1,2))))
#plt.plot(orion_wavelengths_miri, orion_data_miri, color='black', alpha=0.5)
#plt.plot(wavelengths2b, np.log10(image_data_2b[:,i,j]), alpha=0.5)
#plt.plot(wavelengths2c, np.log10(100+image_data_2c[:,i,j]), alpha=0.5)
#plt.xlim(11.6,13.4)

plt.show()

#%%





ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(pah_intensity_62)



plt.colorbar()

plt.scatter(52, 26, s=10, color='red')


ax.invert_yaxis()
plt.show()



#%%

image_data_101_cont = np.zeros((len(image_data_101[:,0,0]), array_length_x, array_length_y))

points101 =  [9.7, 9.85, 10.2, 10.3] 

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_101_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths101, image_data_101[:,i,j], points101) #note image_data_230cs is built out of things with no lines
        
bnf.error_check_imager(wavelengths101, image_data_101, 'PDFtime/spectra_checking/101_check_continuum.pdf', 9, 11, 1.25, continuum=image_data_101_cont)

np.save('Analysis/image_data_101_cont', image_data_101_cont)

#%%

'''
FOR NITROGENATION, LOOK AT 6.0 VS 6.2 VS 11.0 VS 8.6
'''

ax = plt.figure('6.2 intensity, log 10', figsize=(8,8)).add_subplot(111)
plt.title('6.2 intensity, log 10')
plt.imshow(np.log10(pah_intensity_62[30:50, 35:55]), vmax=-4.2, vmin=-5)
plt.colorbar()

ax.invert_yaxis()
plt.show()

#%%

ax = plt.figure('11.0 intensity, log 10', figsize=(8,8)).add_subplot(111)
plt.title('11.0 intensity, log 10')
plt.imshow(np.log10(pah_intensity_110[30:50, 35:55]), vmin=-6.6, vmax=-5.8)
plt.colorbar()

ax.invert_yaxis()
plt.show()

#%%

ax = plt.figure('8.6 intensity, log 10', figsize=(8,8)).add_subplot(111)
plt.title('8.6 intensity, log 10')
plt.imshow(np.log10(pah_intensity_86[30:50, 35:55]), vmin=-5.4, vmax=-4.6)
plt.colorbar()

ax.invert_yaxis()
plt.show()

#%%

ax = plt.figure('6.0 intensity, log 10', figsize=(8,8)).add_subplot(111)
plt.title('6.0 intensity, log 10')
plt.imshow(np.log10(pah_intensity_60[30:50, 35:55]), vmin=-5.0, vmax=-5.8)
plt.colorbar()

ax.invert_yaxis()
plt.show()

#%%

ax = plt.figure('11.2 intensity, log 10', figsize=(8,8)).add_subplot(111)
plt.title('11.2 intensity, log 10')
plt.imshow(np.log10(pah_intensity_112[30:50, 35:55]), vmin=-5.2, vmax=-4.4)
plt.colorbar()

ax.invert_yaxis()
plt.show()

#%%

i=32
j=49
m=40
n=44

ax = plt.figure('11.0 intensity, log 10, points', figsize=(8,8)).add_subplot(111)
plt.title('11.0 intensity, log 10')
plt.imshow(np.log10(pah_intensity_110[30:50, 35:55]), vmin=-6.6, vmax=-5.8)
plt.scatter(j-35, i-30, color='red')
plt.scatter(n-35, m-30, color='red')
plt.colorbar()

ax.invert_yaxis()
plt.show()



#%%

ax = plt.figure('strong 11.0', figsize=(8,8)).add_subplot(111)
plt.title('Strong 11.0: 11.0, 11.2: ' + str(j) + ', ' + str(i))
plt.plot(wavelengths2c, image_data_2c[:,i,j])
plt.ylim(3000, 14000)
#plt.plot(orion_wavelengths_miri, orion_data_miri, color='black', alpha=0.5)
#plt.plot(wavelengths2b, np.log10(image_data_2b[:,i,j]), alpha=0.5)
#plt.plot(wavelengths2c, np.log10(100+image_data_2c[:,i,j]), alpha=0.5)
#plt.xlim(11.6,13.4)

#%%


offset = image_data_2c[100,i,j] - image_data_2c[100,m,n]
scaling = (image_data_2c[775,i,j] - image_data_2c[100,i,j])/(image_data_2c[775,m,n] - image_data_2c[100,m,n])

ax = plt.figure('11.0', figsize=(16,8)).add_subplot(111)
plt.title('11.0, 11.2: ' + str(j) + ', ' + str(i))
plt.plot(wavelengths2c, image_data_2c[:,i,j], label='strong 11.0')
plt.plot(wavelengths2c, image_data_2c[100,i,j] + scaling*(image_data_2c[:,m,n] - image_data_2c[100,m,n]), 
         label='weak 11.0, offset= ' + str(np.round(offset,3)) + ', scaling= ' + str(np.round(scaling, 3)), alpha=0.5)
plt.legend()
plt.ylim(3000, 14000)
plt.show()

#%%


offset = image_data_1b[100,i,j] - image_data_1b[100,m,n]
scaling = (image_data_1b[775,i,j] - image_data_1b[100,i,j])/(image_data_1b[775,m,n] - image_data_1b[100,m,n])

offset = 0
scaling = 1

ax = plt.figure('6.0', figsize=(16,8)).add_subplot(111)
plt.title('6.0, 6.2: ' + str(j) + ', ' + str(i))
plt.plot(wavelengths1b, image_data_1b[:,i,j], label='strong 11.0')
plt.plot(wavelengths1b, image_data_1b[100,i,j] + scaling*(image_data_1b[:,m,n] - image_data_1b[100,m,n]), 
         label='weak 11.0, offset= ' + str(np.round(offset,3)) + ', scaling= ' + str(np.round(scaling, 3)), alpha=0.5)
plt.legend()
plt.ylim(500, 4500)
plt.show()

#%%


offset = image_data_77[100,i,j] - image_data_77[100,m,n]
scaling = (image_data_77[775,i,j] - image_data_77[100,i,j])/(image_data_77[775,m,n] - image_data_77[100,m,n])

offset = 0
scaling = 1

ax = plt.figure('7.7', figsize=(16,8)).add_subplot(111)
plt.title('7.7, 8.6: ' + str(j) + ', ' + str(i))
plt.plot(wavelengths77, image_data_77[:,i,j], label='strong 11.0')
plt.plot(wavelengths77, image_data_77[100,i,j] + scaling*(image_data_77[:,m,n] - image_data_77[100,m,n]), 
         label='weak 11.0, offset= ' + str(np.round(offset,3)) + ', scaling= ' + str(np.round(scaling, 3)), alpha=0.5)
plt.legend()
plt.ylim(1000, 8000)
plt.show()





#%%

#5x5 grid

i = 38
j = 49


ax = plt.figure('6.2', figsize=(8,8)).add_subplot(111)
plt.title('6.0 and 6.2: ' + str(j) + ', ' + str(i))
plt.plot(wavelengths1b, image_data_1b[:,i,j])
#plt.plot(orion_wavelengths_miri, orion_data_miri, color='black', alpha=0.5)
#plt.plot(wavelengths2b, np.log10(image_data_2b[:,i,j]), alpha=0.5)
#plt.plot(wavelengths2c, np.log10(100+image_data_2c[:,i,j]), alpha=0.5)
#plt.xlim(11.6,13.4)

#%%

ax = plt.figure('8.6', figsize=(8,8)).add_subplot(111)
plt.title('8.6: ' + str(j) + ', ' + str(i))
plt.plot(wavelengths77, image_data_77[:,i,j])
#plt.plot(orion_wavelengths_miri, orion_data_miri, color='black', alpha=0.5)
#plt.plot(wavelengths2b, np.log10(image_data_2b[:,i,j]), alpha=0.5)
#plt.plot(wavelengths2c, np.log10(100+image_data_2c[:,i,j]), alpha=0.5)
#plt.xlim(11.6,13.4)

#%%

ax = plt.figure('11.0', figsize=(8,8)).add_subplot(111)
plt.title('11.0: ' + str(j) + ', ' + str(i))
plt.plot(wavelengths2c, image_data_2c[:,i,j])
#plt.plot(orion_wavelengths_miri, orion_data_miri, color='black', alpha=0.5)
#plt.plot(wavelengths2b, np.log10(image_data_2b[:,i,j]), alpha=0.5)
#plt.plot(wavelengths2c, np.log10(100+image_data_2c[:,i,j]), alpha=0.5)
#plt.xlim(11.6,13.4)

#%%

ax = plt.figure(figsize=(8,8)).add_subplot(111)
plt.imshow(pah_intensity_62)
plt.colorbar()

plt.scatter(49, 36, s=10, color='red')


ax.invert_yaxis()
plt.show()

plt.show()


