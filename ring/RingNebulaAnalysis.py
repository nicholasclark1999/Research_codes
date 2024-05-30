
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:21:06 2023

@author: nclark
"""

'''
IMPORTING MODULES
'''
#%%
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

#needed for els' region function
import regions
from astropy.wcs import wcs
from astropy.stats import sigma_clip

#rebinning module
from reproject import reproject_interp

#Functions python script
import RingNebulaFunctions as rnf




'''
LIST OF RNF FUNCTIONS
'''

# loading_function
# loading_function1
# loading_function2
# border_remover
# weighted_mean_finder
# weighted_mean_finder_simple
# flux_aligner
# flux_aligner2
# emission_line_remover
# absorption_line_remover
# line_fitter
# fringe_remover
# extract_spectra_from_regions_one_pointing
# extract_spectra_from_regions_one_pointing_no_bkg
# regrid


####################################



'''
WHAT TO RUN
'''
weighted_mean_north = True
weighted_mean_west = True
weighted_mean_background = True
weighted_mean_north_unsub = False
weighted_mean_west_unsub = False
weighted_mean_nirspec = True
weighted_mean_spitzer = True
weighted_mean_small_fov = False

data_stitching = True





####################################

#%%

'''
LOADING DATA
'''

#calling MIRI_function
#naming is ordered from smallest to largest wavelength range
wavelengths1, image_data1, error_data1 = rnf.loading_function(
    'data/north/ring_neb_north_ch1-short_s3d.fits', 1)
wavelengths2, image_data2, error_data2 = rnf.loading_function(
    'data/north/ring_neb_north_ch1-medium_s3d.fits', 1)
wavelengths3, image_data3, error_data3 = rnf.loading_function(
    'data/north/ring_neb_north_ch1-long_s3d.fits', 1)
wavelengths4, image_data4, error_data4 = rnf.loading_function(
    'data/north/ring_neb_north_ch2-short_s3d.fits', 1)
wavelengths5, image_data5, error_data5 = rnf.loading_function(
    'data/north/ring_neb_north_ch2-medium_s3d.fits',1)
wavelengths6, image_data6, error_data6 = rnf.loading_function(
    'data/north/ring_neb_north_ch2-long_s3d.fits', 1)
wavelengths7, image_data7, error_data7 = rnf.loading_function(
    'data/north/ring_neb_north_ch3-short_s3d.fits', 1)
wavelengths8, image_data8, error_data8 = rnf.loading_function(
    'data/north/ring_neb_north_ch3-medium_s3d.fits', 1)
wavelengths9, image_data9, error_data9 = rnf.loading_function(
    'data/north/ring_neb_north_ch3-long_s3d.fits', 1)
wavelengths10, image_data10, error_data10 = rnf.loading_function(
    'data/north/ring_neb_north_ch4-short_s3d.fits', 1)
wavelengths11, image_data11, error_data11 = rnf.loading_function(
    'data/north/ring_neb_north_ch4-medium_s3d.fits', 1)
wavelengths12, image_data12, error_data12 = rnf.loading_function(
    'data/north/ring_neb_north_ch4-long_s3d.fits', 1)



#now the west region
wavelengths1_west, image_data1_west, error_data1_west = rnf.loading_function(
    'data/west/ring_neb_west_ch1-short_s3d.fits', 1)
wavelengths2_west, image_data2_west, error_data2_west = rnf.loading_function(
    'data/west/ring_neb_west_ch1-medium_s3d.fits', 1)
wavelengths3_west, image_data3_west, error_data3_west = rnf.loading_function(
    'data/west/ring_neb_west_ch1-long_s3d.fits', 1)
wavelengths4_west, image_data4_west, error_data4_west = rnf.loading_function(
    'data/west/ring_neb_west_ch2-short_s3d.fits', 1)
wavelengths5_west, image_data5_west, error_data5_west = rnf.loading_function(
    'data/west/ring_neb_west_ch2-medium_s3d.fits', 1)
wavelengths6_west, image_data6_west, error_data6_west = rnf.loading_function(
    'data/west/ring_neb_west_ch2-long_s3d.fits', 1)
wavelengths7_west, image_data7_west, error_data7_west = rnf.loading_function(
    'data/west/ring_neb_west_ch3-short_s3d.fits', 1)
wavelengths8_west, image_data8_west, error_data8_west = rnf.loading_function(
    'data/west/ring_neb_west_ch3-medium_s3d.fits', 1)
wavelengths9_west, image_data9_west, error_data9_west = rnf.loading_function(
    'data/west/ring_neb_west_ch3-long_s3d.fits', 1)
wavelengths10_west, image_data10_west, error_data10_west = rnf.loading_function(
    'data/west/ring_neb_west_ch4-short_s3d.fits', 1)
wavelengths11_west, image_data11_west, error_data11_west = rnf.loading_function(
    'data/west/ring_neb_west_ch4-medium_s3d.fits', 1)
wavelengths12_west, image_data12_west, error_data12_west = rnf.loading_function(
    'data/west/ring_neb_west_ch4-long_s3d.fits', 1)

#%%


#loading in NIRSPEC data.

#note that obs7 corresponds to 'west' and obs56 corresponds to 'north'

#g395m-f290, note using the aligned version for this one
wavelengths_nirspec4, nirspec_data4, nirspec_error_data4 = rnf.loading_function(
    'data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 1)

wavelengths_nirspec4_west, nirspec_data4_west, nirspec_error_data4_west = rnf.loading_function(
    'data/west/jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked.fits', 1)



#loading in spitzer data
spitzer_file_loc = 'data/spitzer/cube-spitzerSH/ring_spitzer_sh_cube.fits'
spitzer_image_file = get_pkg_data_filename(spitzer_file_loc)
spitzer_image_data = fits.getdata(spitzer_image_file, ext=0)
spitzer_wavelengths = fits.getdata(spitzer_image_file, ext=1)
spitzer_wavelengths = spitzer_wavelengths [0][0]



#loading in template PAH spectra
pah_image_file = np.loadtxt('data/misc/barh_stick_csub.fits.dat', skiprows=1)
pah_wavelengths = pah_image_file[:,0]
pah_data = pah_image_file[:,1]

'''

#loading in new stitched spectra for comparison

file_loc_stitched_north = 'data/MIRI_MRS/stitched_spectra/ring_neb_north_bkgsub_lvl3_asn_combine1dstep.fits'
image_file_stitched_north = get_pkg_data_filename(file_loc_stitched_north)
image_data_stitched_north = fits.getdata(image_file_stitched_north, ext=1)

data_stitched_north = np.zeros(len(image_data_stitched_north))
wavelengths_stitched_north = np.zeros(len(image_data_stitched_north))
for i in range(len(image_data_stitched_north)):
    data_stitched_north[i] = image_data_stitched_north[i][4]
    wavelengths_stitched_north[i] = image_data_stitched_north[i][0]

file_loc_stitched_west = 'data/MIRI_MRS/stitched_spectra/ring_neb_west_bkgsub_lvl3_asn_combine1dstep.fits'
image_file_stitched_west = get_pkg_data_filename(file_loc_stitched_west)
image_data_stitched_west = fits.getdata(image_file_stitched_west, ext=1)

data_stitched_west = np.zeros(len(image_data_stitched_west))
wavelengths_stitched_west = np.zeros(len(image_data_stitched_west))
for i in range(len(image_data_stitched_west)):
    data_stitched_west[i] = image_data_stitched_west[i][4]
    wavelengths_stitched_west[i] = image_data_stitched_west[i][0]

'''

####################################



'''
WEIGHTED MEAN
'''

#%%

#calculating weighted mean of MIRI data, north

#%%

if weighted_mean_north == True:

    #aligned, everything has the same fov
    data1, weighted_mean_error1 =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/north/ring_neb_north_ch1-short_s3d.fits', 
        image_data1, error_data1, wavelengths1, 'NIRSPEC_NORTH_bigbox_new.reg')
    
    data2, weighted_mean_error2 =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/north/ring_neb_north_ch1-medium_s3d.fits', 
        image_data2, error_data2, wavelengths2, 'NIRSPEC_NORTH_bigbox_new.reg')
    
    data3, weighted_mean_error3 =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/north/ring_neb_north_ch1-long_s3d.fits', 
        image_data3, error_data3, wavelengths3, 'NIRSPEC_NORTH_bigbox_new.reg')
    
    data4, weighted_mean_error4 =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/north/ring_neb_north_ch2-short_s3d.fits', 
        image_data4, error_data4, wavelengths4, 'NIRSPEC_NORTH_bigbox_new.reg')
    
    data5, weighted_mean_error5 =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/north/ring_neb_north_ch2-medium_s3d.fits', 
        image_data5, error_data5, wavelengths5, 'NIRSPEC_NORTH_bigbox_new.reg')
    
    data6, weighted_mean_error6 =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/north/ring_neb_north_ch2-long_s3d.fits', 
        image_data6, error_data6, wavelengths6, 'NIRSPEC_NORTH_bigbox_new.reg')
    
    data7, weighted_mean_error7 =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/north/ring_neb_north_ch3-short_s3d.fits', 
        image_data7, error_data7, wavelengths7, 'NIRSPEC_NORTH_bigbox_new.reg')
    
    data8, weighted_mean_error8 =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/north/ring_neb_north_ch3-medium_s3d.fits', 
        image_data8, error_data8, wavelengths8, 'NIRSPEC_NORTH_bigbox_new.reg')
    
    data9, weighted_mean_error9 =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/north/ring_neb_north_ch3-long_s3d.fits', 
        image_data9, error_data9, wavelengths9, 'NIRSPEC_NORTH_bigbox_new.reg')
    
    #saving data
    
    #accounts for Els' function making things a dictionary and then me saving it as an npy file
    data1 = np.array(data1['region_0'])
    data2 = np.array(data2['region_0'])
    data3 = np.array(data3['region_0'])
    data4 = np.array(data4['region_0'])
    data5 = np.array(data5['region_0'])
    data6 = np.array(data6['region_0'])
    data7 = np.array(data7['region_0'])
    data8 = np.array(data8['region_0'])
    data9 = np.array(data9['region_0'])

    np.save('Analysis/data1', data1)
    np.save('Analysis/data2', data2)
    np.save('Analysis/data3', data3)
    np.save('Analysis/data4', data4)
    np.save('Analysis/data5', data5)
    np.save('Analysis/data6', data6)
    np.save('Analysis/data7', data7)
    np.save('Analysis/data8', data8)
    np.save('Analysis/data9', data9)
    #np.save('Analysis/data10', data10)
    #np.save('Analysis/data11', data11)
    #np.save('Analysis/data12', data12)
    
    np.save('Analysis/weighted_mean_error1', weighted_mean_error1)
    np.save('Analysis/weighted_mean_error2', weighted_mean_error2)
    np.save('Analysis/weighted_mean_error3', weighted_mean_error3)
    np.save('Analysis/weighted_mean_error4', weighted_mean_error4)
    np.save('Analysis/weighted_mean_error5', weighted_mean_error5)
    np.save('Analysis/weighted_mean_error6', weighted_mean_error6)
    np.save('Analysis/weighted_mean_error7', weighted_mean_error7)
    np.save('Analysis/weighted_mean_error8', weighted_mean_error8)
    np.save('Analysis/weighted_mean_error9', weighted_mean_error9)
    #np.save('Analysis/weighted_mean_error10', weighted_mean_error10)
    #np.save('Analysis/weighted_mean_error11', weighted_mean_error11)
    #np.save('Analysis/weighted_mean_error12', weighted_mean_error12)

else:
    #loading data
    
    #data1 = np.load('Analysis/data1.npy')
    data2 = np.load('Analysis/data2.npy')
    data3 = np.load('Analysis/data3.npy')
    data4 = np.load('Analysis/data4.npy')
    data5 = np.load('Analysis/data5.npy')
    data6 = np.load('Analysis/data6.npy')
    data7 = np.load('Analysis/data7.npy')
    data8 = np.load('Analysis/data8.npy')
    data9 = np.load('Analysis/data9.npy')
    #data10 = np.load('Analysis/data10.npy')
    #data11 = np.load('Analysis/data11.npy')
    #data12 = np.load('Analysis/data12.npy')

    weighted_mean_error1 = np.load('Analysis/weighted_mean_error1.npy')
    weighted_mean_error2 = np.load('Analysis/weighted_mean_error2.npy')
    weighted_mean_error3 = np.load('Analysis/weighted_mean_error3.npy')
    weighted_mean_error4 = np.load('Analysis/weighted_mean_error4.npy')
    weighted_mean_error5 = np.load('Analysis/weighted_mean_error5.npy')
    weighted_mean_error6 = np.load('Analysis/weighted_mean_error6.npy')
    weighted_mean_error7 = np.load('Analysis/weighted_mean_error7.npy')
    weighted_mean_error8 = np.load('Analysis/weighted_mean_error8.npy')
    weighted_mean_error9 = np.load('Analysis/weighted_mean_error9.npy')
    #weighted_mean_error10 = np.load('Analysis/weighted_mean_error10.npy')
    #weighted_mean_error11 = np.load('Analysis/weighted_mean_error11.npy')
    #weighted_mean_error12 = np.load('Analysis/weighted_mean_error12.npy')

#%%

#calculating weighted mean of MIRI data, north

#%%

#unused version, that excludes the H2 filament
'''
if weighted_mean_west == True:

    #not aligned, everything has same fov
    data1_west, weighted_mean_error1_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg2(
        'data/west/ring_neb_west_ch1-short_s3d.fits', 
        image_data1_west, error_data1_west, wavelengths1_west,
        'ring_west_common_region.reg', 'NIRSPEC_WEST_bigblob.reg')

    data2_west, weighted_mean_error2_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg2(
        'data/west/ring_neb_west_ch1-medium_s3d.fits', 
        image_data2_west, error_data2_west, wavelengths2_west,
        'ring_west_common_region.reg', 'NIRSPEC_WEST_bigblob.reg')
    
    data3_west, weighted_mean_error3_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg2(
        'data/west/ring_neb_west_ch1-long_s3d.fits', 
        image_data3_west, error_data3_west, wavelengths3_west,
        'ring_west_common_region.reg', 'NIRSPEC_WEST_bigblob.reg')
    
    data4_west, weighted_mean_error4_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg2(
        'data/west/ring_neb_west_ch2-short_s3d.fits', 
        image_data4_west, error_data4_west, wavelengths4_west,
        'ring_west_common_region.reg', 'NIRSPEC_WEST_bigblob.reg')
    
    data5_west, weighted_mean_error5_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg2(
        'data/west/ring_neb_west_ch2-medium_s3d.fits', 
        image_data5_west, error_data5_west, wavelengths5_west,
        'ring_west_common_region.reg', 'NIRSPEC_WEST_bigblob.reg')
    
    data6_west, weighted_mean_error6_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg2(
        'data/west/ring_neb_west_ch2-long_s3d.fits', 
        image_data6_west, error_data6_west, wavelengths6_west,
        'ring_west_common_region.reg', 'NIRSPEC_WEST_bigblob.reg')
    
    data7_west, weighted_mean_error7_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg2(
        'data/west/ring_neb_west_ch3-short_s3d.fits', 
        image_data7_west, error_data7_west, wavelengths7_west,
        'ring_west_common_region.reg', 'NIRSPEC_WEST_bigblob.reg')
    
    data8_west, weighted_mean_error8_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg2(
        'data/west/ring_neb_west_ch3-medium_s3d.fits', 
        image_data8_west, error_data8_west, wavelengths8_west,
        'ring_west_common_region.reg', 'NIRSPEC_WEST_bigblob.reg')
    
    data9_west, weighted_mean_error9_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg2(
        'data/west/ring_neb_west_ch3-long_s3d.fits', 
        image_data9_west, error_data9_west, wavelengths9_west,
        'ring_west_common_region.reg', 'NIRSPEC_WEST_bigblob.reg')
'''


if weighted_mean_west == True:

    #not aligned, everything has same fov
    data1_west, weighted_mean_error1_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch1-short_s3d.fits', 
        image_data1_west, error_data1_west, wavelengths1_west,
        'ring_west_common_region.reg')

    data2_west, weighted_mean_error2_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch1-medium_s3d.fits', 
        image_data2_west, error_data2_west, wavelengths2_west,
        'ring_west_common_region.reg')
    
    data3_west, weighted_mean_error3_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch1-long_s3d.fits', 
        image_data3_west, error_data3_west, wavelengths3_west,
        'ring_west_common_region.reg')
    
    data4_west, weighted_mean_error4_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch2-short_s3d.fits', 
        image_data4_west, error_data4_west, wavelengths4_west,
        'ring_west_common_region.reg')
    
    data5_west, weighted_mean_error5_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch2-medium_s3d.fits', 
        image_data5_west, error_data5_west, wavelengths5_west,
        'ring_west_common_region.reg')
    
    data6_west, weighted_mean_error6_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch2-long_s3d.fits', 
        image_data6_west, error_data6_west, wavelengths6_west,
        'ring_west_common_region.reg')
    
    data7_west, weighted_mean_error7_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch3-short_s3d.fits', 
        image_data7_west, error_data7_west, wavelengths7_west,
        'ring_west_common_region.reg')
    
    data8_west, weighted_mean_error8_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch3-medium_s3d.fits', 
        image_data8_west, error_data8_west, wavelengths8_west,
        'ring_west_common_region.reg')
    
    data9_west, weighted_mean_error9_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch3-long_s3d.fits', 
        image_data9_west, error_data9_west, wavelengths9_west,
        'ring_west_common_region.reg')
    
    
    
    
    data1_west_blob, weighted_mean_error1_west_blob =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch1-short_s3d.fits', 
        image_data1_west, error_data1_west, wavelengths1_west,
        'NIRSPEC_WEST_bigblob.reg')
    
    data2_west_blob, weighted_mean_error2_west_blob =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch1-medium_s3d.fits', 
        image_data2_west, error_data2_west, wavelengths2_west,
        'NIRSPEC_WEST_bigblob.reg')
    
    data3_west_blob, weighted_mean_error3_west_blob =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch1-long_s3d.fits', 
        image_data3_west, error_data3_west, wavelengths3_west,
        'NIRSPEC_WEST_bigblob.reg')
    
    data4_west_blob, weighted_mean_error4_west_blob =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch2-short_s3d.fits', 
        image_data4_west, error_data4_west, wavelengths4_west,
        'NIRSPEC_WEST_bigblob.reg')
    
    data5_west_blob, weighted_mean_error5_west_blob =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch2-medium_s3d.fits', 
        image_data5_west, error_data5_west, wavelengths5_west,
        'NIRSPEC_WEST_bigblob.reg')
    
    data6_west_blob, weighted_mean_error6_west_blob =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch2-long_s3d.fits', 
        image_data6_west, error_data6_west, wavelengths6_west,
        'NIRSPEC_WEST_bigblob.reg')
    
    data7_west_blob, weighted_mean_error7_west_blob =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch3-short_s3d.fits', 
        image_data7_west, error_data7_west, wavelengths7_west, 
        'NIRSPEC_WEST_bigblob.reg')
    
    data8_west_blob, weighted_mean_error8_west_blob =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch3-medium_s3d.fits', 
        image_data8_west, error_data8_west, wavelengths8_west,
        'NIRSPEC_WEST_bigblob.reg')
    
    data9_west_blob, weighted_mean_error9_west_blob =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/ring_neb_west_ch3-long_s3d.fits', 
        image_data9_west, error_data9_west, wavelengths9_west, 
        'NIRSPEC_WEST_bigblob.reg')

    #saving data
    
    #accounts for Els' function making things a dictionary and then me saving it as an npy file
    data1_west = np.array(data1_west['region_0'])
    data2_west = np.array(data2_west['region_0'])
    data3_west = np.array(data3_west['region_0'])
    data4_west = np.array(data4_west['region_0'])
    data5_west = np.array(data5_west['region_0'])
    data6_west = np.array(data6_west['region_0'])
    data7_west = np.array(data7_west['region_0'])
    data8_west = np.array(data8_west['region_0'])
    data9_west = np.array(data9_west['region_0'])
    
    np.save('Analysis/data1_west', data1_west)
    np.save('Analysis/data2_west', data2_west)
    np.save('Analysis/data3_west', data3_west)
    np.save('Analysis/data4_west', data4_west)
    np.save('Analysis/data5_west', data5_west)
    np.save('Analysis/data6_west', data6_west)
    np.save('Analysis/data7_west', data7_west)
    np.save('Analysis/data8_west', data8_west)
    np.save('Analysis/data9_west', data9_west)
    #np.save('Analysis/data10_west', data10_west)
    #np.save('Analysis/data11_west', data11_west)
    #np.save('Analysis/data12_west', data12_west)

    np.save('Analysis/weighted_mean_error1_west', weighted_mean_error1_west)
    np.save('Analysis/weighted_mean_error2_west', weighted_mean_error2_west)
    np.save('Analysis/weighted_mean_error3_west', weighted_mean_error3_west)
    np.save('Analysis/weighted_mean_error4_west', weighted_mean_error4_west)
    np.save('Analysis/weighted_mean_error5_west', weighted_mean_error5_west)
    np.save('Analysis/weighted_mean_error6_west', weighted_mean_error6_west)
    np.save('Analysis/weighted_mean_error7_west', weighted_mean_error7_west)
    np.save('Analysis/weighted_mean_error8_west', weighted_mean_error8_west)
    np.save('Analysis/weighted_mean_error9_west', weighted_mean_error9_west)
    #np.save('Analysis/weighted_mean_error10_west', weighted_mean_error10_west)
    #np.save('Analysis/weighted_mean_error11_west', weighted_mean_error11_west)
    #np.save('Analysis/weighted_mean_error12_west', weighted_mean_error12_west)
    
    #accounts for Els' function making things a dictionary and then me saving it as an npy file
    data1_west_blob = np.array(data1_west_blob['region_0'])
    data2_west_blob = np.array(data2_west_blob['region_0'])
    data3_west_blob = np.array(data3_west_blob['region_0'])
    data4_west_blob = np.array(data4_west_blob['region_0'])
    data5_west_blob = np.array(data5_west_blob['region_0'])
    data6_west_blob = np.array(data6_west_blob['region_0'])
    data7_west_blob = np.array(data7_west_blob['region_0'])
    data8_west_blob = np.array(data8_west_blob['region_0'])
    data9_west_blob = np.array(data9_west_blob['region_0'])
    
    np.save('Analysis/data1_west_blob', data1_west_blob)
    np.save('Analysis/data2_west_blob', data2_west_blob)
    np.save('Analysis/data3_west_blob', data3_west_blob)
    np.save('Analysis/data4_west_blob', data4_west_blob)
    np.save('Analysis/data5_west_blob', data5_west_blob)
    np.save('Analysis/data6_west_blob', data6_west_blob)
    np.save('Analysis/data7_west_blob', data7_west_blob)
    np.save('Analysis/data8_west_blob', data8_west_blob)
    np.save('Analysis/data9_west_blob', data9_west_blob)
    
    np.save('Analysis/weighted_mean_error1_west_blob', weighted_mean_error1_west_blob)
    np.save('Analysis/weighted_mean_error2_west_blob', weighted_mean_error2_west_blob)
    np.save('Analysis/weighted_mean_error3_west_blob', weighted_mean_error3_west_blob)
    np.save('Analysis/weighted_mean_error4_west_blob', weighted_mean_error4_west_blob)
    np.save('Analysis/weighted_mean_error5_west_blob', weighted_mean_error5_west_blob)
    np.save('Analysis/weighted_mean_error6_west_blob', weighted_mean_error6_west_blob)
    np.save('Analysis/weighted_mean_error7_west_blob', weighted_mean_error7_west_blob)
    np.save('Analysis/weighted_mean_error8_west_blob', weighted_mean_error8_west_blob)
    np.save('Analysis/weighted_mean_error9_west_blob', weighted_mean_error9_west_blob)

else:
    #loading data
    data1_west = np.load('Analysis/data1_west.npy')
    data2_west = np.load('Analysis/data2_west.npy')
    data3_west = np.load('Analysis/data3_west.npy')
    data4_west = np.load('Analysis/data4_west.npy')
    data5_west = np.load('Analysis/data5_west.npy')
    data6_west = np.load('Analysis/data6_west.npy')
    data7_west = np.load('Analysis/data7_west.npy')
    data8_west = np.load('Analysis/data8_west.npy')
    data9_west = np.load('Analysis/data9_west.npy')
    #data10_west = np.load('Analysis/data10_west.npy')
    #data11_west = np.load('Analysis/data11_west.npy')
    #data12_west = np.load('Analysis/data12_west.npy')
    
    weighted_mean_error1_west = np.load('Analysis/weighted_mean_error1_west.npy')
    weighted_mean_error2_west = np.load('Analysis/weighted_mean_error2_west.npy')
    weighted_mean_error3_west = np.load('Analysis/weighted_mean_error3_west.npy')
    weighted_mean_error4_west = np.load('Analysis/weighted_mean_error4_west.npy')
    weighted_mean_error5_west = np.load('Analysis/weighted_mean_error5_west.npy')
    weighted_mean_error6_west = np.load('Analysis/weighted_mean_error6_west.npy')
    weighted_mean_error7_west = np.load('Analysis/weighted_mean_error7_west.npy')
    weighted_mean_error8_west = np.load('Analysis/weighted_mean_error8_west.npy')
    weighted_mean_error9_west = np.load('Analysis/weighted_mean_error9_west.npy')
    #weighted_mean_error10_west = np.load('Analysis/weighted_mean_error10_west.npy')
    #weighted_mean_error11_west = np.load('Analysis/weighted_mean_error11_west.npy')
    #weighted_mean_error12_west = np.load('Analysis/weighted_mean_error12_west.npy')
    
    data1_west_blob = np.load('Analysis/data1_west_blob.npy')
    data2_west_blob = np.load('Analysis/data2_west_blob.npy')
    data3_west_blob = np.load('Analysis/data3_west_blob.npy')
    data4_west_blob = np.load('Analysis/data4_west_blob.npy')
    data5_west_blob = np.load('Analysis/data5_west_blob.npy')
    data6_west_blob = np.load('Analysis/data6_west_blob.npy')
    data7_west_blob = np.load('Analysis/data7_west_blob.npy')
    data8_west_blob = np.load('Analysis/data8_west_blob.npy')
    data9_west_blob = np.load('Analysis/data9_west_blob.npy')
    
    weighted_mean_error1_west_blob = np.load('Analysis/weighted_mean_error1_west_blob.npy')
    weighted_mean_error2_west_blob = np.load('Analysis/weighted_mean_error2_west_blob.npy')
    weighted_mean_error3_west_blob = np.load('Analysis/weighted_mean_error3_west_blob.npy')
    weighted_mean_error4_west_blob = np.load('Analysis/weighted_mean_error4_west_blob.npy')
    weighted_mean_error5_west_blob = np.load('Analysis/weighted_mean_error5_west_blob.npy')
    weighted_mean_error6_west_blob = np.load('Analysis/weighted_mean_error6_west_blob.npy')
    weighted_mean_error7_west_blob = np.load('Analysis/weighted_mean_error7_west_blob.npy')
    weighted_mean_error8_west_blob = np.load('Analysis/weighted_mean_error8_west_blob.npy')
    weighted_mean_error9_west_blob = np.load('Analysis/weighted_mean_error9_west_blob.npy')

#%%

#calculating weighted mean of NIRSPEC data

#%%

if weighted_mean_nirspec == True:
    nirspec_weighted_mean4, nirspec_error_mean4 =\
        rnf.extract_spectra_from_regions_one_pointing_no_bkg('data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', nirspec_data4, nirspec_error_data4, wavelengths_nirspec4, 'NIRSPEC_NORTH_bigbox_new.reg')

    #accounts for Els' function making things a dictionary and then me saving it as an npy file
    nirspec_weighted_mean4 = np.array(nirspec_weighted_mean4['region_0'])
        
    nirspec_weighted_mean4_west, nirspec_error_mean4_west =\
        rnf.extract_spectra_from_regions_one_pointing_no_bkg2(
            'data/west/jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked.fits', 
            nirspec_data4_west, nirspec_error_data4_west, wavelengths_nirspec4_west, 
            'ring_west_common_region.reg', 'NIRSPEC_WEST_bigblob.reg')
    
    #accounts for Els' function making things a dictionary and then me saving it as an npy file
    nirspec_weighted_mean4_west = np.array(nirspec_weighted_mean4_west['region_0'])
    
    nirspec_weighted_mean4_west_blob, nirspec_error_mean4_west_blob =\
        rnf.extract_spectra_from_regions_one_pointing_no_bkg(
            'data/west/jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked.fits', 
            nirspec_data4_west, nirspec_error_data4_west, wavelengths_nirspec4_west, 'NIRSPEC_WEST_bigblob.reg')
    
    #accounts for Els' function making things a dictionary and then me saving it as an npy file
    nirspec_weighted_mean4_west_blob = np.array(nirspec_weighted_mean4_west_blob['region_0'])
    
    #saving data
    
    np.save('Analysis/nirspec_weighted_mean4', nirspec_weighted_mean4)
    
    np.save('Analysis/nirspec_weighted_mean4_west', nirspec_weighted_mean4_west)
    
    np.save('Analysis/nirspec_weighted_mean4_west_blob', nirspec_weighted_mean4_west_blob)
    
    np.save('Analysis/nirspec_error_mean4', nirspec_error_mean4)
    
    np.save('Analysis/nirspec_error_mean4_west', nirspec_error_mean4_west)
    
    np.save('Analysis/nirspec_error_mean4_west_blob', nirspec_error_mean4_west_blob)
    
else:
    #loading data
    nirspec_weighted_mean4 = np.load('Analysis/nirspec_weighted_mean4.npy')

    nirspec_weighted_mean4_west = np.load('Analysis/nirspec_weighted_mean4_west.npy')
    
    nirspec_weighted_mean4_west_blob = np.load('Analysis/nirspec_weighted_mean4_west_blob.npy')

    nirspec_error_mean4 = np.load('Analysis/nirspec_error_mean4.npy')

    nirspec_error_mean4_west = np.load('Analysis/nirspec_error_mean4_west.npy')
    
    nirspec_error_mean4_west_blob = np.load('Analysis/nirspec_error_mean4_west_blob.npy')

#%%

#calculating weighted mean of spitzer data

#%%

if weighted_mean_spitzer == True:
    #spitzer has no error data so do a regular mean for it
    spitzer_data = np.mean(spitzer_image_data[:,32:37, 4:], axis=(1,2))
    
    #saving data
    
    np.save('Analysis/spitzer_data', spitzer_data)



####################################



'''
REMOVING EMISSION AND ABSORPTION LINES
'''

'''
#applying functions to example spectra for 11.2 region
lines_removed_north = rnf.emission_line_remover(image_data6[:,16,17], 5, 5)
lines_removed_north = rnf.absorption_line_remover(lines_removed_north, 5, 5)

lines_removed_west = rnf.emission_line_remover(image_data6_west[:,16,17], 5, 5)
lines_removed_west = rnf.absorption_line_remover(lines_removed_west, 5, 5)

'''
#%%
#removing emission and absorption lines for NIRSPEC data
#nirspec_weighted_mean1_smooth = rnf.emission_line_remover(nirspec_weighted_mean1, 6, 50)
#nirspec_weighted_mean2_smooth = rnf.emission_line_remover(nirspec_weighted_mean2, 8, 50) 
#nirspec_weighted_mean3_smooth = rnf.emission_line_remover(nirspec_weighted_mean3, 9, 50) 
nirspec_weighted_mean4_smooth = rnf.emission_line_remover(nirspec_weighted_mean4, 4, 4)

#nirspec_weighted_mean1_smooth = rnf.absorption_line_remover(nirspec_weighted_mean1_smooth, 6, 50)
#nirspec_weighted_mean2_smooth = rnf.absorption_line_remover(nirspec_weighted_mean2_smooth, 8, 50) 
#nirspec_weighted_mean3_smooth = rnf.absorption_line_remover(nirspec_weighted_mean3_smooth, 9, 50) 
nirspec_weighted_mean4_smooth = rnf.absorption_line_remover(nirspec_weighted_mean4_smooth, 4, 4)

#nirspec_weighted_mean1_smooth_west = rnf.emission_line_remover(nirspec_weighted_mean1_west, 6, 50)
#nirspec_weighted_mean2_smooth_west = rnf.emission_line_remover(nirspec_weighted_mean2_west, 8, 50) 
#nirspec_weighted_mean3_smooth_west = rnf.emission_line_remover(nirspec_weighted_mean3_west, 9, 50) 
nirspec_weighted_mean4_smooth_west = rnf.emission_line_remover(nirspec_weighted_mean4_west, 4, 4)

#nirspec_weighted_mean1_smooth_west = rnf.absorption_line_remover(nirspec_weighted_mean1_smooth_west, 6, 50)
#nirspec_weighted_mean2_smooth_west = rnf.absorption_line_remover(nirspec_weighted_mean2_smooth_west, 8, 50) 
#nirspec_weighted_mean3_smooth_west = rnf.absorption_line_remover(nirspec_weighted_mean3_smooth_west, 9, 50) 
nirspec_weighted_mean4_smooth_west = rnf.absorption_line_remover(nirspec_weighted_mean4_smooth_west, 4, 4) 

#%%
data9 = rnf.emission_line_remover(data9, 4, 50) 
#%%
#saving data

np.save('Analysis/nirspec_weighted_mean4_smooth', nirspec_weighted_mean4_smooth)
np.save('Analysis/nirspec_weighted_mean4_smooth_west', nirspec_weighted_mean4_smooth_west)

#%%

####################################
#%%
'''
FRINGE REMOVAL
'''
'''
import rfc1d_utils

#get wavenumber
wavenum1 = 10000.0 / wavelengths1
wavenum2 = 10000.0 / wavelengths2
wavenum3 = 10000.0 / wavelengths3
wavenum4 = 10000.0 / wavelengths4
wavenum5 = 10000.0 / wavelengths5
wavenum6 = 10000.0 / wavelengths6
wavenum7 = 10000.0 / wavelengths7
wavenum8 = 10000.0 / wavelengths8
wavenum9 = 10000.0 / wavelengths9
#wavenum10 = 10000.0 / wavelengths10
#wavenum11 = 10000.0 / wavelengths11
#wavenum12 = 10000.0 / wavelengths12

wavenum1_west = 10000.0 / wavelengths1_west
wavenum2_west = 10000.0 / wavelengths2_west
wavenum3_west = 10000.0 / wavelengths3_west
wavenum4_west = 10000.0 / wavelengths4_west
wavenum5_west = 10000.0 / wavelengths5_west
wavenum6_west = 10000.0 / wavelengths6_west
wavenum7_west = 10000.0 / wavelengths7_west
wavenum8_west = 10000.0 / wavelengths8_west
wavenum9_west = 10000.0 / wavelengths9_west

# get weights

#adding 10 because fringe removal hates negatives

weights_data1 = (data1) / np.median(data1)
weights_data2 = (data2 ) / np.median(data2) 
weights_data3 = (data3 ) / np.median(data3)
weights_data4 = (data4) / np.median(data4)
weights_data5 = (data5 ) / np.median(data5)
weights_data6 = (data6 ) / np.median(data6)
weights_data7 = (data7) / np.median(data7)
weights_data8 = (data8) / np.median(data8)
weights_data9 = (data9) / np.median(data9)
#weights_data10 = data10 / np.median(data10)
#weights_data11 = data11 / np.median(data11)
#weights_data12 = data12 / np.median(data12)

weights_data1_west = (data1_west+10) / np.median(data1_west+10)
weights_data2_west = (data2_west+10) / np.median(data2_west+10)
weights_data3_west = (data3_west) / np.median(data3_west)
weights_data4_west = (data4_west+10) / np.median(data4_west+10)
weights_data5_west = (data5_west+10) / np.median(data5_west+10)
weights_data6_west = (data6_west) / np.median(data6_west)
weights_data7_west = (data7_west) / np.median(data7_west)
weights_data8_west = (data8_west) / np.median(data8_west)
weights_data9_west = (data9_west) / np.median(data9_west)
#weights_data10_west = data10_west / np.median(data10_west)
#weights_data11_west = data11_west / np.median(data11_west)
#weights_data12_west = data12_west / np.median(data12_west)

weights_data1_west_blob = (data1_west_blob+10) / np.median(data1_west_blob+10)
weights_data2_west_blob = (data2_west_blob+10) / np.median(data2_west_blob+10)
weights_data3_west_blob = (data3_west_blob) / np.median(data3_west_blob)
weights_data4_west_blob = (data4_west_blob+10) / np.median(data4_west_blob+10)
weights_data5_west_blob = (data5_west_blob+10) / np.median(data5_west_blob+10)
weights_data6_west_blob = (data6_west_blob) / np.median(data6_west_blob)
weights_data7_west_blob = (data7_west_blob) / np.median(data7_west_blob)
weights_data8_west_blob = (data8_west_blob) / np.median(data8_west_blob)
weights_data9_west_blob = (data9_west_blob) / np.median(data9_west_blob)

# set the channel
channel = 1

corrected_data1 = rfc1d_utils.fit_residual_fringes(data1+10, weights_data1, wavenum1, int(channel), plots=False)
corrected_data2 = rfc1d_utils.fit_residual_fringes(data2+10, weights_data2, wavenum2, int(channel), plots=False)
corrected_data3 = rfc1d_utils.fit_residual_fringes(data3, weights_data3, wavenum3, int(channel), plots=False)

corrected_data1_west = rfc1d_utils.fit_residual_fringes(data1_west+10, weights_data1_west, wavenum1_west, int(channel), plots=False)
corrected_data2_west = rfc1d_utils.fit_residual_fringes(data2_west+10, weights_data2_west, wavenum2_west, int(channel), plots=False)
corrected_data3_west = rfc1d_utils.fit_residual_fringes(data3_west, weights_data3_west, wavenum3_west, int(channel), plots=False)

corrected_data1_west_blob = rfc1d_utils.fit_residual_fringes(data1_west_blob+10, weights_data1_west_blob, wavenum1_west, int(channel), plots=False)
corrected_data2_west_blob = rfc1d_utils.fit_residual_fringes(data2_west_blob+10, weights_data2_west_blob, wavenum2_west, int(channel), plots=False)
corrected_data3_west_blob = rfc1d_utils.fit_residual_fringes(data3_west_blob, weights_data3_west_blob, wavenum3_west, int(channel), plots=False)

corrected_data1 = corrected_data1 - 10
corrected_data2 = corrected_data2 - 10

corrected_data1_west = corrected_data1_west - 10
corrected_data2_west = corrected_data2_west - 10

corrected_data1_west_blob = corrected_data1_west_blob - 10
corrected_data2_west_blob = corrected_data2_west_blob - 10

# set the channel
channel = 2

corrected_data4 = rfc1d_utils.fit_residual_fringes(data4, weights_data4, wavenum4, int(channel), plots=False)
corrected_data5 = rfc1d_utils.fit_residual_fringes(data5, weights_data5, wavenum5, int(channel), plots=False)
corrected_data6 = rfc1d_utils.fit_residual_fringes(data6, weights_data6, wavenum6, int(channel), plots=False)

corrected_data4_west = rfc1d_utils.fit_residual_fringes(data4_west+10, weights_data4_west, wavenum4_west, int(channel), plots=False)
corrected_data5_west = rfc1d_utils.fit_residual_fringes(data5_west+10, weights_data5_west, wavenum5_west, int(channel), plots=False)
corrected_data6_west = rfc1d_utils.fit_residual_fringes(data6_west, weights_data6_west, wavenum6_west, int(channel), plots=False)

corrected_data4_west_blob = rfc1d_utils.fit_residual_fringes(data4_west_blob+10, weights_data4_west_blob, wavenum4_west, int(channel), plots=False)
corrected_data5_west_blob = rfc1d_utils.fit_residual_fringes(data5_west_blob+10, weights_data5_west_blob, wavenum5_west, int(channel), plots=False)
corrected_data6_west_blob = rfc1d_utils.fit_residual_fringes(data6_west_blob, weights_data6_west_blob, wavenum6_west, int(channel), plots=False)

corrected_data4_west = corrected_data4_west - 10
corrected_data5_west = corrected_data5_west - 10

corrected_data4_west_blob = corrected_data4_west_blob - 10
corrected_data5_west_blob = corrected_data5_west_blob - 10

# set the channel
channel = 3

corrected_data7 = rfc1d_utils.fit_residual_fringes(data7, weights_data7, wavenum7, int(channel), plots=False)
corrected_data8 = rfc1d_utils.fit_residual_fringes(data8, weights_data8, wavenum8, int(channel), plots=False)
corrected_data9 = rfc1d_utils.fit_residual_fringes(data9, weights_data9, wavenum9, int(channel), plots=False)

corrected_data7_west = rfc1d_utils.fit_residual_fringes(data7_west, weights_data7_west, wavenum7_west, int(channel), plots=False)
corrected_data8_west = rfc1d_utils.fit_residual_fringes(data8_west, weights_data8_west, wavenum8_west, int(channel), plots=False)
corrected_data9_west = rfc1d_utils.fit_residual_fringes(data9_west, weights_data9_west, wavenum9_west, int(channel), plots=False)

corrected_data7_west_blob = rfc1d_utils.fit_residual_fringes(data7_west_blob, weights_data7_west_blob, wavenum7_west, int(channel), plots=False)
corrected_data8_west_blob = rfc1d_utils.fit_residual_fringes(data8_west_blob, weights_data8_west_blob, wavenum8_west, int(channel), plots=False)
corrected_data9_west_blob = rfc1d_utils.fit_residual_fringes(data9_west_blob, weights_data9_west_blob, wavenum9_west, int(channel), plots=False)
'''
#saving data

corrected_data1 = data1
corrected_data2 = data2
corrected_data3 = data3
corrected_data4 = data4
corrected_data5 = data5
corrected_data6 = data6
corrected_data7 = data7
corrected_data8 = data8
corrected_data9 = data9

corrected_data1_west = data1_west
corrected_data2_west = data2_west
corrected_data3_west = data3_west
corrected_data4_west = data4_west
corrected_data5_west = data5_west
corrected_data6_west = data6_west
corrected_data7_west = data7_west
corrected_data8_west = data8_west
corrected_data9_west = data9_west

corrected_data1_west_blob = data1_west_blob
corrected_data2_west_blob = data2_west_blob
corrected_data3_west_blob = data3_west_blob
corrected_data4_west_blob = data4_west_blob
corrected_data5_west_blob = data5_west_blob
corrected_data6_west_blob = data6_west_blob
corrected_data7_west_blob = data7_west_blob
corrected_data8_west_blob = data8_west_blob
corrected_data9_west_blob = data9_west_blob

np.save('Analysis/corrected_data1', corrected_data1)
np.save('Analysis/corrected_data2', corrected_data2)
np.save('Analysis/corrected_data3', corrected_data3)
np.save('Analysis/corrected_data4', corrected_data4)
np.save('Analysis/corrected_data5', corrected_data5)
np.save('Analysis/corrected_data6', corrected_data6)
np.save('Analysis/corrected_data7', corrected_data7)
np.save('Analysis/corrected_data8', corrected_data8)
np.save('Analysis/corrected_data9', corrected_data9)

np.save('Analysis/corrected_data1_west', corrected_data1_west)
np.save('Analysis/corrected_data2_west', corrected_data2_west)
np.save('Analysis/corrected_data3_west', corrected_data3_west)
np.save('Analysis/corrected_data4_west', corrected_data4_west)
np.save('Analysis/corrected_data5_west', corrected_data5_west)
np.save('Analysis/corrected_data6_west', corrected_data6_west)
np.save('Analysis/corrected_data7_west', corrected_data7_west)
np.save('Analysis/corrected_data8_west', corrected_data8_west)
np.save('Analysis/corrected_data9_west', corrected_data9_west)

np.save('Analysis/corrected_data1_west_blob', corrected_data1_west_blob)
np.save('Analysis/corrected_data2_west_blob', corrected_data2_west_blob)
np.save('Analysis/corrected_data3_west_blob', corrected_data3_west_blob)
np.save('Analysis/corrected_data4_west_blob', corrected_data4_west_blob)
np.save('Analysis/corrected_data5_west_blob', corrected_data5_west_blob)
np.save('Analysis/corrected_data6_west_blob', corrected_data6_west_blob)
np.save('Analysis/corrected_data7_west_blob', corrected_data7_west_blob)
np.save('Analysis/corrected_data8_west_blob', corrected_data8_west_blob)
np.save('Analysis/corrected_data9_west_blob', corrected_data9_west_blob)



####################################

#%%

#note nirspec has no background, so make an array of 0's to be it












#%%
'''
data1, weighted_mean_error1 =\
rnf.extract_spectra_from_regions_one_pointing(
        'data/north/ring_neb_north_ch1-short_s3d.fits', 
        image_data1, error_data1, wavelengths1, image_data1_off, error_data1_off, 'NIRSPEC_NORTH_bigbox_new.reg')
'''

#%%

data67, frog =\
rnf.weighted_mean_finder_simple(image_data6[:, 10:30, 10:30], error_data6[:, 10:30, 10:30])


#%%
plt.figure()
plt.ylim(-10,60)
plt.plot(wavelengths6, data6)
plt.plot(wavelengths6, corrected_data6)
#plt.plot(wavelengths6, corrected_data6)
plt.show()


    






