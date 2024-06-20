
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:21:06 2023

@author: nclark
"""
'''
IMPORTANT INFO
'''

# The data used for analysis has been reprojected, to have the same FOV and pixel
# size as nirspec g395m-f290 data. 

# Due to working with large extraction apertures, PSF matching has not been done.

# Currently, fringe correction is not performed.



####################################



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
# bkg_sub_and_weighted_mean_finder
# weighted_mean_finder
# extract_weighted_mean_from_region
# extract_regular_mean_from_region
# extract_weighted_mean_slice_from_region
# extract_pixels_from_region
# flux_aligner_offset
# flux_aligner_manual
# emission_line_remover
# absorption_line_remover
# line_fitter
# linear_continuum_single_channel
# unit_changer
# pah_feature_integrator
# pah_feature_integrator_no_units
# Calculate_R
# error_finder
# border_remover
# regrid




####################################



'''
LOADING DATA
'''

#calling MIRI_function
#naming is according to miri convention of short = a, medium = b, long = c
wavelengths1a, image_data_1a, error_data_1a = rnf.loading_function(
    'data/north/ring_neb_north_ch1-short_s3d.fits', 1)
wavelengths1b, image_data_1b, error_data_1b = rnf.loading_function(
    'data/north/ring_neb_north_ch1-medium_s3d.fits', 1)
wavelengths1c, image_data_1c, error_data_1c = rnf.loading_function(
    'data/north/ring_neb_north_ch1-long_s3d.fits', 1)
wavelengths2a, image_data_2a, error_data_2a = rnf.loading_function(
    'data/north/ring_neb_north_ch2-short_s3d.fits', 1)
wavelengths2b, image_data_2b, error_data_2b = rnf.loading_function(
    'data/north/ring_neb_north_ch2-medium_s3d.fits',1)
wavelengths2c, image_data_2c, error_data_2c = rnf.loading_function(
    'data/north/ring_neb_north_ch2-long_s3d.fits', 1)
wavelengths3a, image_data_3a, error_data_3a = rnf.loading_function(
    'data/north/ring_neb_north_ch3-short_s3d.fits', 1)
wavelengths3b, image_data_3b, error_data_3b = rnf.loading_function(
    'data/north/ring_neb_north_ch3-medium_s3d.fits', 1)
wavelengths3c, image_data_3c, error_data_3c = rnf.loading_function(
    'data/north/ring_neb_north_ch3-long_s3d.fits', 1)
wavelengths4a, image_data_4a, error_data_4a = rnf.loading_function(
    'data/north/ring_neb_north_ch4-short_s3d.fits', 1)
wavelengths4b, image_data_4b, error_data_4b = rnf.loading_function(
    'data/north/ring_neb_north_ch4-medium_s3d.fits', 1)
wavelengths4c, image_data_4c, error_data_4c = rnf.loading_function(
    'data/north/ring_neb_north_ch4-long_s3d.fits', 1)



#now the west region
wavelengths1a_west, image_data_1a_west, error_data_1a_west = rnf.loading_function(
    'data/west/ring_neb_west_ch1-short_s3d.fits', 1)
wavelengths1b_west, image_data_1b_west, error_data_1b_west = rnf.loading_function(
    'data/west/ring_neb_west_ch1-medium_s3d.fits', 1)
wavelengths1c_west, image_data_1c_west, error_data_1c_west = rnf.loading_function(
    'data/west/ring_neb_west_ch1-long_s3d.fits', 1)
wavelengths2a_west, image_data_2a_west, error_data_2a_west = rnf.loading_function(
    'data/west/ring_neb_west_ch2-short_s3d.fits', 1)
wavelengths2b_west, image_data_2b_west, error_data_2b_west = rnf.loading_function(
    'data/west/ring_neb_west_ch2-medium_s3d.fits', 1)
wavelengths2c_west, image_data_2c_west, error_data_2c_west = rnf.loading_function(
    'data/west/ring_neb_west_ch2-long_s3d.fits', 1)
wavelengths3a_west, image_data_3a_west, error_data_3a_west = rnf.loading_function(
    'data/west/ring_neb_west_ch3-short_s3d.fits', 1)
wavelengths3b_west, image_data_3b_west, error_data_3b_west = rnf.loading_function(
    'data/west/ring_neb_west_ch3-medium_s3d.fits', 1)
wavelengths3c_west, image_data_3c_west, error_data_3c_west = rnf.loading_function(
    'data/west/ring_neb_west_ch3-long_s3d.fits', 1)
wavelengths4a_west, image_data_4a_west, error_data_4a_west = rnf.loading_function(
    'data/west/ring_neb_west_ch4-short_s3d.fits', 1)
wavelengths4b_west, image_data_4b_west, error_data_4b_west = rnf.loading_function(
    'data/west/ring_neb_west_ch4-medium_s3d.fits', 1)
wavelengths4c_west, image_data_4c_west, error_data_4c_west = rnf.loading_function(
    'data/west/ring_neb_west_ch4-long_s3d.fits', 1)



#loading in NIRSPEC data.

#note that obs7 corresponds to 'west' and obs56 corresponds to 'north'

#g395m-f290, note using the aligned version for this one
wavelengths_nirspec, nirspec_data, nirspec_error_data = rnf.loading_function(
    'data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 1)

wavelengths_nirspec_west, nirspec_data_west, nirspec_error_data_west = rnf.loading_function(
    'data/west/jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked.fits', 1)



#g235m-f170, note that this is only used for making synthetic cams
wavelengths_nirspec_short, nirspec_data_short, nirspec_error_data_short = rnf.loading_function(
    'data/north/jw01558-o056_t005_nirspec_g235m-f170lp_s3d_masked_aligned.fits', 1)

wavelengths_nirspec_short_west, nirspec_data_short_west, nirspec_error_data_short_west = rnf.loading_function(
    'data/west/jw01558-o008_t007_nirspec_g235m-f170lp_s3d_masked.fits', 1)



#loading in JWST cam data
cam_file_loc_f1000w = 'data/cams/jw01558-o001_t001_miri_f1000w/jw01558-o001_t001_miri_f1000w_i2d.fits'
cam_file_f1000w = get_pkg_data_filename(cam_file_loc_f1000w)
cam_data_f1000w = fits.getdata(cam_file_f1000w, ext=1)
cam_error_data_f1000w = fits.getdata(cam_file_f1000w, ext=2)

cam_file_loc_f1130w = 'data/cams/jw01558005001_04101_00001_nrcb1_combined_i2d.fits'
cam_file_f1130w = get_pkg_data_filename(cam_file_loc_f1130w)
cam_data_f1130w = fits.getdata(cam_file_f1130w, ext=1)
cam_error_data_f1130w = fits.getdata(cam_file_f1130w, ext=2)

cam_file_loc_f300m = 'data/cams/jw01558005001_04103_00001_nrcblong_combined_i2d.fits'
cam_file_f300m = get_pkg_data_filename(cam_file_loc_f300m)
cam_data_f300m = fits.getdata(cam_file_f300m, ext=1)
cam_error_data_f300m = fits.getdata(cam_file_f300m, ext=2)

cam_file_loc_f335m = 'data/cams/jw01558005001_04101_00001_nrcblong_combined_i2d.fits'
cam_file_f335m = get_pkg_data_filename(cam_file_loc_f335m)
cam_data_f335m = fits.getdata(cam_file_f335m, ext=1)
cam_error_data_f335m = fits.getdata(cam_file_f335m, ext=2)



#loading in JWST CAM ratio maps
nircam_file_loc = 'data/cams/nircam_color_F300M_F335M.fits'
nircam_image_file = get_pkg_data_filename(nircam_file_loc)
nircam_data = fits.getdata(nircam_image_file, ext=1)
nircam_error_data = fits.getdata(nircam_image_file, ext=2)

miricam_file_loc = 'data/cams/miri_color_F1000W_F1130W.fits'
miricam_image_file = get_pkg_data_filename(miricam_file_loc)
miricam_data = fits.getdata(miricam_image_file, ext=1)
miricam_error_data = fits.getdata(miricam_image_file, ext=2)



#loading in spitzer ring nebula data (Cox 2016)
spitzer_file_loc = 'data/spitzer/cube-spitzerSH/ring_spitzer_sh_cube.fits'
spitzer_image_file = get_pkg_data_filename(spitzer_file_loc)
spitzer_image_data = fits.getdata(spitzer_image_file, ext=0)
spitzer_wavelengths = fits.getdata(spitzer_image_file, ext=1)
spitzer_wavelengths = spitzer_wavelengths [0][0]



#loading in spitzer ring nebula data (ISO)
iso_sl1_image_file = np.loadtxt('data/misc/ngc6702_all_SL1_peak112_noheader.tbl', skiprows=3)
iso_sl2_image_file = np.loadtxt('data/misc/ngc6702_all_SL2_peak112_noheader.tbl', skiprows=3)
iso_sl1_wavelengths = iso_sl1_image_file[:,0]
iso_sl2_wavelengths = iso_sl2_image_file[:,0]
iso_sl1_data = iso_sl1_image_file[:,1]
iso_sl2_data = iso_sl2_image_file[:,1]



#loading in template PAH spectra, from the Orion bar (JWST)
orion_image_file_nirspec = np.loadtxt('data/misc/templatesT_seg3_crds1084_20230514.dat', skiprows=7)
orion_image_file_miri = np.loadtxt('data/misc/templatesT_MRS_crds1154_add_20231212.dat', skiprows=7)
orion_wavelengths_nirspec = orion_image_file_nirspec[:,0]
orion_wavelengths_miri = orion_image_file_miri[:,0]
orion_data_nirspec = orion_image_file_nirspec[:,6]
orion_data_miri = orion_image_file_miri[:,9]



#loading in horsehead nebula PAH spectra (JWST)
hh_wavelengths2, hh_image_data_2, hh_error_2 = rnf.loading_function(
    'data/misc/jw01192-o010_t002_miri_ch2-shortmediumlong_s3d.fits', 1)
hh_wavelengths3, hh_image_data_3, hh_error_3 = rnf.loading_function(
    'data/misc/jw01192-o010_t002_miri_ch3-shortmediumlong_s3d.fits', 1)



####################################



'''
WEIGHTED MEAN
'''



#calculating weighted mean of MIRI data, north
data1a, weighted_mean_error_1a =\
rnf.extract_weighted_mean_from_region(
    'data/north/ring_neb_north_ch1-short_s3d.fits', 
    image_data_1a, error_data_1a, 'NIRSPEC_NORTH_bigbox_new.reg')

data1b, weighted_mean_error_1b =\
rnf.extract_weighted_mean_from_region(
    'data/north/ring_neb_north_ch1-medium_s3d.fits', 
    image_data_1b, error_data_1b, 'NIRSPEC_NORTH_bigbox_new.reg')

data1c, weighted_mean_error_1c =\
rnf.extract_weighted_mean_from_region(
    'data/north/ring_neb_north_ch1-long_s3d.fits', 
    image_data_1c, error_data_1c, 'NIRSPEC_NORTH_bigbox_new.reg')

data2a, weighted_mean_error_2a =\
rnf.extract_weighted_mean_from_region(
    'data/north/ring_neb_north_ch2-short_s3d.fits', 
    image_data_2a, error_data_2a, 'NIRSPEC_NORTH_bigbox_new.reg')

data2b, weighted_mean_error_2b =\
rnf.extract_weighted_mean_from_region(
    'data/north/ring_neb_north_ch2-medium_s3d.fits', 
    image_data_2b, error_data_2b, 'NIRSPEC_NORTH_bigbox_new.reg')

data2c, weighted_mean_error_2c =\
rnf.extract_weighted_mean_from_region(
    'data/north/ring_neb_north_ch2-long_s3d.fits', 
    image_data_2c, error_data_2c, 'NIRSPEC_NORTH_bigbox_new.reg')

data3a, weighted_mean_error_3a =\
rnf.extract_weighted_mean_from_region(
    'data/north/ring_neb_north_ch3-short_s3d.fits', 
    image_data_3a, error_data_3a, 'NIRSPEC_NORTH_bigbox_new.reg')

data3b, weighted_mean_error_3b =\
rnf.extract_weighted_mean_from_region(
    'data/north/ring_neb_north_ch3-medium_s3d.fits', 
    image_data_3b, error_data_3b, 'NIRSPEC_NORTH_bigbox_new.reg')

data3c, weighted_mean_error_3c =\
rnf.extract_weighted_mean_from_region(
    'data/north/ring_neb_north_ch3-long_s3d.fits', 
    image_data_3c, error_data_3c, 'NIRSPEC_NORTH_bigbox_new.reg')

#accounts for the region weighted mean function making things a dictionary
data1a = np.array(data1a['region_0'])
data1b = np.array(data1b['region_0'])
data1c = np.array(data1c['region_0'])
data2a = np.array(data2a['region_0'])
data2b = np.array(data2b['region_0'])
data2c = np.array(data2c['region_0'])
data3a = np.array(data3a['region_0'])
data3b = np.array(data3b['region_0'])
data3c = np.array(data3c['region_0'])

#saving data
weighted_mean_error_1a = np.array(weighted_mean_error_1a['region_0'])
weighted_mean_error_1b = np.array(weighted_mean_error_1b['region_0'])
weighted_mean_error_1c = np.array(weighted_mean_error_1c['region_0'])
weighted_mean_error_2a = np.array(weighted_mean_error_2a['region_0'])
weighted_mean_error_2b = np.array(weighted_mean_error_2b['region_0'])
weighted_mean_error_2c = np.array(weighted_mean_error_2c['region_0'])
weighted_mean_error_3a = np.array(weighted_mean_error_3a['region_0'])
weighted_mean_error_3b = np.array(weighted_mean_error_3b['region_0'])
weighted_mean_error_3c = np.array(weighted_mean_error_3c['region_0'])

np.save('Analysis/data1a', data1a)
np.save('Analysis/data1b', data1b)
np.save('Analysis/data1c', data1c)
np.save('Analysis/data2a', data2a)
np.save('Analysis/data2b', data2b)
np.save('Analysis/data2c', data2c)
np.save('Analysis/data3a', data3a)
np.save('Analysis/data3b', data3b)
np.save('Analysis/data3c', data3c)

np.save('Analysis/weighted_mean_error_1a', weighted_mean_error_1a)
np.save('Analysis/weighted_mean_error_1b', weighted_mean_error_1b)
np.save('Analysis/weighted_mean_error_1c', weighted_mean_error_1c)
np.save('Analysis/weighted_mean_error_2a', weighted_mean_error_2a)
np.save('Analysis/weighted_mean_error_2b', weighted_mean_error_2b)
np.save('Analysis/weighted_mean_error_2c', weighted_mean_error_2c)
np.save('Analysis/weighted_mean_error_3a', weighted_mean_error_3a)
np.save('Analysis/weighted_mean_error_3b', weighted_mean_error_3b)
np.save('Analysis/weighted_mean_error_3c', weighted_mean_error_3c)



#calculating weighted mean of MIRI data, west
data1a_west, weighted_mean_error_1a_west =\
rnf.extract_weighted_mean_from_region(
    'data/west/ring_neb_west_ch1-short_s3d.fits', 
    image_data_1a_west, error_data_1a_west,
    'ring_west_common_region.reg')

data1b_west, weighted_mean_error_1b_west =\
rnf.extract_weighted_mean_from_region(
    'data/west/ring_neb_west_ch1-medium_s3d.fits', 
    image_data_1b_west, error_data_1b_west,
    'ring_west_common_region.reg')

data1c_west, weighted_mean_error_1c_west =\
rnf.extract_weighted_mean_from_region(
    'data/west/ring_neb_west_ch1-long_s3d.fits', 
    image_data_1c_west, error_data_1c_west,
    'ring_west_common_region.reg')

data2a_west, weighted_mean_error_2a_west =\
rnf.extract_weighted_mean_from_region(
    'data/west/ring_neb_west_ch2-short_s3d.fits', 
    image_data_2a_west, error_data_2a_west,
    'ring_west_common_region.reg')

data2b_west, weighted_mean_error_2b_west =\
rnf.extract_weighted_mean_from_region(
    'data/west/ring_neb_west_ch2-medium_s3d.fits', 
    image_data_2b_west, error_data_2b_west,
    'ring_west_common_region.reg')

data2c_west, weighted_mean_error_2c_west =\
rnf.extract_weighted_mean_from_region(
    'data/west/ring_neb_west_ch2-long_s3d.fits', 
    image_data_2c_west, error_data_2c_west,
    'ring_west_common_region.reg')

data3a_west, weighted_mean_error_3a_west =\
rnf.extract_weighted_mean_from_region(
    'data/west/ring_neb_west_ch3-short_s3d.fits', 
    image_data_3a_west, error_data_3a_west,
    'ring_west_common_region.reg')

data3b_west, weighted_mean_error_3b_west =\
rnf.extract_weighted_mean_from_region(
    'data/west/ring_neb_west_ch3-medium_s3d.fits', 
    image_data_3b_west, error_data_3b_west,
    'ring_west_common_region.reg')

data3c_west, weighted_mean_error_3c_west =\
rnf.extract_weighted_mean_from_region(
    'data/west/ring_neb_west_ch3-long_s3d.fits', 
    image_data_3c_west, error_data_3c_west,
    'ring_west_common_region.reg')

#accounts for the region weighted mean function making things a dictionary
data1a_west = np.array(data1a_west['region_0'])
data1b_west = np.array(data1b_west['region_0'])
data1c_west = np.array(data1c_west['region_0'])
data2a_west = np.array(data2a_west['region_0'])
data2b_west = np.array(data2b_west['region_0'])
data2c_west = np.array(data2c_west['region_0'])
data3a_west = np.array(data3a_west['region_0'])
data3b_west = np.array(data3b_west['region_0'])
data3c_west = np.array(data3c_west['region_0'])

#saving data
weighted_mean_error_1a_west = np.array(weighted_mean_error_1a_west['region_0'])
weighted_mean_error_1b_west = np.array(weighted_mean_error_1b_west['region_0'])
weighted_mean_error_1c_west = np.array(weighted_mean_error_1c_west['region_0'])
weighted_mean_error_2a_west = np.array(weighted_mean_error_2a_west['region_0'])
weighted_mean_error_2b_west = np.array(weighted_mean_error_2b_west['region_0'])
weighted_mean_error_2c_west = np.array(weighted_mean_error_2c_west['region_0'])
weighted_mean_error_3a_west = np.array(weighted_mean_error_3a_west['region_0'])
weighted_mean_error_3b_west = np.array(weighted_mean_error_3b_west['region_0'])
weighted_mean_error_3c_west = np.array(weighted_mean_error_3c_west['region_0'])

np.save('Analysis/data1a_west', data1a_west)
np.save('Analysis/data1b_west', data1b_west)
np.save('Analysis/data1c_west', data1c_west)
np.save('Analysis/data2a_west', data2a_west)
np.save('Analysis/data2b_west', data2b_west)
np.save('Analysis/data2c_west', data2c_west)
np.save('Analysis/data3a_west', data3a_west)
np.save('Analysis/data3b_west', data3b_west)
np.save('Analysis/data3c_west', data3c_west)

np.save('Analysis/weighted_mean_error_1a_west', weighted_mean_error_1a_west)
np.save('Analysis/weighted_mean_error_1b_west', weighted_mean_error_1b_west)
np.save('Analysis/weighted_mean_error_1c_west', weighted_mean_error_1c_west)
np.save('Analysis/weighted_mean_error_2a_west', weighted_mean_error_2a_west)
np.save('Analysis/weighted_mean_error_2b_west', weighted_mean_error_2b_west)
np.save('Analysis/weighted_mean_error_2c_west', weighted_mean_error_2c_west)
np.save('Analysis/weighted_mean_error_3a_west', weighted_mean_error_3a_west)
np.save('Analysis/weighted_mean_error_3b_west', weighted_mean_error_3b_west)
np.save('Analysis/weighted_mean_error_3c_west', weighted_mean_error_3c_west)



#calculating weighted mean, and regular mean, of NIRSPEC data
nirspec_weighted_mean, nirspec_error_mean = rnf.extract_weighted_mean_from_region(
    'data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 
    nirspec_data, nirspec_error_data, 'NIRSPEC_NORTH_bigbox_new.reg')

nirspec_weighted_mean_west, nirspec_error_mean_west = rnf.extract_weighted_mean_from_region(
    'data/west/jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked.fits', 
    nirspec_data_west, nirspec_error_data_west, 'ring_west_common_region.reg')



nirspec_regular_mean, nirspec_error_regular_mean = rnf.extract_weighted_mean_from_region(
    'data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 
    nirspec_data, nirspec_error_data, 'NIRSPEC_NORTH_bigbox_new.reg')

nirspec_regular_mean_west, nirspec_error_regular_mean_west = rnf.extract_weighted_mean_from_region(
    'data/west/jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked.fits', 
    nirspec_data_west, nirspec_error_data_west, 'ring_west_common_region.reg')



nirspec_regular_mean_short, nirspec_error_regular_mean_short = rnf.extract_weighted_mean_from_region(
    'data/north/jw01558-o056_t005_nirspec_g235m-f170lp_s3d_masked_aligned.fits', 
    nirspec_data_short, nirspec_error_data_short, 'NIRSPEC_NORTH_bigbox_new.reg')

nirspec_regular_mean_short_west, nirspec_error_regular_mean_short_west = rnf.extract_weighted_mean_from_region(
    'data/west/jw01558-o008_t007_nirspec_g235m-f170lp_s3d_masked.fits', 
    nirspec_data_short_west, nirspec_error_data_short_west, 'ring_west_common_region.reg')

#accounts for the region weighted mean function making things a dictionary
nirspec_weighted_mean = np.array(nirspec_weighted_mean['region_0'])
nirspec_error_mean = np.array(nirspec_error_mean['region_0'])

nirspec_weighted_mean_west = np.array(nirspec_weighted_mean_west['region_0'])
nirspec_error_mean_west = np.array(nirspec_error_mean_west['region_0'])



nirspec_regular_mean = np.array(nirspec_regular_mean['region_0'])
nirspec_error_regular_mean = np.array(nirspec_error_regular_mean['region_0'])

nirspec_regular_mean_west = np.array(nirspec_regular_mean_west['region_0'])
nirspec_error_regular_mean_west = np.array(nirspec_error_regular_mean_west['region_0'])



nirspec_regular_mean_short = np.array(nirspec_regular_mean_short['region_0'])
nirspec_error_regular_mean_short = np.array(nirspec_error_regular_mean_short['region_0'])

nirspec_regular_mean_short_west = np.array(nirspec_regular_mean_short_west['region_0'])
nirspec_error_regular_mean_short_west = np.array(nirspec_error_regular_mean_short_west['region_0'])

#saving data
np.save('Analysis/nirspec_weighted_mean', nirspec_weighted_mean)
np.save('Analysis/nirspec_error_mean', nirspec_error_mean)

np.save('Analysis/nirspec_weighted_mean_west', nirspec_weighted_mean_west)
np.save('Analysis/nirspec_error_mean_west', nirspec_error_mean_west)



np.save('Analysis/nirspec_regular_mean', nirspec_regular_mean)
np.save('Analysis/nirspec_error_regular_mean', nirspec_error_regular_mean)

np.save('Analysis/nirspec_regular_mean_west', nirspec_regular_mean_west)
np.save('Analysis/nirspec_error_regular_mean_west', nirspec_error_regular_mean_west)



np.save('Analysis/nirspec_regular_mean_short', nirspec_regular_mean_short)
np.save('Analysis/nirspec_regular_error_mean_short', nirspec_error_regular_mean_short)

np.save('Analysis/nirspec_regular_mean_short_west', nirspec_regular_mean_short_west)
np.save('Analysis/nirspec_regular_error_mean_short_west', nirspec_error_regular_mean_short_west)



#calculating REGULAR mean of cams, north
data_f1000w, temp = rnf.extract_regular_mean_slice_from_region(
    cam_file_loc_f1000w, cam_data_f1000w, cam_error_data_f1000w, 'NIRSPEC_NORTH_bigbox_new.reg')
data_f1000w = data_f1000w['region_0']

data_f1130w, temp = rnf.extract_regular_mean_slice_from_region(
    cam_file_loc_f1130w, cam_data_f1130w, cam_error_data_f1130w, 'NIRSPEC_NORTH_bigbox_new.reg')
data_f1130w = data_f1130w['region_0']

data_f300m, temp = rnf.extract_regular_mean_slice_from_region(
    cam_file_loc_f300m, cam_data_f300m, cam_error_data_f300m, 'NIRSPEC_NORTH_bigbox_new.reg')
data_f300m = data_f300m['region_0']

data_f335m, temp = rnf.extract_regular_mean_slice_from_region(
    cam_file_loc_f335m, cam_data_f335m, cam_error_data_f335m, 'NIRSPEC_NORTH_bigbox_new.reg')
data_f335m = data_f335m['region_0']

#saving data
np.save('Analysis/data_f1000w', data_f1000w)
np.save('Analysis/data_f1130w', data_f1130w)
np.save('Analysis/data_f300m', data_f300m)
np.save('Analysis/data_f335m', data_f335m)



#calculating REGULAR mean of cams, west
data_f1000w_west, temp = rnf.extract_regular_mean_slice_from_region(
    cam_file_loc_f1000w, cam_data_f1000w, cam_error_data_f1000w, 'ring_west_common_region.reg')
data_f1000w_west = data_f1000w_west['region_0']

data_f1130w_west, temp = rnf.extract_regular_mean_slice_from_region(
    cam_file_loc_f1130w, cam_data_f1130w, cam_error_data_f1130w, 'ring_west_common_region.reg')
data_f1130w_west = data_f1130w_west['region_0']

data_f300m_west, temp = rnf.extract_regular_mean_slice_from_region(
    cam_file_loc_f300m, cam_data_f300m, cam_error_data_f300m, 'ring_west_common_region.reg')
data_f300m_west = data_f300m_west['region_0']

data_f335m_west, temp = rnf.extract_regular_mean_slice_from_region(
    cam_file_loc_f335m, cam_data_f335m, cam_error_data_f335m, 'ring_west_common_region.reg')
data_f335m_west = data_f335m_west['region_0']

#saving data
np.save('Analysis/data_f1000w_west', data_f1000w_west)
np.save('Analysis/data_f1130w_west', data_f1130w_west)
np.save('Analysis/data_f300m_west', data_f300m_west)
np.save('Analysis/data_f335m_west', data_f335m_west)



#calculating REGULAR mean of cam ratios
nircam_ratio, temp = rnf.extract_regular_mean_slice_from_region(
    nircam_file_loc, nircam_data, nircam_error_data, 'NIRSPEC_NORTH_bigbox_new.reg')
nircam_ratio = nircam_ratio['region_0']

nircam_ratio_west, temp = rnf.extract_regular_mean_slice_from_region(
    nircam_file_loc, nircam_data, nircam_error_data, 'ring_west_common_region.reg')
nircam_ratio_west = nircam_ratio_west['region_0']

miricam_ratio, temp = rnf.extract_regular_mean_slice_from_region(
    miricam_file_loc, miricam_data, miricam_error_data, 'NIRSPEC_NORTH_bigbox_new.reg')
miricam_ratio = miricam_ratio['region_0']

miricam_ratio_west, temp = rnf.extract_regular_mean_slice_from_region(
    miricam_file_loc, miricam_data, miricam_error_data, 'ring_west_common_region.reg')
miricam_ratio_west = miricam_ratio_west['region_0']

#saving data
np.save('Analysis/nircam_ratio', nircam_ratio)
np.save('Analysis/nircam_ratio_west', miricam_ratio_west)
np.save('Analysis/miricam_ratio', nircam_ratio)
np.save('Analysis/miricam_ratio_west', miricam_ratio_west)



#calculating weighted mean of spitzer data


#spitzer has no error data so do a regular mean for it
spitzer_data = np.mean(spitzer_image_data[:,32:37, 4:], axis=(1,2))

#saving data
np.save('Analysis/spitzer_data', spitzer_data)

#calculating weighted mean of horsehead data
hh_data_2, hh_weighted_mean_error_2 = rnf.weighted_mean_finder(hh_image_data_2, hh_error_2)
hh_data_3, hh_weighted_mean_error_3 = rnf.weighted_mean_finder(hh_image_data_3, hh_error_3)

#saving data
np.save('Analysis/hh_data_2', hh_data_2)
np.save('Analysis/hh_data_3', hh_data_3)

np.save('Analysis/hh_weighted_mean_error_2', hh_weighted_mean_error_2)
np.save('Analysis/hh_weighted_mean_error_3', hh_weighted_mean_error_3)



####################################



'''
DATA STITCHING
'''



#11.2

#north
wavelengths112, data112, overlap = rnf.flux_aligner_manual(
    wavelengths2c, wavelengths3a, data2c, data3a + 3)

#west
wavelengths112_west, data112_west, overlap = rnf.flux_aligner_manual(
    wavelengths2c_west, wavelengths3a_west, data2c_west, data3a_west - 1)

#saving data
np.save('Analysis/wavelengths112', wavelengths112)
np.save('Analysis/wavelengths112_west', wavelengths112_west)
np.save('Analysis/data112', data112)
np.save('Analysis/data112_west', data112_west)

#nirspec

#north
wavelengths_nirspec_all_temp, nirspec_data_all_temp, overlap = rnf.flux_aligner_offset(
    wavelengths_nirspec_short, wavelengths_nirspec, nirspec_regular_mean_short, nirspec_regular_mean)

nirspec_data_all = np.zeros((len(nirspec_data_all_temp), nirspec_data.shape[1], nirspec_data.shape[2]))

for x in range(len(nirspec_data[0,0,:])):
    for y in range(len(nirspec_data[0,:,0])):
        wavelengths_nirspec_all, nirspec_data_all[:,y,x], overlap = rnf.flux_aligner_offset(
            wavelengths_nirspec_short, wavelengths_nirspec, nirspec_data_short[:,y,x], nirspec_data[:,y,x])

#west
wavelengths_nirspec_all_west_temp, nirspec_data_all_west_temp, overlap = rnf.flux_aligner_offset(
    wavelengths_nirspec_short_west, wavelengths_nirspec_west, nirspec_regular_mean_short_west, nirspec_regular_mean_west)

nirspec_data_all_west = np.zeros((len(nirspec_data_all_west_temp), nirspec_data_west.shape[1], nirspec_data_west.shape[2]))

for x in range(len(nirspec_data_west[0,0,:])):
    for y in range(len(nirspec_data_west[0,:,0])):
        wavelengths_nirspec_all_west, nirspec_data_all_west[:,y,x], overlap = rnf.flux_aligner_offset(
            wavelengths_nirspec_short_west, wavelengths_nirspec_west, nirspec_data_short_west[:,y,x], nirspec_data_west[:,y,x])
        
        

#horsehead
hh_wavelengths, hh_data, overlap = rnf.flux_aligner_manual(
    hh_wavelengths2, hh_wavelengths3, hh_data_2, hh_data_3-15)

#saving data
np.save('Analysis/hh_wavelengths', hh_wavelengths)
np.save('Analysis/hh_data', hh_data)



#all miri channels for plotting

overlap_array = []

#north
wavelengths_pah, pah, overlap = rnf.flux_aligner_manual(
wavelengths1a, wavelengths1b, data1a, data1b + 2)

if len(overlap) > 1:
    overlap_array.append(overlap[0])
else:
    overlap_array.append(overlap)

wavelengths_pah, pah, overlap = rnf.flux_aligner_manual(
wavelengths_pah, wavelengths1c, pah, data1c + 5)

if len(overlap) > 1:
    overlap_array.append(overlap[0])
else:
    overlap_array.append(overlap)

wavelengths_pah, pah, overlap = rnf.flux_aligner_manual(
wavelengths_pah, wavelengths2a, pah, data2a + 7)

if len(overlap) > 1:
    overlap_array.append(overlap[0])
else:
    overlap_array.append(overlap)

wavelengths_pah, pah, overlap = rnf.flux_aligner_manual(
wavelengths_pah, wavelengths2b, pah, data2b + 4)

if len(overlap) > 1:
    overlap_array.append(overlap[0])
else:
    overlap_array.append(overlap)

wavelengths_pah, pah, overlap = rnf.flux_aligner_manual(
wavelengths_pah, wavelengths2c, pah, data2c + 3)

if len(overlap) > 1:
    overlap_array.append(overlap[0])
else:
    overlap_array.append(overlap)

wavelengths_pah, pah, overlap = rnf.flux_aligner_manual(
wavelengths_pah, wavelengths3a, pah, data3a + 3)

if len(overlap) > 1:
    overlap_array.append(overlap[0])
else:
    overlap_array.append(overlap)

#west
wavelengths_pah_west, pah_west, overlap = rnf.flux_aligner_manual(
wavelengths1a_west, wavelengths1b_west, data1a_west, data1b_west + 1)

wavelengths_pah_west, pah_west, overlap = rnf.flux_aligner_manual(
wavelengths_pah_west, wavelengths1c_west, pah_west, data1c_west + 0)

wavelengths_pah_west, pah_west, overlap = rnf.flux_aligner_manual(
wavelengths_pah_west, wavelengths2a_west, pah_west, data2a_west + 4)

wavelengths_pah_west, pah_west, overlap = rnf.flux_aligner_manual(
wavelengths_pah_west, wavelengths2b_west, pah_west, data2b_west + 3)

wavelengths_pah_west, pah_west, overlap = rnf.flux_aligner_manual(
wavelengths_pah_west, wavelengths2c_west, pah_west, data2c_west + 0)

wavelengths_pah_west, pah_west, overlap = rnf.flux_aligner_manual(
wavelengths_pah_west, wavelengths3a_west, pah_west, data3a_west - 1)

#saving data
np.save('Analysis/wavelengths_pah', wavelengths_pah)
np.save('Analysis/wavelengths_west_pah', wavelengths_pah_west)
np.save('Analysis/pah', pah)
np.save('Analysis/pah_west', pah_west)

np.save('Analysis/overlap_array', overlap_array)



####################################



'''
FITTING CONTINUUM TO DATA
'''



#horsehead
points_hh = [10.99, 11.00, 11.82, 11.83] #first and last points are filler
continuum_hh = rnf.linear_continuum_single_channel(hh_wavelengths, hh_data, points_hh)



#orion nirspec
points_orion_nirspec = [2.99, 3.0, 3.7, 3.71] #first and last points are filler
continuum_orion_nirspec = rnf.linear_continuum_single_channel(orion_wavelengths_nirspec, orion_data_nirspec, points_orion_nirspec)



#orion miri
points_orion_miri = [10.81, 10.82, 11.82, 11.83] #first and last points are filler
continuum_orion_miri = rnf.linear_continuum_single_channel(orion_wavelengths_miri, orion_data_miri, points_orion_miri)



#2016 spitzer
points_spitzer = [10.99, 11.00, 11.82, 11.83]#first and last points are filler
continuum_spitzer = rnf.linear_continuum_single_channel(spitzer_wavelengths, spitzer_data, points_spitzer)



#11.2 feature

#north
points_112 = [10.99, 11.00, 11.82, 11.83]#first and last points are filler
continuum112 = rnf.linear_continuum_single_channel(wavelengths112, data112, points_112)

#west
points_112_west = [11.12, 11.13, 11.85, 11.86]#first and last points are filler
continuum112_west = rnf.linear_continuum_single_channel(wavelengths112_west, data112_west, points_112_west)



#saving data
np.save('Analysis/continuum_hh', continuum_hh)
np.save('Analysis/continuum_orion_nirspec', continuum_orion_nirspec)
np.save('Analysis/continuum_orion_miri', continuum_orion_miri)
np.save('Analysis/continuum_spitzer', continuum_spitzer)
np.save('Analysis/continuum112', continuum112)
np.save('Analysis/continuum112_west', continuum112_west)




####################################



'''
FITTING GAUSSIAN TO 3.3
'''



#gaussian function to fit

#note: this is NOT a normalized gaussian

def gaussian(x, mean, fwhm, a):
    
    std = fwhm/(2*(2*np.log(2))**0.5)
    return a*np.exp(-1*((x - mean)**2)/(2*std**2))



####################################



'''
REMOVING EMISSION AND ABSORPTION LINES
'''

#removing emission lines from miri data, to be integrated

#north
integrand112 = rnf.emission_line_remover(data112 - continuum112, 15, 3)

#west
integrand112_west = rnf.emission_line_remover(data112_west - continuum112_west, 10, 1)



####################################



'''
CALCULATING INTENSITIES AND ERRORS
'''



#3.3 feature

#north

#integration bounds
l_int = np.where(np.round(wavelengths_nirspec, 3) == 3.2)[0][0]
u_int = np.where(np.round(wavelengths_nirspec, 2) == 3.35)[-1][-1]

#sum of gaussians to integrate
integrand033 = gaussian(wavelengths_nirspec[l_int:u_int], 3.29027, 0.0387, 2.15) +\
    gaussian(wavelengths_nirspec[l_int:u_int], 3.2465, 0.0375, 0.6) +\
    gaussian(wavelengths_nirspec[l_int:u_int], 3.32821, 0.0264, 0.35)

integral033 = rnf.pah_feature_integrator(wavelengths_nirspec[l_int:u_int], integrand033)

#error
error033 = rnf.error_finder(wavelengths_nirspec, nirspec_weighted_mean, 3.30, (l_int, u_int), 162)

print('3.3 feature:', integral033, '+/-', error033, 'W/m^2/sr, rms range 3.11 to 3.16 microns')

#west

#integration bounds
l_int = np.where(np.round(wavelengths_nirspec_west, 2) == 3.2)[0][0]
u_int = np.where(np.round(wavelengths_nirspec_west, 2) == 3.35)[-1][-1]

integrand033_west = gaussian(wavelengths_nirspec_west[l_int:u_int], 3.29027, 0.0387, 1.1) +\
    gaussian(wavelengths_nirspec_west[l_int:u_int], 3.2465, 0.0375, 0.1) +\
    gaussian(wavelengths_nirspec_west[l_int:u_int], 3.32821, 0.0264, 0.05)

integral033_west = rnf.pah_feature_integrator(wavelengths_nirspec_west[l_int:u_int], integrand033_west)

#error
error033_west = rnf.error_finder(wavelengths_nirspec_west, nirspec_weighted_mean_west, 3.30, (l_int, u_int), 162)

print('3.3 feature, west:', integral033_west, '+/-', error033_west, 'W/m^2/sr, rms range 3.11 to 3.16 microns')



#11.2 feature

#north

#integration bounds
l_int = np.where(np.round(wavelengths112, 3) == 11.0)[0][0]
u_int = np.where(np.round(wavelengths112, 2) == 11.6)[-1][-1]

integral112 = rnf.pah_feature_integrator(wavelengths112[l_int:u_int], integrand112[l_int:u_int])

#error
error112 = rnf.error_finder(wavelengths112, data112, 11.2, (l_int, u_int), 643)

print('11.2 feature:', integral112, '+/-', error112,  'W/m^2/sr, rms range10.83 to 10.88 microns')

#west

#integration bounds
l_int = np.where(np.round(wavelengths112_west, 3) == 11.13)[0][0]
u_int = np.where(np.round(wavelengths112_west, 2) == 11.6)[-1][-1]

integral112_west = rnf.pah_feature_integrator(wavelengths112_west[l_int:u_int], integrand112_west[l_int:u_int])

#error
error112_west = rnf.error_finder(wavelengths112, data112_west, 11.2, (l_int, u_int), 643)

print('11.2 feature, west:', integral112_west, '+/-', error112_west,  'W/m^2/sr, rms range10.83 to 10.88 microns')



#saving data
np.save('Analysis/integral033', integral033)
np.save('Analysis/integral112', integral112)

np.save('Analysis/integral112_west', integral112_west)
np.save('Analysis/integral033_west', integral033_west)

np.save('Analysis/error033', error033)
np.save('Analysis/error112', error112)

np.save('Analysis/error112_west', error112_west)
np.save('Analysis/error033_west', error033_west)



####################################

#%%


'''
SYNTHETIC IFU CALCULATIONS FOR MIRI
'''



#according to https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters#gsc.tab=0
# table 3, F335M has a wavelength range of 3.177 to 3.537. 
# F300M has a wavelength range of 2.831 to 3.157.

#regions 

region = rnf.extract_pixels_from_region('data/north/ring_neb_north_ch1-short_s3d.fits', 
    image_data_1a, 'NIRSPEC_NORTH_bigbox_new.reg')

region_west = rnf.extract_pixels_from_region('data/west/ring_neb_west_ch1-short_s3d.fits', 
    image_data_1a_west, 'ring_west_common_region.reg')



#F300M

#throughput
throughput_F300M = np.loadtxt('data/cams/nircam_throughput/mean_throughputs/F300M_mean_system_throughput.txt', skiprows=1)

throughput_values_F300M = throughput_F300M[:,1]
wavelengths_throughput_F300M = throughput_F300M[:,0]

#data should match the range of the throughput
ifu_lower_index_F300M = np.where(np.round(wavelengths_nirspec_all, 3) == np.round(wavelengths_throughput_F300M[0], 2))[0][0]
ifu_upper_index_F300M = np.where(np.round(wavelengths_nirspec_all, 3) == np.round(wavelengths_throughput_F300M[-1], 2) - 0.001)[0][0]
#subtracting 0.001 because of a stupid issue where the rounding goes to 0.001 on either side of the rounded value, this is a jank fix



wavelengths_ifu_F300M = wavelengths_nirspec_all[ifu_lower_index_F300M:ifu_upper_index_F300M]

#doing a janky reprojection of sorts, so that throughput has the same wavelength interval as the data
new_throughput_F300M = np.copy(wavelengths_ifu_F300M)

for i in range(len(new_throughput_F300M)):
    #find the nearest wavelength, assign this index to that wavelength
    j = np.argmin(abs(wavelengths_throughput_F300M - wavelengths_ifu_F300M[i]))
    new_throughput_F300M[i] = throughput_values_F300M[j]



#north

#formula is flux density [MJy/sr] = [int(F lambda throughput / hc) dlambda]/[int(lambda throughput / hc) dlambda]
#note that, h and c are constants and so cancel out

numerator_ifu_F300M = np.zeros(nirspec_data.shape[1:])

data_ifu_F300M = nirspec_data_all[ifu_lower_index_F300M:ifu_upper_index_F300M]

number_300 = 0
synth_ifu_F300M = 0

for x in range(len(nirspec_data[0,0,:])):
    for y in range(len(nirspec_data[0,:,0])):
        if region[y,x] == 1:
            number_300 += 1
            numerator_ifu_F300M[y,x] = rnf.pah_feature_integrator_no_units(wavelengths_ifu_F300M, 
                data_ifu_F300M[:,y,x]*wavelengths_ifu_F300M*new_throughput_F300M)
            synth_ifu_F300M += numerator_ifu_F300M[y,x]

denominator_ifu_F300M = rnf.pah_feature_integrator_no_units(wavelengths_ifu_F300M, wavelengths_ifu_F300M*new_throughput_F300M)

synth_ifu_F300M = synth_ifu_F300M/(number_300*denominator_ifu_F300M)

#west

#formula is flux density [MJy/sr] = [int(F lambda throughput / hc) dlambda]/[int(lambda throughput / hc) dlambda]
#note that, h and c are constants and so cancel out

numerator_ifu_F300M_west = np.zeros(nirspec_data_west.shape[1:])

data_ifu_F300M_west = nirspec_data_all_west[ifu_lower_index_F300M:ifu_upper_index_F300M]

number_300_west = 0
synth_ifu_F300M_west = 0

for x in range(len(nirspec_data_west[0,0,:])):
    for y in range(len(nirspec_data_west[0,:,0])):
        if region_west[y,x] == 1:
            number_300_west += 1
            numerator_ifu_F300M_west[y,x] = rnf.pah_feature_integrator_no_units(wavelengths_ifu_F300M, 
                data_ifu_F300M_west[:,y,x]*wavelengths_ifu_F300M*new_throughput_F300M)
            synth_ifu_F300M_west += numerator_ifu_F300M_west[y,x]

denominator_ifu_F300M_west = rnf.pah_feature_integrator_no_units(wavelengths_ifu_F300M, wavelengths_ifu_F300M*new_throughput_F300M)

synth_ifu_F300M_west = synth_ifu_F300M_west/(number_300_west*denominator_ifu_F300M_west)



#F335M

#throughput
throughput_F335M = np.loadtxt('data/cams/nircam_throughput/mean_throughputs/F335M_mean_system_throughput.txt', skiprows=1)

throughput_values_F335M = throughput_F335M[:,1]
wavelengths_throughput_F335M = throughput_F335M[:,0]

#data should match the range of the throughput
ifu_lower_index_F335M = np.where(np.round(wavelengths_nirspec, 3) == np.round(wavelengths_throughput_F335M[0], 2))[0][0]
ifu_upper_index_F335M = np.where(np.round(wavelengths_nirspec, 3) == np.round(wavelengths_throughput_F335M[-1], 2))[0][0]



wavelengths_ifu_F335M = wavelengths_nirspec[ifu_lower_index_F335M:ifu_upper_index_F335M]

#doing a janky reprojection of sorts, so that throughput has the same wavelength interval as the data
new_throughput_F335M = np.copy(wavelengths_ifu_F335M)

for i in range(len(new_throughput_F335M)):
    #find the nearest wavelength, assign this index to that wavelength
    j = np.argmin(abs(wavelengths_throughput_F335M - wavelengths_ifu_F335M[i]))
    new_throughput_F335M[i] = throughput_values_F335M[j]



#north

#formula is flux density [MJy/sr] = [int(F lambda throughput / hc) dlambda]/[int(lambda throughput / hc) dlambda]
#note that h and c are constants and so cancel out

numerator_ifu_F335M = np.zeros(nirspec_data.shape[1:])

data_ifu_F335M = nirspec_data[ifu_lower_index_F335M:ifu_upper_index_F335M]

number_335 = 0
synth_ifu_F335M = 0

for x in range(len(nirspec_data[0,0,:])):
    for y in range(len(nirspec_data[0,:,0])):
        if region[y,x] == 1:
            number_335 += 1
            numerator_ifu_F335M[y,x] = rnf.pah_feature_integrator_no_units(wavelengths_ifu_F335M, 
                data_ifu_F335M[:,y,x]*wavelengths_ifu_F335M*new_throughput_F335M)
            synth_ifu_F335M += numerator_ifu_F335M[y,x]

denominator_ifu_F335M = rnf.pah_feature_integrator_no_units(wavelengths_ifu_F335M, wavelengths_ifu_F335M*new_throughput_F335M)

synth_ifu_F335M = synth_ifu_F335M/(number_335*denominator_ifu_F335M)



#west

#formula is flux density [MJy/sr] = [int(F lambda throughput / hc) dlambda]/[int(lambda throughput / hc) dlambda]
#note that h and c are constants and so cancel out

numerator_ifu_F335M_west = np.zeros(nirspec_data_west.shape[1:])

data_ifu_F335M_west = nirspec_data_west[ifu_lower_index_F335M:ifu_upper_index_F335M]

number_335_west = 0
synth_ifu_F335M_west = 0

for x in range(len(nirspec_data_west[0,0,:])):
    for y in range(len(nirspec_data_west[0,:,0])):
        if region_west[y,x] == 1:
            number_335_west += 1
            numerator_ifu_F335M_west[y,x] = rnf.pah_feature_integrator_no_units(wavelengths_ifu_F335M, 
                data_ifu_F335M_west[:,y,x]*wavelengths_ifu_F335M*new_throughput_F335M)
            synth_ifu_F335M_west += numerator_ifu_F335M_west[y,x]

denominator_ifu_F335M_west = rnf.pah_feature_integrator_no_units(wavelengths_ifu_F335M, wavelengths_ifu_F335M*new_throughput_F335M)

synth_ifu_F335M_west = synth_ifu_F335M_west/(number_335_west*denominator_ifu_F335M_west)



#saving data
np.save('synth_ifu_F300M', synth_ifu_F300M)
np.save('synth_ifu_F335M', synth_ifu_F335M)

np.save('synth_ifu_F300M_west', synth_ifu_F300M_west)
np.save('synth_ifu_F335M_west', synth_ifu_F335M_west)



####################################



'''
IFU COMPARISON
'''

synth_ratio = np.nansum(numerator_ifu_F335M*denominator_ifu_F300M/(denominator_ifu_F335M*numerator_ifu_F300M))/number_335

print('F300M north, synthetic vs real: ',  synth_ifu_F300M, data_f300m)
print('F300M west, synthetic vs real: ',  synth_ifu_F300M_west, data_f300m_west)

print('F335M north, synthetic vs real: ',  synth_ifu_F335M, data_f335m)
print('F335M west, synthetic vs real: ',  synth_ifu_F335M_west, data_f335m_west)

print('F335M/F300M north, synthetic vs real: ',  synth_ifu_F335M/synth_ifu_F300M, nircam_ratio)
print('F335M/F300M west, synthetic vs real: ',  synth_ifu_F335M_west/synth_ifu_F300M_west, nircam_ratio_west)









