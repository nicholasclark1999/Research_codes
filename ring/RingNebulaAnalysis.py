
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
# weighted_mean_finder
# weighted_mean_finder_simple
# flux_aligner_offset
# emission_line_remover
# absorption_line_remover
# line_fitter
# fringe_remover
# extract_weighted_mean_from_region
# regrid
# border_remover



####################################



'''
LOADING DATA
'''

#calling MIRI_function
#naming is according to miri convention of short = a, medium = b, long = c
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



#loading in orion PAH spectra, to serve as a comparison and PAH template
pah_image_file = np.loadtxt('data/misc/barh_stick_csub.fits.dat', skiprows=1)
pah_wavelengths = pah_image_file[:,0]
pah_data = pah_image_file[:,1]



####################################



'''
WEIGHTED MEAN
'''



#calculating weighted mean of MIRI data, north
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

#accounts for the region weighted mean function making things a dictionary
data1 = np.array(data1['region_0'])
data2 = np.array(data2['region_0'])
data3 = np.array(data3['region_0'])
data4 = np.array(data4['region_0'])
data5 = np.array(data5['region_0'])
data6 = np.array(data6['region_0'])
data7 = np.array(data7['region_0'])
data8 = np.array(data8['region_0'])
data9 = np.array(data9['region_0'])

#saving data
weighted_mean_error1 = np.array(weighted_mean_error1['region_0'])
weighted_mean_error2 = np.array(weighted_mean_error2['region_0'])
weighted_mean_error3 = np.array(weighted_mean_error3['region_0'])
weighted_mean_error4 = np.array(weighted_mean_error4['region_0'])
weighted_mean_error5 = np.array(weighted_mean_error5['region_0'])
weighted_mean_error6 = np.array(weighted_mean_error6['region_0'])
weighted_mean_error7 = np.array(weighted_mean_error7['region_0'])
weighted_mean_error8 = np.array(weighted_mean_error8['region_0'])
weighted_mean_error9 = np.array(weighted_mean_error9['region_0'])

np.save('Analysis/data1', data1)
np.save('Analysis/data2', data2)
np.save('Analysis/data3', data3)
np.save('Analysis/data4', data4)
np.save('Analysis/data5', data5)
np.save('Analysis/data6', data6)
np.save('Analysis/data7', data7)
np.save('Analysis/data8', data8)
np.save('Analysis/data9', data9)

np.save('Analysis/weighted_mean_error1', weighted_mean_error1)
np.save('Analysis/weighted_mean_error2', weighted_mean_error2)
np.save('Analysis/weighted_mean_error3', weighted_mean_error3)
np.save('Analysis/weighted_mean_error4', weighted_mean_error4)
np.save('Analysis/weighted_mean_error5', weighted_mean_error5)
np.save('Analysis/weighted_mean_error6', weighted_mean_error6)
np.save('Analysis/weighted_mean_error7', weighted_mean_error7)
np.save('Analysis/weighted_mean_error8', weighted_mean_error8)
np.save('Analysis/weighted_mean_error9', weighted_mean_error9)



#calculating weighted mean of MIRI data, west
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

#accounts for the region weighted mean function making things a dictionary
data1_west = np.array(data1_west['region_0'])
data2_west = np.array(data2_west['region_0'])
data3_west = np.array(data3_west['region_0'])
data4_west = np.array(data4_west['region_0'])
data5_west = np.array(data5_west['region_0'])
data6_west = np.array(data6_west['region_0'])
data7_west = np.array(data7_west['region_0'])
data8_west = np.array(data8_west['region_0'])
data9_west = np.array(data9_west['region_0'])

#saving data
weighted_mean_error1_west = np.array(weighted_mean_error1_west['region_0'])
weighted_mean_error2_west = np.array(weighted_mean_error2_west['region_0'])
weighted_mean_error3_west = np.array(weighted_mean_error3_west['region_0'])
weighted_mean_error4_west = np.array(weighted_mean_error4_west['region_0'])
weighted_mean_error5_west = np.array(weighted_mean_error5_west['region_0'])
weighted_mean_error6_west = np.array(weighted_mean_error6_west['region_0'])
weighted_mean_error7_west = np.array(weighted_mean_error7_west['region_0'])
weighted_mean_error8_west = np.array(weighted_mean_error8_west['region_0'])
weighted_mean_error9_west = np.array(weighted_mean_error9_west['region_0'])

np.save('Analysis/data1_west', data1_west)
np.save('Analysis/data2_west', data2_west)
np.save('Analysis/data3_west', data3_west)
np.save('Analysis/data4_west', data4_west)
np.save('Analysis/data5_west', data5_west)
np.save('Analysis/data6_west', data6_west)
np.save('Analysis/data7_west', data7_west)
np.save('Analysis/data8_west', data8_west)
np.save('Analysis/data9_west', data9_west)

np.save('Analysis/weighted_mean_error1_west', weighted_mean_error1_west)
np.save('Analysis/weighted_mean_error2_west', weighted_mean_error2_west)
np.save('Analysis/weighted_mean_error3_west', weighted_mean_error3_west)
np.save('Analysis/weighted_mean_error4_west', weighted_mean_error4_west)
np.save('Analysis/weighted_mean_error5_west', weighted_mean_error5_west)
np.save('Analysis/weighted_mean_error6_west', weighted_mean_error6_west)
np.save('Analysis/weighted_mean_error7_west', weighted_mean_error7_west)
np.save('Analysis/weighted_mean_error8_west', weighted_mean_error8_west)
np.save('Analysis/weighted_mean_error9_west', weighted_mean_error9_west)



#calculating weighted mean of MIRI data, west h2 region
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

#accounts for the region weighted mean function making things a dictionary
data1_west_blob = np.array(data1_west_blob['region_0'])
data2_west_blob = np.array(data2_west_blob['region_0'])
data3_west_blob = np.array(data3_west_blob['region_0'])
data4_west_blob = np.array(data4_west_blob['region_0'])
data5_west_blob = np.array(data5_west_blob['region_0'])
data6_west_blob = np.array(data6_west_blob['region_0'])
data7_west_blob = np.array(data7_west_blob['region_0'])
data8_west_blob = np.array(data8_west_blob['region_0'])
data9_west_blob = np.array(data9_west_blob['region_0'])

weighted_mean_error1_west_blob = np.array(weighted_mean_error1_west_blob['region_0'])
weighted_mean_error2_west_blob = np.array(weighted_mean_error2_west_blob['region_0'])
weighted_mean_error3_west_blob = np.array(weighted_mean_error3_west_blob['region_0'])
weighted_mean_error4_west_blob = np.array(weighted_mean_error4_west_blob['region_0'])
weighted_mean_error5_west_blob = np.array(weighted_mean_error5_west_blob['region_0'])
weighted_mean_error6_west_blob = np.array(weighted_mean_error6_west_blob['region_0'])
weighted_mean_error7_west_blob = np.array(weighted_mean_error7_west_blob['region_0'])
weighted_mean_error8_west_blob = np.array(weighted_mean_error8_west_blob['region_0'])
weighted_mean_error9_west_blob = np.array(weighted_mean_error9_west_blob['region_0'])

#saving data
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



#calculating weighted mean of NIRSPEC data
nirspec_weighted_mean4, nirspec_error_mean4 =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits',
        nirspec_data4, nirspec_error_data4, wavelengths_nirspec4, 'NIRSPEC_NORTH_bigbox_new.reg')

nirspec_weighted_mean4_west, nirspec_error_mean4_west =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg2(
        'data/west/jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked.fits', 
        nirspec_data4_west, nirspec_error_data4_west, wavelengths_nirspec4_west, 
        'ring_west_common_region.reg', 'NIRSPEC_WEST_bigblob.reg')

nirspec_weighted_mean4_west_blob, nirspec_error_mean4_west_blob =\
    rnf.extract_spectra_from_regions_one_pointing_no_bkg(
        'data/west/jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked.fits', 
        nirspec_data4_west, nirspec_error_data4_west, wavelengths_nirspec4_west, 'NIRSPEC_WEST_bigblob.reg')

#accounts for the region weighted mean function making things a dictionary
nirspec_weighted_mean4 = np.array(nirspec_weighted_mean4['region_0'])
nirspec_error_mean4 = np.array(nirspec_error_mean4['region_0'])

nirspec_weighted_mean4_west = np.array(nirspec_weighted_mean4_west['region_0'])
nirspec_error_mean4_west = np.array(nirspec_error_mean4_west['region_0'])

nirspec_weighted_mean4_west_blob = np.array(nirspec_weighted_mean4_west_blob['region_0'])
nirspec_error_mean4_west_blob = np.array(nirspec_error_mean4_west_blob['region_0'])

#saving data

np.save('Analysis/nirspec_weighted_mean4', nirspec_weighted_mean4)
np.save('Analysis/nirspec_error_mean4', nirspec_error_mean4)

np.save('Analysis/nirspec_weighted_mean4_west', nirspec_weighted_mean4_west)
np.save('Analysis/nirspec_error_mean4_west', nirspec_error_mean4_west)

np.save('Analysis/nirspec_weighted_mean4_west_blob', nirspec_weighted_mean4_west_blob)
np.save('Analysis/nirspec_error_mean4_west_blob', nirspec_error_mean4_west_blob)
   


#calculating weighted mean of spitzer data


#spitzer has no error data so do a regular mean for it
spitzer_data = np.mean(spitzer_image_data[:,32:37, 4:], axis=(1,2))

#saving data
np.save('Analysis/spitzer_data', spitzer_data)



####################################



'''
REMOVING EMISSION AND ABSORPTION LINES
'''

#due to a lack of 
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




