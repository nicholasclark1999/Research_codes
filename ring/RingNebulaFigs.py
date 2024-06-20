
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:06:29 2023

@author: nclark
"""
#%%
'''
IMPORTING MODULES
'''

#import matplotlib
#matplotlib.use('Agg')

#standard stuff
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import  AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter

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
# flux_aligner_offset
# flux_aligner_manual
# emission_line_remover
# absorption_line_remover
# line_fitter
# linear_continuum_single_channel
# unit_changer
# pah_feature_integrator
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
hh_wavelengths2, hh_image_data2, hh_error2 = rnf.loading_function(
    'data/misc/jw01192-o010_t002_miri_ch2-shortmediumlong_s3d.fits', 1)
hh_wavelengths3, hh_image_data3, hh_error3 = rnf.loading_function(
    'data/misc/jw01192-o010_t002_miri_ch3-shortmediumlong_s3d.fits', 1)




####################################

#%%

'''
LOADING ANALYSIS DATA
'''

data1a = np.load('Analysis/data1a.npy', allow_pickle=True)
data1b = np.load('Analysis/data1b.npy', allow_pickle=True)
data1c = np.load('Analysis/data1c.npy', allow_pickle=True)
data2a = np.load('Analysis/data2a.npy', allow_pickle=True)
data2b = np.load('Analysis/data2b.npy', allow_pickle=True)
data2c = np.load('Analysis/data2c.npy', allow_pickle=True)
data3a = np.load('Analysis/data3a.npy', allow_pickle=True)
data3b = np.load('Analysis/data3b.npy', allow_pickle=True)
data3c = np.load('Analysis/data3c.npy', allow_pickle=True)



data1a_west = np.load('Analysis/data1a_west.npy', allow_pickle=True)
data1b_west = np.load('Analysis/data1b_west.npy', allow_pickle=True)
data1c_west = np.load('Analysis/data1c_west.npy', allow_pickle=True)
data2a_west = np.load('Analysis/data2a_west.npy', allow_pickle=True)
data2b_west = np.load('Analysis/data2b_west.npy', allow_pickle=True)
data2c_west = np.load('Analysis/data2c_west.npy', allow_pickle=True)
data3a_west = np.load('Analysis/data3a_west.npy', allow_pickle=True)
data3b_west = np.load('Analysis/data3b_west.npy', allow_pickle=True)
data3c_west = np.load('Analysis/data3c_west.npy', allow_pickle=True)



weighted_mean_error_1a = np.load('Analysis/weighted_mean_error_1a.npy', allow_pickle=True)
weighted_mean_error_1b = np.load('Analysis/weighted_mean_error_1b.npy', allow_pickle=True)
weighted_mean_error_1c = np.load('Analysis/weighted_mean_error_1c.npy', allow_pickle=True)
weighted_mean_error_2a = np.load('Analysis/weighted_mean_error_2a.npy', allow_pickle=True)
weighted_mean_error_2b = np.load('Analysis/weighted_mean_error_2b.npy', allow_pickle=True)
weighted_mean_error_2c = np.load('Analysis/weighted_mean_error_2c.npy', allow_pickle=True)
weighted_mean_error_3a = np.load('Analysis/weighted_mean_error_3a.npy', allow_pickle=True)
weighted_mean_error_3b = np.load('Analysis/weighted_mean_error_3b.npy', allow_pickle=True)
weighted_mean_error_3c = np.load('Analysis/weighted_mean_error_3c.npy', allow_pickle=True)

weighted_mean_error_1a_west = np.load('Analysis/weighted_mean_error_1a_west.npy', allow_pickle=True)
weighted_mean_error_1b_west = np.load('Analysis/weighted_mean_error_1b_west.npy', allow_pickle=True)
weighted_mean_error_1c_west = np.load('Analysis/weighted_mean_error_1c_west.npy', allow_pickle=True)
weighted_mean_error_2a_west = np.load('Analysis/weighted_mean_error_2a_west.npy', allow_pickle=True)
weighted_mean_error_2b_west = np.load('Analysis/weighted_mean_error_2b_west.npy', allow_pickle=True)
weighted_mean_error_2c_west = np.load('Analysis/weighted_mean_error_2c_west.npy', allow_pickle=True)
weighted_mean_error_3a_west = np.load('Analysis/weighted_mean_error_3a_west.npy', allow_pickle=True)
weighted_mean_error_3b_west = np.load('Analysis/weighted_mean_error_3b_west.npy', allow_pickle=True)
weighted_mean_error_3c_west = np.load('Analysis/weighted_mean_error_3c_west.npy', allow_pickle=True)



nirspec_weighted_mean = np.load('Analysis/nirspec_weighted_mean.npy', allow_pickle=True)
nirspec_weighted_mean_west = np.load('Analysis/nirspec_weighted_mean_west.npy', allow_pickle=True)

nirspec_error_mean = np.load('Analysis/nirspec_error_mean.npy', allow_pickle=True)
nirspec_error_mean_west = np.load('Analysis/nirspec_error_mean_west.npy', allow_pickle=True)



nirspec_regular_mean = np.load('Analysis/nirspec_regular_mean', allow_pickle=True)
nirspec_regular_mean_west = np.load('Analysis/nirspec_regular_mean_west', allow_pickle=True)

nirspec_error_regular_mean = np.load('Analysis/nirspec_error_regular_mean', allow_pickle=True)
nirspec_error_regular_mean_west = np.load('Analysis/nirspec_error_regular_mean_west', allow_pickle=True)



nirspec_regular_mean_short = np.load('Analysis/nirspec_regular_mean_short', allow_pickle=True)
nirspec_regular_mean_short_west = np.load('Analysis/nirspec_regular_mean_short_west', allow_pickle=True)

nirspec_error_regular_mean_short = np.load('Analysis/nirspec_regular_error_mean_short', allow_pickle=True)
nirspec_error_regular_mean_short_west = np.load('Analysis/nirspec_regular_error_mean_short_west', allow_pickle=True)










data_f1000w = np.load('Analysis/data_f1000w.npy', allow_pickle=True)
data_f1130w = np.load('Analysis/data_f1130w.npy', allow_pickle=True)
data_f300m = np.load('Analysis/data_f300m.npy', allow_pickle=True)
data_f335m = np.load('Analysis/data_f335m.npy', allow_pickle=True)

data_f1000w_west = np.load('Analysis/data_f1000w_west.npy', allow_pickle=True)
data_f1130w_west = np.load('Analysis/data_f1130w_west.npy', allow_pickle=True)
data_f300m_west = np.load('Analysis/data_f300m_west.npy', allow_pickle=True)
data_f335m_west = np.load('Analysis/data_f335m_west.npy', allow_pickle=True)

nircam_ratio = np.load('Analysis/nircam_ratio.npy', allow_pickle=True)
nircam_ratio_west = np.load('Analysis/nircam_ratio_west.npy', allow_pickle=True)
miricam_ratio = np.load('Analysis/miricam_ratio.npy', allow_pickle=True)
miricam_ratio_west = np.load('Analysis/miricam_ratio_west.npy', allow_pickle=True)



spitzer_data = np.load('Analysis/spitzer_data.npy', allow_pickle=True)



hh_data_2 = np.load('Analysis/hh_data_2.npy', allow_pickle=True)
hh_data_3 = np.load('Analysis/hh_data_3.npy', allow_pickle=True)
hh_weighted_mean_error_2 = np.load('Analysis/hh_weighted_mean_error_2', allow_pickle=True)
hh_weighted_mean_error_3 = np.load('Analysis/hh_weighted_mean_error_3', allow_pickle=True)



wavelengths112 = np.load('Analysis/wavelengths112.npy', allow_pickle=True)
wavelengths112_west = np.load('Analysis/wavelengths112_west.npy', allow_pickle=True)
data112 = np.load('Analysis/data112.npy', allow_pickle=True)
data112_west = np.load('Analysis/data112_west.npy', allow_pickle=True)



hh_wavelengths = np.load('Analysis/hh_wavelengths.npy', allow_pickle=True)
hh_data = np.load('Analysis/hh_data.npy', allow_pickle=True)



wavelengths_pah = np.load('Analysis/wavelengths_pah.npy', allow_pickle=True)
wavelengths_pah_west = np.load('Analysis/wavelengths_west_pah.npy', allow_pickle=True)
pah = np.load('Analysis/pah.npy', allow_pickle=True)
pah_west = np.load('Analysis/pah_west.npy', allow_pickle=True)



continuum_hh = np.load('Analysis/continuum_hh.npy', allow_pickle=True)
continuum_orion_nirspec = np.load('Analysis/continuum_hh.npy', allow_pickle=True)
continuum_spitzer = np.load('Analysis/continuum_spitzer.npy', allow_pickle=True)
continuum112 = np.load('Analysis/continuum112.npy', allow_pickle=True)
continuum112_west = np.load('Analysis/continuum112_west.npy', allow_pickle=True)



integral033 = np.load('Analysis/integral033.npy', allow_pickle=True)
integral112 = np.load('Analysis/integral112.npy', allow_pickle=True)

integral112_west = np.load('Analysis/integral112_west.npy', allow_pickle=True)
integral033_west = np.load('Analysis/integral033_west.npy', allow_pickle=True)

error033 = np.load('Analysis/error033.npy', allow_pickle=True)
error112 = np.load('Analysis/error112.npy', allow_pickle=True)

error112_west = np.load('Analysis/error112_west.npy', allow_pickle=True)
error033_west = np.load('Analysis/error033_west.npy', allow_pickle=True)



synth_ifu_F300M = np.load('synth_ifu_F300M.npy', allow_pickle=True)
synth_ifu_F335M = np.load('synth_ifu_F335M.npy', allow_pickle=True)

synth_ifu_F300M_west = np.load('synth_ifu_F300M_west.npy', allow_pickle=True)
synth_ifu_F335M_west = np.load('synth_ifu_F335M_west.npy', allow_pickle=True)

















####################################

'''
COMPARISON CONTINUA
'''






'''
PLOT INDICES
'''

pahoverlap_miri2_1 = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelengths2[0], 2))[0][0]
pahoverlap_miri2_2 = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelengths2[-1], 2))[0][0]

pahoverlap_miri3_1 = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelengths3[0], 2))[0][0]
pahoverlap_miri3_2 = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelengths3[-1], 2))[0][0]



#index where end of spitzer meets data, for creating aligned plots

spitzer_index_begin = np.where(np.round(spitzer_wavelengths, 2) == np.round(wavelengths6[0], 2))[0][0]
spitzer_index_end = np.where(np.round(spitzer_wavelengths, 2) == np.round(wavelengths6[-1], 2))[0][0]
spitzer_index_end2 = np.where(np.round(spitzer_wavelengths, 2) == np.round(wavelengths7[-1], 2))[0][0]

#index where horsehead meets data, for creating aligned plots

hh_index_begin = np.where(np.round(wavelength_pah_hh, 2) == np.round(wavelengths6[0], 2))[0][0]
hh_index_end = np.where(np.round(wavelength_pah_hh, 2) == np.round(wavelengths6[-1], 2))[0][0]
hh_index_end2 = np.where(np.round(wavelength_pah_hh, 2) == np.round(wavelengths7[-1], 2))[0][0]

ngc7027_index_begin = np.where(np.round(ngc7027_wavelengths, 2) == np.round(wavelengths6[0], 2))[0][0]
ngc7027_index_end = np.where(np.round(ngc7027_wavelengths, 2) == np.round(wavelengths6[-1], 2))[0][0]
ngc7027_index_end2 = np.where(np.round(ngc7027_wavelengths, 2) == np.round(wavelengths7[-1], 2))[0][0]



pahoverlap_low = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelength_pah[0], 2))[0][0]
pahoverlap_high = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelength_pah[-1], 2))[0][0]

nirspec_cutoff = np.where(np.round(wavelengths_nirspec, 2) == 4)[0][0]

pahoverlap_nirspec_1 = np.where(np.round(orion_wavelengths_nirspec, 2) == np.round(wavelengths_nirspec[20], 2))[0][0]
pahoverlap_nirspec_2 = np.where(np.round(orion_wavelengths_nirspec, 2) == np.round(wavelengths_nirspec[nirspec_cutoff], 2))[0][0]

#remaking these variables with new bounds (the others only go down to 10 microns)

ngc7027_index_begin = np.where(np.round(ngc7027_wavelengths, 2) == np.round(wavelengths5[0], 2))[0][0]
ngc7027_index_end = np.where(np.round(ngc7027_wavelengths, 2) == np.round(wavelengths5[-1], 2))[0][0]
ngc7027_index_end2 = np.where(np.round(ngc7027_wavelengths, 2) == np.round(wavelengths7[-1], 2))[0][0]



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
PRINTING INTENSITIES AND ERRORS
'''



print('3.3 feature:', integral033, '+/-', error033, 'W/m^2/sr, rms range 3.11 to 3.16 microns')
print('3.3 feature, west:', integral033_west, '+/-', error033_west, 'W/m^2/sr, rms range 3.11 to 3.16 microns')

print('11.2 feature:', integral112, '+/-', error112,  'W/m^2/sr, rms range10.83 to 10.88 microns')
print('11.2 feature, west:', integral112_west, '+/-', error112_west,  'W/m^2/sr, rms range10.83 to 10.88 microns')



####################################



'''
PRINTING IFU COMPARISON
'''



print('F300M north, synthetic vs real: ',  synth_ifu_F300M, data_f300m)
print('F300M west, synthetic vs real: ',  synth_ifu_F300M_west, data_f300m_west)

print('F335M north, synthetic vs real: ',  synth_ifu_F335M, data_f335m)
print('F335M west, synthetic vs real: ',  synth_ifu_F335M_west, data_f335m_west)

print('F335M/F300M north, synthetic vs real: ',  synth_ifu_F335M/synth_ifu_F300M, nircam_ratio)
print('F335M/F300M west, synthetic vs real: ',  synth_ifu_F335M_west/synth_ifu_F300M_west, nircam_ratio_west)



####################################



'''
PAPER PLOTS
'''



#RNF_paper_continuum_extended_simple_no_legend (FIGURE 5)



#calculate scaling

orion_north_scaling = np.median(everything_removed_3[560:600])/np.max(continuum_removed_orion[13400:13500])

orion_west_scaling = np.median(everything_removed_3_west[560:600])/np.max(continuum_removed_orion[13400:13500])


ax = plt.figure('RNF_paper_continuum_extended_simple_no_legend', figsize=(18,6)).add_subplot(1,1,1)

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
North
'''

ax = plt.figure('RNF_paper_continuum_extended_simple_no_legend', figsize=(18,9)).add_subplot(1,2,1)

plt.plot(wave_cont1, everything_removed_1, color='#dc267f', label='Lines and Continuum removed')
plt.plot(wave_cont2, everything_removed_2, color='#dc267f')
plt.plot(wave_cont3, everything_removed_3, color='#dc267f')

plt.plot(wavelength_pah, 0*pah, color='black', label='zero')

plt.plot(orion_wavelengths_miri[pahoverlap_low:pahoverlap_high], 
         orion_north_scaling*continuum_removed_orion[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='#000000', alpha=1.0)

for data in overlap_array:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-1.0,13.0))
plt.xlim((10.6, 11.8))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.6, 11.8, 0.3), fontsize=14)
plt.yticks(fontsize=14)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'North', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

'''
West
'''

ax = plt.figure('RNF_paper_continuum_extended_simple_no_legend', figsize=(18,9)).add_subplot(1,2,2)

plt.plot(wave_cont1_west, everything_removed_1_west, color='#dc267f', label='Lines and Continuum removed')
plt.plot(wave_cont2_west, everything_removed_2_west, color='#dc267f')
plt.plot(wave_cont3_west, everything_removed_3_west, color='#dc267f')

plt.plot(wavelength_pah_west, 0*pah_west, color='black', label='zero')


plt.plot(orion_wavelengths_miri[pahoverlap_low:pahoverlap_high], 
         orion_west_scaling*continuum_removed_orion[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='#000000', alpha=1.0)

for data in overlap_array_west:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-0.5,6.0))
plt.xlim((10.6, 11.8))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.6, 11.8, 0.3), fontsize=14)
plt.yticks(fontsize=14)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'West', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/paper/RNF_paper_continuum_extended_simple_no_legend.pdf', bbox_inches='tight')
plt.show()




#%%

#RNF_paper_data_extended_simple_no_legend (FIGURE 3)

#%%

ax = plt.figure('RNF_paper_data_extended_simple_no_legend', figsize=(18,9)).add_subplot(111)
#plt.subplots_adjust(right=0.9, left=0.1)

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

ax = plt.figure('RNF_paper_data_extended_simple_no_legend', figsize=(18,9)).add_subplot(211)

'''
North
'''

#plt.title('JWST Continuum Subtracted Data, North, simple', fontsize=20)

plt.plot(wavelength_pah, pah-10, label='data', color='#648fff')

#plt.plot(wave_cont1, everything_removed_1, color='#dc267f', label='Lines and Continuum removed')
#plt.plot(wave_cont2, everything_removed_2, color='#dc267f')
#plt.plot(wave_cont3, everything_removed_3, color='#dc267f')
#plt.plot(wave_nirspec, nirspec_no_line, color='#dc267f')



#plt.plot(wave_cont1, continuum1-10, color='#000000', label='continuum')
#plt.plot(wave_cont2, continuum2-10, color='#785ef0')
plt.plot(wave_cont3, continuum3 - 10, color='#000000', label='continuum')


'''
plt.plot(wavelength_pah, 0*pah, color='black', label='zero')
'''
#plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
#         0.13*continuum_removed_orion[pahoverlap_low:pahoverlap_high], 
#         label='ISO orion spectra, scale=0.13', color='#785ef0', alpha=1.0)

for data in overlap_array:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')
#plt.scatter(overlap_array, -5*np.ones(len(overlap_array)), zorder=100, color='black', label='data overlap')



plt.ylim((-0.0,25.0))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'North', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(5.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(5, 13.5)
#plt.legend(fontsize=11, title='North Common', bbox_to_anchor=(1.02, 1), loc='upper left')

props = dict(boxstyle='round', facecolor='white')
ax.text(0.90, 0.10, 'North', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax = plt.figure('RNF_paper_data_extended_simple_no_legend', figsize=(18,9)).add_subplot(212)

'''
West
'''


#plt.title('JWST Continuum Subtracted Data, West simple', fontsize=20)

plt.plot(wavelength_pah_west, pah_west-20, label='data', color='#648fff')

#plt.plot(wave_cont1_west, everything_removed_1_west, color='#dc267f', label='Lines and Continuum removed')
#plt.plot(wave_cont2_west, everything_removed_2_west, color='#dc267f')
#plt.plot(wave_cont3_west, everything_removed_3_west, color='#dc267f')
#plt.plot(wave_nirspec_west , nirspec_no_line_west, color='#dc267f')



#plt.plot(wave_cont1_west, continuum1_west-20, color='#785ef0', label='continuum')
#plt.plot(wave_cont2_west, continuum2_west-20, color='#785ef0')
plt.plot(wave_cont3_west, continuum3_west-20, color='#000000', label='continuum')


'''
plt.plot(wavelength_pah_west, 0*pah_west, color='black', label='zero')
'''

#plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
#         0.045*continuum_removed_orion[pahoverlap_low:pahoverlap_high], 
#         label='ISO orion spectra, scale=0.045', color='#785ef0', alpha=1.0)

for data in overlap_array_west:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')
#plt.scatter(overlap_array_west, -5*np.ones(len(overlap_array_west)), zorder=100, color='black', label='data overlap')

plt.ylim((-5.0,15.0))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'West', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(5.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(5, 13.5)
#plt.legend(fontsize=11, title='West Common', bbox_to_anchor=(1.02, 1), loc='upper left')

props = dict(boxstyle='round', facecolor='white')
ax.text(0.9, 0.1, 'West', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/paper/RNF_paper_data_extended_simple_no_legend.pdf', bbox_inches='tight')
plt.show()



#######################################



#%%

#RNF_paper_033_gaussian_fit (FIGURE 4)

#%%

ax = plt.figure('RNF_paper_033_gaussian_fit', figsize=(18,18)).add_subplot(111)
#plt.subplots_adjust(right=0.9, left=0.1)
ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.ylabel('Flux (MJy/sr)', fontsize=32, labelpad=90)
plt.xlabel('Wavelength (micron)', fontsize=32, labelpad=60)



'''
North
'''
ax = plt.figure('RNF_paper_033_gaussian_fit', figsize=(18,18)).add_subplot(211)
#plt.title('NIRSPEC Weighted Mean, gaussian fit, North', fontsize=20)
plt.plot(wavelengths_nirspec[:nirspec_cutoff], nirspec_weighted_mean[:nirspec_cutoff] - 2.4, 
         label='g395m-f290, North, offset=-2.4', color='#dc267f')

#3.3 fitting
plt.plot(wavelengths_nirspec[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec[:nirspec_cutoff], 3.29027, 0.0387, 2.15, 0), 
         label ='gaussian fit mean=3.29027, fwhm=0.0387, scale=2.15', color='#648fff')
plt.plot(wavelengths_nirspec[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec[:nirspec_cutoff], 3.2465, 0.0375, 0.6, 0), 
         label ='gaussian fit mean=3.2465, fwhm=0.0375, scale=0.6', color='#648fff')
plt.plot(wavelengths_nirspec[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec[:nirspec_cutoff], 3.32821, 0.0264, 0.35, 0), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.35', color='#648fff')


#3.4 fitting
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4031, 0.0216, 1.15, 0), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.05', color='#648fff')
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4242, 0.0139, 0.50, 0), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.05', color='#648fff')

#sum
plt.plot(wavelengths_nirspec[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec[:nirspec_cutoff], 3.29027, 0.0387, 2.15, 0) +\
         gaussian(wavelengths_nirspec[:nirspec_cutoff], 3.2465, 0.0375, 0.6, 0) +\
         gaussian(wavelengths_nirspec[:nirspec_cutoff], 3.32821, 0.0264, 0.35, 0) +\
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4031, 0.0216, 1.15, 0) +\
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4242, 0.0139, 0.50, 0), 
         label='gaussian fit sum', color='#648fff', alpha=0.7)

plt.plot(orion_wavelengths_nirspec[pahoverlap_nirspec_1:pahoverlap_nirspec_2], 
         orion_north_scaling*continuum_removed_orion_nirspec[pahoverlap_nirspec_1:pahoverlap_nirspec_2], 
         label='ISO orion spectra, scale=0.13', color='#000000', alpha=1.0)
plt.ylim((-0.5,4))
plt.xlim((3.1, 3.6))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'North', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.1, 3.6, 0.05), fontsize=28)
plt.yticks(np.arange(0.0, 4.5, 0.5), fontsize=28)
#plt.legend(fontsize=14)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'North', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)


'''
West
'''

ax = plt.figure('RNF_paper_033_gaussian_fit', figsize=(18,18)).add_subplot(212)
#ax = plt.figure('RNF_paper_033_gaussian_fit_west', figsize=(18,9)).add_subplot(111)
#plt.title('NIRSPEC Weighted Mean, gaussian fit, West', fontsize=20)
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], nirspec_weighted_mean_west[:nirspec_cutoff] - 1.2, 
         label='g395m-f290, West, offset=-1.2', color='#dc267f')

#3.3 fitting
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.29027, 0.0387, 1.1, 0), 
         label ='gaussian fit mean=3.29027, fwhm=0.0387, scale=1.1', color='#648fff')
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.2465, 0.0375, 0.1, 0), 
         label ='gaussian fit mean=3.2465, fwhm=0.0375 scale=0.1', color='#648fff')
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.32821, 0.0264, 0.05, 0), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.05', color='#648fff')

#3.4 fitting
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4031, 0.0216, 0.75, 0), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.05', color='#648fff')
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4242, 0.0139, 0.35, 0), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.05', color='#648fff')

#sum
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.29027, 0.0387, 1.1, 0) +\
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.2465, 0.0375, 0.1, 0) +\
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.32821, 0.0264, 0.05, 0) +\
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4031, 0.0216, 0.75, 0) +\
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4242, 0.0139, 0.35, 0), 
         label='gaussian fit sum', color='#648fff')

plt.plot(orion_wavelengths_nirspec[pahoverlap_nirspec_1:pahoverlap_nirspec_2], 
         orion_west_scaling*continuum_removed_orion_nirspec[pahoverlap_nirspec_1:pahoverlap_nirspec_2], 
         label='ISO orion spectra, scale=0.045', color='#000000', alpha=1.0)
plt.ylim((-0.5,2))
plt.xlim((3.1, 3.6))

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'West', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=32)
plt.xticks(np.arange(3.1, 3.6, 0.05), fontsize=28)
plt.yticks(np.arange(0.0, 2.5, 0.5), fontsize=28)
#plt.legend(fontsize=14)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'West', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/paper/RNF_paper_033_gaussian_fit.pdf', bbox_inches='tight')
plt.show() 


#%%

#RNF_paper_112_comparison (FIGURE 6)

#%%

ax = plt.figure('RNF_paper_112_comparison', figsize=(18,18)).add_subplot(111)
ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.ylabel('Flux (MJy/sr)', fontsize=32, labelpad=90)
plt.xlabel('Wavelength (micron)', fontsize=32, labelpad=60)

'''
North
'''

ax = plt.figure('RNF_paper_112_comparison', figsize=(18,18)).add_subplot(211)
#ax = plt.figure('RNF_paper_112_comparison', figsize=(12,8)).add_subplot(311)
#plt.title('JWST Continuum Subtracted Data, 11.2 feature, North', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
#plt.plot(wavelength_pah_removed_112, pah_112, label='data')
#plt.plot(wavelength_pah_removed_112, continuum_removed6, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112, everything_removed6, label='Lines and Continuum removed', color='#dc267f')
plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.42*continuum_removed_hh[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.42', color='#648fff', alpha=0.5)

plt.plot(spitzer_wavelengths, 
         2.35*continuum_removed_spitzer, 
         label='Spitzer spectra, scale=0.42', color='black')

'''
plt.plot(ngc7027_wavelengths[ngc7027_index_begin:ngc7027_index_end2], 
         0.033*ngc7027_data[ngc7027_index_begin:ngc7027_index_end2] - 8, 
         label='NGC 7027 spectra, scale=0.033, offset=-8', color='#785ef0', alpha=1)
'''

#plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
#         2*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 6, label='Spitzer, scale=2, offset=-6', color='black', alpha=0.8)

#plt.plot(wavelength_pah_removed_112, pah_removed_112, color='black', label='continuum')
'''
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.15*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.15', color='r', alpha=0.5)
'''
'''
plt.plot(11.0*np.ones(len(wavelength_pah_removed_112)), np.linspace(-10, 25, len(pah_removed_112)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(11.6*np.ones(len(wavelength_pah_removed_112)), np.linspace(-10, 25, len(pah_removed_112)), 
         color='black', label='upper integration bound (11.6)')
'''
plt.ylim((-2.5,15))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(10.5, 12., 0.25), fontsize=28)
plt.xlim(10.5,12)
plt.yticks(fontsize=28)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.10, 0.95, 'North', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

#plt.legend(fontsize=11)


'''
West
'''

ax = plt.figure('RNF_paper_112_comparison', figsize=(18,18)).add_subplot(212)
#plt.title('JWST Continuum Subtracted Data, 11.2 feature, North', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
#plt.plot(wavelength_pah_removed_112, pah_112, label='data')
#plt.plot(wavelength_pah_removed_112, continuum_removed6, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112_west, everything_removed6_west, label='Lines and Continuum removed', color='#dc267f')
plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.16*continuum_removed_hh[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.18', color='#648fff', alpha=0.5)

plt.plot(spitzer_wavelengths, 
         0.85*continuum_removed_spitzer, 
         label='Spitzer spectra, scale=0.42', color='black')

'''
plt.plot(ngc7027_wavelengths[ngc7027_index_begin:ngc7027_index_end2], 
         0.033*ngc7027_data[ngc7027_index_begin:ngc7027_index_end2] - 8, 
         label='NGC 7027 spectra, scale=0.033, offset=-8', color='#785ef0', alpha=1)
'''

#plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
#         2*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 6, label='Spitzer, scale=2, offset=-6', color='black', alpha=0.8)

#plt.plot(wavelength_pah_removed_112, pah_removed_112, color='black', label='continuum')
'''
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.15*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.15', color='r', alpha=0.5)
'''
'''
plt.plot(11.0*np.ones(len(wavelength_pah_removed_112)), np.linspace(-10, 25, len(pah_removed_112)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(11.6*np.ones(len(wavelength_pah_removed_112)), np.linspace(-10, 25, len(pah_removed_112)), 
         color='black', label='upper integration bound (11.6)')
'''
plt.ylim((-2.5,7))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(10.5, 12., 0.25), fontsize=28)
plt.xlim(10.5,12)
plt.yticks(fontsize=28)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.10, 0.95, 'West', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/paper/RNF_paper_112_comparison.pdf', bbox_inches='tight')
plt.show()







#%%


#%%

#RNF_paper_ISO_062 (FIGURE 7)


#%%

ax = plt.figure('RNF_paper_ISO_062', figsize=(18,18)).add_subplot(111)
ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
ax.set_xticks([])
ax.set_yticks([])

plt.ylabel('Flux (MJy/sr)', fontsize=32, labelpad=90)
plt.xlabel('Wavelength (micron)', fontsize=32, labelpad=60)

'''
Unscaled
'''

ax = plt.figure('RNF_paper_ISO_062', figsize=(18,18)).add_subplot(211)


#plt.title('JWST Continuum Subtracted Data, West simple', fontsize=20)

plt.plot(iso_sl2_wavelengths, iso_sl2_data, color='#dc267f')
plt.plot(iso_sl1_wavelengths, iso_sl1_data, color='#648fff')



#plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
#         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
#         label='ISO orion spectra, scale=0.045', color='#000000', alpha=1.0)

plt.ylim((-1.0, 40.0))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'West', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(5.5, 13.5, 1.0), fontsize=28)
plt.yticks(fontsize=28)
plt.xlim(5.5, 13.5)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'Unscaled', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

'''
scaled
'''

ax = plt.figure('RNF_paper_ISO_062', figsize=(18,18)).add_subplot(212)


#plt.title('JWST Continuum Subtracted Data, West simple', fontsize=20)

plt.plot(iso_sl2_wavelengths, 0.3*iso_sl2_data, color='#dc267f')
plt.plot(iso_sl1_wavelengths, iso_sl1_data, color='#648fff')



#plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
#         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
#         label='ISO orion spectra, scale=0.045', color='#000000', alpha=1.0)

plt.ylim((-1.0, 30.0))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'West', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(5.5, 13.5, 1.0), fontsize=28)
plt.yticks(fontsize=28)
plt.xlim(5.5, 13.5)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'Scaled', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/paper/RNF_paper_ISO_062.pdf', bbox_inches='tight')
plt.show()

#%%

point1 = 24
point2 = 39

iso_data_062 = iso_sl2_data[point1:point2+1]
iso_wavelengths_062 = iso_sl2_wavelengths[point1:point2+1]

iso_slope = iso_cont = ((iso_sl2_data[point2] - iso_sl2_data[point1])/(iso_sl2_wavelengths[point2] - iso_sl2_wavelengths[point1]))

iso_continuum = iso_slope*(iso_wavelengths_062 - iso_sl2_wavelengths[point1]) + iso_sl2_data[point1]

iso_062_integrand = iso_data_062 - iso_continuum



iso_integrand = rnf.unit_changer(iso_wavelengths_062, iso_062_integrand)

plt.figure()
plt.plot(iso_wavelengths_062, iso_integrand)
#plt.plot(iso_wavelengths_062, iso_data_062)
#plt.plot(iso_wavelengths_062, iso_continuum)
plt.show()

print('estimated flux using 2 triangles with peak at 6.203, 3.83e-7 is 8.9e-8')

#%%

plt.figure()
plt.plot(iso_sl1_wavelengths, iso_sl1_data)
plt.ylim(0,30)
#plt.plot(iso_wavelengths_112, iso_data_112)
#plt.plot(iso_wavelengths_112, iso_continuum)
plt.show()

#%%

point1 = 54
point2 = 66

iso_data_112 = iso_sl1_data[point1:point2+1]
iso_wavelengths_112 = iso_sl1_wavelengths[point1:point2+1]

iso_slope = iso_cont = ((iso_sl1_data[point2] - iso_sl1_data[point1])/(iso_sl1_wavelengths[point2] - iso_sl1_wavelengths[point1]))

iso_continuum = iso_slope*(iso_wavelengths_112 - iso_sl1_wavelengths[point1]) + iso_sl1_data[point1]

iso_112_integrand = iso_data_112 - iso_continuum



iso_integrand = rnf.unit_changer(iso_wavelengths_112, iso_112_integrand)



plt.figure()
plt.plot(iso_wavelengths_112, iso_integrand)
#plt.plot(iso_wavelengths_112, iso_data_112)
#plt.plot(iso_wavelengths_112, iso_continuum)
plt.show()

print('estimated flux using triangle up to 11.259, triangle from 11.383, and rectangle in the middle that starts at the height of the first point, is 11.1e-8')























cam_header_f335m = fits.getheader(cam_file_f335m, ext=1)

pog = wcs.WCS(cam_header_f335m)




import pyregion

from astropy.visualization.wcsaxes import WCSAxes

region_name = "apertures_for_plotting/NIRSPEC_NORTH_bigbox_new.reg"
r1 = pyregion.open(region_name).as_imagecoord(header=cam_header_f335m)
patch_list1, artist_list1 = r1.get_mpl_patches_texts()

region_name = "apertures_for_plotting/ring_west_common_region.reg"
r2 = pyregion.open(region_name).as_imagecoord(header=cam_header_f335m)
patch_list2, artist_list2 = r2.get_mpl_patches_texts()

region_name = "apertures_for_plotting/peak_112_SL1.reg"
r3 = pyregion.open(region_name).as_imagecoord(header=cam_header_f335m)
patch_list3, artist_list3 = r3.get_mpl_patches_texts()

region_name = "apertures_for_plotting/apertures_Cox2016_1.reg"
r4 = pyregion.open(region_name).as_imagecoord(header=cam_header_f335m)
patch_list4, artist_list4 = r4.get_mpl_patches_texts()

region_name = "apertures_for_plotting/apertures_Cox2016_2.reg"
r5 = pyregion.open(region_name).as_imagecoord(header=cam_header_f335m)
patch_list5, artist_list5 = r5.get_mpl_patches_texts()



fig = plt.figure()
ax = WCSAxes(fig, [0.1, 0.1, 0.8, 0.8], wcs=pog)
fig.add_axes(ax)


ax.imshow(data_f335m, vmax=10)
for p in patch_list1 + patch_list2 + patch_list3 + patch_list4 + patch_list5:
    ax.add_patch(p)
    
for t in artist_list1 + artist_list2 + artist_list3 + artist_list4 + artist_list5:
    ax.add_artist(t)

plt.show()


