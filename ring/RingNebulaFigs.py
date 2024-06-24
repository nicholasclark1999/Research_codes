
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
cam_file_loc_f1000w = 'data/cams/ring_nebula_F1000W_i2d.fits'
cam_file_f1000w = get_pkg_data_filename(cam_file_loc_f1000w)
cam_data_f1000w = fits.getdata(cam_file_f1000w, ext=1)
cam_error_data_f1000w = fits.getdata(cam_file_f1000w, ext=2)

cam_file_loc_f1130w = 'data/cams/ring_nebula_F1130W_i2d.fits'
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



nirspec_regular_mean = np.load('Analysis/nirspec_regular_mean.npy', allow_pickle=True)
nirspec_regular_mean_west = np.load('Analysis/nirspec_regular_mean_west.npy', allow_pickle=True)

nirspec_error_regular_mean = np.load('Analysis/nirspec_error_regular_mean.npy', allow_pickle=True)
nirspec_error_regular_mean_west = np.load('Analysis/nirspec_error_regular_mean_west.npy', allow_pickle=True)



nirspec_regular_mean_short = np.load('Analysis/nirspec_regular_mean_short.npy', allow_pickle=True)
nirspec_regular_mean_short_west = np.load('Analysis/nirspec_regular_mean_short_west.npy', allow_pickle=True)

nirspec_error_regular_mean_short = np.load('Analysis/nirspec_regular_error_mean_short.npy', allow_pickle=True)
nirspec_error_regular_mean_short_west = np.load('Analysis/nirspec_regular_error_mean_short_west.npy', allow_pickle=True)










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
hh_weighted_mean_error_2 = np.load('Analysis/hh_weighted_mean_error_2.npy', allow_pickle=True)
hh_weighted_mean_error_3 = np.load('Analysis/hh_weighted_mean_error_3.npy', allow_pickle=True)



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
overlap_array = np.load('Analysis/overlap_array.npy', allow_pickle=True)



continuum_hh = np.load('Analysis/continuum_hh.npy', allow_pickle=True)
continuum_orion_nirspec = np.load('Analysis/continuum_orion_nirspec.npy', allow_pickle=True)
continuum_orion_miri = np.load('Analysis/continuum_orion_miri.npy', allow_pickle=True)
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
INDICES FOR PLOTTING
'''

pahoverlap_miri2_1 = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelengths1b[0], 2))[0][0]
pahoverlap_miri2_2 = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelengths1b[-1], 2))[0][0]

pahoverlap_miri3_1 = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelengths1c[0], 2))[0][0]
pahoverlap_miri3_2 = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelengths1c[-1], 2))[0][0]



#index where end of spitzer meets data, for creating aligned plots

spitzer_index_begin = np.where(np.round(spitzer_wavelengths, 2) == np.round(wavelengths2c[0], 2))[0][0]
spitzer_index_end = np.where(np.round(spitzer_wavelengths, 2) == np.round(wavelengths2c[-1], 2))[0][0]
spitzer_index_end2 = np.where(np.round(spitzer_wavelengths, 2) == np.round(wavelengths3a[-1], 2))[0][0]

#index where horsehead meets data, for creating aligned plots

hh_index_begin = np.where(np.round(hh_wavelengths, 2) == np.round(wavelengths2c[0], 2))[0][0]
hh_index_end = np.where(np.round(hh_wavelengths, 2) == np.round(wavelengths2c[-1], 2))[0][0]
hh_index_end2 = np.where(np.round(hh_wavelengths, 2) == np.round(wavelengths3a[-1], 2))[0][0]



pahoverlap_low = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelengths_pah[0], 2))[0][0]
pahoverlap_high = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelengths_pah[-1], 2))[0][0]

nirspec_cutoff = np.where(np.round(wavelengths_nirspec, 2) == 4)[0][0]

pahoverlap_nirspec_1 = np.where(np.round(orion_wavelengths_nirspec, 2) == np.round(wavelengths_nirspec[20], 2))[0][0]
pahoverlap_nirspec_2 = np.where(np.round(orion_wavelengths_nirspec, 2) == np.round(wavelengths_nirspec[nirspec_cutoff], 2))[0][0]



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



'''
FIGURE 5
'''



#calculate scaling

orion_north_scaling = np.median(rnf.emission_line_remover(data112 - continuum112, 15, 3)[970:1000])/\
    np.max((orion_data_miri - continuum_orion_miri)[13400:13500])

orion_west_scaling = np.median(rnf.emission_line_remover(data112_west - continuum112_west, 10, 1)[1000:1040])/\
    np.max((orion_data_miri - continuum_orion_miri)[13400:13500])


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

plt.plot(wavelengths112, rnf.emission_line_remover(data112 - continuum112, 15, 3), color='#dc267f', label='Lines and Continuum removed')

plt.plot(wavelengths_pah, 0*pah, color='black', label='zero')

plt.plot(orion_wavelengths_miri[pahoverlap_low:pahoverlap_high], 
         orion_north_scaling*(orion_data_miri - continuum_orion_miri)[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='#000000', alpha=1.0)

for data in overlap_array:
    plt.plot([wavelengths_pah[data], wavelengths_pah[data]], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

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

plt.plot(wavelengths112_west, rnf.emission_line_remover(data112_west - continuum112_west, 10, 1), color='#dc267f', label='Lines and Continuum removed')

plt.plot(wavelengths_pah_west, 0*pah_west, color='black', label='zero')


plt.plot(orion_wavelengths_miri[pahoverlap_low:pahoverlap_high], 
         orion_west_scaling*(orion_data_miri - continuum_orion_miri)[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='#000000', alpha=1.0)

for data in overlap_array:
    plt.plot([wavelengths_pah[data], wavelengths_pah[data]], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

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

plt.savefig('Figures/RNF_paper_continuum_extended_simple_no_legend.pdf', bbox_inches='tight')
plt.show()

#%%

'''
FIGURE 3
'''



ax = plt.figure('RNF_paper_data_extended_simple_no_legend', figsize=(18,9)).add_subplot(111)

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

plt.plot(wavelengths_pah, pah, label='data', color='#648fff')

plt.plot(wavelengths112, continuum112, color='#000000', label='continuum')

for data in overlap_array:
    plt.plot([wavelengths_pah[data], wavelengths_pah[data]], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-0.0,25.0))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(5, 13.5)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.90, 0.10, 'North', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

ax = plt.figure('RNF_paper_data_extended_simple_no_legend', figsize=(18,9)).add_subplot(212)

'''
West
'''



plt.plot(wavelengths_pah_west, pah_west, label='data', color='#648fff')

plt.plot(wavelengths112_west, continuum112_west, color='#000000', label='continuum')



for data in overlap_array:
    plt.plot([wavelengths_pah[data], wavelengths_pah[data]], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-5.0,15.0))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(5, 13.5)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.9, 0.1, 'West', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/RNF_paper_data_extended_simple_no_legend.pdf', bbox_inches='tight')
plt.show()

#%%

'''
FIGURE 4
'''

ax = plt.figure('RNF_paper_033_gaussian_fit', figsize=(18,18)).add_subplot(111)

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

plt.plot(wavelengths_nirspec[:nirspec_cutoff], nirspec_weighted_mean[:nirspec_cutoff] - 2.4, 
         label='g395m-f290, North, offset=-2.4', color='#dc267f')

#3.3 fitting
plt.plot(wavelengths_nirspec[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec[:nirspec_cutoff], 3.29027, 0.0387, 2.15), 
         label ='gaussian fit mean=3.29027, fwhm=0.0387, scale=2.15', color='#648fff')
plt.plot(wavelengths_nirspec[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec[:nirspec_cutoff], 3.2465, 0.0375, 0.6), 
         label ='gaussian fit mean=3.2465, fwhm=0.0375, scale=0.6', color='#648fff')
plt.plot(wavelengths_nirspec[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec[:nirspec_cutoff], 3.32821, 0.0264, 0.35), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.35', color='#648fff')


#3.4 fitting
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4031, 0.0216, 1.15), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.05', color='#648fff')
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4242, 0.0139, 0.50), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.05', color='#648fff')

#sum
plt.plot(wavelengths_nirspec[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec[:nirspec_cutoff], 3.29027, 0.0387, 2.15) +\
         gaussian(wavelengths_nirspec[:nirspec_cutoff], 3.2465, 0.0375, 0.6) +\
         gaussian(wavelengths_nirspec[:nirspec_cutoff], 3.32821, 0.0264, 0.35) +\
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4031, 0.0216, 1.15) +\
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4242, 0.0139, 0.50), 
         label='gaussian fit sum', color='#648fff', alpha=0.7)

plt.plot(orion_wavelengths_nirspec[pahoverlap_nirspec_1:pahoverlap_nirspec_2], 
         orion_north_scaling*(orion_data_nirspec - continuum_orion_nirspec)[pahoverlap_nirspec_1:pahoverlap_nirspec_2], 
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
plt.xticks(np.arange(3.1, 3.6, 0.05), fontsize=28)
plt.yticks(np.arange(0.0, 4.5, 0.5), fontsize=28)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'North', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)


'''
West
'''

ax = plt.figure('RNF_paper_033_gaussian_fit', figsize=(18,18)).add_subplot(212)

plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], nirspec_weighted_mean_west[:nirspec_cutoff] - 1.2, 
         label='g395m-f290, West, offset=-1.2', color='#dc267f')

#3.3 fitting
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.29027, 0.0387, 1.1), 
         label ='gaussian fit mean=3.29027, fwhm=0.0387, scale=1.1', color='#648fff')
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.2465, 0.0375, 0.1), 
         label ='gaussian fit mean=3.2465, fwhm=0.0375 scale=0.1', color='#648fff')
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.32821, 0.0264, 0.05), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.05', color='#648fff')

#3.4 fitting
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4031, 0.0216, 0.75), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.05', color='#648fff')
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4242, 0.0139, 0.35), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.05', color='#648fff')

#sum
plt.plot(wavelengths_nirspec_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.29027, 0.0387, 1.1) +\
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.2465, 0.0375, 0.1) +\
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.32821, 0.0264, 0.05) +\
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4031, 0.0216, 0.75) +\
         gaussian(wavelengths_nirspec_west[:nirspec_cutoff], 3.4242, 0.0139, 0.35), 
         label='gaussian fit sum', color='#648fff')

plt.plot(orion_wavelengths_nirspec[pahoverlap_nirspec_1:pahoverlap_nirspec_2], 
         orion_west_scaling*(orion_data_nirspec - continuum_orion_nirspec)[pahoverlap_nirspec_1:pahoverlap_nirspec_2], 
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
plt.xticks(np.arange(3.1, 3.6, 0.05), fontsize=28)
plt.yticks(np.arange(0.0, 2.5, 0.5), fontsize=28)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'West', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/RNF_paper_033_gaussian_fit.pdf', bbox_inches='tight')
plt.show() 

#%%

'''
FIGURE 6
'''



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

plt.plot(wavelengths112, rnf.emission_line_remover(data112 - continuum112, 15, 3), label='Lines and Continuum removed', color='#dc267f')
plt.plot(hh_wavelengths[hh_index_begin:hh_index_end2], 
         0.42*(hh_data - continuum_hh)[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.42', color='#648fff', alpha=0.5)

plt.plot(spitzer_wavelengths, 
         2.35*(spitzer_data - continuum_spitzer[:,0]), 
         label='Spitzer spectra, scale=0.42', color='black')

plt.ylim((-2.5,15))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.5, 12., 0.25), fontsize=28)
plt.xlim(10.5,12)
plt.yticks(fontsize=28)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.10, 0.95, 'North', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)


'''
West
'''

ax = plt.figure('RNF_paper_112_comparison', figsize=(18,18)).add_subplot(212)

plt.plot(wavelengths112_west, rnf.emission_line_remover(data112_west - continuum112_west, 10, 1), label='Lines and Continuum removed', color='#dc267f')
plt.plot(hh_wavelengths[hh_index_begin:hh_index_end2], 
         0.16*(hh_data - continuum_hh)[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.18', color='#648fff', alpha=0.5)

plt.plot(spitzer_wavelengths, 
         0.85*(spitzer_data - continuum_spitzer[:,0]), 
         label='Spitzer spectra, scale=0.42', color='black')

plt.ylim((-2.5,7))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.5, 12., 0.25), fontsize=28)
plt.xlim(10.5,12)
plt.yticks(fontsize=28)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.10, 0.95, 'West', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/RNF_paper_112_comparison.pdf', bbox_inches='tight')
plt.show()

#%%

'''
FIGURE 7
'''

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

plt.plot(iso_sl2_wavelengths, iso_sl2_data, color='#dc267f')
plt.plot(iso_sl1_wavelengths, iso_sl1_data, color='#648fff')

plt.ylim((-1.0, 40.0))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
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

plt.plot(iso_sl2_wavelengths, 0.3*iso_sl2_data, color='#dc267f')
plt.plot(iso_sl1_wavelengths, iso_sl1_data, color='#648fff')

plt.ylim((-1.0, 30.0))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.5, 13.5, 1.0), fontsize=28)
plt.yticks(fontsize=28)
plt.xlim(5.5, 13.5)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'Scaled', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/RNF_paper_ISO_062.pdf', bbox_inches='tight')
plt.show()



#######################################



'''
ISO INTENSITY JANK INTEGRALS
'''

#%%

point1 = 24
point2 = 39

iso_data_062 = iso_sl2_data[point1:point2+1]
iso_wavelengths_062 = iso_sl2_wavelengths[point1:point2+1]

iso_slope = ((iso_sl2_data[point2] - iso_sl2_data[point1])/(iso_sl2_wavelengths[point2] - iso_sl2_wavelengths[point1]))

iso_continuum = iso_slope*(iso_wavelengths_062 - iso_sl2_wavelengths[point1]) + iso_sl2_data[point1]

iso_062_integrand = iso_data_062 - iso_continuum

iso_integrand = np.copy(iso_062_integrand)

#new gaussian that takes std isntead of fwhm

def gaussian_std(x, mean, std, a):
    
    return a*np.exp(-1*((x - mean)**2)/(2*std**2))

for i in range(6):
    iso_integrand[i] = iso_062_integrand[i] - (gaussian_std(iso_wavelengths_062, 6.107, 0.025, 19))[i]

iso_line = np.copy(iso_integrand)

iso_integrand = rnf.unit_changer(iso_wavelengths_062, iso_integrand)

#fitting 6.2ish line

#for spitzer iso sl2: 
    
R = 6.107*16.5333

line_width = 1/16.5333 #can just do this since relation is linear


#%%
plt.figure()
#plt.plot(iso_sl2_wavelengths, iso_sl2_data)
#plt.plot(iso_wavelengths_062, iso_continuum)
plt.plot(iso_wavelengths_062, iso_062_integrand)
plt.plot(iso_wavelengths_062, 1 + 15*(iso_wavelengths_062 - iso_wavelengths_062[0]) + gaussian_std(iso_wavelengths_062, 6.107, 0.025, 19))
plt.plot(orion_wavelengths_miri, 0.002*(orion_data_miri - continuum_orion_miri))
plt.xlim(5, 7)
plt.ylim(0, 25)
plt.show()
#%%
plt.figure()
#plt.plot(iso_sl2_wavelengths, iso_sl2_data)
#plt.plot(iso_wavelengths_062, iso_continuum)
#plt.plot(iso_wavelengths_062, iso_062_integrand)
plt.plot(iso_wavelengths_062, iso_line)
plt.plot(orion_wavelengths_miri, 0.002*(orion_data_miri - continuum_orion_miri))
plt.xlim(5, 7)
plt.ylim(0, 10)
plt.show()
#%%
plt.figure()
plt.plot(iso_wavelengths_062, iso_integrand)
plt.plot([iso_wavelengths_062[0], 6.203], [0, 3.83e-7])
plt.plot([6.203, iso_wavelengths_062[-1]], [3.83e-7, 0])
plt.plot()
plt.show()

iso_intensity_062 = 0.5*(3.83e-7)*(iso_wavelengths_062[-1] - iso_wavelengths_062[0])

print('estimated intensity using 2 triangles with peak at 6.203, 3.83e-7 is ', np.round(iso_intensity_062, 10))



#%%

point1 = 54
point2 = 66

iso_data_112 = iso_sl1_data[point1:point2+1]
iso_wavelengths_112 = iso_sl1_wavelengths[point1:point2+1]

iso_slope = ((iso_sl1_data[point2] - iso_sl1_data[point1])/(iso_sl1_wavelengths[point2] - iso_sl1_wavelengths[point1]))

iso_continuum = iso_slope*(iso_wavelengths_112 - iso_sl1_wavelengths[point1]) + iso_sl1_data[point1]

iso_112_integrand = iso_data_112 - iso_continuum



iso_integrand = rnf.unit_changer(iso_wavelengths_112, iso_112_integrand)



plt.figure()
plt.plot(iso_sl1_wavelengths, iso_sl1_data)
plt.plot(iso_wavelengths_112, iso_continuum)
plt.xlim(10, 12)
plt.show()
#%%
plt.figure()
plt.plot(iso_wavelengths_112, iso_integrand)
plt.plot([iso_wavelengths_112[0], 11.142], [0, 6.41e-8])
plt.plot([11.142, 11.259], [6.41e-8, 2.64e-7])
plt.plot([11.259, 11.259], [6.41e-8, 2.64e-7])
plt.plot([11.259, 11.320], [2.64e-7, 2.64e-7])
plt.plot([11.142, 11.259], [6.41e-8, 6.41e-8])
plt.plot([11.320, iso_wavelengths_112[-1]], [2.64e-7, 0])
plt.show()

iso_intensity_112_rectangles = (2.64e-7)*(11.320 - 11.259) + (2.64e-7)*(11.320 - 11.259)
iso_intesnity_112_triangles = 0.5*((6.41e-8)*(11.142 - iso_wavelengths_112[0]) + (2.64e-7 - 6.41e-8)*(11.259 - 11.142) + (2.64e-7)*(iso_wavelengths_112[-1] - 11.320))
iso_intensity_112 = iso_intensity_112_rectangles + iso_intesnity_112_triangles

# old version print('estimated flux using triangle up to 11.259, triangle from 11.383, and rectangle in the middle that starts at the height of the first point, is 11.1e-8')

print('estimated flux using several triangles and rectangles, is 11.1e-8', iso_intensity_112)















#%%


'''


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


ax.imshow(nircam_data, vmax=10)
for p in patch_list1 + patch_list2 + patch_list3 + patch_list4 + patch_list5:
    ax.add_patch(p)
    
for t in artist_list1 + artist_list2 + artist_list3 + artist_list4 + artist_list5:
    ax.add_artist(t)

ax.invert_xaxis()
ax.invert_yaxis()

plt.show()



#%%

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

fig2 = plt.figure()
ax2 = WCSAxes(fig2, [0.1, 0.1, 0.8, 0.8], wcs=pog)
fig2.add_axes(ax2)


ax2.imshow(miricam_data, vmax=100)

for p in patch_list1 + patch_list2 + patch_list3 + patch_list4 + patch_list5:
    ax2.add_patch(p)
    
for t in artist_list1 + artist_list2 + artist_list3 + artist_list4 + artist_list5:
    ax2.add_artist(t)

plt.show()

#%%
'''

