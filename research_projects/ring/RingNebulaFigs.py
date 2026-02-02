
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



#g235m-f170, note that this is only used for making synthetic cams
wavelengths_nirspec_short, nirspec_data_short, nirspec_error_data_short = rnf.loading_function(
    'data/north/jw01558-o056_t005_nirspec_g235m-f170lp_s3d_masked_aligned.fits', 1)

wavelengths_nirspec_short_west, nirspec_data_short_west, nirspec_error_data_short_west = rnf.loading_function(
    'data/west/jw01558-o008_t007_nirspec_g235m-f170lp_s3d_masked.fits', 1)



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
nircam_data = fits.getdata(nircam_image_file, ext=0)
nircam_error_data = fits.getdata(nircam_image_file, ext=2)

miricam_file_loc = 'data/cams/miri_color_F1000W_F1130W.fits'
miricam_image_file = get_pkg_data_filename(miricam_file_loc)
miricam_data = fits.getdata(miricam_image_file, ext=0)
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

#%%

'''
FIGURE 5
'''



#calculate scaling

orion_north_scaling = np.median(rnf.emission_line_remover(data112 - continuum112, 15, 3)[970:1000])/\
    np.max((orion_data_miri - continuum_orion_miri)[13400:13500])

orion_west_scaling = np.median(rnf.emission_line_remover(data112_west - continuum112_west, 10, 1)[1000:1040])/\
    np.max((orion_data_miri - continuum_orion_miri)[13400:13500])


ax = plt.figure('RNF_paper_continuum_extended_simple_no_legend', figsize=(18,18)).add_subplot(1,1,1)

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

plt.ylabel('Flux (MJy/sr)', fontsize=32, labelpad=90)
plt.xlabel('Wavelength (micron)', fontsize=32, labelpad=60)



'''
North
'''

ax = plt.figure('RNF_paper_continuum_extended_simple_no_legend', figsize=(18,18)).add_subplot(2,1,1)

plt.plot(wavelengths112, rnf.emission_line_remover(data112 - continuum112, 15, 3), color='#dc267f', label='Lines and Continuum removed')

plt.plot(wavelengths_pah, 0*pah, color='black', label='zero')

plt.plot(orion_wavelengths_miri[pahoverlap_low:pahoverlap_high], 
         orion_north_scaling*(orion_data_miri - continuum_orion_miri)[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='#000000', alpha=1.0)

for data in overlap_array:
    plt.plot([wavelengths_pah[data], wavelengths_pah[data]], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-1.0,13.0))
plt.xlim((10.9, 11.8))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.9, 11.8, 0.15), fontsize=28)
plt.yticks(fontsize=28)

props = dict(boxstyle='round', facecolor='none')
ax.text(0.05, 0.95, 'North', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

'''
West
'''

ax = plt.figure('RNF_paper_continuum_extended_simple_no_legend', figsize=(18,18)).add_subplot(2,1,2)

plt.plot(wavelengths112_west, rnf.emission_line_remover(data112_west - continuum112_west, 10, 1), color='#dc267f', label='Lines and Continuum removed')

plt.plot(wavelengths_pah_west, 0*pah_west, color='black', label='zero')


plt.plot(orion_wavelengths_miri[pahoverlap_low:pahoverlap_high], 
         orion_west_scaling*(orion_data_miri - continuum_orion_miri)[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='#000000', alpha=1.0)

for data in overlap_array:
    plt.plot([wavelengths_pah[data], wavelengths_pah[data]], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-0.5,6.0))
plt.xlim((10.9, 11.8))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.9, 11.8, 0.15), fontsize=28)
plt.yticks(fontsize=28)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'West', transform=ax.transAxes, fontsize=28,
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


plt.plot(hh_wavelengths[hh_index_begin:hh_index_end2], 
         0.42*(hh_data - continuum_hh)[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.42', color='#648fff', alpha=0.5)

plt.plot(spitzer_wavelengths, 
         2.35*(spitzer_data - continuum_spitzer[:,0]), 
         label='Spitzer spectra, scale=0.42', color='black', alpha=0.7)

plt.plot(wavelengths112, rnf.emission_line_remover(data112 - continuum112, 15, 3), label='Lines and Continuum removed', color='#dc267f')

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


plt.plot(hh_wavelengths[hh_index_begin:hh_index_end2], 
         0.16*(hh_data - continuum_hh)[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.18', color='#648fff', alpha=0.5)

plt.plot(spitzer_wavelengths, 
         0.85*(spitzer_data - continuum_spitzer[:,0]), 
         label='Spitzer spectra, scale=0.42', color='black', alpha=0.7)

plt.plot(wavelengths112_west, rnf.emission_line_remover(data112_west - continuum112_west, 10, 1), label='Lines and Continuum removed', color='#dc267f')

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




#%%
'''
plt.figure()
#plt.plot(iso_sl2_wavelengths, iso_sl2_data)
#plt.plot(iso_wavelengths_062, iso_continuum)
plt.plot(iso_wavelengths_062, iso_062_integrand)
plt.plot(iso_wavelengths_062, 1 + 15*(iso_wavelengths_062 - iso_wavelengths_062[0]) + gaussian_std(iso_wavelengths_062, 6.107, 0.025, 19))
plt.plot(orion_wavelengths_miri, 0.002*(orion_data_miri - continuum_orion_miri))
plt.xlim(5, 7)
plt.ylim(0, 25)
plt.show()
'''
#%%

orion_iso_continuum = iso_slope*(orion_wavelengths_miri - iso_sl2_wavelengths[point1]) + iso_sl2_data[point1]
'''
plt.figure()
#plt.plot(iso_sl2_wavelengths, iso_sl2_data)
#plt.plot(iso_wavelengths_062, iso_continuum)
#plt.plot(iso_wavelengths_062, iso_062_integrand)
plt.plot(iso_wavelengths_062, iso_line)
plt.plot(orion_wavelengths_miri, 0.002*(orion_data_miri - continuum_orion_miri) + orion_iso_continuum)
plt.xlim(5, 7)
plt.ylim(0, 10)
plt.show()
'''
#%%
'''
plt.figure()
plt.plot(iso_sl2_wavelengths, iso_sl2_data)
plt.plot(iso_wavelengths_062, iso_continuum)
plt.plot(orion_wavelengths_miri, 0.002*(orion_data_miri - continuum_orion_miri) + orion_iso_continuum)
#plt.plot(iso_wavelengths_062, iso_062_integrand)
#plt.plot(iso_wavelengths_062, iso_line)
#plt.plot(orion_wavelengths_miri, 0.002*(orion_data_miri - continuum_orion_miri))
plt.xlim(5, 7)
#plt.ylim(0, 10)
plt.show()
'''
#%%
'''
plt.figure()
plt.plot(orion_wavelengths_miri, orion_data_miri)
plt.plot(orion_wavelengths_miri, continuum_orion_miri)
plt.show()
'''
#%%
'''
plt.figure()
plt.plot(iso_wavelengths_062, iso_integrand)
plt.plot([iso_wavelengths_062[0], 6.203], [0, 3.83e-7])
plt.plot([6.203, iso_wavelengths_062[-1]], [3.83e-7, 0])
plt.plot()
plt.show()
'''
iso_intensity_062 = 0.5*(3.83e-7)*(iso_wavelengths_062[-1] - iso_wavelengths_062[0])

print('estimated intensity using 2 triangles with peak at 6.203, 3.83e-7 is ', np.round(iso_intensity_062, 10))

#%%

orion_scaling_iso = (0.3*(iso_data_062 - iso_continuum)[7])/((orion_data_miri - continuum_orion_miri))[2730]

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
Zoomed out (Scaled + Unscaled)
'''

ax = plt.figure('RNF_paper_ISO_062', figsize=(18,18)).add_subplot(211)

plt.plot(iso_sl2_wavelengths, iso_sl2_data+30, color='#dc267f')
plt.plot(iso_sl1_wavelengths, iso_sl1_data+30, color='#648fff')

plt.plot(iso_sl2_wavelengths, 0.3*iso_sl2_data, color='#dc267f')
plt.plot(iso_sl1_wavelengths, iso_sl1_data, color='#648fff')

plt.ylim((-1.0, 70.0))
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
ax.text(0.02, 0.95, 'Unscaled', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.02, 0.25, 'Scaled', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

'''
Zoomed in (Scaled)
'''

ax = plt.figure('RNF_paper_ISO_062', figsize=(18,18)).add_subplot(212)

plt.plot(iso_sl2_wavelengths, 0.3*iso_sl2_data, color='#dc267f')
#plt.plot(iso_sl1_wavelengths, iso_sl1_data, color='#648fff')

plt.plot(iso_wavelengths_062[:9], (0.7 + 8*(iso_wavelengths_062 - iso_wavelengths_062[0]) + 1*(gaussian_std(iso_wavelengths_062, 6.107, 0.025, 5.9)))[:9], color='black', linestyle='dashed')

plt.plot(orion_wavelengths_miri, orion_scaling_iso*(orion_data_miri - continuum_orion_miri) + 0.3*orion_iso_continuum, color='black', alpha=0.7)

plt.ylim((-0.25, 8.0))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.5, 6.5, 0.1), fontsize=28)
plt.yticks(fontsize=28)
plt.xlim(5.5, 6.5)

props = dict(boxstyle='round', facecolor='white')
ax.text(0.02, 0.95, 'Scaled', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/RNF_paper_ISO_062.pdf', bbox_inches='tight')
plt.show()






#%%

point1 = 54
point2 = 66

iso_data_112 = iso_sl1_data[point1:point2+1]
iso_wavelengths_112 = iso_sl1_wavelengths[point1:point2+1]


iso_continuum = iso_slope*(iso_wavelengths_112 - iso_sl1_wavelengths[point1]) + iso_sl1_data[point1]

iso_112_integrand = iso_data_112 - iso_continuum



iso_integrand = rnf.unit_changer(iso_wavelengths_112, iso_112_integrand)

orion_iso_continuum = iso_slope*(orion_wavelengths_miri - iso_sl1_wavelengths[point1]) + iso_sl1_data[point1]
'''
plt.figure()
plt.plot(iso_sl1_wavelengths, iso_sl1_data)
plt.plot(iso_wavelengths_112, iso_continuum)
plt.plot(orion_wavelengths_miri, 0.0027*(orion_data_miri - continuum_orion_miri) + orion_iso_continuum)
plt.xlim(10.8, 11.8)
plt.ylim(7.5, 27.5)
plt.show()
'''
#%%
'''
plt.figure()
plt.plot(iso_wavelengths_112, iso_integrand)
plt.plot([iso_wavelengths_112[0], 11.142], [0, 6.41e-8])
plt.plot([11.142, 11.259], [6.41e-8, 2.64e-7])
plt.plot([11.259, 11.259], [6.41e-8, 2.64e-7])
plt.plot([11.259, 11.320], [2.64e-7, 2.64e-7])
plt.plot([11.142, 11.259], [6.41e-8, 6.41e-8])
plt.plot([11.320, iso_wavelengths_112[-1]], [2.64e-7, 0])
plt.show()
'''
iso_intensity_112_rectangles = (2.64e-7)*(11.320 - 11.259) + (2.64e-7)*(11.320 - 11.259)
iso_intesnity_112_triangles = 0.5*((6.41e-8)*(11.142 - iso_wavelengths_112[0]) + (2.64e-7 - 6.41e-8)*(11.259 - 11.142) + (2.64e-7)*(iso_wavelengths_112[-1] - 11.320))
iso_intensity_112 = iso_intensity_112_rectangles + iso_intesnity_112_triangles

# old version print('estimated flux using triangle up to 11.259, triangle from 11.383, and rectangle in the middle that starts at the height of the first point, is 11.1e-8')

print('estimated flux using several triangles and rectangles, is 11.1e-8', iso_intensity_112)


#%%













#%%



#%%

import pyregion

from astropy.visualization.wcsaxes import WCSAxes
#%%
'''
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
#ax.set_xticks(np.arange(1000, 2000, 100))
#ax.set_yticks(np.arange(1000, 2000, 100))
fig.add_axes(ax)


ax.imshow(nircam_data, vmax=10)


for p in patch_list1 + patch_list2 + patch_list3:
    ax.add_patch(p)
    
for t in artist_list1 + artist_list2 + artist_list3:
    ax.add_artist(t)

ax.invert_xaxis()
ax.invert_yaxis()



plt.show()



#%%

cam_header_f335m = fits.getheader(miricam_image_file, ext=1)

pog = wcs.WCS(cam_header_f335m)



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

for p in patch_list1 + patch_list2 + patch_list3:
    ax2.add_patch(p)
    
for t in artist_list1 + artist_list2 + artist_list3:
    ax2.add_artist(t)

ax2.invert_xaxis()
ax2.invert_yaxis()

plt.show()
'''
#%%



nirspec_data_new = rnf.extract_pixels_from_region('data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 
                                                  nirspec_data, 'NIRSPEC_NORTH_bigbox_new.reg')


#%%

import RingNebulaFunctions as rnf

wavelengths_nirspec, nirspec_data, nirspec_error_data = rnf.loading_function(
    'data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 1)

nirspec_weighted_mean, nirspec_error_mean = rnf.extract_weighted_mean_from_region(
    'data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 
    nirspec_data, nirspec_error_data, 'NIRSPEC_NORTH_bigbox_new.reg')
nirspec_weighted_mean = np.array(nirspec_weighted_mean['region_0'])
#%%
'''
plt.figure()
plt.plot(wavelengths_nirspec[:nirspec_cutoff], nirspec_weighted_mean[:nirspec_cutoff] - 2.4, 
         label='g395m-f290, North, offset=-2.4', color='#dc267f')
plt.show()

#%%
plt.figure()
plt.imshow(nirspec_error_data[219])
plt.show()

#%%



plt.figure()
plt.plot(wavelengths_nirspec_short, nirspec_regular_mean_short - 1)
plt.plot(wavelengths_nirspec, nirspec_regular_mean)
plt.ylim(0, 10)
plt.show()

#%%

pog = 'data/misc/jw01192-o010_t002_miri_ch2-shortmediumlong_s3d.fits'
frog = get_pkg_data_filename(pog)
pfrog = fits.getheader(frog, ext=0)
'''


#%%
# 1157 (size 2317), 1146 (size 2310)
with fits.open('data/cams/nircam_color_F300M_F335M.fits') as hdul:
    nircam_data = hdul[0].data
    nircam_data = np.rot90(nircam_data[:2292, :2314], 2)
    hdul[0].data = nircam_data
    
    hdul[0].header['PC1_2'] = -1*hdul[0].header['PC1_2']
    hdul[0].header['PC2_2'] = -1*hdul[0].header['PC2_2']
    hdul[0].header['PC1_1'] = -1*hdul[0].header['PC1_1']
    hdul[0].header['PC2_1'] = -1*hdul[0].header['PC2_1']
    
    hdul[0].header['NAXIS1'] = 2292
    hdul[0].header['NAXIS2'] = 2314
    
    hdul.writeto('data/cams/nircam_color_F300M_F335M_flipped.fits', overwrite=True)

# 863.5, 1786 1423.5, 2240
with fits.open('data/cams/miri_color_F1000W_F1130W.fits') as hdul:
    miricam_data = hdul[0].data
    miricam_data = np.rot90(miricam_data[607:, :1727], 2)
    hdul[0].data = miricam_data
    
    hdul[0].header['PC1_2'] = -1*hdul[0].header['PC1_2']
    hdul[0].header['PC2_2'] = -1*hdul[0].header['PC2_2']
    hdul[0].header['PC1_1'] = -1*hdul[0].header['PC1_1']
    hdul[0].header['PC2_1'] = -1*hdul[0].header['PC2_1']
    
    hdul[0].header['NAXIS1'] = 1786
    hdul[0].header['NAXIS2'] = 1633
    
    hdul[0].header['CRPIX2'] = 807.5
    
    hdul.writeto('data/cams/miri_color_F1000W_F1130W_flipped.fits', overwrite=True)


'''
RING PAH MAP
'''

cam_header_f335m = fits.getheader(nircam_file_loc, ext=0)
cam_header_f335m = fits.getheader('data/cams/nircam_color_F300M_F335M_flipped.fits', ext=0)
pog1 = wcs.WCS(cam_header_f335m)

region_name1 = 'physical;polygon(994.89487,1452.2538,953.03342,1447.9915,948.38642,1491.6011,990.07236,1495.9708) # color=#39ff14'
r1 = pyregion.parse(region_name1)
patch_list1, artist_list1 = r1.get_mpl_patches_texts()

region_name2 = 'physical;polygon(1039.8806,1611.6539,1087.974,1577.9429,1020.5521,1481.7551,972.45933,1515.4671) # color=#39ff14'
r2 = pyregion.parse(region_name2)
patch_list2, artist_list2 = r2.get_mpl_patches_texts()

region_name3 = 'physical;ellipse(1164.8768,1086.6737,677.41734,507.18996,184.47065) # color=black'
r3 = pyregion.parse(region_name3)
patch_list3, artist_list3 = r3.get_mpl_patches_texts()

region_name4 = 'physical;vector(1270.2202,1257.6155,58.033503,129.77422) # vector=1 color=white' # 309.47057 320.529426
r4 = pyregion.parse(region_name4)
patch_list4, artist_list4 = r4.get_mpl_patches_texts()

region_name5 = 'physical;vector(1270.3708,1257.621,58.255732,219.09987) # vector=1 color=white' #39.099794 50.529426
r5 = pyregion.parse(region_name5)
patch_list5, artist_list5 = r5.get_mpl_patches_texts()

region_name6 = 'physical;box(1004.8786,1511.2303,1380.2005,58.731937,234.96003) # dash=1 color=white'
r6 = pyregion.parse(region_name6)
patch_list6, artist_list6 = r6.get_mpl_patches_texts()

region_name8 = 'physical;vector(1016,1467,45,180) # vector=1 color=#39ff14'
r8 = pyregion.parse(region_name8)
patch_list8, artist_list8 = r8.get_mpl_patches_texts()

ax = plt.figure('RNF_paper_pah_ring', figsize=(18,18)).add_subplot(211, projection=pog1)

plt.rcParams.update({'font.size': 28})

#ax = plt.subplot(projection=pog)


im = plt.imshow(nircam_data, vmin=0.5, vmax=2, cmap='gnuplot')






# Add the colorbar:
cbar = plt.colorbar(location = "right", fraction=0.05, pad=0.02)
cbar.formatter.set_powerlimits((0, 0))
cbar.ax.yaxis.set_offset_position('left')
            
#Customization of axes' appearance:

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


plt.xlim((700, 1300))
plt.ylim((1200, 1700))

#ax.invert_yaxis()
#ax.invert_xaxis()

ax.tick_params(axis = "y", color = "k", left = True, right = True, direction = "out")
ax.tick_params(axis = "x", color = "k", bottom = True, top = True,  direction = "out")

for p in patch_list1 + patch_list6 + patch_list3 + patch_list4 + patch_list5 + patch_list2 + patch_list8:
    ax.add_patch(p)
    
for t in artist_list1 + artist_list2 + artist_list3 + artist_list4 + artist_list5:
    ax.add_artist(t)

ax.set_facecolor("k")


ax.text(0.80, 0.85, 'PAH Ring', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='black')

ax.text(0.53, 0.55, 'North region', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='#39ff14')

ax.text(0.60, 0.67, 'Spitzer SL1 Peak', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='#39ff14')

ax.text(0.46, 0.43, 'Spitzer SL1', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')

ax.text(0.05, 0.95, 'F335M/F300M', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', color='black')

ax.text(0.855, 0.235, 'N', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')

ax.text(0.84, 0.053, 'E', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')


'''
11.2
'''



cam_header_f335m = fits.getheader('data/cams/miri_color_F1000W_F1130W_flipped.fits', ext=0)

pog2 = wcs.WCS(cam_header_f335m)

region_name1 = 'physical;polygon(623.30008,698.27756,600.05493,692.72046,594.14938,716.92075,617.28763,722.52507) # color=#39ff14'
r1 = pyregion.parse(region_name1)
patch_list1, artist_list1 = r1.get_mpl_patches_texts()

region_name2 = 'physical;polygon(636.60218,791.40792,666.22002,776.0574,635.519,716.82118,605.90146,732.17232) # color=#39ff14'
r2 = pyregion.parse(region_name2)
patch_list2, artist_list2 = r2.get_mpl_patches_texts()

region_name3 = 'physical;ellipse(746.56987,505.2893,384.77209,288.08318,192.10777) # color=black'
r3 = pyregion.parse(region_name3)
patch_list3, artist_list3 = r3.get_mpl_patches_texts()

region_name4 = 'physical;vector(792.98081,609.47001,32.962948,137.41134) # vector=1 color=white'#317.41526 312.8923532292306
r4 = pyregion.parse(region_name4)
patch_list4, artist_list4 = r4.get_mpl_patches_texts()

region_name5 = 'physical;vector(793.06516,609.4845,33.089173,226.73699) # vector=1 color=white' #46.740917 42.89235322923057
r5 = pyregion.parse(region_name5)
patch_list5, artist_list5 = r5.get_mpl_patches_texts()

region_name6 = 'physical;box(625.41911,733.56832,783.95194,33.359657,242.59715) # dash=1 color=white'
r6 = pyregion.parse(region_name6)
patch_list6, artist_list6 = r6.get_mpl_patches_texts()

region_name8 = 'physical;vector(636,707,30,180) # vector=1 color=#39ff14'
r8 = pyregion.parse(region_name8)
patch_list8, artist_list8 = r8.get_mpl_patches_texts()

#fig = plt.figure(figsize=(10, 8))
#ax = plt.subplot(projection=pog)
ax = plt.figure('RNF_paper_pah_ring', figsize=(18,18)).add_subplot(212, projection=pog2)

plt.rcParams.update({'font.size': 28})

im = plt.imshow(miricam_data, vmin=0.3, vmax=1.6, cmap='gnuplot')





# Add the colorbar:
cbar = plt.colorbar(location = "right", fraction=0.05, pad=0.02)
cbar.formatter.set_powerlimits((0, 0))
cbar.ax.yaxis.set_offset_position('left')
            
#Customization of axes' appearance:

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



plt.xlim((450, 810)) #382
plt.ylim((570, 870)) #330

#ax.invert_yaxis()
#ax.invert_xaxis()

for p in patch_list1 + patch_list6 + patch_list3 + patch_list4 + patch_list5 + patch_list2 + patch_list8:
    ax.add_patch(p)
    
for t in artist_list1 + artist_list2 + artist_list3 + artist_list4 + artist_list5:
    ax.add_artist(t)

ax.set_facecolor("k")



ax.text(0.80, 0.82, 'PAH Ring', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='black')

ax.text(0.52, 0.47, 'North region', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='#39ff14')

ax.text(0.57, 0.60, 'Spitzer SL1 Peak', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='#39ff14')

ax.text(0.47, 0.34, 'Spitzer SL1', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')

ax.text(0.05, 0.95, 'F1130W/F1000W', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', color='black')

ax.text(0.85, 0.23, 'N', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')

ax.text(0.86, 0.050, 'E', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')

plt.savefig('Figures/RNF_paper_pah_ring.pdf', bbox_inches='tight')




#%%



# 1157 (size 2317), 1146 (size 2310)
with fits.open('data/cams/nircam_color_F300M_F335M.fits') as hdul:
    nircam_data = hdul[0].data
    nircam_data = np.rot90(nircam_data[:2292, :2314], 2)
    hdul[0].data = nircam_data
    
    hdul[0].header['PC1_2'] = -1*hdul[0].header['PC1_2']
    hdul[0].header['PC2_2'] = -1*hdul[0].header['PC2_2']
    hdul[0].header['PC1_1'] = -1*hdul[0].header['PC1_1']
    hdul[0].header['PC2_1'] = -1*hdul[0].header['PC2_1']
    
    hdul[0].header['NAXIS1'] = 2292
    hdul[0].header['NAXIS2'] = 2314
    
    hdul.writeto('data/cams/nircam_color_F300M_F335M_flipped.fits', overwrite=True)

# 863.5, 1786 1423.5, 2240
with fits.open('data/cams/miri_color_F1000W_F1130W.fits') as hdul:
    miricam_data = hdul[0].data
    miricam_data = np.rot90(miricam_data[607:, :1727], 2)
    hdul[0].data = miricam_data
    
    hdul[0].header['PC1_2'] = -1*hdul[0].header['PC1_2']
    hdul[0].header['PC2_2'] = -1*hdul[0].header['PC2_2']
    hdul[0].header['PC1_1'] = -1*hdul[0].header['PC1_1']
    hdul[0].header['PC2_1'] = -1*hdul[0].header['PC2_1']
    
    hdul[0].header['NAXIS1'] = 1786
    hdul[0].header['NAXIS2'] = 1633
    
    hdul[0].header['CRPIX2'] = 807.5
    
    hdul.writeto('data/cams/miri_color_F1000W_F1130W_flipped.fits', overwrite=True)




#%%

plt.figure()
plt.imshow(nircam_data, vmax=5)

plt.show()

plt.figure()
plt.imshow(miricam_data, vmax=5)

plt.show()

    
#%%

'''
FIGURE 1
'''

'''
Increase font size. Zoom in (just capture the ring, nothing of the dark blue outside the 
ring - can we try to loose less white space for the declination?? perhaps but the 
label between the numbers or at the top??). Change the white color box to something 
that stands out and perhaps make the line thicker (e.g. try red). Label the 2 white 
boxes as JWST North and JWST West in the same color. Change the color of the NIRSpec 
and MIRI FOV so you can see them. Change the label 'Spitzer' to 'Spitzer SH'. Add the 
2nd Spitzer aperture (that I will give you) and label it 'Spitzer SL'.
'''

with fits.open('data/cams/jw01558005001_04101_00001_nrcblong_combined_i2d.fits') as hdul:
    cam_data_f335m = hdul[1].data
    cam_data_f335m = np.rot90(cam_data_f335m[:2292, :2314], 2)
    hdul[1].data = cam_data_f335m
    
    hdul[1].header['PC1_2'] = -1*hdul[1].header['PC1_2']
    hdul[1].header['PC2_2'] = -1*hdul[1].header['PC2_2']
    hdul[1].header['PC1_1'] = -1*hdul[1].header['PC1_1']
    hdul[1].header['PC2_1'] = -1*hdul[1].header['PC2_1']
    
    hdul[1].header['NAXIS1'] = 2292
    hdul[1].header['NAXIS2'] = 2314
    
    hdul.writeto('data/cams/jw01558005001_04101_00001_nrcblong_combined_i2d_flipped.fits', overwrite=True)

cam_header_f335m = fits.getheader('data/cams/jw01558005001_04101_00001_nrcblong_combined_i2d_flipped.fits', ext=1)

pog1 = wcs.WCS(cam_header_f335m)

region_name1 = 'physical;polygon(994.67088,1451.8343,952.81016,1447.5723,948.16346,1491.1812,989.84868,1495.5505) # color=#39ff14 width=2'
r1 = pyregion.parse(region_name1)
patch_list1, artist_list1 = r1.get_mpl_patches_texts()

region_name2 = 'physical;box(1363.7197,1395.0246,47.857734,47.49265,45.470379) # color=white'
r2 = pyregion.parse(region_name2)
patch_list2, artist_list2 = r2.get_mpl_patches_texts()

region_name3 = 'physical;box(1011.5964,1521.4113,1380.1758,58.730885,234.95974) # dash=1 color=white'
r3 = pyregion.parse(region_name3)
patch_list3, artist_list3 = r3.get_mpl_patches_texts()

region_name4 = 'physical;vector(1804.0254,721.4753,90.035003,129.77393) # vector=1 color=white' # #320.900493
r4 = pyregion.parse(region_name4)
patch_list4, artist_list4 = r4.get_mpl_patches_texts()

region_name5 = 'physical;vector(1804.0254,721.4753,90.252148,219.09958) # vector=1 color=white' #50.900493
r5 = pyregion.parse(region_name5)
patch_list5, artist_list5 = r5.get_mpl_patches_texts()

region_name6 = 'physical;box(1684.6144,1149.6802,286.98767,358.73459,94.616863) # color=white'
r6 = pyregion.parse(region_name6)
patch_list6, artist_list6 = r6.get_mpl_patches_texts()

region_name7 = 'physical;polygon(1039.6566,1611.2313,1087.749,1577.5207,1020.3278,1481.335,972.23606,1515.0466) # color=#39ff14 width=2'
r7 = pyregion.parse(region_name7)
patch_list7, artist_list7 = r7.get_mpl_patches_texts()

region_name8 = 'physical;vector(888.3088,1465.6297,86.502547,6.556267) # vector=1 color=#39ff14 width=2'
r8 = pyregion.parse(region_name8)
patch_list8, artist_list8 = r8.get_mpl_patches_texts()

region_name9 = 'physical;ellipse(1164.6479,1086.2599,677.4052,507.18087,184.47036) # dash=1 color=white'
r9 = pyregion.parse(region_name9)
patch_list9, artist_list9 = r9.get_mpl_patches_texts()

region_name10 = 'physical;box(1128.005,1107.0806,286.98767,358.73459,94.616863) # color=white'
r10 = pyregion.parse(region_name10)
patch_list10, artist_list10 = r10.get_mpl_patches_texts()

region_name11 = 'line(749.99835,799.99418,906.28559,800.55263) # color=white'
r11 = pyregion.parse(region_name11)
patch_list11, artist_list11 = r11.get_mpl_patches_texts()








ax = plt.figure('RNF_paper_ring_overview', figsize=(10,8)).add_subplot(111, projection=pog1)

plt.rcParams.update({'font.size': 28})

#ax = plt.subplot(projection=pog)


im = plt.imshow(cam_data_f335m, vmin=0,  vmax=8, cmap='gnuplot')





# Add the colorbar:
cbar = plt.colorbar(location = "right", fraction=0.05, pad=0.02)
cbar.formatter.set_powerlimits((0, 0))
cbar.ax.yaxis.set_offset_position('left')
            
#Customization of axes' appearance:

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



plt.xlim((500, 1900)) #1400
plt.ylim((600, 1700)) #1100


for p in patch_list1 + patch_list2 + patch_list3 + patch_list4 + patch_list5 +\
    patch_list6 + patch_list7 + patch_list8 + patch_list9 + patch_list10 + patch_list11:
    ax.add_patch(p)

ax.set_facecolor("k")


ax.text(0.56, 0.32, 'Spitzer SH', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')

ax.text(0.03, 0.81, 'North region', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='#39ff14')

ax.text(0.51, 0.80, 'West region', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')

ax.text(0.04, 0.93, 'Spitzer SL1 Peak', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='#39ff14')

ax.text(0.02, 0.69, 'Spitzer SL1', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')

ax.text(0.855, 0.215, 'N', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')

ax.text(0.847, 0.065, 'E', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')


ax.text(0.80, 0.85, 'PAH Ring', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')



plt.savefig('Figures/RNF_paper_ring_overview.pdf', bbox_inches='tight')




#%%



coords_x = np.array([0, 1])
coords_y = np.array([0,0])

t_11 = -0.6806230967493241
t_12 = 0.7326337421736457
t_21 = 0.7326337421736457
t_22 = 0.6806230967493241

t_11 = 0
t_12 = 1
t_21 = 1
t_22 = 0



new_coords_x = coords_x*t_11 + coords_y*t_12
new_coords_y = coords_x*t_21 + coords_y*t_21

plt.figure()
plt.plot(coords_x, coords_y)
plt.plot(new_coords_x, new_coords_y)
plt.show()



#%%
# 825.5 (size 1644), 522.5 (size 1043)
with fits.open('data/cams/ring_nebula_F1500W_i2d.fits') as hdul:
    miri_f1500w_data = hdul[1].data
    pog = hdul[1].header
    miri_f1500w_data = np.rot90(miri_f1500w_data[2:, 7:], 2)
    hdul[1].data = miri_f1500w_data
    
    hdul[1].header['PC1_2'] = -1*hdul[1].header['PC1_2']
    hdul[1].header['PC2_2'] = -1*hdul[1].header['PC2_2']
    hdul[1].header['PC1_1'] = -1*hdul[1].header['PC1_1']
    hdul[1].header['PC2_1'] = -1*hdul[1].header['PC2_1']
    
    hdul[1].header['NAXIS1'] = 1637
    hdul[1].header['NAXIS2'] = 1043
    
    hdul[1].header['CRPIX1'] = 818.5
    hdul[1].header['CRPIX2'] = 522.5
    
    hdul.writeto('data/cams/ring_nebula_F1500W_i2d_flipped.fits', overwrite=True)


#%%

'''
RING PAH MAP
'''

cam_header_f335m = fits.getheader(nircam_file_loc, ext=0)
cam_header_f335m = fits.getheader('data/cams/ring_nebula_F1500W_i2d_flipped.fits', ext=1)
pog1 = wcs.WCS(cam_header_f335m)

region_name1 = 'physical;polygon(539.89574,709.26318,516.65096,703.70454,510.74381,727.90444,533.88168,733.51028) # color=#39ff14'
r1 = pyregion.parse(region_name1)
patch_list1, artist_list1 = r1.get_mpl_patches_texts()

region_name2 = 'physical;box(751.94759,705.13324,27.183613,26.976242,53.107595) # color=#39ff14'
r2 = pyregion.parse(region_name2)
patch_list2, artist_list2 = r2.get_mpl_patches_texts()

region_name3 = 'physical;ellipse(663.1783,516.28307,384.77209,288.08318,192.10758) # color=white'
r3 = pyregion.parse(region_name3)
patch_list3, artist_list3 = r3.get_mpl_patches_texts()

region_name4 = 'physical;vector(750,500,32.962948,137.41115) # vector=1 color=white' # 309.47057 320.529426
r4 = pyregion.parse(region_name4)
patch_list4, artist_list4 = r4.get_mpl_patches_texts()

region_name5 = 'physical;vector(750,500,33.089174,226.7368) # vector=1 color=white' #39.099794 50.529426
r5 = pyregion.parse(region_name5)
patch_list5, artist_list5 = r5.get_mpl_patches_texts()







ax = plt.figure('RNF_paper_pah_ring', figsize=(18,18)).add_subplot(111, projection=pog1)

plt.rcParams.update({'font.size': 28})

#ax = plt.subplot(projection=pog)


im = plt.imshow(miri_f1500w_data, cmap='gnuplot')




# Add the colorbar:
cbar = plt.colorbar(location = "right", fraction=0.05, pad=0.02)
cbar.formatter.set_powerlimits((0, 0))
cbar.ax.yaxis.set_offset_position('left')
            
#Customization of axes' appearance:

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


plt.xlim((200, 800)) 
plt.ylim((450, 921))

#ax.invert_yaxis()
#ax.invert_xaxis()

ax.tick_params(axis = "y", color = "k", left = True, right = True, direction = "out")
ax.tick_params(axis = "x", color = "k", bottom = True, top = True,  direction = "out")

for p in patch_list1 + patch_list2 + patch_list3 + patch_list4 + patch_list5:
    ax.add_patch(p)
    
for t in artist_list1 + artist_list2 + artist_list3 + artist_list4 + artist_list5:
    ax.add_artist(t)

ax.set_facecolor("k")


ax.text(0.80, 0.80, 'PAH Ring', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')

ax.text(0.58, 0.54, 'North region', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='#39ff14')

ax.text(0.85, 0.62, 'West region', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='#39ff14')

ax.text(0.05, 0.95, 'F1500W', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', color='white')

ax.text(0.85, 0.17, 'N', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')

ax.text(0.86, 0.050, 'E', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', color='white')


plt.savefig('Figures/RNF_paper_1500_ring.pdf', bbox_inches='tight')





