
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

#rebinning module
from reproject import reproject_interp

#Functions python script
import RingNebulaFunctions as rnf


'''
LIST OF RNF FUNCTIONS
'''

# loading_function
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
LOADING DATA (takes forever to run so not doing this rn)
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

#%%

#loading in template PAH spectra
#pah_image_file = np.loadtxt('data/misc/barh_stick_csub.fits.dat', skiprows=1)
orion_image_file_nirspec = np.loadtxt('data/misc/templatesT_seg3_crds1084_20230514.dat', skiprows=7)
orion_image_file_miri = np.loadtxt('data/misc/templatesT_MRS_crds1154_add_20231212.dat', skiprows=7)
orion_wavelengths_nirspec = orion_image_file_nirspec[:,0]
orion_wavelengths_miri = orion_image_file_miri[:,0]
orion_data_nirspec = orion_image_file_nirspec[:,6]
orion_data_miri = orion_image_file_miri[:,9]
#pah_data = pah_image_file[:,1]

#%%

#loading in horsehead nebula data
#hh_image_file = np.loadtxt('data/misc/horsehead.txt', skiprows=3)
#hh_wavelengths = hh_image_file[:,0]
#hh_data = hh_image_file[:,1]

#loading in better horsehead nebula data from jwst
hh_wavelengths2, hh_image_data2, hh_error2 = rnf.loading_function(
    'data/misc/jw01192-o010_t002_miri_ch2-shortmediumlong_s3d.fits', 1)

hh_wavelengths3, hh_image_data3, hh_error3 = rnf.loading_function(
    'data/misc/jw01192-o010_t002_miri_ch3-shortmediumlong_s3d.fits', 1)


#%%

#weighted mean over entire aperture

hh_data2, hh_weighted_mean_error2 = rnf.weighted_mean_finder_simple(hh_image_data2, hh_error2)
hh_data3, hh_weighted_mean_error3 = rnf.weighted_mean_finder_simple(hh_image_data3, hh_error3)



#%%

#loading in NGC 7027 nebula data
ngc7027_image_file = np.loadtxt('data/misc/ngc7027.fits.dat', skiprows=1)
ngc7027_wavelengths = ngc7027_image_file[:,0]
ngc7027_data = ngc7027_image_file[:,1]
print('pog')

####################################

#%%

'''
LOADING ANALYSIS DATA
'''

data1 = np.load('Analysis/data1.npy', allow_pickle=True)
data2 = np.load('Analysis/data2.npy', allow_pickle=True)
data3 = np.load('Analysis/data3.npy', allow_pickle=True)
data4 = np.load('Analysis/data4.npy', allow_pickle=True)
data5 = np.load('Analysis/data5.npy', allow_pickle=True)
data6 = np.load('Analysis/data6.npy', allow_pickle=True)
data7 = np.load('Analysis/data7.npy', allow_pickle=True)
data8 = np.load('Analysis/data8.npy', allow_pickle=True)
data9 = np.load('Analysis/data9.npy', allow_pickle=True)



data1_west = np.load('Analysis/data1_west.npy', allow_pickle=True)
data2_west = np.load('Analysis/data2_west.npy', allow_pickle=True)
data3_west = np.load('Analysis/data3_west.npy', allow_pickle=True)
data4_west = np.load('Analysis/data4_west.npy', allow_pickle=True)
data5_west = np.load('Analysis/data5_west.npy', allow_pickle=True)
data6_west = np.load('Analysis/data6_west.npy', allow_pickle=True)
data7_west = np.load('Analysis/data7_west.npy', allow_pickle=True)
data8_west = np.load('Analysis/data8_west.npy', allow_pickle=True)
data9_west = np.load('Analysis/data9_west.npy', allow_pickle=True)



weighted_mean_error1 = np.load('Analysis/weighted_mean_error1.npy', allow_pickle=True)
weighted_mean_error2 = np.load('Analysis/weighted_mean_error2.npy', allow_pickle=True)
weighted_mean_error3 = np.load('Analysis/weighted_mean_error3.npy', allow_pickle=True)
weighted_mean_error4 = np.load('Analysis/weighted_mean_error4.npy', allow_pickle=True)
weighted_mean_error5 = np.load('Analysis/weighted_mean_error5.npy', allow_pickle=True)
weighted_mean_error6 = np.load('Analysis/weighted_mean_error6.npy', allow_pickle=True)
weighted_mean_error7 = np.load('Analysis/weighted_mean_error7.npy', allow_pickle=True)
weighted_mean_error8 = np.load('Analysis/weighted_mean_error8.npy', allow_pickle=True)
weighted_mean_error9 = np.load('Analysis/weighted_mean_error9.npy', allow_pickle=True)

weighted_mean_error1_west = np.load('Analysis/weighted_mean_error1_west.npy', allow_pickle=True)
weighted_mean_error2_west = np.load('Analysis/weighted_mean_error2_west.npy', allow_pickle=True)
weighted_mean_error3_west = np.load('Analysis/weighted_mean_error3_west.npy', allow_pickle=True)
weighted_mean_error4_west = np.load('Analysis/weighted_mean_error4_west.npy', allow_pickle=True)
weighted_mean_error5_west = np.load('Analysis/weighted_mean_error5_west.npy', allow_pickle=True)
weighted_mean_error6_west = np.load('Analysis/weighted_mean_error6_west.npy', allow_pickle=True)
weighted_mean_error7_west = np.load('Analysis/weighted_mean_error7_west.npy', allow_pickle=True)
weighted_mean_error8_west = np.load('Analysis/weighted_mean_error8_west.npy', allow_pickle=True)
weighted_mean_error9_west = np.load('Analysis/weighted_mean_error9_west.npy', allow_pickle=True)

data1_west_blob = np.load('Analysis/data1_west_blob.npy', allow_pickle=True)
data2_west_blob = np.load('Analysis/data2_west_blob.npy', allow_pickle=True)
data3_west_blob = np.load('Analysis/data3_west_blob.npy', allow_pickle=True)
data4_west_blob = np.load('Analysis/data4_west_blob.npy', allow_pickle=True)
data5_west_blob = np.load('Analysis/data5_west_blob.npy', allow_pickle=True)
data6_west_blob = np.load('Analysis/data6_west_blob.npy', allow_pickle=True)
data7_west_blob = np.load('Analysis/data7_west_blob.npy', allow_pickle=True)
data8_west_blob = np.load('Analysis/data8_west_blob.npy', allow_pickle=True)
data9_west_blob = np.load('Analysis/data9_west_blob.npy', allow_pickle=True)


    
weighted_mean_error1_west_blob = np.load('Analysis/weighted_mean_error1_west_blob.npy', allow_pickle=True)
weighted_mean_error2_west_blob = np.load('Analysis/weighted_mean_error2_west_blob.npy', allow_pickle=True)
weighted_mean_error3_west_blob = np.load('Analysis/weighted_mean_error3_west_blob.npy', allow_pickle=True)
weighted_mean_error4_west_blob = np.load('Analysis/weighted_mean_error4_west_blob.npy', allow_pickle=True)
weighted_mean_error5_west_blob = np.load('Analysis/weighted_mean_error5_west_blob.npy', allow_pickle=True)
weighted_mean_error6_west_blob = np.load('Analysis/weighted_mean_error6_west_blob.npy', allow_pickle=True)
weighted_mean_error7_west_blob = np.load('Analysis/weighted_mean_error7_west_blob.npy', allow_pickle=True)
weighted_mean_error8_west_blob = np.load('Analysis/weighted_mean_error8_west_blob.npy', allow_pickle=True)
weighted_mean_error9_west_blob = np.load('Analysis/weighted_mean_error9_west_blob.npy', allow_pickle=True)



nirspec_weighted_mean4 = np.load('Analysis/nirspec_weighted_mean4.npy', allow_pickle=True)

nirspec_weighted_mean4_west = np.load('Analysis/nirspec_weighted_mean4_west.npy', allow_pickle=True)

nirspec_error_mean4 = np.load('Analysis/nirspec_error_mean4.npy', allow_pickle=True)

nirspec_error_mean4_west = np.load('Analysis/nirspec_error_mean4_west.npy', allow_pickle=True)

nirspec_weighted_mean4_west_blob = np.load('Analysis/nirspec_weighted_mean4_west_blob.npy', allow_pickle=True)


nirspec_error_mean4_west_blob = np.load('Analysis/nirspec_error_mean4_west_blob.npy', allow_pickle=True)

spitzer_data = np.load('Analysis/spitzer_data.npy', allow_pickle=True)



nirspec_weighted_mean4_smooth = np.load('Analysis/nirspec_weighted_mean4_smooth.npy', allow_pickle=True)
nirspec_weighted_mean4_smooth_west = np.load('Analysis/nirspec_weighted_mean4_smooth_west.npy', allow_pickle=True)



corrected_data1 = np.load('Analysis/corrected_data1.npy')
corrected_data2 = np.load('Analysis/corrected_data2.npy')
corrected_data3 = np.load('Analysis/corrected_data3.npy')
corrected_data4 = np.load('Analysis/corrected_data4.npy')
corrected_data5 = np.load('Analysis/corrected_data5.npy')
corrected_data6 = np.load('Analysis/corrected_data6.npy')
corrected_data7 = np.load('Analysis/corrected_data7.npy')
corrected_data8 = np.load('Analysis/corrected_data8.npy')
corrected_data9 = np.load('Analysis/corrected_data9.npy')

corrected_data1_west = np.load('Analysis/corrected_data1_west.npy')
corrected_data2_west = np.load('Analysis/corrected_data2_west.npy')
corrected_data3_west = np.load('Analysis/corrected_data3_west.npy')
corrected_data4_west = np.load('Analysis/corrected_data4_west.npy')
corrected_data5_west = np.load('Analysis/corrected_data5_west.npy')
corrected_data6_west = np.load('Analysis/corrected_data6_west.npy')
corrected_data7_west = np.load('Analysis/corrected_data7_west.npy')
corrected_data8_west = np.load('Analysis/corrected_data8_west.npy')
corrected_data9_west = np.load('Analysis/corrected_data9_west.npy')

corrected_data1_west_blob = np.load('Analysis/corrected_data1_west_blob.npy')
corrected_data2_west_blob = np.load('Analysis/corrected_data2_west_blob.npy')
corrected_data3_west_blob = np.load('Analysis/corrected_data3_west_blob.npy')
corrected_data4_west_blob = np.load('Analysis/corrected_data4_west_blob.npy')
corrected_data5_west_blob = np.load('Analysis/corrected_data5_west_blob.npy')
corrected_data6_west_blob = np.load('Analysis/corrected_data6_west_blob.npy')
corrected_data7_west_blob = np.load('Analysis/corrected_data7_west_blob.npy')
corrected_data8_west_blob = np.load('Analysis/corrected_data8_west_blob.npy')
corrected_data9_west_blob = np.load('Analysis/corrected_data9_west_blob.npy')



####################################

'''
COMPARISON CONTINUA
'''






#%%

#hh continuum subtraction

#combining 3 different channels

pah_removed_hh, wavelength_pah_removed_hh, overlap = rnf.flux_aligner2(
    hh_wavelengths2, hh_wavelengths3, hh_data2, hh_data3-15)



#first, need to fit continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(wavelength_pah_removed_hh, 2) == 11.0)[0][0]
temp_index_2 = np.where(np.round(hh_wavelengths3, 2) == 11.82)[0][0] #was 11.82 originally

#temp_index_1 = np.where(np.round(wavelength_pah_removed_hh, 2) == 11.0)[0][0]
#temp_index_2 = np.where(np.round(hh_wavelengths3, 2) == 11.65)[0][0] #was 11.82 originally

#calculating the slope of the line to use

#preventing the value found from being on a line or something
pah_slope_1 = np.mean(pah_removed_hh[temp_index_1 - 20:20+temp_index_1])

pah_slope_2 = np.mean(hh_data3[temp_index_2 - 20:20+temp_index_2] - 15)

#value where the wavelengths change interval



pah_slope = (pah_slope_2 - pah_slope_1)/\
(hh_wavelengths3[temp_index_2] - wavelength_pah_removed_hh[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(pah_removed_hh[:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(hh_data3[temp_index_2:]-15))

pah_removed_3 = pah_slope*(wavelength_pah_removed_hh[temp_index_1:overlap[0]] - 
                           wavelength_pah_removed_hh[temp_index_1]) + pah_slope_1
pah_removed_4 = pah_slope*(hh_wavelengths3[overlap[1]:temp_index_2] - 
                           wavelength_pah_removed_hh[temp_index_1]) + pah_slope_1

#putting it all together
pah_removed_hh = np.concatenate((pah_removed_1, pah_removed_3))
pah_removed_hh = np.concatenate((pah_removed_hh, pah_removed_4))
pah_removed_hh = np.concatenate((pah_removed_hh, pah_removed_2))

pah_hh, wavelength_pah_hh, overlap = rnf.flux_aligner2(
    hh_wavelengths2, hh_wavelengths3, hh_data2, hh_data3-15)



continuum_removed_hh = pah_hh - pah_removed_hh# - polynomial6

continuum_hh = np.copy(pah_removed_hh)


plt.figure()
plt.plot(wavelength_pah_hh, pah_hh)
plt.plot(wavelength_pah_hh, pah_removed_hh)
plt.plot(wavelength_pah_hh, continuum_removed_hh)
plt.ylim(-10, 100)
plt.xlim(10, 14)
plt.show()
#plt.close()


#%%

#orion_continuum_subtraction (nirspec)

#first, need to fit continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(orion_wavelengths_nirspec, 2) == 3.0)[0][0]
temp_index_2 = np.where(np.round(orion_wavelengths_nirspec, 2) == 3.7)[0][0] #was 11.82 originally

#temp_index_1 = np.where(np.round(wavelength_pah_removed_hh, 2) == 11.0)[0][0]
#temp_index_2 = np.where(np.round(hh_wavelengths3, 2) == 11.65)[0][0] #was 11.82 originally

#calculating the slope of the line to use

#preventing the value found from being on a line or something
pah_slope_1 = np.mean(orion_data_nirspec[temp_index_1 - 20:20+temp_index_1])

pah_slope_2 = np.mean(orion_data_nirspec[temp_index_2 - 20:20+temp_index_2])

#value where the wavelengths change interval



pah_slope = (pah_slope_2 - pah_slope_1)/\
(orion_wavelengths_nirspec[temp_index_2] - orion_wavelengths_nirspec[temp_index_1])


#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(orion_data_nirspec[:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(orion_data_nirspec[temp_index_2:]-15))

pah_removed_3 = pah_slope*(orion_wavelengths_nirspec[temp_index_1:temp_index_2] - 
                           orion_wavelengths_nirspec[temp_index_1]) + pah_slope_1


#putting it all together
pah_removed_orion_nirspec = np.concatenate((pah_removed_1, pah_removed_3))
pah_removed_orion_nirspec = np.concatenate((pah_removed_orion_nirspec, pah_removed_2))



continuum_removed_orion_nirspec = orion_data_nirspec - pah_removed_orion_nirspec# - polynomial6

continuum_orion_nirspec = np.copy(pah_removed_orion_nirspec)


plt.figure()
plt.plot(wavelength_pah_hh, pah_hh)
plt.plot(wavelength_pah_hh, pah_removed_hh)
plt.plot(wavelength_pah_hh, continuum_removed_hh)
plt.ylim(-10, 100)
plt.xlim(10, 14)
plt.show()

#%%

#orion continuum subtraction (miri)


#first, need to fit continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(orion_wavelengths_miri, 2) == 10.82)[0][0]
temp_index_2 = np.where(np.round(orion_wavelengths_miri, 2) == 11.82)[0][0] #was 11.82 originally

#temp_index_1 = np.where(np.round(wavelength_pah_removed_hh, 2) == 11.0)[0][0]
#temp_index_2 = np.where(np.round(hh_wavelengths3, 2) == 11.65)[0][0] #was 11.82 originally

#calculating the slope of the line to use

#preventing the value found from being on a line or something
pah_slope_1 = np.mean(orion_data_miri[temp_index_1 - 20:20+temp_index_1])

pah_slope_2 = np.mean(orion_data_miri[temp_index_2 - 20:20+temp_index_2])

#value where the wavelengths change interval



pah_slope = (pah_slope_2 - pah_slope_1)/\
(orion_wavelengths_miri[temp_index_2] - orion_wavelengths_miri[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(orion_data_miri[:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(orion_data_miri[temp_index_2:]))

pah_removed_3 = pah_slope*(orion_wavelengths_miri[temp_index_1:temp_index_2] - 
                           orion_wavelengths_miri[temp_index_1]) + pah_slope_1

#putting it all together
pah_removed_orion = np.concatenate((pah_removed_1, pah_removed_3))
pah_removed_orion = np.concatenate((pah_removed_orion, pah_removed_2))

continuum_removed_orion = orion_data_miri - pah_removed_orion# - polynomial6

continuum_orion = np.copy(pah_removed_orion)

plt.figure()
plt.plot(orion_wavelengths_miri, orion_data_miri)
plt.plot(orion_wavelengths_miri, continuum_orion)
plt.xlim(10, 14)
plt.show()
#plt.close()

#%%

#continuum subtracted spitzer data

temp_index_1 = np.where(np.round(spitzer_wavelengths, 2) == 11.0)[0][0]
temp_index_2 = np.where(np.round(spitzer_wavelengths, 2) == 11.82)[0][0]

#calculating the slope of the line to use

#preventing the value found from being on a line or something
pah_slope_1 = np.mean(spitzer_data[temp_index_1 - 5:5+temp_index_1])-0.5

pah_slope_2 = np.mean(spitzer_data[temp_index_2 - 5:5+temp_index_2])


pah_slope = (pah_slope_2 - pah_slope_1)/\
(spitzer_wavelengths[temp_index_2] - spitzer_wavelengths[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(spitzer_data[:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(spitzer_data[temp_index_2:]))

pah_removed_3 = pah_slope*(spitzer_wavelengths[temp_index_1:temp_index_2] - 
                           spitzer_wavelengths[temp_index_1]) + pah_slope_1

pah_removed_3 = pah_removed_3[:,0]

#pah_removed_4 = pah_slope*(spitzer_wavelengths[overlap[1]:temp_index_2] - 
#                           spitzer_wavelengths[temp_index_1]) + pah_slope_1


#putting it all together
pah_removed_spitzer = np.concatenate((pah_removed_1, pah_removed_3))
#pah_removed_spitzer = np.concatenate((pah_removed_spitzer, pah_removed_4))
pah_removed_spitzer = np.concatenate((pah_removed_spitzer, pah_removed_2))


continuum_removed_spitzer = spitzer_data - pah_removed_spitzer# - polynomial6



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

#%%


####################################



'''
5 TO 13 MICRON PLOTS
'''



####################################


#%%

#for smoothing data
from scipy.signal import lfilter

#smoothing data for easier comparison



n = 15  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1



plt.figure()
plt.plot(wavelengths1, lfilter(b, a, corrected_data1))
plt.plot(wavelengths2, lfilter(b, a, corrected_data2)+2, alpha=0.5)
plt.plot(wavelengths3, lfilter(b, a, corrected_data3)+5, alpha=0.5)
plt.plot(wavelengths4, lfilter(b, a, corrected_data4)+7, alpha=0.5)
plt.plot(wavelengths5, lfilter(b, a, corrected_data5)+4, alpha=0.5)
plt.plot(wavelengths6, lfilter(b, a, corrected_data6)+3, alpha=0.5)
plt.plot(wavelengths7, lfilter(b, a, corrected_data7)+3, alpha=0.5)
plt.ylim(0, 20)
plt.show()
#plt.close()

#%%

#making plots of 5 to 13 microns

#north

#combining channels

#array for overlap wavelengths

overlap_array = []

pah_removed, wavelength_pah_removed, overlap = rnf.flux_aligner2(
    wavelengths1, wavelengths2, corrected_data1, corrected_data2 + 2)

overlap_array.append((wavelengths1[overlap[0]] + wavelengths2[overlap[1]])/2)

pah_removed, wavelength_pah_removed, overlap = rnf.flux_aligner2(
    wavelength_pah_removed, wavelengths3, pah_removed, corrected_data3 + 5)

overlap_array.append((wavelength_pah_removed[overlap[0]] + wavelengths3[overlap[1]])/2)

pah_removed, wavelength_pah_removed, overlap = rnf.flux_aligner2(
    wavelength_pah_removed, wavelengths4, pah_removed, corrected_data4 + 7)

overlap_array.append((wavelength_pah_removed[overlap[0]] + wavelengths4[overlap[1]])/2)

pah_removed, wavelength_pah_removed, overlap = rnf.flux_aligner2(
    wavelength_pah_removed, wavelengths5, pah_removed, corrected_data5 + 4)

overlap_array.append((wavelength_pah_removed[overlap[0]] + wavelengths5[overlap[1]])/2)

pah_removed, wavelength_pah_removed, overlap = rnf.flux_aligner2(
    wavelength_pah_removed, wavelengths6, pah_removed, corrected_data6 + 3)

overlap_array.append((wavelength_pah_removed[overlap[0]] + wavelengths6[overlap[1]])/2)

pah_removed, wavelength_pah_removed, overlap = rnf.flux_aligner2(
    wavelength_pah_removed, wavelengths7, pah_removed, corrected_data7 + 3)

overlap_array.append((wavelength_pah_removed[overlap[0]] + wavelengths7[overlap[1]])/2)

#the layout of this code is an illogical mess because of how modified it is from what it used to be 

pah = np.copy(pah_removed) + 10
wavelength_pah = np.copy(wavelength_pah_removed)


#making continuum function and subtracting from data

#6.2

temp_index_1 = np.where(np.round(wavelength_pah_removed, 2) == 5.6)[0][0]
temp_index_2 = np.where(np.round(wavelength_pah_removed, 2) == 6.5)[0][0]

continuum1 = 2.0*np.ones(len(pah_removed[temp_index_1:temp_index_2])) + 10
wave_cont1 = wavelength_pah_removed[temp_index_1:temp_index_2]

pah_temp = pah[temp_index_1:temp_index_2]

continuum_removed_1 = pah_temp - continuum1

everything_removed_1 = rnf.emission_line_remover(continuum_removed_1, 15, 3)

#7.7, 8.6


#for 7.7 amd 8.6 features, exclude 7.1 to 9.2 (funky business from 8.9 to 9.2 that i dont wanna fit so grouping
#it in with the 8.6 feature)

#first, need to fit continuum, do so by eyeballing a polynomial to it, adding 3 to some so everything lines up

pah_removed_077, wavelength_pah_removed_077, overlap1 = rnf.flux_aligner2(
    wavelengths3, wavelengths4, corrected_data3+5, corrected_data4+7)

pah_removed_077, wavelength_pah_removed_077, overlap2 = rnf.flux_aligner2(
    wavelength_pah_removed_077, wavelengths5, pah_removed_077, corrected_data5+4)

pah_077 = np.copy(pah_removed_077)


wavelengths_integrand_077 = wavelength_pah_removed_077.reshape(-1,1)
data_integrand_077 = pah_077.reshape(-1,1)


#first, need to fit continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(wavelength_pah_removed_077, 2) == 7.2)[0][0]
temp_index_2 = np.where(np.round(wavelengths5, 2) == 8.9)[0][0]

temp_index_3 = np.where(np.round(wavelength_pah_removed_077, 2) == 7.0)[0][0]
temp_index_4 = np.where(np.round(wavelengths5, 2) == 9.1)[0][0]

#calculating the slope of the line to use

#preventing the value found from being on a line or something
pah_slope_1 = np.mean(pah_removed_077[temp_index_1 - 20:20+temp_index_1])

pah_slope_2 = np.mean(corrected_data5[temp_index_2 - 20:20+temp_index_2])+3

#value where the wavelengths change interval



pah_slope = (pah_slope_2 - pah_slope_1)/\
(wavelengths5[temp_index_2] - wavelength_pah_removed_077[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(pah_removed_077[temp_index_3:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(corrected_data5[temp_index_2:temp_index_4]))

pah_removed_3 = pah_slope*(wavelength_pah_removed_077[temp_index_1:overlap1[0]] - 
                           wavelength_pah_removed_077[temp_index_1]) + pah_slope_1

#getting the right index for wavelengths4 that corresponds to overlap2[0]

temp_index_5 = np.where(np.round(wavelengths4, 2) == np.round(wavelength_pah_removed_077[overlap2[0]], 2))[0][2]

pah_removed_4 = pah_slope*(wavelengths4[overlap1[1]:temp_index_5] - 
                           wavelength_pah_removed_077[temp_index_1]) + pah_slope_1
pah_removed_5 = pah_slope*(wavelengths5[overlap2[1]:temp_index_2] - 
                           wavelength_pah_removed_077[temp_index_1]) + pah_slope_1


#putting it all together
continuum2 = np.concatenate((pah_removed_1, pah_removed_3))
continuum2 = np.concatenate((continuum2, pah_removed_4))
continuum2 = np.concatenate((continuum2, pah_removed_5))
continuum2 = np.concatenate((continuum2, pah_removed_2)) + 10

temp_index_6 = np.where(np.round(wavelength_pah_removed_077, 2) == 9.1)[0][0]

wave_cont2 = wavelength_pah_removed_077[temp_index_3:temp_index_6]

pah_temp = pah_077[temp_index_3:temp_index_6]

continuum_removed_2 = pah_temp - continuum2 + 10

everything_removed_2 = rnf.emission_line_remover(continuum_removed_2, 15, 3)


#first, need to fit 11.2 continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(wavelength_pah_removed, 2) == 11.0)[0][0]
temp_index_2 = np.where(np.round(wavelengths7, 2) == 11.82)[0][0]
temp_index_3 = np.where(np.round(wavelength_pah_removed, 2) == 10.55)[0][0]
temp_index_4 = np.where(np.round(wavelengths7, 2) == 12.0)[0][0]

#calculating the slope of the line to use

#preventing the value found from being on a line or something
pah_slope_1 = np.mean(pah[temp_index_1 - 20:20+temp_index_1])

#3 takes into account the offset of corrected_data7 applied when it is added to pah
pah_slope_2 = np.mean(corrected_data7[temp_index_2 - 20:20+temp_index_2] + 3 + 10)

pah_slope = (pah_slope_2 - pah_slope_1)/\
(wavelengths7[temp_index_2] - wavelength_pah_removed[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(pah[temp_index_3:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(corrected_data7[temp_index_2:temp_index_4]))

pah_removed_3 = pah_slope*(wavelength_pah[temp_index_1:overlap[0]] - 
                           wavelength_pah[temp_index_1]) + pah_slope_1
pah_removed_4 = pah_slope*(wavelengths7[overlap[1]:temp_index_2] - 
                           wavelength_pah[temp_index_1]) + pah_slope_1




#putting it all together
pah_removed = np.concatenate((pah_removed_1, pah_removed_3))
pah_removed = np.concatenate((pah_removed, pah_removed_4))
pah_removed = np.concatenate((pah_removed, pah_removed_2))

#subtracting continuum for relevant regions only

temp_index_2 = np.where(wavelength_pah == wavelengths7[temp_index_2])[0][0]
temp_index_4 = np.where(wavelength_pah == wavelengths7[temp_index_4])[0][0]



wave_cont3 = wavelength_pah_removed[temp_index_3:temp_index_4]

continuum3 = np.copy(pah_removed) #+10 already taken into account when it is added to data

pah_temp = (pah[temp_index_3:temp_index_4])

continuum_removed_3 = pah_temp - continuum3

everything_removed_3 = rnf.emission_line_remover(continuum_removed_3, 15, 3)

#%%

pahoverlap_low = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelength_pah[0], 2))[0][0]
pahoverlap_high = np.where(np.round(orion_wavelengths_miri, 2) == np.round(wavelength_pah[-1], 2))[0][0]

#%%

plt.figure()
plt.plot(wavelengths1, lfilter(b, a, corrected_data1))
plt.plot(wavelengths2, lfilter(b, a, corrected_data2)+2, alpha=0.5)
plt.plot(wavelengths3, lfilter(b, a, corrected_data3)+5, alpha=0.5)
plt.plot(wavelengths4, lfilter(b, a, corrected_data4)+7, alpha=0.5)
plt.plot(wavelengths5, lfilter(b, a, corrected_data5)+4, alpha=0.5)
plt.plot(wavelengths6, lfilter(b, a, corrected_data6)+3, alpha=0.5)
plt.plot(wavelengths7, lfilter(b, a, corrected_data7)+3, alpha=0.5)
plt.plot(wavelength_pah, lfilter(b, a, pah)-10, alpha=0.5)
plt.ylim(0, 20)
plt.show()





#%%

#west

#%%

plt.figure()
plt.plot(wavelengths1, lfilter(b, a, corrected_data1_west))
plt.plot(wavelengths2, lfilter(b, a, corrected_data2_west)+1, alpha=0.5)
plt.plot(wavelengths3, lfilter(b, a, corrected_data3_west)+0, alpha=0.5)
plt.plot(wavelengths4, lfilter(b, a, corrected_data4_west)+4, alpha=0.5)
plt.plot(wavelengths5, lfilter(b, a, corrected_data5_west)+3, alpha=0.5)
plt.plot(wavelengths6, lfilter(b, a, corrected_data6_west)+0, alpha=0.5)
plt.plot(wavelengths7, lfilter(b, a, corrected_data7_west)-1, alpha=0.5)
#plt.plot(wavelength_pah, lfilter(b, a, pah_west)-20, alpha=0.5)
plt.ylim(-10, 20)
plt.show()

#%%

#combining channels

#array for overlap wavelengths

overlap_array_west = []

pah_removed_west, wavelength_pah_removed_west, overlap = rnf.flux_aligner2(
    wavelengths1_west, wavelengths2_west, corrected_data1_west, corrected_data2_west + 1)

overlap_array_west.append((wavelengths1_west[overlap[0]] + wavelengths2_west[overlap[1]])/2)

pah_removed_west, wavelength_pah_removed_west, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west, wavelengths3_west, pah_removed_west, corrected_data3_west - 0)

overlap_array_west.append((wavelength_pah_removed_west[overlap[0]] + wavelengths3_west[overlap[1]])/2)

pah_removed_west, wavelength_pah_removed_west, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west, wavelengths4_west, pah_removed_west, corrected_data4_west + 4)

overlap_array_west.append((wavelength_pah_removed_west[overlap[0]] + wavelengths4_west[overlap[1]])/2)

pah_removed_west, wavelength_pah_removed_west, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west, wavelengths5_west, pah_removed_west, corrected_data5_west + 3)

overlap_array_west.append((wavelength_pah_removed_west[overlap[0]] + wavelengths5_west[overlap[1]])/2)

pah_removed_west, wavelength_pah_removed_west, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west, wavelengths6_west, pah_removed_west, corrected_data6_west - 0)

overlap_array_west.append((wavelength_pah_removed_west[overlap[0]] + wavelengths6_west[overlap[1]])/2)

pah_removed_west, wavelength_pah_removed_west, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west, wavelengths7_west, pah_removed_west, corrected_data7_west - 1)

overlap_array_west.append((wavelength_pah_removed_west[overlap[0]] + wavelengths7_west[overlap[1]])/2)

#the layout of this code is an illogical mess because of how modified it is from what it used to be

pah_west = np.copy(pah_removed_west) + 20
wavelength_pah_west = np.copy(wavelength_pah_removed_west)


#making continuum function and subtracting from data

#6.2

temp_index_1 = np.where(np.round(wavelength_pah_removed_west, 2) == 5.6)[0][0]
temp_index_2 = np.where(np.round(wavelength_pah_removed_west, 2) == 6.5)[0][0]

continuum1_west = 0*np.ones(len(pah_removed_west[temp_index_1:temp_index_2])) + 20
wave_cont1_west = wavelength_pah_removed_west[temp_index_1:temp_index_2]

pah_temp = pah_west[temp_index_1:temp_index_2]

continuum_removed_1_west = pah_temp - continuum1_west

everything_removed_1_west = rnf.emission_line_remover(continuum_removed_1_west, 20, 3)

#7.7, 8.6

temp_index_1 = np.where(np.round(wavelength_pah_removed_west, 2) == 7.1)[0][0]
temp_index_2 = np.where(np.round(wavelength_pah_removed_west, 2) == 8.9)[0][0]

continuum2_west = 4.0*np.ones(len(pah_removed_west[temp_index_1:temp_index_2])) + 20
wave_cont2_west = wavelength_pah_removed_west[temp_index_1:temp_index_2]

pah_temp = pah_west[temp_index_1:temp_index_2]

continuum_removed_2_west = pah_temp - continuum2_west

everything_removed_2_west = rnf.emission_line_remover(continuum_removed_2_west, 20, 3)

#11.2

#first, need to fit 11.2 continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(wavelength_pah_removed_west, 2) == 11.13)[0][0] #avoids an absorbtion feature
temp_index_2 = np.where(np.round(wavelengths7_west, 2) == 11.85)[0][0]

temp_index_3 = np.where(np.round(wavelength_pah_removed_west, 2) == 10.6)[0][0] #avoids an absorbtion feature
temp_index_4 = np.where(np.round(wavelengths7_west, 2) == 12.0)[0][0]

temp_index_0 = np.where(np.round(wavelength_pah_removed_west, 2) == 10.6)[0][0]

#calculating the slope of the line to use

#removing emission lines that can get in the way
pah_slope_1 = np.mean(rnf.emission_line_remover(pah_west[temp_index_0:temp_index_1], 15, 3))

#preventing the value found from being on a line or something
pah_slope_2 = np.mean(corrected_data7_west[temp_index_2 - 20:20+temp_index_2] - 1 + 20)

pah_slope = (pah_slope_2 - pah_slope_1)/\
    (wavelengths7_west[temp_index_2] - wavelength_pah_removed_west[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(pah_removed_west[temp_index_3:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(corrected_data7_west[temp_index_2:temp_index_4]))

pah_removed_3 = pah_slope*(wavelength_pah_west[temp_index_1:overlap[0]] - 
                           wavelength_pah_west[temp_index_1]) + pah_slope_1
pah_removed_4 = pah_slope*(wavelengths7_west[overlap[1]:temp_index_2] - 
                           wavelength_pah_west[temp_index_1]) + pah_slope_1
    





#putting it all together
pah_removed_west = np.concatenate((pah_removed_1, pah_removed_3))
pah_removed_west = np.concatenate((pah_removed_west, pah_removed_4))
pah_removed_west = np.concatenate((pah_removed_west, pah_removed_2))

#subtracting continuum for relevant regions only

temp_index_2 = np.where(wavelength_pah_west == wavelengths7_west[temp_index_2])[0][0]
temp_index_4 = np.where(wavelength_pah_west == wavelengths7_west[temp_index_4])[0][0]



wave_cont3_west = wavelength_pah_removed_west[temp_index_3:temp_index_4]

continuum3_west = np.copy(pah_removed_west) #+20 already taken into account when it is added to data

pah_temp = pah_west[temp_index_3:temp_index_4]

continuum_removed_3_west = pah_temp - continuum3_west

everything_removed_3_west = rnf.emission_line_remover(continuum_removed_3_west, 10, 1)



#%%

plt.figure()
plt.plot(wavelengths1, lfilter(b, a, corrected_data1_west))
plt.plot(wavelengths2, lfilter(b, a, corrected_data2_west)+1, alpha=0.5)
plt.plot(wavelengths3, lfilter(b, a, corrected_data3_west)+0, alpha=0.5)
plt.plot(wavelengths4, lfilter(b, a, corrected_data4_west)+4, alpha=0.5)
plt.plot(wavelengths5, lfilter(b, a, corrected_data5_west)+3, alpha=0.5)
plt.plot(wavelengths6, lfilter(b, a, corrected_data6_west)+0, alpha=0.5)
plt.plot(wavelengths7, lfilter(b, a, corrected_data7_west)-1, alpha=0.5)
plt.plot(wavelength_pah, lfilter(b, a, pah_west)-20, alpha=0.5)
plt.ylim(-10, 20)
plt.show()


#%%

#H2 region

#%%

plt.figure()
plt.plot(wavelengths1, lfilter(b, a, corrected_data1_west_blob))
plt.plot(wavelengths2, lfilter(b, a, corrected_data2_west_blob)+0, alpha=0.5)
plt.plot(wavelengths3, lfilter(b, a, corrected_data3_west_blob)-2, alpha=0.5)
plt.plot(wavelengths4, lfilter(b, a, corrected_data4_west_blob)+2, alpha=0.5)
plt.plot(wavelengths5, lfilter(b, a, corrected_data5_west_blob)+1, alpha=0.5)
plt.plot(wavelengths6, lfilter(b, a, corrected_data6_west_blob)-2, alpha=0.5)
plt.plot(wavelengths7, lfilter(b, a, corrected_data7_west_blob)-5, alpha=0.5)
#plt.plot(wavelength_pah, lfilter(b, a, pah_west)-10, alpha=0.5)
plt.ylim(-10, 20)
plt.show()

#%%

#array for overlap wavelengths

overlap_array_west_blob = []

pah_removed_west_blob, wavelength_pah_removed_west_blob, overlap = rnf.flux_aligner2(
    wavelengths1_west, wavelengths2_west, corrected_data1_west_blob, corrected_data2_west_blob + 0)

overlap_array_west_blob.append((wavelengths1_west[overlap[0]] + wavelengths2_west[overlap[1]])/2)

pah_removed_west_blob, wavelength_pah_removed_west_blob, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west_blob, wavelengths3_west, 
    pah_removed_west_blob, corrected_data3_west_blob - 2)

overlap_array_west_blob.append((wavelength_pah_removed_west_blob[overlap[0]] + wavelengths3_west[overlap[1]])/2)

pah_removed_west_blob, wavelength_pah_removed_west_blob, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west_blob, wavelengths4_west, 
    pah_removed_west_blob, corrected_data4_west_blob + 2)

overlap_array_west_blob.append((wavelength_pah_removed_west_blob[overlap[0]] + wavelengths4_west[overlap[1]])/2)

pah_removed_west_blob, wavelength_pah_removed_west_blob, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west_blob, wavelengths5_west, 
    pah_removed_west_blob, corrected_data5_west_blob + 1)

overlap_array_west_blob.append((wavelength_pah_removed_west_blob[overlap[0]] + wavelengths5_west[overlap[1]])/2)

pah_removed_west_blob, wavelength_pah_removed_west_blob, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west_blob, wavelengths6_west, 
    pah_removed_west_blob, corrected_data6_west_blob - 2)

overlap_array_west_blob.append((wavelength_pah_removed_west_blob[overlap[0]] + wavelengths6_west[overlap[1]])/2)

pah_removed_west_blob, wavelength_pah_removed_west_blob, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west_blob, wavelengths7_west, 
    pah_removed_west_blob, corrected_data7_west_blob - 5)

overlap_array_west_blob.append((wavelength_pah_removed_west_blob[overlap[0]] + wavelengths7_west[overlap[1]])/2)

#the layout of this code is an illogical mess because of how modified it is from what it used to be

pah_west_blob = np.copy(pah_removed_west_blob) + 20
wavelength_pah_west_blob = np.copy(wavelength_pah_removed_west_blob)



#making continuum function and subtracting from data

#6.2

temp_index_1 = np.where(np.round(wavelength_pah_removed_west_blob, 2) == 5.6)[0][0]
temp_index_2 = np.where(np.round(wavelength_pah_removed_west_blob, 2) == 6.5)[0][0]

continuum1_west_blob = 0*np.ones(len(pah_removed_west_blob[temp_index_1:temp_index_2])) + 20
wave_cont1_west_blob = wavelength_pah_removed_west_blob[temp_index_1:temp_index_2]

pah_temp = pah_west_blob[temp_index_1:temp_index_2]

continuum_removed_1_west_blob = pah_temp - continuum1_west_blob

everything_removed_1_west_blob = rnf.emission_line_remover(continuum_removed_1_west_blob, 20, 3)

#7.7, 8.6

temp_index_1 = np.where(np.round(wavelength_pah_removed_west_blob, 2) == 7.1)[0][0]
temp_index_2 = np.where(np.round(wavelength_pah_removed_west_blob, 2) == 8.9)[0][0]

continuum2_west_blob = -2.0*np.ones(len(pah_removed_west_blob[temp_index_1:temp_index_2])) + 20
wave_cont2_west_blob = wavelength_pah_removed_west_blob[temp_index_1:temp_index_2]

pah_temp = pah_west_blob[temp_index_1:temp_index_2]

continuum_removed_2_west_blob = pah_temp - continuum2_west_blob

everything_removed_2_west_blob = rnf.emission_line_remover(continuum_removed_2_west_blob, 20, 3)

#11.2

#first, need to fit continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(wavelength_pah_removed_west_blob, 2) == 11.13)[0][0]
temp_index_2 = np.where(np.round(wavelengths7_west, 2) == 11.85)[0][0]

temp_index_3 = np.where(np.round(wavelength_pah_removed_west_blob, 2) == 10.6)[0][0]
temp_index_4 = np.where(np.round(wavelengths7_west, 2) == 12.0)[0][0]

temp_index_0 = np.where(np.round(wavelength_pah_west_blob, 2) == 10.5)[0][0]

#calculating the slope of the line to use

#removing emission lines that can get in the way
pah_slope_1 = np.mean(rnf.emission_line_remover(pah_west_blob[temp_index_0:temp_index_1], 15, 3))

#preventing the value found from being on a line or something
pah_slope_2 = np.mean(corrected_data7_west_blob[temp_index_2 - 20:20+temp_index_2]) - 5 + 20

pah_slope = (pah_slope_2 - pah_slope_1)/\
    (wavelengths7_west[temp_index_2] - wavelength_pah_west_blob[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(pah_west_blob[temp_index_3:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(corrected_data7_west_blob[temp_index_2:temp_index_4]))

pah_removed_3 = pah_slope*(wavelength_pah_west_blob[temp_index_1:overlap[0]] - 
                           wavelength_pah_west_blob[temp_index_1]) + pah_slope_1
pah_removed_4 = pah_slope*(wavelengths7_west[overlap[1]:temp_index_2] - 
                           wavelength_pah_west_blob[temp_index_1]) + pah_slope_1




#putting it all together
pah_removed_west_blob = np.concatenate((pah_removed_1, pah_removed_3))
pah_removed_west_blob = np.concatenate((pah_removed_west_blob, pah_removed_4))
pah_removed_west_blob = np.concatenate((pah_removed_west_blob, pah_removed_2))


temp_index_2 = np.where(wavelength_pah_west_blob == wavelengths7_west[temp_index_2])[0][0]
temp_index_4 = np.where(wavelength_pah_west_blob == wavelengths7_west[temp_index_4])[0][0]



wave_cont3_west_blob = wavelength_pah_west_blob[temp_index_3:temp_index_4]

continuum3_west_blob = np.copy(pah_removed_west_blob) #+20 already taken into account when it is added to data

pah_temp = pah_west_blob[temp_index_3:temp_index_4]

continuum_removed_3_west_blob = pah_temp - continuum3_west_blob

everything_removed_3_west_blob = rnf.emission_line_remover(continuum_removed_3_west_blob, 10, 1)



#%%

plt.figure()
plt.plot(wavelengths1, lfilter(b, a, corrected_data1_west_blob))
plt.plot(wavelengths2, lfilter(b, a, corrected_data2_west_blob)+0, alpha=0.5)
plt.plot(wavelengths3, lfilter(b, a, corrected_data3_west_blob)-2, alpha=0.5)
plt.plot(wavelengths4, lfilter(b, a, corrected_data4_west_blob)+2, alpha=0.5)
plt.plot(wavelengths5, lfilter(b, a, corrected_data5_west_blob)+1, alpha=0.5)
plt.plot(wavelengths6, lfilter(b, a, corrected_data6_west_blob)-2, alpha=0.5)
plt.plot(wavelengths7, lfilter(b, a, corrected_data7_west_blob)-5, alpha=0.5)
plt.plot(wavelength_pah, lfilter(b, a, pah_west_blob)-20, alpha=0.5)
plt.ylim(-10, 20)
plt.show()



####################################



'''
FITTING GAUSSIAN TO 3.3
'''



#%%

#gaussian function to fit

#note: this is NOT a normalized gaussian

def gaussian(x, mean, fwhm, a, b):
    
    var = fwhm/(2*(2*np.log(2))**0.5)
    return a*np.exp(-1*((x - mean)**2)/(2*var**2)) + b



#nirspec plot

#mean im working with is 3.29. Can add 3.25 and 3.33 if needed.

#starting fwhm: (note variance is fwhm/2)
    #3.29 is 0.0387
    #3.25 is 0.0375
    #3.33 is 0.0264


nirspec_cutoff = np.where(np.round(wavelengths_nirspec4, 2) == 4)[0][0]

pahoverlap_nirspec4_1 = np.where(np.round(orion_wavelengths_nirspec, 2) == np.round(wavelengths_nirspec4[20], 2))[0][0]
pahoverlap_nirspec4_2 = np.where(np.round(orion_wavelengths_nirspec, 2) == np.round(wavelengths_nirspec4[nirspec_cutoff], 2))[0][0]



#%%

#RNF_paper_033_gaussian_fit_North

#%%
#old stuff: mean=3.29, fwhm=0.015556349, scale=2.1; mean=3.25, fwhm=0.015556349, scale=0.6; mean=3.33, fwhm=0.008485281, scale=0.4'
ax = plt.figure('RNF_paper_033_gaussian_fit_North', figsize=(16,6)).add_subplot(111)
#plt.title('NIRSPEC Weighted Mean, gaussian fit, North', fontsize=20)
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], nirspec_weighted_mean4[:nirspec_cutoff] - 2.4, 
         label='g395m-f290, North, offset=-2.4', color='#000000')

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.29027, 0.0387, 2.15, 0), 
         label ='gaussian fit mean=3.29027, fwhm=0.0387, scale=2.15')
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.2465, 0.0375, 0.6, 0), 
         label ='gaussian fit mean=3.2465, fwhm=0.0375, scale=0.6')
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.32821, 0.0264, 0.35, 0), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.35')
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.29027, 0.0375, 2.15, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.2465, 0.0375, 0.6, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.32821, 0.0264, 0.35, 0), 
         label='gaussian fit sum')

plt.plot(orion_wavelengths_nirspec[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.38*continuum_removed_orion_nirspec[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         label='ISO orion spectra, scale=0.38', color='r', alpha=0.5)
plt.ylim((-2,4))
ax.tick_params(axis='x', which='both', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', right='True')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(2.8, 4.0, 0.1), fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=14)
plt.savefig('Figures/RNF_paper_033_gaussian_fit_North.png')
plt.show() 

#%%

import astropy.units as u

def correct_units_astropy(cube, wavelengths):
        """
        Corrects the units of cubes by changing them from MJy/sr to W/m^2/um/str (with astropy)
        
        Parameters
        ----------
        cube [str or subtract_fits]: the cube(1d array now) whose units need to be corrected (in MJy/sr)
        
        directory_cube_data [str]: the directory of a .fits spectral cube file (in MJy/sr)
    
        directory_wave [str]: the directory of the file with the wavelengths (in micrometers)
    
        function_wave [function]: the function used to read in the wavelengths
                                  get_data: for .xdr files
                                  get_wavlengths: for .tbl files
                                  get_wave_fits: for .fits files
    
        Returns
        -------
        np.array : an array in W/m^2/sr/um
    
        """
        final_cube = np.zeros(cube.shape)
        cube_with_units = (cube*10**6)*(u.Jy/u.sr)
        print('cube with units')
        for i in range(len(wavelengths)):
            final_cube[i] = cube_with_units[i].to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavelengths[i]*u.micron))
        return(final_cube)


#%%

#############

#integrating flux of 3.3 feature

#freq = c/lambda, MJy units MW m^-2 Hz^-1

#final integrated product units MW m^-2 = MW m^-2 Hz^-1 Hz = c MW m^-2 m^-1 m = c MW m^-2 micron^-1 micron

#so multiply by c

#convert to SI units so final integrated units are W/m^2 (also per sr)

#integrate from 3.2 to 3.35 microns

l_int = np.where(np.round(wavelengths_nirspec4, 3) == 3.2)[0][0]
u_int = np.where(np.round(wavelengths_nirspec4, 2) == 3.35)[-1][-1]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand = gaussian(wavelengths_nirspec4[l_int:u_int], 3.29027, 0.0387, 2.15, 0) +\
gaussian(wavelengths_nirspec4[l_int:u_int], 3.2465, 0.0375, 0.6, 0) +\
gaussian(wavelengths_nirspec4[l_int:u_int], 3.32821, 0.0264, 0.35, 0)

#integrand = integrand*(u.MJy/u.sr)
wavelengths_integrand = wavelengths_nirspec4[l_int:u_int]#*(u.micron)


final_cube = np.zeros(integrand.shape)
cube_with_units = (integrand*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies =\
                                u.spectral_density(wavelengths_integrand*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_temp = np.copy(integrand)
for i in range(len(integrand)):
    integrand_temp[i] = float(final_cube[i])

bib, bibb = wavelengths_integrand, final_cube

odd_sum = 0

for i in range(1, len(integrand_temp), 2):
    odd_sum += integrand_temp[i] 

even_sum = 0    

for i in range(2, len(integrand_temp), 2):
    even_sum += integrand_temp[i] 

#stepsize, converted to frequency

h = wavelengths_integrand[1] - wavelengths_integrand[0]

integral033 = (h/3)*(integrand_temp[0] + integrand_temp[-1] + 4*odd_sum + 2*even_sum)

#%%

plt.figure()
plt.plot(bib, bibb)
#plt.plot(iso_wavelengths_062, iso_data_062)
#plt.plot(iso_wavelengths_062, iso_continuum)
plt.show()

#%%



####################################



'''
FITTING GAUSSIAN TO 11.2
'''

#renaming variables 

wavelength_pah_removed_112 = np.copy(wave_cont3)

continuum_removed6 = np.copy(continuum_removed_3)

everything_removed6 = np.copy(everything_removed_3)


#remaking these variables with new bounds (the others only go down to 10 microns)

ngc7027_index_begin = np.where(np.round(ngc7027_wavelengths, 2) == np.round(wavelengths5[0], 2))[0][0]
ngc7027_index_end = np.where(np.round(ngc7027_wavelengths, 2) == np.round(wavelengths5[-1], 2))[0][0]
ngc7027_index_end2 = np.where(np.round(ngc7027_wavelengths, 2) == np.round(wavelengths7[-1], 2))[0][0]



#%%

#RNF_112_continuum_extended_North_simple


#%%

#RNF_112_for_mikako


#%%

#integrating flux of 11.2 feature

#integrate from 11.0 to 11.6 microns

l_int = np.where(np.round(wavelength_pah_removed_112, 3) == 11.0)[0][0]
u_int = np.where(np.round(wavelength_pah_removed_112, 2) == 11.6)[-1][-1]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_112 = np.copy(everything_removed6[l_int:u_int])

#integrand = integrand*(u.MJy/u.sr)
wavelengths_integrand = wavelength_pah_removed_112[l_int:u_int]#*(u.micron)


final_cube = np.zeros(integrand_112.shape)
cube_with_units = (integrand_112*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies =\
                                u.spectral_density(wavelengths_integrand*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_temp_112 = np.copy(integrand_112)
for i in range(len(integrand_112)):
    integrand_temp_112[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_temp_112), 2):
    odd_sum += integrand_temp_112[i] 

even_sum = 0    

for i in range(2, len(integrand_temp_112), 2):
    even_sum += integrand_temp_112[i] 

#stepsize, converted to frequency

h = wavelengths_integrand[1] - wavelengths_integrand[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral112 = (h/3)*(integrand_temp_112[0] + integrand_temp_112[-1] + 4*odd_sum + 2*even_sum)



####################################





#%%

#west area

#3.3



#%%

#RNF_paper_033_gaussian_fit_west

#%%

ax = plt.figure('RNF_paper_033_gaussian_fit_west', figsize=(16,6)).add_subplot(111)
#plt.title('NIRSPEC Weighted Mean, gaussian fit, West', fontsize=20)
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], nirspec_weighted_mean4_west[:nirspec_cutoff] - 1.2, 
         label='g395m-f290, West, offset=-1.2', color='#000000')
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.29027, 0.0387, 1.1, 0), 
         label ='gaussian fit mean=3.29027, fwhm=0.0387, scale=1.1')
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.2465, 0.0375, 0.1, 0), 
         label ='gaussian fit mean=3.2465, fwhm=0.0375 scale=0.1')
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.32821, 0.0264, 0.05, 0), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.05')
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.405, 0.01, 0.3, 0), 
         label ='gaussian fit mean=3.405, fwhm=0.01, scale=0.3')
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.29027, 0.0387, 1.1, 0) +\
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.2465, 0.0375, 0.1, 0) +\
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.32821, 0.0264, 0.05, 0), 
         label='gaussian fit sum')
plt.plot(orion_wavelengths_nirspec[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.17*continuum_removed_orion_nirspec[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         label='ISO orion spectra, scale=0.17', color='r', alpha=0.5)
plt.ylim((-1,2))
ax.tick_params(axis='x', which='both', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', right='True')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(2.8, 4.0, 0.1), fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=14)
plt.savefig('Figures/RNF_paper_033_gaussian_fit_west.png')
plt.show() 

#%%

#integrating flux of 3.3 feature

#freq = c/lambda, MJy units MW m^-2 Hz^-1

#final integrated product units MW m^-2 = MW m^-2 Hz^-1 Hz = c MW m^-2 m^-1 m = c MW m^-2 micron^-1 micron

#so multiply by c

#convert to SI units so final integrated units are W/m^2 (also per sr)

#integrate from 3.2 to 3.35 microns

l_int = np.where(np.round(wavelengths_nirspec4_west, 2) == 3.2)[0][0]
u_int = np.where(np.round(wavelengths_nirspec4_west, 2) == 3.35)[-1][-1]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand = gaussian(wavelengths_nirspec4[l_int:u_int], 3.29027, 0.0387, 1.1, 0) +\
gaussian(wavelengths_nirspec4[l_int:u_int], 3.2465, 0.0375, 0.1, 0) +\
gaussian(wavelengths_nirspec4[l_int:u_int], 3.32821, 0.0264, 0.05, 0)

#integrand = integrand*(u.MJy/u.sr)
wavepog = wavelengths_nirspec4[l_int:u_int]#*(u.micron)


final_cube = np.zeros(integrand.shape)
cube_with_units = (integrand*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog = np.copy(integrand)
for i in range(len(integrand)):
    integrand_pog[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog), 2):
    odd_sum += integrand_pog[i] 

even_sum = 0    

for i in range(2, len(integrand_pog), 2):
    even_sum += integrand_pog[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral033_west = (h/3)*(integrand_pog[0] + integrand_pog[-1] + 4*odd_sum + 2*even_sum)



#%%
#renaming variables 

wavelength_pah_removed_112_west = np.copy(wave_cont3_west)

continuum_removed6_west = np.copy(continuum_removed_3_west)

everything_removed6_west = np.copy(everything_removed_3_west)

#%%

#############

#integrating flux of 11.2 feature



#integrate from 11.0 to 11.6 microns

l_int = np.where(np.round(wavelength_pah_removed_112_west, 3) == 11.13)[0][0]
u_int = np.where(np.round(wavelength_pah_removed_112_west, 2) == 11.6)[-1][-1]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_112_west = np.copy(everything_removed6_west[l_int:u_int])

#integrand = integrand*(u.MJy/u.sr)
wavepog = wavelength_pah_removed_112_west[l_int:u_int]#*(u.micron)


final_cube = np.zeros(integrand_112_west.shape)
cube_with_units = (integrand_112_west*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_112_west = np.copy(integrand_112_west)
for i in range(len(integrand_112_west)):
    integrand_pog_112_west[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_112_west), 2):
    odd_sum += integrand_pog_112_west[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_112_west), 2):
    even_sum += integrand_pog_112_west[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral112_west = (h/3)*(integrand_pog_112_west[0] + integrand_pog_112_west[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavepog) + 2 
#add 2 at the end to include endpoints

new_wavelengths_112_west = np.linspace(l_int, u_int, int(number_of_wavelengths/2))


#%%

#west area, blob

#3.3


#%%

#RNF_paper_033_gaussian_fit_west_blob

#%%

ax = plt.figure('RNF_paper_033_gaussian_fit_west_blob', figsize=(16,6)).add_subplot(111)
#plt.title('NIRSPEC Weighted Mean, gaussian fit, West Blob', fontsize=20)
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], nirspec_weighted_mean4_west_blob[:nirspec_cutoff] - 1.05, 
         label='g395m-f290, West, offset=-1.2', color='#000000')
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.29027, 0.0387, 1.1, 0), 
         label ='gaussian fit mean=3.29027, fwhm=0.0387, scale=1.2')
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.2465, 0.0375, 0.05, 0), 
         label ='gaussian fit 3.2465, fwhm=0.0375, scale=0.05')
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.32821, 0.0264, 0.2, 0), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.2')

plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.29027, 0.0387, 1.1, 0) +\
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.2465, 0.0375, 0.05, 0) +\
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.32821, 0.0264, 0.2, 0), 
         label='gaussian fit sum')
plt.plot(orion_wavelengths_nirspec[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.18*continuum_removed_orion_nirspec[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         label='ISO orion spectra, scale=0.18', color='r', alpha=0.5)
plt.ylim((-1,2))
ax.tick_params(axis='x', which='both', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', right='True')
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(2.8, 4.0, 0.1), fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=14)
plt.savefig('Figures/RNF_paper_033_gaussian_fit_west_blob.png')
plt.show() 

#%%

#############

#integrating flux of 3.3 feature

#convert to SI units so final integrated units are W/m^2 (also per sr)

#integrate from 3.2 to 3.35 microns

l_int = np.where(np.round(wavelengths_nirspec4_west, 2) == 3.2)[0][0]
u_int = np.where(np.round(wavelengths_nirspec4_west, 2) == 3.35)[-1][-1]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_blob = gaussian(wavelengths_nirspec4[l_int:u_int], 3.29027, 0.0387, 1.1, 0) +\
gaussian(wavelengths_nirspec4[l_int:u_int],  3.2465, 0.0375, 0.05, 0) +\
gaussian(wavelengths_nirspec4[l_int:u_int], 3.32821, 0.0264, 0.2, 0)

#integrand = integrand*(u.MJy/u.sr)
wavepog = wavelengths_nirspec4[l_int:u_int]#*(u.micron)


final_cube = np.zeros(integrand_blob.shape)
cube_with_units = (integrand_blob*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_blob = np.copy(integrand_blob)
for i in range(len(integrand_blob)):
    integrand_pog_blob[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_blob), 2):
    odd_sum += integrand_pog_blob[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_blob), 2):
    even_sum += integrand_pog_blob[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral033_west_blob = (h/3)*(integrand_pog_blob[0] + integrand_pog_blob[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavelengths_nirspec4[l_int:u_int]) + 2 
#add 2 at the end to include endpoints

new_wavelengths = np.linspace(l_int, u_int, int(number_of_wavelengths/2))

integrand2 = gaussian(new_wavelengths, 3.29, 0.0220/2, 1.8, 0) +\
gaussian(new_wavelengths, 3.25, 0.0220/2, 0.5, 0) +\
gaussian(new_wavelengths, 3.33, 0.0120/2, 0.2, 0)



final_cube = np.zeros(integrand2.shape)
cube_with_units = (integrand2*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(new_wavelengths*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog2 = np.copy(integrand2)
for i in range(len(integrand_pog2)):
    integrand_pog2[i] = float(final_cube[i])

odd_sum2 = 0

for i in range(1, len(integrand_pog2), 2):
    odd_sum2 += integrand_pog2[i] 

even_sum2 = 0    

for i in range(2, len(integrand_pog2), 2):
    even_sum2 += integrand_pog2[i] 

#stepsize

h2 = (new_wavelengths[1] - new_wavelengths[0])

integral2_west_blob = (h2/3)*(integrand_pog2[0] + integrand_pog2[-1] + 4*odd_sum2 + 2*even_sum2)

integral_error033_west_blob = (integral033_west_blob - integral2_west_blob)/15


#%%
#11.2

#renaming variables 

wavelength_pah_removed_112_west_blob = np.copy(wave_cont3_west_blob)

continuum_removed6_west_blob = np.copy(continuum_removed_3_west_blob)

everything_removed6_west_blob = np.copy(everything_removed_3_west_blob)

#%%

#############

#integrating flux of 11.2 feature



#integrate from 11.0 to 11.6 microns

l_int = np.where(np.round(wavelength_pah_removed_112_west, 3) == 11.13)[0][0]
u_int = np.where(np.round(wavelength_pah_removed_112_west, 2) == 11.6)[-1][-1]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_112_west_blob = np.copy(everything_removed6_west_blob[l_int:u_int])

#integrand = integrand*(u.MJy/u.sr)
wavepog = wavelength_pah_removed_112_west[l_int:u_int]#*(u.micron)


final_cube = np.zeros(integrand_112_west_blob.shape)
cube_with_units = (integrand_112_west_blob*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_112_west_blob = np.copy(integrand_112_west_blob)
for i in range(len(integrand_112_west_blob)):
    integrand_pog_112_west_blob[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_112_west_blob), 2):
    odd_sum += integrand_pog_112_west_blob[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_112_west_blob), 2):
    even_sum += integrand_pog_112_west_blob[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral112_west_blob = (h/3)*(integrand_pog_112_west_blob[0] + integrand_pog_112_west_blob[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavepog) + 2 
#add 2 at the end to include endpoints

new_wavelengths_112_west_blob = np.linspace(l_int, u_int, int(number_of_wavelengths/2))






####################################



'''
SIGNAL TO NOISE RATIO
'''




#%%

#using 3.11 to 3.16 microns (148 - 176)
rms_data033 = rnf.unit_changer(nirspec_weighted_mean4[148:176] - 2.4, wavelengths_nirspec4[148:176])

rms033 = rnf.RMS(rms_data033) 

delta_wave033 = fits.getheader(get_pkg_data_filename('data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits'), 1)["CDELT3"]

snr033 = rnf.SNR(integral033, rms033, delta_wave033, len(nirspec_weighted_mean4[135:163]))

error033 = integral033/snr033

print('3.3 feature:', integral033, '+/-', error033, 'W/m^2/sr, rms range 3.11 to 3.16 microns')

#using 10.83 to 10.88 microns (623 - 662)
rms_data112 = rnf.unit_changer(everything_removed6[623:662], wavelengths6[623:662])

rms112 = rnf.RMS(rms_data112) 

delta_wave112 = fits.getheader(get_pkg_data_filename('data/north/ring_neb_north_ch2-long_s3d.fits'), 1)["CDELT3"]

snr112 = rnf.SNR(integral112, rms112, delta_wave112, len(everything_removed6[623:662]))

error112 = integral112/snr112

print('11.2 feature:', integral112, '+/-', error112,  'W/m^2/sr, rms range10.83 to 10.88 microns')





#using 3.11 to 3.16 microns (148 - 176)
rms_data033_west = rnf.unit_changer(nirspec_weighted_mean4_west[148:176] - 1.3, wavelengths_nirspec4_west[148:176])

rms033_west = rnf.RMS(rms_data033_west) 

delta_wave033_west = fits.getheader(get_pkg_data_filename('data/west/jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked.fits'), 1)["CDELT3"]

snr033_west = rnf.SNR(integral033_west, rms033_west, delta_wave033_west, len(nirspec_weighted_mean4_west[135:163]))

error033_west = integral033_west/snr033_west

print('3.3 feature, west:', integral033_west, '+/-', error033_west, 'W/m^2/sr, rms range 3.11 to 3.16 microns')

#using 10.83 to 10.88 microns (623 - 662)
rms_data112_west = rnf.unit_changer(everything_removed6_west[623:662], wavelengths6_west[623:662])

rms112_west = rnf.RMS(rms_data112_west) 

delta_wave112_west = fits.getheader(get_pkg_data_filename('data/west/ring_neb_west_ch2-long_s3d.fits'), 1)["CDELT3"]

snr112_west = rnf.SNR(integral112_west, rms112_west, delta_wave112_west, len(everything_removed6_west[623:662]))

error112_west = integral112_west/snr112_west

print('11.2 feature, west:', integral112_west, '+/-', error112_west,  'W/m^2/sr, rms range10.83 to 10.88 microns')


#blob in west region

#using 3.11 to 3.16 microns (148 - 176)
rms_data033_west_blob = rnf.unit_changer(nirspec_weighted_mean4_west_blob[148:176] - 1.3, wavelengths_nirspec4_west[148:176])

rms033_west_blob = rnf.RMS(rms_data033_west_blob) 

delta_wave033_west_blob = fits.getheader(get_pkg_data_filename('data/west/jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked.fits'), 1)["CDELT3"]

snr033_west_blob = rnf.SNR(integral033_west_blob, rms033_west_blob, delta_wave033_west_blob, len(nirspec_weighted_mean4_west_blob[135:163]))

error033_west_blob = integral033_west_blob/snr033_west_blob

print('3.3 feature, west blob:', integral033_west_blob, '+/-', error033_west_blob, 'W/m^2/sr, rms range 3.11 to 3.16 microns')

#using 10.83 to 10.88 microns (623 - 662)
rms_data112_west_blob = rnf.unit_changer(everything_removed6_west_blob[623:662], wavelengths6_west[623:662])

rms112_west_blob = rnf.RMS(rms_data112_west_blob) 

delta_wave112_west_blob = fits.getheader(get_pkg_data_filename('data/west/ring_neb_west_ch2-long_s3d.fits'), 1)["CDELT3"]

snr112_west_blob = rnf.SNR(integral112_west_blob, rms112_west_blob, delta_wave112_west_blob, len(everything_removed6_west_blob[623:662]))

error112_west_blob = integral112_west_blob/snr112_west_blob

print('11.2 feature, west blob:', integral112_west_blob, '+/-', error112_west_blob,  'W/m^2/sr, rms range10.83 to 10.88 microns')



####################################





#%%

#RNF_paper_continuum_extended_simple

#%%

ax = plt.figure('RNF_paper_continuum_extended_simple', figsize=(18,9)).add_subplot(311)
plt.subplots_adjust(right=0.8, left=0.1)
'''
North
'''

#plt.title('JWST Continuum Subtracted Data, North, simple', fontsize=20)

plt.plot(wavelength_pah, pah, label='data, offset=+10')

plt.plot(wave_cont1, everything_removed_1, color='#dc267f', label='Lines and Continuum removed')
plt.plot(wave_cont2, everything_removed_2, color='#dc267f')
plt.plot(wave_cont3, everything_removed_3, color='#dc267f')



plt.plot(wave_cont1, continuum1, color='#785ef0', label='continuum, offset=+10')
plt.plot(wave_cont2, continuum2, color='#785ef0')
plt.plot(wave_cont3, continuum3, color='#785ef0')



plt.plot(wavelength_pah, 0*pah, color='black', label='zero')

plt.plot(orion_wavelengths_miri[pahoverlap_low:pahoverlap_high], 
         0.13*continuum_removed_orion[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='#785ef0', alpha=1.0)

plt.scatter(overlap_array, -5*np.ones(len(overlap_array)), zorder=100, color='black', label='data overlap')

plt.ylim((-10,45))
#ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
#ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11, title='North Common', bbox_to_anchor=(1.02, 1), loc='upper left')

'''
West
'''

ax = plt.subplot(312)
#plt.title('JWST Continuum Subtracted Data, West simple', fontsize=20)

plt.plot(wavelength_pah_west, pah_west, label='data, offset=+10')

plt.plot(wave_cont1_west, everything_removed_1_west, color='#dc267f', label='Lines and Continuum removed')
plt.plot(wave_cont2_west, everything_removed_2_west, color='#dc267f')
plt.plot(wave_cont3_west, everything_removed_3_west, color='#dc267f')



plt.plot(wave_cont1_west, continuum1_west, color='#785ef0', label='continuum, offset=+10')
plt.plot(wave_cont2_west, continuum2_west, color='#785ef0')
plt.plot(wave_cont3_west, continuum3_west, color='#785ef0')



plt.plot(wavelength_pah_west, 0*pah_west, color='black', label='zero')


plt.plot(orion_wavelengths_miri[pahoverlap_low:pahoverlap_high], 
         0.045*continuum_removed_orion[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='#785ef0', alpha=1.0)

plt.scatter(overlap_array_west, -5*np.ones(len(overlap_array_west)), zorder=100, color='black', label='data overlap')

plt.ylim((-10,45))
#ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
#ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
#plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11, title='West Common', bbox_to_anchor=(1.02, 1), loc='upper left')

'''
West H2 Filament
'''

ax = plt.subplot(313)
#plt.title('JWST Continuum Subtracted Data, West H2 Filament Simple', fontsize=20)

plt.plot(wavelength_pah_west_blob, pah_west_blob, label='data, offset=+20')

plt.plot(wave_cont1_west_blob, everything_removed_1_west_blob, color='#dc267f', label='Lines and Continuum removed')
plt.plot(wave_cont2_west_blob, everything_removed_2_west_blob, color='#dc267f')
plt.plot(wave_cont3_west_blob, everything_removed_3_west_blob, color='#dc267f')

plt.plot(wave_cont1_west_blob, continuum1_west_blob, color='#785ef0', label='continuum, offset=+20')
plt.plot(wave_cont2_west_blob, continuum2_west_blob, color='#785ef0')
plt.plot(wave_cont3_west_blob, continuum3_west_blob, color='#785ef0')

plt.plot(wavelength_pah_west_blob, 0*pah_west_blob, color='black', label='zero')

plt.plot(orion_wavelengths_miri[pahoverlap_low:pahoverlap_high], 
         0.06*continuum_removed_orion[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.06', color='#785ef0', alpha=1.0)

plt.scatter(overlap_array_west_blob, -5*np.ones(len(overlap_array_west_blob)), zorder=100, color='black', label='data overlap')

plt.ylim((-10,45))
#ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
#ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11, title='West H2 Filament', bbox_to_anchor=(1.02, 1), loc='upper left')



plt.savefig('Figures/RNF_paper_continuum_extended_simple.png')
plt.show()




#%%

#RNF_paper_continuum_extended_simple_no_legend

#%%

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

'''
6.2
'''




'''
ax = plt.figure('RNF_paper_continuum_extended_simple_no_legend', figsize=(18,9)).add_subplot(2,3,1)

plt.plot(wave_cont1, everything_removed_1+continuum1, color='#dc267f', label='Lines and Continuum removed')
plt.plot(wave_cont2, everything_removed_2+continuum2, color='#dc267f')
plt.plot(wave_cont3, everything_removed_3, color='#dc267f')

plt.plot(wavelength_pah, 0*pah, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.13*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='#000000', alpha=1.0)

for data in overlap_array:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

#plt.ylim((-1,15))
plt.ylim((14.0,30.0))
plt.xlim((5.6, 6.5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.6, 6.5, 0.3), fontsize=14)
plt.yticks(fontsize=14)
'''
'''
7.7
'''
'''
ax = plt.figure('RNF_paper_continuum_extended_simple_no_legend', figsize=(18,9)).add_subplot(2,3,2)

plt.plot(wave_cont1, everything_removed_1, color='#dc267f', label='Lines and Continuum removed')
plt.plot(wave_cont2, everything_removed_2+continuum2, color='#dc267f')
plt.plot(wave_cont3, everything_removed_3, color='#dc267f')

plt.plot(wavelength_pah, 0*pah, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.13*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='#000000', alpha=1.0)

for data in overlap_array:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

#plt.ylim((-1,15))
plt.ylim((14.0,30.0))
plt.xlim((7.1, 8.9))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(7.1, 8.9, 0.3), fontsize=14)
plt.yticks(fontsize=14)
'''
'''
11.2
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


'''
West
'''

'''
6.2
'''
'''
ax = plt.figure('RNF_paper_continuum_extended_simple_no_legend', figsize=(18,9)).add_subplot(2,3,4)

plt.plot(wave_cont1_west, everything_removed_1_west+continuum1_west, color='#dc267f', label='Lines and Continuum removed')
plt.plot(wave_cont2_west, everything_removed_2_west, color='#dc267f')
plt.plot(wave_cont3_west, everything_removed_3_west, color='#dc267f')

plt.plot(wavelength_pah_west, 0*pah_west, color='black', label='zero')


plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='#000000', alpha=1.0)

for data in overlap_array_west:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

#plt.ylim((-1,11))
plt.ylim((19.0,31.0))
plt.xlim((5.6, 6.5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.6, 6.5, 0.3), fontsize=14)
plt.yticks(fontsize=14)
'''
'''
7.7
'''
'''
ax = plt.figure('RNF_paper_continuum_extended_simple_no_legend', figsize=(18,9)).add_subplot(2,3,5)

plt.plot(wave_cont1_west, everything_removed_1_west, color='#dc267f', label='Lines and Continuum removed')
plt.plot(wave_cont2_west, everything_removed_2_west+continuum2_west, color='#dc267f')
plt.plot(wave_cont3_west, everything_removed_3_west, color='#dc267f')

plt.plot(wavelength_pah_west, 0*pah_west, color='black', label='zero')


plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='#000000', alpha=1.0)

for data in overlap_array_west:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

#plt.ylim((-1,11))
plt.ylim((19.0,31.0))
plt.xlim((7.1, 8.9))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom='on', top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom='on', top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(7.1, 8.9, 0.3), fontsize=14)
plt.yticks(fontsize=14)
'''
'''
11.2
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

plt.savefig('Figures/paper/RNF_paper_continuum_extended_simple_no_legend.pdf', bbox_inches='tight')
plt.show()




#%%

#RNF_paper_data_extended_simple_no_legend

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

plt.savefig('Figures/paper/RNF_paper_data_extended_simple_no_legend.pdf', bbox_inches='tight')
plt.show()







#######################################



#%%

#RNF_paper_033_gaussian_fit

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
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], nirspec_weighted_mean4[:nirspec_cutoff] - 2.4, 
         label='g395m-f290, North, offset=-2.4', color='#dc267f')

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.29027, 0.0387, 2.15, 0), 
         label ='gaussian fit mean=3.29027, fwhm=0.0387, scale=2.15', color='#648fff')
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.2465, 0.0375, 0.6, 0), 
         label ='gaussian fit mean=3.2465, fwhm=0.0375, scale=0.6', color='#648fff')
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.32821, 0.0264, 0.35, 0), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.35', color='#648fff')
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.29027, 0.0375, 2.15, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.2465, 0.0375, 0.6, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.32821, 0.0264, 0.35, 0), 
         label='gaussian fit sum', color='#648fff')

plt.plot(orion_wavelengths_nirspec[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         orion_north_scaling*continuum_removed_orion_nirspec[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         label='ISO orion spectra, scale=0.13', color='#000000', alpha=1.0)
plt.ylim((-0.5,4))
plt.xlim((3.1, 3.6))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'North', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.1, 3.6, 0.05), fontsize=28)
plt.yticks(np.arange(0.0, 4.5, 0.5), fontsize=28)
#plt.legend(fontsize=14)

'''
West
'''

ax = plt.figure('RNF_paper_033_gaussian_fit', figsize=(18,18)).add_subplot(212)
#ax = plt.figure('RNF_paper_033_gaussian_fit_west', figsize=(18,9)).add_subplot(111)
#plt.title('NIRSPEC Weighted Mean, gaussian fit, West', fontsize=20)
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], nirspec_weighted_mean4_west[:nirspec_cutoff] - 1.2, 
         label='g395m-f290, West, offset=-1.2', color='#dc267f')
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.29027, 0.0387, 1.1, 0), 
         label ='gaussian fit mean=3.29027, fwhm=0.0387, scale=1.1', color='#648fff')
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.2465, 0.0375, 0.1, 0), 
         label ='gaussian fit mean=3.2465, fwhm=0.0375 scale=0.1', color='#648fff')
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.32821, 0.0264, 0.05, 0), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.05', color='#648fff')
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.29027, 0.0387, 1.1, 0) +\
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.2465, 0.0375, 0.1, 0) +\
         gaussian(wavelengths_nirspec4_west[:nirspec_cutoff], 3.32821, 0.0264, 0.05, 0), 
         label='gaussian fit sum', color='#648fff')

plt.plot(orion_wavelengths_nirspec[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         orion_west_scaling*continuum_removed_orion_nirspec[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         label='ISO orion spectra, scale=0.045', color='#000000', alpha=1.0)
plt.ylim((-0.5,2))
plt.xlim((3.1, 3.6))

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
#plt.ylabel('Flux (MJy/sr)', fontsize=32)
plt.xticks(np.arange(3.1, 3.6, 0.05), fontsize=28)
plt.yticks(np.arange(0.0, 2.5, 0.5), fontsize=28)
#plt.legend(fontsize=14)

plt.savefig('Figures/paper/RNF_paper_033_gaussian_fit.pdf', bbox_inches='tight')
plt.show() 



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
plt.ylim((-2,15))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(10.5, 12., 0.25), fontsize=28)
plt.xlim(10.5,12)
plt.yticks(fontsize=28)
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
plt.ylim((-2,7))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(10.5, 12., 0.25), fontsize=28)
plt.xlim(10.5,12)
plt.yticks(fontsize=28)
#plt.legend(fontsize=11)

plt.savefig('Figures/paper/RNF_paper_112_comparison.pdf', bbox_inches='tight')
#plt.show()














#%%



#for smoothing data
from scipy.signal import lfilter

#smoothing data for easier comparison

everything_removed6_smooth = np.copy(everything_removed6)
everything_removed6_west_smooth = np.copy(everything_removed6_west)
everything_removed6_west_blob_smooth = np.copy(everything_removed6_west_blob)

n = 15  # the larger n is, the smoother curve will be
b = [1.0 / n] * n
a = 1


everything_removed6_smooth = lfilter(b, a, everything_removed6)
everything_removed6_west_smooth = lfilter(b, a, everything_removed6_west)
everything_removed6_west_blob_smooth = lfilter(b, a, everything_removed6_west_blob)



#%%

'''
North
'''

ax = plt.figure('RNF_paper_112_comparison_smooth', figsize=(18,12)).add_subplot(311)
#plt.title('JWST Continuum Subtracted Data, 11.2 feature, North', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
#plt.plot(wavelength_pah_removed_112, pah_112, label='data')
#plt.plot(wavelength_pah_removed_112, continuum_removed6, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112, everything_removed6_smooth, label='Lines and Continuum removed')
plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.42*continuum_removed_hh[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.42', color='r', alpha=0.5)

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
plt.ylim((-2,15))

ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(10.5, 12., 0.25), fontsize=14)
plt.xlim(10.5,12)
plt.yticks(fontsize=14)
#plt.legend(fontsize=11)


'''
West
'''

ax = plt.figure('RNF_paper_112_comparison_smooth', figsize=(16,6)).add_subplot(312)
#plt.title('JWST Continuum Subtracted Data, 11.2 feature, North', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
#plt.plot(wavelength_pah_removed_112, pah_112, label='data')
#plt.plot(wavelength_pah_removed_112, continuum_removed6, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112_west, everything_removed6_west_smooth, label='Lines and Continuum removed')
plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.16*continuum_removed_hh[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.18', color='r', alpha=0.5)

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
plt.ylim((-2,7))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(10.5, 12., 0.25), fontsize=14)
plt.xlim(10.5,12)
plt.yticks(fontsize=14)
#plt.legend(fontsize=11)


'''
West H2 Filament
'''

ax = plt.figure('RNF_paper_112_comparison_smooth', figsize=(16,6)).add_subplot(313)
#plt.title('JWST Continuum Subtracted Data, 11.2 feature, North', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
#plt.plot(wavelength_pah_removed_112, pah_112, label='data')
#plt.plot(wavelength_pah_removed_112, continuum_removed6, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112_west_blob, everything_removed6_west_blob_smooth, label='Lines and Continuum removed')
plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.16*continuum_removed_hh[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.18', color='r', alpha=0.5)

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
plt.ylim((-2,8))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(10.5, 12., 0.25), fontsize=14)
plt.xlim(10.5,12)
plt.yticks(fontsize=14)
#plt.legend(fontsize=11)
plt.savefig('Figures/paper/RNF_paper_112_comparison_smooth.pdf', bbox_inches='tight')
plt.show()

#%%

plt.figure()
plt.plot(wavelengths7, corrected_data7-8, label='north, offset=-8')
plt.plot(spitzer_wavelengths, 
         1*continuum_removed_spitzer, 
         label='Spitzer spectra, scale=1', color='black')
plt.xlim(11.5, 13.5)
plt.ylim(-2,6)
plt.legend()
plt.show()

#%%

plt.figure()
plt.plot(orion_wavelengths_nirspec, 
         1*continuum_removed_orion_nirspec, 
         label='ISO orion spectra, scale=0.15', color='r', alpha=1)
plt.xlim(2.5, 4.0)
plt.ylim(0, 10)
plt.show()

#%%

#array for overlap wavelengths


overlap_array_west = []

pah_removed_west, wavelength_pah_removed_west, overlap = rnf.flux_aligner2(
    wavelengths1_west, wavelengths2_west, corrected_data1_west, corrected_data2_west + 1)

overlap_array_west.append((wavelengths1_west[overlap[0]] + wavelengths2_west[overlap[1]])/2)

pah_removed_west, wavelength_pah_removed_west, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west, wavelengths3_west, pah_removed_west, corrected_data3_west - 0)

overlap_array_west.append((wavelength_pah_removed_west[overlap[0]] + wavelengths3_west[overlap[1]])/2)

pah_removed_west, wavelength_pah_removed_west, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west, wavelengths4_west, pah_removed_west, corrected_data4_west + 4)

overlap_array_west.append((wavelength_pah_removed_west[overlap[0]] + wavelengths4_west[overlap[1]])/2)

pah_removed_west, wavelength_pah_removed_west, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west, wavelengths5_west, pah_removed_west, corrected_data5_west + 3)

overlap_array_west.append((wavelength_pah_removed_west[overlap[0]] + wavelengths5_west[overlap[1]])/2)

pah_removed_west, wavelength_pah_removed_west, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west, wavelengths6_west, pah_removed_west, corrected_data6_west - 0)

overlap_array_west.append((wavelength_pah_removed_west[overlap[0]] + wavelengths6_west[overlap[1]])/2)

pah_removed_west, wavelength_pah_removed_west, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_west, wavelengths7_west, pah_removed_west, corrected_data7_west - 1)

overlap_array_west.append((wavelength_pah_removed_west[overlap[0]] + wavelengths7_west[overlap[1]])/2)

pah_west = np.copy(pah_removed_west) + 20
wavelength_pah_west = np.copy(wavelength_pah_removed_west)



#11.2 v2, west

#first, need to fit 11.2 continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(wavelength_pah_removed_west, 2) == 11.13)[0][0] #avoids an absorbtion feature
temp_index_2 = np.where(np.round(wavelengths7_west, 2) == 11.85)[0][0]

temp_index_3 = np.where(np.round(wavelength_pah_removed_west, 2) == 10.6)[0][0] #avoids an absorbtion feature
temp_index_4 = np.where(np.round(wavelengths7_west, 2) == 12.0)[0][0]

temp_index_0 = np.where(np.round(wavelength_pah_removed_west, 2) == 10.6)[0][0]

#calculating the slope of the line to use

#removing emission lines that can get in the way
pah_slope_1 = 1 + np.mean(rnf.emission_line_remover(pah_west[temp_index_0:temp_index_1], 15, 3))

#preventing the value found from being on a line or something
pah_slope_2 = np.mean(corrected_data7_west[temp_index_2 - 20:20+temp_index_2] - 1 + 20)

pah_slope = (pah_slope_2 - pah_slope_1)/\
    (wavelengths7_west[temp_index_2] - wavelength_pah_removed_west[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(pah_removed_west[temp_index_3:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(corrected_data7_west[temp_index_2:temp_index_4]))

pah_removed_3 = pah_slope*(wavelength_pah_west[temp_index_1:overlap[0]] - 
                           wavelength_pah_west[temp_index_1]) + pah_slope_1
pah_removed_4 = pah_slope*(wavelengths7_west[overlap[1]:temp_index_2] - 
                           wavelength_pah_west[temp_index_1]) + pah_slope_1
    





#putting it all together
pah_removed_west = np.concatenate((pah_removed_1, pah_removed_3))
pah_removed_west = np.concatenate((pah_removed_west, pah_removed_4))
pah_removed_west = np.concatenate((pah_removed_west, pah_removed_2))

#subtracting continuum for relevant regions only

temp_index_2 = np.where(wavelength_pah_west == wavelengths7_west[temp_index_2])[0][0]
temp_index_4 = np.where(wavelength_pah_west == wavelengths7_west[temp_index_4])[0][0]



wave_cont3_west_v2 = wavelength_pah_removed_west[temp_index_3:temp_index_4]

continuum3_west_v2 = np.copy(pah_removed_west) #+20 already taken into account when it is added to data

pah_temp = pah_west[temp_index_3:temp_index_4]

continuum_removed_3_west_v2 = pah_temp - continuum3_west_v2

everything_removed_3_west_v2 = rnf.emission_line_remover(continuum_removed_3_west_v2, 10, 1)



#%%

ax = plt.figure('RNF_paper_data_extended_simple_no_legend', figsize=(18,9)).add_subplot(111)
#plt.subplots_adjust(right=0.9, left=0.1)

ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
ax.tick_params(axis='y', which='major', labelleft=False, right=False)



# Hide X and Y axes tick marks


plt.ylabel('Flux (MJy/sr)', fontsize=16, labelpad=60)
plt.xlabel('Wavelength (micron)', fontsize=16, labelpad=30)



'''
West
'''


#plt.title('JWST Continuum Subtracted Data, West simple', fontsize=20)

plt.plot(wavelength_pah_west, pah_west-20, label='data')

#plt.plot(wave_cont1_west, everything_removed_1_west, color='#dc267f', label='Lines and Continuum removed')
#plt.plot(wave_cont2_west, everything_removed_2_west, color='#dc267f')
#plt.plot(wave_cont3_west, everything_removed_3_west, color='#dc267f')
#plt.plot(wave_nirspec_west , nirspec_no_line_west, color='#dc267f')



#plt.plot(wave_cont1_west, continuum1_west-20, color='#785ef0', label='continuum')
#plt.plot(wave_cont2_west, continuum2_west-20, color='#785ef0')
plt.plot(wave_cont3_west, continuum3_west-20, color='#000000', label='continuum')
plt.plot(wave_cont3_west_v2, continuum3_west_v2-20, color='#000000', label='continuum v2')


'''
plt.plot(wavelength_pah_west, 0*pah_west, color='black', label='zero')
'''

#plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
#         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
#         label='ISO orion spectra, scale=0.045', color='#000000', alpha=1.0)

for data in overlap_array_west:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')
#plt.scatter(overlap_array_west, -5*np.ones(len(overlap_array_west)), zorder=100, color='black', label='data overlap')

plt.ylim((-2.5, 7.0))
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
plt.xticks(np.arange(10.0, 13.5, 0.25), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(10, 13.5)
#plt.legend(fontsize=11, title='West Common', bbox_to_anchor=(1.02, 1), loc='upper left')

#plt.savefig('Figures/paper/RNF_paper_data_extended_simple_no_legend.pdf', bbox_inches='tight')
plt.show()

#%%

#############

#%%
#renaming variables 

wavelength_pah_removed_112_west_v2 = np.copy(wave_cont3_west_v2)

continuum_removed6_west_v2 = np.copy(continuum_removed_3_west_v2)

everything_removed6_west_v2 = np.copy(everything_removed_3_west_v2)

#integrating flux of 11.2 feature, v2 cont



#integrate from 11.0 to 11.6 microns

l_int = np.where(np.round(wavelength_pah_removed_112_west_v2, 3) == 11.13)[0][0]
u_int = np.where(np.round(wavelength_pah_removed_112_west_v2, 2) == 11.6)[-1][-1]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_112_west = np.copy(everything_removed6_west_v2[l_int:u_int])

#integrand = integrand*(u.MJy/u.sr)
wavepog = wavelength_pah_removed_112_west_v2[l_int:u_int]#*(u.micron)


final_cube = np.zeros(integrand_112_west.shape)
cube_with_units = (integrand_112_west*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_112_west = np.copy(integrand_112_west)
for i in range(len(integrand_112_west)):
    integrand_pog_112_west[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_112_west), 2):
    odd_sum += integrand_pog_112_west[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_112_west), 2):
    even_sum += integrand_pog_112_west[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral112_west = (h/3)*(integrand_pog_112_west[0] + integrand_pog_112_west[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavepog) + 2 
#add 2 at the end to include endpoints

new_wavelengths_112_west = np.linspace(l_int, u_int, int(number_of_wavelengths/2))

#%%

plt.figure()
plt.plot(wavelength_pah_removed_112, everything_removed6)
plt.plot(wavelength_pah_removed_112_west, everything_removed6_west)
plt.plot(wavelength_pah_removed_112_west_v2, everything_removed6_west_v2)
plt.show()

#%%
'''
image_file1 = get_pkg_data_filename('data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits')
continuum1 = fits.getheader(image_file1, ext=0)

image_file2 = get_pkg_data_filename('data/west/jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked.fits')
continuum2 = fits.getheader(image_file2, ext=0)

image_file3 = get_pkg_data_filename('data/north/ring_neb_north_ch1-short_s3d.fits')
continuum3 = fits.getheader(image_file3, ext=0)

image_file4 = get_pkg_data_filename('data/west/ring_neb_west_ch1-short_s3d.fits')
continuum4 = fits.getheader(image_file4, ext=0)


#%%

#loading in template PAH spectra
line_image_file = np.loadtxt('jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_spec_sum_converted.txt_fit', skiprows=1)
line_wavelengths = line_image_file[:,0]
line_data = line_image_file[:,1]
line_fit = line_image_file[:,2]
line_other = line_image_file[:,3]
line_cont = line_image_file[:,4]

#%%

#use line at 3.003 for scaling, is index 75 for line and 88 for nirspec

scaling = nirspec_weighted_mean4[88]/line_data[75]

plt.figure()
plt.plot(line_wavelengths, scaling*line_data)
#plt.plot(wavelengths_nirspec4, nirspec_weighted_mean4)
plt.plot(line_wavelengths, scaling*line_fit)
#plt.plot(line_wavelengths, scaling*line_other)
plt.plot(line_wavelengths, scaling*line_cont)
plt.xlim(3.1, 3.6)
plt.ylim(0,6)
plt.show()


'''
#%%
'''
import RingNebulaFunctions as rnf

#cam extraction comparison

nircam_file_loc = 'data/cams/nircam_color_F300M_F335M.fits'
nircam_image_file = get_pkg_data_filename(nircam_file_loc)
nircam_data = fits.getdata(nircam_image_file, ext=0)
nircam_data = nircam_data[np.newaxis, :, :]


miricam_file_loc = 'data/cams/miri_color_F1000W_F1130W.fits'
miricam_image_file = get_pkg_data_filename(miricam_file_loc)
miricam_data = fits.getdata(miricam_image_file, ext=0)
miricam_data = miricam_data[np.newaxis, :, :]

#not aligned, everything has same fov
nircam, temp = rnf.extract_spectra_from_regions_one_pointing_no_bkg_2d(
    'data/cams/nircam_color_F300M_F335M.fits', nircam_data, nircam_data, 
    np.array([0]), 'NIRSPEC_NORTH_bigbox_new.reg')
nircam = nircam['region_0']

nircam_west, temp = rnf.extract_spectra_from_regions_one_pointing_no_bkg_2d(
    'data/cams/nircam_color_F300M_F335M.fits', nircam_data, nircam_data, 
    np.array([0]), 'ring_west_common_region.reg')
nircam_west = nircam_west['region_0']

miricam, temp = rnf.extract_spectra_from_regions_one_pointing_no_bkg_2d(
    'data/cams/miri_color_F1000W_F1130W.fits', miricam_data, miricam_data, 
    np.array([0]), 'NIRSPEC_NORTH_bigbox_new.reg')
miricam = miricam['region_0']

miricam_west, temp = rnf.extract_spectra_from_regions_one_pointing_no_bkg_2d(
    'data/cams/miri_color_F1000W_F1130W.fits', miricam_data, miricam_data, 
    np.array([0]), 'ring_west_common_region.reg')
miricam_west = miricam_west['region_0']

print('north cam: ', nircam/miricam)
print('west cam: ', nircam_west/miricam_west)

#%%

print('north cam: ', nircam, miricam)
print('west cam: ', nircam_west, miricam_west)
'''


#%%

#loading in template PAH spectra
iso_sl1_image_file = np.loadtxt('data/misc/ngc6702_all_SL1_peak112_noheader.tbl', skiprows=3)
iso_sl2_image_file = np.loadtxt('data/misc/ngc6702_all_SL2_peak112_noheader.tbl', skiprows=3)
iso_sl1_wavelengths = iso_sl1_image_file[:,0]
iso_sl2_wavelengths = iso_sl2_image_file[:,0]
iso_sl1_data = iso_sl1_image_file[:,1]
iso_sl2_data = iso_sl2_image_file[:,1]

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

plt.ylim((0.0, 40.0))
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
plt.xticks(np.arange(5.5, 13.5, 1.0), fontsize=28)
plt.yticks(fontsize=28)
plt.xlim(5.5, 13.5)

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

plt.ylim((0.0, 30.0))
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
plt.xticks(np.arange(5.5, 13.5, 1.0), fontsize=28)
plt.yticks(fontsize=28)
plt.xlim(5.5, 13.5)

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



iso_integrand = np.copy(iso_062_integrand)
cube_with_units = (iso_062_integrand*10**6)*(u.Jy/u.sr)

iso_integrand = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(iso_wavelengths_062*u.micron))

iso_integrand = iso_integrand*(u.micron)
iso_integrand = iso_integrand*(u.m)
iso_integrand = iso_integrand*(u.m)
iso_integrand = iso_integrand*(u.sr/u.W)

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



iso_integrand = np.copy(iso_112_integrand)
cube_with_units = (iso_112_integrand*10**6)*(u.Jy/u.sr)

iso_integrand = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(iso_wavelengths_112*u.micron))

iso_integrand = iso_integrand*(u.micron)
iso_integrand = iso_integrand*(u.m)
iso_integrand = iso_integrand*(u.m)
iso_integrand = iso_integrand*(u.sr/u.W)

plt.figure()
plt.plot(iso_wavelengths_112, iso_integrand)
#plt.plot(iso_wavelengths_112, iso_data_112)
#plt.plot(iso_wavelengths_112, iso_continuum)
plt.show()

print('estimated flux using triangle up to 11.259, triangle from 11.383, and rectangle in the middle that starts at the height of the first point, is 11.1e-8')

