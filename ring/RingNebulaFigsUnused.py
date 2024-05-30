
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

#standard stuff
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import  AutoMinorLocator

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



#loading in template PAH spectra
pah_image_file = np.loadtxt('data/misc/barh_stick_csub.fits.dat', skiprows=1)
pah_wavelengths = pah_image_file[:,0]
pah_data = pah_image_file[:,1]

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

#hh continuum subtraction

#combining 3 different channels

pah_removed_hh, wavelength_pah_removed_hh, overlap = rnf.flux_aligner2(
    hh_wavelengths2, hh_wavelengths3, hh_data2, hh_data3-15)



#first, need to fit continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(wavelength_pah_removed_hh, 2) == 11.0)[0][0]
temp_index_2 = np.where(np.round(hh_wavelengths3, 2) == 11.65)[0][0] #was 11.82 originally

#calculating the slope of the line to use

#preventing the value found from being on a line or something
pah_slope_1 = np.mean(pah_removed_hh[temp_index_1 - 20:20+temp_index_1])

pah_slope_2 = np.mean(hh_data3[temp_index_2 - 20:20+temp_index_2] - 15)

#value where the wavelengths change interval
'''
overlap_index = np.where(np.round(wavelength_pah_removed_112, 2) == 
                         (wavelength_pah_removed_112[overlap[0]] + wavelengths7[overlap[1]])/2)[0][0]
pah_slope_3 = np.mean(pah_removed_112[overlap_index - 20:20+overlap_index])
'''


pah_slope = (pah_slope_2 - pah_slope_1)/\
(hh_wavelengths3[temp_index_2] - wavelength_pah_removed_hh[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(pah_removed_hh[:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(hh_data3[temp_index_2:]-15))

pah_removed_3 = pah_slope*(wavelength_pah_removed_hh[temp_index_1:overlap[0]] - 
                           wavelength_pah_removed_hh[temp_index_1]) + pah_slope_1
pah_removed_4 = pah_slope*(hh_wavelengths3[overlap[1]:temp_index_2] - 
                           wavelength_pah_removed_hh[temp_index_1]) + pah_slope_1


'''
for i in range(len(pah_removed_3)):
    pah_removed_3[i] = pah_slope*i + pah_slope_1
'''

#putting it all together
pah_removed_hh = np.concatenate((pah_removed_1, pah_removed_3))
pah_removed_hh = np.concatenate((pah_removed_hh, pah_removed_4))
pah_removed_hh = np.concatenate((pah_removed_hh, pah_removed_2))

#wavelength_pah_removed_1 = wavelength_pah_removed_112[:868]
#wavelength_pah_removed_2 = wavelength_pah_removed_112[1220:]
#wavelength_pah_removed_112 = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))



#NOTE: not using my ransac function because i want to use a different
# wavelength array than the one used to make the fit
'''
wavelengths_integrand_112 = wavelength_pah_removed_112.reshape(-1,1)
data_integrand_112 = pah_removed_112.reshape(-1,1)

# Init the RANSAC regressor
ransac = make_pipeline(PolynomialFeatures(20), RANSACRegressor(max_trials=200000, random_state=41))

# Fit with RANSAC
ransac.fit(wavelengths_integrand_112, data_integrand_112)

#now combining 2 different channels, with full data
'''
pah_hh, wavelength_pah_hh, overlap = rnf.flux_aligner2(
    hh_wavelengths2, hh_wavelengths3, hh_data2, hh_data3-15)

'''
wavelengths_integrand_112 = wavelength_pah_removed_112.reshape(-1,1)

# Get the fitted data result
polynomial6 = ransac.predict(wavelengths_integrand_112)

polynomial6 = polynomial6.reshape((1, -1))[0]
'''

continuum_removed_hh = pah_hh - pah_removed_hh# - polynomial6

continuum_hh = np.copy(pah_removed_hh)

#everything_removed6 = rnf.emission_line_remover(continuum_removed6, 15, 1)

plt.figure()
plt.plot(wavelength_pah_hh, pah_hh)
plt.plot(wavelength_pah_hh, pah_removed_hh)
plt.plot(wavelength_pah_hh, continuum_removed_hh)
plt.ylim(-10, 100)
plt.xlim(10, 14)
#plt.plot(hh_wavelengths2, hh_data2, alpha=0.5)
#plt.plot(hh_wavelengths3, hh_data3, alpha=0.5)
plt.show()
plt.close()




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
MIRI PLOTS
'''

pahoverlap_miri2_1 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths2[0], 2))[0][0]
pahoverlap_miri2_2 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths2[-1], 2))[0][0]

pahoverlap_miri3_1 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths3[0], 2))[0][0]
pahoverlap_miri3_2 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths3[-1], 2))[0][0]



#%%

#region where 7.7 and 8.6 features may be

#%%

cutoff_095 = np.where(np.round(wavelengths5, 2) == 9.5)[0][0]

pahoverlap_miri4_1 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths4[0], 2))[0][0]
pahoverlap_miri4_2 = np.where(np.round(pah_wavelengths, 2) == 9.5)[0][0]

#%%

#RNF_086_mean_extended_North

#%%

ax = plt.figure('RNF_086_mean_extended_North', figsize=(16,6)).add_subplot(111)
plt.title('JWST Mean of Data, Potential 7.7 and 8.6 Features, North', fontsize=20)
plt.plot(wavelengths3, data3 - 9, label='Ch1-long, offset=-9')
plt.plot(wavelengths4, data4 - 19, label='Ch2-short, offset=-19')
plt.plot(wavelengths5[:cutoff_095], data5[:cutoff_095] - 19, label='Ch2-medium, offset=-19')
plt.plot(pah_wavelengths[pahoverlap_miri3_1:pahoverlap_miri4_2], 
         0.3*pah_data[pahoverlap_miri3_1:pahoverlap_miri4_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
#plt.plot(wavelengths4, line_ransac_4north, color='black')
#plt.plot(wavelengths3, line_ransac_3north, color='black')
#plt.plot(wavelengths4_west, line_ransac_4west, color='black', linestyle='dashed')
#plt.plot(wavewave[:574], spitzer[:574], label='spitzer', color='g')
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(6.5, 9.5, 0.2), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_086_mean_extended_North.png')
plt.show()
plt.close() 


#%%

#RNF_086_mean_extended_West

#%%

ax = plt.figure('RNF_086_mean_extended_West', figsize=(16,6)).add_subplot(111)
plt.title('JWST Mean of Data, Potential 7.7 and 8.6 Features, West', fontsize=20)
plt.plot(wavelengths3_west, corrected_data3_west - 0, label='Ch1-long, offset=0')
plt.plot(wavelengths4_west, corrected_data4_west + 5, label='Ch2-short, offset=+5')
plt.plot(wavelengths5_west[:cutoff_095], corrected_data5_west[:cutoff_095] + 5, label='Ch2-medium, offset=+5')
plt.plot(pah_wavelengths[pahoverlap_miri3_1:pahoverlap_miri4_2], 
         0.3*pah_data[pahoverlap_miri3_1:pahoverlap_miri4_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
#plt.plot(wavelengths4, line_ransac_4north, color='black')
#plt.plot(wavelengths3, line_ransac_3north, color='black')
#plt.plot(wavelengths4_west, line_ransac_4west, color='black', linestyle='dashed')
#plt.plot(wavewave[:574], spitzer[:574], label='spitzer', color='g')
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(6.5, 9.5, 0.2), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_086_mean_extended_West.png')
plt.show()
plt.close() 

#%%

#region where 11.2 feature may be

#%%

pahoverlap_miri6_1 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths6[0], 2))[0][0]
pahoverlap_miri6_2 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths6[-1], 2))[0][0]

pahoverlap_miri7_1 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths7[0], 2))[0][0]
pahoverlap_miri7_2 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths7[-1], 2))[0][0]

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


#%%

#RNF_112_mean_extended_North

#%%

data_simple6, pog = rnf.weighted_mean_finder_simple(image_data6, error_data6)
data_simple7, pog = rnf.weighted_mean_finder_simple(image_data7, error_data7)
#%%
#offset was originally -25 in ch2-long and -20 in ch3-short.

ax = plt.figure('RNF_112_mean_extended_North', figsize=(16,6)).add_subplot(111)
plt.title('JWST Mean of Data, 11.2 feature, North', fontsize=20)
plt.plot(wavelengths6, corrected_data6, label='Ch2-long, offset=0')
#plt.plot(wavelengths6, data_simple6 - background6, label='Ch2-long, larger weighted mean aperture')
plt.plot(wavelengths7, corrected_data7, label='Ch3-short, offset=0',)
#plt.plot(comparison_wavelengths6, data6_corrected - subtract, label='North, ' + 'A=%.3f' % A6, color='orange')
#plt.plot(comparison_wavelengths7, data7_corrected, label='North, ' + 'A=%.3f' % A7, color='green')
#plt.plot(wavelengths6_west, data6_west, label='West')
#plt.plot(spitzer_wavelengths[:250], spitzer_data[:250], label='Spitzer', color='g')
#plt.plot(comparison_wavelengths6, line_ransac_6north, color='black')
plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
         2*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 0, label='Spitzer, scale=2, offset=0', color='black', alpha=0.8)
'''
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.3*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
'''
#plt.plot(wavelengths6_west, line_ransac_6west, color='black', linestyle='dashed')
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(10, 13.6, 0.2), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_112_mean_extended_North.png')
plt.show()
plt.close() 

#%%

#RNF_112_mean_extended_West

#%%

ax = plt.figure('RNF_112_mean_extended_West', figsize=(16,6)).add_subplot(111)
plt.title('JWST Mean of Data, 11.2 feature, West', fontsize=20)
plt.plot(wavelengths6_west, data6_west - 0, label='Ch2-long, offset=0')
#plt.plot(wavelengths6, error_data6)
plt.plot(wavelengths7_west, data7_west - 0, label='Ch3-short, offset=0',)
#plt.plot(comparison_wavelengths6, data6_corrected - subtract, label='North, ' + 'A=%.3f' % A6, color='orange')
#plt.plot(comparison_wavelengths7, data7_corrected, label='North, ' + 'A=%.3f' % A7, color='green')
#plt.plot(wavelengths6_west, data6_west, label='West')
#plt.plot(spitzer_wavelengths[:250], spitzer_data[:250], label='Spitzer', color='g')
#plt.plot(comparison_wavelengths6, line_ransac_6north, color='black')
plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
         3*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 6, label='Spitzer, scale=3, offset=-6', color='black', alpha=0.8)
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.3*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
#plt.plot(wavelengths6_west, line_ransac_6west, color='black', linestyle='dashed')
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(10, 13.6, 0.2), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_112_mean_extended_West.png')
plt.show()
plt.close() 


#%%

#all relevant MIRI data

#%%

#pahoverlap_miri8_1 = np.where(np.round(pah_wavelengths, 2) == 5.7)[0][0] 
pahoverlap_miri8_2 = np.where(np.round(pah_wavelengths, 2) == 15.2)[0][0]


#%%

#RNF_mean_extended_North_offset

#%%

#mean, offsetting but no scaling, north

ax = plt.figure('RNF_mean_extended_North_offset', figsize=(16,6)).add_subplot(111)
plt.title('JWST Mean of Data, North', fontsize=20)
plt.plot(wavelengths1, data1 - 0, label='Ch1-short, offset=0')
plt.plot(wavelengths2, data2 +4, label='Ch1-medium, offset=4')
plt.plot(wavelengths3, data3 +10, label='Ch1-long, offset=10')
plt.plot(wavelengths4, data4 +10, label='Ch2-short, offset=10')
plt.plot(wavelengths5, data5 + 5, label='Ch2-medium, offset=5')
plt.plot(wavelengths6, data6 + 5, label='Ch2-long, offset=5')
plt.plot(wavelengths7, data7 +4, label='Ch3-short, offset=4',)
plt.plot(wavelengths8, data8 - 20, label='Ch3-medium, offset=-19',)
plt.plot(pah_wavelengths[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         0.3*pah_data[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(5.6, 15.2, 0.8), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_mean_extended_North_offset.png')
plt.show()
plt.close()

#%%

#RNF_mean_extended_West_offset

#%%

#mean, offsetting but no scaling, west
ax = plt.figure('RNF_mean_extended_West_offset', figsize=(16,6)).add_subplot(111)
plt.title('JWST Mean of Data, West', fontsize=20)
plt.plot(wavelengths1_west, data1_west + 0, label='Ch1-medium, offset=+0')
plt.plot(wavelengths2_west, data2_west + 0, label='Ch1-medium, offset=+0')
plt.plot(wavelengths3_west, data3_west - 0, label='Ch1-long, offset=-0')
plt.plot(wavelengths4_west, data4_west + 5, label='Ch2-short, offset=+3')
plt.plot(wavelengths5_west, data5_west + 4, label='Ch2-medium, offset=+3')
plt.plot(wavelengths6_west, data6_west - 2, label='Ch2-long, offset=-3')
plt.plot(wavelengths7_west, data7_west - 2, label='Ch3-short, offset=-2',)
plt.plot(wavelengths8_west, data8_west - 22, label='Ch3-medium, offset=-22',)
plt.plot(pah_wavelengths[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         0.3*pah_data[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(5.6, 15.2, 0.8), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_mean_extended_West_offset.png')
plt.show()
plt.close()

#%%

ax = plt.figure('RNF_mean_extended_West_offset_h2', figsize=(16,6)).add_subplot(111)
plt.title('JWST Mean of Data, West', fontsize=20)
plt.plot(wavelengths1_west, data1_west_blob + 0, label='Ch1-medium, offset=+0')
plt.plot(wavelengths2_west, data2_west_blob + 0, label='Ch1-medium, offset=+0')
plt.plot(wavelengths3_west, data3_west_blob - 0, label='Ch1-long, offset=-0')
plt.plot(wavelengths4_west, data4_west_blob + 5, label='Ch2-short, offset=+3')
plt.plot(wavelengths5_west, data5_west_blob + 4, label='Ch2-medium, offset=+3')
plt.plot(wavelengths6_west, data6_west_blob - 2, label='Ch2-long, offset=-3')
plt.plot(wavelengths7_west, data7_west_blob - 2, label='Ch3-short, offset=-2',)
plt.plot(wavelengths8_west, data8_west_blob - 22, label='Ch3-medium, offset=-22',)
plt.plot(pah_wavelengths[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         0.3*pah_data[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(5.6, 15.2, 0.8), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_mean_extended_West_h2_offset.png')
plt.show()
plt.close()

#%% 

#RNF_mean_extended_North

#%%

#mean, no scaling or offsetting, north
ax = plt.figure('RNF_mean_extended_North', figsize=(16,6)).add_subplot(111)
plt.title('JWST Mean of Data, North, no offset', fontsize=20)
plt.plot(wavelengths2, data2 - 0, label='Ch1-medium')
plt.plot(wavelengths3, data3 - 0, label='Ch1-long')
plt.plot(wavelengths4, data4 - 0, label='Ch2-short')
plt.plot(wavelengths5, data5 - 0, label='Ch2-medium')
plt.plot(wavelengths6, data6 - 0, label='Ch2-long')
plt.plot(wavelengths7, data7 - 0, label='Ch3-short',)
plt.plot(wavelengths8, data8 - 0, label='Ch3-medium',)
plt.plot(pah_wavelengths[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         0.3*pah_data[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(5.6, 15.2, 0.8), fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=14)
plt.savefig('Figures/RNF_mean_extended_North.png')
plt.show()
plt.close()

#%%

#RNF_mean_extended_West

#%%

#mean, no scaling or offsetting, west
ax = plt.figure('RNF_mean_extended_West', figsize=(16,6)).add_subplot(111)
plt.title('JWST Mean of Data, West, no offset', fontsize=20)
plt.plot(wavelengths2_west, data2_west, label='Ch1-medium')
plt.plot(wavelengths3_west, data3_west - 0, label='Ch1-long')
plt.plot(wavelengths4_west, data4_west - 0, label='Ch2-short')
plt.plot(wavelengths5_west, data5_west - 0, label='Ch2-medium')
plt.plot(wavelengths6_west, data6_west - 0, label='Ch2-long')
plt.plot(wavelengths7_west, data7_west - 0, label='Ch3-short',)
plt.plot(wavelengths8_west, data8_west - 0, label='Ch3-medium',)
plt.plot(pah_wavelengths[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         0.3*pah_data[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(5.6, 15.2, 0.8), fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=14)
plt.savefig('Figures/RNF_mean_extended_West.png')
plt.show()
plt.close()



####################################



'''
NIRSPEC PLOTS
'''

#%%

#making a figure of the nirspec weighted means

#%%

nirspec_cutoff = np.where(np.round(wavelengths_nirspec4, 2) == 4)[0][0]

pahoverlap_nirspec4_1 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths_nirspec4[0], 2))[0][0]
pahoverlap_nirspec4_2 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths_nirspec4[nirspec_cutoff], 2))[0][0]

#%%

plt.figure()
plt.plot(wavelengths_nirspec4, nirspec_weighted_mean4_west/110)
plt.plot(wavelengths_nirspec4, nirspec_data4_west[:,20,20])
plt.ylim(0,0.1)
plt.show()
plt.close()

#%%

#RNF_033_N_W

#%%

#Note: I think its one of the NGC now and not orion

ax = plt.figure('RNF_033_N_W', figsize=(16,6)).add_subplot(111)
plt.title('NIRSPEC Weighted Mean, and ISO orion spectra', fontsize=20)
#plt.plot(wavelengths_nirspec1, nirspec_weighted_mean1, label='g140m-f070')
#plt.plot(wavelengths_nirspec2, nirspec_weighted_mean2, label='g140m-f100')
#plt.plot(wavelengths_nirspec3, nirspec_weighted_mean3, label='g235m-f170')
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], nirspec_weighted_mean4[:nirspec_cutoff] - 1.8, 
         label='g395m-f290, North, offset=-1.8', color='purple')
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], nirspec_weighted_mean4_west[:nirspec_cutoff] - 1, 
         label='g395m-f290, West, offset=-1.0', color='orange')
plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.3*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
#plt.plot(wavelengths_nirspec4, line_ransac_nirspec_4north, color='black')
#plt.plot(wavelengths_nirspec4_west, line_ransac_nirspec_4west, color='black', linestyle='dashed')
#plt.plot(wavelengths2, weighted_mean_error033, label='error')
plt.ylim((-2,5))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
#plt.xticks(np.arange(1.0, 5.0, 1.0), fontsize=14)
plt.xticks(np.arange(2.8, 4.0, 0.1), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_033_N_W.png')
plt.show()
plt.close() 



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


#%%

#RNF_033_gaussian_fit_North

#%%
#old stuff: mean=3.29, fwhm=0.015556349, scale=2.1; mean=3.25, fwhm=0.015556349, scale=0.6; mean=3.33, fwhm=0.008485281, scale=0.4'
ax = plt.figure('RNF_033_gaussian_fit_North', figsize=(16,6)).add_subplot(111)
plt.title('NIRSPEC Weighted Mean, gaussian fit, North', fontsize=20)
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], nirspec_weighted_mean4[:nirspec_cutoff] - 2.4, 
         label='g395m-f290, North, offset=-2.4', color='purple')

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

plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.38*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
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
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_033_gaussian_fit_North.png')
plt.show() 
plt.close()

#%%

#RNF_paper_033_gaussian_fit_North

#%%
#old stuff: mean=3.29, fwhm=0.015556349, scale=2.1; mean=3.25, fwhm=0.015556349, scale=0.6; mean=3.33, fwhm=0.008485281, scale=0.4'
ax = plt.figure('RNF_paper_033_gaussian_fit_North', figsize=(16,6)).add_subplot(111)
#plt.title('NIRSPEC Weighted Mean, gaussian fit, North', fontsize=20)
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], nirspec_weighted_mean4[:nirspec_cutoff] - 2.4, 
         label='g395m-f290, North, offset=-2.4', color='purple')

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

plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.38*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
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
plt.savefig('Figures/paper/RNF_paper_033_gaussian_fit_North.png')
plt.show() 

#%%
'''
#old stuff: mean=3.29, fwhm=0.015556349, scale=2.1; mean=3.25, fwhm=0.015556349, scale=0.6; mean=3.33, fwhm=0.008485281, scale=0.4'
ax = plt.figure('RNF_033_gaussian_fit', figsize=(16,6)).add_subplot(111)
plt.title('NIRSPEC Weighted Mean, gaussian fit, North', fontsize=20)

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.29027, 0.0375, 6, 0), 
         label ='gaussian fit mean=3.29027, fwhm=0.03483, scale=2.1')
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.2465, 0.0375, 0.65, 0), 
         label ='gaussian fit mean=3.2465, fwhm=0.0375, scale=0.6')
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.32821, 0.0264, 0.5, 0), 
         label ='gaussian fit mean=3.32821, fwhm=0.0264, scale=0.4')
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.405, 0.01, 1.1, 0), 
         label ='gaussian fit mean=3.405, fwhm=0.01, scale=1.1')
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.29027, 0.0375, 6, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.2465, 0.0375, 0.65, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.32821, 0.0264, 0.5, 0), 
         label='gaussian fit sum')

plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         1*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         label='ISO orion spectra, scale=0.38', color='r', alpha=0.5)
plt.ylim((-2,7))
ax.xaxis.grid(True, which='minor')
ax.tick_params(axis='x', which='both', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', right='True')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(2.8, 4.0, 0.1), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_033_orion_gaussian_fit.png')
plt.show() 
plt.close()
'''
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



####################################



'''
FITTING GAUSSIAN TO 11.2
'''

#%%
#combining 3 different channels

pah_removed_112, wavelength_pah_removed_112, overlap = rnf.flux_aligner2(
    wavelengths5, wavelengths6, corrected_data5, corrected_data6)

pah_removed_112, wavelength_pah_removed_112, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_112, wavelengths7, pah_removed_112, corrected_data7)

#first, need to fit continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(wavelength_pah_removed_112, 2) == 11.0)[0][0]
temp_index_2 = np.where(np.round(wavelengths7, 2) == 11.82)[0][0]

#calculating the slope of the line to use

#preventing the value found from being on a line or something
pah_slope_1 = np.mean(pah_removed_112[temp_index_1 - 20:20+temp_index_1])

pah_slope_2 = np.mean(corrected_data7[temp_index_2 - 20:20+temp_index_2])

#value where the wavelengths change interval
'''
overlap_index = np.where(np.round(wavelength_pah_removed_112, 2) == 
                         (wavelength_pah_removed_112[overlap[0]] + wavelengths7[overlap[1]])/2)[0][0]
pah_slope_3 = np.mean(pah_removed_112[overlap_index - 20:20+overlap_index])
'''


pah_slope = (pah_slope_2 - pah_slope_1)/\
(wavelengths7[temp_index_2] - wavelength_pah_removed_112[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(pah_removed_112[:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(corrected_data7[temp_index_2:]))

pah_removed_3 = pah_slope*(wavelength_pah_removed_112[temp_index_1:overlap[0]] - 
                           wavelength_pah_removed_112[temp_index_1]) + pah_slope_1
pah_removed_4 = pah_slope*(wavelengths7[overlap[1]:temp_index_2] - 
                           wavelength_pah_removed_112[temp_index_1]) + pah_slope_1


'''
for i in range(len(pah_removed_3)):
    pah_removed_3[i] = pah_slope*i + pah_slope_1
'''

#putting it all together
pah_removed_112 = np.concatenate((pah_removed_1, pah_removed_3))
pah_removed_112 = np.concatenate((pah_removed_112, pah_removed_4))
pah_removed_112 = np.concatenate((pah_removed_112, pah_removed_2))

#wavelength_pah_removed_1 = wavelength_pah_removed_112[:868]
#wavelength_pah_removed_2 = wavelength_pah_removed_112[1220:]
#wavelength_pah_removed_112 = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))



#NOTE: not using my ransac function because i want to use a different
# wavelength array than the one used to make the fit
'''
wavelengths_integrand_112 = wavelength_pah_removed_112.reshape(-1,1)
data_integrand_112 = pah_removed_112.reshape(-1,1)

# Init the RANSAC regressor
ransac = make_pipeline(PolynomialFeatures(20), RANSACRegressor(max_trials=200000, random_state=41))

# Fit with RANSAC
ransac.fit(wavelengths_integrand_112, data_integrand_112)

#now combining 2 different channels, with full data
'''
pah_112, wavelength_pah_112, overlap = rnf.flux_aligner2(
    wavelengths5, wavelengths6, corrected_data5, corrected_data6)

pah_112, wavelength_pah_112, overlap = rnf.flux_aligner2(
    wavelength_pah_112, wavelengths7, pah_112, corrected_data7)
'''
wavelengths_integrand_112 = wavelength_pah_removed_112.reshape(-1,1)

# Get the fitted data result
polynomial6 = ransac.predict(wavelengths_integrand_112)

polynomial6 = polynomial6.reshape((1, -1))[0]
'''

continuum_removed6 = pah_112 - pah_removed_112# - polynomial6

everything_removed6 = rnf.emission_line_remover(continuum_removed6, 15, 1)
#everything_removed6 = rnf.absorption_line_remover(everything_removed6, 15, 1)

#remaking these variables with new bounds (the others only go down to 10 microns)

ngc7027_index_begin = np.where(np.round(ngc7027_wavelengths, 2) == np.round(wavelengths5[0], 2))[0][0]
ngc7027_index_end = np.where(np.round(ngc7027_wavelengths, 2) == np.round(wavelengths5[-1], 2))[0][0]
ngc7027_index_end2 = np.where(np.round(ngc7027_wavelengths, 2) == np.round(wavelengths7[-1], 2))[0][0]

#%%

#RNF_112_continuum_extended_North

#%%

ax = plt.figure('RNF_112_continuum_extended_North', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, 11.2 feature, North', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
plt.plot(wavelength_pah_removed_112, pah_112, label='data')
plt.plot(wavelength_pah_removed_112, continuum_removed6, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112, everything_removed6, label='Lines and Continuum removed')
plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.42*continuum_removed_hh[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.42', color='r', alpha=0.5)
'''
plt.plot(ngc7027_wavelengths[ngc7027_index_begin:ngc7027_index_end2], 
         0.033*ngc7027_data[ngc7027_index_begin:ngc7027_index_end2] - 8, 
         label='NGC 7027 spectra, scale=0.033, offset=-8', color='purple', alpha=1)
'''

#plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
#         2*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 6, label='Spitzer, scale=2, offset=-6', color='black', alpha=0.8)

plt.plot(wavelength_pah_removed_112, pah_removed_112, color='black', label='continuum')
'''
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.15*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.15', color='r', alpha=0.5)
'''
plt.plot(11.0*np.ones(len(wavelength_pah_removed_112)), np.linspace(-10, 25, len(pah_removed_112)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(11.6*np.ones(len(wavelength_pah_removed_112)), np.linspace(-10, 25, len(pah_removed_112)), 
         color='black', label='upper integration bound (11.6)')
plt.ylim((-10,25))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(8.6, 13.6, 0.3), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11)
plt.savefig('Figures/RNF_112_continuum_extended_North.png')
plt.show()
plt.close()

#%%

#RNF_112_continuum_extended_North_simple

#%%

ax = plt.figure('RNF_112_continuum_extended_North_simple', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, 11.2 feature, North, simple', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
plt.plot(wavelength_pah_removed_112, pah_112, label='data')
plt.plot(wavelength_pah_removed_112, continuum_removed6, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112, everything_removed6, label='Lines and Continuum removed')
'''
plt.plot(hh_wavelengths[hh_index_begin:hh_index_end2], 
         0.45*hh_data[hh_index_begin:hh_index_end2] - 5, 
         label='HorseHead Nebula spectra, scale=0.45, offset=-5', color='r', alpha=1)
plt.plot(ngc7027_wavelengths[ngc7027_index_begin:ngc7027_index_end2], 
         0.033*ngc7027_data[ngc7027_index_begin:ngc7027_index_end2] - 8, 
         label='NGC 7027 spectra, scale=0.033, offset=-8', color='purple', alpha=1)

plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
         2*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 6, label='Spitzer, scale=2, offset=-6', color='black', alpha=0.8)
'''
plt.plot(wavelength_pah_removed_112, pah_removed_112, color='black', label='continuum')
plt.plot(wavelength_pah_removed_112, 0*pah_removed_112, color='black', label='zero')
plt.plot(11.0*np.ones(len(wavelength_pah_removed_112)), np.linspace(-10, 25, len(pah_removed_112)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(11.6*np.ones(len(wavelength_pah_removed_112)), np.linspace(-10, 25, len(pah_removed_112)), 
         color='black', label='upper integration bound (11.6)')
'''
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.15*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.15', color='r', alpha=0.5)
'''
plt.ylim((-10,25))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(8.6, 13.6, 0.3), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11)
plt.savefig('Figures/RNF_112_continuum_extended_North_simple.png')
plt.show()
plt.close()

#%%

#RNF_112_for_mikako

#%%

ax = plt.figure('RNF_112_for_mikako', figsize=(16,6)).add_subplot(111)
plt.title('JWST 11.2 feature, North', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
plt.plot(wavelength_pah_removed_112, pah_112, label='JWST')
#plt.plot(wavelength_pah_removed_112, continuum_removed6, label='Continuum subtracted')
#plt.plot(wavelength_pah_removed_112, everything_removed6, label='Lines and Continuum removed')
'''
plt.plot(hh_wavelengths[hh_index_begin:hh_index_end2], 
         0.45*hh_data[hh_index_begin:hh_index_end2] - 5, 
         label='HorseHead Nebula spectra, scale=0.45, offset=-5', color='r', alpha=1)
plt.plot(ngc7027_wavelengths[ngc7027_index_begin:ngc7027_index_end2], 
         0.033*ngc7027_data[ngc7027_index_begin:ngc7027_index_end2] - 8, 
         label='NGC 7027 spectra, scale=0.033, offset=-8', color='purple', alpha=1)
'''
plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
         2*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 10, label='Spitzer, scale=2, offset=-10', color='black', alpha=0.8)
'''
plt.plot(wavelength_pah_removed_112, pah_removed_112, color='black', label='continuum')
plt.plot(wavelength_pah_removed_112, 0*pah_removed_112, color='black', label='zero')
plt.plot(11.0*np.ones(len(wavelength_pah_removed_112)), np.linspace(-10, 25, len(pah_removed_112)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(11.6*np.ones(len(wavelength_pah_removed_112)), np.linspace(-10, 25, len(pah_removed_112)), 
         color='black', label='upper integration bound (11.6)')
'''
'''
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.15*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.15', color='r', alpha=0.5)
'''
plt.ylim((-15,25))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(8.6, 13.6, 0.3), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11)
plt.savefig('Figures/RNF_112_for_mikako.png')
plt.show()
plt.close()

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

#%%

plt.figure()
plt.plot(wavelengths_integrand, everything_removed6[l_int:u_int])
plt.show()
plt.close()

#%%
plt.figure()
#plt.plot(wavelengths1, corrected_data1)
#plt.plot(wavelengths2, corrected_data2+1, alpha=0.5)
plt.plot(wavelengths3, corrected_data3, alpha=0.5)
plt.plot(wavelengths4, corrected_data4+3, alpha=0.5)
plt.plot(wavelengths5, corrected_data5, alpha=0.5)
#plt.plot(wavelengths6, corrected_data6+3, alpha=0.5)
#plt.plot(wavelengths7, corrected_data7+3, alpha=0.5)
plt.ylim(-10, 30)
plt.show()
plt.close()

#%%



#for 7.7 amd 8.6 features, exclude 7.1 to 9.2 (funky business from 8.9 to 9.2 that i dont wanna fit so grouping
#it in with the 8.6 feature)

#first, need to fit continuum, do so by eyeballing a polynomial to it, adding 3 to some so everything lines up

pah_removed_077, wavelength_pah_removed_077, overlap1 = rnf.flux_aligner2(
    wavelengths3, wavelengths4, corrected_data3, corrected_data4+3)

pah_removed_077, wavelength_pah_removed_077, overlap2 = rnf.flux_aligner2(
    wavelength_pah_removed_077, wavelengths5, pah_removed_077, corrected_data5)

pah_077 = np.copy(pah_removed_077)

'''
pah_removed_1 = corrected_data3[:837]
pah_removed_2 = corrected_data4 - 1
pah_removed_077 = np.concatenate((pah_removed_1, pah_removed_2))
pah_removed_2 = corrected_data5[408:]
pah_removed_077 = np.concatenate((pah_removed_077, pah_removed_2))

wavelength_pah_removed_1 = wavelengths3[:837]
wavelength_pah_removed_2 = wavelengths4
wavelength_pah_removed_077 = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))
wavelength_pah_removed_2 = wavelengths5[408:]
wavelength_pah_removed_077 = np.concatenate((wavelength_pah_removed_077, wavelength_pah_removed_2))
'''
#NOTE: not using my ransac function because i want to use a different
# wavelength array than the one used to make the fit

wavelengths_integrand_077 = wavelength_pah_removed_077.reshape(-1,1)
data_integrand_077 = pah_077.reshape(-1,1)
'''
wavelength_pah_removed_1 = wavelengths3
wavelength_pah_removed_2 = wavelengths4
wavelength_pah_removed_077 = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))
wavelength_pah_removed_2 = wavelengths5
wavelength_pah_removed_077 = np.concatenate((wavelength_pah_removed_077, wavelength_pah_removed_2))

wavelengths_integrand_077 = wavelength_pah_removed_077.reshape(-1,1)
'''
#now that continuum has been found, redo combined array with nothing removed for subtraction






#first, need to fit continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(wavelength_pah_removed_077, 2) == 7.2)[0][0]
temp_index_2 = np.where(np.round(wavelengths5, 2) == 8.9)[0][0]

#calculating the slope of the line to use

#preventing the value found from being on a line or something
pah_slope_1 = np.mean(pah_removed_077[temp_index_1 - 20:20+temp_index_1])

pah_slope_2 = np.mean(corrected_data5[temp_index_2 - 20:20+temp_index_2])

#value where the wavelengths change interval
'''
overlap_index = np.where(np.round(wavelength_pah_removed_112, 2) == 
                         (wavelength_pah_removed_112[overlap[0]] + wavelengths7[overlap[1]])/2)[0][0]
pah_slope_3 = np.mean(pah_removed_112[overlap_index - 20:20+overlap_index])
'''


pah_slope = (pah_slope_2 - pah_slope_1)/\
(wavelengths5[temp_index_2] - wavelength_pah_removed_077[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(pah_removed_077[:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(corrected_data5[temp_index_2:]))

pah_removed_3 = pah_slope*(wavelength_pah_removed_077[temp_index_1:overlap1[0]] - 
                           wavelength_pah_removed_077[temp_index_1]) + pah_slope_1

#getting the right index for wavelengths4 that corresponds to overlap2[0]

temp_index_3 = np.where(np.round(wavelengths4, 2) == np.round(wavelength_pah_removed_077[overlap2[0]], 2))[0][2]

pah_removed_4 = pah_slope*(wavelengths4[overlap1[1]:temp_index_3] - 
                           wavelength_pah_removed_077[temp_index_1]) + pah_slope_1
pah_removed_5 = pah_slope*(wavelengths5[overlap2[1]:temp_index_2] - 
                           wavelength_pah_removed_077[temp_index_1]) + pah_slope_1


'''
for i in range(len(pah_removed_3)):
    pah_removed_3[i] = pah_slope*i + pah_slope_1
'''

#putting it all together
pah_removed_077 = np.concatenate((pah_removed_1, pah_removed_3))
pah_removed_077 = np.concatenate((pah_removed_077, pah_removed_4))
pah_removed_077 = np.concatenate((pah_removed_077, pah_removed_5))
pah_removed_077 = np.concatenate((pah_removed_077, pah_removed_2))



'''
pah_removed_1 = corrected_data3
pah_removed_2 = corrected_data4+3 # - 1
pah_removed_077 = np.concatenate((pah_removed_1, pah_removed_2))
pah_removed_2 = corrected_data5 # - 5
pah_removed_077 = np.concatenate((pah_removed_077, pah_removed_2))
'''
#ok so the polynomial fit sucks as the fit is somehow finding the 7.7 feature and subtracting it, so
#going to use a constant instead

#%%

continuum_removed4 = pah_077 - pah_removed_077 # - 1.5

everything_removed4 = rnf.emission_line_remover(continuum_removed4, 20, 1)


#%%

#RNF_077_continuum_extended_North

#%%

ax = plt.figure('RNF_077_continuum_extended_North', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum of Data, 7.7 amd 8.6 features, North', fontsize=20)
plt.plot(wavelengths_integrand_077, pah_077, label='Data')
plt.plot(wavelengths_integrand_077, continuum_removed4, label='Continuum subtracted')
plt.plot(wavelengths_integrand_077, everything_removed4, label='Everything removed')
plt.plot(wavelengths_integrand_077, pah_removed_077, label='Continuum')
#plt.plot(wavelength, polynomial4, label='Continuum (not in use)')
plt.plot(wavelengths_integrand_077, 0*everything_removed4, label='zero')
plt.plot(pah_wavelengths[pahoverlap_miri3_1:pahoverlap_miri4_2], 
         0.25*pah_data[pahoverlap_miri3_1:pahoverlap_miri4_2], 
         label='ISO orion spectra, scale=0.25', color='r', alpha=0.5)
'''
plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
         3*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 6, label='Spitzer, scale=3, offset=-6', color='black', alpha=0.8)
'''
#plt.plot(wavelengths6, polynomial6, color='black', label='continuum')
#plt.plot(wavelengths7, polynomial7, color='black', label='continuum')
'''
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.3*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
'''
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(6.6, 10.2, 0.2), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_077_continuum_extended_North.png')
plt.show()
plt.close()





#%%

#integrating flux of 7.7 feature



#integrate from 7.2 to 8.1 microns

l_int = np.where(np.round(wavelengths_integrand_077, 3) == 7.2)[0][0]
u_int = np.where(np.round(wavelengths_integrand_077, 3) == 8.1)[0][0]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_077 = np.copy(everything_removed4[l_int:u_int])

wavelengths_integrand = wavelengths_integrand_077[l_int:u_int]

#wavepog has wrong shape, need to fix

temp = np.copy(wavelengths_integrand)

wavelengths_integrand = np.zeros(len(integrand_077))

for i in range(len(integrand_077)):
    wavelengths_integrand[i] = temp[i,0]

final_cube = np.zeros(integrand_077.shape)
cube_with_units = (integrand_077*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies =\
                                u.spectral_density(wavelengths_integrand*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_temp_077 = np.copy(integrand_077)
for i in range(len(integrand_077)):
    integrand_temp_077[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_temp_077), 2):
    odd_sum += integrand_temp_077[i] 

even_sum = 0    

for i in range(2, len(integrand_temp_077), 2):
    even_sum += integrand_temp_077[i] 

#stepsize, converted to frequency

h = wavelengths_integrand[1] - wavelengths_integrand[0]

integral077 = (h/3)*(integrand_temp_077[0] + integrand_temp_077[-1] + 4*odd_sum + 2*even_sum)



#%%

#integrating flux of 8.6 feature

#integrate from 8.1 to 8.9 microns

l_int = np.where(np.round(wavelengths_integrand_077, 3) == 8.1)[0][0]
u_int = np.where(np.round(wavelengths_integrand_077, 2) == 8.9)[0][0]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_086 = np.copy(everything_removed4[l_int:u_int])

wavelengths_integrand = wavelengths_integrand_077[l_int:u_int]

#wavepog has wrong shape, need to fix

temp = np.copy(wavelengths_integrand)

wavelengths_integrand = np.zeros(len(integrand_086))

for i in range(len(integrand_086)):
    wavelengths_integrand[i] = temp[i,0]

final_cube = np.zeros(integrand_086.shape)
cube_with_units = (integrand_086*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies =\
                                u.spectral_density(wavelengths_integrand*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_temp_086 = np.copy(integrand_086)
for i in range(len(integrand_086)):
    integrand_temp_086[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_temp_086), 2):
    odd_sum += integrand_temp_086[i] 

even_sum = 0    

for i in range(2, len(integrand_temp_086), 2):
    even_sum += integrand_temp_086[i] 

#stepsize, converted to frequency

h = wavelengths_integrand[1] - wavelengths_integrand[0]

integral086 = (h/3)*(integrand_temp_086[0] + integrand_temp_086[-1] + 4*odd_sum + 2*even_sum)



#%%
#for 6.2 feature, exclude 6.0 to 6.5

#first, need to fit continuum, do so by eyeballing a polynomial to it, adding 3 to some so everything lines up

pah_removed_1 = corrected_data2[:425]
pah_removed_2 = corrected_data2[1175:]
pah_removed_062 = np.concatenate((pah_removed_1, pah_removed_2))

wavelength_pah_removed_1 = wavelengths2[:425]
wavelength_pah_removed_2 = wavelengths2[1175:]
wavelength_pah_removed_062 = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))

#NOTE: not using my ransac function because i want to use a different
# wavelength array than the one used to make the fit

wavelengths_integral_062 = wavelength_pah_removed_062.reshape(-1,1)
data_integral_062 = pah_removed_062.reshape(-1,1)

wavelengths_integral_062 = wavelengths2.reshape(-1,1)

#ok so the polynomial fit sucks as the fit is somehow finding the 7.7 feature and subtracting it, so
#going to use a constant instead

#%%

continuum_removed2 = corrected_data2 - 0

everything_removed2 = rnf.emission_line_remover(continuum_removed2, 20, 1)


#%%

#RNF_062_continuum_extended_North

#%%

ax = plt.figure('RNF_062_continuum_extended_North', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, 6.2 feature, North', fontsize=20)
#plt.plot(wavelengths2, corrected_data2, label='Data')
plt.plot(wavelengths2, continuum_removed2, label='Continuum subtracted, continuum=+2')
plt.plot(wavelengths2, everything_removed2, label='Everything removed')
#plt.plot(wavelengths2, 2*np.ones(len(everything_removed2)), label='Continuum')
#plt.plot(wavelengths2, polynomial2, label='Continuum (not in use)')
plt.plot(wavelengths2, 0*everything_removed2, label='zero')


#plt.plot(wavelengths6, polynomial6, color='black', label='continuum')
#plt.plot(wavelengths7, polynomial7, color='black', label='continuum')

plt.plot(pah_wavelengths[pahoverlap_miri2_1:pahoverlap_miri2_2], 
         0.3*pah_data[pahoverlap_miri2_1:pahoverlap_miri2_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)

plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(5.6, 6.6, 0.1), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_062_continuum_extended_North.png')
plt.show()
plt.close()



#%%

#integrating flux of 6.2 feature



#integrate from 6.0 to 6.6 microns

l_int = np.where(np.round(wavelengths2, 3) == 6.0)[0][0]
u_int = np.where(np.round(wavelengths2, 3) == 6.5)[0][0]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_062 = np.copy(everything_removed2[l_int:u_int])

wavelengths_integrand = wavelengths2[l_int:u_int]

final_cube = np.zeros(integrand_062.shape)
cube_with_units = (integrand_062*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies =\
                                u.spectral_density(wavelengths_integrand*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_temp_062 = np.copy(integrand_062)
for i in range(len(integrand_062)):
    integrand_temp_062[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_temp_062), 2):
    odd_sum += integrand_temp_062[i] 

even_sum = 0    

for i in range(2, len(integrand_temp_062), 2):
    even_sum += integrand_temp_062[i] 

#stepsize, converted to frequency

h = wavelengths_integrand[1] - wavelengths_integrand[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral062 = (h/3)*(integrand_temp_062[0] + integrand_temp_062[-1] + 4*odd_sum + 2*even_sum)



#integrating flux of 6.2 feature for orion

#%%

#integrate from 6.0 to 6.6 microns

l_int = np.where(np.round(pah_wavelengths, 3) == 6.0)[0][0]
u_int = np.where(np.round(pah_wavelengths, 3) == 6.5)[0][0]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_062 = np.copy(pah_data[l_int:u_int])

wavepog = pah_wavelengths[l_int:u_int]

final_cube = np.zeros(integrand_062.shape)
cube_with_units = (integrand_062*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_062 = np.copy(integrand_062)
for i in range(len(integrand_062)):
    integrand_pog_062[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_062), 2):
    odd_sum += integrand_pog_062[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_062), 2):
    even_sum += integrand_pog_062[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral062_orion = (h/3)*(integrand_pog_062[0] + integrand_pog_062[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavepog) + 2 
#add 2 at the end to include endpoints

new_wavelengths_062 = np.linspace(l_int, u_int, int(number_of_wavelengths/2))





####################################



'''
FITTING GAUSSIAN TO 11.2 BACKGROUND
'''

#%%
'''
#combining 2 different channels, subtracting 3 so they line up

pah_removed, wavelength_pah_removed, overlap = rnf.flux_aligner2(
    wavelengths6, wavelengths7, corrected_background6 - 3, corrected_background7)

#first, need to fit continuum, do so by eyeballing a polynomial to it

pah_removed_1 = pah_removed[:868]
pah_removed_2 = pah_removed[1220:]
pah_removed = np.concatenate((pah_removed_1, pah_removed_2))

wavelength_pah_removed_1 = wavelength_pah_removed[:868]
wavelength_pah_removed_2 = wavelength_pah_removed[1220:]
wavelength_pah_removed = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))



#NOTE: not using my ransac function because i want to use a different
# wavelength array than the one used to make the fit

wavelength = wavelength_pah_removed.reshape(-1,1)
data = pah_removed.reshape(-1,1)

# Init the RANSAC regressor
ransac = make_pipeline(PolynomialFeatures(20), RANSACRegressor(max_trials=200000, random_state=41))

# Fit with RANSAC
ransac.fit(wavelength, data)

#now combining 2 different channels, with full data

pah_removed, wavelength_pah_removed, overlap = rnf.flux_aligner2(
    wavelengths6, wavelengths7, corrected_background6 - 3, corrected_background7)

wavelength = wavelength_pah_removed.reshape(-1,1)

# Get the fitted data result
polynomial6 = ransac.predict(wavelength)

polynomial6 = polynomial6.reshape((1, -1))[0]


#%%

continuum_removed_background6 = pah_removed - polynomial6

everything_removed_background6 = rnf.emission_line_remover(continuum_removed_background6, 20, 1)

#%%

#Figure 104

#%%

ax = plt.figure('Figure 104', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, 11.2 feature Background, North', fontsize=20)
#plt.plot(wavelengths6, corrected_background6-3, label='Ch2-long, data, offset=-3')
#plt.plot(wavelengths7, corrected_background7, label='Ch3-short, data')
plt.plot(wavelength_pah_removed, pah_removed, label='background')
plt.plot(wavelength_pah_removed, continuum_removed_background6, label='Continuum subtracted')
plt.plot(wavelength_pah_removed, everything_removed_background6, label='Lines and Continuum removed')

#plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
 #        3*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 6, label='Spitzer, scale=3, offset=-6', color='black', alpha=0.8)

plt.plot(wavelength_pah_removed, polynomial6, color='black', label='continuum')
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.03*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.03', color='r', alpha=0.5)
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(10, 13.6, 0.2), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11)
plt.savefig('Figures/RNF_112_continuum_extended_background_North.png')
plt.show()
plt.close()

#%%

#############

#integrating flux of 11.2 feature



#integrate from 11.1 to 11.6 microns

l_int = np.where(np.round(wavelengths6, 3) == 11.1)[0][0]
u_int = np.where(np.round(wavelengths6, 2) == 11.6)[-1][-1]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_112 = np.copy(everything_removed_background6[l_int:u_int])

#integrand = integrand*(u.MJy/u.sr)
wavepog = wavelengths6[l_int:u_int]#*(u.micron)


final_cube = np.zeros(integrand_112.shape)
cube_with_units = (integrand_112*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_112 = np.copy(integrand_112)
for i in range(len(integrand_112)):
    integrand_pog_112[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_112), 2):
    odd_sum += integrand_pog_112[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_112), 2):
    even_sum += integrand_pog_112[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral_background112 = (h/3)*(integrand_pog_112[0] + integrand_pog_112[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavepog) + 2 
#add 2 at the end to include endpoints

new_wavelengths_112 = np.linspace(l_int, u_int, int(number_of_wavelengths/2))
'''
'''
integrand2_112 = gaussian(new_wavelengths, 3.29, 0.0220/2, 1.8, 0) +\
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

integral2 = (h2/3)*(integrand_pog2[0] + integrand_pog2[-1] + 4*odd_sum2 + 2*even_sum2)

integral_error = (integral - integral2)/15

'''

#%%

#16.4 

#combining 2 different channels, subtracting 2 so they line up

pah_removed, wavelength_pah_removed, overlap = rnf.flux_aligner2(
    wavelengths8, wavelengths9, corrected_data8, corrected_data9 - 2)

#first, need to fit continuum, do so by eyeballing a polynomial to it

#if this feature exists its tiny, extending from 16.39 to 16.42

addition = len(wavelengths8)

pah_removed_1 = pah_removed[:addition+327]
pah_removed_2 = pah_removed[addition+337:]
pah_removed = np.concatenate((pah_removed_1, pah_removed_2))

wavelength_pah_removed_1 = wavelength_pah_removed[:addition+327]
wavelength_pah_removed_2 = wavelength_pah_removed[addition+337:]
wavelength_pah_removed = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))



#NOTE: not using my ransac function because i want to use a different
# wavelength array than the one used to make the fit

wavelength = wavelength_pah_removed.reshape(-1,1)
data = pah_removed.reshape(-1,1)

# Init the RANSAC regressor
ransac = make_pipeline(PolynomialFeatures(20), RANSACRegressor(max_trials=200000, random_state=41))

# Fit with RANSAC
ransac.fit(wavelength, data)

#now combining 2 different channels, with full data

pah_removed, wavelength_pah_removed, overlap = rnf.flux_aligner2(
    wavelengths8, wavelengths9, corrected_data8, corrected_data9 - 2)

wavelength = wavelength_pah_removed.reshape(-1,1)

# Get the fitted data result
polynomial9 = ransac.predict(wavelength)

polynomial9 = polynomial9.reshape((1, -1))[0]


#%%

continuum_removed9 = pah_removed - polynomial9

everything_removed9 = rnf.emission_line_remover(continuum_removed9, 20, 1)

#%%

#RNF_164_continuum_extended_North

#%%
pahoverlap_miri8_1 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths8[0], 2))[0][0]
pahoverlap_miri9_2 = np.where(np.round(pah_wavelengths, 2) == np.round(wavelengths9[-1], 2))[0][0]


ax = plt.figure('RNF_164_continuum_extended_North', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, 16.4 feature Background, North', fontsize=20)
#plt.plot(wavelengths6, corrected_background6-3, label='Ch2-long, data, offset=-3')
#plt.plot(wavelengths7, corrected_background7, label='Ch3-short, data')
plt.plot(wavelength_pah_removed, pah_removed, label='data')
plt.plot(wavelength_pah_removed, continuum_removed9, label='Continuum subtracted')
plt.plot(wavelength_pah_removed, everything_removed9, label='Lines and Continuum removed')
'''
plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
         3*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 6, label='Spitzer, scale=3, offset=-6', color='black', alpha=0.8)
'''
plt.plot(wavelength_pah_removed, polynomial9, color='black', label='continuum')
plt.plot(pah_wavelengths[pahoverlap_miri8_1:pahoverlap_miri9_2], 
         0.15*pah_data[pahoverlap_miri8_1:pahoverlap_miri9_2], 
         label='ISO orion spectra, scale=0.15', color='r', alpha=0.5)
plt.ylim((-10,150))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(13.4, 18.2, 0.4), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11)
plt.savefig('Figures/RNF_164_continuum_extended_North.png')
plt.show()
plt.close()

#%%

#############

#integrating flux of 16.4 feature



#integrate from 16.39 to 16.42 microns

#note: no evidence of 16.4 at this point so no point doing flux, currently the code below is copied from 11.2 background so will need
#to update it if i need 16.4 flux
'''
l_int = np.where(np.round(wavelengths6, 3) == 11.1)[0][0]
u_int = np.where(np.round(wavelengths6, 2) == 11.6)[-1][-1]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_112 = np.copy(everything_removed_background6[l_int:u_int])

#integrand = integrand*(u.MJy/u.sr)
wavepog = wavelengths6[l_int:u_int]#*(u.micron)


final_cube = np.zeros(integrand_112.shape)
cube_with_units = (integrand_112*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_112 = np.copy(integrand_112)
for i in range(len(integrand_112)):
    integrand_pog_112[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_112), 2):
    odd_sum += integrand_pog_112[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_112), 2):
    even_sum += integrand_pog_112[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral_background112 = (h/3)*(integrand_pog_112[0] + integrand_pog_112[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavepog) + 2 
#add 2 at the end to include endpoints

new_wavelengths_112 = np.linspace(l_int, u_int, int(number_of_wavelengths/2))

'''















#%%

#west area

#3.3


#%%

#RNF_033_gaussian_fit_west

#%%

ax = plt.figure('RNF_033_gaussian_fit_west', figsize=(16,6)).add_subplot(111)
plt.title('NIRSPEC Weighted Mean, gaussian fit, West', fontsize=20)
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], nirspec_weighted_mean4_west[:nirspec_cutoff] - 1.2, 
         label='g395m-f290, West, offset=-1.2', color='purple')
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
plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.17*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
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
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_033_gaussian_fit_west.png')
plt.show() 
plt.close()

#%%

#RNF_paper_033_gaussian_fit_west

#%%

ax = plt.figure('RNF_paper_033_gaussian_fit_west', figsize=(16,6)).add_subplot(111)
#plt.title('NIRSPEC Weighted Mean, gaussian fit, West', fontsize=20)
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], nirspec_weighted_mean4_west[:nirspec_cutoff] - 1.2, 
         label='g395m-f290, West, offset=-1.2', color='purple')
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
plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.17*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
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
plt.savefig('Figures/paper/RNF_paper_033_gaussian_fit_west.png')
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
#11.2
plt.figure()
plt.plot(wavelengths6_west, corrected_data6_west)
plt.plot(wavelengths7_west, corrected_data7_west + 2.5, alpha=0.5)
plt.ylim(-5,15)
plt.savefig('Figures/000_flux_alignment_west')
plt.show()
plt.close()
#%%
'''
#combining 2 different channels, subtracting 3 so they line up

pah_removed6_west, wavelength_pah_removed6_west, overlap = rnf.flux_aligner2(
    wavelengths6_west, wavelengths7_west, corrected_data6_west, corrected_data7_west - 1)

#first, need to fit continuum, do so by eyeballing a polynomial to it

pah_removed_1 = pah_removed6_west[:868]
pah_removed_2 = pah_removed6_west[1220:]
pah_removed6_west = np.concatenate((pah_removed_1, pah_removed_2))

wavelength_pah_removed_1 = wavelength_pah_removed6_west[:868]
wavelength_pah_removed_2 = wavelength_pah_removed6_west[1220:]
wavelength_pah_removed6_west = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))



#NOTE: not using my ransac function because i want to use a different
# wavelength array than the one used to make the fit

wavelength_continuum6_west = wavelength_pah_removed6_west.reshape(-1,1)
data_continuum6_west = pah_removed6_west.reshape(-1,1)

# Init the RANSAC regressor
ransac_continuum6_west = make_pipeline(PolynomialFeatures(20), RANSACRegressor(max_trials=200000, random_state=41))

# Fit with RANSAC
ransac_continuum6_west.fit(wavelength_continuum6_west, data_continuum6_west)

#now combining 2 different channels, with full data

pah_removed6_west, wavelength_pah_removed6_west, overlap = rnf.flux_aligner2(
    wavelengths6_west, wavelengths7_west, corrected_data6_west, corrected_data7_west - 1)

wavelength_continuum6_west = wavelength_pah_removed6_west.reshape(-1,1)

# Get the fitted data result
polynomial6_west = ransac_continuum6_west.predict(wavelength_continuum6_west)

polynomial6_west = polynomial6_west.reshape((1, -1))[0]


#%%

continuum_removed6_west = pah_removed6_west - polynomial6_west + 1 #continuum removes a bit too much

everything_removed6_west = rnf.emission_line_remover(continuum_removed6_west, 20, 1)
'''

#%%
#combining 3 different channels

pah_removed_112_west, wavelength_pah_removed_112_west, overlap = rnf.flux_aligner2(
    wavelengths5_west, wavelengths6_west, corrected_data5_west + 8.5, corrected_data6_west)

pah_removed_112_west, wavelength_pah_removed_112_west, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_112_west, wavelengths7_west, pah_removed_112_west, corrected_data7_west + 2.5)

#first, need to fit continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(wavelength_pah_removed_112_west, 2) == 11.13)[0][0] #avoids an absorbtion feature
temp_index_2 = np.where(np.round(wavelengths7_west, 2) == 11.85)[0][0]

#calculating the slope of the line to use

#preventing the value found from being on a line or something
pah_slope_1 = np.mean(pah_removed_112_west[temp_index_1 - 20:20+temp_index_1])

pah_slope_2 = np.mean(corrected_data7_west[temp_index_2 - 20:20+temp_index_2]) + 2.5

pah_slope = (pah_slope_2 - pah_slope_1)/\
    (wavelengths7_west[temp_index_2] - wavelength_pah_removed_112_west[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(pah_removed_112_west[:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(corrected_data7_west[temp_index_2:]))

pah_removed_3 = pah_slope*(wavelength_pah_removed_112_west[temp_index_1:overlap[0]] - 
                           wavelength_pah_removed_112_west[temp_index_1]) + pah_slope_1
pah_removed_4 = pah_slope*(wavelengths7_west[overlap[1]:temp_index_2] - 
                           wavelength_pah_removed_112_west[temp_index_1]) + pah_slope_1


'''
for i in range(len(pah_removed_3)):
    pah_removed_3[i] = pah_slope*i + pah_slope_1
'''

#putting it all together
pah_removed_112_west = np.concatenate((pah_removed_1, pah_removed_3))
pah_removed_112_west = np.concatenate((pah_removed_112_west, pah_removed_4))
pah_removed_112_west = np.concatenate((pah_removed_112_west, pah_removed_2))

pah_112_west, wavelength_pah_112_west, overlap = rnf.flux_aligner2(
    wavelengths5, wavelengths6, corrected_data5_west + 8.5, corrected_data6_west)

pah_112_west, wavelength_pah_112_west, overlap = rnf.flux_aligner2(
    wavelength_pah_112_west, wavelengths7, pah_112_west, corrected_data7_west + 2.5)



continuum_removed6_west = pah_112_west - pah_removed_112_west# - polynomial6

everything_removed6_west = rnf.emission_line_remover(continuum_removed6_west, 15, 1)


#%%

#RNF_112_continuum_extended_West

#%%

ax = plt.figure('RNF_112_continuum_extended_West', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, 11.2 feature, West', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
plt.plot(wavelength_pah_removed_112_west, pah_112_west, label='data')
plt.plot(wavelength_pah_removed_112_west, continuum_removed6_west, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112_west, everything_removed6_west, label='Lines and Continuum removed')

plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.3*continuum_removed_hh[hh_index_begin:hh_index_end2] - 3, 
         label='HorseHead Nebula spectra, scale=0.3, offset=-3', color='r', alpha=1)
plt.plot(ngc7027_wavelengths[ngc7027_index_begin:ngc7027_index_end2], 
         0.02*ngc7027_data[ngc7027_index_begin:ngc7027_index_end2] - 5, 
         label='NGC 7027 spectra, scale=0.02, offset=-5', color='purple', alpha=1)

plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
         1.2*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 3, label='Spitzer, scale=1.2, offset=-3', color='black', alpha=0.8)
plt.plot(wavelength_pah_removed_112_west, pah_removed_112_west, color='black', label='continuum')
'''
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.08*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.08', color='r', alpha=0.5)
'''
plt.plot(wavelength_pah_removed_112_west, 0*everything_removed6_west, color='black', label='zero')
plt.plot(11.0*np.ones(len(wavelength_pah_removed_112_west)), np.linspace(-10, 25, len(pah_removed_112_west)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(11.6*np.ones(len(wavelength_pah_removed_112_west)), np.linspace(-10, 25, len(pah_removed_112_west)), 
         color='black', label='upper integration bound (11.6)')
plt.ylim((-5,20))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(8.6, 13.6, 0.3), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11)
plt.savefig('Figures/RNF_112_continuum_extended_West.png')
plt.show()
plt.close()

#%%

#RNF_112_continuum_extended_West_simple

#%%

ax = plt.figure('RNF_112_continuum_extended_West_simple', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, 11.2 feature, West simple', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
plt.plot(wavelength_pah_removed_112_west, pah_112_west, label='data')
plt.plot(wavelength_pah_removed_112_west, continuum_removed6_west, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112_west, everything_removed6_west, label='Lines and Continuum removed')
plt.plot(wavelength_pah_removed_112_west, pah_removed_112_west, color='black', label='continuum')
plt.plot(wavelength_pah_removed_112_west, 0*everything_removed6_west, color='black', label='zero')
plt.plot(11.0*np.ones(len(wavelength_pah_removed_112_west)), np.linspace(-10, 25, len(pah_removed_112_west)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(11.6*np.ones(len(wavelength_pah_removed_112_west)), np.linspace(-10, 25, len(pah_removed_112_west)), 
         color='black', label='upper integration bound (11.6)')
plt.ylim((-5,20))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(8.6, 13.6, 0.3), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11)
plt.savefig('Figures/RNF_112_continuum_extended_West_simple.png')
plt.show()
plt.close()

#%%

#############

#integrating flux of 11.2 feature



#integrate from 11.0 to 11.6 microns

l_int = np.where(np.round(wavelength_pah_removed_112_west, 3) == 11.0)[0][0]
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








#for 7.7 amd 8.6 features, exclude 7.1 to 9.2 (funky business from 8.9 to 9.2 that i dont wanna fit so grouping
#it in with the 8.6 feature)



#%%
#first, need to fit continuum, do so by eyeballing a polynomial to it, adding 3 to some so everything lines up

pah_removed_1 = corrected_data3_west[:837] + 0
pah_removed_2 = corrected_data4_west + 9
pah_removed = np.concatenate((pah_removed_1, pah_removed_2))
pah_removed_2 = corrected_data5_west[408:] + 9
pah_removed = np.concatenate((pah_removed, pah_removed_2))

wavelength_pah_removed_1 = wavelengths3_west[:837]
wavelength_pah_removed_2 = wavelengths4_west
wavelength_pah_removed = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))
wavelength_pah_removed_2 = wavelengths5_west[408:]
wavelength_pah_removed = np.concatenate((wavelength_pah_removed, wavelength_pah_removed_2))

#NOTE: not using my ransac function because i want to use a different
# wavelength array than the one used to make the fit

wavelength = wavelength_pah_removed.reshape(-1,1)
data = pah_removed.reshape(-1,1)

# Init the RANSAC regressor
#ransac = make_pipeline(PolynomialFeatures(20), RANSACRegressor(max_trials=200000, random_state=41))

# Fit with RANSAC
#ransac.fit(wavelength, data)

wavelength_pah_removed_1 = wavelengths3_west
wavelength_pah_removed_2 = wavelengths4_west
wavelength_pah_removed = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))
wavelength_pah_removed_2 = wavelengths5_west
wavelength_pah_removed = np.concatenate((wavelength_pah_removed, wavelength_pah_removed_2))

wavelength = wavelength_pah_removed.reshape(-1,1)

# Get the fitted data result
#polynomial4 = ransac.predict(wavelength)

#polynomial4 = polynomial4.reshape((1, -1))[0]

#now that continuum has been found, redo combined array with nothing removed for subtraction

pah_removed_1 = corrected_data3_west + 0
pah_removed_2 = corrected_data4_west + 9
pah_removed = np.concatenate((pah_removed_1, pah_removed_2))
pah_removed_2 = corrected_data5_west + 9
pah_removed = np.concatenate((pah_removed, pah_removed_2))

#ok so the polynomial fit sucks as the fit is somehow finding the 7.7 feature and subtracting it, so
#going to use a constant instead

#%%

continuum_removed4_west = pah_removed - 5

everything_removed4_west = rnf.emission_line_remover(continuum_removed4_west, 20, 1)


#%%

#RNF_077_continuum_extended_West

#%%

ax = plt.figure('RNF_077_continuum_extended_West', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum of Data, 7.7 amd 8.6 features, West', fontsize=20)
plt.plot(wavelength, pah_removed, label='Data')
plt.plot(wavelength, continuum_removed4_west, label='Continuum subtracted')
plt.plot(wavelength, everything_removed4_west, label='Everything removed')
plt.plot(wavelength, 5*np.ones(len(everything_removed4_west)), label='Continuum (+5)')
#plt.plot(wavelength, polynomial4, label='Continuum (not in use)')
plt.plot(wavelength, 0*everything_removed4_west, label='zero')
plt.plot(pah_wavelengths[pahoverlap_miri3_1:pahoverlap_miri4_2], 
         0.15*pah_data[pahoverlap_miri3_1:pahoverlap_miri4_2], 
         label='ISO orion spectra, scale=0.15', color='r', alpha=0.5)
'''
plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
         3*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 6, label='Spitzer, scale=3, offset=-6', color='black', alpha=0.8)
'''
#plt.plot(wavelengths6, polynomial6, color='black', label='continuum')
#plt.plot(wavelengths7, polynomial7, color='black', label='continuum')
'''
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.3*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
'''
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(6.6, 10.2, 0.2), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_077_continuum_extended_West.png')
plt.show()
plt.close()





#%%

#integrating flux of 7.7 feature



#integrate from 7.2 to 8.1 microns

l_int = np.where(np.round(wavelength, 3) == 7.2)[0][0]
u_int = np.where(np.round(wavelength, 3) == 8.1)[0][0]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_077_west = np.copy(everything_removed4_west[l_int:u_int])

wavepog = wavelength[l_int:u_int]

#wavepog has wrong shape, need to fix

temp = np.copy(wavepog)

wavepog = np.zeros(len(integrand_077_west))

for i in range(len(integrand_077_west)):
    wavepog[i] = temp[i,0]

final_cube = np.zeros(integrand_077_west.shape)
cube_with_units = (integrand_077_west*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_077_west = np.copy(integrand_077_west)
for i in range(len(integrand_077_west)):
    integrand_pog_077_west[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_077_west), 2):
    odd_sum += integrand_pog_077_west[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_077_west), 2):
    even_sum += integrand_pog_077_west[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral077_west = (h/3)*(integrand_pog_077_west[0] + integrand_pog_077_west[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavepog) + 2 
#add 2 at the end to include endpoints

new_wavelengths_077_west = np.linspace(l_int, u_int, int(number_of_wavelengths/2))



#%%

#integrating flux of 8.6 feature



#integrate from 8.1 to 8.9 microns

l_int = np.where(np.round(wavelength, 3) == 8.1)[0][0]
u_int = np.where(np.round(wavelength, 2) == 8.9)[0][0]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_086_west = np.copy(everything_removed4_west[l_int:u_int])

wavepog = wavelength[l_int:u_int]

#wavepog has wrong shape, need to fix

temp = np.copy(wavepog)

wavepog = np.zeros(len(integrand_086_west))

for i in range(len(integrand_086_west)):
    wavepog[i] = temp[i,0]

final_cube = np.zeros(integrand_086_west.shape)
cube_with_units = (integrand_086_west*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_086_west = np.copy(integrand_086_west)
for i in range(len(integrand_086_west)):
    integrand_pog_086_west[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_086_west), 2):
    odd_sum += integrand_pog_086_west[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_086_west), 2):
    even_sum += integrand_pog_086_west[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral086_west = (h/3)*(integrand_pog_086_west[0] + integrand_pog_086_west[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavepog) + 2 
#add 2 at the end to include endpoints

new_wavelengths_086_west = np.linspace(l_int, u_int, int(number_of_wavelengths/2))


#%%

#%%



#for 6.2 feature, exclude 6.0 to 6.5




#%%

#first, need to fit continuum, do so by eyeballing a polynomial to it, adding 3 to some so everything lines up

pah_removed_1 = corrected_data2_west[:425] + 5
pah_removed_2 = corrected_data2_west[1175:] + 5
pah_removed = np.concatenate((pah_removed_1, pah_removed_2))

wavelength_pah_removed_1 = wavelengths2_west[:425]
wavelength_pah_removed_2 = wavelengths2_west[1175:]
wavelength_pah_removed = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))

#NOTE: not using my ransac function because i want to use a different
# wavelength array than the one used to make the fit

wavelength = wavelength_pah_removed.reshape(-1,1)
data = pah_removed.reshape(-1,1)

# Init the RANSAC regressor
#ransac = make_pipeline(PolynomialFeatures(20), RANSACRegressor(max_trials=200000, random_state=41))

# Fit with RANSAC
#ransac.fit(wavelength, data)

wavelength = wavelengths2_west.reshape(-1,1)

# Get the fitted data result
#polynomial2 = ransac.predict(wavelength)

#polynomial2 = polynomial2.reshape((1, -1))[0]

#ok so the polynomial fit sucks as the fit is somehow finding the 7.7 feature and subtracting it, so
#going to use a constant instead

#%%

continuum_removed2_west = corrected_data2_west + 9 - 0

everything_removed2_west = rnf.emission_line_remover(continuum_removed2_west, 20, 1)


#%%

#RNF_062_continuum_extended_West

#%%

ax = plt.figure('RNF_062_continuum_extended_West', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, 6.2 feature, West', fontsize=20)
plt.plot(wavelengths2_west, corrected_data2_west+5, label='Data')
plt.plot(wavelengths2_west, continuum_removed2_west, label='Continuum subtracted, continuum=0, +5 to data')
plt.plot(wavelengths2_west, everything_removed2_west, label='Everything removed')
#plt.plot(wavelengths2, 2*np.ones(len(everything_removed2)), label='Continuum')
#plt.plot(wavelengths2, polynomial2, label='Continuum (not in use)')
plt.plot(wavelengths2_west, 0*everything_removed2_west, label='zero')


#plt.plot(wavelengths6, polynomial6, color='black', label='continuum')
#plt.plot(wavelengths7, polynomial7, color='black', label='continuum')

plt.plot(pah_wavelengths[pahoverlap_miri2_1:pahoverlap_miri2_2], 
         0.15*pah_data[pahoverlap_miri2_1:pahoverlap_miri2_2], 
         label='ISO orion spectra, scale=0.15', color='r', alpha=0.5)

plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(5.6, 6.6, 0.1), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_062_continuum_extended_West.png')
plt.show()
plt.close()





#%%

#integrating flux of 6.2 feature



#integrate from 6.0 to 6.6 microns

l_int = np.where(np.round(wavelengths2_west, 3) == 6.0)[0][0]
u_int = np.where(np.round(wavelengths2_west, 3) == 6.5)[0][0]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_062_west = np.copy(everything_removed2_west[l_int:u_int])

wavepog = wavelengths2_west[l_int:u_int]

final_cube = np.zeros(integrand_062_west.shape)
cube_with_units = (integrand_062_west*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_062_west = np.copy(integrand_062_west)
for i in range(len(integrand_062_west)):
    integrand_pog_062_west[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_062_west), 2):
    odd_sum += integrand_pog_062_west[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_062_west), 2):
    even_sum += integrand_pog_062_west[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral062_west = (h/3)*(integrand_pog_062_west[0] + integrand_pog_062_west[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavepog) + 2 
#add 2 at the end to include endpoints

new_wavelengths_062_west = np.linspace(l_int, u_int, int(number_of_wavelengths/2))

















#%%

#west area, blob

#3.3


#%%

#RNF_033_gaussian_fit_west_blob

#%%
        
ax = plt.figure('RNF_033_gaussian_fit_west_blob', figsize=(16,6)).add_subplot(111)
plt.title('NIRSPEC Weighted Mean, gaussian fit, West Blob', fontsize=20)
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], nirspec_weighted_mean4_west_blob[:nirspec_cutoff] - 1.05, 
         label='g395m-f290, West, offset=-1.2', color='purple')
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
plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.18*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
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
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_033_gaussian_fit_west_blob.png')
plt.show() 
plt.close()

#%%

#RNF_paper_033_gaussian_fit_west_blob

#%%

ax = plt.figure('RNF_paper_033_gaussian_fit_west_blob', figsize=(16,6)).add_subplot(111)
#plt.title('NIRSPEC Weighted Mean, gaussian fit, West Blob', fontsize=20)
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], nirspec_weighted_mean4_west_blob[:nirspec_cutoff] - 1.05, 
         label='g395m-f290, West, offset=-1.2', color='purple')
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
plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.18*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
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
plt.savefig('Figures/paper/RNF_paper_033_gaussian_fit_west_blob.png')
plt.show() 

#%%

#############

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

plt.figure()
plt.plot(wavelengths6_west, corrected_data6_west_blob)
plt.plot(wavelengths7_west, corrected_data7_west_blob - 0.5)
plt.ylim(-10, 30)
plt.show()
plt.close()
#%%
#11.2

pah_removed_112_west_blob, wavelength_pah_removed_112_west_blob, overlap = rnf.flux_aligner2(
    wavelengths5_west, wavelengths6_west, corrected_data5_west_blob + 8.5, corrected_data6_west_blob)

pah_removed_112_west_blob, wavelength_pah_removed_112_west_blob, overlap = rnf.flux_aligner2(
    wavelength_pah_removed_112_west_blob, wavelengths7_west, 
    pah_removed_112_west_blob, corrected_data7_west_blob - 0.5)

#first, need to fit continuum, do so by fitting a linear function to it in relevant region

temp_index_1 = np.where(np.round(wavelength_pah_removed_112_west, 2) == 11.13)[0][0]
temp_index_2 = np.where(np.round(wavelengths7_west, 2) == 11.85)[0][0]

#calculating the slope of the line to use

#preventing the value found from being on a line or something
pah_slope_1 = np.mean(pah_removed_112_west_blob[temp_index_1 - 20:20+temp_index_1])

pah_slope_2 = np.mean(corrected_data7_west_blob[temp_index_2 - 20:20+temp_index_2]) - 0.5

pah_slope = (pah_slope_2 - pah_slope_1)/\
    (wavelengths7_west[temp_index_2] - wavelength_pah_removed_112_west_blob[temp_index_1])

#making area around bounds constant, note name is outdated
pah_removed_1 = pah_slope_1*np.ones(len(pah_removed_112_west_blob[:temp_index_1]))
pah_removed_2 = pah_slope_2*np.ones(len(corrected_data7_west_blob[temp_index_2:]))

pah_removed_3 = pah_slope*(wavelength_pah_removed_112_west_blob[temp_index_1:overlap[0]] - 
                           wavelength_pah_removed_112_west_blob[temp_index_1]) + pah_slope_1
                           
pah_removed_4 = pah_slope*(wavelengths7_west[overlap[1]:temp_index_2] - 
                           wavelength_pah_removed_112_west_blob[temp_index_1]) + pah_slope_1


'''
for i in range(len(pah_removed_3)):
    pah_removed_3[i] = pah_slope*i + pah_slope_1
'''

#putting it all together
pah_removed_112_west_blob = np.concatenate((pah_removed_1, pah_removed_3))
pah_removed_112_west_blob = np.concatenate((pah_removed_112_west_blob, pah_removed_4))
pah_removed_112_west_blob = np.concatenate((pah_removed_112_west_blob, pah_removed_2))

pah_112_west_blob, wavelength_pah_112_west_blob, overlap = rnf.flux_aligner2(
    wavelengths5, wavelengths6, corrected_data5_west_blob + 8.5, corrected_data6_west_blob)

pah_112_west_blob, wavelength_pah_112_west_blob, overlap = rnf.flux_aligner2(
    wavelength_pah_112_west_blob, wavelengths7, pah_112_west_blob, corrected_data7_west_blob - 0.5)

continuum_removed6_west_blob = pah_112_west_blob - pah_removed_112_west_blob# - polynomial6

everything_removed6_west_blob = rnf.emission_line_remover(continuum_removed6_west_blob, 15, 1)



#%%

#RNF_112_continuum_extended_West_H2_filament

#%%

ax = plt.figure('RNF_112_continuum_extended_H2_filament', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, 11.2 feature, H2 Filament', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
plt.plot(wavelength_pah_removed_112_west, pah_112_west_blob, label='data')
plt.plot(wavelength_pah_removed_112_west, continuum_removed6_west_blob, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112_west, everything_removed6_west_blob, label='Lines and Continuum removed')
plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.25*continuum_removed_hh[hh_index_begin:hh_index_end2] - 3, 
         label='HorseHead Nebula spectra, scale=0.25, offset=-3', color='r', alpha=1)
plt.plot(ngc7027_wavelengths[ngc7027_index_begin:ngc7027_index_end2], 
         0.02*ngc7027_data[ngc7027_index_begin:ngc7027_index_end2] - 5, 
         label='NGC 7027 spectra, scale=0.02, offset=-5', color='purple', alpha=1)

plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
         1.1*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 4, label='Spitzer, scale=1.1, offset=-4', color='black', alpha=0.8)

plt.plot(wavelength_pah_removed_112_west, pah_removed_112_west_blob, color='black', label='continuum')
'''
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.08*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.08', color='r', alpha=0.5)
'''
plt.plot(11.0*np.ones(len(wavelength_pah_removed_112_west)), np.linspace(-10, 25, len(pah_removed_112_west_blob)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(11.6*np.ones(len(wavelength_pah_removed_112_west)), np.linspace(-10, 25, len(pah_removed_112_west_blob)), 
         color='black', label='upper integration bound (11.6)')
plt.plot(wavelength_pah_removed_112_west, 0*everything_removed6_west_blob, color='black', label='zero')
plt.ylim((-5,20))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(8.6, 13.6, 0.3), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11)
plt.savefig('Figures/RNF_112_continuum_extended_H2_filament.png')
plt.show()
plt.close()

#%%

#RNF_112_continuum_extended_West_H2_filament_simple

#%%

ax = plt.figure('RNF_112_continuum_extended_West_H2_filament_simple', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, 11.2 feature, West H2 Filament Simple', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
plt.plot(wavelength_pah_removed_112_west, pah_112_west_blob, label='data')
#plt.plot(wavelength_pah_removed_112_west, continuum_removed6_west_blob, label='Continuum subtracted')
#plt.plot(wavelength_pah_removed_112_west, everything_removed6_west_blob, label='Lines and Continuum removed')


plt.plot(wavelength_pah_removed_112_west, pah_removed_112_west_blob, color='black', label='continuum')

plt.plot(11.0*np.ones(len(wavelength_pah_removed_112_west)), np.linspace(-10, 25, len(pah_removed_112_west_blob)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(11.6*np.ones(len(wavelength_pah_removed_112_west)), np.linspace(-10, 25, len(pah_removed_112_west_blob)), 
         color='black', label='upper integration bound (11.6)')
'''
plt.plot(wavelengths7_west[temp_index_2-20]*np.ones(len(wavelength_pah_removed_112_west)), 
         np.linspace(-10, 25, len(pah_removed_112_west_blob)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(wavelengths7_west[temp_index_2+20]*np.ones(len(wavelength_pah_removed_112_west)), 
         np.linspace(-10, 25, len(pah_removed_112_west_blob)), 
         color='black', label='upper integration bound (11.6)')
'''
plt.plot(wavelength_pah_removed_112_west, 0*everything_removed6_west_blob, color='black', label='zero')
plt.ylim((-5,20))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(8.6, 13.6, 0.3), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11)
plt.savefig('Figures/RNF_112_continuum_extended_West_H2_filament_simple.png')
plt.show()
plt.close()

#%%

#############

#integrating flux of 11.2 feature



#integrate from 11.0 to 11.6 microns

l_int = np.where(np.round(wavelength_pah_removed_112_west, 3) == 11.0)[0][0]
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









#for 7.7 amd 8.6 features, exclude 7.1 to 9.2 (funky business from 8.9 to 9.2 that i dont wanna fit so grouping
#it in with the 8.6 feature)



#%%
#first, need to fit continuum, do so by eyeballing a polynomial to it, adding 3 to some so everything lines up

pah_removed_1 = corrected_data3_west_blob[:837] + 0
pah_removed_2 = corrected_data4_west_blob + 9
pah_removed = np.concatenate((pah_removed_1, pah_removed_2))
pah_removed_2 = corrected_data5_west_blob[408:] + 9
pah_removed = np.concatenate((pah_removed, pah_removed_2))

wavelength_pah_removed_1 = wavelengths3_west[:837]
wavelength_pah_removed_2 = wavelengths4_west
wavelength_pah_removed = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))
wavelength_pah_removed_2 = wavelengths5_west[408:]
wavelength_pah_removed = np.concatenate((wavelength_pah_removed, wavelength_pah_removed_2))

#NOTE: not using my ransac function because i want to use a different
# wavelength array than the one used to make the fit

wavelength = wavelength_pah_removed.reshape(-1,1)
data = pah_removed.reshape(-1,1)

# Init the RANSAC regressor
#ransac = make_pipeline(PolynomialFeatures(20), RANSACRegressor(max_trials=200000, random_state=41))

# Fit with RANSAC
#ransac.fit(wavelength, data)

wavelength_pah_removed_1 = wavelengths3_west
wavelength_pah_removed_2 = wavelengths4_west
wavelength_pah_removed = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))
wavelength_pah_removed_2 = wavelengths5_west
wavelength_pah_removed = np.concatenate((wavelength_pah_removed, wavelength_pah_removed_2))

wavelength = wavelength_pah_removed.reshape(-1,1)

# Get the fitted data result
#polynomial4 = ransac.predict(wavelength)

#polynomial4 = polynomial4.reshape((1, -1))[0]

#now that continuum has been found, redo combined array with nothing removed for subtraction

pah_removed_1 = corrected_data3_west_blob + 0
pah_removed_2 = corrected_data4_west_blob + 9
pah_removed = np.concatenate((pah_removed_1, pah_removed_2))
pah_removed_2 = corrected_data5_west_blob + 9
pah_removed = np.concatenate((pah_removed, pah_removed_2))

#ok so the polynomial fit sucks as the fit is somehow finding the 7.7 feature and subtracting it, so
#going to use a constant instead

#%%

continuum_removed4_west_blob = pah_removed - 7

everything_removed4_west_blob = rnf.emission_line_remover(continuum_removed4_west_blob, 20, 1)


#%%

#RNF_077_continuum_extended_West_H2_filament

#%%

ax = plt.figure('RNF_077_continuum_extended_West_H2_filament', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum of Data, 7.7 amd 8.6 features, West H2 Filament', fontsize=20)
plt.plot(wavelength, pah_removed, label='Data')
plt.plot(wavelength, continuum_removed4_west_blob, label='Continuum subtracted')
plt.plot(wavelength, everything_removed4_west_blob, label='Everything removed')
plt.plot(wavelength, 3*np.ones(len(everything_removed4_west_blob)), label='Continuum (+7)') 
#plt.plot(wavelength, polynomial4, label='Continuum (not in use)')
plt.plot(wavelength, 0*everything_removed4_west_blob, label='zero')
plt.plot(pah_wavelengths[pahoverlap_miri3_1:pahoverlap_miri4_2], 
         0.15*pah_data[pahoverlap_miri3_1:pahoverlap_miri4_2], 
         label='ISO orion spectra, scale=0.15', color='r', alpha=0.5)
'''
plt.plot(spitzer_wavelengths[spitzer_index_begin:spitzer_index_end2], 
         3*spitzer_data[spitzer_index_begin:spitzer_index_end2] - 6, label='Spitzer, scale=3, offset=-6', color='black', alpha=0.8)
'''
#plt.plot(wavelengths6, polynomial6, color='black', label='continuum')
#plt.plot(wavelengths7, polynomial7, color='black', label='continuum')
'''
plt.plot(pah_wavelengths[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         0.3*pah_data[pahoverlap_miri6_1:pahoverlap_miri7_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
'''
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(6.6, 10.2, 0.2), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_077_continuum_extended_West_H2_filament.png')
plt.show()
plt.close()





#%%

#integrating flux of 7.7 feature



#integrate from 7.2 to 8.1 microns

l_int = np.where(np.round(wavelength, 3) == 7.2)[0][0]
u_int = np.where(np.round(wavelength, 3) == 8.1)[0][0]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_077_west_blob = np.copy(everything_removed4_west_blob[l_int:u_int])

wavepog = wavelength[l_int:u_int]

#wavepog has wrong shape, need to fix

temp = np.copy(wavepog)

wavepog = np.zeros(len(integrand_077_west_blob))

for i in range(len(integrand_077_west_blob)):
    wavepog[i] = temp[i,0]

final_cube = np.zeros(integrand_077_west_blob.shape)
cube_with_units = (integrand_077_west_blob*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_077_west_blob = np.copy(integrand_077_west_blob)
for i in range(len(integrand_077_west_blob)):
    integrand_pog_077_west_blob[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_077_west_blob), 2):
    odd_sum += integrand_pog_077_west_blob[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_077_west_blob), 2):
    even_sum += integrand_pog_077_west_blob[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral077_west_blob = (h/3)*(integrand_pog_077_west_blob[0] + integrand_pog_077_west_blob[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavepog) + 2 
#add 2 at the end to include endpoints

new_wavelengths_077_west_blob = np.linspace(l_int, u_int, int(number_of_wavelengths/2))



#%%

#integrating flux of 8.6 feature



#integrate from 8.1 to 8.9 microns

l_int = np.where(np.round(wavelength, 3) == 8.1)[0][0]
u_int = np.where(np.round(wavelength, 2) == 8.9)[0][0]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_086_west_blob = np.copy(everything_removed4_west_blob[l_int:u_int])

wavepog = wavelength[l_int:u_int]

#wavepog has wrong shape, need to fix

temp = np.copy(wavepog)

wavepog = np.zeros(len(integrand_086_west_blob))

for i in range(len(integrand_086_west_blob)):
    wavepog[i] = temp[i,0]

final_cube = np.zeros(integrand_086_west_blob.shape)
cube_with_units = (integrand_086_west_blob*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_086_west_blob = np.copy(integrand_086_west_blob)
for i in range(len(integrand_086_west_blob)):
    integrand_pog_086_west_blob[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_086_west_blob), 2):
    odd_sum += integrand_pog_086_west_blob[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_086_west_blob), 2):
    even_sum += integrand_pog_086_west_blob[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral086_west_blob = (h/3)*(integrand_pog_086_west_blob[0] + integrand_pog_086_west_blob[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavepog) + 2 
#add 2 at the end to include endpoints

new_wavelengths_086_west_blob = np.linspace(l_int, u_int, int(number_of_wavelengths/2))








#for 6.2 feature, exclude 6.0 to 6.5




#%%

#first, need to fit continuum, do so by eyeballing a polynomial to it, adding 3 to some so everything lines up

pah_removed_1 = corrected_data2_west_blob[:425] + 5
pah_removed_2 = corrected_data2_west_blob[1175:] + 5
pah_removed = np.concatenate((pah_removed_1, pah_removed_2))

wavelength_pah_removed_1 = wavelengths2_west[:425]
wavelength_pah_removed_2 = wavelengths2_west[1175:]
wavelength_pah_removed = np.concatenate((wavelength_pah_removed_1, wavelength_pah_removed_2))

#NOTE: not using my ransac function because i want to use a different
# wavelength array than the one used to make the fit

wavelength = wavelength_pah_removed.reshape(-1,1)
data = pah_removed.reshape(-1,1)

# Init the RANSAC regressor
#ransac = make_pipeline(PolynomialFeatures(20), RANSACRegressor(max_trials=200000, random_state=41))

# Fit with RANSAC
#ransac.fit(wavelength, data)

wavelength = wavelengths2_west.reshape(-1,1)

# Get the fitted data result
#polynomial2 = ransac.predict(wavelength)

#polynomial2 = polynomial2.reshape((1, -1))[0]

#ok so the polynomial fit sucks as the fit is somehow finding the 7.7 feature and subtracting it, so
#going to use a constant instead

#%%

continuum_removed2_west_blob = corrected_data2_west_blob + 5 - 0

everything_removed2_west_blob = rnf.emission_line_remover(continuum_removed2_west_blob, 20, 1)


#%%

#RNF_062_continuum_extended_West_H2_filament

#%%

ax = plt.figure('RNF_062_continuum_extended_West_H2_filament', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, 6.2 feature, West H2 Filament', fontsize=20)
#plt.plot(wavelengths2, corrected_data2, label='Data')
plt.plot(wavelengths2_west, continuum_removed2_west_blob, label='Continuum subtracted, continuum=0, +5 to data')
plt.plot(wavelengths2_west, everything_removed2_west_blob, label='Everything removed')
#plt.plot(wavelengths2, 2*np.ones(len(everything_removed2)), label='Continuum')
#plt.plot(wavelengths2, polynomial2, label='Continuum (not in use)')
plt.plot(wavelengths2_west, 0*everything_removed2_west_blob, label='zero')


#plt.plot(wavelengths6, polynomial6, color='black', label='continuum')
#plt.plot(wavelengths7, polynomial7, color='black', label='continuum')

plt.plot(pah_wavelengths[pahoverlap_miri2_1:pahoverlap_miri2_2], 
         0.15*pah_data[pahoverlap_miri2_1:pahoverlap_miri2_2], 
         label='ISO orion spectra, scale=0.15', color='r', alpha=0.5)

plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(5.6, 6.6, 0.1), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_062_continuum_extended_West_H2_filament.png')
plt.show()
plt.close()





#%%

#integrating flux of 6.2 feature



#integrate from 6.0 to 6.6 microns

l_int = np.where(np.round(wavelengths2_west, 3) == 6.0)[0][0]
u_int = np.where(np.round(wavelengths2_west, 3) == 6.5)[0][0]

#simspons rule

#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units

integrand_062_west_blob = np.copy(everything_removed2_west_blob[l_int:u_int])

wavepog = wavelengths2_west[l_int:u_int]

final_cube = np.zeros(integrand_062_west_blob.shape)
cube_with_units = (integrand_062_west_blob*10**6)*(u.Jy/u.sr)


final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies = u.spectral_density(wavepog*u.micron))

final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)

integrand_pog_062_west_blob = np.copy(integrand_062_west_blob)
for i in range(len(integrand_062_west_blob)):
    integrand_pog_062_west_blob[i] = float(final_cube[i])



odd_sum = 0

for i in range(1, len(integrand_pog_062_west_blob), 2):
    odd_sum += integrand_pog_062_west_blob[i] 

even_sum = 0    

for i in range(2, len(integrand_pog_062_west_blob), 2):
    even_sum += integrand_pog_062_west_blob[i] 

#stepsize, converted to frequency

h = wavepog[1] - wavepog[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))

integral062_west_blob = (h/3)*(integrand_pog_062_west_blob[0] + integrand_pog_062_west_blob[-1] + 4*odd_sum + 2*even_sum)

#now calculating the error, need an integral estimate with half the data points

number_of_wavelengths = len(wavepog) + 2 
#add 2 at the end to include endpoints

new_wavelengths_062_west_blob = np.linspace(l_int, u_int, int(number_of_wavelengths/2))





















####################################



'''
SIGNAL TO NOISE RATIO
'''

#%%

plt.figure()
plt.plot(wavelengths_nirspec4[148:176], nirspec_weighted_mean4[148:176] - 2.4)
plt.show()
plt.close()


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

#using 9.25 to 9.4 microns (446 - 562)
add077 = len(wavelengths3) + len(wavelengths4)

rms_data077 = rnf.unit_changer(everything_removed4[add077+446:add077+562], wavelengths5[446:562])

rms077 = rnf.RMS(rms_data077) 

delta_wave077 = fits.getheader(get_pkg_data_filename('data/north/ring_neb_north_ch2-medium_s3d.fits'), 1)["CDELT3"]

snr077 = rnf.SNR(integral077, rms077, delta_wave077, len(everything_removed4[446:562]))

error077 = integral077/snr077

print('7.7 feature:', integral077, '+/-', error077, 'W/m^2/sr, rms range 9.25 to 9.4 microns')

#using 9.25 to 9.4 microns (446 - 562)
add086 = len(wavelengths3) + len(wavelengths4)

rms_data086 = rnf.unit_changer(everything_removed4[add086+446:add086+562], wavelengths5[446:562])

rms086 = rnf.RMS(rms_data086) 

delta_wave086 = fits.getheader(get_pkg_data_filename('data/north/ring_neb_north_ch2-medium_s3d.fits'), 1)["CDELT3"]

snr086 = rnf.SNR(integral086, rms086, delta_wave086, len(everything_removed4[446:562]))

error086 = integral086/snr086

print('8.6 feature:', integral086, '+/-', error086, 'W/m^2/sr, rms range using 9.25 to 9.4 microns')

#using 6.51 to 6.56 microns (1063 - 1125)
rms_data062 = rnf.unit_changer(everything_removed2[1063:1125], wavelengths2[1063:1125])

rms062 = rnf.RMS(rms_data062) 

delta_wave062 = fits.getheader(get_pkg_data_filename('data/north/ring_neb_north_ch1-medium_s3d.fits'), 1)["CDELT3"]

snr062 = rnf.SNR(integral062, rms062, delta_wave062, len(everything_removed2[1063:1125]))

error062 = integral062/snr062

print('6.2 feature:', integral062, '+/-', error062, 'W/m^2/sr, rms range 6.51 to 6.56 microns')

#using 5.9 to 5.98 (5501 - 5580)
rms_data_orion062 = rnf.unit_changer(pah_data[5501:5580], pah_wavelengths[5501:5580])

rms_orion062 = rnf.RMS(rms_data_orion062)

delta_wave_orion062 = pah_wavelengths[5502] - pah_wavelengths[5501]

snr_orion062 = rnf.SNR(integral062_orion, rms_orion062, delta_wave_orion062, len(pah_data[5601:5680]))

error_orion062 = integral062_orion/snr_orion062

print('orion 6.2 feature:', integral062_orion, '+/-', error_orion062, 'W/m^2/sr, rms range 5.9 to 5.98 microns')
'''
#using 10.83 to 10.88 microns (623 - 662)
rms_data_background112 = rnf.unit_changer(everything_removed_background6[623:662], wavelengths6[623:662])

rms_background112 = rnf.RMS(rms_data_background112) 

delta_wave_background112 = fits.getheader(get_pkg_data_filename('data/MIRI_MRS/version2_030323/cubes/off-position/ring_neb_obs3_ch2-long_s3d.fits'), 1)["CDELT3"]

snr_background112 = rnf.SNR(integral_background112, rms_background112, delta_wave_background112, len(everything_removed_background6[623:662]))

error_background112 = integral_background112/snr_background112

print('11.2 feature background:', integral_background112, '+/-', error_background112, 'W/m^2/sr, rms range 10.83 to 10.88 microns')
'''













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

#using 9.25 to 9.4 microns (446 - 562)
add077_west = len(wavelengths3_west) + len(wavelengths4_west)

rms_data077_west = rnf.unit_changer(everything_removed4_west[add077_west+446:add077_west+562], wavelengths5_west[446:562])

rms077_west = rnf.RMS(rms_data077_west) 

delta_wave077_west = fits.getheader(get_pkg_data_filename('data/west/ring_neb_west_ch2-medium_s3d.fits'), 1)["CDELT3"]

snr077_west = rnf.SNR(integral077_west, rms077_west, delta_wave077_west, len(everything_removed4_west[446:562]))

error077_west = integral077_west/snr077_west

print('7.7 feature, west:', integral077_west, '+/-', error077_west, 'W/m^2/sr, rms range 9.25 to 9.4 microns')

#using 9.25 to 9.4 microns (446 - 562)
add086_west = len(wavelengths3_west) + len(wavelengths4_west)

rms_data086_west = rnf.unit_changer(everything_removed4_west[add086_west+446:add086_west+562], wavelengths5_west[446:562])

rms086_west = rnf.RMS(rms_data086_west) 

delta_wave086_west = fits.getheader(get_pkg_data_filename('data/west/ring_neb_west_ch2-medium_s3d.fits'), 1)["CDELT3"]

snr086_west = rnf.SNR(integral086_west, rms086_west, delta_wave086_west, len(everything_removed4_west[446:562]))

error086_west = integral086_west/snr086_west

print('8.6 feature, west:', integral086_west, '+/-', error086_west, 'W/m^2/sr, rms range using 9.25 to 9.4 microns')

#using 6.51 to 6.56 microns (1063 - 1125)
rms_data062_west = rnf.unit_changer(everything_removed2_west[1063:1125], wavelengths2_west[1063:1125])

rms062_west = rnf.RMS(rms_data062_west) 

delta_wave062_west = fits.getheader(get_pkg_data_filename('data/west/ring_neb_west_ch1-medium_s3d.fits'), 1)["CDELT3"]

snr062_west = rnf.SNR(integral062_west, rms062_west, delta_wave062_west, len(everything_removed2_west[1063:1125]))

error062_west = integral062_west/snr062_west

print('6.2 feature, west:', integral062_west, '+/-', error062_west, 'W/m^2/sr, rms range 6.51 to 6.56 microns')






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

#using 9.25 to 9.4 microns (446 - 562)
add077_west_blob = len(wavelengths3_west) + len(wavelengths4_west)

rms_data077_west_blob = rnf.unit_changer(everything_removed4_west_blob[add077_west+446:add077_west+562], wavelengths5_west[446:562])

rms077_west_blob = rnf.RMS(rms_data077_west_blob) 

delta_wave077_west_blob = fits.getheader(get_pkg_data_filename('data/west/ring_neb_west_ch2-medium_s3d.fits'), 1)["CDELT3"]

snr077_west_blob = rnf.SNR(integral077_west_blob, rms077_west_blob, delta_wave077_west_blob, len(everything_removed4_west_blob[446:562]))

error077_west_blob = integral077_west_blob/snr077_west_blob

print('7.7 feature, west blob:', integral077_west_blob, '+/-', error077_west_blob, 'W/m^2/sr, rms range 9.25 to 9.4 microns')

#using 9.25 to 9.4 microns (446 - 562)
add086_west = len(wavelengths3_west) + len(wavelengths4_west)

rms_data086_west_blob = rnf.unit_changer(everything_removed4_west_blob[add086_west+446:add086_west+562], wavelengths5_west[446:562])

rms086_west_blob = rnf.RMS(rms_data086_west_blob) 

delta_wave086_west_blob = fits.getheader(get_pkg_data_filename('data/west/ring_neb_west_ch2-medium_s3d.fits'), 1)["CDELT3"]

snr086_west_blob = rnf.SNR(integral086_west_blob, rms086_west_blob, delta_wave086_west_blob, len(everything_removed4_west_blob[446:562]))

error086_west_blob = integral086_west_blob/snr086_west_blob

print('8.6 feature, west blob:', integral086_west_blob, '+/-', error086_west_blob, 'W/m^2/sr, rms range using 9.25 to 9.4 microns')

#using 6.51 to 6.56 microns (1063 - 1125)
rms_data062_west_blob = rnf.unit_changer(everything_removed2_west_blob[1063:1125], wavelengths2_west[1063:1125])

rms062_west_blob = rnf.RMS(rms_data062_west_blob) 

delta_wave062_west_blob = fits.getheader(get_pkg_data_filename('data/west/ring_neb_west_ch1-medium_s3d.fits'), 1)["CDELT3"]

snr062_west_blob = rnf.SNR(integral062_west_blob, rms062_west_blob, delta_wave062_west_blob, len(everything_removed2_west_blob[1063:1125]))

error062_west_blob = integral062_west_blob/snr062_west_blob

print('6.2 feature, west blob:', integral062_west_blob, '+/-', error062_west_blob, 'W/m^2/sr, rms range 6.51 to 6.56 microns')



####################################



'''
3.4 FEATURE INVESTIGATION
'''



####################################



#%%

#RNF_034_investigation_North

#%%
#loading in 3.4 example specrtum

with open('data/misc/IRAS_22272_5435_posA_SW6.txt') as f:
    lines = f.readlines()

    
    wavelengths_034_1 = np.zeros(len(lines))
    data_034_1 = np.zeros(len(lines))
    
    #i dont know why i coded it this way instead of just doing the range len way
    i = 0
    for line in lines:
        temp = line.split(', ')
        wavelengths_034_1[i] = temp[0]
        data_034_1[i] = temp[1]
        i = i+1



#%%


ax = plt.figure('RNF_034_investigation_North', figsize=(16,6)).add_subplot(111)
plt.title('NIRSPEC Weighted Mean, North', fontsize=20)

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], nirspec_weighted_mean4[:nirspec_cutoff] - 2.4, 
         label='g395m-f290, North, offset=-2.4', color='green')

#plt.plot(wavelengths_034_1, 10*(data_034_1 + 1.51), label = 'IRAS 22272+5435 posA SW6, scale=10', color='r', alpha=0.5)

#3.3 feature fit

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.2465, 0.0375, 0.6, 0), color='purple')

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.29027, 0.0387, 2.15, 0), color='purple')

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.32821, 0.0264, 0.35, 0), color='purple')

#3.4 feature fit

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.3944, 0.0076, 0*0.3, 0), color='purple')
         #label ='gaussian fit mean=3.3944, fwhm=0.0076 scale=1', 

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.4031, 0.0216, 1.1, 0), color='purple')
         #label ='gaussian fit mean=3.405, fwhm=0.02691 scale=1', 

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.4242, 0.0139, 0.55, 0), color='purple')
         #label ='gaussian fit mean=3.4242, fwhm=0.0139, scale=1.1', 

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.4649, 0.0500, 0*0.35, 0), color='purple')
         #label ='gaussian fit mean=3.4649, fwhm=0.0500, scale=1.1', 

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.5164, 0.0224, 0*0.35, 0), color='purple')
         #label ='gaussian fit mean=3.5164, fwhm=0.0224, scale=1.1', 
'''
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.5609, 0.0352, 0.5, 0), 
         label ='gaussian fit mean=3.5609, fwhm=0.0352, scale=1.1')
'''
'''
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.4013, 0.2438, 0.1, 0), 
         label ='gaussian fit plateau mean=3.4013, fwhm=0.2438, scale=0.1')
'''

plt.plot(wavelengths_nirspec4[:nirspec_cutoff],
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.2465, 0.0375, 0.6, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.29027, 0.0387, 2.15, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.32821, 0.0264, 0.35, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.3944, 0.0076, 0*0.3, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.4031, 0.0216, 1.1, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.4242, 0.0139, 0.55, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.4649, 0.0500, 0*0.35, 0) +\
         gaussian(wavelengths_nirspec4[:nirspec_cutoff], 3.5164, 0.0224, 0*0.35, 0), 
         label='gaussian fit sum', color='black')


#plt.plot(wavelengths_034_1, 1*(data_034_1 + 1.5), label = 'IRAS_04296+3429, scale=3', color='r', alpha=0.5)

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 0*nirspec_weighted_mean4[:nirspec_cutoff],
         label = 'zero', color='black')
'''
plt.plot(3.39582*np.ones(500), np.linspace(-2, 5, 500), color='black', label = 'H2, 3.39582')
plt.plot(3.40416*np.ones(500), np.linspace(-2, 5, 500), color='black', label = 'H2, 3.40416')
'''

'''
plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.9*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         label='ISO orion spectra, scale=0.9', color='r', alpha=0.5)
'''

plt.ylim((-1,3))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(2.8, 4.0, 0.1), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_034_investigation_North.png')
plt.show()
plt.close() 

#%%

ax = plt.figure('RNF_034_investigation_West', figsize=(16,6)).add_subplot(111)
plt.title('NIRSPEC Weighted Mean, West', fontsize=20)

plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], nirspec_weighted_mean4_west[:nirspec_cutoff] - 1.0, 
         label='g395m-f290, West, offset=-2.0', color='green')

plt.plot(wavelengths_034_1, 5*(data_034_1 - 1.44), label = 'IRAS_04296+3429, scale=10', color='r', alpha=0.5)

plt.plot(wavelengths_034_1, 2*(data_034_1 - 1.44), label = 'IRAS_04296+3429, scale=3', color='r', alpha=0.5)

plt.plot(wavelengths_nirspec4[:nirspec_cutoff], 0*nirspec_weighted_mean4[:nirspec_cutoff],
         label = 'zero', color='black')

'''
plt.plot(3.39582*np.ones(500), np.linspace(-2, 5, 500), color='black', label = 'H2, 3.39582')
plt.plot(3.40416*np.ones(500), np.linspace(-2, 5, 500), color='black', label = 'H2, 3.40416')
'''

'''
plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.9*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         label='ISO orion spectra, scale=0.9', color='r', alpha=0.5)
'''

plt.ylim((-2,5))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(2.8, 4.0, 0.1), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_034_investigation_West.png')
plt.show()
plt.close() 

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
'''
temp_index_1 = np.where(np.round(wavelength_pah_removed, 2) == 7.1)[0][0]
temp_index_2 = np.where(np.round(wavelength_pah_removed, 2) == 8.9)[0][0]

continuum2 = 6.0*np.ones(len(pah_removed[temp_index_1:temp_index_2])) + 10
wave_cont2 = wavelength_pah_removed[temp_index_1:temp_index_2]

pah_temp = pah[temp_index_1:temp_index_2]

continuum_removed_2 = pah_temp - continuum2

everything_removed_2 = rnf.emission_line_remover(continuum_removed_2, 15, 3)
'''

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
'''
overlap_index = np.where(np.round(wavelength_pah_removed_112, 2) == 
                         (wavelength_pah_removed_112[overlap[0]] + wavelengths7[overlap[1]])/2)[0][0]
pah_slope_3 = np.mean(pah_removed_112[overlap_index - 20:20+overlap_index])
'''


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


'''
for i in range(len(pah_removed_3)):
    pah_removed_3[i] = pah_slope*i + pah_slope_1
'''

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



#adding nirspec to data

temp_index_1 = np.where(np.round(wavelengths_nirspec4, 2) == 3.1)[0][0]
temp_index_2 = np.where(np.round(wavelengths_nirspec4, 2) == 3.7)[0][0]

pah, wavelength_pah_big, overlap = rnf.flux_aligner2(
    wavelengths_nirspec4, wavelength_pah, nirspec_weighted_mean4 + 10, pah)

overlap_array.append((wavelengths_nirspec4[overlap[0]] + wavelength_pah[overlap[1]])/2)

overlap_array = np.array(overlap_array)

#adding nirspec continuum

nirspec_no_line = rnf.emission_line_remover(nirspec_weighted_mean4[temp_index_1:temp_index_2] - 2.4, 5, 3)

nirspec_continuum = 2.4*np.ones(len(nirspec_weighted_mean4[temp_index_1:temp_index_2])) + 10
wave_nirspec = wavelengths_nirspec4[temp_index_1:temp_index_2]

pah_removed = np.concatenate((nirspec_continuum, pah_removed))

#%%

plt.figure()
plt.plot(wavelengths1, lfilter(b, a, corrected_data1))
plt.plot(wavelengths2, lfilter(b, a, corrected_data2)+2, alpha=0.5)
plt.plot(wavelengths3, lfilter(b, a, corrected_data3)+5, alpha=0.5)
plt.plot(wavelengths4, lfilter(b, a, corrected_data4)+7, alpha=0.5)
plt.plot(wavelengths5, lfilter(b, a, corrected_data5)+4, alpha=0.5)
plt.plot(wavelengths6, lfilter(b, a, corrected_data6)+3, alpha=0.5)
plt.plot(wavelengths7, lfilter(b, a, corrected_data7)+3, alpha=0.5)
plt.plot(wavelength_pah_big, lfilter(b, a, pah)-10, alpha=0.5)
plt.ylim(0, 20)
plt.show()




#%%

#RNF_continuum_extended_North_simple

#%%

pahoverlap_low = np.where(np.round(pah_wavelengths, 2) == np.round(wavelength_pah_big[0], 2))[0][0]
pahoverlap_high = np.where(np.round(pah_wavelengths, 2) == np.round(wavelength_pah_big[-1], 2))[0][0]

ax = plt.figure('RNF_continuum_extended_North_simple', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, North, simple', fontsize=20)
plt.plot(wavelength_pah_big, pah, label='data, offset=+10')

plt.plot(wave_cont1, everything_removed_1, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2, everything_removed_2, color='red')
plt.plot(wave_cont3, everything_removed_3, color='red')
plt.plot(wave_nirspec, nirspec_no_line, color='red')



plt.plot(wave_cont1, continuum1, color='purple', label='continuum, offset=+10')
plt.plot(wave_cont2, continuum2, color='purple')
plt.plot(wave_cont3, continuum3, color='purple')
plt.plot(wave_nirspec, nirspec_continuum, color='purple')



plt.plot(wavelength_pah_big, 0*pah, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.13*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='green', alpha=1.0)

plt.scatter(overlap_array, -5*np.ones(len(overlap_array)), zorder=100, color='black', label='data overlap')

'''
plt.plot(11.0*np.ones(len(wavelength_pah_removed_112)), np.linspace(-10, 25, len(pah_removed_112)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(11.6*np.ones(len(wavelength_pah_removed_112)), np.linspace(-10, 25, len(pah_removed_112)), 
         color='black', label='upper integration bound (11.6)')
'''
#plt.ylim((-100,100))
plt.ylim((-10,35))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11)
plt.savefig('Figures/RNF_continuum_extended_North_simple.png')
plt.show()
plt.close()



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
#plt.plot(wavelength_pah_big, lfilter(b, a, pah_west)-20, alpha=0.5)
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



#adding nirspec to data

temp_index_1 = np.where(np.round(wavelengths_nirspec4_west, 2) == 3.1)[0][0]
temp_index_2 = np.where(np.round(wavelengths_nirspec4_west, 2) == 3.7)[0][0]

pah_west, wavelength_pah_big_west, overlap = rnf.flux_aligner2(
    wavelengths_nirspec4_west, wavelength_pah_west, nirspec_weighted_mean4_west + 20, pah_west)

overlap_array_west.append((wavelengths_nirspec4_west[overlap[0]] + wavelength_pah_west[overlap[1]])/2)

overlap_array_west = np.array(overlap_array_west)

#adding nirspec continuum

nirspec_no_line_west = rnf.emission_line_remover(nirspec_weighted_mean4_west[temp_index_1:temp_index_2] - 1.2, 5, 1)

nirspec_continuum_west = 1.2*np.ones(len(nirspec_weighted_mean4_west[temp_index_1:temp_index_2])) + 20
wave_nirspec_west = wavelengths_nirspec4_west[temp_index_1:temp_index_2]

pah_removed_west = np.concatenate((nirspec_continuum_west, pah_removed_west))

#%%

plt.figure()
plt.plot(wavelengths1, lfilter(b, a, corrected_data1_west))
plt.plot(wavelengths2, lfilter(b, a, corrected_data2_west)+1, alpha=0.5)
plt.plot(wavelengths3, lfilter(b, a, corrected_data3_west)+0, alpha=0.5)
plt.plot(wavelengths4, lfilter(b, a, corrected_data4_west)+4, alpha=0.5)
plt.plot(wavelengths5, lfilter(b, a, corrected_data5_west)+3, alpha=0.5)
plt.plot(wavelengths6, lfilter(b, a, corrected_data6_west)+0, alpha=0.5)
plt.plot(wavelengths7, lfilter(b, a, corrected_data7_west)-1, alpha=0.5)
plt.plot(wavelength_pah_big, lfilter(b, a, pah_west)-20, alpha=0.5)
plt.ylim(-10, 20)
plt.show()


#%%

#RNF_continuum_extended_West_simple

#%%

pahoverlap_low = np.where(np.round(pah_wavelengths, 2) == np.round(wavelength_pah_big_west[0], 2))[0][0]
pahoverlap_high = np.where(np.round(pah_wavelengths, 2) == np.round(wavelength_pah_big_west[-1], 2))[0][0]

ax = plt.figure('RNF_continuum_extended_West_simple', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, West simple', fontsize=20)
plt.plot(wavelength_pah_big_west, pah_west, label='data, offset=+10')

plt.plot(wave_cont1_west, everything_removed_1_west, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2_west, everything_removed_2_west, color='red')
plt.plot(wave_cont3_west, everything_removed_3_west, color='red')
plt.plot(wave_nirspec_west , nirspec_no_line_west, color='red')



plt.plot(wave_cont1_west, continuum1_west, color='purple', label='continuum, offset=+10')
plt.plot(wave_cont2_west, continuum2_west, color='purple')
plt.plot(wave_cont3_west, continuum3_west, color='purple')
plt.plot(wave_nirspec_west , nirspec_continuum_west, color='purple')



plt.plot(wavelength_pah_big_west, 0*pah_west, color='black', label='zero')


plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='green', alpha=1.0)

plt.scatter(overlap_array_west, -5*np.ones(len(overlap_array_west)), zorder=100, color='black', label='data overlap')


'''
plt.plot(11.0*np.ones(len(wavelength_pah_removed_112_west)), np.linspace(-10, 25, len(pah_removed_112_west)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(11.6*np.ones(len(wavelength_pah_removed_112_west)), np.linspace(-10, 25, len(pah_removed_112_west)), 
         color='black', label='upper integration bound (11.6)')
'''

plt.ylim((-10,45))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11)
plt.savefig('Figures/RNF_continuum_extended_West_simple.png')
plt.show()
plt.close()


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
#plt.plot(wavelength_pah_big, lfilter(b, a, pah_west)-10, alpha=0.5)
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



#adding nirspec to data

temp_index_1 = np.where(np.round(wavelengths_nirspec4_west, 2) == 3.1)[0][0]
temp_index_2 = np.where(np.round(wavelengths_nirspec4_west, 2) == 3.7)[0][0]

pah_west_blob, wavelength_pah_big_west_blob, overlap = rnf.flux_aligner2(
    wavelengths_nirspec4_west, wavelength_pah_west_blob, nirspec_weighted_mean4_west_blob + 20, pah_west_blob)

overlap_array_west_blob.append((wavelengths_nirspec4_west[overlap[0]] + wavelength_pah_west_blob[overlap[1]])/2)

overlap_array_west_blob = np.array(overlap_array_west_blob)

#adding nirspec continuum

nirspec_no_line_west_blob = rnf.emission_line_remover(
    nirspec_weighted_mean4_west_blob[temp_index_1:temp_index_2] - 1.05, 5, 3)

nirspec_continuum_west_blob = 1.05*np.ones(
    len(nirspec_weighted_mean4_west_blob[temp_index_1:temp_index_2])) + 20
wave_nirspec_west_blob = wavelengths_nirspec4_west[temp_index_1:temp_index_2]

pah_removed_west_blob = np.concatenate((nirspec_continuum_west_blob, pah_removed_west_blob))

#%%

plt.figure()
plt.plot(wavelength_pah_big_west_blob, lfilter(b, a, pah_west_blob)-20)
plt.plot(wave_cont3_west_blob, continuum3_west_blob-20)
plt.ylim(-10, 30)
plt.show()

#%%

plt.figure()
plt.plot(wavelength_pah_big_west, lfilter(b, a, pah_west)-20)
plt.plot(wave_cont3_west, continuum3_west-20)
plt.ylim(-10, 30)
plt.show()

#%%

plt.figure()
plt.plot(wavelength_pah_big, lfilter(b, a, pah)-10)
plt.plot(wave_cont3, continuum3-10)
plt.ylim(-10, 30)
plt.show()

#%%

#RNF_continuum_extended_West_H2_filament_simple

#%%

pahoverlap_low = np.where(np.round(pah_wavelengths, 2) == np.round(wavelength_pah_big_west_blob[0], 2))[0][0]
pahoverlap_high = np.where(np.round(pah_wavelengths, 2) == np.round(wavelength_pah_big_west_blob[-1], 2))[0][0]

ax = plt.figure('RNF_continuum_extended_West_H2_filament_simple', figsize=(16,6)).add_subplot(111)
plt.title('JWST Continuum Subtracted Data, West H2 Filament Simple', fontsize=20)
plt.plot(wavelength_pah_big_west_blob, pah_west_blob, label='data, offset=+20')

plt.plot(wave_cont1_west_blob, everything_removed_1_west_blob, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2_west_blob, everything_removed_2_west_blob, color='red')
plt.plot(wave_cont3_west_blob, everything_removed_3_west_blob, color='red')
plt.plot(wave_nirspec_west_blob, nirspec_no_line_west_blob, color='red')

plt.plot(wave_cont1_west_blob, continuum1_west_blob, color='purple', label='continuum, offset=+20')
plt.plot(wave_cont2_west_blob, continuum2_west_blob, color='purple')
plt.plot(wave_cont3_west_blob, continuum3_west_blob, color='purple')
plt.plot(wave_nirspec_west_blob, nirspec_continuum_west_blob, color='purple')

plt.plot(wavelength_pah_big_west_blob, 0*pah_west_blob, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.06*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.06', color='green', alpha=1.0)

plt.scatter(overlap_array_west_blob, -5*np.ones(len(overlap_array_west_blob)), zorder=100, color='black', label='data overlap')

'''
plt.plot(11.0*np.ones(len(wavelength_pah_removed_112_west)), np.linspace(-10, 25, len(pah_removed_112_west_blob)), 
         color='black', label='lower integration bound (11.0)')
plt.plot(11.6*np.ones(len(wavelength_pah_removed_112_west)), np.linspace(-10, 25, len(pah_removed_112_west_blob)), 
         color='black', label='upper integration bound (11.6)')
'''

plt.ylim((-10,45))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11)
plt.savefig('Figures/RNF_continuum_extended_West_H2_filament_simple.png')
plt.show()
plt.close()


#%%

plt.figure()
plt.plot(wavelengths1, lfilter(b, a, corrected_data1_west_blob))
plt.plot(wavelengths2, lfilter(b, a, corrected_data2_west_blob)+0, alpha=0.5)
plt.plot(wavelengths3, lfilter(b, a, corrected_data3_west_blob)-2, alpha=0.5)
plt.plot(wavelengths4, lfilter(b, a, corrected_data4_west_blob)+2, alpha=0.5)
plt.plot(wavelengths5, lfilter(b, a, corrected_data5_west_blob)+1, alpha=0.5)
plt.plot(wavelengths6, lfilter(b, a, corrected_data6_west_blob)-2, alpha=0.5)
plt.plot(wavelengths7, lfilter(b, a, corrected_data7_west_blob)-5, alpha=0.5)
plt.plot(wavelength_pah_big, lfilter(b, a, pah_west_blob)-20, alpha=0.5)
plt.ylim(-10, 20)
plt.show()

#%%

#RNF_paper_continuum_extended_simple

#%%

ax = plt.figure('RNF_paper_continuum_extended_simple', figsize=(18,9)).add_subplot(311)
plt.subplots_adjust(right=0.8, left=0.1)
'''
North
'''

#plt.title('JWST Continuum Subtracted Data, North, simple', fontsize=20)

plt.plot(wavelength_pah_big, pah, label='data, offset=+10')

plt.plot(wave_cont1, everything_removed_1, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2, everything_removed_2, color='red')
plt.plot(wave_cont3, everything_removed_3, color='red')
plt.plot(wave_nirspec, nirspec_no_line, color='red')



plt.plot(wave_cont1, continuum1, color='purple', label='continuum, offset=+10')
plt.plot(wave_cont2, continuum2, color='purple')
plt.plot(wave_cont3, continuum3, color='purple')
plt.plot(wave_nirspec, nirspec_continuum, color='purple')



plt.plot(wavelength_pah_big, 0*pah, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.13*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='green', alpha=1.0)

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

plt.plot(wavelength_pah_big_west, pah_west, label='data, offset=+10')

plt.plot(wave_cont1_west, everything_removed_1_west, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2_west, everything_removed_2_west, color='red')
plt.plot(wave_cont3_west, everything_removed_3_west, color='red')
plt.plot(wave_nirspec_west , nirspec_no_line_west, color='red')



plt.plot(wave_cont1_west, continuum1_west, color='purple', label='continuum, offset=+10')
plt.plot(wave_cont2_west, continuum2_west, color='purple')
plt.plot(wave_cont3_west, continuum3_west, color='purple')
plt.plot(wave_nirspec_west , nirspec_continuum_west, color='purple')



plt.plot(wavelength_pah_big_west, 0*pah_west, color='black', label='zero')


plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='green', alpha=1.0)

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

plt.plot(wavelength_pah_big_west_blob, pah_west_blob, label='data, offset=+20')

plt.plot(wave_cont1_west_blob, everything_removed_1_west_blob, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2_west_blob, everything_removed_2_west_blob, color='red')
plt.plot(wave_cont3_west_blob, everything_removed_3_west_blob, color='red')
plt.plot(wave_nirspec_west_blob, nirspec_no_line_west_blob, color='red')

plt.plot(wave_cont1_west_blob, continuum1_west_blob, color='purple', label='continuum, offset=+20')
plt.plot(wave_cont2_west_blob, continuum2_west_blob, color='purple')
plt.plot(wave_cont3_west_blob, continuum3_west_blob, color='purple')
plt.plot(wave_nirspec_west_blob, nirspec_continuum_west_blob, color='purple')

plt.plot(wavelength_pah_big_west_blob, 0*pah_west_blob, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.06*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.06', color='green', alpha=1.0)

plt.scatter(overlap_array_west_blob, -5*np.ones(len(overlap_array_west_blob)), zorder=100, color='black', label='data overlap')

plt.ylim((-10,45))
#ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
#ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=11, title='West H2 Filament', bbox_to_anchor=(1.02, 1), loc='upper left')



plt.savefig('Figures/paper/RNF_paper_continuum_extended_simple.png')
plt.show()




#%%

#RNF_paper_continuum_extended_simple_no_legend

#%%

ax = plt.figure('RNF_paper_continuum_extended_simple_no_legend', figsize=(18,12)).add_subplot(1,1,1)

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
3.3
'''

ax = plt.subplot(3,4,1)

plt.subplots_adjust(wspace=0.3, hspace=0.2)

plt.plot(wave_cont1, everything_removed_1, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2, everything_removed_2, color='red')
plt.plot(wave_cont3, everything_removed_3, color='red')
plt.plot(wave_nirspec, nirspec_no_line, color='red')

plt.plot(wavelength_pah_big, 0*pah, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.13*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='green', alpha=1.0)

for data in overlap_array:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-0.5,4))
plt.xlim((3.1, 3.7))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'North', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
plt.xticks(np.arange(3.1, 3.7, 0.15), fontsize=14)
plt.yticks(fontsize=14)

'''
6.2
'''

ax = plt.subplot(3,4,2)

plt.plot(wave_cont1, everything_removed_1, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2, everything_removed_2, color='red')
plt.plot(wave_cont3, everything_removed_3, color='red')
plt.plot(wave_nirspec, nirspec_no_line, color='red')

plt.plot(wavelength_pah_big, 0*pah, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.13*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='green', alpha=1.0)

for data in overlap_array:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-1,15))
plt.xlim((5.6, 6.6))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.6, 6.6, 0.25), fontsize=14)
plt.yticks(fontsize=14)

'''
7.7
'''

ax = plt.subplot(3,4,3)

plt.plot(wave_cont1, everything_removed_1, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2, everything_removed_2, color='red')
plt.plot(wave_cont3, everything_removed_3, color='red')
plt.plot(wave_nirspec, nirspec_no_line, color='red')

plt.plot(wavelength_pah_big, 0*pah, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.13*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='green', alpha=1.0)

for data in overlap_array:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-1,15))
plt.xlim((7.1, 8.9))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(7.1, 8.9, 0.3), fontsize=14)
plt.yticks(fontsize=14)

'''
11.2
'''

ax = plt.subplot(3,4,4)

plt.plot(wave_cont1, everything_removed_1, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2, everything_removed_2, color='red')
plt.plot(wave_cont3, everything_removed_3, color='red')
plt.plot(wave_nirspec, nirspec_no_line, color='red')

plt.plot(wavelength_pah_big, 0*pah, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.13*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='green', alpha=1.0)

for data in overlap_array:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-1,13))
plt.xlim((10.6, 11.8))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
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
3.3
'''

ax = plt.subplot(3,4,5)

plt.plot(wave_cont1_west, everything_removed_1_west, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2_west, everything_removed_2_west, color='red')
plt.plot(wave_cont3_west, everything_removed_3_west, color='red')
plt.plot(wave_nirspec_west , nirspec_no_line_west, color='red')

plt.plot(wavelength_pah_big_west, 0*pah_west, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='green', alpha=1.0)

for data in overlap_array_west:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-0.25,1.75))
plt.xlim((3.1, 3.7))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'West', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.1, 3.7, 0.15), fontsize=14)
plt.yticks(fontsize=14)

'''
6.2
'''

ax = plt.subplot(3,4,6)

plt.plot(wave_cont1_west, everything_removed_1_west, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2_west, everything_removed_2_west, color='red')
plt.plot(wave_cont3_west, everything_removed_3_west, color='red')
plt.plot(wave_nirspec_west , nirspec_no_line_west, color='red')

plt.plot(wavelength_pah_big_west, 0*pah_west, color='black', label='zero')


plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='green', alpha=1.0)

for data in overlap_array_west:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-1,11))
plt.xlim((5.6, 6.6))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(5.6, 6.6, 0.25), fontsize=14)
plt.yticks(fontsize=14)

'''
7.7
'''

ax = plt.subplot(3,4,7)

plt.plot(wave_cont1_west, everything_removed_1_west, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2_west, everything_removed_2_west, color='red')
plt.plot(wave_cont3_west, everything_removed_3_west, color='red')
plt.plot(wave_nirspec_west , nirspec_no_line_west, color='red')

plt.plot(wavelength_pah_big_west, 0*pah_west, color='black', label='zero')


plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='green', alpha=1.0)

for data in overlap_array_west:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-1,11))
plt.xlim((7.1, 8.9))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(7.1, 8.9, 0.3), fontsize=14)
plt.yticks(fontsize=14)

'''
11.2
'''

ax = plt.subplot(3,4,8)

plt.plot(wave_cont1_west, everything_removed_1_west, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2_west, everything_removed_2_west, color='red')
plt.plot(wave_cont3_west, everything_removed_3_west, color='red')
plt.plot(wave_nirspec_west , nirspec_no_line_west, color='red')

plt.plot(wavelength_pah_big_west, 0*pah_west, color='black', label='zero')


plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='green', alpha=1.0)

for data in overlap_array_west:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-0.5,6))
plt.xlim((10.6, 11.8))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.6, 11.8, 0.3), fontsize=14)
plt.yticks(fontsize=14)


'''
West H2 Filament
'''

'''
3.3
'''

ax = plt.subplot(3,4,9)

plt.plot(wave_cont1_west_blob, everything_removed_1_west_blob, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2_west_blob, everything_removed_2_west_blob, color='red')
plt.plot(wave_cont3_west_blob, everything_removed_3_west_blob, color='red')
plt.plot(wave_nirspec_west_blob, nirspec_no_line_west_blob, color='red')

plt.plot(wavelength_pah_big_west_blob, 0*pah_west_blob, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.06*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.06', color='green', alpha=1.0)

for data in overlap_array_west_blob:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-0.25,1.75))
plt.xlim((3.1, 3.7))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'H2 Filament', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
plt.xticks(np.arange(3.1, 3.7, 0.15), fontsize=14)
plt.yticks(fontsize=14)

'''
6.2
'''

ax = plt.subplot(3,4,10)

plt.plot(wave_cont1_west_blob, everything_removed_1_west_blob, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2_west_blob, everything_removed_2_west_blob, color='red')
plt.plot(wave_cont3_west_blob, everything_removed_3_west_blob, color='red')
plt.plot(wave_nirspec_west_blob, nirspec_no_line_west_blob, color='red')

plt.plot(wavelength_pah_big_west_blob, 0*pah_west_blob, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.06*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.06', color='green', alpha=1.0)

for data in overlap_array_west_blob:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-1,15))
plt.xlim((5.6, 6.6))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#plt.xlabel('Wavelength (micron)', fontsize=16)
plt.xticks(np.arange(5.6, 6.6, 0.25), fontsize=14)
plt.yticks(fontsize=14)

'''
7.7
'''

ax = plt.subplot(3,4,11)

plt.plot(wave_cont1_west_blob, everything_removed_1_west_blob, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2_west_blob, everything_removed_2_west_blob, color='red')
plt.plot(wave_cont3_west_blob, everything_removed_3_west_blob, color='red')
plt.plot(wave_nirspec_west_blob, nirspec_no_line_west_blob, color='red')

plt.plot(wavelength_pah_big_west_blob, 0*pah_west_blob, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.06*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.06', color='green', alpha=1.0)

for data in overlap_array_west_blob:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-1,15))
plt.xlim((7.1, 8.9))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(7.1, 8.9, 0.3), fontsize=14)
plt.yticks(fontsize=14)

'''
11.2
'''

ax = plt.subplot(3,4,12)

plt.plot(wave_cont1_west_blob, everything_removed_1_west_blob, color='red', label='Lines and Continuum removed')
plt.plot(wave_cont2_west_blob, everything_removed_2_west_blob, color='red')
plt.plot(wave_cont3_west_blob, everything_removed_3_west_blob, color='red')
plt.plot(wave_nirspec_west_blob, nirspec_no_line_west_blob, color='red')

plt.plot(wavelength_pah_big_west_blob, 0*pah_west_blob, color='black', label='zero')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.06*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.06', color='green', alpha=1.0)

for data in overlap_array_west_blob:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')

plt.ylim((-0.5,6))
plt.xlim((10.6, 11.8))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
plt.xticks(np.arange(10.6, 11.8, 0.3), fontsize=14)
plt.yticks(fontsize=14)




plt.savefig('Figures/paper/RNF_paper_continuum_extended_simple_no_legend.pdf', bbox_inches='tight')
plt.show()




#%%
'''
plt.figure()
#plt.plot(wavelength_pah_big, pah_west-20)
plt.plot(wavelengths1, data1_west+10, color='blue', label = 'data+10')
plt.plot(wavelengths2, data2_west+4+10, color='blue')
plt.plot(wavelengths1, corrected_background1+10, color='orange', label = 'background+10')
plt.plot(wavelengths2, corrected_background2+10, color='orange')
plt.plot(wavelengths1, corrected_data1_west, color='purple', label = 'background subtracted')
plt.plot(wavelengths2, corrected_data2_west+4, color='purple')
#plt.plot(wavelengths1, data1_west - corrected_background1, alpha=0.5)
plt.ylim(-20, 40)
plt.legend()
plt.show()
plt.close()
'''
#%%

with fits.open('data/north/ring_neb_north_ch1-medium_s3d.fits') as hdul:
    pog = hdul[0].header
    frog = hdul[1].header
    

    


#%%

#RNF_paper_data_extended_simple_no_legend

#%%

ax = plt.figure('RNF_paper_data_extended_simple_no_legend', figsize=(18,12)).add_subplot(311)
#plt.subplots_adjust(right=0.9, left=0.1)

'''
North
'''

#plt.title('JWST Continuum Subtracted Data, North, simple', fontsize=20)

plt.plot(wavelength_pah_big, pah-10, label='data')

#plt.plot(wave_cont1, everything_removed_1, color='red', label='Lines and Continuum removed')
#plt.plot(wave_cont2, everything_removed_2, color='red')
#plt.plot(wave_cont3, everything_removed_3, color='red')
#plt.plot(wave_nirspec, nirspec_no_line, color='red')



#plt.plot(wave_cont1, continuum1-10, color='purple', label='continuum')
#plt.plot(wave_cont2, continuum2-10, color='purple')
plt.plot(wave_cont3, continuum3-10, color='purple', label='continuum')
plt.plot(wave_nirspec, nirspec_continuum-10, color='purple')


'''
plt.plot(wavelength_pah_big, 0*pah, color='black', label='zero')
'''
#plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
#         0.13*pah_data[pahoverlap_low:pahoverlap_high], 
#         label='ISO orion spectra, scale=0.13', color='green', alpha=1.0)

for data in overlap_array:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')
#plt.scatter(overlap_array, -5*np.ones(len(overlap_array)), zorder=100, color='black', label='data overlap')



plt.ylim((-0,25))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'North', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
#plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(3, 12)
#plt.legend(fontsize=11, title='North Common', bbox_to_anchor=(1.02, 1), loc='upper left')

'''
West
'''

ax = plt.subplot(312)
#plt.title('JWST Continuum Subtracted Data, West simple', fontsize=20)

plt.plot(wavelength_pah_big_west, pah_west-20, label='data')

#plt.plot(wave_cont1_west, everything_removed_1_west, color='red', label='Lines and Continuum removed')
#plt.plot(wave_cont2_west, everything_removed_2_west, color='red')
#plt.plot(wave_cont3_west, everything_removed_3_west, color='red')
#plt.plot(wave_nirspec_west , nirspec_no_line_west, color='red')



#plt.plot(wave_cont1_west, continuum1_west-20, color='purple', label='continuum')
#plt.plot(wave_cont2_west, continuum2_west-20, color='purple')
plt.plot(wave_cont3_west, continuum3_west-20, color='purple', label='continuum')
plt.plot(wave_nirspec_west , nirspec_continuum_west-20, color='purple')


'''
plt.plot(wavelength_pah_big_west, 0*pah_west, color='black', label='zero')
'''

#plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
#         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
#         label='ISO orion spectra, scale=0.045', color='green', alpha=1.0)

for data in overlap_array_west:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')
#plt.scatter(overlap_array_west, -5*np.ones(len(overlap_array_west)), zorder=100, color='black', label='data overlap')

plt.ylim((-5,15))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'West', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
#plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(3, 12)
#plt.legend(fontsize=11, title='West Common', bbox_to_anchor=(1.02, 1), loc='upper left')

'''
West H2 Filament
'''

ax = plt.subplot(313)
#plt.title('JWST Continuum Subtracted Data, West H2 Filament Simple', fontsize=20)

plt.plot(wavelength_pah_big_west_blob, pah_west_blob-20, label='data')

#plt.plot(wave_cont1_west_blob, everything_removed_1_west_blob, color='red', label='Lines and Continuum removed')
#plt.plot(wave_cont2_west_blob, everything_removed_2_west_blob, color='red')
#plt.plot(wave_cont3_west_blob, everything_removed_3_west_blob, color='red')
#plt.plot(wave_nirspec_west_blob, nirspec_no_line_west_blob, color='red')

#plt.plot(wave_cont1_west_blob, continuum1_west_blob-20, color='purple', label='continuum')
#plt.plot(wave_cont2_west_blob, continuum2_west_blob-20, color='purple')
plt.plot(wave_cont3_west_blob, continuum3_west_blob-20, color='purple', label='continuum')
plt.plot(wave_nirspec_west_blob, nirspec_continuum_west_blob-20, color='purple')
'''
plt.plot(wavelength_pah_big_west_blob, 0*pah_west_blob, color='black', label='zero')
'''
#plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
#         0.06*pah_data[pahoverlap_low:pahoverlap_high], 
#         label='ISO orion spectra, scale=0.06', color='green', alpha=1.0)

for data in overlap_array_west_blob:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')
#plt.scatter(overlap_array_west_blob, -5*np.ones(len(overlap_array_west_blob)), zorder=100, color='black', label='data overlap')

plt.ylim((-5,20))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'H2 Filament', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.xlim(3, 12)
plt.yticks(fontsize=14)
#plt.legend(fontsize=11, title='West H2 Filament', bbox_to_anchor=(1.02, 1), loc='upper left')



plt.savefig('Figures/paper/RNF_paper_data_extended_simple_no_legend.pdf', bbox_inches='tight')
plt.show()
















#######################################



#%%

ax = plt.figure('RNF_paper_data_extended_simple_no_legend_seminar', figsize=(18,9)).add_subplot(111)
#plt.subplots_adjust(right=0.9, left=0.1)

'''
North
'''

#plt.title('JWST Continuum Subtracted Data, North, simple', fontsize=20)

plt.plot(wavelength_pah_big, pah-10, label='data')

#plt.plot(wave_cont1, everything_removed_1, color='red', label='Lines and Continuum removed')
#plt.plot(wave_cont2, everything_removed_2, color='red')
#plt.plot(wave_cont3, everything_removed_3, color='red')
#plt.plot(wave_nirspec, nirspec_no_line, color='red')



plt.plot(wave_cont1, continuum1-10, color='purple', label='continuum')
plt.plot(wave_cont2, continuum2-10, color='purple')
plt.plot(wave_cont3, continuum3-10, color='purple')
plt.plot(wave_nirspec, nirspec_continuum-10, color='purple')



plt.plot(wavelength_pah_big, 0*pah, color='black', label='zero')

#plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
#         0.13*pah_data[pahoverlap_low:pahoverlap_high], 
#         label='ISO orion spectra, scale=0.13', color='green', alpha=1.0)

for data in overlap_array:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')
#plt.scatter(overlap_array, -5*np.ones(len(overlap_array)), zorder=100, color='black', label='data overlap')

plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.13*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.13', color='green', alpha=1.0)

plt.ylim((-5,25))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'North', transform=ax.transAxes, fontsize=14,
 #       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=11, title='North Common', bbox_to_anchor=(1.02, 1), loc='upper left')

'''
West
'''
'''
ax = plt.subplot(312)
#plt.title('JWST Continuum Subtracted Data, West simple', fontsize=20)

plt.plot(wavelength_pah_big_west, pah_west-20, label='data')

#plt.plot(wave_cont1_west, everything_removed_1_west, color='red', label='Lines and Continuum removed')
#plt.plot(wave_cont2_west, everything_removed_2_west, color='red')
#plt.plot(wave_cont3_west, everything_removed_3_west, color='red')
#plt.plot(wave_nirspec_west , nirspec_no_line_west, color='red')



plt.plot(wave_cont1_west, continuum1_west-20, color='purple', label='continuum')
plt.plot(wave_cont2_west, continuum2_west-20, color='purple')
plt.plot(wave_cont3_west, continuum3_west-20, color='purple')
plt.plot(wave_nirspec_west , nirspec_continuum_west-20, color='purple')



plt.plot(wavelength_pah_big_west, 0*pah_west, color='black', label='zero')


#plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
#         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
#         label='ISO orion spectra, scale=0.045', color='green', alpha=1.0)

for data in overlap_array_west:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')
#plt.scatter(overlap_array_west, -5*np.ones(len(overlap_array_west)), zorder=100, color='black', label='data overlap')




plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.045*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.045', color='green', alpha=1.0)

plt.ylim((-20,30))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.text(0.375, 0.2, 'West', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
#plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=11, title='West Common', bbox_to_anchor=(1.02, 1), loc='upper left')

'''
#West H2 Filament
'''

ax = plt.subplot(313)
#plt.title('JWST Continuum Subtracted Data, West H2 Filament Simple', fontsize=20)

plt.plot(wavelength_pah_big_west_blob, pah_west_blob-20, label='data')

#plt.plot(wave_cont1_west_blob, everything_removed_1_west_blob, color='red', label='Lines and Continuum removed')
#plt.plot(wave_cont2_west_blob, everything_removed_2_west_blob, color='red')
#plt.plot(wave_cont3_west_blob, everything_removed_3_west_blob, color='red')
#plt.plot(wave_nirspec_west_blob, nirspec_no_line_west_blob, color='red')

plt.plot(wave_cont1_west_blob, continuum1_west_blob-20, color='purple', label='continuum')
plt.plot(wave_cont2_west_blob, continuum2_west_blob-20, color='purple')
plt.plot(wave_cont3_west_blob, continuum3_west_blob-20, color='purple')
plt.plot(wave_nirspec_west_blob, nirspec_continuum_west_blob-20, color='purple')

plt.plot(wavelength_pah_big_west_blob, 0*pah_west_blob, color='black', label='zero')





plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
         0.06*pah_data[pahoverlap_low:pahoverlap_high], 
         label='ISO orion spectra, scale=0.06', color='green', alpha=1.0)

#plt.plot(pah_wavelengths[pahoverlap_low:pahoverlap_high], 
#         0.06*pah_data[pahoverlap_low:pahoverlap_high], 
#         label='ISO orion spectra, scale=0.06', color='green', alpha=1.0)

for data in overlap_array_west_blob:
    plt.plot([data, data], [-100, 100], color='black', alpha=0.5, linestyle='dashed')
#plt.scatter(overlap_array_west_blob, -5*np.ones(len(overlap_array_west_blob)), zorder=100, color='black', label='data overlap')

plt.ylim((-20,30))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.text(0.375, 0.2, 'H2 Filament', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
plt.xlabel('Wavelength (micron)', fontsize=16)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.0, 13.5, 0.5), fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=11, title='West H2 Filament', bbox_to_anchor=(1.02, 1), loc='upper left')

'''

plt.savefig('Figures/RNF_paper_data_extended_simple_no_legend_seminar.png')
plt.show()






























#%%

#RNF_paper_033_gaussian_fit

#%%
ax = plt.figure('RNF_paper_033_gaussian_fit', figsize=(18,18)).add_subplot(311)

'''
North
'''

#plt.title('NIRSPEC Weighted Mean, gaussian fit, North', fontsize=20)
plt.plot(wavelengths_nirspec4[:nirspec_cutoff], nirspec_weighted_mean4[:nirspec_cutoff] - 2.4, 
         label='g395m-f290, North, offset=-2.4', color='purple')

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

plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.38*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         label='ISO orion spectra, scale=0.38', color='r', alpha=0.5)
plt.ylim((-0.5,4))
plt.xlim((3.1, 3.6))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
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
plt.yticks(fontsize=28)
#plt.legend(fontsize=14)

'''
West
'''

ax = plt.subplot(312)
#ax = plt.figure('RNF_paper_033_gaussian_fit_west', figsize=(18,9)).add_subplot(111)
#plt.title('NIRSPEC Weighted Mean, gaussian fit, West', fontsize=20)
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], nirspec_weighted_mean4_west[:nirspec_cutoff] - 1.2, 
         label='g395m-f290, West, offset=-1.2', color='purple')
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
plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.17*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         label='ISO orion spectra, scale=0.17', color='r', alpha=0.5)
plt.ylim((-0.5,2))
plt.xlim((3.1, 3.6))
ax.tick_params(axis='x', which='major', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'West', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
#plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=32)
plt.xticks(np.arange(3.1, 3.6, 0.05), fontsize=28)
plt.yticks(fontsize=28)
#plt.legend(fontsize=14)

'''
West H2 Filament
'''

ax = plt.subplot(313)
#ax = plt.figure('RNF_paper_033_gaussian_fit_west_blob', figsize=(16,6)).add_subplot(111)
#plt.title('NIRSPEC Weighted Mean, gaussian fit, West Blob', fontsize=20)
plt.plot(wavelengths_nirspec4_west[:nirspec_cutoff], nirspec_weighted_mean4_west_blob[:nirspec_cutoff] - 1.05, 
         label='g395m-f290, West, offset=-1.2', color='purple')
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
plt.plot(pah_wavelengths[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         0.18*pah_data[pahoverlap_nirspec4_1:pahoverlap_nirspec4_2], 
         label='ISO orion spectra, scale=0.18', color='r', alpha=0.5)
plt.ylim((-0.5,2))
plt.xlim((3.1, 3.6))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=5, width=2)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=5, width=2)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
#ax.text(0.375, 0.2, 'H2 Filament', transform=ax.transAxes, fontsize=14,
#        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
plt.xlabel('Wavelength (micron)', fontsize=32)
#plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(3.1, 3.6, 0.05), fontsize=28)
plt.yticks(fontsize=28)
#plt.legend(fontsize=14)



plt.savefig('Figures/paper/RNF_paper_033_gaussian_fit.pdf', bbox_inches='tight')
plt.show() 



















'''
MISC STUFF
'''

#%%
'''
file_loc =  'data/nirspec_dec2022/jw01558-o008_t007_nirspec_g235m-f170lp_s3d_masked.fits'

with fits.open(file_loc) as hdul:

    print(hdul[1].header['CRVAL1'])
    print(hdul[1].header['CRVAL2'])
    print(hdul[1].header['CRPIX1'])
    print(hdul[1].header['CRPIX2'])
    print(hdul[1].header['CDELT1'])
    print(hdul[1].header['CDELT2'])
    
file_loc =  'data/west/ring_neb_west_ch2-long_s3d.fits'

with fits.open(file_loc) as hdul:

    print(hdul[1].header['CRVAL1'])
    print(hdul[1].header['CRVAL2'])
    print(hdul[1].header['CRPIX1'])
    print(hdul[1].header['CRPIX2'])
    print(hdul[1].header['CDELT1'])
    print(hdul[1].header['CDELT2'])


'''


#%%

#integral morphology bugtesting


'''
        temp_index_1 = np.where(np.round(wavelength_pah_removed_112, 2) == 11.13)[0][0]
        temp_index_2 = np.where(np.round(wavelength_pah_removed_112, 2) == 11.85)[0][0]

        #calculating the slope of the line to use

        #preventing the value found from being on a line or something
        #for the lower bound, taking the weighted mean all the way up to 10.0 microns because
        #the fit tends to suck over here
        pah_slope_1 = np.mean(to_integrate_west[:temp_index_1,i,j])

        pah_slope_2 = np.mean(to_integrate_west[temp_index_2 - 20:20+temp_index_2,i,j])
        
        pah_slope = (pah_slope_2 - pah_slope_1)/(temp_index_2 - temp_index_1)

        #making area around bounds constant, note name is outdated
        pah_removed_1 = pah_slope_1*np.ones(len(pah_removed_112[:temp_index_1]))
        pah_removed_2 = pah_slope_2*np.ones(len(pah_removed_112[temp_index_2:]))
        pah_removed_3 = pah_removed_112[temp_index_1:temp_index_2]



        for k in range(len(pah_removed_3)):
            pah_removed_3[k] = pah_slope*k + pah_slope_1


        #putting it all together
        continuum_112_west = np.concatenate((pah_removed_1, pah_removed_3))
        continuum_112_west = np.concatenate((continuum_112_west, pah_removed_2))
        
        continuum_array_west[:,i,j] = continuum_112_west
        



#%%
i = 16
j = 36
#simspons rule
        
#working with frequency, but can work with this function as i only change x to freq and this is y, already in freq units
        
integrand_112 = np.copy(to_integrate_north[l_int:u_int, i, j] - continuum_array_north[l_int:u_int,i,j])
        
#integrand = integrand*(u.MJy/u.sr)
wavelengths_integrand = wavelengths6[l_int:u_int]#*(u.micron)
        
        
final_cube = np.zeros(integrand_112.shape)
cube_with_units = (integrand_112*10**6)*(u.Jy/u.sr)
        
        
final_cube = cube_with_units.to(u.W/((u.m**2)*u.micron*u.sr), equivalencies =\
                                u.spectral_density(wavelengths_integrand*u.micron))
        
final_cube = final_cube*(u.micron)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.m)
final_cube = final_cube*(u.sr/u.W)
        
integrand_temp_112 = np.copy(integrand_112)
for k in range(len(integrand_112)):
    integrand_temp_112[k] = float(final_cube[k])
        
        
        
odd_sum = 0
        
for k in range(1, len(integrand_temp_112), 2):
    odd_sum += integrand_temp_112[k] 
        
even_sum = 0    
        
for k in range(2, len(integrand_temp_112), 2):
    even_sum += integrand_temp_112[k] 
        
#stepsize, converted to frequency
        
h = wavelengths_integrand[1] - wavelengths_integrand[0]
#h = c/((wavelengths_nirspec4[1] - wavelengths_nirspec4[0]))
        
integral = (h/3)*(integrand_temp_112[0] + integrand_temp_112[-1] + 4*odd_sum + 2*even_sum)


'''

#%%

mean_data6, pog = rnf.weighted_mean_finder_simple(image_data6, error_data6)

plt.figure()
plt.plot(wavelengths6, data6)
plt.plot(wavelengths6, corrected_data6)
plt.plot(wavelengths6, mean_data6)
plt.ylim(-10, 60)
plt.show()
plt.close()




#%%

#RNF_weighted_mean_extended_North_small

#%%

#background subtracted, entire relevant wavelength range, north small

ax = plt.figure('RNF_weighted_mean_extended_North_small', figsize=(16,10)).add_subplot(111)
plt.title('Background subtracted Spectra, North small')
plt.plot(wavelengths_nirspec4, nirspec_weighted_mean4 - 0, label='Ch1-medium, offset=0')
plt.plot(wavelengths1, corrected_data1 - 0, label='Ch1-short, offset=0')
plt.plot(wavelengths2, corrected_data2 - 0, label='Ch1-medium, offset=0')
plt.plot(wavelengths3, corrected_data3 - 0, label='Ch1-long, offset=0')
plt.plot(wavelengths4, corrected_data4 - 0, label='Ch2-short, offset=0')
plt.plot(wavelengths5, corrected_data5 - 5, label='Ch2-medium, offset=-5')
plt.plot(wavelengths6, corrected_data6 - 5, label='Ch2-long, offset=-5')
plt.plot(wavelengths7, corrected_data7 - 5, label='Ch3-short, offset=-5',)
plt.plot(wavelengths8, corrected_data8 - 30, label='Ch3-medium, offset=-30',)
plt.plot(pah_wavelengths[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         0.3*pah_data[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(2.8, 15.2, 0.8), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_weighted_mean_extended_North_small.png')
plt.show()
plt.close()



#%%
#weighted means of entire FOV
mean_data1, weighted_mean_error1 = rnf.weighted_mean_finder_simple(image_data1, error_data1)
mean_data2, weighted_mean_error2 = rnf.weighted_mean_finder_simple(image_data2, error_data2)
mean_data3, weighted_mean_error3 = rnf.weighted_mean_finder_simple(image_data3, error_data3)
mean_data4, weighted_mean_error4 = rnf.weighted_mean_finder_simple(image_data4, error_data4)
mean_data5, weighted_mean_error5 = rnf.weighted_mean_finder_simple(image_data5, error_data5)
mean_data6, weighted_mean_error6 = rnf.weighted_mean_finder_simple(image_data6, error_data6)
mean_data7, weighted_mean_error7 = rnf.weighted_mean_finder_simple(image_data7, error_data7)
mean_data8, weighted_mean_error8 = rnf.weighted_mean_finder_simple(image_data8, error_data8)


#%%

#RNF_weighted_mean_extended_North_big

#%%
'''
#background subtracted, entire relevant wavelength range, north big

ax = plt.figure('RNF_weighted_mean_extended_North_big', figsize=(16,10)).add_subplot(111)
plt.title('Background subtracted Spectra, North big')
plt.plot(wavelengths_nirspec4, nirspec_weighted_mean4 - 0, label='Ch1-medium, offset=0')
plt.plot(wavelengths1, mean_data1 - background1 - 0, label='Ch1-short, offset=0')
plt.plot(wavelengths2, mean_data2 - background2 - 0, label='Ch1-medium, offset=0')
plt.plot(wavelengths3, mean_data3 - background3 - 0, label='Ch1-long, offset=0')
plt.plot(wavelengths4, mean_data4 - background4 - 0, label='Ch2-short, offset=0')
plt.plot(wavelengths5, mean_data5 - background5 - 5, label='Ch2-medium, offset=-5')
plt.plot(wavelengths6, mean_data6 - background6 - 5, label='Ch2-long, offset=-5')
plt.plot(wavelengths7, mean_data7 - background7 - 5, label='Ch3-short, offset=-5',)
plt.plot(wavelengths8, mean_data8 - background8 - 30, label='Ch3-medium, offset=-30',)
plt.plot(pah_wavelengths[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         0.3*pah_data[pahoverlap_miri2_1:pahoverlap_miri8_2], 
         label='ISO orion spectra, scale=0.3', color='r', alpha=0.5)
plt.ylim((-10,30))
ax.tick_params(axis='x', which='both', labeltop='on', labelbottom='on', top='True')
ax.tick_params(axis='y', which='both', labelleft='on', labelright='on')
plt.xlabel('Wavelength (micron)', fontsize=16)
plt.ylabel('Flux (MJy/sr)', fontsize=16)
plt.xticks(np.arange(2.8, 15.2, 0.8), fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.savefig('Figures/RNF_weighted_mean_extended_North_big.png')
plt.show()
plt.close()
'''


#%%

#921.312

#583.556


file_loc= 'data/west/ring_neb_west_ch3-medium_s3d.fits'

image_file = get_pkg_data_filename(file_loc)
    
header_index=1

science_header = fits.getheader(image_file, header_index)


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

#everything_removed6 = rnf.emission_line_remover(continuum_removed6, 15, 1)
#everything_removed6 = rnf.absorption_line_remover(everything_removed6, 15, 1)

plt.figure()
plt.plot(spitzer_wavelengths, spitzer_data)
plt.plot(spitzer_wavelengths, pah_removed_spitzer)
plt.xlim(10, 13)
plt.ylim(0, 20)
plt.show()
plt.close()

#%%

'''
North
'''

ax = plt.figure('RNF_paper_112_comparison', figsize=(18,12)).add_subplot(311)
#plt.title('JWST Continuum Subtracted Data, 11.2 feature, North', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
#plt.plot(wavelength_pah_removed_112, pah_112, label='data')
#plt.plot(wavelength_pah_removed_112, continuum_removed6, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112, everything_removed6, label='Lines and Continuum removed')
plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.42*continuum_removed_hh[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.42', color='r', alpha=0.5)

plt.plot(spitzer_wavelengths, 
         2.35*continuum_removed_spitzer, 
         label='Spitzer spectra, scale=0.42', color='black')

'''
plt.plot(ngc7027_wavelengths[ngc7027_index_begin:ngc7027_index_end2], 
         0.033*ngc7027_data[ngc7027_index_begin:ngc7027_index_end2] - 8, 
         label='NGC 7027 spectra, scale=0.033, offset=-8', color='purple', alpha=1)
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

ax = plt.figure('RNF_paper_112_comparison', figsize=(16,6)).add_subplot(312)
#plt.title('JWST Continuum Subtracted Data, 11.2 feature, North', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
#plt.plot(wavelength_pah_removed_112, pah_112, label='data')
#plt.plot(wavelength_pah_removed_112, continuum_removed6, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112, everything_removed6_west, label='Lines and Continuum removed')
plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.18*continuum_removed_hh[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.18', color='r', alpha=0.5)

plt.plot(spitzer_wavelengths, 
         0.85*continuum_removed_spitzer, 
         label='Spitzer spectra, scale=0.42', color='black')

'''
plt.plot(ngc7027_wavelengths[ngc7027_index_begin:ngc7027_index_end2], 
         0.033*ngc7027_data[ngc7027_index_begin:ngc7027_index_end2] - 8, 
         label='NGC 7027 spectra, scale=0.033, offset=-8', color='purple', alpha=1)
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

ax = plt.figure('RNF_paper_112_comparison', figsize=(16,6)).add_subplot(313)
#plt.title('JWST Continuum Subtracted Data, 11.2 feature, North', fontsize=20)
#plt.plot(wavelengths6, corrected_data6 - 2, label='Ch2-long, data, offset=-2')
#plt.plot(wavelengths7, corrected_data7, label='Ch3-short, data')
#plt.plot(wavelength_pah_removed_112, pah_112, label='data')
#plt.plot(wavelength_pah_removed_112, continuum_removed6, label='Continuum subtracted')
plt.plot(wavelength_pah_removed_112, everything_removed6_west_blob, label='Lines and Continuum removed')
plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.22*continuum_removed_hh[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.18', color='r', alpha=0.5)

plt.plot(spitzer_wavelengths, 
         1.10*continuum_removed_spitzer, 
         label='Spitzer spectra, scale=0.42', color='black')

'''
plt.plot(ngc7027_wavelengths[ngc7027_index_begin:ngc7027_index_end2], 
         0.033*ngc7027_data[ngc7027_index_begin:ngc7027_index_end2] - 8, 
         label='NGC 7027 spectra, scale=0.033, offset=-8', color='purple', alpha=1)
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
plt.savefig('Figures/paper/RNF_paper_112_comparison.pdf', bbox_inches='tight')
plt.show()














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
         label='NGC 7027 spectra, scale=0.033, offset=-8', color='purple', alpha=1)
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
plt.plot(wavelength_pah_removed_112, everything_removed6_west_smooth, label='Lines and Continuum removed')
plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.18*continuum_removed_hh[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.18', color='r', alpha=0.5)

plt.plot(spitzer_wavelengths, 
         0.85*continuum_removed_spitzer, 
         label='Spitzer spectra, scale=0.42', color='black')

'''
plt.plot(ngc7027_wavelengths[ngc7027_index_begin:ngc7027_index_end2], 
         0.033*ngc7027_data[ngc7027_index_begin:ngc7027_index_end2] - 8, 
         label='NGC 7027 spectra, scale=0.033, offset=-8', color='purple', alpha=1)
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
plt.plot(wavelength_pah_removed_112, everything_removed6_west_blob_smooth, label='Lines and Continuum removed')
plt.plot(wavelength_pah_hh[hh_index_begin:hh_index_end2], 
         0.22*continuum_removed_hh[hh_index_begin:hh_index_end2] - 0, 
         label='HorseHead Nebula spectra, scale=0.18', color='r', alpha=0.5)

plt.plot(spitzer_wavelengths, 
         1.10*continuum_removed_spitzer, 
         label='Spitzer spectra, scale=0.42', color='black')

'''
plt.plot(ngc7027_wavelengths[ngc7027_index_begin:ngc7027_index_end2], 
         0.033*ngc7027_data[ngc7027_index_begin:ngc7027_index_end2] - 8, 
         label='NGC 7027 spectra, scale=0.033, offset=-8', color='purple', alpha=1)
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

plt.plot(wavelength_pah_removed_112, everything_removed6_west, label='Lines and Continuum removed')
plt.plot(wavelengths6_west, corrected_data6_west, label='Lines and Continuum removed')
plt.plot(wavelengths7_west, corrected_data7_west+2.5, label='Lines and Continuum removed')
plt.plot(wavelength_pah_removed_112, pah_112_west)
plt.plot(wavelength_pah_removed_112, pah_removed_112_west)
plt.ylim(0, 20)
plt.show()
plt.close()

#%%

plt.figure()

plt.plot(wavelengths1_west, corrected_data1_west, label='Lines and Continuum removed')
plt.plot(wavelengths2_west, corrected_data2_west, label='Lines and Continuum removed')
plt.plot(wavelengths3_west, corrected_data3_west, label='Lines and Continuum removed')
plt.plot(wavelengths4_west, corrected_data4_west, label='Lines and Continuum removed')
plt.plot(wavelengths5_west, corrected_data5_west, label='Lines and Continuum removed')
plt.plot(wavelengths6_west, data6_west, label='Lines and Continuum removed')
plt.plot(wavelengths7_west, corrected_data7_west+2.5, label='Lines and Continuum removed')

plt.ylim(0, 50)
plt.show()
plt.close()

#%%

plt.figure()
plt.plot(wavelengths5_west, corrected_data5_west+6.5, label='Lines and Continuum removed')
plt.plot(wavelengths6_west, corrected_data6_west, label='Lines and Continuum removed')
plt.plot(wavelengths7_west, corrected_data7_west, label='Lines and Continuum removed')
plt.ylim(0, 15)
plt.show()
plt.close()

#%%

plt.figure()
plt.plot(wavelengths5_west, corrected_data5, label='Lines and Continuum removed')
plt.plot(wavelengths6_west, corrected_data6, label='Lines and Continuum removed')
plt.plot(wavelengths7_west, corrected_data7, label='Lines and Continuum removed')
plt.ylim(0, 25)
plt.show()
plt.close()
