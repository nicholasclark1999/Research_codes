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

import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)

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
plt.show

#%%

#all arrays should have same spacial x and y dimensions, so define variables for this to use in for loops
array_length_x = len(image_data_1a[0,:,0])
array_length_y = len(image_data_1a[0,0,:])


'''
EMISSION LINE REMOVAL
'''



#removing nans
image_data_1a[np.isnan(image_data_1a)] = 0
image_data_1b[np.isnan(image_data_1b)] = 0
image_data_1c[np.isnan(image_data_1c)] = 0
image_data_2a[np.isnan(image_data_2a)] = 0
image_data_2b[np.isnan(image_data_2b)] = 0
image_data_2c[np.isnan(image_data_2c)] = 0
image_data_3a[np.isnan(image_data_3a)] = 0
image_data_3b[np.isnan(image_data_3b)] = 0
image_data_3c[np.isnan(image_data_3c)] = 0

image_data_4a[np.isnan(image_data_4a)] = 0
image_data_4b[np.isnan(image_data_4b)] = 0
image_data_4c[np.isnan(image_data_4c)] = 0



#line lists come from kevin's lists on box. Note that if 2 lines are blended, only the blended one is included here, not the seperated ones.

#%%

wave_list_1a = [
    [5.1228,5.1340],
    [5.2252,5.2292],
    [5.3371,5.3417],
    [5.4460,5.4508],
    [5.4905,5.5124+0.005],
    [5.5708,5.5775],
    [5.5781,5.5848],
    [5.5948,5.6602],
    [5.6547,5.6602],
    [5.70888,5.71423]]

for i in range(len(wave_list_1a)):
    if i == 0:
        image_data_1a_noline = bnf.emission_line_remover_wrapper(wavelengths1a, image_data_1a, np.round(wave_list_1a[i], 3))
    else:
        image_data_1a_noline = bnf.emission_line_remover_wrapper(wavelengths1a, image_data_1a_noline, np.round(wave_list_1a[i], 3))
        
#defining a separate list of lines not on kevins list, note these only appear sometimes (investigate later maybe)
        
wave_list_1a_extra = [
    [5.0487, 5.0620],
    [5.0863, 5.0940],
    [5.1571, 5.1812], 
    [5.2587, 5.2828],
    [5.3739, 5.3853],
    [5.4531, 5.4643],
    [5.4723, 5.5474],
    [5.5601, 5.6622],
    [5.7019, 5.7139]]

for i in range(len(wave_list_1a_extra)):
    image_data_1a_noline = bnf.emission_line_remover_wrapper(wavelengths1a, image_data_1a_noline, np.round(wave_list_1a_extra[i], 3))

bnf.error_check_imager(wavelengths1a, image_data_1a, 'PDFtime/spectra_checking/Channel1A_check.pdf', 4.9, 5.7, 1,
                       data_no_lines=image_data_1a_noline)

print('Channel 1A lines removed')

np.save('Analysis/image_data_1a_noline', image_data_1a_noline)

#%%

wave_list_1b = [
    [5.705,5.7164],
    [5.9020,5.9108],
    [5.9139,5.91725],
    [5.9508,5.9588],
    [5.9596,5.9700],
    [5.9772,5.9876],
    [6.1052,6.1100],
    [6.1476,6.1524],
    [6.2400,6.2420],
    [6.2871,6.2934],
    [6.3770,6.3815],
    [6.4895,6.4958],
    [6.5360,6.5400]]

for i in range(len(wave_list_1b)):
    if i == 0:
        image_data_1b_noline = bnf.emission_line_remover_wrapper(wavelengths1b, image_data_1b, np.round(wave_list_1b[i], 3))
    else:
        image_data_1b_noline = bnf.emission_line_remover_wrapper(wavelengths1b, image_data_1b_noline, np.round(wave_list_1b[i], 3))

#defining a separate list of lines not on kevins list, note these only appear sometimes (investigate later maybe)

wave_list_1b_extra = [
    [5.7347, 5.7423],
    [6.0436, 6.0572],
    [6.1138, 6.1214],
    [6.1458, 6.1579],
    [6.2391, 6.2483],
    [6.4770, 6.5182]]

for i in range(len(wave_list_1b_extra)):
    image_data_1b_noline = bnf.emission_line_remover_wrapper(wavelengths1b, image_data_1b_noline, np.round(wave_list_1b_extra[i], 3))
    
bnf.error_check_imager(wavelengths1b, image_data_1b, 'PDFtime/spectra_checking/Channel1B_check.pdf', 5.7, 6.6, 1,
                       data_no_lines=image_data_1b_noline)

print('Channel 1B lines removed')

np.save('Analysis/image_data_1b_noline', image_data_1b_noline)

#%%

wave_list_1c = [
    [6.5357, 6.5418],
    [6.7015, 6.70915], 
    [6.7192, 6.7264],
    [6.7664,6.7744],
    [6.9056, 6.9112],
    [6.9432, 6.9504],
    [6.9784-0.02, 6.9894+0.02],
    [7.1984, 7.20616],
    [7.3128, 7.3208],
    [7.4496, 7.4624],
    [7.4673, 7.4726],
    [7.4963, 7.5048],
    [7.5052, 7.5104]]

for i in range(len(wave_list_1c)):
    if i == 0:
        image_data_1c_noline = bnf.emission_line_remover_wrapper(wavelengths1c, image_data_1c, np.round(wave_list_1c[i], 3))
    else:
        image_data_1c_noline = bnf.emission_line_remover_wrapper(wavelengths1c, image_data_1c_noline, np.round(wave_list_1c[i], 3))
    
#note that Ch1 ends in a very strong line with pulldown, so to address this set every value after 7.615 microns
#to a constant, that is the median near 7.615 to remove this one

for i in range(array_length_x):
    for j in range(array_length_y):
        temp_index = np.where(np.round(wavelengths1c, 3) == 7.615)[0][0]
        temp_median = np.median(image_data_1c_noline[temp_index-10:temp_index,i,j])
        image_data_1c_noline[temp_index:,i,j] = temp_median

bnf.error_check_imager(wavelengths1c, image_data_1c, 'PDFtime/spectra_checking/Channel1C_check.pdf', 6.6, 7.6, 1,
                       data_no_lines=image_data_1c_noline)

print('Channel 1C lines removed')

np.save('Analysis/image_data_1c_noline', image_data_1c_noline)

#%%

wave_list_2a = [
    [7.6452, 7.6564],
    [7.6405, 7.6447],
    [7.7768, 7.7823],
    [7.8086, 7.8151],
    [7.89437, 7.90587],
    [8.0204, 8.0279],
    [8.0454, 8.0501],
    [8.1510, 8.1582],
    [8.6019, 8.6154],
    [8.6620, 8.6651],
    [8.7524, 8.7620]]

for i in range(len(wave_list_2a)):
    if i == 0:
        image_data_2a_noline = bnf.emission_line_remover_wrapper(wavelengths2a, image_data_2a, np.round(wave_list_2a[i], 3))
    else:
        image_data_2a_noline = bnf.emission_line_remover_wrapper(wavelengths2a, image_data_2a_noline, np.round(wave_list_2a[i], 3))
 
bnf.error_check_imager(wavelengths2a, image_data_2a, 'PDFtime/spectra_checking/Channel2A_check.pdf', 7.6, 8.7, 1,
                        data_no_lines=image_data_2a_noline)       
 
print('Channel 2A lines removed')

np.save('Analysis/image_data_2a_noline', image_data_2a_noline)

#%%

wave_list_2b = [
    [8.7529, 8.7635],
    [8.8243, 8.8330], 
    [8.98135, 8.9980],
    [9.0339, 9.0475],
    [9.1092, 9.1164],
    [9.1299, 9.13475],
    [9.2571, 9.2612],
    [9.2700, 9.2850],
    [9.3880, 9.3933],
    [9.5192, 9.5272],
    [9.6012, 9.6771],
    [9.7001, 9.7185],
    [9.7676, 9.7712]]

for i in range(len(wave_list_2b)):
    if i == 0:
        image_data_2b_noline = bnf.emission_line_remover_wrapper(wavelengths2b, image_data_2b, np.round(wave_list_2b[i], 3))
    else:
        image_data_2b_noline = bnf.emission_line_remover_wrapper(wavelengths2b, image_data_2b_noline, np.round(wave_list_2b[i], 3))

bnf.error_check_imager(wavelengths2b, image_data_2b, 'PDFtime/spectra_checking/Channel2B_check.pdf', 8.7, 10.1, 1,
                       data_no_lines=image_data_2b_noline)        



print('Channel 2B lines removed')

np.save('Analysis/image_data_2b_noline', image_data_2b_noline)

#%%

wave_list_2c = [
    [10.0652, 10.0714],
    [10.4990, 10.5177],
    [11.3254, 11.3367],
    [11.2990, 11.3133]]

for i in range(len(wave_list_2c)):
    if i == 0:
        image_data_2c_noline = bnf.emission_line_remover_wrapper(wavelengths2c, image_data_2c, np.round(wave_list_2c[i], 3))
    else:
        image_data_2c_noline = bnf.emission_line_remover_wrapper(wavelengths2c, image_data_2c_noline, np.round(wave_list_2c[i], 3))

bnf.error_check_imager(wavelengths2c, image_data_2c, 'PDFtime/spectra_checking/Channel2C_check.pdf', 10.1, 11.7, 1,
                       data_no_lines=image_data_2c_noline)

print('Channel 2C lines removed')

np.save('Analysis/image_data_2c_noline', image_data_2c_noline)

#%%

wave_list_3a = [
    [11.7543, 11.7661],
    [12.2717, 12.2836],
    [12.2991, 12.3086],
    [12.3592, 12.4483],
    [12.3809, 12.3912],
    [12.5809, 12.5986],
    [12.8018, 12.8239],
    [13.0942, 13.1076],
    [13.1215, 13.1314],
    [13.3783, 13.3878]]

for i in range(len(wave_list_3a)):
    if i == 0:
        image_data_3a_noline = bnf.emission_line_remover_wrapper(wavelengths3a, image_data_3a, np.round(wave_list_3a[i], 3))
    else:
        image_data_3a_noline = bnf.emission_line_remover_wrapper(wavelengths3a, image_data_3a_noline, np.round(wave_list_3a[i], 3))

bnf.error_check_imager(wavelengths3a, image_data_3a, 'PDFtime/spectra_checking/Channel3A_check.pdf', 11.6, 13.4, 1,
                       data_no_lines=image_data_3a_noline)

print('Channel 3A lines removed')

np.save('Analysis/image_data_3a_noline', image_data_3a_noline)

#%%

wave_list_3b = [
    [13.3763, 13.3896],
    [13.5144, 13.5275],
    [14.3048, 14.3370],
    [14.3592, 14.3713],
    [14.3850, 14.4053+0.01],
    [14.7740, 14.7859],
    [15.3834, 15.3937],
    [15.5425, 15.5671]]

for i in range(len(wave_list_3b)):
    if i == 0:
        image_data_3b_noline = bnf.emission_line_remover_wrapper(wavelengths3b, image_data_3b, np.round(wave_list_3b[i], 3))
    else:
        image_data_3b_noline = bnf.emission_line_remover_wrapper(wavelengths3b, image_data_3b_noline, np.round(wave_list_3b[i], 3))

bnf.error_check_imager(wavelengths3b, image_data_3b, 'PDFtime/spectra_checking/Channel3B_check.pdf', 13.4, 15.5, 1, 
                       data_no_lines=image_data_3b_noline)

print('Channel 3B lines removed')

np.save('Analysis/image_data_3b_noline', image_data_3b_noline)

#%%

wave_list_3c = [
    [15.4643, 15.4766],
    [15.5390, 15.5714],
    [16.1962, 16.2315],
    [17.0266, 17.0410],
    [17.2535, 17.2642],
    [17.9259, 17.9366]]

for i in range(len(wave_list_3c)):
    if i == 0:
        image_data_3c_noline = bnf.emission_line_remover_wrapper(wavelengths3c, image_data_3c, np.round(wave_list_3c[i], 3))
    else:
        image_data_3c_noline = bnf.emission_line_remover_wrapper(wavelengths3c, image_data_3c_noline, np.round(wave_list_3c[i], 3))

bnf.error_check_imager(wavelengths3c, image_data_3c, 'PDFtime/spectra_checking/Channel3C_check.pdf', 15.5, 17.9, 1, 
                       data_no_lines=image_data_3c_noline)

print('Channel 3C lines removed')

np.save('Analysis/image_data_3c_noline', image_data_3c_noline)

#%%

wave_list_4a = [
    [18.6930, 18.7237+0.01],
    [19.0420, 19.0804],
    [20.3007, 20.3134+0.01]]

for i in range(len(wave_list_4a)):
    if i == 0:
        image_data_4a_noline = bnf.emission_line_remover_wrapper(wavelengths4a, image_data_4a, np.round(wave_list_4a[i], 3))
    else:
        image_data_4a_noline = bnf.emission_line_remover_wrapper(wavelengths4a, image_data_4a_noline, np.round(wave_list_4a[i], 3))

bnf.error_check_imager(wavelengths4a, image_data_4a, 'PDFtime/spectra_checking/Channel4A_check.pdf', 17.8, 20.9, 1,
                       data_no_lines=image_data_4a_noline)

print('Channel 4A lines removed')

np.save('Analysis/image_data_4a_noline', image_data_4a_noline)

#%%

wave_list_4b = [
    [21.2488, 21.2629],
    [21.8152, 21.8328],
    [24.2943, 24.3360]]

for i in range(len(wave_list_4b)):
    if i == 0:
        image_data_4b_noline = bnf.emission_line_remover_wrapper(wavelengths4b, image_data_4b, np.round(wave_list_4b[i], 3))
    else:
        image_data_4b_noline = bnf.emission_line_remover_wrapper(wavelengths4b, image_data_4b_noline, np.round(wave_list_4b[i], 3))

bnf.error_check_imager(wavelengths4b, image_data_4b, 'PDFtime/spectra_checking/Channel4B_check.pdf', 20.8, 24.4, 1, 
                       data_no_lines=image_data_4b_noline)

print('Channel 4B lines removed')

np.save('Analysis/image_data_4b_noline', image_data_4b_noline)

#%%

wave_list_4c = [
    [25.2375, 25.2606],
    [25.5615, 25.5848],
    [25.8703, 25.9134+0.01]]

for i in range(len(wave_list_4c)):
    if i == 0:
        image_data_4c_noline = bnf.emission_line_remover_wrapper(wavelengths4c, image_data_4c, np.round(wave_list_4c[i], 3))
    else:
        image_data_4c_noline = bnf.emission_line_remover_wrapper(wavelengths4c, image_data_4c_noline, np.round(wave_list_4c[i], 3))

#note that Ch4 ends with complete chaos, so i am removing it to make the plots prettier

for i in range(array_length_x):
    for j in range(array_length_y):
        temp_index = 700 #28.603
        temp_median = np.median(image_data_4c_noline[temp_index-10:temp_index,i,j])
        image_data_4c_noline[temp_index:,i,j] = temp_median

print('Channel 4C lines removed')

bnf.error_check_imager(wavelengths4c, image_data_4c, 'PDFtime/spectra_checking/Channel4C_check.pdf', 24.5, 28.6, 1, 
                       data_no_lines=image_data_4c_noline)

np.save('Analysis/image_data_4c_noline', image_data_4c_noline)

#%%



'''
SPECTA STITCHING WITH LINES
'''

#11.2 feature

#combining channels 2C and 3A to get proper continua for the 11.2 feature

image_data_112_temp, wavelengths112, overlap112_temp = bnf.flux_aligner3(wavelengths2c, wavelengths3a, image_data_2c[:,50,50], image_data_3a[:,50,50])



#using the above to make an array of the correct size to fill
image_data_112_lines = np.zeros((len(image_data_112_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_112_lines[:,i,j], wavelengths112, overlap112 = bnf.flux_aligner3(wavelengths2c, wavelengths3a, image_data_2c[:,i,j], image_data_3a[:,i,j])

print('11.2 feature stitching complete')

np.save('Analysis/image_data_112_lines', image_data_112_lines)

#%%

#combining channels 1C, 2A, and 2B to get proper continua for the 7.7 and 8.6 features

image_data_77_temp, wavelengths77, overlap77_temp = bnf.flux_aligner4(wavelengths1c, wavelengths2a, image_data_1c[:,50,50], image_data_2a[:,50,50])
image_data_77_temp_temp, wavelengths77, overlap77_temp = bnf.flux_aligner3(wavelengths77, wavelengths2b, image_data_77_temp, image_data_2b[:,50,50])



#using the above to make an array of the correct size to fill
image_data_77_1 = np.zeros((len(image_data_77_temp), array_length_x, array_length_y))

image_data_77_lines = np.zeros((len(image_data_77_temp_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_77_1[:,i,j], wavelengths77_1, overlap77 = bnf.flux_aligner4(wavelengths1c, wavelengths2a, image_data_1c[:,i,j], image_data_2a[:,i,j])
        
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_77_lines[:,i,j], wavelengths77, overlap77 = bnf.flux_aligner3(wavelengths77_1, wavelengths2b, image_data_77_1[:,i,j], image_data_2b[:,i,j])

print('7.7 and 8.6 features stitching complete')

np.save('Analysis/image_data_77_lines', image_data_77_lines)

#%%

#combining channels 3A and 3B to get proper continua for the 13.5 feature

image_data_135_temp, wavelengths135, overlap135_temp = bnf.flux_aligner3(wavelengths3a, wavelengths3b, image_data_3a[:,50,50], image_data_3b[:,50,50])



#using the above to make an array of the correct size to fill
image_data_135_lines = np.zeros((len(image_data_135_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_135_lines[:,i,j], wavelengths135, overlap135 =bnf. flux_aligner3(wavelengths3a, wavelengths3b, image_data_3a[:,i,j], image_data_3b[:,i,j])
    
print('13.5 feature stitching complete')    

np.save('Analysis/image_data_135_lines', image_data_135_lines)

#%%

#combining channels 1A and 1B to get proper continua for the 5.7 feature

image_data_57_temp, wavelengths57, overlap57_temp = bnf.flux_aligner3(wavelengths1a, wavelengths1b, image_data_1a[:,50,50], image_data_1b[:,50,50])



#using the above to make an array of the correct size to fill
image_data_57_lines = np.zeros((len(image_data_57_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_57_lines[:,i,j], wavelengths57, overlap57 =bnf. flux_aligner3(wavelengths1a, wavelengths1b, image_data_1a[:,i,j], image_data_1b[:,i,j])
    
print('5.7 feature stitching complete')    

np.save('Analysis/image_data_57_lines', image_data_57_lines)
    
#%%

#combining all channels to get proper continua for the 23.0 crystalline silicate feature
#note channels 1C, 2A, 2B combined into 77 already, and 2C, 3A combined into 112 already

image_data_230cs_temp_1, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths1a, wavelengths1b, image_data_1a[:,50,50], image_data_1b[:,50,50])
image_data_230cs_temp_2, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths77, image_data_230cs_temp_1, image_data_77_lines[:,50,50])
image_data_230cs_temp_3, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths112, image_data_230cs_temp_2, image_data_112_lines[:,50,50])
#image_data_230cs_temp_4, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths3b, image_data_230cs_temp_3, image_data_3b[:,50,50])

image_data_230cs_temp_4, wavelengths230cs_temp, overlap230cs_temp = bnf.flux_aligner2(wavelengths3b, wavelengths3c, image_data_3b[:,50,50], image_data_3c[:,50,50])
image_data_230cs_temp_5, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths230cs_temp, image_data_230cs_temp_3, image_data_230cs_temp_4)

#image_data_230cs_temp_5, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner2(wavelengths230cs, wavelengths3c, image_data_230cs_temp_4, image_data_3c[:,50,50])
image_data_230cs_temp_6, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths4a, image_data_230cs_temp_5, image_data_4a[:,50,50])
image_data_230cs_temp_7, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths4b, image_data_230cs_temp_6, image_data_4b[:,50,50])
image_data_230cs_temp_8, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths4c, image_data_230cs_temp_7, image_data_4c[:,50,50])

#%%

#using the above to make an array of the correct size to fill
image_data_230cs_1 = np.zeros((len(image_data_230cs_temp_1), array_length_x, array_length_y))
image_data_230cs_2 = np.zeros((len(image_data_230cs_temp_2), array_length_x, array_length_y))
image_data_230cs_3 = np.zeros((len(image_data_230cs_temp_3), array_length_x, array_length_y))
image_data_230cs_4 = np.zeros((len(image_data_230cs_temp_4), array_length_x, array_length_y))
image_data_230cs_5 = np.zeros((len(image_data_230cs_temp_5), array_length_x, array_length_y))
#%%
image_data_230cs_6 = np.zeros((len(image_data_230cs_temp_6), array_length_x, array_length_y))
#%%
image_data_230cs_7 = np.zeros((len(image_data_230cs_temp_7), array_length_x, array_length_y))
image_data_230cs_lines = np.zeros((len(image_data_230cs_temp_8), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_1[:,i,j], wavelengths230cs_1, overlap230cs = bnf.flux_aligner3(wavelengths1a, wavelengths1b, image_data_1a[:,i,j], image_data_1b[:,i,j])
        
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_2[:,i,j], wavelengths230cs_2, overlap230cs = bnf.flux_aligner3(wavelengths230cs_1, wavelengths77, image_data_230cs_1[:,i,j], image_data_77_lines[:,i,j]) 
        
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_3[:,i,j], wavelengths230cs_3, overlap230cs = bnf.flux_aligner3(wavelengths230cs_2, wavelengths112, image_data_230cs_2[:,i,j], image_data_112_lines[:,i,j]) 



for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_4[:,i,j], wavelengths230cs_temp, overlap230cs_temp = bnf.flux_aligner2(wavelengths3b, wavelengths3c, image_data_3b[:,i,j], image_data_3c[:,i,j])

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_5[:,i,j], wavelengths230cs_5, overlap230cs = bnf.flux_aligner3(wavelengths230cs_3, wavelengths230cs_temp, image_data_230cs_3[:,i,j], image_data_230cs_4[:,i,j])

'''
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_4[:,i,j], wavelengths230cs_4, overlap230cs = bnf.flux_aligner3(wavelengths230cs_3, wavelengths3b, image_data_230cs_3[:,i,j], image_data_3b[:,i,j])

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_5[:,i,j], wavelengths230cs_5, overlap230cs = bnf.flux_aligner3(wavelengths230cs_4, wavelengths3c, image_data_230cs_4[:,i,j], image_data_3c[:,i,j])
'''
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_6[:,i,j], wavelengths230cs_6, overlap230cs = bnf.flux_aligner3(wavelengths230cs_5, wavelengths4a, image_data_230cs_5[:,i,j], image_data_4a[:,i,j]) 
        
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_7[:,i,j], wavelengths230cs_7, overlap230cs = bnf.flux_aligner3(wavelengths230cs_6, wavelengths4b, image_data_230cs_6[:,i,j], image_data_4b[:,i,j]) 
        
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_lines[:,i,j], wavelengths230cs, overlap230cs = bnf.flux_aligner3(wavelengths230cs_7, wavelengths4c, image_data_230cs_7[:,i,j], image_data_4c[:,i,j]) 

print('23.0 feature stitching complete')

np.save('Analysis/image_data_230cs_lines', image_data_230cs_lines)

#%%



'''
SPECTA STITCHING WITHOUT LINES
'''


#11.2 feature

#combining channels 2C and 3A to get proper continua for the 11.2 feature

image_data_112_temp, wavelengths112, overlap112_temp = bnf.flux_aligner3(wavelengths2c, wavelengths3a, image_data_2c_noline[:,50,50], image_data_3a_noline[:,50,50])



#using the above to make an array of the correct size to fill
image_data_112 = np.zeros((len(image_data_112_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_112[:,i,j], wavelengths112, overlap112 = bnf.flux_aligner3(wavelengths2c, wavelengths3a, image_data_2c_noline[:,i,j], image_data_3a_noline[:,i,j])

print('11.2 feature stitching complete')

np.save('Analysis/wavelengths112', wavelengths112)
np.save('Analysis/image_data_112', image_data_112)

#%%

#combining channels 1C, 2A, and 2B to get proper continua for the 7.7 and 8.6 features

image_data_77_temp, wavelengths77, overlap77_temp = bnf.flux_aligner4(wavelengths1c, wavelengths2a, image_data_1c_noline[:,50,50], image_data_2a_noline[:,50,50])
image_data_77_temp_temp, wavelengths77, overlap77_temp = bnf.flux_aligner3(wavelengths77, wavelengths2b, image_data_77_temp, image_data_2b_noline[:,50,50])



#using the above to make an array of the correct size to fill
image_data_77_1 = np.zeros((len(image_data_77_temp), array_length_x, array_length_y))

image_data_77 = np.zeros((len(image_data_77_temp_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_77_1[:,i,j], wavelengths77_1, overlap77 = bnf.flux_aligner4(wavelengths1c, wavelengths2a, image_data_1c_noline[:,i,j], image_data_2a_noline[:,i,j])
        
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_77[:,i,j], wavelengths77, overlap77 = bnf.flux_aligner3(wavelengths77_1, wavelengths2b, image_data_77_1[:,i,j], image_data_2b_noline[:,i,j])

print('7.7 and 8.6 features stitching complete')

np.save('Analysis/wavelengths77', wavelengths77)
np.save('Analysis/image_data_77', image_data_77)

#%%

#combining channels 3A and 3B to get proper continua for the 13.5 feature

image_data_135_temp, wavelengths135, overlap135_temp = bnf.flux_aligner3(wavelengths3a, wavelengths3b, image_data_3a_noline[:,50,50], image_data_3b_noline[:,50,50])



#using the above to make an array of the correct size to fill
image_data_135 = np.zeros((len(image_data_135_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_135[:,i,j], wavelengths135, overlap135 =bnf. flux_aligner3(wavelengths3a, wavelengths3b, image_data_3a_noline[:,i,j], image_data_3b_noline[:,i,j])
    
print('13.5 feature stitching complete')    

np.save('Analysis/wavelengths135', wavelengths135)
np.save('Analysis/image_data_135', image_data_135)

#%%

#combining channels 1A and 1B to get proper continua for the 5.7 feature

image_data_57_temp, wavelengths57, overlap57_temp = bnf.flux_aligner3(wavelengths1a, wavelengths1b, image_data_1a_noline[:,50,50], image_data_1b_noline[:,50,50])



#using the above to make an array of the correct size to fill
image_data_57 = np.zeros((len(image_data_57_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_57[:,i,j], wavelengths57, overlap57 =bnf. flux_aligner3(wavelengths1a, wavelengths1b, image_data_1a_noline[:,i,j], image_data_1b_noline[:,i,j])
    
print('5.7 feature stitching complete')    

np.save('Analysis/wavelengths57', wavelengths57)
np.save('Analysis/image_data_57', image_data_57)
    
#%%

#combining all channels to get proper continua for the 23.0 crystalline silicate feature
#note channels 1C, 2A, 2B combined into 77 already, and 2C, 3A combined into 112 already

image_data_230cs_temp_1, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths1a, wavelengths1b, image_data_1a_noline[:,50,50], image_data_1b_noline[:,50,50])
image_data_230cs_temp_2, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths77, image_data_230cs_temp_1, image_data_77[:,50,50])
image_data_230cs_temp_3, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths112, image_data_230cs_temp_2, image_data_112[:,50,50])
#image_data_230cs_temp_4, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths3b, image_data_230cs_temp_3, image_data_3b_noline[:,50,50])

image_data_230cs_temp_4, wavelengths230cs_temp, overlap230cs_temp = bnf.flux_aligner2(wavelengths3b, wavelengths3c, image_data_3b_noline[:,50,50], image_data_3c_noline[:,50,50])
image_data_230cs_temp_5, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths230cs_temp, image_data_230cs_temp_3, image_data_230cs_temp_4)

#image_data_230cs_temp_5, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner2(wavelengths230cs, wavelengths3c, image_data_230cs_temp_4, image_data_3c_noline[:,50,50])
image_data_230cs_temp_6, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths4a, image_data_230cs_temp_5, image_data_4a_noline[:,50,50])
image_data_230cs_temp_7, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths4b, image_data_230cs_temp_6, image_data_4b_noline[:,50,50])
image_data_230cs_temp_8, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths4c, image_data_230cs_temp_7, image_data_4c_noline[:,50,50])



#using the above to make an array of the correct size to fill
image_data_230cs_1 = np.zeros((len(image_data_230cs_temp_1), array_length_x, array_length_y))
image_data_230cs_2 = np.zeros((len(image_data_230cs_temp_2), array_length_x, array_length_y))
image_data_230cs_3 = np.zeros((len(image_data_230cs_temp_3), array_length_x, array_length_y))
image_data_230cs_4 = np.zeros((len(image_data_230cs_temp_4), array_length_x, array_length_y))
image_data_230cs_5 = np.zeros((len(image_data_230cs_temp_5), array_length_x, array_length_y))
image_data_230cs_6 = np.zeros((len(image_data_230cs_temp_6), array_length_x, array_length_y))
image_data_230cs_7 = np.zeros((len(image_data_230cs_temp_7), array_length_x, array_length_y))
image_data_230cs = np.zeros((len(image_data_230cs_temp_8), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_1[:,i,j], wavelengths230cs_1, overlap230cs = bnf.flux_aligner3(wavelengths1a, wavelengths1b, image_data_1a_noline[:,i,j], image_data_1b_noline[:,i,j])
        
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_2[:,i,j], wavelengths230cs_2, overlap230cs = bnf.flux_aligner3(wavelengths230cs_1, wavelengths77, image_data_230cs_1[:,i,j], image_data_77[:,i,j]) 
        
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_3[:,i,j], wavelengths230cs_3, overlap230cs = bnf.flux_aligner3(wavelengths230cs_2, wavelengths112, image_data_230cs_2[:,i,j], image_data_112[:,i,j]) 


for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_4[:,i,j], wavelengths230cs_temp, overlap230cs_temp = bnf.flux_aligner2(wavelengths3b, wavelengths3c, image_data_3b_noline[:,i,j], image_data_3c_noline[:,i,j])

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_5[:,i,j], wavelengths230cs_5, overlap230cs = bnf.flux_aligner3(wavelengths230cs_3, wavelengths230cs_temp, image_data_230cs_3[:,i,j], image_data_230cs_4[:,i,j])


'''
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_4[:,i,j], wavelengths230cs_4, overlap230cs = bnf.flux_aligner3(wavelengths230cs_3, wavelengths3b, image_data_230cs_3[:,i,j], image_data_3b_noline[:,i,j])

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_5[:,i,j], wavelengths230cs_5, overlap230cs = bnf.flux_aligner3(wavelengths230cs_4, wavelengths3c, image_data_230cs_4[:,i,j], image_data_3c_noline[:,i,j])

'''

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_6[:,i,j], wavelengths230cs_6, overlap230cs = bnf.flux_aligner3(wavelengths230cs_5, wavelengths4a, image_data_230cs_5[:,i,j], image_data_4a_noline[:,i,j]) 
        
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_7[:,i,j], wavelengths230cs_7, overlap230cs = bnf.flux_aligner3(wavelengths230cs_6, wavelengths4b, image_data_230cs_6[:,i,j], image_data_4b_noline[:,i,j]) 
        
for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs[:,i,j], wavelengths230cs, overlap230cs = bnf.flux_aligner3(wavelengths230cs_7, wavelengths4c, image_data_230cs_7[:,i,j], image_data_4c_noline[:,i,j]) 

print('23.0 feature stitching complete')

np.save('Analysis/wavelengths230cs', wavelengths230cs)
np.save('Analysis/image_data_230cs', image_data_230cs)

'''

#figure for checking data stitching quality

i = 50
j = 50

ax = plt.figure(figsize=(10,10)).add_subplot(111)
plt.plot(wavelengths230cs, image_data_230cs[:,i,j])
plt.plot(wavelengths3b, image_data_3b_noline[:,i,j])
plt.plot(wavelengths3c, image_data_3c_noline[:,i,j])
plt.plot(wavelengths4a, image_data_4a_noline[:,i,j])
plt.plot(wavelengths4b, image_data_4b_noline[:,i,j])
plt.plot(wavelengths4c, image_data_4c_noline[:,i,j])

plt.show()

'''



#%%



'''
REGION FILE MASKING ARRAYS
'''



#creating an array that indicates where the Ch1 FOV is, so that comparison is only done between pixels with data.

region_indicator = bnf.extract_spectra_from_regions_one_pointing_no_bkg('data/ngc6302_ch1-short_s3d.fits', image_data_1a[0], 'data/ch1Arectangle.reg', do_sigma_clip=True, use_dq=False)

#%%



'''
CRYSTALLINE SILICATE COMPARISON CONTINUUM SUBTRACTION
'''



#HD 100546 continuum; do a subtraction near 11.3 microns and one near 23.0 microns (Cr Si features)

#trim data; have 113 continuum go from 

hd100546_lower_index_113 = 4490 #(7.00)
hd100546_upper_index_113 = 7370 #(13.00)

hd100546_lower_index_230 = np.copy(hd100546_upper_index_113)
hd100546_upper_index_230 = 11210 #(31.00)

hd100546_image_data_113 = np.copy(hd100546_image_data[hd100546_lower_index_113:hd100546_upper_index_113])
hd100546_image_data_230 = np.copy(hd100546_image_data[hd100546_lower_index_230:hd100546_upper_index_230])

hd100546_wavelengths113 = np.copy(hd100546_wavelengths[hd100546_lower_index_113:hd100546_upper_index_113])
hd100546_wavelengths230 = np.copy(hd100546_wavelengths[hd100546_lower_index_230:hd100546_upper_index_230])

hd100546_image_data_113_cont = np.zeros(len(hd100546_image_data_113))
hd100546_image_data_230_cont = np.zeros(len(hd100546_image_data_230))

hd100546_points113 = [10.61, 10.87, 11.65, 11.79] #same as for 11.2 feature for direct comparison
hd100546_points230 = [17.00, 20.35, 25.50, 26.50] #before had an anchor point at 15.40, and 18.2

hd100546_image_data_113_cont = bnf.linear_continuum_single_channel(hd100546_wavelengths113, hd100546_image_data_113, hd100546_points113)
hd100546_image_data_230_cont = bnf.linear_continuum_single_channel(hd100546_wavelengths230, hd100546_image_data_230, hd100546_points230)


np.save('Analysis/hd100546_lower_index_113', hd100546_lower_index_113)
np.save('Analysis/hd100546_upper_index_113', hd100546_upper_index_113)
np.save('Analysis/hd100546_lower_index_230', hd100546_lower_index_230)
np.save('Analysis/hd100546_upper_index_230', hd100546_upper_index_230)
np.save('Analysis/hd100546_image_data_113', hd100546_image_data_113)
np.save('Analysis/hd100546_image_data_230', hd100546_image_data_230)
np.save('Analysis/hd100546_wavelengths113', hd100546_wavelengths113)
np.save('Analysis/hd100546_wavelengths230', hd100546_wavelengths230)
np.save('Analysis/hd100546_image_data_113_cont', hd100546_image_data_113_cont)
np.save('Analysis/hd100546_image_data_230_cont', hd100546_image_data_230_cont)

#%%

#comet Hale-Bopp continuum: make it as similar to HD 100546 as possible 

hale_bopp_image_data_113_cont = np.zeros(len(hale_bopp_image_data_113))
hale_bopp_image_data_230_cont = np.zeros(len(hale_bopp_image_data_230))

hale_bopp_points113 = [10.61, 10.87, 11.65, 11.79] #same as for 11.2 feature for direct comparison
hale_bopp_points230 = [16.0, 18.20, 20.35, 25.50] #dont have data for 15.4

hale_bopp_image_data_113_cont = bnf.linear_continuum_single_channel(hale_bopp_wavelengths113, hale_bopp_image_data_113, hale_bopp_points113)
hale_bopp_image_data_230_cont = bnf.linear_continuum_single_channel(hale_bopp_wavelengths230, hale_bopp_image_data_230, hale_bopp_points230)

np.save('Analysis/hale_bopp_image_data_113_cont', hale_bopp_image_data_113_cont)
np.save('Analysis/hale_bopp_image_data_230_cont', hale_bopp_image_data_230_cont)
        
#%%

#23.0 silicate feature

image_data_230cs_cont_1 = np.zeros((len(image_data_230cs[:,0,0]), array_length_x, array_length_y))

points230cs_1 =  [17.00, 20.35, 25.50, 26.50] #same as hd 100546

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_cont_1[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths230cs, image_data_230cs[:,i,j], points230cs_1) #note image_data_230cs is built out of things with no lines
        
bnf.error_check_imager(wavelengths230cs, image_data_230cs, 'PDFtime/spectra_checking/230cs_check_continuum_1.pdf', 14.0, 28.0, 1.25, continuum=image_data_230cs_cont_1)

np.save('Analysis/image_data_230cs_cont_1', image_data_230cs_cont_1)

#%%

image_data_230cs_cont_2 = np.zeros((len(image_data_230cs[:,0,0]), array_length_x, array_length_y))

points230cs_2 =  [18.10, 20.65, 25.50, 26.50] 

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_230cs_cont_2[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths230cs, image_data_230cs[:,i,j], points230cs_2) #note image_data_230cs is built out of things with no lines
        
bnf.error_check_imager(wavelengths230cs, image_data_230cs, 'PDFtime/spectra_checking/230cs_check_continuum_2.pdf', 14.0, 28.0, 1.25, continuum=image_data_230cs_cont_2)

np.save('Analysis/image_data_230cs_cont_2', image_data_230cs_cont_2)

#%%

image_data_230cs_cont = np.copy(image_data_230cs_cont_1)

#make an array to keep track of which continuum is being used
cont_type_230cs = np.ones((array_length_x, array_length_y))

temp_index_1 = np.where(np.round(wavelengths230cs, 2) == 20.50)[0][0] #approx 20.5
temp_index_2 = np.where(np.round(wavelengths230cs, 2) == 20.70)[0][0] #approx 20.7

for i in range(array_length_x):
    for j in range(array_length_y):
        if np.median(image_data_230cs[temp_index_1:temp_index_2,i,j] - image_data_230cs_cont[temp_index_1:temp_index_2,i,j]) < 0:
            cont_type_230cs[i,j] += 1
            image_data_230cs_cont[:,i,j] = image_data_230cs_cont_2[:,i,j]  

bnf.error_check_imager(wavelengths230cs, image_data_230cs, 'PDFtime/spectra_checking/230cs_check_continuum.pdf', 14.0, 28.0, 1.25, continuum=image_data_230cs_cont)

np.save('Analysis/cont_type_230cs', cont_type_230cs)
np.save('Analysis/image_data_230cs_cont', image_data_230cs_cont)

#%%

#11.3 silicate feature

image_data_113cs_cont = np.zeros((len(image_data_112[:,0,0]), array_length_x, array_length_y)) #uses the same wavelength range, so can use this array instead of making a new one

points113cs = hd100546_points113

for i in range(array_length_x): #note that unless the points change, this will be the same as the continuum for the 11.2 feature
    for j in range(array_length_y):
        image_data_113cs_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths112, image_data_112[:,i,j], points113cs) #note image_data_113cs is built out of things with no lines
  #%%      
bnf.error_check_imager(wavelengths112, image_data_112, 'PDFtime/spectra_checking/113cs_check_continuum.pdf', 10.6, 11.8, 1.5, continuum=image_data_113cs_cont)

np.save('Analysis/image_data_113cs_cont', image_data_113cs_cont)

#%%



'''
INDIVIDUAL PAH FEATURE CONTINUUM SUBTACTION, INTEGRATION AND PLOTS
'''



current_reprojection = '3C'

np.save('Analysis/current_reprojection', current_reprojection)

#%%



'''
11.2 feature
'''



#continuum
image_data_112_cont_1 = np.zeros((len(image_data_112[:,0,0]), array_length_x, array_length_y))

points112_1 = [10.61, 10.87, 11.65, 11.79]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_112_cont_1[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths112, image_data_112[:,i,j], points112_1) #note image_data_112 is built out of things with no lines
        
bnf.error_check_imager(wavelengths112, image_data_112, 'PDFtime/spectra_checking/112_check_continuum_1.pdf', 10.1, 13.1, 1.5, continuum=image_data_112_cont_1)

np.save('Analysis/image_data_112_cont_1', image_data_112_cont_1)

image_data_112_cont_2 = np.zeros((len(image_data_112[:,0,0]), array_length_x, array_length_y))

points112_2 = [10.61, 10.87, 11.79, 11.89] #last point is a filler for now

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_112_cont_2[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths112, image_data_112[:,i,j], points112_2) #note image_data_112 is built out of things with no lines
        
bnf.error_check_imager(wavelengths112, image_data_112, 'PDFtime/spectra_checking/112_check_continuum_2.pdf', 10.1, 13.1, 1.5, continuum=image_data_112_cont_2)

np.save('Analysis/image_data_112_cont_2', image_data_112_cont_2)

#%%
image_data_112_cont = np.copy(image_data_112_cont_1)

#make an array to keep track of which 11.2 continuum is being used
cont_type_112 = np.ones((array_length_x, array_length_y))

temp_index_1 = np.where(np.round(wavelengths112, 2) == 11.65)[0][0] 
temp_index_2 = np.where(np.round(wavelengths112, 2) == 11.75)[0][0] 

#if continuum decreases significantly (more than 100) between these points, use the cont_2

for i in range(array_length_x):
    for j in range(array_length_y):
        if image_data_112_cont[temp_index_2,i,j] - image_data_112_cont[temp_index_1,i,j] < -25:
            cont_type_112[i,j] += 1
            image_data_112_cont[:,i,j] = image_data_112_cont_2[:,i,j]  

bnf.error_check_imager(wavelengths112, image_data_112, 'PDFtime/spectra_checking/112_check_continuum.pdf', 10.1, 13.1, 1.5, continuum=image_data_112_cont)

np.save('Analysis/cont_type_112', cont_type_112)
np.save('Analysis/image_data_112_cont', image_data_112_cont)

#%%

#integration

pah_intensity_112 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_112 = np.zeros((array_length_x, array_length_y))

#have 11.2 feature go from 11.085 (826) to 11.65 (1250), this seems reasonable but theres arguments for slight adjustment i think

#NOTE: 1250 has the wavelength index of channel 3, use 11.621 (1239) instead, as it is the last wavelength of channel 2

lower_index_112 = 826
upper_index_112 = 1239

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_112[i,j] = bnf.pah_feature_integrator(wavelengths112[lower_index_112:upper_index_112], 
                                                            image_data_112[lower_index_112:upper_index_112,i,j] - image_data_112_cont[lower_index_112:upper_index_112,i,j])
    
print('11.2 feature intensity calculated')

np.save('Analysis/lower_index_112', lower_index_112)
np.save('Analysis/upper_index_112', upper_index_112)
np.save('Analysis/pah_intensity_112', pah_intensity_112)

#%%

error_index_112 = 186 #(10.25)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_112[i,j] = bnf.error_finder(wavelengths112, image_data_112[:,i,j] - image_data_112_cont[:,i,j], 11.25, 
                                                            pah_intensity_112[i,j], (lower_index_112, upper_index_112), error_index_112)

print('11.2 feature intensity error calculated')

np.save('Analysis/error_index_112', error_index_112)
np.save('Analysis/pah_intensity_error_112', pah_intensity_error_112)

#%%

snr_cutoff_112 = 300

bnf.single_feature_imager(pah_intensity_112, pah_intensity_112, pah_intensity_error_112, '11.2', '112', snr_cutoff_112, current_reprojection)

np.save('Analysis/snr_cutoff_112', snr_cutoff_112)

#%%



'''
23.0 crystalline silicate feature
'''



#integration

pah_intensity_230cs = np.zeros((array_length_x, array_length_y))
pah_intensity_error_230cs = np.zeros((array_length_x, array_length_y))

#have 23.0 feature go from 22.2 (9644) to 24.4 (10010), this seems reasonable but theres arguments for slight adjustment i think

lower_index_230cs = 9644
upper_index_230cs = 10010

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_230cs[i,j] = bnf.pah_feature_integrator(wavelengths230cs[lower_index_230cs:upper_index_230cs], 
                                                            image_data_230cs[lower_index_230cs:upper_index_230cs,i,j] - image_data_230cs_cont[lower_index_230cs:upper_index_230cs,i,j])
    
print('23.0 crystalline silicate feature intensity calculated')

np.save('Analysis/lower_index_230cs', lower_index_230cs)
np.save('Analysis/upper_index_230cs', upper_index_230cs)
np.save('Analysis/pah_intensity_230cs', pah_intensity_230cs)

#%%

error_index_230cs = 9312 #(20.20) there is a slight bump in some spectra here, namely the ones that use cont_2, but there is no good spot that works for everything it seems

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_230cs[i,j] = bnf.error_finder(wavelengths230cs, image_data_230cs[:,i,j] - image_data_230cs_cont[:,i,j], 23.0, 
                                                            pah_intensity_230cs[i,j], (lower_index_230cs, upper_index_230cs), error_index_230cs)

print('23.0 crystalline silicate feature intensity error calculated')

np.save('Analysis/error_index_230cs', error_index_230cs)
np.save('Analysis/pah_intensity_error_230cs', pah_intensity_error_230cs)

#%%

snr_cutoff_230cs = 1000

bnf.single_feature_imager(pah_intensity_230cs, pah_intensity_112, pah_intensity_error_230cs, '23.0 crystalline silicate', '230cs', snr_cutoff_230cs, current_reprojection)

np.save('Analysis/snr_cutoff_230cs', snr_cutoff_230cs)

#%%



'''
11.0 feature
'''

#integration

pah_intensity_110 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_110 = np.zeros((array_length_x, array_length_y))

#have 11.0 feature go from 10.96 (730) to 11.085 (826)

lower_index_110 = 730
upper_index_110 = np.copy(lower_index_112) #this ensures smooth transition from 11.0 to 11.2 feature

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_110[i,j] = bnf.pah_feature_integrator(wavelengths112[lower_index_110:upper_index_110], 
                                                            image_data_112[lower_index_110:upper_index_110,i,j] - image_data_112_cont[lower_index_110:upper_index_110,i,j])

print('11.0 feature intensity calculated')

np.save('Analysis/lower_index_110', lower_index_110)
np.save('Analysis/upper_index_110', upper_index_110)
np.save('Analysis/pah_intensity_110', pah_intensity_110)

#%%

error_index_110 = np.copy(error_index_112) #uses same error

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_110[i,j] = bnf.error_finder(wavelengths112, image_data_112[:,i,j] - image_data_112_cont[:,i,j], 11.0, 
                                                            pah_intensity_110[i,j], (lower_index_110, upper_index_110),  error_index_110)

print('11.0 feature intensity error calculated')

np.save('Analysis/error_index_110', error_index_110)
np.save('Analysis/pah_intensity_error_110', pah_intensity_error_110)

#%%

snr_cutoff_110 = 25

bnf.single_feature_imager(pah_intensity_110, pah_intensity_112, pah_intensity_error_110, '11.0', '110', snr_cutoff_110, current_reprojection)

np.save('Analysis/snr_cutoff_110', snr_cutoff_110)

#%%



'''
7.7 feature
'''



#continuum

image_data_77_cont = np.zeros((len(image_data_77[:,0,0]), array_length_x, array_length_y))

points77 = [6.55, 7.06, 9.08, 9.30] #last one is a filler value at the moment

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_77_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths77, image_data_77[:,i,j], points77) #note image_data_77 is built out of things with no lines

bnf.error_check_imager(wavelengths77, image_data_77, 'PDFtime/spectra_checking/077_check_continuum.pdf', 6.6, 9.5, 1.5, continuum=image_data_77_cont)

np.save('Analysis/image_data_77_cont', image_data_77_cont)

#%%

#local continuum

image_data_77_cont_local = np.zeros((len(image_data_77[:,0,0]), array_length_x, array_length_y))

points77_local = [7.06, 8.40, 8.90, 9.08] #last one is a filler value at the moment

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_77_cont_local[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths77, image_data_77[:,i,j], points77_local) #note image_data_112 is built out of things with no lines

bnf.error_check_imager(wavelengths77, image_data_77, 'PDFtime/spectra_checking/077_check_continuum_local.pdf', 6.6, 9.5, 1.5, continuum=image_data_77_cont_local)

np.save('Analysis/image_data_77_cont_local', image_data_77_cont_local)

#%%

#integration

pah_intensity_77_1 = np.zeros((array_length_x, array_length_y))
pah_intensity_77_2 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_77_1 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_77_2 = np.zeros((array_length_x, array_length_y))

#have 7.7 feature go from 7.05 (650) to 7.48 (1187) AND 7.80 (1480) to 8.05 (1670)

#note: 650-1187 all has an interval of 0.0008, 1480 to 1670 all has an interval of 0.0013

lower_index_77 = 650
middle_index_77_1 = 1187
middle_index_77_2 = 1480
upper_index_77 = 1670

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_77_1[i,j] = bnf.pah_feature_integrator(wavelengths77[middle_index_77_2:upper_index_77], 
                                                            image_data_77[middle_index_77_2:upper_index_77,i,j] - image_data_77_cont[middle_index_77_2:upper_index_77,i,j])
            pah_intensity_77_2[i,j] = bnf.pah_feature_integrator(wavelengths77[lower_index_77:middle_index_77_1], 
                                                            image_data_77[lower_index_77:middle_index_77_1,i,j] - image_data_77_cont[lower_index_77:middle_index_77_1,i,j])
    
pah_intensity_77 = pah_intensity_77_1 + pah_intensity_77_2

print('7.7 feature intensity calculated')

np.save('Analysis/lower_index_77', lower_index_77)
np.save('Analysis/middle_index_77_1', middle_index_77_1)
np.save('Analysis/middle_index_77_2', middle_index_77_2)
np.save('Analysis/upper_index_77', upper_index_77)
np.save('Analysis/pah_intensity_77', pah_intensity_77)

#%%

#need to be extra careful because the different parts of the intensity have different wavelength intervals

error_index_77 = 2470 #(9.1)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_77_1[i,j] = bnf.error_finder(wavelengths77[middle_index_77_2:], image_data_77[:,i,j] - image_data_77_cont[:,i,j], 7.25, 
                                                           pah_intensity_77_1[i,j], (middle_index_77_2, upper_index_77), error_index_77 - middle_index_77_2)
            pah_intensity_error_77_2[i,j] = bnf.error_finder(wavelengths77, image_data_77[:,i,j] - image_data_77_cont[:,i,j], 7.90, 
                                                           pah_intensity_77_2[i,j], (lower_index_77, middle_index_77_1), error_index_77)
            
pah_intensity_error_77 = (pah_intensity_error_77_1**2 + pah_intensity_error_77_2**2)**0.5

print('7.7 feature intensity error calculated')

np.save('Analysis/error_index_77', error_index_77)
np.save('Analysis/pah_intensity_error_77', pah_intensity_error_77)

#%%

snr_cutoff_77 = 300

bnf.single_feature_imager(pah_intensity_77, pah_intensity_112, pah_intensity_error_77, '7.7', '077', snr_cutoff_77, current_reprojection)

np.save('Analysis/snr_cutoff_77', snr_cutoff_77)

#%%



'''
8.6 feature
'''



#integration

pah_intensity_86 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_86 = np.zeros((array_length_x, array_length_y))

#have 8.6 feature go from 8.4 (1940) to 8.9 (2320)

lower_index_86 = 1940
upper_index_86 = 2320

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_86[i,j] = bnf.pah_feature_integrator(wavelengths77[lower_index_86:upper_index_86], 
                                                            image_data_77[lower_index_86:upper_index_86,i,j] - image_data_77_cont[lower_index_86:upper_index_86,i,j])
            
print('8.6 feature intensity calculated')

np.save('Analysis/lower_index_86', lower_index_86)
np.save('Analysis/upper_index_86', upper_index_86)
np.save('Analysis/pah_intensity_86', pah_intensity_86)

#%%

error_index_86 = np.copy(error_index_77)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_86[i,j] = bnf.error_finder(wavelengths77[lower_index_86:], image_data_77[:,i,j] - image_data_77_cont[:,i,j], 8.60, 
                                                           pah_intensity_86[i,j], (0, upper_index_86-lower_index_86), error_index_86 - lower_index_86)

print('8.6 feature intensity error calculated')

np.save('Analysis/error_index_86', error_index_86)
np.save('Analysis/pah_intensity_error_86', pah_intensity_error_86)

#%%

snr_cutoff_86 = 50

bnf.single_feature_imager(pah_intensity_86, pah_intensity_112, pah_intensity_error_86, '8.6', '086', snr_cutoff_86, current_reprojection)

np.save('Analysis/snr_cutoff_86', snr_cutoff_86)

#%%



'''
8.6 plateau feature
'''



#integration

pah_intensity_86_plat = np.zeros((array_length_x, array_length_y))
pah_intensity_error_86_plat = np.zeros((array_length_x, array_length_y))

#have the plateau the plateau go from 8.05 (1670) to 8.4

lower_index_86_plat = 1670
upper_index_86_plat = np.copy(lower_index_86)


for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_86_plat[i,j] = bnf.pah_feature_integrator(wavelengths77[lower_index_86_plat:upper_index_86_plat], 
                                                            image_data_77[lower_index_86_plat:upper_index_86_plat,i,j] - image_data_77_cont[lower_index_86_plat:upper_index_86_plat,i,j])
            
print('8.6 plateau feature intensity calculated')

np.save('Analysis/lower_index_86_plat', lower_index_86_plat)
np.save('Analysis/upper_index_86_plat', upper_index_86_plat)
np.save('Analysis/pah_intensity_86_plat', pah_intensity_86_plat)
            
#%%

error_index_86_plat = np.copy(error_index_77)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_86_plat[i,j] = bnf.error_finder(wavelengths77[lower_index_86_plat:], image_data_77[:,i,j] - image_data_77_cont[:,i,j], 8.2, 
                                                                pah_intensity_86_plat[i,j], (0, upper_index_86_plat-lower_index_86_plat), error_index_86_plat - lower_index_86_plat)

print('8.6 plateau feature intensity error calculated')

np.save('Analysis/error_index_86_plat', error_index_86_plat)
np.save('Analysis/pah_intensity_error_86_plat', pah_intensity_error_86_plat)

#%%

snr_cutoff_86_plat = 50

bnf.single_feature_imager(pah_intensity_86_plat, pah_intensity_112, pah_intensity_error_86_plat, '8.6 plateau', '086_plat', snr_cutoff_86_plat, current_reprojection)

np.save('Analysis/snr_cutoff_86_plat', snr_cutoff_86_plat)

#%%



'''
8.6 feature local continuum
'''

#integration

pah_intensity_86_local = np.zeros((array_length_x, array_length_y))
pah_intensity_error_86_local = np.zeros((array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_86_local[i,j] = bnf.pah_feature_integrator(wavelengths77[lower_index_86:upper_index_86], 
                                                            image_data_77[lower_index_86:upper_index_86,i,j] - image_data_77_cont_local[lower_index_86:upper_index_86,i,j])
    
print('8.6 feature local continuum intensity calculated')

np.save('Analysis/pah_intensity_86_local', pah_intensity_86_local)

#%%

error_index_86_local = np.copy(error_index_77)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_86_local[i,j] = bnf.error_finder(wavelengths77[lower_index_86:], image_data_77[:,i,j] - image_data_77_cont[:,i,j], 8.6, 
                                                                 pah_intensity_86_local[i,j], (0, upper_index_86-lower_index_86), error_index_86_local - lower_index_86)

print('8.6 feature local continuum intensity error calculated')

np.save('Analysis/error_index_86_local', error_index_86_local)
np.save('Analysis/pah_intensity_error_86_local', pah_intensity_error_86_local)

#%%

snr_cutoff_86_local = 20

bnf.single_feature_imager(pah_intensity_86_local, pah_intensity_112, pah_intensity_error_86_local, '8.6 local continuum', '086_local', snr_cutoff_86_local, current_reprojection)

np.save('Analysis/snr_cutoff_86_local', snr_cutoff_86_local)

#%%



'''
13.5 feature
'''



#continuum

image_data_135_cont = np.zeros((len(image_data_135[:,0,0]), array_length_x, array_length_y))

points135 = [13.21, 13.31, 13.83, 14.00] #last one is a filler value at the moment

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_135_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths135, image_data_135[:,i,j], points135) #note image_data_135 is built out of things with no lines

bnf.error_check_imager(wavelengths135, image_data_135, 'PDFtime/spectra_checking/135_check_continuum.pdf', 13.2, 14.0, 1.25, continuum=image_data_135_cont)

np.save('Analysis/image_data_135_cont', image_data_135_cont)

#%%

#integration

pah_intensity_135 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_135 = np.zeros((array_length_x, array_length_y))

#have 13.5 feature go from 13.31 (650) to 13.83 (910)

lower_index_135 = 705
upper_index_135 = 910

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_135[i,j] = bnf.pah_feature_integrator(wavelengths135[lower_index_135:upper_index_135], 
                                                            image_data_135[lower_index_135:upper_index_135,i,j] - image_data_135_cont[lower_index_135:upper_index_135,i,j])

print('13.5 feature intensity calculated')

np.save('Analysis/lower_index_135', lower_index_135)
np.save('Analysis/upper_index_135', upper_index_135)
np.save('Analysis/pah_intensity_135', pah_intensity_135)

#%%

error_index_135 = 981 #(14.01)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_135[i,j] = bnf.error_finder(wavelengths135, image_data_135[:,i,j] - image_data_135_cont[:,i,j], 13.5, 
                                                            pah_intensity_135[i,j], (lower_index_135, upper_index_135), error_index_135)
            
print('13.5 feature intensity error calculated')

np.save('Analysis/error_index_135', error_index_135)
np.save('Analysis/pah_intensity_error_135', pah_intensity_error_135)

#%%

snr_cutoff_135 = 50

bnf.single_feature_imager(pah_intensity_135, pah_intensity_112, pah_intensity_error_135, '13.5', '135', snr_cutoff_135, current_reprojection)

np.save('Analysis/snr_cutoff_135', snr_cutoff_135)

#%%



'''
5.7 feature
'''



#continuum

image_data_57_cont = np.zeros((len(image_data_57[:,0,0]), array_length_x, array_length_y))

points57 = [5.39, 5.55, 5.81, 5.94] #first point is used as an upper bound for 5.25 feature (5.39)

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_57_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths57, image_data_57[:,i,j], points57) #note image_data_57 is built out of things with no lines

bnf.error_check_imager(wavelengths57, image_data_57, 'PDFtime/spectra_checking/057_check_continuum.pdf', 5.3, 6.0, 1.5, continuum=image_data_57_cont)

np.save('Analysis/image_data_57_cont', image_data_57_cont)

#%%

#integration

pah_intensity_57 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_57 = np.zeros((array_length_x, array_length_y))

#have 5.7 feature go from 5.55 (810) to 5.81 (1130)

lower_index_57 = 810
upper_index_57 = 1130

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_57[i,j] = bnf.pah_feature_integrator(wavelengths57[lower_index_57:upper_index_57], 
                                                            image_data_57[lower_index_57:upper_index_57,i,j] - image_data_57_cont[lower_index_57:upper_index_57,i,j])

print('5.7 feature intensity calculated')

np.save('Analysis/lower_index_57', lower_index_57)
np.save('Analysis/upper_index_57', upper_index_57)
np.save('Analysis/pah_intensity_57', pah_intensity_57)

#%%

error_index_57 = 650 #(5.42, same as 5.25)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_57[i,j] = bnf.error_finder(wavelengths57, image_data_57[:,i,j] - image_data_57_cont[:,i,j], 5.7, 
                                                            pah_intensity_57[i,j], (lower_index_57, upper_index_57), error_index_57)
            
print('5.7 feature intensity error calculated')

np.save('Analysis/error_index_57', error_index_57)
np.save('Analysis/pah_intensity_error_57', pah_intensity_error_57)

#%%

snr_cutoff_57 = 20

bnf.single_feature_imager(pah_intensity_57, pah_intensity_112, pah_intensity_error_57, '5.7', '057', snr_cutoff_57, current_reprojection)

np.save('Analysis/snr_cutoff_57', snr_cutoff_57)

#%%



'''
5.9 feature
'''



#integration

pah_intensity_59 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_59 = np.zeros((array_length_x, array_length_y))

#have 5.9 feature go from 5.81 (1130) to 5.94 (1290)

lower_index_59 = np.copy(upper_index_57)
upper_index_59 = 1290

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_59[i,j] = bnf.pah_feature_integrator(wavelengths57[lower_index_59:upper_index_59], 
                                                            image_data_57[lower_index_59:upper_index_59,i,j] - image_data_57_cont[lower_index_59:upper_index_59,i,j])

print('5.9 feature intensity calculated')

np.save('Analysis/lower_index_59', lower_index_59)
np.save('Analysis/upper_index_59', upper_index_59)
np.save('Analysis/pah_intensity_59', pah_intensity_59)

#%%

error_index_59 = np.copy(error_index_57) #(5.42, same as 5.25)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_59[i,j] = bnf.error_finder(wavelengths57, image_data_57[:,i,j] - image_data_57_cont[:,i,j], 5.9, 
                                                            pah_intensity_59[i,j], (lower_index_59, upper_index_59), error_index_59)
            
print('5.9 feature intensity error calculated')

np.save('Analysis/error_index_59', error_index_59)
np.save('Analysis/pah_intensity_error_59', pah_intensity_error_59)

#%%

snr_cutoff_59 = 20

bnf.single_feature_imager(pah_intensity_59, pah_intensity_112, pah_intensity_error_59, '5.9', '059', snr_cutoff_59, current_reprojection)

np.save('Analysis/snr_cutoff_59', snr_cutoff_59)

#%%



'''
6.2 feature
'''



#continuum

image_data_1b_cont = np.zeros((len(image_data_1b[:,0,0]), array_length_x, array_length_y))

points62 = [5.68, 5.945, 6.53, 6.61]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_1b_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths1b, image_data_1b_noline[:,i,j], points62)
        
bnf.error_check_imager(wavelengths1b, image_data_1b_noline, 'PDFtime/spectra_checking/062_check_continuum.pdf', 5.7, 6.6, 1.5, continuum=image_data_1b_cont)

np.save('Analysis/image_data_1b_cont', image_data_1b_cont)

#%%

#integration

pah_intensity_62 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_62 = np.zeros((array_length_x, array_length_y))

#have the 6.2 feature go from 6.125 (581) to 6.5 (1050)

#note: upper limit changed from 6.6 (1175) to 6.5, due to there being no feature between 6.5 and 6.6

lower_index_62 = 581
upper_index_62 = 1050

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_62[i,j] = bnf.pah_feature_integrator(wavelengths1b[lower_index_62:upper_index_62], 
                                                            image_data_1b_noline[lower_index_62:upper_index_62,i,j] - image_data_1b_cont[lower_index_62:upper_index_62,i,j])
            
print('6.2 feature intensity calculated')

np.save('Analysis/lower_index_62', lower_index_62)
np.save('Analysis/upper_index_62', upper_index_62)
np.save('Analysis/pah_intensity_62', pah_intensity_62)

#%%

error_index_62 = 1110 # (6.55)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_62[i,j] = bnf.error_finder(wavelengths1b, image_data_1b_noline[:,i,j] - image_data_1b_cont[:,i,j], 6.2, 
                                                           pah_intensity_62[i,j], (lower_index_62, upper_index_62), error_index_62)
            
print('6.2 feature intensity error calculated')

np.save('Analysis/error_index_62', error_index_62)
np.save('Analysis/pah_intensity_error_62', pah_intensity_error_62)

#%%

snr_cutoff_62 = 200

bnf.single_feature_imager(pah_intensity_62, pah_intensity_112, pah_intensity_error_62, '6.2', '062', snr_cutoff_62, current_reprojection)

np.save('Analysis/snr_cutoff_62', snr_cutoff_62)

#%%



'''
6.0 feature
'''



#integration

pah_intensity_60 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_60 = np.zeros((array_length_x, array_length_y))

#have the 6.2 feature go from 5.968 (385) to 6.125 (581)

lower_index_60 = 385
upper_index_60 = np.copy(lower_index_62)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_60[i,j] = bnf.pah_feature_integrator(wavelengths1b[lower_index_60:upper_index_60], 
                                                            image_data_1b_noline[lower_index_60:upper_index_60,i,j] - image_data_1b_cont[lower_index_60:upper_index_60,i,j])

print('6.0 feature intensity calculated')

np.save('Analysis/lower_index_60', lower_index_60)
np.save('Analysis/upper_index_60', upper_index_60)
np.save('Analysis/pah_intensity_60', pah_intensity_60)
        
#%%

error_index_60 = np.copy(error_index_62) # (6.55)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_60[i,j] = bnf.error_finder(wavelengths1b, image_data_1b_noline[:,i,j] - image_data_1b_cont[:,i,j], 6.0, 
                                                           pah_intensity_60[i,j], (lower_index_60, upper_index_60), error_index_60)
            
print('6.0 feature intensity error calculated')

np.save('Analysis/error_index_60', error_index_60)
np.save('Analysis/pah_intensity_error_60', pah_intensity_error_60)

#%%

snr_cutoff_60 = 50

bnf.single_feature_imager(pah_intensity_60, pah_intensity_112, pah_intensity_error_60, '6.0', '060', snr_cutoff_60, current_reprojection)

np.save('Analysis/snr_cutoff_60', snr_cutoff_60)

#%%



'''
6.0 and 6.2 feature
'''



#integration

pah_intensity_60_and_62 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_60_and_62 = np.zeros((array_length_x, array_length_y))

lower_index_60_and_62 = np.copy(lower_index_60)
upper_index_60_and_62 = np.copy(upper_index_62)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_60_and_62[i,j] = bnf.pah_feature_integrator(wavelengths1b[lower_index_60_and_62:upper_index_60_and_62], 
                                                            image_data_1b_noline[lower_index_60_and_62:upper_index_60_and_62,i,j] - image_data_1b_cont[lower_index_60_and_62:upper_index_60_and_62,i,j])

print('6.0 and 6.2 feature intensity calculated')

np.save('Analysis/pah_intensity_60_and_62', pah_intensity_60_and_62)
        
#%%

error_index_60_and_62 = np.copy(error_index_62) # (6.55)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_60_and_62[i,j] = bnf.error_finder(wavelengths1b, image_data_1b_noline[:,i,j] - image_data_1b_cont[:,i,j], 6.2, 
                                                           pah_intensity_60[i,j], (lower_index_60_and_62, upper_index_60_and_62), error_index_60_and_62)
            
print('6.0 and 6.2 feature intensity error calculated')

np.save('Analysis/error_index_60_and_62', error_index_60_and_62)
np.save('Analysis/pah_intensity_error_60_and_62', pah_intensity_error_60_and_62)

#%%

snr_cutoff_60_and_62 = 200

bnf.single_feature_imager(pah_intensity_60_and_62, pah_intensity_112, pah_intensity_error_60_and_62, '6.0 and 6.2', '060_and_062', snr_cutoff_60_and_62, current_reprojection)

np.save('Analysis/snr_cutoff_60_and_62', snr_cutoff_60_and_62)

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

bnf.error_check_imager(wavelengths3a, image_data_3a_noline, 'PDFtime/spectra_checking/120_check_continuum.pdf', 11.6, 12.3, 1.25, continuum=image_data_3a_cont)

np.save('Analysis/image_data_3a_cont', image_data_3a_cont)

#%%

#integration

pah_intensity_120 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_120 = np.zeros((array_length_x, array_length_y))

#have 12.0 feature go from 11.80 (96) to 12.24 (276), this seems reasonable but theres arguments for slight adjustment i think

lower_index_120 = 96
upper_index_120 = 276

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_120[i,j] = bnf.pah_feature_integrator(wavelengths3a[lower_index_120:upper_index_120], 
                                                            image_data_3a_noline[lower_index_120:upper_index_120,i,j] - image_data_3a_cont[lower_index_120:upper_index_120,i,j])

print('12.0 feature intensity calculated')

np.save('Analysis/lower_index_120', lower_index_120)
np.save('Analysis/upper_index_120', upper_index_120)
np.save('Analysis/pah_intensity_120', pah_intensity_120)

#%%

error_index_120 = 60 #(11.70)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_120[i,j] = bnf.error_finder(wavelengths3a, image_data_3a_noline[:,i,j] - image_data_3a_cont[:,i,j], 12.0, 
                                                              pah_intensity_120[i,j], (lower_index_120, upper_index_120), error_index_120)

print('12.0 feature intensity error calculated')

np.save('Analysis/error_index_120', error_index_120)
np.save('Analysis/pah_intensity_error_120', pah_intensity_error_120)

#%%

snr_cutoff_120 = 50

bnf.single_feature_imager(pah_intensity_120, pah_intensity_112, pah_intensity_error_120, '12.0', '120', snr_cutoff_120, current_reprojection)

np.save('Analysis/snr_cutoff_120', snr_cutoff_120)

#%%



'''
16.4 feature
'''

import ButterflyNebulaFunctions as bnf

#continuum

#make 3 continua, as the 16.4 feature seems to be present as a strong version with a red wing, and a weaker version with no red wing.

image_data_3c_cont_1 = np.zeros((len(image_data_3c[:,0,0]), array_length_x, array_length_y))

points164_1 = [16.12, 16.27, 16.73, 16.85]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_3c_cont_1[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths3c, image_data_3c_noline[:,i,j], points164_1)
        
bnf.error_check_imager(wavelengths3c, image_data_3c, 'PDFtime/spectra_checking/164_check_continuum_1.pdf', 16.1, 16.9, 1, continuum=image_data_3c_cont_1)

np.save('Analysis/image_data_3c_cont_1', image_data_3c_cont_1)
'''
image_data_3c_cont_2 = np.zeros((len(image_data_3c[:,0,0]), array_length_x, array_length_y))

points164_2 = [16.12, 16.27, 16.63, 16.75]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_3c_cont_2[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths3c, image_data_3c_noline[:,i,j], points164_2)
        
bnf.error_check_imager(wavelengths3c, image_data_3c, 'PDFtime/spectra_checking/164_check_continuum_2.pdf', 16.1, 16.9, 1, continuum=image_data_3c_cont_2)

np.save('Analysis/image_data_3c_cont_2', image_data_3c_cont_2)

image_data_3c_cont_3 = np.zeros((len(image_data_3c[:,0,0]), array_length_x, array_length_y))

points164_3 = [16.12, 16.27, 16.53, 16.65]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_3c_cont_3[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths3c, image_data_3c_noline[:,i,j], points164_3)
        
import ButterflyNebulaFunctions as bnf
bnf.error_check_imager(wavelengths3c, image_data_3c, 'PDFtime/spectra_checking/164_check_continuum_3.pdf', 16.1, 16.9, 1, continuum=image_data_3c_cont_3)

np.save('Analysis/image_data_3c_cont_3', image_data_3c_cont_3)
'''
#%%
'''
image_data_3c_cont = np.copy(image_data_3c_cont_1)

#make an array to keep track of which 16.4 continuum is being used
cont_type_164 = np.ones((array_length_x, array_length_y))

#note these are named based off of which continuum the point belongs to, and so appear backwards than what would otherwise be expected; [1:2] is expected but this is [2:1]
temp_index_1 = np.where(np.round(wavelengths3c, 2) == points164_1[2])[0][0] #approx 16.73
temp_index_2 = np.where(np.round(wavelengths3c, 2) == points164_2[2])[0][0] #approx 16.63
temp_index_3 = np.where(np.round(wavelengths3c, 2) == points164_3[2])[0][0] #approx 16.53

for i in range(array_length_x):
    for j in range(array_length_y):
        if np.median(image_data_3c_noline[temp_index_2:temp_index_1,i,j] - image_data_3c_cont[temp_index_2:temp_index_1,i,j]) < 0:
            cont_type_164[i,j] += 1
            image_data_3c_cont[:,i,j] = image_data_3c_cont_2[:,i,j]
        if np.median(image_data_3c_noline[temp_index_3:temp_index_2,i,j] - image_data_3c_cont[temp_index_3:temp_index_2,i,j]) < 0:
            cont_type_164[i,j] += 1
            image_data_3c_cont[:,i,j] = image_data_3c_cont_3[:,i,j]      
'''
image_data_3c_cont = np.copy(image_data_3c_cont_1)

bnf.error_check_imager(wavelengths3c, image_data_3c, 'PDFtime/spectra_checking/164_check_continuum.pdf', 16.1, 16.9, 1, continuum=image_data_3c_cont)

#np.save('Analysis/cont_type_164', cont_type_164)
np.save('Analysis/image_data_3c_cont', image_data_3c_cont)

#%%

#integration

pah_intensity_164 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_164 = np.zeros((array_length_x, array_length_y))

#have 16.4 feature go from 16.27 (344) to 16.70 (514), this seems reasonable but theres arguments for slight adjustment i think

#originally went to 16.67 (504)

lower_index_164 = 344
upper_index_164 = 514

#pah integral negative value flag
pah_intensity_164_flag = np.zeros((array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_164[i,j] = bnf.pah_feature_integrator(wavelengths3c[lower_index_164:upper_index_164], 
                                                            image_data_3c_noline[lower_index_164:upper_index_164,i,j] - image_data_3c_cont[lower_index_164:upper_index_164,i,j])
            #this one has a lot of integrals that go negative, set them to zero for now, i can properly address this when doing a better
            #continuum fit down the road.
        if pah_intensity_164[i,j] < 0:
            pah_intensity_164[i,j] = 0
            pah_intensity_164_flag[i,j] = 1

print('16.4 feature intensity calculated')

np.save('Analysis/lower_index_164', lower_index_164)
np.save('Analysis/upper_index_164', upper_index_164)
np.save('Analysis/pah_intensity_164', pah_intensity_164)

#%%

import ButterflyNebulaFunctions as bnf

error_index_164 = 560 # (16.81)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_164[i,j] = bnf.error_finder(wavelengths3c, image_data_3c_noline[:,i,j] - image_data_3c_cont[:,i,j], 16.4, 
                                                            pah_intensity_164[i,j], (lower_index_164, upper_index_164), error_index_164)

print('16.4 feature intensity error calculated')

np.save('Analysis/error_index_164', error_index_164)
np.save('Analysis/pah_intensity_error_164', pah_intensity_error_164)

#%%

snr_cutoff_164 = 50

bnf.single_feature_imager(pah_intensity_164, pah_intensity_112, pah_intensity_error_164, '16.4', '164', snr_cutoff_164, current_reprojection)

np.save('Analysis/snr_cutoff_164', snr_cutoff_164)

#%%

ax = plt.figure(figsize=(10,10)).add_subplot(111)
plt.imshow(pah_intensity_164/pah_intensity_error_164, vmin=30)

plt.colorbar() 
plt.scatter([100], [61], color='black')
ax.invert_yaxis()
plt.show()

#%%
i = 80
j = 64

plt.figure()
plt.plot(wavelengths3c, image_data_3c_noline[:,i,j] - image_data_3c_cont[:,i,j])
plt.show()

#%%

'''
5.25 feature
'''



#continuum

image_data_1a_cont = np.zeros((len(image_data_1a[:,0,0]), array_length_x, array_length_y))

points52 = [5.06, 5.15, 5.39, 5.55]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_1a_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths1a, image_data_1a_noline[:,i,j], points52)
        
bnf.error_check_imager(wavelengths1a, image_data_1a, 'PDFtime/spectra_checking/052_check_continuum.pdf', 5.0, 5.6, 1.5, continuum=image_data_1a_cont)

np.save('Analysis/image_data_1a_cont', image_data_1a_cont)

#%%

#integration

pah_intensity_52 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_52 = np.zeros((array_length_x, array_length_y))

#have 5.25 feature go from 5.17 (337) to 5.40 (624), this seems reasonable but theres arguments for slight adjustment i think

#originally went to 16.67 (504)

lower_index_52 = 337
upper_index_52 = 624

#pah integral negative value flag
pah_intensity_52_flag = np.zeros((array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_52[i,j] = bnf.pah_feature_integrator(wavelengths1a[lower_index_52:upper_index_52], 
                                                            image_data_1a_noline[lower_index_52:upper_index_52,i,j] - image_data_1a_cont[lower_index_52:upper_index_52,i,j])

print('5.25 feature intensity calculated')

np.save('Analysis/lower_index_52', lower_index_52)
np.save('Analysis/upper_index_52', upper_index_52)
np.save('Analysis/pah_intensity_52', pah_intensity_52)

#%%

error_index_52 = np.copy(error_index_57) # (5.42)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_52[i,j] = bnf.error_finder(wavelengths1a, image_data_1a_noline[:,i,j] - image_data_1a_cont[:,i,j], 5.25, 
                                                            pah_intensity_52[i,j], (lower_index_52, upper_index_52), error_index_52)

print('5.25 feature intensity error calculated')

np.save('Analysis/error_index_52', error_index_52)
np.save('Analysis/pah_intensity_error_52', pah_intensity_error_52)

#%%

snr_cutoff_52 = 20

bnf.single_feature_imager(pah_intensity_52, pah_intensity_112, pah_intensity_error_52, '5.25', '052', snr_cutoff_52, current_reprojection)

np.save('Analysis/snr_cutoff_52', snr_cutoff_52)

#%%

'''
6.9 feature
'''



#continuum

image_data_1c_cont = np.zeros((len(image_data_1c[:,0,0]), array_length_x, array_length_y))

points69 = [6.70, 6.80, 6.90, 7.05]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_1c_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths1c, image_data_1c_noline[:,i,j], points69)
        
bnf.error_check_imager(wavelengths1c, image_data_1c_noline, 'PDFtime/spectra_checking/069_check_continuum.pdf', 6.6, 7.3, 1, continuum=image_data_1c_cont)

np.save('Analysis/image_data_1c_cont', image_data_1c_cont)

#%%

#integration

pah_intensity_69 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_69 = np.zeros((array_length_x, array_length_y))

#have 6.9 feature go from 6.80 (335) to 6.90 (460), normally this feature should go to like 7.1 but that stuff is clearly missing, so instead im integrating over a little bump that is present.

lower_index_69 = 335
upper_index_69 = 460

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_69[i,j] = bnf.pah_feature_integrator(wavelengths1c[lower_index_69:upper_index_69], 
                                                            image_data_1c_noline[lower_index_69:upper_index_69,i,j] - image_data_1c_cont[lower_index_69:upper_index_69,i,j])

print('6.9 feature intensity calculated')

np.save('Analysis/lower_index_69', lower_index_69)
np.save('Analysis/upper_index_69', upper_index_69)
np.save('Analysis/pah_intensity_69', pah_intensity_69)

#%%

error_index_69 = 150 # (6.65)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_69[i,j] = bnf.error_finder(wavelengths1c, image_data_1c_noline[:,i,j] - image_data_1c_cont[:,i,j], 6.9, 
                                                            pah_intensity_69[i,j], (lower_index_69, upper_index_69), error_index_69)

print('6.9 feature intensity error calculated')

np.save('Analysis/error_index_69', error_index_69)
np.save('Analysis/pah_intensity_error_69', pah_intensity_error_69)

#%%

snr_cutoff_69 = 20

bnf.single_feature_imager(pah_intensity_69, pah_intensity_112, pah_intensity_error_69, '6.9', '069', snr_cutoff_69, current_reprojection)

np.save('Analysis/snr_cutoff_69', snr_cutoff_69)

#%%



'''
15.8 feature
'''



#continuum

#make 2 continua, as the 16.4 feature seems to be present as a strong version with a red wing, and a weaker version with no red wing.

image_data_158_cont = np.zeros((len(image_data_3c[:,0,0]), array_length_x, array_length_y))

points158 = [15.65, 15.74, 16.04, 16.22]

for i in range(array_length_x):
    for j in range(array_length_y):
        image_data_158_cont[:,i,j] = bnf.linear_continuum_single_channel(
            wavelengths3c, image_data_3c_noline[:,i,j], points158)
#%%
bnf.error_check_imager(wavelengths3c, image_data_3c, 'PDFtime/spectra_checking/158_check_continuum.pdf', 15.6, 16.2, 1, continuum=image_data_158_cont)

np.save('Analysis/image_data_158_cont', image_data_158_cont)

#%%

#integration

pah_intensity_158 = np.zeros((array_length_x, array_length_y))
pah_intensity_error_158 = np.zeros((array_length_x, array_length_y))

#have 15.8 feature go from 15.70 (115) to 16.02 (243), this seems reasonable but theres arguments for slight adjustment i think

#originally went to 16.67 (504)

lower_index_158 = 115
upper_index_158 = 243

#pah integral negative value flag
pah_intensity_158_flag = np.zeros((array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_158[i,j] = bnf.pah_feature_integrator(wavelengths3c[lower_index_158:upper_index_158], 
                                                            image_data_3c_noline[lower_index_158:upper_index_158,i,j] - image_data_158_cont[lower_index_158:upper_index_158,i,j])
            #this one has a lot of integrals that go negative, set them to zero for now, i can properly address this when doing a better
            #continuum fit down the road.
        if pah_intensity_158[i,j] < 0:
            pah_intensity_158[i,j] = 0
            pah_intensity_158_flag[i,j] = 1

print('15.8 feature intensity calculated')

np.save('Analysis/lower_index_158', lower_index_158)
np.save('Analysis/upper_index_158', upper_index_158)
np.save('Analysis/pah_intensity_158', pah_intensity_158)

#%%

error_index_158 = 95 # (15.65)

for i in range(array_length_x):
    for j in range(array_length_y):
        if region_indicator[i,j] == 1:
            pah_intensity_error_158[i,j] = bnf.error_finder(wavelengths3c, image_data_3c_noline[:,i,j] - image_data_158_cont[:,i,j], 15.8, 
                                                            pah_intensity_158[i,j], (lower_index_158, upper_index_158), error_index_158)

#removing some nans that show up
pah_intensity_error_158[np.isnan(pah_intensity_error_158)] = 0

print('15.8 feature intensity error calculated')

np.save('Analysis/error_index_158', error_index_158)
np.save('Analysis/pah_intensity_error_158', pah_intensity_error_158)

#%%

snr_cutoff_158 = 45

bnf.single_feature_imager(pah_intensity_158, pah_intensity_112, pah_intensity_error_158, '15.8', '158', snr_cutoff_158, current_reprojection)

np.save('Analysis/snr_cutoff_158', snr_cutoff_158)

#%%



'''
CENTROID CALCULATIONS
'''



#%%
#6.2

poi_list_62 = [
    [57, 46], 
    [48, 40]]

wavelengths_centroid_62, wavelengths_centroid_error_62 = bnf.feature_centroid(wavelengths1b[lower_index_62:upper_index_62], 
                                                                              image_data_1b_noline[lower_index_62:upper_index_62] - image_data_1b_cont[lower_index_62:upper_index_62], 
                                                                              wavelengths1b, image_data_1b_noline - image_data_1b_cont, 
                                                                              pah_intensity_62, pah_intensity_error_62, region_indicator, '6.2', '062', snr_cutoff_62, current_reprojection,
                                                                              [lower_index_62, upper_index_62], poi_list_62)

np.save('Analysis/wavelengths_centroid_62', wavelengths_centroid_62)
np.save('Analysis/wavelengths_centroid_error_62', wavelengths_centroid_error_62)

#%%

#8.6 local

poi_list_86_local = [
    [57, 46], 
    [48, 40]]

wavelengths_centroid_86_local, wavelengths_centroid_error_86_local = bnf.feature_centroid(wavelengths77[lower_index_86:upper_index_86], 
                                                                              image_data_77[lower_index_86:upper_index_86] - image_data_77_cont_local[lower_index_86:upper_index_86], 
                                                                              wavelengths77[lower_index_86-100:upper_index_86+100], 
                                                                              image_data_77[lower_index_86-100:upper_index_86+100] - image_data_77_cont_local[lower_index_86-100:upper_index_86+100], 
                                                                              pah_intensity_86_local, pah_intensity_error_86_local, region_indicator, '8.6 local', '086_local', snr_cutoff_86_local, current_reprojection, 
                                                                              [100, -100], poi_list_86_local)

np.save('Analysis/wavelengths_centroid_86_local', wavelengths_centroid_86_local)
np.save('Analysis/wavelengths_centroid_error_86_local', wavelengths_centroid_error_86_local)

#%%
import ButterflyNebulaFunctions as bnf
#11.2

poi_list_112 = [
    [41, 51], 
    [49, 42],
    [66, 63]]

#making an index to use with plotting

index_centroid_lower_112 = np.where(np.round(wavelengths112, 2) == 10.1)[0][0]
index_centroid_upper_112 = np.where(np.round(wavelengths112, 2) == 11.9)[0][0]

wavelengths_centroid_112, wavelengths_centroid_error_112 = bnf.feature_centroid(wavelengths112[lower_index_112:upper_index_112], 
                                                                              image_data_112[lower_index_112:upper_index_112] - image_data_112_cont[lower_index_112:upper_index_112], 
                                                                              wavelengths112[:index_centroid_upper_112], 
                                                                              image_data_112[:index_centroid_upper_112] - image_data_112_cont[:index_centroid_upper_112], 
                                                                              pah_intensity_112, pah_intensity_error_112, region_indicator, '11.2', '112', snr_cutoff_112, current_reprojection, 
                                                                              [lower_index_112, upper_index_112], poi_list_112)

np.save('Analysis/wavelengths_centroid_112', wavelengths_centroid_112)
np.save('Analysis/wavelengths_centroid_error_112', wavelengths_centroid_error_112)

#%%
