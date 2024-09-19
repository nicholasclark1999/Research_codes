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

#fringe removal
from jwst.residual_fringe.utils import fit_residual_fringes_1d as rf1d

    

    
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




#%%

#all arrays should have same spacial x and y dimensions, so define variables for this to use in for loops
array_length_x = len(image_data_1a[0,:,0])
array_length_y = len(image_data_1a[0,0,:])

#%%

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

'''
FRINGE REMOVAL
'''

#%%

#need to remove fringes for channel 2C, channel 3


#using dq array for data with funky pixels
image_file = get_pkg_data_filename('data/ngc6302_ch2-long_s3d.fits')
dq_data_2c = fits.getdata(image_file, ext=3)



image_data_2c_fringes = np.copy(image_data_2c)

for i in range(array_length_x):
    for j in range(array_length_y):
        print(i,j)
        if (np.max(image_data_2c_fringes[:,i,j]) != np.min(image_data_2c_fringes[:,i,j])) and\
            (np.max(dq_data_2c[:,i,j]) == np.min(dq_data_2c[:,i,j])):
            print('passed check')
            image_data_2c[:,i,j] = rf1d(image_data_2c_fringes[:,i,j], wavelengths2c, channel=2)

#%%

image_data_3a_fringes = np.copy(image_data_3a)

for i in range(array_length_x):
    for j in range(array_length_y):
        if np.max(image_data_3a_fringes[:,i,j]) != np.min(image_data_3a_fringes[:,i,j]):
            image_data_3a[:,i,j] = rf1d(image_data_3a_fringes[:,i,j], wavelengths3a, channel=3)
        
image_data_3b_fringes = np.copy(image_data_3b)

for i in range(array_length_x):
    for j in range(array_length_y):
        if np.max(image_data_3b_fringes[:,i,j]) != np.min(image_data_3b_fringes[:,i,j]):
            image_data_3b[:,i,j] = rf1d(image_data_3b_fringes[:,i,j], wavelengths3b, channel=3)
        
image_data_3c_fringes = np.copy(image_data_3c)

for i in range(array_length_x):
    for j in range(array_length_y):
        if np.max(image_data_3c_fringes[:,i,j]) != np.min(image_data_3c_fringes[:,i,j]):
            image_data_3c[:,i,j] = rf1d(image_data_3c_fringes[:,i,j], wavelengths3c, channel=3)



#saving data
np.save('Analysis/image_data_2c_fringes', image_data_2c_fringes)
np.save('Analysis/image_data_3a_fringes', image_data_3a_fringes)
np.save('Analysis/image_data_3b_fringes', image_data_3b_fringes)
np.save('Analysis/image_data_3c_fringes', image_data_3c_fringes)

print('Fringe removal Complete')



'''
EMISSION LINE REMOVAL
'''


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

#defining a separate list of lines not on kevins list, note these only appear sometimes (investigate later maybe)

wave_list_3b_extra = [
    [13.5230, 13.5310],
    [13.5930, 13.6140]]

for i in range(len(wave_list_3b_extra)):
    image_data_3b_noline = bnf.emission_line_remover_wrapper(wavelengths3b, image_data_3b_noline, np.round(wave_list_3b_extra[i], 3))

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

#defining a separate list of lines not on kevins list, note these only appear sometimes (investigate later maybe)

wave_list_3c_extra = [
    [16.3960, 16.4160]]

for i in range(len(wave_list_3c_extra)):
    image_data_3c_noline = bnf.emission_line_remover_wrapper(wavelengths3c, image_data_3c_noline, np.round(wave_list_3c_extra[i], 3))

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
SPECTA STITCHING ERRORS
'''

#11.2 feature

#combining channels 2C and 3A to get proper continua for the 11.2 feature

error_data_112_temp, wavelengths112, overlap112_temp = bnf.flux_aligner3(wavelengths2c, wavelengths3a, error_data_2c[:,50,50], error_data_3a[:,50,50])



#using the above to make an array of the correct size to fill
error_data_112 = np.zeros((len(error_data_112_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_112[:,i,j], wavelengths112, overlap112 = bnf.flux_aligner3(wavelengths2c, wavelengths3a, error_data_2c[:,i,j], error_data_3a[:,i,j])

print('11.2 feature stitching complete')

np.save('Analysis/error_data_112', error_data_112)

#%%

#combining channels 1C, 2A, and 2B to get proper continua for the 7.7 and 8.6 features

error_data_77_temp, wavelengths77, overlap77_temp = bnf.flux_aligner4(wavelengths1c, wavelengths2a, error_data_1c[:,50,50], error_data_2a[:,50,50])
error_data_77_temp_temp, wavelengths77, overlap77_temp = bnf.flux_aligner3(wavelengths77, wavelengths2b, error_data_77_temp, error_data_2b[:,50,50])



#using the above to make an array of the correct size to fill
error_data_77_1 = np.zeros((len(error_data_77_temp), array_length_x, array_length_y))

error_data_77 = np.zeros((len(error_data_77_temp_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_77_1[:,i,j], wavelengths77_1, overlap77 = bnf.flux_aligner4(wavelengths1c, wavelengths2a, error_data_1c[:,i,j], error_data_2a[:,i,j])
        
for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_77[:,i,j], wavelengths77, overlap77 = bnf.flux_aligner3(wavelengths77_1, wavelengths2b, error_data_77_1[:,i,j], error_data_2b[:,i,j])

print('7.7 and 8.6 features stitching complete')

np.save('Analysis/error_data_77', error_data_77)

#%%

#combining channels 3A and 3B to get proper continua for the 13.5 feature

error_data_135_temp, wavelengths135, overlap135_temp = bnf.flux_aligner3(wavelengths3a, wavelengths3b, error_data_3a[:,50,50], error_data_3b[:,50,50])



#using the above to make an array of the correct size to fill
error_data_135 = np.zeros((len(error_data_135_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_135[:,i,j], wavelengths135, overlap135 =bnf. flux_aligner3(wavelengths3a, wavelengths3b, error_data_3a[:,i,j], error_data_3b[:,i,j])
    
print('13.5 feature stitching complete')    

np.save('Analysis/error_data_135', error_data_135)

#%%

#combining channels 1A and 1B to get proper continua for the 5.7 feature

error_data_57_temp, wavelengths57, overlap57_temp = bnf.flux_aligner3(wavelengths1a, wavelengths1b, error_data_1a[:,50,50], error_data_1b[:,50,50])



#using the above to make an array of the correct size to fill
error_data_57 = np.zeros((len(error_data_57_temp), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_57[:,i,j], wavelengths57, overlap57 =bnf. flux_aligner3(wavelengths1a, wavelengths1b, error_data_1a[:,i,j], error_data_1b[:,i,j])
    
print('5.7 feature stitching complete')    

np.save('Analysis/error_data_57', error_data_57)
    
#%%

#combining all channels to get proper continua for the 23.0 crystalline silicate feature
#note channels 1C, 2A, 2B combined into 77 already, and 2C, 3A combined into 112 already

error_data_230cs_temp_1, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths1a, wavelengths1b, error_data_1a[:,50,50], error_data_1b[:,50,50])
error_data_230cs_temp_2, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths77, error_data_230cs_temp_1, error_data_77[:,50,50])
error_data_230cs_temp_3, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths112, error_data_230cs_temp_2, error_data_112[:,50,50])
#error_data_230cs_temp_4, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths3b, error_data_230cs_temp_3, error_data_3b[:,50,50])

error_data_230cs_temp_4, wavelengths230cs_temp, overlap230cs_temp = bnf.flux_aligner2(wavelengths3b, wavelengths3c, error_data_3b[:,50,50], error_data_3c[:,50,50])
error_data_230cs_temp_5, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths230cs_temp, error_data_230cs_temp_3, error_data_230cs_temp_4)

#error_data_230cs_temp_5, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner2(wavelengths230cs, wavelengths3c, error_data_230cs_temp_4, error_data_3c[:,50,50])
error_data_230cs_temp_6, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths4a, error_data_230cs_temp_5, error_data_4a[:,50,50])
error_data_230cs_temp_7, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths4b, error_data_230cs_temp_6, error_data_4b[:,50,50])
error_data_230cs_temp_8, wavelengths230cs, overlap230cs_temp = bnf.flux_aligner3(wavelengths230cs, wavelengths4c, error_data_230cs_temp_7, error_data_4c[:,50,50])

#%%

#using the above to make an array of the correct size to fill
error_data_230cs_1 = np.zeros((len(error_data_230cs_temp_1), array_length_x, array_length_y))
error_data_230cs_2 = np.zeros((len(error_data_230cs_temp_2), array_length_x, array_length_y))
error_data_230cs_3 = np.zeros((len(error_data_230cs_temp_3), array_length_x, array_length_y))
error_data_230cs_4 = np.zeros((len(error_data_230cs_temp_4), array_length_x, array_length_y))
error_data_230cs_5 = np.zeros((len(error_data_230cs_temp_5), array_length_x, array_length_y))
error_data_230cs_6 = np.zeros((len(error_data_230cs_temp_6), array_length_x, array_length_y))
#%%
error_data_230cs_7 = np.zeros((len(error_data_230cs_temp_7), array_length_x, array_length_y))
error_data_230cs = np.zeros((len(error_data_230cs_temp_8), array_length_x, array_length_y))

for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_230cs_1[:,i,j], wavelengths230cs_1, overlap230cs = bnf.flux_aligner3(wavelengths1a, wavelengths1b, error_data_1a[:,i,j], error_data_1b[:,i,j])
        
for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_230cs_2[:,i,j], wavelengths230cs_2, overlap230cs = bnf.flux_aligner3(wavelengths230cs_1, wavelengths77, error_data_230cs_1[:,i,j], error_data_77[:,i,j]) 
        
for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_230cs_3[:,i,j], wavelengths230cs_3, overlap230cs = bnf.flux_aligner3(wavelengths230cs_2, wavelengths112, error_data_230cs_2[:,i,j], error_data_112[:,i,j]) 



for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_230cs_4[:,i,j], wavelengths230cs_temp, overlap230cs_temp = bnf.flux_aligner2(wavelengths3b, wavelengths3c, error_data_3b[:,i,j], error_data_3c[:,i,j])

for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_230cs_5[:,i,j], wavelengths230cs_5, overlap230cs = bnf.flux_aligner3(wavelengths230cs_3, wavelengths230cs_temp, error_data_230cs_3[:,i,j], error_data_230cs_4[:,i,j])

'''
for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_230cs_4[:,i,j], wavelengths230cs_4, overlap230cs = bnf.flux_aligner3(wavelengths230cs_3, wavelengths3b, error_data_230cs_3[:,i,j], error_data_3b[:,i,j])

for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_230cs_5[:,i,j], wavelengths230cs_5, overlap230cs = bnf.flux_aligner3(wavelengths230cs_4, wavelengths3c, error_data_230cs_4[:,i,j], error_data_3c[:,i,j])
'''
for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_230cs_6[:,i,j], wavelengths230cs_6, overlap230cs = bnf.flux_aligner3(wavelengths230cs_5, wavelengths4a, error_data_230cs_5[:,i,j], error_data_4a[:,i,j]) 
        
for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_230cs_7[:,i,j], wavelengths230cs_7, overlap230cs = bnf.flux_aligner3(wavelengths230cs_6, wavelengths4b, error_data_230cs_6[:,i,j], error_data_4b[:,i,j]) 
        
for i in range(array_length_x):
    for j in range(array_length_y):
        error_data_230cs[:,i,j], wavelengths230cs, overlap230cs = bnf.flux_aligner3(wavelengths230cs_7, wavelengths4c, error_data_230cs_7[:,i,j], error_data_4c[:,i,j]) 

print('23.0 feature stitching complete')

np.save('Analysis/error_data_230cs', error_data_230cs)

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



