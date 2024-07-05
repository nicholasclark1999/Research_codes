#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 17:30:34 2024

@author: nclark
"""

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

import lmfit

wavelengths_nirspec, nirspec_data, nirspec_error_data = rnf.loading_function(
    'data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 1)

nirspec_data_new, _ = rnf.regrid(nirspec_data, nirspec_error_data, 2)

nirspec_data_new[np.isnan(nirspec_data_new)] = 0

def gaussian_plus_line(x, a, b):
    mean = 3.29027
    std = 0.016434376835573167
    return a*np.exp(-1*((x - mean)**2)/(2*std**2)) + b



#lmfit

fit_params = lmfit.Parameters()
fit_params.add(value=2.5, name = 'amp')
fit_params.add(value=0, name= 'constant')



def myfunc(params, x, data):
    amp = params['amp'].value
    constant = params['constant'].value
    
    model = gaussian_plus_line(x, amp, constant)
    
    residual_array = data - model
    
    return residual_array


gaussians = np.copy(nirspec_data_new)

amplitudes = np.copy(nirspec_data_new[0])
constants = np.copy(nirspec_data_new[0])





for y in range(len(nirspec_data_new[0,:,0])):
    for x in range(len(nirspec_data_new[0,0,:])):
        data = np.hstack((nirspec_data_new[140:290,y,x], nirspec_data_new[365:420,y,x]))
        wave = np.hstack((wavelengths_nirspec[140:290], wavelengths_nirspec[365:420]))
        result = lmfit.minimize(myfunc, fit_params, args=(wave, rnf.emission_line_remover(data, 5, 1)))
        is_fit_good = result.params
        amp = is_fit_good['amp'].value
        constant = is_fit_good['constant'].value
        
        model_best2 = gaussian_plus_line(wavelengths_nirspec, amp, constant)
        
        gaussians[:,y,x] = model_best2
        amplitudes[y,x] = amp
        constants[y,x] = constant
        
  
#%%
        
ax = plt.figure().add_subplot(1,1,1)
plt.title('fit amplitudes')
plt.imshow(amplitudes, vmin=0)
ax.invert_yaxis()
plt.colorbar()
plt.show()

ax = plt.figure().add_subplot(1,1,1)
plt.title('fit constants')
plt.imshow(constants)
ax.invert_yaxis()
plt.colorbar()
plt.show()
'''
plt.figure()
plt.title('gaussian integrals')
plt.imshow(amplitudes)
plt.show()
'''
#%%

x = 11
y = 5


plt.figure()
plt.title(str(x) + ', ' + str(y))
plt.plot(wavelengths_nirspec, nirspec_data_new[:,y,x])
plt.plot(wavelengths_nirspec, gaussians[:,y,x])
plt.ylim(0, 10)
plt.xlim(3.1, 3.7)
plt.show()

#%%

#now doing west region

wavelengths_nirspec_west, nirspec_data_west, nirspec_error_data_west = rnf.loading_function(
    'data/west/jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked.fits', 1)

nirspec_data_new_west, _ = rnf.regrid(nirspec_data_west, nirspec_error_data_west, 2)

nirspec_data_new_west[np.isnan(nirspec_data_new_west)] = 0

def gaussian_plus_line(x, a, b):
    mean = 3.29027
    std = 0.016434376835573167
    return a*np.exp(-1*((x - mean)**2)/(2*std**2)) + b



#lmfit

fit_params = lmfit.Parameters()
fit_params.add(value=2.5, name = 'amp')
fit_params.add(value=0, name= 'constant')



def myfunc(params, x, data):
    amp = params['amp'].value
    constant = params['constant'].value
    
    model = gaussian_plus_line(x, amp, constant)
    
    residual_array = data - model
    
    return residual_array


gaussians_west = np.copy(nirspec_data_new_west)

amplitudes_west = np.copy(nirspec_data_new_west[0])
constants_west = np.copy(nirspec_data_new_west[0])





for y in range(len(nirspec_data_new_west[0,:,0])):
    for x in range(len(nirspec_data_new_west[0,0,:])):
        data = np.hstack((nirspec_data_new_west[140:290,y,x], nirspec_data_new_west[365:420,y,x]))
        wave = np.hstack((wavelengths_nirspec_west[140:290], wavelengths_nirspec_west[365:420]))
        result_west = lmfit.minimize(myfunc, fit_params, args=(wave, rnf.emission_line_remover(data, 5, 1)))
        is_fit_good_west = result_west.params
        amp_west = is_fit_good_west['amp'].value
        constant_west = is_fit_good_west['constant'].value
        
        model_best2_west = gaussian_plus_line(wavelengths_nirspec_west, amp_west, constant_west)
        
        gaussians_west[:,y,x] = model_best2_west
        amplitudes_west[y,x] = amp_west
        constants_west[y,x] = constant_west
        
        
#%%
        
ax = plt.figure().add_subplot(1,1,1)
plt.title('fit amplitudes')
plt.imshow(amplitudes_west, vmin=0)
ax.invert_yaxis()
plt.colorbar()
plt.show()

ax = plt.figure().add_subplot(1,1,1)
plt.title('fit constants')
plt.imshow(constants_west)
ax.invert_yaxis()
plt.colorbar()
plt.show()
'''
plt.figure()
plt.title('gaussian integrals')
plt.imshow(amplitudes)
plt.show()
'''
#%%

x = 12
y = 11


plt.figure()
plt.title(str(x) + ', ' + str(y))
plt.plot(wavelengths_nirspec_west, nirspec_data_new_west[:,y,x])
plt.plot(wavelengths_nirspec_west, gaussians_west[:,y,x])
plt.ylim(0, 10)
plt.xlim(3.1, 3.7)
plt.show()
