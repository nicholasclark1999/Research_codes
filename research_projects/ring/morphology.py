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

nirspec_data_new[np.isnan(nirspec_data_new)] = -10

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

amplitudes_pah = np.copy(amplitudes)


#%%

def gaussian_plus_line(x, a, b):
    mean = 3.28380
    std = 2*0.0018/(2*(2*np.log(2))**0.5)
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
        data = nirspec_data_new[239:247,y,x]
        wave = wavelengths_nirspec[239:247]
        result = lmfit.minimize(myfunc, fit_params, args=(wave, data))
        is_fit_good = result.params
        amp = is_fit_good['amp'].value
        constant = is_fit_good['constant'].value
        
        model_best2 = gaussian_plus_line(wavelengths_nirspec, amp, constant)
        
        gaussians[:,y,x] = model_best2
        amplitudes[y,x] = amp
        constants[y,x] = constant

amplitudes_civ = np.copy(amplitudes)


gaussian_civ = np.copy(gaussians)

#%%


def gaussian_plus_line(x, a, b):
    mean = 3.23519
    std = 2*0.0018/(2*(2*np.log(2))**0.5)
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
        data = nirspec_data_new[207:222,y,x]
        wave = wavelengths_nirspec[207:222]
        result = lmfit.minimize(myfunc, fit_params, args=(wave, data))
        is_fit_good = result.params
        amp = is_fit_good['amp'].value
        constant = is_fit_good['constant'].value
        
        model_best2 = gaussian_plus_line(wavelengths_nirspec, amp, constant)
        
        gaussians[:,y,x] = model_best2
        amplitudes[y,x] = amp
        constants[y,x] = constant

amplitudes_h2 = np.copy(amplitudes)

constants_h2 = np.copy(constants)

gaussian_h2 = np.copy(gaussians)

#%%


# 'data/north/jw01558-o056_t005_nirspec_g140m-f100lp_s3d_masked_aligned.fits'



wavelengths_nirspec_c, nirspec_data_c, nirspec_error_data_c = rnf.loading_function(
    'data/north/jw01558-o056_t005_nirspec_g140m-f100lp_s3d_masked_aligned.fits', 1)

nirspec_data_new_c, _ = rnf.regrid(nirspec_data_c, nirspec_error_data_c, 2)

nirspec_data_new_c[np.isnan(nirspec_data_new_c)] = 0

def gaussian_plus_line(x, a, b):
    mean = 0.9854
    std = 2*0.0006/(2*(2*np.log(2))**0.5)
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


gaussians = np.copy(nirspec_data_new_c)

amplitudes = np.copy(nirspec_data_new_c[0])
constants = np.copy(nirspec_data_new_c[0])





for y in range(len(nirspec_data_new_c[0,:,0])):
    for x in range(len(nirspec_data_new_c[0,0,:])):
        data = nirspec_data_new_c[34:50,y,x]
        wave = wavelengths_nirspec_c[34:50]
        result = lmfit.minimize(myfunc, fit_params, args=(wave, data))
        is_fit_good = result.params
        amp = is_fit_good['amp'].value
        constant = is_fit_good['constant'].value
        
        model_best2 = gaussian_plus_line(wavelengths_nirspec_c, amp, constant)
        
        gaussians[:,y,x] = model_best2
        amplitudes[y,x] = amp
        constants[y,x] = constant

amplitudes_ci = np.copy(amplitudes)
gaussians_ci = np.copy(gaussians)

#%%

x = 10
y = 10


plt.figure()
plt.title(str(x) + ', ' + str(y))
plt.plot(wavelengths_nirspec_c, nirspec_data_new_c[:,y,x])
plt.plot(wavelengths_nirspec_c, gaussians_ci[:,y,x])
#plt.ylim(0, 10)
plt.xlim(0.98, 1.0)
plt.show()


#%%

# F212N (H2)
with fits.open('data/cams/jw01558005001_04101_00001_nrcb1_combined_i2d.fits') as hdul:
    miri_f1500w_data = hdul[1].data
    pog = hdul[0].header



with fits.open('data/north/jw01558-o056_t005_nirspec_g140m-f100lp_s3d_masked_aligned.fits') as hdul:
    miri_f1500w_data = hdul[1].data
    pog = hdul[1].header














#%%


#ax = plt.figure('RNF_paper_morphology', figsize=(18,9)).add_subplot(111)

plt.rcParams.update({'font.size': 14})

#ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
#ax.tick_params(axis='y', which='major', labelleft=False, right=False)

#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
#ax.set_xticks([])
#ax.set_yticks([])

#plt.ylabel('Flux (MJy/sr)', fontsize=16, labelpad=60)
#plt.xlabel('Wavelength (micron)', fontsize=16, labelpad=30)

'''
PAH
'''

ax = plt.figure('RNF_paper_morphology', figsize=(18,6)).add_subplot(131)

plt.imshow(amplitudes_pah, cmap='gnuplot', vmin=0)
ax.invert_yaxis()
plt.colorbar()
plt.scatter([8.5], [12.5], s=900, facecolors='none', edgecolors='#39ff14')


ax.set_xticks([])
ax.set_yticks([])


props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'PAH', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


'''
H2
'''

ax = plt.figure('RNF_paper_morphology', figsize=(18,6)).add_subplot(132)

plt.imshow(amplitudes_h2, cmap='gnuplot', vmin=0)
ax.invert_yaxis()
plt.colorbar()
plt.scatter([8.5], [12.5], s=900, facecolors='none', edgecolors='#39ff14')


ax.set_xticks([])
ax.set_yticks([])


props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'H$_2$ 1-0 S(1)', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.arrow(3, 4, 0, -2, width=0.5, head_length=1, color='white')

'''
C I
'''

ax = plt.figure('RNF_paper_morphology', figsize=(18,6)).add_subplot(133)

plt.imshow(amplitudes_ci, cmap='gnuplot', vmin=0)
ax.invert_yaxis()
plt.colorbar()
plt.scatter([8.5], [12.5], s=900, facecolors='none', edgecolors='#39ff14')


ax.set_xticks([])
ax.set_yticks([])


props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'C I', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/RNF_paper_morphology.pdf', bbox_inches='tight')
plt.show()




#%%


wavelengths_nirspec, nirspec_data, nirspec_error_data = rnf.loading_function(
    'data/north/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 1)

nirspec_data_new = np.copy(nirspec_data)

nirspec_data_new[np.isnan(nirspec_data_new)] = -10

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

amplitudes_pah_small = np.copy(amplitudes)




#%%


def gaussian_plus_line(x, a, b):
    mean = 3.23519
    std = 2*0.0018/(2*(2*np.log(2))**0.5)
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
        data = nirspec_data_new[207:222,y,x]
        wave = wavelengths_nirspec[207:222]
        result = lmfit.minimize(myfunc, fit_params, args=(wave, data))
        is_fit_good = result.params
        amp = is_fit_good['amp'].value
        constant = is_fit_good['constant'].value
        
        model_best2 = gaussian_plus_line(wavelengths_nirspec, amp, constant)
        
        gaussians[:,y,x] = model_best2
        amplitudes[y,x] = amp
        constants[y,x] = constant

amplitudes_h2_small = np.copy(amplitudes)



#%%


# 'data/north/jw01558-o056_t005_nirspec_g140m-f100lp_s3d_masked_aligned.fits'



wavelengths_nirspec_c, nirspec_data_c, nirspec_error_data_c = rnf.loading_function(
    'data/north/jw01558-o056_t005_nirspec_g140m-f100lp_s3d_masked_aligned.fits', 1)

nirspec_data_new_c = np.copy(nirspec_data_c)

nirspec_data_new_c[np.isnan(nirspec_data_new_c)] = 0

def gaussian_plus_line(x, a, b):
    mean = 0.9854
    std = 2*0.0006/(2*(2*np.log(2))**0.5)
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


gaussians = np.copy(nirspec_data_new_c)

amplitudes = np.copy(nirspec_data_new_c[0])
constants = np.copy(nirspec_data_new_c[0])





for y in range(len(nirspec_data_new_c[0,:,0])):
    for x in range(len(nirspec_data_new_c[0,0,:])):
        data = nirspec_data_new_c[:290,y,x]
        wave = wavelengths_nirspec_c[:290]
        result = lmfit.minimize(myfunc, fit_params, args=(wave, data))
        is_fit_good = result.params
        amp = is_fit_good['amp'].value
        constant = is_fit_good['constant'].value
        
        model_best2 = gaussian_plus_line(wavelengths_nirspec_c, amp, constant)
        
        gaussians[:,y,x] = model_best2
        amplitudes[y,x] = amp
        constants[y,x] = constant

amplitudes_ci_small = np.copy(amplitudes)



#%%

# F212N (H2)
with fits.open('data/cams/jw01558005001_04101_00001_nrcb1_combined_i2d.fits') as hdul:
    miri_f1500w_data = hdul[1].data
    pog = hdul[0].header



with fits.open('data/north/jw01558-o056_t005_nirspec_g140m-f100lp_s3d_masked_aligned.fits') as hdul:
    miri_f1500w_data = hdul[1].data
    pog = hdul[1].header














#%%


#ax = plt.figure('RNF_paper_morphology', figsize=(18,9)).add_subplot(111)

plt.rcParams.update({'font.size': 14})

#ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
#ax.tick_params(axis='y', which='major', labelleft=False, right=False)

#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
#ax.set_xticks([])
#ax.set_yticks([])

#plt.ylabel('Flux (MJy/sr)', fontsize=16, labelpad=60)
#plt.xlabel('Wavelength (micron)', fontsize=16, labelpad=30)

'''
PAH
'''

ax = plt.figure('RNF_paper_morphology_small', figsize=(18,6)).add_subplot(131)

plt.imshow(amplitudes_pah_small, cmap='gnuplot', vmin=0)
ax.invert_yaxis()
plt.colorbar()
plt.scatter([18], [24], s=500, facecolors='none', edgecolors='#39ff14')


ax.set_xticks([])
ax.set_yticks([])


props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'PAH', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


'''
H2
'''

ax = plt.figure('RNF_paper_morphology_small', figsize=(18,6)).add_subplot(132)
plt.imshow(amplitudes_h2_small, cmap='gnuplot', vmin=0)
ax.invert_yaxis()
plt.colorbar()
plt.scatter([18], [24], s=500, facecolors='none', edgecolors='#39ff14')


ax.set_xticks([])
ax.set_yticks([])


props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'H$_2$ 1-0 S(1)', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.arrow(6, 8, 0, -4, width=1, head_length=2, color='white')

'''
C I
'''

ax = plt.figure('RNF_paper_morphology_small', figsize=(18,6)).add_subplot(133)

plt.imshow(amplitudes_ci_small, cmap='gnuplot', vmin=0)
ax.invert_yaxis()
plt.colorbar()
plt.scatter([18], [24], s=500, facecolors='none', edgecolors='#39ff14')


ax.set_xticks([])
ax.set_yticks([])


props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'C I', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/RNF_paper_morphology_small.pdf', bbox_inches='tight')
plt.show()



#%%

'''
VERSION FOR POSTER
'''


#ax = plt.figure('RNF_paper_morphology', figsize=(18,9)).add_subplot(111)

plt.rcParams.update({'font.size': 14})

#ax.tick_params(axis='x', which='major', labelbottom=False, top=False)
#ax.tick_params(axis='y', which='major', labelleft=False, right=False)

#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
#ax.spines['left'].set_visible(False)

# Hide X and Y axes tick marks
#ax.set_xticks([])
#ax.set_yticks([])

#plt.ylabel('Flux (MJy/sr)', fontsize=16, labelpad=60)
#plt.xlabel('Wavelength (micron)', fontsize=16, labelpad=30)

'''
PAH 
'''

ax = plt.figure('RNF_paper_morphology_small', figsize=(18,6)).add_subplot(131)

plt.axis('off')

plt.imshow(amplitudes_pah_small, cmap='gnuplot', vmin=0)
ax.invert_yaxis()
plt.scatter([18], [24], s=500, facecolors='none', edgecolors='#39ff14')


ax.set_xticks([])
ax.set_yticks([])


props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'PAH', transform=ax.transAxes, fontsize=36,
        verticalalignment='top', bbox=props)


'''
H2
'''

ax = plt.figure('RNF_paper_morphology_small', figsize=(18,6)).add_subplot(132)
plt.imshow(amplitudes_h2_small, cmap='gnuplot', vmin=0)
ax.invert_yaxis()
plt.scatter([18], [24], s=500, facecolors='none', edgecolors='#39ff14')


ax.set_xticks([])
ax.set_yticks([])


props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'H$_2$ 1-0 S(1)', transform=ax.transAxes, fontsize=36,
        verticalalignment='top', bbox=props)

plt.arrow(5, 14, 0, -4, width=1, head_length=2, color='white')

ax.text(0.05, 0.13, 'Towards center', transform=ax.transAxes, fontsize=28,
        verticalalignment='top', bbox=props)

'''
C I
'''

ax = plt.figure('RNF_paper_morphology_small', figsize=(18,6)).add_subplot(133)

plt.imshow(amplitudes_ci_small, cmap='gnuplot', vmin=0)
ax.invert_yaxis()
plt.scatter([18], [24], s=500, facecolors='none', edgecolors='#39ff14')


ax.set_xticks([])
ax.set_yticks([])

props = dict(boxstyle='round', facecolor='white')
ax.text(0.05, 0.95, 'C I', transform=ax.transAxes, fontsize=36,
        verticalalignment='top', bbox=props)

plt.savefig('Figures/RNF_paper_morphology_small_poster.png', bbox_inches='tight', transparent=True, dpi=1000)
plt.show()



















#%%
'''
ax = plt.figure().add_subplot(1,1,1)
plt.title('fit amplitudes')
plt.imshow(amplitudes_ci, cmap='gnuplot', vmin=0)
ax.invert_yaxis()
plt.colorbar()
plt.scatter([8.5], [12.5], s=900, facecolors='none', edgecolors='#39ff14')
plt.show()

#%%

ax = plt.figure().add_subplot(1,1,1)
plt.title('fit amplitudes')
plt.imshow(nirspec_data[253], cmap='gnuplot', vmin=0)
ax.invert_yaxis()
plt.colorbar()
plt.scatter([8.5], [12.5], s=900, facecolors='none', edgecolors='#39ff14')
plt.show()

#%%

ax = plt.figure().add_subplot(1,1,1)
plt.title('fit amplitudes')
plt.imshow(amplitudes_pah, cmap='gnuplot', vmin=0)
ax.invert_yaxis()
plt.colorbar()
plt.scatter([8.5], [12.5], s=900, facecolors='none', edgecolors='#39ff14')
plt.show()

#%%

ax = plt.figure().add_subplot(1,1,1)
plt.title('fit constants')
plt.imshow(constants)
ax.invert_yaxis()
plt.colorbar()
plt.show()
'''
'''
plt.figure()
plt.title('gaussian integrals')
plt.imshow(amplitudes)
plt.show()
'''


#%%

#now doing west region
'''
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
'''