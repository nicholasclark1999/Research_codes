#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 13:14:16 2023

@author: nclark
"""



'''
IMPORTING MODULES
'''



#standard stuff
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import  AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter

#used for fits file handling
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

#needed for Ryan's reprojection function
from jwst import datamodels
from astropy import wcs
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_exact



####################################



'''
TABLE OF CONTENTS
'''



# reproject_slice:                 outputs reprojected slices (data OR error). 
#                                    fits files CAN have modified headers (from JWST default)

# reproject_cube:                  outputs reprojected cubes (data OR error). 
#                                    fits files CAN have modified headers

# loading_function_reproject_cube: outputs reprojected cubes (data AND error), and corresponding wavelength arrays. 
#                                    fits files CANNOT have modified headers

# fits_reprojection_cube:          saves reprojected cubes (data AND error), without creating output variables.
#                                    fits files CANNOT have modified headers



####################################



'''
HELPER FUNCTIONS
'''



#needed for Ryan's reproject function
def get_2d_wcs_from_data(fits_to_reproject, header_index): 
    '''
    Gets a wcs object from header data.
    Works with fits files containing cubes and slices.
    
    Note from Nicholas: This function seems to have been built to fix a problem with spitzer data. Since this fix
    doesnt do anything to JWST data I am leaving the fix in.
    
    Gets 2D (spatial) WCS from IRS cube.
    For some reason, extracting WCS from cubism cubes doesn't work well
    (the spectral axis messes things up).
​
    Parameters
    ----------
    fits_to_reproject
        TYPE: string
        DESCRIPTION: where the fits file to be reprojected is located.        
    header_index
        TYPE: positive integer
        DESCRIPTION: header that corresponds to image data, for JWST is 1 by default
            
    Returns
    -------
    w_in
        TYPE: WCS object
        DESCRIPTION: contains WCS info about RA and Dec to be used by other functions
    '''
    fits_in = fits.open(fits_to_reproject)
    
    #naxis=2, corresponds to RA and Dec
    w_in = wcs.WCS(fits_in[header_index].header, fobj=fits_in, naxis=2)
    
    # Turn WCS into a header, and then back to WCS again (this cleans up garbage to do with the 3rd axis we don't want anyway)
    w_in = wcs.WCS(w_in.to_header())
    
    return w_in



#needed for ryans reproject function
def clip_spikes_spec(spec, spec_unc):
    '''
    Tidies up data by removing anything that is not finite, and anything less than zero, which are assumed to be bad pixels.
​
    Parameters
    ----------
    spec
        TYPE: 1d array of floats
        DESCRIPTION: a spectra
    spec_unc
        TYPE: 1d array of floats
        DESCRIPTION: the error assiciated with spec

    Returns
    -------
    spec_out
        TYPE: 1d array of floats
        DESCRIPTION: a spectra, artifacts removed.
    spec_unc_out
        TYPE: 1d array of floats
        DESCRIPTION:  the error assiciated with spec, artifacts removed.
    '''
    
    #defining output varaibles
    spec_out = np.zeros(spec.shape)
    spec_unc_out = np.zeros(spec.shape)
    
    #defining variables to serve as exclusion criteria
    snr = np.log10(np.abs(spec / spec_unc))
    x = snr
    y = np.log10(np.abs(spec))
    
    #creating array which has bad pixels identified
    good = np.isfinite(x) & np.isfinite(y) & (spec > 0)
    
    #new array has values of old array for good pixels, 0 for bad pixels
    spec_out[good] = spec[good]
    spec_unc_out[good] = spec_unc[good]

    return spec_out, spec_unc_out



#needed for ryans reproject function
def clip_spikes_cube(cube, cube_unc):
    '''
    A wrapper function for clip_spikes_spec, meant to work on data cubes. 
    To work with a slice, change dimentions from (x,y) to (1,x,y)
    For example: data = data[np.newaxis, :, :]
​
    Parameters
    ----------
    cube
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [k,i,j] k is wavelength index, i and j are position index.
    cube_unc
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [k,i,j] k is wavelength index, i and j are position index.
            
    Returns
    -------
    cube_out
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, artifacts removed.
            for [k,i,j] k is wavelength index, i and j are position index.
    cube_unc_out
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data, artifacts removed.
                for [k,i,j] k is wavelength index, i and j are position index.
    '''
    
    #defining output variables
    cube_out = np.zeros(cube.shape)
    cube_unc_out = np.zeros(cube.shape)
    nw, nx, ny = cube.shape
    for ix in range(nx):
        for iy in range(ny):
            #calling clip_spikes_spec
            spec = cube[:, ix, iy]
            spec_unc = cube_unc[:, ix, iy]
            
            #saving output
            res = clip_spikes_spec(spec, spec_unc)
            cube_out[:, ix, iy] = res[0]
            cube_unc_out[:, ix, iy] = res[1]
            
    return cube_out, cube_unc_out



####################################



'''
REPROJECTION FUNCTIONS
'''



def reproject_slice(fits_to_reproject, fits_reference,
                     header_index, data_index,
                     progress_updates=True):
    '''
    This function loads in fits data slices (not necessarily JWST data products). It then reprojects the fits files
    to match the header of the fits file located at fits_reference, and returns the reprojected data OR error slice,
    but not both at the same time.
    
    If you do not have error or dq extentions, set them to None.
    
    Call this function if you are using modified data, and not JWST s3d data cubes whose header and overall shape is unmodified.
    
    Parameters
    ----------
    fits_to_reproject
        TYPE: string
        DESCRIPTION: where the fits file to be reprojected is located.        
    fits_reference
        TYPE: string
        DESCRIPTION: where the fits file with the header to be used as a reference is located
    header_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index to get wavelength data from in the header; usually 1. This header will also
            store info about WCS coordinates, and so it is used for 'fits reference', in the event that 'fits_to_reproject' 
            has its data in different extensions
    data_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index where the primary data in fits_to_reproject is stored (the default for JWST is 1)
    error_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index where the error data in fits_to_reproject is stored (the default for JWST is 2)
    dq_index
        TYPE: index (nonzero integer)
        DESCRIPTION: the index where the dq (data quality) data in fits_to_reproject is stored (the default for JWST is 3)
    uncertainty
        TYPE: boolean
        DESCRIPTION: whether or not to treat the data to be reprojected as an error array, is False by default.
    clip_spikes
        TYPE: boolean
        DESCRIPTION: whether or not to run the clip_spikes_cube function. Requires error data, is False by default.
    progress_updates
        TYPE: boolean
        DESCRIPTION: whether or not to include the various print statements built in to the function that indicate progress.
            Is useful for larger cubes and bug-testing, but would be annoying if the cubes are small and the function finishes quickly.
            Is True by default

    Returns
    -------
    cube
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, reprojected.
            for [i,j,k] k is wavelength index, i and j are position index. (check this, ryans code loads the data weird)
    '''
    
    #loading in data
    input_data = []
    
    data = fits.open(fits_reference)[header_index].data
    hdr = fits.open(fits_reference)[header_index].header
    w = wcs.WCS(hdr)
    input_data += [(data, w)]
    
    #getting wcs coords
    wcs_out, shape_out = find_optimal_celestial_wcs(input_data, auto_rotate=True)


    #loading in data
    data = fits.open(fits_to_reproject)[data_index].data
    

        
    wcs_cube = get_2d_wcs_from_data(fits_to_reproject, header_index)




        
    if progress_updates == True:
        print("Reprojection")
        
    #note reproject_exact takes a tuple of the data and an astropy wcs object
    reprojected_slice, _ = reproject_exact((data, wcs_cube),
                             wcs_out,
                             shape_out,
                             parallel=False)
        
    if progress_updates == True:
        print("Done")
            
    return reprojected_slice



fits_to_reproject = 'data/cams/miri_color_F1000W_F1130W.fits'
fits_reference = 'data/spitzer/ngc6720_112PAH_intensity_map_original.fits'

reprojected_slice = reproject_slice(fits_to_reproject, fits_reference, 0, 0)

#%%


miricam_image_file = get_pkg_data_filename(fits_reference)
miricam_data = fits.getdata(miricam_image_file, ext=0)
miricam_header = fits.getheader(miricam_image_file, ext=0)



#collapsing

spitzer_data = np.mean(miricam_data, axis=0)



miri_data = np.mean(reprojected_slice, axis=0)



miri_data = np.flip(miri_data)


fits_to_reproject = 'data/cams/nircam_color_F300M_F335M.fits'
fits_reference = 'data/spitzer/ngc6720_112PAH_intensity_map_original.fits'

reprojected_slice = reproject_slice(fits_to_reproject, fits_reference, 0, 0)


nir_data = np.mean(reprojected_slice, axis=0)



nir_data = np.flip(nir_data)
#%%

x_array = np.arange(0, 47*0.000513888895512, 0.000513888895512)*3600

ax = plt.figure('RNF_paper_ring_confirmation', figsize=(18,9)).add_subplot(111)

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

plt.ylabel('Band Strength', fontsize=32, labelpad=90)
plt.xlabel('Distance along slit (arcsec)', fontsize=32, labelpad=60)



ax = plt.figure('RNF_paper_ring_confirmation', figsize=(18,9)).add_subplot(111)

plt.plot(x_array, spitzer_data[40:]/spitzer_data[62], color='black')
plt.plot(x_array, miri_data[40:]/miri_data[61], color='#dc267f')
plt.plot(x_array, nir_data[40:]/nir_data[60], color='#648fff')

plt.plot([x_array[24], x_array[24]], [-2, 2], linestyle='dashed', color='black')
plt.plot([x_array[26], x_array[26]], [-2, 2], linestyle='dashed', color='black')

plt.arrow(x_array[35], 1, 0.002*3600, 0, width=0.02, head_length=0.001*3600, color='black')

props = dict(boxstyle='round', facecolor='none')
ax.text(0.67, 0.8, 'Increasing ' +  r'Right Ascension ($\alpha$)', transform=ax.transAxes, fontsize=20,
        verticalalignment='top', bbox=props)



ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.tick_params(axis='x', which='major', labelbottom=True, top=True, length=10, width=4)
ax.tick_params(axis='x', which='minor', labelbottom=False, top=True, length=5, width=2)
ax.tick_params(axis='y', which='major', labelleft='on', right=True, length=10, width=4)
ax.tick_params(axis='y', which='minor', labelleft='on', right=True, length=5, width=2)
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())

#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)

plt.ylim((-0.2, 1.2))
plt.xlim((0, 46*0.000513888895512*3600))

plt.savefig('Figures/RNF_paper_ring_confirmation.pdf', bbox_inches='tight')
plt.show() 

print('spit, mir, nir: ', spitzer_data[62], miri_data[61], nir_data[60])





spitzer_at_north = (spitzer_data[64]+spitzer_data[65])/2