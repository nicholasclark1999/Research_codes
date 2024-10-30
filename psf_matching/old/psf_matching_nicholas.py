'''
Code to smooth a stitched NIRSpec cube to the resolution of MRS at a wavelength of choice

- Uses webbpsf to compute NIRSpec and MIRI monochromatic PSFs, fits them with 
  Gaussians. Calculates a kernel to go from a monochromatic NIRSpec PSF at wave_in 
  to the MIRI PSF at the specified reference wavelength (11.7 microns by default, since
  Amelie's stitched MRS cube uses CH2 Long as the reference resolution).
- For uncertainties, this script will generate a number of noise realizations (random numbers such that 
  their mean is zero and their standard deviation equals the instrumental uncertainty) at a given wavelength 
  based on the NIRSpec uncertainty array (2D) at that wavelength. Then it will smooth all of these realizations
  to the MIRI resolution. The uncertainty on the smoothed NIRSpec slice at this wavelength is given by the RMS
  of the noise realizations (i.e. calculate the RMS of the smoothed noise realizations pixel by pixel)

Author: Ryan Chown (rchown3@uwo.ca)
Date: 1 Aug 23
'''

'''
Modified and upkept by Nicholas Clark
'''

'''
TO DO LIST

fix 2d gaussian approximation (if possible)

make it spit out the fwhm of the psf
'''

'''
RUN THIS BEFORE RUNNING CODE (assuming you are on linux)
'''
#    webbpsf installation instructions
#    https://webbpsf.readthedocs.io/en/latest/installation.html
#    export WEBBPSF_PATH=~/webbpsf/webbpsf-data
#    then relaunch spyder in terminal, or else it wont reconise the environment variable. i could also add this to the bash startup script but meh




#%%
import convolver as cvr
import helpers as hlp
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
import numpy as np
from astropy.utils.data import get_pkg_data_filename


#%%

def data_loader(fname_stitched_cube):
    '''
    This function loads in JWST MIRI and NIRSPEC fits data cubes, and extracts wavelength 
    data from the header and builds the corresponding wavelength array. 
    
    Parameters
    ----------
    fname_stitched_cube
        TYPE: string
        DESCRIPTION: where the fits file is located.

    Returns 
    -------
    waves
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns.
    cube
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data.
            for [k,i,j] k is wavelength index, i and j are position index.
    cube_unc
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral error data.
                for [k,i,j] k is wavelength index, i and j are position index.
    pixsize_arcsec
        TYPE: float
        DESCRIPTION pixelsize of the data cube, in arcseconds.
    '''

    image_file = get_pkg_data_filename(fname_stitched_cube)
        
    #header data
    science_header = fits.getheader(image_file, 1)
        
    #wavelength data from header
    number_wavelengths = science_header["NAXIS3"]
    wavelength_increment = science_header["CDELT3"]
    wavelength_start = science_header["CRVAL3"]
        
    #constructing the ending point using given data
    #subtracting 1 so wavelength array is the right size.
    wavelength_end = wavelength_start + (number_wavelengths - 1)*wavelength_increment
    
    #making wavelength array, in micrometers
    waves = np.arange(wavelength_start, wavelength_end, wavelength_increment)
    
    # Load data cube to be psf matched, as well as corresponding error cube
    cube = fits.open(fname_stitched_cube)[1].data
    cube_unc = fits.open(fname_stitched_cube)[2].data
    
    #sometimes wavelength array is 1 element short, this will fix that
    
    #note to self: this was a problem in the early days of JWST, but it may have been fixed now and 
    #wavelength arrays can be made of a specific shape without needing to check just in case; i should
    #investigate this at some point
    
    if len(waves) != len(cube):
        wavelength_end = wavelength_start + number_wavelengths*wavelength_increment
        waves = np.arange(wavelength_start, wavelength_end, wavelength_increment)
    
    #setting nans to 0
    cube[np.isnan(cube)] = 0
    cube_unc[np.isnan(cube_unc)] = 0
    pixsize_arcsec = fits.open(fname_stitched_cube)[1].header['CDELT2'] * 3600
    
    return waves, cube, cube_unc, pixsize_arcsec



#%%



def make_noise_realizations(arr_unc, n_reals=100):
    '''
    This function generates noise from a normal distribution based off of the input error data.
    
    Parameters
    ----------
    arr_unc
        TYPE: 2d array of floats
        DESCRIPTION: position and spectral error data.
                for [i,j] i and j are position index.
    n_reals
        TYPE: positive integer
        DESCRIPTION: When performing rigorous error calculations, this is the number of  samples to 

    Returns 
    -------

    comb_cube
        TYPE: 3d array of floats
        DESCRIPTION: a cube of the same shape [i,j,n_reals] with random values drawn from a normal distribution

    '''
    SEED = 1234567
    rng = np.random.default_rng(SEED)
    nx, ny = arr_unc.shape
    mu = np.zeros((nx, ny))
    noise_cube = np.zeros((nx, ny, n_reals))
    for i in range(n_reals):
        noise_cube[:, :, i] = rng.normal(loc=mu, scale=arr_unc)
    return noise_cube



def smooth_slice(cube, cube_unc, waves, band_in, i, pixsize_arcsec, wave_out, band_out, jwst_inst_in, jwst_inst_out, n_reals, advanced_noise, regul, method, enable_printout, psf_out=None):
    '''
    This function performs PSF matching of a single wavelength slice. 
    The index and 3d cubes are input instead of inputting a 2d slice of a cube, to make the inputs of the function more readable.
    
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
    waves
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns.
    i
        TYPE: positive integer
        DESCRIPTION: the index of the specific wavelength slice beign calculated. 
    pixsize_arcsec
        TYPE: float
        DESCRIPTION pixelsize of the data cube, in arcseconds.
    wave_out
        TYPE: float
        DESCRIPTION desired wavelength to psf match to, in microns.
    jwst_inst_in
        TYPE: string
        DESCRIPTION whether or not the input array to be psf matched comes from 'nirspec' or 'miri'
    jwst_inst_out
        TYPE: string
        DESCRIPTION whether or not the output array that is getting psf matched comes from 'nirspec' or 'miri'
    n_reals
        TYPE: positive integer
        DESCRIPTION: When performing rigorous error calculations, this is the number of  samples to 
    advanced_noise
        TYPE: boolean
        DESCRIPTION: whether or not to perform the advanced error calculation. This is very lengthy to run (multiplies time by about 100) 
            and so an option is presented of just smoothing the error array. The advanced version creates a normal distribution and draws n_reals
            values from it, then smooths each of them and combines them. 
    regul
        TYPE: float
        DESCRIPTION: regularization paramater used in kernel generation
    method
        TYPE: string
        DESCRIPTION: method used for noise suppresion in kernel generation. can be 'A+11' or 'B+16'
    enable_printout
        TYPE: boolean
        DESCRIPTION: whether or not to print out diagnostic information while psf matching. 
            Useful for bugtesting, but spams the console with print statements.
    psf_out
        TYPE: psf object
        DESCRIPTION: The wrapper function calculates the output psf and inserts it in order to save time, as it is expected
            that this function will be ran repeatedly; the output function only needs to be calculated once for a given set of psf matching.

    Returns
    -------
    smoothed_slice
        TYPE: 2d array of floats
        DESCRIPTION: psf matched array corresponding to the slice of the input array specified by i.
            for [i,j] i and j are position index.
    smoothed_slice_unc
        TYPE: 2d array of floats
        DESCRIPTION: error array for smoothed_slice
                for [i,j] i and j are position index.
    '''

    
    wave_in = waves[i] * 1e-6 #converts from microns to meters
    
    #only calculate psf_out if it isnt input
    if psf_out==None:
        if jwst_inst_out == 'nirspec':
            psf_out = hlp.nirspec_monochrom_psf(wave_out*1e-6, fov_arcsec=10, norm='last')
            
        elif jwst_inst_out == 'miri':
            psf_out = hlp.miri_monochrom_psf([wave_out*1e-6], band_out, fov_arcsec=10, norm='last')
    
    
        '''
        test1 = psf_out[0].data
        ax = plt.figure(figsize=(9,9)).add_subplot(111)
        plt.title('diagnostic output psf')
        plt.imshow(test1)
        plt.colorbar()
        ax.invert_yaxis()
        plt.show()
        '''
            
        pix_arcsec_psf_out = psf_out[0].header['PIXELSCL'] # equals pixelscale / oversample
        psf_out = cvr.PSF(psf_out[0].data, pix_arcsec_psf_out, regul=regul, method=method)
        #gfit = psf_out.fit_gauss2d()
    
    unsmoothed_slice = cube[i]
    unsmoothed_slice_unc = cube_unc[i]

    #verifying that the pixel is nonzero and finite
    good_pix = (unsmoothed_slice != 0) & np.isfinite(unsmoothed_slice)
    n_good = np.sum(good_pix)
    if n_good > 0:
        # PSF of this slice
        if jwst_inst_in == 'nirspec':
            psf_in = hlp.nirspec_monochrom_psf([wave_in], fov_arcsec=10, norm='last')
            
        elif jwst_inst_in == 'miri':
            psf_in = hlp.miri_monochrom_psf([wave_in], band_in, fov_arcsec=10, norm='last')
            
            ax = plt.figure(figsize=(9,9)).add_subplot(111)
            plt.title('diagnostic input psf')
            plt.imshow(psf_out.arr)
            plt.colorbar()
            ax.invert_yaxis()
            plt.show()
            
            psf_out.compare_fwhm(psf_out)
            
            print(psf_out.fwhm_arcsec_x, psf_out.fwhm_arcsec_y)
            

            
            
        pix_arcsec_psf_in = psf_in[0].header['PIXELSCL'] # equals pixelscale / oversample
        psf_in = cvr.PSF(psf_in[0].data, pix_arcsec_psf_in, regul=regul, method=method)
        
        #gfit_in = psf_in.fit_gauss2d() # get gaussian beam parameters
        # Smooth to MRS res
        image_in = cvr.Image(unsmoothed_slice, pixsize_arcsec)
        # Provide Convolver with 2 PSFs
        c = cvr.Convolver(image_in, enable_printout=enable_printout, psf_in=psf_in, psf_out=psf_out)
        c.prepare_image()
        # Make a kernel
        c.create_kernel() # this will accept any optional arguments of helpers.create_kernel (e.g. low pass filter settings)
        c.prepare_kernel()
        c.do_the_convolution()
        smoothed_slice = c.image_out.arr
        
        #commented out the original for loop, replace with 1 iteration
        if advanced_noise == True:
            
            noise_realizations =  make_noise_realizations(unsmoothed_slice_unc,n_reals=n_reals)
            noise_realizations_smoothed = np.zeros(noise_realizations.shape)
            
            # Now smooth all of the noise realizations
            for i_noise in range(n_reals):
                nmap_i = cvr.Image(noise_realizations[:, :, i_noise], pixsize_arcsec)
                # Provide Convolver with 2 PSFs
                c = cvr.Convolver(nmap_i, enable_printout=enable_printout, psf_in=psf_in, psf_out=psf_out)
                c.prepare_image()
                # Make a kernel
                c.create_kernel() # this will accept any optional arguments of helpers.create_kernel (e.g. low pass filter settings)
                c.prepare_kernel()
                c.do_the_convolution()
                noise_realizations_smoothed[:, :, i_noise] = c.image_out.arr
                
                # Calculate the RMS of the smoothed noise realizations
                smoothed_slice_unc = np.sqrt(np.sum(noise_realizations_smoothed**2, axis=2) / n_reals)
        else: 
            
            noise_realizations = np.copy(unsmoothed_slice_unc)
            noise_realizations_smoothed = np.zeros(noise_realizations.shape)
            
            nmap_i = cvr.Image(noise_realizations, pixsize_arcsec)
                # Provide Convolver with 2 PSFs
            c = cvr.Convolver(nmap_i, enable_printout=enable_printout, psf_in=psf_in, psf_out=psf_out)
            c.prepare_image()
                # Make a kernel
            c.create_kernel() # this will accept any optional arguments of helpers.create_kernel (e.g. low pass filter settings)
            c.prepare_kernel()
            c.do_the_convolution()
    
            smoothed_slice_unc = c.image_out.arr

    else:
        print("Not enough pixels to smooth image")
        smoothed_slice = unsmoothed_slice
        smoothed_slice_unc = unsmoothed_slice_unc

    return smoothed_slice, smoothed_slice_unc



def smooth_slice_cube(cube, cube_unc, waves, band_in, pixsize_arcsec, wave_out, band_out, jwst_inst_in, jwst_inst_out, n_reals=100, advanced_noise=False, regul=0.1, method='A+11', enable_printout=False):
    '''
    This function is a wrapper for smooth_slice, meant to operate on entire data cubes. This will take a long time to run; 
    For example on Nicholas' work pc in simple error mode it takes 30-60 minutes for a single subchannel of miri data to be psf matched (input array approx 150 by 150)'
    
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
    waves
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns.
    pixsize_arcsec
        TYPE: float
        DESCRIPTION pixelsize of the data cube, in arcseconds.
    wave_out
        TYPE: float
        DESCRIPTION desired wavelength to psf match to, in microns
    jwst_inst_in
        TYPE: string
        DESCRIPTION whether or not the input array to be psf matched comes from 'nirspec' or 'miri'
    jwst_inst_out
        TYPE: string
        DESCRIPTION whether or not the output array that is getting psf matched comes from 'nirspec' or 'miri'
    n_reals
        TYPE: positive integer
        DESCRIPTION: When performing rigorous error calculations, this is the number of  samples to 
    advanced_noise
        TYPE: boolean
        DESCRIPTION: whether or not to perform the advanced error calculation. This is very lengthy to run (multiplies time by about 100) 
            and so an option is presented of just smoothing the error array. The advanced version creates a normal distribution and draws n_reals
            values from it, then smooths each of them and combines them. 
    regul
        TYPE: float
        DESCRIPTION: regularization paramater used in kernel generation
    method
        TYPE: string
        DESCRIPTION: method used for noise suppresion in kernel generation. can be 'A+11' or 'B+16'
    enable_printout
        TYPE: boolean
        DESCRIPTION: whether or not to print out diagnostic information while psf matching. 
            Useful for bugtesting, but spams the console with print statements.

    Returns
    -------
    cube_sm
        TYPE: 3d array of floats
        DESCRIPTION: psf matched data cube
            for [k,i,j] k is wavelength index, i and j are position index.
    cube_sm_unc
        TYPE: 3d array of floats
        DESCRIPTION: error cube for cube_sm
                for [k,i,j] k is wavelength index, i and j are position index.
    '''
    
    # This function will take a long time to run
    
    if jwst_inst_out == 'nirspec':
        psf_out = hlp.nirspec_monochrom_psf(wave_out*1e-6, fov_arcsec=10, norm='last')
        
    elif jwst_inst_out == 'miri':
        psf_out = hlp.miri_monochrom_psf(wave_out*1e-6, band_out, fov_arcsec=10, norm='last')
        
    pix_arcsec_psf_out = psf_out[0].header['PIXELSCL'] # equals pixelscale / oversample
    psf_out = cvr.PSF(psf_out[0].data, pix_arcsec_psf_out, regul=regul, method=method)
    #gfit = psf_out.fit_gauss2d()
    
    
    cube_sm = np.zeros(cube.shape)
    cube_sm_unc = np.zeros(cube.shape)
    n_waves = waves.size
    for i in range(n_waves):
        #printout, if enabled
        if enable_printout == True:
            print('beginning psf matching of index ' + str(i) + ' out of ' + str(n_waves - 1))
        
        cube_sm[i], cube_sm_unc[i] = smooth_slice(cube, 
                                                  cube_unc, 
                                                  waves, 
                                                  band_in,
                                                  i, 
                                                  pixsize_arcsec, 
                                                  wave_out, 
                                                  band_out,
                                                  jwst_inst_in, 
                                                  jwst_inst_out, 
                                                  n_reals, 
                                                  advanced_noise, 
                                                  regul,
                                                  method,
                                                  enable_printout,
                                                  psf_out=psf_out)
    return cube_sm, cube_sm_unc



#%%

#wave_out = 17.95 * 1e-6 #hard coded, should show up way later if this is how its gonna be

#new_cube, new_error = smooth_nirspec_stitched_and_unc(comb_cube, comb_cube_unc, waves, pixsize_arcsec, n_reals=100)
#%%
'''

    
#%%
'''
#%%
waves1, comb_cube1, comb_cube_unc1, pixsize_arcsec1 = data_loader('ngc6302_ch1-short_s3d.fits')
waves2, comb_cube2, comb_cube_unc2, pixsize_arcsec2 = data_loader('ngc6302_ch1-medium_s3d.fits')
waves3, comb_cube3, comb_cube_unc3, pixsize_arcsec3 = data_loader('ngc6302_ch1-long_s3d.fits')
waves4, comb_cube4, comb_cube_unc4, pixsize_arcsec4 = data_loader('ngc6302_ch2-short_s3d.fits')
waves5, comb_cube5, comb_cube_unc5, pixsize_arcsec5 = data_loader('ngc6302_ch2-medium_s3d.fits')
waves6, comb_cube6, comb_cube_unc6, pixsize_arcsec6 = data_loader('ngc6302_ch2-long_s3d.fits')
waves7, comb_cube7, comb_cube_unc7, pixsize_arcsec7 = data_loader('ngc6302_ch3-short_s3d.fits')
waves8, comb_cube8, comb_cube_unc8, pixsize_arcsec8 = data_loader('ngc6302_ch3-medium_s3d.fits')
waves9, comb_cube9, comb_cube_unc9, pixsize_arcsec9 = data_loader('ngc6302_ch3-long_s3d.fits')



#%%
'''
new_cube1, new_error1 = smooth_slice_cube(comb_cube1, comb_cube_unc1, waves1, '1A', pixsize_arcsec1, 17.95, '3C', 'miri', 'miri' )

with fits.open('ngc6302_ch1-short_s3d.fits') as hdul:
        
    hdul[1].data = new_cube1
    hdul[2].data = new_error1
   
    hdul.writeto('new_data/ngc6302_ch1-short_s3d.fits', overwrite=True)
    

    
new_cube2, new_error2 = smooth_slice_cube(comb_cube2, comb_cube_unc2, waves2, '1B', pixsize_arcsec2, 17.95, '3C', 'miri', 'miri' )

with fits.open('ngc6302_ch1-medium_s3d.fits') as hdul:
        
    hdul[1].data = new_cube2
    hdul[2].data = new_error2
   
    hdul.writeto('new_data/ngc6302_ch1-medium_s3d.fits', overwrite=True)



new_cube3, new_error3 = smooth_slice_cube(comb_cube3, comb_cube_unc3, waves3, '1C', pixsize_arcsec3, 17.95, '3C', 'miri', 'miri' )

with fits.open('ngc6302_ch1-long_s3d.fits') as hdul:
        
    hdul[1].data = new_cube3
    hdul[2].data = new_error3
   
    hdul.writeto('new_data/ngc6302_ch1-long_s3d.fits', overwrite=True)



new_cube4, new_error4 = smooth_slice_cube(comb_cube4, comb_cube_unc4, waves4, '2A', pixsize_arcsec4, 17.95, '3C', 'miri', 'miri' )

with fits.open('ngc6302_ch2-short_s3d.fits') as hdul:
        
    hdul[1].data = new_cube4
    hdul[2].data = new_error4
   
    hdul.writeto('new_data/ngc6302_ch2-short_s3d.fits', overwrite=True)

#%%

new_cube5, new_error5 = smooth_slice_cube(comb_cube5, comb_cube_unc5, waves5, '2B', pixsize_arcsec5, 17.95, '3C', 'miri', 'miri' )

with fits.open('ngc6302_ch2-medium_s3d.fits') as hdul:
        
    hdul[1].data = new_cube5
    hdul[2].data = new_error5
   
    hdul.writeto('new_data/ngc6302_ch2-medium_s3d.fits', overwrite=True)
    
    
new_cube6, new_error6 = smooth_slice_cube(comb_cube6, comb_cube_unc6, waves6, '2C', pixsize_arcsec6, 17.95, '3C', 'miri', 'miri' )

with fits.open('ngc6302_ch2-long_s3d.fits') as hdul:
        
    hdul[1].data = new_cube6
    hdul[2].data = new_error6
   
    hdul.writeto('new_data/ngc6302_ch2-long_s3d.fits', overwrite=True)

#%%

new_cube7, new_error7 = smooth_slice_cube(comb_cube7, comb_cube_unc7, waves7, '3A', pixsize_arcsec7, 17.95, '3C', 'miri', 'miri' )

with fits.open('ngc6302_ch3-short_s3d.fits') as hdul:
        
    hdul[1].data = new_cube7
    hdul[2].data = new_error7
   
    hdul.writeto('new_data/ngc6302_ch3-short_s3d.fits', overwrite=True)

#%%

new_cube8, new_error8 = smooth_slice_cube(comb_cube8, comb_cube_unc8, waves8, '3B', pixsize_arcsec8, 17.95, '3C', 'miri', 'miri' )

with fits.open('ngc6302_ch3-medium_s3d.fits') as hdul:
        
    hdul[1].data = new_cube8
    hdul[2].data = new_error8
   
    hdul.writeto('new_data/ngc6302_ch3-medium_s3d.fits', overwrite=True)

#%%

new_cube9, new_error9 = smooth_slice_cube(comb_cube9, comb_cube_unc9, waves9, '3C', pixsize_arcsec9, 17.95, '3C', 'miri', 'miri' )

with fits.open('ngc6302_ch3-long_s3d.fits') as hdul:
        
    hdul[1].data = new_cube9
    hdul[2].data = new_error9
   
    hdul.writeto('new_data/ngc6302_ch3-long_s3d.fits', overwrite=True)


'''




'''
smoothed_slice, smoothed_slice_unc = smooth_slice(comb_cube1, comb_cube_unc1, waves1, 314, pixsize_arcsec1, 17.95, 'miri', 'miri', 100, False)



ax = plt.figure(figsize=(9,9)).add_subplot(111)
plt.imshow(smoothed_slice, vmin=0, vmax=5000)
plt.colorbar()
ax.invert_yaxis()

plt.show()

ax = plt.figure(figsize=(9,9)).add_subplot(111)
plt.imshow(comb_cube1[314], vmin=0, vmax=5000)
plt.colorbar()
ax.invert_yaxis()

plt.show()
#%%
ax = plt.figure(figsize=(9,9)).add_subplot(111)
plt.imshow(smoothed_slice - comb_cube2[100])
plt.colorbar()
ax.invert_yaxis()

plt.show()

'''
'''
#%%
print('hello')



#%%
k = 8
for i in range(len(comb_cube[0,:,0])):
    for j in range(len(comb_cube[0,0,:])):
        pog = np.isinf(comb_cube[k,i,j])
        if pog == True:
            print(i,j)
            
#%%
i = 314
    # Look at it
pl.figure()
pl.subplot(131)
pl.imshow(comb_cube[i], vmin=0, vmax=1500)
pl.subplot(132)
pl.imshow(new_cube[i], vmin=0, vmax=1500)
pl.subplot(133)
pl.imshow((new_cube[i] - comb_cube[i]), vmin=0, vmax=200)


with fits.open('ngc6302_ch1-short_s3d.fits') as hdul:
        
    hdul[1].data = new_cube
    hdul[2].data = new_error
   
    hdul.writeto('pog.fits', overwrite=True)
    
    
#%%

with fits.open('new_data/ngc6302_ch2-short_s3d.fits') as hdul:
        
    pog1 = hdul[1].data
    pog2 = hdul[2].data
   
   # hdul.writeto('pog.fits', overwrite=True)
    
i = 50
j = 70
    
pl.figure()
pl.plot(waves4, comb_cube4[:,i,j])
pl.plot(waves4, new_cube4[:,i,j])
pl.plot(waves4, pog1[:,i,j], alpha=0.5)
pl.ylim((0,5000))
pl.show()
    
    '''
    
    
#%%
i = 43

new_cube6, new_error6 = smooth_slice(comb_cube6, comb_cube_unc6, waves6, '2C', i, pixsize_arcsec6, 17.95, '3C', 'miri', 'miri' , 100, False, 0.69, 'B+16', False)


#%%


    
    
    
    
