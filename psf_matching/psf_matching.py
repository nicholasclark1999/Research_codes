'''
RUN THIS BEFORE RUNNING CODE (assuming you are on linux)
'''

#    webbpsf installation instructions
#    https://webbpsf.readthedocs.io/en/latest/installation.html
#    export WEBBPSF_PATH=~/webbpsf/webbpsf-data
#    then relaunch spyder in terminal, or else it wont reconise the environment variable.

# github for webbpsf: https://github.com/spacetelescope/webbpsf/blob/develop/webbpsf/webbpsf_core.py#L3399


'''
IMPORTING MODULES
'''

from astropy.io import fits
import numpy as np
import os
import webbpsf

from scipy.ndimage import rotate
from photutils.psf.matching import TopHatWindow
from photutils.psf.matching import create_matching_kernel
from astropy.convolution import convolve_fft


import time
import matplotlib.pyplot as plt

import new_pypher as pp



'''
THE CODE
'''



class DataCube:
    """
    NIRSpec IFU cube, or MIRI MRS cube
    should be a single Disperser-filter combination for NIRSPec, or a single 
        subchannel for MIRI MRS.

    Attributes:
        fits_file
            TYPE: string
            DESCRIPTION: LOCAL file location of fits file containing JWST cube 
                to be psf-matched. This will be used to create a new fits file
                of identical format that has been psf matched, at the file location
                'new_data/fits_file'
        fits_file_data 
            TYPE: HDUList object
            DESCRIPTION: loaded fits data of object to be psf matched, for convinient
                saving of data after psf matching.
        data
            TYPE: 3d array of floats
            DESCRIPTION: data cube to be psf-matched, corresponding to the data 
                in fits_file_data
        psf
            TYPE: 3d array of floats
            DESCRIPTION: PSFs of each wavelength in the data cube
            
        kernel
            TYPE: 3d array of floats
            DESCRIPTION: kernel made with 2 PSF's
    """
    
    def __init__(self, 
                 fits_file, 
                 wavelengths,
                 data, 
                 header, 
                 pixelsize, 
                 data_rotation,
                 instrument,
                 subchannel, 
                 channel_rotation):

        self.fits_file = fits_file
        self.wavelengths = wavelengths
        self.data = data
        self.header = header
        self.pixelsize = pixelsize
        self.data_rotation = data_rotation
        self.instrument = instrument
        self.subchannel = subchannel
        self.channel_rotation = channel_rotation

        #self.pixsize_arcsec = pixsize_arcsec
        #self.prepared = prepared
        #self.padding = padding
    
        
    
    #loading in the data from a fits file
    @staticmethod
    def load_fits(fits_file):
        '''
        used to initialize an instance of the DataCube class using fits_file
        '''
        
        with fits.open(fits_file) as hdul:
            
            data = hdul[1].data
            header = hdul[1].header
            
            data_rotation = hdul[1].header['PA_APER'] # rotation of image w.r.t. JWST V3 axis
            
            pixelsize = header['CDELT1'] # units of degrees
            pixelsize *= 3600 # convert to arcseconds
            
            number_wavelengths = header["NAXIS3"]
            wavelength_increment = header["CDELT3"]
            wavelength_start = header["CRVAL3"] # units of microns

                
        # building wavelength array
        # final wavelength, subtracting 1 so wavelength array is the right size.
        wavelength_end = wavelength_start + (number_wavelengths - 1)*wavelength_increment
        wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment)
        
        #sometimes wavelength array is 1 element short, this will fix that
        if len(wavelengths) != len(data[:,0,0]):
            wavelength_end = wavelength_start + number_wavelengths*wavelength_increment
            wavelengths = np.arange(wavelength_start, wavelength_end, wavelength_increment)
        
        # removing rounding error from arange
        wavelengths = np.round(wavelengths, 4)
        
        # using starting_wavelength to determine instrument:
        if wavelength_start < 4.0:
            instrument = 'nirspec'
            subchannel = 'nirspec'
        else:
            instrument = 'miri'
        
            # for miri, use starting_wavelength to determine subchannel and channel_rotation
            if wavelength_start < 5.5:
                subchannel = '1A'
                channel_rotation = 8.4
            elif wavelength_start < 6.0:
                subchannel = '1B'
                channel_rotation = 8.4
            elif wavelength_start < 7.0:
                subchannel = '1C'
                channel_rotation = 8.4
            elif wavelength_start < 8.0:
                subchannel = '2A'
                channel_rotation = 8.2
            elif wavelength_start < 9.0:
                subchannel = '2B'
                channel_rotation = 8.2
            elif wavelength_start < 11.0:
                subchannel = '2C'
                channel_rotation = 8.2
            elif wavelength_start < 12.0:
                subchannel = '3A'
                channel_rotation = 7.5
            elif wavelength_start < 14.0:
                subchannel = '3B'
                channel_rotation = 7.5
            elif wavelength_start < 16.0:
                subchannel = '3C'
                channel_rotation = 7.5
            elif wavelength_start < 19.0:
                subchannel = '4A'
                channel_rotation = 8.3
            elif wavelength_start < 22.0:
                subchannel = '4B'
                channel_rotation = 8.3
            else:
                subchannel = '4C'
                channel_rotation = 8.3
            
        return DataCube(fits_file, wavelengths, data, header, pixelsize, data_rotation, 
                        instrument, subchannel, channel_rotation)



    # building psf array
    def psf(self, **kwargs):
        
        if self.instrument == 'miri':
            PsfSetup = webbpsf.MIRI()
        
        PsfSetup.options['parity'] = 'odd' # ensures PSF will have an odd number of pixels on each side, with the centre of the PSF in the middle of a pixel
        PsfSetup.mode = 'IFU' # PSF for data cube, not imager
        PsfSetup.band = self.subchannel # specific subchannel to use in the calculation, e.g. 2A or 3C
        
        # might need to use a non-default pixelsize, specified by 'pixelsize' kwarg:
        pixelsize = kwargs.get('pixelsize')
        if pixelsize is not None:
            # subchannel will be a float, containing pixelsize of PSF.

            PsfSetup.pixelscale = pixelsize

        else:
            pix = self.pixelsize
            PsfSetup.pixelscale = pix
            
        # if a single wavelength is specified, use this instead of the entire wavelength array
        output_psf_wavelength = kwargs.get('output_psf_wavelength')
        if output_psf_wavelength is not None:
            wavelengths = np.array([output_psf_wavelength]) # needs to be in array format, not a float
        else:
            wavelengths = self.wavelengths
        
        self.psf_fits = PsfSetup.calc_datacube_fast(
            wavelengths*1e-6, # webbpsf takes wavelength units in m, not microns
            oversample=1, # default oversampling is 4, made it 5 so that parity is actually odd # was 5
            fov_arcsec = 3) # 10 arcseconds is sufficiently wide for any channel # was 20
        
    
    
    # rotating the PSF array to match the image rotation, before convolution
    # note that this will be the rotation of the input data, if it is different from the output data
    
    # output psf will need to have the same rotation as the input psf
    # then, both need to be rotated to match input data, or be 180deg of input data, 
    # due to a symmetry of the psf
    
    # XXX data does not seem to have channel rotation included in data_rotation
    
    # the output PSF should have the same subchannel rotation as the input data
    def _psf_rotation(self, OutputDataCube):

        
        # rotation to apply will be the rotation of the image w.r.t. JWST V3 axis, 
        # which includes mrs rotation, so this must be removed from the PSF.
        
        # total_input_rotation = self.data_rotation - self.channel_rotation
        # total_output_rotation = self.data_rotation - OutputDataCube.channel_rotation
        
        total_input_rotation = -1*self.data_rotation  - 2*self.channel_rotation
        total_output_rotation = -1*self.data_rotation - OutputDataCube.channel_rotation - self.channel_rotation
        
        print(total_input_rotation, total_output_rotation)
                 
        # rotating PSF, will be the same shape after rotation. Since the PSF is centred
        # on a pixel, the rotation occurs about the center of the PSF
        
        # for i in range(self.psf_fits[0].data.shape[0])
        
        self.psf_fits[0].data = rotate(
            self.psf_fits[0].data, total_input_rotation, axes=(1,2), reshape=False)
        
        OutputDataCube.psf_fits[0].data = rotate(
            OutputDataCube.psf_fits[0].data, total_output_rotation, axes=(1,2), reshape=False)

    
    
    # makes the kernel between two PSFs
    def _kernel_calculator(self, OutputDataCube):

        # both PSF will have the pixelsize of self, and both will have the size of self
    
        # TODO (): add different regul, method support 
    
            # from original kernel function:
            # regul : float : regularisation parameter (default is 0.11). 1/SNR for B+16.
            # method : str : 'A+11' (Aniano+2011, default) or 'B+16' (Boucaud+2016) method
    
        # at the moment, hard code regul to be 0.11 and use the A+11 method
        
        # the input and output PSF
        input_psf = self.psf_fits[0].data
        output_psf = OutputDataCube.psf_fits[0].data[0] # output_psf contains only a single psf
        
        print(input_psf.shape)
        
        # create_matching_kernel requires 2d inputs. So, must be done one slice at a time.
        kernel = np.zeros(input_psf.shape)
        kernel_fourier = np.zeros(input_psf.shape)
        for i in range(input_psf.shape[0]):
            kernel[i], kernel_fourier[i] = pp.homogenization_kernel(output_psf, input_psf[i], reg_fact=1e-5)
        return kernel
    
    
    # performs convolution between self and a PSF of specified wavelength
    def convolve(self, output_fits_file, output_psf_wavelength, output_fits_file_save_loc):
        
        # sanity check variable for pixelsize
        pix = self.pixelsize
        
        # define new class instance for output image and PSF
        OutputDataCube = DataCube.load_fits(output_fits_file)

        # make PSF for specified wavelength, using pixelsize of input data
        OutputDataCube.psf(pixelsize=pix, output_psf_wavelength=output_psf_wavelength)
        print(OutputDataCube.psf_fits[0].data.shape)
        # verify PSFs have the same rotation as input data
        self._psf_rotation(OutputDataCube)
        print(OutputDataCube.psf_fits[0].data.shape)
        # calculate kernel of PSFs
        kernel = self._kernel_calculator(OutputDataCube)
        
        # perform the convolution
        # note that if a 3d array is given, this function assumes it is 1 3 dimensional
        # kernel, not a series of 2d kernels. So, this must be done one slice at a time.
        convolution = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            convolution[i] = convolve_fft(self.data[i], kernel[i], preserve_nan=True)
        
        # replace data with convolution in output, and save new fits file
        OutputDataCube.data = convolution 
        
        with fits.open(OutputDataCube.fits_file) as hdul:
            hdul[1].data = convolution
            
            #saves to specified local file location
            hdul.writeto(output_fits_file_save_loc, overwrite=True)
            
        return OutputDataCube
    
#%%

time_start = time.time() # time at start

# file names
file_loc_ch1a = 'ngc6302_ch1-short_s3d.fits'
file_loc_ch1b = 'ngc6302_ch1-medium_s3d.fits'
file_loc_ch1c = 'ngc6302_ch1-long_s3d.fits'
file_loc_ch2a = 'ngc6302_ch2-short_s3d.fits'
file_loc_ch2b = 'ngc6302_ch2-medium_s3d.fits'
file_loc_ch2c = 'ngc6302_ch2-long_s3d.fits'
file_loc_ch3a = 'ngc6302_ch3-short_s3d.fits'
file_loc_ch3b = 'ngc6302_ch3-medium_s3d.fits'
file_loc_ch3c = 'ngc6302_ch3-long_s3d.fits'

# building unconvolved cubes
DataCube_1a = DataCube.load_fits(file_loc_ch1a)
DataCube_1b = DataCube.load_fits(file_loc_ch1b)
DataCube_1c = DataCube.load_fits(file_loc_ch1c)
DataCube_2a = DataCube.load_fits(file_loc_ch2a)
DataCube_2b = DataCube.load_fits(file_loc_ch2b)
DataCube_2c = DataCube.load_fits(file_loc_ch2c)
DataCube_3a = DataCube.load_fits(file_loc_ch3a)
DataCube_3b = DataCube.load_fits(file_loc_ch3b)
DataCube_3c = DataCube.load_fits(file_loc_ch3c)

# generating PSFs
time_pre_psf = time.time() # pre PSF step time

DataCube_1a.psf()
sanity_time = time.time()
print(sanity_time - time_pre_psf)
print(DataCube_1a.psf_fits[0].data.shape)

DataCube_1b.psf()
DataCube_1c.psf()
DataCube_2a.psf()
#%%
DataCube_2b.psf()
#%%
DataCube_2c.psf()
DataCube_3a.psf()
DataCube_3b.psf()
DataCube_3c.psf()

# performing convolution
time_pre_convolve = time.time() # pre convolve step time
#%%
# interested in convolving up to beginning of ch4, so go to 18 microns since ch3a ends at 17.95 microns
DataCube_1a_Convolved = DataCube_1a.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch1a)

#%%
DataCube_1b_Convolved = DataCube_1b.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch1b)
DataCube_1c_Convolved = DataCube_1c.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch1c)
DataCube_2a_Convolved = DataCube_2a.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch2a)
#%%
DataCube_2b_Convolved = DataCube_2b.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch2b)
#%%
DataCube_2c_Convolved = DataCube_2c.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch2c)
DataCube_3a_Convolved = DataCube_3a.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch3a)
DataCube_3b_Convolved = DataCube_3b.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch3b)
DataCube_3c_Convolved = DataCube_3c.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch3c)

time_final = time.time() # time at end

# time benchmarks
print('Initialization Step: ', time_pre_psf - time_start, 's')
print('PSF Generation Step: ', time_pre_convolve - time_pre_psf, 's')
print('Convolution Step: ', time_final - time_pre_convolve, 's')
print('Total Time: ', time_final - time_start, 's')


#%%




#%%


plt.figure('original')
plt.imshow(DataCube_1b.data[740])

#%%

plt.figure('psf matched')
plt.imshow(DataCube_1b_Convolved.data[740])

#%%

plt.figure()
plt.imshow((DataCube_2b.data[940] - DataCube_2b_Convolved.data[940])/DataCube_2b.data[940], vmin=-0.05, vmax=0.05)



#%%

# PogPog.psf_fits[0].data[0].shape


#%%

plt.figure('original')
plt.imshow(DataCube_1a.data[10])

#%%

plt.figure('psf matched')
plt.imshow(DataCube_1a_Convolved.data[10])

#%%

plt.figure('difference (percent)')
plt.imshow((DataCube_1a.data[10] - DataCube_1a_Convolved.data[10])/DataCube_1a.data[10], vmin=-0.1, vmax=0.1)

#%%


frog = DataCube_2b_Convolved.psf_fits[0].data[0]

pog = DataCube_2b.psf_fits[0].data
plt.figure()
plt.imshow(pog[940])

plt.figure()
plt.imshow(frog)



#%%


with fits.open(file_loc_ch2b) as hdul:
    
    data = hdul[1].data
    header = hdul[1].header
    
    data_rotation = hdul[1].header['PA_APER'] # rotation of image w.r.t. JWST V3 axis

#%%

DataCube_2c = DataCube.load_fits(file_loc_ch2c)

DataCube_2c.psf()






#%%



PsfSetup = webbpsf.MIRI()
        
PsfSetup.options['parity'] = 'odd' # ensures PSF will have an odd number of pixels on each side, with the centre of the PSF in the middle of a pixel
PsfSetup.mode = 'IFU' # PSF for data cube, not imager
PsfSetup.band = '3C' # specific subchannel to use in the calculation, e.g. 2A or 3C
        



PsfSetup.pixelscale = 0.13

            

boof = PsfSetup.calc_datacube_fast(np.array([18])*1e-6, oversample=1, fov_arcsec = 3) 

#%%
ax = plt.figure().add_subplot(111)
plt.imshow(boof[0].data[0])
ax.invert_yaxis()

#%%

ax = plt.figure().add_subplot(111)
plt.imshow(DataCube_1a.psf_fits[0].data[10])
ax.invert_yaxis()

#%%


total_input_rotation = -8.2# -8.2-101.74+4.7 # data_rotation
spoof = rotate(boof[0].data, total_input_rotation, axes=(1,2), reshape=False)

ax = plt.figure().add_subplot(111)
plt.imshow(spoof[0], vmax=0.0001)
ax.invert_yaxis()

#%%
where_are_NaNs = np.isnan(data) 
data[where_are_NaNs] = 0
#%%
ax = plt.figure().add_subplot(111)
plt.imshow(data[100], vmax=1, vmin=0)
ax.invert_yaxis()
#%%

# data_roation+4.7-8.2

madoof = rotate(data[100], 101.74)

ax = plt.figure().add_subplot(111)
plt.imshow(madoof)
ax.invert_yaxis()


#%%

# data_roation+4.7-8.2

of = rotate(np.ones((75,50)), -101.74)

ax = plt.figure().add_subplot(111)
plt.imshow(of)
ax.invert_yaxis()

#%%


# trying to make an oversampled psf and rebin myself

PsfSetup1 = webbpsf.MIRI()
        
PsfSetup1.options['parity'] = 'odd' # ensures PSF will have an odd number of pixels on each side, with the centre of the PSF in the middle of a pixel
PsfSetup1.mode = 'IFU' # PSF for data cube, not imager
PsfSetup1.band = '2B' # specific subchannel to use in the calculation, e.g. 2A or 3C
        



PsfSetup1.pixelscale = 0.13

            

boof1 = PsfSetup1.calc_datacube_fast(np.array([10])*1e-6, oversample=5, fov_arcsec = 5) 

PsfSetup2 = webbpsf.MIRI()
        
PsfSetup2.options['parity'] = 'odd' # ensures PSF will have an odd number of pixels on each side, with the centre of the PSF in the middle of a pixel
PsfSetup2.mode = 'IFU' # PSF for data cube, not imager
PsfSetup2.band = '2B' # specific subchannel to use in the calculation, e.g. 2A or 3C
        



PsfSetup2.pixelscale = 0.13

            

boof2 = PsfSetup2.calc_datacube_fast(np.array([9.89135])*1e-6, oversample=5, fov_arcsec = 5) 

old_boof1 = np.copy(boof1[0].data)



