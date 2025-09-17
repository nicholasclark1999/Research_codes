'''
RUN THIS BEFORE RUNNING CODE (assuming you are on linux)
'''

#    webbpsf (now stpsf) installation instructions
#    https://stpsf.readthedocs.io/en/latest/installation.html
#    export STPSF_PATH=~/webbpsf/stpsf-data

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
from astropy.convolution import convolve_fft
from copy import deepcopy


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

    Attributes
    ----------
    fits_file
        TYPE: string
        DESCRIPTION: LOCAL file location of fits file containing JWST cube 
            to be psf-matched. This will be used to create a new fits file
            of identical format that has been psf matched, at the file location
            'new_data/fits_file'
    wavelengths
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array corresponding to the JWST data cube
    data
        TYPE: 3d array of floats
        DESCRIPTION: data cube to be psf-matched, corresponding to the data 
            in fits_file_data
    header
        TYPE: fits header object
        DESCRIPTIONS: the header of EXT 1 of the JWST fits file
    pixelsize
        TYPE: float
        DESCRIPTION: pixelsize of the JWST data array, in units of arcseconds
    data_rotation
        TYPE: float
        DESCRIPTION: the rotation of the JWST data array with respect to the JWST
            V3 axis, in units of degrees
    instrument
        TYPE: string
        DESCRIPTION: the type of JWST instrument used, can be nirspec of miri
    subchannel
        TYPE: string
        DESCRIPTION: the subchannel of the JWST data cube
    channel_rotation
        TYPE: float
        DESCRIPTION: the rotation of the JWST data with respect to the JWST V3 axis 
            attributed to the particular MIRI MRS channel used, in units of degrees. 
            total rotation angle is channel_rotation + data_rotation
    psf_fits
        TYPE: HDUList object
        DESCRIPTION: fits file containing cube of psf's, using the same index
            conventions as jwst data. Each wavelength index is the psf of the 
            corresponding data slice wavelength. In the case of the output 
            psf, a single psf is generated, corresponding to the output wavelength
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

    
    
    #loading in the data from a fits file
    @staticmethod
    def load_fits(fits_file):
        '''
        used to initialize an instance of the DataCube class using fits_file
        
        Attributes
        ----------
        fits_file
            TYPE: string
            DESCRIPTION: LOCAL file location of fits file containing JWST cube 
                to be psf-matched. 
                    
        Returns
        -------
        DataCube
            TYPE: class object
            DESCRIPTION: corresponds to a JWST data cube
        '''
        
        with fits.open(fits_file) as hdul:
            
            data = hdul[1].data
            header = hdul[1].header
            
            data_rotation = hdul[1].header['PA_APER'] # rotation of image w.r.t. JWST V3 axis
            # NOTE: data_rotation does NOT contain individual MIRI MRS rotation w.r.t. JWST V3 axis
            
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
        """
        Generates a cube of PSFs, corresponding to wavelengths of the wavelengths
        attribute. In the case of output PSFs, a single PSF is generated that corresponds
        to the output wavelength. PSF is stored in fits file format, the data
        cube is in ext 0
    
        Returns
        -------
        psf_fits
            TYPE: HDUList object
            DESCRIPTION: fits file containing cube of psf's, using the same index
                conventions as jwst data. Each wavelength index is the psf of the 
                corresponding data slice wavelength. In the case of the output 
                psf, a single psf is generated, corresponding to the output wavelength

        """
        
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
            oversample=1, # do not change this, this value has the most accurate convolution
            fov_arcsec = 3) # do not change this, this value has the most accurate convolution
        
    

    def _psf_rotation(self, OutputDataCube):
        # WARNING 
        # performing a convolution rotates the psf of the input data. running 
        # this step multiple times with the same input data object will result 
        # in additional psf rotations, causing an innacurate convolution. 
        """
        Rotates the PSF to match image data. In order to ensure that the output PSF 
        and input PSF both have the same final rotation, this step is intended
        to be performed immediately before making the kernel. Note that because 
        webbpsf generates MIRI MRS PSFs with an applied rotation that depends on
        the channel in the opposite direction to how the data is rotated, this rotation
        need to be undone. Then, a rotation must be applied to match the input data,
        in addition to the corresponding channel rotation of the input data.
    

        """
        
        total_input_rotation = -1*self.data_rotation  - 2*self.channel_rotation
        total_output_rotation = -1*self.data_rotation - OutputDataCube.channel_rotation - self.channel_rotation
        
        self.psf_fits[0].data = rotate(
            self.psf_fits[0].data, total_input_rotation, axes=(1,2), reshape=False)
        
        OutputDataCube.psf_fits[0].data = rotate(
            OutputDataCube.psf_fits[0].data, total_output_rotation, axes=(1,2), reshape=False)

    
    
    def _kernel_calculator(self, OutputDataCube):
        """
        makes the kernel array between two PSFs. Note that in this case, a kernel is
        the ratio of two optical transfer functions; the fourier transformed PSFs.
        The kernel cube will be the same shape as the input PSF data cube.
    
        Returns
        -------
        kernel
            TYPE: 3d array of floats
            DESCRIPTION: kernel made with 2 PSF's
        """
    
        # the input and output PSF
        input_psf = self.psf_fits[0].data
        output_psf = OutputDataCube.psf_fits[0].data[0] # output_psf contains only a single psf

        # create_matching_kernel requires 2d inputs. So, must be done one slice at a time.
        kernel = np.zeros(input_psf.shape)
        kernel_fourier = np.zeros(input_psf.shape)
        for i in range(input_psf.shape[0]):
            kernel[i], kernel_fourier[i] = pp.homogenization_kernel(output_psf, input_psf[i], reg_fact=1e-5)
            
        return kernel
    
    
    # performs convolution between self and a PSF of specified wavelength
    def convolve(self, output_fits_file, output_psf_wavelength, output_fits_file_save_loc):
        """
        makes the kernel array between two PSFs. Note that in this case, a kernel is
        the ratio of two optical transfer functions; the fourier transformed PSFs.
        The kernel cube will be the same shape as the input PSF data cube.
        
        Attributes
        ----------
        output_fits_file
            TYPE: string
            DESCRIPTION: LOCAL file location of fits file containing JWST cube 
                that acts as the reference. This reference is used for the channel
                rotation
        output_psf_wavelength
            TYPE: float
            DESCRIPTION: the wavelength in microns to PSF match the data to. It should
                ideally be at the end the of the subchannel with the largest wavelengths
                you are using, i.e. if using data up to 3C, 18 microns is a good value
                to use
        output_fits_file_save_loc
            TYPE: string
            DESCRIPTION: where OutputDataCube is saved. It is convinient for this
                to have the same name as the input file, but in a different folder;
                this allows for seamless integration of this code with pre-existing
                codes that work on JWST data products
        
        Returns
        -------
        OutputDataCube
            TYPE: class object
            DESCRIPTION: corresponds to the PSF-matched JWST data cube. Will appear
                identical to the input data cube object, except for channel_rotation, 
                psf_fits, and data attributes.
        """
        
        # sanity check variable for pixelsize
        pix = self.pixelsize
        
        # define new class instance for output image and PSF, based on input class
        OutputDataCube = deepcopy(self)
        
        # make sure OutputDataCube uses channel rotation that corresponds to output psf
        Temp = DataCube.load_fits(output_fits_file)
        OutputDataCube.channel_rotation = Temp.channel_rotation

        # make PSF for specified wavelength, using pixelsize of input data
        OutputDataCube.psf(pixelsize=pix, output_psf_wavelength=output_psf_wavelength)

        # verify PSFs have the same rotation as input data
        self._psf_rotation(OutputDataCube)

        # calculate kernel of PSFs
        kernel = self._kernel_calculator(OutputDataCube)
        
        # perform the convolution
        # note that if a 3d array is given, convolve_fft assumes it is 1 3-dimensional
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
    
    
    
'''
EXAMPLE OF THE CODE IN USE
'''

# below is an example of PSF matching miri channels 1, 2 and 3, to a wavelength
# corresponding to the end of channel 3C. At the end, are some plots showing
# the image and psf before and after convolution.

'''

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
DataCube_1b.psf()
DataCube_1c.psf()
DataCube_2a.psf()
DataCube_2b.psf()
DataCube_2c.psf()
DataCube_3a.psf()
DataCube_3b.psf()
DataCube_3c.psf()

# performing convolution
time_pre_convolve = time.time() # pre convolve step time

# interested in convolving up to beginning of ch4, so go to 18 microns since ch3a ends at 17.95 microns
# saving each file to the new data folder, with the same name as the pre convolved version

DataCube_1a_Convolved = DataCube_1a.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch1a)
DataCube_1b_Convolved = DataCube_1b.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch1b)
DataCube_1c_Convolved = DataCube_1c.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch1c)
DataCube_2a_Convolved = DataCube_2a.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch2a)
DataCube_2b_Convolved = DataCube_2b.convolve(file_loc_ch3c, 18, 'new_data/' + file_loc_ch2b)
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



# showing some diagnostic figures, of before and after PSF matching

plt.figure('original')
plt.imshow(DataCube_1b.data[100])
plt.show()

plt.figure('psf matched')
plt.imshow(DataCube_1b_Convolved.data[100])
plt.show()

plt.figure('original psf')
plt.imshow(DataCube_1b.psf_fits[0].data[100])

plt.figure('18 micron psf')
plt.imshow(DataCube_1b_Convolved.psf_fits[0].data[0])

'''
