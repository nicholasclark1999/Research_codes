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
            
        # if a single wavelength is specified, use this instead of the entire wavelength array
        output_psf_wavelength = kwargs.get('output_psf_wavelength')
        if output_psf_wavelength is not None:
            wavelengths = np.array([output_psf_wavelength]) # needs to be in array format, not a float
        else:
            wavelengths = self.wavelengths
        
        self.psf_fits = PsfSetup.calc_datacube_fast(
            wavelengths*1e-6, # webbpsf takes wavelength units in m, not microns
            oversample=5, # default oversampling is 4, made it 5 so that parity is actually odd
            fov_arcsec = 20) # 10 arcseconds is sufficiently wide for any channel
    
    
    
    # rotating the PSF array to match the image rotation, before convolution
    # note that this will be the rotation of the input data, if it is different from the output data
    
    # the output PSF should have the same subchannel rotation as the input data
    def _psf_rotation(self, OutputDataCube): 
    
       # rotation to apply will be the rotation of the image w.r.t. JWST V3 axis, 
       # which includes mrs rotation, so this must be removed from the PSF.
       
       total_input_rotation = self.data_rotation - self.channel_rotation
       total_output_rotation = self.data_rotation - OutputDataCube.channel_rotation
                
       # rotating PSF, will be the same shape after rotation. Since the PSF is centred
       # on a pixel, the rotation occurs about the center of the PSF
       
       # for i in range(self.psf_fits[0].data.shape[0])
       
       self.psf_fits[0].data = rotate(
           self.psf_fits[0].data[0], total_input_rotation, axes=(1,2))
       
       OutputDataCube.psf_fits[0].data = rotate(
           OutputDataCube.psf_fits[0].data[0], total_output_rotation, axes=(1,2))
                
    
    
    # makes the kernel between two PSFs
    def _kernel_calculator(self, OutputDataCube):

        # both PSF will have the pixelsize of self, and both will have the size of self
    
        # TODO (): add different regul, method support 
    
            # from original kernel function:
            # regul : float : regularisation parameter (default is 0.11). 1/SNR for B+16.
            # method : str : 'A+11' (Aniano+2011, default) or 'B+16' (Boucaud+2016) method
    
        # at the moment, hard code regul to be 0.11 and use the A+11 method
        
        # the input and output PSF
        input_psf = self.psf_fits[0].data[0]
        output_psf = OutputDataCube.psf_fits[0].data[0]
        
        # define window function
        regul = 0.11
        window = TopHatWindow(regul) # top hat function: a centred circle of 1; the other values are 0.
        
        # create_matching_kernel requires 2d inputs. So, must be done one slice at a time.
        kernel = np.zeros(input_psf)
        for i in range(input_psf.shape[0]):
            kernel[i] = create_matching_kernel(input_psf[i], output_psf[i], window=window)
        return kernel
        
    
    
    # performs convolution between self and a PSF of specified wavelength
    def convolve(self, output_fits_file, output_psf_wavelength):
        
        # define new class instance for output image and PSF
        OutputDataCube = DataCube.load_fits(output_fits_file)
        
        # make PSF for specified wavelength, using pixelsize of input data
        OutputDataCube.psf(pixelsize=self.pixelsize, output_psf_wavelength=output_psf_wavelength)
        
        # verify PSFs have the same rotation as input data
        
        
        # calculate kernel of PSFs
        kernel = self._kernel_calculator(OutputDataCube)
        
        # perform the convolution
        # note that if a 3d array is given, this function assumes it is 1 3 dimensional
        # kernel, not a series of 2d kernels. So, this must be done one slice at a time.
        convolution = np.zeros(self.data.shape)
        for i in range(self.data.shape[0]):
            convolution[i] = convolve_fft(self.data[i], kernel[i])
        
        # replace data with convolution in output, and save new fits file
        OutputDataCube.data = convolution 
        
        with fits.open(OutputDataCube.fits_file) as hdul:
            hdul[1].data = convolution
            
            # verify new_data folder exists, create it if not 
            cwd = os.getcwd()
            new_data_path = os.path.join(cwd, '/new_data')
            if not os.path.exists(new_data_path):
                os.makedirs(new_data_path)
            
            #saves to new_data folder with the same name as output fits file
            hdul.writeto('new_data/' + OutputDataCube.fits_file)
            
        return OutputDataCube
    
    
        
    


PogPog = DataCube.load_fits('data_ch1c.fits')



PogPog.wavelengths = PogPog.wavelengths[:10]

#%%
q = time.time()
PogPog.psf()
w = time.time()
print(w-q)


FrogFrog = PogPog.convolve('data_ch3c.fits', 18.0)























#%%

plt.imshow(PogPog.psf_fits[0].data[0])

#%%

qwerty = PogPog.data[0]





#%%

erty = np.copy(qwerty)

where_are_NaNs = np.isnan(erty) 
erty[where_are_NaNs] = 0

plt.imshow(erty)
#
#%%

werty = rotate(erty, (8.4 + 93.54553500726477), reshape=False) #

plt.imshow(werty)

#%%

plt.imshow(rotate(PogPog.psf_fits[0].data[0], 90-8.4), vmax=0.001)


#%%
with fits.open('data_ch1c.fits') as hdul:
    
    fro = hdul[0].header


#%%

def test(**kwargs):
    
    print(kwargs['hello'])





test(hello = 4)




#%%

pog = np.ones((3,4,5))

pog[1] *=2
pog[2] *=3

for slice in pog[:]:
    print(pog)









