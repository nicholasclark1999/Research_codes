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
import webbpsf

import time
import matplotlib.pyplot as plt

from scipy.ndimage import rotate


'''
import convolver as cvr
import helpers as hlp


from astropy.convolution import Gaussian2DKernel

from astropy.utils.data import get_pkg_data_filename


import photutils
# import scipy
from . import helpers
import helpers

import numpy as np
from astropy.convolution import convolve_fft
from helpers import miri_monochrom_psf
from helpers import create_kernel
from helpers import prep_image
from helpers import prep_kernel
from astropy.modeling import models, fitting
from astropy.stats import gaussian_sigma_to_fwhm


import os

import photutils
from astropy.convolution import convolve_fft
from astropy.io import fits
import numpy as np
import scipy.interpolate
import scipy.ndimage
from photutils.psf.matching import TopHatWindow
from synphot import SourceSpectrum
from synphot.models import Empirical1D
from astropy import units as u
#webbpsf.setup_logging()
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
                 instrument,
                 subchannel):

        self.fits_file = fits_file
        self.wavelengths = wavelengths
        self.data = data
        self.header = header
        self.pixelsize = pixelsize
        self.instrument = instrument
        self.subchannel = subchannel

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
        
            # for miri, use starting_wavelength to determine subchannel
            if wavelength_start < 5.5:
                subchannel = '1A'
            elif wavelength_start < 6.0:
                subchannel = '1B'
            elif wavelength_start < 7.0:
                subchannel = '1C'
            elif wavelength_start < 8.0:
                subchannel = '2A'
            elif wavelength_start < 9.0:
                subchannel = '2B'
            elif wavelength_start < 11.0:
                subchannel = '2C'
            elif wavelength_start < 12.0:
                subchannel = '3A'
            elif wavelength_start < 14.0:
                subchannel = '3B'
            elif wavelength_start < 16.0:
                subchannel = '3C'
            elif wavelength_start < 19.0:
                subchannel = '4A'
            elif wavelength_start < 22.0:
                subchannel = '4B'
            else:
                subchannel = '4C'
            
        return DataCube(fits_file, wavelengths, data, header, pixelsize, 
                        instrument, subchannel)
    
    # building psf array
    def psf(self):
        
        if self.instrument == 'miri':
            PsfSetup = webbpsf.MIRI()
        
        PsfSetup.options['parity'] = 'odd' # ensures PSF will have an odd number of pixels on each side, with the centre of the PSF in the middle of a pixel
        PsfSetup.mode = 'IFU' # PSF for data cube, not imager
        PsfSetup.band = self.subchannel # specific subchannel to use in the calculation, e.g. 2A or 3C
        self.psf_fits = PsfSetup.calc_datacube_fast(self.wavelengths*1e-6, # webbpsf takes wavelength units in m, not microns
                                                    oversample=5, # default oversampling is 4, made it 5 so that parity is actually odd
                                                    fov_arcsec = 20) # 10 arcseconds is sufficiently wide for any channel
        
    # at some point will want to rotate psf by header['PA_APER'] to be aligned with image, since psf already has mrs angle applied and image has both
    # do this rotation right before kernel is calculated
    
    # @staticmethod def kernel and convolver, will build a new image/psf to convolve to, containing an input subchannel to psf match 2, and 
    # an input for the already defined psf/image class. Will then seamlessly calculate kernel and then convolve. do this in 2 functions, with _kernel so its
    # only for debugging purposes
    
'''
def miri_monochrom_psf(wave, band, fov_arcsec=10, norm='last'):
    """Calculates a monochromatic PSF for MIRI.
    Here monochromatic means a single wavelength slice of a data cube
    
    note that because pixelscale and fov_arcsec depend on JWST psf and detectors, they are unlikely to change and so they can be safely left alone (for now at least)
    Args:
        wave: Wavelength at which the psf is calculated in units of m.
        oversample: factor by which the PSF will be oversampled. (removed at the moment)
        pixelscale: Pixel scale in arcsec (default is 0.11 for the imager) (removed at the moment)
        fov_arcsec: FOV of the PSF.
        norm: PSF normalization. The default value of 'last' will 
        normalize PSF after the calculation.
    Returns:
        psf: The output PSF is returned as a fits.HDUlist object.
    """
    miri = webbpsf.MIRI()
    miri.options['parity'] = 'odd' #because the input array is odd in length, refers to how things are centered in webbpsf
    miri.mode = 'IFU' # PSF for MIRI MRS
    miri.band = band # specific subchannel to use in the calculation, e.g. 2A or 3C
    #miri.pixelscale= pixelscale
    miri.options['output mode'] = 'both' #can output oversampled, detector sampled, or both. 
    print('test')
    psf = miri.calc_datacube(wave)
    

    psf[0].data = psf[0].data[0] # makes the psf a 2d array for now

    return psf 
'''

PogPog = DataCube.load_fits(')



PogPog.wavelengths = PogPog.wavelengths[:10]

#%%
q = time.time()
PogPog.psf()
w = time.time()
print(w-q)


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

























