'''
RUN THIS BEFORE RUNNING CODE (assuming you are on linux)
'''

#    webbpsf installation instructions
#    https://webbpsf.readthedocs.io/en/latest/installation.html
#    export WEBBPSF_PATH=~/webbpsf/webbpsf-data
#    then relaunch spyder in terminal, or else it wont reconise the environment variable.



'''
IMPORTING MODULES
'''

import convolver as cvr
import helpers as hlp
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
import numpy as np
from astropy.utils.data import get_pkg_data_filename

import photutils
# import scipy
# from . import helpers
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
import webbpsf
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
        
        
        
        arr (np.ndarray) : The image array
        pixsize_arcsec (float): pixel scale of the input image to be convolved.
        prepared (bool) : Has the image been prepared?
        padding (np.ndarray) : Array with 1's everywhere there is data, and
            0's around the edges.
    """
    def __init__(self, arr, pixsize_arcsec, prepared=False, padding=None):
        self.arr = arr
        self.pixsize_arcsec = pixsize_arcsec
        self.prepared = prepared
        self.padding = padding
    
    #calls the prep_image function from helpers
    def prepare(self, **kwargs):
        if not self.prepared:
            self.arr, self.padding = helpers.prep_image(
                self.arr, self.pixsize_arcsec, **kwargs)
            self.prepared = True




















































