#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 13:49:08 2024

@author: nclark
"""

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

from copy import deepcopy

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
    """
    
    def __init__(self, 
                 fits_file, 
                 wavelengths,
                 data, 
                 header, 
                 pixelsize, 
                 header_wcs):

        self.fits_file = fits_file
        self.wavelengths = wavelengths
        self.data = data
        self.header = header
        self.pixelsize = pixelsize
        self.header_wcs = header_wcs

    
    
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
            
            pixelsize = header['CDELT1'] # units of degrees
            pixelsize *= 3600 # convert to arcseconds
            
            number_wavelengths = header["NAXIS3"]
            wavelength_increment = header["CDELT3"]
            wavelength_start = header["CRVAL3"] # units of microns
            
            # calculate WCS object
            header_wcs = wcs.WCS(header, fobj=hdul, naxis=2) #naxis=2, corresponds to RA and Dec
            
            # Turn WCS into a header object, and then back to WCS again 
            # this makes NAXIS correspond to 2d instead of 3d
            header_wcs = wcs.WCS(header_wcs.to_header())
            

                
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

        return DataCube(fits_file, wavelengths, data, header, pixelsize, header_wcs)
        


    def reproject(self, output_fits_file, output_fits_file_save_loc):
        """
        Performs the reprojection
        
        Attributes
        ----------
        output_fits_file
            TYPE: string
            DESCRIPTION: LOCAL file location of fits file containing JWST cube 
                that acts as the reference. 
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
            DESCRIPTION: corresponds to the reprojected JWST data cube. 
        """
        
        # define new class instance for output image
        OutputDataCube = deepcopy(self)
        
        # define a temp object that contains reference info
        Temp = DataCube.load_fits(output_fits_file)
              
        shape_out = Temp.data[0].shape
        wcs_out = Temp.header_wcs
        
        # need to remove nans
        input_nans = np.isnan(self.data)
        self.data[input_nans] = 0 
        
        #getting wcs coords (input)
        wcs_cube = self.header_wcs
        reprojected_data = np.zeros((len(self.data[:,0,0]), len(Temp.data[0,:,0]), len(Temp.data[0,0,:])))
        
        N = len((self.data[:,0,0]))
        for i in range(N):
            #note reproject_exact takes a tuple of the data and an astropy wcs object
            reprojected_data[i], _ = reproject_exact((self.data[i], wcs_cube),
                                     wcs_out,
                                     shape_out,
                                     parallel=False) # refers to multithreading
    
        # updating OutputDataCube to contain reprojected data
        OutputDataCube.data = reprojected_data
        
        with fits.open(OutputDataCube.fits_file) as hdul:
            # updating header variables

            # CRVAL1,2 = wcs coordinates of fits file
            OutputDataCube.header['CRVAL1'] = Temp.header['CRVAL1']
            OutputDataCube.header['CRVAL2'] = Temp.header['CRVAL2']
            
            # CRPIX1,2 = pixel coordinate that corresponds to CRVAL1,2
            OutputDataCube.header['CRPIX1'] = Temp.header['CRPIX1']
            OutputDataCube.header['CRPIX2'] = Temp.header['CRPIX2'] 
            
            # CDELT1,2 = pixel size
            OutputDataCube.header['CDELT1'] = Temp.header['CDELT1']
            OutputDataCube.header['CDELT2'] = Temp.header['CDELT2']
            
            #saves to specified local file location
            hdul.writeto(output_fits_file_save_loc, overwrite=True)
        
        return OutputDataCube
    
    

'''
EXAMPLE OF THE CODE IN USE
'''

# below is an example of reprojecting miri channels 1, 2 and 3, to match
# channel 3C. This is intended to be done after the PSF matching step.

'''

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

# building non reprojected cubes
DataCube_1a = DataCube.load_fits(file_loc_ch1a)
DataCube_1b = DataCube.load_fits(file_loc_ch1b)
DataCube_1c = DataCube.load_fits(file_loc_ch1c)
DataCube_2a = DataCube.load_fits(file_loc_ch2a)
DataCube_2b = DataCube.load_fits(file_loc_ch2b)
DataCube_2c = DataCube.load_fits(file_loc_ch2c)
DataCube_3a = DataCube.load_fits(file_loc_ch3a)
DataCube_3b = DataCube.load_fits(file_loc_ch3b)
DataCube_3c = DataCube.load_fits(file_loc_ch3c)

# performing reprojection
DataCube_1a_Reprojected = DataCube_1a.reproject(file_loc_ch3c, 'new_data/' + file_loc_ch1a)
DataCube_1b_Reprojected = DataCube_1b.reproject(file_loc_ch3c, 'new_data/' + file_loc_ch1b)
DataCube_1c_Reprojected = DataCube_1c.reproject(file_loc_ch3c, 'new_data/' + file_loc_ch1c)
DataCube_2a_Reprojected = DataCube_2a.reproject(file_loc_ch3c, 'new_data/' + file_loc_ch2a)
DataCube_2b_Reprojected = DataCube_2b.reproject(file_loc_ch3c, 'new_data/' + file_loc_ch2b)
DataCube_2c_Reprojected = DataCube_2c.reproject(file_loc_ch3c, 'new_data/' + file_loc_ch2c)
DataCube_3a_Reprojected = DataCube_3a.reproject(file_loc_ch3c, 'new_data/' + file_loc_ch3a)
DataCube_3b_Reprojected = DataCube_3b.reproject(file_loc_ch3c, 'new_data/' + file_loc_ch3b)
DataCube_3c_Reprojected = DataCube_3c.reproject(file_loc_ch3c, 'new_data/' + file_loc_ch3c)

'''
