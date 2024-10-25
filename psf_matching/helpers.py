#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 19:52:53 2022
@author: Ameek Sidhu and Ryan Chown

Modified and upkept by Nicholas Clark
"""
#%%
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

#%%

def residuals(psf1, psf2, kernel):
    '''Compute kernel residuals |psf2-psf1*K|/psf2
    
    note: a residual is the difference between an actual value and the value predicted by a model.
    args:
        psf1: np.ndarray : narrow psf
        psf2: np.ndarray : wider psf
        kernel: np.ndarray : matching kernel K from psf1 to psf2
    return:
        residuals : np.ndarray : kernel residuals
    '''
    convolve = convolve_fft(psf1, kernel, allow_huge=True)
    return abs(psf2 - convolve) / psf2

#%%

def pixelscale(fits_file):
    """Find pixel scale keywords in FITS file
    
    Note that keyword names are standardized, and so are unlikely to change and can be safely searched for in this manor.
    Args:
        fits_file : str : Path to a FITS image file
    Returns:
        pixel_scale: float : pixel scale in arcsec
    """
    hdul = fits.open(fits_file)
    hdr = fits.getheader(fits_file, 0)
    if len(hdul) > 1:
        hdr += fits.getheader(fits_file, 1)
    hdul.close()
    pix_key = [
        key for key in
        ['CDELT1', 'CD1_1', 'PIXELSCL', 'PIXSCALE', 'PXLSCALE', 'PIXSCL']
        if key in list(hdr.keys())
    ]

    if not pix_key:
        raise IOError("Pixel scale not found in {0}.".format(fits_file))

    units = [hdr[unit] for unit in ['CUNIT1'] if unit in list(hdr.keys())]
    for key in pix_key:
        if 'arcsec' in hdr.comments[key]:
            units += ['arcsec']
        elif 'deg' in hdr.comments[key]:
            units += ['deg']

    if not units:
        units += ['arcsec']
        raise IOError("Pixel scale units not found in {0}.".format(fits_file))

    pixel_scale = abs(hdr[pix_key.pop()])

    if 'deg' in units.pop():
        pixel_scale *= 3600

    return pixel_scale
#%%

def source_spectrum(wave, flux):
    '''Derive a SourceSpectrum object as an input for miri_broad_psf
    Args:
        wave: array: wavelength array in microns
        flux: array: integrated flux array in Jy
    Returns:
        spec: SourceSpectrum object
    '''
    sp = SourceSpectrum(Empirical1D,
                        points=wave * u.um,
                        lookup_table=flux * u.Jy,
                        keep_neg=True,
                        meta={'name': 'PDR'})
    return sp

#%%

#originally had oversample=4 and pixelscale=0.11

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
    
    '''
    psf = miri.calc_psf(monochromatic=wave,
                        oversample=oversample,
                        normalize=norm,
                        fov_arcsec=fov_arcsec)
    '''
    psf[0].data = psf[0].data[0] # makes the psf a 2d array for now

    return psf 

#%%

#originally had oversample=4 and pixelscale=0.11

def miri_broad_psf(mirifilter, fov_arcsec=10, norm='last', source=None):
    """Calculates a monochromatic PSF for MIRI.
    Here broad means a MIRI image is used, instead of a slice of a data cube.
    
    note that because pixelscale and fov_arcsec depend on JWST psf and detectors, they are unlikely to change and so they can be safely left alone (for now at least)
    Args:
        mirifilter: str: MIRI broad filter 
        oversample: int: factor by which the PSF will be oversampled.  (removed at the moment)
        pixelscale: int: Pixel scale in arcsec (default is 0.11 for the imager)  (removed at the moment)
        fov_arcsec: int: FOV of the PSF.
        norm: str: PSF normalization. The default value of 'last' will 
        normalize PSF after the calculation.
        source: input spectrum as a synphot.SourceSpectrum object
    Returns:
        psf: The output PSF is returned as a fits.HDUlist object.
    """
    miri = webbpsf.MIRI()
    miri.options['parity'] = 'odd' #because the input array is odd in length, refers to how things are centered in webbpsf
    miri.options['output mode'] = 'both' #can output oversampled, detector sampled, or both. 
    miri.filter = mirifilter
    #miri.pixelscale= pixelscale
    psf = miri.calc_psf(monochromatic=None,
                        normalize=norm,
                        fov_arcsec=fov_arcsec,
                        source=source)
    
    '''
    psf = miri.calc_psf(monochromatic=None,
                        oversample=oversample,
                        normalize=norm,
                        fov_arcsec=fov_arcsec,
                        source=source)
    '''

    return psf

#%%

#originally had oversample=4

def nirspec_monochrom_psf(wave, fov_arcsec=10, norm='last', mode='ifu'):
    """Calculates a monochromatic PSF for NIRSpec.
    Here monochromatic means a single wavelength slice of a data cube
    
    note that because pixelscale and fov_arcsec depend on JWST psf and detectors, they are unlikely to change and so they can be safely left alone (for now at least)
    Args:
        wave: Wavelength at which the psf is calculated in units of m.wave: Wavelength at which the psf is calculated in units of m.
        oversample: factor by which the PSF will be oversampled. (removed at the moment)
        fov_arcsec: FOV of the PSF.
        norm: PSF normalization. The default value of 'last' will 
        normalize PSF after the calculation.
    Returns:
        psf: The output PSF is returned as a fits.HDUlist object.
    """
    nirspec = webbpsf.NIRSpec()
    nirspec.options['parity'] = 'odd' #because the input array is odd in length, refers to how things are centered in webbpsf
    nirspec.options['output mode'] = 'both' #can output oversampled, detector sampled, or both. 
    psf = nirspec.calc_psf(monochromatic=wave,
                        normalize=norm,
                        fov_arcsec=fov_arcsec)
    
    '''
    psf = nirspec.calc_psf(monochromatic=wave,
                        oversample=oversample,
                        normalize=norm,
                        fov_arcsec=fov_arcsec)\
    '''
    
    return psf

#%%

def psf_rotation(psf, target_channel='Ch4'):
    """Rotates the psf by an angle relative to
    the MIRI Imager. Rotation of MIRI Imager and the NIRSpec is
    applied in the webbpsf.
    
    
    Args:
        psf: 2-D array containing the PSF to be rotated.
        target_channel: Name of the channel along which the psf will be rotated.
    Returns:
        psf_rot: The rotated psf is returned as a 2-D array.
    """
    list_channels = ['Ch1', 'Ch2', 'Ch3', 'Ch4']
    mapping_rotation_angle = {'Ch1': 3.9, 'Ch2': 3.6, 'Ch3': 3.2, 'Ch4': 3.8}
    rotation_angle = mapping_rotation_angle[target_channel]

    psf_rot = scipy.ndimage.rotate(psf,
                                   angle=rotation_angle,
                                   reshape=False,
                                   prefilter=False)

    return psf_rot

#%%

def otf(psf, shape):
    """ Convert point-spread function to optical transfer function.
    
    note: optical transfer function is the fourier transform of the psf
    
    Args:
        psf: `np.ndarray`: PSF array
        shape : int : Output shape of the OTF array
    Returns:
         otf : `np.ndarray`: OTF array
    """
    inshape = psf.shape
    psf = pad(psf, shape, position='corner')
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)
    otf = np.fft.fft2(psf)
    return otf

#%%

def wiener_filter(psf, mu):
    """Create a Wiener filter using a PSF image.
    The signal is l2 penalized by a 2D Laplacian operator that
    serves as a high-pass filter for the regularization process.
    
    Note: 'L2 penalized' is a means of dealing with oversampling in machine learning.
    
    A wiener filter creates an unknown 'nonnoisy' image using a known 'noisy' image, for example. 
    
    Args:
        psf: `np.ndarray`: PSF array
        mu : float : Regularisation parameter for the Wiener filter
    Returns:
         wiener: np.ndarray`: Fourier space Wiener filter
    """
    LAPLACIAN = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    
    #here trans means fourier transformed
    trans_psf = otf(psf, psf.shape)
    trans_lap = otf(LAPLACIAN, psf.shape)
    wiener = np.conj(trans_psf) / (np.abs(trans_psf)**2 +
                                   mu * np.abs(trans_lap)**2)
    return wiener
#%%

def pad(image, shape, position='corner'):
    """This function pads image to a given shape
    Args:
        image : np.ndarray`: Input image
        shape : tuple of int : Desired output shape of the image
        position : str, optional : The position of the input image in the output one
    Returns:
        padded image: np.ndarray`: zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)
    if np.alltrue(imshape == shape):
        return image
    dshape = shape - imshape
    pad_img = np.zeros(shape, dtype=image.dtype)
    idx, idy = np.indices(imshape)
    if position == 'center':
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)
    pad_img[idx + offx, idy + offy] = image
    return pad_img
#%%

def trim(image, shape):
    """This function trims image to a given shape
    Args:
        image : np.ndarray`: Input image
        shape: tuple of int : Desired output shape of the image
    Returns:
        trimmed image: np.ndarray`: Input image trimmed
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image
    
    #nicholas added this to fix dshape not being defined
    dshape = imshape - shape

    if np.any(shape <= 0) or np.any(dshape < 0) or np.any(dshape % 2 != 0):
        raise ValueError("Shape error for trim step")

    dshape = imshape - shape

    idx, idy = np.indices(shape)
    offx, offy = dshape // 2

    return image[idx + offx, idy + offy]

#%%
''' NOT DONE'''
def create_kernel(psf_source, psf_target, regul, method):
    """
    This function creates a kernel.
    Args:
        psf_source = The source PSF.
                 The source PSF should have higher
                 resolution (i.e., narrower) than the target PSF.
        psf_target = The target PSF.
                 The target PSF should have lower resolution
                 (i.e., broader) than the source PSF.
        regul : float : regularisation parameter (default is 0.11). 1/SNR for B+16.
        method : str : 'A+11' (Aniano+2011, default) or 'B+16' (Boucaud+2016) method
    Returns:
        kernel = 2-D array containing the matching kernel
             to go from source_psf to target_psf.
    """
    print(regul, method)
    # There should be some checks in this function
    # TODO (RC): incorporate low pass filter
    # TODO (RC): add more arguments, e.g. a "default" option where some hard-coded options are used,
    #           and another option for users to supply their own filters/options to create_matching_kernel
    # TODO (AS): Hard code the value of parameters for different channels
    if method == 'A+11':
        window = TopHatWindow(regul) #defines a 2d top hat window function, which is a centred circle of 1s, the other values are 0.
        kernel = photutils.psf.matching.create_matching_kernel(
            psf_source, psf_target, window=window) #modified this from window to window=window, since in the function its described as window=None
    elif method == 'B+16':
        #generates fourier transformed wiener filter
        wiener = wiener_filter(psf_source, regul)
        norm = np.sqrt(psf_target.size)
        kernel = wiener * np.fft.fft2(psf_target) / norm 
        norm = np.sqrt(kernel.size)
        kernel = np.real(np.fft.ifft2(kernel) * norm)
        kernel.clip(-1, 1)
    return kernel
#%%

def prep_image(image, pixel_scale, padding_arcseconds=100.0):
    """
    This function prepares the image for convolution. This is done by adding padding (by default, 100 arcseconds worth) around the input image.
    args:
        image = 2-D array containg the image to be convolved.
        pixel_scale = pixel scale of the input image to be convolved.
        padding_arcseconds = float containing the black sky in arcseconds
                         which is added to the each side of image to be able to
                         include the boundaries contributions in the convolutions.
                         By default, padding_arcseconds is set to 100 arcseconds.
    RETURN:
    prepped_image = 2-D array containing the image ready for convolution.
    flag = 2-D array containing the flags (1's and 0's) for the indices
           added to the prepped_image due to padding. 1's are written at the indices
           with original data, and 0's are written at indices with padded data.
    """

    pixels_added = int(padding_arcseconds / pixel_scale)
    prepped_image = np.pad(image, (pixels_added),
                           'constant',
                           constant_values=(0))
    flag_array = np.ones_like(image)
    flag_array = np.pad(flag_array, (pixels_added),
                        'constant',
                        constant_values=(0))
    return prepped_image, flag_array
#%%

def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''
    This function performs the rebinning of the input array.
    It is adopted from https://scipy-cookbook.readthedocs.io/items/Rebinning.html
    Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).
    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.
    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates
    
    args:
        a = input array to be rebinned
        newdims = dimensions of the rebinned array in the form of (x, y).
        method = method adopted for rebinning.
             By default, this variable is set to 'linear'.
        centre = True: interpolation points are at the centres of the bins.
             False: interpolation points are at the front edge of the bin.
        minusone = For example- inarray.shape = (i,j) & new dimensions = (x,y)
               False: inarray is resampled by factors of (i/x) * (j/y)
               True: inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
               This prevents extrapolation one element beyond bounds of input array.
    returns
        newa = Rebinned array
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array(a.shape)
    ndims = len(a.shape)
    if len(newdims) != ndims:
        print(
            "Congrid error: can only rebin to the same number of dimensions.")
        return None
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    if method == 'neighbour':
        for i in range(ndims):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array(dimlist).round().astype(int)
        newa = a[list(cd)]
        return newa

    elif method in ['nearest', 'linear']:
        # calculate new dims
        for i in range(ndims):
            base = np.arange(newdims[i])
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype=float) for i in list(a.shape)] #modified this from depreciated np.float to float

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d(olddims[-1],
                                          a,
                                          kind=method,
                                          fill_value='extrapolate')
        newa = mint(dimlist[-1])

        trorder = [ndims - 1] + [i for i in range(ndims - 1)]
        for i in range(ndims - 2, -1, -1):
            newa = newa.transpose(trorder)

            mint = scipy.interpolate.interp1d(olddims[i],
                                              newa,
                                              kind=method,
                                              fill_value='extrapolate')
            newa = mint(dimlist[i])

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose(trorder)

        return newa
    elif method in ['spline']:
        oslices = [slice(0, j) for j in old]
        oldcoords = np.ogrid[oslices]
        nslices = [slice(0, j) for j in list(newdims)]
        newcoords = np.mgrid[nslices]

        newcoords_dims = range(np.rank(newcoords))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.")
        print(
            "Must be one of: \'neighbour\', \'nearest\', \'linear\', or \'spline\'"
        )
        return None

#%%

def center_psf(image):
    """
    This function recenters the psf/kernel.
    Args:
        image = 2-D array containg the psf/kernel to be recentered.
    Returns:
        image_centered = 2-D array with the psf re-cenetered
    """
    #obscure operator time: | is 'bitwise or'. For x | y, output is 0 if x AND y is 0, otherwise it is 1.
    #mask is 0 unless there is no nan, inf, or 0 in that index.
    mask = np.isnan(image) | np.isinf(image) | (image == 0) 
    x0 = np.shape(image)[0]
    y0 = np.shape(image)[1]

    x0 = (x0 - 1) // 2
    y0 = (y0 - 1) // 2

    xc, yc = photutils.centroids.centroid_2dg(image, mask=mask)

    shift_x = x0 - int(xc)
    shift_y = y0 - int(yc)

    # Now shift to the center
    image_centered = np.roll(image, (shift_x, shift_y))

    return image_centered

#%%

def prep_kernel(kernel, pixel_scale_kernel, pixel_scale_image):
    """
    This function prepares the kernel for convolution.
    (1): make kernal odd square if length is even (i.e. size is 2N+1 by 2N+1, N a positive integer)
    (2): resamples kernel and image grid to same pixel size
    (3): recentres the resampled kernel
    Args:
        image = 2-D array containg the kernel.
        pixel_scale_kernel = pixel scale of the kernel in arcseconds.
        pixel_scale_image = pixel scale of the image in arcseconds.
    Returns:
        prepped_kernel = 2-D array containing the kernel ready for convolution.
    """
    x_size_old = np.shape(kernel)[0]
    y_size_old = np.shape(kernel)[1] #this is technically not needed since it is a square shape

    size_new = x_size_old

    if size_new % 2 == 0:
        size_new = size_new + 1

    if (size_new > x_size_old): #new row and column is added at the beginning of the array, so that these new ones are the first row and column
        kernel = np.pad(kernel, ((1, 0), (1, 0)),
                        'constant',
                        constant_values=(0))
    
    #resizes kernel to match image if the difference in size between kernel and image is greater than 5%
    if (abs(pixel_scale_kernel - pixel_scale_image) /
            pixel_scale_image) > 0.05:
        size_ker = np.shape(kernel)[0]
        size_new = round((size_ker) * pixel_scale_kernel / pixel_scale_image)

    if size_new % 2 == 0:
        size_new = size_new + 1

    resampled_kernel = congrid(kernel, (size_new, size_new), centre=True)

    new_image_size_x = np.shape(resampled_kernel)[0]
    new_image_size_y = np.shape(resampled_kernel)[1]
    max_ker_size = min(new_image_size_x, new_image_size_y)

    if (max_ker_size % 2) == 0:
        max_ker_size = max_ker_size - 1

    size_ker = np.shape(resampled_kernel)[0]
    
    #making sure length of new kernal is odd, trimming it if it isnt
    if max_ker_size < size_ker:
        trim_side = int(size_ker - max_ker_size) / 2
        resampled_kernel = resampled_kernel[trim_side:trim_side +
                                            max_ker_size - 1,
                                            trim_side:trim_side +
                                            max_ker_size - 1]
    
    #making sure kernel is centered
    prepped_kernel = center_psf(resampled_kernel)

    return prepped_kernel

import time

pog1 = time.time()

psf = miri_monochrom_psf([5*1e-6], '1A', fov_arcsec=10, norm='last')

pog2 = time.time()

print(pog2 - pog1)

#%%

import matplotlib.pyplot as plt

plt.imshow(psf[0].data)