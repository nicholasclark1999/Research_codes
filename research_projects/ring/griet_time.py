# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 00:25:37 2024

@author: nickj
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 13:33:30 2023

@author: gsteene
"""
import numpy as np

import pyspeckit

from specutils import Spectrum1D

from astropy import units as u
from astropy.io import fits
#import astropy.io.fits as pyfits
from astropy.wcs import WCS
from astropy.nddata import StdDevUncertainty
#from astropy import stats
#from astropy.modeling import models

from spectral_cube import SpectralCube

import regions

from matplotlib import pyplot as plt

# Fitting H2 1-0 S(1) line in NIR data cube
#https://pyspeckit.readthedocs.io/en/latest/example_NIR_cube.html

# for PAH ... https://pyspeckit.readthedocs.io/en/latest/example_sdss.html

# NIRSPEC north
inputdir0 = 'input/'

# NIRSPEC west
#inputdir0 = '/Users/gsteene/JWST/RingNebula/MAST_2023-04-06/extraction_nirspec_west_common_region/'

inputdir = 'input/'
outdir = 'output/'

fname = 'jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits'

# read in csv file and plot spectral region of interest
#fname1= 'jw01558-o008_t007_nirspec_g395m-f290lp_s3d_masked_spec_sum.csv'
fname1 = 'jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned_spec_sum.csv'

area='north'
#area='west'


# FWHM=2√2ln2σ≈2.355 sigma

# load the extracted spectrum and plot to have a look at the wavelength region
wl, fl, err= np.loadtxt(inputdir0+fname1,dtype='float',usecols=(0,1,2),skiprows=1,unpack=True)
lamb = wl * u.micron
flux = fl *  1.e3 *u.mJy
fluxerr = (err * 1.e3)**2
#print(flux)
spec = Spectrum1D(spectral_axis=lamb, flux=flux,uncertainty=StdDevUncertainty(fluxerr))
#spec_err = Spectrum1D(spectral_axis=lamb, flux=fluxerr)

f, ax = plt.subplots()  
#ax.step, ax.plot, ax.scatter all work, but ax.scatter doesn't autscale correctly
#plt.errorbar only works on unitless data !
ax.step(spec.spectral_axis, spec.flux)
#ax.errorbar(spec1.spectral_axis.value, spec1.flux.value, yerr=spec1.uncertainty,label='Data', color='k')
#ax.plot(spec1.spectral_axis.value, spec1.flux.value, yerr=spec1.uncertainty,label='Data', color='k')
#plt.xlim(1.955,1.960) #line
#plt.xlim(1.950,1.965) S3
plt.xlim(3.21,3.36)
plt.ylim(-0.1,1.0)
plt.xlabel('wavelength (micron)')
plt.ylabel('Flux*(mJy)')
#plt.savefig(outdir+'NIRSPEC_WEST_spec_PAH.png',bbox_inches='tight')
plt.show()
plt.close()



# read in first masked cube
# Open the FITS file for reading
nirsp = fits.open(inputdir+fname)  
#nirsp1.info()
# header and data
hdr= nirsp[1].header
data_nirsp= nirsp[1].data *  u.MJy/u.sr
#data_nirspec= data_nirsp * 10**9 * (2.35040007004737E-13 / 0.01000000029802323) *u.mJy/u.arcsec**2
wcs_nirspec = WCS(hdr)
cb = SpectralCube(data=data_nirsp,wcs=wcs_nirspec).with_spectral_unit(u.micron)
cb_mask = cb > 0.0 * u.MJy/u.sr
cube = cb.with_mask(cb_mask).with_fill_value(0.0)

scube = cube.spectral_slab(3.21*u.micron,3.36*u.micron)
spectral_axis = scube.with_spectral_unit(u.micron).spectral_axis
good_channels = (spectral_axis < 3.22*u.micron) #| (spectral_axis > 3.35*u.micron)
masked_scube = scube.with_mask(good_channels[:, np.newaxis, np.newaxis])
cont = masked_scube.median(axis=0)
cube_cont = scube - cont


# error
hdr_err=nirsp['ERR'].header
err_nirsp = nirsp['ERR'].data  * u.MJy/u.sr
print(err_nirsp.shape)
cube_err = SpectralCube(data=err_nirsp,wcs=wcs_nirspec).with_spectral_unit(u.micron)

# take subcubes over common region

#WEST
#regionfile=inputdir+'ring_west_common_region.reg'
#regionfile=inputdir+'NIRSPEC_WEST_bigblob.reg'
#regionfile=inputdir+'NIRSPEC_HST_absregion.reg'
#NORTH
regionfile=inputdir+'nirspec_miri_north_common_region_new.reg'

region_list = regions.Regions.read(regionfile)  
print(len(region_list))
if len(region_list)  == 1 :
  for region in region_list:
      subcube = cube_cont.subcube_from_regions([region])
      #subcube = cube.subcube_from_regions([region])
      subcube_err = cube_err.subcube_from_regions([region])



# Create a pyspeckit cube for the fitting
#pcube = pyspeckit.Cube(cube=subcube)
pcube = pyspeckit.Cube(cube=subcube)

# Slice the cube over the wavelength range you'd like to fit
# I add an offset to avoid negative values
#
cube_PAH = pcube.slice(3.24,3.36,unit='micron')


std = cube_PAH.stats((3.24, 3.36))['std']
med = cube_PAH.stats((3.24, 3.36))['median']
mea = cube_PAH.stats((3.24, 3.36))['mean']

# Create a pyspeckit cube
pcube_err = pyspeckit.Cube(cube=subcube_err)
cube_PAH_err = pcube_err.slice(3.24,3.36,unit='micron')

# Do an initial plot & fit of a single spectrum
# at a pixel with good S/N
# Here I'm fitting two gaussians with 4 parameters each (background offset,
# amplitude, wavelength centroid, linewidth).
# I find that if I let both backgrounds be free parameters, pyspeckit returns
# unrealistic values for both backgrounds, so I fix the 2nd gaussian's background
# level to 0.  The actual command to fix the parameter comes in the fiteach call.
#cube_PAH.plot_spectrum(20,23) # west
#cube_PAH.plot_spectrum(23,39)  # north
#cube_PAH.plot_spectrum(10,10)
cube_PAH.specfit(fittype='vheightgaussian',guesses=[0.,2.0,3.29,0.0387,
         0.,20.,3.29720,0.002],quiet=False,save=False)

# Get ready for the interactive plots that come up after fiteach finishes
cube_PAH.mapplot.makeplane(estimator=np.nansum)

# covers a part of the spectrum that is free of
# spectral lines.  The std variable will be used to estimate the S/N of a line
# during fiteach.
#std = cube_PAH.stats((1.950,1.952))['std']
#print(std)


# other lines in the region
#wlen (obs)  (rest)      flux     uncert.        peak        FWHM   Ion     Multiplet   UpperTerm LowerTerm g1  g2
#3.23499 H2 1–0 O(5) 3.23519 2.24(3) − 17 47.2(6) 3.23465 1.142(13) − 17 37.8(4) 3.23478 2.89(3) − 18 48.4(5)
#3.28180 C iv 1𝑠210𝑖 2I – 1𝑠211ℎ 2H◦ — — — 3.28146 4.0(19) − 19 1.3(6) 3.28158 1.1(5) − 19 1.8(8)
#3.29699 H i 5 – 9 3.29720 5.7(4) − 18 12.1(9) 3.29665 3.24(18) − 18 10.7(6) 3.29677 6.1(4) − 19 10.2(6)
#3.36871 H2 0–0 S(21) 3.36865 3.4(8) − 19 0.71(16) 3.36831 1.1(4) − 19 0.36(12) 3.36830 3.2(8) − 20 0.53(14)
#3.38122 H2 0–0 S(20) 3.38116 2.5(5) − 19 0.53(11) 3.38082 1.1(3) − 19 0.36(11) 3.38081 3.2(12) − 20 0.54(20)
#3.39582 H2 0-0 S(19)

#### Here's where all the fitting happens.
## With the "parlimited" and "parlimits" keywords, I have restricted
## the range for the wavelength centroid and linewidth parameters.
## With the "fixed" keyword, I have held the 2nd gaussian's background level
## to zero, and the "signal_cut" keyword rejects fits for voxels below a
## user-specified S/N threshold.
#4 parameters each (background offset,amplitude, wavelength centroid, linewidth)
#                   0           1                 2                     3
#                   4           5                 6                     7
#errspec=np.ones(cube_PAH.shape)*std
cube_PAH.fiteach(use_nearest_as_guess=False,
                guesses=[0.,2.0,3.29,0.035,
                         0.,8.,3.29720,0.002],
                fittype='vheightgaussian',
                integral=False,
                multicore=4,
                negamp=False,
                verbose_level=2,
                errspec=np.ones(cube_PAH.shape)*std,
                parlimited=[ (True,False), (True,False), (True,True),(True,True),
                            (True,False), (True,False), (True,True),(True,True)],
                parlimits=[(0.0,0.5), (0.1,10.), (3.280,3.305), (0.008,0.12),
                           (0.0,0.5), (0.1,50.), (3.292,3.302), (0.0009,0.005)],
                fixed=[True, False, False, False,True, False, False, False],
                signal_cut=5.,
                start_from_point=(10,10))

# plot the fits as images (you can click on background image to see the spectra + fits)


cube_PAH.mapplot.figure=plt.figure(5)
cube_PAH.mapplot(estimator=3, vmax=0.06, vmin=0.01)
cube_PAH.mapplot.axis.set_title("Line Width")

cube_PAH.mapplot.figure=plt.figure(6)
cube_PAH.mapplot(estimator=2,cmap='bwr',vmin=3.25,vmax=3.33)
cube_PAH.mapplot.axis.set_title("Line Center")

#cube_PAH.mapplot.figure=plt.figure(7)
#cube_PAH.mapplot(estimator=0,vmax=0.7,vmin=-0.001)
#cube_PAH.mapplot.axis.set_title("Background")

cube_PAH.mapplot.figure=plt.figure(8)
amp_med = np.median(cube_PAH.parcube[1,:,:])
cube_PAH.mapplot(estimator=1,vmax=5.0,vmin=0)
cube_PAH.mapplot.axis.set_title("Amplitude")

plt.show()
plt.close()


## Create the images
def gaussian_fwhm(sigma):
    return np.sqrt(8*np.log(2)) * sigma

def gaussian_integral(amplitude, sigma):
    """ Integral of a Gaussian """
    return amplitude * np.sqrt(2*np.pi*sigma**2)


## Create the images
background = cube_PAH.parcube[0,:,:]
PAH_amplitude = cube_PAH.parcube[1,:,:]
PAH_sigma = cube_PAH.parcube[3,:,:]
PAH_linecenter = cube_PAH.parcube[2,:,:]
PAH_fwhm = gaussian_fwhm(PAH_sigma) * 2.355
PAH_image = gaussian_integral(PAH_amplitude, PAH_sigma)


# Write pyspeckit parcube and errcube to file
# 0: background, 1: amplitude, 2: linecenter, 3: sima
pyspeckit_fits_filename = outdir + fname.replace(".fits","pyspeckitfits_PA_cont.fits")
#pyspeckit_fits_filename = outdir + fname.replace(".fits","pyspeckitfits_PA.fits")
cube_PAH.write_fit(pyspeckit_fits_filename,overwrite=True)


# Write pyspeckit parcube and errcube to file
pyspeckit_fits_filename = outdir + fname.replace(".fits","_pyspeckitfits_PAH_cont.fits")
#pyspeckit_fits_filename = outdir + fname.replace(".fits","_pyspeckitfits_PAH.fits")
cube_PAH.write_fit(pyspeckit_fits_filename,overwrite=True)


#print(cube_PAH.header)
cube0_PAH = subcube.spectral_slab(3.25*u.micron,3.33*u.micron)
mom0_PAH = cube0_PAH.moment(order=0, axis=0)
newheader = mom0_PAH.header

del newheader['COMMENT']
print(newheader)

# Write the images to file
#data_nirsp=PAH_image
PAHfilename = outdir + fname.replace(".fits","_PAH_flux_gf_cont.fits")
#PAHfilename = outdir + fname.replace(".fits","_PAH_flux_gf.fits")
fits.writeto(PAHfilename,data=PAH_image,  header = newheader, overwrite=True)

PAHfilename = outdir + fname.replace(".fits","_PAH_lcenter_cont.fits")
#PAHfilename = outdir + fname.replace(".fits","_PAH_lcenter.fits")
fits.writeto(PAHfilename,data=PAH_linecenter,  header = newheader, overwrite=True)

# Write the images to file
PAHfilename = outdir + fname.replace(".fits","_PAH_fwhm_gf_cont.fits")
#PAHfilename = outdir + fname.replace(".fits","_PAH_fwhm_gf.fits")
fits.writeto(PAHfilename,data=PAH_fwhm, header=newheader, overwrite=True)
