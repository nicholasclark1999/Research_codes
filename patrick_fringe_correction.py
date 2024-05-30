#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:12:11 2023

@author: nclark
"""

'''
Purpose of this notebook
As is common in IR detectors, effects in the MIRI MRS detectors results in the input signal being modulated as a function of wavelength, 
so-called 'fringing'. Two fringe components have been identified in MRS data, the primary fringe originating in the detector substrate
 (see Argyriou et. al, 2020 A&A, 641, 150) and a second high frequency, low amplitude fringe in channels 3 and 4, thought to originate in the MRS dichroics.

The JWST calibration pipeline contains two steps to remove fringes from MRS data. These are the fringe step in Spec2Pipeline and the 
residual_fringe step which should be run before cube building at the level 3 stage. The fringe step divides the detector level data
 by a static fringe flat derived from an extended source for the primary fringe component only. However, this can leave residuals 
 for either spatially unresolved sources or a source with spatial structure. These residuals are corrected for by the residual_fringe 
 step on detector level data, which also attemps to correct for the dichroic fringe in channels 3 and 4. However, this step is time
 consuming and a user may want a faster indication of what improvements may be achieved.

This notebook is intended to demonstrate how reisdual fringes are removed from spectrum level products, i.e. the output of the 
'extract_1d' step. Note that the accompanying rfc1d_utils.py file is expected to be in the same folder as this notebook.

Here only one file is processed and some additional plots are shown to illustrate how the residuals are identified and removed.
'''

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))

import logging, sys
logging.disable(sys.maxsize)

# basic imports
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# import the jwst datamodels
from jwst import datamodels

# import utils that accompany this notebook
import rfc1d_utils

# set the jwst pipeline crds context (leave blank to use latest)
os.environ["CRDS_CONTEXT"] = ""

'''
%matplotlib notebook
'''

'''
Read and display the test spectrum
'''

# set filename
test_file = 'ring_neb_west_spectra_23.1.0/ring_neb_west_ch3-long_src.txt'

# read the file
wavelength, flux, sb = np.loadtxt(test_file, unpack=True)

# set the channel
channel = 3

# get weights and wavenumber
weights = flux / np.median(flux)
weights[weights == np.inf] = 0
weights[np.isnan(weights)] = 0
wavenum = 10000.0 / wavelength

# plot the spectrum
fig, axs = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
axs.step(wavelength[20:-20], flux[20:-20], c='b', linestyle='-', linewidth=1, label='flux')
axs.set_xlabel('Wavelength (micron)', fontsize=14)
axs.set_ylabel('Flux unit', fontsize=14)
axs.legend(fontsize=10)
plt.tight_layout(h_pad=0)
plt.show()

# perform the correction
corrected_sb = rfc1d_utils.fit_residual_fringes(sb, weights, wavenum, int(channel), plots=False)
corrected_flux = rfc1d_utils.fit_residual_fringes(flux, weights, wavenum, int(channel), plots=False)

np.savetxt(test_file, np.array([wavelength, corrected_flux, corrected_sb]).T, header='Wavelength  flux_Jy  surfbright_MJy/sr')

