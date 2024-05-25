#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:42:21 2024

@author: nclark
"""

'''
IMPORTING MODULES
'''

#standard stuff
import numpy as np



'''
LINE FINDING FUNCTION
'''

def emission_line_remover(data, wavelengths, width, limit):
    '''
    A function that finds emission lines in data, using an inputted 
    maximum width and height defintion. It assumes that emission lines are areas
    where the flux density rises by a given amount (limit) above the surrounding continuum
    a fixed number of indices away in either direction (width).
    
    Parameters
    ----------
    data
        TYPE: 1d array of floats
        DESCRIPTION: a spectra.
    wavelengths
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array corresponding to spectra.
    width
        TYPE: integer
        DESCRIPTION: max index width of spectral peaks.
    limit
        TYPE: float
        DESCRIPTION: defines the min flux dif between an emission line and continuum.

    Returns
    -------
    line_indices 
        TYPE: 1d array of indices
        DESCRIPTION: array containing the index of each line found, corresponding to the peak flux density.
    lines
        TYPE: 1d array of floats
        DESCRIPTION: array containing the wavelength of each line found, correspondint to the peak flux density.
    '''
    
    #list to store indices that correspond to a line
    line_indices = []
    
    #list to store the lines
    lines = []
    
    i = 0
    
    #left edge case
    while i < width:
        if data[i] - data[i+width] > limit:
            line_index = np.argmax(data[i:i+width]) + i - width
            
            line_indices.append(line_index)
            lines.append(wavelengths[line_index])
            
            #move index outside of this line
            i = i+width
        i+=1
        
    #central regions away from the edges
    while i < len(data) - 1 - width:
        if data[i] - data[i-width] > limit and data[i] - data[i+width] > limit:
            line_index = np.argmax(data[i-width:i+width]) + i - width
            
            line_indices.append(line_index)
            lines.append(wavelengths[line_index])
            print(i-width, line_index, i+width)
                
            #move index outside of this line
            i = i+width
        i+=1

    #right edge case
    while i < len(data):
        if data[i] - data[i-width] > limit:
            line_index = np.argmax(data[i-width:i]) + i - width
                
            line_indices.append(line_index)
            lines.append(wavelengths[line_index])
                
            #move index outside of this line
            i = i+width
        i+=1
    
    #converting lists to arrays
    line_indices = np.array(line_indices)
    lines = np.array(lines)
    
    return line_indices, lines

