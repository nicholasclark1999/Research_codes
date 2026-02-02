#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:18:53 2024

@author: nclark
"""

import numpy as np

def cube_stitching_no_offset(wave_a, wave_b, data_a, data_b):
    '''
    This function takes in 2 adjacent wavelength and image data arrays, presumably 
    from the same part of the image fov (field of view), so they correspond to 
    the same location in the sky. It then finds which indices in the lower wavelength 
    data overlap with the beginning of the higher wavelength data, and combines 
    the 2 data arrays in the middle of this region. For this function, no offset is done.
    
    It needs to work with arrays that may have different intervals, so it is split into 2
    to take a longer running but more careful approach if needed.
    
    Note that in the latter case, the joining is not perfect, and currently a gap
    of ~0.005 microns is present; this is much better than the previous gap of ~0.05 microns,
    as 0.005 microns corresponds to 2-3 indices.
    
    Parameters
    ----------
    wave_a
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the smaller wavelengths.
        wave_b
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the larger wavelengths.
    data_a
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_a.
            for [k,i,j] k is wavelength index, i and j are position index.
    data_b
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_b.
            for [k,i,j] k is wavelength index, i and j are position index.
            
    Returns
    -------
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_a and data_b joined together as described above.
            for [k,i,j] k is wavelength index, i and j are position index.
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined together as described above.
    overlap
        TYPE: integer (index) OR tuple (index)
        DESCRIPTION: index of the wavelength value in wave_a that equals the first element in wave_b. In the 
        case of the two wavelength arrays having different intervals, overlap is instead a tuple of the regular
        overlap, followed by the starting index in the 2nd array.
        
        UPDATE: now returns lower_index and upper_index, as opposed to the indices where wave_a first meets wave_b,
        i.e. it now returns the indices where the connection happens.
        
    '''
    
    #check if wavelength interval is the same or different
    check_a = np.round(wave_a[1] - wave_b[0], 4)
    check_b = np.round(wave_b[1] - wave_b[0], 4)
    
    if check_a == check_b:
    
        #check where the overlap is
        overlap = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many entries are overlapped, subtract 1 for index
        overlap_length = len(wave_a) -1 - overlap
        
        #making a temp array to scale
        temp = np.copy(data_b)
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
        
        split_index = overlap_length/2
        
        #check if even or odd, do different things depending on which
        if overlap_length%2 == 0: #even
            lower_index = overlap + split_index
            upper_index = split_index

        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap + split_index + 0.5
            upper_index = split_index - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        
    else:
        #check where the overlap is, only works for wave_a
        overlap_a = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many microns the overlap is
        overlap_micron = wave_a[-1] - wave_a[overlap_a]
        
        #find how many entries of wave_a are overlapped, subtract 1 for index
        overlap_length_a = len(wave_a) -1 - overlap_a
        split_index_a = overlap_length_a/2
        
        #number of indices in wave_B over the wavelength range
        overlap_length_b = int(overlap_micron/check_b)
        split_index_b = overlap_length_b/2
        
        #making a temp array to scale
        temp = np.copy(data_b)
        
        #check if even or odd, do different things depending on which
        if overlap_length_a%2 == 0: #even
            lower_index = overlap_a + split_index_a
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap_a + split_index_a + 0.5
            
        if overlap_length_b%2 == 0: #even
            upper_index = split_index_b
        else: #odd, so split_index is a number of the form int+0.5
            upper_index = split_index_b - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)


        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        overlap = (lower_index, upper_index)
    
    return image_data, wavelengths, overlap


def cube_stitching_offset(wave_a, wave_b, data_a, data_b):
    '''
    This function takes in 2 adjacent wavelength and image data arrays, presumably 
    from the same part of the image fov (field of view), so they correspond to 
    the same location in the sky. It then finds which indices in the lower wavelength 
    data overlap with the beginning of the higher wavelength data, and combines 
    the 2 data arrays in the middle of this region. For this function, an offset is applied.
    
    It needs to work with arrays that may have different intervals, so it is split into 2
    to take a longer running but more careful approach if needed.
    
    Note that in the latter case, the joining is not perfect, and currently a gap
    of ~0.005 microns is present; this is much better than the previous gap of ~0.05 microns,
    as 0.005 microns corresponds to 2-3 indices.
    
    Parameters
    ----------
    wave_a
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the smaller wavelengths.
        wave_b
        TYPE: 1d array of floats
        DESCRIPTION: wavelength array in microns, contains the larger wavelengths.
    data_a
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_a.
            for [k,i,j] k is wavelength index, i and j are position index.
    data_b
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data corresponding to wave_b.
            for [k,i,j] k is wavelength index, i and j are position index.
            
    Returns
    -------
    image_data
        TYPE: 3d array of floats
        DESCRIPTION: position and spectral data, data_a and data_b joined together as described above.
            for [k,i,j] k is wavelength index, i and j are position index.
    wavelengths
        TYPE: 1d numpy array of floats
        DESCRIPTION: the wavelength array in microns, data_a and data_b joined together as described above.
    overlap
        TYPE: integer (index) OR tuple (index)
        DESCRIPTION: index of the wavelength value in wave_a that equals the first element in wave_b. In the 
        case of the two wavelength arrays having different intervals, overlap is instead a tuple of the regular
        overlap, followed by the starting index in the 2nd array.
        
        UPDATE: now returns lower_index and upper_index, as opposed to the indices where wave_a first meets wave_b,
        i.e. it now returns the indices where the connection happens.
        
    '''
    
    #check if wavelength interval is the same or different
    check_a = np.round(wave_a[1] - wave_b[0], 4)
    check_b = np.round(wave_b[1] - wave_b[0], 4)
    
    if check_a == check_b:
    
        #check where the overlap is
        overlap = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many entries are overlapped, subtract 1 for index
        overlap_length = len(wave_a) -1 - overlap
        
        #making a temp array to scale
        temp = np.copy(data_b)
                
        #combine arrays such that the first half of one is used, and the second half
        #of the other is used. This way data at the end of the wavelength range is avoided
        
        split_index = overlap_length/2
        
        #check if even or odd, do different things depending on which
        if overlap_length%2 == 0: #even
            lower_index = overlap + split_index
            upper_index = split_index
            
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap + split_index + 0.5
            upper_index = split_index - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        
    else:
        #check where the overlap is, only works for wave_a
        overlap_a = np.where(np.round(wave_a, 2) == np.round(wave_b[0], 2))[0][0]
        
        #find how many microns the overlap is
        overlap_micron = wave_a[-1] - wave_a[overlap_a]
        
        #find how many entries of wave_a are overlapped, subtract 1 for index
        overlap_length_a = len(wave_a) -1 - overlap_a
        split_index_a = overlap_length_a/2
        
        #number of indices in wave_B over the wavelength range
        overlap_length_b = int(overlap_micron/check_b)
        split_index_b = overlap_length_b/2
        
        #making a temp array to scale
        temp = np.copy(data_b)
        
        #check if even or odd, do different things depending on which
        if overlap_length_a%2 == 0: #even
            lower_index = overlap_a + split_index_a
        else: #odd, so split_index is a number of the form int+0.5
            lower_index = overlap_a + split_index_a + 0.5
            
        if overlap_length_b%2 == 0: #even
            upper_index = split_index_b
        else: #odd, so split_index is a number of the form int+0.5
            upper_index = split_index_b - 0.5
        
        #make sure they are integers
        lower_index = int(lower_index)
        upper_index = int(upper_index)
        
        offset1 = np.mean(data_a[lower_index:])
        offset2 = np.mean(temp[:upper_index])
        
        offset = offset1 - offset2
        
        temp = temp + offset
        
        image_data = np.concatenate((data_a[:lower_index], temp[upper_index:]), axis=0)
        wavelengths = np.hstack((wave_a[:lower_index], wave_b[upper_index:]))
        overlap = (lower_index, upper_index)
    
    return image_data, wavelengths, overlap

