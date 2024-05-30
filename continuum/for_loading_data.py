# Imports

from astropy.io import fits
import numpy as np
import pandas
import scipy.io as sio

# Functions

def read_in_fits(directory):
    """
    Turns .fits files into np.arrays.
    
    Parameters
    ----------
    directory [str]: the directory of the .fits file

    Returns
    -------
    np.array : the .fits file's data in a np.array of the same dimensions
    
    """
    return(fits.getdata(directory))

def get_wavelengths(directory_tbl, directory_fits):
    """
    Gets the wavelengths from .tbl file.
    
    Parameters
    ----------
    directory_tbl [str]: the directory of the .tbl file with the wavelengths
    
    directory_fits [str]: the directory of the .fits file for which the wavelengths are used

    Returns
    -------
    np.array : a np.array of the wavelengths
    
    """
    wavelengths = []
    data = pandas.read_table(directory_tbl) 
    length = read_in_fits(directory_fits).shape[0] # finds number of wavelengths
    data = data.tail(length) # wavelegths start to appear towards the end of the file
    data.rename(columns = {data.columns[0]:'wavelength'}, inplace = True) # current name has apostrophies
    for wavelength in range(length):
        wavelengths.append(float(data.wavelength.iloc[wavelength].strip().split(" ")[0]))
    return(np.array(wavelengths))

def get_data(file):
    """
    Takes .xdr files and turns them into np.arrays.
    
    Parameters
    ----------
    file [str]: the directory of the .xdr file

    Returns
    -------
    np.array : the .xdr file's data in a np.array 
    
    """
    data = sio.readsav(file, idict=None, python_dict=False, uncompressed_file_name=None, verbose=False)
    data = np.array(list(data.values())[0])
    return(data)

def read_dat(directory_dat):
    """
    Turns the wavelength and flux columns of a dat file into np.arrays.
    
    Parameters
    ----------
    directory_dat [str]: the directory of the .dat file (normally used for scaling the 12.7 feature)

    Returns
    -------
    wavelengths [np.array]: the array of wavelengths
    
    brightnesses [np.array]: the array of surface brightnesses
    
    """
    wavelengths = np.array([])
    brightnesses = np.array([])
    with open(directory_dat) as file:
        for line in file:
             wavelengths = np.append(wavelengths, float(line.split()[0]))
             brightnesses = np.append(brightnesses, float(line.split()[1]))
    return(wavelengths, brightnesses)

def get_wave_fits(file):
    """
    Takes .fits files and turns them into np.arrays.
    
    Parameters
    ----------
    file [str]: the directory of the .fits file
    Returns
    -------
    np.array : the .fits file's data in a np.array 
    
    """
    data = fits.open(file)
    data = np.array(data[1].data[0])[0]
    data = np.squeeze(data)
    data = np.float64(data)
    return(data)
