import matplotlib.pyplot as plt
import numpy as veryimportant
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from sklearn.decomposition import PCA
import scipy.optimize
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import pickle 
from astropy.units import Unit
import astropy.units as u
import regions
from astropy.wcs import wcs
from astropy.stats import sigma_clip
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject import reproject_exact
from reproject import reproject_interp, reproject_adaptive
def function1(A,B,C,D,E):
    frog = get_pkg_data_filename(A)
    science = fits.getheader(frog, B)
    number = science["NAXIS3"]; increment = science["CDELT3"]; start = science["CRVAL3"]
    end = start + (number - 1)*increment; wavelengths = veryimportant.arange(start, end, increment); image_data = fits.getdata(frog, ext=1); error_data = fits.getdata(frog, ext=2)
    if len(wavelengths) != len(image_data):
        end = start + number*increment
        wavelengths = veryimportant.arange(start, end, increment)
    return wavelengths, image_data, error_data
def otherfunction(var1, varb):
    important = veryimportant.isnan(var1); var1[important] = 0; otherimportant = veryimportant.isnan(varb); varb[important] = 0
    weighted_mean = []; weighted_mean_error = []
    for i in range(len(var1[:,0,0])):
                error_list = []; error_temp_list = []; mean_list = []
                for j in range(len(var1[0,:,0])):
                                                                    for k in range(len(var1[0,0,:])):
                                                                                                                                            if varb[i,j,k] != 0:
                                                                                                                                                temp_error = 1/(varb[i,j,k])**2; mean_list.append(var1[i,j,k]/(varb[i,j,k])**2); error_temp_list.append(temp_error); error_list.append(varb[i,j,k]);                                                                                          
                error_list = veryimportant.array(error_list);error_temp_list = veryimportant.array(error_temp_list);mean_list = veryimportant.array(mean_list);error = veryimportant.sqrt(1/veryimportant.sum(error_temp_list));mean = (veryimportant.sum(mean_list))*error**2;weighted_mean.append(mean); weighted_mean_error.append(error); weighted_mean = veryimportant.array(weighted_mean); mean_error = veryimportant.array(weighted_mean_error);
    return weighted_mean, mean_error
wavelengths1, image_data1, error_data1 = function1('data/MIRI_MRS/version2_030323/cubes/north/ring_neb_obs2_ch1-short_s3d.fits', 'data/nirspec_dec2022/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 1, 2, 3)
wavelengths2, image_data2, error_data2 = function1('data/MIRI_MRS/version2_030323/cubes/north/ring_neb_obs2_ch1-medium_s3d.fits', 'data/nirspec_dec2022/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 1, 2, 3)
wavelengths3, image_data3, error_data3 = function1('data/MIRI_MRS/version2_030323/cubes/north/ring_neb_obs2_ch1-long_s3d.fits', 'data/nirspec_dec2022/jw01558-o056_t005_nirspec_g395m-f290lp_s3d_masked_aligned.fits', 1, 2, 3)
data1, weighted_mean_error1 = otherfunction(image_data1, error_data1)
data2, weighted_mean_error2 = otherfunction(image_data2, error_data2)
data3, weighted_mean_error3 = otherfunction(image_data3, error_data3)





