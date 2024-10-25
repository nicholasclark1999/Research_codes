"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Authors: Ameek Sidhu & Ryan Chown

  Example 1:

    import convolver as cvr

    psf_in = cvr.PSF(psf_in_arr, pixsize_psf_in)
    psf_out = cvr.PSF(psf_out_arr, pixsize_psf_out)
    image_in = cvr.Image(image_in_arr, pixsize_image_in)
    # Provide Convolver with 2 PSFs
    c = cvr.Convolver(image_in, psf_in=psf_in, psf_out=psf_out)
    c.prepare_image()
    # Make a kernel
    c.create_kernel() # this will accept any optional arguments of helpers.create_kernel (e.g. low pass filter settings)
    c.prepare_kernel()
    c.do_the_convolution()


  Example 2:

    psf_in = cvr.PSF(psf_in_arr, pixsize_psf_in)
    psf_out = cvr.PSF(psf_out_arr, pixsize_psf_out)
    kernel = psf_in.to_kernel(psf_out, recenter=True)
    image_in = cvr.Image(image_in_arr, pixsize_image_in)

    # provide Convolver with a kernel
    c = cvr.Convolver(image_in, kernel=kernel)
    c.prepare_image()
    c.prepare_kernel()
    c.do_the_convolution()

  Example 3 (more concrete, note the PSFs and image were just randomly chosen for illustration):

    import convolver as cvr
    import matplotlib.pyplot as pl
    from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel
    from astropy.utils.data import get_pkg_data_filename

    filename = get_pkg_data_filename('galactic_center/gc_msx_e.fits')
    hdu = fits.open(filename)[0]
    img = hdu.data[50:90, 60:100] * 1e5

    psf_in_arr = Gaussian2DKernel(10, x_size=100).array
    psf_out_arr = Gaussian2DKernel(20, x_size=200).array
    psf_in = cvr.PSF(psf_in_arr, 1.3)
    psf_out = cvr.PSF(psf_out_arr, 2.5)
    image_in_arr = img
    image_in = cvr.Image(image_in_arr, 1.)
    # Provide Convolver with 2 PSFs
    c = cvr.Convolver(image_in, psf_in=psf_in, psf_out=psf_out)
    c.prepare_image()
    # Make a kernel
    c.create_kernel() # this will accept any optional arguments of helpers.create_kernel (e.g. low pass filter settings)
    c.prepare_kernel()
    c.do_the_convolution()



  Another example:

    c = cvr.Convolver()
    c.check_if_ready_for_convolution()

    # Prints:
    # Kernel is not ready...
    # and we can't make one (PSFs have not been homogenized)
    # Not ready for convolution (image_prepped=False and kernel_prepped=False)

  One more example:

    c = cvr.Convolver(image_prepped=True, kernel_prepped=True)
    c.check_if_ready_for_convolution()

# Prints:
# Kernel is ready
# Ready for convolution


psf_in = conv.PSF(psf_in_arr, pixsize_psf_in)
psf_out = conv.PSF(psf_out_arr, pixsize_psf_out)
image_in = conv.Image(image_in_arr, pixsize_image_in)
c = conv.Convolver(psf_in=psf_in, psf_out=psf_out, image_in=image_in)

"""

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


class Image:
    """Image (e.g. a monochromatic slice of an IFU cube, or a broadband image)

    prepare calls the prep_image function, which gets the image ready for convolution


    Attributes:
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


class PSF:
    """Point spread function

    

    Attributes:
        arr (np.ndarray) : The PSF array
        pixsize_arcsec (float) #intended pixelsize of psf in arcseconds
        fwhm_arcsec_x (float)  x fwhm of the 2d gaussian approximation of PSF
        fwhm_arcsec_y (float) y fwhm of the 2d gaussian approximation of PSF
        theta (float): position angle of the PSF (increases counterclockwise)
        gauss2d_fit (float): an output of the 2d gaussian function
        regul (float): regularization parameter for kernel generation
        method (string): method of noise supression for kernel generation
    """
    def __init__(self,
                 arr,
                 pixsize_arcsec,
                 fwhm_arcsec_x=None,
                 fwhm_arcsec_y=None,
                 theta=None,
                 gauss2d_fit=None,
                 regul=None,
                 method=None):
        self.arr = arr
        self.pixsize_arcsec = pixsize_arcsec
        self.fwhm_arcsec_x = fwhm_arcsec_x
        self.fwhm_arcsec_y = fwhm_arcsec_y
        self.theta = theta
        self.gauss2d_fit = gauss2d_fit
        self.regul = regul
        self.method = method
        
    
    #resizes psf
    def resample(self, new_pixsize_arcsec):
        self.arr = photutils.psf.matching.resize_psf(self.arr,
                                                     self.pixsize_arcsec,
                                                     new_pixsize_arcsec,
                                                     order=1)
        self.pixsize_arcsec = new_pixsize_arcsec
        # old_size = self.arr.shape[0]
        # new_size = old_size * self.pixsize_arcsec / new_pixsize_arcsec
        # new_size = int(np.ceil(new_size))
        # if (old_size - new_size) % 2 == 1:
        #     new_size += 1
        # ratio = new_size / old_size
        # self.arr = scipy.ndimage.zoom(self.arr, ratio, order=1) / ratio**2
        return self
    
    #calls the recenter helper function
    def recenter(self):
        self.arr = helpers.center_psf(self.arr)
        return self

    def compare_fwhm(self, psf_out):
        """Compares the FWHM of this PSF with another one"""
        # Check if Gaussian fits have been done.
        # If they haven't been done, do them.
        if not self.gauss2d_fit:
            self.fit_gauss2d()
        if not psf_out.gauss2d_fit:
            psf_out.fit_gauss2d()

        # Compare
        result = 0
        #print(self.fwhm_arcsec_x, psf_out.fwhm_arcsec_x)
        #print(self.fwhm_arcsec_y, psf_out.fwhm_arcsec_y)
        if self.fwhm_arcsec_x > psf_out.fwhm_arcsec_x:
            print("WARNING: input PSF FWHM_x > output PSF FWHM_x")
            result += 1
        if self.fwhm_arcsec_y > psf_out.fwhm_arcsec_y:
            print("WARNING: input PSF FWHM_y > output PSF FWHM_y")
            result += 1
        return result

    def fit_gauss2d(self):
        """Fits PSF with a 2D Gaussian"""


        # Make a Gaussian 2D model with a rough initial guess at parameters
        g2d = models.Gaussian2D(amplitude=1,
                                x_mean=self.arr.shape[1] / 2,
                                y_mean=self.arr.shape[0] / 2,
                                x_stddev=self.arr.shape[1] / 4,
                                y_stddev=self.arr.shape[0] / 4)
        fitter = fitting.LevMarLSQFitter()

        x, y = np.mgrid[:self.arr.shape[0], :self.arr.shape[1]]

        gauss2d_fit = fitter(g2d, x, y, self.arr)

        self.gauss2d_fit = gauss2d_fit

        self.fwhm_arcsec_x = gauss2d_fit.x_stddev * gaussian_sigma_to_fwhm * self.pixsize_arcsec
        self.fwhm_arcsec_y = gauss2d_fit.y_stddev * gaussian_sigma_to_fwhm * self.pixsize_arcsec
        self.theta = gauss2d_fit.theta

        #print("FWHM = (" + str(self.fwhm_arcsec_x) + ", " +
         #     str(self.fwhm_arcsec_y) + ")")
        #print("theta = " + str(self.theta))

        return gauss2d_fit
    
    #pads psf to match shape of another psf
    def pad_to(self, other_psf):
        out_shape = other_psf.arr.shape

        pixels_added = out_shape[0] - self.arr.shape[0]
        new_size = pixels_added + self.arr.shape[0]

        new_array = np.zeros((new_size, new_size))
        new_array[:self.arr.shape[0], :self.arr.shape[1]] = self.arr
        self.arr = new_array
        self = self.recenter()
        return self

    def to_kernel(self, psf_out, recenter=True, **kwargs):
        """Makes a kernel from this PSF and another PSF"""
        
        #this line is giving errors in a particular wavelength (314 of ch1a i think)
        #update: now it gives errors in channel 2B
        #because the gaussian fit seems to suck and has infs or something, so disabling
        '''
        # How close are their FWHMs?
        comparison_flag = self.compare_fwhm(psf_out)
        if comparison_flag >= 1:
            print("Input PSF is broader than output PSF")
            print("Returning 0")
            return 0
        '''
        # Do their pixel scales match?
        if self.pixsize_arcsec < psf_out.pixsize_arcsec:
            psf_out.resample(self.pixsize_arcsec)
            recenter = True
        
        if self.pixsize_arcsec > psf_out.pixsize_arcsec:
            self.resample(psf_out.pixsize_arcsec)
            recenter = True
        
        # Do their shapes match?
        nx_in = self.arr.shape[0]
        nx_out = psf_out.arr.shape[0]
        if nx_in < nx_out:
            self = self.pad_to(psf_out)
            recenter = True
        if nx_in > nx_out:
            psf_out = psf_out.pad_to(self)
            recenter = True

        # Recenter last
        if recenter:
            psf_out = psf_out.recenter()
            self = self.recenter()

        kernel = helpers.create_kernel(self.arr, psf_out.arr, self.regul, self.method, **kwargs)
        return Kernel(kernel, self.pixsize_arcsec)


class Kernel:
    """Class for a kernel
    
    note a kernel is the ratio of the fourier transformed psfs


    Attributes:
        arr (np.ndarray) : The image array
        pixsize_arcsec (float)
        prepared (bool) : Has the kernel been prepared to match an image?
    """
    def __init__(self, arr, pixsize_arcsec, prepared=False):
        self.arr = arr
        self.pixsize_arcsec = pixsize_arcsec
        self.prepared = prepared
        
        #calls the prep_kernel helper function
    def prepare(self, image_pixsize_arcsec):
        if not self.prepared:
            self.arr = helpers.prep_kernel(self.arr, self.pixsize_arcsec,
                                           image_pixsize_arcsec)
            self.pixsize_arcsec = image_pixsize_arcsec
            self.prepared = True


class Convolver:
    # https://google.github.io/styleguide/pyguide.html#384-classes
    """Interface for convolution.

    Takes as input PSFs, a kernel, and input image.
    Can compute kernel from PSFs, and perform the convolution,
    and performs checks at each step.
    
    Note: a convolution is when one function is given as an input into a second function, for example f(g(x)) is a convolution of the functions f and g

    Attributes:
        enable_printout: Boolean flag for if some 'this is done' prints should be done 
        psf_in: An instance of PSF (see class written above) corresponding to image_in
        psf_out: An instance of PSF (see class written above) corresponding to image_out
        kernel: An instance of Kernel
        image_in: An instance of Image
        image_out: An instance of Image
        image_prepped: Boolean flag which is True if image_in has been prepared for convolution
        kernel_prepped: Boolean flag which is True if kernel has been prepared for convolution
        ready_for_convolution: Boolean flag which is True if all required checks have passed and we are ready to call do_the_convolution
    """

    # def __init__(self, filename='', wave_source=None, wave_target=None, psf_in=None, psf_out=None, kernel=None, image_in=None, image_out=None):
    def __init__(self,
                 image_in,
                 enable_printout=None,
                 psf_in=None,
                 psf_out=None,
                 kernel=None,
                 image_out=None):
        """Inits Convolver. Required arguments are image_in (Image) plus either:
            a) psf_in (PSF) and psf_out (PSF), or
            b) a kernel (Kernel)
        """

        if not (type(image_in) == Image):
            print("image_in must be an Image")
        if any((not isinstance(psf, PSF)) for psf in [psf_in, psf_out]):
            print("psf_in and psf_out must be PSF objects")

        # self.filename = filename
        # self.wave_source = wave_source
        # self.wave_target = wave_target
        self.enable_printout = enable_printout
        self.psf_in = psf_in
        self.psf_out = psf_out
        self.kernel = kernel
        self.image_in = image_in
        self.image_out = image_out

        self.image_prepped = image_in.prepared

        self.kernel_made = kernel is not None

        self.kernel_prepped = False
        if self.kernel_made:
            if not (type(kernel) == Kernel):
                print("kernel must be a Kernel")
            else:
                self.kernel_prepped = kernel.prepared

        # Note (RC): left out option to initialize Convolver with ready_for_convolution = True
        # Did this intentionally to avoid issues e.g. if someone says it's ready but all the pieces aren't actually there.
        self.ready_for_convolution = False
    
    def create_kernel(self, recenter=True, **kwargs):
        self.kernel = self.psf_in.to_kernel(self.psf_out,
                                            recenter=recenter,
                                            **kwargs)
        self.kernel_made = True #nicholas added this

    def prepare_kernel(self):
        self.kernel.prepare(self.image_in.pixsize_arcsec)
        self.kernel_prepped = True

    def prepare_image(self, **kwargs):
        self.image_in.prepare(**kwargs)
        self.image_prepped = True

    def check_if_ready_for_convolution(self):
        """Checks if all ingredients are is ready to do a convolution"""

        if self.ready_for_convolution:
            # No checks needed.
            if self.enable_printout == True:
                print("Ready for convolution.")
        else:
            ready = True
            # Has a kernel been made?
            if not self.kernel_made:
                print("Kernel needs to be provided or made")
                ready = False
            else:
                if self.kernel_prepped:
                    if self.enable_printout == True:
                        print("Kernel has been provided or made and is ready")
                else:
                    print(
                        "Kernel has been provided or made but needs to be prepared"
                    )
                    ready = False

            # Is the image ready?
            if not self.image_prepped:
                print("Image needs to be prepared")
                ready = False
            else:
                if self.kernel_prepped:
                    if self.enable_printout == True:
                        print("Ready for convolution")
                    self.ready_for_convolution = True

    def do_the_convolution(self):
        """Performs the convolution.
        """
        
        self.check_if_ready_for_convolution()

        if self.ready_for_convolution:
            kernel = self.kernel.arr
            image = self.image_in.arr
            padding = self.image_in.padding
            result_image = helpers.convolve_fft(image, kernel, allow_huge=True)

            # remove padding
            ind = np.where(padding == 1)
            xlim_start = ind[0][0]
            xlim_stop = ind[0][-1] + 1
            ylim_start = ind[1][0]
            ylim_stop = ind[1][-1] + 1
            convolved_image = result_image[xlim_start:xlim_stop,
                                           ylim_start:ylim_stop]
            self.image_out = Image(convolved_image, self.kernel.pixsize_arcsec)

            return self.image_out
        else:
            return None

    def run(self):
        """
            Does as many steps as possible.

        """
        self.check_if_ready_for_convolution()

        if ~self.psfs_matched:
            self.match_psf_centers_and_pixel()

        if ~self.image_prepped:
            self.prepare_image()

        if ~self.kernel_prepped:
            self.prepare_kernel()

        if self.ready_for_convolution:
            self.do_the_convolution()




#2d gaussian breaks at index 314 of channel 1A, i.e. wavelength 5.1508