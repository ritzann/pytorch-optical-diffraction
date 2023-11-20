import numpy as np
import torch
from scipy import constants as const


"""
Copyright (c) 2023, Ritz Aguilar
All rights reserved.

v1.0 With both CPU and GPU support

References:
1. J. Goodman (2005). Introduction to Fourier Optics.
2. J. Schmidt (2010). Numerical Simulation of Optical Wave Propagation With examples in MATLAB.
3. D. Voelz (2011). Computational Fourier Optics: A MATLAB Tutorial.
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Propagate():
    def __init__(self,
                 device="cuda",
                 pixel_number=None, 
                 pixel_size=None,
                 photon_energy = None,
                 photon_energy_bandwidth = 0.,
                 photon_energy_bandwidth_sample_width = 0.):
        
        self._category = ""
        
        # Express the field as cos(phi) + i*sin(phi)
        self.realPart = []
        self.imagPart = []

        # Express the field as A * exp(i*phi)
        self.amplitude = []
        self.phase = []

        self._pixel_number = None
        self._pixel_size = None

        self._photon_energy=photon_energy
        self._photon_energy_bandwidth = photon_energy_bandwidth
        self._photon_energy_bandwidth_sample_width = photon_energy_bandwidth_sample_width

        if pixel_number != None:
            self.set_pixel_number(pixel_number)
        if pixel_size != None:
            self.set_pixel_size(pixel_size)
            
        self.device = device
        print(self.device)
        # self.device = "cpu"
    
    def rs_spectral(self,
                    field_in=None,
                    position=None,
                    pixel_number=None, 
                    pixel_size=None,
                    photon_energy=None,
                    has_evanescent=False,
                    reset=True,
                    pad=0,
                    backend=torch):
        """
        Implements Rayleigh-Sommerfeld (RS) convolution using the Angular Spectrum Method (ASM).
        Provides the most accurate optical wave propagation results as it does NOT make use of 
        paraxial approximation.
        
        Has two modes for (1) calculating both traveling and evanescent waves, or
        (2) ignoring evanescent waves, i.e. only takes into account traveling waves.

        NOTE: Currently the Spectral Propagator in TK's code!

        Parameters
        ----------
        field_in : input field (source plane)
        position : propagation distance (z)
        photon_energy : photon energy in keV
        has_evanescent : evanescent mode (boolean); default value = False
        
        
        Returns
        -------
        field_out : output field (observation plane)
        """
        
        z = position
        # old_pix_size = self._pixel_size # input plane grid spacing (spatial domain)
        # old_pix_number = self._pixel_number # pixel number (spatial domain)
        old_pix_size = pixel_size # input plane grid spacing (spatial domain)
        old_pix_number = pixel_number # pixel number (spatial domain)
        Lambda = self.get_lambda(photon_energy = photon_energy) # field wavelength
        # print(Lambda)
        k = self.get_k(photon_energy = photon_energy) # wave number = 2*pi/Lambda or other fixed value
        # print(k)
                
        if photon_energy is None:
            photon_energy = self.get_photon_energy()
        
        if field_in is None:
            field_in = self._realPart + 1j*self._imagPart
        
        if pad != 0: # pad the input
            if backend == np:
                field_in = np.pad(field_in, ((pad, pad), (pad, pad)))
                old_pix_number = field_in.shape[0]
            elif backend == torch:
                if type(field_in)==np.ndarray:
                    field_in = torch.from_numpy(field_in).to(self.device)
                field_in = torch.nn.functional.pad(field_in, (pad, pad, pad, pad)).to(self.device)
                old_pix_number = field_in.cpu().shape[0]

        fx = backend.fft.fftshift(backend.fft.fftfreq(old_pix_number, d=old_pix_size))
        if backend == torch:
            # Generate grid in the Fourier domain
            fx = fx.to(self.device)

        fx, fy = backend.meshgrid(fx, fx)

        # Perform angular spectrum method (ASM) to propagate the field
        # define transfer function H of the ASM
        sqrt_arg = 1 - (Lambda*fx)**2 - (Lambda*fy)**2 # square root argument of the transfer function
        tmp = k*backend.sqrt(backend.abs(sqrt_arg))
        if has_evanescent:
            # calculate the propagating and the evanescent (complex) modes
            kz = backend.where(sqrt_arg >= 0, tmp, 1j*tmp) 
        else:
            # without evanescent waves
            kz = backend.where(sqrt_arg >= 0, tmp, -tmp) 
        

        fft_field_in = backend.fft.fftshift(backend.fft.fft2(field_in))
        if backend == np:
            H = backend.exp(1j*kz*z,dtype='complex64') # transfer function
            # propagate the angular spectrum a distance z
            field_out = backend.fft.ifft2(backend.fft.ifftshift(fft_field_in * H)).astype('complex64')
        elif backend == torch:
            H = backend.exp(1j*kz*z).to(self.device) # transfer function
            # propagate the angular spectrum a distance z
            field_out = backend.fft.ifft2(backend.fft.ifftshift(fft_field_in * H))

        if reset:
            self._realPart = field_out.real
            self._imagPart = field_out.imag

        # return field_out
        return field_out.real + 1j*field_out.imag

    
    def fresnel_spectral(self,
                         field_in=None,
                         position=None,
                         pixel_number=None, 
                         pixel_size=None,
                         photon_energy=None,
                         pad=0,
                         backend=torch,
                         reset=True):
        """
        Implements Fresnel propagation using the Angular Spectrum Method (ASM).
        Unlike in the RS version, it makes use of paraxial approximation.

        Parameters
        ----------
        field_in : input field (source plane)
        position : propagation distance (z)
        photon_energy : photon energy in keV
        
        
        Returns
        -------
        field_out : output field (observation plane)
        """
        
        if self.device == "cpu":
            backend = np # numpy
        else:  # gpu
            backend = torch
            field_in = torch.from_numpy(field_in).to(self.device)
            
        # print(backend)

        if photon_energy is None:
            photon_energy = self.get_photon_energy()
        
        if field_in is None:
            field_in = self._realPart + 1j*self._imagPart
            
        z = position
        # old_pix_size = self._pixel_size # input plane grid spacing (spatial domain)
        # old_pix_number = self._pixel_number # pixel number (spatial domain)
        old_pix_size = pixel_size # input plane grid spacing (spatial domain)
        old_pix_number = pixel_number # pixel number (spatial domain)
        Lambda = self.get_lambda(photon_energy = photon_energy) # field wavelength
        k = self.get_k(photon_energy = photon_energy) # wave number = 2*pi/Lambda or other fixed value
        
        # Generate grid for the source plane (spatial domain)
        x = backend.arange(-old_pix_number/2, old_pix_number/2)*old_pix_size
        y1, x1 = backend.meshgrid(x,x)
        r1sq = x1**2 + y1**2
        
        # Generate grid in the Fourier domain
        fx = backend.fft.fftshift(backend.fft.fftfreq(old_pix_number, d=old_pix_size))
        fx, fy = backend.meshgrid(fx, fx)
        fsq = fx**2 + fy**2
        
        new_pix_size = Lambda*backend.abs(z)/(old_pix_number*old_pix_size)                           
        m = new_pix_size/old_pix_size # scaling parameter

        # Generate grid for the observation plane 
        x = backend.arange(-old_pix_number/2, old_pix_number/2)*new_pix_size 
        y2, x2 = backend.meshgrid(x,x)
        r2sq = x2**2 + y2**2

        # Quadratic phase factors
        Q1 = backend.exp(1j*k/2*(1-m)/z*r1sq)
        Q2 = backend.exp(-1j*2*const.pi**2*z/m/k*fsq)
        Q3 = backend.exp(1j*k/2*(m-1)/(m*z)*r2sq)
        
        if self.device == "cpu":
            fft_field_in = backend.fft.fftshift(backend.fft.fft2(Q1 * field_in / m)).astype('complex64')
            field_out = Q3 * backend.fft.ifft2(backend.fft.ifftshift(Q2 * fft_field_in)).astype('complex64')
        else:  # gpu
            fft_field_in = backend.fft.fftshift(backend.fft.fft2(Q1 * field_in / m)).to(self.device)
            field_out = Q3 * backend.fft.ifft2(backend.fft.ifftshift(Q2 * fft_field_in)).cpu()
            
        if reset:
            self._realPart = field_out.real
            self._imagPart = field_out.imag

        return field_out.real + 1j*field_out.imag
    
            
    def fresnel_onestep(self,
                        field_in=None,
                        position=None,
                        pixel_number=None, 
                        pixel_size=None,
                        photon_energy=None,
                        reset=True):
        """
        Implements Fresnel approximation in a single step using the impulse response method.
        Does NOT allow the user to control the grid spacing at the observation plane.
        
        NOTE: Currently the Lorentz Propagator in TK's code!
        
        ----
        Assumptions/conditions: 
        1. Scalar diffraction
        2. r >> lambda (the distance (r) between the source and observation plane, 
        must be much greater than the source wavelength (lambda))
        3. sqrt(fx**2 + fy**2) < 1/lambda 
        ----

        Parameters
        ----------
        field_in : input field (source plane)
        position : propagation distance (z)
        photon_energy : photon energy in keV

        Returns
        -------
        field_out : output field (observation plane)
        """

        if self.device == "cpu": # gpu
            backend = np # numpy
        else:
            backend = cp # cupy

        if photon_energy is None:
            photon_energy = self.get_photon_energy()
            
        if field_in is None:
            field_in = self._realPart + 1j*self._imagPart
            
        z = position
        # old_pix_size = self._pixel_size # source plane grid spacing (spatial domain)
        # old_pix_number = self._pixel_number # pixel number (spatial domain)
        Lambda = self.get_lambda(photon_energy = photon_energy) # field wavelength
        k = self.get_k(photon_energy = photon_energy) # wave number = 2*pi/Lambda or other fixed value

        # Generate grid for the source plane (spatial domain)
        x = np.arange(-old_pix_number/2, old_pix_number/2)*old_pix_size
        y1, x1 = backend.meshgrid(x,x) 
        
        # # Obtain source plane coordinates (spatial domain)
        # x1 = self._x
        # y1 = self._y
        
        # Generate grid for the observation plane 
        new_pix_size = Lambda*z/(old_pix_number*old_pix_size) 
        x2 = np.arange(-old_pix_number/2, old_pix_number/2)*new_pix_size
        y2, x2 = np.meshgrid(x2,x2)
        # x2, y2 = self._define_image_scale(pixel_size=new_pixel_size,reset=reset)
        
        # Perform Fresnel propagation to evaluate the field at the observation plane
        # Define the impulse response function h
        h = np.exp(1j*k*z) / (1j*Lambda*z) * np.exp(1j*k/(2*z) * (x2**2 + y2**2))
        fft_field_in = backend.fft.fft2(field_in * np.exp(1j*k/(2*z) * (x1**2 + y1**2)))
        field_out = h * backend.fft.fftshift(fft_field_in)

        if reset:
            self._realPart = field_out.real
            self._imagPart = field_out.imag

        return field_out.real + 1j*field_out.imag
                                            
                                               
    def fresnel_twostep(self,
                        field_in=None,
                        position=None,
                        pixel_number=None, 
                        pixel_size=None,
                        photon_energy=None,
                        reset=True):
        """
        Implements Fresnel approximation in two steps using the impulse response method.
        Allows the user to control the grid spacing at the observation plane.

        Assumptions/conditions: 
        ----
        1. Scalar diffraction
        2. r >> lambda (the distance (r) between the source and observation plane, 
        must be much greater than the source wavelength (lambda))
        3. sqrt(fx**2 + fy**2) < 1/lambda 

        Parameters
        ----------
        field_in : input field (source plane)
        position : propagation distance (z)
        photon_energy : photon energy in keV
        
        
        Returns
        -------
        field_out : output field (observation plane)
        """

        if self.device == "cpu": # gpu
            backend = np # numpy
        else:
            backend = cp # cupy

        if photon_energy is None:
            photon_energy = self.get_photon_energy()
            
        if field_in is None:
            field_in = self._realPart + 1j*self._imagPart
            
        z = position # propagation distance
        # old_pix_size = self._pixel_size # source plane grid spacing (spatial domain)
        # old_pix_number = self._pixel_number # pixel number (spatial domain)
        Lambda = self.get_lambda(photon_energy = photon_energy) # field wavelength
        k = self.get_k(photon_energy = photon_energy) # wave number = 2*pi/Lambda or other fixed value
    
        new_pix_size = Lambda*np.abs(z)/(old_pix_number*old_pix_size)                       
        m = new_pix_size/old_pix_size # magnification
                                            
        # Generate grid for the source plane (spatial domain)
        x = np.arange(-old_pix_number/2, old_pix_number/2)*old_pix_size
        y1, x1 = backend.meshgrid(x, x) 
                                            
        # Perform first Fresnel propagation to evaluate the field at some intermediate plane
        z1 = z / (1 - m) # propagation distance to intermediate plane
        # Generate grid for the intermediate plane
        # define grid spacing on the intermediate plane 
        itm_pix_size = Lambda*np.abs(z1)/(old_pix_number*old_pix_size) 
        x_itm = np.arange(-old_pix_number/2, old_pix_number/2)*itm_pix_size
        y_itm, x_itm = np.meshgrid(x_itm, x_itm) 
        # Define the impulse response function for the intermediate plane
        h_itm = np.exp(1j*k*z1) / (1j*Lambda*z1) * np.exp(1j*k/(2*z1) * (x_itm**2 + y_itm**2))
        field_itm = h_itm * backend.fft.fftshift(backend.fft.fft2(field_in * 
                                                                    np.exp(1j*k/(2*z1) * (x1**2 + y1**2))))                
        # Perform second Fresnel propagation to evaluate the field at the observation plane
        z2 = z - z1 # propagation distance to observation plane
        # Generate grid for the observation plane 
        x2 = np.arange(-old_pix_number/2, old_pix_number/2)*new_pix_size
        y2, x2 = np.meshgrid(x2,x2)
                                                   
                                                   
        # Define the impulse response function for the observation plane
        h_out = np.exp(1j*k*z2)/(1j*Lambda*z2) * np.exp(1j*k/(2*z2) * (x2**2 + y2**2))
        field_out = h_out * backend.fft.fftshift(backend.fft.fft2(field_itm * 
                                                                  np.exp(1j*k/(2*z2) * (x_itm**2 + y_itm**2))))
                                            
        if reset:
            self._realPart = field_out.real
            self._imagPart = field_out.imag

        return field_out.real + 1j*field_out.imag  

             
    def fraunhofer(self, 
                   field_in=None, 
                   position=None, 
                   pixel_number=None, 
                   pixel_size=None,
                   reset=True, 
                   photon_energy=None):   
        """
        Fraunhofer diffraction propagator (far-field approximation)
        ---------------------------------
        Calculate E field at position X, Y, Z=position.
            - X, Y are on the old coordinates x, y or those defined by 
            new pixel_number and pixel_size E(X,Y,Z) = Eqn. 13
            
        Fraunhofer approximation:
        Limit: 
        1. delta_z > 2 * D**2 / Lambda (Eq. 4.2) [2], (Eq ) [1]
            delta_z : propagation distance
            D : maximum spatial extent of the source-plane field
            lambda : source wavelength
        2. N_F << 1
            N_F : Fresnel number (N_F = w**2 / lambda * z)
                w : half-width of a square aperture in the source plane, or the radius of a circular aperture
                z : distance to the observation plane
            

        Citations:
        - Fraunhofer Diffraction Integral (Eq 1.67) [2]
        - 
        
        """
        
        if self.device == "cpu": # gpu
            backend = np # numpy
        else:
            backend = cp # cupy
        
        if photon_energy is None:
            photon_energy = self.get_photon_energy()
        if field_in is None:
            field_in = self._realPart + 1j*self._imagPart

        Lambda = self.get_lambda()
        k = self.get_k()
        z = position
        # old_pixel_size = self._pixel_size
        # old_pixel_number = self._pixel_number
                                                 
        new_pixel_size = Lambda*np.abs(z)/(old_pix_number*old_pix_size)   
        x2, y2 = self._define_image_scale(pixel_size=new_pixel_size,reset=reset)
                                                 
        # define convolution kernel h
        h = backend.exp(1j*k*z) / (1j*Lambda*z) * backend.exp(1j*k/(2*z) * (x2**2 + y2**2))
        fft_field_in = backend.fft.fft2(field_in)
        field_out = h * backend.fft.fftshift(fft_field_in)

        if reset:
            self._realPart = field_out.real
            self._imagPart = field_out.imag

        return field_out.real + 1j*field_out.imag
                                                              
                                                            
                                                              
    def set_pixel_number(self,pixel_number):
        self._pixel_number = pixel_number
        if self._pixel_size is not None:
            self._define_image_scale()

            
    def set_pixel_size(self,pixel_size):
        self._pixel_size = pixel_size
        if self._pixel_number is not None:
            self._define_image_scale()

            
    def set_parameters(self,parameters):
        self._parameters=parameters
        
        
    def get_k(self,photon_energy=None,Lambda=None):
        if Lambda is None:
            if photon_energy is None:
                photon_energy=self.get_photon_energy()
            factor = 2 * const.pi * const.e / const.h / const.c / 1e6 # # 2*pi*e/hc (in MeV)
            # return 5.06773076 * photon_energy 
            return factor * photon_energy
        else:
            2 * const.pi / Lambda

    def get_lambda(self, k=None, photon_energy=None):
        if photon_energy is None:
            photon_energy = self.get_photon_energy() 
        if k is None:
            k = self.get_k(photon_energy=photon_energy) 
        return (2 * const.pi) / k

    def get_photon_energy(self):
        return self._photon_energy
    
    def get_field(self):
        return self._realPart,self._imagPart
        
    def _define_image_scale(self,reset=True, pixel_number=None, pixel_size=None):
        if pixel_number is None:
            pixel_number=self._pixel_number
        if pixel_size is None:
            pixel_size=self._pixel_size
        #xaxis = np.arange(-self._pixel_number/2,self._pixel_number/2) * self._pixel_size
        xaxis = np.arange(-pixel_number/2,pixel_number/2) * pixel_size # by Lingen
        x, y = np.meshgrid(xaxis, xaxis)
        
        if reset:
            self._x = x
            self._y = y
            self._pixel_number=pixel_number
            self._pixel_size=pixel_size
        return x, y

    
