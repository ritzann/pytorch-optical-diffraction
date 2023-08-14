import numpy
import torch


class Objects():
    def __init__(self, backend = torch, device = "cuda"):
        """
        Parameters
        ----------
        backend : module
            two options: `numpy` or `torch`
        device : string
            two options: "cpu" or "cuda"
        nx : int
            array size along the x-axis in pixels
        ny : int
            array size along the y-axis in pixels
        """
    
        self.device = device
        self.backend = backend
        # assert self.backend == torch, "numpy has no gpu support"

    
    def create_2Darray(self, nx, ny):
        
        x = self.backend.linspace(-1,1,nx)
        y = self.backend.linspace(1,-1,ny)
        arr = self.backend.zeros((nx,ny))
            
        if self.backend == torch:
            x = x.to(self.device)
            y = y.to(self.device)
            
        X,Y = self.backend.meshgrid(x,y)
        
        self.X = X
        self.Y = Y
        self.arr = arr
        
    
    def create_circle(self, r = 0.5, x0 = 0, y0 = 0):
        """
        Creates a circular aperture of radius r centered at (x0, y0).

        Parameters
        ----------
        x0 : int
            
        y0 : int
        
        r : float
            circular aperture radius (0 < r < 1)

        -------
        circ : logical 2D array
            circular aperture
        radius : float 2D array
            polar coordinates (radius)
        theta : float 2D array
            polar coordinates (angle)
        """

        radius = self.backend.abs(self.X + 1j*self.Y) # polar radius
        theta = self.backend.angle(self.X + 1j*self.Y) # polar angle (if needed, return)
        circ = radius < r
        return circ
    
    
    def create_rect(self, w=0.5, h=0.25):
        """
        Creates a rectangular aperture with specified width w and height h.

        Parameters
        ----------
        w : float
            width of the rectangular aperture
        h : float
            height of the rectangular aperture

        -------
        rec : float 2D array
            rectangular aperture
        """
        
        rect = self.arr
        indices = self.backend.where((self.backend.abs(self.X) < w) & 
                                     (self.backend.abs(self.Y) < h))
        rect[indices] = 1
        return rect
        