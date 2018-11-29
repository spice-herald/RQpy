import numpy as np

__all__ = ["gaussian", "gaussian_background", "double_gaussian"]



def gaussian(x,amp, mean, sd):
    """
    Functional form for Gaussian distribution
    
    Parameters
    ----------
    x: array
        Array corresponding to x data
    amp: float
        Normilization factor (or amplitude) for function
    mean: float
        The first moment of the distribution
    sd: float
        The second moment of the distribution
        
    Return
    ------
    gauss: array
        Array y values corresponding to the given x values
        
    """
    
    gauss = amp*np.exp(-(x - mean)**2/(2*sd**2))
    return gauss 

def gaussian_background(x,amp, mean, sd, offset):
    """
    Functional form for Gaussian distribution plus a background offset 
    
    Parameters
    ----------
    x: array
        Array corresponding to x data
    amp: float
        Normilization factor (or amplitude) for function
    mean: float
        The first moment of the distribution
    sd: float
        The second moment of the distribution
    offset: float
        The offset (in the y-direction)
        
    Return
    ------
    gauss_background: array
        Array y values corresponding to the given x values
        
    """
    
    gauss_background =  gaussian(x,amp, mean, sd) + background
    return gauss_background


def double_gaussian(x, *params):
    """
    Functional form for two Gaussian distributions added together
    
    Parameters
    ----------
    x: array
        Array corresponding to x data
    params: list
        A list of the paramters to be passed to gaussian()
        in the following order:
            amp1, amp2, mean1, mean2, sd1, sd2 = params
            
    Return
    ------
    double_gauss: array
        Array y values corresponding to the given x values
        
    """
    
    a1, a2, m1, m2, sd1, sd2 = params
    double_gauss = gaussian(x,a1, m1, sd1) + gaussian(x,a2, m2, sd2)
    return double_gauss