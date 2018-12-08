import numpy as np


__all__ = ["bindata", "gaussian", "n_gauss", "gaussian_background", "saturation_func", 
           "sat_func_expansion", "prop_sat_err", "prop_sat_err_lin"]


def bindata(arr, xrange=None, bins='sqrt'):
    """
    Helper function to convert 1d array into binned (x,y) data
    
    Parameters
    ----------
    arr : ndarray
        Input array
    xrange : tuple, optional
        Range over which to bin data
    bins : int or str, optional
        Number of bins, or type of automatic binning scheme 
        (see numpy.histogram())
    
    Returns
    -------
    x : ndarray
        Array of x data
    y : ndarray
        Array of y data
    bins : ndarray
        Array of bins returned by numpy.histogram()
    
    """
    
    if xrange is not None:
        y, bins = np.histogram(arr, bins=bins, range=xrange)
    else:
        y, bins = np.histogram(arr, bins=bins)
    x = (bins[1:]+bins[:-1])/2
    
    return x, y, bins

def gaussian(x, amp, mean, sd):
    """
    Functional form for Gaussian distribution
    
    Parameters
    ----------
    x : ndarray
        Array corresponding to x data
    amp : float
        Normilization factor (or amplitude) for function
    mean : float
        The first moment of the distribution
    sd : float
        The second moment of the distribution
        
    Return
    ------
    gauss : array
        Array y values corresponding to the given x values
        
    """
    
    gauss = amp*np.exp(-(x - mean)**2/(2*sd**2))
    
    return gauss 

def n_gauss(x, params, n):
    """
    Function to sum n Gaussian distributions
    
    Parameters
    ----------
    x : ndarray
        Array corresponding to x data
    params : tuple
        The order must be as follows:
        (amplitude_i, mu_i, std_i,
        ....,
        ....,
        background),
        where the guess for the background is the last element
    n : int
        The number of Gaussian distributions to be summed
        
    Returns
    -------
    results : ndarray
        2D array of Gaussians, where the first dimension corresponds
        to each Gaussian. 
            
    Raises
    ------
    ValueError
        If the number or parameters given is in conflict with n,
        a ValueError is raised.
        
    """
    
    if n != int((len(params)-1)/3):
        raise ValueError('Number of parameters must match the number of Gaussians')

    results = []
    for ii in range(n):
        results.append(gaussian(x, *params[ii*3:(ii*3)+3]))
    results.append(np.ones(x.shape)*params[-1])
    results = np.array(results)
    
    return results

def gaussian_background(x, amp, mean, sd, background):
    """
    Functional form for Gaussian distribution plus a background offset 
    
    Parameters
    ----------
    x : ndarray
        Array corresponding to x data
    amp : float
        Normilization factor (or amplitude) for function
    mean : float
        The first moment of the distribution
    sd : float
        The second moment of the distribution
    background : float
        The offset (in the y-direction)
        
    Returns
    -------
    gauss_background : ndarray
        Array y values corresponding to the given x values
        
    """
    
    gauss_background = gaussian(x, amp, mean, sd) + background
    
    return gauss_background

def saturation_func(x, a, b):
    """
    Function to describe the saturation of a signal in a 
    detector as a function of energy 
    
    
    Parameters
    ----------
    x : ndarray
        Array of x-data
    a : float
        Amplitude parameter
    b : float
        Saturation parameter
        
    Returns
    -------
    sat_func : ndarray
        Array of y-values 
        
    Notes
    -----
    This function has the following criteria imposed on it
    in order to be physically consistant with saturation
    in a system. It must be monotonic, and must asymptote to
    a fixed value for large values of x. 
    
    The functional form is as follows:
    
    y = a(1-exp(-x/b))
    
    """

    sat_func = a*(1-np.exp(-x/b))
    
    return sat_func

def sat_func_expansion(x, a, b):
    """
    Taylor expansion of saturation_func()
    
    Parameters
    ----------
    x : ndarray
        Array of x-data
    a : float
        Amplitude parameter
    b : float
        Saturation parameter
        
    Returns
    -------
    lin_func : ndarray
        Array of y-values 
    
    """
    
    lin_func = a*x/b
    
    return lin_func

def prop_sat_err(x, params,cov):
    """
    Helper function to propagate errors for saturation_func()
    
    Parameters
    ----------
    x : ndarray
        Array of x-data
    params : ndarray
        Best fit parameters for saturation_func()
    cov : ndarray
        Covariance matrix for parameters
        
    Returns
    -------
    errors : ndarray
        Array of 1 sigma errors as a function of x
        
    """
    
    a, b = params
    deriv = np.array([(1-np.exp(-x/b)), -a*x*np.exp(-x/b)/(b**2)])
    sig_func = []
    for ii in range(len(deriv)):
        for jj in range(len(deriv)):
            sig_func.append(deriv[ii]*cov[ii][jj]*deriv[jj])
    sig_func = np.array(sig_func)
    errors = sig_func.sum(axis=0) 
    
    return errors



def prop_sat_err_lin(x, params, cov):
    """
    Helper function to propagate errors for the taylor expantion of 
    saturation_func()
    
    Parameters
    ----------
    x : ndarray
        Array of x-data
    params : ndarray
        Best fit parameters for _saturation_func()
    cov : ndarray
        Covariance matrix for parameters
        
    Returns
    -------
    errors : ndarray
        Array of 1 sigma errors as a function of x
        
    """
    
    a, b = params
    deriv = np.array([x/b, -a*x/(b**2)])
    sig_func = []
    for ii in range(len(deriv)):
        for jj in range(len(deriv)):
            sig_func.append(deriv[ii]*cov[ii][jj]*deriv[jj])
    sig_func = np.array(sig_func)
    errors = sig_func.sum(axis=0)
    
    return errors

