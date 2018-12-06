import numpy as np

__all__ = ["gaussian", "gaussian_background", "double_gaussian", "n_gauss", "saturation_func"]



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

def n_gauss(x, params, n):
    """
    Function to sum n Gaussian distributions
    
    Parameters
    ----------
    x: array
        Array corresponding to x data
    params: tuple
        The order must be as follows:
        (amplitude_i, mu_i, std_i,
        ....,
        ....,
        background),
        where the guess for the background is the last element
    n: int
        The number of Gaussian distributions to be summed
        
    Returns
    -------
        results: array
            2D array of Gaussians, where the first dimension corresponds
            to each Gaussian. 
            
    Raises
    ------
    ValueError:
        If the number or parameters given is in conflict with n,
        a ValueError is raised.
        
    """
    
    if n != int((len(params)-1)/3):
        raise ValueError('Number of parameters must match the number of Gaussians')

    results = []
    for ii in range(n):
        results.append(gaussian(x, *params[ii*3:(ii*3)+3]))
    results.append(np.ones(shape = x.shape)*params[-1])
    results =  np.array(results)
    return results

def gaussian_background(x,amp, mean, sd, background):
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
    background: float
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


    
def saturation_func(x, a, b):
    """
    Function to describe the saturation of a signal in a 
    detector as a function of energy 
    
    
    Parameters
    ----------
    x : array
        Array of x-data
    a : float
        Amplitude parameter
    b : float
        Saturation parameter
        
    Returns
    -------
    sat_func : array
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
    x : array
        Array of x-data
    a : float
        Amplitude parameter
    b : float
        Saturation parameter
        
    Returns
    -------
    lin_func : array
        Array of y-values 
    
    """
    
    lin_func = a*x/b
    
    return lin_func

def prop_sat_err(x,params,cov):
    """
    Helper function to propagate errors for saturation_func()
    
    Parameters
    ----------
    x : array
        Array of x-data
    params : array
        Best fit parameters for saturation_func()
    cov : array
        Covariance matrix for parameters
        
    Returns
    -------
    errors : array
        Array of 1 sigma errors as a function of x
        
    """
    
    a, b = params
    deriv = np.array([(1-np.exp(-x/b)), -a*x*np.exp(-x/b)/(b**2)])
    sig_func = []
    for ii in range(len(deriv)):
        for jj in range(len(deriv)):
            sig_func.append(deriv[ii]*cov[ii][jj]*deriv[jj])
    sig_func = np.array(sig_func)
    errors = sig_func.sum(axis = 0) 
    
    return errors



def prop_sat_err_lin(x, params, cov):
    """
    Helper function to propagate errors for the taylor expantion of 
    saturation_func()
    
    Parameters
    ----------
    x : array
        Array of x-data
    params : array
        Best fit parameters for saturation_func()
    cov : array
        Covariance matrix for parameters
        
    Returns
    -------
    errors : array
        Array of 1 sigma errors as a function of x
        
    """
    
    a, b = params
    deriv = np.array([x/b, -a*x/(b**2)])
    sig_func = []
    for ii in range(len(deriv)):
        for jj in range(len(deriv)):
            sig_func.append(deriv[ii]*cov[ii][jj]*deriv[jj])
    sig_func = np.array(sig_func)
    errors = sig_func.sum(axis = 0)
    
    return errors







