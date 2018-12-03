import numpy as np
from ._utils import _bindata
from ._functions import gaussian_background, n_gauss
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from rqpy.plotting._plotting import _plot_gauss, _plot_n_gauss







def fit_multi_gauss(arr, guess, ngauss, xrange = None, lgcplot = True, labeldict = None, lgcfullreturn = False):
    """
    Function to multiple Gaussians plus a flat background. Note, depending on
    the spectrum, this function can ber very sensitive to the inital guess parameters. 
    
    
    Parameters
    ----------
    arr: array
        Array of values to be binned
    guess: tuple
        The initial guesses for the Gaussian peaks. The order must be as follows:
        (amplitude_i, mu_i, std_i,
        ....,
        ....,
        background),
        where the guess for the background is the last element
    ngauss: int
        The number of peaks to fit
    xrange: tuple, optional
        The range over which to fit the peaks
    lgcplot: bool, optional
        If True, the fit and spectrum will be plotted 
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are : 
            labels = {'title' : 'Histogram', 'xlabel' : 'variable', 'ylabel' : 'Count'}
            Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to fig_gauss()
    lgcfullreturn: bool, optional
        If True, the binned data is returned along with the fit parameters
        
    Returns
    -------
    fitparams: array
        The best fit parameters, in the same order as the input guess
    errors: array
        The uncertainty in the best fit parameters
    cov: array
        The covariance matrix returned by scipy.optimize.curve_fit()
    bindata: tuple, optional
        The binned data from _bindata(), in order (x, y, bins)
        
    Raises
    ------
    ValueError:
        If the number or parameters given in the guess is in conflict with ngauss,
        a ValueError is raised.
        
    """
    
    if ngauss != (len(guess)-1)/3:
        raise ValueError('Number of parameters in guess must match the number of Gaussians being fit (ngauss)')

    def fit_n_gauss(x, *params):
        return n_gauss(x, params, ngauss).sum(axis = 0)
    
        
    
    x,y, bins = _bindata(arr,  xrange = xrange, bins = 'sqrt')
    yerr = np.sqrt(y)
    yerr[yerr == 0] = 1 #make errors 1 if bins are empty
    
   
    fitparams, cov = curve_fit(fit_n_gauss, x, y, guess, sigma = yerr,absolute_sigma = True)
    errors = np.sqrt(np.diag(cov))    
    
    if lgcplot:
        _plot_n_gauss(x, y, bins, fitparams, labeldict)
    if lgcfullreturn:
        return fitparams, errors, cov, (x, y, bins)
    else:
        return fitparams, errors, cov


def fit_gauss(arr ,xrange = None, noiserange = None, lgcplot = False, labeldict = None):
    """
    Function to fit Gaussian distribution with background to peak in spectrum. 
    Errors are assumed to be poissonian. 
    
    
    Parameters
    ----------
        arr: ndarray
            Array of data to bin and fit to gaussian
        xrange: tuple, optional
            The range of data to use when binning
        noiserange: tuple, optional
            nested 2-tuple. should contain the range before 
            and after the peak to be used for subtracting the 
            background
        lgcplot: bool, optional
            If True, the fit and spectrum will be plotted 
        labeldict : dict, optional
            Dictionary to overwrite the labels of the plot. defaults are : 
                labels = {'title' : 'Histogram', 'xlabel' : 'variable', 'ylabel' : 'Count'}
            Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to fig_gauss()
            
    Returns
    -------
        peakloc: float
            The mean of the distribution
        peakerr: float
            The full error in the location of the peak
        fitparams: tuple
            The best fit parameters of the fit; A, mu, sigma
        errors: ndarray
            The uncertainty in the fit parameters
        
            
    """
    
    x,y, bins = _bindata(arr,  xrange, bins = 'sqrt')
    yerr = np.sqrt(y)
    yerr[yerr == 0] = 1 #make errors 1 if bins are empty
    if noiserange is not None:
        if noiserange[0][0] >= xrange[0]:
            clowl = noiserange[0][0]
        else:
            clow = xrange[0]
        clowh = noiserange[0][1]
        chighl = noiserange[1][0]
        if noiserange[1][1] <= xrange[1]:
            chighh = noiserange[1][1] 
        else:
            chighh = xrange[1]          
        indlowl = (np.abs(x - clowl)).argmin()
        indlowh = (np.abs(x - clowh)).argmin() 
        indhighl = (np.abs(x - chighl)).argmin()
        indhighh = (np.abs(x - chighh)).argmin() - 1
        background = np.mean(np.concatenate((y[indlowl:indlowh],y[indhighl:indhighh])))  
    else:
        background = 0
    y_noback = y - background
    # get starting values for guess   
    A0 = np.max(y_noback)
    mu0 = x[np.argmax(y_noback)]
    sig0 = np.abs(mu0 - x[np.abs(y_noback - np.max(y_noback)/2).argmin()])
    p0 = (A0, mu0, sig0, background)
    #do fit
    fitparams, cov = curve_fit(gaussian_background, x, y, p0, sigma = yerr,absolute_sigma = True)
    errors = np.sqrt(np.diag(cov))    
    peakloc = fitparams[1]
    peakerr = np.sqrt((fitparams[2]/np.sqrt(fitparams[0]))**2)
    
    if lgcplot:
        _plot_gauss(x, bins, y, fitparams, errors, background, labeldict)
    
    return peakloc, peakerr, fitparams, errors
