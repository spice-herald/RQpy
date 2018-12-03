import numpy as np
from ._utils import _bindata
from ._functions import gaussian_background
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from rqpy.plotting._plotting import _plot_gauss

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
