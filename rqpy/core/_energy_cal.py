import numpy as np
from scipy.optimize import curve_fit
from rqpy import plotting, utils
import matplotlib.pyplot as plt 


__all__ = ["fit_integral_ofamp", "fit_saturation"]

def fit_saturation(x, y, yerr, guess, labeldict=None, lgcplot=True, ax=None):
    """
    Function to fit the saturation of the measured calibration spectrum. 
    
    Parameters
    ----------
    x : array_like
        The true energy of the spectral peaks in eV
    y : array_like
        The measured energy (or similar quantity) of the spectral peaks in eV
    yerr : array_like
        The errors in the measured energy of the spectral peaks in eV
    guess : array_like
        Array of initial guess parameters (a,b) to be passed to saturation_func(). See Notes
        for the functional form these parameters apply to.
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are : 
            labels = {'title' : 'Energy Saturation Correction', 
                      'xlabel' : 'True Energy [eV]',
                      'ylabel' : 'Measured Energy [eV]',
                      'nsigma' : 2} # Note, nsigma is the number of sigma error bars 
                                      to plot 
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}
    lgcplot : bool, optional
            If True, the fit and spectrum will be plotted    
    ax : axes.Axes object, optional
        Option to pass an existing Matplotlib Axes object to plot over, if it already exists.
        
    Returns
    -------
    popt : ndarray
        Array of best fit paramters
    pcov : ndarray
        Covariance matrix from fit
    slope_linear : float
        The slope of the Taylor expansion of the saturation function 
        evaluated at the best fit parameters
        
    Notes
    -----
    This function fits the function y = a(1-exp(-x/b)) to the given data. This function
    is then Taylor expanded about x=0 to find the linear part of the calibration at
    low energies. There errors in this taylor expanded function y~ax/b, are determined
    via the covariance matrix returned from the initial fit.
    
    """
    
    popt, pcov = curve_fit(utils.saturation_func, x, y, sigma = yerr, p0 = guess, 
                           absolute_sigma=True, maxfev = 10000)
    if lgcplot:
        plotting.plot_saturation_correction(x, y, yerr, popt, pcov, labeldict, ax)
        
    slope_linear = utils.sat_func_expansion(1, *popt)
    
    return popt, pcov, slope_linear
    
def fit_integral_ofamp(xarr, yarr, clinearx, clineary, guess = (2e-6, 2e-10), yerr=None, nsigma=2):


    x = xarr[(xarr < clinearx) & (yarr < clineary)]
    y = yarr[(xarr < clinearx) & (yarr < clineary)]
    x_fit = np.linspace(0, max(x), 50)
    if yerr is None:
        err = np.ones_like(y)
    elif (isinstance(yerr, float) or isinstance(yerr, int)):
        err = np.ones_like(y)*yerr
    else:
        err = yerr
        
    popt, pcov = curve_fit(utils.invert_saturation_func, xdata = x, ydata = y,
                                   sigma = err ,p0 = guess, maxfev=100000, absolute_sigma = True)

    sat_errors = utils.prop_invert_sat_err(x_fit, popt, pcov)
    y_fit = utils.invert_saturation_func(x_fit, *popt)
    linear_approx = popt[1]/popt[0]
    
    #dfda = popt_sat[1]/popt_sat[0]**2
    #dfdb = -1/popt_sat[0]
   



    #linear_approx_error = np.sqrt(dfda**2*pcov_sat[0,0] + dfdb**2*pcov_sat[1,1] + dfda*pcov_sat[0,1]*dfdb + dfdb*pcov_sat[1,0]*dfda)

    plt.figure(figsize=(9,6))

    plt.errorbar(x,y, marker = '.', linestyle = ' ', yerr = err, label = 'Data used for Fit',
                 elinewidth=0.3, alpha =.5, ms = 5,zorder = 50)

    plt.grid(True, linestyle = '--')
    plt.plot(x_fit, y_fit, color = 'k',  label = r'$y = -b*ln(1-y/a)$')
    plt.fill_between(x_fit, y_fit+nsigma*sat_errors, y_fit-nsigma*sat_errors, color = 'k' , alpha= .5)
    
    plt.plot(x_fit, linear_approx*x_fit,zorder = 200, c = 'r', linestyle = '--', label = 'linear approximation (2σ bounds) ')

    plt.legend()
    return linear_approx#, linear_approx_error
