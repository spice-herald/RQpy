import numpy as np
from scipy.optimize import curve_fit
from rqpy import plotting, utils
import matplotlib.pyplot as plt 


__all__ = ["fit_integral_ofamp", "fit_saturation", "scale_integral", "scale_of_to_integral"]

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
    linear_approx : float
        The slope of the Taylor expansion of the saturation function 
        evaluated at the best fit parameters
    slope_error : float
        The error in the linear slope
        
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
        
    linear_approx = utils.sat_func_expansion(1, *popt)
    slope_error = utils.prop_sat_err_lin(1, popt, pcov)
    
    
    return popt, pcov, linear_approx, slope_error
    
def fit_integral_ofamp(xarr, yarr, clinearx, clineary, guess = (2e-6, 2e-10),
                       yerr=None, labeldict=None, lgcplot=True, ax=None):
    """
    Function to fit the saturation of the Optimum Filter metric to an Integral 
    type metric for Low Energy Events. 
    
    Parameters
    ----------
    xarr : array_like
        Array of OF amplitudes
    yarr : array_like
        Array of Integrals (or Integral like measurement)
    clinearx : float
        The cut-off in x to include in the fit
    clineary : float
        The cut-off in y to include in the fit
    guess : tuple, optional
        Intial guess for the fit. Note, can be very sensitive to guess,
        may take multiple tries to get good fit
    yerr : float, array, or NoneType, optional
        The errors in the integral metric. Can be a single float applied
        to all events, or an array of values of the same shape as xarr. 
        Can also be left as None. In this case, all data point will be 
        weighted equally, Note that the error returned from the fit is meaningless 
        in this case.
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
    linear_approx : float
        The slope of the Taylor expansion of the inverted saturation function 
        evaluated at the best fit parameters
    slope_error : float
        The error in the linear slope
    """
    
    
    
    
    cut = (xarr < clinearx) & (yarr < clineary) & (yarr > 0) & (xarr > 0)
    x = xarr[cut]
    y = yarr[cut]

    x_fit = np.linspace(0, max(x), 50)
    if yerr is None:
        err = np.ones(y.shape)
    elif (isinstance(yerr, float) or isinstance(yerr, int)):
        err = np.ones(y.shape)*yerr
    else:
        err = yerr[cut]
        
    popt, pcov = curve_fit(utils.invert_saturation_func, xdata = x, ydata = y,
                                   sigma = err, p0 = guess, maxfev=100000, absolute_sigma = True)
    y_fit = utils.invert_saturation_func(x_fit, *popt)
    sat_errors = utils.prop_invert_sat_err(x_fit, popt, pcov)
    
    linear_approx = popt[1]/popt[0]
    linear_approx_errs = utils.prop_sat_err_lin(x_fit, popt[::-1], pcov[::-1,::-1])
    slope_error = utils.prop_sat_err_lin(1, popt[::-1], pcov[::-1,::-1])

    if lgcplot:
        plotting._plot_fit_integral_ofamp(x=x, y=y, err=err, y_fit=y_fit, sat_errors=sat_errors,
                                          linear_approx=linear_approx, linear_approx_errs=linear_approx_errs,
                                          labeldict=labeldict, ax=ax)
    
    return popt, pcov, linear_approx, slope_error



def scale_integral(vals, lgcsaturated=False, linparams=None, satparams=None):
    """
    Function to convert saturated measured energy into 
    true energy
    
    Parameters
    ----------
    vals : ndarray
        Array of measured energies to be converted to true energies
    lgcsaturated : bool, Defaults to False
        If True, the saturated correction is done. Note, the user must 
        provide the fit parameters from the saturated energy fit.
        If False, the linear scaling is done. The user must suply the
        slope in this case
    linparams : list, optional
        List containing the slope for the linear approximation, and 
        the error in the slope. If the error in the slope is not known, 
        then just put 0 for linparams[0]. 
        linparams must be in units of [(units of vals)/(eV)]
    satparams : list, optional
        List containing the best fit parameters from the fit_saturation() 
        fuction. fitparams[0] should correspond to the optimum parameters
        and fitparams[1] should be the covariance matrix from the fit
        
    Returns
    -------
    energy_true : ndarray
        Array of saturation corrected energies
    errors : ndarray
        Array of uncertainties for each value in energy_true
        
    """
    
    if not lgcsaturated:
        if linparams is None:
            raise ValueError('Must provide linparamsto do the linear scaling')
        else:
            energy_true = vals/linparams[0]
            errors = np.sqrt(abs(-vals/linparams[0])**2*linparams[1])
            
    else:
        if satparams is None:
            raise ValueError('Must provide satparamsto do the saturation correction')
        else:
            params = satparams[0]
            cov = satparams[1]

            energy_true = utils.invert_saturation_func(vals, *params)
            errors = utils.prop_invert_sat_err(vals, params, cov)
    
    return energy_true, errors





def scale_of_to_integral(vals, satparams):
    """
    Function to calibrate the OF amplitude to an integral type metric
    
    Parameters
    ----------
    vals : ndarray
        Array of OF amplitudes to be calibrated
    satparams : list
        List containing the best fit parameters from the fit_saturation() 
        fuction. fitparams[0] should correspond to the optimum parameters
        and fitparams[1] should be the covariance matrix from the fit
        
    Returns
    -------
    energy_true : ndarray
        Array of saturation corrected energies
    errors : ndarray
        Array of uncertainties for each value in energy_true
        
    """
           
    
    params = satparams[0]
    cov = satparams[1]

    energy_true = utils.invert_saturation_func(vals, *params)
    try:
        errors = utils.prop_invert_sat_err(vals, params, cov)
    except:
        print('Problem calculating errors')
        errors = np.ones(vals.shape)
    return energy_true, errors




