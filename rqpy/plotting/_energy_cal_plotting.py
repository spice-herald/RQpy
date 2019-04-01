import numpy as np
import matplotlib.pyplot as plt
from rqpy import utils


__all__ = ["plot_saturation_correction",
           "_plot_fit_integral_ofamp",
          ]


def plot_saturation_correction(x, y, yerr, popt, pcov, labeldict, ax=None):
    """
    Helper function to plot the fit for the saturation correction

    Parameters
    ----------
    x : array_like
        Array of x data
    y : array_like
        Array of y data
    yerr : array_like
        The errors in the measured energy of the spectral peaks
    guess : array_like
        Array of initial guess parameters (a,b) to be passed to saturation_func()
    popt : array_like
        Array of best fit parameters from fit_saturation()
    pcov : array_like
        Covariance matrix returned by fit_saturation()
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are : 
            labels = {'title' : 'Energy Saturation Correction', 
                      'xlabel' : 'True Energy [eV]',
                      'ylabel' : 'Measured Energy [eV]'}
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}
    ax : axes.Axes object, optional
        Option to pass an existing Matplotlib Axes object to plot over, if it already exists.

    Returns
    -------
    fig : Figure
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object

    """

    labels = {'title'  : 'Energy Saturation Correction',
              'xlabel' : 'True Energy [eV]', 
              'ylabel' : 'Measured Energy [eV]',
              'nsigma' : 2} 

    if labeldict is not None:
        for key in labeldict:
            labels[key] = labeldict[key]
    n = labels['nsigma'] 

    x_fit = np.linspace(0, x[-1], 100)
    y_fit = utils.saturation_func(x_fit, *popt)
    y_fit_lin = utils.sat_func_expansion(x_fit, *popt)

    err_full = utils.prop_sat_err(x_fit,popt,pcov)
    err_lin = utils.prop_sat_err_lin(x_fit,popt,pcov)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = None

    ax.set_title(labels['title'], fontsize = 16)
    ax.set_xlabel(labels['xlabel'], fontsize = 14)
    ax.set_ylabel(labels['ylabel'], fontsize = 14)
    ax.grid(True, linestyle = 'dashed')

    ax.scatter(x,y, marker = 'x', label = 'Spectral Peaks' , s = 100, zorder = 100, color ='b')
    ax.errorbar(x,y, yerr=yerr, linestyle = ' ')
    ax.plot(x_fit, y_fit, label = f'$y = a[1-exp(x/b)]$ $\pm$ {n} $\sigma$', color = 'g')
    ax.fill_between(x_fit, y_fit - n*err_full, y_fit + n*err_full, alpha = .5, color = 'g')
    ax.plot(x_fit, y_fit_lin, linestyle = '--', color = 'r', 
            label = f'Taylor Expansion of Saturation Function $\pm$ {n} $\sigma$')
    ax.fill_between(x_fit, y_fit_lin - n*err_lin, y_fit_lin + n*err_lin, alpha = .2, color = 'r')

    ax.legend(loc = 2, fontsize = 14)
    ax.tick_params(which="both", direction="in", right=True, top=True)
    plt.tight_layout()

    return fig, ax


def _plot_fit_integral_ofamp(x, y, err, y_fit, sat_errors, linear_approx,
                             linear_approx_errs, labeldict, ax=None):
    """
    Helper function to plot the fit for fit_integral_ofamp()

    Parameters
    ----------
    x : array_like
        Array of x data
    y : array_like
        Array of y data
    err : array_like
        The errors in the measured energy of the spectral peaks
    y_fit : array_like
        Array of y data from fit
    sat_errors : array_like
        Array of errors for the fit
    linear_approx : float
        The slope of the linear approximation of the saturated function
    linear_approx_errs : array
        Array of errors for the approximation of the fit
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are : 
            labels = {'title' : 'Energy Saturation Correction', 
                      'xlabel' : 'True Energy [eV]',
                      'ylabel' : 'Measured Energy [eV]'}
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}
    ax : axes.Axes object, optional
        Option to pass an existing Matplotlib Axes object to plot over, if it already exists.

    Returns
    -------
    fig : Figure
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object

    """

    labels = {'title'  : 'OF Amplitude vs Integral Saturation Correction',
              'xlabel' : 'OF Amplitude [A]', 
              'ylabel' : 'Integrated Charge [C]',
              'nsigma' : 2} 

    if labeldict is not None:
        for key in labeldict:
            labels[key] = labeldict[key]
    nsigma = labels['nsigma'] 

    x_fit = np.linspace(0, max(x), 50)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = None
    ax.set_title(labels['title'], fontsize = 16)
    ax.set_xlabel(labels['xlabel'], fontsize = 14)
    ax.set_ylabel(labels['ylabel'], fontsize = 14)
    ax.grid(True, linestyle = 'dashed')

    ax.grid(True, linestyle = '--')
    ax.set_xlim(0, max(x)*1.05)
    ax.set_ylim(0, max(y)*1.05)

    ax.errorbar(x,y, marker = '.', linestyle = ' ', yerr = err, label = 'Data used for Fit',
                 elinewidth=0.3, alpha =.5, ms = 5,zorder = 50)

    ax.plot(x_fit, y_fit, color = 'k',  label = f'Fit : $y = -b*ln(1-x/a)$ ({nsigma}σ bounds)')
    ax.fill_between(x_fit, y_fit+nsigma*sat_errors, y_fit-nsigma*sat_errors, color = 'k' , alpha= .5)

    ax.plot(x_fit, linear_approx*x_fit,zorder = 200, c = 'r', linestyle = '--', 
            label = f'Linear approximation ({nsigma}σ bounds) ')
    ax.fill_between(x_fit, linear_approx*x_fit+nsigma*linear_approx_errs,
                    linear_approx*x_fit-nsigma*linear_approx_errs, color = 'r' , alpha= .5)
    ax.legend()    

    return fig, ax

