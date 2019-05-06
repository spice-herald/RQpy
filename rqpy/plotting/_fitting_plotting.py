import numpy as np
import matplotlib.pyplot as plt
import rqpy as rp
from rqpy import utils
from scipy.special import erf


__all__ = ["plot_gauss",
           "plot_gauss_asymbkg",
           "plot_n_gauss",
          ]

def plot_gauss(x, bins, y, fitparams, errors, background, labeldict=None):
    """
    Hidden helper function to plot Gaussian plus background fits

    Parameters
    ----------
    x : array_like
        Array of x data
    bins : array_like
        Array of binned data
    y : array_like
        Array of y data
    fitparams : tuple
        The best fit parameters from the fit
    errors : tuple
        The unccertainty in the best fit parameters
    background : float
        The average background rate
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are : 
            labels = {'title' : 'Histogram', 'xlabel' : 'variable', 'ylabel' : 'Count'}
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to _plot_gauss()

    Returns
    -------
    fig : Figure
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object

    """

    x_fit = np.linspace(x[0], x[-1], 250) #make x data for fit

    labels = {'title'  : 'Gaussian Fit',
              'xlabel' : 'x variable', 
              'ylabel' : 'Count'} 
    if labeldict is not None:
        for key in labeldict:
            labels[key] = labeldict[key]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(labels['title'])
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['ylabel'])
    ax.plot([],[], linestyle = ' ', label = f' μ = {fitparams[1]:.2f} $\pm$ {errors[1]:.3f}')
    ax.plot([],[], linestyle = ' ', label = f' σ = {fitparams[2]:.2f} $\pm$ {errors[2]:.3f}')
    ax.plot([],[], linestyle = ' ', label = f' A = {fitparams[0]:.2f} $\pm$ {errors[0]:.3f}')
    ax.plot([],[], linestyle = ' ', label = f' Offset = {fitparams[3]:.2f} $\pm$ {errors[3]:.3f}')

    ax.hist(x, bins = bins, weights = y, histtype = 'step', linewidth = 1, label ='Raw Data', alpha = .9)
    ax.axhline(background, label = 'Average Background Rate', linestyle = '--', alpha = .3)

    ax.plot(x_fit, utils.gaussian_background(x_fit, *fitparams), label = 'Gaussian Fit')
    ax.legend()
    ax.grid(True, linestyle = 'dashed')

    return fig, ax

def plot_gauss_asymbkg(x, bins, y, fitparams, errors,labeldict=None):
    """
    Hidden helper function to plot Gaussian plus asymmetric background fit

    Parameters
    ----------
    x : array_like
        Array of x data
    bins : array_like
        Array of binned data
    y : array_like
        Array of y data
    fitparams : tuple
        The best fit parameters from the fit
    errors : tuple
        The unccertainty in the best fit parameters
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are : 
            labels = {'title' : 'Histogram', 'xlabel' : 'variable', 'ylabel' : 'Count'}
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to _plot_gauss()

    Returns
    -------
    fig : Figure
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object

    """

    x_fit = np.linspace(x[0], x[-1], 250) #make x data for fit

    labels = {'title'  : 'Gaussian Fit',
              'xlabel' : 'x variable', 
              'ylabel' : 'Count'} 
    if labeldict is not None:
        for key in labeldict:
            labels[key] = labeldict[key]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(labels['title'])
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['ylabel'])
    ax.plot([],[], linestyle = ' ', label = f' μ = {fitparams[1]:.2f} $\pm$ {errors[1]:.3f}')
    ax.plot([],[], linestyle = ' ', label = f' σ = {fitparams[2]:.2f} $\pm$ {errors[2]:.3f}')
    ax.plot([],[], linestyle = ' ', label = f' A = {fitparams[0]:.2f} $\pm$ {errors[0]:.3f}')
    
    ax.hist(x, bins = bins, weights = y, histtype = 'step', linewidth = 1, label ='Raw Data', alpha = .9)
    ax.plot(x_fit, utils.gaussian_asym_background(x_fit, *fitparams), label = 'Fit')
    
    erfarg = (x_fit-fitparams[1])/(np.sqrt(2)*fitparams[2])
    b1 = fitparams[3]*(1-(1/2)*(1+erf(erfarg)))
    b2 = fitparams[4]*((1/2)*(1+erf(erfarg)))
    
    ax.plot(x_fit, b1, linestyle = '--', color='k', alpha = .3, label = f'Bkg 1 = {fitparams[3]:.2f} $\pm$ {errors[3]:.3f}')
    ax.plot(x_fit, b2, linestyle = '--', color='k',label = f'Bkg 2 = {fitparams[4]:.2f} $\pm$ {errors[4]:.3f}')

    ax.legend()
    ax.grid(True, linestyle = 'dashed')

    return fig, ax

def plot_n_gauss(x, y, bins, fitparams, labeldict=None, ax=None):
    """
    Helper function to plot and arbitrary number of Gaussians plus background fit

    Parameters
    ----------
    x : array_like
        Array of x data
    y : array_like
        Array of y data
    bins : array_like
        Array of binned data
    fitparams : tuple
        The best fit parameters from the fit
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are : 
            labels = {'title' : 'Histogram', 'xlabel' : 'variable', 'ylabel' : 'Count'}
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to _plot_n_gauss()

    Returns
    -------
    fig : Figure
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object

    """

    n = int((len(fitparams)-1)/3)

    x_fit = np.linspace(x[0], x[-1], 250) #make x data for fit

    labels = {'title'  : 'Gaussian Fit',
              'xlabel' : 'x variable', 
              'ylabel' : 'Count'} 
    for ii in range(n):
        labels[f'peak{ii+1}'] = f'Peak{ii+1}'
    if labeldict is not None:
        for key in labeldict:
            labels[key] = labeldict[key]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))
    else:
        fig = None


    ax.set_title(labels['title'])
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['ylabel'])
    ax.hist(x, bins = bins, weights = y, histtype = 'step', linewidth = 1, label ='Raw Data', alpha = .9)


    y_fits = utils.n_gauss(x_fit, fitparams, n)
    ax.plot(x_fit, y_fits.sum(axis = 0), label = 'Total Fit')
    for ii in range(y_fits.shape[0] - 1):
        ax.plot(x_fit, y_fits[ii], alpha = .5, label = labels[f'peak{ii+1}'])
    ax.plot(x_fit, y_fits[-1], alpha = .5, linestyle = '--', label = 'Background')
    ax.grid(True, linestyle = 'dashed')
    ax.set_ylim(1, y.max()*1.05)
    ax.legend()

    return fig, ax    
