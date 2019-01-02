import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from rqpy import utils


__all__ = ["hist", "scatter", "densityplot", "plot_gauss", "plot_n_gauss", "plot_saturation_correction"]


def hist(arr, nbins='sqrt', xlims=None, cuts=None, lgcrawdata=True, 
         lgceff=True, lgclegend=True, labeldict=None, ax=None, cmap="viridis"):
    """
    Function to plot histogram of RQ data with multiple cuts.
    
    Parameters
    ----------
    arr : array_like
        Array of values to be binned and plotted
    nbins : int, str, optional
        This is the same as plt.hist() bins parameter. Defaults is 'sqrt'.
    xlims : list of float, optional
        The xlimits of the histogram. This is passed to plt.hist() range parameter.
    cuts : list, optional
        List of masks of values to be plotted. The cuts will be applied in the order that they are listed, 
        such that any number of cuts can be plotted
    lgcrawdata : bool, optional
        If True, the raw data is plotted
    lgceff : bool, optional
        If True, the cut efficiencies are printed in the legend. 
    lgclegend : bool, optional
        If True, the legend is plotted.
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are: 
            labels = {'title' : 'Histogram', 
                      'xlabel' : 'variable', 
                      'ylabel' : 'Count', 
                      'cut0' : '1st', 
                      'cut1' : '2nd', 
                      ...}
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to hist()
    ax : axes.Axes object, optional
        Option to pass an existing Matplotlib Axes object to plot over, if it already exists.
    cmap : str, optional
        The colormap to use for plotting each cut. Default is 'viridis'.
    
    Returns
    -------
    fig : Figure
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object
        
    """
    
    if not isinstance(cuts, list):
        cuts = [cuts]
    
    labels = {'title'  : 'Histogram', 
              'xlabel' : 'variable', 
              'ylabel' : 'Count'}
    
    
    for ii in range(len(cuts)):
        
        num_str = str(ii+1)
        
        if num_str[-1]=='1':
            num_str+="st"
        elif num_str[-1]=='2':
            num_str+="nd"
        elif num_str[-1]=='3':
            num_str+="rd"
        else:
            num_str+="th"
        
        labels[f"cut{ii}"] = num_str
        
    if labeldict is not None:
        labels.update(labeldict)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = None
        
    ax.set_title(labels['title'])
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['ylabel'])

    if lgcrawdata:
        if xlims is None:
            hist, bins, _ = ax.hist(arr, bins=nbins, histtype='step', 
                                    label='Full data', linewidth=2, color=plt.cm.get_cmap(cmap)(0))
            xlims = (bins.min(), bins.max())
        else:
            hist, bins, _ = ax.hist(arr, bins=nbins, range=xlims, histtype='step', 
                                    label='Full data', linewidth=2, color=plt.cm.get_cmap(cmap)(0))
            
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.1, 0.9, len(cuts)))
        
    ctemp = np.ones(len(arr), dtype=bool)
    
    for ii, cut in enumerate(cuts):
        oldsum = ctemp.sum()
        ctemp = ctemp & cut
        newsum = ctemp.sum()
        cuteff = newsum/oldsum * 100
        label = f"Data passing {labels[f'cut{ii}']} cut"
        
        if lgceff:
            label+=f", Eff = {cuteff:.1f}%"
            
        if xlims is not None:
            ax.hist(arr[ctemp], bins=nbins, range=xlims, histtype='step', 
                    label=label, linewidth=2, color=colors[ii])
        else:
            res = ax.hist(arr[ctemp], bins=nbins, histtype='step', 
                    label=label, linewidth=2, color=colors[ii])
            xlims = (res[1].min(), res[1].max())
    
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.grid(linestyle="dashed")
    
    if lgclegend:
        ax.legend(loc="best")
        
    return fig, ax
    

def scatter(xvals, yvals, xlims=None, ylims=None, cuts=None, lgcrawdata=True, lgceff=True, 
            lgclegend=True, labeldict=None, ms=1, a=.3, ax=None, cmap="viridis"):
    """
    Function to plot RQ data as a scatter plot.
    
    Parameters
    ----------
    xvals : array_like
        Array of x values to be plotted
    yvals : array_like
        Array of y values to be plotted
    xlims : list of float, optional
        This is passed to the plot as the x limits. Automatically determined from range of data
        if not set.
    ylims : list of float, optional
        This is passed to the plot as the y limits. Automatically determined from range of data
        if not set.
    cuts : list, optional
        List of masks of values to be plotted. The cuts will be applied in the order that they are listed,
        such that any number of cuts can be plotted
    lgcrawdata : bool, optional
        If True, the raw data is plotted
    lgceff : bool, optional
        If True, the efficiencies of each cut, with respect to the data that survived 
        the previous cut, are printed in the legend. 
    lgclegend : bool, optional
        If True, the legend is included in the plot.
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are: 
            labels = {'title' : 'Scatter Plot', 
                      'xlabel' : 'x variable', 
                      'ylabel' : 'y variable', 
                      'cut0' : '1st', 
                      'cut1' : '2nd', 
                      ...}
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to scatter()
    ms : float, optional
        The size of each marker in the scatter plot. Default is 1
    a : float, optional
        The opacity of the markers in the scatter plot, i.e. alpha. Default is 0.3
    ax : axes.Axes object, optional
        Option to pass an existing Matplotlib Axes object to plot over, if it already exists.
    cmap : str, optional
        The colormap to use for plotting each cut. Default is 'viridis'.
    
    Returns
    -------
    fig : Figure
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object
        
    """

    if not isinstance(cuts, list):
        cuts = [cuts]
    
    labels = {'title'  : 'Scatter Plot',
              'xlabel' : 'x variable', 
              'ylabel' : 'y variable'}
    
    for ii in range(len(cuts)):
        
        num_str = str(ii+1)
        
        if num_str[-1]=='1':
            num_str+="st"
        elif num_str[-1]=='2':
            num_str+="nd"
        elif num_str[-1]=='3':
            num_str+="rd"
        else:
            num_str+="th"
        
        labels[f"cut{ii}"] = num_str
        
    if labeldict is not None:
        labels.update(labeldict)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = None
    
    ax.set_title(labels['title'])
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['ylabel'])
    
    if xlims is not None:
        xlimitcut = (xvals>xlims[0]) & (xvals<xlims[1])
    else:
        xlimitcut = np.ones(len(xvals), dtype=bool)
        
    if ylims is not None:
        ylimitcut = (yvals>ylims[0]) & (yvals<ylims[1])
    else:
        ylimitcut = np.ones(len(yvals), dtype=bool)
    
    limitcut = xlimitcut & ylimitcut
    
    if lgcrawdata and cuts is not None: 
        ax.scatter(xvals[limitcut & ~cuts[0]], yvals[limitcut & ~cuts[0]], 
                   label='Full Data', c='b', s=ms, alpha=a)
    elif lgcrawdata:
        ax.scatter(xvals[limitcut], yvals[limitcut], 
                   label='Full Data', c='b', s=ms, alpha=a)
    
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.1, 0.9, len(cuts)))
        
    ctemp = np.ones(len(xvals), dtype=bool)
    
    for ii, cut in enumerate(cuts):
        oldsum = ctemp.sum()
        ctemp = ctemp & cut
        newsum = ctemp.sum()
        cuteff = newsum/oldsum * 100
        label = f"Data passing {labels[f'cut{ii}']} cut"
        
        if lgceff:
            label+=f", Eff = {cuteff:.1f}%"
            
        cplot = ctemp & limitcut
        
        if ii+1<len(cuts):
            cplot = cplot & ~cuts[ii+1]
        
        ax.scatter(xvals[cplot], yvals[cplot], 
                   label=label, c=colors[ii], s=ms, alpha=a)
        
    if xlims is None:
        if lgcrawdata and cuts is None:
            xrange = xvals.max()-xvals.min()
            ax.set_xlim([xvals.min()-0.05*xrange, xvals.max()+0.05*xrange])
        elif cuts is not None:
            xrange = xvals[cuts[0]].max()-xvals[cuts[0]].min()
            ax.set_xlim([xvals[cuts[0]].min()-0.05*xrange, xvals[cuts[0]].max()+0.05*xrange])
    else:
        ax.set_xlim(xlims)
        
    if ylims is None:
        if lgcrawdata and cuts is None:
            yrange = yvals.max()-yvals.min()
            ax.set_ylim([yvals.min()-0.05*yrange, yvals.max()+0.05*yrange])
        elif cuts is not None:
            yrange = yvals[cuts[0]].max()-yvals[cuts[0]].min()
            ax.set_ylim([yvals[cuts[0]].min()-0.05*yrange, yvals[cuts[0]].max()+0.05*yrange])
    else:
        ax.set_ylim(ylims)
        
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.grid(linestyle="dashed")
    
    if lgclegend:
        ax.legend(markerscale=6, framealpha=.9)
    
    return fig, ax



def densityplot(xvals, yvals, xlims=None, ylims=None, nbins = (500,500), cut=None, 
                labeldict=None, lgclognorm = True, ax=None):
    """
    Function to plot RQ data as a density plot.
    
    Parameters
    ----------
    xvals : array_like
        Array of x values to be plotted
    yvals : array_like
        Array of y values to be plotted
    xlims : list of float, optional
        This is passed to the plot as the x limits. Automatically determined from range of data
        if not set.
    ylims : list of float, optional
        This is passed to the plot as the y limits. Automatically determined from range of data
        if not set.
    nbins : tuple, optional
        The number of bins to use to make the 2d histogram (nx, ny).
    cut : array of bool, optional
        Mask of values to be plotted
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are : 
            labels = {'title' : 'Histogram', 'xlabel' : 'variable', 'ylabel' : 'Count'}
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to densityplot()
    lgclognorm : bool, optional
        If True (default), the color normilization for the density will be log scaled, rather 
        than linear
    ax : axes.Axes object, optional
        Option to pass an existing Matplotlib Axes object to plot over, if it already exists.
    
    Returns
    -------
    fig : Figure
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object
        
    """
    
    labels = {'title'  : 'Density Plot',
              'xlabel' : 'x variable', 
              'ylabel' : 'y variable'}
    
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
    
    if xlims is not None:
        xlimitcut = (xvals>xlims[0]) & (xvals<xlims[1])
    else:
        xlimitcut = np.ones(len(xvals), dtype=bool)
    if ylims is not None:
        ylimitcut = (yvals>ylims[0]) & (yvals<ylims[1])
    else:
        ylimitcut = np.ones(len(yvals), dtype=bool)

    limitcut = xlimitcut & ylimitcut
    
    if cut is None:
        cut = np.ones(shape = xvals.shape, dtype=bool)

    cax = ax.hist2d(xvals[limitcut & cut], yvals[limitcut & cut], bins = nbins, 
              norm = colors.LogNorm(), cmap = 'icefire')
    cbar = fig.colorbar(cax[-1], label = 'Density of Data')
    cbar.ax.tick_params(direction="in")
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.grid(linestyle="dashed")

    return fig, ax

def plot_gauss(x, bins, y, fitparams, errors, background, labeldict=None):
    """
    Hidden helper function to plot Gaussian plus background fits
    
    Parameters
    ----------
    x : array
        Array of x data
    bins : array
        Array of binned data
    y : array
        Array of y data
    fitparams : tuple
        The best fit parameters from the fit
    errors : tuple
        The unccertainy in the best fit parameters
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
    
def plot_n_gauss(x, y, bins, fitparams, labeldict=None, ax=None):
    """
    Helper function to plot and arbitrary number of Gaussians plus background fit
    
    Parameters
    ----------
    x : ndarray
        Array of x data
    y : ndarray
        Array of y data
    bins : ndarray
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



def plot_saturation_correction(x, y, yerr, popt, pcov, labeldict, ax = None):
    
    """
    Helper function to plot the fit for the saturation correction
    
    Parameters
    ----------
    x : array
        Array of x data
    y : array
        Array of y data
    yerr : array-like
        The errors in the measured energy of the spectral peaks
    guess : array-like
        Array of initial guess parameters (a,b) to be passed to saturation_func()
    popt : array
        Array of best fit parameters from fit_saturation()
    pcov : array
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
    fig : matrplotlib figure object
    
    ax : matplotlib axes object
    
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

    
