import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import rqpy as rp
from rqpy import utils


__all__ = ["hist", "scatter", "densityplot", "passageplot", "plot_gauss", "plot_n_gauss", 
           "plot_saturation_correction", "_make_iv_noiseplots", "_plot_energy_res_vs_bias", 
           "_plot_n_noise", "_plot_sc_noise", "_plot_rload_rn_qetbias", "_plot_fit_integral_ofamp"]


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
    
    if cuts is None:
        cuts = []
    elif not isinstance(cuts, list):
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
        if nbins=="sqrt":
            nbins = len(bins)
            
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
            
        if xlims is None:
            hist, bins, _  = ax.hist(arr[ctemp], bins=nbins, histtype='step', 
                                     label=label, linewidth=2, color=colors[ii])
            xlims = (bins.min(), bins.max())
        else:
            hist, bins, _  = ax.hist(arr[ctemp], bins=nbins, range=xlims, histtype='step', 
                                     label=label, linewidth=2, color=colors[ii])
            
        if nbins=="sqrt":
            nbins = len(bins)
    
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

    if cuts is None:
        cuts = []
    elif not isinstance(cuts, list):
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
    
    if lgcrawdata and len(cuts) > 0: 
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
        if lgcrawdata and len(cuts)==0:
            xrange = xvals.max()-xvals.min()
            ax.set_xlim([xvals.min()-0.05*xrange, xvals.max()+0.05*xrange])
        elif len(cuts)>0:
            xrange = xvals[cuts[0]].max()-xvals[cuts[0]].min()
            ax.set_xlim([xvals[cuts[0]].min()-0.05*xrange, xvals[cuts[0]].max()+0.05*xrange])
    else:
        ax.set_xlim(xlims)
        
    if ylims is None:
        if lgcrawdata and len(cuts)==0:
            yrange = yvals.max()-yvals.min()
            ax.set_ylim([yvals.min()-0.05*yrange, yvals.max()+0.05*yrange])
        elif len(cuts)>0:
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

def passageplot(arr, cuts, basecut=None, nbins=100, lgcequaldensitybins=False, xlims=None, ylims=(0, 1),
                lgceff=True, lgclegend=True, labeldict=None, ax=None, cmap="viridis"):
    """
    Function to plot histogram of RQ data with multiple cuts.
    
    Parameters
    ----------
    arr : array_like
        Array of values to be binned and plotted
    cuts : list, optional
        List of masks of values to be plotted. The cuts will be applied in the order that 
        they are listed, such that any number of cuts can be plotted.
    basecut : NoneType, array_like, optional
        The base cut for comparison of the first cut in `cuts`. If left as None, then the 
        passage fraction is calculated using all of the inputted data for the first cut.
    nbins : int, str, optional
        This is the same as plt.hist() bins parameter. Defaults is 'sqrt'.
    lgcequaldensitybins : bool, optional
        If set to True, the bin widths are set such that each bin has the same number
        of data points within it. If left as False, then a constant bin width is used.
    xlims : list of float, optional
        The xlimits of the passage fraction plot. 
    ylims : list of float, optional
        This is passed to the plot as the y limits. Set to (0, 1) by default.
    lgceff : bool, optional
        If True, the total cut efficiencies are printed in the legend. 
    lgclegend : bool, optional
        If True, the legend is plotted.
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are: 
            labels = {'title' : 'Passage Fraction Plot', 
                      'xlabel' : 'variable', 
                      'ylabel' : 'Passage Fraction', 
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
    
    labels = {'title'  : 'Passage Fraction Plot', 
              'xlabel' : 'variable', 
              'ylabel' : 'Passage Fraction'}
    
    
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

    if basecut is None:
        basecut = np.ones(len(arr), dtype=bool)
            
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.1, 0.9, len(cuts)))

    ctemp = np.ones(len(arr), dtype=bool) & basecut
    
    for ii, cut in enumerate(cuts):
        oldsum = ctemp.sum()
        x_binned, passage_binned = rp.passage_fraction(arr, cut, basecut=ctemp, nbins=nbins,
                                                       lgcequaldensitybins=lgcequaldensitybins)
        ctemp = ctemp & cut
        newsum = ctemp.sum()
        cuteff = newsum/oldsum * 100
        label = f"Data passing {labels[f'cut{ii}']} cut"
        
        if lgceff:
            label+=f", Total Passage: {cuteff:.1f}%"
            
        if xlims is None:
            xlims = (x_binned.min()*0.9, x_binned.max()*1.1)
            
        ax.step(x_binned, passage_binned, where='mid', color=colors[ii], label=label)
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.grid(linestyle="dashed")
    
    if lgclegend:
        ax.legend(loc="best")
    
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


def _plot_fit_integral_ofamp(x, y, err, y_fit, sat_errors, linear_approx, linear_approx_errs, labeldict, ax):
     
    """
    Helper function to plot the fit for fit_integral_ofamp()
    
    Parameters
    ----------
    x : array
        Array of x data
    y : array
        Array of y data
    err : array-like
        The errors in the measured energy of the spectral peaks
    y_fit : array
        Array of y data from fit
    sat_errors : array
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
    fig : matrplotlib figure object
    
    ax : matplotlib axes object
    
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

    ax.plot(x_fit, y_fit, color = 'k',  label = f'Fit : $y = -b*ln(1-y/a)$ ({nsigma}σ bounds)')
    ax.fill_between(x_fit, y_fit+nsigma*sat_errors, y_fit-nsigma*sat_errors, color = 'k' , alpha= .5)
    
    ax.plot(x_fit, linear_approx*x_fit,zorder = 200, c = 'r', linestyle = '--', 
            label = f'Linear approximation ({nsigma}σ bounds) ')
    ax.fill_between(x_fit, linear_approx*x_fit+nsigma*linear_approx_errs,
                    linear_approx*x_fit-nsigma*linear_approx_errs, color = 'r' , alpha= .5)
    ax.legend()    

    
    
    
    
def _make_iv_noiseplots(IVanalysisOBJ, lgcsave=False):
    """
    Helper function to plot average noise/didv traces in time domain, as well as 
    corresponding noise PSDs, for all QET bias points in IV/dIdV sweep.

    Parameters
    ----------
    IVanalysisOBJ : rqpy.IVanalysis
         The IV analysis object that contains the data to use for plotting.
    lgcsave : bool, optional
        If True, all the plots will be saved in the a folder
        Avetrace_noise/ within the user specified directory

    Returns
    -------
    None

    """

    for (noiseind, noiserow), (didvind, didvrow) in zip(IVanalysisOBJ.df[IVanalysisOBJ.noiseinds].iterrows(), IVanalysisOBJ.df[IVanalysisOBJ.didvinds].iterrows()):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

        t = np.arange(0,len(noiserow.avgtrace))/noiserow.fs
        tdidv = np.arange(0, len(didvrow.avgtrace))/noiserow.fs
        axes[0].set_title(f"{noiserow.seriesnum} Avg Trace, QET bias = {noiserow.qetbias*1e6:.2f} $\mu A$")
        axes[0].plot(t*1e6, noiserow.avgtrace * 1e6, label=f"{self.chname} Noise", alpha=0.5)
        axes[0].plot(tdidv*1e6, didvrow.avgtrace * 1e6, label=f"{self.chname} dIdV", alpha=0.5)
        axes[0].grid(which="major")
        axes[0].grid(which="minor", linestyle="dotted", alpha=0.5)
        axes[0].tick_params(axis="both", direction="in", top=True, right=True, which="both")
        axes[0].set_ylabel("Current [μA]", fontsize = 14)
        axes[0].set_xlabel("Time [μs]", fontsize = 14)
        axes[0].legend()

        axes[1].loglog(noiserow.f, noiserow.psd**0.5 * 1e12, label=f"{self.chname} PSD")
        axes[1].set_title(f"{noiserow.seriesnum} PSD, QET bias = {noiserow.qetbias*1e6:.2f} $\mu A$")
        axes[1].grid(which="major")
        axes[1].grid(which="minor", linestyle="dotted", alpha=0.5)
        axes[1].set_ylim(1, 1e3)
        axes[1].tick_params(axis="both", direction="in", top=True, right=True, which="both")
        axes[1].set_ylabel(r"PSD [pA/$\sqrt{\mathrm{Hz}}$]", fontsize = 14)
        axes[1].set_xlabel("Frequency [Hz]", fontsize = 14)
        axes[1].legend()

        plt.tight_layout()
        if lgcsave:
            if not savepath.endswith('/'):
                savepath += '/'
            fullpath = f'{IVanalysisOBJ.figsavepath}avetrace_noise/'
            if not os.path.isdir(fullpath):
                os.makedirs(fullpath)

            plt.savefig(fullpath + f'{noiserow.qetbias*1e6:.2f}_didvnoise.png')
        plt.show()
            
def _plot_rload_rn_qetbias(IVanalysisOBJ, lgcsave, xlims_rl, ylims_rl, xlims_rn, ylims_rn):
    """
    Helper function to plot rload and rnormal as a function of
    QETbias from the didv fits of SC and Normal data for IVanalysis object.

    Parameters
    ----------
    IVanalysisOBJ : rqpy.IVanalysis
         The IV analysis object that contains the data to use for plotting.
    lgcsave : bool, optional
        If True, all the plots will be saved 
    xlims_rl : NoneType, tuple, optional
        Limits to be passed to ax.set_xlim()for the 
        rload plot
    ylims_rl : NoneType, tuple, optional
        Limits to be passed to ax.set_ylim() for the
        rload plot
    xlims_rn : NoneType, tuple, optional
        Limits to be passed to ax.set_xlim()for the 
        rtot plot
    ylims_rn : NoneType, tuple, optional
        Limits to be passed to ax.set_ylim() for the
        rtot plot
    

    Returns
    -------
    None

    """

    fig, axes = plt.subplots(1,2, figsize = (16,6))
    fig.suptitle("Rload and Rtot from dIdV Fits", fontsize = 18)
    
    if xlims_rl is not None:
        axes[0].set_xlim(xlims_rl)
    if ylims_rl is not None:
        axes[0].set_ylim(ylims_rl)
    if xlims_rn is not None:
        axes[1].set_xlim(xlims_rn)
    if ylims_rn is not None:
        axes[1].set_ylim(ylis_rn)

    axes[0].errorbar(IVanalysisOBJ.vb[0,0,IVanalysisOBJ.scinds]*1e6,
                     np.array(IVanalysisOBJ.rload_list)*1e3, 
                     yerr = IVanalysisOBJ.rshunt_err*1e3, linestyle = '', marker = '.', ms = 10)
    axes[0].grid(True, linestyle = 'dashed')
    axes[0].set_title('Rload vs Vbias', fontsize = 14)
    axes[0].set_ylabel(r'$R_ℓ$ [mΩ]', fontsize = 14)
    axes[0].set_xlabel(r'$V_{bias}$ [μV]', fontsize = 14)
    axes[0].tick_params(axis="both", direction="in", top=True, right=True, which="both")

    axes[1].errorbar(IVanalysisOBJ.vb[0,0,IVanalysisOBJ.norminds]*1e6,
                     np.array(IVanalysisOBJ.rtot_list)*1e3, 
                     yerr = IVanalysisOBJ.rshunt_err*1e3, linestyle = '', marker = '.', ms = 10)
    axes[1].grid(True, linestyle = 'dashed')
    axes[1].set_title('Rtotal vs Vbias', fontsize = 14)
    axes[1].set_ylabel(r'$R_{N} + R_ℓ$ [mΩ]', fontsize = 14)
    axes[1].set_xlabel(r'$V_{bias}$ [μV]', fontsize = 14)
    axes[1].tick_params(axis="both", direction="in", top=True, right=True, which="both")

    plt.tight_layout()
    if lgcsave:
        plt.savefig(IVanalysisOBJ.figsavepath + 'rload_rtot_variation.png')
            
            
def _plot_energy_res_vs_bias(r0s, energy_res, qets, optimum_r0, figsavepath, lgcsave,
                            xlims, ylims):
    """
    Helper function for the IVanalysis class to plot the expected energy resolution as 
    a function of QET bias and TES resistance.
    
    Parameters
    ----------
    r0s : array
        Array of r0 values
    energy_res : array
        Array of expected energy resolutions
    qets : array
        Array of QET bias values
    optimum_r0 : float
        The TES resistance corresponding to the 
        lowest energy resolution
    figsavepath : str
        Directory to save the figure
    lgcsave : bool
        If true, the figure is saved
    xlims : NoneType, tuple, optional
            Limits to be passed to ax.set_xlim()
    ylims : NoneType, tuple, optional
        Limits to be passed to ax.set_ylim()
        
    Returns
    -------
    None

    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
        
    ax.plot(r0s, energy_res, linestyle = ' ', marker = '.', ms = 10, c='g')
    ax.plot(r0s, energy_res, linestyle = '-', marker = ' ', alpha = .3, c='g')
    ax.grid(True, which = 'both', linestyle = '--')
    ax.set_xlabel('$R_0$ [mΩ]')
    ax.set_ylabel(r'$σ_E$ [eV]')
    ax2 = ax.twiny()
    ax2.plot(qets[::-1], energy_res, linestyle = ' ')
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom') 
    ax2.spines['bottom'].set_position(('outward', 36))
    ax2.set_xlabel('QET bias [μA]')
    ax3 = plt.gca()
    plt.draw()
    ax3.get_xticklabels()
    newlabels = [thing for thing in ax3.get_xticklabels()][::-1]
    ax2.set_xticklabels(newlabels)
    ax.axvline(optimum_r0, linestyle = '--', color = 'r', label = r'Optimum QET bias (minumum $σ_E$)')
    ax.set_title('Expected Energy Resolution vs QET bias and $R_0$')
    ax.legend()

    if lgcsave:
        plt.savefig(f'{figsavepath}energy_res_vs_bias.png')
        
        
        
def _plot_sc_noise(f, psd, noise_sim, qetbias, figsavepath, lgcsave, xlims, ylims):
    """
    Helper function to plot SC noise for IVanalysis class
    
    Parameters
    ----------
    f : array
        Array of frequency values
    psd : array
        One sided Power spectral density
    noise_sim : TESnoise object
        The noise simulation object
    qetbias : float
        Applied QET bias
    figsavepath : str
        Directory to save the figure
    lgcsave : bool
        If true, the figure is saved
    xlims : NoneType, tuple, optional
        Limits to be passed to ax.set_xlim()
    ylims : NoneType, tuple, optional
        Limits to be passed to ax.set_ylim()
    
    Returns
    -------
    None

    """

    f = f[1:]
    psd = psd[1:]
    fig, ax = plt.subplots(1,1, figsize=(11,6))
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
        
    ax.grid(True, linestyle = '--')
    ax.loglog(f, np.sqrt(psd), alpha = .5, label = 'Raw Data')
    ax.loglog(f, np.sqrt(noise_sim.s_isquid(f)), label = 'Squid+Electronics Noise')
    ax.loglog(f, np.sqrt(noise_sim.s_iloadsc(f)),label= 'Load Noise')
    ax.loglog(f, np.sqrt(noise_sim.s_itotsc(f)),label= 'Total Noise')
    ax.legend()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Input Referenced Current Noise [A/$\sqrt{\mathrm{Hz}}$]')
    ax.set_title(f'Normal State noise for QETbias: {qetbias*1e6} $\mu$A')
    
    if lgcsave:
        plt.savefig(f'{figsavepath}SC_noise_qetbias{qetbias}.png')
        
        
def _plot_n_noise(f, psd, noise_sim, qetbias, figsavepath, lgcsave, xlims, ylims):
    """
    Helper function to plot normal state noise for IVanalysis class
    
    Parameters
    ----------
    f : array
        Array of frequency values
    psd : array
        One sided Power spectral density
    noise_sim : TESnoise object
        The noise simulation object
    qetbias : float
        Applied QET bias
    figsavepath : str
        Directory to save the figure
    lgcsave : bool
        If true, the figure is saved
    xlims : NoneType, tuple, optional
        Limits to be passed to ax.set_xlim()
    ylims : NoneType, tuple, optional
        Limits to be passed to ax.set_ylim()    
    
    Returns
    -------
    None

    """   

    f = f[1:]
    psd = psd[1:]
    fig, ax = plt.subplots(1,1, figsize=(11,6))
    if xlims is not None:
        ax.set_xlim(xlims)
    if ylims is not None:
        ax.set_ylim(ylims)
        
    ax.grid(True, linestyle = '--')
    ax.loglog(f, np.sqrt(psd), alpha = .5, label = 'Raw Data')
    ax.loglog(f, np.sqrt(noise_sim.s_isquid(f)), label = 'Squid+Electronics Noise')
    ax.loglog(f, np.sqrt(noise_sim.s_itesnormal(f)),label= 'TES johnson Noise')
    ax.loglog(f, np.sqrt(noise_sim.s_iloadnormal(f)),label= 'Load Noise')
    ax.loglog(f, np.sqrt(noise_sim.s_itotnormal(f)),label= 'Total Noise')
    ax.legend()
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Input Referenced Current Noise [A/$\sqrt{\mathrm{Hz}}$]')
    ax.set_title(f'Normal State noise for QETbias: {qetbias*1e6} $\mu$A')
    
    if lgcsave:
        plt.savefig(f'{figsavepath}Normal_noise_qetbias{qetbias}.png')
