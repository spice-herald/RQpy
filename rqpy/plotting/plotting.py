import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


__all__ = ["hist", "scatter", "densityplot"]


def hist(arr, nbins='sqrt', xlims=None, cutold=None, cutnew=None, lgcrawdata=True, 
         lgceff=True, lgclegend=True, labeldict=None, ax=None):
    """
    Function to plot histogram of RQ data. The bins are set such that all bins have the same size
    as the raw data
    
    Parameters
    ----------
    arr : array_like
        Array of values to be binned and plotted
    nbins : int, str, optional
        This is the same as plt.hist() bins parameter. Defaults is 'sqrt'.
    xlims : list of float, optional
        The xlimits of the histogram. This is passed to plt.hist() range parameter.
    cutold : array of bool, optional
        Mask of values to be plotted
    cutnew : array of bool, optional
        Mask of values to be plotted. This mask is added to cutold if cutold is not None. 
    lgcrawdata : bool, optional
        If True, the raw data is plotted
    lgceff : bool, optional
        If True, the cut efficiencies are printed in the legend. The total eff will be the sum of all the 
        cuts divided by the length of the data. The current cut eff will be the sum of the current cut 
        divided by the sum of all the previous cuts, if any.
    lgclegend : bool, optional
        If True, the legend is plotted.
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are : 
            labels = {'title' : 'Histogram', 'xlabel' : 'variable', 'ylabel' : 'Count', 
            'cutnew' : 'current', 'cutold' : 'previous'}
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to histrq()
    ax : axes.Axes object, optional
        Option to pass an existing Matplotlib Axes object to plot over, if it already exists.
    
    Returns
    -------
    fig : Figure
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object
        
    """
    
    labels = {'title'  : 'Histogram', 
              'xlabel' : 'variable', 
              'ylabel' : 'Count', 
              'cutnew' : 'current', 
              'cutold' : 'previous'}
    
    if labeldict is not None:
        for key in labeldict:
            labels[key] = labeldict[key]
    
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
                                    label='full data', linewidth=2, color='b')
        else:
            hist, bins, _ = ax.hist(arr, bins=nbins, range=xlims, histtype='step', 
                                    label='full data', linewidth=2, color='b')            
    if cutold is not None:
        oldsum = cutold.sum()
        if cutnew is None:
            cuteff = oldsum/cutold.shape[0]
            cutefftot = cuteff
        if lgcrawdata:
            nbins = bins
        label = f"Data passing {labels['cutold']} cut"
        if xlims is not None:
            ax.hist(arr[cutold], bins=nbins, range=xlims, histtype='step', 
                    label=label, linewidth=2, color='r')
        else:
            ax.hist(arr[cutold], bins=nbins, histtype='step', 
                    label=label, linewidth=2, color='r')
    if cutnew is not None:
        newsum = cutnew.sum()
        if cutold is not None:
            cutnew = cutnew & cutold
            cuteff = cutnew.sum()/oldsum
            cutefftot = cutnew.sum()/cutnew.shape[0]
        else:
            cuteff = newsum/cutnew.shape[0]
            cutefftot = cuteff
        if lgcrawdata:
            nbins = bins
        if lgceff:
            label = f"Data passing {labels['cutnew']} cut, eff :  {cuteff:.3f}"
        else:
            label = f"Data passing {labels['cutnew']} cut "
        if xlims is not None:
            ax.hist(arr[cutnew], bins=nbins, range=xlims, histtype='step', 
                    linewidth=2, color='g', label=label)
        else:
            ax.hist(arr[cutnew], bins=nbins, histtype='step', 
                    linewidth=2, color='g', label=label)
    elif (cutnew is None) & (cutold is None):
        cuteff = 1
        cutefftot = 1
    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    
    if lgceff:
        ax.plot([], [], linestyle=' ', label=f'Efficiency of total cut: {cutefftot:.3f}')
    if lgclegend:
        ax.legend()
    return fig, ax
    



def scatter(xvals, yvals, xlims=None, ylims=None, cutold=None, cutnew=None, 
            lgcrawdata=True, lgceff=True, lgclegend=True, labeldict=None, ms=1, a=.3, ax=None):
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
    cutold : array of bool, optional
        Mask of values to be plotted
    cutnew : array of bool, optional
        Mask of values to be plotted. This mask is added to cutold if cutold is not None. 
    lgcrawdata : bool, optional
        If True, the raw data is plotted
    lgceff : bool, optional
        If True, the cut efficiencies are printed in the legend. The total eff will be the sum of all the 
        cuts divided by the length of the data. The current cut eff will be the sum of the current cut 
        divided by the sum of all the previous cuts, if any.
    lgclegend : bool, optional
        If True, the legend is plotted.
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are : 
            labels = {'title' : 'Histogram', 'xlabel' : 'variable', 'ylabel' : 'Count', 
            'cutnew' : 'current', 'cutold' : 'previous'}
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to histrq()
    ms : float, optional
        The size of each marker in the scatter plot. Default is 1
    a : float, optional
        The opacity of the markers in the scatter plot, i.e. alpha. Default is 0.3
    ax : axes.Axes object, optional
        Option to pass an existing Matplotlib Axes object to plot over, if it already exists.
    
    Returns
    -------
    fig : Figure
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object
        
    """

    labels = {'title'  : 'Scatter Plot',
              'xlabel' : 'x variable', 
              'ylabel' : 'y variable', 
              'cutnew' : 'current', 
              'cutold' : 'previous'}
    
    if labeldict is not None:
        for key in labeldict:
            labels[key] = labeldict[key]
    
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
    
    if lgcrawdata and cutold is not None: 
        ax.scatter(xvals[limitcut & ~cutold], yvals[limitcut & ~cutold], 
                   label='Full Data', c='b', s=ms, alpha=a)
    elif lgcrawdata and cutnew is not None: 
        ax.scatter(xvals[limitcut & ~cutnew], yvals[limitcut & ~cutnew], 
                   label='Full Data', c='b', s=ms, alpha=a)
    elif lgcrawdata:
        ax.scatter(xvals[limitcut], yvals[limitcut], 
                   label='Full Data', c='b', s=ms, alpha=a)
        
    if cutold is not None:
        oldsum = cutold.sum()
        if cutnew is None:
            cuteff = cutold.sum()/cutold.shape[0]
            cutefftot = cuteff
            
        label = f"Data passing {labels['cutold']} cut"
        if cutnew is None:
            ax.scatter(xvals[cutold & limitcut], yvals[cutold & limitcut], 
                       label=label, c='r', s=ms, alpha=a)
        else: 
            ax.scatter(xvals[cutold & limitcut & ~cutnew], yvals[cutold & limitcut & ~cutnew], 
                       label=label, c='r', s=ms, alpha=a)
        
    if cutnew is not None:
        newsum = cutnew.sum()
        if cutold is not None:
            cutnew = cutnew & cutold
            cuteff = cutnew.sum()/oldsum
            cutefftot = cutnew.sum()/cutnew.shape[0]
        else:
            cuteff = newsum/cutnew.shape[0]
            cutefftot = cuteff
            
        if lgceff:
            label = f"Data passing {labels['cutnew']} cut, eff : {cuteff:.3f}"
        else:
            label = f"Data passing {labels['cutnew']} cut"
            
        ax.scatter(xvals[cutnew & limitcut], yvals[cutnew & limitcut], 
                   label=label, c='g', s=ms, alpha=a)
    
    elif (cutnew is None) & (cutold is None):
        cuteff = 1
        cutefftot = 1
        
    if xlims is None:
        if lgcrawdata:
            xrange = xvals.max()-xvals.min()
            ax.set_xlim([xvals.min()-0.05*xrange, xvals.max()+0.05*xrange])
        elif cutold is not None:
            xrange = xvals[cutold].max()-xvals[cutold].min()
            ax.set_xlim([xvals[cutold].min()-0.05*xrange, xvals[cutold].max()+0.05*xrange])
        elif cutnew is not None:
            xrange = xvals[cutnew].max()-xvals[cutnew].min()
            ax.set_xlim([xvals[cutnew].min()-0.05*xrange, xvals[cutnew].max()+0.05*xrange])
    else:
        ax.set_xlim(xlims)
        
    if ylims is None:
        if lgcrawdata:
            yrange = yvals.max()-yvals.min()
            ax.set_ylim([yvals.min()-0.05*yrange, yvals.max()+0.05*yrange])
        elif cutold is not None:
            yrange = yvals[cutold].max()-yvals[cutold].min()
            ax.set_ylim([yvals[cutold].min()-0.05*yrange, yvals[cutold].max()+0.05*yrange])
        elif cutnew is not None:
            yrange = yvals[cutnew].max()-yvals[cutnew].min()
            ax.set_ylim([yvals[cutnew].min()-0.05*yrange, yvals[cutnew].max()+0.05*yrange])
        
    else:
        ax.set_ylim(ylims)
        
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.grid(True)
    
    if lgceff:
        ax.plot([], [], linestyle=' ', label=f'Efficiency of total cut: {cutefftot:.3f}')
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
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to histrq()
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
    fig.colorbar(cax[-1], label = 'Density of Data')
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax.grid(True)

    return fig, ax

