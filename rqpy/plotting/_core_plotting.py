import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as clrs
from matplotlib.patches import Ellipse
from scipy import stats
import rqpy as rp
import types


__all__ = [
    "hist",
    "scatter",
    "passageplot",
    "densityplot",
    "RatePlot",
    "conf_ellipse",
]


def hist(arr, nbins='auto', xlims=None, cuts=None, lgcrawdata=True,
         lgceff=True, lgclegend=True, labeldict=None, ax=None, cmap="viridis"):
    """
    Function to plot histogram of RQ data with multiple cuts.

    Parameters
    ----------
    arr : array_like
        Array of values to be binned and plotted
    nbins : int, str, optional
        This is the same as `plt.hist` bins parameter, where the number of bins, an array of
        bin edges, or a string can be passed. Default is 'auto'. See `numpy.histogram_bin_edges`
        for more information for different strings that can be passed.
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

    labels = {
        'title' : 'Histogram',
        'xlabel' : 'variable',
        'ylabel' : 'Count',
    }

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

    bins = None

    ax.set_title(labels['title'])
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['ylabel'])

    if lgcrawdata:
        if bins is None:
            bins = np.histogram_bin_edges(arr, bins=nbins, range=xlims)

        hist, _, _ = ax.hist(
            arr,
            bins=bins,
            histtype='step',
            label='Full data',
            linewidth=2,
            color=plt.cm.get_cmap(cmap)(0),
        )

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

        if bins is None:
            bins = np.histogram_bin_edges(arr[ctemp], bins=nbins, range=xlims)

        hist, _, _  = ax.hist(
            arr[ctemp],
            bins=bins,
            histtype='step',
            label=label,
            linewidth=2,
            color=colors[ii],
        )

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

    labels = {
        'title' : 'Scatter Plot',
        'xlabel' : 'x variable',
        'ylabel' : 'y variable',
    }

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
        ax.scatter(
            xvals[limitcut & ~cuts[0]], yvals[limitcut & ~cuts[0]],
            label='Full Data',
            c='b',
            s=ms,
            alpha=a,
        )

    elif lgcrawdata:
        ax.scatter(
            xvals[limitcut],
            yvals[limitcut],
            label='Full Data',
            c='b',
            s=ms,
            alpha=a,
        )

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

        ax.scatter(
            xvals[cplot],
            yvals[cplot],
            label=label,
            c=colors[ii][np.newaxis,...],
            s=ms,
            alpha=a,
        )

    if xlims is None:
        if lgcrawdata and len(cuts)==0:
            xrange = xvals.max() - xvals.min()
            ax.set_xlim([xvals.min() - 0.05 * xrange, xvals.max() + 0.05 * xrange])
        elif len(cuts)>0:
            xrange = xvals[cuts[0]].max() - xvals[cuts[0]].min()
            ax.set_xlim([xvals[cuts[0]].min() - 0.05 * xrange, xvals[cuts[0]].max() + 0.05 * xrange])
    else:
        ax.set_xlim(xlims)

    if ylims is None:
        if lgcrawdata and len(cuts)==0:
            yrange = yvals.max() - yvals.min()
            ax.set_ylim([yvals.min() - 0.05 * yrange, yvals.max() + 0.05 * yrange])
        elif len(cuts)>0:
            yrange = yvals[cuts[0]].max() - yvals[cuts[0]].min()
            ax.set_ylim([yvals[cuts[0]].min() - 0.05 * yrange, yvals[cuts[0]].max() + 0.05 * yrange])
    else:
        ax.set_ylim(ylims)

    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.grid(linestyle="dashed")

    if lgclegend:
        ax.legend(markerscale=6, framealpha=.9, loc='upper left')

    return fig, ax

def passageplot(arr, cuts, basecut=None, nbins=100, lgcequaldensitybins=False, xlims=None, ylims=(0, 1),
                lgceff=True, lgclegend=True, labeldict=None, ax=None, cmap="viridis",
                showerrorbar=False, nsigmaerrorbar=1):
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
    showerrorbar : bool, optional
        Boolean flag for also plotting the error bars for the passage fraction in each bin.
        Default is False.
    nsigmaerrorbar : float, optional
        The number of sigma to show for the error bars if `showerrorbar` is True. Default is 1.

    Returns
    -------
    fig : Figure
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object

    """

    if not isinstance(cuts, list):
        cuts = [cuts]

    labels = {
        'title' : 'Passage Fraction Plot',
        'xlabel' : 'variable',
        'ylabel' : 'Passage Fraction',
    }

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

    if xlims is None:
        xlimitcut = np.ones(len(arr), dtype=bool)
    else:
        xlimitcut = rp.inrange(arr, xlims[0], xlims[1])

    for ii, cut in enumerate(cuts):
        oldsum = ctemp.sum()

        if ii==0:
            passage_output = rp.passage_fraction(
                arr,
                cut,
                basecut=ctemp & xlimitcut,
                nbins=nbins,
                lgcequaldensitybins=lgcequaldensitybins,
            )
        else:
            passage_output = rp.passage_fraction(
                arr,
                cut,
                basecut=ctemp & xlimitcut,
                nbins=x_binned,
            )

        x_binned = passage_output[0]
        passage_binned = passage_output[1]

        ctemp = ctemp & cut
        newsum = ctemp.sum()
        cuteff = newsum/oldsum * 100
        label = f"Data passing {labels[f'cut{ii}']} cut"

        if showerrorbar:
            label += f" $\pm$ {nsigmaerrorbar}$\sigma$"

        if lgceff:
            label+=f", Total Passage: {cuteff:.1f}%"

        if xlims is None:
            xlims = (x_binned.min()*0.9, x_binned.max()*1.1)

        bin_centers = (x_binned[1:]+x_binned[:-1])/2

        ax.hist(
            bin_centers,
            bins=x_binned,
            weights=passage_binned,
            histtype='step',
            color=colors[ii],
            label=label,
            linewidth=2,
        )

        if showerrorbar:
            passage_binned_biased = passage_output[2]
            passage_binned_err = passage_output[3]

            err_top = passage_binned_biased + passage_binned_err * nsigmaerrorbar
            err_bottom = passage_binned_biased - passage_binned_err * nsigmaerrorbar

            err_top = np.pad(err_top, (0, 1), mode='constant', constant_values=(0, err_top[-1]))
            err_bottom = np.pad(err_bottom, (0, 1), mode='constant', constant_values=(0, err_bottom[-1]))

            ax.fill_between(
                x_binned,
                err_top,
                y2=err_bottom,
                step='post',
                linewidth=1,
                alpha=0.5,
                color=colors[ii],
            )

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.grid(linestyle="dashed")

    if lgclegend:
        ax.legend(loc="best")

    return fig, ax


def densityplot(xvals, yvals, xlims=None, ylims=None, nbins = (500,500), cut=None, labeldict=None,
                lgclognorm=True, ax=None, cmap='icefire', plot_cut_data=False, basecut=None):
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
        Dictionary to overwrite the labels of the plot. defaults are:
            labels = {
                'title' : 'Histogram',
                'xlabel' : 'variable',
                'ylabel' : 'Count',
                'basecut' : '',
            }
        Ex: to change just the title, pass: labeldict = {'title' : 'new title'}, to densityplot()
    lgclognorm : bool, optional
        If True (default), the color normilization for the density will be log scaled, rather
        than linear.
    ax : axes.Axes object, optional
        Option to pass an existing Matplotlib Axes object to plot over, if it already exists.
    cmap : str, optional
        The colormap to use for plotting each cut. Default is 'icefire'.
    plot_cut_data : bool, optional
        Boolean value for whether or not to plot the data cut by `cut`. If only a subset of the original
        data is desired to be plotted, then use `basecut` to trim the plotted data.
    basecut : array of bool, optional
        The base cut for comparison of `cut`. If left as None, then this is set as an array of
        Trues of the same length as the input data. Only used if `plot_cut_data` is True.

    Returns
    -------
    fig : Figure
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object

    Raises
    ------
    ValueError
        If there are no `xvals` in the range specified by `xlims`.
        If there are no `yvals` in the range specified by `ylims`.

    """

    labels = {
        'title'  : 'Density Plot',
        'xlabel' : 'x variable',
        'ylabel' : 'y variable',
        'basecut' : '',
    }

    if labeldict is not None:
        for key in labeldict:
            labels[key] = labeldict[key]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))
    else:
        fig = plt.gcf()

    ax.set_title(labels['title'])
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['ylabel'])

    if cut is None:
        cut = np.ones(shape=xvals.shape, dtype=bool)

    if basecut is None:
        basecut = np.ones(shape=xvals.shape, dtype=bool)

    if xlims is not None:
        xlimitcut = (xvals > xlims[0]) & (xvals < xlims[1])
    else:
        xlims = (np.min(xvals[cut]), np.max(xvals[cut]))
        xlimitcut = np.ones(len(xvals), dtype=bool)
    if ylims is not None:
        ylimitcut = (yvals > ylims[0]) & (yvals < ylims[1])
    else:
        ylims = (np.min(yvals[cut]), np.max(yvals[cut]))
        ylimitcut = np.ones(len(yvals), dtype=bool)

    if np.sum(xlimitcut)==0:
        raise ValueError("There are no x values in the specified range.")
    if np.sum(ylimitcut)==0:
        raise ValueError("There are no y values in the specified range.")

    limitcut = xlimitcut & ylimitcut

    if lgclognorm:
        norm = clrs.LogNorm()
    else:
        norm = clrs.Normalize()

    cax = ax.hist2d(
        xvals[limitcut & cut],
        yvals[limitcut & cut],
        bins=nbins,
        range=(xlims, ylims),
        norm=norm,
        cmap=cmap,
    )

    if plot_cut_data:
        cmap_grey_arr = np.ones((1, 4))
        cmap_grey_arr[:, :-1] = 0.9
        cmap_grey = clrs.ListedColormap(cmap_grey_arr)

        ax.hist2d(
            xvals[limitcut & basecut],
            yvals[limitcut & basecut],
            bins=nbins,
            range=(xlims, ylims),
            norm=norm,
            cmap=cmap_grey,
            zorder=0,
        )

        if len(labels['basecut']) > 0:
            labels['basecut'] += ' '

        ax.scatter(
            [], [], color=cmap_grey(0), label=f"Data removed by {labels['basecut']}cut", marker='s',
        )
        ax.legend()

    cbar = fig.colorbar(cax[-1], label='Density of Data')
    cbar.ax.tick_params(which="both", direction="in")
    ax.tick_params(which="both", direction="in", right=True, top=True)
    ax.grid(linestyle="dashed")

    return fig, ax


class RQpyPlot(object):
    """
    Base class for making plots in RQpy.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object
    ax : matplotlib.axes.AxesSubplot
        Matplotlib Axes object

    """

    def __init__(self, figsize=(10, 6)):
        """
        Initialization of the RQpyPlot base class.

        Parameters
        ----------
        figsize : tuple, optional
            Width and height of the figure in inches. Default is (10, 6).

        """

        self.fig, self.ax = plt.subplots(figsize=figsize)

        self.ax.grid()
        self.ax.grid(which="minor", axis="both", linestyle="dotted")
        self.ax.tick_params(which="both", direction="in", right=True, top=True)


class RatePlot(RQpyPlot):
    """
    Class for making a plot of different dark matter spectra with the ability of comparing
    different data and with theoretical dR/dE curves.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib Figure object
    ax : matplotlib.axes.AxesSubplot
        Matplotlib Axes object
    _energy_range : array_like
        The energy range of the plot in keV.
    _spectrum_cmap : str
        The colormap to use for plotting each spectra from data.
    _drde_cmap : str
        The colormap to use for plotting each theoretical drde curve using the WIMP model.

    """

    def __init__(self, energy_range, spectrum_cmap="inferno", drde_cmap="viridis", figsize=(10, 6)):
        """
        Initialization of the RatePlot class for plotting dark matter spectra.

        Parameters
        ----------
        energy_range : array_like
            The energy range of the plot in keV.
        spectrum_cmap : str, optional
            The colormap to use for plotting each spectra from data. Default is "inferno".
        drde_cmap : str, optional
            The colormap to use for plotting each theoretical drde curve using the WIMP
            model. Default is "viridis".
        figsize : tuple, optional
            Width and height of the figure in inches. Default is (10, 6).

        Returns
        -------
        None

        """

        self._energy_range = energy_range
        self._spectrum_cmap = spectrum_cmap
        self._drde_cmap = drde_cmap

        super().__init__(figsize=figsize)

        self.ax.set_yscale('log')
        self.ax.set_xlim(self._energy_range)
        self.ax.set_ylabel("$\partial R/\partial E_r$ [evts/keV/kg/day]")
        self.ax.set_xlabel("Energy [keV]")
        self.ax.set_title(
            f"Spectrum of Events from {self._energy_range[0]:.1f} to "
            f"{self._energy_range[1]:.1f} keV"
        )


    def _update_colors(self, linestyle):
        """
        Helper method for updating the line colors whenever a new line is added.

        Parameters
        ----------
        linestyle : str
            The linestyle to update all the plot colors for. Should be "-" or "--".

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `linestyle` is not "-" or "--".

        Notes
        -----
        Given a linestyle, this method checks how many lines have the specified style.
        This method assumes that data curves have a solid linestyle and theoretical
        DM curves have a dashed linestyle.

        """

        if linestyle == "-":
            cmap = self._spectrum_cmap
        elif linestyle == "--":
            cmap = self._drde_cmap
        else:
            raise ValueError("The inputted linestyle is not supported.")

        n_lines = sum(line.get_linestyle() == linestyle for line in self.ax.lines)
        line_colors = plt.cm.get_cmap(cmap)(np.linspace(0.1, 0.9, n_lines))
        ii = 0

        for line in self.ax.lines:
            if line.get_linestyle() == linestyle:
                line.set_color(line_colors[ii])
                ii += 1

        self.ax.legend(loc="upper right")

    def add_data(self, energies, exposure, efficiency=None, label=None, nbins=100, **kwargs):
        """
        Method for plotting a single spectrum in evts/keV/kg/day from inputted data.

        Parameters
        ----------
        energies : array_like
            The energies of the events that will be used when plotting, in units of keV.
        exposure : float
            The exposure of DM search with respect to the inputted spectrum, in units
            of kg-days.
        efficiency : float, FunctionType, NoneType, optional
            The efficiency of the cuts that were applied to the inputted energies. This
            can be passed as a float, as a function of energy in keV, or left as None if
            no efficiency correction will be done.
        label : str, optional
            The label for this data to be used in the plot. If left as None, no label is added.
        nbins : int, optional
            The number of bins to use in the plot.
        kwargs
            The keyword arguments to pass to `matplotlib.pyplot.step`.

        Returns
        -------
        None

        """

        hist, bin_edges = np.histogram(energies, bins=nbins, range=self._energy_range)
        bin_cen = (bin_edges[:-1]+bin_edges[1:])/2
        rate = hist / np.diff(bin_cen).mean() / exposure

        if np.isscalar(efficiency):
            rate /= efficiency
        elif isinstance(efficiency, types.FunctionType):
            rate /= efficiency(bin_cen)

        self.ax.step(bin_cen, rate, where='mid', label=label, linestyle='-', **kwargs)
        self._update_colors("-")
        
    def add_drde(self, masses, sigmas, tm="Si", npoints=1000, res=None, gauss_width=10, **kwargs):
        """
        Method for plotting the expected dark matter spectrum for specified masses and
        cross sections in evts/keV/kg/day.

        Parameters
        ----------
        masses : float, array_like
            The dark matter mass at which to calculate/plot the expected differential
            scattering rate. Expected units are GeV.
        sigmas : float, array_like
            The dark matter cross section at which to calculate/plot the expected
            differential scattering rate. Expected units are cm^2.
        tm : str, int, optional
            The target material of the detector. Can be passed as either the atomic symbol, the
            atomic number, or the full name of the element. Default is 'Si'.
        npoints : int, optional
            The number of points to use in the dR/dE plot. Default is 1000.
        res : float, NoneType, optional
            The width of the gaussian (1 standard deviation) to be used to smear differential
            scatter rate in the plot. Should have units of keV. None by default, in which no
            smearing is done.
        gauss_width : float, optional
            If `res` is not None, this is the number of standard deviations of the Gaussian
            distribution that the smearing will go out to. Default is 10.
        kwargs
            The keyword arguments to pass to `matplotlib.pyplot.plot`.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `masses` and `sigmas` are not the same length.

        """

        if np.isscalar(masses):
            masses = [masses]
        if np.isscalar(sigmas):
            sigmas = [sigmas]
        if len(masses) != len(sigmas):
            raise ValueError("masses and sigmas must be the same length.")

        xvals = np.linspace(self._energy_range[0], self._energy_range[1], npoints)

        for m, sig in zip(masses, sigmas):
            drde = rp.limit.drde(xvals, m, sig, tm=tm)
            label = f"DM Mass = {m:.2f} GeV, σ = {sig:.2e} cm$^2$"
            if res is not None:
                drde = rp.limit.gauss_smear(xvals, drde, res, gauss_width=gauss_width)
                label += f"\nWith {gauss_width}$\sigma_E$ Smearing"

            self.ax.plot(
                xvals,
                drde,
                linestyle='--',
                label=label,
                **kwargs,
            )

        self._update_colors("--")

def conf_ellipse(mu, cov, conf=0.683, ax=None, **kwargs):
    """
    Draw a 2-D confidence level ellipse based on a mean, covariance matrix,
    and specified confidence level.

    Parameters
    ----------
    mu : array_like
        The x and y values of the mean, where the ellipse will be centered.
    cov : ndarray
        A 2-by-2 covariance matrix describing the relation of the x and y variables.
    conf : float
        The confidence level at which to draw the ellipse. Should be a value between
        0 and 1. Default is 0.683. See Notes for more information
    ax : axes.Axes object, NoneType, optional
        Option to pass an existing Matplotlib Axes object to plot over, if it already exists.
    **kwargs
        Keyword arguments to pass to `Ellipse`. See Notes for more information.

    Returns
    -------
    fig : Figure, NoneType
        Matplotlib Figure object. Set to None if ax is passed as a parameter.
    ax : axes.Axes object
        Matplotlib Axes object

    Raises
    ------
    ValueError
        If `conf` is not in the range [0, 1]

    Notes
    -----
    When deciding the value for `conf`, the standard frequentist statement about what this contour means is:

        "If the experiment is repeated many times with the same statistical analysis, then the
        contour (which will in general be different for each realization of the experiment) will
        define a region which contains the true value in 68.3% of the experiments."

    Note that the 68.3% confidence level contour in 2 dimensions is not the same as 1-sigma contour. The
    1-sigma contour for 2 dimensions (i.e. the value by which the chi^2 value increases by 1) contains
    the true value in 39.3% of the experiments.

    More discussion on multi-parameter errors can be found here:
        http://seal.web.cern.ch/seal/documents/minuit/mnerror.pdf

    The valid keyword arguments are below (taken from the Ellipse docstring). In this function, `fill` is
    defaulted to False and 'zorder' is defaulted so that the ellipse is be on top of previous plots.
        agg_filter: a filter function, which takes a (m, n, 3) float array and a dpi value, and returns a (m, n, 3) array
        alpha: float or None
        animated: bool
        antialiased: unknown
        capstyle: {'butt', 'round', 'projecting'}
        clip_box: `.Bbox`
        clip_on: bool
        clip_path: [(`~matplotlib.path.Path`, `.Transform`) | `.Patch` | None]
        color: color
        contains: callable
        edgecolor: color or None or 'auto'
        facecolor: color or None
        figure: `.Figure`
        fill: bool
        gid: str
        hatch: {'/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
        in_layout: bool
        joinstyle: {'miter', 'round', 'bevel'}
        label: object
        linestyle: {'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
        linewidth: float or None for default
        path_effects: `.AbstractPathEffect`
        picker: None or bool or float or callable
        rasterized: bool or None
        sketch_params: (scale: float, length: float, randomness: float)
        snap: bool or None
        transform: `.Transform`
        url: str
        visible: bool
        zorder: float

    """

    if isinstance(mu, np.ndarray):
        mu = mu.tolist()

    if not rp.inrange(conf, 0, 1):
        raise ValueError("conf should be in the range [0, 1]")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.grid()
        ax.grid(which="minor", axis="both", linestyle="dotted")
        ax.tick_params(which="both", direction="in", right=True, top=True)
        autoscale_axes = True
    else:
        fig = None
        autoscale_axes = False

    if 'fill' not in kwargs:
        kwargs['fill'] = False

    if 'zorder' not in kwargs and len(ax.lines + ax.collections) > 0:
        kwargs['zorder'] = max(lc.get_zorder() for lc in ax.lines + ax.collections) + 0.1

    a, v = np.linalg.eig(cov)
    v0 = v[:,0]
    v1 = v[:,1]

    theta = np.arctan2(v1[1], v1[0])

    quantile = stats.chi2.ppf(conf, 2)

    ell = Ellipse(
        mu,
        2 * (quantile * a[1])**0.5,
        2 * (quantile * a[0])**0.5,
        angle=theta * 180/np.pi,
        **kwargs,
    )

    ax_ell = ax.add_patch(ell)

    if autoscale_axes:
        ax.autoscale()

    return fig, ax
