import numpy as np
from scipy import integrate, interpolate, stats
import types
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


__all__ = [
    'pdf_sampling',
    'sample_from_data',
]


def pdf_sampling(function, xrange, nsamples=1000, npoints=10000):
    """
    Produces randomly sampled values based on the arbitrary PDF defined
    by `function`, done using inverse transform sampling.

    Parameters
    ----------
    function : FunctionType
        The 1D probability density function to be randomly sampled from.
    xrange : array_like
        A 1D array of length 2 that defines the range over which the PDF
        in `function` is defined. Outside of this range, it is assumed that
        the PDF is zero.
    nsamples : int, optional
        The number of random samples that we wish to create from the PDF
        defined by `function`.
    npoints : int, optional
        The number of points to use in the numerical integration to evaluate
        the CDF of `function`. This is also the number of points used in the
        interpolation of the inverse of the CDF.

    Returns
    -------
    rvs : ndarray
        The random samples that were taken from the inputted PDF defined by
        `function`. This is 1D array of length `nsamples`.

    Notes
    -----
    For a discussion of inverse transform sampling, see the Wikipedia page:
        https://en.wikipedia.org/wiki/Inverse_transform_sampling

    """

    if not isinstance(function, types.FunctionType):
        raise TypeError("Inputted variable function is not FunctionType.")

    x = np.linspace(xrange[0], xrange[1], num=npoints)
    pdf = function(x)

    cdf = integrate.cumtrapz(pdf, x=x, initial=0.0)
    cdf_normed = cdf / cdf[-1]

    inv_cdf = interpolate.interp1d(cdf_normed, x)

    samples = np.random.rand(nsamples)

    return inv_cdf(samples)

def _sample_from_kde(data, xrange, kernel, bw_method, bandwidths, cv, npoints, plot_pdf):
    """
    Helper function for sampling from a kernel density estimate of `data`. Returns a
    function that can be used for inverse transform sampling. See `sample_from_data`
    for detailed documentation.

    """

    if xrange is not None:
        data = data[rp.inrange(data, xrange[0], xrange[1])]

    ndata = len(data)

    if bw_method == 'scott':
        bandwidth = ndata**(-1 / 5) * np.std(data, ddof=1)
    elif bw_method == 'silverman':
        bandwidth = (ndata * 3 / 4)**(-1 / 5) * np.std(data, ddof=1)
    elif bw_method == 'cv':
        if bandwidths is None:
            bandwidths = np.std(data, ddof=1) ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(), {'bandwidth': bandwidths}, cv=cv)
        grid.fit(data[:, np.newaxis])
        bandwidth = grid.best_params_['bandwidth']
    elif np.isscalar(bw_method):
        bandwidth = bw_method
    else:
        raise ValueError("Unrecognized input for bw_method.")

    x_interp = np.linspace(np.min(data), np.max(data), num=npoints)

    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(data[:, np.newaxis])
    pdf = np.exp(kde.score_samples(x_interp[:, np.newaxis]))
    
    cdf = integrate.cumtrapz(pdf, x=x_interp, initial=0)
    cdf /= cdf[-1]

    if plot_pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_interp, pdf, color='k', label='Estimated PDF')
        ax.hist(
            data,
            bins='auto',
            range=xrange,
            histtype='step',
            density=True,
            color='k',
            alpha=0.3,
            linewidth=2,
            label="Data",
        )
        ax.set_xlim(x_interp[0], x_interp[-1])
        ax.set_ylim(bottom=0)
        ax.grid()
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.set_title(f"Estimated PDF of data from KDE: bandwidth = {bandwidth:.2e}")
        ax.legend()

    inv_cdf = interpolate.interp1d(cdf, x_interp)

    return inv_cdf

def _sample_from_hist(data, xrange, nbins, plot_pdf):
    """
    Helper function for sampling directly from a histogram defined by `data`. Returns a
    function that can be used for inverse transform sampling. See `sample_from_data`
    for detailed documentation.

    """

    hist, bin_edges = np.histogram(data, bins=nbins, density=True, range=xrange)

    if plot_pdf:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(
            data,
            bins=bin_edges,
            range=xrange,
            histtype='step',
            density=True,
            color='k',
            alpha=0.3,
            linewidth=2,
            label="Data",
        )
        ax.set_ylim(bottom=0)
        ax.grid()
        ax.tick_params(which='both', direction='in', right=True, top=True)
        ax.set_title("Histogram of Data to use as PDF")
        ax.legend()

    cdf = np.zeros(len(bin_edges))
    cdf[1:] = np.cumsum(hist * np.diff(bin_edges))

    # add tiny amounts so that cdf is always increasing 
    # avoids passing duplicate x-values to interpolate.interp1d
    cdf += np.arange(0, len(cdf), dtype=float) * 1e-12
    cdf /= cdf[-1]

    inv_cdf = interpolate.interp1d(cdf, bin_edges)

    return inv_cdf

def sample_from_data(data, use_kde=False, nsamples=1000, npoints=10000, nbins='auto', xrange=None,
                     kernel='gaussian', bw_method='scott', bandwidths=None, cv=20, plot_pdf=False):
    """
    Function for sampling from a 1D probability distribution estimated using `data`.
    Can either directly sample from a histogram of the data or a kernel density estimate
    (KDE) of the data.

    Parameters
    ----------
    data : ndarray
        A 1D ndarray of values to use as an estimate of underlying PDF.
    use_kde : bool, optional
        A boolean flag on whether or not to use KDE to estimate the underlying PDF (True)
        or to directly use the histogram of the data (False).
    nsamples : int, optional
        The number of random samples that we wish to create from the estimated PDF.
    npoints : int, optional
        The number of points to use in the numerical integration to evaluate the
        estimated CDF of the data. This is also the number of points used in the
        interpolation of the inverse of the CDF.
    nbins : str, int, optional
        The number of bins to use when creating a histogram of the data. Only used if
        `use_kde` is False. Can also pass a string which specifies the method to
        determine the method to determine the number of bins. The available methods are:
            ['auto'|'fd'|'doane'|'scott'|'stone'|'rice'|'sturges'|'sqrt']
        See the `numpy.histogram` docstring for more information. Default is 'auto'.
    xrange : array_like, NoneType, optional
        The range of values to define the PDF over. Should be an array_like of length 2.
        If left as None, this is essentially set as `(data.min(), data.max())`.
    kernel : str, optional
        The type of kernel to use for the KDE. Only used if `use_kde` is True.
        Valid kernels are:
            ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
        Default is 'gaussian'.
    bw_method : str, float, optional
        The method to use to determine the bandwidth for the KDE. Only used if `use_kde`
        is True. Valid methods are:
            ['scott'|'silverman'|'cv'|float]
        Default is 'scott'. See Notes for more.
    bandwidths : array_like, optional
        The bandwidths over which a grid search will be performed with cross-validation.
        Only used if `use_kde` is True and `bw_method = 'cv'`. Default is an ndarray, that is
        set as `bandwidths = np.std(data, ddof=1) ** np.linspace(-1, 1, 100)`.
    cv : int, cross-validation generator, iterable, optional
        Determines the cross-validation splitting strategy. See Notes for more. Default is 20.
    plot_pdf : bool, optional
        Option to plot the PDF estimated from the data, which will also give the bandwidth
        used if `use_kde` is True. Default is False.

    Returns
    -------
    rvs : ndarray
        The random samples that were taken from the inputted PDF estimated from `data`. This is a
        1D array of length `nsamples`.

    Raises
    ------
    ValueError
        If `bw_method` is not a recognized method.

    Notes
    -----
    For `bw_method`, the `'scott'` method uses Scott's Rule, as defined in [1]_. The `'silverman'` method
    uses Silverman's Rule, as defined in [2]_. Note that for each rule, we have multiplied by
    the standard deviation of the data to match the `sklearn` convention for bandwidth.

    If the `'cv'` method is used, the algorithm attempts a grid-search using a cross-validation method
    via `sklearn.model_selection.GridSearchCV` ([3]_). See the `bandwidths` and `cv` keyword arguments
    for tuning the grid-search.

    For `cv`, see [3]_ for more information on options for this keyword argument.

    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    """

    samples = np.random.rand(nsamples)

    if use_kde:
        inv_cdf = _sample_from_kde(data, xrange, kernel, bw_method, bandwidths, cv, npoints, plot_pdf)
    else:
        inv_cdf = _sample_from_hist(data, xrange, nbins, plot_pdf)

    return inv_cdf(samples)