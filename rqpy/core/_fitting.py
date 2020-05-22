import numpy as np
from scipy.optimize import curve_fit
from rqpy import plotting, utils
import iminuit


__all__ = [
    "fit_multi_gauss",
    "fit_gauss",
    "ext_max_llhd",
    "NormBackground",
]


def fit_multi_gauss(arr, guess, ngauss, xrange=None, nbins='sqrt',
                    lgcplot=True, labeldict=None, lgcfullreturn=False):
    """
    Function to multiple Gaussians plus a flat background. Note,
    depending on the spectrum, this function can ber very sensitive to
    the inital guess parameters.

    Parameters
    ----------
    arr : array
        Array of values to be binned
    guess : tuple
        The initial guesses for the Gaussian peaks. The order must be as
        follows:
            (amplitude_i, mu_i, std_i,
            ....,
            ....,
            background),
        where the guess for the background is the last element.
    ngauss : int
        The number of peaks to fit
    xrange : tuple, optional
        The range over which to fit the peaks
    nbins : int, str, optional
        This is the same as plt.hist() bins parameter. Defaults is 'sqrt'.
    lgcplot : bool, optional
        If True, the fit and spectrum will be plotted 
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are:
            labels = {
                'title' : 'Histogram',
                'xlabel' : 'variable',
                'ylabel' : 'Count',
            }
        Ex: to change just the title, pass:
            labeldict = {'title' : 'new title'}
        to fit_multi_gauss()
    lgcfullreturn : bool, optional
        If True, the binned data is returned along with the fit
        parameters

    Returns
    -------
    peaks : array
        Array of locations of Gaussians maximums, sorted by magnitude
        in increasing order
    amps : array
        Array of amplitudes, corresponding to order of 'peaks'
    stds : array
        Array of sqrt of variance, corresponding to order of 'peaks'
    background_fit : float
        The magnitude of the background
    fitparams : array, optional
        The best fit parameters, in the same order as the input guess
    errors : array, optional
        The uncertainty in the best fit parameters
    cov : array, optional
        The covariance matrix returned by scipy.optimize.curve_fit()
    bindata : tuple, optional
        The binned data from _bindata(), in order (x, y, bins)

    Raises
    ------
    ValueError
        If the number or parameters given in the guess is in conflict
        with ngauss, a ValueError is raised.

    """

    if ngauss != (len(guess)-1)/3:
        raise ValueError(
            'Number of parameters in guess must match the number of '
            'Gaussians being fit (ngauss)'
        )

    fit_n_gauss = lambda x, *params: utils.n_gauss(
        x, params, ngauss,
    ).sum(axis=0)

    x,y, bins = utils.bindata(arr,  xrange=xrange, bins=nbins)
    yerr = np.sqrt(y)
    yerr[yerr == 0] = 1 # make errors 1 if bins are empty

    fitparams, cov = curve_fit(
        fit_n_gauss, x, y, guess, sigma=yerr, absolute_sigma=True,
    )
    errors = np.sqrt(np.diag(cov))

    peaks = fitparams[1:-1:3]
    amps =  fitparams[0:-1:3]
    stds = fitparams[2:-1:3]
    bkgd_fit = fitparams[-1]

    peakssort = peaks.argsort()
    peaks = peaks[peakssort]
    amps = amps[peakssort]
    stds = stds[peakssort]

    if lgcplot:
        plotting.plot_n_gauss(x, y, bins, fitparams, labeldict)

    if lgcfullreturn:
        bininfo = (x, y, bins)
        return peaks, amps, stds, bkgd_fit, fitparams, errors, cov, bininfo
    else:
        return peaks, amps, stds, bkgd_fit


def fit_gauss(arr, xrange=None, nbins='sqrt', noiserange=None, lgcplot=False,
              labeldict=None, lgcasymbkg=False):
    """
    Function to fit Gaussian distribution with background to peak in
    spectrum. Errors are assumed to be poissonian.

    Parameters
    ----------
    arr : ndarray
        Array of data to bin and fit to gaussian
    xrange : tuple, optional
        The range of data to use when binning
    nbins : int, str, optional
        This is the same as plt.hist() bins parameter.
        Default is 'sqrt'.
    noiserange : tuple, optional
        nested 2-tuple. should contain the range before
        and after the peak to be used for subtracting the
        background. Only used when lgcasymbkg is False
    lgcplot : bool, optional
        If True, the fit and spectrum will be plotted
    labeldict : dict, optional
        Dictionary to overwrite the labels of the plot. defaults are:
            labels = {
                'title' : 'Histogram',
                'xlabel' : 'variable',
                'ylabel' : 'Count',
            }
        Ex: to change just the title, pass:
            labeldict = {'title' : 'new title'}
        to fit_gauss()
    lgcasymbkg : bool, optional
        If True, fit different background amplitudes on either side of
        Gaussian peak.

    Returns
    -------
    peakloc : float
        The mean of the distribution
    peakerr : float
        The full error in the location of the peak
    fitparams : tuple
        The best fit parameters of the fit; A, mu, sigma, background
    errors : ndarray
        The uncertainty in the fit parameters

    """

    x,y, bins = utils.bindata(arr,  xrange, bins=nbins)
    yerr = np.sqrt(y)
    yerr[yerr == 0] = 1 # make errors 1 if bins are empty

    if (noiserange is not None) and (lgcasymbkg is False):
        if noiserange[0][0] >= xrange[0]:
            clowl = noiserange[0][0]
        else:
            clowl = xrange[0]

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
        background = np.mean(
            np.concatenate(
                (y[indlowl:indlowh], y[indhighl:indhighh]),
            )
        )  
    else:
        background = 0

    y_noback = y - background

    # get starting values for guess
    A0 = np.max(y_noback)
    mu0 = x[np.argmax(y_noback)]
    sig0 = np.abs(mu0 - x[np.abs(y_noback - np.max(y_noback)/2).argmin()])


    if lgcasymbkg:
        b10=0
        b20=0
        p0 = (A0, mu0, sig0, b10, b20)
        #do fit
        fitparams, cov = curve_fit(
            utils.gaussian_asym_background,
            x,
            y,
            p0,
            sigma=yerr,
            absolute_sigma=True,
        )
    else:
        p0 = (A0, mu0, sig0, background)
        #do fit
        fitparams, cov = curve_fit(
            utils.gaussian_background,
            x,
            y,
            p0,
            sigma=yerr,
            absolute_sigma=True,
        )

    errors = np.sqrt(np.diag(cov))
    peakloc = fitparams[1]
    peakerr = np.sqrt((fitparams[2] / np.sqrt(fitparams[0]))**2)

    if lgcplot:
        if lgcasymbkg:
            plotting.plot_gauss_asymbkg(
                x, bins, y, fitparams, errors, labeldict,
            )
        else:
            plotting.plot_gauss(
                x, bins, y, fitparams, errors, background, labeldict,
            )

    return peakloc, peakerr, fitparams, errors


def ext_max_llhd(x, func, guess, guess_err=None, limits=None):
    """
    Routine for finding the Extended Unbinned Maximum Likelihood of an
    inputted spectrum, giving an inputted (arbitrary) function.

    Parameters
    ----------
    x : array_like
        The energies of each inputted event.
    func : FunctionType
        The negative log-likelihood for the normalized PDF, see Notes.
    guess : array_like
        Guesses for the true values of each parameter.
    guess_err : array_like, optional
        Guess for the errors of each parameter. Default is to
        simply use the guesses.
    limits : array_like, optional
        The limits to set on each parameter. Default is to set no
        limits. Should be of form:
            [(lower0, upper0), (lower1, upper1), ...]

    Returns
    -------
    m : iminuit.Minuit
        The Minuit object that contains all information on the fit,
        after the MINUIT algorithm has completed.

    Notes
    -----
    For a normalized PDF of form f(x, p) / Norm(p), where p is a vector
    of the fit parameters, the negative-log likelihood for the Extended
    Unbinned Maximum Likelihood method is:

        -log(L) = Norm(p) - sum(log(f(x, p)))

    """

    fit_dict = {f'p{ii}': g for ii, g in enumerate(guess)}

    if guess_err is None:
        err_dict = {f'error_p{ii}': g for ii, g in enumerate(guess)}
    else:
        err_dict = {f'error_p{ii}': g for ii, g in enumerate(guess_err)}

    if limits is None:
        limit_dict = {
            f'limit_p{ii}': (None, None) for ii in range(len(guess))
        }
    else:
        limit_dict = {f'limit_p{ii}': l for ii, l in enumerate(limits)}

    input_dict = {**fit_dict, **err_dict, **limit_dict}

    m = iminuit.Minuit(
        lambda p: func(x, p),
        use_array_call=True,
        errordef=1,
        forced_parameters=[f'p{ii}' for ii in range(len(guess))],
        **input_dict,
    )

    m.migrad()
    m.hesse()

    return m


class NormBackground(object):
    """
    Class for calculating a normalized spectrum from specified
    background shapes.

    """

    def __init__(self, lwrbnd, uprbnd, flatbkgd=True, nexpbkgd=0,
                 ngaussbkgd=0):
        """
        Initalization of the NormBackground class.

        Parameters
        ----------
        lwrbnd : float
            The lower bound of the background spectra, in energy.
        uprbnd : float
            The upper bound of the background spectra, in energy.
        flatbkgd : bool, optional
            If True, then the background spectrum will have a flat
            background component. If False, there will not be one.
            Default is True.
        nexpbkgd : int, optional
            The number of exponential spectra in the background
            spectrum. Default is 0.
        ngaussbkgd : int, optional
            The number of Gaussian spectra in the background spectrum.
            Default is 0.

        """

        self._lwrbnd = lwrbnd
        self._uprbnd = uprbnd
        self._flatbkgd = flatbkgd
        self._nexpbkgd = nexpbkgd
        self._ngaussbkgd = ngaussbkgd

        self._nparams = flatbkgd + nexpbkgd * 2 + ngaussbkgd * 3

    @staticmethod
    def _flatbkgd(x, *p):
        """Hidden method to calculate a flat spectrum."""

        if np.isscalar(x):
            return p[0]

        return p[0] * np.ones(len(x))

    @staticmethod
    def _expbkgd(x, *p):
        """Hidden method to calculate an exponential spectrum."""

        return p[0] * np.exp(-x / p[1])

    @staticmethod
    def _gaussbkgd(x, *p):
        """Hidden method to calculate a Gaussian spectrum."""

        return p[0] * np.exp(-(x - p[1])**2 / (2 * p[2]**2))

    def background(self, x, p):
        """
        Method for calculating the differential background (in units of
        1 / [energy]) given the inputted parameters and initialized
        background shape.

        Parameters
        ----------
        x : ndarray
            The energies at which the background will be calculated.
        p : array_like
            The parameters that determine the shape of each component
            of the background. See Notes for order of parameters.

        Returns
        -------
        output : ndarray
            The differential background spectrum at each `x` value,
            given the inputted shape parameters `p`. Units of
            1 / [energy].

        Notes
        -----
        The order of parameters should be

            1) The flat background rate (if there is one, otherwise
                skip)
            2) The exponential shape parameters for each exponential
                background (if nonzero):
                    (amplitude, exponential coefficient)
            3) The Gaussian shape parameters for each Gaussian
                background (if nonzero):
                    (ampltiude, mean, standard deviation)

        """

        if len(p) != self._nparams:
            raise ValueError(
                'Length of p does not match expected number of parameters'
            )

        output = np.zeros(len(x))
        ii = 0

        if self._flatbkgd:
            output += self._flatbkgd(x, *(p[ii], ))
            ii += 1

        for jj in range(self._nexpbkgd):
            output += self._expbkgd(x, *(p[ii], p[ii + 1]))
            ii += 2

        for jj in range(self._ngaussbkgd):
            output += self._gaussbkgd(x, *(p[ii], p[ii + 1], p[ii + 2]))
            ii += 3

        return output

    def _normalization(self, p):
        """
        Hidden method for calculating the normalization of the
        background spectrum.

        """

        norm = 0
        ii = 0

        if self._flatbkgd:
            norm += self._flatbkgd(
                0, *(p[ii], ),
            ) * (self._uprbnd - self._lwrbnd)
            ii += 1

        for jj in range(self._nexpbkgd):
            norm += p[ii + 1] * (
                self._expbkgd(
                    self._lwrbnd, *(p[ii], p[ii + 1]),
                ) - self._expbkgd(
                    self._uprbnd, *(p[ii], p[ii + 1]),
                )
            )
            ii += 2

        for jj in range(self._ngaussbkgd):
            norm += p[ii] * np.sqrt(2 * np.pi * p[ii + 2]**2)
            ii += 3

        return norm

    def neglogllhd(self, x, p):
        """
        Method for calculating the negative log-likelihood for use with
        an extended maximum likelihood method, e.g.
        `rqpy.ext_max_llhd`.

        Parameters
        ----------
        x : ndarray
            The energies at which the background will be calculated.
        p : array_like
            The parameters that determine the shape of each component
            of the background. See Notes for order of parameters.

        Returns
        -------
        out : float
            The extended maximum likelihood for inputted spectrum
            parameters.

        Notes
        -----
        The order of parameters should be

            1) The flat background rate (if there is one, otherwise
                skip)
            2) The exponential shape parameters for each exponential
                background (if nonzero):
                    (amplitude, exponential coefficient)
            3) The Gaussian shape parameters for each Gaussian
                background (if nonzero):
                    (ampltiude, mean, standard deviation)

        """

        return -sum(np.log(self.background(x, p))) + self._normalization(p)

