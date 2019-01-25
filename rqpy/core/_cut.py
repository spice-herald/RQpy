import warnings
import numpy as np
import pandas as pd
from scipy import stats, interpolate, optimize
from skimage import measure

from qetpy.cut import removeoutliers

__all__ = ["binnedcut", "baselinecut", "inrange", "passage_fraction"]


def baselinecut(arr, r0, i0, rload, dr=0.1e-3, cut=None):
    """
    Function to automatically generate the pre-pulse baseline cut. 
    The value where the cut is placed is set by dr, which is the user
    specified change in resistance from R0.
    
    Parameters
    ----------
    arr : ndarray
        Array of values to generate cut with
    r0 : float
        Operating resistance of TES
    i0 : float
        Quiescent operating current of TES
    rload : float
        The load resistance of the TES circuit, (Rp+Rsh)
    dr : float, optional
        The change in operating resistance where the
        cut should be placed
    cut : ndarray, optional
        Initial cut mask to use in the calculation of the pre-pulse
        baseline cut
            
    Returns:
    --------
    cbase : ndarray
        Array of type bool, corresponding to values which pass the 
        pre-pulse baseline cut
            
    """
    
    if cut is None:
        cut = np.ones_like(arr, dtype = bool)
    
    base_inds = removeoutliers(arr[cut])
    meanval = np.mean(arr[cut][base_inds])
    
    di = -(dr/(r0+dr+rload)*i0)
    
    cbase = (arr < (meanval + di))
    
    return cbase

class GenericModel(object):
    """
    A generic model class to be used with `skimage.measure.ransac` to allow the user to 
    create their own model for the data.
    
    Attributes
    ----------
    params : NoneType, ndarray
        The parameters that are returned by the fit.
    model : function
        A user-defined function to use as the model for the data.
    guess : tuple
        A guess for the best-fit parameters of the model.
    
    """
    
    def __init__(self, model, guess):
        """
        Initialization of the `GenericModel` object
        
        Parameters
        ----------
        model : function
            A user-defined function to use as the model for the data.
        guess : tuple
            A guess for the best-fit parameters of the model.
        
        """
        
        self.params = None
        self.model = model
        self.guess = guess
        
    def estimate(self, data):
        """
        Estimate the generic model from data using `scipy.optimize.curve_fit`.
        
        Parameters
        ----------
        data : (N, 2) ndarray
            The x and y values of the data to be used to estimate the model.
        
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        
        """
        
        self.params, _ = optimize.curve_fit(self.model, data[:, 0], data[:, 1], p0=self.guess)
        
        return True
        
    def residuals(self, data):
        """
        Determine the residuals of data to the generic model.
        
        Parameters
        ----------
        data : (N, 2) ndarray
            The x and y values of the data to be used to estimate the model.
        
        Returns
        -------
        residuals : (N, ) ndarray
            The residuals for each data point.
        
        """
        
        return self.model(data[:, 0], *self.params) - data[:, 1]
    

def binnedcut(x, y, cut=None, nbins=100, cut_eff=0.9, keep_large_vals=True, lgcequaldensitybins=False,
              xlwrlim=None, xuprlim=None, model=None, guess=None, residual_threshold=2, **kwargs):
    """
    Function for calculating a cut given a desired passage fraction, based on binning the data.
    
    Parameters
    ----------
    x : array_like
        Array of x-values to bin in.
    y : array_like
        Array of y-values to cut.
    cut : array_like, optional
        Boolean mask of values to keep for determination of the binned cut. Useful if 
        doing cut in a certain order. The binned cut will be added to this cut. Default is None.
    nbins : float, optional
        The number of bins to use in the cut. Default is 100.
    cut_eff : float, optional
        The desired efficiency/passage fraction of the cut, should be a value between 0 and 1.
        Default is 0.9.
    keep_large_vals : bool, optional
        Whether or not the cut should keep the smaller values or the larger values
        of `y`. If True, the larger values of `y` pass the cut based on `cut_eff`. 
        If False, the smaller values of `y` pass the cut based on `cut_eff`. Default
        is True.
    lgcequaldensitybins : bool, optional
        If set to True, the bin widths are set such that each bin has the same number
        of data points within it. If set to False, then a constant bin width is used. Default
        is False.
    xlwrlim : NoneType, float, optional
        The lower limit on `x` such that the cut at this value is applied to any values of `x`
        less than this. Default is None, where no lower limit is applied.
    xuprlim : NoneType, float, optional
        The upper limit on `x` such that the cut at this value is applied to any values of `x`
        greater than this. Default is None, where no upper limit is applied.
    model : NoneType, str, function, optional
        The model to use for determining the functional form of the cut. If set to None, the 
        bins are used as the bounds for this cut. If set to "linear", then
        `skimage.measure.LineModelND` is used as the model, which is a linear model, and 
        `skimage.measure.ransac` is used to estimate the model parameters. Can also be set to a 
        user-defined function, of the form f(x, *params), and `skimage.measure.ransac` is used 
        to estimate the model parameters. If the user-defined function is used, then the `guess` 
        parameter also needs to be passed.
    guess : NoneType, tuple, optional
        If model is set to a user-defined function, then a guess for the parameters must be 
        specified. This is only used if a user-defined function is passed to `model`.
    residual_threshold : float, optional
        Maximum distance for adata point to be classified as an inlier. This is passed directly
        to the `skimage.measure.ransac` function, of which this is a parameter. Default is 2.
    kwargs
        Keyword arguments passed to `skimage.measure.ransac`. See [1].
        
    Returns
    -------
    cbinned : array_like
        A boolean mask indicating which data points passed the baseline cut.
        
    Notes
    -----
    
    [1] http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.ransac
    
    """
    
    if (cut_eff > 1) or (cut_eff < 0):
        raise ValueError("cut_eff must be a value between 0 and 1")

    if cut is None:
        cut = np.ones(len(x), dtype=bool)

    if keep_large_vals:
        cut_eff = 1 - cut_eff

    st = lambda var: np.partition(var, int(len(var)*cut_eff))[int(len(var)*cut_eff)]

    if nbins==1:
        f = lambda var: st(y[cut])
    else:
        bin_cut = cut
        
        if xlwrlim is not None:
            lwr_cut = x >= xlwrlim
            bin_cut = bin_cut & lwr_cut
            
        if xuprlim is not None:
            upr_cut = x <= xuprlim
            bin_cut = bin_cut & upr_cut
            
        if lgcequaldensitybins:
            histbins_equal = lambda var, nbin: np.interp(np.linspace(0, len(var), nbin + 1),
                                                         np.arange(len(var)),
                                                         np.sort(var))
            nbins = histbins_equal(x[bin_cut], nbins)
        
        cutoffs, bin_edges, _ = stats.binned_statistic(x[bin_cut], y[bin_cut], bins=nbins,
                                                       statistic=st)
        if model is None:
            f = interpolate.interp1d(bin_edges[1:-1], cutoffs[1:], kind='previous', 
                                     bounds_error=False, fill_value=(cutoffs[0], cutoffs[-1]),
                                     assume_sorted=True)
        else: 
            if model=="linear":
                ModelClass = measure.LineModelND
                min_samples = 2
            else:
                if guess is None:
                    raise ValueError("guess was not set, a guess is required to use the generic model.")
                
                min_samples = len(guess)
                
                class ModelClass(GenericModel):
                    def __init__(self):
                        super().__init__(model, guess)
                    
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=optimize.OptimizeWarning)
                model_robust, _ = measure.ransac(np.stack((bin_edges[1:-1], cutoffs[1:]), axis=1),
                                                 ModelClass, min_samples, residual_threshold, 
                                                 **kwargs)
            if model=="linear":
                f = lambda var: ModelClass().predict_y(var, model_robust.params)
            else:
                f = lambda var: model(var, *model_robust.params)
                
    if keep_large_vals:
        cbinned = (y > f(x)) & cut
    else:
        cbinned = (y < f(x)) & cut
    
    return cbinned

def inrange(vals, lwrbnd, uprbnd):
    """
    Function for returning a boolean mask that specifies which values
    in an array are between the specified bounds (inclusive of the bounds).
    
    Parameters
    ----------
    vals : array_like
        A 1-d array of values.
    lwrbnd : float
        The lower bound of the range that we are checking if vals is between.
    uprbnd : float
        The upper bound of the range that we are checking if vals is between.
            
    Returns
    -------
    mask : ndarray
        A boolean array of the same shape as vals. True means that the
        value was between the bounds, False means that the value was not.
    
    """
    
    mask = (vals >= lwrbnd) & (vals <= uprbnd)
    
    return mask

def passage_fraction(x, cut, basecut=None, nbins=100, lgcequaldensitybins=False):
    """
    Function for returning the passage fraction of a cut as a function of the specified
    variable `x`.
    
    Parameters
    ----------
    x : array_like
        Array of values to be binned and plotted
    cut : array_like
        Mask of values to calculate passage fraction for.
    basecut : NoneType, array_like, optional
        A cut to use as the comparison for the passage fraction. If left as None,
        then the passage fraction is calculated using all of the inputted data.
    nbins : int, optional
        The number of bins that should be created. Default is 100.
    lgcequaldensitybins : bool, optional
        If set to True, the bin widths are set such that each bin has the same number
        of data points within it. If left as False, then a constant bin width is used.
    
    Returns
    -------
    x_binned : ndarray
        The corresponding `x` values for each passage fraction, corresponding to the
        center of the bin.
    frac_binned : ndarray
        The passage fractions for each value of `x_binned` for the given `cut` and `basecut`.
    
    """
    
    if basecut is None:
        basecut = np.ones(len(x), dtype=bool)
    
    if lgcequaldensitybins:
        histbins_equal = lambda var, nbin: np.interp(np.linspace(0, len(var), nbin + 1),
                                                     np.arange(len(var)),
                                                     np.sort(var))
        nbins = histbins_equal(x[basecut], nbins)
    
    hist_vals_base, x_binned = np.histogram(x[basecut], bins=nbins)
    hist_vals, _ = np.histogram(x[basecut & cut], bins=x_binned)
    
    frac_binned = hist_vals/hist_vals_base
    x_binned = (x_binned[:-1]+x_binned[1:])/2
    
    return x_binned, frac_binned