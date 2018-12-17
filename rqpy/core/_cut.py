import numpy as np
import pandas as pd
from qetpy.cut import removeoutliers
from scipy import stats, interpolate

__all__ = ["binnedcut", "baselinecut", "inrange"]


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


def binnedcut(x, y, cut=None, nbins=100, cut_eff=0.9, keep_large_vals=True):
    """
    Function for calculating a baseline cut over time based on a given percentile.
    
    Parameters
    ----------
    x : array_like
        Array of x-values to bin in.
    y : array_like
        Array of y-values to cut.
    cut : array_like, optional
        Boolean mask of values to keep for determination of the binned cut. Useful if 
        doing cut in a certain order. The binned cut will be added to this cut.
    nbins : float, optional
        The number of bins to use in the cut
    cut_eff : float, optional
        The desired cut efficiency, should be a value between 0 and 1.
    keep_large_vals : bool, optional
        Whether or not the cut should keep the smaller values or the larger values
        of `y`. If True, the larger values of `y` pass the cut based on `cut_eff`. 
        If False, the smaller values of `y` pass the cut based on `cut_eff`. Default
        is True.
        
    Returns
    -------
    cbinned : array_like
        A boolean mask indicating which data points passed the baseline cut.
    
    """
    
    if (cut_eff > 1) or (cut_eff < 0):
        raise ValueError("cut_eff must be a value between 0 and 1")

    if cut is None:
        cut = np.ones(len(x), dtype=bool)

    if keep_large_vals:
        cut_eff = 1 - cut_eff

    st = lambda var: np.partition(var, int(len(var)*cut_eff))[int(len(var)*cut_eff)]

    if nbins==1:
        f = lambda var: st(x)*np.ones(len(x))
    else:
        cutoffs, bin_edges, _ = stats.binned_statistic(x[cut], y[cut], bins=nbins,
                                                       statistic=st)
        cutoffs = np.pad(cutoffs, (1, 0), 'constant', constant_values=(cutoffs[0], 0))
        f = interpolate.interp1d(bin_edges, cutoffs, kind='next', 
                                 bounds_error=False, fill_value=(cutoffs[0], cutoffs[-1]),
                                 assume_sorted=True)

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
