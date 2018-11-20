import numpy as np
import pandas as pd
from qetpy.utils import removeoutliers
from scipy import stats, interpolate

__all__ = ["baselinecut_tdep", "baselinecut_dr"]


def baselinecut_dr(arr, r0, i0, rload, dr = 0.1e-3, cut = None):
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
    cbase_pre: ndarray
        Array of type bool, corresponding to values which pass the 
        pre-pulse baseline cut
            
    """
    
    if cut is None:
        cut = np.ones_like(arr, dtype = bool)
    
    base_inds = removeoutliers(arr[cut])
    meanval = np.mean(arr[cut][base_inds])
    
    di = -(dr/(r0+dr+rload)*i0)
    
    cbase_pre = (arr < (meanval + di))
    
    return cbase_pre


def baselinecut_tdep(t, b, cut=None, dt=1000, cut_eff=90, positive_pulses=True):
    """
    Function for calculating a baseline cut over time based on a given percentile.
    
    Parameters
    ----------
    t : array_like
        Array of time values, should be in units of s.
    b : array_like
        Array of baselines to cut, any units.
    cut : array_like, optional
        Boolean mask of values to keep for determination of baseline cut. Useful if 
        doing cut in a certain order. The baseline cut will be added to this cut.
    dt : float, optional
        Length in time that the baselines should be binned in. Should be in units of s.
        Determines the number of bins by (time elapsed)/dt.
    cut_eff : float, optional
        The desired cut efficiency, should be a value between 0 and 100. This is the 
        percentile used to cut on.
    positive_pulses : bool, optional
        The direction of the pulses in the data, which determines the direction of the 
        tails of the baseline distributions and which values should be kept.
        
    Returns
    -------
    cbase : array_like
        A boolean mask indicating which data points passed the baseline cut.
    
    """
    
    if cut is None:
        cut = np.ones(len(b), dtype=bool)
    
    if isinstance(t, pd.core.series.Series):
        t_elapsed = t[cut].iloc[-1] - t[cut].iloc[0]
    else:
        t_elapsed = t[cut][-1] - t[cut][0]
    
    nbins = int(t_elapsed/dt)
    
    if not positive_pulses:
        cut_eff = 100 - cut_eff
    
    cutoffs, bin_edges, _ = stats.binned_statistic(t[cut], b[cut], bins=nbins,
                                                   statistic=lambda x: np.percentile(x, cut_eff))
    
    f = interpolate.interp1d(bin_edges[:-1], cutoffs, kind='next', 
                             bounds_error=False, fill_value=(cutoffs[0], cutoffs[-1]),
                             assume_sorted=True)
    
    if positive_pulses:
        cbase = (b < f(t)) & cut
    else:
        cbase = (b > f(t)) & cut
    
    return cbase

