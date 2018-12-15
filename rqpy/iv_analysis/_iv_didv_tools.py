import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from glob import glob
import sys
from collections import Counter
import pprint

from qetpy import IV, DIDV, Noise, didvinitfromdata, autocuts
from qetpy.sim import TESnoise, loadfromdidv
from qetpy.plotting import plot_noise_sim
from qetpy.utils import align_traces
import rqpy as rp


__all__ = ["IVanalysis"]

def _check_df(df, channels=None):
    """
    Simple helper function to check number of 
    occurances of qet bias values. It should be 2
    per channel
    
    Parameters
    ----------
    df : Pandas.core.DataFrame
        DataFrame of processed IV/dIdV sweep data
    channels : list, optional
        The channel name to analyze. If None, then 
        all the cahnnels are checked
        
    Returns
    -------
    gooddf : array
        Array of booleans corresponding to each
        channel in DF passing or failing check
        
    """
    
    if not channels:
        channels = set(df.channels.values)
    
    gooddf = np.ones(shape = len(channels), dtype = bool)

    for ii, chan in enumerate(channels):
        chancut = df.channels == chan
        check_data = Counter(df.qetbias[chancut].values)
        if np.all(np.array(list(check_data.values())) == 2):
            check = True
        else:
            check = False
        gooddf[ii] = check
    return gooddf


class IVanalysis(object):
    
    
    
    def __init__(self, df, channels=None):
        
  
        check = _check_df(df, channels)
        if np.all(check):
            self.df = df
        else:
            raise ValueError('The DF is not the correct shape. \n There is either an extra series, or missing data on one or more channels')
            
        self.channels = channels
        
        self.noiseinds = (df.datatype == "noise")
        self.didvinds = (df.datatype == "didv")
            
        
    
    


        