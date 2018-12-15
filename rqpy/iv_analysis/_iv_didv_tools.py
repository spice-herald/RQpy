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
    
    
    
    def __init__(self, df, channels=None, channelname = '', figsavepath = ''):
        
  
        check = _check_df(df, channels)
        if np.all(check):
            self.df = df
        else:
            raise ValueError('The DF is not the correct shape. \n There is either an extra series, or missing data on one or more channels')
            
        self.channels = channels
        self.chname = channelname
        self.figsavepath = figsavepath
        
        self.noiseinds = (df.datatype == "noise")
        self.didvinds = (df.datatype == "didv")
            
    
    def remove_bad_series(self):
        """
        Function to remove series where the the squid lost lock, or the 
        amplifier railed. This method will overwrite the parameter
        self.df with a DF that has the bad series removed. 
        
        """
        ccutfail = ~self.df.cut_pass 
        cstationary = np.array([len(set(trace)) for trace in self.df.avgtrace]) < 100
        cstd = self.df.offset_err == 0
        cbad = ccutfail | cstationary | cstd
        self.df = self.df[~cbad]

        
    def make_noiseplots(self, lgcsave=False):
        """
        Helper function to plot average noise/didv traces in time domain, as well as 
        corresponding noise PSDs, for all QET bias points in IV/dIdV sweep.
        Note, this function expects a DF with the parameters returned by
        rqpy.process.process_ivsweep()

        Parameters
        ----------
        lgcsave : Bool, optional
            If True, all the plots will be saved in the a folder
            Avetrace_noise/ within the user specified directory

        Returns
        -------
        None

        """
        for (noiseind, noiserow), (didvind, didvrow) in zip(self.df[self.noiseinds].iterrows(), self.df[self.didvinds].iterrows()):
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

            t = np.arange(0,len(noiserow.avgtrace))/noiserow.fs
            tdidv = np.arange(0, len(didvrow.avgtrace))/noiserow.fs
            axes[0].set_title(f"{noiserow.seriesnum} Avg Trace, QET bias = {noiserow.qetbias*1e6:.2f} $\mu A$")
            axes[0].plot(t*1e6, noiserow.avgtrace * 1e6, label=f"{self.chname} Noise", alpha=0.5)
            axes[0].plot(tdidv*1e6, didvrow.avgtrace * 1e6, label=f"{self.chname} dIdV", alpha=0.5)
            axes[0].grid(which="major")
            axes[0].grid(which="minor", linestyle="dotted", alpha=0.5)
            axes[0].tick_params(axis="both", direction="in", top=True, right=True, which="both")
            axes[0].set_ylabel("Current [μA]")
            axes[0].set_xlabel("Time [μs]")
            axes[0].legend()

            axes[1].loglog(noiserow.f, noiserow.psd**0.5 * 1e12, label=f"{self.chname} PSD")
            axes[1].set_title(f"{noiserow.seriesnum} PSD, QET bias = {noiserow.qetbias*1e6:.2f} $\mu A$")
            axes[1].grid(which="major")
            axes[1].grid(which="minor", linestyle="dotted", alpha=0.5)
            axes[1].set_ylim(1, 1e3)
            axes[1].tick_params(axis="both", direction="in", top=True, right=True, which="both")
            axes[1].set_ylabel(r"PSD [pA/$\sqrt{\mathrm{Hz}}$]")
            axes[1].set_xlabel("Frequency [Hz]")
            axes[1].legend()

            plt.tight_layout()
            if lgcsave:
                if not savepath.endswith('/'):
                    savepath += '/'
                fullpath = f'{savepath}avetrace_noise/'
                if not os.path.isdir(fullpath):
                    os.makedirs(fulpath)

                plt.savefig(fullpath + f'{noiserow.qetbias*1e6:.2f}_didvnoise.png')
            plt.show()
    


        