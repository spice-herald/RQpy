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
    """
    Class to aid in the analysis of an IV/dIdV sweep as processed by
    rqpy.proccess.process_ivsweep()
    
    Attributes
    ----------
    df : Pandas.core.DataFrame
        DataFrame with the parameters returned by
        rqpy.process.process_ivsweep()
    channels : str or list
        Channel names to analyze. If only interested in a single
        channel, channels can be in the form a string. 
    chname : str or list
        The corresponding name of the channels if a 
        different label from channels is desired for
        plotting. Ex: PBS1 -> G147 channel 1
    figsavepath : str
        Path to where figures should be saved
    noiseinds : array
        Array of booleans corresponding to the rows
        of the df that are noise type data
    didvinds : array
        Array of booleans corresponding to the rows
        of the df that are didv type data
    norminds : array
        Array of booleans corresponding to the rows
        of the didv df and noise df that are normal 
        state
    scinds : array
        Array of booleans corresponding to the rows
        of the didv df and noise df that are SC 
        state
    rshunt : float
        The value of the shunt resistor in the TES circuit
        in Ohms
    rload : float
        The value of the load resistor (rshunt + rp)
    rp : float
        The parasitic resistance in the TES line
    rn_didv : float
        The normal state resistance of the TES,
        calculated from fitting the dIdV
    rn_iv : float
        The normal state resistance of the TES,
        calculated from the IV curve
    
    """
    
    
    
    def __init__(self, df, nnorm, nsc, channels=None, channelname='', rshunt=5e-3, figsavepath=''):
        
  
        check = _check_df(df, channels)
        if np.all(check):
            self.df = df
        else:
            raise ValueError('The DF is not the correct shape. \n There is either an extra series, or missing data on one or more channels')
            
        self.channels = channels
        self.chname = channelname
        self.figsavepath = figsavepath
        self.rshunt = rshunt 
        self.rload = None
        self.rp = None
        self.rn_didv = None
        self.rn_iv = None
        
        self.noiseinds = (df.datatype == "noise")
        self.didvinds = (df.datatype == "didv")
        
        self.norminds = range(nnorm)
        self.scinds = range(len(df)//2-nsc, len(df)//2)
        
        
            
    
    def _remove_bad_series(self):
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
        self.noiseinds = self.noiseinds[~cbad]
        self.didvinds = self.didvinds[~cbad]
        self.norminds = range(len(self.norminds))
        self.scinds = range(len(self.df)//2-len(self.scinds), len(self.df)//2)

    def _fit_rload_didv(self, lgcplot=False, lgcsave=False, **kwargs):
        """
        Function to fit the SC dIdV series data and calculate rload. 
        Note, if the fit is not good, you may need to speficy an initial
        time offset using the **kwargs. Pass {'dt0' : 1.5e-6}# (or other value) 
        or additionally try {'add180phase' : False}
        
        Parameters
        ----------
        lgcplot : bool, optional
            If True, the plots are shown for each fit
        lgcsave : Bool, optional
            If True, all the plots will be saved in the a folder
            Avetrace_noise/ within the user specified directory
        lgcsave : 
        **kwargs : dict
            Additional key word arguments to be passed to didvinitfromdata()
        
        Returns
        -------
        None
        """
        
        rload_list = []
        for ind in (self.scinds):
            didvsc = self.df[self.didvinds].iloc[ind]
            didvobjsc = didvinitfromdata(didvsc.avgtrace[:len(didvsc.didvmean)], didvsc.didvmean, 
                                         didvsc.didvstd, didvsc.offset, didvsc.offset_err, 
                                         didvsc.fs, didvsc.sgfreq, didvsc.sgamp, 
                                         rshunt = self.rshunt, **kwargs)
            didvobjsc.dofit(1)
            rload_list.append(didvobjsc.get_irwinparams_dict(1)["rtot"])
            
            if lgcplot:
                didvobjsc.plot_full_trace(lgcsave=lgcsave, savepath=self.figsavepath,
                                          savename=f'didv_{didvsc.qetbias:.3e}')
        self.rload = np.mean(rload_list)
        self.rp = self.rload - self.rshunt
        
    def _fit_rn_didv(self, lgcplot=False, lgcsave=False, **kwargs):
        """
        Function to fit the Normal dIdV series data and calculate rn. 
        Note, if the fit is not good, you may need to speficy an initial
        time offset using the **kwargs. Pass {'dt0' : 1.5e-6}# (or other value) 
        or additionally try {'add180phase' : False}

        Parameters
        ----------
        lgcplot : bool, optional
            If True, the plots are shown for each fit
        lgcsave : Bool, optional
            If True, all the plots will be saved in the a folder
            Avetrace_noise/ within the user specified directory
        lgcsave : 
        **kwargs : dict
            Additional key word arguments to be passed to didvinitfromdata()

        Returns
        -------
        None
        """
        if self.rload is None:
            raise ValueError('rload has not been calculated yet, please fit rload first')
        rn_list = []
        for ind in (self.norminds):
            didvn = self.df[self.didvinds].iloc[ind]
            didvobjn = didvinitfromdata(didvn.avgtrace[:len(didvn.didvmean)], didvn.didvmean, 
                                         didvn.didvstd, didvn.offset, didvn.offset_err, 
                                         didvn.fs, didvn.sgfreq, didvn.sgamp, 
                                         rshunt = self.rshunt, rload=self.rload, **kwargs)
            didvobjn.dofit(1)
            rn=didvobjn.fitparams1[0]
            rn_list.append(rn)

            if lgcplot:
                didvobjn.plot_full_trace(lgcsave=lgcsave, savepath=self.figsavepath,
                                          savename=f'didv_{didvn.qetbias:.3e}')
        self.rn_didv = np.mean(rn_list) - self.rload
 
        
    def fit_rload_rn(self, lgcplot=False, lgcsave=False, **kwargs):
        """
        Function to fit the SC dIdV series data  and the Normal dIdV series 
        data and calculate rload, rp, and rn. 
        
        This is just a wrapper function that calls _fit_rload_didv() and
        _fit_rn_didv().
        
        Note, if the fit is not good, you may need to speficy an initial
        time offset using the **kwargs. Pass {'dt0' : 1.5e-6}# (or other value)
        or additionally try {'add180phase' : False}

        Parameters
        ----------
        lgcplot : bool, optional
            If True, the plots are shown for each fit
        lgcsave : Bool, optional
            If True, all the plots will be saved in the a folder
            Avetrace_noise/ within the user specified directory
        lgcsave : 
        **kwargs : dict
            Additional key word arguments to be passed to didvinitfromdata()

        Returns
        -------
        None
        """     
        
        self._fit_rload_didv(lgcplot, lgcsave, **kwargs)
        self._fit_rn_didv(lgcplot, lgcsave, **kwargs)
        
        
        
    def make_noiseplots(self, lgcsave=False):
        """
        Helper function to plot average noise/didv traces in time domain, as well as 
        corresponding noise PSDs, for all QET bias points in IV/dIdV sweep.
        

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
    


        