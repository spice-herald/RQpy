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

def _remove_bad_series(df):
    """
    Helper function to remove series where the the squid lost lock, or the 
    amplifier railed. This method will overwrite the parameter
    self.df with a DF that has the bad series removed. 
    
    Parameters
    ----------
    df : Pandas.core.DataFrame
        DataFrame of processed IV/dIdV sweep data


    Returns
    -------
    newdf : Pandas.core.DataFrame
        New dataframe with railed events removed
        
    """
    
    ccutfail = ~df.cut_pass 
    cstationary = np.array([len(set(trace)) for trace in df.avgtrace]) < 100
    cstd = df.offset_err == 0
    cbad = ccutfail | cstationary | cstd
    newdf = df[~cbad]

    return newdf

def _sort_df(df):
    """
    Helper function to sort data frame
    
    Parameters
    ----------
    df : Pandas.core.DataFrame
        DataFrame of processed IV/dIdV sweep data

    Returns
    -------
    sorteddf : Pandas.core.DataFrame
        New sorted dataframe
        
    """
    
    sorteddf = df.sort_values(['qetbias', 'seriesnum'], ascending=[True, True])
    
    return sorteddf


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
    rshunt_err : float
        The uncertainty in the shunt resistor in the TES circuit
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
    rn_iv_err : float
        The uncertainty in the normal state resistance 
        of the TES, calculated from the IV curve
    vb : array
        Array of bias voltages
    vb_err : array
        Array of uncertainties in the bais voltage
    dites : array
        Array of DC offsets for IV/didv data
    dites : array
        Array of uncertainties in the DC offsets
    
    """
    
    
    
    def __init__(self, df, nnorm, nsc, channels=None, channelname='', rshunt=5e-3, 
                 rshunt_err = 0.05*5e-3, lgcremove_badseries = True, figsavepath=''):
        
        df = _sort_df(df)
        
        check = _check_df(df, channels)
        if np.all(check):
            self.df = df
        else:
            raise ValueError('The DF is not the correct shape. \n There is either an extra series, or missing data on one or more channels')
        
        self.channels = channels
        self.chname = channelname
        self.figsavepath = figsavepath
        self.rshunt = rshunt 
        self.rshunt_err = rshunt_err
        self.rload = None
        self.rload_list = None
        self.rp = None
        self.rn_didv = None
        self.rn_iv = None
        self.rn_iv_err = None
        self.rtot_list = None
        
        if lgcremove_badseries:
            self.df = _remove_bad_series(df)
        
        self.noiseinds = (self.df.datatype == "noise")
        self.didvinds = (self.df.datatype == "didv")
        self.norminds = range(nnorm)
        self.scinds = range(len(self.df)//2-nsc, len(self.df)//2)
    
        vb = np.zeros((1,2,self.noiseinds.sum()))
        vb_err = np.zeros(vb.shape)
        vb[0,0,:] = self.df[self.noiseinds].qetbias.values * rshunt
        vb[0,1,:] = (self.df[self.didvinds].qetbias.values) * rshunt
        dites = np.zeros((1,2,self.noiseinds.sum()))
        dites_err = np.zeros((1,2,self.noiseinds.sum()))
        dites[0,0,:] = self.df[self.noiseinds].offset.values
        dites_err[0,0,:] = self.df[self.noiseinds].offset_err.values
        dites[0,1,:] = self.df[self.didvinds].offset.values
        dites_err[0,1,:] = self.df[self.didvinds].offset_err.values
        
        self.vb = vb
        self.vb_err = vb_err
        self.dites = dites
        self.dites_err = dites_err
        
     
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
        self.rload_list = rload_list
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
        rtot_list = []
        for ind in (self.norminds):
            didvn = self.df[self.didvinds].iloc[ind]
            didvobjn = didvinitfromdata(didvn.avgtrace[:len(didvn.didvmean)], didvn.didvmean, 
                                         didvn.didvstd, didvn.offset, didvn.offset_err, 
                                         didvn.fs, didvn.sgfreq, didvn.sgamp, 
                                         rshunt = self.rshunt, rload=self.rload, **kwargs)
            didvobjn.dofit(1)
            rtot=didvobjn.fitparams1[0]
            rtot_list.append(rtot)

            if lgcplot:
                didvobjn.plot_full_trace(lgcsave=lgcsave, savepath=self.figsavepath,
                                          savename=f'didv_{didvn.qetbias:.3e}')
        self.rn_didv = np.mean(rtot_list) - self.rload
        self.rtot_list = rtot_list
        
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
        **kwargs : dict
            Additional key word arguments to be passed to didvinitfromdata()

        Returns
        -------
        None
        """     
        
        self._fit_rload_didv(lgcplot, lgcsave, **kwargs)
        self._fit_rn_didv(lgcplot, lgcsave, **kwargs)
        
        
    def analyze_sweep(self, lgcplot=False, lgcsave=False):
        """
        Function to correct for the offset in current and calculate
        R0, P0 and make plots of IV sweeps.
        
        The following parameters are added to self.df:
            ptes
            ptes_err
            r0
            r0_err
        and rn_iv and rn_iv_err are added to self. All of these 
        parameters are calculated from the noise data, and the 
        didv data.
        
        Parameters
        ----------
        lgcplot : bool, optional
            If True, the plots are shown for each fit
        lgcsave : Bool, optional
            If True, all the plots will be saved in the a folder
            Avetrace_noise/ within the user specified directory
        
        Returns
        -------
        None
        
        """
        
        ivobj = IV(dites = self.dites, dites_err = self.dites_err, vb = self.vb, vb_err = self.vb_err, 
                   rload = self.rload, rload_err = self.rshunt_err, 
                   chan_names = [f'{self.chname} Noise',f'{self.chname} dIdV'], 
                   normalinds = self.norminds)
        ivobj.calc_iv()
        
        self.df.loc[self.noiseinds, 'ptes'] =  ivobj.ptes[0,0]
        self.df.loc[self.didvinds, 'ptes'] =  ivobj.ptes[0,1]
        self.df.loc[self.noiseinds, 'ptes_err'] =  ivobj.ptes_err[0,0]
        self.df.loc[self.didvinds, 'ptes_err'] =  ivobj.ptes_err[0,1]
        self.df.loc[self.noiseinds, 'r0'] =  ivobj.r0[0,0]
        self.df.loc[self.didvinds, 'r0'] =  ivobj.r0[0,1]
        self.df.loc[self.noiseinds, 'r0_err'] =  ivobj.r0_err[0,0]
        self.df.loc[self.didvinds, 'r0_err'] =  ivobj.r0_err[0,1]
        
        self.rn_iv = ivobj.rnorm[0,0]
        self.rn_iv_err = ivobj.rnorm_err[0,0]

        if lgcplot:
            ivobj.plot_all_curves(lgcsave=lgcsave, savepath=self.figsavepath, savename=self.chname)
            
        
        
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
            axes[0].set_ylabel("Current [μA]", fontsize = 14)
            axes[0].set_xlabel("Time [μs]", fontsize = 14)
            axes[0].legend()

            axes[1].loglog(noiserow.f, noiserow.psd**0.5 * 1e12, label=f"{self.chname} PSD")
            axes[1].set_title(f"{noiserow.seriesnum} PSD, QET bias = {noiserow.qetbias*1e6:.2f} $\mu A$")
            axes[1].grid(which="major")
            axes[1].grid(which="minor", linestyle="dotted", alpha=0.5)
            axes[1].set_ylim(1, 1e3)
            axes[1].tick_params(axis="both", direction="in", top=True, right=True, which="both")
            axes[1].set_ylabel(r"PSD [pA/$\sqrt{\mathrm{Hz}}$]", fontsize = 14)
            axes[1].set_xlabel("Frequency [Hz]", fontsize = 14)
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
    
    def plot_rload_rn_qetbias(self, lgcsave=False):
        """
        Helper function to plot rload and rnormal as a function of
        QETbias from the didv fits of SC and Normal data
        

        Parameters
        ----------
        lgcsave : Bool, optional
            If True, all the plots will be saved 
            
        Returns
        -------
        None
        """
        
        fig, axes = plt.subplots(1,2, figsize = (16,6))
        fig.suptitle("Rload and Rtot from dIdV Fits", fontsize = 18)
        
        axes[0].errorbar(self.vb[0,0,self.scinds]*1e6,np.array(self.rload_list)*1e3, 
                       yerr = self.rshunt_err*1e3, linestyle = '', marker = '.', ms = 10)
        axes[0].grid(True, linestyle = 'dashed')
        axes[0].set_title('Rload vs Vbias', fontsize = 14)
        axes[0].set_ylabel(r'$R_ℓ$ [mΩ]', fontsize = 14)
        axes[0].set_xlabel(r'$V_{bias}$ [μV]', fontsize = 14)
        axes[0].tick_params(axis="both", direction="in", top=True, right=True, which="both")
        
        axes[1].errorbar(self.vb[0,0,self.norminds]*1e6,np.array(self.rtot_list)*1e3, 
                       yerr = self.rshunt_err*1e3, linestyle = '', marker = '.', ms = 10)
        axes[1].grid(True, linestyle = 'dashed')
        axes[1].set_title('Rtotal vs Vbias', fontsize = 14)
        axes[1].set_ylabel(r'$R_{N} + R_ℓ$ [mΩ]', fontsize = 14)
        axes[1].set_xlabel(r'$V_{bias}$ [μV]', fontsize = 14)
        axes[1].tick_params(axis="both", direction="in", top=True, right=True, which="both")
        
        plt.tight_layout()
        if lgcsave:
            plt.savefig(self.figsavepath + 'rload_rtot_variation.png')
            

        