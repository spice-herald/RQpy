import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from glob import glob
import sys
from collections import Counter
import pprint
from scipy import constants
from scipy.signal import savgol_filter
from lmfit import Model

import rqpy as rp
from rqpy.plotting import _plot_rload_rn_qetbias, _make_iv_noiseplots, _plot_energy_res_vs_bias, _plot_n_noise, _plot_sc_noise
from qetpy import IV, DIDV, Noise, didvinitfromdata, autocuts
from qetpy.sim import TESnoise, loadfromdidv, energy_res_estimate
from qetpy.plotting import plot_noise_sim
from qetpy.utils import align_traces, make_decreasing



__all__ = ["IVanalysis"]



class Error(Exception):
   """Base class for other exceptions"""
   pass


class AnalysisError(Error):
   """
   Raised when there is an error in one of the steps 
   in the automated analysis
   """
   pass

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

def _flatten_psd(f, psd):
    """
    Helper function to smooth out all the spikes
    in a single sided psd in order to more easily fit the 
    SC and Normal state noise
    
    Parameters
    ----------
    f: ndarray
        Array of frequency values
    psd : ndarray
        Array of one sided psd values
        
    Returns
    -------
    flattened_psd : ndarray
        Array of values of smoothed psd
        
    """
    
    sav = np.zeros(psd.shape)
    div = int(.0025*len(psd))
    sav_lower = savgol_filter(psd[1:], 3, 1, mode = 'interp', deriv=0)
    sav_upper = savgol_filter(psd[1:], 45, 1, mode = 'interp', deriv=0)
    sav[1:div+1] = sav_lower[:div]
    sav[1+div:] = sav_upper[div:]
    sav[0] = psd[0]
    flattened_psd = make_decreasing(sav, x=f)
    
    return flattened_psd

def _normal_noise(freqs, squiddc, squidpole, squidn, rload, tload, rn, tc, inductance):
    """
    Functional form of the normal state noise. Including
    the johnson noise for the load resistor, the johnson 
    noise for the TES, and the SQUID + downstream electronics 
    noise. See qetpy.sim.TESnoise class for more info.

    Parameters
    ----------
    freqs : array
        Array of frequencies
    squiddc : float
        The average value for the white noise from the squid 
        (ignoring the 1/f component)
    squidpole : float
        The knee for the 1/f component of the noise
    squidn : float
        The factor for the 1/f^n noise
    rload : float
        Value of the load resistor in Ohms
    tload : float
        The temeperature of the load resistor in Kelvin
    rn : float
        The value of the resistance of the TES when normal
    tc : float
        The SC transistion temperature of the TES
    inductance : float
        The inductance of the TES line

    Returns
    -------
    s_tot : array
        Array of values corresponding to the theortical 
        normal state noise. 

    """
    
    omega = 2.0*np.pi*freqs
    dIdVnormal = 1.0/(rload+rn+1.0j*omega*inductance)
    s_vload = 4.0*constants.k*tload*rload * np.ones_like(freqs)
    s_iloadnormal = s_vload*np.abs(dIdVnormal)**2.0
    s_vtesnormal = 4.0*constants.k*tc*rn * np.ones_like(freqs)
    s_itesnormal = s_vtesnormal*np.abs(dIdVnormal)**2.0
    s_isquid = (squiddc*(1.0+(squidpole/freqs)**squidn))**2.0
    s_tot = s_iloadnormal+s_itesnormal+s_isquid

    return s_tot

def _sc_noise(freqs, tload, squiddc, squidpole, squidn, rload, inductance):
    """
    Functional form of the Super Conducting state noise. Including
    the johnson noise for the load resistor and the SQUID + downstream 
    electronics noise. See qetpy.sim.TESnoise class for more info.

    Parameters
    ----------
    freqs : array
        Array of frequencies
    tload : float
        The temeperature of the load resistor in Kelvin
    squiddc : float
        The average value for the white noise from the squid 
        (ignoring the 1/f component)
    squidpole : float
        The knee for the 1/f component of the noise
    squidn : float
        The factor for the 1/f^n noise
    rload : float
        Value of the load resistor in Ohms
    inductance : float
        The inductance of the TES line

    Returns
    -------
    s_tot : array
        Array of values corresponding to the theortical 
        SC state noise. 

    """
    
    omega = 2.0*np.pi*freqs
    dIdVsc = 1.0/(rload+1.0j*omega*inductance)
    s_vload = 4.0*constants.k*tload*rload * np.ones_like(freqs)    
    s_iloadsc = s_vload*np.abs(dIdVsc)**2.0 
    s_isquid = (squiddc*(1.0+(squidpole/freqs)**squidn))**2.0
    return s_iloadsc+s_isquid
    

class IVanalysis(object):
    """
    Class to aid in the analysis of an IV/dIdV sweep as processed by
    rqpy.proccess.process_ivsweep(). Currently only supports a single
    channel
    
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
    noiseinds : ndarray
        Array of booleans corresponding to the rows
        of the df that are noise type data
    didvinds : ndarray
        Array of booleans corresponding to the rows
        of the df that are didv type data
    norminds : range
        python built-in range type corresponding 
        to normal data points of the didv
    scinds : range
        python built-in range type corresponding 
        to SC data points of the didv
    traninds : range
        python built-in range type corresponding 
        to transition data points of the didv
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
    vb : ndarray
        Array of bias voltages
    vb_err : ndarray
        Array of uncertainties in the bais voltage
    dites : ndarray
        Array of DC offsets for IV/didv data
    dites : ndarray
        Array of uncertainties in the DC offsets
    tbath : float
        The bath temperature of the fridge
    tc : float
        The super conducting temperature of the 
        TESs
    Gta : float
        The thermal conductance of the TESs to the
        absorber
    squiddc : float
        The DC component of the squid+electronics noise
    squidpole : float
        The knee for the squid 1/f noise
    squidn : float
        The power of the 1/f^n noise for the squid
    inductance : float
        The inductance of the TES line
        
    """
    
    def __init__(self, df, nnorm, nsc, channels=None, channelname='', rshunt=5e-3, 
                 rshunt_err = 0.05*5e-3, tbath=0, tc=0, Gta=0, lgcremove_badseries = True, figsavepath=''):
        """
        Initialization of IVanalysis object. Note, currently only single channel analysis is supported
        
        Parameters
        ----------
        df : Pandas.core.DataFrame
            DataFrame of a processed IV/dIdV sweep returned from 
            rqpy._process_iv_didv.process_ivsweep()
        nnorm : int
            Number bias values where the TES was normal,
            Note: count only one per noise and didv point (don't double count!)
        nsc : int
            Number of bias values where the TES was Super Conducting,
            Note: count only one per noise and didv point (don't double count!)
        channels : list, optional
            A list of strings correponding to the channels to analyze. 
            Note, currently only single channel analysis is supported
        channelname : str, optional
            This is used if the user wished to label the channel as something
            other than the stored channel name. ie. channel = PBS1, channelname = PD2
        rshunt : float, optional
            The value of the shunt resistor in Ohms
        rshunt_err : float, optional
            The unccertainty in the value of the shunt resistor
        tbath : float, optional
            The temperature of the detector stack in Kelvin
        tc : float, optional
            The temperature of the SC transition for the TES
        Gta : float, optional
            The theremal conductance between the TES and the 
            absorber
        lgcremove_badseries : bool, optional
            If True, series where the SQUID lost lock, or the amplifier railed 
            are removed
        figsavepath : str, optional
            The path to the directory where the figures should be saved.
        
        """
        
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
        self.squiddc = None
        self.squidpole = None
        self.squidn = None
        
        if lgcremove_badseries:
            self.df = _remove_bad_series(df)
        
        self.noiseinds = (self.df.datatype == "noise")
        self.didvinds = (self.df.datatype == "didv")
        self.norminds = range(nnorm)
        self.scinds = range(len(self.df)//2-nsc, len(self.df)//2)
        self.traninds = range(self.norminds[-1]+1, self.scinds[0])
    
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
        
        self.tbath = tbath
        self.tc = tc
        self.Gta = Gta
        
        tempdidv = DIDV(1,1,1,1,1)
        self.df = self.df.assign(didvobj = tempdidv)
        
    
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
        lgcsave : bool, optional
            If True, all the plots will be saved in the a folder
            Avetrace_noise/ within the user specified directory
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
            
            self.df.iat[int(np.flatnonzero(self.didvinds)[ind]), self.df.columns.get_loc('didvobj')] = didvobjsc
            self.df.iat[int(np.flatnonzero(self.noiseinds)[ind]), self.df.columns.get_loc('didvobj')] = didvobjsc
            
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
        time offset using the **kwargs. Pass {'dt0' : 1.5e-6} (or other value) 
        or additionally try {'add180phase' : False}

        Parameters
        ----------
        lgcplot : bool, optional
            If True, the plots are shown for each fit
        lgcsave : bool, optional
            If True, all the plots will be saved in the a folder
            Avetrace_noise/ within the user specified directory
        **kwargs : dict
            Additional key word arguments to be passed to didvinitfromdata()

        Returns
        -------
        None
        
        """
        
        if self.rload is None:
            raise ValueError('rload has not been calculated yet, please fit rload first')
        rtot_list = []
        for ind in self.norminds:
            didvn = self.df[self.didvinds].iloc[ind]
            didvobjn = didvinitfromdata(didvn.avgtrace[:len(didvn.didvmean)], didvn.didvmean, 
                                         didvn.didvstd, didvn.offset, didvn.offset_err, 
                                         didvn.fs, didvn.sgfreq, didvn.sgamp, 
                                         rshunt = self.rshunt, rload=self.rload, **kwargs)
            didvobjn.dofit(1)
            rtot=didvobjn.fitparams1[0]
            rtot_list.append(rtot)

            self.df.iat[int(np.flatnonzero(self.didvinds)[ind]), self.df.columns.get_loc('didvobj')] = didvobjn
            self.df.iat[int(np.flatnonzero(self.noiseinds)[ind]), self.df.columns.get_loc('didvobj')] = didvobjn

            
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
        lgcsave : bool, optional
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
        lgcsave : bool, optional
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
            
        
    
    
    
    def fit_tran_didv(self, lgcplot=False, lgcsave=False):
        """
        Function to fit all the didv data in the IV sweep data
  
        Parameters
        ----------
        lgcplot : bool, optional
            If True, the plots are shown for each fit
        lgcsave : bool, optional
            If True, all the plots will be saved in the a folder
            Avetrace_noise/ within the user specified directory
        
        Returns
        -------
        None
        
        """ 
    
        for ind in (self.traninds):
    
            row = self.df[self.didvinds].iloc[ind]
            r0 = row.r0
            dr0 = row.r0_err
            priors = np.zeros(7)
            invpriorsCov = np.zeros((7,7))
            priors[0] = self.rload
            priors[1] = row.r0
            invpriorsCov[0,0] = 1.0/self.rshunt_err**2
            invpriorsCov[1,1] = 1.0/(dr0)**2


            didvobj = didvinitfromdata(row.avgtrace[:len(row.didvmean)], row.didvmean, row.didvstd, row.offset, 
                                       row.offset_err, row.fs, row.sgfreq, row.sgamp, rshunt=self.rshunt,  
                                       rload=self.rload, rload_err = self.rshunt_err, r0=r0, r0_err=dr0,
                                       priors = priors, invpriorscov = invpriorsCov)

            didvobj.dopriorsfit()
            didvobj.dofit(poles=2)
            
            self.df.iat[int(np.flatnonzero(self.didvinds)[ind]), self.df.columns.get_loc('didvobj')] = didvobj
            self.df.iat[int(np.flatnonzero(self.noiseinds)[ind]), self.df.columns.get_loc('didvobj')] = didvobj
            
            
            if lgcplot:
                didvobj.plot_full_trace(lgcsave=lgcsave, savepath=self.figsavepath,
                                          savename=f'didv_{row.qetbias:.3e}')
                didvobj.plot_re_im_didv(poles='all', plotpriors=True, lgcsave=lgcsave, 
                                        savepath=self.figsavepath,
                                        savename=f'didv_{row.qetbias:.3e}')

                
    
    def fit_normal_noise(self, fit_range=(10, 3e4), squiddc0=6e-12, squidpole0=200, squidn0=0.7,
                        lgcplot=False, lgcsave=False, xlims = None, ylims = None):
        """
        Function to fit the noise components of the SQUID+Electronics. Fits all normal noise PSDs
        and stores the average value for squiddc, squidpole, and squidn as attributes of the class.
        
        Parameters
        ----------
        fit_range : tuple, optional
            The frequency range over which to do the fit
        squiddc0 : float, optional
            Initial guess for the squiddc parameter
        squidpole0 : float, optional
            Initial guess for the squidpole parameter
        squidn0 : float, optional
            Initial guess for the squidn paramter
        lgcplot : bool, optional
            If True, a plot of the fit is shown
        lgcsave : bool, optional
            If True, the figure is saved
        xlims : NoneType, tuple, optional
            Limits to be passed to ax.set_xlim()
        ylims : NoneType, tuple, optional
            Limits to be passed to ax.set_ylim()
        
        Returns
        -------
        None
        
        """
        
        squiddc_list = []
        squidpole_list = []
        squidn_list = []
        
        
        for ind in self.norminds:
            noise_row = self.df[self.noiseinds].iloc[ind]
            f = noise_row.f
            psd = noise_row.psd
            
            inductance = noise_row.didvobj.get_irwinparams_dict(1)["L"]
            
            ind_lower = (np.abs(f - fit_range[0])).argmin()
            ind_upper = (np.abs(f - fit_range[1])).argmin()

            xdata = f[ind_lower:ind_upper]
            ydata = _flatten_psd(f,psd)[ind_lower:ind_upper]

            model = Model(_normal_noise, independent_vars=['freqs'])
            params = model.make_params(squiddc=squiddc0, squidpole=squidpole0,squidn=squidn0,
                                        rload = self.rload, tload = 0.0, rn = self.rn_iv, tc = self.tc,
                                        inductance = inductance)
            params['tc'].vary = False
            params['tload'].vary = False
            params['rload'].vary = False
            params['rn'].vary = False
            params['inductance'].vary = False
            result = model.fit(ydata, params, freqs = xdata)
            
            fitvals = result.values
    
            noise_sim = TESnoise(rload=self.rload, r0=self.rn_iv, rshunt=self.rshunt, inductance=inductance, 
                          beta=0, loopgain=0, tau0=0, G=0,qetbias=noise_row.qetbias, tc=self.tc, tload=0,
                          tbath=self.tbath, squiddc=fitvals['squiddc'], squidpole=fitvals['squidpole'], 
                          squidn=fitvals['squidn'])
            
            squiddc_list.append(fitvals['squiddc'])
            squidpole_list.append(fitvals['squidpole'])
            squidn_list.append(fitvals['squidn'])
            
            if lgcplot:
                _plot_n_noise(f, psd, noise_sim, noise_row.qetbias, self.figsavepath, lgcsave, xlims, ylims)
                
            
        self.squiddc = np.mean(squiddc_list)
        self.squidpole = np.mean(squidpole_list)
        self.squidn = np.mean(squidn_list)
                       
                       
    def fit_sc_noise(self, fit_range=(3e3, 1e5), lgcplot=False, lgcsave=False, xlims = None, ylims = None):
        """
        Function to fit the components of the SC Noise. Fits all SC noise PSDs
        and stores the average value for tload as an attribute of the class.
        
        Parameters
        ----------
        fit_range : tuple, optional
            The frequency range over which to do the fit
        lgcplot : bool, optional
            If True, a plot of the fit is shown
        lgcsave : bool, optional
            If True, the figure is saved
        xlims : NoneType, tuple, optional
            Limits to be passed to ax.set_xlim()
        ylims : NoneType, tuple, optional
            Limits to be passed to ax.set_ylim()    
            
        Returns
        -------
        None
        
        """
        
        if self.squidpole is None:
            raise AttributeError('You must fit the normal noise before fitting the SC noise')
                       
        tload_list = []
        
        
        for ind in self.scinds:
            noise_row = self.df[self.noiseinds].iloc[ind]
            f = noise_row.f
            psd = noise_row.psd
            inductance = noise_row.didvobj.get_irwinparams_dict(1)["L"]
            
            ind_lower = (np.abs(f - fit_range[0])).argmin()
            ind_upper = (np.abs(f - fit_range[1])).argmin()

            xdata = f[ind_lower:ind_upper]
            ydata = _flatten_psd(f,psd)[ind_lower:ind_upper]

            model = Model(_sc_noise, independent_vars=['freqs'])
            params = model.make_params(tload = 0.03, squiddc=self.squiddc, squidpole=self.squidpole,
                                       squidn=self.squidn, rload=self.rload, inductance=inductance)

            params['squiddc'].vary = False
            params['squidpole'].vary = False
            params['squidn'].vary = False
            params['rload'].vary = False
            params['inductance'].vary = False
            result = model.fit(ydata, params, freqs = xdata)
            
            fitvals = result.values
    
            noise_sim = TESnoise(rload=self.rload, r0=0.0001, rshunt=self.rshunt, inductance=inductance, 
                          beta=0, loopgain=0, tau0=0, G=0,qetbias=noise_row.qetbias, tc=self.tc, 
                          tload=fitvals['tload'], tbath=self.tbath, squiddc=self.squiddc, 
                          squidpole=self.squidpole, squidn=self.squidn)
            
            tload_list.append(fitvals['tload'])
            
            if lgcplot:
                _plot_sc_noise(f, psd, noise_sim, noise_row.qetbias, self.figsavepath, lgcsave, xlims, ylims)
            
        self.tload = np.mean(tload_list)
        
    def model_noise(self, tau_collect=20e-6, collection_eff=1, lgcplot=False, lgcsave=False, 
                    ylims_current = None, ylims_power = None):
        """
        Function to plot noise PSD with all the theoretical noise
        components (calculated from the didv fits). This function also estimates
        the expected energy resolution based on the power noise spectrum
        
        Parameters
        ----------
        tau_collect : float, optional
            The phonon collection time of the detector
        collection_eff : float, optional
            The absolute phonon collection efficiency of the detector
        lgcplot : bool, optional
            If True, a plot of the fit is shown
        lgcsave : bool, optional
            If True, the figure is saved
        ylims_current : NoneType, tuple, optional
            Limits to be passed to ax.set_ylim()
            for the current nosie plots
        ylims_power : NoneType, tuple, optional
            Limits to be passed to ax.set_ylim()  
            for the power noise plots
            
        Returns
        -------
        None 
        
        """
        
        
        energy_res_arr = np.full(shape = sum(self.noiseinds), fill_value = np.nan)
        tau_eff_arr = np.full(shape = sum(self.noiseinds), fill_value = np.nan)
        for ind in self.traninds:
            noise_row = self.df[self.noiseinds].iloc[ind]
            f = noise_row.f
            psd = noise_row.psd
            didvobj = noise_row.didvobj
            
            noise_sim = loadfromdidv(didvobj, G=self.Gta, qetbias=noise_row.qetbias, tc=self.tc, 
                                     tload=self.tload, tbath=self.tbath, squiddc=self.squiddc, 
                                     squidpole=self.squidpole, squidn=self.squidn,
                                     noisetype='transition', lgcpriors = True)
            if lgcplot:
                fig, ax = plot_noise_sim(f, psd, noise_sim, 'current')
                if ylims_current is not None:
                    ax.set_ylim(ylims_current)
                fig, ax = plot_noise_sim(f, psd, noise_sim, 'power')
                if ylims_power is not None:
                    ax.set_ylim(ylims_power)
                    
                if lgcsave:
                    plt.savefig(f'{self.figsavepath}T_noise_qetbias{noise_row.qetbias}.png')
                
            res = energy_res_estimate(freqs= f, tau_collect = tau_collect,
                                      Sp = psd/(np.abs(noise_sim.dIdP(f))**2),
                                      collection_eff = collection_eff)
            energy_res_arr[ind] = res
            
            tau_eff = didvobj.get_irwinparams_dict(2)['tau_eff']
            tau_eff_arr[ind] = tau_eff
            
            
        self.df.loc[self.noiseinds, 'energy_res'] =  energy_res_arr
        self.df.loc[self.didvinds, 'energy_res'] =  energy_res_arr
        self.df.loc[self.noiseinds, 'tau_eff'] =  tau_eff_arr
        self.df.loc[self.didvinds, 'tau_eff'] =  tau_eff_arr
        
    
    def find_optimum_bias(self, lgcplot=False, lgcsave=False, xlims = None, ylims = None):
        """
        Function to find the QET bias with the lowest energy 
        resolution. 
        
        Parameters
        ----------
        lgcplot : bool, optional
            If True, a plot of the fit is shown
        lgcsave : bool, optional
            If True, the figure is saved
        xlims : NoneType, tuple, optional
            Limits to be passed to ax.set_xlim()
        ylims : NoneType, tuple, optional
            Limits to be passed to ax.set_ylim()   
        
        Returns
        -------
        optimum_bias : float
            The QET bias (in Amperes) corresponding to the 
            lowest energy resolution
        optimum_r0 : float
            The resistance of the TES (in Ohms) corresponding to the 
            lowest energy resolution
        optimum_e : float
            The energy resolution (in eV) at the optimum bias point
        
        """
        
        trandf = self.df.loc[self.noiseinds].iloc[self.traninds]
        r0s = trandf.r0.values*1e3
        energy_res = trandf.energy_res.values
        qets = trandf.qetbias.values*1e6
        qets = qets.astype(int)


        emin_list = []
        r0min_list = []
        qetmin_list = []

        for ii in range(len(r0s)):
            if (ii+2) < len(r0s):
                x = r0s[ii:ii+3]
                q = qets[ii:ii+3]
                y = energy_res[ii:ii+3]

                xfit = np.linspace(x[0], x[-1], 25)
                qetfit = np.linspace(q[0], q[-1], 25)
                poly = np.polyfit(x, y, 2)
                yfit = np.poly1d(poly)(xfit)

                eminind = np.argmin(yfit)
                emin_list.append(yfit[eminind])
                r0min_list.append(xfit[eminind])
                qetmin_list.append(qetfit[eminind])

        eminind = np.argmin(emin_list)
        optimum_bias = qetmin_list[eminind]*1e-6
        optimum_r0 = r0min_list[eminind]*1e-3
        optimum_e = emin_list[eminind]
        
        if lgcplot:
            _plot_energy_res_vs_bias(r0s, energy_res, qets, optimum_r0, self.figsavepath, lgcsave, xlims, ylims)
        
        return optimum_bias, optimum_r0, optimum_e

        
    def make_noiseplots(self, lgcsave=False):
        """
        Helper function to plot average noise/didv traces in time domain, as well as 
        corresponding noise PSDs, for all QET bias points in IV/dIdV sweep.

        Parameters
        ----------
        lgcsave : bool, optional
            If True, all the plots will be saved in the a folder
            Avetrace_noise/ within the user specified directory

        Returns
        -------
        None

        """
        
        _make_iv_noiseplots(self, lgcsave)
    
    def plot_rload_rn_qetbias(self, lgcsave=False, xlims_rl=None, ylims_rl=None, xlims_rn=None, ylims_rn=None):
        """
        Helper function to plot rload and rnormal as a function of
        QETbias from the didv fits of SC and Normal data

        Parameters
        ----------
        lgcsave : bool, optional
            If True, all the plots will be saved 
        xlims_rl : NoneType, tuple, optional
            Limits to be passed to ax.set_xlim()for the 
            rload plot
        ylims_rl : NoneType, tuple, optional
            Limits to be passed to ax.set_ylim() for the
            rload plot
        xlims_rn : NoneType, tuple, optional
            Limits to be passed to ax.set_xlim()for the 
            rtot plot
        ylims_rn : NoneType, tuple, optional
            Limits to be passed to ax.set_ylim() for the
            rtot plot
            
        Returns
        -------
        None
        
        """
        
        _plot_rload_rn_qetbias(self, lgcsave, xlims_rl, ylims_rl, xlims_rn, ylims_rn)
        
        
            
    def do_full_analysis(self, collection_eff=1, lgcplot=True, lgcsave=True):
        """
        Function to perform full IV/dIdV analysis. This function simply 
        calls all the individual functions in this class in the 
        proper order.
        
        Note, if any step in the analysis fails, the analysis will stop
        and the user will need to do the analysis in parts. 
        
        Parameters
        ----------
        collection_eff : float, optional
            The absolute phonon collection efficiency of the detector, should be a float from 0 to 1.
        lgcplot : bool, optional
            If True, a plot of the fit is shown
        lgcsave : bool, optional
            If True, the figure is saved
        
        Returns
        -------
        success : bool
            Will return True if the full analysis is successful.
            
        """
        
        success = False
        
        try:
            self._fit_rload_didv(lgcplot=lgcplot, lgcsave=lgcsave)
        except:
            raise AnalysisError('An error occurred when fitting the SC dIdV to determine Rload. \n Please make sure the SC indices (scinds) are correct')
        
        try:
            self._fit_rn_didv(lgcplot=lgcplot, lgcsave=lgcsave)
        except:
            raise AnalysisError('An error occurred when fitting the Normal dIdV to determine Rn. \n Please make sure the Normal indices (norminds) are correct')
            
        try:
            self.analyze_sweep(lgcplot=lgcplot, lgcsave=lgcsave)
        except:
            raise AnalysisError('An error occurred when fitting the IV curve')
        
        try:
            self.plot_rload_rn_qetbias(lgcsave=lgcsave)
        except:
            raise AnalysisError('An error occurred when plotting Rload and Rn vs applied QETbias')
            
        try:
            self.fit_normal_noise(lgcplot=lcgplot, lgcsave=lgcsave)
        except:
            raise AnalysisError('An error occurred when trying to fit the SQUID and Electronics components \n of the normal state noise')
        
        try:
            self.fit_sc_noise(lgcplot=lgcplot, lgcsave=lgcsave)
        except:
            raise AnalysisError('An error occurred when trying to fit the temperature of the load resistor \n from the SC state noise')
            
        try:
            self.fit_tran_didv(lgcplot=lgcplot, lgcsave=lgcsave)
        except:
            raise AnalysisError('An error occurred when trying to fit the Transition state dIdV. \n Try changing the dt0 or add180phase parameter for the DIDV fits')
        
        try:
            self.model_noise(collection_eff=collection_eff, lgcplot=lgcplot, lgcsave=lgcsave)
        except:
            raise AnalysisError('An error occurred when trying to model the Transition state noise and estimate the theoretical energy resolution')
        
        try:
            self.find_optimum_bias()
        except:
            raise AnalysisError('An error occurred when fiding the optimum QET bias')
            
         
        success = True
        
        return success
            
            
        
