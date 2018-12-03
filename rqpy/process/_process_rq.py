import numpy as np
import pandas as pd
import os
import multiprocessing
from itertools import repeat
from rqpy import io
from qetpy.detcal import _fitting as fitting
from rqpy import HAS_SCDMSPYTOOLS

if HAS_SCDMSPYTOOLS:
    from scdmsPyTools.BatTools.IO import getRawEvents, getDetectorSettings

__all__ = ["SetupRQ", "rq"]

class SetupRQ(object):
    """
    Class for setting up the calculation of RQs when processing data.
    
    Attributes
    ----------
    templates : list
        List of pulse templates corresponding to each channel. The pulse templates should
        be normalized.
    psds : list
        List of PSDs coresponding to each channel. Should be two-sided PSDs, with units of A^2/Hz.
    fs : float
        The digitization rate of the data in Hz.
    summed_template : ndarray
        The pulse template for all of the channels summed together to be used when calculating
        RQs. Should be normalized to have a maximum height of 1. If not set, then the RQs for 
        the sum of the channels will not be calculated.
    summed_psd : ndarray
        The PSD corresponding to all of the channels summed together to be used when calculating
        RQs. Should be a two-sided PSD, with units of A^2/Hz. If not set, then the RQs for 
        the sum of the channels will not be calculated.
    calcchans : bool
        Boolean flag for whether or not to calculate the RQs for each of the individual 
        channels.
    calcsum : bool
        Boolean flag for whether or not calculate the RQs for the sum of the channels.
        Requires summed_template and summed_psd to be set when initializing the SetupRQ
        object.
    nchan : int
        The number of channels to be processed.
    do_ofamp_nodelay : bool
        Boolean flag for whether or not to do the optimum filter fit with no time
        shifting.
    ofamp_nodelay_lowfreqchi2 : bool
        Boolean flag for whether or not to calculate the low frequency chi-squared for 
        the optimum filter fit with no time shifting.
    do_ofamp_unconstrained : bool
        Boolean flag for whether or not to do the optimum filter fit with unconstrained time
        shifting.
    ofamp_unconstrained_lowfreqchi2 : bool
        Boolean flag for whether or not to calculate the low frequency chi-squared for 
        the optimum filter fit with unconstrained time shifting.
    do_ofamp_constrained : bool
        Boolean flag for whether or not to do the optimum filter fit with constrained time
        shifting.
    ofamp_constrained_lowfreqchi2 : bool
        Boolean flag for whether or not to calculate the low frequency chi-squared for 
        the optimum filter fit with constrained time shifting.
    ofamp_constrained_nconstrain : list
        The length of the window (in bins), centered on the middle of the trace, to constrain 
        the possible time shift values to when doing the optimum filter fit with constrained time shifting.
    do_ofamp_pileup : bool
        Boolean flag for whether or not to do the pileup optimum filter fit.
    ofamp_pileup_nconstrain : list
        The length of the window (in bins), centered on the middle of the trace, outside of which to 
        constrain the possible time shift values to when searching for a pileup pulse using ofamp_pileup.
    do_chi2_nopulse : bool
        Boolean flag for whether or not to calculate the chi^2 for no pulse.
    do_chi2_lowfreq : bool
        Boolean flag for whether or not to calculate the low frequency chi^2 for any of the fits.
    chi2_lowfreq_fcutoff : list
        The frequency cutoff for the calculation of the low frequency chi^2, units of Hz.
    do_baseline : bool
        Boolean flag for whether or not to calculate the DC baseline for each trace.
    baseline_indbasepre : int
        The number of indices up to which a trace should be averaged to determine the baseline.
    do_integral : bool
        Boolean flag for whether or not to calculate the baseline-subtracted integral of each trace.
    
    """
    
    def __init__(self, templates, psds, fs, summed_template=None, summed_psd=None):
        """
        Initialization of the SetupRQ class.
        
        Parameters
        ----------
        templates : list
            List of pulse templates corresponding to each channel. The pulse templates should
            be normalized to have a maximum height of 1.
        psds : list
            List of PSDs coresponding to each channel. Should be two-sided PSDs, with units of A^2/Hz.
        fs : float
            The digitization rate of the data in Hz.
        summed_template : ndarray, optional
            The pulse template for all of the channels summed together to be used when calculating
            RQs. Should be normalized to have a maximum height of 1. If not set, then the RQs for 
            the sum of the channels will not be calculated.
        summed_psd : ndarray, optional
            The PSD corresponding to all of the channels summed together to be used when calculating
            RQs. Should be a two-sided PSD, with units of A^2/Hz. If not set, then the RQs for 
            the sum of the channels will not be calculated.
        
        """
        
        if len(templates) != len(psds):
            raise ValueError("templates and psds should have the same length")
        
        self.templates = templates
        self.psds = psds
        self.fs = fs
        self.nchan = len(templates)
        
        self.summed_template = summed_template
        self.summed_psd = summed_psd
        
        self.calcchans=True
        if summed_template is None or summed_psd is None:
            self.calcsum=False
        else:
            self.calcsum=True
        
        self.do_ofamp_nodelay = True
        self.ofamp_nodelay_lowfreqchi2 = False
        
        self.do_ofamp_unconstrained = True
        self.ofamp_unconstrained_lowfreqchi2 = False
        
        self.do_ofamp_constrained = True
        self.ofamp_constrained_lowfreqchi2 = True
        self.ofamp_constrained_nconstrain = [80]*self.nchan
        
        self.do_ofamp_pileup = True
        self.ofamp_pileup_nconstrain = [80]*self.nchan
        
        self.do_chi2_nopulse = True
        
        self.do_chi2_lowfreq = True
        self.chi2_lowfreq_fcutoff = [10000]*self.nchan
        
        self.do_baseline = True
        self.baseline_indbasepre = [16000]*self.nchan
        
        self.do_integral = True
        
    def adjust_calc(self, lgcchans=True, lgcsum=True):
        """
        Method for adjusting the calculation of RQs for each individual channel and the sum
        of the channels.
        
        Parameters
        ----------
        lgcchans : bool, optional
            Boolean flag for whether or not to calculate the RQs for each of the individual 
            channels. Default is True.
        lgcsum : bool, optional
            Boolean flag for whether or not calculate the RQs for the sum of the channels.
            Requires summed_template and summed_psd to be set when initializing the SetupRQ
            object. Default is True.
            
        Raises
        ------
        ValueError
            A ValueError is raised if lgcsum is set to True when the SetupRQ Object was not
            initialized with summed_template or summed_psd.
        
        """
        
        self.calcchans = lgcchans
        
        if (self.summed_template is None or self.summed_psd is None) and lgcsum:
            raise ValueError("SetupRQ was not initialized with summed_template or summed_psd, cannot calculate the summed RQs")
        else:
            self.calcsum = lgcsum
        
    def adjust_ofamp_nodelay(self, lgcrun=True, calc_lowfreqchi2=False):
        """
        Method for adjusting the calculation of the optimum filter fit with no time
        shifting.
        
        Parameters
        ----------
        lgcrun : bool, optional
            Boolean flag for whether or not the optimum filter fit with no time
            shifting should be calculated.
        calc_lowfreqchi2 : bool, optional
            Boolean flag for whether or not the low frequency chi^2 of this fit 
            should be calculated. The low frequency chi^2 calculation should be adjusted
            using the adjust_chi2_lowfreq method.
        
        """
        
        self.do_ofamp_nodelay = lgcrun
        self.ofamp_nodelay_lowfreqchi2 = calc_lowfreqchi2
        
    def adjust_ofamp_unconstrained(self, lgcrun=True, calc_lowfreqchi2=False):
        """
        Method for adjusting the calculation of the optimum filter fit with unconstrained 
        time shifting.
        
        Parameters
        ----------
        lgcrun : bool, optional
            Boolean flag for whether or not the optimum filter fit with unconstrained 
            time shifting should be calculated.
        calc_lowfreqchi2 : bool, optional
            Boolean flag for whether or not the low frequency chi^2 of this fit 
            should be calculated. The low frequency chi^2 calculation should be adjusted
            using the adjust_chi2_lowfreq method. Default is False.
        
        """
        
        self.do_ofamp_unconstrained = lgcrun
        self.ofamp_unconstrained_lowfreqchi2 = calc_lowfreqchi2
        
    def adjust_ofamp_constrained(self, lgcrun=True, calc_lowfreqchi2=True, nconstrain=80):
        """
        Method for adjusting the calculation of the optimum filter fit with constrained 
        time shifting.
        
        Parameters
        ----------
        lgcrun : bool, optional
            Boolean flag for whether or not the optimum filter fit with constrained 
            time shifting should be calculated.
        calc_lowfreqchi2 : bool, optional
            Boolean flag for whether or not the low frequency chi^2 of this fit 
            should be calculated. The low frequency chi^2 calculation should be adjusted
            using the adjust_chi2_lowfreq method. Default is True.
        nconstrain : int, list of int, optional
            The length of the window (in bins), centered on the middle of the trace, 
            to constrain the possible time shift values to when doing the optimum filter 
            fit with constrained time shifting. Can be set to a list of values, if the 
            constrain window should be different for each channel. The length of the list should
            be the same length as the number of channels.
        
        """
        
        if np.isscalar(nconstrain):
            nconstrain = [nconstrain]*self.nchan
        
        if len(nconstrain)!=self.nchan:
            raise ValueError("The length of nconstrain is not equal to the number of channels")
        
        self.do_ofamp_constrained = lgcrun
        self.ofamp_constrained_lowfreqchi2 = calc_lowfreqchi2
        self.ofamp_constrained_nconstrain = nconstrain
        
    def adjust_ofamp_pileup(self, lgcrun=True, nconstrain=80):
        """
        Method for adjusting the calculation of the pileup optimum filter fit.
        
        Parameters
        ----------
        lgcrun : bool, optional
            Boolean flag for whether or not the pileup optimum filter fit should be calculated.
        nconstrain : int, list of int, optional
            The length of the window (in bins), centered on the middle of the trace, outside 
            of which to constrain the possible time shift values to when searching for a 
            pileup pulse using ofamp_pileup. Can be set to a list of values, if the constrain 
            window should be different for each channel. The length of the list should
            be the same length as the number of channels.
        
        """
        
        if np.isscalar(nconstrain):
            nconstrain = [nconstrain]*self.nchan
        
        if len(nconstrain)!=self.nchan:
            raise ValueError("The length of nconstrain is not equal to the number of channels")
        
        self.do_ofamp_pileup = lgcrun
        self.ofamp_pileup_nconstrain = nconstrain
        
    def adjust_chi2_nopulse(self, lgcrun=True):
        """
        Method for adjusting the calculation of the no pulse chi^2.
        
        Parameters
        ----------
        lgcrun : bool, optional
            Boolean flag for whether or not to calculate the chi^2 for no pulse.
        
        """
        
        self.do_chi2_nopulse = lgcrun
        
    def adjust_chi2_lowfreq(self, lgcrun=True, fcutoff=10000):
        """
        Method for adjusting the calculation of the low frequency chi^2.
        
        Parameters
        ----------
        lgcrun : bool, optional
            Boolean flag for whether or not the low frequency chi^2 should be calculated
            for any of the optimum filter fits.
        fcutoff : float, list of float, optional
            The frequency cutoff for the calculation of the low frequency chi^2, units of Hz.
            Can be set to a list of values, if the frequency cutoff should be different for 
            each channel. The length of the list should be the same length as the number 
            of channels.
            
        """
        
        if np.isscalar(fcutoff):
            fcutoff = [fcutoff]*self.nchan
        
        if len(fcutoff)!=self.nchan:
            raise ValueError("The length of fcutoff is not equal to the number of channels")
            
        self.do_chi2_lowfreq = lgcrun
        self.chi2_lowfreq_fcutoff = fcutoff
        
    def adjust_baseline(self, lgcrun=True, indbasepre=16000):
        """
        Method for adjusting the calculation of the DC baseline.
        
        Parameters
        ----------
        lgcrun : bool, optional
            Boolean flag for whether or not the DC baseline should be calculated. It is highly
            recommended to set this to true if the integral will be calculated, so that the
            baseline can be subtracted.
        indbasepre : int, list of int, optional
            The number of indices up to which a trace should be averaged to determine the baseline.
            Can be set to a list of values, if indbasepre should be different for each channel. 
            The length of the list should be the same length as the number of channels.
            
        """
        
        if np.isscalar(indbasepre):
            indbasepre = [indbasepre]*self.nchan
        
        if len(indbasepre)!=self.nchan:
            raise ValueError("The length of indbasepre is not equal to the number of channels")
            
        self.do_baseline = lgcrun
        self.baseline_indbasepre = indbasepre
        
    def adjust_integral(self, lgcrun=True):
        """
        Method for adjusting the calculation of the integral.
        
        Parameters
        ----------
        lgcrun : bool, optional
            Boolean flag for whether or not the integral should be calculated. If self.do_baseline
            is True, then the baseline is subtracted from the integral. If self.do_baseline is False,
            then the baseline is not subtracted. It is recommended that the baseline should be subtracted.
            
        """
        
        self.do_integral = lgcrun
        
def _calc_rq_single_channel(signal, template, psd, setup, readout_inds, chan, chan_num):
    """
    Helper function for calculating RQs for an array of traces corresponding to a single channel.
    
    Parameters
    ----------
    signal : ndarray
        Array of traces to use in calculation of RQs. Should be of shape (number of traces,
        length of trace)
    template : ndarray
        The pulse template to be used for the optimum filter (should be normalized beforehand).
    psd : ndarray
        The two-sided psd that will be used to describe the noise in the signal (in Amps^2/Hz)
    setup : SetupRQ
        A SetupRQ class object. This object defines all of the different RQs that should be calculated 
        and specifies relevant parameters.
    readout_inds : ndarray of bool
        Boolean mask that specifies which traces should be used to calculate the RQs. RQs for the 
        excluded traces are set to -999999.0. 
    chan : str
        Name of the channel that is being processed.
    chan_num : int
        The corresponding number for the channel being processed.
    
    Returns
    -------
    rq_dict : dict
        A dictionary containing all of the RQs that were calculated (as specified by the setup object).
    
    """
    
    rq_dict = {}
    
    fs = setup.fs
    
    if setup.do_baseline:
        baseline = np.mean(signal[:, :setup.baseline_indbasepre[chan_num]], axis=-1)
        rq_dict[f'baseline_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'baseline_{chan}'][readout_inds] = baseline

    if setup.do_integral:
        if setup.do_baseline:
            integral = np.trapz(signal - baseline[:, np.newaxis], axis=-1)/fs
        else:
            integral = np.trapz(signal, axis=-1)/fs
        rq_dict[f'integral_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'integral_{chan}'][readout_inds] = integral

    if setup.do_chi2_nopulse:
        chi0 = np.zeros(len(signal))
        for jj, s in enumerate(signal):
            chi0[jj] = fitting.chi2_nopulse(s, psd, fs)

        rq_dict[f'chi2_nopulse_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_nopulse_{chan}'][readout_inds] = chi0

    if setup.do_ofamp_nodelay:
        amp_nodelay = np.zeros(len(signal))
        chi2_nodelay = np.zeros(len(signal))
        for jj, s in enumerate(signal):
            amp_nodelay[jj], _, chi2_nodelay[jj] = fitting.ofamp(s, template, psd, fs, withdelay=False)

        rq_dict[f'ofamp_nodelay_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_nodelay_{chan}'][readout_inds] = amp_nodelay
        rq_dict[f'chi2_nodelay_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_nodelay_{chan}'][readout_inds] = chi2_nodelay

        if setup.ofamp_nodelay_lowfreqchi2 and setup.do_chi2_lowfreq:
            chi2low = np.zeros(len(signal))
            for jj, s in enumerate(signal):
                chi2low[jj] = fitting.chi2lowfreq(s, template, amp_nodelay[jj], 
                                                  0, psd, fs, fcutoff=setup.chi2_lowfreq_fcutoff[chan_num])

            rq_dict[f'chi2_nodelay_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'chi2lowfreq_nodelay_{chan}'][readout_inds] = chi2low

    if setup.do_ofamp_unconstrained:
        amp_noconstrain = np.zeros(len(signal))
        t0_noconstrain = np.zeros(len(signal))
        chi2_noconstrain = np.zeros(len(signal))
        for jj, s in enumerate(signal):
            amp_noconstrain[jj], t0_noconstrain[jj], chi2_noconstrain[jj] = fitting.ofamp(s, template, 
                                                                                      psd, fs, withdelay=True)

        rq_dict[f'ofamp_unconstrain_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_unconstrain_{chan}'][readout_inds] = amp_noconstrain
        rq_dict[f't0_unconstrain_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_unconstrain_{chan}'][readout_inds] = t0_noconstrain
        rq_dict[f'chi2_unconstrain_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_unconstrain_{chan}'][readout_inds] = chi2_noconstrain

        if setup.ofamp_unconstrained_lowfreqchi2 and setup.do_chi2_lowfreq:
            chi2low = np.zeros(len(signal))
            for jj, s in enumerate(signal):
                chi2low[jj] = fitting.chi2lowfreq(s, template, amp_noconstrain[jj], t0_noconstrain[jj], 
                                                  psd, fs, fcutoff=setup.chi2_lowfreq_fcutoff[chan_num])

            rq_dict[f'chi2lowfreq_unconstrain_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'chi2lowfreq_unconstrain_{chan}'][readout_inds] = chi2low

    if setup.do_ofamp_constrained:
        amp_constrain = np.zeros(len(signal))
        t0_constrain = np.zeros(len(signal))
        chi2_constrain = np.zeros(len(signal))
        for jj, s in enumerate(signal):
            amp_constrain[jj], t0_constrain[jj], chi2_constrain[jj] = fitting.ofamp(s, template, 
                                                               psd, fs, withdelay=True,
                                                               nconstrain=setup.ofamp_constrained_nconstrain[chan_num])

        rq_dict[f'ofamp_constrain_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_constrain_{chan}'][readout_inds] = amp_constrain
        rq_dict[f't0_constrain_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_constrain_{chan}'][readout_inds] = t0_constrain
        rq_dict[f'chi2_constrain_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_constrain_{chan}'][readout_inds] = chi2_constrain

        if setup.ofamp_constrained_lowfreqchi2 and setup.do_chi2_lowfreq:
            chi2low = np.zeros(len(signal))
            for jj, s in enumerate(signal):
                chi2low[jj] = fitting.chi2lowfreq(s, template, amp_constrain[jj], t0_constrain[jj], 
                                                  psd, fs, fcutoff=setup.chi2_lowfreq_fcutoff[chan_num])

            rq_dict[f'chi2lowfreq_constrain_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'chi2lowfreq_constrain_{chan}'][readout_inds] = chi2low

    if setup.do_ofamp_pileup:
        amp_pileup = np.zeros(len(signal))
        t0_pileup = np.zeros(len(signal))
        chi2_pileup = np.zeros(len(signal))
        for jj, s in enumerate(signal):
            _,_,amp_pileup[jj], t0_pileup[jj], chi2_pileup[jj] = fitting.ofamp_pileup(s, template, 
                                                               psd, fs, a1=amp_constrain[jj], t1=t0_constrain[jj],
                                                               nconstrain2=setup.ofamp_pileup_nconstrain[chan_num])

        rq_dict[f'ofamp_pileup_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_pileup_{chan}'][readout_inds] = amp_pileup
        rq_dict[f't0_pileup_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_pileup_{chan}'][readout_inds] = t0_pileup
        rq_dict[f'chi2_pileup_{chan}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_pileup_{chan}'][readout_inds] = chi2_pileup
        
    return rq_dict
    
def _calc_rq(traces, channels, setup, readout_inds=None):
    """
    Helper function for calculating RQs for arrays of traces.
    
    Parameters
    ----------
    traces : ndarray
        Array of traces to use in calculation of RQs. Should be of shape (number of traces,
        number of channels, length of trace)
    channels : list
        List of the channels that will be processed
    setup : SetupRQ
        A SetupRQ class object. This object defines all of the different RQs that should be calculated 
        and specifies relevant parameters.
    readout_inds : ndarray of bool, optional
        Boolean mask that specifies which traces should be used to calculate the RQs. RQs for the 
        excluded traces are set to -999999.0. 
    
    Returns
    -------
    rq_dict : dict
        A dictionary containing all of the RQs that were calculated (as specified by the setup object).
    
    """
    
    if readout_inds is None:
        readout_inds = np.ones(len(traces), dtype=bool)
    
    rq_dict = {}
    
    if setup.calcchans:
        for ii, chan in enumerate(channels):

            signal = traces[readout_inds, ii]
            template = setup.templates[ii]
            psd = setup.psds[ii]

            chan_dict = _calc_rq_single_channel(signal, template, psd, setup, readout_inds, chan, ii)

            rq_dict.update(chan_dict)
            
    if setup.calcsum:
        signal = traces[readout_inds].sum(axis=1)
        template = setup.summed_template
        psd = setup.summed_psd
        chan = "sum"

        sum_dict = _calc_rq_single_channel(signal, template, psd, setup, readout_inds, chan, 0)

        rq_dict.update(sum_dict)
    
    return rq_dict

def _rq(file, channels, det, setup, convtoamps, savepath, lgcsavedumps, filetype):
    """
    Helper function for processing raw data to calculate RQs for single files.
    
    Parameters
    ----------
    file : str
        Path to a file that should be opened and processed
    channels : list of str
        List of the channels that will be processed
    det : str
        The detector ID that corresponds to the channels that will be processed
    setup : SetupRQ
        A SetupRQ class object. This object defines all of the different RQs that should be calculated 
        and specifies relevant parameters.
    convtoamps : list
        List of the factors for each channel that will convert the units to Amps.
    savepath : str
        The path to where each dump should be saved, if lgcsavedumps is set to True.
    lgcsavedumps : bool
        Boolean flag for whether or not the DataFrame for each dump should be saved individually.
        Useful for saving data as the processing routine is run, allowing checks of the data during
        run time.
    filetype : str
        The string that corresponds to the file type that will be opened. Supports two 
        types -"mid.gz" and "npz".
    
    Returns
    -------
    rq_df : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for the dataset specified.
    
    """
    
    if filetype == "mid.gz" and not HAS_SCDMSPYTOOLS:
        raise ImportError("Cannot use filetype mid.gz because scdmsPyTools is not installed.")
        
    if filetype == "mid.gz":
        seriesnum = file.split('/')[-2]
        dump = file.split('/')[-1].split('_')[-1].split('.')[0]
    elif filetype == "npz":
        seriesnum = file.split('/')[-1].split('.')[0]
        dump = int(seriesnum.split('_')[-1])
        
    print(f"On Series: {seriesnum},  dump: {dump}")
    
    if isinstance(channels, str):
        channels = [channels]
    
    if filetype == "mid.gz":
        traces, info_dict = io.get_traces_midgz([file], chan=channels, det=det, convtoamps=convtoamps)
    elif filetype == "npz":
        traces, info_dict = io.get_traces_npz([file])
    
    data = {}
    
    data.update(info_dict)
    
    if filetype == "mid.gz":
        readout_inds = np.array(data["readoutstatus"]) == 1
    elif filetype == "npz":
        readout_inds = None
    
    rq_dict = _calc_rq(traces, channels, setup, readout_inds=readout_inds)
    
    data.update(rq_dict)
    
    rq_df = pd.DataFrame.from_dict(data)
    
    if lgcsavedumps:
        rq_df.to_pickle(f'{savepath}rq_df_{seriesnum}_d{dump}.pkl')   

    return rq_df


def rq(filelist, channels, setup, det="Z1", savepath='', lgcsavedumps=False, nprocess=1, filetype="mid.gz"):
    """
    Function for processing raw data to calculate RQs. Supports multiprocessing.
    
    Parameters
    ----------
    filelist : list
        List of paths to each file that should be opened and processed
    channels : list of str
        List of the channel names that will be processed. Used when naming RQs. When filetype is "mid.gz", 
        this is also used when reading the traces from each file.
    setup : SetupRQ
        A SetupRQ class object. This object defines all of the different RQs that should be calculated 
        and specifies relevant parameters.
    det : str, optional
        The detector ID that corresponds to the channels that will be processed. Set to "Z1" by default.
        Only used when filetype is "mid.gz"
    savepath : str
        The path to where each dump should be saved, if lgcsavedumps is set to True.
    lgcsavedumps : bool
        Boolean flag for whether or not the DataFrame for each dump should be saved individually.
        Useful for saving data as the processing routine is run, allowing checks of the data during
        run time.
    nprocess : int, optional
        The number of processes that should be used when multiprocessing. The default is 1.
    filetype : str, optional
        The string that corresponds to the file type that will be opened. Supports two 
        types -"mid.gz" and "npz". "mid.gz" is the default.
    
    Returns
    -------
    rq_df : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for each dataset in filelist.
    
    """
    
    if filetype == "mid.gz" and not HAS_SCDMSPYTOOLS:
        raise ImportError("Cannot use filetype mid.gz because scdmsPyTools is not installed.")
    
    if isinstance(filelist, str):
        filelist = [filelist]
    
    folder = os.path.split(filelist[0])[0]
    
    if filetype == "mid.gz":
        convtoamps = []
        for ch in channels:
            convtoamps.append(io.get_trace_gain(folder, ch, det)[0])
    elif filetype == "npz":
        convtoamps = [1]*len(channels)
    
    if nprocess == 1:
        results = []
        for f in filelist:
            results.append(_rq(f, channels, det, setup, convtoamps, savepath, lgcsavedumps, filetype))
    else:
        pool = multiprocessing.Pool(processes = nprocess)
        results = pool.starmap(_rq, zip(filelist, repeat(channels), repeat(det), repeat(setup), 
                                        repeat(convtoamps), repeat(savepath), repeat(lgcsavedumps),
                                        repeat(filetype)))
        pool.close()
        pool.join()
    
    rq_df = pd.concat([df for df in results], ignore_index = True)
    
    return rq_df

