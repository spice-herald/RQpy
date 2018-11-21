import numpy as np
import pandas as pd
import os
import multiprocessing
from itertools import repeat
from rqpy import io
from qetpy import fitting

__all__ = ["SetupRQ", "rq"]

class SetupRQ(object):
    def __init__(self, templates, psds, fs):
        
        if len(templates) != len(psds):
            raise ValueError("templates and psds should have the same length")
        
        self.templates = templates
        self.psds = psds
        self.fs = fs
        self.nchan = len(templates)
        
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
        
    def adjust_ofamp_nodelay(self, lgcrun, calc_lowfreqchi2=False):
        self.do_ofamp_nodelay = lgcrun
        self.ofamp_nodelay_lowfreqchi2 = calc_lowfreqchi2
        
    def adjust_ofamp_unconstrained(self, lgcrun, calc_lowfreqchi2=False):
        self.do_ofamp_unconstrained = lgcrun
        self.ofamp_unconstrained_lowfreqchi2 = calc_lowfreqchi2
        
    def adjust_ofamp_constrained(self, lgcrun, calc_lowfreqchi2=True, nconstrain=80):
        if np.isscalar(nconstrain):
            nconstrain = [nconstrain]*self.nchan
        
        if len(nconstrain)!=self.nchan:
            raise ValueError("The length of nconstrain is not equal to the number of channels")
        
        self.do_ofamp_constrained = lgcrun
        self.ofamp_constrained_lowfreqchi2 = calc_lowfreqchi2
        self.ofamp_constrained_nconstrain = nconstrain
        
    def adjust_ofamp_pileup(self, lgcrun, nconstrain=80):
        if np.isscalar(nconstrain):
            nconstrain = [nconstrain]*self.nchan
        
        if len(nconstrain)!=self.nchan:
            raise ValueError("The length of nconstrain is not equal to the number of channels")
        
        self.do_ofamp_pileup = lgcrun
        self.ofamp_pileup_nconstrain = nconstrain
        
    def adjust_chi2_nopulse(self, lgcrun):
        self.do_chi2_nopulse = lgcrun
        
    def adjust_chi2_lowfreq(self, lgcrun, fcutoff=10000):
        if np.isscalar(fcutoff):
            fcutoff = [fcutoff]*self.nchan
        
        if len(fcutoff)!=self.nchan:
            raise ValueError("The length of fcutoff is not equal to the number of channels")
            
        self.do_chi2_lowfreq = lgcrun
        self.chi2_lowfreq_fcutoff = fcutoff
        
    def adjust_baseline(self, lgcrun, indbasepre=16000):
        if np.isscalar(indbasepre):
            indbasepre = [indbasepre]*self.nchan
        
        if len(indbasepre)!=self.nchan:
            raise ValueError("The length of indbasepre is not equal to the number of channels")
            
        self.do_baseline = lgcrun
        self.baseline_indbasepre = indbasepre
        
    def adjust_integral(self, lgcrun):
        self.do_integral = lgcrun
    
def _calc_rq(traces, channels, setup, readout_inds=None):
    
    if readout_inds is None:
        readout_inds = np.ones(len(traces), dtype=bool)
    
    rq_dict = {}
    
    for ii, chan in enumerate(channels):
        
        signal = traces[readout_inds, ii]
        template = setup.templates[ii]
        psd = setup.psds[ii]
        
        if setup.do_baseline:
            baseline = np.mean(signal[:, :setup.baseline_indbasepre[ii]], axis=-1)
            rq_dict[f'baseline_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f'baseline_{chan}'][readout_inds] = baseline
        
        if setup.do_integral:
            integral = np.trapz(signal - baseline[:, np.newaxis], axis=-1)/fs
            rq_dict[f'integral_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f'integral_{chan}'][readout_inds] = integral
        
        if setup.do_chi2_nopulse:
            chi0 = np.zeros(len(signal))
            for jj, s in enumerate(signal):
                chi0[jj] = fitting.chi2_nopulse(s, psd, fs)
            
            rq_dict[f'chi2_nopulse_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f'chi2_nopulse_{chan}'][readout_inds] = chi0
        
        if setup.do_ofamp_nodelay:
            amp_nodelay = np.zeros(len(signal))
            chi2_nodelay = np.zeros(len(signal))
            for jj, s in enumerate(signal):
                amp_nodelay[jj], _, chi2_nodelay[jj] = fitting.ofamp(s, template, psd, fs, withdelay=False)
            
            rq_dict[f'ofamp_nodelay_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f'ofamp_nodelay_{chan}'][readout_inds] = amp_nodelay
            rq_dict[f'chi2_nodelay_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f'chi2_nodelay_{chan}'][readout_inds] = chi2_nodelay
            
            if setup.ofamp_nodelay_lowfreqchi2 and setup.do_chi2_lowfreq:
                chi2low = np.zeros(len(signal))
                for jj, s in enumerate(signal):
                    chi2low[jj] = fitting.chi2lowfreq(s, template, amp_nodelay[jj], 
                                                      0, psd, fs, fcutoff=setup.chi2_lowfreq_fcutoff[ii])
                
                rq_dict[f'chi2_nodelay_{chan}'] = np.ones(len(traces))*(-999999.0)
                rq_dict[f'chi2lowfreq_nodelay_{chan}'][readout_inds] = chi2low

        if setup.do_ofamp_unconstrained:
            amp_noconstrain = np.zeros(len(signal))
            t0_noconstrain = np.zeros(len(signal))
            chi2_noconstrain = np.zeros(len(signal))
            for jj, s in enumerate(signal):
                amp_noconstrain[jj], t0_noconstrain[jj], chi2_noconstrain[jj] = fitting.ofamp(s, template, 
                                                                                          psd, fs, withdelay=True)
            
            rq_dict[f'ofamp_unconstrain_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f'ofamp_unconstrain_{chan}'][readout_inds] = amp_noconstrain
            rq_dict[f't0_unconstrain_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f't0_unconstrain_{chan}'][readout_inds] = t0_noconstrain
            rq_dict[f'chi2_unconstrain_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f'chi2_unconstrain_{chan}'][readout_inds] = chi2_noconstrain
            
            if setup.ofamp_unconstrained_lowfreqchi2 and setup.do_chi2_lowfreq:
                chi2low = np.zeros(len(signal))
                for jj, s in enumerate(signal):
                    chi2low[jj] = fitting.chi2lowfreq(s, template, amp_noconstrain[jj], t0_noconstrain[jj], 
                                                      psd, fs, fcutoff=setup.chi2_lowfreq_fcutoff[ii])
                
                rq_dict[f'chi2lowfreq_unconstrain_{chan}'] = np.ones(len(traces))*(-999999.0)
                rq_dict[f'chi2lowfreq_unconstrain_{chan}'][readout_inds] = chi2low

        if setup.do_ofamp_constrained:
            amp_constrain = np.zeros(len(signal))
            t0_constrain = np.zeros(len(signal))
            chi2_constrain = np.zeros(len(signal))
            for jj, s in enumerate(signal):
                amp_constrain[jj], t0_constrain[jj], chi2_constrain[jj] = fitting.ofamp(s, template, 
                                                                   psd, fs, withdelay=True,
                                                                   nconstrain=setup.ofamp_constrained_nconstrain[ii])
            
            rq_dict[f'ofamp_constrain_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f'ofamp_constrain_{chan}'][readout_inds] = amp_constrain
            rq_dict[f't0_constrain_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f't0_constrain_{chan}'][readout_inds] = t0_constrain
            rq_dict[f'chi2_constrain_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f'chi2_constrain_{chan}'][readout_inds] = chi2_constrain
            
            if setup.ofamp_constrained_lowfreqchi2 and setup.do_chi2_lowfreq:
                chi2low = np.zeros(len(signal))
                for jj, s in enumerate(signal):
                    chi2low[jj] = fitting.chi2lowfreq(s, template, amp_constrain[jj], t0_constrain[jj], 
                                                      psd, fs, fcutoff=setup.chi2_lowfreq_fcutoff[ii])
                
                rq_dict[f'chi2lowfreq_constrain_{chan}'] = np.ones(len(traces))*(-999999.0)
                rq_dict[f'chi2lowfreq_constrain_{chan}'][readout_inds] = chi2low

        if setup.do_ofamp_pileup:
            amp_pileup = np.zeros(len(signal))
            t0_pileup = np.zeros(len(signal))
            chi2_pileup = np.zeros(len(signal))
            for jj, s in enumerate(signal):
                _,_,amp_pileup[jj], t0_pileup[jj], chi2_pileup[jj] = fitting.ofamp_pileup(s, template, 
                                                                   psd, fs, a1=amp_constrain[jj], t1=t0_constrain[jj],
                                                                   nconstrain2=setup.ofamp_pileup_nconstrain[ii])
                
            rq_dict[f'ofamp_pileup_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f'ofamp_pileup_{chan}'][readout_inds] = amp_pileup
            rq_dict[f't0_pileup_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f't0_pileup_{chan}'][readout_inds] = t0_pileup
            rq_dict[f'chi2_pileup_{chan}'] = np.ones(len(traces))*(-999999.0)
            rq_dict[f'chi2_pileup_{chan}'][readout_inds] = chi2_pileup

    
    return rq_dict

def _rq(file, chan, det, setup, convtoamps, savepath='', lgcsavedumps=False):
    
    dump = file.split('/')[-1].split('_')[-1].split('.')[0]
    seriesnum = file.split('/')[-2]
    print(f"On Series: {seriesnum},  dump: {dump}")
    
    if isinstance(chan, str):
        chan = [chan]
    
    traces, info_dict = io.get_traces_per_dump([file], chan=chan, det=det, convtoamps=convtoamps)
    
    data = {}
    
    for key in info_dict.keys():
        data[key] = info_dict[key]
        
    readout_inds = np.array(data["readoutstatus"]) == 1
    
    rq_dict = _calc_rq(traces, chan, setup, readout_inds=readout_inds)
    
    for key in rq_dict.keys():
        data[key] = rq_dict[key]
    
    rq_df = pd.DataFrame.from_dict(data)
    
    if lgcsavedumps:
        rq_df.to_pickle(f'{savepath}rq_df_{seriesnum}_d{dump}.pkl')   

    return rq_df


def rq(filelist, chan, det, setup, savepath='', lgcsavedumps=False, nprocess=1):
    """
    Function for processing raw data to calculate RQs. Supports multiprocessing. Currently
    only supports a three channel set up.
    
    Parameters
    ----------
    filelist : list
        List of paths to each file that should be opened and processed
    chan : list
        List of the channels that will be processed
    det : str
        The detector ID that corresponds to the channels that will be processed
    setup : SetupRQ
        A SetupRQ class object. This object defines all of the different RQs that should be calculated 
        and specifies relevant parameters.
    savepath : str
        The path to where each dump should be saved, if lgcsavedumps is set to True.
    lgcsavedumps : bool
        Boolean flag for whether or not the DataFrame for each dump should be saved individually.
        Useful for saving data as the processing routine is run, allowing checks of the data during
        run time.
    nprocess : int, optional
        The number of processes that should be used when multiprocessing. The default is 1.
    
    Returns
    -------
    rq_df : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for each dataset.
    
    """
    
    if isinstance(filelist, str):
        filelist = [filelist]
    
    convtoamps = []
    folder = os.path.split(filelist[0])[0]
    
    for ch in chan:
        convtoamps.append(io.get_trace_gain(folder, ch, det)[0])
    
    pool = multiprocessing.Pool(processes = nprocess)
    results = pool.starmap(_rq, zip(filelist, repeat([chan, det, setup, convtoamps, savepath, lgcsavedumps])))
    pool.close()
    pool.join()
    rq_df = pd.concat([df for df in results], ignore_index = True)
    
    return rq_df

