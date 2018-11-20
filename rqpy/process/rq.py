import numpy as np
import pandas as pd
import multiprocessing
from itertools import repeat
from qetpy.fitting import ofamp, OFnonlin, MuonTailFit, chi2lowfreq, ofamp_pileup

__all__ = ["process_rq"]

def _process_rq(file, params):
    """
    Function for processing raw data to calculate RQs, which is called by process_rq. Currently
    only supports a three channel set up.
    
    Parameters
    ----------
    file : str
        The path to the file that should be opened and processed
    params : list
        List of the relevant parameters for calculation of RQs. The parameters are chan, det, 
        convtoamps, template, psds, fs, time, indbasepre, indbasepost, savepath, and lgcsavedumps.
        The process_rq function documents each of these parameters.
    
    Returns
    -------
    df_temp : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for this file.
    
    """
    chan, det, convtoamps, template, psds, fs, time, indbasepre, indbasepost, savepath, lgcsavedumps = params
    
    traces, rq_dict = get_traces_per_dump([file], chan=chan, det=det, convtoamps=convtoamps)
    
    columns = [f'baseline_{chan[0]}', f'baseline_{chan[1]}', f'baseline_{chan[2]}',
               f'int_bsSub_{chan[0]}', f'int_bsSub_{chan[1]}', f'int_bsSub_{chan[2]}',
               f'ofAmps_nodelay_{chan[0]}', f'ofAmps_nodelay_{chan[1]}', f'ofAmps_nodelay_{chan[2]}',
               f'chi2_nodelay_{chan[0]}', f'chi2_nodelay_{chan[1]}', f'chi2_nodelay_{chan[2]}',
               f'ofAmps_delay_{chan[0]}', f'ofAmps_delay_{chan[1]}', f'ofAmps_delay_{chan[2]}',
               f'chi2_delay_{chan[0]}', f'chi2_delay_{chan[1]}', f'chi2_delay_{chan[2]}',
               f't0_delay_{chan[0]}', f't0_delay_{chan[1]}', f't0_delay_{chan[2]}',
               f'ofAmps_uncon_{chan[0]}', f'ofAmps_uncon_{chan[1]}', f'ofAmps_uncon_{chan[2]}',
               f'chi2_uncon_{chan[0]}', f'chi2_uncon_{chan[1]}', f'chi2_uncon_{chan[2]}',
               f't0_uncon_{chan[0]}', f't0_uncon_{chan[1]}', f't0_uncon_{chan[2]}',
               f'ofAmps_pileup_{chan[0]}', f'ofAmps_pileup_{chan[1]}', f'ofAmps_pileup_{chan[2]}',
               f'chi2_pileup_{chan[0]}', f'chi2_pileup_{chan[1]}', f'chi2_pileup_{chan[2]}',
               f't0_pileup_{chan[0]}', f't0_pileup_{chan[1]}', f't0_pileup_{chan[2]}',
               'chi2_lowfreq', 'eventNumber', 'eventTime', 'seriesNumber', 'triggerType', 
               'triggerAmp',"readoutStatus", "pollingEndTime", "triggerTime", "deadTime", "liveTime", 
               "seriesTime", "triggerVetoReadoutTime", "waveformReadEndTime", "waveformReadStartTime"]
    
    temp_data = {}
    for item in columns:
        temp_data[item] = []
    dump = file.split('/')[-1].split('_')[-1].split('.')[0]
    seriesnum = file.split('/')[-2]
    print(f"On Series: {seriesnum},  dump: {dump}")
    
    psd1 = psds[0]
    psd2 = psds[1]
    psd3 = psds[2]
    
    template1 = template[0]
    template2 = template[1]
    template3 = template[2]
    
    for ii, trace_full in enumerate(traces):
        
        temp_data['eventNumber'].append(rq_dict["eventnumber"][ii])
        temp_data['eventTime'].append(rq_dict["eventtime"][ii])
        temp_data['seriesNumber'].append(rq_dict["seriesnumber"][ii])
        temp_data['triggerType'].append(rq_dict["triggertype"][ii])
        temp_data['triggerAmp'].append(rq_dict["triggeramp"][ii])
        temp_data['readoutStatus'].append(rq_dict["readoutstatus"][ii])
        temp_data['pollingEndTime'].append(rq_dict["pollingendtime"][ii])
        temp_data['triggerTime'].append(rq_dict["triggertime"][ii])
        temp_data['deadTime'].append(rq_dict["deadtime"][ii])
        temp_data['liveTime'].append(rq_dict["livetime"][ii])
        temp_data['seriesTime'].append(rq_dict["seriestime"][ii])
        temp_data['triggerVetoReadoutTime'].append(rq_dict["triggervetoreadouttime"][ii])
        temp_data['waveformReadEndTime'].append(rq_dict["waveformreadendtime"][ii])
        temp_data['waveformReadStartTime'].append(rq_dict["waveformreadstarttime"][ii])
        
        if temp_data['readoutStatus'][ii] == 1:
        
            trace1 = trace_full[0]
            trace2 = trace_full[1]
            trace3 = trace_full[2]

            baseline1 = np.mean(np.hstack((trace1[:indbasepre], trace1[indbasepost:])))
            baseline2 = np.mean(np.hstack((trace2[:indbasepre], trace2[indbasepost:])))
            baseline3 = np.mean(np.hstack((trace3[:indbasepre], trace3[indbasepost:])))

            trace_bsSub1 = trace1 - baseline1
            trace_bsSub2 = trace2 - baseline2
            trace_bsSub3 = trace3 - baseline3

            amp1_delay, t01_delay, chi21_delay = ofamp(trace1, template1, psd1, fs, nconstrain=80)
            amp2_delay, t02_delay, chi22_delay = ofamp(trace2, template2, psd2, fs, nconstrain=80)
            amp3_delay, t03_delay, chi23_delay = ofamp(trace3, template3, psd3, fs, nconstrain=80)

            template2_rolled = np.roll(template2, int(t01_delay*fs))
            template3_rolled = np.roll(template3, int(t01_delay*fs))

            amp2_rolled, _, chi22_rolled = ofamp(trace2, template2_rolled, psd2, fs, withdelay=False)
            amp3_rolled, _, chi23_rolled = ofamp(trace3, template3_rolled, psd3, fs, withdelay=False)

            amp1_nodelay, _, chi21_nodelay = ofamp(trace1, template1, psd1, fs, withdelay=False)
            amp2_nodelay, _, chi22_nodelay = ofamp(trace2, template2, psd2, fs, withdelay=False)
            amp3_nodelay, _, chi23_nodelay = ofamp(trace3, template3, psd3, fs, withdelay=False)

            amp1_uncon, t01_uncon, chi21_uncon = ofamp(trace1, template1, psd1, fs)
            amp2_uncon, t02_uncon, chi22_uncon = ofamp(trace2, template2, psd2, fs)
            amp3_uncon, t03_uncon, chi23_uncon = ofamp(trace3, template3, psd3, fs)

            _, _, amp1_pileup, t01_pileup, chi21_pileup = ofamp_pileup(trace1, template1, psd1, fs, 
                                                                       a1=amp1_delay, t1=t01_delay, 
                                                                       nconstrain2=80)
            _, _, amp2_pileup, t02_pileup, chi22_pileup = ofamp_pileup(trace2, template2, psd2, fs, 
                                                                       a1=amp2_uncon, t1=t02_uncon, 
                                                                       nconstrain2=1000)
            _, _, amp3_pileup, t03_pileup, chi23_pileup = ofamp_pileup(trace3, template3, psd3, fs, 
                                                                       a1=amp3_uncon, t1=t03_uncon, 
                                                                       nconstrain2=1000)

            chi2_10000 = chi2lowfreq(trace1, template1, amp1_delay, t01_delay, psd1, fs, fcutoff=10000)

            temp_data[f'baseline_{chan[0]}'].append(baseline1)
            temp_data[f'baseline_{chan[1]}'].append(baseline2)
            temp_data[f'baseline_{chan[2]}'].append(baseline3)

            temp_data[f'int_bsSub_{chan[0]}'].append(np.trapz(trace_bsSub1, time))
            temp_data[f'int_bsSub_{chan[1]}'].append(np.trapz(trace_bsSub2, time))
            temp_data[f'int_bsSub_{chan[2]}'].append(np.trapz(trace_bsSub3, time))

            temp_data[f'ofAmps_nodelay_{chan[0]}'].append(amp1_nodelay)
            temp_data[f'ofAmps_nodelay_{chan[1]}'].append(amp2_nodelay)
            temp_data[f'ofAmps_nodelay_{chan[2]}'].append(amp3_nodelay)

            temp_data[f'chi2_nodelay_{chan[0]}'].append(chi21_nodelay)
            temp_data[f'chi2_nodelay_{chan[1]}'].append(chi22_nodelay)
            temp_data[f'chi2_nodelay_{chan[2]}'].append(chi23_nodelay)

            temp_data[f'ofAmps_delay_{chan[0]}'].append(amp1_delay)
            temp_data[f'ofAmps_delay_{chan[1]}'].append(amp2_delay)
            temp_data[f'ofAmps_delay_{chan[2]}'].append(amp3_delay)

            temp_data[f'chi2_delay_{chan[0]}'].append(chi21_delay)
            temp_data[f'chi2_delay_{chan[1]}'].append(chi22_delay)
            temp_data[f'chi2_delay_{chan[2]}'].append(chi23_delay)

            temp_data[f't0_delay_{chan[0]}'].append(t01_delay)
            temp_data[f't0_delay_{chan[1]}'].append(t02_delay)
            temp_data[f't0_delay_{chan[2]}'].append(t03_delay)

            temp_data[f'ofAmps_uncon_{chan[0]}'].append(amp1_uncon)
            temp_data[f'ofAmps_uncon_{chan[1]}'].append(amp2_uncon)
            temp_data[f'ofAmps_uncon_{chan[2]}'].append(amp3_uncon)

            temp_data[f'chi2_uncon_{chan[0]}'].append(chi21_uncon)
            temp_data[f'chi2_uncon_{chan[1]}'].append(chi22_uncon)
            temp_data[f'chi2_uncon_{chan[2]}'].append(chi23_uncon)

            temp_data[f't0_uncon_{chan[0]}'].append(t01_uncon)
            temp_data[f't0_uncon_{chan[1]}'].append(t02_uncon)
            temp_data[f't0_uncon_{chan[2]}'].append(t03_uncon)

            temp_data[f'ofAmps_pileup_{chan[0]}'].append(amp1_pileup)
            temp_data[f'ofAmps_pileup_{chan[1]}'].append(amp2_pileup)
            temp_data[f'ofAmps_pileup_{chan[2]}'].append(amp3_pileup)

            temp_data[f'chi2_pileup_{chan[0]}'].append(chi21_pileup)
            temp_data[f'chi2_pileup_{chan[1]}'].append(chi22_pileup)
            temp_data[f'chi2_pileup_{chan[2]}'].append(chi23_pileup)

            temp_data[f't0_pileup_{chan[0]}'].append(t01_pileup)
            temp_data[f't0_pileup_{chan[1]}'].append(t02_pileup)
            temp_data[f't0_pileup_{chan[2]}'].append(t03_pileup)

            temp_data['chi2_lowfreq'].append(chi2_10000)
        
        else:
            
            temp_data[f'baseline_{chan[0]}'].append(-999999.0)
            temp_data[f'baseline_{chan[1]}'].append(-999999.0)
            temp_data[f'baseline_{chan[2]}'].append(-999999.0)

            temp_data[f'int_bsSub_{chan[0]}'].append(-999999.0)
            temp_data[f'int_bsSub_{chan[1]}'].append(-999999.0)
            temp_data[f'int_bsSub_{chan[2]}'].append(-999999.0)

            temp_data[f'ofAmps_nodelay_{chan[0]}'].append(-999999.0)
            temp_data[f'ofAmps_nodelay_{chan[1]}'].append(-999999.0)
            temp_data[f'ofAmps_nodelay_{chan[2]}'].append(-999999.0)

            temp_data[f'chi2_nodelay_{chan[0]}'].append(-999999.0)
            temp_data[f'chi2_nodelay_{chan[1]}'].append(-999999.0)
            temp_data[f'chi2_nodelay_{chan[2]}'].append(-999999.0)

            temp_data[f'ofAmps_delay_{chan[0]}'].append(-999999.0)
            temp_data[f'ofAmps_delay_{chan[1]}'].append(-999999.0)
            temp_data[f'ofAmps_delay_{chan[2]}'].append(-999999.0)

            temp_data[f'chi2_delay_{chan[0]}'].append(-999999.0)
            temp_data[f'chi2_delay_{chan[1]}'].append(-999999.0)
            temp_data[f'chi2_delay_{chan[2]}'].append(-999999.0)

            temp_data[f't0_delay_{chan[0]}'].append(-999999.0)
            temp_data[f't0_delay_{chan[1]}'].append(-999999.0)
            temp_data[f't0_delay_{chan[2]}'].append(-999999.0)

            temp_data[f'ofAmps_uncon_{chan[0]}'].append(-999999.0)
            temp_data[f'ofAmps_uncon_{chan[1]}'].append(-999999.0)
            temp_data[f'ofAmps_uncon_{chan[2]}'].append(-999999.0)

            temp_data[f'chi2_uncon_{chan[0]}'].append(-999999.0)
            temp_data[f'chi2_uncon_{chan[1]}'].append(-999999.0)
            temp_data[f'chi2_uncon_{chan[2]}'].append(-999999.0)

            temp_data[f't0_uncon_{chan[0]}'].append(-999999.0)
            temp_data[f't0_uncon_{chan[1]}'].append(-999999.0)
            temp_data[f't0_uncon_{chan[2]}'].append(-999999.0)

            temp_data[f'ofAmps_pileup_{chan[0]}'].append(-999999.0)
            temp_data[f'ofAmps_pileup_{chan[1]}'].append(-999999.0)
            temp_data[f'ofAmps_pileup_{chan[2]}'].append(-999999.0)

            temp_data[f'chi2_pileup_{chan[0]}'].append(-999999.0)
            temp_data[f'chi2_pileup_{chan[1]}'].append(-999999.0)
            temp_data[f'chi2_pileup_{chan[2]}'].append(-999999.0)

            temp_data[f't0_pileup_{chan[0]}'].append(-999999.0)
            temp_data[f't0_pileup_{chan[1]}'].append(-999999.0)
            temp_data[f't0_pileup_{chan[2]}'].append(-999999.0)

            temp_data['chi2_lowfreq'].append(-999999.0)
            
    df_temp = pd.DataFrame.from_dict(temp_data)
    
    if lgcsavedumps:
        df_temp.to_pickle(f'{savepath}rq_df_{seriesnum}_d{dump}.pkl')   

    return df_temp


def process_rq(filelist, chan, det, convtoamps, template, psds, fs, time, indbasepre, 
               indbasepost, savepath, lgcsavedumps, nprocess=1):
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
    convtoamps : array_like
        The conversion factor that each channel should be multiplied by to convert from ADC bins to Amps
    template : array_like
        An array that contains the corresponding pulse templates for each channel
    psds : array_like
        An array that contains the corresponding PSDs for each channel
    fs : float
        The sample rate of the data being taken (in Hz).
    time : ndarray
        The corresponding time values for the template.
    indbasepre : int
        This index defines the number of bins from the beginning of the trace that will be used for
        the baseline calculation, i.e. the beginning of the trace up to this index.
    indbasepost : int
        This index defines the number of bins from the end of the trace that will be used for
        the baseline calculation, i.e. this index up to the end of the trace.
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
    
    path = filelist[0]
    pathgain = path.split('.')[0][:-19]
    
    pool = multiprocessing.Pool(processes = nprocess)
    results = pool.starmap(_process_rq, zip(filelist, repeat([chan, det, convtoamps, template, psds, 
                                            fs, time, indbasepre, indbasepost, savepath, lgcsavedumps])))
    pool.close()
    pool.join()
    rq_df = pd.concat([df for df in results], ignore_index = True)
    
    return rq_df

