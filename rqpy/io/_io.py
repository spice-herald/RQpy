import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from rqpy import HAS_SCDMSPYTOOLS

if HAS_SCDMSPYTOOLS:
    from scdmsPyTools.BatTools.IO import getRawEvents, getDetectorSettings


__all__ = ["getrandevents", "get_trace_gain", "get_traces_midgz", "get_traces_npz", "loadstanfordfile"]


def getrandevents(basepath, evtnums, seriesnums, cut=None, channels=["PDS1"], det="Z1", sumchans=False, 
                  convtoamps=1, fs=625e3, lgcplot=False, ntraces=1, nplot=20, seed=None, indbasepre=None,
                  filetype="mid.gz"):
    """
    Function for loading (and plotting) random events from a datasets. Has functionality to pull 
    randomly from a specified cut. For use with scdmsPyTools.BatTools.IO.getRawEvents
    
    Parameters
    ----------
    basepath : str
        The base path to the directory that contains the folders that the event dumps 
        are in. The folders in this directory should be the series numbers.
    evtnums : array_like
        An array of all event numbers for the events in all datasets.
    seriesnums : array_like
        An array of the corresponding series numbers for each event number in evtnums.
    cut : array_like, optional
        A boolean array of the cut that should be applied to the data. If left as None,
        then no cut is applied.
    channels : list, optional
        A list of strings that contains all of the channels that should be loaded.
    det : str
        String that specifies the detector name. Only used if filetype=='mid.gz'. Default
        is 'Z1'.
    sumchans : bool, optional
        A boolean flag for whether or not to sum the channels when plotting. If False, each 
        channel is plotted individually.
    convtoamps : float or list of floats, optional
        The factor that the traces should be multiplied by to convert ADC bins to Amperes.
    fs : float, optional
        The sample rate in Hz of the data.
    ntraces : int, optional
        The number of traces to randomly load from the data (with the cut, if specified)
    lgcplot : bool, optional
        Logical flag on whether or not to plot the pulled traces.
    nplot : int, optional
        If lgcplot is True, the number of traces to plot.
    seed : int, optional
        A value to pass to np.random.seed if the user wishes to use the same random seed
        each time getrandevents is called.
    indbasepre : NoneType, int, optional
        The number of indices up to which a trace should be averaged to determine the baseline.
        This baseline will then be subtracted from the traces when plotting. If left as None, no
        baseline subtraction will be done.
    filetype : str, optional
        The string that corresponds to the file type that will be opened. Supports two 
        types -"mid.gz" and ".npz". "mid.gz" is the default.
        
    Returns
    -------
    t : ndarray
        The time values for plotting the events.
    x : ndarray
        Array containing all of the events that were pulled.
    crand : ndarray
        Boolean array that contains the cut on the loaded data.
    
    """
    
    if filetype == "mid.gz" and not HAS_SCDMSPYTOOLS:
        raise ImportError("Cannot use filetype mid.gz because scdmsPyTools is not installed.")
    
    if seed is not None:
        np.random.seed(seed)
    
    if type(evtnums) is not pd.core.series.Series:
        evtnums = pd.Series(data=evtnums)
    if type(seriesnums) is not pd.core.series.Series:
        seriesnums = pd.Series(data=seriesnums)
        
    if not isinstance(convtoamps, list):
        convtoamps = [convtoamps]
    convtoamps_arr = np.array(convtoamps)
    convtoamps_arr = convtoamps_arr[np.newaxis,:,np.newaxis]
        
    if cut is None:
        cut = np.ones(len(evtnums), dtype=bool)
        
    if np.sum(cut) == 0:
        raise ValueError("The inputted cut has no events, cannot load any traces.")
        
    if ntraces > np.sum(cut):
        ntraces = np.sum(cut)
        
    inds = np.random.choice(np.flatnonzero(cut), size=ntraces, replace=False)
        
    crand = np.zeros(len(evtnums), dtype=bool)
    crand[inds] = True
    
    arrs = list()
    for snum in seriesnums[crand].unique():
        cseries = crand & (seriesnums == snum)
        
        if filetype == "mid.gz":
            if np.issubdtype(type(snum), np.integer):
                snum_str = f"{snum:012}"
                snum_str = snum_str[:8] + '_' + snum_str[8:]
            else:
                snum_str = snum

            arr = getRawEvents(f"{basepath}{snum_str}/", "", channelList=channels, outputFormat=3, 
                               eventNumbers=evtnums[cseries].astype(int).tolist())
        elif filetype == "npz":
            inds = np.mod(evtnums[cseries], 10000) - 1
            with np.load(f"{basepath}/{snum}.npz") as f:
                arr = f["traces"][inds]
    
        arrs.append(arr)
        
    if filetype == "mid.gz":
        if channels != arr[det]["pChan"]:
            chans = [arr[det]["pChan"].index(val) for val in channels]
            x = arr[det]["p"][:, chans].astype(float)
        else:
            x = arr[det]["p"].astype(float)
        
    elif filetype == "npz":
        x = np.vstack(arrs).astype(float)
        chans = list(range(x.shape[1]))
        
    t = np.arange(x.shape[-1])/fs
    
    x*=convtoamps_arr
    
    if lgcplot:
        if nplot>ntraces:
            nplot = ntraces
    
        for ii in range(nplot):
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if sumchans:
                trace_sum = x[ii].sum(axis=0)
                
                if indbasepre is not None:
                    baseline = np.mean(trace_sum[..., :indbasepre])
                else:
                    baseline = 0
                
                ax.plot(t * 1e6, trace_sum * 1e6, label="Summed Channels")
            else:
                colors = plt.cm.viridis(np.linspace(0, 1, num=x.shape[1]), alpha=0.5)
                for jj, chan in enumerate(channels):
                    if filetype == "mid.gz":
                        label = f"Channel {chan}"
                    elif filetype == "npz":
                        label = f"Channel {chan}"
                    
                    if indbasepre is not None:
                        baseline = np.mean(x[ii, jj, :indbasepre])
                    else:
                        baseline = 0
                        
                    ax.plot(t * 1e6, x[ii, jj] * 1e6 - baseline * 1e6, color=colors[jj], label=label)
            ax.grid()
            ax.set_ylabel("Current [μA]")
            ax.set_xlabel("Time [μs]")
            ax.set_title(f"Pulses, Evt Num {evtnums[crand].iloc[ii]}, Series Num {seriesnums[crand].iloc[ii]}");
            ax.legend()
    
    return t, x, crand


def get_trace_gain(path, chan, det, gainfactors = {'rfb': 5000, 'loopgain' : 2.4, 'adcpervolt' : 2**(16)/2}):
    """
    Calculates the conversion from ADC bins to TES current for mid.gz files.
    
    Parameters
    ----------
    path : str, list of str
        Absolute path, or list of paths, to the dump to open.
    chan : str
        Channel name, i.e. 'PDS1'
    det : str
        Detector name, i.e. 'Z1'
    gainfactors : dict, optional
        Dictionary containing phonon amp parameters.
        The keys for dictionary are as follows.
            'rfb' : resistance of feedback resistor
            'loopgain' : gain of loop of the feedback amp
            'adcpervolt' : the bitdepth divided by the voltage range of the ADC
    
    Returns
    -------
    convtoamps : float
        Conversion factor from ADC bins to TES current in Amps (units are [Amps]/[ADC bins])
    drivergain : float
        Gain setting of the driver amplifier
    qetbias : float
        The current bias of the QET in Amps.
        
    """
    
    if not HAS_SCDMSPYTOOLS:
        raise ImportError("Cannot use get_trace_gain because scdmsPyTools is not installed.")
    
    series = path.split('/')[-1]
    settings = getDetectorSettings(path, series)
    qetbias = settings[det][chan]['qetBias']
    drivergain = settings[det][chan]['driverGain']*2
    convtoamps = 1/(gainfactors['rfb'] * gainfactors['loopgain'] * drivergain * gainfactors['adcpervolt'])
    
    return convtoamps, drivergain, qetbias

def get_traces_midgz(path, chan, det, convtoamps = 1, lgcskip_empty = False):
    """
    Function to return raw traces and event information for a single channel for mid.gz files.
    
    Parameters
    ----------
    path : str, list of str
        Absolute path, or list of paths, to the dump to open.
    chan : str, list of str
        Channel name(s), i.e. 'PDS1'. If a list of channels, the outputted traces will be sorted to match the order
        the getRawEvents reports in events[det]['pChan'], which can cause slow downs. It is recommended to match
        this order if opening many or large files.
    det : str
        Detector name, i.e. 'Z1'
    convtoamps : float, list of floats, optional
        Conversion factor from ADC bins to TES current in Amps (units are [Amps]/[ADC bins]). Default is to 
        keep in units of ADC bins (i.e. the traces are left in units of ADC bins)
    lgcskip_empty : bool, optional
        Boolean flag on whether or not to skip empty events. Should be set to false if user only wants the traces.
        If the user also wants to pull extra timing information (primarily for live time calculations), then set
        to True. Default is True.
    
    Returns
    -------
    traces : ndarray
        Array of traces in the specified dump. Dimensions are (number of traces, number of channels, bins in each trace)
    rq_dict : dict
        Dictionary that contains extra information on each event. Includes timing and trigger information.
        The keys in the dictionary are as follows.
            'eventnumber' : The event number for each event
            'seriesnumber' : The corresponding series number for each event
            'eventtime' : The time of the event (in s). Only has resolution up seconds place.
            'triggertype' : The type of the trigger (e.g. random, pulse trigger, no trigger)
            'triggeramp' : The amplitude of the trigger. Only useful if triggertype = 1 (pulse trigger)
            'pollingendtime' : The end time for events being polled by the DAQ for reading.
            'triggertime' : The time of the trigger, this has better resolution than eventtime.
            'readoutstatus' : The status of the readout, discerns between good/stale/no trigger
            'deadtime' : The accrued DAQ dead time.
            'livetime' : The accrued DAQ live time.
            'triggervetoreadouttime' : The time of the trigger veto information readout.
            'seriestime' : Identical to the eventtime.
            'waveformreadendtime' : The time that a waveform readout completed.
            'waveformreadstarttime' : The time that a waveform readout began.
    
    """
    
    if not HAS_SCDMSPYTOOLS:
        raise ImportError("Cannot use get_traces_midgz because scdmsPyTools is not installed.")
    
    if not isinstance(path, list):
        path = [path]
    if not isinstance(chan, list):
        chan = [chan]

    if not isinstance(convtoamps, list):
        convtoamps = [convtoamps]
    convtoamps_arr = np.array(convtoamps)
    convtoamps_arr = convtoamps_arr[np.newaxis,:,np.newaxis]
    
    events = getRawEvents(filepath='',files_series = path, channelList=chan, 
                          detectorList=[int(''.join(x for x in det if x.isdigit()))],
                          skipEmptyEvents=lgcskip_empty, outputFormat=3)
    
    columns = ["eventnumber", "seriesnumber", "eventtime", "triggertype", "readoutstatus", "pollingendtime", 
               "triggertime", "deadtime", "livetime", "seriestime", "triggervetoreadouttime",
               "waveformreadendtime", "waveformreadstarttime", "triggeramp"]

    rq_dict = {}
    for item in columns:
        rq_dict[item] = []
    
    for ev, trig, trigv in zip(events["event"], events["trigger"], events["trigger_veto"]):
        rq_dict["eventnumber"].append(ev["EventNumber"])
        rq_dict["seriesnumber"].append(ev["SeriesNumber"])
        rq_dict["eventtime"].append(ev["EventTime"])
        rq_dict["triggertype"].append(ev["TriggerType"])
        rq_dict["triggeramp"].append(trig['TriggerAmplitude'])
        rq_dict["pollingendtime"].append(ev["PollingEndTime"])

        rq_dict["triggertime"].append(trig["TriggerTime"])

        try:
            rq_dict["readoutstatus"].append(trigv[det]["ReadoutStatus"])
        except:
            rq_dict["readoutstatus"].append(-999999.0)

        try:
            rq_dict["deadtime"].append(trigv[det]["DeadTime0"])
        except:
            rq_dict["deadtime"].append(-999999.0)

        try:
            rq_dict["livetime"].append(trigv[det]["LiveTime0"])
        except:
            rq_dict["livetime"].append(-999999.0)

        try:
            rq_dict["triggervetoreadouttime"].append(trigv[det]["TriggerVetoReadoutTime0"])
        except:
            rq_dict["triggervetoreadouttime"].append(-999999.0)

        try:
            rq_dict["seriestime"].append(trigv[det]["SeriesTime"])
        except:
            rq_dict["seriestime"].append(-999999.0)

        try:
            rq_dict["waveformreadendtime"].append(trigv[det]["WaveformReadEndTime"])
        except:
            rq_dict["waveformreadendtime"].append(-999999.0)

        try:
            rq_dict["waveformreadstarttime"].append(trigv[det]["WaveformReadStartTime"])
        except:
            rq_dict["waveformreadstarttime"].append(-999999.0)
            
    if chan != events[det]["pChan"]:
        inds = [events[det]["pChan"].index(val) for val in chan]
        traces = events[det]["p"][:, inds]*convtoamps_arr
    else:
        traces = events[det]["p"]*convtoamps_arr
    
    
    return traces, rq_dict


def get_traces_npz(path):
    """
    Function to return raw traces and event information for a single channel for mid.gz files.
    
    Parameters
    ----------
    path : str, list of str
        Absolute path, or list of paths, to the dump to open.
    chan : str
        Channel name, i.e. 'PDS1'
    det : str
        Detector name, i.e. 'Z1'
    convtoamps : float, list of floats, optional
        Conversion factor from ADC bins to TES current in Amps (units are [Amps]/[ADC bins]). Default is to 
        keep in units of ADC bins (i.e. the traces are left in units of ADC bins)
    lgcskip_empty : bool, optional
        Boolean flag on whether or not to skip empty events. Should be set to false if user only wants the traces.
        If the user also wants to pull extra timing information (primarily for live time calculations), then set
        to True. Default is True.
    
    Returns
    -------
    traces : ndarray
        Array of traces in the specified dump. Dimensions are (number of traces, number of channels, bins in each trace)
    rq_dict : dict
        Dictionary that contains extra information on each event. Includes timing and trigger information.
        The keys in the dictionary are as follows.
            'eventnumber' : The event number for each event
            'seriesnumber' : The corresponding series number for each event
            'ttltimes' : If we triggered due to ttl, the time of the ttl trigger in seconds. Otherwise this is zero.
            'ttlamps' : If we triggered due to ttl, the optimum amplitude at the ttl trigger time. Otherwise this is zero.
            'pulsetimes' : If we triggered on a pulse, the time of the pulse trigger in seconds. Otherwise this is zero.
            'pulseamps' : If we triggered on a pulse, the optimum amplitude at the pulse trigger time. Otherwise this is zero.
            'randomstimes' : Array of the corresponding event times for each section
            'randomstrigger' : If we triggered due to randoms, this is True. Otherwise, False.
            'pulsestrigger' : If we triggered on a pulse, this is True. Otherwise, False.
            'ttltrigger' : If we triggered due to ttl, this is True. Otherwise, False.
    
    """
    
    if not isinstance(path, list):
        path = [path]
    
    columns = ["eventnumber", "seriesnumber", "ttltimes", "ttlamps", "pulsetimes", "pulseamps", 
               "randomstimes", "randomstrigger", "pulsestrigger", "ttltrigger"]
    
    info_dict = {}
    
    traces = []
    eventnumber = []
    seriesnumber = []
    trigtimes = []
    trigamps = []
    pulsetimes = []
    pulseamps = []
    randomstimes = []
    trigtypes = []
    
    for file in path:
        seriesnum = file.split('/')[-1].split('.')[0]
        dumpnum = int(seriesnum.split('_')[-1])
        
        with np.load(file) as data:
            trigtimes.append(data["trigtimes"])
            trigamps.append(data["trigamps"])
            pulsetimes.append(data["pulsetimes"])
            pulseamps.append(data["pulseamps"])
            randomstimes.append(data["randomstimes"])
            trigtypes.append(data["trigtypes"])
            traces.append(data["traces"])
            nevts = len(data["traces"])
        
        eventnumber.append(10000*dumpnum + 1 + np.arange(nevts))
        seriesnumber.extend([seriesnum] * nevts)

    info_dict["eventnumber"] = np.concatenate(eventnumber)
    info_dict["ttltimes"] = np.concatenate(trigtimes)
    info_dict["ttlamps"] = np.concatenate(trigamps)
    info_dict["pulsetimes"] = np.concatenate(pulsetimes)
    info_dict["pulseamps"] = np.concatenate(pulseamps)
    info_dict["randomstimes"] = np.concatenate(randomstimes)
    
    info_dict["seriesnumber"] = seriesnumber
    trigtypes = np.vstack(trigtypes)
    info_dict["randomstrigger"] = trigtypes[:, 0]
    info_dict["pulsestrigger"] = trigtypes[:, 1]
    info_dict["ttltrigger"] = trigtypes[:, 2]
    
    traces = np.vstack(traces)
        
    return traces, info_dict


def loadstanfordfile(f, convtoamps=1/1024, lgcfullrtn=False):
    """
    Function that opens a Stanford .mat file and extracts the useful parameters. 
    There is an option to return a dictionary that includes all of the data.
    
    Parameters
    ----------
    f : list, str
        A list of filenames that should be opened (or just one filename). These
        files should be Stanford DAQ .mat files.
    convtoamps : float, optional
        Correction factor to convert the data to Amps. The traces are multiplied by this
        factor, as is the TTL channel (if it exists). Default is 1/1024.
    lgcfullrtn : bool, optional
        Boolean flag that also returns a dict of all extracted data from the file(s).
        Set to False by default.
            
    Returns
    -------
    traces : ndarray
        An array of shape (# of traces, # of channels, # of bins) that contains
        the traces extracted from the .mat file.
    times : ndarray
        An array of shape (# of traces,) that contains the starting time (in s) for 
        each trace in the traces array. The zero point of the times is arbitrary. 
    fs : float
        The digitization rate (in Hz) of the data.
    ttl : ndarray, None
        The TTL channel data, if it exists in the inputted data. This is set to None
        if there is no TTL data.
    data : dict, optional
        The dictionary of all of the data in the data file(s). Only returned if 
        lgcfullrtn is set to True.
    
    """
    
    data = _getchannels(f)
    fs = data["prop"]["sample_rate"][0][0][0][0]
    times = data["time"]
    traces = np.stack((data["A"], data["B"]), axis=1)*convtoamps
    try:
        ttl = data["T"]*convtoamps
    except:
        ttl = None
        
    if lgcfullrtn:
        return traces, times, fs, ttl, data
    else:
        return traces, times, fs, ttl
        
def _getchannels_singlefile(filename):
    """
    Function for opening a .mat file from the Stanford DAQ and returns a dictionary
    that contains the data.
    
    Parameters
    ----------
    filename : str
        The filename that will be opened. Should be a Stanford DAQ .mat file.
            
    Returns
    -------
    res : dict
        A dictionary that has all of the needed data taken from a Stanford DAQ 
        .mat file. 
    
    """
    
    res = loadmat(filename, squeeze_me = False)
    prop = res['exp_prop']
    data = res['data_post']

    exp_prop = dict()
    for line in prop.dtype.names:
        try:
            val = prop[line][0][0][0]
        except IndexError:
            val = 'Nothing'
        if type(val) is str:
            exp_prop[line] = val
        elif val.size == 1:
            exp_prop[line] = val[0]
        else:
            exp_prop[line] = np.array(val, dtype = 'f')

    gains = np.array(prop['SRS'][0][0][0], dtype = 'f')
    rfbs = np.array(prop['Rfb'][0][0][0], dtype = 'f')
    turns = np.array(prop['turn_ratio'][0][0][0], dtype = 'f')
    fs = float(prop['sample_rate'][0][0][0])
    minnum = min(len(gains), len(rfbs), len(turns))
    
    ch1 = data[:,:,0]
    ch2 = data[:,:,1]
    try:
        trig = data[:,:,2]
    except IndexError:
        trig = np.array([])
    ai0 = ch1[:]
    ai1 = ch2[:]
    ai2 = trig[:]
    try:
        ai3 = data[:, :, 3]
    except:
        pass
    
    try:
        ttable  = np.array([24*3600.0, 3600.0, 60.0, 1.0])
        reltime = res['t_rel_trig'].squeeze()
        abstime = res['t_abs_trig'].squeeze()
        timestamp = abstime[:,2:].dot(ttable)+reltime
    except:
        timestamp = np.arange(0,len(ch1))

    dvdi = turns[:minnum]*rfbs[:minnum]*gains[:minnum]
    didv = 1.0/dvdi
    
    res = dict()
    res['A'] = ch1*didv[0]
    res['B'] = ch2*didv[1]
    res['Total'] = res['A']+res['B']
    res['T'] = trig
    res['dVdI'] = dvdi
    res['Fs'] = fs
    res['prop'] = prop
    res['filenum'] = 1
    res['time'] = timestamp
    res['exp_prop'] = exp_prop
    res['ai0'] = ai0
    res['ai1'] = ai1
    res['ai2'] = ai2
    try:
        res['ai3'] = ai3
    except:
        pass
    return res

def _getchannels(filelist):
    """
    Function for opening multiple .mat files from the Stanford DAQ and returns a dictionary
    that contains the data.
    
    Parameters
    ----------
    filelist : list, str
        The list of files that will be opened. Should be Stanford DAQ .mat files.
            
    Returns
    -------
    combined : dict
        A dictionary that has all of the needed data taken from all of the 
        inputted Stanford DAQ .mat files. 
    
    """
    
    if(type(filelist) == str):
        return _getchannels_singlefile(filelist)
    else:
        res1=_getchannels_singlefile(filelist[0])
        combined=dict()
        combined['A']=[res1['A']]
        combined['B']=[res1['B']]
        combined['Total']=[res1['Total']]
        combined['T']=[res1['T']]
        combined['dVdI']=res1['dVdI']
        combined['Fs']=res1['Fs']
        combined['prop']=res1['prop']
        combined['time']=[res1['time']]

        for i in range(1,len(filelist)):
            try:
                res=_getchannels_singlefile(filelist[i])
                combined['A'].append(res['A'])
                combined['B'].append(res['B'])
                combined['Total'].append(res['Total'])
                combined['T'].append(res['T'])
                combined['time'].append(res['time'])
            except:
                pass

        combined['A']=np.concatenate(combined['A'])
        combined['B']=np.concatenate(combined['B'])
        combined['Total']=np.concatenate(combined['Total'])
        combined['T']=np.concatenate(combined['T'])
        combined['time']=np.concatenate(combined['time'])
        
        combined['filenum']=len(filelist)
        
        return combined
