import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scdmsPyTools.BatTools.IO import getRawEvents, getDetectorSettings


__all__ = ["getrandevents", "get_trace_gain", "get_traces_per_dump"]


def getrandevents(basepath, evtnums, seriesnums, cut=None, channels=["PDS1"], convtoamps=1, fs=625e3, 
                  lgcplot=False, ntraces=1, nplot=20, seed=None):
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
        
    Returns
    -------
    t : ndarray
        The time values for plotting the events.
    x : ndarray
        Array containing all of the events that were pulled.
    crand : ndarray
        Boolean array that contains the cut on the loaded data.
    
    """
    
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
        
    if ntraces > np.sum(cut):
        ntraces = np.sum(cut)
        
    inds = np.random.choice(np.flatnonzero(cut), size=ntraces, replace=False)
        
    crand = np.zeros(len(evtnums), dtype=bool)
    crand[inds] = True
    
    arrs = list()
    for snum in seriesnums[crand].unique():
        cseries = crand & (seriesnums == snum)
        arr = getRawEvents(f"{basepath}{snum}/", "", channelList=channels, outputFormat=3, 
                           eventNumbers=evtnums[cseries].astype(int).tolist())
        arrs.append(arr)
        
    chans = list()
    for chan in channels:
        chans.append(arr["Z1"]["pChan"].index(chan))
    chans = sorted(chans)

    x = np.vstack([a["Z1"]["p"][:, chans] for a in arrs]).astype(float)
    t = np.arange(x.shape[-1])/fs
    
    x*=convtoamps_arr
    
    if lgcplot:
        
        if nplot>ntraces:
            nplot = ntraces
    
        colors = plt.cm.viridis(np.linspace(0, 1, num=len(chans)), alpha=0.5)

        for ii in range(nplot):
            
            fig, ax = plt.subplots(figsize=(10, 6))
            for jj, chan in enumerate(chans):
                ax.plot(t * 1e6, x[ii, chan] * 1e6, color=colors[jj], label=f"Channel {arr['Z1']['pChan'][chan]}")
            ax.grid()
            ax.set_ylabel("Current [μA]")
            ax.set_xlabel("Time [μs]")
            ax.set_title(f"Pulses, Evt Num {evtnums[crand].iloc[ii]}, Series Num {seriesnums[crand].iloc[ii]}");
            ax.legend()
    
    return t, x, crand


def get_trace_gain(path, chan, det, gainfactors = {'rfb': 5000, 'loopgain' : 2.4, 'adcpervolt' : 2**(16)/2}):
    """
    Calculates the conversion from ADC bins to TES current.
    
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
    
    series = path.split('/')[-1]
    settings = getDetectorSettings(path, series)
    qetbias = settings[det][chan]['qetBias']
    drivergain = settings[det][chan]['driverGain']*2
    convtoamps = 1/(gainfactors['rfb'] * gainfactors['loopgain'] * drivergain * gainfactors['adcpervolt'])
    
    return convtoamps, drivergain, qetbias

def get_traces_per_dump(path, chan, det, convtoamps = 1, lgcskip_empty = False):
    """
    Function to return raw traces and event information for a single channel.
    
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
    
    if not isinstance(path, list):
        path = [path]
    if not isinstance(chan, list):
        chan = [chan]

    if not isinstance(convtoamps, list):
        convtoamps = [convtoamps]
    convtoamps_arr = np.array(convtoamps)
    convtoamps_arr = convtoamps_arr[np.newaxis,:,np.newaxis]
    
    events = getRawEvents(filepath='',files_series = path, channelList=chan, skipEmptyEvents=lgcskip_empty, outputFormat=3)
    
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
    
    traces = events[det]['p']*convtoamps_arr
    
    return traces, rq_dict

