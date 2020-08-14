import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from glob import glob
import warnings
import deepdish as dd

from rqpy import HAS_SCDMSPYTOOLS

if HAS_SCDMSPYTOOLS:
    from rawio.IO import getRawEvents, getDetectorSettings


__all__ = [
    "getrandevents",
    "get_trace_gain",
    "get_traces_midgz",
    "get_traces_npz",
    "loadstanfordfile",
    "load_h5_dump",
]


def getrandevents(basepath, evtnums, seriesnums, cut=None, channels=["PDS1"], det="Z1", sumchans=False, 
                  convtoamps=1, fs=625e3, lgcplot=False, ntraces=1, nplot=20, seed=None, indbasepre=None,
                  filetype="mid.gz"):
    """
    Function for loading (and plotting) random events from a datasets. Has functionality to pull
    randomly from a specified cut. For use with `rawio.IO.getRawEvents`

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
    det : str or list of str, optional
        String or list of strings that specifies the detector name. Only used if filetype=='mid.gz'. 
        If a list of strings, then should each value should directly correspond to the channel names.
        If a string is inputted and there are multiple channels, then it is assumed that the detector
        name is the same for each channel.
    sumchans : bool, optional
        A boolean flag for whether or not to sum the channels when plotting. If False, each
        channel is plotted individually.
    convtoamps : float or list of floats, optional
        The factor that the traces should be multiplied by to convert ADC bins to Amperes.
    fs : float, optional
        The sample rate in Hz of the data.
    ntraces : int, str, optional
        The number of traces to randomly load from the data (with the cut, if specified). If all traces
        from a specfied cut are desired, pass the string "all". Default is 1.
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
        types -"mid.gz" and "npz". "mid.gz" is the default.

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
        raise ImportError("Cannot use filetype mid.gz because cdms rawio is not installed.")

    if seed is not None:
        np.random.seed(seed)

    if isinstance(channels, str):
        channels = [channels]

    if not isinstance(evtnums, pd.Series):
        evtnums = pd.Series(data=evtnums)
    if not isinstance(seriesnums, pd.Series):
        seriesnums = pd.Series(data=seriesnums)

    if not isinstance(convtoamps, list):
        convtoamps = [convtoamps]
    convtoamps_arr = np.array(convtoamps)
    convtoamps_arr = convtoamps_arr[np.newaxis,:,np.newaxis]

    if cut is None:
        cut = np.ones(len(evtnums), dtype=bool)

    if np.sum(cut) == 0:
        raise ValueError("The inputted cut has no events, cannot load any traces.")

    if ntraces == 'all' or ntraces > np.sum(cut):
        ntraces = np.sum(cut)
        if ntraces > 1000:
            warnings.warn(f"You are loading a large number of traces ({ntraces}). Be careful with your RAM usage.")

    inds = np.random.choice(np.flatnonzero(cut), size=ntraces, replace=False)

    crand = np.zeros(len(evtnums), dtype=bool)
    crand[inds] = True

    arrs = list()
    for snum in seriesnums[crand].unique():
        cseries = crand & (seriesnums == snum)

        if filetype == "mid.gz":

            if isinstance(det, str):
                det = [det]*len(channels)

            if len(det) != len(channels):
                raise ValueError("channels and det should have the same length")

            if np.isscalar(snum):
                snum_str = f"{int(snum):012}"
                snum_str = snum_str[:8] + '_' + snum_str[8:]
            else:
                snum_str = snum

            dets = [int("".join(filter(str.isdigit, d))) for d in det]

            arr = getRawEvents(
                f"{basepath}{snum_str}/",
                "",
                channelList=channels,
                detectorList=list(set(dets)),
                outputFormat=3,
                eventNumbers=evtnums[cseries].astype(int).tolist(),
            )
        elif filetype == "npz":
            dumpnums = np.asarray(evtnums/10000, dtype=int)

            snum_str = f"{snum:010}"
            snum_str = snum_str[:6] + '_' + snum_str[6:]

            arr = list()

            for dumpnum in set(dumpnums[cseries]):
                cdump = dumpnums == dumpnum
                inds = np.mod(evtnums[cseries & cdump], 10000) - 1

                matching_files = sorted(glob(f"{basepath}/{snum_str}/{snum_str}_*_{dumpnum:04d}.npz"))
                if len(matching_files) > 1:
                    raise IOError(
                        f"There are multiple files with series number {snum_str} "
                        f"and dump number {dumpnum:04d} at this location, making "
                        "it unclear which one to open."
                    )

                with np.load(matching_files[0]) as f:
                    arr.append(f["traces"][inds])

            arr = np.vstack(arr)

        arrs.append(arr)

    if filetype == "mid.gz":
        xs = []
        for arr in arrs:
            if len(set(dets))==1:
                if channels != arr[det[0]]["pChan"]:
                    chans = [arr[det[0]]["pChan"].index(ch) for ch in channels]
                    x = arr[det[0]]["p"][:, chans].astype(float)
                else:
                    x = arr[det[0]]["p"].astype(float)
            else:
                chans = [arr[d]["pChan"].index(ch) for d, ch in zip(det, channels)]
                x = [arr[d]["p"][:, ch].astype(float) for d, ch in zip(det, chans)]
                x = np.stack(x, axis=1)

            xs.append(x)

        x = np.vstack(xs)

    elif filetype == "npz":
        x = np.vstack(arrs).astype(float)
        channels = list(range(x.shape[1]))

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


def get_trace_gain(path, chan, det, gainfactors={'rfb': 5000, 'loopgain' : 2.4, 'adcpervolt' : 2**(16)/2, 'lowpassgain' : 2}):
    """
    Calculates the conversion from ADC bins to TES current for mid.gz files.

    Parameters
    ----------
    path : str, list of str
        Absolute path, or list of paths, to the dump to open.
    chan : str
        Channel name, i.e. 'PDS1'
    det : str
        Detector name, i.e. 'Z1'. 
    gainfactors : dict, optional
        Dictionary containing phonon amp parameters.
        The keys for dictionary are as follows.
            'rfb' : resistance of feedback resistor
            'loopgain' : gain of loop of the feedback amp
            'adcpervolt' : the bitdepth divided by the voltage range of the ADC
            'lowpassgain' : the final gain of the low pass filter
                            (Note, this is a specific value depending on the DCRC
                            version used: RevD = 2, RevE = 4)

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
        raise ImportError("Cannot use get_trace_gain because cdms rawio is not installed.")

    series = path.split('/')[-1]

    if os.path.splitext(path)[-1]:
        path = os.path.dirname(path)

    settings = getDetectorSettings(path, series)
    qetbias = settings[det][chan]['qetBias']
    drivergain = settings[det][chan]['driverGain']
    convtoamps = 1 / (gainfactors['rfb'] * gainfactors['loopgain'] * drivergain * gainfactors['lowpassgain'] * gainfactors['adcpervolt'])

    return convtoamps, drivergain, qetbias

def get_traces_midgz(path, channels, det, convtoamps=None, lgcskip_empty=True, lgcreturndict=False):
    """
    Function to return raw traces and event information for a single channel for mid.gz files.

    Parameters
    ----------
    path : str, list of str
        Absolute path, or list of paths, to the dump to open.
    channels : str, list of str
        Channel name(s), i.e. 'PDS1'. If a list of channels, the outputted traces will be sorted to match the order
        the getRawEvents reports in events[det]['pChan'], which can cause slow downs. It is recommended to match
        this order if opening many or large files.
    det : str, list of str
        Detector name, i.e. 'Z1'. If a list of strings, then should each value should directly correspond to
        the channel names. If a string is inputted and there are multiple channels, then it
        is assumed that the detector name is the same for each channel.
    convtoamps : float, list of floats, Nonetype, optional
        Conversion factor from ADC bins to TES current in Amps (units are [Amps]/[ADC bins]). If units of ADC bins
        are desired, convtoamps should be set to 1. Default is None, which will call get_trace_gain() to get the
        conversion to amps
    lgcskip_empty : bool, optional
        Boolean flag on whether or not to skip empty events. Should be set to True if user only wants the traces.
        If the user also wants to pull extra timing information (primarily for live time calculations), then set
        to False. Default is True.
    lgcreturndict : bool, optional
        Boolean flag on whether or not to return the info_dict that has extra information on every event.
        By default, this is False

    Returns
    -------
    x : ndarray
        Array of traces in the specified dump. Dimensions are (number of traces, number of channels, bins in each trace)
    info_dict : dict, optional
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
        raise ImportError("Cannot use get_traces_midgz because cdms rawio is not installed.")

    if not isinstance(path, list):
        path = [path]

    if not isinstance(channels, list):
        channels = [channels]

    if isinstance(det, str):
        det = [det]*len(channels)

    if len(det) != len(channels):
        raise ValueError("channels and det should have the same length")

    if convtoamps is None:
        convtoamps = []
        for ii in range(len(channels)):
            conv, _, _ = get_trace_gain(path=path[0], chan=channels[ii], det=det[ii])
            convtoamps.append(conv)

    if not isinstance(convtoamps, list):
        convtoamps = [convtoamps]
    convtoamps_arr = np.array(convtoamps)
    convtoamps_arr = convtoamps_arr[np.newaxis,:,np.newaxis]

    dets = [int("".join(filter(str.isdigit, d))) for d in det]

    events = getRawEvents(
        filepath='',
        files_series=path,
        channelList=channels,
        detectorList=dets,
        skipEmptyEvents=lgcskip_empty,
        outputFormat=3,
    )

    if len(set(dets))==1:
        if channels != events[det[0]]["pChan"]:
            chans = [events[det[0]]["pChan"].index(ch) for ch in channels]
            x = events[det[0]]["p"][:, chans].astype(float)
        else:
            x = events[det[0]]["p"].astype(float)
    else:
        chans = [events[d]["pChan"].index(ch) for d, ch in zip(det, channels)]
        x = [events[d]["p"][:, ch].astype(float) for d, ch in zip(det, chans)]
        x = np.stack(x, axis=1)

    x*=convtoamps_arr

    if lgcreturndict:

        columns = [
            "eventnumber",
            "seriesnumber",
            "eventtime",
            "triggertype",
            "pollingendtime",
            "triggertime",
            "triggeramp",
            "triggermask",
            "triggerdetnum",
        ]

        columns_trigveto = [
            "readoutstatus",
            "deadtime",
            "livetime",
            "triggervetoreadouttime",
            "seriestime",
            "waveformreadendtime",
            "waveformreadstarttime",
        ]

        for item in columns_trigveto:
            for d in set(det):
                columns.append(f"{item}{d}")

        info_dict = {}
        for item in columns:
            info_dict[item] = []

        for ev, trig, trigv in zip(events["event"], events["trigger"], events["trigger_veto"]):
            info_dict["eventnumber"].append(ev["EventNumber"])
            info_dict["seriesnumber"].append(ev["SeriesNumber"])
            info_dict["eventtime"].append(ev["EventTime"])
            info_dict["triggertype"].append(ev["TriggerType"])
            info_dict["triggeramp"].append(trig['TriggerAmplitude'])
            info_dict["pollingendtime"].append(ev["PollingEndTime"])
            info_dict["triggertime"].append(trig["TriggerTime"])
            info_dict["triggermask"].append(trig["TriggerMask"])
            info_dict["triggerdetnum"].append(trig["TriggerDetNum"])

            for d in set(det):
                try:
                    info_dict[f"readoutstatus{d}"].append(trigv[d]["ReadoutStatus"])
                except:
                    info_dict[f"readoutstatus{d}"].append(-999999.0)

                try:
                    info_dict[f"deadtime{d}"].append(trigv[d]["DeadTime0"])
                except:
                    info_dict[f"deadtime{d}"].append(-999999.0)

                try:
                    info_dict[f"livetime{d}"].append(trigv[d]["LiveTime0"])
                except:
                    info_dict[f"livetime{d}"].append(-999999.0)

                try:
                    info_dict[f"triggervetoreadouttime{d}"].append(trigv[d]["TriggerVetoReadoutTime0"])
                except:
                    info_dict[f"triggervetoreadouttime{d}"].append(-999999.0)

                try:
                    info_dict[f"seriestime{d}"].append(trigv[d]["SeriesTime"])
                except:
                    info_dict[f"seriestime{d}"].append(-999999.0)

                try:
                    info_dict[f"waveformreadendtime{d}"].append(trigv[d]["WaveformReadEndTime"])
                except:
                    info_dict[f"waveformreadendtime{d}"].append(-999999.0)

                try:
                    info_dict[f"waveformreadstarttime{d}"].append(trigv[d]["WaveformReadStartTime"])
                except:
                    info_dict[f"waveformreadstarttime{d}"].append(-999999.0)

        return x, info_dict
    else:
        return x


def get_traces_npz(path):
    """
    Function to return raw traces and event information for a single channel for `npz` files.

    Parameters
    ----------
    path : str, list of str
        Absolute path, or list of paths, to the dump to open.

    Returns
    -------
    traces : ndarray
        Array of traces in the specified dump. Dimensions are (number of traces, number of channels, bins in each trace)
    info_dict : dict
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
    truthamps = []
    truthtdelay = []

    for file in path:
        filename = file.split('/')[-1].split('.')[0]
        seriesnum = int(str().join(filename.split('_')[:2]))
        dumpnum = int(filename.split('_')[-1])
    
        with np.load(file) as data:
            has_sim_data = "truthamps" in data.keys() and "truthtdelay" in data.keys()

            trigtimes.append(data["trigtimes"])
            trigamps.append(data["trigamps"])
            pulsetimes.append(data["pulsetimes"])
            pulseamps.append(data["pulseamps"])
            randomstimes.append(data["randomstimes"])
            trigtypes.append(data["trigtypes"])
            traces.append(data["traces"])
            nevts = len(data["traces"])

            if has_sim_data:
                truthamps.append(data["truthamps"])
                truthtdelay.append(data["truthtdelay"])

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

    if has_sim_data:
        truthamps = np.vstack(truthamps)
        truthtdelay = np.vstack(truthtdelay)

        for ii in range(truthamps.shape[-1]):
            info_dict[f"truthamps{ii+1}"] = truthamps[:, ii]
            info_dict[f"truthtdelay{ii+1}"] = truthtdelay[:, ii]

    return traces, info_dict

def load_h5_dump(path, lgcskip_empty=True, lgcreturndict=False):
    """
    Function to load HDF5 dumps

    Parameters
    ----------
    path : str
        Absolute path to dump of traces
    lgcskip_empty : bool, optional
        Boolean flag on whether or not to skip empty events. Should be set to True if user only wants the traces.
        If the user also wants to pull extra timing information (primarily for live time calculations), then set
        to False. Default is True.
    lgcreturndict : bool, optional
        Boolean flag on whether or not to return the info_dict that has extra information on every event.
        By default, this is False

    Returns
    -------
    traces : ndarray
        Array of traces in the specified dump. Dimensions are (number of traces, number of channels, bins in each trace)
    info_dict : dict, optional
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

    info_dict = dd.io.load(path)
    traces = info_dict.pop('traces')
    if lgcskip_empty:
        cchans = ~np.all(traces[:,:,:] == 0, axis=-1)
        cut = np.all(cchans, axis = -1)
        traces = traces[cut]
        for key in info_dict:
            info_dict[key] = info_dict[key][cut]
    if lgcreturndict:
        return traces, info_dict
    return traces


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

    res = loadmat(filename, squeeze_me=False)
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
            exp_prop[line] = np.array(val, dtype='f')

    gains = np.array(prop['SRS'][0][0][0], dtype='f')
    rfbs = np.array(prop['Rfb'][0][0][0], dtype='f')
    turns = np.array(prop['turn_ratio'][0][0][0], dtype='f')
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
