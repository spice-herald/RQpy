import numpy as np
import pandas as pd
import os
from glob import glob
from math import log10, floor

import rqpy as rp
from rqpy import io
from rqpy import HAS_SCDMSPYTOOLS

if HAS_SCDMSPYTOOLS:
    from scdmsPyTools.BatTools.IO import getDetectorSettings


__all__ = ["buildfakepulses"]


def buildfakepulses(rq, cut, templates, amplitudes, tdelay, basepath,
                    channels="PDS1", det="Z1", relcal=None, convtoamps=1,
                    fs=625e3, neventsperdump=1000, filetype="mid.gz",
                    lgcsavefile=False, savefilepath=None, savefilename=None):
    """
    Function for building fake pulses by adding a template, scaled to certain amplitudes and
    certain time delays, to an existing trace (typically a random).
    
    This function calls `_buildfakepulses_seg` which does the heavy lifting
              
    Parameters
    ----------
    rq : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for the dataset specified.
    cut : array_like
        A boolean array for the cut that selects the traces that will be loaded from the dump
        files. These traces serve as the underlying data to which a template is added.
    templates : ndarray, list of ndarray
        The template(s) to be added to the traces, assumed to be normalized to a max height
        of 1. The template start time should be centered on the center bin. If a list of templates,
        then each template will be added to the traces in succession, using the corresponding
        `amplitudes` and `tdelay`.
    amplitudes : ndarray, list of ndarray
        The amplitudes, in Amps, by which to scale the template to add to the traces. Must be 
        same length as cut. A list of ndarray can be passed, where each ndarray corresponds to
        the amplitudes of the corresponding template in the list of templates.
    tdelay : ndarray, list of ndarray
        The time delay offset, in seconds, by which to shift the template to add to the traces.
        Bin interpolation is implemented for values that are not a multiple the reciprocal of
        the digitization rate. A list of ndarray can be passed, where each ndarray corresponds to
        the tdelays of the corresponding template in the list of templates.
    basepath : str
        The base path to the directory that contains the folders that the event dumps 
        are in. The folders in this directory should be the series numbers.
    channels : str, list of str, optional
        A list of strings that contains all of the channels that should be loaded. Only used if
        filetype=='mid.gz'.
    det : str, list of str, optional
        String or list of strings that specifies the detector name. Only used if filetype=='mid.gz'. 
        If a list of strings, then should each value should directly correspond to the channel names.
        If a string is inputted and there are multiple channels, then it is assumed that the detector
        name is the same for each channel.
    relcal : ndarray, optional
        An array with the amplitude scalings between channels used when making the total 
        If channels is supplied, relcal indices correspond to that list
        Default is all 1 (no relative scaling).
    convtoamps : float, optional
        The factor that the traces should be multiplied by to convert ADC bins to Amperes.
    fs : float, optional
        The sample rate in Hz of the data.
    neventsperdump : int, optional
        The number of events to be saved per dump file.
    filetype : str, optional
        The string that corresponds to the file type that will be opened. Supports two 
        types: "mid.gz" and "npz". "mid.gz" is the default.
    lgcsavefile : bool, optional
        A boolean flag for whether or not to save the fake data to a file.
    savefilepath : str, optional
        The string that corresponds to the file path that will be saved.
    savefilename : str, optional
        The string that corresponds to the file name that will be adjoined to dumpnum and saved.
        
    Returns
    -------
    None
    
    """
    
    if filetype == "mid.gz" and not HAS_SCDMSPYTOOLS:
        raise ImportError("Cannot use filetype mid.gz because scdmsPyTools is not installed.")
    
    if isinstance(channels, str):
        channels = [channels]
    
    if isinstance(det, str):
        det = [det]*len(channels)

    if isinstance(templates, np.ndarray):
        templates = [templates]
    
    if isinstance(amplitudes, np.ndarray):
        amplitudes = [amplitudes]
        
    if isinstance(tdelay, np.ndarray):
        tdelay = [tdelay]
        
    if not len(tdelay) == len(amplitudes) == len(templates):
        raise ValueError("The lists of tdelay, amplitudes, and templates must have the "
                         "same number of ndarray.")
    
    if len(det)!=len(channels):
        raise ValueError("channels and det should have the same length.")
    
    if len(set(rq.seriesnumber[cut])) > 1:
        raise ValueError("There cannot be multiple series numbers included in the "
                         "inputted cut.")
    
    if lgcsavefile and (savefilename is None or savefilepath is None):
        raise ValueError("In order to save the simulated data, you must specify savefilename "
                         "and savefilepath.")

    ntraces = np.sum(cut)
    cutlen = len(cut)

    last_dump_ind = -(ntraces%neventsperdump) if ntraces%neventsperdump else None

    indices = np.arange(ntraces, dtype=int)
    nonzerocutinds = np.flatnonzero(cut)
    
    split_inds = []
    
    if ntraces//neventsperdump > 0:
        split_inds.extend(np.split(indices[:last_dump_ind], ntraces//neventsperdump))

    if last_dump_ind is not None:
        split_inds.append(indices[last_dump_ind:])

    for ii, c in enumerate(split_inds):
        cut_seg = np.zeros(cutlen, dtype=bool)
        cut_seg[nonzerocutinds[c]] = True

        split_amplitudes = [a[c] for a in amplitudes]
        split_tdelay = [t[c] for t in tdelay]

        _buildfakepulses_seg(rq, cut_seg, templates, split_amplitudes, split_tdelay, 
                             basepath, channels=channels, relcal=relcal,
                             det=det, convtoamps=convtoamps, fs=fs, dumpnum=ii+1,
                             filetype=filetype, lgcsavefile=lgcsavefile,
                             savefilepath=savefilepath, savefilename=savefilename)
    
    
def _buildfakepulses_seg(rq, cut, templates, amplitudes, tdelay, basepath,
                         channels="PDS1", relcal=None, det="Z1", convtoamps=1,
                         fs=625e3, dumpnum=1, filetype="mid.gz", lgcsavefile=False,
                         savefilepath=None, savefilename=None):
    """
    Hidden helper function for building fake pulses.
              
    Parameters
    ----------
    rq : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for the dataset specified.
    cut : array_like
        A boolean array for the cut that selects the traces that will be loaded from the dump
        files. These traces serve as the underlying data to which a template is added.
    templates : ndarray, list of ndarray
        The template(s) to be added to the traces, assumed to be normalized to a max height
        of 1. The template start time should be centered on the center bin. If a list of templates,
        then each template will be added to the traces in succession, using the corresponding
        `amplitudes` and `tdelay`.
    amplitudes : ndarray, list of ndarray
        The amplitudes, in Amps, by which to scale the template to add to the traces. Must be 
        same length as cut. A list of ndarray can be passed, where each ndarray corresponds to
        the amplitudes of the corresponding template in the list of templates.
    tdelay : ndarray, list of ndarray
        The time delay offset, in seconds, by which to shift the template to add to the traces.
        Bin interpolation is implemented for values that are not a multiple the reciprocal of
        the digitization rate. A list of ndarray can be passed, where each ndarray corresponds to
        the tdelays of the corresponding template in the list of templates.
    basepath : str
        The base path to the directory that contains the folders that the event dumps 
        are in. The folders in this directory should be the series numbers.
    channels : str, list of str, optional
        A list of strings that contains all of the channels that should be loaded. Only used if
        filetype=='mid.gz'.
    det : str, list of str, optional
        String or list of strings that specifies the detector name. Only used if
        filetype=='mid.gz'. If a list of strings, then should each value should directly
        correspond to the channel names. If a string is inputted and there are multiple
        channels, then it is assumed that the detector name is the same for each channel.
    relcal : ndarray, optional
        An array with the amplitude scalings between channels used when making the total 
        If channels is supplied, relcal indices correspond to that list
        Default is all 1 (no relative scaling).
    convtoamps : float, optional
        The factor that the traces should be multiplied by to convert ADC bins to Amperes.
    fs : float, optional
        The sample rate in Hz of the data.
    filetype : str, optional
        The string that corresponds to the file type that will be opened. Supports two 
        types: "mid.gz" and "npz". "mid.gz" is the default.
    lgcsavefile : bool, optional
        A boolean flag for whether or not to save the fake data to a file.
    savefilepath : str, optional
        The string that corresponds to the file path that will be saved.
    savefilename : str, optional
        The string that corresponds to the file name that will be adjoined to dumpnum and saved.
        
    Returns
    -------
    None
    
    """
    
    seriesnumber = list(set(rq.seriesnumber[cut]))[0]
    
    ntraces = np.sum(cut)
    t, traces, _ = io.getrandevents(basepath, rq.eventnumber, rq.seriesnumber, cut=cut, 
                                    channels=channels, det=det, convtoamps=convtoamps, fs=fs, 
                                    ntraces=ntraces, filetype=filetype)
    
    nchan = traces.shape[1]
    
    if relcal is None:
        relcal = np.ones(nchan)
    elif nchan != len(relcal):
        raise ValueError('relcal must have length equal to number of channels')
        
    tracessum = np.sum(traces, axis=1)
    fakepulses = np.zeros(traces.shape)
    newtrace = np.zeros(tracessum.shape[-1])
    
    for ii in range(ntraces):
        newtrace[:] = tracessum[ii]
        
        for temp, amp, td in zip(templates, amplitudes, tdelay):
            newtrace += amp[ii] * rp.shift(temp, td[ii] * fs)

        # multiply by reciprocal of the relative calibration such that when the processing script 
        # creates the total channel pulse, it will be equal to newtrace
        for jj in range(nchan):
            if relcal[jj]!=0:
                fakepulses[ii, jj] = newtrace/(relcal[jj]*nchan)

    if lgcsavefile:
        if filetype=='npz':
            trigtypes = np.zeros((ntraces, 3), dtype=bool)
            # save the data. note that we are storing the truth
            # information in some of the inputs intended for use
            # by the continuous trigger code.
            # TODO: save truth information in a better way
            io.saveevents_npz(pulsetimes=tdelay[0],
                              pulseamps=amplitudes[0],
                              trigtimes=tdelay[1] if len(tdelay)>1 else None,
                              trigamps=amplitudes[1] if len(amplitudes)>1 else None,
                              traces=fakepulses,
                              trigtypes=trigtypes,
                              savepath=savefilepath,
                              savename=savefilename,
                              dumpnum=dumpnum)
            
        elif filetype=="mid.gz":
            
            if np.issubdtype(type(seriesnumber), np.integer):
                snum_str = f"{seriesnumber:012}"
                snum_str = snum_str[:8] + '_' + snum_str[8:]
            else:
                snum_str = seriesnumber
                
            full_settings_dict = getDetectorSettings(f"{basepath}{snum_str}", "")
            
            settings_dict = {d: full_settings_dict[d] for d in det}
            
            for ch, d in zip(channels, det):
                settings_dict[d]["detectorType"] = 710
                settings_dict[d]["phononTraceLength"] = int(settings_dict[d][ch]["binsPerTrace"])
                settings_dict[d]["phononPreTriggerLength"] = settings_dict[d]["phononTraceLength"]//2
                settings_dict[d]["phononSampleRate"] = int(1/settings_dict[d][ch]["timePerBin"])
            
            events_list = _create_events_list(tdelay[0], amplitudes[0], fakepulses, channels, 
                                              det, convtoamps, seriesnumber, dumpnum)
            
            io.saveevents_midgz(events=events_list, settings=settings_dict, 
                                savepath=savefilepath, savename=savefilename, dumpnum=dumpnum)
        else:
            raise ValueError('Inputted filetype is not supported.')
    

def _round_sig(x, sig=2):
    """
    Function for rounding a float to the specified number of significant figures.
    
    Parameters
    ----------
    x : float
        Number to round to the specified number of significant figures.
    sig : int
        The number of significant figures to round.
        
    Returns
    -------
    y : float
        `x` rounded to the number of significant figures specified by `sig`.
    
    """
    
    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)
    

def _create_events_list(pulsetimes, pulseamps, traces, channels, det, convtoamps, seriesnumber, dumpnum):
    """
    Function for structuring the events list correctly for use with `rqpy.io.save_events_midgz` when 
    saving `mid.gz` files.
    
    Parameters
    ----------
    pulsetimes : ndarray
        The true values of the time of the inputted pulses in the simulated data, in seconds.
    pulseamps : ndarray
        The true values of the amplitudes of the inputted pulses in the simulated data, in Amps.
    traces : ndarray
        The array of traces after adding the specified pulses, in Amps. Has shape (number of traces,
        number of channels, 
    channels : list of str
        The list of channels that were used when making the simulated data. The order is assumed to correspond
        to the order of channels in `traces`.
    det : list of str
        The corresponding detector IDs for each channel in `channels`.
    convtoamps : float
        The factor that converts from ADC bins to Amps. This is used to convert back to ADC bins.
    seriesnumber : int
        The series number that the data was pulled from before adding the pulses.
    dumpnum : int
        The dump number for this file, used for correctly setting the event number.
    
    Returns
    -------
    events : list
        List of all of the simulated events with the required fields for saving as a MIDAS file.
    
    """
    
    events = list()
    
    pchans = ["PAS1", "PBS1", "PCS1", "PDS1", "PES1", "PFS1", "PAS2", "PBS2", "PCS2", "PDS2", "PES2", "PFS2"]
    
    for ii, (pulsetime, pulseamp, trace) in enumerate(zip(pulsetimes, pulseamps, traces)):

        event_dict = {'SeriesNumber': seriesnumber,
                      'EventNumber' : dumpnum*(10000)+ii,
                      'EventTime'   : 0,
                      'TriggerType' : 1,
                      'SimAvgX'     : 0,
                      'SimAvgY'     : _round_sig(pulseamp, sig=6),
                      'SimAvgZ'     : 0}

        trigger_dict = {'TriggerUnixTime1'  : 0,
                        'TriggerTime1'      : 0, 
                        'TriggerTimeFrac1'  : 0, 
                        'TriggerDetNum1'    : 0, 
                        'TriggerAmplitude1' : 0,
                        'TriggerStatus1'    : 3,
                        'TriggerUnixTime2'  : 0,
                        'TriggerTime2'      : 0,
                        'TriggerTimeFrac2'  : int(pulsetime/100e-9),
                        'TriggerDetNum2'    : 1,
                        'TriggerAmplitude2' : (pulseamp/convtoamps).astype(np.int32),
                        'TriggerStatus2'    : 1,
                        'TriggerUnixTime3'  : 0,
                        'TriggerTime3'      : 0,
                        'TriggerTimeFrac3'  : 0,
                        'TriggerDetNum3'    : 0,
                        'TriggerAmplitude3' : 0,
                        'TriggerStatus3'    : 8}
        
        events_dict = {'event'   : event_dict,
                       'trigger' : trigger_dict}
        
        for d in set(det):
            events_dict[d] = dict()
        
        for jj, (ch, d) in enumerate(zip(channels, det)):
            for pch in pchans:
                if pch==ch:
                    events_dict[d][ch] = (trace[jj]/convtoamps).astype(np.int32)
                else:
                    events_dict[d][pch] = np.zeros(trace.shape[-1], dtype=np.int32)
        
        events.append(events_dict)
    
    return events

