import numpy as np
import pandas as pd
import os
from glob import glob

from rqpy import io
from rqpy import core
from rqpy import process
from rqpy import HAS_SCDMSPYTOOLS



__all__ = ["buildfakepulses"]


def buildfakepulses(rq, cut, template1, amplitudes1, tdelay1, basepath, evtnums, seriesnums,
                    template2=None, amplitudes2=None, tdelay2=None, channels=["PDS1"], relcal=None,
                    det="Z1", sumchans=False, convtoamps=1, fs=625e3, neventsperdump = 1000,
                    filetype="mid.gz", lgcsavefile=False,savefilepath="/galbascratch/wpage/",savefilename=""):

    """
    Function for building fake pulses by adding a template, scaled to certain amplitudes and
    certain time delays, to an existing trace (typically a random).
    
    This function calls _buildfakepulses_seg which does the heavy lifting
              
    Parameters
    ----------
    rq : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for the dataset specified
    cut : array_like
        A boolean array for the cut that selects the traces that will be loaded from the dump files. These
        traces serve as the underlying data to which a template is added.
    template1 : ndarray
        The template to be added to the traces. The template start time should be centered on the center bin
    amplitudes1 : ndarray
        The amplitudes by which to scale the template to add the the traces. Must be same length as cut
    telay1 : ndarray
        The time delay offset, in seconds, by which to shift the template to add to the traces. Bin interpolation
        not implemented
    basepath : str
        The base path to the directory that contains the folders that the event dumps 
        are in. The folders in this directory should be the series numbers.
    evtnums : array_like
        An array of all event numbers for the events in all datasets.
    seriesnums : array_like
        An array of the corresponding series numbers for each event number in evtnums.
    template2 : ndarray, optional
        The 2nd template to be added to the traces, otherwise same as template1
    amplitudes2 : ndarray, optional
        The amplitudes by which to scale the 2nd template, otherwise same as amplitudes1
    telay2 : ndarray, optional
        The time delay offset for the 2nd template, otherwise same as tdelay1
    channels : list, optional
        A list of strings that contains all of the channels that should be loaded.
    relcal : ndarray, optional
        An array with the amplitude scalings between channels used when making the total 
        If channels is supplied, relcal indices correspond to that list
        Default is all 1 (no relative scaling)
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
    neventsperdump : int, optional
        The number of events to be saved per dump file
    filetype : str, optional
        The string that corresponds to the file type that will be opened. Supports two 
        types -"mid.gz" and ".npz". "mid.gz" is the default.
    lgcsavefile : bool, optional
        A boolean flag for whether or not to save the fake data to a file
    savefilepath : str, optional
        The string that corresponds to the file path that will be saved
    savefilename : str, optional
        The string that corresponds to the file name that will be adjoined to dumpnum and saved
        
    Returns
    -------
    
    
    """
    
    # number traces passing cut
    nTraces = np.sum(cut)

    # size of cut
    sizeCut = np.size(cut)
    
    # segment the fake event building and file saving if
    # there are more traces than number of events per file
    nDump = nTraces//neventsperdump + 1
    
    # get indices of non-zero entries in the cut
    nonZeroCutInd = np.where(cut)[0]
 
    for iDump in range(nDump):
        
        # get indices for events for this dump
        jstart = (iDump*neventsperdump)
        jstop = ((iDump + 1)*neventsperdump - 1)
        
        # handle case for last dump which will have
        # fewer than neventsperdump in the dump
        if (jstop >= nTraces):
            jstop = nTraces - 1
       
        # get first and last indices for the cut mask
        maskIndStart = nonZeroCutInd[jstart]
        maskIndEnd = nonZeroCutInd[jstop] + 1
        
        # create mask for cut
        mask = np.zeros(sizeCut,dtype=bool)
        mask[maskIndStart:maskIndEnd] = True
        
        # apply mask
        cut_seg = cut & mask
        
        # call hidden buildfakepulses function with masked cut
        _buildfakepulses_seg(rq, cut_seg, template1, amplitudes1, tdelay1, basepath, evtnums, seriesnums,
                             template2=template2, amplitudes2=amplitudes2, tdelay2=tdelay2, channels=channels,
                             relcal=relcal, det=det, sumchans=sumchans, convtoamps=convtoamps, fs=fs,
                             dumpnum=(iDump+1), filetype=filetype, lgcsavefile=lgcsavefile, savefilepath=savefilepath,
                             savefilename=savefilename)
    
    return
    
def _buildfakepulses_seg(rq, cut, template1, amplitudes1, tdelay1, basepath, evtnums, seriesnums,
                    template2=None, amplitudes2=None, tdelay2=None, channels=["PDS1"], relcal=None,
                    det="Z1", sumchans=False, convtoamps=1, fs=625e3, dumpnum=1, filetype="mid.gz",
                    lgcsavefile=False,savefilepath="",savefilename=""):
    
    """
    Hidden helper function for building fake pulses
              
    Parameters
    ----------
    rq : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for the dataset specified
    cut : array_like
        A boolean array for the cut that selects the traces that will be loaded from the dump files. These
        traces serve as the underlying data to which a template is added.
    template1 : ndarray
        The template to be added to the traces. The template start time should be centered on the center bin
    amplitudes1 : ndarray
        The amplitudes by which to scale the template to add the the traces. Must be same length as cut
    telay1 : ndarray
        The time delay offset, in seconds, by which to shift the template to add to the traces. Bin interpolation
        not implemented
    basepath : str
        The base path to the directory that contains the folders that the event dumps 
        are in. The folders in this directory should be the series numbers.
    evtnums : array_like
        An array of all event numbers for the events in all datasets.
    seriesnums : array_like
        An array of the corresponding series numbers for each event number in evtnums.
    template2 : ndarray, optional
        The 2nd template to be added to the traces, otherwise same as template1
    amplitudes2 : ndarray, optional
        The amplitudes by which to scale the 2nd template, otherwise same as amplitudes1
    telay2 : ndarray, optional
        The time delay offset for the 2nd template, otherwise same as tdelay1
    channels : list, optional
        A list of strings that contains all of the channels that should be loaded.
    relcal : ndarray, optional
        An array with the amplitude scalings between channels used when making the total 
        If channels is supplied, relcal indices correspond to that list
        Default is all 1 (no relative scaling)
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
    dumpnum : int, optional
        The dump number used in the file name
    filetype : str, optional
        The string that corresponds to the file type that will be opened. Supports two 
        types -"mid.gz" and ".npz". "mid.gz" is the default.
    lgcsavefile : bool, optional
        A boolean flag for whether or not to save the fake data to a file
    savefilepath : str, optional
        The string that corresponds to the file path that will be saved
    savefilename : str, optional
        The string that corresponds to the file name that will be adjoined to dumpnum and saved
        
    Returns
    -------
    
    
    """
    
    # load all traces selected by cut
    nTraces = np.sum(cut)
    t, traces, _ = io.getrandevents(basepath, rq.eventnumber, rq.seriesnumber,
                                 cut=cut, channels=channels,det=det,sumchans=sumchans,
                                 convtoamps=convtoamps,fs=fs,ntraces = nTraces,filetype=filetype,
                                 lgcplot=False)
    
    shapeTraces = np.shape(traces)
    nchan = shapeTraces[1]
    
    if relcal is None:
        relcal = np.ones(nchan)
    else:
        if(nchan != np.size(relcal)):
            print('Error: relcal must have size equal to number of channels')
        
    # sum traces along channel dimension
    tracesSum = np.sum(traces, axis=1)

    # convert tdelay1 to bins
    tdelay1Bin = tdelay1*(1/fs)
    
    # initialize ndarray with same dimension
    # as traces to be saved
    fakepulses = np.zeros(shapeTraces)
    
    for i in range(nTraces):
        # scale and shift template1 by amplitudes1 and tdelay1
        newTrace = tracesSum[i] + amplitudes1[i]*core.shift(template1,tdelay1Bin[i])
        # multiply by reciprocal of the relative calibration such that
        # when the processing script creates the total channel pulse
        # it will be equal to newTrace
        for j in range(nchan):
            if (relcal[j]!=0):
                fakepulses[i,j,:]=newTrace/(relcal[j]*nchan)
            else:
                # the relative calibration of this channel is zero
                # thus it does not matter what the trace is equal to
                # so set to zeros to avoid division by zero
                fakepulses[i,j,:]=np.zeros(shapeTraces[2])

    if lgcsavefile:
        if (filetype=='npz'):
            trigtypes = np.zeros((nTraces,3))
            process._trigger._saveevents(pulsetimes=rq.pulsetimes[cut],
                                         pulseamps=rq.pulseamps[cut],
                                         trigtimes=rq.pulsetimes[cut],
                                         trigamps =rq.pulseamps[cut],
                                         traces=fakepulses,
                                         trigtypes=trigtypes,
                                         savepath=savefilepath,
                                         savename=savefilename,
                                         dumpnum=dumpnum)

        else:
            print('WORK IN PROGRESS: midas file writing here')
            
    return


 
            