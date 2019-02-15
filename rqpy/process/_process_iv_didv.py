import numpy as np
import pandas as pd
import os
import multiprocessing
from itertools import repeat
from glob import glob

from rqpy import io
from rqpy import HAS_SCDMSPYTOOLS
from qetpy import calc_psd, autocuts, DIDV
from qetpy.utils import calc_offset

if HAS_SCDMSPYTOOLS:
    from scdmsPyTools.BatTools.IO import getRawEvents, getDetectorSettings
    from scdmsPyTools.BatTools import rawdata_reader as rawdata

__all__ = ["process_ivsweep"]


def _process_ivfile(filepath, chans, detectorid, rfb, loopgain, binstovolts, 
                    rshunt, rbias, lgcHV, lgcverbose):
    """
    Helper function to process data from noise or dIdV series as part of an IV/dIdV sweep. See Notes for 
    more details on what parameters are calculated
    
    Parameters
    ----------
    filepath : str
        Absolute path to the series folder
    chans : list
        List containing strings corresponding to the names of all the channels of interest
    detectorid : str
        The label of the detector, i.e. Z1, Z2, .. etc
    rfb : int
        The resistance of the feedback resistor in the phonon amplifier
    loopgain : float
        The ratio of number of turns in the squid input coil vs feedback coil
    binstovolts : int
        The bit depth divided by the dynamic range of the ADC in Volts
    rshunt : float
        The value of the shunt resistor in the TES circuit
    rbias : int
        The value of the bias resistor on the test signal line
    lgcHV : bool
        If False (default), the detector is assumed to be operating in iZip mode, 
        If True, HV mode. Note, the channel names will be different between the 
        two modes, it is up to the user to make sure the channel names are correct
    lgcverbose : bool
        If True, the series number being processed will be displayed
        
    Returns
    -------
    data_list : list
        The list of calculated parameters
        
    Notes
    -----
    For each series passed to this function, the following paramers are calculated/looked up:
    
        Channel name, 
        Series number, 
        Sample rate, 
        Qetbias, 
        Amplitude of signal generator (referenced in terms of QET bias current) #If didv data, 
        Frequency of signal generator #If didv data, 
        DC offset, 
        STD of the DC offset,
        PSD (folded over),
        Corresponding frequencies for the PSD,
        Average trace,
        dIdV mean (calculated using DIDV class),
        dIdV STD (calculated using DIDV class),
        data type ('noise' or 'didv'),
        Efficiency of the auto cuts to the data, 
        The boolean cut mask,
        A boolean saying whether or not the auto cuts were successful or not.

    
    """
    
    if lgcverbose:
        print(f'------------------\n Processing dumps in file: {filepath} \n------------------')
    
    
    if isinstance(chans, str):
        chans = [chans]
    
    detnum = int(detectorid[-1])
    
    nchan = len(chans)
    
    data_list = []
    if filepath[-1] == '/':
        seriesnum = filepath[:-1].split('/')[-1]
    else:
        seriesnum = filepath.split('/')[-1]
    reader = rawdata.DataReader()
    settings_path = glob(f"{filepath}*")[0]
    reader.set_filename(settings_path)

    if not lgcHV:
        odb_list = [f"/Detectors/Det0{detnum}/Readback/TestSignal/Amplitude (mV)",
                    f"/Detectors/Det0{detnum}/Readback/TestSignal/Frequency (Hz)",
                    f"/Detectors/Det0{detnum}/Readback/TestSignal/GeneratorEnable",
                    f"/Detectors/Det0{detnum}/Readback/TestSignal/QETTestEnable",
                    f"/Detectors/Det0{detnum}/Readback/TestSignal/QETTestSelect"]
    else:
        odb_list = [f"/Detectors/Det0{detnum}/Readback/TestSignal/Amplitude (mV)",
                    f"/Detectors/Det0{detnum}/Readback/TestSignal/Frequency (Hz)",
                    f"/Detectors/Det0{detnum}/Readback/TestSignal/GeneratorEnable",
                    f"/Detectors/Det0{detnum}/Readback/TestSignal/QETTestEnable",
                    f"/Detectors/Det0{detnum}/Readback/TestSignal/QETTestSelect[0]",
                    f"/Detectors/Det0{detnum}/Readback/TestSignal/QETTestSelect[1]"]
    reader.set_odb_list(odb_list)
    odb_dict = reader.get_odb_dict()
    
    lgcGenEnable = odb_dict[odb_list[2]] #True/False
    lgcQETEnable = odb_dict[odb_list[3]] #True/False
    qetChanSelect0 =  odb_dict[odb_list[4]] #channel number
    if lgcHV:
        qetChanSelect1 =  odb_dict[odb_list[5]] #channel number
    else:
        qetChanSelect1 =  odb_dict[odb_list[4]] #channel number
        
    ### Load traces and channel names in order of loaded traces
    events = getRawEvents(filepath, "", channelList=chans, detectorList=[detnum], outputFormat=3)
    channels = events[detectorid]["pChan"] # we use the returned channels rather than the user 
                                           # provided channel list because the order returned 
                                           # from getRawEvents() is not nessesarily in the 
                                           # expected order
    traces = events[detectorid]["p"]
    
    settings = getDetectorSettings(filepath, "")
    detcodes = [settings[detectorid][ch]["channelNum"] for ch in channels]
    drivergain = [2*settings[detectorid][ch]["driverGain"] for ch in channels] # extra factor of two from filters
    qetbias = [settings[detectorid][ch]["qetBias"] for ch in channels]
    fs = [1/settings[detectorid][ch]["timePerBin"] for ch in channels]
    
    
    
    
    
    if (not lgcGenEnable or not lgcQETEnable):
        for ii in range(nchan):
            
            convtoamps = 1/(drivergain[ii] * rfb * loopgain * binstovolts)
            traces_temp = traces[:,ii]*convtoamps
            
            cut_pass = True
            try:
                cut = autocuts(traces_temp, fs=fs[ii])
            except:
                cut = np.ones(shape = traces_temp.shape[0], dtype=bool)
                cut_pass = False 
            
            f, psd = calc_psd(traces_temp[cut], fs=fs[ii])
            
            offset, offset_err = calc_offset(traces_temp[cut], fs=fs[ii])
            
            avgtrace = np.mean(traces_temp[cut], axis = 0)
            sgamp = None
            sgfreq = None
            datatype = 'noise'
            cut_eff = np.sum(cut)/len(cut)
            didvmean = None
            didvstd = None
            
            data = [channels[ii], seriesnum, fs[ii], qetbias[ii], sgamp, sgfreq, offset, offset_err, 
                    f, psd, avgtrace, didvmean, didvstd, datatype, cut_eff, cut, cut_pass]
            data_list.append(data)
            
    elif (lgcGenEnable and lgcQETEnable):
        for ii in range(nchan):
            if (qetChanSelect0 == detcodes[ii] or qetChanSelect1 == detcodes[ii]):

                
                convtoamps = drivergain[ii] * rfb * loopgain * binstovolts
                sgamp = odb_dict[odb_list[0]]*1e-3/rbias # conversion from mV to V, convert to qetbias jitter
                sgfreq = int(odb_dict[odb_list[1]])
                
                traces_temp = traces[:,ii]/convtoamps
                
                # get rid of traces that are all zero
                zerocut = np.all(traces_temp!=0, axis=1)
                
                traces_temp = traces_temp[zerocut]
                
                cut_pass = True
                try:
                    cut = autocuts(traces_temp, fs=fs[ii], is_didv=True, sgfreq=sgfreq)
                except:
                    cut = np.ones(shape = traces_temp.shape[0], dtype=bool)
                    cut_pass = False 

                offset, offset_err = calc_offset(traces_temp[cut], fs=fs[ii], sgfreq=sgfreq, is_didv=True)
                avgtrace = np.mean(traces_temp[cut], axis = 0)
                
                didvobj = DIDV(traces_temp[cut], fs[ii], sgfreq, sgamp, rshunt)
                didvobj.processtraces()
                
                didvmean = didvobj.didvmean
                didvstd = didvobj.didvstd

                f = None
                psd = None
                datatype = 'didv'
                cut_eff = np.sum(cut)/len(cut)
                
                data = [channels[ii], seriesnum, fs[ii], qetbias[ii], sgamp, sgfreq, offset, offset_err, 
                        f, psd, avgtrace, didvmean, didvstd, datatype, cut_eff, cut, cut_pass]
                data_list.append(data)
    
    return data_list


def process_ivsweep(ivfilepath, chans, detectorid="Z1", rfb=5000, loopgain=2.4, binstovolts=65536/2, 
                    rshunt=0.005, rbias=20000, lgcHV=False, lgcverbose=False, lgcsave=True,
                    nprocess=1, savepath='', savename='IV_dIdV_DF'):
    """
    Function to process data for an IV/dIdV sweep. See Notes for 
    more details on what parameters are calculated
    
    Parameters
    ----------
    ivfilepath : str
        Absolute path to the directory containing all the series in the sweep
    chans : list
        List containing strings corresponding to the names of all the channels of interest
    detectorid : str, optional
        The label of the detector, i.e. Z1, Z2, .. etc
    rfb : int
        The resistance of the feedback resistor in the phonon amplifier
    loopgain : float, optional
        The ratio of number of turns in the squid input coil vs feedback coil
    binstovolts : int, optional
        The bit depth divided by the dynamic range of the ADC in Volts
    rshunt : float, optional
        The value of the shunt resistor in the TES circuit
    rbias : int, optional
        The value of the bias resistor on the test signal line
    lgcHV : bool, optional
        If False (default), the detector is assumed to be operating in iZip mode, 
        If True, HV mode. Note, the channel names will be different between the 
        two modes, it is up to the user to make sure the channel names are correct
    lgcverbose : bool, optional
        If True, the series number being processed will be displayed
    lgcsave : bool, optional
        If True, the processed DF is saved in the user specified directory
    nprocess : int, optional
        Number of jobs to use to process IV dIdV sweep. If nprocess = 1, only a single
        core will be used. If more than one, Pool will be used for multiprocessing. 
        Note, if you are running this on a shared computer, no more than 4 jobs should be
        used, idealy 2, as it will significantly slow down the computer.
    lgcsave : bool, optional
        If True, the processed DataFrame will be saved
    savepath : str, optional
        Abosolute path to save DataFrame
    savename : str, optional
        The name of the processed DataFrame to be saved
        
    Returns
    -------
    df : pandas.DataFrame
        A pandas DataFrame with all the processed parameters for the IV dIdV sweep
        
    Notes
    -----
    For each series in the IV/dIdV sweep, the following paramers are calculated/looked up:
    
        Channel name, 
        Series number, 
        Sample rate, 
        Qetbias, 
        Amplitude of signal generator (referenced in terms of QET bias current) #If didv data, 
        Frequency of signal generator #If didv data, 
        DC offset, 
        STD of the DC offset,
        PSD (folded over),
        Corresponding frequencies for the PSD,
        Average trace,
        dIdV mean (calculated using DIDV class),
        dIdV STD (calculated using DIDV class),
        data type ('noise' or 'didv'),
        Efficiency of the auto cuts to the data, 
        The boolean cut mask,
        A boolean saying whether or not the auto cuts were successful or not.

    
    """
    if not HAS_SCDMSPYTOOLS:
        raise ImportError("""Cannot use this IV processing because scdmsPyTools is not installed. 
                          More file types will be supported in future releases of RQpy.""")
    
    files = sorted(glob(ivfilepath +'*/'))
    
    if nprocess == 1:
        results = []
        for filepath in files:
            results.append(_process_ivfile(filepath, chans, detectorid, rfb, loopgain, binstovolts, 
                 rshunt, rbias, lgcHV, lgcverbose))
    else:
        pool = multiprocessing.Pool(processes = int(nprocess))
        results = pool.starmap(_process_ivfile, zip(files, repeat(chans, detectorid, rfb, loopgain, binstovolts, 
                 rshunt, rbias, lgcHV, lgcverbose))) 
        pool.close()
        pool.join()
        
    flat_result = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flat_result, columns=["channels", "seriesnum", "fs", "qetbias", "sgamp", "sgfreq", "offset", 
                                       "offset_err", "f", "psd", "avgtrace", "didvmean", "didvstd", "datatype", 
                                       "cut_eff", "cut", "cut_pass"])
    
    if lgcsave:
        df.to_pickle(f"{savepath}{savename}.pkl")
    
    return df
    
    
