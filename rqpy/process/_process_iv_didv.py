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
    import rawio.IO  as midasio


__all__ = ["process_ivsweep"]


def _process_ivfile(filepath, chans, detectorid, rfb, loopgain, binstovolts, 
                    rshunt, rbias, lowpassgain, lgcverbose):
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
    lowpassgain: int
        The value of fix low pass filter driver gain (DCRC RevD = 2, DCRC RevE = 4)
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
        print(f'\n\n============================')
        print(f'Processing dumps in file: {filepath}')
        print(f'============================\n')
    
    if isinstance(chans, str):
        chans = [chans]
    
    detnum = int(detectorid[1:])
    nchan = len(chans)
    

    # Data list
    data_list = []
    if filepath[-1] == '/':
        seriesnum = filepath[:-1].split('/')[-1]
    else:
        seriesnum = filepath.split('/')[-1]


    # Load traces for user provided detector and channels
    events = []
    try:
        events = midasio.getRawEvents(filepath, "", channelList=chans, detectorList=[detnum], 
                                      outputFormat=3)
    except:
        print('ERROR in process_iv_didv: Unable to get traces!')
        return

    traces = events[detectorid]["p"]
    channels = events[detectorid]["pChan"]

    # Get detector settings (-> use first file)
    settings_file = glob(f"{filepath}*")[0]

    detector_settings =[]
    signal_gen_settings = []
    try:
        detector_settings = midasio.getDetectorSettings('',settings_file)[detectorid]
        signal_gen_settings = midasio.getTestSignalInfo('',settings_file,detectorList=[detnum])[detectorid]
    except:
        print('ERROR in process_iv_didv: Unable to get detector settings!')
        return
   
    
    # Is it an IV or dIdV data?
    # if signal generator enable AND one channel connected to QET
    #    -> dIdV

    is_didv = False
    for chan in channels:
        if (signal_gen_settings[chan]['GeneratorEnable'] and 
            signal_gen_settings[chan]['QETConnection']):
            is_didv = True
    
    
     
    # process...
    if not is_didv:
        
        # =====================
        # IV processing
        # =====================
        if lgcverbose:
            print(f'--------------\nIV processing\n--------------')
    
        # LOOP channels
        for chan in channels:
            
            # channel array index
            chan_index = channels.index(chan)
            

            if lgcverbose:
                print(f'Processing channel ' + chan + ' (ndarray index = ' + str(chan_index) + ')')
                
            continue

            # settings
            qetbias = detector_settings[chan]['qetBias']
            fs = 1/detector_settings[chan]['timePerBin']
                  
            # conversion factors
            drivergain = lowpassgain * detector_settings[chan]['driverGain']
            convtoamps = 1/(drivergain * rfb * loopgain * binstovolts)
        
            # normalize traces
            traces_temp = traces[:,chan_index]*convtoamps

            # apply cut
            cut_pass = True
            try:
                cut = autocuts(traces_temp, fs=fs)
            except:
                cut = np.ones(shape = traces_temp.shape[0], dtype=bool)
                cut_pass = False 

                
                
            # PSD calculation
            f, psd = calc_psd(traces_temp[cut], fs=fs)

            # Offset calculation
            offset, offset_err = calc_offset(traces_temp[cut], fs=fs)
        

            # Pulse average
            avgtrace = np.mean(traces_temp[cut], axis = 0)

            # Store data
            sgamp = None
            sgfreq = None
            datatype = 'noise'
            cut_eff = np.sum(cut)/len(cut)
            didvmean = None
            didvstd = None
            
            data = [chan, seriesnum, fs, qetbias, sgamp, sgfreq, offset, offset_err, 
                    f, psd, avgtrace, didvmean, didvstd, datatype, cut_eff, cut, cut_pass]
            data_list.append(data)

    else:
            
          
        # =====================
        # dIdV processing
        # =====================
        if lgcverbose:
            print(f'--------------\ndIdV processing\n--------------')
          
        # LOOP channels
        for chan in channels:
            
            # check if signal generator enabled on that channel
            if  not signal_gen_settings[chan]['QETConnection']:
                continue
                
            # channel array index
            chan_index = channels.index(chan)


            if lgcverbose:
                print(f'Processing channel ' + chan + ' (ndarray index = ' + str(chan_index) +')')
            

            # settings
            qetbias = detector_settings[chan]['qetBias']
            fs = 1/detector_settings[chan]['timePerBin']
                  
            # conversion factors
            drivergain = lowpassgain * detector_settings[chan]['driverGain']
            convtoamps = 1/(drivergain * rfb * loopgain * binstovolts)


            # normalize traces
            traces_temp = traces[:,chan_index]*convtoamps


            # signal generator conversion, from mV and Amps
            sgamp = signal_gen_settings[chan]['Amplitude']*1e-3/rbias
            sgfreq = int(signal_gen_settings[chan]['Frequency'])
         

            # get rid of traces that are all zero
            zerocut = np.all(traces_temp!=0, axis=1)
            traces_temp = traces_temp[zerocut]
                
            
            # pile-up cuts
            cut_pass = True
            try:
                cut = autocuts(traces_temp, fs=fs, is_didv=True, sgfreq=sgfreq)
            except:
                cut = np.ones(shape = traces_temp.shape[0], dtype=bool)
                cut_pass = False 


            # Offset calculation
            offset, offset_err = calc_offset(traces_temp[cut], fs=fs, sgfreq=sgfreq, is_didv=True)
         

            # Average pulse
            avgtrace = np.mean(traces_temp[cut], axis = 0)
                
            # dIdV fit
            didvobj = DIDV(traces_temp[cut], fs, sgfreq, sgamp, rshunt)
            didvobj.processtraces()
            
            # store data
            didvmean = didvobj.didvmean
            didvstd = didvobj.didvstd
            f = None
            psd = None
            datatype = 'didv'
            cut_eff = np.sum(cut)/len(cut)
                
            data = [chan, seriesnum, fs, qetbias, sgamp, sgfreq, offset, offset_err, 
                    f, psd, avgtrace, didvmean, didvstd, datatype, cut_eff, cut, cut_pass]
            data_list.append(data)

    
    return data_list




def process_ivsweep(ivfilepath, chans, detectorid="Z1", rfb=5000, loopgain=2.4, binstovolts=65536/8, 
                    rshunt=0.005, rbias=20000, lowpassgain=4, lgcverbose=False, lgcsave=True,
                    nprocess=1, savepath='', savename='IV_dIdV_DF'):
    """
    Function to process data for an IV/dIdV sweep. See Notes for 
    more details on what parameters are calculated
    
    Parameters
    ----------
    ivfilepath : str, list of str
        Absolute path to the directory containing all the series in the sweep. Can also pass a list of specific
        paths to the parent directories of files to process.
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
    lowpassgain: int, optional
         The value of fix low pass filter driver gain (DCRC RevD = 2, DCRC RevE = 4)
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
        raise ImportError("""Cannot use this IV processing because cdms rawio is not installed. 
                          More file types will be supported in future releases of RQpy.""")
    
    if isinstance(ivfilepath, str):
        files = sorted(glob(ivfilepath +'*/'))
    else:
        files = ivfilepath
    
    if nprocess == 1:
        results = []
        for filepath in files:
            results.append(_process_ivfile(filepath, chans, detectorid, rfb, loopgain, binstovolts, 
                 rshunt, rbias, lowpassgain, lgcverbose))
    else:
        pool = multiprocessing.Pool(processes = int(nprocess))
        results = pool.starmap(_process_ivfile, zip(files, repeat(chans, detectorid, rfb, loopgain, binstovolts, 
                 rshunt, rbias, lowpassgain, lgcverbose))) 
        pool.close()
        pool.join()
        
    flat_result = [item for sublist in results for item in sublist]
    df = pd.DataFrame(flat_result, columns=["channels", "seriesnum", "fs", "qetbias", "sgamp", "sgfreq", "offset", 
                                       "offset_err", "f", "psd", "avgtrace", "didvmean", "didvstd", "datatype", 
                                       "cut_eff", "cut", "cut_pass"])
    
    if lgcsave:
        df.to_pickle(f"{savepath}{savename}.pkl")
    
    return df
    
    
