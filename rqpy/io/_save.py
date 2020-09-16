import numpy as np
from rqpy.io import get_traces_midgz
from rqpy import HAS_RAWIO
import deepdish as dd

if HAS_RAWIO:
    from rawio import DataWriter
    from rawio.IO import getRawEvents


__all__ = ["saveevents_npz", "saveevents_midgz", "convert_midgz_to_h5"]


def _check_kwargs_npz(**kwargs):
    """
    Helper function for extracting the array length that is being saved.
    
    Parameters
    ----------
    kwargs : dict
        The keyword arguments from `rqpy.io.saveevents_npz` that correspond to
        the inputted ndarrays.
        
    Returns
    -------
    arr_len : int
        The length of the arrays (assuming they all have the same length).
    
    """

    for key in kwargs.keys():
        if kwargs[key] is not None:
            return len(kwargs[key])
    
    raise IOError("Cannot save file, all arrays appear to be None.")


def saveevents_npz(pulsetimes=None, pulseamps=None, trigtimes=None, trigamps=None, randomstimes=None, 
                   traces=None, trigtypes=None, truthamps=None, truthtdelay=None,
                   savepath=None, savename=None, dumpnum=None):
    """
    Function for simple saving of events to .npz file.
    
    Parameters
    ----------
    pulsetimes : ndarray, NoneType, optional
        If we triggered on a pulse, the time of the pulse trigger in seconds. Otherwise this is zero.
    pulseamps : ndarray, NoneType, optional
        If we triggered on a pulse, the optimum amplitude at the pulse trigger time. Otherwise this is zero.
    trigtimes : ndarray, NoneType, optional
        If we triggered due to ttl, the time of the ttl trigger in seconds. Otherwise this is zero.
    trigamps : ndarray, NoneType, optional
        If we triggered due to ttl, the optimum amplitude at the ttl trigger time. Otherwise this is zero.
    randomstimes : ndarray, NoneType, optional
        Array of the corresponding event times for each section, if this is a random.
    traces : ndarray, NoneType, optional
        The corresponding trace for each detected event.
    trigtypes : ndarray, NoneType, optional
        Array of boolean vectors each of length 3. The first value indicates if the trace is a random or not.
        The second value indicates if we had a pulse trigger. The third value indicates if we had a ttl trigger.
    truthamps : ndarray, NoneType, optional
        If the data being saved is simulated data, this is a 2-d ndarray of the true amplitudes for each trace,
        where the shape is (number of traces, number of templates). Otherwise, this is zero.
    truthtdelay : ndarray, NoneType, optional
        If the data being saved is simulated data, this is a 2-d ndarray of the true tdelay for each trace,
        where the shape is (number of traces, number of templates). Otherwise, this is zero.
    savepath : str, NoneType, optional
        Path to save the events to.
    savename : str, NoneType, optional
        Filename to save the events as.
    dumpnum : int, optional
        The dump number of the current file.
        
    """
    
    filename = f"{savepath}{savename}_{dumpnum:04d}.npz"
    
    arr_len = _check_kwargs_npz(pulsetimes=pulsetimes, pulseamps=pulseamps, trigtimes=trigtimes, 
                                trigamps=trigamps, randomstimes=randomstimes, traces=traces,
                                trigtypes=trigtypes, truthamps=truthamps, truthtdelay=truthtdelay)
    
    if randomstimes is None:
        randomstimes = np.zeros(arr_len)
        
    if pulsetimes is None:
        pulsetimes = np.zeros(arr_len)
        pulseamps = np.zeros(arr_len)
        trigtimes = np.zeros(arr_len)
        trigamps = np.zeros(arr_len)
    
    if truthamps is None:
        truthamps = np.zeros((arr_len, 1))
    
    if truthtdelay is None:
        truthtdelay = np.zeros((arr_len, 1))
    
    np.savez(filename, 
             pulsetimes=pulsetimes, 
             pulseamps=pulseamps, 
             trigtimes=trigtimes, 
             trigamps=trigamps, 
             randomstimes=randomstimes, 
             traces=traces, 
             trigtypes=trigtypes,
             truthamps=truthamps,
             truthtdelay=truthtdelay)


def saveevents_midgz(events, settings, savepath, savename, dumpnum):
    """
    Function for writing events to MIDAS files.
    
    Parameters
    ----------
    events : list
        The list of events, which has been set up by `rqpy.sim._pulsesim._create_events_list`. This list
        must have a very specific format in order to be successfully saved to a MIDAS file.
    settings : dict
        Dictionary that contains all of the relevant detector settings for saving a MIDAS file.
    savepath : str
        Path to save the events to.
    savename : str
        Filename to save the events as.
    dumpnum : int
        The dump number of the current file.
    
    Returns
    -------
    None
    
    """
    
    if not HAS_RAWIO:
        raise ImportError("Cannot use save mid.gz files because cdms rawio is not installed.")
    
    mywriter = DataWriter()
    
    filename_out = f"{savename}_F{dumpnum:04}.mid.gz"
    mywriter.open_file(filename_out, savepath)
    mywriter.write_settings_from_dict(settings)
    mywriter.write_events(events)
    mywriter.close_file()  


def convert_midgz_to_h5(path, savepath, channels, det, lgcskip_empty=False):
    """
    Function to convert raw traces and event numbers for a single dump from mid.gz to HDF5.
    
    Saves the ndarray of traces, converted to units of TES current, and saves a corresponding
    array of event numbers. Note, since event numbers are not unique, the series number is
    appended to the front of the event number, ie. seriesnumber_eventnumber (as an integer
    withough the underscore) 
    
    Parameters
    ----------
    path : str, list of str
        Absolute path, or list of paths, to the dump to open. 
    savepath : str
        Absolute path to where the dump should be saved
    channels : str, list of str
        Channel name(s), i.e. 'PDS1'. If a list of channels, the outputted traces will be sorted to match the order
        the getRawEvents reports in events[det]['pChan'], which can cause slow downs. It is recommended to match
        this order if opening many or large files.
    det : str, list of str
        Detector name, i.e. 'Z1'. If a list of strings, then should each value should directly correspond to 
        the channel names. If a string is inputted and there are multiple channels, then it 
        is assumed that the detector name is the same for each channel.
    lgcskip_empty : bool, optional
        Boolean flag on whether or not to skip empty events. Should be set to True if user only wants the traces.
        If the user also wants to pull extra timing information (primarily for live time calculations), then set
        to False. Default is False.
    
    Returns
    -------
    None
    
    """
    
    if not isinstance(path, list):
        path = [path]

    savename = []
    for p in path:
        savename.append(p.split('/')[-1].split('.')[0])

    for jj, p in enumerate(path):
        x, info_dict = get_traces_midgz(p, channels, det, lgcskip_empty=lgcskip_empty, lgcreturndict=True)
        for key in info_dict:
            info_dict[key] = np.asarray(info_dict[key])
        info_dict['traces'] = x
        dd.io.save(f'{savepath}{savename[jj]}.h5', info_dict)
        
        
    

