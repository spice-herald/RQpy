import numpy as np
from rqpy import HAS_SCDMSPYTOOLS

if HAS_SCDMSPYTOOLS:
    from scdmsPyTools.BatTools import rawdata_writer as writer


__all__ = ["saveevents_npz", "saveevents_midgz"]


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
    
    if not HAS_SCDMSPYTOOLS:
        raise ImportError("Cannot use save mid.gz files because scdmsPyTools is not installed.")
    
    mywriter = writer.DataWriter()
    
    filename_out = f"{savename}_F{dumpnum:04}.mid.gz"
    mywriter.open_file(filename_out, savepath)
    mywriter.write_settings_from_dict(settings)
    mywriter.write_events(events)
    mywriter.close_file()

