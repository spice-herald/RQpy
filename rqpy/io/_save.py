import numpy as np
from rqpy import HAS_SCDMSPYTOOLS

if HAS_SCDMSPYTOOLS:
    from scdmsPyTools.BatTools import rawdata_writer as writer


__all__ = ["saveevents_npz", "saveevents_midgz"]


def saveevents_npz(pulsetimes=None, pulseamps=None, trigtimes=None, trigamps=None, randomstimes=None, 
                   traces=None, trigtypes=None, savepath=None, savename=None, dumpnum=None):
    """
    Function for simple saving of events to .npz file.
    
    Parameters
    ----------
    pulsetimes : ndarray, optional
        If we triggered on a pulse, the time of the pulse trigger in seconds. Otherwise this is zero.
    pulseamps : ndarray, optional
        If we triggered on a pulse, the optimum amplitude at the pulse trigger time. Otherwise this is zero.
    trigtimes : ndarray, optional
        If we triggered due to ttl, the time of the ttl trigger in seconds. Otherwise this is zero.
    trigamps : ndarray, optional
        If we triggered due to ttl, the optimum amplitude at the ttl trigger time. Otherwise this is zero.
    randomstimes : ndarray, optional
        Array of the corresponding event times for each section
    traces : ndarray, optional
        The corresponding trace for each detected event.
    trigtypes: ndarray, optional
        Array of boolean vectors each of length 3. The first value indicates if the trace is a random or not.
        The second value indicates if we had a pulse trigger. The third value indicates if we had a ttl trigger.
    savepath : NoneType, str, optional
        Path to save the events to.
    savename : NoneType, str, optional
        Filename to save the events as.
    dumpnum : int, optional
        The dump number of the current file.
        
    """
    
    if randomstimes is None:
        randomstimes = np.zeros_like(pulsetimes)
        
    if pulsetimes is None:
        pulsetimes = np.zeros_like(randomstimes)
        pulseamps = np.zeros_like(randomstimes)
        trigtimes = np.zeros_like(randomstimes)
        trigamps = np.zeros_like(randomstimes)
    
    filename = f"{savepath}{savename}_{dumpnum:04d}.npz"
    np.savez(filename, 
             pulsetimes=pulsetimes, 
             pulseamps=pulseamps, 
             trigtimes=trigtimes, 
             trigamps=trigamps, 
             randomstimes=randomstimes, 
             traces=traces, 
             trigtypes=trigtypes)

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
    
    filename_out = f"{savename}_S{dumpnum:04}.mid.gz"
    mywriter.open_file(filename_out, savepath)
    mywriter.write_settings_from_dict(settings)
    mywriter.write_events(events)
    mywriter.close_file()

