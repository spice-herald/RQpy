import numpy as np
from scipy.signal import decimate


__all__ = ["pulse_func", "make_ideal_template", "ds_trunc"]


def pulse_func(time, tau_r, tau_f):
    """
    Simple function to make an ideal pulse shape in time domain 
    with a single pole rise and a signle pole fall
    
    Prameters
    ---------
    time: array
        Array of time values to make the pulse with
    tau_r: float
        The time constant for the exponential rise of the pulse
    tau_f: float
        The time constant for the exponential fall of the pulse
        
    Returns
    -------
    pulse: array
        The pulse magnitude as a function of time. Note, the normalization is
        arbitrary. 
        
    """
    
    pulse = np.exp(-time/tau_f)-np.exp(-time/tau_r)
    return pulse 


def make_ideal_template(x, tau_r, tau_f, offset):
    """
    Function to make ideal pulse template in time domain with single pole exponential rise
    and fall times and a given time offset. The template will be returned with maximum
    pulse hight normalized to one. The template is rolled by the specified offset to define
    the location of the peak. 
    
    Parameters
    ----------
    time: array
        Array of time values to make the pulse with
    tau_r: float
        The time constant for the exponential rise of the pulse
    tau_f: float
        The time constant for the exponential fall of the pulse
    offset: int
        The number of bins the pulse template should be shifted
        
    Returns
    -------
    template_normed: array
        the pulse template in time domain
        
    """
    
    pulse = pulse_func(x, tau_r,tau_f)
    pulse_shifted = np.roll(pulse, offset)
    pulse_shifted[:offset] = 0
    template_normed = pulse_shifted/pulse_shifted.max()
    return template_normed




def ds_trunc(traces, fs, trunc, ds, template = None):
    """
    Function to downsample and/or truncate time series data. 
    Note, this will likely change the DC offset of the traces
    
    Parameters
    ----------
    traces: ndarray
        array or time series traces
    fs: int
        sample rate
    trunc: int
        index of where the trace should be truncated
    ds: int
        scale factor for how much downsampling to be done
        ex: ds = 16 means traces will be downsampled by a factor
        of 16
    template: ndarray, optional
        pulse template to be downsampled
    
    Returns
    -------
    traces_ds: ndarray
        downsampled/truncated traces
    psd_ds: ndarray
        psd made from downsampled traces
    fs_ds: int
        downsampled frequency
    template_ds: ndarray, optional
        downsampled template
        
    """
    # truncate the traces/template
    
    traces_trunc = traces[..., :trunc]
    
    trunc_time = trunc/fs
    
    # low pass filter and downsample the traces/template
    if template is not None:
        template_trunc = template[(len(template)-trunc)//2:(len(template)-trunc)//2+trunc]
        template_ds = decimate(template_trunc, ds, zero_phase=True)
    traces_ds = decimate(traces_trunc, ds, zero_phase=True)
    
    fs_ds = len(traces_ds)/trunc_time
    
    f_ds, psd_ds = calc_psd(traces_ds, fs=fs_ds, folded_over=False)
    if template is not None:
        return traces_ds, template_ds, psd_ds, fs_ds
    else:
        return traces_ds, psd_ds, fs_ds