import numpy as np
import qetpy as qp
from scipy import signal
from scipy import ndimage


__all__ = ["shift", "make_ideal_template", "downsample_truncate"]


def shift(arr, num, fill_value=0):
    """
    Function for shifting the values in an array by a certain number of indices, filling
    the values of the bins at the head or tail of the array with fill_value.
    
    Parameters
    ----------
    arr : array_like
        Array to shift values in.
    num : float
        The number of values to shift by. If positive, values shift to the right. If negative, 
        values shift to the left.
        If num is a non-whole number of bins, arr is linearly interpolated
    fill_value : scalar, optional
        The value to fill the bins at the head or tail of the array with.
    
    Returns
    -------
    result : ndarray
        The resulting array that has been shifted and filled in.
    
    """
    
    result = np.empty_like(arr)
    
    if float(num).is_integer():
        num = int(num) # force num to int type for slicing
        
        if num > 0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num < 0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result[:] = arr
    else:
        result = ndimage.shift(arr, num, order=1, mode='constant', cval=fill_value)
        
    return result


def make_ideal_template(t, tau_r, tau_f, offset=0):
    """
    Function to make an ideal pulse template in time domain with single pole exponential rise
    and fall times, and a given time offset. The template will be returned with the maximum
    pulse height normalized to one. The pulse, by default, begins at the center of the trace, 
    which can be left or right shifted via the `offset` optional argument.
    
    Parameters
    ----------
    t : ndarray
        Array of time values to make the pulse with
    tau_r : float
        The time constant for the exponential rise of the pulse
    tau_f : float
        The time constant for the exponential fall of the pulse
    offset : int
        The number of bins the pulse template should be shifted
        
    Returns
    -------
    template_normed : array
        the pulse template in time domain
        
    """
    
    pulse = np.exp(-t/tau_f)-np.exp(-t/tau_r)
    pulse_shifted = shift(pulse, len(t)//2 + offset)
    template_normed = pulse_shifted/pulse_shifted.max()
    
    return template_normed


def downsample_truncate(traces, fs, trunc, ds, template=None, calcpsd=False):
    """
    Function to downsample and/or truncate time series data. Note that this will 
    likely change the DC offset of the traces.
    
    Parameters
    ----------
    traces : ndarray
        Array of time series traces.
    fs : int
        Digitization rate of the inputted traces in Hz.
    trunc : int
        Length of the truncated trace in bins.
    ds : int
        Scale factor for how much downsampling to be done, e.g. ds = 16 means 
        traces will be downsampled by a factor of 16.
    template : ndarray, optional
        Pulse template to be downsampled and truncated.
    calcpsd : bool, optional
        Boolean flag for whether or not to calculate the PSD of the downsampled
        and truncated traces.
    
    Returns
    -------
    res : dict
        The results of the downsampling/truncation of the traces.
        The returned keys are:
            'traces_ds'   : Array of the downsampled and truncated traces.
            'fs_ds'       : Downsampled digitization frequency in Hz.
            'psd_ds'      : PSD made from the downsampled and truncated traces. (optional)
            'template_ds' : Downsampled and truncated pulse template. (optional)
        
    """
    
    traces_trunc = traces[..., (traces.shape[-1]-trunc)//2:(traces.shape[-1]-trunc)//2+trunc]
    
    trunc_time = trunc/fs
    
    traces_ds = signal.decimate(traces_trunc, ds, zero_phase=True)
    
    fs_ds = traces_ds.shape[-1]/trunc_time
    
    res = {}
    res['traces_ds'] = traces_ds
    res['fs_ds'] = fs_ds
    
    if template is not None:
        template_trunc = template[(len(template)-trunc)//2:(len(template)-trunc)//2+trunc]
        template_ds = signal.decimate(template_trunc, ds, zero_phase=True)
        res['template_ds'] = template_ds
    
    if calcpsd:
        f_ds, psd_ds = qp.calc_psd(traces_ds, fs=fs_ds, folded_over=False)
        res['psd_ds'] = pds_ds
    
    return res


