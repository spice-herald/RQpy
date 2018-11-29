import numpy as np


__all__ = ["pulse_func", "make_ideal_template"]


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
    pulse hight normalized to one. 
    
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