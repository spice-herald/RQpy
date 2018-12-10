import numpy as np
import matplotlib.pyplot as plt
from rqpy import plotting, utils
import rqpy as rp

__all__ = ["scale_saturated_energy"]

def scale_saturated_energy(vals, fitparams):
    """
    Thing to do stuff
    """
    params = fitparams[0]
    cov = fitparams[1]
    
    energy_true = utils.invert_saturation_func(vals, *params)
    errors = utils.prop_sat_err(vals,popt,pcov)
    
    return energy_true, errors
    
    
    
    
    
    
    