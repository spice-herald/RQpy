import numpy as np
import matplotlib.pyplot as plt
from rqpy import plotting, utils
import rqpy as rp

__all__ = ["scale_saturated_energy"]

def scale_saturated_energy(vals, fitparams):
    """
    Function to convert saturated measured enrgy into 
    true energy
    
    Parameters
    ----------
    vals: array
        Array of measured energies to be converted to true energies
    fitparams: list
        List containing the best fit parameters from the fit_saturation() 
        fuction. fitparams[0] should correspond to the optimum parameters
        and fitparams[1] should be the covariance matrix from the fit
        
    Returns
    -------
    energy_true: array
        Array of saturation corrected energies
    errors: array
        Array of uncertainties for each value in energy_true
        
    """
    
    params = fitparams[0]
    cov = fitparams[1]
    
    energy_true = utils.invert_saturation_func(vals, *params)
    errors = utils.prop_sat_err(vals, params, cov)
    
    return energy_true, errors
    
    
    
    
    
    
    