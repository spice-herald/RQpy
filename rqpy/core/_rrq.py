from rqpy import utils

__all__ = ["scale_saturated_energy"]

def scale_saturated_energy(vals, fitparams):
    """
    Function to convert saturated measured energy into 
    true energy
    
    Parameters
    ----------
    vals : ndarray
        Array of measured energies to be converted to true energies
    fitparams : list
        List containing the best fit parameters from the fit_saturation() 
        fuction. fitparams[0] should correspond to the optimum parameters
        and fitparams[1] should be the covariance matrix from the fit
        
    Returns
    -------
    energy_true : ndarray
        Array of saturation corrected energies
    errors : ndarray
        Array of uncertainties for each value in energy_true
        
    """
    
    params = fitparams[0]
    cov = fitparams[1]
    
    energy_true = utils.invert_saturation_func(vals, *params)
    errors = utils.prop_sat_err(vals, params, cov)
    
    return energy_true, errors
    
    
    
    
    
    
    
