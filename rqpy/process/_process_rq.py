import numpy as np
import pandas as pd
import os
import multiprocessing
from itertools import repeat
import warnings

import rqpy as rp
from rqpy import io
import qetpy as qp
from rqpy import HAS_SCDMSPYTOOLS

if HAS_SCDMSPYTOOLS:
    from scdmsPyTools.BatTools.IO import getRawEvents, getDetectorSettings

__all__ = ["SetupRQ", "rq"]

class SetupRQ(object):
    """
    Class for setting up the calculation of RQs when processing data.
    
    Attributes
    ----------
    templates : list
        List of pulse templates corresponding to each channel. The pulse templates should
        be normalized.
    psds : list
        List of PSDs coresponding to each channel. Should be two-sided PSDs, with units of A^2/Hz.
    fs : float
        The digitization rate of the data in Hz.
    summed_template : ndarray
        The pulse template for all of the channels summed together to be used when calculating
        RQs. Should be normalized to have a maximum height of 1. If not set, then the RQs for 
        the sum of the channels will not be calculated.
    summed_psd : ndarray
        The PSD corresponding to all of the channels summed together to be used when calculating
        RQs. Should be a two-sided PSD, with units of A^2/Hz. If not set, then the RQs for 
        the sum of the channels will not be calculated.
    trigger : float, NoneType, optional
        The index corresponding to which channel is the trigger channel in the list of templates
        and psds. If left as None, then no channel is assumed to be the trigger channel.
    calcchans : bool
        Boolean flag for whether or not to calculate the RQs for each of the individual 
        channels.
    calcsum : bool
        Boolean flag for whether or not calculate the RQs for the sum of the channels.
        Requires summed_template and summed_psd to be set when initializing the SetupRQ
        object.
    indstart : int, NoneType
        The index at we should truncate the beginning of the traces up to when calculating RQs.
    indstop : int, NoneType
        The index at we should truncate the end of the traces up to when calculating RQs.
    nchan : int
        The number of channels to be processed.
    do_ofamp_nodelay : list of bool
        Boolean flag for whether or not to do the optimum filter fit with no time
        shifting. Each value in the list specifies this attribute for each channel.
    do_ofamp_nodelay_smooth : list of bool
        Boolean flag for whether or not the optimum filter fit with no time
        shifting should be calculated with a smoothed PSD. Useful in the case 
        where the PSD for a channel has large spike(s) in order to suppress echoes 
        elsewhere in the trace. Each value in the list specifies this attribute for each channel.
    ofamp_nodelay_lowfreqchi2 : bool
        Boolean flag for whether or not to calculate the low frequency chi-squared for 
        the optimum filter fit with no time shifting.
    do_ofamp_unconstrained : list of bool
        Boolean flag for whether or not to do the optimum filter fit with unconstrained time
        shifting. Each value in the list specifies this attribute for each channel.
    do_ofamp_unconstrained_smooth : list of bool
        Boolean flag for whether or not the optimum filter fit with unconstrained time
        shifting should be calculated with a smoothed PSD. Useful in the case 
        where the PSD for a channel has large spike(s) in order to suppress echoes 
        elsewhere in the trace. Each value in the list specifies this attribute for each channel.
    ofamp_unconstrained_lowfreqchi2 : bool
        Boolean flag for whether or not to calculate the low frequency chi-squared for 
        the optimum filter fit with unconstrained time shifting.
    ofamp_unconstrained_pulse_constraint : list of int
        Sets a constraint on the direction of the fitted pulse. If 0, then no constraint on the 
        pulse direction is set. If 1, then a positive pulse constraint is set for all fits. 
        If -1, then a negative pulse constraint is set for all fits. If any other value, then
        an ValueError will be raised. Each value in the list specifies this attribute for each channel.
    do_ofamp_constrained : list of bool
        Boolean flag for whether or not to do the optimum filter fit with constrained time
        shifting. Each value in the list specifies this attribute for each channel.
    do_ofamp_constrained_smooth : list of bool
        Boolean flag for whether or not the optimum filter fit with constrained time
        shifting should be calculated with a smoothed PSD. Useful in the case 
        where the PSD for a channel has large spike(s) in order to suppress echoes 
        elsewhere in the trace. Each value in the list specifies this attribute for each channel.
    ofamp_constrained_lowfreqchi2 : bool
        Boolean flag for whether or not to calculate the low frequency chi-squared for 
        the optimum filter fit with constrained time shifting.
    ofamp_constrained_nconstrain : list of int
        The length of the window (in bins), centered on the middle of the trace, to constrain 
        the possible time shift values to when doing the optimum filter fit with constrained time shifting. 
        Each value in the list specifies this attribute for each channel.
    ofamp_constrained_pulse_constraint : list of int
        Sets a constraint on the direction of the fitted pulse. If 0, then no constraint on the 
        pulse direction is set. If 1, then a positive pulse constraint is set for all fits. 
        If -1, then a negative pulse constraint is set for all fits. If any other value, then
        an ValueError will be raised. Each value in the list specifies this attribute for each channel.
    do_ofamp_pileup : list of bool
        Boolean flag for whether or not to do the pileup optimum filter fit. Each value in the list specifies 
        this attribute for each channel.
    do_ofamp_pileup_smooth : list of bool
        Boolean flag for whether or not the pileup optimum filter fit should be calculated 
        with a smoothed PSD. Useful in the case where the PSD for a channel has large spike(s) 
        in order to suppress echoes elsewhere in the trace. Each value in the list specifies this
        attribute for each channel.
    ofamp_pileup_nconstrain : list of int
        The length of the window (in bins), centered on the middle of the trace, outside of which to 
        constrain the possible time shift values to when searching for a pileup pulse using ofamp_pileup. Each 
        value in the list specifies this attribute for each channel.
    ofamp_pileup_pulse_constraint : list of int
        Sets a constraint on the direction of the fitted pulse. If 0, then no constraint on the 
        pulse direction is set. If 1, then a positive pulse constraint is set for all fits. 
        If -1, then a negative pulse constraint is set for all fits. If any other value, then
        an ValueError will be raised. Each value in the list specifies this attribute for each channel.
    do_chi2_nopulse : list of bool
        Boolean flag for whether or not to calculate the chi^2 for no pulse. Each value in the list specifies
        this attribute for each channel.
    do_chi2_nopulse_smooth : list of bool
        Boolean flag for whether or not the chi^2 for no pulse should be calculated 
        with a smoothed PSD. Useful in the case where the PSD for a channel has large 
        spike(s) in order to suppress echoes elsewhere in the trace. Each value in the list specifies this 
        attribute for each channel.
    do_chi2_lowfreq : list of bool
        Boolean flag for whether or not to calculate the low frequency chi^2 for any of the fits. Each value 
        in the list specifies this attribute for each channel.
    chi2_lowfreq_fcutoff : list of int
        The frequency cutoff for the calculation of the low frequency chi^2, units of Hz. Each value in the 
        list specifies this attribute for each channel.
    do_ofamp_baseline : list of bool
        Boolean flag for whether or not to do the optimum filter fit with fixed baseline. Each value in 
        the list specifies this attribute for each channel.
    do_ofamp_baseline_smooth : list of bool
        Boolean flag for whether or not the optimum filter fit with fixed baseline
        should be calculated with a smoothed PSD. Useful in the case where the PSD for a 
        channel has large spike(s) in order to suppress echoes elsewhere in the trace. Each value in the 
        list specifies this attribute for each channel.
    ofamp_baseline_nconstrain : list of int
        The length of the window (in bins), centered on the middle of the trace, to constrain 
        the possible time shift values to when doing the optimum filter fit with fixed baseline. Each 
        value in the list specifies this attribute for each channel.
    ofamp_baseline_pulse_constraint : list of int
        Sets a constraint on the direction of the fitted pulse. If 0, then no constraint on the 
        pulse direction is set. If 1, then a positive pulse constraint is set for all fits. 
        If -1, then a negative pulse constraint is set for all fits. If any other value, then
        an ValueError will be raised. Each value in the list specifies this attribute for each channel.
    do_baseline : list of bool
        Boolean flag for whether or not to calculate the DC baseline for each trace. Each value in the 
        list specifies this attribute for each channel.
    baseline_indbasepre : list of int
        The number of indices up to which a trace should be averaged to determine the baseline. Each value
        in the list specifies this attribute for each channel.
    do_integral : list of bool
        Boolean flag for whether or not to calculate the baseline-subtracted integral of each trace. Each value
        in the list specifies this attribute for each channel.
    indstart_integral : list of int
        The index at which the integral should start being calculated from in order to reduce noise by 
        truncating the beginning of the trace. Default is 1/3 of the trace length. Each value in the 
        list specifies this attribute for each channel.
    indstop_integral : list of int
        The index at which the integral should be calculated up to in order to reduce noise by 
        truncating the rest of the trace. Default is 2/3 of the trace length. Each value in the list
        specifies this attribute for each channel.
    do_energy_absorbed : list of bool
        Boolean flag for whether or not to calculate the energy absorbed for each trace. Each value
        in the list specifies this attribute for each channel.
    ioffset : list of float
        The offset in the measured TES current, units of Amps. Each value in the list specifies this attribute
        for each channel.
    qetbias : list of float
        Applied QET bias current, units of Amps. Each value in the list specifies this attribute for each
        channel.
    rload : list of float
        Load resistance of TES circuit (defined as sum of parasitic resistance and shunt 
        resistance, i.e. rp+rsh), units of Ohms. Each value in the list specifies this attribute for each
        channel.
    rsh : list of float
        Shunt resistance for TES circuit, units of Ohms. Each value in the list specifies this attribute for
        each channel.
    indstart_energy_absorbed : list of int
        The index at which the integral should start being calculated from in order to reduce noise by 
        truncating the beginning of the trace. Default is one-third of the trace length. Each value in the 
        list specifies this attribute for each channel.
    indstop_energy_absorbed : list of int
        The index at which the integral should be calculated up to in order to reduce noise by 
        truncating the rest of the trace. Default is two-thirds of the trace length. Each value in the list
        specifies this attribute for each channel.
    do_ofamp_shifted : list of bool
        Boolean flag for whether or not the shifted optimum filter fit should be calculated for
        the non-trigger channels. If set to True, then self.trigger must have been set to a value. Each 
        value in the list specifies this attribute for each channel.
    do_ofamp_shifted_smooth : list of bool
        Boolean flag for whether or not the shifted optimum filter fit should be calculated 
        with a smoothed PSD. Useful in the case where the PSD for a channel has large spike(s) 
        in order to suppress echoes elsewhere in the trace. Each value in the list specifies this
        attribute for each channel.
    shifted_fit : str
        String specifying which fit that the time shift should be pulled from if the shifted
        optimum filter fit will be calculated. Should be "nodelay", "constrained", or "unconstrained",
        referring the the no delay OF, constrained OF, and unconstrained OF, respectively. Default
        is "constrained".
    t0_shifted : ndarray
        Attribute used to save the times to shift the non-trigger channels. Only used if `do_ofamp_shifted`
        is True.
    do_ofnonlin : list of bool
        Boolean flag for whether or not the nonlinear optimum filter fit with floating rise and fall time 
        should be calculated. Default is False. Each value in the list specifies this attribute for each channel.
    ofnonlin_positive_pulses : list of bool
        If True, then the pulses are assumed to be in the positive direction. If False, then the 
        pulses are assumed to be in the negative direction. Default is True. Each value in the list specifies
        this attribute for each channel.
    do_optimumfilters : list of bool
        Boolean flag for whether or not any of the optimum filters will be calculated. If only
        calculating non-OF-related RQs, then this will be False, and processing time will not
        be spent on initializing the OF. Each value in the list specifies this attribute for each channel.
    do_optimumfilters_smooth : list of bool
        Boolean flag for whether or not any of the smoothed-PDS optimum filters will be calculated. 
        If only calculating non-OF-related RQs, then this will be False, and processing time will not
        be spent on initializing the OF. Each value in the list specifies this attribute for each channel.
    
    """
    
    def __init__(self, templates, psds, fs, summed_template=None, summed_psd=None, trigger=None,
                 indstart=None, indstop=None):
        """
        Initialization of the SetupRQ class.
        
        Parameters
        ----------
        templates : list
            List of pulse templates corresponding to each channel. The pulse templates should
            be normalized to have a maximum height of 1.
        psds : list
            List of PSDs coresponding to each channel. Should be two-sided PSDs, with units of A^2/Hz.
        fs : float
            The digitization rate of the data in Hz.
        summed_template : ndarray, optional
            The pulse template for all of the channels summed together to be used when calculating
            RQs. Should be normalized to have a maximum height of 1. If not set, then the RQs for 
            the sum of the channels will not be calculated.
        summed_psd : ndarray, optional
            The PSD corresponding to all of the channels summed together to be used when calculating
            RQs. Should be a two-sided PSD, with units of A^2/Hz. If not set, then the RQs for 
            the sum of the channels will not be calculated.
        trigger : float, NoneType, optional
            The index corresponding to which channel is the trigger channel in the list of templates
            and psds. If left as None, then no channel is assumed to be the trigger channel.
        indstart : int, NoneType, optional
            The index at we should truncate the beginning of the traces up to when calculating RQs.
            If left as None, then we do not truncate the beginning of the trace. See `indstop`.
        indstop : int, NoneType, optional
            The index at we should truncate the end of the traces up to when calculating RQs. If left as 
            None, then we do not truncate the end of the trace. See `indstart`.
        
        """
        
        if not isinstance(templates, list):
            templates = [templates]
            
        if not isinstance(psds, list):
            psds = [psds]
            
        if len(templates) != len(psds):
            raise ValueError("Different numbers of templates and psds were inputted")
        
        if len(templates[0]) != len(psds[0]):
            raise ValueError("templates and psds should have the same length")
        
        self.templates = templates
        self.psds = psds
        self.fs = fs
        self.nchan = len(templates)
        
        self.indstart = indstart
        self.indstop = indstop
        
        if self.indstart is not None and self.indstop is not None and (self.indstop - self.indstart != len(self.templates[0])):
            raise ValueError("The indices specified indstart and indstop will result in each "+\
                             "truncated trace having a different length than their corresponding "+\
                             "psd and template. Make sure indstart-indstop = the length of the "+\
                             "template/psd")
        
        self.summed_template = summed_template
        self.summed_psd = summed_psd
        
        if trigger is None or trigger in list(range(self.nchan)):
            self.trigger = trigger
        else:
            raise ValueError("trigger must be either None, or an integer"+\
                             f" from zero to the number of channels - 1 ({self.nchan-1})")
            
        self.calcchans=True
        
        if summed_template is None or summed_psd is None:
            self.calcsum=False
        else:
            self.calcsum=True
        
        self.do_ofamp_nodelay = [True]*self.nchan
        self.do_ofamp_nodelay_smooth = [False]*self.nchan
        self.ofamp_nodelay_lowfreqchi2 = False
        
        self.do_ofamp_unconstrained = [True]*self.nchan
        self.do_ofamp_unconstrained_smooth = [False]*self.nchan
        self.ofamp_unconstrained_lowfreqchi2 = False
        self.ofamp_unconstrained_pulse_constraint = [0]*self.nchan
        
        self.do_ofamp_constrained = [True]*self.nchan
        self.do_ofamp_constrained_smooth = [False]*self.nchan
        self.ofamp_constrained_lowfreqchi2 = True
        self.ofamp_constrained_nconstrain = [80]*self.nchan
        self.ofamp_constrained_pulse_constraint = [0]*self.nchan
        
        self.do_ofamp_pileup = [True]*self.nchan
        self.do_ofamp_pileup_smooth = [False]*self.nchan
        self.ofamp_pileup_nconstrain = [80]*self.nchan
        self.ofamp_pileup_pulse_constraint = [0]*self.nchan
        
        self.do_chi2_nopulse = [True]*self.nchan
        self.do_chi2_nopulse_smooth = [False]*self.nchan
        
        self.do_chi2_lowfreq = [True]*self.nchan
        self.chi2_lowfreq_fcutoff = [10000]*self.nchan
        
        self.do_ofamp_baseline = [False]*self.nchan
        self.do_ofamp_baseline_smooth = [False]*self.nchan
        self.ofamp_baseline_nconstrain = [80]*self.nchan
        self.ofamp_baseline_pulse_constraint = [0]*self.nchan
        
        self.do_baseline = [True]*self.nchan
        self.baseline_indbasepre = [len(self.templates[0])//3]*self.nchan
        
        self.do_integral = [True]*self.nchan
        self.indstart_integral = [len(self.templates[0])//3]*self.nchan
        self.indstop_integral = [2*len(self.templates[0])//3]*self.nchan
        
        self.do_energy_absorbed = [False]*self.nchan
        
        self.ioffset = None
        self.qetbias = None
        self.rload = None
        self.rsh = None
        self.indstart_energy_absorbed = [len(self.templates[0])//3]*self.nchan
        self.indstop_energy_absorbed = [2*len(self.templates[0])//3]*self.nchan
        
        self.do_ofamp_shifted = [False]*self.nchan
        self.do_ofamp_shifted_smooth = [False]*self.nchan
        self.which_fit = "constrained"
        self.t0_shifted = None
        
        self.do_maxmin = [True]*self.nchan
        self.use_min = [False]*self.nchan
        self.indstart_maxmin = [0]*self.nchan
        self.indstop_maxmin = [len(self.templates[0])]*self.nchan
        
        self.do_ofnonlin = [False]*self.nchan
        self.ofnonlin_positive_pulses = [True]*self.nchan
        
        self.do_optimumfilters = [True]*self.nchan
        self.do_optimumfilters_smooth = [False]*self.nchan
        
    def _check_of(self):
        """
        Helper function for checking if any of the optimum filters are going to be calculated.
        
        """
        
        do_optimumfilters = [False]*self.nchan
        
        if any(self.do_ofamp_nodelay):
            do_optimumfilters = [ii or jj for ii, jj in zip(do_optimumfilters, self.do_ofamp_nodelay)]
        if any(self.do_ofamp_unconstrained):
            do_optimumfilters = [ii or jj for ii, jj in zip(do_optimumfilters, self.do_ofamp_unconstrained)]
        if any(self.do_ofamp_constrained):
            do_optimumfilters = [ii or jj for ii, jj in zip(do_optimumfilters, self.do_ofamp_constrained)]
        if any(self.do_ofamp_pileup):
            do_optimumfilters = [ii or jj for ii, jj in zip(do_optimumfilters, self.do_ofamp_pileup)]
        if any(self.do_chi2_nopulse):
            do_optimumfilters = [ii or jj for ii, jj in zip(do_optimumfilters, self.do_chi2_nopulse)]
        if any(self.do_chi2_lowfreq):
            do_optimumfilters = [ii or jj for ii, jj in zip(do_optimumfilters, self.do_chi2_lowfreq)]
        if any(self.do_ofamp_baseline):
            do_optimumfilters = [ii or jj for ii, jj in zip(do_optimumfilters, self.do_ofamp_baseline)]
        
        self.do_optimumfilters = do_optimumfilters
        
        do_optimumfilters_smooth = [False]*self.nchan
            
        if any(self.do_ofamp_nodelay_smooth):
            do_optimumfilters_smooth = [ii or jj for ii, jj in zip(do_optimumfilters_smooth, self.do_ofamp_nodelay_smooth)]
        if any(self.do_ofamp_unconstrained_smooth):
            do_optimumfilters_smooth = [ii or jj for ii, jj in zip(do_optimumfilters_smooth, self.do_ofamp_unconstrained_smooth)]
        if any(self.do_ofamp_constrained_smooth):
            do_optimumfilters_smooth = [ii or jj for ii, jj in zip(do_optimumfilters_smooth, self.do_ofamp_constrained_smooth)]
        if any(self.do_ofamp_pileup_smooth):
            do_optimumfilters_smooth = [ii or jj for ii, jj in zip(do_optimumfilters_smooth, self.do_ofamp_pileup_smooth)]
        if any(self.do_chi2_nopulse_smooth):
            do_optimumfilters_smooth = [ii or jj for ii, jj in zip(do_optimumfilters_smooth, self.do_chi2_nopulse_smooth)]
        if any(self.do_ofamp_baseline_smooth):
            do_optimumfilters_smooth = [ii or jj for ii, jj in zip(do_optimumfilters_smooth, self.do_ofamp_baseline_smooth)]
            
        self.do_optimumfilters_smooth = do_optimumfilters_smooth
        
    def _check_arg_length(self, **kwargs):
        """
        Helper function for checking if the inputted `**kwargs` are list type and if they have same
        length as the number of channels.
        
        Parameters
        ----------
        **kwargs : Arbitrary keyword arguments
            The inputted keyword arguments to check list type and length.
        
        Returns
        -------
        out : list
            The list of values for each inputted kwarg, in the
            order that they were inputted.
            
        Raises
        ------
        ValueError
            A ValueError is raised if the length of one of the `**kwargs` is not equal
            to `self.chan`.
        ValueError
            A ValueError is raised if the `**kwargs` pulse_direction_constraint is inputted and 
            one of the values is not set to 1, 0, or -1.
        
        """
    
        for key, value in kwargs.items():
            if np.isscalar(value):
                kwargs[key] = [value]*self.nchan
                
            if key == "pulse_direction_constraint" and not all(x in [-1, 0, 1] for x in kwargs[key]):
                raise ValueError(f"{key} should be set to 0, 1, or -1")

            if len(kwargs[key])!=self.nchan:
                raise ValueError(f"The length of {key} is not equal to the number of channels")
        
        out = list(kwargs.values())
        
        if len(out)==1:
            out = out[0]
        
        return out

    def adjust_calc(self, lgcchans=True, lgcsum=True):
        """
        Method for adjusting the calculation of RQs for each individual channel and the sum
        of the channels.
        
        Parameters
        ----------
        lgcchans : bool, optional
            Boolean flag for whether or not to calculate the RQs for each of the individual 
            channels. Default is True.
        lgcsum : bool, optional
            Boolean flag for whether or not calculate the RQs for the sum of the channels.
            Requires summed_template and summed_psd to be set when initializing the SetupRQ
            object. Default is True.
            
        Raises
        ------
        ValueError
            A ValueError is raised if lgcsum is set to True when the SetupRQ Object was not
            initialized with summed_template or summed_psd.
        
        """
        
        self.calcchans = lgcchans
        
        if (self.summed_template is None or self.summed_psd is None) and lgcsum:
            raise ValueError("SetupRQ was not initialized with summed_template or summed_psd, cannot calculate the summed RQs")
        else:
            self.calcsum = lgcsum
        
    def adjust_ofamp_nodelay(self, lgcrun=True, lgcrun_smooth=False, calc_lowfreqchi2=False):
        """
        Method for adjusting the calculation of the optimum filter fit with no time
        shifting.
        
        Parameters
        ----------
        lgcrun : bool, list of bool, optional
            Boolean flag for whether or not the optimum filter fit with no time
            shifting should be calculated.
        lgcrun_smooth : bool, list of bool, optional
            Boolean flag for whether or not the optimum filter fit with no time
            shifting should be calculated with a smoothed PSD. Useful in the case 
            where the PSD for a channel has large spike(s) in order to suppress echoes 
            elsewhere in the trace.
        calc_lowfreqchi2 : bool, optional
            Boolean flag for whether or not the low frequency chi^2 of this fit 
            should be calculated. The low frequency chi^2 calculation should be adjusted
            using the adjust_chi2_lowfreq method.
        
        """
        
        lgcrun, lgcrun_smooth = self._check_arg_length(lgcrun=lgcrun, lgcrun_smooth=lgcrun_smooth)
        
        self.do_ofamp_nodelay = lgcrun
        self.do_ofamp_nodelay_smooth = lgcrun_smooth
        self.ofamp_nodelay_lowfreqchi2 = calc_lowfreqchi2
        
        self._check_of()
        
    def adjust_ofamp_unconstrained(self, lgcrun=True, lgcrun_smooth=False, calc_lowfreqchi2=False, 
                                   pulse_direction_constraint=0):
        """
        Method for adjusting the calculation of the optimum filter fit with unconstrained 
        time shifting.
        
        Parameters
        ----------
        lgcrun : bool, list of bool, optional
            Boolean flag for whether or not the optimum filter fit with unconstrained 
            time shifting should be calculated.
        lgcrun_smooth : bool, list of bool, optional
            Boolean flag for whether or not the optimum filter fit with unconstrained time
            shifting should be calculated with a smoothed PSD. Useful in the case 
            where the PSD for a channel has large spike(s) in order to suppress echoes 
            elsewhere in the trace.
        calc_lowfreqchi2 : bool, optional
            Boolean flag for whether or not the low frequency chi^2 of this fit 
            should be calculated. The low frequency chi^2 calculation should be adjusted
            using the adjust_chi2_lowfreq method. Default is False.
        pulse_direction_constraint : int, list of int, optional
            Sets a constraint on the direction of the fitted pulse. If 0, then no constraint on the 
            pulse direction is set. If 1, then a positive pulse constraint is set for all fits. 
            If -1, then a negative pulse constraint is set for all fits. If any other value, then
            an ValueError will be raised.
        
        """
        
        lgcrun, lgcrun_smooth, pulse_direction_constraint = self._check_arg_length(lgcrun=lgcrun,
                                                                                   lgcrun_smooth=lgcrun_smooth,
                                                            pulse_direction_constraint=pulse_direction_constraint)
        
        self.do_ofamp_unconstrained = lgcrun
        self.do_ofamp_unconstrained_smooth = lgcrun_smooth
        self.ofamp_unconstrained_lowfreqchi2 = calc_lowfreqchi2
        
        self.ofamp_unconstrained_pulse_constraint = pulse_direction_constraint
        
        self._check_of()
        
    def adjust_ofamp_constrained(self, lgcrun=True, lgcrun_smooth=False, calc_lowfreqchi2=True, 
                                 nconstrain=80, pulse_direction_constraint=0):
        """
        Method for adjusting the calculation of the optimum filter fit with constrained 
        time shifting.
        
        Parameters
        ----------
        lgcrun : bool, list of bool, optional
            Boolean flag for whether or not the optimum filter fit with constrained 
            time shifting should be calculated.
        lgcrun_smooth : bool, list of bool, optional
            Boolean flag for whether or not the optimum filter fit with constrained time
            shifting should be calculated with a smoothed PSD. Useful in the case 
            where the PSD for a channel has large spike(s) in order to suppress echoes 
            elsewhere in the trace.
        calc_lowfreqchi2 : bool, optional
            Boolean flag for whether or not the low frequency chi^2 of this fit 
            should be calculated. The low frequency chi^2 calculation should be adjusted
            using the adjust_chi2_lowfreq method. Default is True.
        nconstrain : int, list of int, optional
            The length of the window (in bins), centered on the middle of the trace, 
            to constrain the possible time shift values to when doing the optimum filter 
            fit with constrained time shifting. Can be set to a list of values, if the 
            constrain window should be different for each channel. The length of the list should
            be the same length as the number of channels.
        pulse_direction_constraint : int, list of int, optional
            Sets a constraint on the direction of the fitted pulse. If 0, then no constraint on the 
            pulse direction is set. If 1, then a positive pulse constraint is set for all fits. 
            If -1, then a negative pulse constraint is set for all fits. If any other value, then
            an ValueError will be raised.
        
        """
        
        lgcrun, lgcrun_smooth, nconstrain, pulse_direction_constraint = self._check_arg_length(lgcrun=lgcrun,
                                                                                               lgcrun_smooth=lgcrun_smooth,
                                                                                               nconstrain=nconstrain, 
                                                                        pulse_direction_constraint=pulse_direction_constraint)
        
        self.do_ofamp_constrained = lgcrun
        self.do_ofamp_constrained_smooth = lgcrun_smooth
        self.ofamp_constrained_lowfreqchi2 = calc_lowfreqchi2
        self.ofamp_constrained_nconstrain = nconstrain
        
        self.ofamp_constrained_pulse_constraint = pulse_direction_constraint
        
        self._check_of()
        
    def adjust_ofamp_baseline(self, lgcrun=True, lgcrun_smooth=False, 
                              nconstrain=80, pulse_direction_constraint=0):
        """
        Method for adjusting the calculation of the optimum filter fit with fixed 
        baseline.
        
        Parameters
        ----------
        lgcrun : bool, list of bool, optional
            Boolean flag for whether or not the optimum filter fit with fixed baseline 
            should be calculated.
        lgcrun_smooth : bool, list of bool, optional
            Boolean flag for whether or not the optimum filter fit with fixed baseline
            should be calculated with a smoothed PSD. Useful in the case where the PSD for a 
            channel has large spike(s) in order to suppress echoes elsewhere in the trace.
        nconstrain : int, list of int, optional
            The length of the window (in bins), centered on the middle of the trace, 
            to constrain the possible time shift values to when doing the optimum filter 
            fit with fixed baseline. Can be set to a list of values, if the 
            constrain window should be different for each channel. The length of the list should
            be the same length as the number of channels.
        pulse_direction_constraint : int, list of int, optional
            Sets a constraint on the direction of the fitted pulse. If 0, then no constraint on the 
            pulse direction is set. If 1, then a positive pulse constraint is set for all fits. 
            If -1, then a negative pulse constraint is set for all fits. If any other value, then
            an ValueError will be raised.
        
        """
        
        lgcrun, lgcrun_smooth, nconstrain, pulse_direction_constraint = self._check_arg_length(lgcrun=lgcrun,
                                                                                               lgcrun_smooth=lgcrun_smooth,
                                                                                               nconstrain=nconstrain, 
                                                                        pulse_direction_constraint=pulse_direction_constraint)
        
        self.do_ofamp_baseline = lgcrun
        self.do_ofamp_baseline_smooth = lgcrun_smooth
        self.ofamp_baseline_nconstrain = nconstrain
        
        self.ofamp_baseline_pulse_constraint = pulse_direction_constraint
        
        self._check_of()
        
    def adjust_ofamp_pileup(self, lgcrun=True, lgcrun_smooth=False, 
                            nconstrain=80, pulse_direction_constraint=0):
        """
        Method for adjusting the calculation of the pileup optimum filter fit.
        
        Parameters
        ----------
        lgcrun : bool, list of bool, optional
            Boolean flag for whether or not the pileup optimum filter fit should be calculated.
        lgcrun_smooth : bool, list of bool, optional
            Boolean flag for whether or not the pileup optimum filter fit should be calculated 
            with a smoothed PSD. Useful in the case where the PSD for a channel has large spike(s) 
            in order to suppress echoes elsewhere in the trace.
        nconstrain : int, list of int, optional
            The length of the window (in bins), centered on the middle of the trace, outside 
            of which to constrain the possible time shift values to when searching for a 
            pileup pulse using ofamp_pileup. Can be set to a list of values, if the constrain 
            window should be different for each channel. The length of the list should
            be the same length as the number of channels.
        pulse_direction_constraint : int, list of int, optional
            Sets a constraint on the direction of the fitted pulse. If 0, then no constraint on the 
            pulse direction is set. If 1, then a positive pulse constraint is set for all fits. 
            If -1, then a negative pulse constraint is set for all fits. If any other value, then
            an ValueError will be raised.
        
        """
        
        lgcrun, lgcrun_smooth, nconstrain, pulse_direction_constraint = self._check_arg_length(lgcrun=lgcrun,
                                                                                               lgcrun_smooth=lgcrun_smooth,
                                                                                               nconstrain=nconstrain, 
                                                                        pulse_direction_constraint=pulse_direction_constraint)
        
        self.do_ofamp_pileup = lgcrun
        self.do_ofamp_pileup_smooth = lgcrun_smooth
        self.ofamp_pileup_nconstrain = nconstrain
        
        self.ofamp_pileup_pulse_constraint = pulse_direction_constraint
        
        self._check_of()
        
    def adjust_chi2_nopulse(self, lgcrun=True, lgcrun_smooth=False):
        """
        Method for adjusting the calculation of the no pulse chi^2.
        
        Parameters
        ----------
        lgcrun : bool, list of bool, optional
            Boolean flag for whether or not to calculate the chi^2 for no pulse.
        lgcrun_smooth : bool, list of bool, optional
            Boolean flag for whether or not the chi^2 for no pulse should be calculated 
            with a smoothed PSD. Useful in the case where the PSD for a channel has large 
            spike(s) in order to suppress echoes elsewhere in the trace.
        
        """
        
        lgcrun, lgcrun_smooth = self._check_arg_length(lgcrun=lgcrun, lgcrun_smooth=lgcrun_smooth)
        
        self.do_chi2_nopulse = lgcrun
        self.do_chi2_nopulse_smooth = lgcrun_smooth
        
        self._check_of()
        
    def adjust_chi2_lowfreq(self, lgcrun=True, fcutoff=10000):
        """
        Method for adjusting the calculation of the low frequency chi^2.
        
        Parameters
        ----------
        lgcrun : bool, list of bool, optional
            Boolean flag for whether or not the low frequency chi^2 should be calculated
            for any of the optimum filter fits.
        fcutoff : float, list of float, optional
            The frequency cutoff for the calculation of the low frequency chi^2, units of Hz.
            Can be set to a list of values, if the frequency cutoff should be different for 
            each channel. The length of the list should be the same length as the number 
            of channels.
            
        """
        
        lgcrun, fcutoff = self._check_arg_length(lgcrun=lgcrun, fcutoff=fcutoff)
            
        self.do_chi2_lowfreq = lgcrun
        self.chi2_lowfreq_fcutoff = fcutoff
        
        self._check_of()
        
    def adjust_baseline(self, lgcrun=True, indbasepre=None):
        """
        Method for adjusting the calculation of the DC baseline.
        
        Parameters
        ----------
        lgcrun : bool, list of bool, optional
            Boolean flag for whether or not the DC baseline should be calculated. It is highly
            recommended to set this to true if the integral will be calculated, so that the
            baseline can be subtracted.
        indbasepre : int, list of int, optional
            The number of indices up to which a trace should be averaged to determine the baseline.
            Can be set to a list of values, if indbasepre should be different for each channel. 
            The length of the list should be the same length as the number of channels. Default
            is one-third of the trace length.
            
        """
        
        if indbasepre is None:
            indbasepre = len(self.templates[0])//3
        
        lgcrun, indbasepre = self._check_arg_length(lgcrun=lgcrun, indbasepre=indbasepre)
            
        self.do_baseline = lgcrun
        self.baseline_indbasepre = indbasepre
        
    def adjust_integral(self, lgcrun=True, indstart=None, indstop=None):
        """
        Method for adjusting the calculation of the integral.
        
        Parameters
        ----------
        lgcrun : bool, list of bool, optional
            Boolean flag for whether or not the integral should be calculated. If self.do_baseline
            is True, then the baseline is subtracted from the integral. If self.do_baseline is False,
            then the baseline is not subtracted. It is recommended that the baseline should be subtracted.
        indstart : int, list of int, optional
            The index at which the integral should start being calculated from in order to reduce noise by 
            truncating the beginning of the trace. Default is one-third of the trace length.
        indstop : int, list of int, optional
            The index at which the integral should be calculated up to in order to reduce noise by 
            truncating the rest of the trace. Default is two-thirds of the trace length.
            
        """
        
        if indstart is None:
            indstart = len(self.templates[0])//3
            
        if indstop is None:
            indstop = 2*len(self.templates[0])//3
        
        lgcrun, indstart, indstop = self._check_arg_length(lgcrun=lgcrun, indstart=indstart, indstop=indstop)
        
        self.do_integral = lgcrun
        self.indstart_integral = indstart
        self.indstop_integral = indstop
        
    def adjust_energy_absorbed(self, ioffset, qetbias, rload, rsh, lgcrun=True, indstart=None, indstop=None):
        """
        Method for calculating the energy absorbed by the TES, in the limit of infinite loop gain.
        
        Parameters
        ----------
        ioffset : float, list of float
            The offset in the measured TES current, units of Amps.
        qetbias : float, list of float
            Applied QET bias current, units of Amps.
        rload : float, list of float
            Load resistance of TES circuit (defined as sum of parasitic resistance and shunt 
            resistance, i.e. rp+rsh), units of Ohms.
        rsh : float, list of float
            Shunt resistance for TES circuit, units of Ohms.
        lgcrun : bool, list of bool, optional
            Boolean flag for whether or not the energy absorbed should be calculated. If self.do_baseline
            is True, then the baseline is subtracted from the integral. If self.do_baseline is False,
            then the baseline is not subtracted. It is recommended that the baseline should be subtracted.
        indstart : int, list of int, optional
            The index at which the integral should start being calculated from in order to reduce noise by 
            truncating the beginning of the trace. Default is one-third of the trace length.
        indstop : int, list of int, optional
            The index at which the integral should be calculated up to in order to reduce noise by 
            truncating the rest of the trace. Default is two-thirds of the trace length.
        
        """
        
        if indstart is None:
            indstart = len(self.templates[0])//3
            
        if indstop is None:
            indstop = 2*len(self.templates[0])//3
        
        lgcrun, ioffset, qetbias, rload, rsh, indstart, indstop = self._check_arg_length(lgcrun=lgcrun, 
                                                                                         ioffset=ioffset, 
                                                                                         qetbias=qetbias,
                                                                                         rload=rload,
                                                                                         rsh=rsh,
                                                                                         indstart=indstart,
                                                                                         indstop=indstop)
        
        self.do_energy_absorbed = lgcrun
        
        self.ioffset = ioffset
        self.qetbias = qetbias
        self.rload = rload
        self.rsh = rsh
        self.indstart_energy_absorbed = indstart
        self.indstop_energy_absorbed = indstop
        
    def adjust_maxmin(self, lgcrun=True, use_min=False, indstart=None, indstop=None):
        """
        Method for adjusting the calculation of the range of a trace.
        
        Parameters
        ----------
        lgcrun : bool, list of bool, optional
            Boolean flag for whether or not the maximum (or minimum) of the trace should be calculated.
        use_min : bool, list of bool, optional
            Whether or not to use the maximum or the minimum when calculating maxmin. If True, then the
            minimum is used. If False, then the maximum is used. Default is False.
        indstart : NoneType, int, list of int, optional
            The starting index for the range of values that maxmin should be calculated over. Default is
            None, which sets the starting index to the beginning of the trace.
        indstop : int, list of int, optional
            The end index for the range of values that that maxmin should be calculated over. Default is 
            None, which sets the end index to the end of the trace.
            
        """
        
        if indstart is None:
            indstart = 0
        
        if indstop is None:
            indstop = len(self.templates[0])
            
        lgcrun, use_min, indstart, indstop = self._check_arg_length(lgcrun=lgcrun, use_min=use_min, 
                                                                    indstart=indstart, indstop=indstop)
        
        self.do_maxmin = lgcrun
        self.use_min = use_min
        self.indstart_maxmin = indstart
        self.indstop_maxmin = indstop
        
    def adjust_ofamp_shifted(self, lgcrun=True, lgcrun_smooth=False, which_fit="constrained"):
        """
        Method for adjusting the calculation of the shifted optimum filter fit.
        
        Parameters
        ----------
        lgcrun : bool, optional
            Boolean flag for whether or not the shifted optimum filter fit should be calculated for
            the non-trigger channels. If set to True, then self.trigger must have been set to a value.
        lgcrun_smooth : bool, optional
            Boolean flag for whether or not the shifted optimum filter fit should be calculated 
            with a smoothed PSD. Useful in the case where the PSD for a channel has large spike(s) 
            in order to suppress echoes elsewhere in the trace.
        which_fit : str, optional
            String specifying which fit that the time shift should be pulled from if the shifted
            optimum filter fit will be calculated. Should be "nodelay", "constrained", or "unconstrained",
            referring the the no delay OF, constrained OF, and unconstrained OF, respectively. Default
            is "constrained".
            
        """
        
        lgcrun, lgcrun_smooth = self._check_arg_length(lgcrun=lgcrun, lgcrun_smooth=lgcrun_smooth)
        
        self.do_ofamp_shifted = lgcrun
        
        if self.do_ofamp_shifted:
            if which_fit not in ["constrained", "unconstrained", "nodelay"]:
                raise ValueError("which_fit should be set to 'constrained', 'unconstrained', or 'nodelay'")

            if which_fit == "constrained" and not self.do_ofamp_constrained:
                raise ValueError("which_fit was set to 'constrained', but that fit has been set to not be calculated")

            if which_fit == "unconstrained" and not self.do_ofamp_unconstrained:
                raise ValueError("which_fit was set to 'constrained', but that fit has been set to not be calculated")

            if which_fit == "nodelay" and not self.do_ofamp_nodelay:
                raise ValueError("which_fit was set to 'nodelay', but that fit has been set to not be calculated")
                
        self.do_ofamp_shifted_smooth = lgcrun_smooth
        
        if self.do_ofamp_shifted_smooth:
            if which_fit not in ["constrained", "unconstrained", "nodelay"]:
                raise ValueError("which_fit should be set to 'constrained', 'unconstrained', or 'nodelay'")

            if which_fit == "constrained" and not self.do_ofamp_constrained_smooth:
                raise ValueError("""which_fit was set to 'constrained', but that fit (using the smoothed PSD) 
                                 has been set to not be calculated""")

            if which_fit == "unconstrained" and not self.do_ofamp_unconstrained_smooth:
                raise ValueError("""which_fit was set to 'constrained', but that fit (using the smoothed PSD) 
                                 has been set to not be calculated""")

            if which_fit == "nodelay" and not self.do_ofamp_nodelay_smooth:
                raise ValueError("""which_fit was set to 'nodelay', but that fit (using the smoothed PSD) 
                                 has been set to not be calculated""")
            
        self.shifted_fit = which_fit
        
    def adjust_ofnonlin(self, lgcrun=True, positive_pulses=True):
        """
        Method for adjusting the calculation of the nonlinear optimum filter fit with rise and 
        fall time floating.
        
        Parameters
        ----------
        lgcrun : bool, list of bool, optional
            Boolean flag for whether or not the nonlinear optimum filter fit should be calculated.
        positive_pulses : bool, list of bool, optional
            If True, then the pulses are assumed to be in the positive direction. If False, then the 
            pulses are assumed to be in the negative direction. Default is True.
        
        """
        
        lgcrun, npolefit, taurise, positive_pulses = self._check_arg_length(lgcrun=lgcrun, 
                                                                            positive_pulses=positive_pulses)
        
        if any(lgcrun):
            warnings.warn("The nonlinear OF should only be run on a cluster due to the slow computation speed.")
        
        self.do_ofnonlin = lgcrun
        self.ofnonlin_positive_pulses = positive_pulses
        
def _calc_rq_single_channel(signal, template, psd, setup, readout_inds, chan, chan_num, det):
    """
    Helper function for calculating RQs for an array of traces corresponding to a single channel.
    
    Parameters
    ----------
    signal : ndarray
        Array of traces to use in calculation of RQs. Should be of shape (number of traces,
        length of trace)
    template : ndarray
        The pulse template to be used for the optimum filter (should be normalized beforehand).
    psd : ndarray
        The two-sided psd that will be used to describe the noise in the signal (in Amps^2/Hz)
    setup : SetupRQ
        A SetupRQ class object. This object defines all of the different RQs that should be calculated 
        and specifies relevant parameters.
    readout_inds : ndarray of bool
        Boolean mask that specifies which traces should be used to calculate the RQs. RQs for the 
        excluded traces are set to -999999.0. 
    chan : str
        Name of the channel that is being processed.
    chan_num : int
        The corresponding number for the channel being processed.
    det : str
        Name of the detector corresponding to the channel that is being processed.
    
    Returns
    -------
    rq_dict : dict
        A dictionary containing all of the RQs that were calculated (as specified by the setup object).
    
    """
    
    rq_dict = {}
    
    fs = setup.fs
    
    if setup.do_baseline[chan_num]:
        baseline = np.mean(signal[:, :setup.baseline_indbasepre[chan_num]], axis=-1)
        rq_dict[f'baseline_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'baseline_{chan}{det}'][readout_inds] = baseline
    
    if setup.do_integral[chan_num]:
        if setup.do_baseline[chan_num]:
            integral = np.trapz(signal[:, setup.indstart_integral[chan_num]:setup.indstop_integral[chan_num]]\
                                - baseline[:, np.newaxis], axis=-1)/fs
        else:
            integral = np.trapz(signal[:, setup.indstart_integral[chan_num]:setup.indstop_integral[chan_num]], axis=-1)/fs
        rq_dict[f'integral_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'integral_{chan}{det}'][readout_inds] = integral
        
    if setup.do_energy_absorbed[chan_num]:
        if setup.do_baseline[chan_num]:
            energy_absorbed = qp.utils.energy_absorbed(
                              signal[:, setup.indstart_energy_absorbed[chan_num]:setup.indstop_energy_absorbed[chan_num]],
                                                       setup.ioffset[chan_num], 
                                                       setup.qetbias[chan_num], 
                                                       setup.rload[chan_num], 
                                                       setup.rsh[chan_num],
                                                       fs=fs,
                                                       baseline=baseline[:, np.newaxis])
        else:
            energy_absorbed = qp.utils.energy_absorbed(signal[:, :setup.indstop_energy_absorbed[chan_num]],
                                                       setup.ioffset[chan_num], 
                                                       setup.qetbias[chan_num], 
                                                       setup.rload[chan_num], 
                                                       setup.rsh[chan_num],
                                                       indbasepre=setup.indstart_energy_absorbed[chan_num],
                                                       fs=fs)
        rq_dict[f'energy_absorbed_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'energy_absorbed_{chan}{det}'][readout_inds] = energy_absorbed
        
    if setup.do_maxmin[chan_num]:
        if setup.use_min[chan_num]:
            maxmin = np.amin(signal[:, setup.indstart_maxmin[chan_num]:setup.indstop_maxmin[chan_num]], axis=-1)
        else:
            maxmin = np.amax(signal[:, setup.indstart_maxmin[chan_num]:setup.indstop_maxmin[chan_num]], axis=-1)
        rq_dict[f'maxmin_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'maxmin_{chan}{det}'][readout_inds] = maxmin
    
    # initialize variables for the various OFs
    if setup.do_chi2_nopulse[chan_num]:
        chi0 = np.zeros(len(signal))
    if setup.do_chi2_nopulse_smooth[chan_num]:
        chi0_smooth = np.zeros(len(signal))
    
    if setup.do_ofamp_nodelay[chan_num]:
        amp_nodelay = np.zeros(len(signal))
        chi2_nodelay = np.zeros(len(signal))
    if setup.do_ofamp_nodelay_smooth[chan_num]:
        amp_nodelay_smooth = np.zeros(len(signal))
        chi2_nodelay_smooth = np.zeros(len(signal))
    
    if setup.do_ofamp_unconstrained[chan_num]:
        amp_noconstrain = np.zeros(len(signal))
        t0_noconstrain = np.zeros(len(signal))
        chi2_noconstrain = np.zeros(len(signal))
        if setup.ofamp_unconstrained_pulse_constraint[chan_num]!=0:
            amp_noconstrain_pcon = np.zeros(len(signal))
            t0_noconstrain_pcon  = np.zeros(len(signal))
            chi2_noconstrain_pcon  = np.zeros(len(signal))
    if setup.do_ofamp_unconstrained_smooth[chan_num]:
        amp_noconstrain_smooth = np.zeros(len(signal))
        t0_noconstrain_smooth = np.zeros(len(signal))
        chi2_noconstrain_smooth = np.zeros(len(signal))
    
    if setup.do_ofamp_constrained[chan_num]:
        amp_constrain = np.zeros(len(signal))
        t0_constrain = np.zeros(len(signal))
        chi2_constrain = np.zeros(len(signal))
        if setup.ofamp_constrained_pulse_constraint[chan_num]!=0:
            amp_constrain_pcon = np.zeros(len(signal))
            t0_constrain_pcon = np.zeros(len(signal))
            chi2_constrain_pcon = np.zeros(len(signal))
    if setup.do_ofamp_constrained_smooth[chan_num]:
        amp_constrain_smooth = np.zeros(len(signal))
        t0_constrain_smooth = np.zeros(len(signal))
        chi2_constrain_smooth = np.zeros(len(signal))
    
    if setup.do_ofamp_pileup[chan_num]:
        amp_pileup = np.zeros(len(signal))
        t0_pileup = np.zeros(len(signal))
        chi2_pileup = np.zeros(len(signal))
        if setup.ofamp_pileup_pulse_constraint[chan_num]!=0:
            amp_pileup_pcon = np.zeros(len(signal))
            t0_pileup_pcon = np.zeros(len(signal))
            chi2_pileup_pcon = np.zeros(len(signal))
    if setup.do_ofamp_pileup_smooth[chan_num]:
        amp_pileup_smooth = np.zeros(len(signal))
        t0_pileup_smooth = np.zeros(len(signal))
        chi2_pileup_smooth = np.zeros(len(signal))
    
    if setup.do_ofamp_baseline[chan_num]:
        amp_baseline = np.zeros(len(signal))
        t0_baseline = np.zeros(len(signal))
        chi2_baseline = np.zeros(len(signal))
        if setup.ofamp_baseline_pulse_constraint[chan_num]!=0:
            amp_baseline_pcon = np.zeros(len(signal))
            t0_baseline_pcon = np.zeros(len(signal))
            chi2_baseline_pcon = np.zeros(len(signal))
    if setup.do_ofamp_baseline_smooth[chan_num]:
        amp_baseline_smooth = np.zeros(len(signal))
        t0_baseline_smooth = np.zeros(len(signal))
        chi2_baseline_smooth = np.zeros(len(signal))
        
    if setup.do_chi2_lowfreq[chan_num]:
        if setup.ofamp_nodelay_lowfreqchi2:
            chi2low_nodelay = np.zeros(len(signal))
        if setup.ofamp_unconstrained_lowfreqchi2:
            chi2low_unconstrain = np.zeros(len(signal))
            if setup.ofamp_unconstrained_pulse_constraint[chan_num]!=0:
                chi2low_unconstrain_pcon = np.zeros(len(signal))
        if setup.ofamp_constrained_lowfreqchi2:
            chi2low_constrain = np.zeros(len(signal))
            if setup.ofamp_constrained_pulse_constraint[chan_num]!=0:
                chi2low_constrain_pcon = np.zeros(len(signal))
    
    if setup.do_ofnonlin[chan_num]:
        amp_nonlin = np.zeros(len(signal))
        amp_nonlin_err = np.zeros(len(signal))
        taurise_nonlin = np.zeros(len(signal))
        taurise_nonlin_err = np.zeros(len(signal))
        taufall_nonlin = np.zeros(len(signal))
        taufall_nonlin_err = np.zeros(len(signal))
        t0_nonlin = np.zeros(len(signal))
        t0_nonlin_err = np.zeros(len(signal))
        chi2_nonlin = np.zeros(len(signal))
    
    # run the OF class for each trace
    if setup.do_optimumfilters[chan_num]:
        OF = qp.OptimumFilter(signal[0], template, psd, fs)
    if setup.do_optimumfilters_smooth[chan_num]:
        psd_smooth = qp.smooth_psd(psd)
        OF_smooth = qp.OptimumFilter(signal[0], template, psd_smooth, fs)
    
    for jj, s in enumerate(signal):
        if jj!=0:
            if setup.do_optimumfilters[chan_num]:
                OF.update_signal(s)
            if setup.do_optimumfilters_smooth[chan_num]:
                OF_smooth.update_signal(s)
        
        if setup.do_chi2_nopulse[chan_num]:
            chi0[jj] = OF.chi2_nopulse()
            
        if setup.do_chi2_nopulse_smooth[chan_num]:
            chi0_smooth[jj] = OF_smooth.chi2_nopulse()
        
        if setup.do_ofamp_nodelay[chan_num]:
            amp_nodelay[jj], chi2_nodelay[jj] = OF.ofamp_nodelay()
            
        if setup.do_ofamp_nodelay_smooth[chan_num]:
            amp_nodelay_smooth[jj], chi2_nodelay_smooth[jj] = OF_smooth.ofamp_nodelay()
        
        if setup.ofamp_nodelay_lowfreqchi2 and setup.do_chi2_lowfreq[chan_num]:
            chi2low_nodelay[jj] = OF.chi2_lowfreq(amp_nodelay[jj], 0, 
                                          fcutoff=setup.chi2_lowfreq_fcutoff[chan_num])
        
        if setup.do_ofamp_unconstrained[chan_num]:
            amp_noconstrain[jj], t0_noconstrain[jj], chi2_noconstrain[jj] = OF.ofamp_withdelay()
            if setup.ofamp_unconstrained_pulse_constraint[chan_num]!=0:
                amp_noconstrain_pcon[jj], t0_noconstrain_pcon[jj], chi2_noconstrain_pcon[jj] = OF.ofamp_withdelay(
                                        pulse_direction_constraint=setup.ofamp_unconstrained_pulse_constraint[chan_num])
            
        if setup.do_ofamp_unconstrained_smooth[chan_num]:
            amp_noconstrain_smooth[jj], t0_noconstrain_smooth[jj], chi2_noconstrain_smooth[jj] = OF_smooth.ofamp_withdelay()
        
        if setup.ofamp_unconstrained_lowfreqchi2 and setup.do_chi2_lowfreq[chan_num]:
            chi2low_unconstrain[jj] = OF.chi2_lowfreq(amp_noconstrain[jj], t0_noconstrain[jj], 
                                                      fcutoff=setup.chi2_lowfreq_fcutoff[chan_num])
            if setup.ofamp_unconstrained_pulse_constraint[chan_num]!=0:
                chi2low_unconstrain_pcon[jj] = OF.chi2_lowfreq(amp_noconstrain_pcon[jj], t0_noconstrain_pcon[jj], 
                                                               fcutoff=setup.chi2_lowfreq_fcutoff[chan_num])

        if setup.do_ofamp_constrained[chan_num]:
            amp_constrain[jj], t0_constrain[jj], chi2_constrain[jj] = OF.ofamp_withdelay(
                                                                      nconstrain=setup.ofamp_constrained_nconstrain[chan_num])
            if setup.ofamp_constrained_pulse_constraint[chan_num]!=0:
                amp_constrain_pcon[jj], t0_constrain_pcon[jj], chi2_constrain_pcon[jj] = OF.ofamp_withdelay(
                                                             nconstrain=setup.ofamp_constrained_nconstrain[chan_num],
                                        pulse_direction_constraint=setup.ofamp_constrained_pulse_constraint[chan_num])
            
        if setup.do_ofamp_constrained_smooth[chan_num]:
            amp_constrain_smooth[jj], t0_constrain_smooth[jj], chi2_constrain_smooth[jj] = OF_smooth.ofamp_withdelay(
                                                                      nconstrain=setup.ofamp_constrained_nconstrain[chan_num])
        
        if setup.ofamp_constrained_lowfreqchi2 and setup.do_chi2_lowfreq[chan_num]:
            chi2low_constrain[jj] = OF.chi2_lowfreq(amp_constrain[jj], t0_constrain[jj], 
                                                    fcutoff=setup.chi2_lowfreq_fcutoff[chan_num])
            if setup.ofamp_constrained_pulse_constraint[chan_num]!=0:
                chi2low_constrain_pcon[jj] = OF.chi2_lowfreq(amp_constrain_pcon[jj], t0_constrain_pcon[jj], 
                                                            fcutoff=setup.chi2_lowfreq_fcutoff[chan_num])
        
        if setup.do_ofamp_pileup[chan_num]:
            amp_pileup[jj], t0_pileup[jj], chi2_pileup[jj] = OF.ofamp_pileup_iterative(amp_constrain[jj], t0_constrain[jj],
                                                               nconstrain=setup.ofamp_pileup_nconstrain[chan_num])
            if setup.ofamp_pileup_pulse_constraint[chan_num]!=0:
                amp_pileup_pcon[jj], t0_pileup_pcon[jj], chi2_pileup_pcon[jj] = OF.ofamp_pileup_iterative(amp_constrain[jj], 
                                                             t0_constrain[jj],
                                                             nconstrain=setup.ofamp_pileup_nconstrain[chan_num],
                                        pulse_direction_constraint=setup.ofamp_pileup_pulse_constraint[chan_num])
            
        if setup.do_ofamp_pileup_smooth[chan_num]:
            amp_pileup_smooth[jj], t0_pileup_smooth[jj], chi2_pileup_smooth[jj] = OF_smooth.ofamp_pileup_iterative(
                                                               amp_constrain_smooth[jj], t0_constrain_smooth[jj],
                                                               nconstrain=setup.ofamp_pileup_nconstrain[chan_num])
        
        if setup.do_ofamp_baseline[chan_num]:
            amp_baseline[jj], t0_baseline[jj], chi2_baseline[jj] = OF.ofamp_baseline(
                                                                   nconstrain=setup.ofamp_baseline_nconstrain[chan_num])
            if setup.ofamp_baseline_pulse_constraint[chan_num]!=0:
                amp_baseline_pcon[jj], t0_baseline_pcon[jj], chi2_baseline_pcon[jj] = OF.ofamp_baseline(
                                                             nconstrain=setup.ofamp_baseline_nconstrain[chan_num],
                                        pulse_direction_constraint=setup.ofamp_baseline_pulse_constraint[chan_num])
            
        if setup.do_ofamp_baseline_smooth[chan_num]:
            amp_baseline_smooth[jj], t0_baseline_smooth[jj], chi2_baseline_smooth[jj] = OF_smooth.ofamp_baseline(
                                                                   nconstrain=setup.ofamp_baseline_nconstrain[chan_num])
        
        if setup.do_ofnonlin[chan_num]:
            nlin = qp.OFnonlin(psd, fs, template=template)
            params_nlin, errors_nlin, _, reducedchi2_nlin = nlin.fit_falltimes(s, npolefit=2, lgcfullrtn=True)
            
            amp_nonlin[jj] = params_nlin[0]
            amp_nonlin_err[jj] = errors_nlin[0]
            taurise_nonlin[jj] = params_nlin[1]
            taurise_nonlin_err[jj] = errors_nlin[1]
            taufall_nonlin[jj] = params_nlin[2]
            taufall_nonlin_err[jj] = errors_nlin[2]
            t0_nonlin[jj] = params_nlin[3]
            t0_nonlin_err[jj] = errors_nlin[3]
            chi2_nonlin[jj] = reducedchi2_nlin * (len(nlin.data)-nlin.dof)
    
    # save variables to dict
    if setup.do_chi2_nopulse[chan_num]:
        rq_dict[f'chi2_nopulse_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_nopulse_{chan}{det}'][readout_inds] = chi0
    if setup.do_chi2_nopulse_smooth[chan_num]:
        rq_dict[f'chi2_nopulse_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_nopulse_smooth_{chan}{det}'][readout_inds] = chi0_smooth
    
    if setup.do_ofamp_nodelay[chan_num]:
        rq_dict[f'ofamp_nodelay_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_nodelay_{chan}{det}'][readout_inds] = amp_nodelay
        rq_dict[f'chi2_nodelay_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_nodelay_{chan}{det}'][readout_inds] = chi2_nodelay
    if setup.do_ofamp_nodelay_smooth[chan_num]:
        rq_dict[f'ofamp_nodelay_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_nodelay_smooth_{chan}{det}'][readout_inds] = amp_nodelay_smooth
        rq_dict[f'chi2_nodelay_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_nodelay_smooth_{chan}{det}'][readout_inds] = chi2_nodelay_smooth
    
    if setup.do_ofamp_unconstrained[chan_num]:
        rq_dict[f'ofamp_unconstrain_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_unconstrain_{chan}{det}'][readout_inds] = amp_noconstrain
        rq_dict[f't0_unconstrain_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_unconstrain_{chan}{det}'][readout_inds] = t0_noconstrain
        rq_dict[f'chi2_unconstrain_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_unconstrain_{chan}{det}'][readout_inds] = chi2_noconstrain
        if setup.ofamp_unconstrained_pulse_constraint[chan_num]!=0:
            rq_dict[f'ofamp_unconstrain_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'ofamp_unconstrain_pcon_{chan}{det}'][readout_inds] = amp_noconstrain_pcon
            rq_dict[f't0_unconstrain_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f't0_unconstrain_pcon_{chan}{det}'][readout_inds] = t0_noconstrain_pcon
            rq_dict[f'chi2_unconstrain_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'chi2_unconstrain_pcon_{chan}{det}'][readout_inds] = chi2_noconstrain_pcon
    if setup.do_ofamp_unconstrained_smooth[chan_num]:
        rq_dict[f'ofamp_unconstrain_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_unconstrain_smooth_{chan}{det}'][readout_inds] = amp_noconstrain_smooth
        rq_dict[f't0_unconstrain_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_unconstrain_smooth_{chan}{det}'][readout_inds] = t0_noconstrain_smooth
        rq_dict[f'chi2_unconstrain_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_unconstrain_smooth_{chan}{det}'][readout_inds] = chi2_noconstrain_smooth
    
    if setup.do_ofamp_constrained[chan_num]:
        rq_dict[f'ofamp_constrain_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_constrain_{chan}{det}'][readout_inds] = amp_constrain
        rq_dict[f't0_constrain_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_constrain_{chan}{det}'][readout_inds] = t0_constrain
        rq_dict[f'chi2_constrain_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_constrain_{chan}{det}'][readout_inds] = chi2_constrain
        if setup.ofamp_constrained_pulse_constraint[chan_num]!=0:
            rq_dict[f'ofamp_constrain_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'ofamp_constrain_pcon_{chan}{det}'][readout_inds] = amp_constrain_pcon
            rq_dict[f't0_constrain_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f't0_constrain_pcon_{chan}{det}'][readout_inds] = t0_constrain_pcon
            rq_dict[f'chi2_constrain_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'chi2_constrain_pcon_{chan}{det}'][readout_inds] = chi2_constrain_pcon
    if setup.do_ofamp_constrained_smooth[chan_num]:
        rq_dict[f'ofamp_constrain_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_constrain_smooth_{chan}{det}'][readout_inds] = amp_constrain_smooth
        rq_dict[f't0_constrain_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_constrain_smooth_{chan}{det}'][readout_inds] = t0_constrain_smooth
        rq_dict[f'chi2_constrain_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_constrain_smooth_{chan}{det}'][readout_inds] = chi2_constrain_smooth
        
    if setup.do_chi2_lowfreq[chan_num]:
        if setup.ofamp_nodelay_lowfreqchi2:
            rq_dict[f'chi2lowfreq_nodelay_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'chi2lowfreq_nodelay_{chan}{det}'][readout_inds] = chi2low_nodelay
        
        if setup.ofamp_unconstrained_lowfreqchi2:
            rq_dict[f'chi2lowfreq_unconstrain_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'chi2lowfreq_unconstrain_{chan}{det}'][readout_inds] = chi2low_unconstrain
            if setup.ofamp_unconstrained_pulse_constraint[chan_num]!=0:
                rq_dict[f'chi2lowfreq_unconstrain_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
                rq_dict[f'chi2lowfreq_unconstrain_pcon_{chan}{det}'][readout_inds] = chi2low_unconstrain_pcon
        
        if setup.ofamp_constrained_lowfreqchi2:
            rq_dict[f'chi2lowfreq_constrain_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'chi2lowfreq_constrain_{chan}{det}'][readout_inds] = chi2low_constrain
            if setup.ofamp_constrained_pulse_constraint[chan_num]!=0:
                rq_dict[f'chi2lowfreq_constrain_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
                rq_dict[f'chi2lowfreq_constrain_pcon_{chan}{det}'][readout_inds] = chi2low_constrain_pcon
    
    if setup.do_ofamp_pileup[chan_num]:
        rq_dict[f'ofamp_pileup_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_pileup_{chan}{det}'][readout_inds] = amp_pileup
        rq_dict[f't0_pileup_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_pileup_{chan}{det}'][readout_inds] = t0_pileup
        rq_dict[f'chi2_pileup_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_pileup_{chan}{det}'][readout_inds] = chi2_pileup
        if setup.ofamp_pileup_pulse_constraint[chan_num]!=0:
            rq_dict[f'ofamp_pileup_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'ofamp_pileup_pcon_{chan}{det}'][readout_inds] = amp_pileup_pcon
            rq_dict[f't0_pileup_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f't0_pileup_pcon_{chan}{det}'][readout_inds] = t0_pileup_pcon
            rq_dict[f'chi2_pileup_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'chi2_pileup_pcon_{chan}{det}'][readout_inds] = chi2_pileup_pcon
    if setup.do_ofamp_pileup_smooth[chan_num]:
        rq_dict[f'ofamp_pileup_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_pileup_smooth_{chan}{det}'][readout_inds] = amp_pileup_smooth
        rq_dict[f't0_pileup_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_pileup_smooth_{chan}{det}'][readout_inds] = t0_pileup_smooth
        rq_dict[f'chi2_pileup_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_pileup_smooth_{chan}{det}'][readout_inds] = chi2_pileup_smooth
        
    if setup.do_ofamp_baseline[chan_num]:
        rq_dict[f'ofamp_baseline_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_baseline_{chan}{det}'][readout_inds] = amp_baseline
        rq_dict[f't0_baseline_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_baseline_{chan}{det}'][readout_inds] = t0_baseline
        rq_dict[f'chi2_baseline_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_baseline_{chan}{det}'][readout_inds] = chi2_baseline
        if setup.ofamp_baseline_pulse_constraint[chan_num]!=0:
            rq_dict[f'ofamp_baseline_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'ofamp_baseline_pcon_{chan}{det}'][readout_inds] = amp_baseline_pcon
            rq_dict[f't0_baseline_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f't0_baseline_pcon_{chan}{det}'][readout_inds] = t0_baseline_pcon
            rq_dict[f'chi2_baseline_pcon_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'chi2_baseline_pcon_{chan}{det}'][readout_inds] = chi2_baseline_pcon
    if setup.do_ofamp_baseline_smooth[chan_num]:
        rq_dict[f'ofamp_baseline_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_baseline_smooth_{chan}{det}'][readout_inds] = amp_baseline_smooth
        rq_dict[f't0_baseline_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_baseline_smooth_{chan}{det}'][readout_inds] = t0_baseline_smooth
        rq_dict[f'chi2_baseline_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_baseline_smooth_{chan}{det}'][readout_inds] = chi2_baseline_smooth
    
    if setup.do_ofnonlin[chan_num]:
        rq_dict[f'ofamp_nlin_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_nlin_{chan}{det}'][readout_inds] = amp_nonlin
        rq_dict[f'ofamp_nlin_err_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'ofamp_nlin_err_{chan}{det}'][readout_inds] = amp_nonlin_err
        rq_dict[f'oftaurise_nlin_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'oftaurise_nlin_{chan}{det}'][readout_inds] = taurise_nonlin
        rq_dict[f'oftaurise_nlin_err_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'oftaurise_nlin_err_{chan}{det}'][readout_inds] = taurise_nonlin_err
        rq_dict[f'oftaufall_nlin_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'oftaufall_nlin_{chan}{det}'][readout_inds] = taufall_nonlin
        rq_dict[f'oftaufall_nlin_err_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'oftaufall_nlin_err_{chan}{det}'][readout_inds] = taufall_nonlin_err
        rq_dict[f't0_nlin_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_nlin_{chan}{det}'][readout_inds] = t0_nonlin
        rq_dict[f't0_nlin_err_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f't0_nlin_err_{chan}{det}'][readout_inds] = t0_nonlin_err
        rq_dict[f'chi2_nlin_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
        rq_dict[f'chi2_nlin_{chan}{det}'][readout_inds] = chi2_nonlin
    
    if any(setup.do_ofamp_shifted) and setup.trigger is not None:
        # do the shifted OF on each trace
        if chan_num==setup.trigger:
            if setup.shifted_fit=="nodelay" and any(setup.do_ofamp_nodelay):
                setup.t0_shifted = np.zeros(len(signal))
            elif setup.shifted_fit=="constrained" and any(setup.do_ofamp_constrained):
                setup.t0_shifted = t0_constrain
            elif setup.shifted_fit=="unconstrained" and any(setup.do_ofamp_unconstrained):
                setup.t0_shifted = t0_unconstrain
        elif setup.do_ofamp_shifted[chan_num]:
            amp_shifted = np.zeros(len(signal))
            chi2_shifted = np.zeros(len(signal))

            for jj, s in enumerate(signal):
                amp_shifted[jj], _, chi2_shifted[jj] = qp.ofamp(s, rp.shift(template, int(setup.t0_shifted[jj]*fs)), 
                                                                            psd, fs, withdelay=False)

            rq_dict[f'ofamp_shifted_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'ofamp_shifted_{chan}{det}'][readout_inds] = amp_shifted
            rq_dict[f't0_shifted_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f't0_shifted_{chan}{det}'][readout_inds] = setup.t0_shifted
            rq_dict[f'chi2_shifted_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'chi2_shifted_{chan}{det}'][readout_inds] = chi2_shifted
            
    if any(setup.do_ofamp_shifted_smooth) and setup.trigger is not None:
        # do the shifted OF on each trace
        if chan_num==setup.trigger:
            if not setup.do_ofamp_shifted:
                if setup.shifted_fit=="nodelay" and any(setup.do_ofamp_nodelay_smooth):
                    setup.t0_shifted = np.zeros(len(signal))
                elif setup.shifted_fit=="constrained" and any(setup.do_ofamp_constrained_smooth):
                    setup.t0_shifted = t0_constrain
                elif setup.shifted_fit=="unconstrained" and any(setup.do_ofamp_unconstrained_smooth):
                    setup.t0_shifted = t0_unconstrain
        elif setup.do_ofamp_shifted_smooth[chan_num]:
            amp_shifted_smooth = np.zeros(len(signal))
            chi2_shifted_smooth = np.zeros(len(signal))

            for jj, s in enumerate(signal):
                amp_shifted_smooth[jj], _, chi2_shifted_smooth[jj] = qp.ofamp(s, rp.shift(template, int(setup.t0_shifted[jj]*fs)), 
                                                                              psd_smooth, fs, withdelay=False)

            rq_dict[f'ofamp_shifted_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'ofamp_shifted_smooth_{chan}{det}'][readout_inds] = amp_shifted_smooth
            rq_dict[f't0_shifted_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f't0_shifted_smooth_{chan}{det}'][readout_inds] = setup.t0_shifted
            rq_dict[f'chi2_shifted_smooth_{chan}{det}'] = np.ones(len(readout_inds))*(-999999.0)
            rq_dict[f'chi2_shifted_smooth_{chan}{det}'][readout_inds] = chi2_shifted_smooth
    
    return rq_dict
    
def _calc_rq(traces, channels, det, setup, readout_inds=None):
    """
    Helper function for calculating RQs for arrays of traces.
    
    Parameters
    ----------
    traces : ndarray
        Array of traces to use in calculation of RQs. Should be of shape (number of traces,
        number of channels, length of trace)
    channels : list of str
        List of the channels that will be processed
    det : list of str
        The detector ID that corresponds to the channels that will be processed.
    setup : SetupRQ
        A SetupRQ class object. This object defines all of the different RQs that should be calculated 
        and specifies relevant parameters.
    readout_inds : ndarray of bool, optional
        Boolean mask that specifies which traces should be used to calculate the RQs. RQs for the 
        excluded traces are set to -999999.0. 
    
    Returns
    -------
    rq_dict : dict
        A dictionary containing all of the RQs that were calculated (as specified by the setup object).
    
    """
    
    if readout_inds is None:
        readout_inds = np.ones(len(traces), dtype=bool)
    
    rq_dict = {}
    
    if setup.calcchans:
        vals = list(enumerate(zip(channels, det)))
        
        if setup.do_ofamp_shifted and setup.trigger is not None:
            # change order so that trigger is processed to be able get the shifted times
            # to be able to shift the non-trigger channels to the right time
            vals[setup.trigger], vals[0] = vals[0], vals[setup.trigger]
        
        for ii, (chan, d) in vals:
            signal = traces[readout_inds, ii, setup.indstart:setup.indstop]
            template = setup.templates[ii]
            psd = setup.psds[ii]

            chan_dict = _calc_rq_single_channel(signal, template, psd, setup, readout_inds, chan, ii, d)
            
            rq_dict.update(chan_dict)
            
    if setup.calcsum:
        signal = traces[readout_inds, :, setup.indstart:setup.indstop].sum(axis=1)
        template = setup.summed_template
        psd = setup.summed_psd
        chan = "sum"

        sum_dict = _calc_rq_single_channel(signal, template, psd, setup, readout_inds, chan, 0, "")

        rq_dict.update(sum_dict)
    
    return rq_dict

def _rq(file, channels, det, setup, convtoamps, savepath, lgcsavedumps, filetype):
    """
    Helper function for processing raw data to calculate RQs for single files.
    
    Parameters
    ----------
    file : str
        Path to a file that should be opened and processed.
    channels : list of str
        List of the channels that will be processed.
    det : list of str
        The detector ID that corresponds to the channels that will be processed.
    setup : SetupRQ
        A SetupRQ class object. This object defines all of the different RQs that should be calculated 
        and specifies relevant parameters.
    convtoamps : list
        List of the factors for each channel that will convert the units to Amps.
    savepath : str
        The path to where each dump should be saved, if lgcsavedumps is set to True.
    lgcsavedumps : bool
        Boolean flag for whether or not the DataFrame for each dump should be saved individually.
        Useful for saving data as the processing routine is run, allowing checks of the data during
        run time.
    filetype : str
        The string that corresponds to the file type that will be opened. Supports two 
        types -"mid.gz" and "npz".
    
    Returns
    -------
    rq_df : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for the dataset specified.
    
    """
    
    if filetype == "mid.gz" and not HAS_SCDMSPYTOOLS:
        raise ImportError("Cannot use filetype mid.gz because scdmsPyTools is not installed.")
        
    if filetype == "mid.gz":
        seriesnum = file.split('/')[-2]
        dump = file.split('/')[-1].split('_')[-1].split('.')[0]
    elif filetype == "npz":
        seriesnum = file.split('/')[-1].split('.')[0]
        dump = f"{int(seriesnum.split('_')[-1]):04d}"
        
    print(f"On Series: {seriesnum},  dump: {dump}")
    
    if isinstance(channels, str):
        channels = [channels]
        
    if isinstance(det, str):
        det = [det]*len(channels)
        
    if len(det)!=len(channels):
        raise ValueError("channels and det should have the same length")
    
    if filetype == "mid.gz":
        traces, info_dict = io.get_traces_midgz([file], channels=channels, det=det, convtoamps=convtoamps,
                                                lgcskip_empty=False, lgcreturndict=True)
    elif filetype == "npz":
        traces, info_dict = io.get_traces_npz([file])
    
    data = {}
    
    data.update(info_dict)
    
    if filetype == "mid.gz":
        readout_inds = []
        for d in set(det):
            readout_inds.append(np.array(data[f'readoutstatus{d}'])==1)
        readout_inds = np.logical_and.reduce(readout_inds)
    elif filetype == "npz":
        readout_inds = None
    
    rq_dict = _calc_rq(traces, channels, det, setup, readout_inds=readout_inds)
    
    data.update(rq_dict)
    
    rq_df = pd.DataFrame.from_dict(data)
    
    if lgcsavedumps:
        rq_df.to_pickle(f'{savepath}rq_df_{seriesnum}_d{dump}.pkl')   

    return rq_df


def rq(filelist, channels, setup, det="Z1", savepath='', lgcsavedumps=False, nprocess=1, filetype="mid.gz"):
    """
    Function for processing raw data to calculate RQs. Supports multiprocessing.
    
    Parameters
    ----------
    filelist : list
        List of paths to each file that should be opened and processed
    channels : str, list of str
        List of the channel names that will be processed. Used when naming RQs. When filetype is "mid.gz", 
        this is also used when reading the traces from each file.
    setup : SetupRQ
        A SetupRQ class object. This object defines all of the different RQs that should be calculated 
        and specifies relevant parameters.
    det : str, list of str, optional
        The detector ID that corresponds to the channels that will be processed. Set to "Z1" by default.
        Only used when filetype is "mid.gz". If a list of strings, then should each value should directly 
        correspond to the channel names. If a string is inputted and there are multiple channels, then it
        is assumed that the detector name is the same for each channel.
    savepath : str
        The path to where each dump should be saved, if lgcsavedumps is set to True.
    lgcsavedumps : bool
        Boolean flag for whether or not the DataFrame for each dump should be saved individually.
        Useful for saving data as the processing routine is run, allowing checks of the data during
        run time.
    nprocess : int, optional
        The number of processes that should be used when multiprocessing. The default is 1.
    filetype : str, optional
        The string that corresponds to the file type that will be opened. Supports two 
        types -"mid.gz" and "npz". "mid.gz" is the default.
    
    Returns
    -------
    rq_df : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for each dataset in filelist.
    
    """
    
    if filetype == "mid.gz" and not HAS_SCDMSPYTOOLS:
        raise ImportError("Cannot use filetype mid.gz because scdmsPyTools is not installed.")
    
    if isinstance(filelist, str):
        filelist = [filelist]
        
    if isinstance(channels, str):
        channels = [channels]
        
    if isinstance(det, str):
        det = [det]*len(channels)
        
    if len(det)!=len(channels):
        raise ValueError("channels and det should have the same length")
    
    folder = os.path.split(filelist[0])[0]
    
    if filetype == "mid.gz":
        convtoamps = []
        for ch, d in zip(channels, det):
            convtoamps.append(io.get_trace_gain(folder, ch, d)[0])
    elif filetype == "npz":
        convtoamps = [1]*len(channels)
    
    if nprocess == 1:
        results = []
        for f in filelist:
            results.append(_rq(f, channels, det, setup, convtoamps, savepath, lgcsavedumps, filetype))
    else:
        pool = multiprocessing.Pool(processes = nprocess)
        results = pool.starmap(_rq, zip(filelist, repeat(channels), repeat(det), repeat(setup), 
                                        repeat(convtoamps), repeat(savepath), repeat(lgcsavedumps),
                                        repeat(filetype)))
        pool.close()
        pool.join()
    
    rq_df = pd.concat([df for df in results], ignore_index = True)
    
    return rq_df

