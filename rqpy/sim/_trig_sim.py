from scipy import signal
import qetpy as qp
import rqpy as rp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import pickle as pkl
from rqpy import HAS_TRIGSIM

if HAS_TRIGSIM:
    import trigsim


__all__ = ["TrigSim"]


class TrigSim(object):
    """
    Class for setting up the DCRC FIR filter trigger simulation.
    
    Attributes
    ----------
    trig : Object
        The `trigsim.Trigger` object which is used to build and run the FIR filter.
    resolution : ndarray
        The resolution of the FIR filter in the arbitrary units of the filter.
    
    """
    
    def __init__(self, psd, template, fs, nsigma=5):
        """
        Initialization of the TrigSim class.
        
        Parameters
        ----------
        psd : ndarray
            The input power spectral density to use for the FIR filter. Should be in
            units of ADC bins.
        template : ndarray
            The pulse template to use for the FIR filter. Should be normalized to have a
            height of 1.
        fs : float
            The digitization rate of the data in Hz.
        nsigma : float, optional
            The number of standard deviations of the energy resolution to use as the threshold 
            for which events will be detected as a pulse in the FIR filter.
        
        """
        
        if not HAS_TRIGSIM:
            raise ImportError("Cannot run the trigger simulation because trigsim is not installed.")
        
        input_PSDs = [psd]*12 + [np.ones(4*len(psd))]*4
        input_pulse_shapes = [template]*12 + [np.zeros(4*template.shape[0])]*4

        raw_LC_coeffs = np.zeros((4,16))
        which_channel = 0
        raw_LC_coeffs[0, which_channel] = 1 # only one phonon channel
        LC_coeffs = trigsim.scale_max(raw_LC_coeffs, bits=8, axis=1)
        TrL_requires = np.zeros((8,16), dtype=bool)
        TrL_vetos = np.zeros((8,16), dtype=bool)

        self.trig = trigsim.Trigger(bits_phonon=16, bits_charge=16,
                                    phonon_DF_R=16, phonon_DF_N=3, phonon_DF_M=1, phonon_start=0,
                                    charge_DF_R=64, charge_DF_N=3, charge_DF_M=1, charge_start=0,
                                    LC_coeffs=LC_coeffs, LC_bits_out=46,
                                    LC_bits_coeff=8, LC_discard_MSBs=[0,0,0,0],
                                    FIR_coeffs=np.zeros((4,1024), 'i8'), FIR_bits_out=32, 
                                    FIR_bits_coeff=16, FIR_discard_MSBs=[4,0,0,0],
                                    ThL_selectors=np.array([0,1,2,3,1,1,2,3], dtype='uint'),
                                    ThL_activation_thresholds   = (2**15 - 1)*np.ones(8, int),
                                    ThL_deactivation_thresholds = np.zeros(8, int),
                                    PS_max_window_lengths=(2**31 - 1)*np.ones(4, dtype='uint'),
                                    PS_saturated_pulse_offsets=np.zeros(4, dtype='uint'),
                                    TrL_selectors=np.array([0,1,2,3,1,1,2,3], dtype='uint'),
                                    TrL_enables=np.array([1,0,0,0,0,0,0,0], dtype=bool),
                                    TrL_requires=TrL_requires, TrL_vetos=TrL_vetos,
                                    TrL_prescales=np.ones(8, dtype='float'))
        
        OF_coeffs = self.trig.build_OF_coeffs(input_PSDs, input_pulse_shapes)
        self.trig.set_FIR_coeffs(OF_coeffs)
        self.resolution = self.trig.resolution(input_PSDs, deltaT_phonon=1/fs)
        self.trig.ThL.ThLs[which_channel].set_thresholds(np.int64(nsigma*self.resolution), 0)
        
    def trigger(self, x, k=12):
        """
        Method for sending a trace to run the trigger simulation on.
        
        Parameters
        ----------
        x : ndarray
            The input array to run the FIR filter on. Should be a 1-d ndarray in units of 
            ADC bins.
        k : int, optional
            The bin number to start the FIR filter at. Since the filter downsamples the data
            by a factor of 16, the starting bin has a small effect on the calculated amplitude.
        
        Returns
        -------
        triggeramp : int
            The trigger amplitude of the pulse as calculated by the FIR filter, in the arbitrary 
            units of the filter. Taken from the maximum amplitude in the trace.
        fir_out : ndarray
            The complete, filtered trace corresponding to the inputted trace, in the arbitrary
            units of the filter.
        
        """
        
        input_traces = [(x[k:] - 32768).astype('int64')] + [np.zeros(len(x[k:]),dtype='int64')]*11 + \
                       [np.zeros(4*len(x[k:]),dtype='int64')]*4
        
        self.trig.reset()
        
        pipe = trigsim.pipeline(self.trig.DF, self.trig.LC, self.trig.LC_trunc, 
                                 self.trig.FIR, self.trig.FIR_trunc)
        
        fir_out = pipe.send(input_traces)
        
        triggeramp = fir_out[0, 1024:].max()
        
        return triggeramp, fir_out

