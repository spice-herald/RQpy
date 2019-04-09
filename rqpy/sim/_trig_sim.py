from scipy import signal, special
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


__all__ = [
    "TrigSim",
    "trigger_efficiency",
]


def trigger_efficiency(x, offset, width):
    """
    The expected functional form of the trigger efficiency, using an Error Function.

    Parameters
    ----------
    x : array_like
        The x-values at which to calculate the trigger efficiency, e.g. energy values.
    offset : float
        The offset of the trigger efficiency curve, such that the midpoint of the Error
        Function occurs at this point. Should have same units as `x`.
    width : float
        The width of the Error Function, with the same units as `x`. This width can be
        thought of as the width of the Gaussian distribution that, when convolved with
        a step function placed at `offset`, gives us the resulting Error Function.

    Returns
    -------
    trig_eff : array_like
        The corresponding trigger efficiency (between 0 and 1) for each value of the
        inputted `x`.

    """

    trig_eff = (1 - special.erf((offset - x) / (np.sqrt(2) * width))) / 2

    return trig_eff


class TrigSim(object):
    """
    Class for setting up the DCRC FIR filter trigger simulation.

    Attributes
    ----------
    _Trigger : Object
        The `trigsim.Trigger` object which is used to build and run the FIR filter.
    input_psds : list
        The input power spectral density for the data, re-arranged for input into the FIR
        filter code.
    input_pulse_shapes : list
        The input pulse shape to use for the FIR filter, normalized to have a height of 1,
        and re-arranged for input into the FIR filter code.
    fs : float
        The digitization rate of the data in Hz.
    can_run_trigger : bool
        A boolean flag for whether or not the `TrigSim.trigger` method can be run.
    which_channel : int
        The channel that the FIR filter will be triggering. Always set to zero, should not be
        changed.
    of_coeffs : ndarray
        The coefficients of the FIR filter that will be applied to the downsampled
        data.
    _resolution : list
        The calculated resolution of the FIR filter.
    threshold : int
        The threshold to set for triggering on pulses, should be in the arbitrary units of the
        filter.
    convtoadcbins : float
        The conversion factor for expressing FIR filter amplitudes in units of ADC bins.

    """

    def __init__(self, psd, template, fs, fir_bits_out=32, fir_discard_msbs=4):
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
        fir_bits_out : int, optional
            The number of bits to use in the integer values of the FIR. Default is 32, corresponding
            to 32-bit integer trigger amplitudes. This is the recommended value, smaller values
            may result in saturation fo the trigger amplitude (where the true amplitude would be
            larger than the largest integer).
        fir_discard_msbs : int, optional
            The FIR pre-truncation shift of the bits for the FIR module. Default is 4, which is the
            recommended value.

        """

        if not HAS_TRIGSIM:
            raise ImportError("Cannot run the trigger simulation because trigsim is not installed.")

        self.input_psds = [psd]*12 + [np.ones(4*len(psd))]*4
        self.input_pulse_shapes = [template]*12 + [np.zeros(4*template.shape[0])]*4
        self.fs = fs
        self.can_run_trigger = False
        self.threshold = 0

        raw_LC_coeffs = np.zeros((4,16))
        self.which_channel = 0
        raw_LC_coeffs[0, self.which_channel] = 1 # only one phonon channel
        LC_coeffs = trigsim.scale_max(raw_LC_coeffs, bits=8, axis=1)
        TrL_requires = np.zeros((8,16), dtype=bool)
        TrL_vetos = np.zeros((8,16), dtype=bool)

        self._Trigger = trigsim.Trigger(bits_phonon=16, bits_charge=16,
                                    phonon_DF_R=16, phonon_DF_N=3, phonon_DF_M=1, phonon_start=0,
                                    charge_DF_R=64, charge_DF_N=3, charge_DF_M=1, charge_start=0,
                                    LC_coeffs=LC_coeffs, LC_bits_out=46,
                                    LC_bits_coeff=8, LC_discard_MSBs=[0,0,0,0],
                                    FIR_coeffs=np.zeros((4,1024), 'i8'), FIR_bits_out=fir_bits_out, 
                                    FIR_bits_coeff=16, FIR_discard_MSBs=[fir_discard_msbs,0,0,0],
                                    ThL_selectors=np.array([0,1,2,3,1,1,2,3], dtype='uint'),
                                    ThL_activation_thresholds   = (2**15 - 1)*np.ones(8, int),
                                    ThL_deactivation_thresholds = np.zeros(8, int),
                                    PS_max_window_lengths=(2**31 - 1)*np.ones(4, dtype='uint'),
                                    PS_saturated_pulse_offsets=np.zeros(4, dtype='uint'),
                                    TrL_selectors=np.array([0,1,2,3,1,1,2,3], dtype='uint'),
                                    TrL_enables=np.array([1,0,0,0,0,0,0,0], dtype=bool),
                                    TrL_requires=TrL_requires, TrL_vetos=TrL_vetos,
                                    TrL_prescales=np.ones(8, dtype='float'))
        
        self.of_coeffs = self._Trigger.build_OF_coeffs(self.input_psds, self.input_pulse_shapes)
        self._Trigger.set_FIR_coeffs(self.of_coeffs)
        self.convtoadcbins = self._convtoadcbins(fir_bits_out, fir_discard_msbs)

    def _convtoadcbins(self, fir_bits_out, fir_discard_msbs):
        """
        Hidden function for calculating the conversion factor from FPGA amplitude to ADC bins.

        Parameters
        ----------
        fir_bits_out : int
            The number of bits to use in the integer values of the FIR.
        fir_discard_msbs : int
            The FIR pre-truncation shift of the bits for the FIR module.

        Returns
        -------
        scale_factor : float
            The conversion factor for expressing FIR filter amplitudes in units of ADC bins.

        """

        fir_pulse_shapes = self._Trigger.compute_FIR_pulse_shapes(self.input_pulse_shapes)
        fir_length = fir_pulse_shapes.shape[1]
        fir_psds = self._Trigger.compute_FIR_PSDs(self.input_psds, fir_length)
        fir_psds[fir_psds == 0] = np.inf

        pulses_freq = np.fft.fft(fir_pulse_shapes, axis=1)
        ofs = np.nan_to_num(pulses_freq.conj() / fir_psds)

        of_norms = np.sum(np.abs(ofs * pulses_freq), axis=1, keepdims=True)
        of_norms[of_norms==0] = 1
        ofs /= of_norms
        ofs[:,0] = 0 # Kill the DC component

        of_coeffs = np.real(np.fft.ifft(ofs, axis=1))
        fir_coeff_bits = self._Trigger.FIR.modules[0].bits_coeff

        num = np.max(np.abs(of_coeffs))
        denom = 2**(fir_coeff_bits) - 1

        if num == np.abs(np.max(of_coeffs)):
            denom -= 1

        scale_factor = num/denom
        scale_factor *= 2 * fir_length * 2**(self._Trigger.FIR.modules[0].bits_sum - fir_bits_out - fir_discard_msbs)

        return scale_factor

    def resolution(self):
        """
        Method for calculation the resolution of the FIR filter in the arbitrary units of the filter.

        Returns
        -------
        resolution : ndarray
            The resolution of the FIR filter.

        """

        self._resolution = self._Trigger.resolution(self.input_psds, deltaT_phonon=1/self.fs)
        
        return self._resolution

    def set_threshold(self, threshold):
        """
        Method for calculation the resolution of the FIR filter in the arbitrary units of the filter.

        Parameters
        ----------
        threshold : float
            The threshold to set for triggering on pulses, should be in the arbitrary units of the
            filter.

        Notes
        -----
        For this function, if the trigger is already known, then it is recommended that the user
        calculate the resolution of the FIR filter using the `TrigSim.resolution` method. The output
        of that function times the number of standard deviations desired can then be directly passed 
        to this function.

        Examples
        --------
        >>> import rqpy as rp

        Load the PSD, template, and fs for passing to `TrigSim`.

        >>> psd, template, fs = my_load_function('/path/to/data/file.ext')
        >>> TS = rp.sim.TrigSim(psd, template, fs)
        >>> resolution = TS.resolution()

        Set the treshold to a 5-sigma threshold.

        >>> TS.set_threshold(5*resolution[0])

        Now the `TrigSim.trigger` method can be run on some data, where we have a 5-sigma trigger
        threshold.

        """

        self.threshold = threshold

        if np.isscalar(threshold):
            threshold = (~np.all(~self.of_coeffs, axis=1)).astype(float)*threshold

        self._Trigger.ThL.ThLs[self.which_channel].set_thresholds(np.int64(threshold), 0)

        self.can_run_trigger = True

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
        triggertime : float
            The time of the triggered pulse as calculated by the FIR filter, in s. Taken as the bin
            number of the maximum amplitude within the valid part of the trace divided by the
            downsampled digitization rate.
        fir_out : ndarray
            The complete, filtered trace corresponding to the inputted trace, in the arbitrary
            units of the filter.

        Raises
        ------
        ValueError
            If the `TrigSim.set_threshold` method was not used to set the trigger threshold,
            then an error is thrown because the trigger is not properly set up.

        """

        if not self.can_run_trigger:
            raise ValueError("The threshold was not set, please use the TrigSim.set_threshold method.")

        input_traces = [(x[k:] - 32768).astype('int64')] + [np.zeros(len(x[k:]),dtype='int64')]*11 + \
                       [np.zeros(4*len(x[k:]),dtype='int64')]*4

        self._Trigger.reset()

        pipe = trigsim.pipeline(self._Trigger.DF, self._Trigger.LC, self._Trigger.LC_trunc, 
                                self._Trigger.FIR, self._Trigger.FIR_trunc)

        fir_out = pipe.send(input_traces)

        bins_to_keep = (len(x) - k)//16 - 1024

        if any(fir_out[0, -bins_to_keep:] >= self.threshold):
            triggeramp = fir_out[0, -bins_to_keep:].max()
            triggertime = (np.argmax(fir_out[0, -bins_to_keep:]) + 512 + k / 16) / (self.fs / 16)

            return triggeramp, triggertime, fir_out[0, -bins_to_keep:]
        else:
            return 0, 0, fir_out[0, -bins_to_keep:]
