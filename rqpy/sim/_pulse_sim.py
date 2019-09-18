import numpy as np
import pandas as pd
import os
from glob import glob
from math import log10, floor
from scipy import stats

import deepdish as dd
import rqpy as rp
from rqpy import io
from rqpy import HAS_SCDMSPYTOOLS

if HAS_SCDMSPYTOOLS:
    from scdmsPyTools.BatTools.IO import getDetectorSettings

pd.io.pytables._tables()

__all__ = ["PulseSim", "buildfakepulses"]


class PulseSim(object):
    """
    Helper class for easier use of setting up `rqpy.sim.buildfakepulses`.

    Attributes
    ----------
    rq : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for the dataset specified.
    basepath : str
        The base path to the directory that contains the folders that the event dumps
        are in. The folders in this directory should be the series numbers.
    filetype : str
        The string that corresponds to the file type that will be opened. Supports two
        types: "mid.gz" and "npz". "mid.gz" is the default.
    templates : list
        The list of template(s) to be added to the traces, assumed to be normalized to a max
        height of 1. The template start time should be centered on the center bin. If `taufalls`
        and `taurises` are set, then the `templates` are only used to determine how many pulses
        there are.
    fs : float
        The digitization rate in Hz of the data.
    cut : array_like of bool, NoneType
        A boolean array for the cut that selects the traces that will be loaded from the dump
        files. These traces serve as the underlying data to which a template is added.
    ntraces : int
        The number of traces included in the cut.
    amplitudes : list
        The list of amplitudes, in Amps, by which to scale the template to add to the traces.
        Must be same length as cut. Each ndarray in the list corresponds to the amplitudes
        of the corresponding template in the list of templates.
    tdelay : list
        The time delay offset, in seconds, by which to shift the template to add to the traces.
        Bin interpolation is implemented for values that are not a multiple the reciprocal of
        the digitization rate. Each ndarray in the list can be passed, where each ndarray
        corresponds to the tdelays of the corresponding template in the list of templates.
    taurises : list, NoneType
        The rise times to use for each simulated pulse in seconds. Each ndarray in the list can
        be passed, where each ndarray corresponds to the taurises of the corresponding pulse.
        This will supersede the `templates` attribute if used. `taufalls` must also be
        specified to use this.
    taufalls : list, NoneType
        The fall times to use for each simulated pulse in seconds. Each ndarray in the list can
        be passed, where each ndarray corresponds to the taurises of the corresponding pulse.
        This will supersede the `templates` attribute if used. `taurises` must also be
        specified to use this.

    """

    def __init__(self, rq, basepath, filetype, templates, fs, cut=None):
        """
        Initialization of the PulseSim class.

        Parameters
        ----------
        rq : pandas.DataFrame
            A pandas DataFrame object that contains all of the RQs for the dataset specified.
        basepath : str
            The base path to the directory that contains the folders that the event dumps
            are in. The folders in this directory should be the series numbers.
        filetype : str
            The string that corresponds to the file type that will be opened. Supports two
            types: "mid.gz" and "npz". "mid.gz" is the default.
        templates : list, numpy.ndarray
            The template(s) to be added to the traces, assumed to be normalized to a max height
            of 1. The template start time should be centered on the center bin. If a list of
            templates, then each template will be added to the traces in succession, using
            the corresponding `amplitudes` and `tdelay`. If `taufalls` and `taurises` are set,
            then the `templates` are only used to determine how many pulses there are.
        fs : float
            The digitization rate in Hz of the data.
        cut : array_like of bool, NoneType, optional
            A boolean array for the cut that selects the traces that will be loaded from the dump
            files. These traces serve as the underlying data to which a template is added.

        """

        self.rq = rq
        self.basepath = basepath
        self.fs = fs

        if filetype not in ["npz", "mid.gz"]:
            raise ValueError("Only npz and mid.gz file types are currently supported by PulseSim")

        self.filetype = filetype
        self.cut = cut

        self.ntraces = self.cut.sum() if self.cut is not None else None

        self.amplitudes = []
        self.tdelay = []
        self.taurises = None
        self.taufalls = None

        if isinstance(templates, np.ndarray):
            templates = [templates]

        self.templates = templates

    @staticmethod
    def _check_valid_attr(attr):
        """
        Helper method for checking if an attribute is valid when generating simulated data.

        Parameters
        ----------
        attr : str
            The attribute that will be checked.

        Raises
        ------
        ValueError
            If `attr` is not a string.
            If `attr` is not "amplitudes", "tdelay", "taurises", or "taufalls".

        """

        if not isinstance(attr, str):
            raise TypeError("The inputted attr is not a string.")
        if attr not in ["amplitudes", "tdelay", "taurises", "taufalls"]:
            raise ValueError(
                "The inputted attr is not a valid option. "
                "Please see the docstring for valid values."
            )

    @staticmethod
    def _check_basedumpnum(basedumpnum):
        """
        Helper method for checking if basedumpnum is a positive integer (inclusive of zero).

        Parameters
        ----------
        basedumpnum : int
            Value to check if it is a positive integer (zero-inclusive).

        """

        if not isinstance(basedumpnum, int):
            raise TypeError("The inputted basedumpnum is not an int.")
        if basedumpnum < 0:
            raise ValueError("basedumpnum must be 0 or greater.")

    def _check_if_cut_set(self):
        """
        Helper method for checking if the cut has been loaded.

        Raises
        ------
        ValueError
            If the cut has not yet been set, but is still None.

        """

        if self.cut is None:
            raise ValueError(
                "The cut has not been set, consider setting it "
                "via the PulseSim.update_cut method."
            )

    def _check_sim_data(self):
        """
        Helper method for checking if the size of the simulated data matches
        the number of templates.

        Raises
        ------
        ValueError
            If the length of the list of amplitudes does not match the length of the list
            of templates.
            If the length of the list of tdelay does not match the length of the list of
            templates.

        """

        if len(self.amplitudes) != len(self.templates):
            raise ValueError(
                f"There are {len(self.templates)} templates, but only "
                f"{len(self.amplitudes)} sets of amplitudes data. Consider "
                "adding more using PulseSim.generate_sim_data."
            )
        elif len(self.tdelay) != len(self.templates):
            raise ValueError(
                f"There are {len(self.templates)} templates, but only "
                f"{len(self.tdelay)} sets of tdelay data. Consider adding "
                "more using PulseSim.generate_sim_data."
            )
        elif self.taurises is not None and len(self.taurises) != len(self.templates):
            raise ValueError(
                f"There are {len(self.templates)} pulses specified, but only "
                f"{len(self.taurises)} sets of taurises data. Consider adding "
                "more using PulseSim.generate_sim_data."
            )
        elif self.taufalls is not None and len(self.taufalls) != len(self.templates):
            raise ValueError(
                f"There are {len(self.templates)} pulses specified, but only "
                f"{len(self.taufalls)} sets of taufalls data. Consider adding "
                "more using PulseSim.generate_sim_data."
            )

    def _check_channel_det(self, channel, det):
        """
        Helper method for checking if the `channel` and `det` args are set.

        Parameters
        ----------
        channel : any_type
            A channel variable to check against None for filetype "mid.gz".
        det : any_type
            A det variable to check against None for filetype "mid.gz".

        Raises
        ------
        ValueError
            If filetype is "mid.gz" and either channel or det are None.

        """

        if self.filetype == "mid.gz" and (not isinstance(channel, str) or not isinstance(det, str)):
            raise ValueError("For filetype mid.gz, the channel and det kwargs must be set and be strings.")

    def _check_convtoamps(self, convtoamps, channel, det):
        """
        Helper method for automatically determining the `convtoamps` variable if it hasn't been set.

        Parameters
        ----------
        convtoamps : float, NoneType
            The factor that would convert the units of the data to amplitude.
        channel : str, NoneType
            The name the channel that should be loaded. Only used if filetype=="mid.gz".
        det : str, NoneType
            String that specifies the detector name. Only used if filetype=='mid.gz'.

        Returns
        -------
        convtoamps_auto : float
            The convtoamps value. Same as the inputted value if it was not None, otherwise
            it was automatically set based on filetype.

        """

        self._check_channel_det(channel, det)

        if self.filetype == "mid.gz" and convtoamps is None:
            snum = list(set(self.rq.seriesnumber))[0]
            snum_str = f"{snum:012}"
            snum_str = snum_str[:8] + '_' + snum_str[8:]
            return rp.io.get_trace_gain(f"{self.basepath}{snum_str}/", channel, det)[0]

        elif self.filetype == "npz" and convtoamps is None:
            return 1

        return convtoamps

    def _check_data_size(self, attr):
        """
        Helper method for checking if the data size is less than or equal to the template size.

        Raises
        ------
        ValueError
            If the attr that is being set already has the maximum length.

        """

        if len(getattr(self, attr)) == len(self.templates):
            raise ValueError(f"Cannot add any more {attr}, as this would "
                             f"result in more {attr} data than templates.")

    def _reset_sim_data(self):
        """
        Helper method for resetting the `amplitudes` and `tdelay` attributes to empty lists,
        as well as the `taurises` and `taufalls` attributes to None.

        """

        self.amplitudes = []
        self.tdelay = []
        self.taurises = None
        self.taufalls = None

    def _check_taus_set(self, attr):
        """
        Helper method for checking if `taurises` or `taufalls` are being set.

        """

        valid_attrs = ["taurises", "taufalls"]

        if attr in valid_attrs and getattr(self, attr) is None:
            setattr(self, attr, list())

    def update_cut(self, cut):
        """
        Method for updating the inputted cut. Useful for either changing the cut or setting it,
        if it was not set in the initialization.

        Parameters
        ----------
        cut : array_like of bool
            A boolean array for the cut that selects the traces that will be loaded from the dump
            files. These traces serve as the underlying data to which a template is added.

        """

        self.cut = cut
        self.ntraces = np.sum(self.cut)

        self._reset_sim_data()

    def generate_sim_data(self, attr, *args, distribution=None, values=None, **kwargs):
        """
        Method for generating simulated data and adding it to the specified attribute.

        Parameters
        ----------
        attr : str
            The attribute that will be updated with the simulated data. Can be either
            "amplitudes", "tdelay", "taurises", or "taufalls".
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution (see docstring of the
            instance object for more information).
        distribution : NoneType, scipy.stats distribution, optional
            The `scipy.stats` distribution to use for generating the simulated data. If left
            as None, then the `scipy.stats.uniform` distribution is defaulted. This parameter
            will be overridden by `values` if `values` is not None.
        values : array_like, float, optional
            An array of specified values to use for the data, rather than generating simulated
            data from a probaility distribution. Can also pass a single value.
        loc : array_like, optional
            Location parameter for `scipy.stats` distribution. Default is 0.
        scale : array_like, optional
            Scale parameter for `scipy.stats` continuous distribution. Default is 1.
        random_state : NoneType, int, `numpy.random.RandomState` instance, optional
            Definition of the random state for the generated data. If int or RandomState,
            use it for drawing the random variates. If None, rely on `self.random_state`.
            Default is None.

        """

        self._check_valid_attr(attr)
        self._check_if_cut_set()
        self._check_taus_set(attr)
        self._check_data_size(attr)

        if values is None:
            if distribution is None:
                distribution = stats.uniform

            if "size" in kwargs.keys() and kwargs["size"]!=self.ntraces:
                raise ValueError(
                    f"The inputted size does not match the cut length ({self.ntraces}), "
                    "The size is automatically set, consider not passing it."
                )
            else:
                kwargs["size"] = self.ntraces

            sim_data = distribution.rvs(*args, **kwargs)
        elif np.isscalar(values):
            sim_data = np.ones(self.ntraces) * values
        else:
            if len(values)!=self.ntraces:
                raise ValueError(
                    "The length of the inputted values argument "
                    f"does not match the cut length ({self.ntraces})"
                )
            sim_data = values

        val = getattr(self, attr)
        val.append(sim_data)

    def run_sim(self, savefilepath, convtoamps=None, channel=None, det=None, 
                relcal=None, neventsperdump=1000, basedumpnum=0):
        """
        Method for running the pulse simulation after the data has been generated.

        Parameters
        ----------
        savefilepath : str
            The path where the simulated files should be saved.
        convtoamps : NoneType, float, optional
            The factor to convert the loaded data to units of Amps. If left as None,
            then the conversion factor is loaded automatically.
        channel : NoneType, str, optional
            The name the channel that should be loaded. Only used if filetype=="mid.gz"
        det : NoneType, str, optional
            String that specifies the detector name. Only used if filetype=='mid.gz'.
        neventsperdump : int, optional
            The number of events to be saved per dump file. Default is 1000. This should
            not be made much larger than 1000 to avoid loading too much data into RAM.
        basedumpnum : int, optional
            The base value for the `dumpnum` variable. When saving dumps, the first dump
            will start with this value. Should be an integer of value zero or greater.
            Default is 0.

        """

        self._check_if_cut_set()
        self._check_sim_data()
        self._check_channel_det(channel, det)
        self._check_basedumpnum(basedumpnum)
        convtoamps_auto = self._check_convtoamps(convtoamps, channel, det)

        buildfakepulses(
            self.rq,
            self.cut,
            self.templates,
            self.amplitudes,
            self.tdelay,
            self.basepath,
            taurises=self.taurises,
            taufalls=self.taufalls,
            channels=channel,
            det=det,
            relcal=relcal,
            convtoamps=convtoamps_auto,
            fs=self.fs,
            neventsperdump=neventsperdump,
            filetype=self.filetype,
            lgcsavefile=True,
            savefilepath=savefilepath,
            basedumpnum=basedumpnum,
        )


def buildfakepulses(rq, cut, templates, amplitudes, tdelay, basepath, taurises=None, taufalls=None,
                    channels="PDS1", det="Z1", relcal=None, convtoamps=1, fs=625e3, neventsperdump=1000,
                    basedumpnum=0, filetype="mid.gz", lgcsavefile=False, savefilepath=None):
    """
    Function for building fake pulses by adding a template, scaled to certain amplitudes and
    certain time delays, to an existing trace (typically a random).

    This function calls `_buildfakepulses_seg` which does the heavy lifting.

    Parameters
    ----------
    rq : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for the dataset specified.
    cut : array_like
        A boolean array for the cut that selects the traces that will be loaded from the dump
        files. These traces serve as the underlying data to which a template is added.
    templates : ndarray, list of ndarray
        The template(s) to be added to the traces, assumed to be normalized to a max height
        of 1. The template start time should be centered on the center bin. If a list of templates,
        then each template will be added to the traces in succession, using the corresponding
        `amplitudes` and `tdelay`.
    amplitudes : ndarray, list of ndarray
        The amplitudes, in Amps, by which to scale the template to add to the traces. Must be
        same length as cut. A list of ndarray can be passed, where each ndarray corresponds to
        the amplitudes of the corresponding template in the list of templates.
    tdelay : ndarray, list of ndarray
        The time delay offset, in seconds, by which to shift the template to add to the traces.
        Bin interpolation is implemented for values that are not a multiple the reciprocal of
        the digitization rate. A list of ndarray can be passed, where each ndarray corresponds to
        the tdelays of the corresponding template in the list of templates.
    basepath : str
        The base path to the directory that contains the folders that the event dumps
        are in. The folders in this directory should be the series numbers.
    channels : str, list of str, optional
        A list of strings that contains all of the channels that should be loaded. Only used if
        filetype=='mid.gz'.
    det : str, list of str, optional
        String or list of strings that specifies the detector name. Only used if filetype=='mid.gz'.
        If a list of strings, then should each value should directly correspond to the channel names.
        If a string is inputted and there are multiple channels, then it is assumed that the detector
        name is the same for each channel.
    relcal : ndarray, optional
        An array with the amplitude scalings between channels used when making the total
        If channels is supplied, relcal indices correspond to that list
        Default is all 1 (no relative scaling).
    convtoamps : float, optional
        The factor that the traces should be multiplied by to convert ADC bins to Amperes.
    fs : float, optional
        The sample rate in Hz of the data.
    neventsperdump : int, optional
        The number of events to be saved per dump file.
    basedumpnum : int, optional
        The base value for the `dumpnum` variable. When saving dumps, the first dump
        will start with this value. Should be an integer of value zero or greater.
        Default is 0.
    filetype : str, optional
        The string that corresponds to the file type that will be opened. Supports two
        types: "mid.gz" and "npz". "mid.gz" is the default.
    lgcsavefile : bool, optional
        A boolean flag for whether or not to save the fake data to a file.
    savefilepath : str, optional
        The string that corresponds to the file path where the data will be saved.

    Returns
    -------
    None

    """

    if filetype == "mid.gz" and not HAS_SCDMSPYTOOLS:
        raise ImportError("Cannot use filetype mid.gz because scdmsPyTools is not installed.")

    if isinstance(channels, str):
        channels = [channels]

    if isinstance(det, str):
        det = [det]*len(channels)

    if isinstance(templates, np.ndarray):
        templates = [templates]

    if isinstance(amplitudes, np.ndarray):
        amplitudes = [amplitudes]

    if isinstance(tdelay, np.ndarray):
        tdelay = [tdelay]

    if isinstance(taurises, np.ndarray):
        taurises = [taurises]

    if isinstance(taufalls, np.ndarray):
        taufalls = [taufalls]

    if not len(tdelay) == len(amplitudes) == len(templates):
        raise ValueError(
            "The lists of tdelay, amplitudes, and templates must have the "
            "same number of ndarray."
        )
    elif taurises is not None and len(taurises) != len(tdelay):
        raise ValueError(
            "The lists of taurises, taufalls, tdelay, amplitudes, and templates "
            "must have the same number of ndarray."
        )
    elif taufalls is not None and len(taufalls) != len(tdelay):
        raise ValueError(
            "The lists of taurises, taufalls, tdelay, amplitudes, and templates "
            "must have the same number of ndarray."
        )

    if len(det)!=len(channels):
        raise ValueError("channels and det should have the same length.")

    if len(set(rq.seriesnumber[cut])) > 1:
        raise ValueError(
            "There cannot be multiple series numbers included in the inputted cut."
        )

    if lgcsavefile and savefilepath is None:
        raise ValueError("In order to save the simulated data, you must specify savefilepath.")

    ntraces = np.sum(cut)
    cutlen = len(cut)

    last_dump_ind = -(ntraces%neventsperdump) if ntraces%neventsperdump else None

    indices = np.arange(ntraces, dtype=int)
    nonzerocutinds = np.flatnonzero(cut)

    split_inds = []

    if ntraces//neventsperdump > 0:
        split_inds.extend(np.split(indices[:last_dump_ind], ntraces//neventsperdump))

    if last_dump_ind is not None:
        split_inds.append(indices[last_dump_ind:])

    for ii, c in enumerate(split_inds):
        cut_seg = np.zeros(cutlen, dtype=bool)
        cut_seg[nonzerocutinds[c]] = True

        split_amplitudes = [a[c] for a in amplitudes]
        split_tdelay = [t[c] for t in tdelay]

        if taurises is not None and taufalls is not None:
            split_taurises = [tr[c] for tr in taurises]
            split_taufalls = [tf[c] for tf in taufalls]
        else:
            split_taurises = None
            split_taufalls = None

        _buildfakepulses_seg(
            rq,
            cut_seg,
            templates,
            split_amplitudes,
            split_tdelay,
            basepath,
            taurises=split_taurises,
            taufalls=split_taufalls,
            channels=channels,
            relcal=relcal,
            det=det,
            convtoamps=convtoamps,
            fs=fs,
            dumpnum=ii + 1 + basedumpnum,
            filetype=filetype,
            lgcsavefile=lgcsavefile,
            savefilepath=savefilepath,
        )

    if lgcsavefile:
        _save_truth_info(
            savefilepath,
            basepath=basepath,
            basedumpnum=basedumpnum,
            seriesnumber=rq.seriesnumber[cut],
            eventnumber=rq.eventnumber[cut],
            templates=templates,
            amplitudes=amplitudes,
            tdelay=tdelay,
            taurises=taurises,
            taufalls=taufalls,
            channels=channels,
            relcal=relcal,
            det=det,
            convtoamps=convtoamps,
            fs=fs,
            filetype=filetype,
        )


def _buildfakepulses_seg(rq, cut, templates, amplitudes, tdelay, basepath, taurises=None, taufalls=None,
                         channels="PDS1", relcal=None, det="Z1", convtoamps=1, fs=625e3, dumpnum=1,
                         filetype="mid.gz", lgcsavefile=False, savefilepath=None):
    """
    Hidden helper function for building fake pulses.

    Parameters
    ----------
    rq : pandas.DataFrame
        A pandas DataFrame object that contains all of the RQs for the dataset specified.
    cut : array_like
        A boolean array for the cut that selects the traces that will be loaded from the dump
        files. These traces serve as the underlying data to which a template is added.
    templates : ndarray, list of ndarray
        The template(s) to be added to the traces, assumed to be normalized to a max height
        of 1. The template start time should be centered on the center bin. If a list of templates,
        then each template will be added to the traces in succession, using the corresponding
        `amplitudes` and `tdelay`.
    amplitudes : ndarray, list of ndarray
        The amplitudes, in Amps, by which to scale the template to add to the traces. Must be
        same length as cut. A list of ndarray can be passed, where each ndarray corresponds to
        the amplitudes of the corresponding template in the list of templates.
    tdelay : ndarray, list of ndarray
        The time delay offset, in seconds, by which to shift the template to add to the traces.
        Bin interpolation is implemented for values that are not a multiple the reciprocal of
        the digitization rate. A list of ndarray can be passed, where each ndarray corresponds to
        the tdelays of the corresponding template in the list of templates.
    basepath : str
        The base path to the directory that contains the folders that the event dumps
        are in. The folders in this directory should be the series numbers.
    taurises : ndarray, list of ndarray, NoneType
        If specified, the rise times for the simulated pulses, in seconds. If left as None, then
        the `templates` argument is used.
    taufalls : ndarray, list of ndarray, NoneType
        If specified, the rise times for the simulated pulses, in seconds. If left as None, then
        the `templates` argument is used.
    channels : str, list of str, optional
        A list of strings that contains all of the channels that should be loaded. Only used if
        filetype=='mid.gz'.
    det : str, list of str, optional
        String or list of strings that specifies the detector name. Only used if
        filetype=='mid.gz'. If a list of strings, then should each value should directly
        correspond to the channel names. If a string is inputted and there are multiple
        channels, then it is assumed that the detector name is the same for each channel.
    relcal : ndarray, optional
        An array with the amplitude scalings between channels used when making the total
        If channels is supplied, relcal indices correspond to that list
        Default is all 1 (no relative scaling).
    convtoamps : float, optional
        The factor that the traces should be multiplied by to convert ADC bins to Amperes.
    fs : float, optional
        The sample rate in Hz of the data.
    filetype : str, optional
        The string that corresponds to the file type that will be opened. Supports two
        types: "mid.gz" and "npz". "mid.gz" is the default.
    lgcsavefile : bool, optional
        A boolean flag for whether or not to save the fake data to a file.
    savefilepath : str, optional
        The string that corresponds to the file path that will be saved.

    Returns
    -------
    None

    """

    seriesnumber = list(set(rq.seriesnumber[cut]))[0]

    ntraces = np.sum(cut)
    t, traces, _ = io.getrandevents(
        basepath,
        rq.eventnumber,
        rq.seriesnumber,
        cut=cut,
        channels=channels,
        det=det,
        convtoamps=convtoamps,
        fs=fs,
        ntraces=ntraces,
        filetype=filetype,
    )

    nchan = traces.shape[1]
    nbins = traces.shape[-1]
    t = np.arange(nbins)/fs

    if relcal is None:
        relcal = np.ones(nchan)
    elif nchan != len(relcal):
        raise ValueError('relcal must have length equal to number of channels')

    tracessum = np.sum(traces, axis=1)
    fakepulses = np.zeros(traces.shape)
    newtrace = np.zeros(tracessum.shape[-1])

    for ii in range(ntraces):
        newtrace[:] = tracessum[ii]

        if taurises is not None and taufalls is not None:
            for tr, tf, amp, td in zip(taurises, taufalls, amplitudes, tdelay):
                newtrace += amp[ii] * rp.make_ideal_template(t, tr[ii], tf[ii], offset=td[ii] * fs)
        else:
            for temp, amp, td in zip(templates, amplitudes, tdelay):
                newtrace += amp[ii] * rp.shift(temp, td[ii] * fs)

        # multiply by reciprocal of the relative calibration such that when the processing script 
        # creates the total channel pulse, it will be equal to newtrace
        for jj in range(nchan):
            if relcal[jj]!=0:
                fakepulses[ii, jj] = newtrace/(relcal[jj]*nchan)

    if lgcsavefile:
        if filetype=='npz':
            savefilename = f"{seriesnumber:010}"
            savefilename = savefilename[:6] + '_' + savefilename[6:]
            savefilename = savefilename + "_fake_pulses"

            truthamps = np.stack(amplitudes, axis=1)
            truthtdelay = np.stack(tdelay, axis=1)
            trigtypes = np.zeros((ntraces, 3), dtype=bool)

            io.saveevents_npz(
                traces=fakepulses,
                trigtypes=trigtypes,
                truthamps=truthamps,
                truthtdelay=truthtdelay,
                savepath=savefilepath,
                savename=savefilename,
                dumpnum=dumpnum,
            )

        elif filetype=="mid.gz":
            savefilename = f"{seriesnumber:012}"
            savefilename = savefilename[:8] + '_' + savefilename[8:]

            if np.issubdtype(type(seriesnumber), np.integer):
                snum_str = f"{seriesnumber:012}"
                snum_str = snum_str[:8] + '_' + snum_str[8:]
            else:
                snum_str = seriesnumber

            full_settings_dict = getDetectorSettings(f"{basepath}{snum_str}", "")

            settings_dict = {d: full_settings_dict[d] for d in det}

            for ch, d in zip(channels, det):
                settings_dict[d]["detectorType"] = 710
                settings_dict[d]["phononTraceLength"] = int(settings_dict[d][ch]["binsPerTrace"])
                settings_dict[d]["phononPreTriggerLength"] = settings_dict[d]["phononTraceLength"]//2
                settings_dict[d]["phononSampleRate"] = int(1/settings_dict[d][ch]["timePerBin"])

            events_list = _create_events_list(
                tdelay[0],
                amplitudes[0],
                fakepulses,
                channels,
                det,
                convtoamps,
                seriesnumber,
                dumpnum,
            )

            io.saveevents_midgz(
                events=events_list,
                settings=settings_dict,
                savepath=savefilepath,
                savename=savefilename,
                dumpnum=dumpnum,
            )
        else:
            raise ValueError('Inputted filetype is not supported.')


def _save_truth_info(savefilepath, **kwargs):
    """
    Function for saving the truth information for the pulse simulation to an HDF5 file.

    Parameters
    ----------
    savefilepath : str
        The string that corresponds to the file path where the data will be saved.
    kwargs
        The kwargs are used as a dictionary containing the different truth information
        that will be saved.

    Notes
    -----
    The expected kwargs are:
        savefilepath
        basepath
        basedumpnum
        seriesnumber
        eventnumber
        templates
        amplitudes
        tdelay
        taurises
        taufalls
        channels
        relcal
        det
        convtoamps
        fs
        filetype

    See `rqpy.buildfakepulses` for more information on these values.

    """

    seriesnumber = list(set(kwargs["seriesnumber"]))[0]

    if kwargs['filetype']=="npz":
        savefilename = f"{seriesnumber:010}"
        savefilename = savefilename[:6] + '_' + savefilename[6:]
    elif kwargs['filetype']=="mid.gz":
        savefilename = f"{seriesnumber:012}"
        savefilename = savefilename[:8] + '_' + savefilename[8:]

    basedumpnum = kwargs['basedumpnum']
    dd.io.save(f"{savefilepath}{savefilename}_truth_info_{basedumpnum:04d}.h5", kwargs)


def _round_sig(x, sig=2):
    """
    Function for rounding a float to the specified number of significant figures.

    Parameters
    ----------
    x : float
        Number to round to the specified number of significant figures.
    sig : int
        The number of significant figures to round.

    Returns
    -------
    y : float
        `x` rounded to the number of significant figures specified by `sig`.

    """

    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)


def _create_events_list(pulsetimes, pulseamps, traces, channels, det, convtoamps, seriesnumber, dumpnum):
    """
    Function for structuring the events list correctly for use with `rqpy.io.save_events_midgz` when 
    saving `mid.gz` files.

    Parameters
    ----------
    pulsetimes : ndarray
        The true values of the time of the inputted pulses in the simulated data, in seconds.
    pulseamps : ndarray
        The true values of the amplitudes of the inputted pulses in the simulated data, in Amps.
    traces : ndarray
        The array of traces after adding the specified pulses, in Amps. Has shape (number of traces,
        number of channels, number of bins).
    channels : list of str
        The list of channels that were used when making the simulated data. The order is assumed to correspond
        to the order of channels in `traces`.
    det : list of str
        The corresponding detector IDs for each channel in `channels`.
    convtoamps : float
        The factor that converts from ADC bins to Amps. This is used to convert back to ADC bins.
    seriesnumber : int
        The series number that the data was pulled from before adding the pulses.
    dumpnum : int
        The dump number for this file, used for correctly setting the event number.

    Returns
    -------
    events : list
        List of all of the simulated events with the required fields for saving as a MIDAS file.

    """

    events = list()

    pchans = ["PAS1", "PBS1", "PCS1", "PDS1", "PES1", "PFS1", "PAS2", "PBS2", "PCS2", "PDS2", "PES2", "PFS2"]

    for ii, (pulsetime, pulseamp, trace) in enumerate(zip(pulsetimes, pulseamps, traces)):

        event_dict = {
            'SeriesNumber': seriesnumber,
            'EventNumber' : dumpnum * (10000) + ii,
            'EventTime'   : 0,
            'TriggerType' : 1,
            'SimAvgX'     : 0,
            'SimAvgY'     : _round_sig(pulseamp, sig=6),
            'SimAvgZ'     : 0,
        }

        trigger_dict = {
            'TriggerUnixTime1'  : 0,
            'TriggerTime1'      : 0,
            'TriggerTimeFrac1'  : 0,
            'TriggerDetNum1'    : 0,
            'TriggerAmplitude1' : 0,
            'TriggerStatus1'    : 3,
            'TriggerUnixTime2'  : 0,
            'TriggerTime2'      : 0,
            'TriggerTimeFrac2'  : int(pulsetime/100e-9),
            'TriggerDetNum2'    : 1,
            'TriggerAmplitude2' : (pulseamp/convtoamps).astype(np.int32),
            'TriggerStatus2'    : 1,
            'TriggerUnixTime3'  : 0,
            'TriggerTime3'      : 0,
            'TriggerTimeFrac3'  : 0,
            'TriggerDetNum3'    : 0,
            'TriggerAmplitude3' : 0,
            'TriggerStatus3'    : 8,
        }

        events_dict = {
            'event'   : event_dict,
            'trigger' : trigger_dict,
        }

        for d in set(det):
            events_dict[d] = dict()

        for jj, (ch, d) in enumerate(zip(channels, det)):
            for pch in pchans:
                if pch==ch:
                    events_dict[d][ch] = (trace[jj]/convtoamps).astype(np.int32)
                else:
                    events_dict[d][pch] = np.zeros(trace.shape[-1], dtype=np.int32)

        events.append(events_dict)

    return events
