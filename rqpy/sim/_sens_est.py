import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special

import rqpy as rp
import mendeleev


__all__ = [
    "calculate_substrate_mass",
    "trigger_pdf",
    "SensEst",
]


def calculate_substrate_mass(vol, tm):
    """
    Helper function for calculating the mass of substrate given its
    volume.

    Parameters
    ----------
    vol : float
        The volume of the substrate that we want to know the mass of,
        in units of meters^3.
    tm : str, int
        The target material of the detector. Can be passed as either
        the atomic symbol, the atomic number, or the full name of the
        element. Default is 'Si'.

    Returns
    -------
    mass : float
        The mass, in kg, of the substrate based on the inputted volume
        and target material.

    """

    conv_factor = rp.constants.centi**3 * rp.constants.kilo
    # density in kg/m^3
    rho = mendeleev.element(tm).density / conv_factor
    mass = rho * vol

    return mass


def trigger_pdf(x, sigma, n_win):
    """
    Function for calculating the expected PDF due to a finite trigger
    window, based on the Optimum Filter method. Outputs in units of
    1 / [units of `sigma`].

    Parameters
    ----------
    x : ndarray
        The values at which the PDF will be evaluated. Units are
        arbitrary, but note that this function is usually used with
        regards to energy.
    sigma : float
        The detector resolution in the same units of `x`.
    n_win : float
        The number of independent samples in a trigger search window.
        If there is correlated noise, this value can be shorter than
        the number of samples in the trigger window.

    Returns
    -------
    pdf : ndarray
        The PDF for noise triggers from the given inputs.

    """

    normal_dist = lambda xx: stats.norm.pdf(xx, scale=sigma)
    erf_scale = lambda xx: special.erf(xx / np.sqrt(2 * sigma**2))

    pdf = n_win * normal_dist(x) * (0.5 * (1 + erf_scale(x)))**(n_win - 1)

    return pdf


class SensEst(object):
    """
    Class for setting up and running an estimate of the sensitivity of
    a device, given expected backgrounds.

    Attributes
    ----------
    m_det : float
        Mass of the detector in kg.
    tm : str, int
        The target material of the detector. Can be passed as either
        the atomic symbol, the atomic number, or the full name of the
        element.
    exposure : float
        The total exposure of the detector in units of kg*days.

    """

    def __init__(self, m_det, time_elapsed, eff=1, tm="Si"):
        """
        Initialization of the SensEst class.

        Parameters
        ----------
        m_det : float
            Mass of the detector in kg.
        time_elapsed : float
            The time elapsed for the simulated experiment, in days.
        eff : float, optional
            The estimated efficiency due to data selection, live time
            losses, etc. Default is 1.
        tm : str, int, optional
            The target material of the detector. Can be passed as
            either the atomic symbol, the atomic number, or the full
            name of the element. Default is 'Si'.

        """

        self.m_det = m_det
        self.tm = tm
        self.exposure = m_det * time_elapsed * eff

        self._backgrounds = []

    def add_flat_bkgd(self, flat_rate):
        """
        Method for adding a flat background to the simulation.

        Parameters
        ----------
        flat_rate : float
            The flat background rate, in units of events/kg/kev/day
            (DRU).

        """

        flat_bkgd = lambda x: flat_rate * np.ones(len(x))
        self._backgrounds.append(flat_bkgd)


    def add_noise_bkgd(self, sigma, n_win, fs):
        """
        Method for adding a noise background to the simulation.

        Parameters
        ----------
        sigma : float
            The detector resolution in units of keV.
        n_win : float
            The number of independent samples in a trigger search
            window. If there is correlated noise, this value can be
            shorter than the number of samples in the trigger window.
        fs : float
            The digitization rate of the data being used in the trigger
            algorithm, units of Hz.

        """

        norm = self.m_det * n_win / fs / rp.constants.day
        noise_bkgd = lambda x: trigger_pdf(x, sigma, n_win) / norm
        self._backgrounds.append(noise_bkgd)


    def add_dm_bkgd(self, m_dm, sig0):
        """
        Method for adding a DM background to the simulation.

        Parameters
        ----------
        m_dm : float
            The dark matter mass at which to calculate the expected
            differential event rate. Expected units are GeV/c^2.
        sig0 : float
            The dark matter cross section at which to calculate the
            expected differential event rate. Expected units are cm^2.

        """

        dm_bkgd = lambda x: rp.limit.drde(x, m_dm, sig0, tm=self.tm)
        self._backgrounds.append(dm_bkgd)


    def add_arb_bkgd(self, function):
        """
        Method for adding an arbitrary background to the simulation.

        Parameters
        ----------
        function : FunctionType
            A function that returns a background rate in units of
            events/kg/keV/day when inputted energy, where the energies
            are in units of keV.

        """

        self._backgrounds.append(function)


    def reset_sim(self):
        """Method for resetting the simulation to its initial state."""

        self._backgrounds = []


    def run_sim(self, threshold, e_high, e_low=1e-6, m_dms=None, nexp=1, npts=1000,
                plot_bkgd=False):
        """
        Method for running the simulation for getting the sensitivity
        estimate.

        Parameters
        ----------
        threshold : float
            The energy threshold of the experiment, units of keV.
        e_high : float
            The high energy cutoff of the analysis range, as we need
            some cutoff to the event energies that we generate.
        m_dms : ndarray, optional
            Array of dark matter masses (in GeV/c^2) to run the Optimum
            Interval code. Default is 50 points from 0.05 to 2 GeV/c^2.
        nexp : int, optional
            The number of experiments to run - the median of the
            outputs will be taken. Recommended to set to 1 for
            diagnostics, which is default.
        npts : int, optional
            The number of points to use when interpolating the
            simulated background rates. Default is 1000.
        plot_bkgd : bool, optional
            Option to plot the background being used on top of the
            generated data, for diagnostic purposes. If `nexp` is
            greater than 1, then only the first generated dataset is
            plotted.

        Returns
        -------
        m_dms : ndarray
            The dark matter masses in GeV/c^2 that upper limit was set
            at.
        sig : ndarray
            The cross section in cm^2 that the upper limit was
            determined to be.

        """

        sigs = []

        if m_dms is None:
            m_dms = np.geomspace(0.5, 2, num=50)

        en_interp = np.geomspace(e_low, e_high, num=npts)

        for ii in range(nexp):
            evts_sim = self._generate_background(
                en_interp, plot_bkgd=plot_bkgd and ii==0,
            )

            sig_temp, _, _ = rp.limit.optimuminterval(
                evts_sim[evts_sim >= threshold],
                en_interp,
                np.heaviside(en_interp - threshold, 1),
                m_dms,
                self.exposure,
                tm=self.tm,
                hard_threshold=threshold,
            )

            sigs.append(sig_temp)

        sig = np.median(np.stack(sigs, axis=1), axis=1)

        return m_dms, sig

    def generate_background(self, e_high, e_low=0, npts=1000,
                            plot_bkgd=False):
        """
        Method for generating events based on the inputted background.

        Parameters
        ----------
        e_high : float
            The high energy cutoff of the analysis range, as we need
            some cutoff to the event energies that we generate.
        e_low : float, optional
            The low energy cutoff of the analysis range, default is 0.
        npts : int, optional
            The number of points to use when interpolating the
            simulated background rates. Default is 1000.
        plot_bkgd : bool, optional
            Option to plot the background being used on top of the
            generated data, for diagnostic purposes. If `nexp` is
            greater than 1, then only the first generated dataset is
            plotted.

        Returns
        -------
        evts_sim : ndarray
            The array of all the simulated events based on the inputted
            backgrounds. Units are keV.

        Raises
        ------
        ValueError
            If `self._backgrounds` is an empty list (no backgrounds
            have been added).

        """

        en_interp = np.geomspace(e_low, e_high, num=npts)
        evts_sim = self._generate_background(en_interp, plot_bkgd=plot_bkgd)

        return evts_sim

    def _generate_background(self, en_interp, plot_bkgd=False):
        """
        Hidden method for generating events based on the inputted
        background.

        Parameters
        ----------
        en_interp : ndarray
            The energies at which the total simulated background rate
            will be interpolated, in units of keV.
        plot_bkgd : bool, optional
            Option to plot the background being used on top of the
            generated data, for diagnostic purposes. If `nexp` is
            greater than 1, then only the first generated dataset is
            plotted.

        Returns
        -------
        evts_sim : ndarray
            The array of all the simulated events based on the inputted
            backgrounds. Units are keV.

        Raises
        ------
        ValueError
            If `self._backgrounds` is an empty list (no backgrounds
            have been added).

        """

        if len(self._backgrounds) == 0:
            raise ValueError(
                "No backgrounds have been added, "
                "add some using the methods of SensEst."
            )

        e_high = en_interp.max()
        npts = len(en_interp)

        tot_bkgd = np.zeros(npts)

        for bkgd in self._backgrounds:
            tot_bkgd += bkgd(en_interp)

        tot_bkgd_func = lambda x: np.stack(
            [bkgd(x) for bkgd in self._backgrounds], axis=1,
        ).sum(axis=1)

        rtot = np.trapz(tot_bkgd_func(en_interp), x=en_interp)

        nevts_exp = rtot * self.exposure
        nevts_sim = np.random.poisson(nevts_exp)

        evts_sim = rp.sim.pdf_sampling(
            tot_bkgd_func, (0, e_high), npoints=npts, nsamples=nevts_sim,
        )

        if plot_bkgd:
            self._plot_bkgd(evts_sim, en_interp, tot_bkgd_func)

        return evts_sim

    def _plot_bkgd(self, evts, en_interp, tot_bkgd_func):
        """
        Hidden Method for plotting the generated events on top of the
        inputted backgrounds.

        """

        ratecomp = rp.RatePlot(
            (en_interp.min(), en_interp.max()), figsize=(9, 6),
        )
        ratecomp.add_data(
            evts,
            self.exposure,
            label="Simulated Events Spectrum",
        )

        if len(self._backgrounds) > 1:
            for ii, bkgd in enumerate(self._backgrounds):
                ratecomp.ax.plot(
                    en_interp,
                    bkgd(en_interp),
                    linestyle='--',
                    label=f"Background {ii + 1}",
                )

        ratecomp.ax.plot(
            en_interp,
            tot_bkgd_func(en_interp),
            linestyle='--',
            label="Total Background",
        )

        ratecomp._update_colors('--')

        ratecomp.ax.set_xlabel("Event Energy [keV]")
        ratecomp.ax.set_title("Spectrum of Simulated Events")
        ratecomp.ax.set_ylim(
            tot_bkgd_func(en_interp).min() * 0.1,
            tot_bkgd_func(en_interp).max() * 10,
        )

        ratecomp.ax.legend(fontsize=14)
        list_of_text = [
            ratecomp.ax.title,
            ratecomp.ax.xaxis.label,
            ratecomp.ax.yaxis.label,
        ]

        for item in list_of_text:
            item.set_fontsize(14)

        ratecomp.fig.tight_layout()
