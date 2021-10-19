import numpy as np
from glob import glob
import time
import os
from pathlib import Path
import types
import contextlib
from scipy import stats, signal, interpolate, special, integrate

import rqpy as rp
from rqpy import constants
from rqpy.limit import _upper
import mendeleev


__all__ = [
    "optimuminterval",
    "gauss_smear",
    "drde",
    "drde_max_q",
    "helmfactor",
    "upper",
    "drde_gauss_smear2d",
    "optimuminterval_2dsmear",
]


@contextlib.contextmanager
def _working_directory(path):
    """
    Changes working directory and returns to previous on exit.

    Parameters
    ----------
    path : str
        The directory that the current working directory will temporarily be switched to.

    """

    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def upper(fc, cl=0.9):
    """
    Fortran wrapper function for Steve Yellin's Optimum Interval code `Upper.f`. In this case,
    it calls a version of `UpperLim.f` that allows a larger range of confidence levels.

    Parameters
    ----------
    fc : array_like
        Given the foreground distribution whose shape is known, but whose normalization is
        to have its upper limit total expected number of events determined, fc(0) to fc(N+1),
        with fc(0)=0, fc(N+1)=1, and with  fc(i) the increasing ordered set of cumulative
        probabilities for the foreground distribution for event i, i=1 to N.
    cl : float, optional
        The confidence level desired for the upper limit. Default is 0.9. Can be any value
        between 0.00001 and 0.99999. However, the algorithm requires less than 100 upper
        limit events when outside the range 0.8 to 0.995 in order to work, so an error may
        be raised.

    Returns
    -------
    ulout : float
        The output of the Upper Fortran code, corresponding to the upper limit expected number of
        events. To convert to cross section, the output should be divided by the total rate of the
        signal and multiplied by the expected cross section for that rate.
    endpoints0 : int
        An integer giving the index of FC at which the optimum interval started.
    endpoints1 : int
        An integer giving the index of FC at which the optimum interval ended.

    Notes
    -----
    This is a wrapper around Steve Yellin's Optimum Interval Fortran code, which was compiled via f2py to
    be callable by Python. Because the Fortran code expects look-up tables in the current working directory,
    we need to use a context manager to switch directories to where the look-up tables are when running the
    algorithm.

    Read more about Steve Yellin's Optimum Interval code here:
        - http://titus.stanford.edu/Upper/
        - https://arxiv.org/abs/physics/0203002
        - https://arxiv.org/abs/0709.2701

    """

    file_path = os.path.dirname(os.path.realpath(__file__))

    # make sure fc starts with 0 and ends with 1
    fc_new = fc
    if fc[0]!=0:
        fc_new = np.concatenate(([0], fc_new))
    if fc[-1]!=1:
        fc_new = np.concatenate((fc_new, [1]))

    method = 0
    nexp = 1
    maxp1 = len(fc_new) - 1
    nevts = np.array([maxp1 - 1])
    mu = 1
    icode = 0

    with _working_directory(f"{file_path}/_upper/"):
        ulout = _upper.upper(
            method=method,
            cl=cl,
            nexp=nexp,
            maxp1=maxp1,
            nevts=nevts,
            mu=np.asarray([mu]),
            fc=fc_new[:, np.newaxis],
            icode=icode,
        )

    endpoints = _upper.upperlimcom.endpoints

    return ulout, endpoints[0], endpoints[1]


def helmfactor(er, tm='Si'):
    """
    The analytic nuclear form factor via the Helm approximation.

    Parameters
    ----------
    er : array_like
        The recoil energy to use in the form factor calculation, units of keV.
    tm : str, int, optional
        The target material of the detector. Can be passed as either the atomic symbol, the
        atomic number, or the full name of the element. Default is 'Si'.

    Returns
    -------
    ffactor2 : ndarray
        The square of the dimensionless form factor for the inputted recoil energies and target
        material.

    Notes
    -----
    This form factor uses Helm's approximation to the charge density of the nucleus, as explained by
    Lewin and Smith in section 4 of their paper:
        - https://doi.org/10.1016/S0927-6505(96)00047-3

    """

    er = np.atleast_1d(er)

    hbarc = constants.hbar * constants.c / constants.e * 1e-6 * 1e15 # [MeV fm]
    mn = constants.atomic_mass * constants.c**2 / constants.e * 1e-9 # 1 amu in [GeV]
    atomic_weight = mendeleev.element(tm).atomic_weight

    # dimensionless momentum transfer
    q = np.sqrt(2 * mn * atomic_weight * er) # [MeV]

    # using the parameters defined in L&S
    s = 0.9 # [fm]
    a = 0.52 # [fm]
    c = 1.23 * atomic_weight**(1 / 3) - 0.60 # [fm]

    # approximation of rn [Eq. 4.11 of L&S]
    rn = np.sqrt(c**2 + 7 / 3 * np.pi**2 * a**2 - 5 * s**2)

    qrn = q * rn / hbarc
    qs = q * s / hbarc

    # Helm approximation of form facter [Eq. 4.7 of L&S]
    ffactor2 = (3 * special.spherical_jn(1, qrn) / qrn * np.exp(-qs**2 / 2))**2 

    return ffactor2

def _mixed_tm(tm):
    """
    Helper function for extracting the element names and number
    of them from an inputted chemical formula.

    """

    pos = [i for i, e in enumerate(tm + 'A') if e.isupper()]
    parts = [tm[pos[j]:pos[j + 1]] for j in range(len(pos) - 1)]
    tms = []
    for item in parts:
        for ii, letter in enumerate(item):
            if letter.isdigit():
                tm_temp = [item[:ii], int(item[ii:])]
                break
            elif ii == len(item) - 1:
                tm_temp = [item, 1]
        tms.append(tm_temp)

    return tms


def drde(q, m_dm, sig0, tm='Si'):
    """
    The differential event rate of an expected WIMP.

    Parameters
    ----------
    q : array_like
        The recoil energies at which to calculate the dark matter differential
        event rate. Expected units are keV.
    m_dm : float
        The dark matter mass at which to calculate the expected differential
        event rate. Expected units are GeV.
    sig0 : float
        The dark matter cross section at which to calculate the expected differential
        event rate. Expected units are cm^2.
    tm : str, int, optional
        The target material of the detector. Must be passed as the atomic
        symbol. Can also pass a compound, but must be its chemical formula
        (e.g. sapphire is 'Al2O3'). Default value is 'Si'.

    Returns
    -------
    rate : ndarray
        The expected dark matter differential event rate for the inputted recoil energies,
        dark matter mass, and dark matter cross section. Units are events/keV/kg/day, 
        or "DRU".

    Notes
    -----
    The derivation of the expected dark matter differential event rate is done in Lewin and
    Smith's paper "Review of mathematics, numerical factors, and corrections dark matter experiments
    based on elastic nuclear recoil", which can be found here:
        - https://doi.org/10.1016/S0927-6505(96)00047-3

    The derivation by L&S is incomplete, see Eq. 22 of R. Schnee's paper "Introduction to Dark Matter
    Experiments", which includes the correct rate for `vmin` in the range (`vesc` - `ve`, `vesc` + `ve`)
        - https://arxiv.org/abs/1101.5205

    Another citation for this correction can be found in Savage, et. al.'s paper "Compatibility of
    DAMA/LIBRA dark matter detection with other searches", see Eq. 19. This is a different parameterization,
    but is the same solution.
        - https://doi.org/10.1088/1475-7516/2009/04/010

    """

    totalmassnum = sum([mendeleev.element(t).mass_number * num for t, num in _mixed_tm(tm)])
    rate = sum(
        [mendeleev.element(t).mass_number * num / totalmassnum * _drde(
            q, m_dm, sig0, t,
        ) for t, num in _mixed_tm(tm)]
    )

    return rate


def _drde(q, m_dm, sig0, tm):
    """
    The differential event rate of an expected WIMP for a single target material.
    See `drde` for the full explanation of each parameter.

    """

    q = np.atleast_1d(q) # convert to recoil energy in keV

    v0 = constants.v0_sun # sun velocity about galactic center [m/s]
    ve = constants.ve_orbital # mean orbital velocity of Earth [m/s]
    vesc = constants.vesc_galactic # galactic escape velocity [m/s]
    rho0 = constants.rho0_dm # local DM density [GeV/cm^3]

    a = mendeleev.element(tm).atomic_weight
    mn = constants.atomic_mass * constants.c**2 / constants.e * 1e-9 # nucleon mass (1 amu) [GeV]
    mtarget = a * mn # nucleon mass for tm [GeV]
    r = 4 * m_dm * mtarget / (m_dm + mtarget)**2 # unitless reduced mass parameter
    e0 = 0.5 * m_dm * (v0 / constants.c)**2 * 1e6 # kinetic energy of dark matter [keV]
    vmin = np.sqrt(q / (e0 * r)) * v0 # DM velocity for smallest particle energy to give recoil energy q

    form_factor = helmfactor(q, tm=tm)

    # spin-independent cross section on entire nucleus
    sigma = form_factor * sig0 * a**2 * (mtarget/(m_dm + mtarget))**2 / (mn / (m_dm + mn))**2

    # event rate per unit mass for ve= 0 and vesc = infinity [Eq. 3.1 of L&S]
    r0con = 2 * constants.N_A / np.sqrt(np.pi) * 1e5 * constants.day
    r0 = r0con * sigma * rho0 * v0 / (a * m_dm)

    # ratio of k0/k1 [Eq. 2.2 of L&S]
    k0_over_k1 = 1 / (special.erf(vesc / v0) - 2 / np.sqrt(np.pi) * vesc / v0 * np.exp(-(vesc / v0)**2))

    # rate integrated to infinity [Eq. 3.12 of L&S]
    rate_inf = r0 * np.sqrt(np.pi) * v0 / (4 * e0 * r * ve) * (special.erf((vmin + ve) / v0) - special.erf((vmin - ve) / v0))
    # rate integrated to vesc [Eq. 3.13 of L&S]
    rate_vesc = k0_over_k1 * (rate_inf - r0 / (e0 * r) * np.exp(-(vesc / v0)**2))

    # rate calculation correction to L&S for `vmin` in range (`vesc` - `ve`, `vesc` + `ve`) [Eq. 22 of Schnee]
    rate_inf2 = r0 * np.sqrt(np.pi) * v0 / (4 * e0 * r * ve) * (special.erf(vesc / v0) - special.erf((vmin - ve) / v0))
    rate_high_vmin = k0_over_k1 * (rate_inf2 - r0 / (e0 * r) * (vesc + ve - vmin) / (2 * ve) * np.exp(-(vesc / v0)**2))

    # combine the calculations based on their regions of validity
    rate = np.zeros(q.shape)
    rate[(vmin < vesc - ve) & (vmin > 0)] = rate_vesc[(vmin < vesc - ve) & (vmin > 0)]
    rate[(vmin > vesc - ve) & (vmin < vesc + ve)] = rate_high_vmin[(vmin > vesc - ve) & (vmin < vesc + ve)]

    return rate


def drde_max_q(m_dm, tm='Si'):
    """
    Function for calculating the energy corresponding to the largest nonzero value of the differential rate,
    i.e. `rqpy.limit.drde`.

    Parameters
    ----------
    m_dm : float, ndarray
        The dark matter mass at which to calculate the expected differential
        event rate. Expected units are GeV.
    tm : str, int, optional
        The target material of the detector. Must be passed as the atomic
        symbol. Can also pass a compound, but must be its chemical formula
        (e.g. sapphire is 'Al2O3'). Default value is 'Si'.

    Returns
    -------
    qmax : float, ndarray
        The energy corresponding to the largest nonzero value of the differential rate, where recoil energies
        above this value will have a differential rate of zero.

    """

    qmax = max([_drde_max_q(m_dm, t) for t, num in _mixed_tm(tm)])

    return qmax

def _drde_max_q(m_dm, tm):
    """
    Function for calculating the energy corresponding to the largest nonzero
    value of the differential rate, i.e. `rqpy.limit.drde`. See `drde_max_q` for
    the full documentation.

    """

    a = mendeleev.element(tm).mass_number
    mn = constants.atomic_mass * constants.c**2 / constants.e * 1e-9 # nucleon mass (1 amu) [GeV]
    mtarget = a * mn # nucleon mass for largest_tm [GeV]
    r = 4 * m_dm * mtarget / (m_dm + mtarget)**2 # unitless reduced mass parameter
    e0 = 0.5 * m_dm * (constants.v0_sun / constants.c)**2 * 1e6 # kinetic energy of dark matter [keV]
    qmax = e0 * r * ((constants.vesc_galactic + constants.ve_orbital) / constants.v0_sun)**2

    return qmax

def gauss_smear(x, f, res, nres=1e5, gauss_width=10):
    """
    Function for smearing an array of values by a Gaussian.

    Parameters
    ----------
    x : array_like
        The x-values of the array `f` that will be smeared.
    f : array_like
        The array of value to smear via a Gaussian distribution.
    res : float
        The width of the Gaussian (1 standard deviation) that will be
        used to smear the inputted array. Should have the same units as `x`.
    nres : float, optional
        The size of the array that the Gaussian distribution will be saved to.
        Default is 1e5.
    gauss_width : float, optional
        The number of standard deviations of the Gaussian distribution that the
        smearing will go out to. Default is 10.

    Returns
    -------
    sx : ndarray
        The inputted array `f` after being smeared by the Gaussian distribution.

    """

    x2 = np.linspace(min(x), max(x), num=int(nres))
    spacing = np.mean(np.diff(x2))
    f2 = interpolate.interp1d(x, f)

    xgauss = np.arange(-gauss_width*res, gauss_width*res, spacing)
    gauss = stats.norm.pdf(xgauss, scale=res)

    sce = signal.convolve(f2(x2), gauss, mode="full", method="direct") * spacing
    e_conv = np.arange(-gauss_width * res + x2[0], gauss_width * res + x2[-1], spacing)
    s = interpolate.interp1d(e_conv, sce)

    return s(x)


def optimuminterval(eventenergies, effenergies, effs, masslist, exposure,
                    tm="Si", cl=0.9, res=None, gauss_width=10, verbose=False,
                    drdefunction=None, hard_threshold=0.0):
    """
    Function for running Steve Yellin's Optimum Interval code on an inputted spectrum and efficiency curve.

    Parameters
    ----------
    eventenergies : ndarray
        Array of all of the event energies (in keV) to use for calculating the sensitivity.
    effenergies : ndarray 
        Array of the energy values (in keV) of the efficiency curve.
    effs : ndarray
        Array of the efficiencies (unitless) corresponding to `effenergies`.
        If `drdefunction` argument is provided, the `effs` argument is ignored. It is kept as
        a positional argument for backward compatibility
    masslist : ndarray
        List of candidate DM masses (in GeV/c^2) to calculate the sensitivity at.
    exposure : float
        The total exposure of the detector (kg*days).
    tm : str, int, optional
        The target material of the detector. Must be passed as the atomic
        symbol. Can also pass a compound, but must be its chemical formula
        (e.g. sapphire is 'Al2O3'). Default value is 'Si'.
    cl : float, optional
        The confidence level desired for the upper limit. Default is 0.9. Can be any value
        between 0.00001 and 0.99999. However, the algorithm requires less than 100 upper
        limit events when outside the range 0.8 to 0.995 in order to work, so an error may
        be raised.
    res : float, NoneType, optional
        The detector resolution in units of keV. If passed, then the differential event
        rate of the dark matter is convoluted with a Gaussian with width `res`, which results
        in a smeared spectrum. If left as None, no smearing is performed.
        If `drdefunction` is provided, this argument is ignored
    gauss_width : float, optional
        If `res` is not None, this is the number of standard deviations of the Gaussian
        distribution that the smearing will go out to. Default is 10.
        If `drdefunction` is provided, this argument is ignored
    verbose : bool, optional
        If True, then the algorithm prints out which mass is currently being used in the calculation.
        If False, no information is printed. Default is False.
    drdefunction : list, optional
        List of callables of type float(float). Every element of the list represents the signal model
        rate as a function of reconstructed energy for the corresponding Dark Matter mass from the
        `masslist` and the cross section sigma=10^-41 cm^2. The experiment efficiency must be taken
        into account. The energy unit is keV, the rate unit is 1/keV/kg/day.
        By default (or if None is provided) the standard Lewin&Smith signal model is used with gaussian
        smearing of width `res`, truncated at `gauss_width` standard deviations.
    hard_threshold : float, optional
        The energy value (keV) below which the efficiency is zero.
        This argument is not required in a case of smooth efficiency curve, however it must be provided 
        in a case of step-function-like efficiency.

    Returns
    -------
    sigma : ndarray
        The corresponding cross sections of the sensitivity curve (in cm^2).
    oi_energy0 : ndarray
        The energies in keV at which each optimum interval started.
    oi_energy1 : ndarray
        The energies in keV at which each optimum interval ended.

    Notes
    -----
    This function is a wrapper for Steve Yellin's Optimum Interval code. His code can be found
    here: titus.stanford.edu/Upper/

    Read more about the Optimum Interval code in these two papers:
        - https://arxiv.org/abs/physics/0203002
        - https://arxiv.org/abs/0709.2701

    """

    if np.isscalar(masslist):
        masslist = [masslist]

    eventenergies = np.sort(eventenergies)

    elow = max(hard_threshold, min(effenergies))
    ehigh = max(effenergies)

    en_interp = np.logspace(np.log10(elow), np.log10(ehigh), int(1e5))

    sigma0 = 1e-41

    event_inds = rp.inrange(eventenergies, elow, ehigh)

    sigma = np.ones(len(masslist)) * np.inf
    oi_energy0 = np.zeros(len(masslist))
    oi_energy1 = np.zeros(len(masslist))

    for ii, mass in enumerate(masslist):
        if verbose:
            print(f"On mass {ii+1} of {len(masslist)}.")

        if drdefunction is None:
            exp = effs * exposure

            curr_exp = interpolate.interp1d(
                effenergies, exp, kind="linear", bounds_error=False, fill_value=(0, exp[-1]),
            )
    
            init_rate = drde(
                en_interp, mass, sigma0, tm=tm,
            )
            if res is not None:
                init_rate = gauss_smear(en_interp, init_rate, res, gauss_width=gauss_width)
            rate = init_rate * curr_exp(en_interp)
        else:
            rate = drdefunction[ii](en_interp) * exposure

        integ_rate = integrate.cumtrapz(rate, x=en_interp, initial=0)

        tot_rate = integ_rate[-1]

        x_val_fcn = interpolate.interp1d(
            en_interp,
            integ_rate,
            kind="linear",
            bounds_error=False,
            fill_value=(0, tot_rate),
        )

        x_vals = x_val_fcn(eventenergies[event_inds])

        if tot_rate != 0:
            fc = x_vals/tot_rate
            fc[fc > 1] = 1

            cdf_max = 1 - 1e-6
            possiblewimp = fc <= cdf_max
            fc = fc[possiblewimp]

            if len(fc) == 0:
                fc = np.asarray([0, 1])

            try:
                uloutput, endpoint0, endpoint1 = upper(fc, cl=cl)

                sigma[ii] = (sigma0 / tot_rate) * uloutput

                oi_energy0[ii] = eventenergies[event_inds][possiblewimp][endpoint0-1] if endpoint0>0 else elow # endpoint==0 means the start of the SM integration range
                oi_energy1[ii] = eventenergies[event_inds][possiblewimp][endpoint1-1] if endpoint1-1 < len(fc) else ehigh
            except:
                pass

    return sigma, oi_energy0, oi_energy1

def _norm2d(x0, x1, mu, cov, return_ellipse=False):
    """
    Two-dimensional normal probability density function.

    Parameters
    ----------
    x0 : float, ndarray
        Value or array of values at which to calculate the PDF.
    x1 : float, ndarray
        Value or array of values at which to calculate the PDF.
    mu : float, ndarray
        The center (mean) of the PDF, assumed to be the same for `x0` and `x1`.
    cov : ndarray
        The covariance matrix corresponding to `x0` and `x1`.
    return_ellipse : bool, optional
        Boolean flag for whether or not the `ell` value should also be returned. Default is False.

    Returns
    -------
    norm2d : ndarray
        The two-dimensional normal probability density function
    ell : ndarray, optional
        The chi-squared calculated for each data point, equivalent to `[x0, x1] * inv(cov) * [x0, x1]'`.

    """

    cov_inv = np.linalg.inv(cov)

    # multiplied out 2d normal distribution (for vectorization)
    ell = (x0 - mu)**2 * cov_inv[0, 0] + 2 * (x0 - mu) * (x1 - mu) * cov_inv[1, 0] + (x1 - mu)**2 * cov_inv[1, 1]
    ell = np.atleast_1d(ell)

    norm2d = np.exp(-0.5 * ell) / np.sqrt((2 * np.pi)**2 * np.linalg.det(cov))

    if return_ellipse:
        return norm2d, ell
    else:
        return norm2d

def _norm2d_trunc(x0, x1, mu, cov, nsig):
    """
    Truncated two-dimensional normal probability density function, where we set the PDF to zero
    for any points outside the confidence region ellipse defined by `nsig`.

    Parameters
    ----------
    x0 : float, ndarray
        Value or array of values at which to calculate the PDF.
    x1 : float, ndarray
        Value or array of values at which to calculate the PDF.
    mu : float, ndarray
        The center (mean) of the PDF, assumed to be the same for `x0` and `x1`.
    cov : ndarray
        The covariance matrix corresponding to `x0` and `x1`.
    nsig : float
        The number of sigma outside of which the PDF will be set to zero. This defines
        an elliptical confidence region, whose shape comes from the covariance matrix.

    Returns
    -------
    norm2d : ndarray
        The truncated two-dimensional normal probability density function

    """

    # get the perimeter of the ellipse for the specified number of sigma
    sig = lambda n: stats.norm.cdf(n) - stats.norm.cdf(-n)
    max_ell = stats.chi2.ppf(sig(nsig), 2)

    norm2d, ell = _norm2d(x0, x1, mu, cov, return_ellipse=True)

    inds = ell > max_ell
    norm2d[inds] = 0

    return norm2d

def _gauss2d_integrand(recon_energy, et, e0, delta, cov, nsig, m_dm, sig0, subtract_zero=False, tm="Si"):
    """Helper function for calculating the integrand for doing 2D smearing of the dRdE."""

    step_function = np.heaviside(
        et - delta,
        1,
    )

    normal_dist = _norm2d_trunc(
        recon_energy,
        et,
        e0,
        cov,
        nsig,
    )

    if subtract_zero:
        normal_dist -= _norm2d(
            recon_energy,
            et,
            0,
            cov,
        )

    scattering_rate = drde(
        e0,
        m_dm,
        sig0,
        tm=tm,
    )

    res = step_function * normal_dist * scattering_rate

    res[res < 0] = 0

    return res

def drde_gauss_smear2d(x, cov, delta, m_dm, sig0, nsig=3, tm="Si", subtract_zero=False):
    """
    Function for smearing the differential rate for DM, given that we have a covariance matrix
    for two energy estimators, where we have set a trigger threshold on one and a measured energy
    for the other.

    Parameters
    ----------
    x : float, ndarray
        The measured energies at which to calculate the differential event rate.
    cov : ndarray
        The covariance matrix relating the measured energy and the trigger energy.
    delta : float
        The threshold value (in keV) for the trigger energy.
    m_dm : float
        The dark matter mass at which to calculate the expected differential
        event rate. Expected units are GeV.
    sig0 : float
        The dark matter cross section at which to calculate the expected differential
        event rate. Expected units are cm^2.
    nsig : float, optional
        The number of sigma outside of which the PDF will be set to zero. This defines
        an elliptical confidence region, whose shape comes from the covariance matrix.
    tm : str, int, optional
        The target material of the detector. Must be passed as the atomic
        symbol. Can also pass a compound, but must be its chemical formula
        (e.g. sapphire is 'Al2O3'). Default value is 'Si'.
    subtract_zero : bool, optional
        Option to subtract out the zero-energy multivariate normal distribution in true energy for
        a more conservative estimate of the 2D Gaussian smeared limit. This will have only a small
        effect. Default is False.

    Returns
    -------
    out : ndarray
        The expected dark matter differential rate for the inputted recoil energies,
        dark matter mass, and dark matter cross section, taking into account the smearing
        by a two-dimensional normal distribution. Units are events/keV/kg/day, or "DRU".

    """

    x = np.atleast_1d(x)

    sig = lambda n: stats.norm.cdf(n) - stats.norm.cdf(-n)
    conf = stats.chi2.ppf(sig(nsig), 2)

    cov_inv = np.linalg.inv(cov)

    a = cov_inv[0, 0]
    b = 2 * cov_inv[1, 0]
    c = cov_inv[1, 1]

    # get the deltas of the confidence ellipse for trigger and reconstructed energies
    et_top = np.sqrt(conf / (c - b**2 / (4 * a)))
    ep_top = np.sqrt(conf / (a - b**2 / (4 * c)))

    # get range of nonzero true energies in integral
    start = 0.0
    end = drde_max_q(m_dm)
    d2 = 0.001
    y = np.linspace(start, end, num=int((end - start) / d2) + 1)
    ydiff = np.diff(y).mean()

    out = np.zeros(len(x))

    for ii, val in enumerate(x):
        # define function that we will be integrating over
        func = lambda et, e0: _gauss2d_integrand(
            val,
            et,
            e0,
            delta,
            cov,
            nsig,
            m_dm,
            sig0,
            subtract_zero=subtract_zero,
            tm=tm,
        )

        # get x values inside ellipse for each y value
        etvals = []
        for en in y:
            if rp.inrange(val, en - ep_top, en + ep_top):
                ets = np.linspace(en - et_top, en + et_top, num=100)
                etvals.append([ets, en])

        # evaluate double integral
        if len(etvals) > 0:
            temp_out = 0
            for ets, en in etvals:
                temp_out += np.sum(func(ets, en)) * np.diff(ets).mean() * ydiff
            out[ii] = temp_out

    return out

def optimuminterval_2dsmear(eventenergies, masslist, passagefraction, exposure,
                            cov, delta, tm="Si", cl=0.9, nsig=3, verbose=False,
                            npts=1000, subtract_zero=False):
    """
    Function for running Steve Yellin's Optimum Interval code on an
    inputted spectrum, using the two-dimensional normal distribution
    defined by the inputted covariance matrix to model the trigger
    efficiency. This is a more complicated version of
    `rqpy.limit.optimuminterval`.

    Parameters
    ----------
    eventenergies : ndarray
        Array of all of the event energies (in keV) to use for
        calculating the sensitivity.
    masslist : ndarray
        List of candidate DM masses (in GeV/c^2) to calculate the upper
        limit at.
    passagefraction : float, functionType
        The passage fraction of the cuts being applied to the data.
        Excludes the trigger efficiency, since that is wrapped up in the
        2D smearing. If a float, then it should be a number between 0
        and 1, meaning that the passage fraction is energy independent.
        If a function, then the input should be in units of keV.
    exposure : float
        The total exposure of the detector (kg*days).
    cov : ndarray
        The covariance matrix relating the measured/reconstructed energy
        and the trigger energy (both in keV).
    delta : float
        The threshold value (in keV) for the trigger energy.
    tm : str, int, optional
        The target material of the detector. Must be passed as the atomic
        symbol. Can also pass a compound, but must be its chemical formula
        (e.g. sapphire is 'Al2O3'). Default value is 'Si'.
    cl : float, optional
        The confidence level desired for the upper limit. Default is
        0.9. Can be any value between 0.00001 and 0.99999. However, the
        algorithm requires less than 100 upper limit events when outside
        the range 0.8 to 0.995 in order to work, so an error may be
        raised.
    nsig : float
        The number of sigma outside of which the two-dimensional normal
        PDF defined by the inputted covariance matrix will be set to
        zero. This defines an elliptical confidence region. This is used
        to restrict the amount of smearing that is applied to the DM
        spectrum to avoid calculate artificially low upper limits.
    verbose : bool, optional
        If True, then the algorithm prints out which mass is currently
        being used in the calculation. If False, no information is
        printed. Default is False.
    npts : float, optional
        The number of energies at which to evaluate the smeared
        differential rate. Large values result in long computation
        times. Default is 1e3.
    subtract_zero : bool, optional
        Option to subtract out the zero-energy multivariate normal
        distribution in true energy for a more conservative estimate of
        the 2D Gaussian smeared limit. This will have only a small
        effect. Default is False.

    Returns
    -------
    sigma : ndarray
        The corresponding cross sections of the sensitivity curve (in
        cm^2).
    oi_energy0 : ndarray
        The energies in keV at which each optimum interval started.
    oi_energy1 : ndarray
        The energies in keV at which each optimum interval ended.

    Notes
    -----
    This function is a wrapper for Steve Yellin's Optimum Interval code.
    His code can be found here: titus.stanford.edu/Upper/

    Read more about the Optimum Interval code in these two papers:
        - https://arxiv.org/abs/physics/0203002
        - https://arxiv.org/abs/0709.2701

    """

    if np.isscalar(masslist):
        masslist = [masslist]

    eventenergies = np.sort(eventenergies)

    elow = max(0.001, min(eventenergies))
    ehigh = max(eventenergies)

    en_interp = np.logspace(
        np.log10(0.9 * elow), np.log10(1.1 * ehigh), npts,
    )

    delta_e = np.concatenate(([(en_interp[1] - en_interp[0])/2],
                              (en_interp[2:] - en_interp[:-2])/2,
                              [(en_interp[-1] - en_interp[-2])/2]))

    sigma0 = 1e-41

    event_inds = rp.inrange(eventenergies, elow, ehigh)
    inlim = rp.inrange(en_interp, elow, ehigh)

    sigma = np.ones(len(masslist)) * np.inf
    oi_energy0 = np.zeros(len(masslist))
    oi_energy1 = np.zeros(len(masslist))

    for ii, mass in enumerate(masslist):
        if verbose:
            print(f"On mass {ii+1} of {len(masslist)}.")

        init_rate = drde_gauss_smear2d(
            en_interp,
            cov,
            delta,
            mass,
            sigma0,
            nsig=nsig,
            tm=tm,
            subtract_zero=subtract_zero,
        )

        if isinstance(passagefraction, types.FunctionType):
            rate = init_rate * exposure * passagefraction(en_interp)
        else:
            rate = init_rate * exposure * passagefraction

        integ_rate = integrate.cumtrapz(
            rate[inlim], x=en_interp[inlim], initial=0,
        )

        tot_rate = integ_rate[-1]

        x_val_fcn = interpolate.interp1d(
            en_interp[inlim],
            integ_rate,
            kind="linear",
            bounds_error=False,
            fill_value=(0, tot_rate),
        )

        x_vals = x_val_fcn(eventenergies[event_inds])

        if tot_rate != 0:
            fc = x_vals/tot_rate
            fc[fc > 1] = 1

            cdf_max = 1 - 1e-6
            possiblewimp = fc <= cdf_max
            fc = fc[possiblewimp]

            if len(fc) == 0:
                fc = np.asarray([0, 1])

            try:
                uloutput, endpoint0, endpoint1 = upper(fc, cl=cl)
                sigma[ii] = (sigma0 / tot_rate) * uloutput
                energies_used = eventenergies[event_inds][possiblewimp]
                oi_energy0[ii] = energies_used[endpoint0]

                if endpoint1 < len(fc):
                    oi_energy1[ii] = energies_used[endpoint1]
                else:
                    oi_energy1[ii] = energies_used[-1]
            except:
                pass

    return sigma, oi_energy0, oi_energy1
