import numpy as np
from scipy import integrate, interpolate
import types


__all__ = [
    'pdf_sampling',
]


def pdf_sampling(function, xrange, nsamples=1000, npoints=10000):
    """
    Produces randomly sampled values based on the arbitrary PDF defined
    by `function`, done using inverse transform sampling.

    Parameters
    ----------
    function : FunctionType
        The 1D probability density function to be randomly sampled from.
    xrange : array_like
        A 1D array of length 2 that defines the range over which the PDF
        in `function` is defined. Outside of this range, it is assumed that
        the PDF is zero.
    nsamples : int, optional
        The number of random samples that we wish to create from the PDF
        defined by `function`.
    npoints : int, optional
        The number of points to use in the numerical integration to evaluate
        the CDF of `function`. This is also the number of points used in the
        interpolation of the inverse of the CDF.

    Returns
    -------
    rvs : ndarray
        The random samples that were taken from the inputted PDF defined by
        `function`. This is 1D array of length `nsamples`.

    Notes
    -----
    For a discussion of inverse transform sampling, see the Wikipedia page:
        https://en.wikipedia.org/wiki/Inverse_transform_sampling

    """

    if not isinstance(function, types.FunctionType):
        raise TypeError("Inputted variable function is not FunctionType.")

    x = np.linspace(xrange[0], xrange[1], num=npoints)
    pdf = function(x)

    cdf = integrate.cumtrapz(pdf, x=x, initial=0.0)
    cdf_normed = cdf / cdf[-1]

    inv_cdf = interpolate.interp1d(cdf_normed, x)

    samples = np.random.rand(nsamples)

    return inv_cdf(samples)
