from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
import numpy as np

__all__ = ['random_floats']


def random_floats(low, high=None, size=None):
    """Return random floats in the half-open interval [0.0, 1.0) between *low* and *high*,
    inclusive.

    Return random floats from the "continuous uniform" distribution over the stated interval.
    If *high* is None (the default), the results are from [0, *low*]

    Parameters
    ----------
    low : float
        Lowest (signed) float to be drawn from the distribution (unless ``high=None``, in which
        case this parameter is the *highest such float).
    high : float, optional
        If provided the largest (signed) float to be drawn from the distribution (see above for
        behavior if ``high=None``).
    size : int or tuple of ints, optional
        Output shape. If the given shape is, e.g., ``(m, n, k)``, then ``m * n * k`` samples are
        drawn. Default is None, in which case a single value is returned.

    Returns
    -------
    out : float or ndarray of floats
        `size`-shaped array of random floats from teh appropriate distribution or a single such
        random float if *size* not provided.

    """
    if high is None:
        high = low
        low = 0
    return low + (np.random.random(size) * (high - low))
