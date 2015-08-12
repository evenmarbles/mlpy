"""
.. module:: mlpy.optimize.util
   :platform: Unix, Windows
   :synopsis: Optimization function utilities.

.. moduleauthor:: Astrid Jackson <ajackson@eecs.ucf.edu>
"""
from __future__ import division, print_function, absolute_import

import numpy as np


def is_converged(fval, prev_fval, thresh=1e-4, warn=False):
    """Check if an objective function has converged.

    Parameters
    ----------
    fval : float
        The current value.
    prev_fval : float
        The previous value.
    thresh : float
        The convergence threshold.
    warn : bool
        Flag indicating whether to warn the user when the
        fval decreases.

    Returns
    -------
    bool :
        Flag indicating whether the objective function has converged or not.

    Notes
    -----
    The test returns true if the slope of the function falls below the threshold; i.e.

    .. math::

        \\frac{|f(t) - f(t-1)|}{\\text{avg}} < \\text{thresh},

    where

    .. math::

         \\text{avg} = \\frac{|f(t)| + |f(t+1)|}{2}

    .. note::
        Ported from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    converged = False
    delta_fval = abs(fval - prev_fval)
    # noinspection PyTypeChecker
    avg_fval = np.true_divide(abs(fval) + abs(prev_fval) + np.spacing(1), 2)

    if float(delta_fval) / float(avg_fval) < thresh:
        converged = True

    if warn and (fval - prev_fval) < -2 * np.spacing(1):
        print("is_converged: fval decrease, objective decreased!")

    return converged
