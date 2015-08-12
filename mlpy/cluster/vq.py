"""
.. module:: mlpy.cluster.vq
   :platform: Unix, Windows
   :synopsis: K-means clustering and vector quantization.

.. moduleauthor:: Astrid Jackson <ajackson@eecs.ucf.edu>
"""
from __future__ import division, print_function, absolute_import

import numpy as np

from ..stats import sq_distance, partitioned_mean, partitioned_sum, multivariate_normal
from ..optimize.utils import is_converged


def kmeans(x, k, n_iter=None, thresh=None, mean=None, fn_plot=None, return_assignment=False, return_err_hist=False,
           verbose=False):
    """Hard cluster data using kmeans.

    Parameters
    ----------
    x : array_like, shape (`n`, `dim`)
        List of dim-dimensional data points. Each row corresponds to a
        single data point.
    k : int
        The number of clusters to fit.
    n_iter : int, optional
        Number of iterations to perform. Default is 100.
    thresh : float, optional
        Convergence threshold. Default is 1e-3.
    mean : array_like, shape (ncomponents,), optional
        Initial guess for the cluster centers.
    fn_plot : callable, optional
        A plotting callback function.
    return_assignment : bool, optional
        Whether to return the assignments or not. Default is False.
    return_err_hist : bool, optional
        Whether to return the error history. Default is False.
    verbose : bool, optional
        Controls if debug information is printed to the console.
        Default is False.

    Returns
    -------
    ndarray or tuple :
        The cluster centers and optionally the assignments and error history.

    Examples
    --------
    >>> from mlpy.cluster.vq import kmeans


    .. note::
        Ported from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    optional_returns = return_assignment or return_err_hist

    n_iter = 100 if n_iter is None else n_iter
    thresh = 1e-3 if thresh is None else thresh

    n = x.shape[0]

    # Initialize
    if mean is None:
        # Initialize using k data points chosen at random
        perm = np.random.permutation(range(n))

        # In the unlikely event of a tie, ensure the means are different
        v = np.var(x, 0)
        noise = multivariate_normal.rvs(np.zeros(v.size), 0.01 * np.diag(v), size=k)
        mean = x[perm[0:k]] + noise

    # Setup loop
    i = 0
    err_hist = np.zeros(n_iter)
    prev_err = np.inf

    assign = None

    while True:
        dist = sq_distance(x, mean)
        assign = np.argmin(dist, 1)
        mean = partitioned_mean(x, assign, k)
        current_err = np.sum(np.min(partitioned_sum(dist, assign, k), 1)) / n
        """:type: float"""

        # display progress
        err_hist[i] = current_err
        if fn_plot:
            fn_plot(x, mean, assign, current_err, i)
        if verbose:
            print("iteration {0}, err={1}".format(i + 1, current_err))

        # check convergence
        if is_converged(current_err, prev_err, thresh=thresh) or i >= n_iter - 1:
            break

        i += 1
        prev_err = current_err

    err_hist = err_hist[0:i + 1]

    if not optional_returns:
        ret = mean
    else:
        ret = (mean,)
        if return_assignment:
            ret += (assign,)
        if return_err_hist:
            ret += (err_hist,)
    return ret
