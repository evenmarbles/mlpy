from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
# noinspection PyPackageRequirements
from sklearn.utils.extmath import logsumexp

from ..auxiliary.array import nunique

__all__ = ['is_posdef', 'randpd', 'stacked_randpd', 'normalize_logspace', 'sq_distance',
           'partitioned_mean', 'partitioned_cov', 'partitioned_sum', 'shrink_cov',
           'canonize_labels']


def is_posdef(a):
    """Test if matrix `a` is positive definite.

    The method uses Cholesky decomposition to determine if
    the matrix is positive definite.

    Parameters
    ----------
    a : ndarray
        A matrix.

    Returns
    -------
    bool :
        Whether the matrix is positive definite.

    Examples
    --------
    >>> is_posdef()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    try:
        np.linalg.cholesky(np.asarray(a))
        return True
    except np.linalg.LinAlgError:
        return False


def randpd(dim):
    """Create a random positive definite matrix of size `dim`-by-`dim`.

    Parameters
    ----------
    dim : int
        The dimension of the matrix to create.

    Returns
    -------
    ndarray :
        A `dim`-by-`dim` positive definite matrix.

    Examples
    --------
    >>> randpd()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    x = np.random.randn(dim, dim)
    a = x * x.T
    while not is_posdef(a):
        a = a + np.diag(0.001 * np.ones(dim))

    return a


def stacked_randpd(dim, k, p=0):
    """Create stacked positive definite matrices.

    Create multiple random positive definite matrices of size
    dim-by-dim and stack them.

    Parameters
    ----------
    dim : int
        The dimension of each matrix.
    k : int
        The number of matrices.
    p : int
        The diagonal value of each matrix.

    Returns
    -------
    ndarray :
        Multiple stacked random positive definite matrices.

    Examples
    --------
    >>> stacked_randpd()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    s = np.zeros((k, dim, dim))
    for i in range(k):
        s[i] = randpd(dim) + np.diag(p * np.ones(dim))

    return s


def normalize_logspace(a):
    """Normalizes the array `a` in the log domain.

    Each row of `a` is a log discrete distribution. Returns
    the array normalized in the log domain while minimizing the
    possibility of numerical underflow.

    Parameters
    ----------
    a : ndarray
        The array to normalize in the log domain.

    Returns
    -------
    a : ndarray
        The array normalized in the log domain.
    lnorm : float
        log normalization constant.

    Examples
    --------
    >>> normalize_logspace()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    l = logsumexp(a, 1)
    y = a.T - l
    return y.T, l


def sq_distance(p, q, p_sos=None, q_sos=None):
    """Efficiently compute squared Euclidean distances between stats of vectors.

    Compute the squared Euclidean distances between every d-dimensional point
    in `p` to every `d`-dimensional point in q. Both `p` and `q` are n-point-by-n-dimensions.

    Parameters
    ----------
    p : array_like, shape (`n`, `dim`)
        Array where `n` is the number of points and `dim` is the number of
        dimensions.
    q : array_like, shape (`n`, `dim`)
        Array where `n` is the number of points and `dim` is the number of
        dimensions.
    p_sos : array_like, shape (`dim`,)
    q_sos : array_like, shape (`dim`,)

    Returns
    -------
    ndarray :
        The squared Euclidean distance.

    Examples
    --------
    >>> sq_distance()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    p_sos = np.sum(np.power(p, 2), 1) if p_sos is None else p_sos
    # noinspection PyTypeChecker
    q_sos = np.sum(np.power(q, 2), 1) if q_sos is None else q_sos

    # noinspection PyUnresolvedReferences
    n = q_sos.shape[0]
    # noinspection PyUnresolvedReferences
    return (q_sos.reshape((n, 1)) + p_sos).T - 2 * np.dot(p, q.T)


def partitioned_mean(x, y, c=None, return_counts=False):
    """Mean of groups.

    Groups the rows of `x` according to the class labels in y and
    takes the mean of each group.

    Parameters
    ----------
    x : array_like, shape (`n`, `dim`)
        The data to group, where `n` is the number of data points and
        `dim` is the dimensionality of each data point.
    y : array_like, shape (`n`,)
        The class label for each data point.
    return_counts : bool
        Whether to return the number of elements in each group or not.

    Returns
    -------
    mean : array_like
        The mean of each group.
    counts : int
        The number of elements in each group.

    Examples
    --------
    >>> partitioned_mean()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    c = nunique(y) if c is None else c

    dim = x.shape[1]
    m = np.zeros((c, dim))

    for i in range(c):
        ndx = y == i
        m[i] = np.mean(x[ndx], 0)

    if not return_counts:
        ret = m
    else:
        ret = (m,)
        # noinspection PyTupleAssignmentBalance
        _, counts = np.unique(y, return_counts=True)
        ret += (counts,)

    return ret


def partitioned_cov(x, y, c=None):
    """Covariance of groups.

    Partition the rows of `x` according to class labels in `y` and
    take the covariance of each group.

    Parameters
    ----------
    x : array_like, shape (`n`, `dim`)
        The data to group, where `n` is the number of data points and
        `dim` is the dimensionality of each data point.
    y : array_like, shape (`n`,)
        The class label for each data point.
    c : int
        The number of components in `y`.

    Returns
    -------
    cov : array_like
        The covariance of each group.

    Examples
    --------
    >>> partitioned_cov()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    .. warning::
        Implementation of this function is not finished yet.

    """
    c = nunique(y) if c is None else c

    dim = x.shape[1]
    cov = np.zeros((c, dim, dim))

    for i in range(c):
        cov[i] = np.cov(x[y == c])


def partitioned_sum(x, y, c=None):
    """Sums of groups.

    Groups the rows of `x` according to the class labels in `y`
    and sums each group.

    Parameters
    ----------
    x : array_like, shape (`n`, `dim`)
        The data to group, where `n` is the number of data points and
        `dim` is the dimensionality of each data point.
    y : array_like, shape (`n`,)
        The class label for each data point.
    c : int
        The number of components in `y`.

    Returns
    -------
    sums : array_like
        The sum of each group.

    Examples
    --------
    >>> partitioned_sum()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    c = nunique(y) if c is None else c
    # noinspection PyTypeChecker
    return np.dot(np.arange(0, c).reshape(c, 1) == y, x)


def shrink_cov(x, return_lambda=False, return_estimate=False):
    """Covariance shrinkage estimation.

    Ledoit-Wolf optimal shrinkage estimator for cov(X)
    :math:`C = \\lambda*t + (1 - \\lambda) * s`
    using the diagonal variance 'target' t=np.diag(s) with the
    unbiased sample cov `s` as the unconstrained estimate.

    Parameters
    ----------
    x : array_like, shape (`n`, `dim`)
        The data, where `n` is the number of data points and
        `dim` is the dimensionality of each data point.
    return_lambda : bool
        Whether to return lambda or not.
    return_estimate : bool
        Whether to return the unbiased estimate or not.

    Returns
    -------
    C : array
        The shrunk final estimate
    lambda_ : float, optional
        Lambda
    estimate : array, optional
        Unbiased estimate.

    Examples
    --------
    >>> shrink_cov()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    optional_returns = return_lambda or return_estimate

    n, p = x.shape

    x_mean = np.mean(x, 0)
    x = x - x_mean

    # noinspection PyTypeChecker
    s = np.asarray(np.dot(x.T, x) / (n - 1))      # unbiased estimate
    s_bar = (n - 1) * s / n

    s_var = np.zeros((p, p))
    for i in range(n):
        # noinspection PyTypeChecker
        s_var += np.power(x[i].reshape(p, 1) * x[i] - s_bar, 2)
    s_var = np.true_divide(n, (n - 1)**3) * s_var

    # calculate optimal shrinkage
    o_shrink = np.triu(np.ones((p, p))) - np.eye(p)

    # Ledoit-Wolf formula
    lambda_ = np.sum(s_var[o_shrink.astype(np.bool)]) / np.sum(np.power(s[o_shrink.astype(np.bool)], 2))

    # bound-constrain lambda
    lambda_ = np.max([0, np.min([1, lambda_])])

    # shrunk final estimate C
    c = lambda_ * np.diag(np.diag(s)) + (1 - lambda_) * s

    if not optional_returns:
        ret = c
    else:
        ret = (c,)
        if return_lambda:
            ret += (lambda_,)
        if return_estimate:
            ret += (s,)
    return ret


def canonize_labels(labels, support=None):
    """Transform labels to 1:k.

    The size of canonized is the same as ladles but every label is
    transformed to its corresponding 1:k. If labels does not span
    the support, specify the support explicitly as the 2nd argument.

    Parameters
    ----------
    labels : array_like
    support : optional

    Returns
    -------
    Transformed labels.

    Examples
    --------
    >>> canonize_labels()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    .. warning::
        This is only a stub function. Implementation is still missing
    """
    raise NotImplementedError
