"""
.. module:: mlpy.auxiliary.array
   :platform: Unix, Windows
   :synopsis: Numpy array utility functions.

.. moduleauthor:: Astrid Jackson <ajackson@eecs.ucf.edu>
"""
from __future__ import division, print_function, absolute_import

from itertools import product
import numpy as np


def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
    # noinspection PyTypeChecker
    """An accumulation function similar to Matlab's `accumarray` function.

    Parameters
    ----------
    accmap : array_like
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1)
    a : array_like or float or int
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed.
    size : array_like or tuple
        The size of the output array.  If None, the size will be determined
        from `accmap`.
    fill_value : scalar
        The default value for elements of the output array.
    dtype : dtype
        The data type of the output array.  If None, the data type of
        `a` is used.

    Returns
    -------
    array_like :
        The accumulated results.

        The shape of `out` is `size` if `size` is given.  Otherwise the
        shape is determined by the (lexicographically) largest indices of
        the output found in `accmap`.

    Examples
    --------

    >>> from numpy import array, prod, float64
    >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    >>> a
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])

    Sum the diagonals:

    >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
    >>> s = accum(accmap, a)
    array([9, 7, 15])

    A 2D output, from sub-arrays with shapes and positions like this:

    | [ (2,2) (2,1)]
    | [ (1,2) (1,1)]

    >>> accmap = array([
    ...     [[0,0],[0,0],[0,1]],
    ...     [[0,0],[0,0],[0,1]],
    ...     [[1,0],[1,0],[1,1]],
    ... ])

    Accumulate using a product:

    >>> accum(accmap, a, func=prod, dtype=float64)
    array([[ -8.,  18.],
           [ -8.,   9.]])

    Same accmap, but create an array of lists of values:

    >>> accum(accmap, a, func=lambda x: x, dtype='O')
    array([[[1, 2, 4, -1], [3, 6]],
           [[-1, 8], [9]]], dtype=object)

    .. note::
        Adapted from

        | Project: Code from `SciPy Cookbook <http://wiki.scipy.org/Cookbook/AccumarrayLike>`_.
        | Code author: Warren Weckesser
        | License: `CC-Wiki <http://creativecommons.org/licenses/by-sa/3.0/>`_

    """

    # Check for bad arguments and handle the defaults.
    if hasattr(a, "__len__") and accmap.shape[:a.ndim] != a.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = np.float64
        if hasattr(a, "__len__"):
            # noinspection PyUnresolvedReferences
            dtype = a.dtype
    if hasattr(a, "__len__") and accmap.shape == a.shape:
        accmap = np.expand_dims(accmap, -1)

    if not hasattr(a, "__len__"):
        c = np.ascontiguousarray(accmap).view(np.dtype((np.void, accmap.dtype.itemsize * accmap.shape[1])))
        unique_x = np.unique(c).view(accmap.dtype).reshape(-1, accmap.shape[1])

        if size is None:
            adims = unique_x.shape[1]
            size = (adims, adims)
        size = np.atleast_1d(size)
        out = np.zeros(size, dtype=dtype)

        for seq in unique_x:
            cmd = "accmap[np.where("
            idx = ""
            for i, ele in enumerate(seq):
                if i > 0:
                    cmd += " * "
                    idx += ", "
                cmd += "(accmap[:,%d] == %d)" % (i, ele)
                idx += "%d" % ele
            cmd += ")]"
            out[eval(idx)] = eval(cmd).shape[0]

        return out

    adims = tuple(range(a.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in a.shape]):
        indx = tuple(accmap[s])
        # noinspection PyUnresolvedReferences
        val = a[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if not vals[s]:
            out[s] = fill_value
        else:
            # noinspection PyCallingNonCallable
            out[s] = func(vals[s])

    return out


def normalize(a, axis=None, return_scale=False):
    """Normalize the input array to sum to `1`.

    Parameters
    ----------
    a : array_like, shape (`nsamples`, `nfeatures`)
        Non-normalized input data array.
    axis : int
        Dimension along which normalization is performed.

    Returns
    -------
    array_like, shape (`nsamples`, `nfeatures`) :
        An array with values normalized (summing to 1) along the
        prescribed axis.

    Examples
    --------
    >>>

    .. attention::
        The input array `a` is modified inplace.

    """
    a += np.finfo(float).eps
    asum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        asum[asum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        asum.shape = shape
    a = np.true_divide(a, asum)
    # TODO: should return nothing, since the operation is inplace.
    if not return_scale:
        ret = a
    else:
        ret = (a,)
        ret += (asum,)
    return ret


def nunique(x, axis=None):
    """Efficiently count the unique elements of `x` along the given axis.

    Parameters
    ----------
    x : array_like
        The array for which to count the unique elements.
    axis : int
        Dimension along which to count the unique elements.

    Returns
    -------
    int or array_like :
        The number of unique elements along the given axis.

    Examples
    --------
    >>>

    .. note::
        Ported from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    axis = 0 if axis is None else axis
    n = np.sum(np.diff(np.sort(x, axis=axis), axis=axis) > 0, axis=axis) + 1
    return n
