from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
from scipy import linalg, special
from scipy.misc import doccer

__all__ = ["multivariate_normal", "multivariate_student", "invwishart", "normal_invwishart", "multigammaln"]


def multigammaln(a, d):
    """
    Returns the log of multivariate gamma, also sometimes called the
    generalized gamma.

    Parameters
    ----------
    a : ndarray
        The multivariate gamma is computed for each item of `a`.
    d : int
        The dimension of the space fo integration.

    Returns
    -------
    ndarray :
        The values of the log multivariate gamma at the given points `a`.

    Notes
    -----

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    a = np.asarray(a)
    if not np.isscalar(d) or (np.floor(d) != d):
        raise ValueError("d should be a positive integer (dimension)")
    if np.any(a <= 0.5 * (d - 1)):
        # noinspection PyStringFormat
        raise ValueError("condition a (%f) > 0.5 * (d-1) (%f) not met"
                         % (a, 0.5 * (d-1)))

    res = (d * (d - 1) * 0.25) * np.log(np.pi)
    return res + np.sum(special.gammaln(a + 0.5 * (1 - np.arange(1, d))))


def _process_parameters(dim=None, mean=None, cov=None):
    """
    Infer dimensionality from mean or covariance matrix, ensure that
    mean and covariance are full vector resp. matrix.

    """
    # Try to infer dimensionality
    if dim is None:
        if mean is None:
            if cov is None:
                dim = 1
            else:
                cov = np.asarray(cov, dtype=float)
                if cov.ndim < 2:
                    dim = 1
                else:
                    dim = cov.shape[0]
        else:
            mean = np.asarray(mean, dtype=float)
            dim = mean.size
    else:
        if not np.isscalar(dim):
            raise ValueError("Dimension of random variable must be a scalar.")

    # Check input sizes and return full arrays for mean and cov if necessary
    if mean is None:
        mean = np.zeros(dim)
    mean = np.asarray(mean, dtype=float)

    if cov is None:
        cov = 1.0
    cov = np.asarray(cov, dtype=float)

    if dim == 1:
        mean.shape = (1,)
        cov.shape = (1, 1)

    if mean.ndim != 1 or mean.shape[0] != dim:
        raise ValueError("Array 'mean' must be a vector of length %d." % dim)
    if cov.ndim == 0:
        cov = cov * np.eye(dim)
    elif cov.ndim == 1:
        cov = np.diag(cov)
    elif cov.ndim == 2 and cov.shape != (dim, dim):
        rows, cols = cov.shape
        if rows != cols:
            msg = ("Array 'cov' must be square if it is two dimensional,"
                   " but cov.shape = %s." % str(cov.shape))
        else:
            msg = ("Dimension mismatch: array 'cov' is of shape %s,"
                   " but 'mean' is a vector of length %d.")
            msg = msg % (str(cov.shape), len(mean))
        raise ValueError(msg)
    elif cov.ndim > 2:
        raise ValueError("Array 'cov' must be at most two-dimensional,"
                         " but cov.ndim = %d" % cov.ndim)

    return dim, mean, cov


def _process_quantiles(x, dim):
    """
    Adjust quantiles array so that last axis labels the components of
    each data point.

    """
    x = np.asarray(x, dtype=float)

    if x.ndim == 0:
        x = x[np.newaxis]
    elif x.ndim == 1:
        if dim == 1:
            x = x[:, np.newaxis]
        else:
            x = x[np.newaxis, :]

    return x


def _squeeze_output(out):
    """
    Remove single-dimensional entries from array and convert to scalar,
    if necessary.

    """
    out = out.squeeze()
    if out.ndim == 0:
        out = out[()]
    return out


_doc_default_callparams = """\
mean : array_like, optional
mean : array_like, optional
    Mean of the distribution (default zero)
cov : array_like, optional
    Covariance matrix of the distribution (default one)
"""

_doc_callparams_note = \
    """Setting the parameter `mean` to `None` is equivalent to having `mean`
    be the zero-vector. The parameter `cov` can be a scalar, in which case
    the covariance matrix is the identity times that value, a vector of
    diagonal entries for the covariance matrix, or a two-dimensional
    array_like.
    """

_doc_frozen_callparams = ""

_doc_frozen_callparams_note = \
    """See class definition for a detailed description of parameters."""

docdict_params = {
    '_doc_default_callparams': _doc_default_callparams,
    '_doc_callparams_note': _doc_callparams_note,
}

docdict_noparams = {
    '_doc_default_callparams': _doc_frozen_callparams,
    '_doc_callparams_note': _doc_frozen_callparams_note,
}


# noinspection PyPep8Naming
class multi_rv_generic(object):

    def __init__(self):
        super(multi_rv_generic, self).__init__()


# noinspection PyPep8Naming
class multi_rv_frozen(object):

    def __init__(self):
        super(multi_rv_frozen, self).__init__()


# noinspection PyPep8Naming
class multivariate_normal_gen(multi_rv_generic):
    # noinspection PyTypeChecker
    """
    A multivariate Normal random variable.

    The `mean` keyword specifies the mean. The `cov` keyword specifies
    the covariance matrix.

    This implementation supports both classical statistics via maximum
    likelihood estimate (MLE) and Bayesian statistics using maximum
    a-posteriori (MAP) estimation for fitting the distribution from
    observation.

    Methods
    -------
    pdf(x, mean=None, cov=1)
        Probability density function.
    logpdf(x, mean=None, cov=1)
        Log of the probability density function.
    rvs(mean=None, cov=1, size=1)
        Draw random samples from a multivariate normal distribution.
    fit(x, prior=None, algorithm="map")
        Fit a multivariate normal via MLE or MAP.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_doc_default_callparams)s

    Alternatively, the object may be called (as a function) to fix the mean
    and covariance parameters, returning a "frozen" multivariate normal
    random variable:

    rv = multivariate_normal(mean=None, cov=1)
        - Frozen object with the same methods but holding the given
          mean and covariance fixed.

    Notes
    -----
    Setting the parameter `mean` to `None` is equivalent to having `mean`
    be the zero-vector. The parameter `cov` can be a scalar, in which case
    the covariance matrix is the identity times that value, a vector of
    diagonal entries for the covariance matrix, or a two-dimensional
    array_like.

    The covariance matrix `cov` must be a (symmetric) positive
    semi-definite matrix. The determinant and inverse of `cov` are computed
    as the pseudo-determinant and pseudo-inverse, respectively, so
    that `cov` does not need to have full rank.

    The probability density function for `multivariate_normal` is

    .. math::

        f(x) = \\frac{1}{\\sqrt{(2 \\pi)^k \\det \\Sigma}}
               \\exp\\left( -\\frac{1}{2} (x - \\mu)^T \\Sigma^{-1} (x - \\mu) \\right),

    where :math:`\mu` is the mean, :math:`\\Sigma` the covariance matrix,
    and :math:`k` is the dimension of the space where :math:`x` takes values.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from mlpy.stats import multivariate_normal

    >>> x = np.linspace(0, 5, 10, endpoint=False)
    >>> y = multivariate_normal.pdf(x, mean=2.5, cov=0.5)
    array([ 0.00108914,  0.01033349,  0.05946514,  0.20755375,  0.43939129,
            0.56418958,  0.43939129,  0.20755375,  0.05946514,  0.01033349])
    >>> fig1 = plt.figure()
    >>> ax = fig1.add_subplot(111)
    >>> ax.plot(x, y)

    The input quantiles can be any shape of array, as long as the last
    axis labels the components.  This allows us for instance to
    display the frozen pdf for a non-isotropic random variable in 2D as
    follows:

    >>> x, y = np.mgrid[-1:1:.01, -1:1:.01]
    >>> pos = np.empty(x.shape + (2,))
    >>> pos[:, :, 0] = x; pos[:, :, 1] = y
    >>> rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> ax2.contourf(x, y, rv.pdf(pos))

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    def __init__(self):
        super(multivariate_normal_gen, self).__init__()
        self.__doc__ = doccer.docformat(self.__doc__, docdict_params)

    def __call__(self, mean=None, cov=1):
        return multivariate_normal_frozen(mean, cov)

    def _logpdf(self, x, mean, cov):
        if np.any(np.isnan(np.ravel(x))):
            return self._handle_missing_data(x, mean, cov)

        if not hasattr(mean, "__len__"):
            x = np.ravel(x)  # mean is a scalar

        dim = 1
        n = x.shape[0]

        if x.ndim == 1:
            x = np.ravel(x) - np.ravel(mean)
        else:
            dim = x.shape[1]
            x = x - mean

        if cov.ndim == 1 and cov.size > 1:
            # diagonal case
            cov2 = np.tile(cov, (n, 1))
            # noinspection PyTypeChecker
            tmp = -np.true_divide(np.power(x, 2), 2 * cov2) - 0.5 * np.log(2 * np.pi * cov2)
            logp = np.sum(tmp, 2)
            return logp

        # full covariance case
        c_decomp = linalg.cholesky(cov, lower=False)

        # noinspection PyUnboundLocalVariable
        logp = -0.5 * np.sum(np.power(linalg.solve(c_decomp.T, x.T).T, 2), x.ndim-1)
        logz = 0.5 * dim * np.log(2 * np.pi) + np.sum(np.log(np.diag(c_decomp)))
        return logp - logz

    def logpdf(self, x, mean, cov):
        """
        Log of the multivariate normal probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability
            density function.
        mean : array_like
            Mean of the distribution.
        cov : array_like
            Covariance matrix of the distribution.

        Returns
        -------
        ndarray :
            Log of the probability density function evaluated at `x`.

        """
        dim, mean, cov = _process_parameters(None, mean, cov)
        x = _process_quantiles(x, dim)
        out = self._logpdf(x, mean, cov)
        return _squeeze_output(out)

    def pdf(self, x, mean, cov):
        """
        Multivariate normal probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability
            density function.
        mean : array_like
            Mean of the distribution.
        cov : array_like
            Covariance matrix of the distribution.

        Returns
        -------
        ndarray :
            Log of the probability density function. evaluated at `x`.

        """
        dim, mean, cov = _process_parameters(None, mean, cov)
        x = _process_quantiles(x, dim)
        return np.exp(self._logpdf(x, mean, cov))

    def rvs(self, mean=None, cov=None, size=1):
        """
        Draw random samples from a multivariate normal distribution.

        Parameters
        ----------
        mean : array_like
            Mean of the distribution.
        cov : array_like
            Covariance matrix of the distribution.
        size : int
            Number of samples to draw. Defaults to `1`.

        Returns
        -------
        ndarray or scalar :
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        """
        dim, mean, cov = _process_parameters(None, mean, cov)

        a = linalg.cholesky(cov, lower=False)
        z = np.random.randn(np.size(mean, axis=mean.ndim-1), size)
        mean = np.ravel(mean)
        out = np.dot(a, z).T + mean
        return _squeeze_output(out)

    def _fit_mle(self, x):
        return np.mean(x), np.cov(x)

    def _fit_map(self, x, prior):
        n, dim = x.shape

        xbar = np.mean(x)
        kappa0 = prior.kappa
        m0 = np.ravel(prior.mean)

        kappa = kappa0 + n
        mean = np.true_divide(n * xbar + kappa0 * m0, kappa)
        cov = prior.sigma + x.T * x + kappa0 * (m0 * m0.T) - kappa * (mean * mean.T)
        # noinspection PyTypeChecker
        cov = np.true_divide(cov, (prior.df + n) - dim - 1)
        return mean, cov

    def fit(self, x, prior=None, algorithm="map"):
        """
        Fit a multivariate Gaussian via MLE or MAP.

        MLE stands for Maximum Likelihood Estimate which chooses a value
        for :math:`\\mu` that maximizes the likelihood function given the
        observed data.
        MAP stands for Maximum a-Posteriori estimate is a Bayesian approach
        that tries to reflect our belief about :math:`\\mu`. Using Bayes' law
        a prior belief about the parameter :math:`\\mu`, :math:`p(\\mu)`,
        (before seeing the data :math:`X`) is converted into a posterior
        probability, :math:`p(\\mu|X)`, by using the likelihood function
        :math:`p(X|\\mu)`. The maximum a-posteriori estimate is defined as:

        .. math::

            \\hat{\\mu}_{MAP}=\\underset{x}{\\arg\\max}p(\\mu|X)

        Parameters
        ----------
        x: array_like
            Data to use to calculate the MLEs or MAPs.
        prior: normal_invwishart
            The prior (a normal-inverse-Wishart model).
            Set `prior` to ``None`` for MLE algorithm. For MAP, if `prior`
            is set to ``None``, a weak prior is used.
        algorithm: str, optional
            The estimation algorithm to use (map or mle). Default is `map`.

        Returns
        -------
        mean : array
            The mean.
        cov : array
            The covariance matrix.

        """
        algorithm = algorithm if algorithm in frozenset(("mle", "map")) else "map"

        if algorithm == "map":
            if prior is None:
                n, dim = x.shape
                prior = normal_invwishart(np.zeros(dim), 0, dim + 2, np.diag(np.true_divide(np.var(x), n)))
            return self._fit_map(x, prior)

        return self._fit_mle(x)

    def _handle_missing_data(self, x, mean, cov):
        miss_rows = np.isnan(x)

        mean = mean[~miss_rows]
        cov = cov[np.ix_(~miss_rows, ~miss_rows)]

        return self.logpdf(x[~miss_rows], mean, cov)

multivariate_normal = multivariate_normal_gen()


# noinspection PyPep8Naming
class multivariate_normal_frozen(multi_rv_frozen):

    def __init__(self, mean=None, cov=1):
        super(multivariate_normal_frozen, self).__init__()

        # noinspection PyTypeChecker
        self.dim, self.mean, self.cov = _process_parameters(None, mean, cov)
        self._dist = multivariate_normal_gen()

    def logpdf(self, x):
        """
        Log of the multivariate normal probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability
            density function.

        Returns
        -------
        ndarray :
            Log of the probability density function. evaluated at `x`.

        """
        x = _process_quantiles(x, self.dim)
        # noinspection PyProtectedMember
        out = self._dist._logpdf(x, self.mean, self.cov)
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1):
        """
        Draw random samples from a multivariate normal distribution.

        Parameters
        ----------
        size: int
            Number of samples to draw. Defaults to `1`.

        Returns
        -------
        ndarray or scalar :
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        """
        return self._dist.rvs(self.mean, self.cov, size)


# noinspection PyPep8Naming
class multivariate_student_gen(multi_rv_generic):
    """
    A multivariate Student random variable.

    The `mean` keyword specifies the mean. The `cov` keyword specifies
    the covariance matrix. The `df` keyword specifies the degrees of
    freedom.

    Methods
    -------
    pdf(x, mean=None, cov=1)
        Probability density function.
    logpdf(x, mean=None, cov=1)
        Log of the probability density function.
    rvs(mean=None, cov=1, size=1)
        Draw random samples from a multivariate Student distribution.
    fit(x, prior=None, algorithm="map")
        Fit a multivariate Student via MLE or MAP.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_doc_default_callparams)s
    df : int
        Degrees of freedom.

    Alternatively, the object may be called (as a function) to fix the mean
    and covariance parameters, returning a "frozen" multivariate normal
    random variable:

    rv = multivariate_student(mean=None, cov=1, df=None)
        - Frozen object with the same methods but holding the given
          mean and covariance fixed.

    Notes
    -----
    Setting the parameter `mean` to `None` is equivalent to having `mean`
    be the zero-vector. The parameter `cov` can be a scalar, in which case
    the covariance matrix is the identity times that value, a vector of
    diagonal entries for the covariance matrix, or a two-dimensional
    array_like.

    The covariance matrix `cov` must be a (symmetric) positive
    semi-definite matrix. The determinant and inverse of `cov` are computed
    as the pseudo-determinant and pseudo-inverse, respectively, so
    that `cov` does not need to have full rank.

    .. warning::
        This is only a stub class. Implementation still missing!

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """

    def __init__(self):
        super(multivariate_student_gen, self).__init__()
        self.__doc__ = doccer.docformat(self.__doc__, docdict_params)

    def __call__(self, mean=None, cov=None, df=None):
        multivariate_student_frozen(mean, cov, df)

    def logpdf(self, x, mean, cov, df):
        """
        Log of the multivariate Student probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability
            density function.
        mean : array_like
            Mean of the distribution.
        cov : array_like
            Covariance matrix of the distribution.
        df : int
            df: Degrees of freedom.

        Returns
        -------
        ndarray :
            Log of the probability density function. evaluated at `x`.

        Raises
        ------
        NotImplementedError
            This function is not yet implemented.

        """
        return NotImplementedError

    def pdf(self, x, mean, cov, df):
        """
        Multivariate Student probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the probability
            density function.
        mean : array_like
            Mean of the distribution.
        cov : array_like
            Covariance matrix of the distribution.
        df : int
            df: Degrees of freedom.

        Returns
        -------
        ndarray :
            Log of the probability density function. evaluated at `x`.

        Raises
        ------
        NotImplementedError
            This function is not yet implemented.

        """
        return NotImplementedError

    def rvs(self, mean, cov, df, size=1):
        """
        Draw random samples from a multivariate normal distribution.

        Parameters
        ----------
        mean : array_like
            Mean of the distribution.
        cov : array_like
            Covariance matrix of the distribution.
        df : int
            df: Degrees of freedom.
        size : int
            size: Number of samples to draw. Defaults to `1`.

        Returns
        -------
        ndarray or scalar :
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.

        Raises
        ------
        NotImplementedError
            This function is not yet implemented.

        """
        return NotImplementedError

    def fit(self, x, df=None):
        """
        Fit a multivariate Student.

        Parameters
        ----------
        x : array_like
            Data to use to calculate the MLEs or MAPs.
        df : int
            Degrees of freedom.

        Returns
        -------
        mean : array
            The mean.
        cov : array
            The covariance matrix.
        df : array
            The degrees of freedom.

        Raises
        ------
        NotImplementedError
            This function is not yet implemented.

        """
        return NotImplementedError

multivariate_student = multivariate_student_gen()


# noinspection PyPep8Naming
class multivariate_student_frozen(multi_rv_frozen):

    def __init__(self, mean, cov, df):
        super(multivariate_student_frozen, self).__init__()

        self._dist = multivariate_student_gen()
        self.mean = mean
        self.cov = cov
        self.df = df

    def logpdf(self, x):
        return self._dist.logpdf(x, self.mean, self.cov, self.df)

    def pdf(self, x):
        # noinspection PyTypeChecker
        return np.exp(self.logpdf(x))

    def rvs(self, size=1):
        self._dist.rvs(self.mean, self.cov, self.df, size)


_wishart_doc_default_callparams = """\
df : int
    Degrees of freedom, must be greater than or equal to dimension of the
    scale matrix
scale : array_like
    Symmetric positive definite scale matrix of the distribution
"""

_wishart_doc_callparams_note = ""

_wishart_doc_frozen_callparams = ""

_wishart_doc_frozen_callparams_note = \
    """See class definition for a detailed description of parameters."""

wishart_docdict_params = {
    '_doc_default_callparams': _wishart_doc_default_callparams,
    '_doc_callparams_note': _wishart_doc_callparams_note,
}

wishart_docdict_noparams = {
    '_doc_default_callparams': _wishart_doc_frozen_callparams,
    '_doc_callparams_note': _wishart_doc_frozen_callparams_note,
}


# noinspection PyPep8Naming
class invwishart_gen(multi_rv_generic):
    """
    Inverse Wishart random variable.

    The `df` keyword specifies the degrees of freedom. The `scale` keyword
    specifies the scale matrix, which must be symmetric and positive definite.
    In this context, the scale matrix is often interpreted in terms of a
    multivariate normal covariance matrix.

    Methods
    -------
    pdf(x, mean=None, cov=1)
        Probability density function.
    logpdf(x, mean=None, cov=1)
        Log of the probability density function.
    rvs(mean=None, cov=1, size=1)
        Draw random samples from a multivariate Student distribution.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_doc_default_callparams)s

    Alternatively, the object may be called (as a function) to fix the degrees
    of freedom and scale parameters, returning a "frozen" inverse Wishart
    random variable:

    rv = invwishart(df=1, scale=1)
        - Frozen object with the same methods but holding the given
          degrees of freedom and scale fixed.

    See Also
    --------
    :class:`normal_invwishart`

    Notes
    -----
    Setting the parameter `mean` to `None` is equivalent to having `mean`
    be the zero-vector. The parameter `cov` can be a scalar, in which case
    the covariance matrix is the identity times that value, a vector of
    diagonal entries for the covariance matrix, or a two-dimensional
    array_like.

    The scale matrix `scale` must be a symmetric positive defined matrix.
    Singular matrices, including the symmetric positive semi-definite case,
    are not supported.

    The inverse Wishart distribution is often denoted

    .. math::

        W_p^{-1}(\\Psi, \\nu)

    where :math:`\\nu` is the degrees of freedom and :math:`\\Psi` is the
    :math:`p \\times p` scale matrix.

    The probability density function for `invwishart` has support over positive
    definite matrices :math:`S`; if :math:`S \\sim W^{-1}_p(\\Sigma, \\nu)`,
    then its PDF is given by:

    .. math::

        f(S) = \\frac{|\\Sigma|^\\frac{\\nu}{2}}{2^{ \\frac{\\nu p}{2} }
               |S|^{\\frac{\\nu + p + 1}{2}} \\Gamma_p \\left(\\frac{\\nu}{2} \\right)}
               \\exp\\left( -\\frac{1}{2} tr(\\Sigma S^{-1}) \\right)

    If :math:`S \\sim W_p^{-1}(\\Psi, \\nu)` (inverse Wishart) then
    :math:`S^{-1} \\sim W_p(\\Psi^{-1}, \\nu)` (Wishart).

    If the scale matrix is 1-dimensional and equal to one, then the inverse
    Wishart distribution :math:`W_1(\\nu, 1)` collapses to the
    inverse Gamma distribution with parameters shape = :math:`\\frac{\\nu}{2}`
    and scale = :math:`\\frac{1}{2}`.

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """

    def __init__(self):
        super(invwishart_gen, self).__init__()
        self.__doc__ = doccer.docformat(self.__doc__, wishart_docdict_params)

    def __call__(self, df, scale):
        return invwishart_frozen(df, scale)

    def _process_parameters(self, df, scale):
        if scale is None:
            scale = 1.0
        scale = np.asarray(scale, dtype=float)

        if scale.ndim == 0:
            scale = scale[np.newaxis, np.newaxis]
        elif scale.ndim == 1:
            scale = np.diag(scale)
        elif scale.ndim == 2 and not scale.shape[0] == scale.shape[1]:
            raise ValueError("Array 'scale' must be square if it is two"
                             " dimensional, but scale.scale = %s."
                             % str(scale.shape))
        elif scale.ndim > 2:
            raise ValueError("Array 'scale' must be at most two-dimensional,"
                             " but scale.ndim = %d" % scale.ndim)

        dim = scale.shape[0]

        if df is None:
            df = dim
        elif not np.isscalar(df):
            raise ValueError("Degrees of freedom must be a scalar.")
        elif df < dim:
            raise ValueError("Degrees of freedom cannot be less than dimension"
                             " of scale matrix, but df = %d" % df)

        return dim, df, scale

    def _process_quantiles(self, x, dim):
        """
        Adjust quantiles array so that last axis labels the components of
        each data point.

        """
        x = np.asarray(x, dtype=float)

        if x.ndim == 0:
            x = x * np.eye(dim)[:, :, np.newaxis]
        if x.ndim == 1:
            if dim == 1:
                x = x[np.newaxis, np.newaxis, :]
            else:
                x = np.diag(x)[:, :, np.newaxis]
        elif x.ndim == 2:
            if not x.shape[0] == x.shape[1]:
                raise ValueError("Quantiles must be square if they are two"
                                 " dimensional, but x.shape = %s."
                                 % str(x.shape))
            x = x[:, :, np.newaxis]
        elif x.ndim == 3:
            if not x.shape[1] == x.shape[2]:
                raise ValueError("Quantiles must be square in the second and third"
                                 " dimensions if they are three dimensional"
                                 ", but x.shape = %s." % str(x.shape))
        elif x.ndim > 3:
            raise ValueError("Quantiles must be at most two-dimensional with"
                             " an additional dimension for multiple"
                             "components, but x.ndim = %d" % x.ndim)

        # Now we have 3-dim array; should have shape [dim, dim, *]
        if not x.shape[1:3] == (dim, dim):
            raise ValueError('Quantiles have incompatible dimensions: should'
                             ' be %s, got %s.' % ((dim, dim), x.shape[0:2]))

        return x

    def _logpdf(self, x, df, scale, logdet_scale):
        n = 1
        if x.ndim > 2:
            n = x.shape[0]
        dim = scale.shape[1]

        # noinspection PyTypeChecker
        logz = (df * dim * 0.5) * np.log(2) + multigammaln(0.5*df, dim) - (0.5*df) * logdet_scale

        out = np.zeros(n)
        for i in range(n):
            _, logdet_x = self._cholesky_logdet(x[i])
            out[i] = -(df + dim + 1) * 0.5 * logdet_x - 0.5 * np.trace(
                np.linalg.lstsq(x[i].T, scale.T)[0]) - logz
        return out

    def logpdf(self, x, df, scale):
        """
        Log of the inverse Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability
            density function.
        df : int
            Degrees of freedom.
        scale : ndarray
            Scale matrix.

        Returns
        -------
        ndarray :
            Log of the probability density function evaluated at `x`.

        """
        dim, df, scale = self._process_parameters(df, scale)
        x = self._process_quantiles(x, dim)
        _, logdet_scale = self._cholesky_logdet(scale)
        out = self._logpdf(x, df, scale, logdet_scale)
        return _squeeze_output(out)

    def pdf(self, x, df, scale):
        """
        Inverse Wishart probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability
            density function.
        df : int
            Degrees of freedom.
        scale : ndarray
            Scale matrix.

        Returns
        -------
        ndarray :
            Probability density function evaluated at `x`.

        """
        return np.exp(self.logpdf(x, df, scale))

    def rvs(self, df, scale, size=1):
        """
        Draw random samples from teh inverse Wishart distribution.

        Parameters
        ----------
        df : int
            Degrees of freedom.
        scale : ndarray
            Scale matrix.

        Returns
        -------
        ndarray :
            Random variates of shape (`size`) + (`dim`, `dim`), where
            `dim` is the dimension of the scale matrix.

        Raises
        ------
        NotImplementedError
            This function is not yet implemented.

        """
        raise NotImplementedError

    def _mean(self, dim, df, scale):
        if df > dim + 1:
            out = scale / (df - dim - 1)
        else:
            out = None
        return out

    def mean(self, df, scale):
        """
        Mean of the inverse Wishart distribution

        Only valid if the degrees of freedom are greater than the dimension of
        the scale matrix plus one.

        Parameters
        ----------
        df : int
            Degrees of freedom.
        scale : ndarray
            Scale matrix.

        Returns
        -------
        float :
            The mean of the distribution

        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mean(dim, df, scale)
        # noinspection PyTypeChecker
        return _squeeze_output(out) if out is not None else out

    def _mode(self, dim, df, scale):
        return scale / (df + dim + 1)

    def mode(self, df, scale):
        """
        Mode of the inverse Wishart distribution

        Parameters
        ----------
        df : int
            Degrees of freedom.
        scale : ndarray
            Scale matrix.

        Returns
        -------
        float :
            The Mode of the distribution

        """
        dim, df, scale = self._process_parameters(df, scale)
        out = self._mode(dim, df, scale)
        # noinspection PyTypeChecker
        return _squeeze_output(out)

    def _cholesky_logdet(self, scale):
        c_decomp = linalg.cholesky(scale, lower=True)
        logdet = 2 * np.sum(np.log(c_decomp.diagonal()))
        return c_decomp, logdet

invwishart = invwishart_gen()


# noinspection PyPep8Naming,PyProtectedMember
class invwishart_frozen(multi_rv_frozen):
    def __init__(self, df, scale):
        """
        Create a frozen inverse Wishart distribution.

        Parameters
        ----------
        df : int
            Degrees of freedom.
        scale : ndarray
            Scale matrix.

        """
        super(invwishart_frozen, self).__init__()

        self._dist = invwishart_gen()
        self.dim, self.df, self.scale = self._dist._process_parameters(df, scale)
        _, self.logdet_scale = self._dist._cholesky_logdet(self.scale)

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        out = self._dist._logpdf(x, self.df, self.scale, self.logdet_scale)
        return _squeeze_output(out)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1):
        return self._dist.rvs(self.df, self.scale, size)


_normal_wishart_doc_default_callparams = """\
m0 : array_like
    The prior mean.
k0 : int
    Kappa: The strength of the believe in m0.
nu0 : int
    The strength of the believe in s0.
s0 : ndarray
    The prior scale matrix.
"""

_normal_wishart_doc_callparams_note = ""

_normal_wishart_doc_frozen_callparams = ""

_normal_wishart_doc_frozen_callparams_note = \
    """See class definition for a detailed description of parameters."""

normal_wishart_docdict_params = {
    '_doc_default_callparams': _normal_wishart_doc_default_callparams,
    '_doc_callparams_note': _normal_wishart_doc_callparams_note,
}

normal_wishart_docdict_noparams = {
    '_doc_default_callparams': _normal_wishart_doc_frozen_callparams,
    '_doc_callparams_note': _normal_wishart_doc_frozen_callparams_note,
}


# noinspection PyPep8Naming
class normal_invwishart_gen(multi_rv_generic):
    """
    A normal inverse Wishart random variable.

    The `m0` keyword specifies the prior mean for :math:`\mu`. The `k0` keyword
    specifies the strength in the believe of the prior mean. The `s0` keyword
    specifies the prior scale matrix and the `nu0` keyword specifies the strength
    in the believe of the prior scale. The `mean` keyword specifies the mean and the
    `scale` keyword specifies the scale matrix, which must be symmetric and positive
    definite.

    Methods
    -------
    pdf(x, mean=None, cov=1)
        Probability density function.
    logpdf(x, mean=None, cov=1)
        Log of the probability density function.
    rvs(mean=None, cov=1, size=1)
        Draw random samples from a multivariate Student distribution.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_doc_default_callparams)s


    Alternatively, the object may be called (as a function) to fix the degrees
    of freedom and scale parameters, returning a "frozen" inverse Wishart
    random variable:

    rv = normal_invwishart(m0=None, k0=0.5, nu0=0.5, s0=1)
        - Frozen object with the same methods but holding the given
          degrees of freedom and scale fixed.

    See Also
    --------
    :class:`invwishart`

    Notes
    -----
    Setting the parameter `mean` to `None` is equivalent to having `mean`
    be the zero-vector. The parameter `cov` can be a scalar, in which case
    the covariance matrix is the identity times that value, a vector of
    diagonal entries for the covariance matrix, or a two-dimensional
    array_like.

    The scale matrix `scale` must be a symmetric positive defined matrix.
    Singular matrices, including the symmetric positive semi-definite case,
    are not supported.

    The normal-inverse Wishart distribution is often denoted

    .. math::

        NIW_p^{-1}(\\mu_0, \\kappa_0, \\Psi, \\nu)

    where :math:`\\mu` is the prior mean, :math:`kappa_0` is the believe of this prior,
    :math:`\\Psi` is the :math:`p \\times p` scale matrix, and  :math:`\\nu` is the believe
    of this prior.


    The probability density function for `normal_invwishart` has support over positive
    definite matrices :math:`\\Sigma`; if :math:`(\\mu, \\Sigma) \\sim NIW^{-1}_p(\\mu_0, \\kappa_0, \\Psi, \\nu)`,
    then its PDF is given by:

    .. math::

        f(\\mu, \\Sigma|\\mu_0, \\kappa_0, \\Psi, \\nu) = \\mathcal{N} \\left) \\mu | \\mu_0, \\frac{1}{\\kappa_0}
        \\Sigma\\right) W^{-1}(\\Sigma | \\Psi, \\nu)

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """

    def __init__(self):
        super(normal_invwishart_gen, self).__init__()
        self.__doc__ = doccer.docformat(self.__doc__, normal_wishart_docdict_params)

    def __call__(self, m0, k0, nu0, s0, pseudo_counts=None):
        return normal_invwishart_frozen(m0, k0, nu0, s0, pseudo_counts)

    def _process_parameters(self, mean, sigma):
        sdim1 = 1
        if sigma.ndim > 2:
            sdim1, sdim2, sdim3 = sigma.shape
        else:
            sdim2, sdim3 = sigma.shape
        d = min(sdim2, sdim3)
        mean = np.reshape(mean, (-1, d))
        sigma = np.reshape(sigma, (-1, d, d))

        mdim1, mdim2 = mean.shape
        n = max(mdim1, sdim1)

        if mdim1 < n:
            mean = np.tile(mean, (1, n))
        if sdim1 < n:
            sigma = np.tile(sigma, (n, 1, 1))

        return n, mean, sigma

    def _logpdf(self, m0, k0, nu0, s0, mean, sigma, ncomponents):
        pgauss = np.zeros(ncomponents)
        for i in range(ncomponents):
            pgauss[i] = multivariate_normal.logpdf(mean[i], m0, np.true_divide(sigma[i], k0))

        out = pgauss + invwishart.logpdf(sigma, nu0, s0)
        return out

    def logpdf(self, m0, k0, nu0, s0, mean, sigma):
        """
        Log of the normal inverse Wishart probability density function.

        Parameters
        ----------
        m0 : ndarray
            The prior mean.
        k0 : int
            The strength of the believe in m0.
        nu0 : int
            The strength of believe in s0.
        s0 : ndarray
            The prior scale matrix.
        mean : ndarray
            The mean of the distribution.
        sigma : ndarray
            Scale matrix.

        Returns
        -------
        ndarray :
            Log of the probability density function evaluated at `x`.

        """
        ncomponents, mean, sigma = self._process_parameters(mean, sigma)
        out = self._logpdf(m0, k0, nu0, s0, mean, sigma, ncomponents)
        return _squeeze_output(out)

normal_invwishart = normal_invwishart_gen()


# noinspection PyPep8Naming,PyProtectedMember
class normal_invwishart_frozen(multi_rv_frozen):
    def __init__(self, m0, k0, nu0, s0, pseudo_counts=None):
        """
        Create a frozen normal inverse Wishart distribution.

        Parameters
        ----------
        m0 : ndarray
            The prior mean.
        k0 : int
            The strength of the believe in m0.
        nu0 : int
            The strength of the believe in s0.
        s0 : ndarray
            The prior scale matrix.

        """
        super(normal_invwishart_frozen, self).__init__()

        self._dist = normal_invwishart_gen()
        self.mean = m0
        self.kappa = k0
        self.df = nu0
        self.sigma = s0

        self.pseudo_counts = pseudo_counts

    def logpdf(self, mean, sigma):
        ncomponents, mean, sigma = self._dist._process_parameters(mean, sigma)
        out = self._dist._logpdf(self.mean, self.kappa, self.df, self.sigma, mean, sigma, ncomponents)
        return _squeeze_output(out)
