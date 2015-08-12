from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
from scipy.misc import doccer
# noinspection PyPackageRequirements
from sklearn.utils.extmath import logsumexp

from ..auxiliary.array import nunique
from . import partitioned_mean, partitioned_cov, randpd, stacked_randpd, multivariate_normal, multivariate_student, \
    normal_invwishart

__all__ = ["conditional_normal", "conditional_student", "conditional_mix_normal"]


def _process_parameters(ncomponents=None, dim=None, mean=None, cov=None):
    """
    Infer number of components and dimensionality from mean or covariance
    matrix, ensure that mean and covariance are full vector resp. matrix.

    """
    if ncomponents is None and dim is None:
        if mean is None:
            if cov is None:
                ncomponents = dim = 1
            else:
                cov = np.asarray(cov, dtype=float)
                if cov.ndim < 3:
                    ncomponents = 1
                    if cov.ndim < 2:
                        dim = 1
                    else:
                        dim = cov.shape[0]
                else:
                    ncomponents = cov.shape[0]
                    dim = cov.shape[1]

        else:
            mean = np.asarray(mean, dtype=float)
            if mean.ndim < 2:
                ncomponents = dim = 1
            else:
                ncomponents, dim = mean.shape
    elif not np.isscalar(ncomponents):
        raise ValueError("Number of mixture components of random variable must be scalar.")
    elif not np.isscalar(dim):
        raise ValueError("Dimension of random variable must be a scalar.")

    # Check input sizes and return full arrays for mean and cov if necessary
    if mean is None:
        mean = np.zeros((ncomponents, dim))
    mean = np.asarray(mean, dtype=float)

    if cov is None:
        cov = 1.0
    cov = np.asarray(cov, dtype=float)

    if cov.shape[0] != ncomponents and cov.shape[1] == ncomponents:
        cov = cov.T

    if dim == 1:
        mean.shape = (ncomponents, 1,)
        cov.shape = (ncomponents, 1, 1)

    if mean.shape[0] != ncomponents:
        raise ValueError("Array 'mean' must have %d mixture components." % ncomponents)
    if mean.ndim != 2 or mean.shape[1] != dim:
        raise ValueError("Each mixture component of array 'mean' must be a vector of length %d." % dim)
    if cov.ndim == 0:
        c = cov
        cov = np.zeros((ncomponents, dim, dim))
        for i in range(ncomponents):
            cov[i] = c * np.eye(dim)
    elif cov.ndim == 1:
        c = cov
        cov = np.zeros((ncomponents, dim, dim))
        for i in range(ncomponents):
            cov[i] = [np.diag(c)]
    elif cov.ndim == 2:
        if cov[0].shape != (dim, dim):
            rows, cols = cov[0].shape
            if rows != cols:
                msg = ("Each mixture component of array 'cov' must be square if it is two dimensional,"
                       " but cov[0].shape = %s." % str(cov[0].shape))
            else:
                msg = ("Dimension mismatch: each mixture component of array 'cov' is of shape %s,"
                       " but each mixture component of 'mean' is a vector of length %d.")
                msg = msg % (str(cov[0].shape), len(mean[0]))
            raise ValueError(msg)
        else:
            c = cov
            cov = np.zeros((ncomponents, dim, dim))
            for i in range(ncomponents):
                cov[i] = c
    elif cov.ndim == 3:
        n, rows, cols = cov.shape
        if n != ncomponents:
            msg = ("Dimension mismatch: array 'cov' has %d mixture components,"
                   " but 'mean' has %d mixture components" % (cov.shape[0], mean.shape[0]))
            raise ValueError(msg)
        if cov[0].shape != (dim, dim):
            rows, cols = cov[0].shape
            if rows != cols:
                msg = ("Each mixture component of array 'cov' must be square if it is two dimensional,"
                       " but cov[0].shape = %s." % str(cov[0].shape))
            else:
                msg = ("Dimension mismatch: each mixture component of array 'cov' is of shape %s,"
                       " but each mixture component of 'mean' is a vector of length %d.")
                msg = msg % (str(cov[0].shape), len(mean[0]))
            raise ValueError(msg)
    elif cov.ndim > 3:
        raise ValueError("Array 'cov' must be at most three-dimensional,"
                         " but cov.ndim = %d" % cov.ndim)

    return ncomponents, dim, mean, cov


_doc_default_callparams = """\
mean : array_like, optional
    Mean of the distribution (default zero)
cov : array_like, optional
    Covariance matrix of the distribution (default one)
"""

_doc_callparams_note = """\
Setting the parameter `mean` to `None` is equivalent to having `mean`
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
class cond_rv_generic(object):
    """
    Class which encapsulates common functionality between all
    conditional distributions.

    """
    def __init__(self):
        super(cond_rv_generic, self).__init__()


# noinspection PyPep8Naming
class cond_rv_frozen(object):
    """
    Class which encapsulates common functionality between all
    conditional distributions.

    """
    def __init__(self):
        super(cond_rv_frozen, self).__init__()


# noinspection PyProtectedMember,PyPep8Naming
class conditional_normal_gen(cond_rv_generic):
    """A conditional normal random variable.

    The `mean` keyword specifies the mean. the `cov` keyword specifies the
    covariance matrix. The `prior` keyword is the prior probability used
    when fitting the distribution via maximum-a-posteriori (MAP). The
    `algorithm` keyword identifies whether to use maximum likelihood estimation
    (MLE) or maximum-a-posteriori (MAP) for fitting the distribution.

    Methods
    -------
    logprior(mean, cov, prior)
        Log of the prior distribution.
    logpdf(x, mean, cov)
        Log of the probability density function.
    pdf(x, mean, cov)
        Probability density function.
    random(ncomponents, dim)
        Randomly initialize.
    fit(z, y, prior=None, algorithm='map')
        Fit a conditional normal probability distribution.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_doc_default_callparams)s
    prior : normal_invwishart
        The distribution's prior, a Normal-inverse-Wishart distribution.
    ncomponents : int
        The distribution's number of components.
    dim : int
        Dimensionality of each component.
    algorithm : {'map', 'mle'}
        Distribution fitting algorithm.

    Alternatively, the object may be called (as a function) to fix the mean,
    covariance, prior, and algorithm parameters, returning a "frozen" conditional
    normal random variable:

    rv = conditional_normal(mean=None, cov=1, algorithm='mle')
        - Frozen object with the same methods but holding the given
          mean, covariance, prior and algorithm fixed.

    Notes
    -----
    %(_doc_callparams_note)s

    Examples
    --------
    >>> from mlpy.stats import conditional_normal

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    def __init__(self):
        super(conditional_normal_gen, self).__init__()
        self.__doc__ = doccer.docformat(self.__doc__, docdict_params)

    def __call__(self, mean=None, cov=None, prior=None, algorithm='map'):
        return conditional_normal_frozen(mean, cov, prior, algorithm)

    def logprior(self, mean, cov, prior):
        """Log of the prior distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        prior : normal_invwishart
            The distribution's prior, a Normal-inverse-Wishart
            distribution.

        Returns
        -------
        float :
            The log of the prior.

        """
        out = 0
        if prior is not None:
            out = prior.logpdf(mean, cov)
        return np.sum(out)

    def _logpdf(self, x, mean, cov, ncomponents):
        nseq = x.shape[0]
        obs = np.logical_not(np.any(np.isnan(x.T), 0))
        xobs = x[obs]

        lpr = np.empty((ncomponents, nseq))
        lpr.fill(np.nan)

        for i in range(ncomponents):
            lpr[i, obs] = multivariate_normal.logpdf(xobs, mean[i], cov[i])
        return lpr

    def logpdf(self, x, mean, cov):
        """Log of the conditional normal probability density function.

        Return the soft evidence matrix for the given observations `x`.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function.
        %(_doc_default_callparams)s

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at `x`.

        Notes
        -----
        %(_doc_callparams_note)s

        """
        ncomponents, dim, mean, cov = _process_parameters(None, None, mean, cov)
        return self._logpdf(x, mean, cov, ncomponents)

    def pdf(self, x, mean, cov):
        """Conditional normal probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`

        """
        return np.exp(self.logpdf(x, mean, cov))

    def random(self, ncomponents, dim):
        """Randomly initialize the conditional normal distribution.

        Parameters
        ----------
        ncomponents : int
            The number of components of the distribution.
        dim : int
            The dimension of each component in the distribution.

        Returns
        -------
        mean : array_like, shape (`ncomponents`, dim`)
            The mean of the distribution.
        cov : array_like, shape (`ncomponents`, `dim`, `dim`)
            The covariance matrix of the distribution.

        """
        mean = np.random.randn(ncomponents, dim)
        # noinspection PyTypeChecker
        cov = np.zeros(ncomponents, dim, dim)
        for i in range(ncomponents):
            cov[i] = randpd(dim) + 2 * np.eye(dim)
        return mean, cov

    def _fit_mle(self, z, y, ncomponents):
        mean = partitioned_mean(y, z, ncomponents)
        cov = partitioned_cov(y, z, ncomponents)
        return mean, cov

    def _fit_map(self, z, y, prior, ncomponents, dim):
        mean = np.zeros(ncomponents, dim)
        cov = np.zeros(ncomponents, dim, dim)
        for i in range(ncomponents):
            mean[i], cov[i] = multivariate_normal.fit(y[z == i], prior, algorithm='map')
        return mean, cov

    def fit(self, z, y, prior=None, algorithm='map'):
        """Fit a conditional normal probability distribution.

        By default the parameters are lightly regularized, thus map
        estimation is performed rather than mle. The Gaussian-invWishart
        prior is set at instantiation of the class.

        Parameters
        ----------
        z : array_like, shape (`nsamples`, 1)
            z[i] is the state of the parent z in observation i
        y : array_like, shape (`nsamples`, `dim`)
            y[i, :] is the ith 1-by-`dim` observation of the child corresponding to z[i]
        prior: normal_invwishart
            A `normal_invwishart` distribution.

        """
        algorithm = algorithm if algorithm in frozenset(('mle', 'map')) else 'map'
        ncomponents = nunique(z)

        if algorithm == 'map':
            dim = y.shape[1]
            prior = prior if prior is not None else normal_invwishart(np.zeros(dim), 0.01, dim + 1,
                                                                      0.1 * np.eye(dim))
            return self._fit_map(z, y, prior, ncomponents, dim)
        return self._fit_mle(z, y, ncomponents)


conditional_normal = conditional_normal_gen()


# noinspection PyProtectedMember,PyPep8Naming
class conditional_normal_frozen(cond_rv_frozen):
    @property
    def wsum(self):
        return self._wsum

    def __init__(self, mean, cov, prior=None, algorithm="map"):
        """Create a "frozen" conditional normal random variable.

        Parameters
        ----------
        mean : array_like, shape (`ncomponents`, `dim`)
            Mean parameters for each component.
        cov : array_like, shape (`ncomponents`, `dim`, `dim`)
            Covariance parameters for each mixture component.
        prior : normal_invwishart
            A :data:`normal_invwishart` distribution.
        algorithm : {'map', 'mle'}
            The estimation algorithm to use (map or mle).

        Examples
        --------
        When called with the default parameters, this will create a 1D random
        variable with mean 0 and covariance 1:

        >>> from mlpy.stats import conditional_normal
        >>> r = conditional_normal()
        >>> r.mean
        array([ 0.])
        >>> r.cov
        array([[1.]])

        """
        super(conditional_normal_frozen, self).__init__()

        self.ncomponents, self.dim, self.mean, self.cov = _process_parameters(None, None, mean, cov)
        self._algorithm = algorithm if algorithm in frozenset(("mle", "map")) else "map"
        self._dist = conditional_normal_gen()

        self.prior = None
        if algorithm == "map":
            self.prior = prior if prior is not None else normal_invwishart(np.zeros(self.dim), 0.01, self.dim + 1,
                                                                           0.1 * np.eye(self.dim))
        # expected sufficient statistics
        self._wsum = None
        self._xbar = None
        self._xx = None

    def logprior(self):
        return self._dist.logprior(self.mean, self.cov, self.prior)

    def logpdf(self, x):
        return self._dist._logpdf(x, self.mean, self.cov, self.ncomponents)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def expected_sufficient_statistics(self, data, weights):
        """Compute the expected sufficient statistics.

        Compute the expected sufficient statistics for the conditional
        probability distribution.

        Parameters
        ----------
        data : array_like, shape (`nsamples`, `dim`)
            The observations.
        weights : array_like, shape (`nsamples`, `ncomponents`)
            The marginal probability of the discrete parent for each observation.

        """
        self._wsum = np.sum(weights, 0)
        x = np.dot(data.T, weights)
        self._xbar = x / self._wsum
        self._xx = np.zeros((self.ncomponents, self.dim, self.dim))

        for i in range(self.ncomponents):
            xc = data - self._xbar[:, i]
            self._xx[i] = np.dot(xc.T * weights[:, i], xc)

    def _fit_mle_ess(self):
        self.mean = np.reshape(self._xbar, self.dim, self.ncomponents)
        self.cov = [np.true_divide(xi, self._wsum[i]) for i, xi in self._xx]

    def _fit_map_ess(self):
        kappa0 = self.prior.kappa
        m0 = np.ravel(self.prior.mean)
        nu0 = self.prior.df
        s0 = self.prior.sigma

        for i in range(self.ncomponents):
            xbar_i = self._xbar[:, i]
            w_i = self._wsum[i]
            a = np.true_divide(kappa0 * w_i, kappa0 + w_i)
            b = nu0 + w_i + self.dim + 2
            sprior = np.dot(xbar_i - m0, xbar_i - m0)
            self.mean[i] = np.true_divide(w_i * xbar_i + kappa0 * m0, w_i + kappa0)
            # noinspection PyTypeChecker
            self.cov[i] = np.true_divide(s0 + self._xx[i] + a * sprior, b)

    def fit(self, z=None, y=None):
        """Fit a conditional normal probability distribution.

        If no parameters are given, the condition normal probability
        distribution is fit with the expected sufficient statistics.

        By default the parameters are lightly regularized, thus map
        estimation is performed rather than mle. The Gaussian-invWishart
        prior is set at instantiation of the class.

        Parameters
        ----------
        z : array_like, shape (`nsamples`, 1)
            z[i] is the state of the parent z in observation i
        y : array_like, shape (`nsamples`, `dim`)
            y[i, :] is the ith 1-by-`dim` observation of the child corresponding to z[i]

        """
        if z is not None and y is not None:
            if self._algorithm == "map":
                self.mean, self.cov = self._dist._fit_map(z, y, self.prior, self.ncomponents, self.dim)
            else:
                self.mean, self.cov = self._dist._fit_mle(z, y, self.ncomponents)
        else:
            {
                "mle": self._fit_mle_ess,
                "map": self._fit_map_ess
            }[self._algorithm]()


_student_doc_default_callparams = """\
mean : array_like, optional
    Mean of the distribution (default zero)
cov : array_like, optional
    Covariance matrix of the distribution (default one)
df : array_like, shape (`ncomponents`,)
    The degrees of freedom.
"""

_student_doc_callparams_note = """\
Setting the parameter `mean` to `None` is equivalent to having `mean`
be the zero-vector. The parameter `cov` can be a scalar, in which case
the covariance matrix is the identity times that value, a vector of
diagonal entries for the covariance matrix, or a two-dimensional
array_like.
"""

_student_doc_frozen_callparams = ""

_student_doc_frozen_callparams_note = \
    """See class definition for a detailed description of parameters."""

student_docdict_params = {
    '_doc_default_callparams': _student_doc_default_callparams,
    '_doc_callparams_note': _student_doc_callparams_note,
}

student_docdict_noparams = {
    '_doc_default_callparams': _student_doc_frozen_callparams,
    '_doc_callparams_note': _student_doc_frozen_callparams_note,
}


# noinspection PyPep8Naming
class conditional_student_gen(cond_rv_generic):
    """A conditional student random variable.

    The `mean` keyword specifies the mean. the `cov` keyword specifies the
    covariance matrix. The `df` keyword is the degrees of freedom. The `prior`
    keyword is the prior probability used when fitting the distribution via
    maximum-a-posteriori (MAP). The `algorithm` keyword identifies whether to
    use maximum likelihood estimation (MLE) or maximum-a-posteriori (MAP)
    for fitting the distribution.

    Methods
    -------
    logpdf(x, mean, cov, df)
        Log of the probability density function.
    pdf(x, mean, cov, df)
        Probability density function.
    random(ncomponents, dim)
        Randomly initialize.
    fit(z, y, prior=None, algorithm='map')
        Fit a conditional normal probability distribution.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_doc_default_callparams)s
    prior : normal_invwishart
        The distribution's prior, a Normal-inverse-Wishart distribution.
    ncomponents : int
        The distribution's number of components.
    dim : int
        Dimensionality of each component.
    algorithm : {'map', 'mle'}
        Distribution fitting algorithm.

    Alternatively, the object may be called (as a function) to fix the mean,
    covariance, prior, and algorithm parameters, returning a "frozen" conditional
    student random variable:

    rv = conditional_student(mean=None, cov=1, algorithm='mle')
        - Frozen object with the same methods but holding the given
          mean, covariance, df, prior and algorithm fixed.

    Notes
    -----
    %(_doc_callparams_note)s

    Examples
    --------
    >>> from mlpy.stats import conditional_student

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    def __init__(self):
        super(conditional_student_gen, self).__init__()
        self.__doc__ = doccer.docformat(self.__doc__, student_docdict_params)

    def __call__(self, mean=None, cov=None, df=None, prior=None, algorithm="map"):
        return conditional_student_frozen(mean, cov, df, prior, algorithm)

    def _logpdf(self, x, mean, cov, df, ncomponents):
        nseq = x.shape[0]
        obs = np.logical_not(np.any(np.isnan(x.T), 0))
        xobs = x[obs]

        lpr = np.empty((ncomponents, nseq))
        lpr.fill(np.nan)

        for i in range(ncomponents):
            lpr[i, obs] = multivariate_student.logpdf(xobs, mean[i], cov[i], df)
        return lpr

    def logpdf(self, x, mean, cov, df):
        """Log of the conditional student probability density function.

        Return the soft evidence matrix for the given observations `x`.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function.
        %(_doc_default_callparams)s

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at `x`.

        Notes
        -----
        %(_doc_callparams_note)s

        """
        ncomponents, dim, mean, cov = _process_parameters(None, None, mean, cov)
        return self._logpdf(x, mean, cov, df, ncomponents)

    def pdf(self, x, mean, cov, df):
        """Conditional student probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`

        """
        # noinspection PyTypeChecker
        return np.exp(self.logpdf(x, mean, cov, df))

    def random(self, ncomponents, dim):
        """Randomly initialize the conditional student distribution.

        Parameters
        ----------
        ncomponents : int
            The number of components of the distribution.
        dim : int
            The dimension of each component in the distribution.

        Returns
        -------
        mean : array_like, shape (`ncomponents`, dim`)
            The mean of the distribution.
        cov : array_like, shape (`ncomponents`, `dim`, `dim`)
            The covariance matrix of the distribution.
        df : int
            The degrees of freedom.

        """
        mean = np.random.randn(ncomponents, dim)
        cov = stacked_randpd(dim, ncomponents, 2)
        df = 10 * np.ones(ncomponents)
        return mean, cov, df

    def _fit(self, z, y, ncomponents, dim):
        mean = np.zeros((ncomponents, dim))
        cov = np.zeros((ncomponents, dim, dim))
        df = np.zeros(ncomponents)

        for i in range(ncomponents):
            # noinspection PyArgumentList
            mean[i], cov[i], df[i] = multivariate_student.fit(x=y[z == i])
        return mean, cov, df

    def fit(self, z, y, prior=None, algorithm='map'):
        """Fit a conditional student probability distribution.

        By default the parameters are lightly regularized, thus map
        estimation is performed rather than mle. The Gaussian-invWishart
        prior is set at instantiation of the class.

        Parameters
        ----------
        z : array_like, shape (`nsamples`, 1)
            z[i] is the state of the parent z in observation i
        y : array_like, shape (`nsamples`, `dim`)
            y[i, :] is the ith 1-by-`dim` observation of the child corresponding to z[i]
        prior: normal_invwishart
            A `normal_invwishart` distribution.

        """
        algorithm = algorithm if algorithm in frozenset(('mle', 'map')) else 'map'
        ncomponents = nunique(z)
        dim = y.shape[1]

        if algorithm == 'map':
            # noinspection PyUnusedLocal
            prior = prior if prior is not None else normal_invwishart(np.zeros(dim), 0.01, dim + 1,
                                                                      0.1 * np.eye(dim))
        return self._fit(z, y, ncomponents, dim)


conditional_student = conditional_student_gen()


# noinspection PyPep8Naming,PyProtectedMember
class conditional_student_frozen(cond_rv_frozen):
    def __init__(self, mean, cov, df, prior=None, algorithm='map'):
        """Create a "frozen" conditional student random variable.

        Parameters
        ----------
        mean : array_like, shape (`ncomponents`, `dim`)
            Mean parameters for each component.
        cov : array_like, shape (`ncomponents`, `dim`, `dim`)
            Covariance parameters for each mixture component.
        df : array_like, shape (`ncomponents`,)
            The degrees of freedom
        prior : normal_invwishart
            A :data:`normal_invwishart` distribution.
        algorithm : {'map', 'mle'}
            The estimation algorithm to use (map or mle).

        Examples
        --------
        When called with the default parameters, this will create a 1D random
        variable with mean 0 and covariance 1:

        >>> from mlpy.stats import conditional_student
        >>> r = conditional_student()
        >>> r.mean
        array([ 0.])
        >>> r.cov
        array([[1.]])

        """
        super(conditional_student_frozen, self).__init__()

        self.ncomponents, self.dim, self.mean, self.cov = _process_parameters(None, None, mean, cov)
        self._algorithm = algorithm if algorithm in frozenset(('mle', 'map')) else 'map'
        self._dist = conditional_student_gen()

        self.df = df
        self.prior = None
        if algorithm == 'map':
            self.prior = prior if prior is not None else normal_invwishart(np.zeros(self.dim), 0.01, self.dim + 1,
                                                                           0.1 * np.eye(self.dim))

        # expected sufficient statistics
        self._wsum = None
        self._sw = None
        self._sx = None
        self._sxx = None
        self._data = None

    def logpdf(self, x):
        return self._dist._logpdf(x, self.mean, self.cov, self.df, self.ncomponents)

    def pdf(self, x):
        # noinspection PyTypeChecker
        return np.exp(self.logpdf(x))

    def expected_sufficient_statistics(self, data, weights):
        """Compute the expected sufficient statistics.

        Compute the expected sufficient statistics for the conditional
        probability distribution.

        Parameters
        ----------
        data : array_like, shape (`nsamples`, `dim`)
            The observations.
        weights : array_like, shape (`nsamples`, `ncomponents`)
            The marginal probability of the discrete parent for each observation.

        """
        self._wsum = np.sum(weights, 0)
        self._sw = np.zeros(self.ncomponents)
        self._sx = np.zeros(self.ncomponents, self.dim)
        self._sxx = np.zeros((self.ncomponents, self.dim, self.dim))

        for i in range(self.ncomponents):
            xc = data - self.mean[:, i]
            delta = np.sum((xc / self.cov[i]) * xc, axis=1)
            # noinspection PyTypeChecker
            w = np.true_divide(self.df[i] + self.dim, self.df[i] + delta)
            xw = weights[:, i] * data * w[:]
            self._sw[i] = np.dot(weights[:, i].T, w)
            self._sx[i] = np.sum(xw, axis=0)
            self._sxx[i] = np.dot(xw.T, data)
        self._data = data

    def _fit(self):
        for i in range(self.ncomponents):
            sxc = self._sx[i].T
            factor = np.true_divide(1, self._wsum[i])
            self.cov[i] = factor * (self._sxx[i] - np.true_divide(np.dot(sxc, sxc.T), self._sw[i]))
        self.mean = np.linalg.lstsq(self._sx.T, self._sw)

    def fit(self, z=None, y=None):
        """Fit a conditional student probability distribution.

        If no parameters are given, the condition student probability
        distribution is fit with the expected sufficient statistics.

        By default the parameters are lightly regularized, thus map
        estimation is performed rather than mle. The Gaussian-invWishart
        prior is set at instantiation of the class.

        Parameters
        ----------
        z : array_like, shape (`nsamples`, 1)
            z[i] is the state of the parent z in observation i
        y : array_like, shape (`nsamples`, `dim`)
            y[i, :] is the ith 1-by-`dim` observation of the child corresponding to z[i]

        """
        if z is not None and y is not None:
            self.mean, self.cov, self.df = self._dist._fit(z, y, self.ncomponents, self.dim)
        else:
            self._fit()


_mix_normal_doc_default_callparams = """\
mean : array_like, optional
    Mean of the distribution (default zero)
cov : array_like, optional
    Covariance matrix of the distribution (default one)
m: array_like, shape (`nmix`, `ncomponents`)
    Matrix, where each row sums to 1 and `ncomponents` is
    the number of states of the parent:
    :math:`m[j, k] = p(m_t = k | s_t = j)`
"""

_mix_normal_doc_callparams_note = """\
Setting the parameter `mean` to `None` is equivalent to having `mean`
be the zero-vector. The parameter `cov` can be a scalar, in which case
the covariance matrix is the identity times that value, a vector of
diagonal entries for the covariance matrix, or a two-dimensional
array_like.
"""

_mix_normal_doc_frozen_callparams = ""

_mix_normal_doc_frozen_callparams_note = \
    """See class definition for a detailed description of parameters."""

mix_normal_docdict_params = {
    '_doc_default_callparams': _mix_normal_doc_default_callparams,
    '_doc_callparams_note': _mix_normal_doc_callparams_note,
}

mix_normal_docdict_noparams = {
    '_doc_default_callparams': _mix_normal_doc_frozen_callparams,
    '_doc_callparams_note': _mix_normal_doc_frozen_callparams_note,
}


# noinspection PyPep8Naming
class conditional_mix_normal_gen(cond_rv_generic):
    """A conditional mix-normal random variable.

    The `mean` keyword specifies the mean. the `cov` keyword specifies the
    covariance matrix. The `prior` keyword is the prior probability used
    when fitting the distribution via maximum-a-posteriori (MAP). The
    `algorithm` keyword identifies whether to use maximum likelihood estimation
    (MLE) or maximum-a-posteriori (MAP) for fitting the distribution.

    Methods
    -------
    logprior(mean, cov, m, prior)
        Log of the prior distribution.
    logpdf(x, mean, cov, m)
        Log of the probability density function.
    pdf(x, mean, cov, m)
        Probability density function.
    random(nmix, ncomponents, dim)
        Randomly initialize.
    fit(z, y, prior=None, algorithm='map')
        Fit a conditional normal probability distribution.

    Parameters
    ----------
    x : array_like
        Quantiles, with the last axis of `x` denoting the components.
    %(_doc_default_callparams)s
    prior : normal_invwishart
        The distribution's prior, a Normal-inverse-Wishart distribution.
    nmix : int
        The number of mixtures.
    ncomponents : int
        The distribution's number of components.
    dim : int
        Dimensionality of each component.
    algorithm : {'map', 'mle'}
        Distribution fitting algorithm.

    Alternatively, the object may be called (as a function) to fix the mean,
    covariance, m, prior, and algorithm parameters, returning a "frozen" conditional
    mix-normal random variable:

    rv = conditional_mix_normal(mean=None, cov=1, m=None, algorithm='mle')
        - Frozen object with the same methods but holding the given
          mean, covariance, m, prior and algorithm fixed.

    Notes
    -----
    %(_doc_callparams_note)s

    Examples
    --------
    >>> from mlpy.stats import conditional_mix_normal

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    def __init__(self):
        super(conditional_mix_normal_gen, self).__init__()
        self.__doc__ = doccer.docformat(self.__doc__, mix_normal_docdict_params)

    def __call__(self, mean=None, cov=None, m=None, prior=None, algorithm="map"):
        conditional_mix_normal_frozen(mean, cov, m, prior, algorithm)

    def logprior(self, mean, cov, m, prior):
        """Log of the prior distribution.

        Parameters
        ----------
        %(_doc_default_callparams)s
        prior : normal_invwishart
            The distribution's prior, a Normal-inverse-Wishart
            distribution.

        Returns
        -------
        float :
            The log of the prior.

        """
        nmix, = m.shape[0]

        out = 0
        if prior is not None:
            for i in range(nmix):
                out += prior.logpdf(mean, cov)

        # noinspection PyTypeChecker
        return out + np.dot(np.log(m.ravel()+np.finfo(np.float64).eps), prior.pseudo_counts.ravel() - 1)

    def _logpdf(self, x, mean, cov, m, nmix, ncomponents):
        nseq = x.shape[0]
        obs = np.logical_not(np.any(np.isnan(x.T), 0))
        xobs = x[obs]

        log_m = np.log(m)
        log_p = np.empty((ncomponents, nseq))
        log_p.fill(np.nan)

        for i in range(nmix):
            log_p[i, obs] = multivariate_normal.logpdf(xobs, mean[i], cov[i])

        lpr = np.empty((ncomponents, nseq))
        lpr.fill(np.nan)

        for i in range(ncomponents):
            lpr_i = log_m[i] + log_p[:, obs]
            lpr[i, obs] = logsumexp(lpr_i, 0)
        return lpr

    def logpdf(self, x, mean, cov, m):
        """Log of the conditional mix-normal probability density function.

        Return the soft evidence matrix for the given observations `x`.

        Parameters
        ----------
        x : ndarray
            Points at which to evaluate the log of the probability
            density function.
        %(_doc_default_callparams)s

        Returns
        -------
        logpdf : ndarray
            Log of the probability density function evaluated at `x`.

        Notes
        -----
        %(_doc_callparams_note)s

        """
        nmix, ncomponents = m.shape
        return self._logpdf(x, mean, cov, ncomponents, nmix, ncomponents)

    def pdf(self, x, mean, cov, m):
        """Conditional mix-normal probability density function.

        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        %(_doc_default_callparams)s

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`

        """
        # noinspection PyTypeChecker
        return np.exp(self.logpdf(x, mean, cov, m))

    def random(self, nmix, ncomponents, dim):
        """Randomly initialize the conditional mix-normal distribution.

        Parameters
        ----------
        nmix : int
            The number of mixtures.
        ncomponents : int
            The number of components of the distribution.
        dim : int
            The dimension of each component in the distribution.

        Returns
        -------
        mean : array_like, shape (`ncomponents`, dim`)
            The mean of the distribution.
        cov : array_like, shape (`ncomponents`, `dim`, `dim`)
            The covariance matrix of the distribution.

        Raises
        ------
        NotImplementedError
            This function is not yet implemented.

        """
        raise NotImplementedError

    def fit(self, z, y, prior=None, algorithm="map"):
        """Fit a conditional mix-normal probability distribution.

        By default the parameters are lightly regularized, thus map
        estimation is performed rather than mle. The Gaussian-invWishart
        prior is set at instantiation of the class.

        Parameters
        ----------
        z : array_like, shape (`nsamples`, 1)
            z[i] is the state of the parent z in observation i
        y : array_like, shape (`nsamples`, `dim`)
            y[i, :] is the ith 1-by-`dim` observation of the child corresponding to z[i]
        prior: normal_invwishart
            A `normal_invwishart` distribution.

        Raises
        ------
        NotImplementedError
            This function is not yet implemented.

        """
        raise NotImplementedError

conditional_mix_normal = conditional_mix_normal_gen()


# noinspection PyPep8Naming,PyProtectedMember
class conditional_mix_normal_frozen(cond_rv_frozen):
    def __init__(self, mean, cov, m, prior=None, algorithm="map"):
        """Create a "frozen" conditional mix-normal random variable.

        Parameters
        ----------
        mean : array_like, shape (`ncomponents`, `dim`)
            Mean parameters for each component.
        cov : array_like, shape (`ncomponents`, `dim`, `dim`)
            Covariance parameters for each mixture component.
        m: array_like, shape (`nmix`, `ncomponents`)
            Matrix, where each row sums to 1 and `ncomponents` is
            the number of states of the parent:
            :math:`m[j, k] = p(m_t = k | s_t = j)`
        prior : normal_invwishart
            A :data:`normal_invwishart` distribution.
        algorithm : {'map', 'mle'}
            The estimation algorithm to use (map or mle).

        Examples
        --------
        When called with the default parameters, this will create a 1D random
        variable with mean 0 and covariance 1:

        >>> from mlpy.stats import conditional_mix_normal
        >>> r = conditional_mix_normal()
        >>> r.mean
        array([ 0.])
        >>> r.cov
        array([[1.]])

        """
        super(conditional_mix_normal_frozen, self).__init__()

        self.dim = cov[1]
        self.nmix, self.ncomponents = m.shape

        self.mean = mean
        self.cov = cov
        self.m = m

        self.prior = None
        if algorithm == "map":
            pseudo_counts = 2 * np.ones(m.shape)
            self.prior = prior if prior is not None else normal_invwishart(np.zeros(self.dim), 0.01, self.dim + 1,
                                                                           0.1 * np.eye(self.dim), pseudo_counts)
        self._dist = conditional_mix_normal_gen()

    def logprior(self):
        return self._dist.logprior(self.mean, self.cov, self.m, self.prior)

    def logpdf(self, x):
        return self._dist._logpdf(x, self.mean, self.cov, self.m, self.nmix, self.ncomponents)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def expected_sufficient_statistics(self, data, weights):
        """Compute the expected sufficient statistics.

        Compute the expected sufficient statistics for the conditional
        probability distribution.

        Parameters
        ----------
        data : array_like, shape (`nsamples`, `dim`)
            The observations.
        weights : array_like, shape (`nsamples`, `ncomponents`)
            The marginal probability of the discrete parent for each observation.

        Raises
        ------
        NotImplementedError
            This function is not yet implemented.

        """
        raise NotImplementedError

    def fit(self, z=None, y=None):
        """Fit a conditional mix-normal probability distribution.

        If no parameters are given, the condition normal probability
        distribution is fit with the expected sufficient statistics.

        By default the parameters are lightly regularized, thus map
        estimation is performed rather than mle. The Gaussian-invWishart
        prior is set at instantiation of the class.

        Parameters
        ----------
        z : array_like, shape (`nsamples`, 1)
            z[i] is the state of the parent z in observation i
        y : array_like, shape (`nsamples`, `dim`)
            y[i, :] is the ith 1-by-`dim` observation of the child corresponding to z[i]

        Raises
        ------
        NotImplementedError
            This function is not yet implemented.

        """
        raise NotImplementedError
