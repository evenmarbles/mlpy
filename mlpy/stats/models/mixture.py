from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

from abc import ABCMeta, abstractmethod
import numpy as np

from ...optimize.algorithms import EM
from ...auxiliary.array import normalize
from ...cluster.vq import kmeans

from ...stats import multivariate_normal, multivariate_student, conditional_normal, conditional_student
from ...stats import canonize_labels, normalize_logspace, shrink_cov, randpd, stacked_randpd

__all__ = ['MixtureModel', 'DiscreteMM', 'GMM', 'StudentMM']


def _process_parameters(ncomponents, mix_prior=None, mix_weight=None):
    """

    """
    if mix_prior is None:
        mix_prior = 2
    mix_prior = np.asarray(mix_prior, dtype=float)

    if mix_prior.ndim == 0:
        m = mix_prior
        mix_prior = m * np.ones(ncomponents)
    if mix_prior.ndim > 1:
        raise ValueError("Array 'mix_prior' must be at most one-dimensional,"
                         " but mix_prior.ndim = %d" % mix_prior.ndim)
    if mix_prior.shape[0] != ncomponents:
        raise ValueError("Array 'mix_prior' must have %d elements,"
                         " but mix_prior.shape[0] = %d" % (ncomponents, mix_prior.shape[0]))

    if mix_weight is not None:
        if mix_weight.ndim > 1:
            raise ValueError("Array 'mix_weight' must be at most one-dimensional,"
                             " but mix_weight.ndim = %d" % mix_weight.ndim)
    if mix_weight.shape[0] != ncomponents:
        raise ValueError("Array 'mix_weight' must have %d elements,"
                         " but mix_weight.shape[0] = %d" % (ncomponents, mix_weight.shape[0]))

    return ncomponents, mix_prior, mix_weight


# noinspection PyAbstractClass
class MixtureModel(EM):
    """Mixture model base class.

    Representation of a mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a distribution.

    Parameters
    ----------
    ncomponents : int, optional
        Number of mixture components. Default is 1.
    prior : normal_invwishart, optional
        A :data:`.normal_invwishart` distribution.
    mix_prior : float or array_like, shape (`ncomponents`,), optional
        Prior mixture probabilities.
    mix_weight : array_like, shape (`ncomponents`,), optional
        Mixture weights.
    n_iter : int, optional
        Number of EM iterations to perform. Default is 100.
    thresh : float, optional
        Convergence threshold. EM iterations will stop when average
        gain in log-likelihood is below this threshold. Default is 1e-4.
    verbose : bool, optional
        Controls if debug information is printed to the console. Default
        is False.

    Attributes
    ----------
    ncomponents : int
        Number of mixture components.
    dim : int
        Dimensionality of the each component.
    prior : normal_invwishart
        A :data:`.normal_invwishart` distribution.
    mix_prior : array_like, shape (`ncomponents`,)
        Prior mixture probabilities.
    mix_weight : array_like, shape (`ncomponents`,)
        Mixture weights.
    cond_proba : cond_rv_frozen
        Conditional probability distribution.
    n_iter : int
        Number of EM iterations to perform.
    thresh : float
        Convergence threshold.
    verbose : bool
        Controls if debug information is printed to the console.

    Examples
    --------
    >>> from mlpy.stats.models.mixture import GMM

    >>> m = GMM()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    __metaclass__ = ABCMeta

    def __init__(self, ncomponents=1, prior=None, mix_prior=None, mix_weight=None, n_iter=None,
                 thresh=None, verbose=None):
        super(MixtureModel, self).__init__(n_iter, thresh, verbose)

        self.dim = 1
        self.cond_proba = None

        self.prior = prior
        self.ncomponents, self.mix_prior, self.mix_weight = _process_parameters(ncomponents, mix_prior, mix_weight)

    @abstractmethod
    def sample(self, size=1):
        """Generate random samples from the model.

        Parameters
        ----------
        size : int, optional
            Number of samples to generate. Default is 1.

        Returns
        -------
        x : array_like, shape (`size`, `dim`)
            List of samples

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    @abstractmethod
    def score_samples(self, x):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of x under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of x.

        Parameters
        ----------
        x : array_like, shape (`size`, `dim`)
            List of `dim`-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        responsibilities : array_like, shape (`size`, `ncomponents`)
            Posterior probabilities of each mixture component for each
            observation.

        loglik : array_like, shape (size,)
            Log probabilities of each data point in `x`.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    # noinspection PyUnusedLocal
    def score(self, x, y=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        x : array_like, shape (size, dim)
            List of dim-dimensional data points.  Each row
            corresponds to a single data point.
        y : Not used.

        Returns
        -------
        logp : array_like, shape (`size`,)
            Log probabilities of each data point in `x`.

        """
        _, logp = self.score_samples(x)
        return logp

    def predict(self, x):
        """Predict label for data.

        Parameters
        ----------
        x : array_like, shape (`size`, `dim`)

        Returns
        -------
        C : array, shape = (`size`,)

        """
        responsibilities, _ = self.score_samples(x)
        return responsibilities.argmax(axis=1)

    def predict_proba(self, x):
        """
        Predict posterior probability of data under the model.

        Parameters
        ----------
        x : array_like, shape (`size`, `dim`)

        Returns
        -------
        responsibilities : array_like, shape = (`nsamples`, `ncomponents`)
            Returns the probability of the sample for each Gaussian
            (state) in the model.

        """
        responsibilities, logp = self.score_samples(x)
        return responsibilities

    def fit(self, x, n_init=1):
        """Fit the mixture model from the data `x`.

        Estimate model parameters with the expectation-maximization
        algorithm.

        Parameters
        ----------
        x : array_like, shape (`n`, `dim`)
            List of dim-dimensional data points.  Each row
            corresponds to a single data point.
        n_init : int, optional
            Number of random restarts to avoid a local minimum.
            Default is 1.

        """
        self.dim = x.shape[1]
        return self._em(x, n_init=n_init)

    def _compute_mix_prior(self):
        """
        Compute the weighted mixture prior probabilities.

        Returns
        -------
        float :
            The weighted mixture priors.

        """
        if np.all(self.mix_prior == 1):
            return 0
        return np.dot(np.log(self.mix_weight).T, (self.mix_prior - 1))

    def _estep(self, x):
        mix_weight, ll = self.score_samples(x)
        self.cond_proba.expected_sufficient_statistics(x, mix_weight)
        loglik = np.sum(ll) + self.cond_proba.logprior() + self._compute_mix_prior()
        return loglik

    def _mstep(self):
        self.cond_proba.fit()
        self.mix_weight = normalize(self.cond_proba.wsum + self.mix_prior - 1)


class DiscreteMM(MixtureModel):
    """Discrete mixture model class.

    Representation of a discrete mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a distribution.

    Parameters
    ----------
    ncomponents : int, optional
        Number of mixture components. Default is 1.
    prior : normal_invwishart, optional
        A :data:`.normal_invwishart` distribution.
    mix_prior : float or array_like, shape (`ncomponents`,), optional
        Prior mixture probabilities.
    mix_weight : array_like, shape (`ncomponents`,), optional
        Mixture weights.
    transmat : array_like, shape (`ncomponents`, `ncomponents`), optional
        Matrix of transition probabilities between states.
    alpha : float
        Value of Dirichlet prior on observations. Default is 1.1 (1=MLE)
    n_iter : int, optional
        Number of EM iterations to perform. Default is 100.
    thresh : float, optional
        Convergence threshold. EM iterations will stop when average
        gain in log-likelihood is below this threshold. Default is 1e-4.
    verbose : bool, optional
        Controls if debug information is printed to the console. Default
        is False.

    Attributes
    ----------
    ncomponents : int
        Number of mixture components.
    dim : int
        Dimensionality of the each component.
    prior : normal_invwishart
        A :data:`.normal_invwishart` distribution.
    mix_prior : array_like, shape (`ncomponents`,)
        Prior mixture probabilities.
    mix_weight : array_like, shape (`ncomponents`,)
        Mixture weights.
    transmat : array_like, shape (`ncomponents`, `ncomponents`)
        Matrix of transition probabilities between states.
    alpha : float
        Value of Dirichlet prior on observations.
    cond_proba : cond_rv_frozen
        Conditional probability distribution.
    n_iter : int
        Number of EM iterations to perform.
    thresh : float
        Convergence threshold.
    verbose : bool
        Controls if debug information is printed to the console.

    Examples
    --------
    >>> from mlpy.stats.models.mixture import DiscreteMM

    >>> m = DiscreteMM()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    def __init__(self, ncomponents=1, prior=None, mix_prior=None, mix_weight=None, transmat=None, alpha=None,
                 n_iter=None, thresh=None, verbose=None):
        super(DiscreteMM, self).__init__(ncomponents, prior, mix_prior, mix_weight, n_iter, thresh, verbose)

        self.transmat = transmat
        self.alpha = alpha if alpha is not None else 1.1

    def sample(self, size=1):
        """Generate random samples from the model.

        Parameters
        ----------
        size : int, optional
            Number of samples to generate. Default is 1.

        Returns
        -------
        x : array_like, shape (`size`, `dim`)
            List of samples

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    def score_samples(self, x):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of x under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of x.

        Parameters
        ----------
        x : array_like, shape (`size`, `dim`)
            List of `dim`-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        responsibilities : array_like, shape (`size`, `ncomponents`)
            Posterior probabilities of each mixture component for each
            observation.

        loglik : array_like, shape (size,)
            Log probabilities of each data point in `x`.

        """
        n, dim = x.shape
        logp = np.log(self.mix_weight)
        logt = np.log(self.cond_proba.T + np.finfo(np.float64).eps)
        lijk = np.zeros((n, dim, self.ncomponents))

        x = canonize_labels(x)

        for i in range(dim):
            ndx = ~np.isnan(x[:, i])
            lijk[ndx, i] = logt[:, x[ndx, i], i].T
        logpz = logp + np.squeeze(np.sum(lijk, 1))

        logpz, ll = normalize_logspace(logpz)
        pz = np.exp(logpz)
        return pz, ll

    def _initialize(self, x, init_count):
        raise NotImplementedError


class GMM(MixtureModel):
    """Gaussian mixture model class.

    Representation of a gaussian mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a distribution.

    Parameters
    ----------
    ncomponents : int, optional
        Number of mixture components. Default is 1.
    prior : normal_invwishart, optional
        A :data:`.normal_invwishart` distribution.
    mix_prior : float or array_like, shape (`ncomponents`,), optional
        Prior mixture probabilities.
    mix_weight : array_like, shape (`ncomponents`,), optional
        Mixture weights.
    mean : array, shape (`ncomponents`, `nfeatures`)
        Mean parameters for each state.
    cov : array, shape (`ncomponents`, `nfeatures`, `nfeatures`)
        Covariance parameters for each state.
    n_iter : int, optional
        Number of EM iterations to perform. Default is 100.
    thresh : float, optional
        Convergence threshold. EM iterations will stop when average
        gain in log-likelihood is below this threshold. Default is 1e-4.
    verbose : bool, optional
        Controls if debug information is printed to the console. Default
        is False.

    Attributes
    ----------
    ncomponents : int
        Number of mixture components.
    dim : int
        Dimensionality of the each component.
    prior : normal_invwishart
        A :data:`.normal_invwishart` distribution.
    mix_prior : array_like, shape (`ncomponents`,)
        Prior mixture probabilities.
    mix_weight : array_like, shape (`ncomponents`,)
        Mixture weights.
    mean : array, shape (`ncomponents`, `nfeatures`)
        Mean parameters for each state.
    cov : array, shape (`ncomponents`, `nfeatures`, `nfeatures`)
        Covariance parameters for each state.
    cond_proba : conditional_normal
        A :data:`.conditional_normal` probability distribution.
    n_iter : int
        Number of EM iterations to perform.
    thresh : float
        Convergence threshold.
    verbose : bool
        Controls if debug information is printed to the console.

    Examples
    --------
    >>> from mlpy.stats.models.mixture import GMM

    >>> m = GMM()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    @property
    def mean(self):
        """Mean parameters of the emission.

        Returns
        -------
        array, shape (`ncomponents`, `nfeatures`) :
            The mean parameters.

        """
        return self.cond_proba.mean

    @property
    def cov(self):
        """Covariance parameters of the emission.

        array, shape (`ncomponents`, `nfeatures`, `nfeatures`):
            Covariance parameters.

        """
        return self.cond_proba.cov

    def __init__(self, ncomponents=1, prior=None, mix_prior=None, mix_weight=None, mean=None, cov=None,
                 n_iter=None, thresh=None, verbose=None):
        super(GMM, self).__init__(ncomponents, prior, mix_prior, mix_weight, n_iter, thresh, verbose)

        if mean is not None and cov is not None:
            self.cond_proba = conditional_normal(mean, cov, self.prior)

    def sample(self, size=1):
        """Generate random samples from the model.

        Parameters
        ----------
        size : int, optional
            Number of samples to generate. Default is 1.

        Returns
        -------
        x : array_like, shape (`size`, `dim`)
            List of samples

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    def score_samples(self, x):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of x under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of x.

        Parameters
        ----------
        x : array_like, shape (`size`, `dim`)
            List of `dim`-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        responsibilities : array_like, shape (`size`, `ncomponents`)
            Posterior probabilities of each mixture component for each
            observation.

        loglik : array_like, shape (size,)
            Log probabilities of each data point in `x`.

        """
        n = x.shape[0]
        logp = np.log(self.mix_weight)
        logpz = np.zeros((n, self.ncomponents))

        for i in range(self.ncomponents):
            logpz[:, i] = logp[i] + multivariate_normal.logpdf(x, self.cond_proba.mean[i], self.cond_proba.cov[i])

        logpz, ll = normalize_logspace(logpz)
        pz = np.exp(logpz)
        return pz, ll

    def _initialize(self, x, init_count):
        if init_count == 0:
            if self.cond_proba is None and self.mix_weight is None:
                mean, assign = kmeans(x, self.ncomponents, return_assignment=True)
                cov = np.zeros((self.ncomponents, self.dim, self.dim))
                counts = np.zeros(self.ncomponents)
                for i in range(self.ncomponents):
                    # noinspection PyTypeChecker
                    ndx = np.nonzero(assign == i)
                    counts[i] = np.size(ndx)
                    c = shrink_cov(x[ndx])
                    cov[i] = c
                self.mix_weight = normalize(counts)
                for i in range(self.ncomponents):
                    if np.any(np.isnan(cov[i])):
                        cov[i] = randpd(self.dim)
                self.cond_proba = conditional_normal(mean, cov, self.prior, algorithm="map")
        else:
            mean = np.random.randn(self.dim, self.ncomponents)
            regularizer = 2
            cov = stacked_randpd(self.dim, self.ncomponents, regularizer)
            self.mix_weight = normalize(np.random.rand(self.ncomponents, 1) + 2)
            self.cond_proba = conditional_normal(mean, cov, self.prior, algorithm="map")


class StudentMM(GMM):
    """Student mixture model class.

    Representation of a student mixture model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a distribution.

    Parameters
    ----------
    ncomponents : int, optional
        Number of mixture components. Default is 1.
    prior : normal_invwishart, optional
        A :data:`.normal_invwishart` distribution.
    mix_prior : float or array_like, shape (`ncomponents`,), optional
        Prior mixture probabilities.
    mix_weight : array_like, shape (`ncomponents`,), optional
        Mixture weights.
    mean : array, shape (`ncomponents`, `nfeatures`)
        Mean parameters for each state.
    cov : array, shape (`ncomponents`, `nfeatures`, `nfeatures`)
        Covariance parameters for each state.
    df : array, shape (`ncomponents`,)
        Degrees of freedom.
    n_iter : int, optional
        Number of EM iterations to perform. Default is 100.
    thresh : float, optional
        Convergence threshold. EM iterations will stop when average
        gain in log-likelihood is below this threshold. Default is 1e-4.
    verbose : bool, optional
        Controls if debug information is printed to the console. Default
        is False.

    Attributes
    ----------
    ncomponents : int
        Number of mixture components.
    dim : int
        Dimensionality of the each component.
    prior : normal_invwishart
        A :data:`.normal_invwishart` distribution.
    mix_prior : array_like, shape (`ncomponents`,)
        Prior mixture probabilities.
    mix_weight : array_like, shape (`ncomponents`,)
        Mixture weights.
    mean : array, shape (`ncomponents`, `nfeatures`)
        Mean parameters for each state.
    cov : array, shape (`ncomponents`, `nfeatures`, `nfeatures`)
        Covariance parameters for each state.
    df : array, shape (`ncomponents`,)
        Degrees of freedom.
    cond_proba : conditional_student
        A :data:`.conditional_student` probability distribution.
    n_iter : int
        Number of EM iterations to perform.
    thresh : float
        Convergence threshold.
    verbose : bool
        Controls if debug information is printed to the console.

    Examples
    --------
    >>> from mlpy.stats.models.mixture import StudentMM

    >>> m = StudentMM()

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    def __init__(self, ncomponents=1, prior=None, mix_prior=None, mix_weight=None, mean=None, cov=None, df=None,
                 n_iter=None, thresh=None, verbose=None):
        super(StudentMM, self).__init__(ncomponents, prior, mix_prior, mix_weight, mean, cov, n_iter, thresh,
                                        verbose)

        self.df = 10 * np.ones(self.ncomponents) if df is None else df

        if mean is not None and cov is not None:
            self.cond_proba = conditional_student(mean, cov, self.df, self.prior)

    def sample(self, size=1):
        """Generate random samples from the model.

        Parameters
        ----------
        size : int, optional
            Number of samples to generate. Default is 1.

        Returns
        -------
        x : array_like, shape (`size`, `dim`)
            List of samples

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    def score_samples(self, x):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of x under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of x.

        Parameters
        ----------
        x : array_like, shape (`size`, `dim`)
            List of `dim`-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        responsibilities : array_like, shape (`size`, `ncomponents`)
            Posterior probabilities of each mixture component for each
            observation.

        loglik : array_like, shape (size,)
            Log probabilities of each data point in `x`.

        """
        n = x.shape[0]
        logp = np.log(self.mix_weight)
        logpz = np.zeros((n, self.ncomponents))

        for i in range(self.ncomponents):
            logpz[:, i] = logp[i] + multivariate_student.logpdf(x, self.cond_proba.mean[i], self.cond_proba.cov[i],
                                                                self.cond_proba.df)

        logpz, ll = normalize_logspace(logpz)
        pz = np.exp(logpz)
        return pz, ll

    def _initialize(self, x, init_count):
        if self.cond_proba is None:
            super(StudentMM, self)._initialize(x, init_count)
            self.cond_proba = conditional_student(self.cond_proba.mean, self.cond_proba.cov, self.df, self.prior)
