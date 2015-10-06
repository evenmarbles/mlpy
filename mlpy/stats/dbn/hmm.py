"""
Hidden Markov Models
====================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   HMM
   DiscreteHMM
   GaussianHMM
   StudentHMM
   GMMHMM

"""
from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

from abc import ABCMeta, abstractmethod

import numpy as np
# noinspection PyPackageRequirements
from sklearn.mixture import GMM

from ...auxiliary.array import accum, normalize
from ...optimize.algorithms import EM
from ...libs import hmmc
from ..models.mixture import StudentMM
from ..models import markov
from .. import normalize_logspace, nonuniform
from .. import conditional_normal, conditional_student, conditional_mix_normal
from .. import multivariate_normal, multivariate_student


class HMM(EM):
    """Hidden Markov Model base class.

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    See the instance documentation for details specific to a
    particular object.

    Parameters
    ----------
    ncomponents : int
        Number of states in the model.
    startprob_prior : array, shape (`ncomponents`,)
        Initial state occupation prior distribution.
    startprob : array, shape (`ncomponents`,)
        Initial state occupation distribution.
    transmat_prior : array, shape (`ncomponents`, `ncomponents`)
        Matrix of prior transition probabilities between states.
    transmat : array, shape (`ncomponents`, `ncomponents`)
        Matrix of transition probabilities between states.
    emission_prior : normal_invwishart
        Initial emission parameters, a normal-inverse Wishart distribution.
    emission : cond_rv_frozen
        The conditional probability distribution used for the emission.
    n_iter : int
        Number of iterations to perform during training, optional.
    thresh : float
        Convergence threshold, optional.
    verbose : bool
        Controls if debug information is printed to the console, optional.

    Attributes
    ----------
    ncomponents : int
        The number of hidden states.
    nfeatures : int
        Dimensionality of the Gaussian emission.
    startprob_prior : array, shape (`ncomponents`,)
        Initial state occupation prior distribution.
    startprob : array, shape (`ncomponents`,)
        Initial state occupation distribution.
    transmat_prior : array, shape (`ncomponents`, `ncomponents`)
        Matrix of prior transition probabilities between states.
    transmat : array, shape (`ncomponents`, `ncomponents`)
        Matrix of transition probabilities between states.
    emission_prior : normal_invwishart
        Initial emission parameters, a normal-inverse Wishart distribution.
    emission : cond_rv_frozen
        The conditional probability distribution used for the emission.

    Examples
    --------
    >>> from mlpy.stats.dbn.hmm import GaussianHMM

    >>> model = GaussianHMM(ncomponents=2, startprob_prior=[3, 2])

    Create a gaussian hidden Markov model

    >>> import scipy.io
    >>> mat = scipy.io.loadmat('data/speechDataDigits4And5.mat'))
    >>> x = np.hstack([mat['train4'][0], mat['train5'][0]])

    Load data used for fitting the HMM and fit the HMM:

    >>> model.fit(x, n_init=3)

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    __metaclass__ = ABCMeta

    @property
    def startprob_prior(self):
        """Vector of initial probabilities for each state.

        Returns
        -------
        startprob_prior : array, shape (`ncomponents`,)
            The initial probabilities.

        """
        return self._startprob_prior

    @startprob_prior.setter
    def startprob_prior(self, pi_prior):
        self._startprob_prior = np.ones(self.ncomponents,
                                        dtype=float) if pi_prior is None else np.asarray(pi_prior, dtype=np.float64)

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(self._startprob_prior):
            normalize(self._startprob_prior)

        if len(self._startprob_prior) != self.ncomponents:
            raise ValueError('pi_prior must have length ncomponents')

    @property
    def transmat_prior(self):
        """Transition probability matrix.

        Returns
        -------
        transmat_prior : array, shape (`ncomponents`, `ncomponents`)
            Matrix of transition probabilities from each state to every other state.

        """
        return self._transmat_prior

    @transmat_prior.setter
    def transmat_prior(self, trans_prior):
        self._transmat_prior = np.ones((self.ncomponents, self.ncomponents),
                                       float) if trans_prior is None else np.asarray(trans_prior, dtype=np.float64)

        # check if there exists a component whose value is exactly zero
        # if so, add a small number and re-normalize
        if not np.alltrue(self._transmat_prior):
            normalize(self._transmat_prior)

        if np.asarray(self._transmat_prior).shape != (self.ncomponents, self.ncomponents):
            self._transmat_prior = np.tile(self._transmat_prior, (self.ncomponents, 1))

    def __init__(self, ncomponents=1, startprob_prior=None, startprob=None, transmat_prior=None, transmat=None,
                 emission_prior=None, emission=None, n_iter=None, thresh=None, verbose=None):
        super(HMM, self).__init__(n_iter, thresh, verbose)

        self.ncomponents = ncomponents
        self.nfeatures = 1

        self._startprob_prior = None
        self._transmat_prior = None

        self._fit_X = None
        """:type:ndarray[ndarray[float]]"""

        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior

        self.startprob = startprob
        self.transmat = transmat

        self.emission_prior = emission_prior
        self.emission = emission

    def score_samples(self, obs):
        """Compute the log probability of the evidence.

        Compute the log probability of the evidence (likelihood) under the
        model and the posteriors.

        Parameters
        ----------
        obs : array_like, shape (`n`, `len`, `nfeatures`)
            Sequence of `nfeatures`-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        logp : float
            Log likelihood of the sequence `obs`.

        posteriors : array_like, shape (`n`, `ncomponents`)
            Posterior probabilities of each state for each observation

        """
        n = obs.shape[0]
        logp = np.zeros(n)
        posterior = np.zeros((n, obs.shape[1], self.ncomponents), dtype=np.float64)

        for i, x in enumerate(obs):
            log_b = self.emission.logpdf(x)
            log_b, scale = normalize_logspace(log_b.T)
            b = np.exp(log_b)
            logp[i], alpha = self._forward(self.startprob, self.transmat, b.T)
            _, gamma = self._backward(self.transmat, b.T, alpha)
            logp[i] += np.sum(scale)
            posterior[i] = gamma.T

        return logp, posterior

    def score(self, obs):
        """
        Compute log probability of the evidence (likelihood) under the model.

        Parameters
        ----------
        obs : array_like, shape (`n`, `len`, `nfeatures`)
            Sequence of `nfeatures`-dimensional data points. Each row
            corresponds to a single point in the sequence.

        Returns
        -------
        logp : float
            Log likelihood of the sequence `obs`.

        """
        n = obs.shape[0]
        logp = np.zeros(n)

        for i, x in enumerate(obs):
            log_b = self.emission.logpdf(x)
            log_b, scale = normalize_logspace(log_b.T)
            b = np.exp(log_b)
            logp[i], alpha = self._forward(self.startprob, self.transmat, b.T)
            logp[i] += np.sum(scale)

        return logp

    def predict_proba(self, obs):
        """Compute the posterior probability for each state in the model.

        Parameters
        ----------
        obs : array_like, shape (`n`, `len`, `nfeatures`)
                Sequence of `nfeatures`-dimensional data points. Each row
                corresponds to a single point in the sequence.

        Returns
        -------
        posteriors : array_like, shape (`n`, `ncomponents`)
            Posterior probabilities of each state for each observation

        """
        _, posteriors = self.score_samples(obs)
        return posteriors

    def sample(self, length, size=1):
        """Generates random samples from the model.

        Parameters
        ----------
        length : int or ndarray[int]
            Length of a sample
        size : int, optional
            Number of samples to generate. Default is 1.

        Returns
        -------
        obs : array_like, shape (`n`, `ni`, `nfeatures`)
            List of samples, where `n` is the number of samples, `ni` is the
            length of the i-th sample, and each observation has `nfeatures`.

        hidden_states : array_like, shape (`n`, `ni`)
            List of hidden states, where `n` is the number of samples, `ni` is
            the i-th hidden state.

        """
        # noinspection PyTypeChecker
        length = np.tile(length, size) if not hasattr(length, "__len__") else length
        assert (length.size == size)

        hidden_states = np.empty((size, np.max(length)), dtype=np.int32)
        obs = np.empty((size, np.max(length), self.nfeatures), dtype=np.float64) * np.nan

        for i in range(size):
            hidden_states[i] = markov.sample(self.startprob, self.transmat, size=length[i])
            for t in range(length[i]):
                obs[i][t] = self._generate_sample_from_state(hidden_states[i][t])

        if size == 1:
            hidden_states = hidden_states[0]
            obs = obs[0]

        return obs, hidden_states

    def decode(self, obs, algorithm="viterbi"):
        """Find the most likely state sequence.

        Find the most likely state sequence corresponding to the observation `obs`.
        Uses the given algorithm for decoding.

        Parameters
        ----------
        obs : array_like, shape (`nfeatures`, `T`)
                The local evidence vector.
        algorithm : {'viterbi', 'map'}
            Decoder algorithm to be used.

        Returns
        -------
        best_path : array_like, shape (`n`,)
            The most likely states for each observation

        loglik : float
            Log probability of the maximum likelihood path through the HMM

        """
        algorithm = algorithm if algorithm in frozenset(("viterbi", "map")) else "viterbi"
        return {
            "viterbi": self._decode_viterbi,
            "map": self._decode_map
        }[algorithm](obs)

    def fit(self, obs, n_init=1):
        """Estimate model parameters.

        Parameters
        ----------
        obs : array_like, shape (`n`, `ni`, `nfeatures`)
            List of observation sequences, where `n` is the number of sequences, `ni` is
            the length of the i_th observation, and each observation has `nfeatures` features.

        Returns
        -------
        float :
            log likelihood of the sequence `obs`

        """
        self.nfeatures = obs[0].shape[0]
        self._fit_X = obs
        return self._em(obs, n_init=n_init)

    @abstractmethod
    def _initialize(self, obs, init_count):
        """Perform initialization step before entering the EM algorithm.

        Parameters
        ----------
        obs : array_like, shape (`n`, `ni`, `nfeatures`)
            List of observation sequences, where `n` is the number of sequences, `ni` is
            the length of the i_th observation, and each observation has `nfeatures` features.
        init_count : int
            Restart counter

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    @abstractmethod
    def _generate_sample_from_state(self, state):
        """Generate a sample from the given current state.

        Parameters
        ----------
        state : int
            Current state.

        Returns
        -------
        sample: int
            An observation sampled for the given state

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    def _decode_viterbi(self, obs, scale=True):
        """Find most likely (Viterbi) state sequence corresponding to `obs`.

        Parameters
        ----------
        obs : array_like, shape (`nfeatures`, `T`)
            The local evidence vector.

        Returns
        -------
        best_path: array_like, shape (`n`,)
            The most likely states for each observation

        loglik: float
            Log probability of the maximum likelihood path through the HMM

        """
        log_b = self.emission.logpdf(obs.T)
        obslik = np.exp(log_b)

        seq_len = obslik.shape[1]
        delta = np.zeros((self.ncomponents, seq_len), dtype=np.float64)
        psi = np.zeros((self.ncomponents, seq_len), dtype=np.int32)
        best_path = np.zeros(seq_len, dtype=np.int32)
        scales = np.ones(seq_len, dtype=np.float64)

        delta[:, 0] = self.startprob * obslik[:, 0]
        if scale:
            delta[:, 0], n = normalize(delta[:, 0], return_scale=True)
            scales[0] = 1.0 / n

        psi[:, 0] = 0  # arbitrary, since there is no predecessor to t=0

        for t in range(1, seq_len):
            for j in range(self.ncomponents):
                m = delta[:, t - 1] * self.transmat[:, j]
                delta[j, t] = np.max(m)
                psi[j, t] = np.argmax(m)
                delta[j, t] = delta[j, t] * obslik[j, t]
            if scale:
                delta[:, t], n = normalize(delta[:, t], return_scale=True)
                scales[t] = 1.0 / n

        best_path[seq_len - 1] = np.argmax(delta[:, seq_len - 1])
        for t in range(seq_len - 2, -1, -1):
            best_path[t] = psi[best_path[t + 1], t + 1]

        if scale:
            loglik = -np.sum(np.log(scales))
        else:
            p = np.max(delta[:, seq_len - 1])
            loglik = np.log(p)

        return best_path, loglik

    def _decode_map(self, obs):
        """Find most likely (MAP) state sequence corresponding to `obs`.

        Uses the maximum a posteriori estimation.

        Parameters
        ----------
        obs : array_like, shape (`nfeatures`, `T`)
            The local evidence vector.

        Returns
        -------
        best_path: array_like, shape (`n`,)
            The most likely states for each observation

        loglik: float
            Log probability of the maximum likelihood path through the HMM

        """
        _, posteriors = self.score_samples(np.array([obs.T]))
        best_path = np.argmax(posteriors[0], axis=1)
        loglik = np.max(posteriors[0], axis=1).sum()
        return best_path, loglik

    def _estep(self, obs):
        """Perform expectation step of the EM algorithm.

        Parameters
        ----------
        obs : array_like, shape (`n`, `ni`, `nfeatures`)
            List of observation sequences, where `n` is the number of sequences, `ni` is
            the length of the i_th observation, and each observation has `nfeatures` features.

        Returns
        -------
        loglik : float
            Log likelihood of the observation `obs`

        """
        stacked_obs = self._stack_obs(obs)
        seq_idx = np.cumsum([0] + [x.shape[1] for x in obs])
        nstacked = stacked_obs.shape[0]
        start_counts = np.zeros(self.ncomponents)
        trans_counts = np.zeros((self.ncomponents, self.ncomponents))
        weights = np.zeros((nstacked, self.ncomponents))
        loglik = 0
        nobs = obs.shape[0]
        log_b = self.emission.logpdf(stacked_obs)
        log_b, scale = normalize_logspace(log_b.T)
        b = np.exp(log_b)

        for i in range(nobs):
            ndx = np.arange(seq_idx[i], seq_idx[i + 1])
            bi = b[ndx]
            logp, alpha = hmmc.forward(self.startprob, self.transmat, bi.T)
            # logp, alpha = self._forward(self.startprob, self.transmat, bi.T)
            beta, gamma = hmmc.backward(self.transmat, bi.T, alpha)
            # beta, gamma = self._backward(self.transmat, bi.T, alpha)
            loglik += logp
            xi_summed = hmmc.computeTwoSliceSum(alpha, beta, self.transmat, bi.T)
            # xi_summed = self._compute_two_slice_sum(alpha, beta, self.transmat, bi.T)
            start_counts += gamma[:, 0]
            trans_counts += xi_summed
            weights[ndx] += gamma.T

        loglik += np.sum(scale)
        log_prior = np.dot(np.log(np.ravel(self.transmat) + np.spacing(1)),
                           np.ravel(self.transmat_prior)) + np.dot(np.log(self.startprob + np.spacing(1)),
                                                                   self.startprob_prior)
        loglik += log_prior

        # emission component
        self.startprob = normalize(start_counts + self.startprob_prior)
        self.transmat = normalize(trans_counts + self.transmat_prior, 1)
        self.emission.expected_sufficient_statistics(stacked_obs, weights)
        loglik += self.emission.logprior()
        return loglik

    def _mstep(self):
        """Perform maximization step of the EM algorithm."""
        self.emission.fit()

    def _stack_obs(self, obs):
        """Stack observations.

        Stack observations to a sequence of (`n`*`ni`)-by-`nfeatures`-dimensional
        data points. Each row corresponds to a single data point in a sequence.

        Parameters
        ----------
        obs : array_like, shape (`n`, `ni`, `nfeatures`)
            List of observation sequences, where `n` is the number of sequences, `ni` is
            the length of the i_th observation, and each observation has `nfeatures` features.

        Returns
        -------
        stacked_obs : array_like, shape (`n`*`ni`, `nfeatures`)
            List of observation sequences

        """
        stacked_obs = np.empty((0, self.nfeatures), dtype=np.float64)
        for i in range(obs.shape[0]):
            # d = obs[i].T if obs[i].shape[0] == self.nfeatures else obs[i]
            stacked_obs = np.vstack([stacked_obs, obs[i].T])
        return stacked_obs

    def _rand_init(self):
        """Randomly initialize the prior and the transition probabilities.
        """
        # noinspection PyArgumentList
        self.startprob = normalize(np.random.rand(self.ncomponents) + self.startprob_prior - 1)
        self.transmat = normalize(
            np.random.rand(self.ncomponents, self.ncomponents) + self.transmat_prior - 1, 1)

    def _init_with_mix_model(self, obs, pz):
        """Initialize the prior probabilities and transition probabilities.

        Initialize the prior probabilities and transition probabilities during the
        initialization step before the EM algorithm.

        Parameters
        ----------
        obs : array_like, shape (`n`, `ni`, `nfeatures`)
            List of observation sequences, where `n` is the number of sequences, `ni` is
            the length of the i_th observation, and each observation has `nfeatures` features.
        pz : array_like, shape (`n`*`ni`, `ncomponents`)
            Posterior probability of the observation

        """
        z = np.argmax(pz, axis=1)

        if self.transmat is None:
            self.transmat = accum(np.vstack([z[0:-1], z[1::]]).T, 1,
                                  size=(self.ncomponents, self.ncomponents))
            # regularize
            self.transmat = normalize(self.transmat + np.ones(self.transmat.shape), 1)

        if self.startprob is None:
            seq_idx = np.cumsum([0] + [x.shape[1] for x in obs])
            self.startprob = np.bincount(z[seq_idx[0:-1]], minlength=self.ncomponents)
            self.startprob = normalize(self.startprob + np.ones(self.startprob.shape))  # regularize

    # noinspection PyMethodMayBeStatic
    def _forward(self, pi, transmat, softev):
        n, m = softev.shape
        scale = np.zeros(m)
        at = transmat.T
        alpha = np.zeros((n, m))
        alpha[:, 0], scale[0] = normalize(np.ravel(pi) * softev[:, 0], return_scale=True)
        for t in range(1, m):
            alpha[:, t], scale[t] = normalize(np.dot(at, alpha[:, t - 1]) * softev[:, t], return_scale=True)
        loglik = np.sum(np.log(scale + np.finfo(float).eps))
        return loglik, alpha

    # noinspection PyMethodMayBeStatic
    def _backward(self, transmat, softev, alpha):
        n, m = softev.shape
        beta = np.zeros((n, m))
        beta[:, m - 1] = np.ones(n)
        for t in reversed(range(m)):
            beta[:, t - 1] = normalize(np.dot(transmat, beta[:, t] * softev[:, t]))
        gamma = normalize(alpha * beta, 0)
        return beta, gamma

    # noinspection PyMethodMayBeStatic
    def _compute_two_slice_sum(self, alpha, beta, transmat, softev):
        n, m = softev.shape
        xi_summed = np.zeros((n, n))
        for t in reversed(range(1, m)):
            b = beta[:, t] * softev[:, t]
            xit = transmat * np.dot(alpha[:, t - 1].reshape(n, 1), b.reshape(1, n))
            xi_summed += np.true_divide(xit, np.sum(np.ravel(xit)))
        return xi_summed


class DiscreteHMM(HMM):
    """Hidden Markov Model with discrete(multinomial) emissions.

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    Parameters
    ----------
    ncomponents : int
        Number of states in the model.
    startprob_prior : array, shape (`ncomponents`,)
        Initial state occupation prior distribution.
    startprob : array, shape (`ncomponents`,)
        Initial state occupation distribution.
    transmat_prior : array, shape (`ncomponents`, `ncomponents`)
        Matrix of prior transition probabilities between states.
    transmat : array, shape (`ncomponents`, `ncomponents`)
        Matrix of transition probabilities between states.
    emission_prior : normal_invwishart
        Initial emission parameters, a normal-inverse Wishart distribution.
    emission : cond_rv_frozen
        The conditional probability distribution used for the emission.
    n_iter : int
        Number of iterations to perform during training, optional.
    thresh : float
        Convergence threshold, optional.
    verbose : bool
        Controls if debug information is printed to the console, optional.

    Attributes
    ----------
    ncomponents : int
        The number of hidden states.
    nfeatures : int
        Dimensionality of the Gaussian emission.
    startprob_prior : array, shape (`ncomponents`,)
        Initial state occupation prior distribution.
    startprob : array, shape (`ncomponents`,)
        Initial state occupation distribution.
    transmat_prior : array, shape (`ncomponents`, `ncomponents`)
        Matrix of prior transition probabilities between states.
    transmat : array, shape (`ncomponents`, `ncomponents`)
        Matrix of transition probabilities between states.
    emission_prior : normal_invwishart
        Initial emission parameters, a normal-inverse Wishart distribution.
    emission : cond_rv_frozen
        The conditional probability distribution used for the emission.

    Examples
    --------
    >>> from mlpy.stats.dbn.hmm import DiscreteHMM
    >>> DiscreteHMM(ncomponents=2)
    ...

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    def __init__(self, ncomponents=1, startprob_prior=None, startprob=None, transmat_prior=None, transmat=None,
                 emission_prior=None, emission=None, n_iter=None, thresh=None, verbose=None):
        super(DiscreteHMM, self).__init__(ncomponents, startprob_prior, startprob, transmat_prior, transmat,
                                          emission_prior, emission, n_iter, thresh, verbose)

    def _generate_sample_from_state(self, state):
        """Generate a sample from the given current state.

        Parameters
        ----------
        state : int
            Current state.

        Returns
        -------
        sample: int
            An observation sampled for the given state

        """
        return nonuniform.rvs(self.emission.T[state])

    def _initialize(self, obs, init_count):
        """Perform initialization step before entering the EM algorithm.

        Parameters
        ----------
        obs : array_like, shape (`n`, `ni`, `nfeatures`)
            List of observation sequences, where `n` is the number of sequences, `ni` is
            the length of the i_th observation, and each observation has `nfeatures` features.
        init_count : int
            Restart counter

        Raises
        ------
        NotImplementedError
            This function is not implemented yet.

        """
        raise NotImplementedError


class GaussianHMM(HMM):
    """
    Hidden Markov Model with Gaussian emissions.

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    Parameters
    ----------
    ncomponents : int
        Number of states in the model.
    startprob_prior : array, shape (`ncomponents`,)
        Initial state occupation prior distribution.
    startprob : array, shape (`ncomponents`,)
        Initial state occupation distribution.
    transmat_prior : array, shape (`ncomponents`, `ncomponents`)
        Matrix of prior transition probabilities between states.
    transmat : array, shape (`ncomponents`, `ncomponents`)
        Matrix of transition probabilities between states.
    emission_prior : normal_invwishart
        Initial emission parameters, a normal-inverse Wishart distribution.
    emission : conditional_normal_frozen
        The conditional probability distribution used for the emission.
    n_iter : int
        Number of iterations to perform during training, optional.
    thresh : float
        Convergence threshold, optional.
    verbose : bool
        Controls if debug information is printed to the console, optional.

    Attributes
    ----------
    ncomponents : int
        The number of hidden states.
    nfeatures : int
        Dimensionality of the Gaussian emission.
    startprob_prior : array, shape (`ncomponents`,)
        Initial state occupation prior distribution.
    startprob : array, shape (`ncomponents`,)
        Initial state occupation distribution.
    transmat_prior : array, shape (`ncomponents`, `ncomponents`)
        Matrix of prior transition probabilities between states.
    transmat : array, shape (`ncomponents`, `ncomponents`)
        Matrix of transition probabilities between states.
    emission_prior : normal_invwishart
        Initial emission parameters, a normal-inverse Wishart distribution.
    emission : cond_rv_frozen
        The conditional probability distribution used for the emission.
    mean : array, shape (`ncomponents`, `nfeatures`)
        Mean parameters for each state.
    cov : array, shape (`ncomponents`, `nfeatures`, `nfeatures`)
        Covariance parameters for each state.

    Examples
    --------
    >>> from mlpy.stats.dbn.hmm import GaussianHMM
    >>> GaussianHMM(ncomponents=2)
    ...

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    @property
    def mean(self):
        """The mean parameters for each state

        Returns
        -------
        array, shape (`ncomponents`, `nfeatures`) :
            Mean parameters for each state.
        """
        return self.emission.mean

    @property
    def cov(self):
        """Covariance parameters for each state.

        Returns
        -------
        array, shape (`ncomponents`, `nfeatures`, `nfeatures`) :
            Covariance parameters for each state as a full matrix

        """
        return self.emission.cov

    def __init__(self, ncomponents=1, startprob_prior=None, startprob=None, transmat_prior=None, transmat=None,
                 emission_prior=None, emission=None, n_iter=None, thresh=None, verbose=None):
        super(GaussianHMM, self).__init__(ncomponents, startprob_prior, startprob, transmat_prior, transmat,
                                          emission_prior, emission, n_iter, thresh, verbose)
        if emission:
            self.nfeatures = emission.dim

    def _generate_sample_from_state(self, state):
        """Generate a sample from the given current state.

        Parameters
        ----------
        state : int
            Current state.

        Returns
        -------
        sample: int
            An observation sampled for the given state

        """
        return multivariate_normal.rvs(self.emission.mean[state], self.emission.cov[state], size=1)

    def _initialize(self, obs, init_count):
        """Perform initialization step before entering the EM algorithm.

        Parameters
        ----------
        obs : array_like, shape (`n`, `ni`, `nfeatures`)
            List of observation sequences, where `n` is the number of sequences, `ni` is
            the length of the i_th observation, and each observation has `nfeatures` features.
        init_count : int
            Restart counter

        """
        if self.emission is None or self.startprob is None or self.transmat is None:
            if init_count == 0:
                stacked_obs = self._stack_obs(obs)

                gmm = GMM(n_components=self.ncomponents, covariance_type='full')
                gmm.fit(stacked_obs)

                if self.transmat is None or self.startprob is None:
                    pz = gmm.predict_proba(stacked_obs)
                    self._init_with_mix_model(obs, pz)

                if self.emission is None:
                    self.emission = conditional_normal(gmm.means_, gmm.covars_, algorithm='map')
                    # regularize MLE
                    for i in range(self.emission.ncomponents):
                        self.emission.cov[i] += np.eye(self.emission.dim)
            else:
                stacked_obs = self._stack_obs(obs)
                mean = np.zeros((self.ncomponents, self.nfeatures))
                cov = np.zeros((self.ncomponents, self.nfeatures, self.nfeatures))
                for i in range(self.ncomponents):
                    xx = stacked_obs + np.random.randn(stacked_obs.shape[0], stacked_obs.shape[1])
                    mean[i] = np.mean(xx, 0)
                    cov[i, :, :] = np.cov(xx, rowvar=0)

                if self.emission is None:
                    self.emission = conditional_normal(mean, cov, algorithm='map')
                else:
                    self.emission.mean = mean
                    self.emission.cov = cov
                self._rand_init()

        if self.emission_prior:
            self.emission.prior = self.emission_prior


class StudentHMM(HMM):
    """
    Hidden Markov Model with Student emissions

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    Parameters
    ----------
    ncomponents : int
        Number of states in the model.
    startprob_prior : array, shape (`ncomponents`,)
        Initial state occupation prior distribution.
    startprob : array, shape (`ncomponents`,)
        Initial state occupation distribution.
    transmat_prior : array, shape (`ncomponents`, `ncomponents`)
        Matrix of prior transition probabilities between states.
    transmat : array, shape (`ncomponents`, `ncomponents`)
        Matrix of transition probabilities between states.
    emission_prior : normal_invwishart
        Initial emission parameters, a normal-inverse Wishart distribution.
    emission : conditional_student_frozen
        The conditional probability distribution used for the emission.
    n_iter : int
        Number of iterations to perform during training, optional.
    thresh : float
        Convergence threshold, optional.
    verbose : bool
        Controls if debug information is printed to the console, optional.

    Attributes
    ----------
    ncomponents : int
        The number of hidden states.
    nfeatures : int
        Dimensionality of the Gaussian emission.
    startprob_prior : array, shape (`ncomponents`,)
        Initial state occupation prior distribution.
    startprob : array, shape (`ncomponents`,)
        Initial state occupation distribution.
    transmat_prior : array, shape (`ncomponents`, `ncomponents`)
        Matrix of prior transition probabilities between states.
    transmat : array, shape (`ncomponents`, `ncomponents`)
        Matrix of transition probabilities between states.
    emission_prior : normal_invwishart
        Initial emission parameters, a normal-inverse Wishart distribution.
    emission : cond_rv_frozen
        The conditional probability distribution used for the emission.

    Examples
    --------
    >>> from mlpy.stats.dbn.hmm import StudentHMM
    >>> StudentHMM(ncomponents=2)
    ...

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    def __init__(self, ncomponents=1, startprob_prior=None, startprob=None, transmat_prior=None,
                 transmat=None, emission_prior=None, emission=None, n_iter=None, thresh=None, verbose=None):
        super(StudentHMM, self).__init__(ncomponents, startprob_prior, startprob, transmat_prior, transmat,
                                         emission_prior, emission, n_iter, thresh, verbose)

    def _generate_sample_from_state(self, state):
        """Generate a sample from the given current state.

        Parameters
        ----------
        state : int
            Current state.

        Returns
        -------
        sample: int
            An observation sampled for the given state

        """
        return multivariate_student.rvs(self.emission.mean[state], self.emission.cov[state], self.emission.df, size=1)

    def _initialize(self, obs, init_count):
        """Perform initialization step before entering the EM algorithm.

        Parameters
        ----------
        obs : array_like, shape (`n`, `ni`, `nfeatures`)
            List of observation sequences, where `n` is the number of sequences, `ni` is
            the length of the i_th observation, and each observation has `nfeatures` features.
        init_count : int
            Restart counter

        """
        df = 10 * np.ones(self.ncomponents)
        fix_df = False
        if self.emission and self.emission.df:
            df = self.emission.df
            fix_df = True

        if self.emission is None or self.startprob is None or self.transmat is None:
            if init_count == 0:
                stacked_obs = self._stack_obs(obs)

                model = StudentMM(ncomponents=self.ncomponents, n_iter=10)
                model.fit(stacked_obs)

                if self.transmat is None or self.startprob is None:
                    pz = model.predict_proba(stacked_obs)
                    self._init_with_mix_model(obs, pz)

                cov = model.cond_proba.cov + np.eye(self.nfeatures)
                mean = model.cond_proba.mean
                if not fix_df:
                    df = model.cond_proba.df

                self.emission = conditional_student(mean, cov, df, self.emission_prior)
            else:
                stacked_obs = self._stack_obs(obs)
                mean = np.zeros((self.ncomponents, self.nfeatures))
                cov = np.zeros((self.ncomponents, self.nfeatures, self.nfeatures))
                for i in range(self.ncomponents):
                    xx = stacked_obs + np.random.randn(stacked_obs.shape[0], stacked_obs.shape[1])
                    mean[i] = np.mean(xx, 0)
                    cov[i, :, :] = np.cov(xx, rowvar=0)

                if self.emission is None:
                    self.emission = conditional_student(mean, cov, df, self.emission_prior)
                else:
                    self.emission.mean = mean
                    self.emission.cov = cov
                self._rand_init()

        if self.emission_prior:
            self.emission.prior = self.emission_prior


class GMMHMM(HMM):
    """
    Hidden Markov Model with Gaussian mixture emissions.

    Representation of a hidden Markov model probability distribution.
    This class allows for easy evaluation of, sampling from, and
    maximum-likelihood estimation of the parameters of a HMM.

    Parameters
    ----------
    ncomponents : int
        Number of states in the model.
    nmix : int
        Number of mixtures.
    startprob_prior : array, shape (`ncomponents`,)
        Initial state occupation prior distribution.
    startprob : array, shape (`ncomponents`,)
        Initial state occupation distribution.
    transmat_prior : array, shape (`ncomponents`, `ncomponents`)
        Matrix of prior transition probabilities between states.
    transmat : array, shape (`ncomponents`, `ncomponents`)
        Matrix of transition probabilities between states.
    emission_prior : normal_invwishart
        Initial emission parameters, a normal-inverse Wishart distribution.
    emission : conditional_mix_normal_frozen
        The conditional probability distribution used for the emission.
    n_iter : int
        Number of iterations to perform during training, optional.
    thresh : float
        Convergence threshold, optional.
    verbose : bool
        Controls if debug information is printed to the console, optional.

    Attributes
    ----------
    ncomponents : int
        The number of hidden states.
    nmix : int
        Number of mixtures.
    nfeatures : int
        Dimensionality of the Gaussian emission.
    startprob_prior : array, shape (`ncomponents`,)
        Initial state occupation prior distribution.
    startprob : array, shape (`ncomponents`,)
        Initial state occupation distribution.
    transmat_prior : array, shape (`ncomponents`, `ncomponents`)
        Matrix of prior transition probabilities between states.
    transmat : array, shape (`ncomponents`, `ncomponents`)
        Matrix of transition probabilities between states.
    emission_prior : normal_invwishart
        Initial emission parameters, a normal-inverse Wishart distribution.
    emission : cond_rv_frozen
        The conditional probability distribution used for the emission.

    Examples
    --------
    >>> from mlpy.stats.dbn.hmm import GMMHMM
    >>> GMMHMM(ncomponents=2)
    ...

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    def __init__(self, ncomponents=1, nmix=1, startprob_prior=None, startprob=None, transmat_prior=None,
                 transmat=None, emission_prior=None, emission=None, n_iter=None, thresh=None, verbose=None):
        super(GMMHMM, self).__init__(ncomponents, startprob_prior, startprob, transmat_prior, transmat,
                                     emission_prior, emission, n_iter, thresh, verbose)
        self.nmix = 1 if nmix is None else nmix

    def _generate_sample_from_state(self, state):
        """Generate a sample from the given current state.

        Parameters
        ----------
        state : int
            Current state.

        Returns
        -------
        sample: int
            An observation sampled for the given state

        Raises
        ------
        NotImplementedError
            This functionality is not implemented yet.

        """
        raise NotImplementedError

    def _initialize(self, obs, init_count):
        """Perform initialization step before entering the EM algorithm.

        Parameters
        ----------
        obs : array_like, shape (`n`, `ni`, `nfeatures`)
            List of observation sequences, where `n` is the number of sequences, `ni` is
            the length of the i_th observation, and each observation has `nfeatures` features.
        init_count : int
            Restart counter

        """
        if self.emission is None or self.startprob is None or self.transmat is None:
            stacked_obs = self._stack_obs(obs)

            if init_count == 0:

                gmm = GMM(n_components=self.ncomponents, covariance_type='full')
                gmm.fit(stacked_obs)

                if self.emission is None:
                    cov = np.eye(self.nfeatures) + gmm.covars_
                    # noinspection PyTypeChecker
                    m = np.tile(gmm.weights_, self.ncomponents)
                    m = normalize(m + np.random.rand(m.shape[0], m.shape[1]), 0)
                    self.emission = conditional_mix_normal(gmm.means_, cov, m, self.emission_prior)
            else:
                stacked_obs = self._stack_obs(obs)
                mean = np.zeros((self.ncomponents, self.nfeatures))
                cov = np.zeros((self.ncomponents, self.nfeatures, self.nfeatures))
                for i in range(self.ncomponents):
                    xx = stacked_obs + np.random.randn(stacked_obs.shape[0], stacked_obs.shape[1])
                    mean[i] = np.mean(xx, 0)
                    cov[i, :, :] = np.cov(xx, rowvar=0)

                m = normalize(np.random.rand(self.nmix, self.ncomponents), 0)

                if self.emission is None:
                    self.emission = conditional_mix_normal(mean, cov, m, self.emission_prior)
                else:
                    self.emission.mean = mean
                    self.emission.cov = cov
                    self.m = m
            self._rand_init()

        if self.emission_prior:
            self.emission.prior = self.emission_prior
