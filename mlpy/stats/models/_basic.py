from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
from scipy.misc import doccer

from ...stats import nonuniform
from ...auxiliary.array import normalize, nunique, accum

__all__ = ['markov']


_doc_default_callparams = """\
startprob : array_like
    Start probabilities.
transmat : array_like
    Transition matrix.
"""

_doc_frozen_callparams = ""

_doc_frozen_callparams_note = \
    """See class definition for a detailed description of parameters."""

docdict_params = {
    '_doc_default_callparams': _doc_default_callparams,
}

docdict_noparams = {
    '_doc_default_callparams': _doc_frozen_callparams,
}


# noinspection PyPep8Naming
class markov_gen(object):
    """Markov model.

    The `startprob` keyword specifies the start probabilities for the model.
    The `transmat` keyword specifies the transition probabilities the model
    follows.

    Methods
    -------
    score(x, startprob, transmat)
        Log probability of the given data `x`.
    sample(x, startprob, transmat, size=1)
        Draw random samples from a Markov model.
    fit(x)
        Fits a Markov model from data via MLE or MAP.

    Parameters
    ----------
    %(_doc_default_callparams)s


    Alternatively, the object may be called (as a function) to fix the degrees
    of freedom and scale parameters, returning a "frozen" Markov model:

    rv = normal_invwishart(startprob=None, transmat=None)
        - Frozen object with the same methods but holding the given
          start probabilities and transitions fixed.

    Examples
    --------
    >>> from mlpy.stats.models import markov

    >>> startprob = np.array([0.1, 0.4, 0.5])
    >>> transmat = np.array([[0.3, 0.2, 0.5], [0.6, 0.3, 0.1], [0.1, 0.5, 0.4]])

    >>> m = markov(startprob, transmat)
    >>> m.sample(size=2)
    [[2 2]]

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """

    def __init__(self):
        super(markov_gen, self).__init__()
        self.__doc__ = doccer.docformat(self.__doc__, docdict_params)

    def __call__(self, startprob, transmat):
        markov_frozen(startprob, transmat)

    def score(self, x, startprob, transmat):
        """Log probability for a given data `x`.

        Attributes
        ----------
        x : ndarray
            Data to evaluate.
        %(_doc_default_callparams)s

        Returns
        -------
        log_prob : float
            The log probability of the data.

        """
        log_transmat = np.log(transmat + np.finfo(float).eps)
        log_startprob = np.log(startprob + np.finfo(float).eps)
        log_prior = log_startprob[x[:, 0]]

        n = x.shape[0]
        nstates = log_startprob.shape[0]

        logp = np.zeros(n)
        for i in range(n):
            njk = accum(np.vstack([x[i, 0:-1], x[i, 1::]]).T, 1, size=(nstates, nstates), dtype=np.int32)
            logp[i] = np.sum(njk * log_transmat)
        return logp + log_prior

    def sample(self, startprob, transmat, size=1):
        """Sample from a Markov model.

        Attributes
        ----------
        size: int
            Defining number of sampled variates. Defaults to `1`.

        Returns
        -------
        vals: ndarray
            The sampled sequences of size (nseq, seqlen).

        """
        if np.isscalar(size):
            size = (1, size)

        vals = np.zeros(size, dtype=np.int32)

        nseq, seqlen = size
        for i in range(nseq):
            vals[i][0] = nonuniform.rvs(startprob)
            for t in range(1, seqlen):
                vals[i][t] = nonuniform.rvs(transmat[vals[i][t - 1]])
        return vals

    def fit(self, x):
        """Fit a Markov model from data via MLE or MAP.

        Attributes
        ----------
        x : ndarray[int]
            Observed data

        Returns
        -------
        %(_doc_default_callparams)s

        """
        # TODO: allow to pass pseudo_counts as parameter?
        nstates = nunique(x.ravel())
        pi_pseudo_counts = np.ones(nstates)
        transmat_pseudo_counts = np.ones((nstates, nstates))

        n = x.shape[0]

        startprob = normalize(np.bincount(x[:, 0])) + pi_pseudo_counts - 1
        counts = np.zeros((nstates, nstates))
        for i in range(n):
            counts += accum(np.vstack([x[i, 0:-1], x[i, 1::]]).T, 1, size=(nstates, nstates))
        transmat = normalize(counts + transmat_pseudo_counts - 1, 1)
        return startprob, transmat

markov = markov_gen()


# noinspection PyPep8Naming
class markov_frozen(object):

    def __init__(self, startprob, transmat):
        """Create a "frozen" Markov model.

        Parameters
        ----------
        startprob : array_like
            Start probabilities
        transmat : array_like
            Transition matrix

        """
        self._model = markov_gen()

        self.startprob = startprob
        self.transmat = transmat

    def score(self, x):
        return self._model.score(x, self.startprob, self.transmat)

    def sample(self, size=1):
        return self._model.sample(self.startprob, self.transmat, size)
