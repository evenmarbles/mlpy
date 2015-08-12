"""
.. module:: mlpy.optimize.algorithms
   :platform: Unix, Windows
   :synopsis: Optimization algorithms.

.. moduleauthor:: Astrid Jackson <ajackson@eecs.ucf.edu>
"""
from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

from abc import ABCMeta, abstractmethod
import numpy as np
from .utils import is_converged


class EM(object):
    """Expectation-Maximization module base class.

    Representation of the expectation-maximization (EM) model.
    This class allows for the execution of the expectation- maximization
    algorithm by providing functionality for random restarts and convergence
    checking.

    See the instance documentation for details specific to a particular
    implementation of the EM algorithm.

    Parameters
    ----------
    n_iter : int, optional
        The number of iterations to perform. Default is 100.
    thresh : float, optional
        The convergence threshold. Default is 1e-4.
    verbose : bool, optional
        Controls if debug information is printed to the console.
        Default is False.

    See Also
    --------
    :class:`.HMM`, :class:`.GMM`

    Notes
    -----
    Classes that deriving from the EM base class must overwrite the following
    private functions:

        _initialize(obs, init_count)
            Perform initialization before entering the EM algorithm. The expected parameters are:

                obs : array_like, shape (`n`, `ni`, `nfeatures`)
                    List of observation sequences, where `n` is the number of sequences, `ni` is
                    the length of the i_th observation, and each observation has `nfeatures` features.
                init_count : int
                    Restart counter.

        _estep(obs)
            Perform the expectation step of the EM algorithm and return the log likelihood of the
            observation `obs`. The expected parameters are:

                obs : array_like, shape (`n`, `ni`, `nfeatures`)
                    List of observation sequences, where `n` is the number of sequences, `ni` is
                    the length of the i_th observation, and each observation has `nfeatures` features.

        _mstep()
            Perform maximization step of the EM algorithm.

    Optionally, the private function :meth:`_plot` can be overwritten to visualize
    the results at each iteration. The :meth:`_plot` function is called by the EM
    algorithm before the maximization step is performed.

    The deriving class must call the private method ``_em(x, n_init=None)``
    to initiate the the EM algorithm. Pass the following parameters:

        x : array_like, shape (`n`, `ni`, `ndim`)
                List of data sequences, where `n` is the number of sequences, `ni` is
                the length of the i_th sequence, and each data point in the sequence
                has `ndim` dimensions.
        n_init : int, optional
            Number of restarts to prevent getting stuck in a local minimum. Default is 1.

    The function returns the log likelihood of the data sequences `x`.


    Examples
    --------
    >>> from mlpy.optimize.algorithms import EM
    >>>
    >>> class MyEM(EM):
    ...     def _initialize(self, obs, init_count):
    ...         pass
    ...
    ...     def _estep(self, obs):
    ...         pass
    ...
    ...     def _mstep(self):
    ...         pass
    ...
    ...     def _plot(self):
    ...         pass
    ...
    ...     def fit(self, x):
    ...         return self._em(x, n_init=5)
    ...

    This creates a new class capable of performing the expectation-maximization
    algorithm.


    .. note::
        Adapted from:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_
    """
    __metaclass__ = ABCMeta

    def __init__(self, n_iter=None, thresh=None, verbose=None):
        self._n_iter = 100 if n_iter is None else n_iter
        self._thresh = 1e-4 if thresh is None else thresh
        self._verbose = False if verbose is None else verbose

    @abstractmethod
    def _initialize(self, obs, init_count):
        """Perform an initialization step before entering the EM algorithm.

        Parameters
        ----------
        obs : array_like, shape (`n`, `ni`, `nfeatures`)
            List of observation sequences, where `n` is the number of sequences, `ni` is
            the length of the i_th observation, and each observation has `nfeatures` features.
        init_count : int
            Restart counter.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.


        """
        raise NotImplementedError

    @abstractmethod
    def _estep(self, obs):
        """Perform the expectation step of the EM algorithm.

        Parameters
        ----------
        obs : array_like, shape (`n`, `ni`, `nfeatures`)
            List of observation sequences, where `n` is the number of sequences, `ni` is
            the length of the i_th observation, and each observation has `nfeatures` features.

        Returns
        -------
        sufficient_stats:
            Expected sufficient statistics according to emission.
        loglik:
            Log likelihood of the observation `obs`.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.


        """
        raise NotImplementedError

    @abstractmethod
    def _mstep(self):
        """Perform maximization step of the EM algorithm.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.


        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def _plot(self):
        """Plot the data.

        Notes
        -----
        Overwrite this function to plot data during the EM step.
        """
        pass

    def _em(self, x, n_init=None, init_count=None):
        """Perform the expectation-maximization algorithm.

        Parameters
        ----------
        x : array_like, shape (`n`, `ni`, `ndim`)
                List of data sequences, where `n` is the number of sequences, `ni` is
                the length of the i_th sequence, and each data point in the sequence
                has `ndim` dimensions.
        n_init : int, optional
            Number of restarts to prevent getting stuck in a local minimum. Default is 1.
        init_count : int, optional
            The number of random restarts already performed. Default is 0.

        Returns
        -------
        float :
            Log likelihood of the data sequences `x`.

        """
        n_init = 1 if n_init is None else n_init
        if n_init < 1:
            raise ValueError('{0} estimation requires at least one run'.format(self.__class__.__name__))
        init_count = 0 if init_count is None else init_count

        if n_init > 1:
            models = []
            ll_hists = []
            best_ll = np.zeros(n_init)
            init_model = self.__dict__.copy()
            for i in range(n_init):
                if self._verbose:
                    print("\n********** Random Restart {0} **********".format(i+1))
                ll_hists.append(self._em(x, init_count=i))
                models.append(self.__dict__.copy())
                self.__dict__ = init_model
                best_ll[i] = ll_hists[i][-1]
            best_ndx = best_ll.argmax()
            self.__dict__ = models[best_ndx]
            return ll_hists[best_ndx]

        if self._verbose:
            print("initializing model for EM")
        self._initialize(x, init_count)
        ndx = 0
        done = False
        loglik_hist = np.zeros(self._n_iter)

        while not done:
            # stats, loglik_hist[ndx] = self._estep(x)
            loglik_hist[ndx] = self._estep(x)
            if self._verbose:
                print("\t{0} loglik: {1}".format(ndx+1, loglik_hist[ndx]))

            self._plot()

            self._mstep()

            if ndx >= self._n_iter - 1:
                break
            elif ndx > 0:
                done = is_converged(loglik_hist[ndx], loglik_hist[ndx - 1], thresh=self._thresh)

            ndx += 1

        return loglik_hist[0:ndx]
