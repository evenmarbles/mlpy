from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

from scipy.stats import rv_discrete
import numpy as np

__all__ = ['nonuniform', 'gibbs']


# noinspection PyMethodOverriding,PyPep8Naming
class nonuniform_gen(rv_discrete):
    """A nonuniform discrete random variable.

    %(before_notes)s

    %(example)s

    .. note::
        Adapted from Matlab:

        | Project: `Probabilistic Modeling Toolkit for Matlab/Octave <https://github.com/probml/pmtk3>`_.
        | Copyright (2010) Kevin Murphy and Matt Dunham
        | License: `MIT <https://github.com/probml/pmtk3/blob/5fefd068a2e84ae508684d3e4750bd72a4164ba0/license.txt>`_

    """
    def _argcheck(self, a):
        self.a = a
        return abs(a.sum() - 1) <= np.finfo(np.float16).eps

    def _pmf(self, x, a):
        # port discreteLogprob
        raise NotImplementedError

    def _ppf(self, q, a):
        raise NotImplementedError

    def _stats(self, a):
        raise NotImplementedError

    # noinspection PyArgumentList
    def _rvs(self, a):
        r = np.random.rand()
        if self._size is not None:
            r = np.random.rand(self._size)
        s = np.zeros(self._size, dtype=np.int32)

        cum_prob = np.cumsum(a.ravel())

        if self._size is None:
            cum_prob2 = cum_prob[0:-1]
            s = np.sum(r > cum_prob2)
        else:
            n = a.size
            if n < self._size:
                for i in range(n - 1):
                    s += r > cum_prob[i]
            else:
                cum_prob2 = cum_prob[0:-1]
                for i in range(self._size):
                    # noinspection PyTypeChecker
                    s[i] = np.sum(r[i] > cum_prob2)

        return s

nonuniform = nonuniform_gen(name='nonuniform', longname='A discrete non-uniform '
                            '(random integer)')


# noinspection PyMethodOverriding,PyPep8Naming
class gibbs_gen(rv_discrete):
    """A Gibbs distributed discrete random variable.

    %(before_notes)s

    Notes
    -----
    The probability mass function for `gibbs` is::

                        exp(a/t)
        gibbs.pmf(x) = ------------
                       sum(exp(a/t)

    %(example)s

    """
    def _argcheck(self, t):
        """Default check for correct values on args and keywords.

        Returns condition array of 1's where arguments are correct
        and 0's where they are not.

        """
        return t >= 0

    def _nonzero(self, k, t):
        return k == k

    def _pmf(self, a, t):
        values = np.exp(a / t)

        # noinspection PyTypeChecker
        if np.any(t <= np.finfo(np.float16).eps):
            max_value = max(a)
            values = np.asarray([val if val == max_value else 0. for val in a])
        return values / np.sum(values)

    def _ppf(self, a, t):
        raise NotImplementedError

    def _stats(self, t):
        raise NotImplementedError

gibbs = gibbs_gen(name='gibbs', longname='Gibbs distribution '
                  '(random integer)')
