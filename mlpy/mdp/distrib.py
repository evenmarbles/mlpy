from __future__ import division, print_function, absolute_import

import numpy as np

from abc import abstractmethod
from ..modules.patterns import RegistryInterface


class ProbaCalcMethodFactory(object):
    """The probability calculation method factory.

    An instance of a probability calculation method can be created by passing
    the probability calculation method type.

    Examples
    --------
    >>> from mlpy.mdp.distrib import ProbaCalcMethodFactory
    >>> ProbaCalcMethodFactory.create('defaultprobacalcmethod')

    This creates a :class:`.DefaultProbaCalcMethod` instance.

    """
    @staticmethod
    def create(_type, *args, **kwargs):
        """Create a probability calculation method of the given type.

        Parameters
        ----------
        _type : str
            The probability calculation method type. The method type should
            be equal to the class name of the method.
        args : tuple
            Positional arguments passed to the class of the given type for
            initialization.
        kwargs : dict
            Non-positional arguments passed to the class of the given type
            for initialization.

        Returns
        -------
        IProbaCalcMethod :
            A probability calculation method instance of the given type.

        """
        # noinspection PyUnresolvedReferences
        return IProbaCalcMethod.registry[_type.lower()](*args, **kwargs)


class IProbaCalcMethod(object):
    """The Probability calculation method interface.

    The probability calculation method is responsible for calculating the
    probability distribution based on the state transitions seen so far.

    Notes
    -----
    To create custom probability calculation methods, derive
    from this class.

    """
    __metaclass__ = RegistryInterface

    @abstractmethod
    def execute(self, states):
        """Execute the calculation.

        Parameters
        ----------
        states : dict[State, dict[str, int | float]]
            The list of next states to consider.

        Returns
        -------
        dict[State, dict[str, int | float]] :
            The updated states information including the probabilities.

        Raises
        ------
        NotImplementedError :
            If the child class does not implement this function.

        """
        raise NotImplementedError


class DefaultProbaCalcMethod(IProbaCalcMethod):
    """The default probability calculation method.

    The default probability calculation method determines
    the probability distribution by normalizing the state count
    over all state.

    """
    def execute(self, states):
        """Execute the calculation.

        Calculate the probability distribution based on
        the number of times the states have been seen so far.

        Parameters
        ----------
        states : dict[State, dict[str, int | float]]
            The list of next states to consider.

        Returns
        -------
        dict[State, dict[str, int | float]] :
            The updated states information including the probabilities.

        """
        count = np.array([v['count'] for v in states.values()])
        proba = np.true_divide(count, np.sum(count))
        return {k: {'count': c, 'proba': p} for k, c, p in
                zip(states.keys(), count, proba)}


class ProbabilityDistribution(object):
    """Probability Distribution.

    This class handles evaluation of empirically derived states and
    calculates the probability distribution from them.

    Parameters
    ----------
    proba_calc_method : str
        The method used to calculate the probability distribution for
        the initial state. Defaults to 'defaultprobacalcmethod'.

    """
    __slots__ = ('_states', '_dirty', '_proba_calc_method')

    def __init__(self, proba_calc_method=None):
        self._states = {}
        """:type : dict[State,dict[str,int|float]]"""
        self._dirty = False
        """:type: bool"""

        proba_calc_method = proba_calc_method if proba_calc_method is not None else 'defaultprobacalcmethod'
        try:
            self._proba_calc_method = ProbaCalcMethodFactory.create(proba_calc_method)
        except:
            raise ValueError("%s is not a valid probability calculation method" % proba_calc_method)

    def __getstate__(self):
        return {
            "_states": self._states,
            "_proba_calc_method": {
                "module": self._proba_calc_method.__class__.__module__,
                "name": self._proba_calc_method.__class__.__name__
            }
        }

    def __setstate__(self, d):
        for name, value in d.iteritems():
            if name == "_proba_calc_method":
                module = __import__(value["module"])
                try:
                    value = getattr(module, value["name"])()
                except:
                    path = value["module"].split(".")
                    mod = "module"
                    for i, ele in enumerate(path):
                        if i != 0:
                            mod += '.'
                            mod += ele
                    value = getattr(eval(mod), value["name"])()

            setattr(self, name, value)

        self._dirty = False

    def __len__(self):
        return len(self._states)

    def __iter__(self):
        if self._dirty:
            self._states = self._proba_calc_method.execute(self._states)
        return {k: p["proba"] for k, p in self._states.iteritems()}.iteritems()

    def __getitem__(self, state):
        try:
            return self._states[state]["proba"]
        except KeyError:
            return None

    def __setitem__(self, state, proba):
        dirty = self._dirty
        self.add_state(state)
        self._dirty = dirty

        self._states[state] = proba

    def add_state(self, state):
        """Adds a state to the states list.

        Adds a state to the states list in order to build the probability distribution.

        Parameters
        ----------
        state : State
            An initial state.

        """
        if state not in self._states:
            self._states[state] = {
                "count": 0,
                "proba": 0.0
            }
        self._states[state]["count"] += 1
        self._dirty = True

    def iadd(self, state, proba):
        """In-place addition of the probability to the states probability.

        If the state does not exist in the list of states, it will
        be added.

        Parameters
        ----------
        state : State
            The state for which the probability is updated.
        proba : float
            The probability value to add to the state's probability.

        """
        dirty = self._dirty
        self.add_state(state)
        self._dirty = dirty

        self._states[state]["proba"] += proba

    def get(self):
        """Retrieve the probability distribution.

        Returns
        -------
        dict[State, float] :
            A list of probabilities for all possible transitions.
        """
        if self._dirty:
            self._states = self._proba_calc_method.execute(self._states)
        return {k: p["proba"] for k, p in self._states.iteritems()}

    def clear(self):
        """Clear the probability distribution."""
        self._states.clear()
        self._dirty = True

    def sample(self):
        """Returns a next state according to the probability distribution.

        Returns
        -------
        State :
            The next state sampled from the probability distribution.
        """
        keys = self._states.keys()

        if not keys:
            raise UserWarning("No initial states defined.")

        if self._dirty:
            self._states = self._proba_calc_method.execute(self._states)
            self._dirty = False

        idx = np.random.choice(range(len(keys)), p=[v['proba'] for v in self._states.values()])
        return keys[idx]
