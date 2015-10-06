"""
.. currentmodule:: mlpy.planners.explorers

Explorers
=========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ExplorerFactory
   IExplorer

Discrete explorers
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~discrete.DiscreteExplorer
   ~discrete.EGreedyExplorer
   ~discrete.SoftmaxExplorer

"""
from __future__ import division, print_function, absolute_import

from ...modules.patterns import RegistryInterface


class ExplorerFactory(object):
    """The explorer factory.

    An instance of an explorer can be created by passing the explorer type.

    Examples
    --------
    >>> from mlpy.planners.explorers import ExplorerFactory
    >>> ExplorerFactory.create('egreedyexplorer', 0.8)

    This creates a :class:.EGreedyExplorer` instance with epsilon set
    to 0.8.

    >>> ExplorerFactory.create('softmaxexplorer', tau=3.0, decay=0.4)

    This creates a :class:`.SoftmaxExplorer` instance with tau set to
    3.0 and decay set to 0.4

    """
    @staticmethod
    def create(_type, *args, **kwargs):
        """
        Create an explorer of the given type .

        Parameters
        ----------
        _type : str
            The explorer type. Valid explorer types:

            egreedyexplorer
                With :math:`\\epsilon` probability, a random action is
                chosen, otherwise the action resulting in the highest
                q-value is selected. An :class:`.EGreedyExplorer` is created.

            softmaxexplorer
                The softmax explorer varies the action probability as a
                graded function of estimated value. The greedy action is
                still given the highest selection probability, but all the others
                are ranked and weighted according to their value estimates.
                A :class:`.SoftmaxExplorer` is created.

        args : tuple
            Positional arguments passed to the class of the given type for
            initialization.
        kwargs : dict
            Non-positional arguments passed to the class of the given type
            for initialization.

        Returns
        -------
        IExplorer :
            An explorer instance of the given type.

        """
        # noinspection PyUnresolvedReferences
        return IExplorer.registry[_type.lower()](*args, **kwargs)


class IExplorer(object):
    """The explorer interface class.

    The explorer class executes the exploration policy.

    Notes
    -----
    All explorers should derive from this class.

    """
    __metaclass__ = RegistryInterface

    def __init__(self):
        self._is_active = True

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, d):
        for name, value in d.iteritems():
            setattr(self, name, value)

    def activate(self):
        """Turn on exploration mode."""
        self._is_active = True

    def deactivate(self):
        """Turn off exploration mode."""
        self._is_active = False

    def choose_action(self, *args, **kwargs):
        """Choose the next action according to the exploration strategy.

        Parameters
        ----------
        args : tuple
            Positional arguments.
        kwargs : dict
            Non-positional arguments.

        Returns
        -------
        MDPAction :
            The next action to taken.

        Raises
        ------
        NotImplementedError:
            If the child class does not implement this function.

        """
        return NotImplementedError


from .discrete import *

__all__ = [s for s in dir() if not (s.startswith('_') or s.endswith('cython'))]
