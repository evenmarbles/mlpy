"""
========================================
Planning tools (:mod:`mlpy.planners`)
========================================


.. automodule:: mlpy.planners.explorers
   :noindex:


.. currentmodule:: mlpy.planners

Planners
========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   IPlanner


Discrete planners
-----------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~discrete.ValueIteration

"""
from __future__ import division, print_function, absolute_import

from abc import ABCMeta, abstractmethod
from ..modules import UniqueModule
from ..mdp.stateaction import Action

__all__ = ['explorers', 'discrete']


class IPlanner(UniqueModule):
    """The planner interface class.

    Parameters
    ----------
    explorer : Explorer
        The exploration strategy to employ. Available explorers are:

        :class:`.EGreedyExplorer`
            With :math:`\\epsilon` probability, a random action is
            chosen, otherwise the action resulting in the highest
            q-value is selected.

        :class:`.SoftmaxExplorer`
            The softmax explorer varies the action probability as a
            graded function of estimated value. The greedy action is
            still given the highest selection probability, but all the others
            are ranked and weighted according to their value estimates.

    """
    __metaclass__ = ABCMeta

    def __init__(self, explorer=None):
        """
        Initialization of the planner class.
        """
        super(IPlanner, self).__init__()

        self._history = {}
        """:type : dict[State,list[str]]"""
        self._current = -1

        self._explorer = explorer
        """:type: Explorer"""

    def __getstate__(self):
        return {'_history': self._history}

    def __setstate__(self, d):
        super(IPlanner, self).__setstate__(d)

        setattr(self, '_history', d['_history'])
        self._current = len(self._history[next(iter(self._history))]) - 1 if len(self._history) > 0 else -1

    def activate_exploration(self):
        """Turn the explorer on. """
        if self._explorer is not None:
            self._explorer.activate()

    def deactivate_exploration(self):
        """ Turn the explorer off. """
        if self._explorer is not None:
            self._explorer.deactivate()

    @abstractmethod
    def get_best_action(self, state):
        """ Choose the best next action for the agent to take.

        Parameters
        ----------
        state : State
            The state for which to choose the action for.

        Returns
        -------
        Action :
            The best action.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    @abstractmethod
    def plan(self):
        """ Plan for the optimal policy.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    def get_next_action(self, state, use_policy=False):
        """ Returns the optimal action for a state according to the current policy.

        Parameters
        ----------
        state : State
            The state for which to choose the next action for.
        use_policy : bool, optional
            When using a policy the next action is chosen according to the
            current policy, otherwise the best action is selected. Default
            is False.

        Returns
        -------
        Action :
            The next action.

        """
        if not use_policy:
            action = self.get_best_action(state)
        else:
            if not self._history:
                self.create_policy()

            action = self._history[state][self._current]
        return action

    def create_policy(self, func=None):
        """ Creates a policy (i.e., a state-action association).

        Parameters
        ----------
        func : callable, optional
            A callback function for mixing policies.

        """
        policy = self._create_policy(func)

        states = set(self._history).union(policy)

        # noinspection PyUnresolvedReferences
        n = len(self._history.itervalues().next()) if self._history else 0
        self._history = dict((s, (self._history.get(s, []) if self._history.get(s) is not None else [
                              Action.get_noop_action()] * n) + policy.get(s, [])) for s in states)
        self._current += 1

    def visualize(self):
        """ Visualize of the planning data.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    def _create_policy(self, func=None):
        raise NotImplementedError
