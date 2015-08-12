from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range
import random

import numpy as np

from ...stats import gibbs
from . import IExplorer

__all__ = ['DiscreteExplorer', 'EGreedyExplorer', 'SoftmaxExplorer']


class DiscreteExplorer(IExplorer):
    """The discrete explorer base class.

    The explorer class executes the exploration policy by choosing
    a next action based on the current qvalues of the state-action pairs.

    Notes
    -----
    All discrete explorers should derive from this class.

    """
    def __init__(self):
        super(DiscreteExplorer, self).__init__()

    def choose_action(self, actions, qvalues):
        """Choose the next action according to the exploration strategy.

        Parameters
        ----------
        actions : list[Actions]
            The available actions.
        qvalues : list[float]
            The q-value for each action.

        Returns
        -------
        Action :
            The action with maximum q-value that can be taken
            from the given state.

        """
        return self._get_maxq_action(actions, qvalues)

    # noinspection PyMethodMayBeStatic
    def _get_maxq_action(self, actions, qvalues):
        """Find the highest valued action available for the given state.

        Parameters
        ----------
        actions : list[Actions]
            The available actions.
        qvalues : list[float]
            The q-value for each action.

        Returns
        -------
        Action :
            The action with maximum qvalue that can be taken from
            the given state.

        """
        maxq = max(qvalues)
        count = qvalues.count(maxq)
        if count > 1:
            action = actions[np.random.choice([i for i in range(len(actions)) if qvalues[i] == maxq])]
        else:
            index = qvalues.index(maxq)
            action = actions[index]
        return action


class EGreedyExplorer(DiscreteExplorer):
    """The :math:`\\epsilon`-greedy explorer.

    The :math:`\\epsilon`-greedy explorer policy chooses as next action
    the action with the highest q-value, however with
    :math:`\\epsilon`-probability a random action is chosen to
    drive exploration of unknown states.

    Parameters
    ----------
    epsilon : float, optional
        The :math:`\\epsilon` probability. Default is 0.5.
    decay : float, optional
        The value by which :math:`\\epsilon` decays. This value should be
        between 0 and 1. The probability :math:`\\epsilon` to decreases
        over time with a factor of `decay`. Set this value to 1 if
        :math:`\\epsilon` should remain the same throughout the experiment.
        Default is 1.

    """
    def __init__(self, epsilon=None, decay=None):
        super(EGreedyExplorer, self).__init__()

        self._epsilon = epsilon if epsilon is not None else 0.5
        self._decay = decay if decay is not None else 1
        self._decay = max(0, self._decay)
        self._decay = min(self._decay, 1)

    def choose_action(self, actions, qvalues):
        """Choose the next action.

        With :math:`\\epsilon` probability, a random action is
        chosen, otherwise the action resulting in the highest
        q-value is selected.

        Parameters
        ----------
        actions : list[Actions]
            The available actions.
        qvalues : list[float]
            The q-value for each action.

        Returns
        -------
        Action :
            The action with maximum qvalue that can be taken from
            the given state.

        """
        action = self._get_maxq_action(actions, qvalues)

        if self._is_active and np.random.random() < self._epsilon:
            self._epsilon *= self._decay

            action = random.choice(actions)

        return action


class SoftmaxExplorer(DiscreteExplorer):
    """The softmax explorer.

    The softmax explorer varies the action probability as a
    graded function of estimated value. The greedy action is
    still given the highest selection probability, but all the others
    are ranked and weighted according to their value estimates.

    Parameters
    ----------
    tau : float, optional
        The temperature value. Default is 2.0.
    decay : float, optional
        The value by which :math:`\\tau` decays. This value should
        be between 0 and 1. The temperature :math:`\\tau` to decrease
        over time with a factor of `decay`. Set this value to 1 if
        :math:`\\tau` should remain the same throughout the experiment.
        Default is 1.

    Notes
    -----
    The softmax function implemented uses the Gibbs distribution. It
    chooses action `a` on the `t`-th play with probability:

    .. math::

        \\frac{e^{Q_t(a)/\\tau}}{\\sum_{b=1}^ne^{Q_t(b)/\\tau}}

    where :math:`\\tau` is a positive parameter called the `temperature`.
    High temperatures cause all actions to be equiprobable. Low temperatures
    cause a greater difference in the selection probability. For :math:`\\tau`
    close to zero, the action selection because the same as greedy.

    """
    def __init__(self, tau=None, decay=None):
        super(SoftmaxExplorer, self).__init__()

        self._tau = tau if tau is not None else 2.0
        self._decay = decay if decay is not None else 1
        self._decay = max(0, self._decay)
        self._decay = min(self._decay, 1)

    def choose_action(self, actions, qvalues):
        """Choose the next action.

        Choose the next action according to the Gibbs
        distribution.

        Parameters
        ----------
        actions : list[Actions]
            The available actions.
        qvalues : list[float]
            The q-value for each action.

        Returns
        -------
        Action :
            The action with maximum q-value that can be taken
            from the given state.

        """
        action = self._get_maxq_action(actions, qvalues)

        if self._is_active:
            self._tau *= self._decay

            pmf = gibbs.pmf(np.asarray(qvalues), self._tau)
            action = actions[np.random.choice(np.arange(len(np.asarray(qvalues))), p=pmf)]

        return action
