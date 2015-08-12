from __future__ import division, print_function, absolute_import

import os

from ...tools.log import LoggingMgr
from ...planners.explorers.discrete import DiscreteExplorer
from ...mdp.discrete import DiscreteModel
from . import IOnlineLearner

__all__ = ['RLLearner', 'QLearner', 'RLDTLearner']


# noinspection PyAbstractClass
class RLLearner(IOnlineLearner):
    """The reinforcement learning learner interface.

    Parameters
    ----------
    max_steps : int, optional
        The maximum number of steps in an iteration. Default is 100.
    filename : str, optional
        The name of the file to save the learner state to after each iteration.
        If None is given, the learner state is not saved. Default is None.
    profile : bool, optional
        Turn on profiling at which point profiling data is collected
        and saved to a text file. Default is False.

    """
    def __init__(self, max_steps=None, filename=None, profile=False):
        super(RLLearner, self).__init__(filename)
        self._logger = LoggingMgr().get_logger(self._mid)

        self._step_iter = 0

        self._episode_cntr = 1
        self._cum_reward = 0
        self._num_wins = 0

        self._max_steps = max_steps if max_steps is not None else 100
        self._profile = profile

    def __getstate__(self):
        data = super(RLLearner, self).__getstate__()
        data.update(self.__dict__.copy())

        remove_list = ('_id', '_logger')
        for key in remove_list:
            if key in data:
                del data[key]

        return data

    def __setstate__(self, d):
        super(RLLearner, self).__setstate__(d)

        for name, value in d.iteritems():
            setattr(self, name, value)

        self._logger = LoggingMgr().get_logger(self._mid)
        self._logger.debug("Episode=%d", self._episode_cntr)

    def reset(self, t, **kwargs):
        """Reset reinforcement learner.

        Parameters
        ----------
        t : float
            The current time (sec)
        kwargs : dict, optional
            Non-positional parameters, optional.

        """
        super(RLLearner, self).reset(t, **kwargs)

        self._step_iter = 0

    def save(self, filename):
        """Save the learners state.

        If profiling is turned on, profile information is saved to a `txt` file
        with the same name.

        Parameters
        ----------
        filename : str
            The filename to save the information to.

        """
        super(RLLearner, self).save(filename)

        if self._profile:
            filename = os.path.splitext(self._filename)[0]
            with open(filename + ".txt", "a") as f:
                win_ratio = float(self._num_wins) / float(self._episode_cntr)
                f.write("%d, %d, %.2f, %.2f\n" % (self._episode_cntr, self._num_wins, self._cum_reward, win_ratio))

    def learn(self, experience=None):
        """Learn a policy from the experience.

        Parameters
        ----------
        experience : Experience
            The agent's experience consisting of the previous state, the action performed
            in that state, the current state and the reward awarded.

        """
        self._logger.info(experience)

        if self._profile and experience.reward is not None:
            if experience.reward > 0.0:
                self._num_wins += 1
            self._cum_reward += experience.reward
            self._logger.debug("cumReward: %.2f", self._cum_reward)


class QLearner(RLLearner):
    """Performs q-learning.

    Q-learning is a reinforcement learning variant.

    Parameters
    ----------
    explorer : Explorer, optional
        The exploration strategy used. Default is no exploration.
    max_steps : int, optional
        The maximum number of steps in an iteration. Default is 100
    alpha : float, optional
        The learning rate. Default is 0.5.
    gamma : float, optional
        The discounting factor. Default is 0.9.
    filename : str, optional
        The name of the file to save the learner state to after each iteration.
        If None is given, the learner state is not saved. Default is None.
    profile : bool, optional
        Turn on profiling at which point profiling data is collected
        and saved to a text file. Default is False.

    """
    def __init__(self, explorer=None, max_steps=None, alpha=None, gamma=None, filename=None, profile=False):
        super(QLearner, self).__init__(max_steps, filename, profile)

        self._model = DiscreteModel()
        self._explorer = explorer if explorer is not None else DiscreteExplorer()
        """:type: Explorer"""

        self._alpha = alpha if alpha is not None else 0.5
        self._gamma = gamma if gamma is not None else 0.9

    def execute(self, experience):
        """Execute learning specific updates.

        Learning specific updates are performed, e.g. model updates.

        Parameters
        ----------
        experience : Experience
            The actor's current experience consisting of previous state, the action
            performed in that state, the current state, and the reward awarded.

        """
        self._model.update(experience)

    def learn(self, experience=None):
        """ Learn a policy from the experience.

        By updating the Q table according to the experience a policy is learned.

        Parameters
        ----------
        experience : Experience
            The actor's current experience consisting of previous state, the action
            performed in that state, the current state, and the reward awarded.

        """
        super(QLearner, self).learn(experience)

        info = self._model.statespace[experience.state]
        info2 = self._model.statespace[experience.next_state]

        qvalue = info.q[experience.action]
        maxq = max([info2.q[a] for a in self._model.get_actions(experience.next_state)])

        delta = experience.reward + self._gamma * maxq - qvalue
        info.q[experience.action] = qvalue + self._alpha * delta

        self._logger.debug("%s action=%s reward=%.2f %s d=%.2f", experience.state, experience.action, experience.reward,
                           experience.next_state, delta)
        self._logger.debug("\tq_old=%.2f visits=%d", qvalue, info.models[experience.action].visits)
        self._logger.debug("\tq_new=%.2f", info.q[experience.action])

    def choose_action(self, state):
        """Choose the next action

        The next action is chosen according to the current policy and the
        selected exploration strategy.

        Parameters
        ----------
        state : State
            The current state.

        Returns
        -------
        Action :
            The chosen action.

        """
        self._model.add_state(state)

        action = None
        if self._step_iter < self._max_steps:
            actions = self._model.get_actions(state)
            info = self._model.statespace[state]

            action = self._explorer.choose_action(actions, [info.q[a] for a in actions])
            self._logger.debug("state=%s act=%s value=%.2f", state, action, self._model.statespace[state].q[action])

        return action


class RLDTLearner(RLLearner):
    """Performs reinforcement learning using decision trees.

    Reinforcement learning using decision trees (RL-DT) use decision trees
    to build the transition and reward models as described by Todd Hester and
    Peter Stone [1]_.

    Parameters
    ----------
    planner : IPlanner
        The planner to use to determine the best action.
    max_steps : int, optional
        The maximum number of steps in an iteration. Default is 100.
    filename : str, optional
        The name of the file to save the learner state to after each iteration.
        If None is given, the learner state is not saved. Default is None.
    profile : bool, optional
        Turn on profiling at which point profiling data is collected
        and saved to a text file. Default is False.

    References
    ----------
    .. [1] Hester, Todd, and Peter Stone. "Generalized model learning for reinforcement
        learning in factored domains." Proceedings of The 8th International Conference on
        Autonomous Agents and Multiagent Systems-Volume 2. International Foundation for Autonomous
        Agents and Multiagent Systems, 2009.

    """
    def __init__(self, planner, max_steps=None, filename=None, profile=False):
        super(RLDTLearner, self).__init__(max_steps, filename, profile)

        self._do_plan = True
        self._planner = planner

    def __getstate__(self):
        data = super(RLDTLearner, self).__getstate__()
        data.update({'_planner': self._planner})
        return data

    def __setstate__(self, d):
        super(RLDTLearner, self).__setstate__(d)

        for name, value in d.iteritems():
            setattr(self, name, value)

        self._do_plan = False

    def execute(self, experience):
        """Execute learning specific updates.

        Learning specific updates are performed, e.g. model updates.

        Parameters
        ----------
        experience : Experience
            The actor's current experience consisting of previous state, the action
            performed in that state, the current state, and the reward awarded.

        """
        self._do_plan = self._planner.model.update(experience)

    def learn(self, experience=None):
        """Learn a policy from the experience.

        A policy is learned from the experience by building the MDP model.

        Parameters
        ----------
        experience : Experience
            The actor's current experience consisting of previous state, the action
            performed in that state, the current state, and the reward awarded.

        """
        super(RLDTLearner, self).learn(experience)

        if self._do_plan:
            self._planner.plan()

    def choose_action(self, state):
        """Choose the next action

        The next action is chosen according to the current policy and the
        selected exploration strategy.

        Parameters
        ----------
        state : State
            The current state.

        Returns
        -------
        Action :
            The chosen action.

        """
        action = None

        if self._step_iter < self._max_steps:
            action = self._planner.get_next_action(state)
            self._step_iter += 1

        return action
