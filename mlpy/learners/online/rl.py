from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.stats import bernoulli

from ...planners.explorers.discrete import DiscreteExplorer
from ...planners.explorers.discrete import IExplorer
from ...mdp.discrete import DiscreteModel
from ...mdp.stateaction import MDPState, MDPAction
from ...stats.models.ann.neuralnet import NeuralNetwork
from . import IOnlineLearner

__all__ = ['QLearner', 'ModelBasedLearner']


class QLearner(IOnlineLearner):
    """Performs q-learning.

    Q-learning is a reinforcement learning variant.

    Parameters
    ----------
    explorer : Explorer, optional
        The exploration strategy used. Default is no exploration.
    alpha : float, optional
        The learning rate. Default is 0.5.
    gamma : float, optional
        The discounting factor. Default is 0.9.
    filename : str, optional
        The name of the file to save the learner state to after each iteration.
        If None is given, the learner state is not saved. Default is None.

    """
    @property
    def type(self):
        return super(QLearner, self).type

    def __init__(self, explorer=None, alpha=None, gamma=None, filename=None):
        super(QLearner, self).__init__(filename)

        self._model = DiscreteModel()
        self._explorer = explorer if explorer is not None else DiscreteExplorer()
        """:type: Explorer"""
        if not isinstance(self._explorer, IExplorer):
            raise TypeError("'explorer' must be of type 'IExplorer'")

        self._alpha = alpha if alpha is not None else 0.5
        self._gamma = gamma if gamma is not None else 0.9

    def init(self):
        """Initialize the learner."""
        self._model.init()

    def step(self, experience):
        """Execute learning specific updates.

        Learning specific updates are performed, e.g. model updates.

        Parameters
        ----------
        experience : Experience
            The actor's current experience consisting of previous state, the action
            performed in that state, the current state, and the reward awarded.

        """
        self._model.update(experience)

    def learn(self, experience):
        """ Learn a policy from the experience.

        By updating the Q table according to the experience a policy is learned.

        Parameters
        ----------
        experience : Experience
            The actor's current experience consisting of previous state, the action
            performed in that state, the current state, and the reward awarded.

        """
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
        state : MDPState
            The current state.

        Returns
        -------
        MDPAction :
            The chosen action.

        """
        self._model.add_state(state)

        actions = self._model.get_actions(state)
        info = self._model.statespace[state]

        action = self._explorer.choose_action(actions, [info.q[a] for a in actions])
        self._logger.debug("state=%s act=%s value=%.2f", state, action, self._model.statespace[state].q[action])


class ModelBasedLearner(IOnlineLearner):
    """Performs model based reinforcement learning.

    Model based reinforcement learning uses the model and planner provided
    to make decisions on which action to perform next.

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

    """
    @property
    def type(self):
        return super(ModelBasedLearner, self).type

    def __init__(self, planner, filename=None):
        super(ModelBasedLearner, self).__init__(filename)

        self._do_plan = True
        self._planner = planner

    def __setstate__(self, d):
        super(ModelBasedLearner, self).__setstate__(d)
        self._do_plan = False

    def init(self):
        """Initialize the learner."""
        self._planner.init()

    def step(self, experience):
        """Execute learning specific updates.

        Learning specific updates are performed, e.g. model updates.

        Parameters
        ----------
        experience : Experience
            The actor's current experience consisting of previous state, the action
            performed in that state, the current state, and the reward awarded.

        """
        self._do_plan = self._planner.model.update(experience)

    def learn(self, experience):
        """Learn a policy from the experience.

        A policy is learned from the experience by building the MDP model.

        Parameters
        ----------
        experience : Experience
            The actor's current experience consisting of previous state, the action
            performed in that state, the current state, and the reward awarded.

        """
        if self._do_plan:
            self._planner.plan()

    def choose_action(self, state):
        """Choose the next action

        The next action is chosen according to the current policy and the
        selected exploration strategy.

        Parameters
        ----------
        state : MDPState
            The current state.

        Returns
        -------
        MDPAction :
            The chosen action.

        """
        return self._planner.choose_action(state)


class Cacla(IOnlineLearner):
    @property
    def type(self):
        return super(Cacla, self).type

    def __init__(self, nhidden_q, nhidden_v, explorer_type=None, gamma=None, alpha=None, beta=None, explore_rate=None,
                 filename=None):
        super(Cacla, self).__init__(filename)

        self._g1 = 0.
        self._g2 = 0.
        self._stored_gauss = False

        self._action = None
        self._value = None

        self._v_target = None

        self._nhidden_q = nhidden_q
        self._nhidden_v = nhidden_v

        self._gamma = gamma if gamma is not None else .99
        self._alpha = alpha if alpha is not None else .01
        self._beta = beta if beta is not None else .01

        self._explorer_type = explorer_type if explorer_type is not None else "gaussian"
        self._explore_rate = explore_rate if explore_rate is not None else 5000

    def init(self):
        """Initialize the learner."""
        if MDPState.discretized:
            num_states = 1
            for states_per_dim in zip(MDPState.states_per_dim):
                num_states *= states_per_dim

            self._action = np.zeros((num_states, MDPAction.nfeatures))
            self._value = np.zeros((num_states,))
        else:
            if self._nhidden_q == 0:
                self._action = np.random.random((MDPState.nfeatures + 1, MDPAction.nfeatures))
            else:
                self._action = NeuralNetwork([MDPState.nfeatures, self._nhidden_q, MDPAction.nfeatures])
            self._value = NeuralNetwork([MDPState.nfeatures, self._nhidden_v, 1])

            self._v_target = np.zeros((1,))

        self._stored_gauss = False

    def step(self, experience):
        """Execute learning specific updates.

        Learning specific updates are performed, e.g. model updates.

        Parameters
        ----------
        experience : Experience
            The actor's current experience consisting of previous state, the action
            performed in that state, the current state, and the reward awarded.

        """
        if not MDPState.discretized and self._nhidden_q > 0:
            self._action.feed_forward(experience.next_state)

    def end(self, experience):
        """End the episode.

        Perform all end of episode tasks and save the state of the
        learner to file.

        Parameters
        ----------
        experience : Experience
            The agent's experience consisting of the previous state, the action performed
            in that state, the current state and the reward awarded.

        """
        if MDPState.discretized:
            vt = self._value[experience.state][0]

            self._value[experience.state] += self._alpha * (experience.reward - self._value[experience.state])

            if self._value[experience.state] > vt:
                self._action[experience.state] += self._beta * (
                    experience.action - self._action[experience.state])
        else:
            self._v_target[0] = experience.reward
            vt = self._value.feed_forward(experience.state)[0]
            self._value.back_propagate(experience.state, self._v_target, self._alpha)

            if self._v_target[0] > vt:
                if self._nhidden_q == 0:
                    st = np.ones((experience.state.get().shape[0] + 1,))
                    st[:-1] = experience.state

                    self._action += self._beta * np.outer(st, (experience.action - self._get_action(experience.state)))
                else:
                    self._action.back_propagate(experience.state, experience.action, self._beta)

    def learn(self, experience):
        """Learn a policy from the experience.

        Perform the learning step to derive a new policy taking the
        latest experience into account.

        Parameters
        ----------
        experience : Experience
            The agent's experience consisting of the previous state, the action performed
            in that state, the current state and the reward awarded.

        """
        if MDPState.discretized:
            vt = self._value[experience.state][0]

            self._value[experience.state] += self._alpha * (
                experience.reward + self._gamma * self._value[experience.next_state] - self._value[experience.state])

            if self._value[experience.state] > vt:
                self._action[experience.state] += self._beta * (
                    experience.action - self._action[experience.state])
        else:
            vs = self._value.feed_forward(experience.next_state)[0]
            self._v_target[0] = experience.reward + self._gamma * vs

            vt = self._value.feed_forward(experience.state)[0]
            self._value.back_propagate(experience.state, self._v_target, self._alpha)

            if self._v_target[0] > vt:
                if self._nhidden_q == 0:
                    st = np.ones((experience.state.get().shape[0] + 1,))
                    st[:-1] = experience.state

                    self._action += self._beta * np.outer(st, (experience.action - self._get_action(experience.state)))
                else:
                    self._action.back_propagate(experience.state, experience.action, self._beta)

    def choose_action(self, state):
        """Choose the next action

        The next action is chosen according to the current policy and the
        selected exploration strategy.

        Parameters
        ----------
        state : MDPState
            The current state.

        Returns
        -------
        MDPAction :
            The chosen action.

        """
        if MDPState.discretized:
            action = self._action[state]
        else:
            if self._nhidden_q > 0:
                action = self._action.get_activations(-1)
            else:
                action = self._get_action(state)

        if self._explorer_type == "egreedy":
            if bernoulli.rvs(self._explore_rate):
                for i, (min_, max_) in enumerate(zip(MDPAction.min_features, MDPAction.max_features)):
                    action[i] = np.random.uniform(min_, max_)
        if self._explorer_type == "gaussian":
            for i in range(MDPAction.nfeatures):
                action[i] += self._explore_rate * self._gaussian_random()

        return MDPAction(action)

    def _get_action(self, state):
        st = np.ones((state.get().shape[0] + 1,))
        st[:-1] = state

        action = np.asarray(np.dot(self._action.T, st))
        if action.ndim == 0:
            action = action[np.newaxis]

        return action

    def _gaussian_random(self):
        if self._stored_gauss:
            self._stored_gauss = False
            return self._g2

        x = y = 0.
        z = 1.
        while z >= 1.0:
            # x = np.random.uniform(-1., 1.)
            # y = np.random.uniform(-1., 1.)
            x = 2.0 * np.random.random() - 1.0
            y = 2.0 * np.random.random() - 1.0
            z = x * x + y * y

        z = np.sqrt(-2. * np.log(z) / z)

        self._g1 = x * z
        self._g2 = y * z

        self._stored_gauss = True
        return self._g1
