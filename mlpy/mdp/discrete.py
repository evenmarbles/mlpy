from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import sys
import copy

import numpy as np

from ..tools.log import LoggingMgr
from ..modules import UniqueModule
from ..modules.patterns import RegistryInterface
from ..tools.misc import Waiting
from ..libs import classifier
from .stateaction import StateData, State, Action, RewardFunction
from . import IMDPModel

__all__ = ['ExplorerFactory', 'RMaxExplorer', 'LeastVisitedBonusExplorer', 'UnknownBonusExplorer',
           'DiscreteModel', 'DecisionTreeModel']


class ExplorerFactory(object):
    """The model explorer factory.

    An instance of an explorer can be created by passing the
    explorer type.

    Examples
    --------
    >>> from mlpy.mdp.discrete import ExplorerFactory
    >>> ExplorerFactory.create('unknownbonusexplorer', 1.0)

    This creates a :class:`.UnknownBonusExplorer` with `rmax`
    set to 1.0.

    """
    @staticmethod
    def create(_type, *args, **kwargs):
        """Create an MDP model of the given type.

        Parameters
        ----------
        _type : str
            The model explorer type. Valid model types:

                leastvisitedbonusexplorer:
                    In least-visited-bonus exploration mode, the states
                    that have been visited the least are given a bonus of RMax.
                    A :class:`.LeastVisitedBonusExplorer` instance is create.

                unknownbonusexplorer:
                    In unknown-bonus exploration mode states for which
                    the decision tree was unable to predict a reward are considered
                    unknown and are given a bonus of RMax. A :class:`.UnknownBonusExplorer`
                    instance is create.

        args : tuple, optional
            Positional arguments to pass to the class of the given type for
            initialization.
        kwargs : dict, optional
            Non-positional arguments to pass to the class of the given type
            for initialization.

        Returns
        -------
        RMaxExplorer :
            An explorer instance of the given type.

        """
        # noinspection PyUnresolvedReferences
        return RMaxExplorer.registry[_type.lower()](*args, **kwargs)


class RMaxExplorer(UniqueModule):
    """RMax based exploration base class.

    Parameters
    ----------
    rmax : float
        The maximum achievable reward.

    """
    __metaclass__ = RegistryInterface

    def __init__(self, rmax):
        super(RMaxExplorer, self).__init__()

        self._rmax = rmax if rmax is not None else 0.0
        RewardFunction.activate_bonus = True
        RewardFunction.rmax = self._rmax

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, d):
        for name, value in d.iteritems():
            setattr(self, name, value)

        RewardFunction.activate_bonus = True
        RewardFunction.rmax = self._rmax

    def activate(self, *args, **kwargs):
        """Turn on exploration mode."""
        RewardFunction.activate_bonus = True

    # noinspection PyMethodMayBeStatic
    def deactivate(self):
        """Turn off exploration mode."""
        RewardFunction.activate_bonus = False

    def update(self, model):
        """Update the reward model according to a RMax based exploration policy.

        Parameters
        ----------
        model : StateActionInfo
            The state-action information.

        """
        pass


class LeastVisitedBonusExplorer(RMaxExplorer):
    """Least visited bonus explorer, a RMax based exploration model.

    Least visited bonus exploration only goes into exploration mode
    whether it is predicted that only states with rewards less than
    a given threshold can be reached. Once in exploration mode, states
    that have been visited least are given a bonus of RMax to drive
    exploration.

    Parameters
    ----------
    rmax : float
        The maximum achievable reward.
    func : callable
        Callback function to retrieve the minimum number of times a
        state has been visited.
    thresh : float
        If all states that can be reached from the current state have
        a value less than the threshold, exploration mode is turned on.

    """
    def __init__(self, rmax, func, thresh=None):
        super(LeastVisitedBonusExplorer, self).__init__(rmax)

        self._is_active = True

        self._min_visits = sys.maxint
        self._thresh = thresh if thresh is not None else self._rmax * 0.4
        self._func = func

    def __getstate__(self):
        data = super(LeastVisitedBonusExplorer, self).__getstate__()
        del data['_func']
        return data

    def activate(self, qvalues=None, *args, **kwargs):
        """Turn on exploration mode.

        If it is predicted that only states with rewards less than the threshold
        can be reached then the agent goes into exploration mode.

        Parameters
        ----------
        qvalues : dict
            The qvalues for all actions from the current state

        """
        if qvalues is None:
            RewardFunction.activate_bonus = True

        self._is_active = True
        for qvalue in qvalues.itervalues():
            if qvalue > self._thresh:
                self._is_active = False
                break

        RewardFunction.activate_bonus = self._is_active

        self._min_visits = self._func()

    def update(self, model):
        """Update the reward model.

        Update the reward model according to a RMax based exploration policy.
        To drive exploration a bonus of RMax is given to the least visited states.

        Parameters
        ----------
        model : StateActionInfo
            The states-action information.

        """
        model.reward_func.bonus = 0.0
        if model.visits <= self._min_visits:
            model.reward_func.bonus = self._rmax


class UnknownBonusExplorer(RMaxExplorer):
    """Unknown bonus explorer, a RMax based exploration model.

    States for which the decision tree was unable to predict a reward
    are given a bonus of RMax to drive exploration, since these states
    are considered to be unknown under the model.

    Parameters
    ----------
    rmax : float
        The maximum achievable reward.

    """

    def __init__(self, rmax):
        super(UnknownBonusExplorer, self).__init__(rmax)

    def update(self, model):
        """Update the reward model.

        Update the reward model according to a RMax based exploration policy.
        States for which the decision tree was unable to predict a reward
        are considered unknown. These states are given a bonus of RMax to drive
        exploration.

        Parameters
        ----------
        model : StateActionInfo
            The states-action information.

        """
        model.reward_func.bonus = 0.0
        if not model.known:
            model.reward_func.bonus = self._rmax


class DiscreteModel(IMDPModel):
    """The MDP model for discrete states and actions.

    Parameters
    ----------
    actions : list[Action] or dict[State, list[Action]
        The available actions. If not given, the actions are read
        from the Action description.

    """
    @property
    def statespace(self):
        """Collection of states and their state-action information.

        Returns
        -------
        dict[State, StateData] :
            The state space.
        """
        return self._statespace

    def __init__(self, actions=None, **kwargs):
        super(DiscreteModel, self).__init__(**kwargs)

        #: The number of states in the model.
        self._nstates = 0
        self._statespace = {}
        """:type: dict[State, StateData]"""

        self._actions = self._set_actions(actions)
        """:type: dict[State, list[Action]] | list[Action]"""

    def __getstate__(self):
        data = super(DiscreteModel, self).__getstate__()
        data.update({
            '_statespace': self._statespace,
            'actions': [a.name for a in self._actions]
        })
        return data

    def __setstate__(self, d):
        super(DiscreteModel, self).__setstate__(d)

        for name, value in d.iteritems():
            if name == 'actions':
                name = '_actions'
                value = self._set_actions(value)
            setattr(self, name, value)

        self._nstates = len(self._statespace)

    def get_actions(self, state=None):
        """Retrieve the available actions for the given state.

        Parameters
        ----------
        state : State
            The state for which to get the actions.

        Returns
        -------
        list :
            The actions that can be taken in this state.

        """
        if isinstance(self._actions, dict):
            return self._actions[state]
        return self._actions

    def add_state(self, state):
        """Add a new state to the statespace.

        Add a new state to the statespace (a collection of states
        that have already been seen).

        Parameters
        ----------
        state : State
            The state to add to the state space.

        Returns
        -------
        bool :
            Whether the state was a new state or not.

        """
        if state is not None and state not in self._statespace:
            self._nstates += 1
            self._statespace[state] = StateData(self._nstates, self.get_actions(state))
            return True

        return False

    def fit(self, obs, actions, labels=None):
        """Fit the model to the observations and actions of the trajectory.

        Parameters
        ----------
        obs : array_like, shape (`nfeatures`, `n`)
            Trajectory of observations, where each observation has `nfeatures` features
            and `n` is the length of the trajectory.
        actions : array_like, shape (`nfeatures`, `n`)
            Trajectory of actions, where each action has `nfeatures` features and `n` is
            the length of the trajectory.
        labels : array_like, shape (`n`,)
            Label identifying each step in the trajectory, where `n` is the length of the
            trajectory.

        """
        n = obs.shape[1]
        for i in range(n - 1):
            state = State(obs[:, i], labels[i])
            self.add_state(state)

            if i == 0:
                self._initial_dist.add_state(State(obs[:, i]))

            action = Action(actions[:, i])

            next_state = State(obs[:, i + 1], labels[i + 1])
            self.add_state(next_state)

            proba = self._statespace[state].models[action].transition_proba
            proba.add_state(next_state)

            if next_state.is_terminal():
                for a in self.get_actions(next_state):
                    proba = self._statespace[next_state].models[a].transition_proba
                    proba.add_state(next_state)

    def update(self, experience=None):
        """Update the model with the agent's experience.

        Parameters
        ----------
        experience : Experience
            The agent's experience, consisting of state, action, next state(, and reward).

        Returns
        -------
        bool :
            Return True if the model has changed, False otherwise.

        """
        if experience is None:
            return False

        if experience.state is None:
            self._initial_dist.add_state(experience.next_state)
            return False

        self.add_state(experience.state)
        self.add_state(experience.next_state)

        model = self._statespace[experience.state].models[experience.action]
        model.transition_proba.add_state(experience.next_state)

        if experience.reward is not None:
            model.reward_func.set(experience.reward)

        model.visits += 1
        return True

    def predict_proba(self, state, action):
        """Predict the probability distribution.

        Predict the probability distribution for state transitions
        given a state and an action.

        Parameters
        ----------
        state : State
            The current state the robot is in.
        action : Action
            The action perform in state `state`.

        Returns
        -------
        dict[tuple[float]], float] :
            The probability distribution for the state-action pair.

        """
        return self._statespace[state].models[action].transition_proba.get()

    # noinspection PyShadowingNames
    def print_transitions(self):
        """Print the state transitions for debugging purposes."""
        if self._logger.level > LoggingMgr.LOG_DEBUG:
            return

        sorted_states = [None] * len(self.statespace)
        for state, info in self._statespace.iteritems():
            sorted_states[info.id - 1] = state.encode()
        decorated = [(i, tup[0], tup) for i, tup in enumerate(sorted_states)]
        decorated.sort(key=lambda tup: tup[1])
        sorted_states = [tup for i, second, tup in decorated]

        self._logger.debug("============================== Transition probabilities ==============================")
        for state_rep in sorted_states:
            # noinspection PyTypeChecker
            state = State.decode(state_rep)
            info = self._statespace[state]
            self._logger.debug("state={0}".format(state_rep))
            for act, model in info.models.iteritems():
                self._logger.debug("   act={0}\ttransitions=".format(act))
                for key, prob in model.transition_proba:
                    self._logger.debug("      {0} : {1}".format(key, prob))

    # noinspection PyShadowingNames
    def print_rewards(self):
        """Print the state rewards for debugging purposes."""
        if self._logger.level > LoggingMgr.LOG_DEBUG:
            return

        sorted_states = [None] * len(self.statespace)
        for state, info in self._statespace.iteritems():
            sorted_states[info.id - 1] = state.encode()
        decorated = [(i, tup[0], tup) for i, tup in enumerate(sorted_states)]
        decorated.sort(key=lambda tup: tup[1])
        sorted_states = [tup for i, second, tup in decorated]

        self._logger.debug("============================== Rewards ==============================")
        for state_rep in sorted_states:
            # noinspection PyTypeChecker
            state = State.decode(state_rep)
            info = self._statespace[state]
            self._logger.debug("state={0}".format(state_rep))

            for act, model in info.models.iteritems():
                self._logger.debug("   act={0}\treward={1}".format(act, model.reward_func.get(state)))

    # noinspection PyMethodMayBeStatic
    def _set_actions(self, actions):
        if not isinstance(Action.description, dict):
            raise ValueError('Action.description not set')

        act_names = actions if actions is not None else Action.description.keys()

        if act_names is not None:
            if isinstance(act_names, dict):
                actions = {}
                for state, action_names in act_names.iteritems():
                    s = State(eval(state))
                    actions[s] = []
                    for a in action_names:
                        # noinspection PyTypeChecker
                        actions[s].append(Action(Action.description[a]["value"], a))
            else:
                actions = []
                for a in act_names:
                    actions.append(Action(Action.description[a]["value"], a))
        else:
            raise ValueError('Actions required')

        return actions


class ClassPair(object):
    @property
    def in_(self):
        return self._in

    @property
    def out(self):
        return self._out

    def __init__(self, in_, out):
        self._in = in_
        self._out = out


class DecisionTreeModel(DiscreteModel):
    """The MDP model for discrete states and actions realized with decision trees.

    The MDP model with decision trees is implemented as described by Todd Hester and
    Peter Stone [1]_. Transitions are learned for each feature; i.e. there is a decision
    tree for each state feature, and the predictions :math:`P(x_i^r|s,a)` for the ``n``
    state features are combined to create a prediction of probabilities of the relative
    change of the state :math:`s^r=\\langle x_1^r, x_2^r, \\ldots, x_n^r \\rangle` by
    calculating:

    .. math::

        P(s^r|s, a) = \\Pi_{i=0}^n P(x_i^r|s,a)

    Optionally, the reward can also be learned by generating a decision tree for it.

    The MDP model with decision trees can optionally specify an RMax based
    exploration model to drive exploration of unseen states.

    Parameters
    ----------
    actions : list[Action] | dict[State, list[Action]
        The available actions. If not given, the actions are read from the
        Action description.
    explorer_type : str
        The type of exploration policy to perform. Valid explorer types:

        unvisitedbonusexplorer:
            In unvisited-bonus exploration mode, if a state is
            experienced that has not been seen before the decision
            trees are considered to have changed and thus are being updated,
            otherwise, the decision trees are only considered to have changed
            based on the C45Tree algorithm.

        leastvisitedbonusexplorer:
            In least-visited-bonus exploration mode, the states
            that have been visited the least are given a bonus of RMax.
            A :class:`.LeastVisitedBonusExplorer` instance is create.

        unknownbonusexplorer:
            In unknown-bonus exploration mode states for which
            the decision tree was unable to predict a reward are considered
            unknown and are given a bonus of RMax. A :class:`.UnknownBonusExplorer`
            instance is create.

    use_reward_trees : bool
        If True, decision trees are used for the rewards model, otherwise a
        standard reward function is used.
    args: tuple
        Positional parameters passed to the model explorer.
    kwargs: dict
        Non-positional parameters passed to the model explorer.

    Other Parameters
    ----------------
    explorer_params : dict
        Parameters specific to the given exploration type.

    Raises
    ------
    ValueError
        If explorer type is not valid.

    Notes
    -----
    A C4.5 algorithm is used to generate the decision trees. The implementation of
    the algorithm that was improved to make the algorithm incremental. This is realized
    by checking at each node whether the new experience changes the optimal split and
    only rebuilds the the tree from that node if it does.

    References
    ----------
    .. [1] Hester, Todd, and Peter Stone. "Generalized model learning for reinforcement
        learning in factored domains." Proceedings of The 8th International Conference on
        Autonomous Agents and Multiagent Systems-Volume 2. International Foundation for Autonomous
        Agents and Multiagent Systems, 2009.

    """
    DEBUG_TREE = False

    def __init__(self, actions=None, explorer_type=None, use_reward_trees=None, *args, **kwargs):
        super(DecisionTreeModel, self).__init__(actions)

        if explorer_type is not None and explorer_type not in ['unvisitedbonusexplorer',
                                                               'leastvisitedbonusexplorer',
                                                               'unknownbonusexplorer']:
            raise ValueError("%s is not a valid exploration model" % explorer_type)

        try:
            if explorer_type == "leastvisitedbonusexplorer":
                kwargs.update({"func": self._get_min_visits})
            # noinspection PyTypeChecker
            self._explorer = ExplorerFactory.create(explorer_type, *args, **kwargs)
            """:type: RMaxExplorer"""
        except:
            self._explorer = None

        #: If True, the decision trees are considered to have changed and
        #: thus are being updated if a state is experienced that has not
        #: been seen before, otherwise, the decision trees are only considered
        #: to be changed based on the C45Tree algorithm.
        self._unvisited_bonus = True if explorer_type == "unvisitedbonusexplorer" else False
        """:type: bool"""

        #: If True, decision trees are used for the rewards model,
        #: otherwise a standard reward function is used.
        self._use_reward_trees = use_reward_trees if use_reward_trees is not None else True
        """:type: bool"""

        self._rng = classifier.Random(2)

        #: The raw transition data used to fit the decision trees.
        self._fit_transition = []
        """:type: list[list[ClassPair]]"""

        #: The raw reward data used to fit the decision trees.
        self._fit_reward = None
        """:type: list[ClassPair]"""

        #: The decision trees for predicting the transition model.
        #: Each state feature is handled by one decision tree.
        self._output_models = []
        """:type: list[C45Tree]"""

        #: The decision tree predicting the reward value.
        self._reward_model = None
        """:type: C45Tree"""

    def __getstate__(self):
        data = super(DiscreteModel, self).__getstate__()
        data.update(self.__dict__.copy())

        remove_list = ('_output_models', '_reward_model', '_rng', '_id', '_logger')
        for key in remove_list:
            if key in data:
                del data[key]

        return data

    def __setstate__(self, d):
        super(DiscreteModel, self).__setstate__(d)

        for name, value in d.iteritems():
            setattr(self, name, value)

        if self._explorer is not None and self._explorer.__class__.__name__ == 'LeastVisitedBonusExplorer':
            setattr(self._explorer, '_func', self._get_min_visits)

        self._rng = classifier.Random(2)

        self._output_models = []
        """:type: list[C45Tree]"""
        self._reward_model = None
        """:type: C45Tree"""

        waiting = Waiting("Training models")
        waiting.start()

        transition_data = []
        reward_data = None

        for i, tree in enumerate(self._fit_transition):
            transition_data.append(classifier.ClassPairList())
            for tp in tree:
                transition_data[i].append(classifier.ClassPair(tp.in_, tp.out))

        if self._fit_reward is not None:
            reward_data = classifier.ClassPairList()
            for tp in self._fit_reward:
                reward_data.append(classifier.ClassPair(tp.in_, tp.out))

        self._train(transition_data, reward_data)
        waiting.stop()

        self._update_sainfo()

    def activate_exploration(self):
        """Turn the explorer on."""
        if self._explorer is not None:
            self._explorer.activate()

    def deactivate_exploration(self):
        """Turn the explorer off."""
        if self._explorer is not None:
            self._explorer.deactivate()

    def fit(self, obs, actions, rewards=None):
        """Fit the model to the trajectory data.

        Parameters
        ----------
        obs : array_like, shape (`nfeatures`, `n`)
            Trajectory of observations, where each observation has `nfeatures` features
            and `n` is the length of the trajectory.
        actions : array_like, shape (`nfeatures`, `n`)
            Trajectory of actions, where each action has `nfeatures` features and `n` is
            the length of the trajectory.
        rewards : array_like, shape (`n`,)
            List of rewards, a reward is awarded for each observation.

        """
        waiting = Waiting("Preparing to train models")
        waiting.start()

        transition_data = []
        reward_data = None

        n = obs.shape[1]

        for i in range(n - 1):
            state = State(obs[:, i])
            self.add_state(state)

            action = Action(actions[:, i])

            next_state = State(obs[:, i + 1])
            self.add_state(next_state)

            if i == 0:
                self._initial_dist.add_state(State(obs[:, i]))

                if self._use_reward_trees and rewards is not None:
                    reward_data = classifier.ClassPairList()

            cp = classifier.ClassPair(self._prepare_tree_input(state, action))

            for j in range(State.nfeatures):
                if i == 0:
                    transition_data.append(classifier.ClassPairList())
                cp.out = float(next_state[j] - state[j])
                transition_data[j].append(cp)
                if len(self._fit_transition) - 1 < j:
                    self._fit_transition.append([])
                self._fit_transition[j].append(ClassPair(cp.in_, cp.out))

            if reward_data is not None:
                cp.out = float(rewards[i])
                reward_data.append(cp)
                self._fit_reward.append(ClassPair(cp.in_, cp.out))

        self._train(transition_data, reward_data)
        waiting.stop()

        self._update_sainfo()

    def update(self, experience=None):
        """Update the model with the agent's experience.

        The decision trees for transition and reward functions are being updated.

        Parameters
        ----------
        experience : Experience
            The agent's experience, consisting of state, action, next state(, and reward).

        Returns
        -------
        bool :
            Return True if the model has changed, False otherwise.

        """
        if experience is None:
            return False

        if experience.state is None:
            self._initial_dist.add_state(experience.next_state)
            return False

        changed = self.add_state(experience.state) and self._unvisited_bonus
        self.add_state(experience.next_state)

        info = self._statespace[experience.state]
        info.models[experience.action].visits += 1

        if experience.next_state.is_terminal():
            info = self._statespace[experience.next_state]
            info.models[experience.action].visits += 1

        waiting = Waiting("Training models")
        waiting.start()

        self._init_mdp()

        cp = classifier.ClassPair(self._prepare_tree_input(experience.state, experience.action))

        for i, tree in enumerate(self._output_models):
            cp.out = float(experience.next_state[i] - experience.state[i])
            self._fit_transition[i].append(ClassPair(cp.in_, cp.out))
            changed = tree.train_instance(cp) or changed

        if self._use_reward_trees:
            cp.out = float(experience.reward)
            changed = self._reward_model.train_instance(cp) or changed
            self._fit_reward.append(ClassPair(cp.in_, cp.out))

        if self._explorer is not None:
            self._explorer.activate(info.q)

        waiting.stop()

        if changed:
            self._update_sainfo()

        return changed

    def _init_mdp(self):
        """Initializes the decision trees for the MDP model."""
        if len(self._output_models) == 0:
            for i in range(State.nfeatures):
                self._output_models.append(classifier.C45Tree(i, 1, 5, 0, 0, self._rng))
                if self.DEBUG_TREE:
                    # noinspection PyPep8Naming
                    self._output_models[i].DTDEBUG = True

                if len(self._fit_transition) < State.nfeatures:
                    self._fit_transition.append([])

        if len(self._output_models) != State.nfeatures:
            self._logger.error(
                "Error size mismatch between input vector and # trees {0}, {1}".format(len(self._output_models),
                                                                                       State.nfeatures))
            return False

        if self._use_reward_trees and self._reward_model is None:
            self._reward_model = classifier.C45Tree(State.nfeatures, 1, 5, 0, 0, self._rng)

            if self._fit_reward is None:
                self._fit_reward = []

    def _train(self, transition_data, reward_data):
        """Train the models (decision trees) with the transition and reward data.

        Parameters
        ----------
        transition_data : list[classifier.ClassPairList[classifier.ClassPair]]
            The input data for the decision tree to predict the transition model.
        reward_data : classifier.ClassPairList[classifier.ClassPair]
            The input data for the decision tree to predict the reward model.

        """
        self._init_mdp()

        for i, sd in enumerate(transition_data):
            self._output_models[i].train_instances(sd)

        if reward_data is not None:
            self._reward_model.train_instances(reward_data)

    def _update_sainfo(self):
        """Build the transition and reward models.

        Build the transition and reward models based on predictions returned
        by the decision trees.

        """
        waiting = Waiting("Combining results")
        waiting.start()

        for state in self._statespace.keys():
            info = self._statespace[state]
            for act, model in info.models.iteritems():
                if len(self._output_models) == 0:
                    model.transition_proba[state] = 1.0
                    model.known = False
                else:
                    model.known = True
                    model.transition_proba.clear()

                    predictions = []
                    tree_input = self._prepare_tree_input(state, act)
                    for i, omodel in enumerate(self._output_models):
                        predictions.append(omodel.test_instance(tree_input))

                    self._combine_results(0, [0.0] * State.nfeatures, State(np.asarray([0] * State.nfeatures)),
                                          tree_input, predictions, model, state)

                    if self._use_reward_trees:
                        tree_input = self._prepare_tree_input(state, act)

                        reward_preds = self._reward_model.test_instance(tree_input)

                        if len(reward_preds) == 0:
                            model.known = False
                        else:
                            reward_sum = 0.0
                            num_visits = 0.0
                            for val, prob in reward_preds.iteritems():
                                num_visits = num_visits + prob
                                reward_sum = reward_sum + (val * prob)
                            model.reward_func.set(reward_sum / num_visits)

                if self._explorer is not None:
                    self._explorer.update(model)

        waiting.stop()

    def _prepare_tree_input(self, state, action):
        """Prepares the decision tree input.

        Parameters
        ----------
        state : State
            The state.
        action : Action
            The action.
\
        Returns
        -------
        ndarray[float] :
            The decision tree input vector.

        """
        tree_input = state.get()
        for act in self.get_actions(state):
            if action == act:
                tree_input = np.append(tree_input, [1.0])
            else:
                tree_input = np.append(tree_input, [0.0])
        return tree_input

    def _combine_results(self, index, cum_probs, t_next, tree_input, predictions, model, state):
        """Combine the feature predictions.

        Combine the feature predictions to compute the state transition based on the combined
        value of all state features.

        Parameters
        ----------
        index: int
            The current index into the predictions.
        cum_probs: list[float]
            The cumulative probability.
        t_next: State
            The next state.
        tree_input: list[float]
            The original tree input.
        predictions: list[FloatMap]
            The predictions from the tree.
        model: StateActionInfo
            The model.
        state: State
            The current state.
        """
        for feature_val, prob in predictions[index].iteritems():
            # ignore if it has probability 0
            if prob == 0.0:
                continue

            t_next[index] = int(feature_val + tree_input[index])
            cum_probs[index] = prob if index == 0 else cum_probs[index - 1] * prob

            # if this is the last feature, remember it in transition probabilities
            if index == (State.nfeatures - 1) and cum_probs[index] > 0.0:
                n = copy.deepcopy(t_next)
                if n not in self._statespace:
                    if n.is_valid():
                        self._logger.debug("Unknown state {0} in transitioning model".format(n))
                        self.add_state(n)
                    else:
                        n = copy.deepcopy(state)

                model.transition_proba.iadd(n, cum_probs[index])
                continue

            self._combine_results(index + 1, cum_probs, t_next, tree_input, predictions, model, state)

    def _get_min_visits(self):
        """Calculates the number of visits of the least visited state.

        Returns
        -------
        int :
            The number of visits of the least visited state.

        """
        min_visits = sys.maxint
        for state, info in self._statespace.iteritems():
            for model in info.models.values():
                if model.visits < min_visits:
                    min_visits = model.visits

        self._logger.debug("min visits={0}".format(min_visits))
        return min_visits
