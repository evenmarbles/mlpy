from __future__ import division, print_function, absolute_import

import time
import copy
import numpy as np

from rlglued.agent.agent import Agent
from rlglued.utils.taskspecvrlglue3 import TaskSpecParser
from rlglued.types import Action

from .modules import AgentModuleFactory
from ..modules import UniqueModule
from ..auxiliary.datasets import DataSet
from ..mdp.stateaction import Experience, MDPState, MDPAction

__all__ = ['Bot', 'BotWrapper', 'ModelBasedAgent']


class Bot(UniqueModule):
    """The bot base class.

    """

    def __init__(self, mid):
        super(Bot, self).__init__(mid)
        self._is_ready = False

    def check_is_ready(self):
        is_ready = self._is_ready
        self._is_ready = False
        return is_ready

    def set_is_ready(self, value):
        self._is_ready = value

    def init(self):
        pass

    def enter(self, t):
        raise NotImplementedError

    def update(self, dt, action):
        raise NotImplementedError

    def exit(self):
        raise NotImplementedError


class BotWrapper(object):
    """The bot wrapper class

    The bot wrapper is useful for experiments containing bots whose actions
    take multiple time steps to complete.

    """

    def __init__(self, bot):
        self._bot = bot
        self._t = 0.0

    def init(self):
        """Initialize the bot."""
        self._bot.init()

    def start(self):
        """Start the bot.

        This moves the bots in its initial position.
        """
        self._t = time.time()

        self._bot.enter(self._t)
        while not self._bot.check_is_ready():
            self._bot.update(self._delta_time())

    def next_behavior(self, action):
        """Perform next behavior.

        The next behavior is based on the given action.

        Parameters
        ----------
        action : MDPAction
            The action on which the behavior is based on.

        """
        self._bot.update(self._delta_time(), action)
        while not self._bot.check_is_ready():
            self._bot.update(self._delta_time())

    def exit(self):
        """Perform cleanup tasks."""
        self._bot.exit()

    def _delta_time(self):
        dt = time.time() - self._t
        self._t += dt
        return dt


class ModelBasedAgent(UniqueModule, Agent):
    """The agent base class.

    Agents act in an environment (:class:`.Environment`) usually performing a task
    (:class:`.Task`). Depending on the agent module they either follow a policy
    (:class:`.FollowPolicyModule`), are user controlled via the keyboard or a PS2
    controller (:class:`.UserModule`), or learn according to a learner
    (:class:`.LearningModule`). New agent modules can be created by inheriting from the
    :class:`.IAgentModule` class.

    Parameters
    ----------
    module_type : str
        The agent module type, which is the name of the class.
    record: bool, optional
        Flag indicating whether the experiences are being recorded or not.
        Default is False.
    dataset_args: tuple, optional
        Positional parameters for the :class:`.DataSet` class, which records the
        experiences. Default is None.
    dataset_kwargs: dict, optional
        Non-positional parameters for the :class:`.DataSet` class, which records
        the experiences. Default is None.
    bot : BotWrapper, optional
        The wrapper for the (physical) robot underlying the agent. Default is None.
    args: tuple
        Positional parameters passed to the agent module.
    kwargs: dict
        Non-positional parameters passed to the agent module.

    Examples
    --------
    >>> from mlpy.agents.modelbased import ModelBasedAgent
    >>> ModelBasedAgent('learningmodule', None, None, None, None, 'qlearner', alpha=0.5)

    This creates an agent with a :class:`.LearningModule` agent module that performs
    qlearning. The parameters are given in the order in which the objects are created.
    Internally the agent creates the learning agent module and the learning agent module
    creates the qlearner.

    Alternatively, non-positional arguments can be used:

    >>> ModelBasedAgent('learningmodule', learner_type='qlearner', alpha=0.5)

    """

    @property
    def module(self):
        """The agent module controlling the actions of the agent.

        Returns
        -------
        IAgentModule :
            The agent module instance.

        """
        return self._module

    def __init__(self, module_type, record=False, dataset_args=None, dataset_kwargs=None,
                 bot=None, *args, **kwargs):
        super(ModelBasedAgent, self).__init__()

        self._last_state = None
        self._last_action = None

        self._bot = bot
        self._record = record

        if self._record:
            dataset_args = tuple(dataset_args) if dataset_args is not None else ()
            dataset_kwargs = dict(dataset_kwargs) if dataset_kwargs is not None else {}

            self._history = DataSet(*dataset_args, **dataset_kwargs)
            self._history.load()

        self._module = AgentModuleFactory.create(module_type, *args, **kwargs)

    def __setstate__(self, d):
        self.__dict__.update(d)

    def init(self, taskspec):
        """Initializes the agent.

        Parameters
        ----------
        taskspec : str
            The task specification.

        """
        ts = TaskSpecParser(taskspec)
        if ts.valid:
            # Todo need to validate values
            gamma = ts.discount_factor
            reward_range = ts.get_reward_range()

            obs = list(ts.get_int_obs())
            obs += list(ts.get_double_obs())
            if len(ts.get_double_obs()) == 0:
                MDPState.set_dtype(MDPState.DTYPE_INT)
            if len(obs) > 0:
                MDPState.set_minmax_features(*obs)

            act = list(ts.get_int_act())
            act += list(ts.get_double_act())
            if len(ts.get_double_act()) == 0:
                MDPAction.set_dtype(MDPAction.DTYPE_INT)
            if len(act) > 0:
                MDPAction.set_minmax_features(*act)

            extra = ts.get_extra()

            v = ['STATEDESCR', 'ACTIONDESCR', 'STATES_PER_DIM', 'ACTIONS_PER_DIM', 'COPYRIGHT']
            pos = []
            for i, id_ in enumerate(list(v)):
                try:
                    pos.append(extra.index(id_))
                except:
                    v.remove(id_)
            sorted_v = sorted(zip(pos, v))
            v = [s[1] for s in sorted_v]

            for i, id_ in enumerate(v):
                val = ts.get_value(i, extra, v)
                if id_ == 'STATEDESCR':
                    MDPState.set_description(eval(val))
                elif id_ == 'ACTIONDESCR':
                    MDPAction.set_description(eval(val))
                elif id_ == 'STATES_PER_DIM':
                    MDPState.set_states_per_dim(eval(val))
                elif id_ == 'ACTIONS_PER_DIM':
                    MDPAction.set_states_per_dim(eval(val))

        self._module.init()

        if self._bot is not None:
            self._bot.init()

        self._last_state = None
        self._last_action = None

    def setup(self):
        # move bot into its initial position
        if self._bot is not None:
            self._bot.start()

    def start(self, observation):
        """Enter the agent and the agent module.

        Perform initialization tasks here.

        Parameters
        ----------
        observation : Observation
            The current observation.

        Returns
        -------
        action : MDPAction
            The next action to perform.

        """
        if self._record:
            self._history.new_sequence()

        self._module.start()

        self._last_state = None
        self._last_action = None

        return self._process_observation(observation)

    def step(self, reward, observation):
        """Update the agent and the agent module.

        The agent and the agent module are updated at every time step
        in the program loop.

        Parameters
        ----------
        reward : float
            The reward awarded for the previous observation and
            preformed action.
        observation : Observation
            The new observation.

        Returns
        -------
        action : MDPAction
            The next action to perform.

        """
        return self._process_observation(observation, reward)

    def end(self, reward):
        """End the episode.

        Parameters
        ----------
        reward : float
            The reward awarded for the previous observation and
            preformed action.

        """
        state = self.obs2state(self._last_state.get())
        self._module.end(Experience(self._last_state, self._last_action, state, reward))

        if self._record:
            self._history.append("state", state)
            self._history.append("label", state.name)
            self._history.save()

    def cleanup(self):
        """Exit the agent and the agent module."""
        if self._bot is not None:
            self._bot.exit()

        self._module.cleanup()

    def obs2state(self, observation):
        """Convert the observation into a state.

        Parameters
        ----------
        observation : Observation
            The new observation

        Returns
        -------
        state : MDPState
            The state corresponding to the observation.

        """
        return MDPState(observation)

    def _process_observation(self, observation, reward=None):
        state = self.obs2state(list(observation.intArray) + list(observation.doubleArray))
        self._module.step(Experience(self._last_state, self._last_action, state, reward))

        action = self._module.choose_action(state)

        self._last_state = copy.deepcopy(state)
        self._last_action = copy.deepcopy(action)
        if not self._module.is_complete():
            self._record_experience(state, action)

        if self._bot is not None:
            self._bot.next_behavior(action)

        return_action = Action()
        if MDPAction.dtype == MDPAction.DTYPE_INT:
            if action is None:
                return_action.intArray = []
            else:
                return_action.intArray = action.tolist()
        elif MDPAction.dtype == MDPAction.DTYPE_FLOAT:
            if action is None:
                return_action.doubleArray = []
            else:
                return_action.doubleArray = action.tolist()
        return return_action

    def _record_experience(self, state, action):
        if self._record:
            if not self._history.has_field("state"):
                self._history.add_field("state", len(state),
                                        dtype=state.dtype,
                                        description=state.description)
            self._history.append("state", state)

            if not self._history.has_field("label"):
                self._history.add_field("label", 1, dtype=DataSet.DTYPE_OBJECT)
            self._history.append("label", state.name)

            if action is not None:
                if not self._history.has_field("act"):
                    self._history.add_field("act", len(action),
                                            dtype=action.dtype,
                                            description=action.description)
                self._history.append("act", action)
