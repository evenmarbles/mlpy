from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import math
from abc import abstractmethod

import numpy as np

from ..modules import Module
from ..modules.patterns import RegistryInterface
from ..tools.log import LoggingMgr
from ..auxiliary.misc import stdout_redirected, listify
from ..constants import eps
from ..mdp.stateaction import Experience, Action
from ..learners import LearnerFactory


class AgentModuleFactory(object):
    # noinspection PyUnusedLocal
    """The agent module factory.

        An instance of an agent module can be created by passing the agent
        module type. The module type is the name of the module. The set of
        agent modules can be extended by inheriting from :class:`IAgentModule`.
        However, for the agent module to be registered, the custom module must
        be imported by the time the agent module factory is called.

        Examples
        --------
        >>> from mlpy.agents.modules import AgentModuleFactory
        >>> AgentModuleFactory.create('learningmodule', 'qlearner', max_steps=10)

        This creates a :class:`.LearningModule` instance performing
        q-learning with max_steps set to 10.

        >>> def get_reward(state, action):
        ...     return 1.0
        ...
        >>> AgentModuleFactory().create('learningmodule', 'qlearner', get_reward,
        ...                             max_steps=10)

        This creates a q-learning learning module instance passing a reward callback
        function and sets max_steps to 10.

        >>> from mlpy.mdp.discrete import DiscreteModel
        >>> from mlpy.planners.discrete import ValueIteration
        >>>
        >>> planner = ValueIteration(DiscreteModel(['out', 'in', 'kick']))
        >>>
        >>> AgentModuleFactory().create('learningmodule', 'rldtlearner', None, planner,
        ...                             max_steps=10)

        This creates a learning module using the :class:`.RLDTLearner`. The parameters for
        the learner are appended to the end of the argument list. Notice that since positional
        arguments are used to pass the planner, the reward callback must be accounted for by
        setting it to `None`.

        Alternatively non-positional arguments can be used:

        >>> AgentModuleFactory().create('learningmodule', 'rldtlearner', planner=planner,
        ...                             max_steps=10)

        Notes
        -----
        The agent module factory is being called by the :class:`.Agent` during
        initialization to create the agents controller.

        """
    @staticmethod
    def create(_type, *args, **kwargs):
        """Create an agent module of the given type.

        Parameters
        ----------
        _type : str
            The agent module type. Valid agent module types:

            followpolicymodule
                The agent follows a given policy: a :class:`.FollowPolicyModule`
                is created.

            learningmodule
                The agent learns according to a specified learner: a
                :class:`.LearningModule` is created.

            usermodule
                The agent is user controlled via keyboard or PS2 controller:
                a :class:`.UserModule` is created.

        args : tuple, optional
            Positional arguments passed to the class of the given type for
            initialization.
        kwargs : dict, optional
            Non-positional arguments passed to the class of the given type
            for initialization.

        Returns
        -------
        IAgentModule :
            Agent model instance of given type.

        """
        # noinspection PyUnresolvedReferences
        return IAgentModule.registry[_type.lower()](*args, **kwargs)


class IAgentModule(Module):
    """Agent module base interface class.

    The agent (:class:`~mlpy.agents.Agent`) uses an agent module, which specifies how
    the agent is controlled. Valid agent module types are:

        followpolicymodule
            The agent follows a given policy (:class:`FollowPolicyModule`)

        learningmodule
            The agent learns according to a specified learner
            (:class:`LearningModule`).

        usermodule
            The agent is user controlled via keyboard or PS2 controller
            (:class:`UserModule`).

    Notes
    -----
    Every class inheriting from IAgentModule must implement :meth:`get_next_action`.

    """
    __metaclass__ = RegistryInterface

    def __init__(self):
        super(IAgentModule, self).__init__()
        self._logger = LoggingMgr().get_logger(self._mid)

        self._completed = False

        self._state = None

    def reset(self, t, **kwargs):
        """Reset the agent module.

        Parameters
        ----------
        t : float
            The current time (sec)
        kwargs : dict, optional
            Non-positional parameters, optional.

        """
        super(IAgentModule, self).reset(t, **kwargs)
        self._state = None

    def terminate(self, value):
        """Set the termination flag.

        Parameters
        ----------
        value : bool
            The value of the termination flag.

        """
        self._completed = value

    def is_complete(self):
        """Check if the agent module has completed.

        Returns
        -------
        bool :
            Whether the agent module has completed or not.

        """
        return self._completed

    # noinspection PyMethodMayBeStatic
    def execute(self, state):
        """Execute the agent module. This method can optionally be overwritten.

        Parameters
        ----------
        state : State
            The current state

        """
        pass

    @abstractmethod
    def get_next_action(self):
        """Return the next action the agent will execute.

        Returns
        -------
        Action :
            The next action

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        Notes
        -----
        This is an abstract method and *must* be implemented by its deriving class.

        """
        raise NotImplementedError


class LearningModule(IAgentModule):
    # noinspection PyUnusedLocal
    """Learning agent module.

    The learning agent module allows the agent to learn from passed
    experiences.

    Parameters
    ----------
    learner_type : str
        The learning type. Based on the type the appropriate learner module is created.
        Valid learning types are:

            qlearner
                The learner performs q-learning, a reinforcement learning variant
                (:class:`~mlpy.learners.online.rl.QLearner`).

            rldtlearner
                The learner performs reinforcement learning with decision trees (RLDT),
                a method introduced by Hester, Quinlan, and Stone which builds a generalized
                model for the transitions and rewards of the environment
                (:class:`~mlpy.learners.online.rl.RLDTLearner`).

            apprenticeshiplearner
                The learner performs apprenticeship learning via inverse reinforcement
                learning, a method introduced by Abbeel and Ng which strives to imitate
                the demonstrations given by an expert
                (:class:`~mlpy.learners.offline.irl.ApprenticeshipLearner`).

            incrapprenticeshiplearner
                The learner incrementally performs apprenticeship learning via inverse
                reinforcement learning. Inverse reinforcement learning assumes knowledge
                of the underlying model. However, this is not always feasible. The
                incremental apprenticeship learner updates its model after every iteration
                by executing the current policy
                (:class:`~mlpy.learners.offline.irl.IncrApprenticeshipLearner`).

    cb_get_reward : callable, optional
        A callback function to retrieve the reward based on the current state and action.
        Default is `None`.

        The function must be of the following format:

            >>> def callback(state, action):
            >>>     pass

    learner_params : dict, optional
        Parameters passed to the learner for initialization. See the appropriate learner
        type for more information. Default is None.

    """
    def __init__(self, learner_type, cb_get_reward=None, *args, **kwargs):
        super(LearningModule, self).__init__()

        self._learner = LearnerFactory.create(learner_type, *args, **kwargs)

        self._cb_get_reward = cb_get_reward
        if self._cb_get_reward is not None and not hasattr(self._cb_get_reward, '__call__'):
            raise ValueError("Expected a function or None, but got %r" % type(cb_get_reward))

        self._state = None
        self._action = None

    def reset(self, t, **kwargs):
        """Reset the module for the next iteration.

        Offline learning is performed at the end of the iteration.

        Parameters
        ----------
        t : float
            The current time (sec).
        kwargs : dict, optional
            Non-positional parameters.

        """
        super(LearningModule, self).reset(t, **kwargs)

        if self._learner.type == 'offline':
            self.terminate(self._learner.learn())

        self._learner.reset(t, **kwargs)

    def execute(self, state):
        """Execute the learner.

        Update models with the current experience (:class:`~mlpy.mdp.stateaction.Experience`).
        Additionally, online learning is performed at this point.

        Parameters
        ----------
        state : State
            The current state.

        """
        experience = Experience(self._state, self._action, state, self._cb_get_reward(self._state, self._action))
        self._learner.execute(experience)

        if self._state is not None:
            if self._learner.type == 'online':
                if self._state is not None:
                    self._learner.learn(experience)

        self._state = state

    def get_next_action(self):
        """Return the next action.

        The next action the agent will execute is selected based on the current
        state and the policy that has been derived by the learner so far.

        Returns
        -------
        Action :
            The next action

        """
        self._action = Action.get_noop_action()
        if self._state is not None:
            self._action = self._learner.choose_action(self._state)
        return self._action


class FollowPolicyModule(IAgentModule):
    """The follow policy agent module.

    The follow policy agent module follows a given policy choosing the next action
    based on that policy.

    Parameters
    ----------
    policies : array_like, shape (`n`, `nfeatures`, `ni`)
        A list of policies (i.e., action sequences), where `n` is the
        number of policies, `nfeatures` is the number of action features,
        and `ni` is the sequence length.
    niter : int, optional
        The number of times each policy is repeated. Default is 1.
    start : int, optional
        The first policy to execute. Default is 0.

    """
    def __init__(self, policies, niter=None, start=None):
        super(FollowPolicyModule, self).__init__()

        self._policies = None
        """:type: ndarray"""
        self._npolicies = None

        self._policy_cntr = None
        self._policy_len = None
        self._iter = 0

        self._current = start if start is not None else 0
        self._niter = niter if niter is not None else 1

        self.change_policies(policies)
        self._logger.info("Current policy id: {0}".format(self._current))

    def reset(self, t, **kwargs):
        """Reset the module for the next iteration.

        Offline learning is performed at the end of the iteration.

        Parameters
        ----------
        t : float
            The current time (sec).
        kwargs : dict, optional
            Non-positional parameters.

        """
        super(FollowPolicyModule, self).reset(t, **kwargs)

        self._iter += 1
        if self._iter >= self._niter:
            self._iter = 0
            self._current += 1

            if self._current >= self._npolicies:
                self.terminate(True)
                return

            self._policy_len = self._policies[self._current].shape[1]

        self._policy_cntr = -1
        self._logger.info("Current policy id: {0}".format(self._current))

    def change_policies(self, policies):
        """ Exchange the list of policies.

        Parameters
        ----------
        policies : array_like, shape (`n`, `nfeatures`, `ni`)
            A list of policies (i.e., action sequences), where `n` is the
            number of policies, `nfeatures` is the number of action features,
            and `ni` is the sequence length.

        Raises
        ------
        IndexError
            If the list is empty.

        """
        self._npolicies = policies.shape[0]
        if self._npolicies < 1:
            raise IndexError("No policies available.")

        self._policies = policies

        self._policy_cntr = -1
        self._policy_len = self._policies[self._current].shape[1]

    def get_next_action(self):
        """Return the next action.

        The next action the agent will execute is selected based on the current
        state and the policy that has been derived by the learner so far.

        Returns
        -------
        Action :
            The next action

        """
        action = None

        self._policy_cntr += 1

        if self._policy_cntr < self._policy_len:
            action = Action(self._policies[self._current][:, self._policy_cntr])
            self._logger.debug("'%s'" % action)

        return action


class UserModule(IAgentModule):
    """The user agent module.

    With the user agent module the agent is controlled by the user via the
    keyboard or a PS2 controller. The mapping of keyboard/joystick keys
    to events is given through a configuration file.

    Parameters
    ----------
    events_map : ConfigMgr
        The configuration mapping keyboard/joystick keys to events that
        are translated into actions.

        :Example:

            ::

                {
                    "keyboard": {
                        "down": {
                            "pygame.K_ESCAPE": "QUIT",
                            "pygame.K_SPACE": [-1.0],
                            "pygame.K_LEFT" : [-0.004],
                            "pygame.K_RIGHT":  [0.004]
                        }
                    }
                }

    niter : int
        The number of episodes to capture..

    Notes
    -----
    This agent module is requires the `PyGame <http://www.pygame.org/>`_ library.

    """
    def __init__(self, events_map, niter=None):

        super(UserModule, self).__init__()

        import pygame
        self.pygame = pygame

        self._niter = niter if niter is not None else 1
        """:type: int"""
        self._iter = 0

        self._effectors = ["LArm"]
        self._effector_iter = 0

        self._events_map = events_map
        """:type: ConfigMgr"""
        try:
            self._discretize_joystick_axis = self._events_map.get("joystick.axis.discretize")
        except KeyError:
            self._discretize_joystick_axis = False

        self.pygame.init()

        self._js = None
        if self.pygame.joystick.get_count() > 0:
            self._js = self.pygame.joystick.Joystick(0)
            self._js.init()

        # if not self._js:
        self.pygame.display.set_mode((320, 240))
        self.pygame.display.set_caption('Read Keyboard Input')
        self.pygame.mouse.set_visible(0)

    def reset(self, t, **kwargs):
        """Reset the module for the next iteration.

        Offline learning is performed at the end of the iteration.

        Parameters
        ----------
        t : float
            The current time (sec).
        kwargs : dict, optional
            Non-positional parameters.

        """
        super(UserModule, self).reset(t, **kwargs)

        self._effector_iter = 0

        self._iter += 1
        if self._iter >= self._niter:
            self.terminate(True)

    def exit(self):
        """ Exit the agent module. """
        super(UserModule, self).exit()
        self.pygame.quit()

    def get_next_action(self):
        """Return the next action.

        Return the next action the agent will execute depending on the
        key/button pressed.

        Returns
        -------
        Action :
            The next action

        """
        action = np.zeros(Action.nfeatures)

        for event in self.pygame.event.get():
            if event.type == self.pygame.QUIT:
                return None

            if event.type == self.pygame.KEYDOWN:
                try:
                    button_value = self._events_map.get("keyboard.down." + str(event.key))
                    if button_value:
                        if button_value == "QUIT":
                            self.pygame.event.post(self.pygame.event.Event(self.pygame.QUIT))
                            break
                        else:
                            action = np.asarray(button_value)
                except KeyError:
                    pass

            if event.type == self.pygame.JOYBUTTONDOWN:
                try:
                    button_value = self._events_map.get("joystick.button.down." + str(event.dict['button']))
                    if button_value:
                        if button_value == "QUIT":
                            self.pygame.event.post(self.pygame.event.Event(self.pygame.QUIT))
                            break
                        else:
                            if isinstance(button_value, dict):
                                operation = button_value["operation"]
                                if operation == "effectors":
                                    self._effector_iter += 1
                                    if self._effector_iter >= len(button_value["effectors"]):
                                        self._effector_iter = 0
                                    self._effectors = listify(button_value["effectors"][self._effector_iter])
                                    self._logger.info(self._effectors)
                except KeyError:
                    pass

        name = Action.get_name(action)

        if self._js:
            with stdout_redirected():
                num_axis = self._js.get_numaxes()
            for i in range(num_axis):
                try:
                    config = self._events_map.get("joystick.axis." + str(i))
                    if config:
                        with stdout_redirected():
                            axis = self._js.get_axis(i)
                        axis = axis if abs(axis) > eps else 0.0

                        if not axis == 0:
                            descr = Action.description[name]["descr"]
                            for e in self._effectors:
                                if e in config["effectors"]:
                                    action[descr[e][config["label"]]] = axis * config["scale"]
                except KeyError:
                    pass

            with stdout_redirected():
                num_hats = self._js.get_numhats()
            for i in range(num_hats):
                try:
                    with stdout_redirected():
                        hat = self._js.get_hat(i)
                    for j, hat in enumerate(hat):
                        config = None
                        if not hat == 0:
                            config = self._events_map.get("joystick.hat." + ("x", "y")[j])
                        if config:
                            descr = Action.description[name]["descr"]
                            for e in self._effectors:
                                if e in config["effectors"]:
                                    action[descr[e][config["label"]]] = hat * config["scale"]
                except KeyError:
                    pass

        if self._discretize_joystick_axis:
            action = self._discretize(action)

        action = Action(action, name)
        return action

    # noinspection PyMethodMayBeStatic
    def _discretize(self, vec):
        new_vec = [0.0 if -0.5 <= vec[0] <= 0.5 else vec[0]/abs(vec[0]),
                   0.0 if -0.5 <= vec[1] <= 0.5 else vec[1]/abs(vec[1]),
                   0.0 if -0.5 <= vec[2] <= 0.5 else vec[2]/abs(vec[2])]

        uvec = [0.0, 0.0, 0.0]
        mag = math.sqrt(math.pow(new_vec[0], 2) + math.pow(new_vec[1], 2) + math.pow(new_vec[2], 2))
        if not mag == 0.0:
            # noinspection PyTypeChecker
            uvec = np.true_divide(np.asarray(vec), mag)

        uvec = np.asarray(uvec) * 0.01

        # noinspection PyUnresolvedReferences
        return uvec.tolist() + [0.0, 0.0, 0.0]
