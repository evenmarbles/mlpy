from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import math
from abc import abstractmethod

import numpy as np

from ..modules import UniqueModule
from ..modules.patterns import RegistryInterface
from ..tools.log import LoggingMgr
from ..auxiliary.misc import stdout_redirected, listify
from ..constants import eps
from ..mdp.stateaction import MDPAction
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
        >>> AgentModuleFactory.create('learningmodule', 'qlearner', alpha=0.5)

        This creates a :class:`.LearningModule` instance performing
        q-learning with the learning rate alpha set to 0.5.

        >>> from mlpy.mdp.discrete import DiscreteModel
        >>> from mlpy.planners.discrete import ValueIteration
        >>>
        >>> planner = ValueIteration(DiscreteModel(['out', 'in', 'kick']))
        >>>
        >>> AgentModuleFactory().create('learningmodule', 'modelbasedlearner', planner)

        This creates a learning module using the :class:`.ModelBasedLearner`. The parameters for
        the learner are appended to the end of the argument list.

        Alternatively non-positional arguments can be used:

        >>> AgentModuleFactory().create('learningmodule', 'modelbasedlearner', planner=planner)

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


# noinspection PyMethodMayBeStatic
class IAgentModule(UniqueModule):
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

    def __getstate__(self):
        d = dict(self.__dict__)
        remove_list = ['_logger']
        for key in remove_list:
            del d[key]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

        self._logger = LoggingMgr().get_logger(self._mid)

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

    def init(self):
        """Initialize the agent module."""
        pass

    def start(self):
        """"Start an episode."""
        pass

    def step(self, state):
        """Execute the agent module. This method can optionally be overwritten.

        Parameters
        ----------
        state : MDPState
            The current state

        """
        pass

    def end(self, experience):
        """End the learning agent module.

        Parameters
        ----------
        experience : Experience
            The agent's experience consisting of the previous state, the action performed
            in that state, the current state and the reward awarded.

        """
        pass

    def cleanup(self):
        """Cleanup the agent module. """
        pass

    @abstractmethod
    def choose_action(self, state):
        """Choose the next action the agent will execute.

        Parameters
        ----------
        state : MDPState
            The current state the agent is in.

        Returns
        -------
        MDPAction :
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

            modelbasedlearner
                The model based learner performs reinforcement learning using the provided planner
                and model (:class:`~mlpy.learners.online.rl.ModelBasedLearner`).

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

    args : tuple, optional
        Positional parameters passed to the learner for initialization. See the appropriate learner
        type for more information. Default is None.
    kwargs : dict, optional
        Non-positional parameters passed to the learner for initialization. See the appropriate learner
        type for more information. Default is None.

    """
    def __init__(self, learner_type, *args, **kwargs):
        super(LearningModule, self).__init__()

        self._learner = LearnerFactory.create(learner_type, *args, **kwargs)

    def init(self):
        """Initialize the learning agent module."""
        self._learner.init()

    def start(self):
        """"Start an episode."""
        self._learner.start()

    def step(self, experience):
        """Execute the learner.

        Update models with the current experience (:class:`~mlpy.mdp.stateaction.Experience`).
        Additionally, online learning is performed at this point.

        Parameters
        ----------
        experience : Experience
            The current experience, consisting of previous state and action,
            current state and the awarded reward.

        """
        self._learner.step(experience)

        if experience.state is not None:
            if self._learner.type == 'online':
                self._learner.learn(experience)

    def end(self, experience):
        """End the episode.

        Offline learning is performed at the end of an episode.

        Parameters
        ----------
        experience : Experience
            The agent's experience consisting of the previous state, the action performed
            in that state, the current state and the reward awarded.

        """
        super(LearningModule, self).end(experience)

        self._learner.step(experience)

        if self._learner.type == 'offline':
            self.terminate(self._learner.learn())
            experience = None

        self._learner.end(experience)

    def choose_action(self, state):
        """Choose the next action.

        The next action the agent will execute is selected based on the current
        state and the policy that has been derived by the learner so far.

        Parameters
        ----------
        state : MDPState
            The current state the agent is in.

        Returns
        -------
        MDPAction :
            The next action

        """
        action = MDPAction.get_noop_action()
        if state is not None:
            action = self._learner.choose_action(state)

        return action


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

    """
    def __init__(self, policies):
        super(FollowPolicyModule, self).__init__()

        self._policies = None
        """:type: ndarray"""
        self._current = None
        self._cntr = None

        self.change_policies(policies)

    def init(self):
        """Initialize the follow policy agent module."""
        self._current += 1
        if self._current >= self._policies.shape[0]:
            self._current = 0
        self._logger.info("Running policy id: {0}".format(self._current))

    def start(self):
        """"Start an episode."""
        self._cntr = -1

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
        if policies.shape[0] < 1:
            raise IndexError("No policies available.")

        self._policies = policies
        self._current = -1
        self._cntr = -1

    def choose_action(self, _):
        """Choose the next action.

        The next action the agent will execute is selected based on the current
        state and the policy that has been derived by the learner so far.

        Returns
        -------
        MDPAction :
            The next action

        """
        action = None

        self._cntr += 1

        if self._cntr < self._policies[self._current].shape[1]:
            action = MDPAction(self._policies[self._current][:, self._cntr])
            self._logger.debug("'%s'" % action)

        self.terminate(action is None)
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

    Notes
    -----
    This agent module is requires the `PyGame <http://www.pygame.org/>`_ library.

    """
    def __init__(self, events_map):

        super(UserModule, self).__init__()

        self.pygame = None
        self._js = None

        self._effectors = ["LArm"]
        self._effector_iter = 0

        self._events_map = events_map
        """:type: ConfigMgr"""
        try:
            self._discretize_joystick_axis = self._events_map.get("joystick.axis.discretize")
        except KeyError:
            self._discretize_joystick_axis = False

    def init(self):
        """Initialize the user agent module."""
        import pygame
        self.pygame = pygame

        self.pygame.init()

        if self.pygame.joystick.get_count() > 0:
            self._js = self.pygame.joystick.Joystick(0)
            self._js.init()

        # if not self._js:
        self.pygame.display.set_mode((320, 240))
        self.pygame.display.set_caption('Read Keyboard Input')
        self.pygame.mouse.set_visible(0)

    def start(self):
        """"Start an episode."""
        self._effector_iter = 0

    def cleanup(self):
        """ Exit the agent module. """
        super(UserModule, self).cleanup()
        self.pygame.quit()

    def choose_action(self, _):
        """Choose the next action.

        Return the next action the agent will execute depending on the
        key/button pressed.

        Returns
        -------
        MDPAction :
            The next action

        """
        action = np.zeros(MDPAction.nfeatures)

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

        name = MDPAction.get_name(action)

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
                            descr = MDPAction.description[name]["descr"]
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
                            descr = MDPAction.description[name]["descr"]
                            for e in self._effectors:
                                if e in config["effectors"]:
                                    action[descr[e][config["label"]]] = hat * config["scale"]
                except KeyError:
                    pass

        if self._discretize_joystick_axis:
            action = self._discretize(action)

        action = MDPAction(action, name)
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
