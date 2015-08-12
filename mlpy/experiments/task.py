from __future__ import division, print_function, absolute_import

import numpy as np
from abc import ABCMeta, abstractmethod

from ..mdp.stateaction import State


class Task(object):
    """The task description base class.

    A task description describes the task the agent is to perform. The task
    description allows to configure :class:`.State` and :class:`.Action` by
    setting the number of features, the description and by overwriting the
    static functions :func:`~mlpy.mdp.stateaction.State.is_valid`,
    :func:`~mlpy.mdp.stateaction.MDPPrimitive.encode`, and
    :func:`~mlpy.mdp.stateaction.MDPPrimitive.decode` at runtime.

    Parameters
    ----------
    env : Environment, optional
        The environment in which the agent performs the task.

    See Also
    --------
    :class:`EpisodicTask`, :class:`SearchTask`

    Notes
    -----
    Any task should inherit from this base class or any class deriving from
    this class. Every deriving class must overwrite the methods :meth:`_configure_state`
    and :meth:`_configure_action` to configure the classes :class:`.State` and
    :class:`.Action`, respectively.

    For both :class:`.State` and :class:`.Action` the appropriate class variables
    can be set by calling the following functions:

    * :func:`~mlpy.mdp.stateaction.State.set_nfeatures`

    * :func:`~mlpy.mdp.stateaction.State.set_dtype`

    * :func:`~mlpy.mdp.stateaction.State.set_description`

    * :func:`~mlpy.mdp.stateaction.State.set_discretized`

    * :func:`~mlpy.mdp.stateaction.State.set_minmax_features`

    * :func:`~mlpy.mdp.stateaction.State.set_states_per_dim`


    Overwrite the following :class:`.State` and :class:`.Action` methods to allow
    for more readable descriptions:

    * :func:`~mlpy.mdp.stateaction.State.encode`

    * :func:`~mlpy.mdp.stateaction.State.decode`


    Additionally, the :class:`.State` class provides a method to check a state's
    validity:

    * :func:`~mlpy.mdp.stateaction.State.is_valid`

    """
    @property
    def is_episodic(self):
        """Identifies if the task is episodic or not.

        Returns
        -------
        bool :
            Whether this task is episodic or not.

        """
        return self._is_episodic

    @property
    def event_delay(self):
        """Event delay.

        The time in milliseconds (ms) by which the fsm event is delayed
        once termination is requested.

        Returns
        -------
        float :
            The time in milliseconds.

        """
        return self._event_delay_on_term

    def __init__(self, env=None):
        self._env = env

        self._configure_state()
        self._configure_action()

        self._is_episodic = False
        self._request_termination = False
        self._completed = False

        self._event_delay_on_term = 0.0

    # noinspection PyUnusedLocal
    def reset(self, t, **kwargs):
        """Reset the task.

        Parameters
        ----------
        t : float
            The current time (sec).
        kwargs : dict, optional
            Non-positional parameters.

        """
        self._request_termination = False
        self._completed = False

    def request_termination(self, value):
        """Request termination of the task.

        Parameters
        ----------
        value : bool
            The value to set the termination requested flag to.

        """
        self._request_termination = value

    def termination_requested(self):
        """Check if termination was requested.

        Returns
        -------
        bool :
            Whether termination was requested or not.

        """
        return self._request_termination

    def terminate(self, value):
        """Set the termination flag.

        Parameters
        ----------
        value : bool
            The value to set the termination flag to.

        """
        self._completed = value

    def is_complete(self):
        """Check if the task has completed.

        Returns
        -------
        bool :
            Whether the task has completed or not.

        """
        return self._completed

    def sensation(self, **kwargs):
        """Gather the state feature information.

        Gather the state information (i.e. features) according to
        the task from the agent's senses.

        Parameters
        ----------
        kwargs: dict
            Non-positional arguments needed for gathering the
            information.

        Returns
        -------
        features : array, shape (`nfeatures`,)
            The sensed features

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def get_reward(self, state, action):
        """Retrieve the reward.

        Retrieve the reward for the given state and action from
        the environment.

        Parameters
        ----------
        state : State
            The current state.
        action : Action
            The current action.

        Returns
        -------
        float :
            The reward.

        """
        return None

    # noinspection PyMethodMayBeStatic
    def _configure_state(self):
        """Configure :class:`.State`.

        Notes
        -----
        The appropriate class variables can be set by calling the
        following functions:

        * :func:`~mlpy.mdp.stateaction.State.set_nfeatures`

        * :func:`~mlpy.mdp.stateaction.State.set_dtype`

        * :func:`~mlpy.mdp.stateaction.State.set_description`

        * :func:`~mlpy.mdp.stateaction.State.set_discretized`

        * :func:`~mlpy.mdp.stateaction.State.set_minmax_features`

        * :func:`~mlpy.mdp.stateaction.State.set_states_per_dim`


        Overwrite the following methods to allow for more readable
        descriptions and to validate the state:

        * :func:`~mlpy.mdp.stateaction.State.encode`

        * :func:`~mlpy.mdp.stateaction.State.decode`

        * :func:`~mlpy.mdp.stateaction.State.is_valid`

        """
        pass

    # noinspection PyMethodMayBeStatic
    def _configure_action(self):
        """Configure :class:`.Action`.

        Notes
        -----
        The appropriate class variables can be set by calling the
        following functions:

        * :func:`~mlpy.mdp.stateaction.State.set_nfeatures`

        * :func:`~mlpy.mdp.stateaction.State.set_dtype`

        * :func:`~mlpy.mdp.stateaction.State.set_description`

        * :func:`~mlpy.mdp.stateaction.State.set_discretized`

        * :func:`~mlpy.mdp.stateaction.State.set_minmax_features`

        * :func:`~mlpy.mdp.stateaction.State.set_states_per_dim`


        Overwrite the following methods to allow for more readable
        descriptions:

        * :func:`~mlpy.mdp.stateaction.State.encode`

        * :func:`~mlpy.mdp.stateaction.State.decode`

        """
        pass


# noinspection PyAbstractClass
class EpisodicTask(Task):
    """The episodic task description base class.

    This class automatically identifies the task as an episodic task.
    An episodic task has a set of actions that transitions the agent
    into a terminal state. Once a terminal state is reached the task
    is complete.

    Parameters
    ----------
    initial_states : str or State or list[str or State]
        List of possible initial states.
    terminal_states : str or State or list[str or State]
        List of terminal states.
    env : Environment, optional
        The environment in which the agent performs the task.

    Notes
    -----
    Every deriving class must overwrite the methods :meth:`_configure_state`
    and :meth:`_configure_action` to configure the classes :class:`.State` and
    :class:`.Action`, respectively.

    For both :class:`.State` and :class:`.Action` the appropriate class variables
    can be set by calling the following functions:

    * :func:`~mlpy.mdp.stateaction.State.set_nfeatures`

    * :func:`~mlpy.mdp.stateaction.State.set_dtype`

    * :func:`~mlpy.mdp.stateaction.State.set_description`

    * :func:`~mlpy.mdp.stateaction.State.set_discretized`

    * :func:`~mlpy.mdp.stateaction.State.set_minmax_features`

    * :func:`~mlpy.mdp.stateaction.State.set_states_per_dim`


    Overwrite the following :class:`.State` and :class:`.Action` methods to allow
    for more readable descriptions:

    * :func:`~mlpy.mdp.stateaction.State.encode`

    * :func:`~mlpy.mdp.stateaction.State.decode`


    Additionally, the :class:`.State` class provides a method to check a state's
    validity. Overwrite this function to specify valid states:

    * :func:`~mlpy.mdp.stateaction.State.is_valid`

    """
    def __init__(self, initial_states, terminal_states, env=None):
        super(EpisodicTask, self).__init__(env)

        self._is_episodic = True
        State.initial_states = initial_states
        State.terminal_states = terminal_states

    @staticmethod
    def random_initial_state():
        """Return a random initial state.

        Returns
        -------
        str or State :
            A random initial state.

        """
        if isinstance(State.initial_states, list):
            return np.random.choice(State.initial_states)
        return State.initial_states


# noinspection PyAbstractClass
class SearchTask(EpisodicTask):
    """The abstract class for a search task definition.

    Parameters
    ----------
    initial_states : str or State or list[str or State]
        List of possible initial states.
    terminal_states : str or State or list[str or State]
        List of terminal states.
    env : Environment, optional
        The environment in which the agent performs the task.

    """
    __metaclass__ = ABCMeta

    def __init__(self, initial_states, terminal_states=None, env=None):
        super(SearchTask, self).__init__(initial_states, terminal_states, env)

    @abstractmethod
    def get_successor(self, state):
        """Find valid successors.

        Finds all valid successors (state-action pairs) for the given ``state``.

        Parameters
        ----------
        state : int or tuple[int]
            The state from which to find successors.

        Returns
        -------
        list[tuple(str, str or tuple[str])] :
            A list of all successor.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    @staticmethod
    def get_path_cost(c, _):
        """Returns the cost for the current path.

        Parameters
        ----------
        c : float
            The current cost for the path.

        Returns
        -------
        float :
            The updated cost.

        """
        return c + 1
