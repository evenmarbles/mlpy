"""
====================================
Agent design (:mod:`mlpy.agents`)
====================================

.. currentmodule:: mlpy.agents

This module contains functionality for designing agents
navigating inside an :class:`.Environment`.

Control of the agents is specified by an agent module which
is handled by the :class:`Agent` base class.

An agent class deriving from :class:`Agent` can also make use
of a finite state machine (FSM) to control the agent's behavior
and a world model to maintain a notion of the current state of
the world.


Agents
======

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Agent
   ~modules.AgentModuleFactory
   ~modules.IAgentModule
   ~modules.LearningModule
   ~modules.FollowPolicyModule
   ~modules.UserModule


World Model
===========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~world.WorldObject
   ~world.WorldModel


Finite State Machine
====================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~fsm.Event
   ~fsm.EmptyEvent
   ~fsm.FSMState
   ~fsm.Transition
   ~fsm.OnUpdate
   ~fsm.StateMachine

"""
from __future__ import division, print_function, absolute_import

from ..modules import Module
from .modules import AgentModuleFactory

__all__ = ['fsm', 'modules', 'world']


class Agent(Module):
    """The agent base class.

    Agents act in an environment (:class:`.Environment`) usually performing a task
    (:class:`.Task`). Depending on the agent module they either follow a policy
    (:class:`.FollowPolicyModule`), are user controlled via the keyboard or a PS2
    controller (:class:`.UserModule`), or learn according to a learner
    (:class:`.LearningModule`). New agent modules can be created by inheriting from the
    :class:`.IAgentModule` class.

    Parameters
    ----------
    mid : str, optional
        The agent's unique identifier.
    module_type : str
        The agent module type, which is the name of the class.
    task: Task
        The task the agent must complete.
    args: tuple
        Positional parameters passed to the agent module.
    kwargs: dict
        Non-positional parameters passed to the agent module.

    Examples
    --------
    >>> from mlpy.agents import Agent
    >>> Agent(module_type='learningmodule', learner_type='qlearner', max_steps=10)
    >>> Agent(None, 'learningmodule', None, 'qlearner', max_steps=10)

    This creates an agent with a :class:`.LearningModule` agent module that performs
    qlearning. The parameters are given in the order in which the objects are created.
    Internally the agent creates the learning agent module and the learning agent module
    creates the qlearner.

    Alternatively, non-positional arguments can be used:

    >>> Agent(module_type='learningmodule', learner_type='qlearner', max_steps=10)

    >>> from mlpy.experiments.task import Task
    >>> task = Task()
    >>> Agent(None, 'learningmodule', task, 'qlearner', max_steps=10)

    This creates an agent that performs the given task. If a task is given,
    the method :meth:`.Task.get_reward` is passed as the reward callback to the
    learning module to retrieve the reward. By default :meth:`.Task.get_reward`
    returns `None`.

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

    @property
    def task(self):
        """The task the agent is to perform.

        Returns
        -------
        Task :
            The task to perform.

        """
        return self._task

    def __init__(self, mid=None, module_type=None, task=None, *args, **kwargs):
        super(Agent, self).__init__(mid)

        self._module = None
        if module_type is not None:
            if module_type == 'learningmodule' and task is not None:
                if "learner_type" in kwargs:
                    kwargs.update({'cb_get_reward': task.get_reward})
                else:
                    args_ = list(args)
                    args_.insert(1, task.get_reward)
                    args = tuple(args_)
            self._module = AgentModuleFactory.create(module_type, *args, **kwargs)

        self._task = task

    def reset(self, t, **kwargs):
        """Reset the agent's state.

        Parameters
        ----------
        t : float
            The current time (sec).
        kwargs : dict, optional
            Non-positional parameters.

        """
        super(Agent, self).reset(t, **kwargs)

        if self._task is not None:
            self._task.reset(t, **kwargs)

        if self._module is not None:
            self._module.reset(t, **kwargs)
            if self._module.is_complete():
                self._task.terminate(True)

    def enter(self, t):
        """Enter the agent and the agent module.

        Perform initialization tasks here.

        Parameters
        ----------
        t : float
            The current time (sec).

        """
        super(Agent, self).enter(t)

        if self._module is not None:
            self._module.enter(t)

    def update(self, dt):
        """Update the agent and the agent module.

        The agent and the agent module are updated at every time step
        in the program loop.

        Parameters
        ----------
        dt : float
            The elapsed time (sec)

        """
        super(Agent, self).update(dt)

        if self._module is not None:
            self._module.update(dt)

    def exit(self):
        """Exit the agent and the agent module.

        Perform cleanup tasks here.

        """
        super(Agent, self).exit()

        if self._module is not None:
            self._module.exit()

    def is_task_complete(self):
        """Check if the agent's task is completed.

        This could be because a terminal state was reached.

        Returns
        -------
        bool :
            The value of the termination flag.

        """
        if self._task is None:
            raise UserWarning("No task registered")

        return self._task.is_complete()
