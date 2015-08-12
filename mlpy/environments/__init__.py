"""
==========================================
Environments (:mod:`mlpy.environments`)
==========================================

.. currentmodule:: mlpy.environments

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Environment


Gridworld
=========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~gridworld.Cell
   ~gridworld.GridWorld

Nao
===

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~nao.NaoEnvFactory
   ~nao.PhysicalWorld
   ~nao.Webots

"""
from __future__ import division, print_function, absolute_import

from ..auxiliary.misc import listify
from ..modules import Module

__all__ = ['gridworld', 'nao']


class Environment(Module):
    """The environment base class.

    The environment specifies the setting in which the agent(s) act.
    The class is responsible to update the agent(s) at each time step of the
    program loop and keeps track if the agents' task is complete.

    Parameters
    ----------
    agents : Agent or list[Agent], optional
        A list of agents that act in the environment.

    """

    def __init__(self, agents=None):
        super(Environment, self).__init__()

        self._agents = {}
        """:type": dict[str|int, Agent]"""

        for agent in listify(agents):
            self._agents[agent.mid] = agent

    def __str__(self):
        return self.__class__.__name__

    def reset(self, t, **kwargs):
        """Resets the environment.

        The environment and all agents are reset.

        Parameters
        ----------
        t : float
            The current time (sec)
        kwargs : dict, optional
            Non-positional parameters, optional.

        """
        super(Environment, self).reset(t, **kwargs)

        for agent in self._agents.itervalues():
            agent.reset(t, **kwargs)

    def is_complete(self):
        """Checks if the environment has completed.

        This is dependent on whether the agent(s) have
        completed their task.

        Returns
        -------
        bool :
            Whether the environment has reached some end goal.

        """
        for agent in self._agents.itervalues():
            if not agent.is_task_complete():
                return False

        return True

    def add_agents(self, agent):
        """Add an agent to the environment.

        Parameters
        ----------
        agents : Agent or list[Agent], optional
            A list of agents added to the environment.

        """
        for a in listify(agent):
            self._agents[a.mid] = a

    def get_agent(self, mid):
        """Return the agent specified by the id.

        Returns
        -------
        Agent :
            The agent identified by the id.

        """
        return self._agents[mid]

    def enter(self, t):
        """Enter the environment and all agents.

        Parameters
        ----------
        t : float
            The current time (sec).

        """
        super(Environment, self).enter(t)

        for agent in self._agents.itervalues():
            agent.enter(t)

    def update(self, dt):
        """Update the environment and all agents.

        Parameters
        ----------
        dt : float
            The elapsed time (sec)

        """
        super(Environment, self).update(dt)

        for agent in self._agents.itervalues():
            agent.update(dt)

    def exit(self):
        """Exit the environment and all agents.

        Perform cleanup tasks here.

        """
        super(Environment, self).exit()

        for agent in self._agents.itervalues():
            agent.exit()
