"""
======================================================
Experiment Infrastructure (:mod:`mlpy.experiments`)
======================================================

.. currentmodule:: mlpy.experiments

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Experiment


Tasks
=====

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~task.Task
   ~task.EpisodicTask
   ~task.SearchTask

"""
from __future__ import division, print_function, absolute_import

import time

__all__ = ['task']


class Experiment(object):
    """The experiment class.

    An experiment sets up an agent in an environment and runs
    until the environment is considered to have completed. This can
    be the case when all agents acting in the environment have reached
    their goal state.

    An experiment can consist of multiple episodes and rests itself
    at the end of each episode.

    Parameters
    ----------
    env : Environment
        The environment in which to run the agent(s).

    """
    def __init__(self, env):
        self._t = 0.0
        """:type: float"""

        self._env = env

    def reset(self):
        """Reset the experiment."""
        self._t = time.time()
        self._env.reset(self._t)

    def enter(self):
        """Enter the experiment."""
        self._t = time.time()
        self._env.enter(self._t)

    def update(self):
        """Update all modules during the program loop."""
        dt = time.time() - self._t
        self._t += dt

        self._env.update(dt)

    def exit(self):
        """Exit the experiment."""
        self._env.exit()

    def run(self):
        """Run the experiment.

        The experiment finishes when the environment
        is considered to have completed. Possible causes for completing
        the environment is that all agents have reached a goal state.

        """
        self.enter()

        while True:
            while not self._env.is_complete():
                self.update()

            self.reset()
            if self._env.is_complete():
                self.exit()
                break
