"""
Online learners (:mod:`mlpy.learners.online`)
================================================

.. currentmodule:: mlpy.learners.online

.. autosummary::
   :toctree: generated/
   :nosignatures:

   IOnlineLearner

Reinforcement learning
----------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~rl.QLearner
   ~rl.Cacla
   ~rl.ModelBasedLearner

"""
from .. import ILearner


# noinspection PyAbstractClass
class IOnlineLearner(ILearner):
    """The online learner base class.

    The learning step is performed during the episode or iteration
    after each step.

    Parameters
    ----------
    filename : str, optional
        The name of the file to save the learner state to after each iteration.
        If None is given, the learner state is not saved. Default is None.

    """
    @property
    def type(self):
        """This learner is of type `online`.

        Returns
        -------
        str :
            The learner type

        """
        return 'online'

    def __init__(self, filename=None):
        super(IOnlineLearner, self).__init__(filename)

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
        super(IOnlineLearner, self).end()

    def learn(self, experience):
        """Learn a policy from the experience.

        Perform the learning step to derive a new policy taking the
        latest experience into account.

        Parameters
        ----------
        experience : Experience
            The agent's experience consisting of the previous state, the action performed
            in that state, the current state and the reward awarded.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError


from .rl import *

__all__ = [s for s in dir() if not (s.startswith('_') or s.endswith('cython'))]
