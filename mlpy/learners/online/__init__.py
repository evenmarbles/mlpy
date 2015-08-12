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

   ~rl.RLLearner
   ~rl.QLearner
   ~rl.RLDTLearner

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


from .rl import *

__all__ = [s for s in dir() if not (s.startswith('_') or s.endswith('cython'))]
