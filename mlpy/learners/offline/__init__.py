"""
Offline learners (:mod:`mlpy.learners.offline`)
==================================================

.. currentmodule:: mlpy.learners.offline

.. autosummary::
   :toctree: generated/
   :nosignatures:

   IOfflineLearner


Inverse reinforcement learning
------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~irl.ApprenticeshipLearner
   ~irl.IncrApprenticeshipLearner

"""
from .. import ILearner


# noinspection PyAbstractClass
class IOfflineLearner(ILearner):
    """The offline learner base class.

    In offline learning the learning step is performed at the end of
    the episode or iteration.

    Parameters
    ----------
    filename : str, optional
        The name of the file to save the learner state to after each iteration.
        If None is given, the learner state is not saved. Default is None.

    """
    @property
    def type(self):
        """This learner is of type `offline`.

        Returns
        -------
        str :
            The learner type

        """
        return 'offline'

    def __init__(self, filename=None):
        super(IOfflineLearner, self).__init__(filename)

    def end(self):
        """End the episode.

        Perform all end of episode tasks and save the state of the
        learner to file.

        """
        super(IOfflineLearner, self).end()

    def learn(self):
        """Learn a policy from the experience.

        Perform the learning step to derive a new policy taking the
        latest experience into account.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError


from .irl import *

__all__ = [s for s in dir() if not (s.startswith('_') or s.endswith('cython'))]
