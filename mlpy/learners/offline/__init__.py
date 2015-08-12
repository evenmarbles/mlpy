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


from .irl import *

__all__ = [s for s in dir() if not (s.startswith('_') or s.endswith('cython'))]
