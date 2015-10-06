"""
=============================================
Learning algorithms (:mod:`mlpy.learners`)
=============================================

.. currentmodule:: mlpy.learners


.. autosummary::
   :toctree: generated/
   :nosignatures:

   LearnerFactory
   ILearner


.. automodule:: mlpy.learners.online
   :noindex:


.. automodule:: mlpy.learners.offline
   :noindex:

"""
from __future__ import division, print_function, absolute_import

from abc import abstractmethod, abstractproperty
from ..modules.patterns import RegistryInterface
from ..modules import UniqueModule
from ..tools.log import LoggingMgr


class LearnerFactory(object):
    """The learner factory.

    An instance of a learner can be created by passing the learner type.

    Examples
    --------
    >>> from mlpy.learners import LearnerFactory
    >>> q0 = LearnerFactory.create('qlearner')

    This creates a :class:`.QLearner` instance with default parameters.

    >>> q1 = LearnerFactory.create('qlearner', max_steps=10)

    This creates a :class:`.QLearner` instance with max_steps set to 10.

    """

    @staticmethod
    def create(_type, *args, **kwargs):
        """
        Create an learner of the given type.

        A new learner of the given type is created. If `progress` is
        among the keywords in `kwargs`, the factory attempts to recover
        the learner from the learner state saved to file `filename`. If
        the factory fails to load the learners state from file, a new
        learner is created.

        Parameters
        ----------
        _type : str
            The learner type. Valid learner types:

            qlearner
                Performs q-learning, a reinforcement learning variant. A :class:`.QLearner`
                module is created.

            modelbasedlearner
                The model based learner performs reinforcement learning using the specified model
                and planner. A :class:`.ModelBasedLearner` module is created.

            apprenticeshiplearner
                The learner performs apprenticeship learning via inverse reinforcement
                learning, a method introduced by Abbeel and Ng which strives to imitate
                the demonstrations given by an expert. A :class:`.ApprenticeshipLearner`
                module is create.

            incrapprenticeshiplearner
                The learner incrementally performs apprenticeship learning via inverse
                reinforcement learning. Inverse reinforcement learning assumes knowledge
                of the underlying model. However, this is not always feasible. The
                incremental apprenticeship learner updates its model after every iteration
                by executing the current policy. A :class:`.IncrApprenticeshipLearner` module
                is create.

        args : tuple, optional
            Positional arguments passed to the class of the given type for
            initialization.
        kwargs : dict, optional
            Non-positional arguments passed to the class of the given type
            for initialization.

        Returns
        -------
        ILearner :
            A learner instance of the given type.

        """
        loaded = False
        # noinspection PyUnresolvedReferences
        learner = ILearner.registry[_type.lower()]

        if 'progress' in kwargs:
            if kwargs['progress']:
                try:
                    learner = learner.load(kwargs['filename'])
                    loaded = True
                except IOError:
                    pass
                except KeyError:
                    import sys
                    import traceback

                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback)
                    sys.exit(1)

            del kwargs['progress']

        if not loaded:
            learner = learner(*args, **kwargs)

        return learner


# noinspection PyMethodMayBeStatic
class ILearner(UniqueModule):
    """
    The learner interface.

    Both online and offline learner inherit from this interface.

    Parameters
    ----------
    filename : str, optional
        The name of the file to save the learner state to after each iteration.
        If None is given, the learner state is not saved. Default is None.

    """
    __metaclass__ = RegistryInterface

    @abstractproperty
    def type(self):
        """The type of the learner (i.e., `online` and `offline`).

        During online learning the learning is performed during the
        episode or iteration, while offline learner do not perform the
        learning step until the end of the episode or iteration.

        This property must be overwritten by its deriving class.

        Returns
        -------
        str :
            The type. Values can be either `online` or `offline`.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    def __init__(self, filename=None):
        super(ILearner, self).__init__()
        self._logger = LoggingMgr().get_logger(self._mid)

        self._filename = filename

    def __getstate__(self):
        data = super(ILearner, self).__getstate__()
        del data['_logger']
        return data

    def __setstate__(self, d):
        super(ILearner, self).__setstate__(d)
        self._logger = LoggingMgr().get_logger(self._mid)

    def init(self):
        """Initialize the learner."""
        pass

    def start(self):
        """Start an episode."""
        pass

    def step(self, experience):
        """Execute learning specific updates.

        Learning specific updates are performed, e.g. model updates.

        Parameters
        ----------
        experience : Experience
            The actor's current experience consisting of previous state, the action
            performed in that state, the current state, and the reward awarded.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    def end(self, *args, **kwargs):
        """End the episode.

        Perform all end of episode tasks and save the state of the
        learner to file.

        Parameters
        ----------
        args : tuple
            Positional arguments specific to the implementation of
            the learner
        kwargs : dict
            Non-positional arguments specific to the implementation of
            the learner

        """
        if self._filename is not None:
            self.save(self._filename)

    @abstractmethod
    def learn(self, *args, **kwargs):
        """Learn a policy from the experience.

        Perform the learning step to derive a new policy taking the
        latest experience into account.

        Parameters
        ----------
        args : tuple
            Positional arguments specific to the implementation of
            the learner
        kwargs : dict
            Non-positional arguments specific to the implementation of
            the learner

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    def choose_action(self, state):
        """Choose the next action

        The next action is chosen according to the current policy and the
        selected exploration strategy.

        Parameters
        ----------
        state : MDPState
            The current state.

        Returns
        -------
        MDPAction :
            The chosen action.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError


from .online import *
from .offline import *

__all__ = [s for s in dir() if not (s.startswith('_') or s.endswith('cython'))]
