"""
==================================================
Markov decision process (MDP) (:mod:`mlpy.mdp`)
==================================================

.. currentmodule:: mlpy.mdp


Transition and reward models
============================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~MDPModelFactory
   ~IMDPModel


Discrete models
---------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~discrete.DiscreteModel
   ~discrete.DecisionTreeModel


Model explorer
++++++++++++++

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~discrete.ExplorerFactory
   ~discrete.RMaxExplorer
   ~discrete.LeastVisitedBonusExplorer
   ~discrete.UnknownBonusExplorer


Contiguous models
-----------------

.. autosummary::
   :toctree: generated/

   ~continuous.casml


Probability distributions
=========================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~distrib.ProbaCalcMethodFactory
   ~distrib.IProbaCalcMethod
   ~distrib.DefaultProbaCalcMethod
   ~distrib.ProbabilityDistribution


State and action information
============================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~stateaction.Experience
   ~stateaction.RewardFunction
   ~stateaction.MDPStateActionInfo
   ~stateaction.MDPStateData
   ~stateaction.MDPPrimitive
   ~stateaction.MDPState
   ~stateaction.MDPAction

"""
from __future__ import division, print_function, absolute_import

from abc import abstractmethod
import numpy as np

from ..tools.log import LoggingMgr
from ..modules import UniqueModule
from ..modules.patterns import RegistryInterface
from .distrib import ProbabilityDistribution


class MDPModelFactory(object):
    """The Markov decision process (MDP) model factory.

    An instance of an MDP model can be created by passing
    the MDP model type.

    Examples
    --------
    >>> from mlpy.mdp import MDPModelFactory
    >>> MDPModelFactory.create('discretemodel')

    This creates a :class:`.DiscreteModel` instance with default settings.

    >>> MDPModelFactory.create('decisiontreemodel', explorer_type='leastvisitedbonusexplorer',
    ...                        explorer_params={'rmax': 1.0})

    This creates a :class:`.DecisionTreeModel` instance using :class:`.LeastVisitedBonusExplorer`
    with `rmax` set to 1.0.

    Notes
    -----

    """
    @staticmethod
    def create(_type, *args, **kwargs):
        """Create an MDP model of the given type.

        Parameters
        ----------
        _type : str
            The MDP model type. Valid model types:

            discretemodel
                A model for discrete state and actions deriving transition
                and reward information from empirical data. A :class:`.DiscreteModel`
                instance is created.

            decisiontreemodel
                A model for discrete state and actions deriving transition
                and reward information from empirical data generalized using
                decision trees. A :class:`.DecisionTreeModel` instance model is
                created.

            casml
                A model for continuous state and actions deriving transition
                information from empirical data fit to a case base and a Hidden
                Markov Model (:class:`.HMM`). Rewards are derived from empirical
                data. A :class:`.CASML` instance is created.

        args : tuple, optional
            Positional arguments to pass to the class of the given type for
            initialization.
        kwargs : dict, optional
            Non-positional arguments to pass to the class of the given type
            for initialization.

        Returns
        -------
        IMDPModel :
            A MDP model instance of the given type.

        """
        # noinspection PyUnresolvedReferences
        return IMDPModel.registry[_type.lower()](*args, **kwargs)


class IMDPModel(UniqueModule):
    """The Markov decision process interface.

    All Markov decision process (MDP) models are derived from
    the base class. The base class maintains an initial probability distribution
    from which the initial state can be sampled.

    Parameters
    ----------
    proba_calc_method : str
        The method used to calculate the probability
        distribution for the initial state. Defaults to DefaultProbaCalcMethod.

    """
    __metaclass__ = RegistryInterface

    def __init__(self, proba_calc_method=None):
        super(IMDPModel, self).__init__()
        self._logger = LoggingMgr().get_logger(self._mid)

        self._initial_dist = ProbabilityDistribution(proba_calc_method)
        """:type: ProbabilityDistribution"""

    def __getstate__(self):
        data = super(IMDPModel, self).__getstate__()
        del data['_logger']
        return data

    def __setstate__(self, d):
        super(IMDPModel, self).__setstate__(d)
        self._logger = LoggingMgr().get_logger(self._mid)

    def init(self):
        """Initialize the MDP model."""
        pass

    @abstractmethod
    def fit(self, obs, actions, **kwargs):
        """Fit the model to the observations and actions of the trajectory.

        Parameters
        ----------
        obs : array_like, shape (`nfeatures`, `n`)
            Trajectory of observations, where each observation has `nfeatures`
            features and `n` is the length of the trajectory.
        actions : array_like, shape (`nfeatures`, `n`)
            Trajectory of actions, where each action has `nfeatures` features
            and `n` is the length of the trajectory.
        kwargs: dict, optional
            Non-positional parameters, optional

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        Notes
        -----
        This is an abstract method and *must* be implemented by its deriving class.

        """
        raise NotImplementedError

    def update(self, experience):
        """Update the model with the agent's experience.

        Parameters
        ----------
        experience : Experience
            The agent's experience, consisting of state,
            action, next state(, and reward).

        Returns
        -------
        bool :
            Return True if the model has changed, False otherwise.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        Notes
        -----
        Optionally this method can be overwritten if the model supports
        incrementally updating the model.

        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, state, action):
        """Predict the probability distribution.

        The probability distribution for state transitions is predicted
        for the given state and an action.

        Parameters
        ----------
        state : MDPState
            The current state the robot is in.
        action : MDPAction
            The action perform in state `state`.

        Returns
        -------
        dict[tuple[float], float] :
            The probability distribution for the state-action pair.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        Notes
        -----
        This is an abstract method and *must* be implemented by its deriving class.

        """
        raise NotImplementedError

    def sample(self, state=None, action=None):
        """Sample from the probability distribution.

        The next state is sampled for the given state and action from the probability
        distribution. If either state or action is ``None`` the next state is
        sampled from the initial distribution.

        Parameters
        ----------
        state : MDPState, optional
            The current state the robot is in.
        action : MDPAction, optional
            The action perform in state `state`.

        Returns
        -------
        MDPState :
            The sampled next state.

        """
        if state is None or action is None:
            return self._initial_dist.sample()

        transition_proba = self.predict_proba(state, action)
        if not transition_proba:
            # a state is reached for which no empirical transition data exists
            self._logger.debug("MDPState {0} is unknown and has no transition probabilities".format(state))
            return None

        next_state = None
        if transition_proba.keys():
            proba = np.array(transition_proba.values())
            if not sum(proba) == 1:
                assert ((proba.sum() - 1) <= np.finfo(np.float16).eps), "Probabilities do not sum to 1"
                proba /= proba.sum()

            idx = np.random.choice(range(len(proba)), p=proba)
            next_state = transition_proba.keys()[idx]

        return next_state


from .discrete import *
from .continuous import *

__all__ = [s for s in dir() if not (s.startswith('_') or s.endswith('cython'))]
__all__ += ['distrib', 'stateaction']
