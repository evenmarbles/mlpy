"""
Continuous Action and State Model Learner (CASML)
=================================================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   CASMLReuseMethod
   CASMLRevisionMethod
   CASMLRetentionMethod
   CASML

"""
from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from ...auxiliary.array import normalize
from ...auxiliary.plotting import Arrow3D
from ...knowledgerep.cbr.engine import CaseBase
from ...knowledgerep.cbr.methods import IReuseMethod, IRevisionMethod, IRetentionMethod
from ...stats.dbn.hmm import GaussianHMM
from ..stateaction import Experience, State
from .. import IMDPModel

__all__ = ['CASML']


class CASMLReuseMethod(IReuseMethod):
    """The reuse method implementation for :class:`CASML`.

    The solutions of the best (or set of best) retrieved cases are used to construct
    the solution for the query case; new generalizations and specializations may occur
    as a consequence of the solution transformation.

    The CASML reuse method further specializes the solution by identifying cases similar
    in both state and action.

    """

    # noinspection PyUnusedLocal
    def __init__(self):
        super(CASMLReuseMethod, self).__init__()

    # noinspection PyMethodMayBeStatic
    def execute(self, case, case_matches, fn_retrieve=None):
        """Execute reuse step.

        Take both similarity in state and in actions into account.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.
        fn_retrieve : callable
            Callback for accessing the case base's 'retrieval' method.

        Returns
        -------
        revised_matches : dict[int, CaseMatch]
            The revised solution.

        """
        cluster = []
        id_map = {}
        for i, m in enumerate(case_matches.itervalues()):
            cluster.append(m.case["act"])
            id_map[i] = m.case.id

        revised_matches = fn_retrieve(case, "act", False, **{"data": cluster, "id_map": id_map})
        for id_, m in revised_matches.items():
            m.set_similarity("state", case_matches[id_].get_similarity("state"))

        return revised_matches
    

class CASMLRevisionMethod(IRevisionMethod):
    """The revision method implementation for :class:`CASML`.

    The solutions provided by the query case is evaluated and information about whether the solution
    has or has not provided a desired outcome is gathered.

    Parameters
    ----------
    rho : float, optional
        The permitted error of the similarity measure. Default is 0.99.
    plot_revision : bool, optional
        Whether to plot the vision step or not. Default is False.
    plot_revision_params : {'origin_to_query', 'original_origin'}
        Parameters used for plotting. Valid parameters are:

            origin_to_query
                Which moves the origins of all actions to the query case's origin.

            original_origin
                The origins remain unchanged and the states are plotted at their
                original origins.

    Notes
    -----
    The CASML revision method further narrows down the solutions to the query case by identifying
    whether the actions are similar by ensuring that the actions are cosine similar within the
    permitted error :math:`\\rho`:

    .. math::

        d(c_{q.\\text{action}}, c.\\text{action}) >= \\rho

    """

    def __init__(self, rho=None, plot_revision=None, plot_revision_params=None):
        super(CASMLRevisionMethod, self).__init__()

        self._fig1, self._fig2, self._fig3 = None, None, None
        self._ax1, self._ax2, self._ax3 = None, None, None

        self._rho = rho if rho is not None else 0.99
        """:type: float"""

        self._plot = plot_revision if plot_revision is not None else False
        """:type: bool"""

        if plot_revision_params is not None:
            if plot_revision_params not in ["origin_to_query", "original_origin"]:
                raise ValueError(
                    "%s is not a valid plot parameter for revision method" % plot_revision_params)
            self._plot_params = plot_revision_params
        else:
            self._plot_params = "origin_to_query"

    def execute(self, case, case_matches, **kwargs):
        """Execute the revision step.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        Returns
        -------
        case_matches : dict[int, CaseMatch]
            the corrected solution.

        """

        for key, cm in case_matches.iteritems():
            if cm.get_similarity('act') >= self._rho:
                cm.is_solution = True

        if self._plot:
            self.plot_data(case, case_matches)

        return case_matches

    def plot_data(self, case, case_matches, **kwargs):
        """Plot the data during the revision step.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        """
        if self._fig1 is None or not plt.fignum_exists(self._fig1.number):
            self._fig1 = plt.figure()
            plt.rcParams['legend.fontsize'] = 10
            self._fig1.suptitle('Similarity: state')
            self._ax1 = self._fig1.add_subplot(1, 1, 1, projection='3d')
            self._fig1.show()

        # Plot axis 1
        self._ax1.cla()

        [x, y, z] = case["state"]
        self._ax1.scatter(x, y, z, edgecolors='r', c='r', marker='o')

        for cm in case_matches.itervalues():
            [xs, ys, zs] = cm.case["state"]
            if not cm.is_solution:
                self._ax1.scatter(xs, ys, zs, edgecolors='k', c='k', marker='o')
            else:
                self._ax1.scatter(xs, ys, zs, edgecolors='g', c='g', marker='o')

        scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='r', c='r', marker='o')
        scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='k', c='k', marker='o')
        scatter3_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='g', c='g', marker='o')
        self._ax1.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy],
                         ['query case', 'no solution', 'solution'], numpoints=1)

        self._ax1.set_xlabel('X position')
        self._ax1.set_ylabel('Y position')
        self._ax1.set_zlabel('Z position')
        self._ax1.set_title("States")

        self._fig1.canvas.draw()

        # Plot figure 2
        if self._fig2 is None or not plt.fignum_exists(self._fig2.number):
            self._fig2 = plt.figure()
            plt.rcParams['legend.fontsize'] = 10
            self._fig2.suptitle('Similarity: action')
            self._ax2 = self._fig2.add_subplot(1, 1, 1, projection='3d')
            self._fig2.show()

        self._ax2.cla()

        # add query case' action
        [x, y, z] = case["state"]
        [vx, vy, vz] = case["act"]
        a = Arrow3D([x, x + vx], [y, y + vy], [z, z + vz], mutation_scale=10, lw=1, arrowstyle="-|>", color='r')
        self._ax2.add_artist(a)
        self._ax2.scatter(x, y, z, edgecolors='r', c='r', marker='o')

        for cm in case_matches.itervalues():
            if self._plot_params == "original_origin":
                [x, y, z] = cm.case["state"]
                self._ax2.scatter(x, y, z, c='k', marker='o')

            [vx, vy, vz] = cm.case["act"].get()

            color = "k"
            if cm.is_solution:
                color = "g"
            a = Arrow3D([x, x + vx], [y, y + vy], [z, z + vz], 
                        mutation_scale=10, 
                        lw=1, 
                        arrowstyle="-|>", 
                        color=color)
            self._ax2.add_artist(a)

        proxies = []
        legend = []
        if self._plot_params == "original_origin":
            proxies.append(matplotlib.lines.Line2D([0], [0], 
                                                   linestyle="none", 
                                                   markeredgecolor='k', c='k', 
                                                   marker='o'))
            legend.append('case')

        scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='r', c='r', marker='o')
        scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="-", c='r')
        scatter3_proxy = matplotlib.lines.Line2D([0], [0], linestyle="-", c='k')
        scatter4_proxy = matplotlib.lines.Line2D([0], [0], linestyle="-", c='g')
        proxies.extend([scatter1_proxy, scatter2_proxy, scatter3_proxy, scatter4_proxy])
        legend.extend(['query case', 'query', 'not similar', 'similar'])

        self._ax2.legend(proxies, legend, numpoints=1)

        self._ax2.set_xlabel('X position')
        self._ax2.set_ylabel('Y position')
        self._ax2.set_zlabel('Z position')

        self._fig2.canvas.draw()

        # Plot figure 3
        if self._fig3 is None or not plt.fignum_exists(self._fig3.number):
            self._fig3 = plt.figure()
            plt.rcParams['legend.fontsize'] = 10
            self._fig3.suptitle('Similarity: action')
            self._ax3 = self._fig3.add_subplot(1, 1, 1, projection='3d')
            self._fig3.show()

        self._ax3.cla()

        if self._plot_params == "origin_to_query":
            [x, y, z] = [0, 0, 0]
            self._ax3.scatter(x, y, z, edgecolors='r', c='r', marker='o')

        for i, cm in enumerate(case_matches.itervalues()):
            [vx, vy, vz] = cm.case["delta_state"]

            color = 'k'
            if cm.is_solution:
                color = 'g'

            if self._plot_params == "original_origin":
                [x, y, z] = cm.case["state"]
                self._ax3.scatter(x, y, z, edgecolors=color, c=color, marker='o')

            a = Arrow3D([x, x + vx], [y, y + vy], [z, z + vz], 
                        mutation_scale=10, 
                        lw=1, 
                        arrowstyle="-|>", 
                        color=color)
            self._ax3.add_artist(a)

        scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='r', c='r', marker='o')
        scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="-", c='g')
        scatter3_proxy = matplotlib.lines.Line2D([0], [0], linestyle="-", c='k')
        self._ax3.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy],
                         ['query case', 'solution', 'not solution'], numpoints=1)

        self._ax3.set_xlabel('X position')
        self._ax3.set_ylabel('Y position')
        self._ax3.set_zlabel('Z position')
        self._ax3.set_title("Delta state")

        self._fig3.canvas.draw()


class CASMLRetentionMethod(IRetentionMethod):
    """The retention method implementation for :class:`CASML`.

    When the new problem-solving experience can be stored or not stored in memory,
    depending on the revision outcomes and the CBR policy regarding case retention.

    Parameters
    ----------
    tau : float, optional
        The maximum permitted error when comparing most similar solution.
        Default is 0.8.
    sigma : float, optional
        The maximum permitted error when comparing actual with estimated
        transitions. Default is 0.2
    plot_retention : bool, optional
        Whether to plot the data during the retention step or not. Default
        is False.

    Notes
    -----
    The CASML retention method considers query cases as predicted correctly if

    1. the query case is within the maximum permitted error :math:`\\tau` of
       the most similar solution case:

       .. math::

          d(\\text{case}, 1\\text{NN}(C_T, \\text{case})) < \\tau

    2. the difference between the actual and the estimated transitions are less
       than the permitted error :math:`\\sigma`:

       .. math::

          d(\\text{case}.\\Delta_\\text{state}, T(s_{i-1}, a_{i-1}) < \\sigma

    """

    def __init__(self, tau=None, sigma=None, plot_retention=None):
        super(CASMLRetentionMethod, self).__init__()

        self._fig = None
        self._ax1, self._ax2, self._ax3 = None, None, None

        self._tau = tau if tau is not None else 0.8
        """:type: float"""

        self._sigma = sigma if sigma is not None else 0.2
        """:type: float"""

        self._plot = plot_retention if plot_retention is not None else False
        """:type: bool"""

    def execute(self, case, case_matches, fn_add=None):
        """Execute the retention step.

        Parameters
        ----------
        case : Case
            The query case
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.
        fn_add : callable
            Callback for accessing the case base's 'add' method.

        """
        if self._plot:
            self.plot_data(case, case_matches)

        do_add = True
        for cm in case_matches.itervalues():
            if cm.is_solution:
                cm.error = np.linalg.norm(np.asarray(cm.case["delta_state"]) - np.asarray(case["delta_state"]))
                if cm.error <= self._sigma:
                    # At least one of the cases in the case base
                    # correctly estimated the query case, the query case
                    # does not add any new information, do not add.
                    cm.predicted = True
                    do_add = False

        # noinspection PyUnresolvedReferences
        do_add = do_add or case_matches[min(case_matches, key=lambda x: case_matches[x].get_similarity(
            "state") if case_matches[x].is_solution else np.inf)].get_similarity("state") > self._tau

        if do_add:
            fn_add(case)
        else:
            print("Case {0} was not added".format(case.id))

    def plot_data(self, case, case_matches, **kwargs):
        """Plot the data during the retention step.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        """
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._fig = plt.figure()
            plt.rcParams['legend.fontsize'] = 10
            self._fig.suptitle('Retention')
            self._ax1 = self._fig.add_subplot(1, 3, 1, projection='3d')
            self._ax2 = self._fig.add_subplot(1, 3, 2, projection='3d')
            self._ax3 = self._fig.add_subplot(1, 3, 3, projection='3d')
            self._fig.show()

        # Plot axis 1
        self._ax1.cla()

        [x, y, z] = case["state"]
        self._ax1.scatter(x, y, z, edgecolors='r', c='r', marker='o')

        for cm in case_matches.itervalues():
            [xs, ys, zs] = cm.case["state"]
            if not cm.is_solution:
                self._ax1.scatter(xs, ys, zs, edgecolors='k', c='k', marker='o')
            elif cm.get_similarity("state") > self._tau:
                self._ax1.scatter(xs, ys, zs, edgecolors='g', c='g', marker='^')
            else:
                self._ax1.scatter(xs, ys, zs, edgecolors='y', c='y', marker='v')

        scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='r', c='r', marker='o')
        scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='k', c='k', marker='o')
        scatter3_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='y', c='y', marker='v')
        scatter4_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='g', c='g', marker='^')
        self._ax1.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy, scatter4_proxy],
                         ['query case', 'non solution', 'less than tau', 'greater than tau'], numpoints=1)

        self._ax1.set_xlabel('X position')
        self._ax1.set_ylabel('Y position')
        self._ax1.set_zlabel('Z position')
        self._ax1.set_title("Tau")

        # Plot axis 2
        self._ax2.cla()

        [x, y, z] = [0, 0, 0]

        # add query case's action
        [vx, vy, vz] = case["act"]
        a = Arrow3D([x, x + vx], [y, y + vy], [z, z + vz], mutation_scale=10, lw=1, arrowstyle="-|>", color='r')
        self._ax2.add_artist(a)
        self._ax2.scatter(x, y, z, edgecolors='r', c='r', marker='o')

        # add solution case' action
        for cm in case_matches.itervalues():
            [vx, vy, vz] = cm.case["act"]

            color = "k"
            if cm.is_solution:
                color = "g"
            a = Arrow3D([x, x + vx], [y, y + vy], [z, z + vz], 
                        mutation_scale=10, 
                        lw=1, 
                        arrowstyle="-|>", 
                        color=color)
            self._ax2.add_artist(a)

        scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='r', c='r', marker='o')
        scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="-", c='r')
        scatter3_proxy = matplotlib.lines.Line2D([0], [0], linestyle="-", c='k')
        scatter4_proxy = matplotlib.lines.Line2D([0], [0], linestyle="-", c='g')
        self._ax2.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy, scatter4_proxy],
                         ['query case', 'query', 'not similar', 'similar'], numpoints=1)

        self._ax2.set_xlabel('X position')
        self._ax2.set_ylabel('Y position')
        self._ax2.set_zlabel('Z position')
        self._ax2.set_title("Action")

        # Plot axis 3
        self._ax3.cla()

        # add query case delta state
        [dx, dy, dz] = case["delta_state"]
        a = Arrow3D([x, dx], [y, dy], [z, dz], mutation_scale=10, lw=1, arrowstyle="-|>", color='r')
        self._ax3.add_artist(a)
        self._ax3.scatter(x, y, z, edgecolors='r', c='r', marker='o')

        # add solution cases' delta state
        error = np.zeros(len(case_matches))
        for i, cm in enumerate(case_matches.itervalues()):
            if cm.is_solution:
                [vx, vy, vz] = cm.case["delta_state"]
                error[i] = cm.error
                color = 'b'
                if cm.predicted:
                    color = 'g'
                a = Arrow3D([x, vx], [y, vy], [z, vz], mutation_scale=10, lw=1, arrowstyle="-|>", color=color)
                self._ax3.add_artist(a)

        scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='r', c='r', marker='o')
        scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="-", c='r')
        scatter3_proxy = matplotlib.lines.Line2D([0], [0], linestyle="-", c='g')
        scatter4_proxy = matplotlib.lines.Line2D([0], [0], linestyle="-", c='b')
        self._ax3.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy, scatter4_proxy],
                         ['query case', 'query delta', 'less than sigma', 'greater than sigma'], numpoints=1)

        self._ax3.set_xlabel('X position')
        self._ax3.set_ylabel('Y position')
        self._ax3.set_zlabel('Z position')
        self._ax3.set_title("Delta state: {0}".format(np.trim_zeros(error, 'b')))

        self._fig.canvas.draw()


# noinspection PyAbstractClass
class CASML(IMDPModel):
    """Continuous Action and State Model Learner (CASML).

    Parameters
    ----------
    case_template : dict
        The template from which to create a new case.

            :Example:

                An example template for a feature named `state` with the
                specified feature parameters. `data` is the data from which
                to extract the case from. In this example it is expected that
                `data` has a member variable `state`.

                ::

                    {
                        "state": {
                            "type": "float",
                            "value": "data.state",
                            "is_index": True,
                            "retrieval_method": "radius-n",
                            "retrieval_method_params": 0.01
                        },
                        "delta_state": {
                            "type": "float",
                            "value": "data.next_state - data.state",
                            "is_index": False,
                        }
                    }

    rho : float, optional
        The maximum permitted error when comparing cosine  similarity of
        actions. Default is 0.99.
    tau : float, optional
        The maximum permitted error when comparing most similar solution.
        Default is 0.8.
    sigma : float, optional
        The maximum permitted error when comparing actual with estimated
        transitions. Default is 0.2.
    ncomponents : int, optional
        Number of states of the hidden Markov model. Default is 1.
    revision_method_params : dict, optional
        Additional initialization parameters for :class:`CASMLRevisionMethod`.
    retention_method_params : dict, optional
        Additional initialization parameters for :class:`CASMLRetentionMethod`.
    case_base_params : dict, optional
        Initialization parameters for :class:`.CaseBase`.
    hmm_params : dict, optional
        Additional initialization parameters for :class:`.GaussianHMM`.
    proba_calc_method : str, optional
        The method used to calculate the probability distribution for the initial
        states. Default is DefaultProbaCalcMethod.

    """
            
    def __init__(self, case_template, rho=None, tau=None, sigma=None, ncomponents=1,
                 revision_method_params=None, retention_method_params=None, 
                 case_base_params=None, hmm_params=None, proba_calc_method=None):
        super(CASML, self).__init__(proba_calc_method)
        
        revision_method_params = revision_method_params if revision_method_params is not None else {}
        revision_method_params.update({'rho': rho})
        
        retention_method_params = retention_method_params if retention_method_params is not None else {}
        retention_method_params.update({'tau': tau, 'sigma': sigma})
        
        case_base_params = case_base_params if case_base_params is not None else {}

        #: The case base maintaining the observations in the form
        #:     c = <s, a, ds>, where ds = s_{i+1} - s_i
        #: in order to reason on the possible next states.
        self._cb_t = CaseBase(case_template,
                              reuse_method='CASMLReuseMethod',
                              revision_method='CASMLRevisionMethod', revision_method_params=revision_method_params,
                              retention_method='CASMLRetentionMethod', retention_method_params=retention_method_params,
                              **case_base_params)
        """:type: CaseBase"""

        hmm_params = hmm_params if hmm_params is not None else {}
        hmm_params.update({'ncomponents': ncomponents})
        #: The hidden Markov model maintaining the observations in the form
        #:     seq = <s_{i}, s_{i+1}>
        #: in order to reason on the probability distribution of the possible
        #: next states.
        self._hmm = GaussianHMM(**hmm_params)
        """:type: GaussianHMM"""

    # noinspection PyProtectedMember
    def fit(self, obs, actions, n_init=1, **kwargs):
        """Fit the :class:`.CaseBase` and the :class:`.HMM`.

        The model is fit to the observations and actions of the trajectory by
        updating the case base and the HMM.

        Parameters
        ----------
        obs : array_like, shape (`nfeatures`, `n`)
            Trajectory of observations, where each observation has `nfeatures`
            features and `n` is the length of the trajectory.
        actions : array_like, shape (`nfeatures`, `n`)
            Trajectory of actions, where each action has `nfeatures` features
            and `n` is the length of the trajectory.
        n_init : int, optional
            Number of restarts to prevent the HMM from getting stuck in a local
            minimum. Default is 1.

        """
        n = obs.shape[1]
        for i in range(n - 1):
            self._cb_t.run(self._cb_t.case_from_data(
                Experience(obs[:, i], actions[:, i], obs[:, i + 1])))

        # build initial state distribution
        self._initial_dist.add_state(State(obs[:, 0]))

        if self._hmm._fit_X is None:
            x = np.array([obs])
        else:
            x = np.vstack([self._hmm._fit_X, [obs]])
        self._hmm.fit(x, n_init=n_init)

    def predict_proba(self, state, action):
        """Predict the probability distribution.

        Predict the probability distribution for state transitions
        given a state and an action.

        Parameters
        ----------
        state : State
            The current state the robot is in.
        action : Action
            The action perform in state `state`.

        Returns
        -------
        dict[tuple[float]], float] :
            The probability distribution for the state-action pair.

        """
        case = self._cb_t.case_from_data(Experience(state, action, state))

        case_matches = self._cb_t.retrieve(case)
        revised_matches = self._cb_t.reuse(case, case_matches)
        solution = self._cb_t.revision(case, revised_matches)
        solution = [cm for cm in solution.itervalues() if cm.is_solution]
        # if not solution:
        #     self._cb_t.plot_retrieval(case, [cm.case.id for cm in case_matches.itervalues()], 'state')
        #     self._cb_t.plot_revision(case, solution)

        # calculate next states from current state and solution delta state
        current_state = case["state"]
        sequences = np.zeros((len(solution), 2, len(current_state)), dtype=float)

        for i, cm in enumerate(solution):
            if cm.is_solution:
                sequences[i, 0] = np.array(current_state)

                delta_state = cm.case["delta_state"]
                sequences[i, 1] = np.array(np.add(current_state, delta_state))

        # use HMM to calculate probability for observing sequence <current_state, next_state>
        # noinspection PyTypeChecker
        proba = normalize(np.exp(self._hmm.score(sequences)))
        return {State(s[1]): l for s, l in zip(sequences, proba)}
