"""
Continuous Action and State Model Learner (CASML)
=================================================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   CbTReuseMethod
   CbTRetentionMethod
   CbVRevisionMethod
   CbVRetentionMethod
   CbTData
   CbVData
   CASML

"""
from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import copy
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from ...auxiliary.array import normalize
from ...auxiliary.plotting import Arrow3D
from ...knowledgerep.cbr.engine import CaseBase
from ...knowledgerep.cbr.methods import IReuseMethod, IRevisionMethod, IRetentionMethod
from ...stats.dbn.hmm import GaussianHMM
from ..stateaction import Experience, MDPState, MDPStateData
from .. import IMDPModel

__all__ = ['CbTData', 'CbVData', 'CASML']


class CbTReuseMethod(IReuseMethod):
    """The reuse method for the transition case base implementation for :class:`CASML`.

    The solutions of the best (or set of best) retrieved cases are used to construct
    the solution for the query case; new generalizations and specializations may occur
    as a consequence of the solution transformation.

    The CASML reuse method for the transition case base further specializes the solution
    by identifying cases similar in both state and action.

    Parameters
    ----------
    owner : CaseBase
        A pointer to the owning case base.
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
    The CASML reuse method for the transition case base further narrows down the solutions to
    the query case by identifying whether the actions are similar by ensuring that the actions
    are cosine similar within the permitted error :math:`\\rho`:

    .. math::

        d(c_{q.\\text{action}}, c.\\text{action}) >= \\rho

    """

    def __init__(self, owner, rho=None, plot_reuse=None, plot_reuse_params=None):
        super(CbTReuseMethod, self).__init__(owner)

        self._fig1, self._fig2, self._fig3 = None, None, None
        self._ax1, self._ax2, self._ax3 = None, None, None

        self._rho = rho if rho is not None else 0.99
        """:type: float"""

        self._plot = plot_reuse if plot_reuse is not None else False
        """:type: bool"""
        if self._plot:
            self._plot_params = "origin_to_query"
            if plot_reuse_params is not None:
                if plot_reuse_params not in ["origin_to_query", "original_origin"]:
                    raise ValueError(
                        "%s is not a valid plot parameter for revision method" % plot_reuse_params)
                self._plot_params = plot_reuse_params

    # noinspection PyMethodMayBeStatic
    def execute(self, case, case_matches):
        """Execute reuse step.

        Take both similarity in state and in actions into account.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

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

        revised_matches = self._owner.retrieve(case, "act", False, **{"data": cluster, "id_map": id_map})
        for id_, m in revised_matches.items():
            m.set_similarity("state", case_matches[id_].get_similarity("state"))

        for key, cm in revised_matches.iteritems():
            if cm.get_similarity('act') >= self._rho:
                cm.is_solution = True

        if self._plot:
            self.plot_data(case, revised_matches)

        return revised_matches

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


class CbTRetentionMethod(IRetentionMethod):
    """The retention method for the transition case base implementation for :class:`CASML`.

    When the new problem-solving experience can be stored or not stored in memory,
    depending on the revision outcomes and the CBR policy regarding case retention.

    Parameters
    ----------
    owner : CaseBase
        A pointer to the owning case base.
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
    The CASML retention method for the transition case base considers query cases as
    predicted correctly if:

    1. the query case is within the maximum permitted error :math:`\\tau` of
       the most similar solution case:

       .. math::

          d(\\text{case}, 1\\text{NN}(C_T, \\text{case})) < \\tau

    2. the difference between the actual and the estimated transitions are less
       than the permitted error :math:`\\sigma`:

       .. math::

          d(\\text{case}.\\Delta_\\text{state}, T(s_{i-1}, a_{i-1}) < \\sigma

    """

    def __init__(self, owner, tau=None, sigma=None, plot_retention=None):
        super(CbTRetentionMethod, self).__init__(owner)

        self._fig = None
        self._ax1, self._ax2, self._ax3 = None, None, None

        self._tau = tau if tau is not None else 0.8
        """:type: float"""

        self._sigma = sigma if sigma is not None else 0.2
        """:type: float"""

        self._plot = plot_retention if plot_retention is not None else False
        """:type: bool"""

    def execute(self, case, case_matches):
        """Execute the retention step.

        Parameters
        ----------
        case : Case
            The query case
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

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
            self._owner.add(case)
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


class CbVRevisionMethod(IRevisionMethod):
    """The revision method for the value case base implementation for :class:`CASML`.

    The solutions provided by the query case is evaluated and information about whether the solution
    has or has not provided a desired outcome is gathered.

    Parameters
    ----------
    owner : CaseBase
        A pointer to the owning case base.

    Notes
    -----
    The CASML revision method for the value case base revises the state value :math:`v_k` associated
    with each nearest neighbor :math:`c_{V,k} \\in \\text{kNN}:(V, c_V)` to its contribution to the
    error in estimating the value of :math:`v_{i-1}`:

    .. math::

        v_k += \\alpha(r_{i-1} + \\gamma v_i - v_k) * \\frac{K(d(c_{V, i}, c_{V, k}))}{\\sum_{c_{V, j} \\in k\\text{NN}(V, c_v)} K(d(c_{V, j}, c_{V, j}))}

    where :math:`v_k` is the value associated with neighbor math:`c_{V, k}`, :math:`\\alpha (0 \\le \\alpha \\le 1)`
    is the learning rate, :math:`\\gamma (0 \\le \\gamma \\le 1)` is a geometric discount factor, and Gaussian kernel
    function :math:`K(d) = exp(-d^2)` determines the relative contribution of the k-nearest neighbors.

    """

    def __init__(self, owner):
        super(CbVRevisionMethod, self).__init__(owner)

    def execute(self, case, case_matches):
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
        return case_matches


class CbVRetentionMethod(IRetentionMethod):
    """The retention method for the value case base implementation for :class:`CASML`.

    When the new problem-solving experience can be stored or not stored in memory,
    depending on the revision outcomes and the CBR policy regarding case retention.

    Parameters
    ----------
    owner : CaseBase
        A pointer to the owning case base.
    tau : float, optional
        The maximum permitted error when comparing most similar solution.
        Default is 0.05.

    Notes
    -----
    The CASML retention method for the value case base considers query cases as predicted correctly if
    the query case is within the maximum permitted error :math:`\\tau` of
    the most similar solution case:

    .. math::

       d(\\text{case}, 1\\text{NN}(C_V, \\text{case})) < \\tau

    """

    def __init__(self, owner, tau=None):
        super(CbVRetentionMethod, self).__init__(owner)
        self._tau = tau if tau is not None else 0.05
        """:type: float"""

    def execute(self, case, case_matches):
        """Execute the retention step.

        Parameters
        ----------
        case : Case
            The query case
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        """
        if not case_matches or case_matches[min(case_matches, key=lambda x: case_matches[x].get_similarity(
                "state") if case_matches[x].is_solution else np.inf)].get_similarity("state") > self._tau:
            self._owner.add(case)
        else:
            print("Case {0} was not added".format(case.id))


class CbTData(object):
    """Transition case base data.

    Parameters
    ----------
    case_template : dict
        The template from which to create a new case for the **transition case base**.

            :Example:

                An example template for a feature named `state` with the
                specified feature parameters and a feature named `state`.
                `data` is the data from which to extract the case from.
                In this example it is expected that `data` has a member
                variable `state`.

                ::

                    {
                        "state": {
                            "type": "float",
                            "value": "data.state",
                            "is_index": True,
                            "retrieval_method": "radius-n",
                            "retrieval_method_params": 0.01
                        },
                        "act": {
                            "type": "float",
                            "value": "data.action",
                            "is_index": False,
                            "retrieval_method": "cosine",
                        },
                        "delta_state": {
                            "type": "float",
                            "value": "data.next_state - data.state",
                            "is_index": False,
                        }
                    }

    rho : float, optional
        The maximum permitted error when comparing cosine similarity of
        actions in the transition case base. Default is 0.99.
    tau : float, optional
        The maximum permitted error when comparing most similar solutions in
        the transition case base.
        Default is 0.8.
    sigma : float, optional
        The maximum permitted error when comparing actual with estimated
        transitions. Default is 0.2.

    Additional Parameters
    ---------------------
    plot_retrieval : bool, optional
        Whether to plot the result of the retrieval method. Default is False.
    plot_retrieval_names : str or list[str], optional
        The names of the feature which to plot.
    plot_reuse : bool, optional
        Whether to plot the result of the reuse method. Default is False.
    plot_reuse_params : {"origin_to_query", "original_origin"}, optional
        The location of the origin. Default is "origin_to_query"
    plot_retention : bool, optional
        Whether to plot the results of the retention method. Default is False.

    """

    def __init__(self, case_template, rho=None, tau=None, sigma=None, **kwargs):
        self.case_template = case_template

        self.reuse_method = 'CbTReuseMethod'
        self.reuse_method_params = {'rho': rho}
        for k, v in kwargs.iteritems():
            if k in ["plot_reuse", "plot_reuse_params"]:
                self.reuse_method_params[k] = kwargs.pop(k)

        self.retention_method = 'CbTRetentionMethod'
        self.retention_method_params = {'tau': tau, 'sigma': sigma}
        for k, v in kwargs.iteritems():
            if k in ['plot_retention']:
                self.retention_method_params[k] = kwargs.pop(k)

        for key, value in kwargs.iteritems():
            setattr(self, key, value)


class CbVData(object):
    """Value case base data.

    Parameters
    ----------
    case_template : dict
        The template from which to create a new case for the **transition case base**.

            :Example:

                An example template for a feature named `state` with the
                specified feature parameters and a feature named `state`.
                `data` is the data from which to extract the case from.
                In this example it is expected that `data` has a member
                variable `state`.

                ::

                    {
                        "state": {
                            "type": "float",
                            "value": "data.state",
                            "is_index": True,
                            "retrieval_method": "radius-n",
                            "retrieval_method_params": 0.01
                        },
                        "value": {
                            "type": "float",
                            "value": "data.action",
                            "is_index": False,
                            "retrieval_method": "cosine",
                        },
                        "delta_state": {
                            "type": "float",
                            "value": "data.next_state - data.state",
                            "is_index": False,
                        }
                    }

    tau : float, optional
        The maximum permitted error when comparing most similar solutions in
        the transition case base.
        Default is 0.8.

    """

    def __init__(self, case_template, tau=None, **kwargs):
        self.case_template = case_template

        self.reuse_method = 'CbVReuseMethod'

        self.revision_method = 'CbVRevisionMethod'

        self.retention_method = 'CbVRetentionMethod'
        self.retention_method_params = {'tau': tau}

        for key, value in kwargs.iteritems():
            setattr(self, key, value)


# noinspection PyAbstractClass
class CASML(IMDPModel):
    """Continuous Action and State Model Learner (CASML).

    Parameters
    ----------
    cbtdata : CbTData
        The transition case base data.
    cbvdata : CbVData
        The value case base data.
    ncomponents : int, optional
        Number of states of the hidden Markov model. Default is 1.
    proba_calc_method : str, optional
        The method used to calculate the probability distribution for the initial
        states. Default is DefaultProbaCalcMethod.

    Additional Parameters
    ---------------------
    startprob_prior : array, shape (`ncomponents`,)
        Initial state occupation prior distribution.
    startprob : array, shape (`ncomponents`,)
        Initial state occupation distribution.
    transmat_prior : array, shape (`ncomponents`, `ncomponents`)
        Matrix of prior transition probabilities between states.
    transmat : array, shape (`ncomponents`, `ncomponents`)
        Matrix of transition probabilities between states.
    emission_prior : normal_invwishart
        Initial emission parameters, a normal-inverse Wishart distribution.
    emission : conditional_normal_frozen
        The conditional probability distribution used for the emission.
    n_iter : int
        Number of iterations to perform during training, optional.
    thresh : float
        Convergence threshold, optional.
    verbose : bool
        Controls if debug information is printed to the console, optional.

    """
    @property
    def statespace(self):
        return self._statespace

    @property
    def transition_cases(self):
        return self._cb_t.cases

    @property
    def value_cases(self):
        return self._cb_v.cases

    def __init__(self, cbtdata, cbvdata=None, ncomponents=1, proba_calc_method=None, n_init=1, actions=None, **kwargs):
        super(CASML, self).__init__(proba_calc_method)

        self._nstates = 0
        self._statespace = {}
        """:type: dict[MDPState, MDPStateData]"""
        self._actions = actions
        """:type: dict[MDPState, list[MDPAction]] | list[MDPAction]"""
        self._last_state = None
        """:type: MDPState"""
        self._last_action = None
        """:type: MDPAction"""

        self._n_init = n_init

        #: The case base maintaining the observations in the form
        #:     c = <s, a, ds>, where ds = s_{i+1} - s_i
        #: in order to reason on the possible next states.
        self._cb_t = CaseBase(**cbtdata.__dict__)

        self._cb_v = None
        if cbvdata is not None:
            #: The case base maintaining the observations in the form
            #:     c = <s, v>
            #: in order to reason on the next best action.
            self._cb_v = CaseBase(**cbvdata.__dict__)

        hmm_param_names = ['startprob_prior', 'startprob', 'transmat_prior', 'transmat',
                           'emission_prior', 'emission', 'niter', 'thresh', 'verbose']
        hmm_params = {k: v for k, v in kwargs.iteritems() if k in hmm_param_names}
        # hmm_params = hmm_params if hmm_params is not None else {}
        hmm_params.update({'ncomponents': ncomponents})
        #: The hidden Markov model maintaining the observations in the form
        #:     seq = <s_{i}, s_{i+1}>
        #: in order to reason on the probability distribution of the possible
        #: next states.
        self._hmm = GaussianHMM(**hmm_params)
        """:type: GaussianHMM"""

    # noinspection PyProtectedMember
    def fit(self, obs, actions, rewards=None, n_init=1):
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

            if self._cb_v is not None:
                self._cb_v.run(self._cb_v.case_from_data(
                    Experience(obs[:, i], actions[:, i], obs[:, i + 1], rewards[:, i])))

        # build initial state distribution
        self._initial_dist.add_state(MDPState(obs[:, 0]))

        if self._hmm._fit_X is None:
            x = np.array([obs])
        else:
            # self._hmm.startprob = None
            # self._hmm.transmat = None
            # self._hmm.emission = None
            x = np.vstack([self._hmm._fit_X, [obs]])
        self._hmm.fit(x, n_init=n_init)

    # noinspection PyProtectedMember
    def update(self, experience):
        """Update the model with the agent's experience.

        Parameters
        ----------
        experience : Experience
            The agent's experience, consisting of state, action, next state(, and reward).

        Returns
        -------
        bool :
            Return True if the model has changed, False otherwise.

        """
        if experience.state is None:
            self._initial_dist.add_state(experience.next_state)
            self._last_state = copy.deepcopy(experience.next_state)
            self._last_action = copy.deepcopy(experience.action)
            return False

        self._cb_t.run(self._cb_t.case_from_data(experience))

        if self._cb_v is not None:
            self._cb_v.run(self._cb_v.case_from_data(experience))

        if self._hmm._fit_X is None:
            x = np.array([np.concatenate((np.reshape(experience.state, (-1, experience.state.nfeatures)).T,
                          np.reshape(experience.next_state, (-1, experience.next_state.nfeatures)).T), axis=1)])
        else:
            # self._hmm.startprob = None
            # self._hmm.transmat = None
            # self._hmm.emission = None
            x = np.array([np.hstack([self._hmm._fit_X[0], np.reshape(experience.state, (-1, experience.state.nfeatures)).T])])
        self._hmm.fit(x, n_init=self._n_init)

        self.add_state(experience.state)
        self.add_state(experience.next_state)

        for state in self._statespace.keys():
            info = self._statespace[state]

            for act, model in info.models.iteritems():
                model.transition_proba.clear()
                for next_state, prob in self.predict_proba(state, act).iteritems():
                    if next_state not in self._statespace:
                        if next_state.is_valid():
                            self._logger.debug("Unknown state {0} in transitioning model".format(next_state))
                            self.add_state(next_state)
                        else:
                            next_state = copy.deepcopy(state)
                    model.transition_proba.iadd(next_state, prob)

                model.reward_func.set(experience.reward)

        # self._statespace = {}
        # for i, c in self._cb_t.cases.iteritems():
        #     case_matches = self._cb_t.retrieve(c, 'state', False)
        #
        #     actions = []
        #     for m in case_matches.itervalues():
        #         actions.append(m.case["act"])
        #
        #     state = MDPState(c['state'])
        #     self.add_state(i, state, actions)
        #
        #     info = self._statespace[state]
        #     for act, model in info.models.iteritems():
        #         model.transition_proba[state] = self.predict_proba(state, act)

        self._last_state = copy.deepcopy(experience.next_state)
        self._last_action = copy.deepcopy(experience.action)
        return True

    def add_state(self, state):
        if state is not None and state not in self._statespace:
            self._nstates += 1
            self._statespace[state] = MDPStateData(self._nstates, self._actions)
            return True

        return False

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
        #     self._cb_T.plot_retrieval(case, [cm.case.id for cm in case_matches.itervalues()], 'state')
        #     self._cb_T.plot_revision(case, solution)

        # calculate next states from current state and solution delta state
        current_state = case["state"]
        sequences = np.zeros((len(solution), 2, len(current_state)), dtype=float)

        for i, cm in enumerate(solution):
            if cm.is_solution:
                sequences[i, 0] = np.array(current_state)

                delta_state = cm.case["delta_state"]
                sequences[i, 1] = np.asarray(current_state + delta_state)

        # use HMM to calculate probability for observing sequence <current_state, next_state>
        # noinspection PyTypeChecker
        proba = normalize(np.exp(self._hmm.score(sequences)))
        return {MDPState(s[1]): l for s, l in zip(sequences, proba)}
