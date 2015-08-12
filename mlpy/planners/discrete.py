from __future__ import division, print_function, absolute_import

import math
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from . import IPlanner
from .explorers.discrete import DiscreteExplorer
from ..tools.log import LoggingMgr
from ..tools.misc import Waiting


class ValueIteration(IPlanner):
    """ Planning through value Iteration.

    Parameters
    ----------
    model : DiscreteModel
        The Markov decision model.
    explorer : Explorer, optional
        The exploration strategy to employ. Available explorers are:

        :class:`.EGreedyExplorer`
            With :math:`\\epsilon` probability, a random action is
            chosen, otherwise the action resulting in the highest
            q-value is selected.

        :class:`.SoftmaxExplorer`
            The softmax explorer varies the action probability as a
            graded function of estimated value. The greedy action is
            still given the highest selection probability, but all the others
            are ranked and weighted according to their value estimates.

        By default no explorer is used and the greedy action is chosen.
    gamma : float, optional
        The discount factor. Default is 0.9.
    ignore_unreachable : bool, optional
        Whether to ignore unreachable states or not. Unreachability is determined
        by how many steps a state is are away from the closest neighboring state.
        Default is False.

    Raises
    ------
    AttributeError
        If both the Markov model and the planner define an explorer.
        Only one explorer can be specified.

    """
    MAX_STEPS = 100

    @property
    def model(self):
        """ The Markov decision process model.

        The Markov decision process model contain information about
        the states, actions, and their transitions and the reward
        function.

        Returns
        -------
        IMDPModel :
            The model.

        """
        return self._model

    def __init__(self, model, explorer=None, gamma=None, ignore_unreachable=False):
        super(ValueIteration, self).__init__(explorer)
        self._logger = LoggingMgr().get_logger(self._mid)

        self._plot_num = 0

        self._model = model
        """:type: IMDPModel"""

        if self._explorer is not None:
            variable = getattr(self._model, "_explorer", None)
            if variable is not None:
                raise AttributeError("There can be only one explorer. Either based on the model or of the planner.")

        if self._explorer is None:
            self._explorer = DiscreteExplorer()

        self._gamma = 0.9 if gamma is None else gamma
        self._ignore_unreachable = ignore_unreachable if ignore_unreachable is not None else False

    def __getstate__(self):
        data = super(ValueIteration, self).__getstate__()
        data.update({
            '_model': self._model,
            '_explorer': self._explorer,
            '_gamma': self._gamma,
            '_ignore_unreachable': self._ignore_unreachable,
            '_plot_num': self._plot_num
        })
        return data

    def __setstate__(self, d):
        super(ValueIteration, self).__setstate__(d)

        for name, value in d.iteritems():
            setattr(self, name, value)

        self._logger = LoggingMgr().get_logger(self._mid)

    def activate_exploration(self):
        """Turn the explorer on."""
        super(ValueIteration, self).activate_exploration()

        func = getattr(self._model, "activate_exploration", None)
        if callable(func):
            func()

    def deactivate_exploration(self):
        """Turn the explorer off."""
        super(ValueIteration, self).deactivate_exploration()

        func = getattr(self._model, "deactivate_exploration", None)
        if callable(func):
            func()

    def get_best_action(self, state):
        """Choose the best next action for the agent to take.

        Parameters
        ----------
        state : State
            The state for which to choose the action for.

        Returns
        -------
        Action :
            The best action.

        """
        self._model.add_state(state)

        actions = self._model.get_actions(state)
        info = self._model.statespace[state]

        action = self._explorer.choose_action(actions, [info.q[a] for a in actions])
        self._logger.debug("state=%s\tact=%s\tvalue=%.2f", state, action, self._model.statespace[state].q[action])

        return action

    def plan(self):
        """Plan for the optimal policy.

        Perform value iteration and build the Q-table.

        """
        if self._ignore_unreachable:
            self._calculate_reachable_states()

        nloops = 0
        max_error = 5000
        min_error = 0.1

        states_updated = 0

        waiting = None
        if self._logger.level > LoggingMgr.LOG_DEBUG:
            waiting = Waiting("Perform value iteration")
            waiting.start()

        s0 = datetime.now()

        while max_error > min_error:
            self._logger.debug("max error: %0.5f nloops: %d", max_error, nloops)

            max_error = 0
            nloops += 1

            for state in self._model.statespace.keys():
                info = self._model.statespace[state]
                self._logger.debug("\tState: id: %d: %s, Steps: %d", info.id, state, info.steps_away)

                states_updated += 1

                if self._ignore_unreachable and info.steps_away > 99999:
                    self._logger.debug("\tState not reachable, ignoring")
                    continue

                for action, mdl in info.models.iteritems():
                    newq = mdl.reward_func.get(state)

                    for state2, prob in mdl.transition_proba:
                        self._logger.debug("\t\tNext state is: %s, prob: %.2f", state2, prob)

                        real_state = state2.is_valid()

                        next_state = state2
                        if not real_state:
                            next_state = state
                        elif self._ignore_unreachable and info.steps_away >= ValueIteration.MAX_STEPS:
                            next_state = state
                        else:
                            self._model.add_state(next_state)

                        info2 = self._model.statespace[next_state]

                        next_steps = info.steps_away + 1
                        if next_steps < info2.steps_away:
                            info2.steps_away = next_steps

                        maxq = max([info2.q[a] for a in self._model.get_actions(state2)])
                        newq += self._gamma * prob * maxq

                    tderror = math.fabs(info.q[action] - newq)
                    info.q[action] = newq

                    if tderror > max_error:
                        max_error = tderror

                    self._logger.debug("\t\tTD error: %.5f Max error: %.5f", tderror, max_error)

        s1 = datetime.now()
        delta = s1 - s0

        if waiting is not None:
            waiting.stop()

        self._logger.info("\tvalues computed with maxError: %.5f nloops: %d time: %d:%d states: %d", max_error, nloops,
                          delta.seconds, delta.microseconds, states_updated)

        self._remove_unreachable_states()

    # noinspection PyShadowingNames
    def visualize(self):
        """Visualize of the planning data.

        The results in the Q table are visualized via a heat map.

        """
        nrows = 30
        actions = self._model.get_actions()
        ncols = len(actions)

        num_states = len(self._model.statespace)
        data = np.zeros((num_states, len(actions)))

        ylabels = [None] * num_states
        for state, info in self._model.statespace.iteritems():
            ylabels[info.id - 1] = state        # TODO: check if that is correct: .encode()
            for i, act in enumerate(actions):
                data[info.id - 1][i] = info.q[act]

        decorated = [(i, tup[0], tup) for i, tup in enumerate(ylabels)]
        decorated.sort(key=lambda tup: tup[1])
        ylabels = [tup for i, second, tup in decorated]
        indices = [i for i, second, tup in decorated]
        data = np.array([data[i] for i in indices])

        self._logger.debug("Q-table data".format(data[::-1]))

        h, w = data.shape
        nsubplots = int(math.ceil(h / float(nrows)))
        diff = (nsubplots * nrows) - h
        # noinspection PyTypeChecker
        data = np.lib.pad(data, ((0, diff), (0, 0)), 'constant', constant_values=0)
        # noinspection PyTypeChecker
        ylabels.extend([""] * diff)
        h, w = data.shape

        # noinspection PyArgumentList,PyTypeChecker
        sdata = (data.reshape(h // nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

        dt = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        with PdfPages("savedata/figures/plot {0}.pdf".format(dt)) as pdf:
            fig, axes = plt.subplots(1, nsubplots, figsize=(10, 7), tight_layout=True)

            if nsubplots > 1:
                for i, ax in enumerate(axes.flat):
                    self._add_subplot(fig, ax, sdata[i], ylabels[i * nrows:i * nrows + nrows])
            else:
                self._add_subplot(fig, axes, sdata[0], ylabels[0:nrows])

            fig.subplots_adjust(right=1.2, top=0.2)
            fig.suptitle("Plot #{0}".format(self._plot_num + 1), fontsize=10)
            self._plot_num += 1
            pdf.savefig()
            plt.close()

    def _create_policy(self, func=None):
        """Creates a policy (i.e., a state-action association).

        Parameters
        ----------
        func : callable, optional
            A callback function for mixing policies.

        """
        policy = {}
        # noinspection PyUnresolvedReferences
        if func and self._history and len(self._history.itervalues().next()) >= 2:
            lmda = np.cumsum(func(), dtype=float)
            for state, info in self._model.statespace.iteritems():
                idx = np.argmax(lmda > np.random.random())
                policy[state] = [self._history[state][idx]]
        else:
            for state, info in self._model.statespace.iteritems():
                policy[state] = [self.get_best_action(state)]

        return policy

    def _calculate_reachable_states(self):
        """Identify the reachable states."""
        for state, info in self._model.statespace.iteritems():
            info.steps_away = 100000
            for mdl in info.models.values():
                if mdl.visits > 0:
                    info.steps_away = 0
                    break

    def _remove_unreachable_states(self):
        """Remove unreachable states."""
        if False and self._ignore_unreachable:
            for state in self._model.statespace.keys():
                info = self._model.statespace[state]

                if info.steps_away > ValueIteration.MAX_STEPS:
                    self._model.statespace.pop(state, None)

    # noinspection PyMethodMayBeStatic
    def _add_subplot(self, fig, ax, data, ylabels):
        """Add a subplot."""
        h, w = data.shape

        # noinspection PyUnresolvedReferences
        heatmap = ax.pcolormesh(data,
                                edgecolors='w',  # put white lines between squares in heatmap
                                cmap=plt.cm.Blues)

        ax.autoscale(tight=True)  # get rid of whitespace in margins of heatmap
        ax.set_aspect('equal')  # ensure heatmap cells are square
        ax.tick_params(bottom='on', top='off', left='on', right='off')  # turn off ticks

        ax.set_yticks(np.arange(h) + 0.5)
        ax.set_yticklabels(np.arange(1, h + 1), size=7)
        ax.set_xticks(np.arange(w) + 0.5)
        ax.set_xticklabels(np.arange(1, w + 1), size=7)

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", "20%", pad="15%")
        cbar = fig.colorbar(heatmap, cax=cax)
        cbar.ax.tick_params(labelsize=7)

        # Set the labels
        ax.set_xticklabels(self._model.get_actions(), minor=False, rotation=90)
        ax.set_yticklabels(ylabels, minor=False)
