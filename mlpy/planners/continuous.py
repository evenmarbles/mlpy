from __future__ import division, print_function, absolute_import

import math
from datetime import datetime
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from scipy.optimize import fmin

from . import IPlanner
from ..mdp.stateaction import MDPState, MDPAction, Experience, RewardFunction
from ..mdp.distrib import ProbabilityDistribution
from ..tools.log import LoggingMgr
from ..tools.misc import Waiting

__all__ = ['CASMLPlanner']


class CASMLPlanner(IPlanner):

    @property
    def model(self):
        """ The Markov decision process model.

        The CASML model containing information
        about the states, actions, and their transitions and the
        reward function.

        Returns
        -------
        CASML :
            The continuous action and state model learner.

        """
        return self._model

    def __init__(self, model, gamma=None, alpha=None, explorer=None, discrete_action=True):
        super(CASMLPlanner, self).__init__(explorer)
        # LoggingMgr().change_level(self.mid, LoggingMgr.LOG_DEBUG)

        self._model = model
        """:type: IMDPModel"""

        self._alpha = 0.5 if alpha is None else alpha
        self._gamma = 0.9 if gamma is None else gamma

        self._discrete_action = discrete_action

    def init(self):
        self._model.init()

    def get_random_action(self):
        action = np.random.random(MDPAction.nfeatures)
        for i, (min_, max_) in enumerate(zip(MDPAction.min_features, MDPAction.max_features)):
            action[i] = (max_ - min_) * action[i] + min_
        return MDPAction(action)

    def _get_qvalues(self, state, action, reward, actions):
        q = {a: 0.0 for a in actions}

        if action is not None:
            transition_proba = {a: ProbabilityDistribution() for a in actions}
            reward_func = {a: RewardFunction() for a in actions}
            if self._discrete_action:
                reward_func[action].set(reward)
            else:
                for a in actions:
                    reward_func[a].set(reward)

            self._update_transitions(state, transition_proba)

            for a, transition_proba in transition_proba.iteritems():
                q[a] = reward_func[a].get()

                for (id_, state2), prob in transition_proba:
                    q[a] += self._gamma * prob * self._model.cases[id_]['value']

        return q

    def get_best_action(self, state):
        action = self._model.experience.action
        next_state = self._model.experience.next_state
        reward = self._model.experience.reward

        # determine q-values for state
        actions = self._model.get_actions(state)
        q = self._get_qvalues(next_state, action, reward, actions)

        # find best action
        if self._discrete_action:
            action = self._explorer.choose_action(actions, q.values())
            msg = "state=%s\tact=%s\tvalue=%.2f" % (state, action, q[action])
            self._logger.info(msg)
            return action

        # 1. retrieve cases from cb_t whose state are similar to state
        case_t_matches = self._model.retrieve('transition', state, self._model.action)

        if len(case_t_matches) <= 2:
            return self.get_random_action()

        r = self._model.case.reward_func.get()

        X = np.zeros((len(case_t_matches), MDPAction.nfeatures))
        y = np.zeros(len(case_t_matches))

        for i, cm_t in enumerate(case_t_matches.itervalues()):
            # 2. compute next state
            next_state = state + cm_t.case['delta_state']
            v_i = self._model.statespace[MDPState(next_state)].v

            # 3. retrieve k nearest neighbors of next state in cb_v
            case_v = self._model._cb_v.case_from_data(
                Experience(next_state, MDPAction(np.zeros(MDPAction.nfeatures)), next_state))
            case_v_matches = self._model._cb_v.retrieve(case_v)

            # 4. revise value of k nearest neighbors
            s = 0
            for cm_v in case_v_matches.itervalues():
                s += self._kernel(np.linalg.norm(np.asarray(next_state) - np.asarray(cm_v.case["state"])))

            X[i] = cm_t.case['act']
            for cm_v in case_v_matches.itervalues():
                v_k = cm_v.case['value']
                contrib = self._kernel(np.linalg.norm(np.asarray(next_state) - np.asarray(cm_v.case["state"]))) / s
                y[i] += v_k + self._alpha * (r + self._gamma * v_i - v_k) * contrib
            y[i] /= len(case_v_matches)

        # 5. perform quadratic regression
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(X, y)

        # 6. locate action maximizing the model
        def qd_1d(x):
            x = np.asarray(x)
            r = -np.sum(model.named_steps['linearregression'].coef_ * np.asarray([1, x[0], x[0]**2]))
            return r

        action = fmin(qd_1d, np.random.random(1))
        for i, (min_, max_) in enumerate(zip(MDPAction.min_features, MDPAction.max_features)):
            action[i] = min(max(action[i], min_), max_)
        return MDPAction(action)

    def plan(self):
        """Plan for the optimal policy.

        Perform value iteration to calculate the value for the cases in
        CASML's value case base.

        """
        nloops = 0
        max_error = 5000
        min_error = 0.1

        states_updated = 0

        waiting = None
        if self._logger.level > LoggingMgr.LOG_DEBUG:
            waiting = Waiting("Perform value iteration")
            waiting.start()

        s0 = datetime.now()

        for case in self._model.cases.itervalues():
            self._update_transitions(case['state'], case.transition_proba)

        # perform exact value iteration on X using math:`\tilde{P}\ and math:`\tilde{R}`
        while max_error > min_error:
            self._logger.debug("max error: %0.5f nloops: %d", max_error, nloops)

            max_error = 0
            nloops += 1

            for case in self._model.cases.itervalues():
                state = case['state']
                self._logger.debug("\tState: id: %d: %s", case.id, state)

                states_updated += 1

                newv = 0.0
                for action, transition_proba in case.transition_proba.iteritems():
                    newq = case.reward_func[action].get(state)

                    v = 0.0
                    for (id_, state2), prob in transition_proba:
                        self._logger.debug("\t\tNext state is: %s, prob: %.2f", state2, prob)

                        # if self._discrete_action:
                        #     maxq = max([self._model.cases[id_].q[a] for a in self._model.get_actions(state2)])
                        #     newq += self._gamma * prob * maxq

                        v += prob * (newq + self._gamma * self._model.cases[id_]['value'])

                    newv = max(v, newv)

                    # if self._discrete_action:
                    #     tderror = math.fabs(case.q[action] - newq)
                    #     if tderror > max_error:
                    #         max_error = tderror
                    #     self._logger.debug("\t\tTD error: %.5f Max error: %.5f", tderror, max_error)
                    #
                    #     case.q[action] = newq

                # if not self._discrete_action:
                tderror = math.fabs(case['value'] - newv)
                if tderror > max_error:
                    max_error = tderror
                self._logger.debug("\t\tTD error: %.5f Max error: %.5f", tderror, max_error)

                case['value'] = newv

        s1 = datetime.now()
        delta = s1 - s0

        if waiting is not None:
            waiting.stop()

        self._logger.info("\tvalues computed with maxError: %.5f nloops: %d time: %d:%d states: %d", max_error, nloops,
                          delta.seconds, delta.microseconds, states_updated)

    def visualize(self):
        pass

    def _update_transitions(self, state, transitions):
        # build math:`\tilde{P}` used for fitted value iteration
        for act, transition_proba in transitions.iteritems():
            transition_proba.clear()
            for next_state, prob in self._model.predict_proba(state, act).iteritems():
                case_matches = self._model.retrieve('transition', next_state, act)

                cm = {id_: MDPState(cm.case['state']) for id_, cm in case_matches.iteritems() if id_ in self._model.cases}

                # only consider exact match as solution if it exists in case base
                if next_state in cm.itervalues():
                    for id_, s in cm.iteritems():
                        if s == next_state:
                            transition_proba.iadd((id_, next_state), prob)
                    continue

                # build math:`\Phi[s', x']: S x X \rightarrow W`, mapping state to weights for math:`x \in X`
                w = np.asarray(
                    [self._kernel(np.linalg.norm(next_state - s)) for s in cm.itervalues()])
                w /= w.sum()

                # math:`\tilde{P}[xa, x'] = \sum_{s' \in S} P[xa, s'] \times \Phi[s', x']`
                for i, s in enumerate(cm.iteritems()):
                    transition_proba.iadd(s, prob * w[i])

    def _kernel(self, d):
        return np.exp(-d ** 2)

    def _create_policy(self, func=None):
        pass
