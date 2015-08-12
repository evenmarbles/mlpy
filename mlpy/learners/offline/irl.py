from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import sys
from datetime import datetime

import numpy as np

from ...tools.log import LoggingMgr
from ...mdp.stateaction import State, RewardFunction
from . import IOfflineLearner

__all__ = ['ApprenticeshipLearner', 'IncrApprenticeshipLearner']


# noinspection PyAbstractClass
class ApprenticeshipLearner(IOfflineLearner):
    """The apprenticeship learner.

    The apprenticeship learner is an inverse reinforcement learner, a method introduced
    by Abbeel and Ng [1]_ which strives to imitate the demonstrations given by an expert.

    Parameters
    ----------
    obs : array_like, shape (`n`, `nfeatures`, `ni`)
        List of trajectories provided by demonstrator, which the learner
        is trying to emulate, where `n` is the number of sequences, `ni` is
        the length of the i_th demonstration, and each demonstration has
        `nfeatures` features.
    planner : IPlanner
        The planner to use to determine the best action.
    method : {'projection', 'maxmargin'}, optional
        The IRL method to employ. Default is `projection`.
    max_iter : int, optional
        The maximum number of iteration after which learning
        will be terminated. It is assumed that a policy close enough to
        the experts demonstrations was found. Default is `inf`.
    thresh : float, optional
        The learning is considered having converged to the
        demonstrations once the threshold has been reach. Default is `eps`.
    gamma : float, optional
        The discount factor. Default is 0.9.
    nsamples : int, optional
        The number of samples taken during Monte Carlo sampling. Default is 100.
    max_steps : int, optional
        The maximum number of steps in an iteration (during MonteCarlo sampling).
        Default is 100.
    filename : str, optional
        The name of the file to save the learner state to after each iteration.
        If None is given, the learner state is not saved. Default is None.

    Other Parameters
    ----------------
    mix_policies : bool
        Whether to create a new policy by mixing from policies seen so far or by
        considering the best valued action. Default is False.
    rescale : bool
        If set to True, the feature expectations are rescaled to be between 0 and 1.
        Default is False.
    visualize : bool
        Visualize each iteration of the IRL step if set to True. Default is False.

    See Also
    --------
    :class:`IncrApprenticeshipLearner`

    Notes
    -----
    Method **maxmargin** using a QP solver to solve the following equation:

        .. math::

            \\begin{aligned}
            & \\underset{t, w}{\\text{maximize}} & & t \\\\
            & \\text{subject to} & & w^T \\mu_E > w^T \\mu^{(j)} + t, j=0, \\ldots, i-1 \\\\
            & & & ||w||_2 \\le 1.
            \\end{aligned}

    and mixing policies is realized by solving the quadratic problem:

        .. math::

            \\begin{aligned}
            & \\text{minimize} & &  ||\\mu_E - \\mu||_2 \\\\
            & \\text{subject to} & & \\mu = \\sum_i (\\lambda_i \\mu^{(i)}) \\\\
            & & & \\lambda_i \\ge 0 \\\\
            & & & \\sum_i \\lambda_i = 1
            \\end{aligned}

    The QP solver used for the implementation is the IBM ILOG CPLEX Optimizer which
    requires a separate license. If you are unable to obtain a license, the 'projection'
    method can be used instead.

    References
    ----------
    .. [1] Abbeel, Pieter, and Andrew Y. Ng. "Apprenticeship learning via inverse
        reinforcement learning." Proceedings of the twenty-first international
        conference on Machine learning. ACM, 2004.

    """

    def __init__(self, obs, planner, method=None, max_iter=None, thresh=None, gamma=None,
                 nsamples=None, max_steps=None, filename=None, **kwargs):
        super(ApprenticeshipLearner, self).__init__(filename)
        self._logger = LoggingMgr().get_logger(self._mid)

        self._iter = 0
        self._i = 0
        self._t = 0.0

        # noinspection PyTypeChecker
        RewardFunction.cb_get = staticmethod(lambda r, s: np.dot(s, RewardFunction.reward))

        self._planner = planner
        if self._planner is None:
            raise AttributeError("The apprenticeship learner requires a planner.")

        self._method = method if method is not None else 'projection'
        if self._method not in ['maxmargin', 'projection']:
            raise ValueError("%s is not a valid IRL method" % self._method)

        self._max_iter = max_iter if max_iter is not None else sys.maxint
        self._thresh = thresh if thresh is not None else np.finfo(float).eps
        self._gamma = gamma if gamma is not None else 0.9
        self._nsamples = nsamples if nsamples is not None else 100
        self._max_steps = max_steps if max_steps is not None else 100

        self._mix_policies = kwargs["mix_policies"] if "mix_policies" in kwargs else False
        self._rescale = kwargs["rescale"] if "rescale" in kwargs else False
        self._visualize = kwargs["visualize"] if "visualize" in kwargs else False

        assert (len(obs) > 0)
        if self._logger.level <= LoggingMgr.LOG_DEBUG:
            for i, o in enumerate(obs):
                self._logger.debug("Demonstration #{0}:".format(i + 1))
                for j, state in enumerate(o.T):
                    self._logger.debug("    {0}: {1}".format(j + 1, State(state)))

        self._mu_E = self._estimate_expert_mu(obs)
        """:type : ndarray[float]"""

        nfeatures = obs[0].shape[0]
        self._mu = np.empty((0, nfeatures), float)
        """:type : ndarray[ndarray[float]]"""
        self._weights = np.empty((0, nfeatures), float)
        """:type : ndarray[ndarray[float]]"""

        if self._method == 'projection':
            self._mu_bar = np.array([])
            """:type: ndarray[float]"""

        if self._method == 'maxmargin' or self._mix_policies:
            import cplex
            from cplex.exceptions import CplexError
            self.cplex = cplex
            self.CplexError = CplexError

    def __getstate__(self):
        data = super(ApprenticeshipLearner, self).__getstate__()
        data.update(self.__dict__.copy())

        remove_list = ('_id', '_logger', '_i')
        for key in remove_list:
            if key in data:
                del data[key]

        data['_iter'] = self._i
        return data

    def __setstate__(self, d):
        super(ApprenticeshipLearner, self).__setstate__(d)

        for name, value in d.iteritems():
            setattr(self, name, value)

        # noinspection PyTypeChecker
        RewardFunction.cb_get = staticmethod(lambda r, s: np.dot(s, RewardFunction.reward))
        RewardFunction.reward = self._weights[self._iter - 1]

        self._logger = LoggingMgr().get_logger(self._mid)

    def learn(self):
        """Learn the optimal policy via apprenticeship learning.

        The apprenticeship learning algorithm for finding a policy :math:`\\tilde{\\pi}`,
        that induces feature expectations :math:`\\mu(\\tilde{\\pi})` close to :math:`\\mu_E`
        is as follows:

        1. Randomly pick some policy :math:`\\pi^{(0)}`, compute (or approximate via Monte Carlo)
           :math:`\\mu^{(0)} = \\mu(\\pi^{(0)})`, and set :math:`i=1`.

        2. Compute :math:`t^{(i)} = \\underset{w:||w||_2 \\le 1}{\\text{max}}\\underset{j \in {0 \ldots (i-1)}}{\\text{min}} w^T(\\mu_E = \\mu^{(j)})`,
           and let :math:`w^{(i)}` be the value of :math:`w` that attains this maximum. This can be achieved
           by either the **max-margin** method or by the **projection** method.

        3. If :math:`t^{(i)} \\le \\epsilon`, then terminate.

        4. Using the RL algorithm, compute the optimal policy :math:`\\pi^{(i)}` for the MDP using rewards
           :math:`R = (w^{(i)})^T \\phi`.

        5. Compute (or estimate) :math:`\\mu^{(i)} = \\mu(\\pi^{(i)})`.

        6. Set :math:`i = i + 1`, and go back to step 2.
        """
        for self._i in range(self._iter, self._max_iter):
            self._logger.info(
                "Starting iteration {0}/{1} (error was {2})".format(self._i + 1, self._max_iter, self._t))
            if self._perform_irl(self._i):
                break

            # save to file
            self.save(self._filename)

    def _perform_irl(self, i):
        """Perform the inverse reinforcement learning algorithm.

        Parameters
        ----------
        i : int
            The current iteration count.

        Returns
        -------
        bool :
            In the case of the algorithm having converged on the optimal policy,
            True is returned otherwise False. The algorithm is considered to have
            converged to the optimal policy if either the performance is within a
            certain threshold or if the maximum number of iterations has been reached.
        """

        # 2. Estimate mu
        self._logger.info("Estimate mu...")
        self._mu = np.vstack([self._mu, self._estimate_mu()])

        # 3. Compute maximum weights
        if self._method == "projection":
            self._t, weights, self._mu_bar = self._compute_projection(self._mu[i], self._mu_bar)
        else:
            self._t, weights = self._compute_max_margin(self._i + 1, self._mu)
        self._weights = np.vstack([self._weights, weights])

        self._logger.debug("\nweights = \n{0}\nmu = \n{1}\nmu_E = \n{2}".format(weights, self._mu[i], self._mu_E))
        self._logger.info("Delta error is {0}".format(abs(self._t - self._thresh)))

        # 4. Check termination
        if self._t <= self._thresh:
            self._logger.info("Converged to optimal solution")

            # save to file
            self.save(self._filename)
            return True

        # 5. Compute optimal policy \pi^(i)

        # update reward model
        RewardFunction.reward = weights

        # value iteration
        self._planner.plan()

        if self._visualize:
            self._planner.visualize()

        return False

    def _estimate_expert_mu(self, obs):
        """Estimate the experts feature expectations.

        Calculate the empirical estimate for the experts feature expectation mu
        from the demonstration trajectories.

        Parameters
        ----------
         obs: array_like, shape (`n`, `nfeatures`, `ni`)
            List of trajectories provided by demonstrator, which the learner
            is trying to emulate, where `n` is the number of sequences, `ni` is
            the length of the i_th demonstration, and each demonstration has `nfeatures` features.

        Returns
        -------
        ndarray[float] :
            The experts feature expectations

        """
        n = obs.shape[0]
        nfeatures = obs[0].shape[0]

        mu = np.zeros(nfeatures)
        for o in obs:
            for t, sample in enumerate(o.T):
                mu += self._gamma ** t * np.array(sample)
        mu /= n

        if self._rescale:
            mu *= (1 - self._gamma)

        return mu

    def _estimate_mu(self):
        """Estimate the feature expectations for the current policy.

        Perform Monte Carlo sampling to estimate the feature expectations, mu,
        for the policy.

        Returns
        -------
        ndarray[float] :
            The feature expectations.

        """
        s0 = datetime.now()

        mu = np.zeros(State.nfeatures, float)

        self._planner.create_policy(self._find_closest if self._mix_policies else None)

        for i in range(self._nsamples):
            self._logger.info("Sample #{0}...".format(i + 1))

            # select an initial state according to the initial state distribution
            state = self._planner.model.sample()

            mu = np.add(mu, state.get())

            for t in range(2, self._max_steps + 1):
                # choose the next state according to the chosen policy
                action = self._planner.get_next_action(state, use_policy=True)

                next_state = self._planner.model.sample(state, action)
                if next_state is None:
                    # a state is reached for which no empirical transition data exists, it is uncertain where
                    # to go from here, so break out of the loop and this sample will be discarded
                    break
                self._logger.info("state={0}, act={1}, next= {2}".format(state, action, next_state))

                # calculate mu here
                fe = self._gamma ** (t - 1) * np.array(next_state.get())
                mu = np.add(mu, fe)

                state = next_state

        mu /= self._nsamples

        if self._rescale:
            mu *= (1 - self._gamma)

        s1 = datetime.now()
        delta = s1 - s0
        self._logger.info("Estimation of feature expectations in %d:%d\n", delta.seconds, delta.microseconds)

        return mu

    def _compute_projection(self, mu, mu_bar):
        """ Inverse reinforcement learning step.

        Computation of orthogonal projection of mu_E onto the line through mu_bar(i-2)
        and mu(i-1).

        Parameters
        ----------
        mu : array_like, shape (`nfeatures`,)
            Feature expectations found in the previous iteration, with `nfeatures`
            being the number of features.
        mu_bar : array_like, shape (`nfeatures`,)
            Vector with `nfeatures` being the number of features.

        Returns
        -------
        t : float
            The margin.
        w : ndarray[float]
            The feature weights.
        mu_bar : ndarray[float]
            Vector of the current iteration.

        """
        if mu_bar.size == 0:
            mu_bar = np.copy(mu)
        else:
            diff = mu - mu_bar
            mu_bar = mu_bar + (np.dot(diff, self._mu_E - mu_bar) / np.dot(diff, diff)) * diff

        w = self._mu_E - mu_bar
        t = np.linalg.norm(w)
        return t, w, mu_bar

    def _compute_max_margin(self, idx, mu):
        """ Inverse reinforcement learning step.

        Guesses the reward function being optimized by the expert; i.e. find the reward
        on which the expert does better by a 'margin' of `t`, then any of the policies
        found previously.

        Parameters
        ----------
        idx : int
            The current iteration step
        mu : array_like, shape (`n`, `nfeatures`)
            The set of feature expectations, where `n` is the number of iterations and
            `nfeatures` is the number of features.

        Returns
        -------
        t : float
            The margin.
        w : ndarray[float]
            The feature weights

        Notes
        -----
        Using the QP solver (CPLEX) solve the following equation:

        .. math::

            \\begin{aligned}
            & \\underset{t, w}{\\text{maximize}} & & t \\
            & \\text{subject to} & & w^T * mu_E > w^T * mu^j + t, j=0, \\ldots, idx-1 \\
            & & & ||w||_2 <= 1.
            \\end{aligned}

        """
        try:
            n = mu[0, :].shape[0]

            cpx = self.cplex.Cplex()
            cpx.objective.set_sense(cpx.objective.sense.maximize)

            obj = [1.0] + [0.0] * n
            lb = [0.0] + [-self.cplex.infinity] * n
            ub = [self.cplex.infinity] * (n + 1)
            names = ["t"]
            for i in range(n):
                names.append("w{0}".format(i))
            cpx.variables.add(obj=obj, lb=lb, ub=ub, names=names)

            # add linear constraints:
            # w^T * mu_E >= w^T * mu^j + t, j=0,...,idx-1
            #       => -t + w^T * (mu_E - mu^j) >= 0, j=0,...,idx-1
            # populated by row
            expr = []
            for j in range(idx):
                # noinspection PyTypeChecker
                row = [names, [-1.0] + (self._mu_E - mu[j, :]).tolist()]
                expr.append(row)
            senses = "G" * idx
            rhs = [0.0] * idx
            cpx.linear_constraints.add(expr, senses, rhs)

            # add quadratic constraints:
            #   w * w^T <= 1
            q = self.cplex.SparseTriple(ind1=names, ind2=names, val=[0.0] + [1.0] * n)
            cpx.quadratic_constraints.add(rhs=1.0, quad_expr=q, sense="L")

            cpx.solve()
            if not cpx.solution.get_status() == cpx.solution.status.optimal:
                raise Exception("No optimal solution found")

            t, w = cpx.solution.get_values(0), cpx.solution.get_values(1, n)
            w = np.array(w)
            return t, w

        except self.CplexError as e:
            self._logger.exception(e.message)
            return None

    def _find_closest(self):
        """Find the point closest to the experts feature expectation.

        Find the point closest to the experts feature expectation in the convex closure
        of the feature expectation by solving the quadratic problem:

        .. math::

            min     ||self._mu_E - mu||_2
            s.t.    mu = sum(lambda_i * mu_i)
                    lambda >=0
                    sum(lambda_i) = 1

        Returns
        -------
        lambda_ : array_like, shape (`nfeatures`, `n`)
            Set of feature expectations found by the algorithm, with `nfeatures` being the number
            of features and `n` being the number of iterations (until the margin `t` was epsilon close).

        """
        try:
            n = self._mu.shape[0]

            cpx = self.cplex.Cplex()
            cpx.objective.set_sense(cpx.objective.sense.minimize)

            # Solve non-negative least square problem
            # min     ||A * x - y||_2
            #   s.t.    x >= 0
            #           sum(x_i) = 1
            #
            # by solving the quadratic program problem
            #   min     1/2 x^T * Q * x + c^T * x
            #   s.t.    x >= 0
            #           sum(x_i) = 1
            # where Q = A^T * A, and c = -A^T * y
            #
            # Let A = mu, y = mu_E, and x = lambda

            # set linear terms:
            obj = -(np.dot(self._mu, self._mu_E))
            obj = obj.tolist()

            # set linear boundaries
            #   lambda >= 0 (true for all components of the vector lambda)
            lb = [0.0] * n
            ub = [self.cplex.infinity] * n

            # set linear constraints:
            #   sum_i(lambda_i) = 1
            # names = []
            # for i in range(n):
            #     names.append("lambda{0}".format(i))
            cols = []
            for i in range(n):
                cols.append([[0], [1]])
            cpx.linear_constraints.add(rhs=[1.0], senses="E")

            cpx.variables.add(obj=obj, lb=lb, ub=ub, columns=cols)

            # add quadratic terms
            qmat = []
            q = 2 * np.dot(self._mu, self._mu.transpose())
            for j in range(n):
                # noinspection PyUnresolvedReferences
                row = [range(n), q[j, :].tolist()]
                qmat.append(row)
            cpx.objective.set_quadratic(qmat)

            cpx.solve()

            if not cpx.solution.get_status() == cpx.solution.status.optimal:
                raise Exception("No optimal solution found")

            lambda_ = np.array(cpx.solution.get_values())

            self._logger.info("Cplex solution value: {0}".format(cpx.solution.get_objective_value()))
            self._logger.info("lambda={0}".format(lambda_))

            return lambda_

        except self.CplexError as e:
            self._logger.exception(e.message)
            return None


class IncrApprenticeshipLearner(ApprenticeshipLearner):
    """ Incremental apprenticeship learner.

    The model under which the apprenticeship is operating is updated incrementally
    while learning a policy that emulates the expert's demonstrations.

    Parameters
    ----------
    obs : array_like, shape (`n`, `nfeatures`, `ni`)
        List of trajectories provided by demonstrator, which the learner
        is trying to emulate, where `n` is the number of sequences, `ni` is
        the length of the i_th demonstration, and each demonstration has
        `nfeatures` features.
    planner : IPlanner
        The planner to use to determine the best action.
    method : {'projection', 'maxmargin'}, optional
        The IRL method to employ. Default is `projection`.
    max_iter : int, optional
        The maximum number of iteration after which learning
        will be terminated. It is assumed that a policy close enough to
        the experts demonstrations was found. Default is `inf`.
    thresh : float, optional
        The learning is considered having converged to the
        demonstrations once the threshold has been reach. Default is `eps`.
    gamma : float, optional
        The discount factor. Default is 0.9.
    nsamples : int, optional
        The number of samples taken during Monte Carlo sampling. Default is 100.
    max_steps : int, optional
        The maximum number of steps in an iteration (during MonteCarlo sampling).
        Default is 100.
    filename : str, optional
        The name of the file to save the learner state to after each iteration.
        If None is given, the learner state is not saved. Default is None.

    Other Parameters
    ----------------
    mix_policies : bool
        Whether to create a new policy by mixing from policies seen so far or by
        considering the best valued action. Default is False.
    rescale : bool
        If set to True, the feature expectations are rescaled to be between 0 and 1.
        Default is False.
    visualize : bool
        Visualize each iteration of the IRL step if set to True. Default is False.

    Notes
    -----
    Inverse reinforcement learning assumes knowledge of the underlying model. However,
    this is not always feasible. The incremental apprenticeship learner updates its model
    after every iteration by executing the current policy. Thus, it provides an extension to
    the original apprenticeship learner.

    See Also
    --------
    :class:`ApprenticeshipLearner`

    """

    def __init__(self, obs, planner, method=None, max_iter=None, thresh=None, gamma=None,
                 nsamples=None, max_steps=None, filename=None, **kwargs):
        super(IncrApprenticeshipLearner, self).__init__(obs, planner, method, max_iter, thresh, gamma, nsamples,
                                                        max_steps, filename, **kwargs)
        self._step_iter = 0

    def __getstate__(self):
        return super(IncrApprenticeshipLearner, self).__getstate__()

    def __setstate__(self, d):
        super(IncrApprenticeshipLearner, self).__setstate__(d)

        self._i = self._iter
        self._step_iter = 0

    def reset(self, t, **kwargs):
        """Reset the apprenticeship learner.

        Parameters
        ----------
        t : float
            The current time (sec)
        kwargs : dict, optional
            Non-positional parameters, optional.

        """
        super(IncrApprenticeshipLearner, self).reset(t, **kwargs)
        self._step_iter = 0

    def execute(self, experience):
        """Execute learning specific updates.

        Learning specific updates are performed, e.g. model updates.

        Parameters
        ----------
        experience : Experience
            The actor's current experience consisting of previous state, the action
            performed in that state, the current state, and the reward awarded.

        """
        self._planner.model.update(experience)

    def learn(self):
        """ Learn a policy from the experience.

        Learn the optimal policy using an apprenticeship learning algorithm
        incrementally.

        Returns
        -------
        bool :
            Whether the found policy is considered to have converged. The algorithm is
            considered to have converged on the optimal policy if either the performance
            is within a certain threshold or if the maximum number of iterations has been
            reached.

        """
        self._logger.info("Starting iteration {0}/{1} (error was {2})".format(self._i + 1, self._max_iter, self._t))

        self._planner.deactivate_exploration()
        converged = self._perform_irl(self._i)

        self._i += 1
        return converged or (self._i >= self._max_iter)

    def choose_action(self, state):
        """Choose the next action

        The next action is chosen according to the current policy and the
        selected exploration strategy.

        Parameters
        ----------
        state : State
            The current state.

        Returns
        -------
        Action :
            The chosen action.

        """
        action = None

        if self._step_iter < self._max_steps:
            self._planner.activate_exploration()
            action = self._planner.get_next_action(state)
            self._step_iter += 1

        return action
