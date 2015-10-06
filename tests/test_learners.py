#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_learners
----------------------------------

Tests for `mlpy.learners` module.
"""


class TestOnlineLearners(object):
    def setup_method(self, _):
        pass

    def test_learner_creation(self):
        from mlpy.learners import LearnerFactory

        # create qlearner
        LearnerFactory.create('qlearner')
        LearnerFactory.create('qlearner', alpha=0.5)

        # create modelbasedlearner
        from mlpy.mdp.discrete import DiscreteModel
        from mlpy.planners.discrete import ValueIteration
        planner = ValueIteration(DiscreteModel())

        LearnerFactory.create('modelbasedlearner', planner)

    def teardown_method(self, _):
        pass


class TestOfflineLearners(object):
    def setup_method(self, _):
        pass

    def test_apprenticeshiplearner(self):
        pass

    def test_incrapprenticeshiplearner(self):
        pass

    def teardown_method(self, _):
        pass
