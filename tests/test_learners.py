#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_learners
----------------------------------

Tests for `mlpy.learners` module.
"""


class TestOnlineLearners(object):
    def setup_method(self, _):
        from mlpy.mdp.stateaction import Action
        Action.set_description({
            'out': {'value': [-0.004]},
            'in': {'value': [0.004]},
            'kick': {'value': [-1.0]}
        })

    def test_learner_creation(self):
        from mlpy.learners import LearnerFactory

        # create qlearner
        LearnerFactory.create('qlearner')
        LearnerFactory.create('qlearner', max_steps=10)

        # create rldtlearner
        from mlpy.mdp.discrete import DiscreteModel
        from mlpy.planners.discrete import ValueIteration
        planner = ValueIteration(DiscreteModel())

        LearnerFactory.create('rldtlearner', planner)
        LearnerFactory.create('rldtlearner', planner, 10)
        LearnerFactory.create('rldtlearner', planner, max_steps=10)

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
