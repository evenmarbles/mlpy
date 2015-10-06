#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_agents
----------------------------------

Tests for `mlpy.agents` module.
"""
import os
import pytest


class TestAgentModule(object):
    def setup_method(self, _):
        pass

    def test_agentmodule_creation(self):
        from mlpy.agents.modules import AgentModuleFactory

        # create follow policy module
        with pytest.raises(TypeError):
            AgentModuleFactory().create('followpolicymodule')

        from mlpy.auxiliary.io import load_from_file
        data = load_from_file(os.path.join(os.getcwd(), 'tests', 'data/policies.pkl'))
        with pytest.raises(AttributeError):
            AgentModuleFactory().create('followpolicymodule', data)

        AgentModuleFactory().create('followpolicymodule', data['act'][0:2])

        # create learning module
        with pytest.raises(TypeError):
            AgentModuleFactory().create('learningmodule')

        # create `qlearner` learning module
        AgentModuleFactory().create('learningmodule', learner_type='qlearner', alpha=0.5)

        with pytest.raises(TypeError):
            AgentModuleFactory().create('learningmodule', 'qlearner', 0.5)

        # create `modelbasedlearner` learner module
        from mlpy.mdp.discrete import DiscreteModel
        from mlpy.planners.discrete import ValueIteration
        planner = ValueIteration(DiscreteModel(['out', 'in', 'kick']))

        AgentModuleFactory().create('learningmodule', 'modelbasedlearner', planner)
        AgentModuleFactory().create('learningmodule', learner_type='modelbasedlearner', planner=planner)

        with pytest.raises(TypeError):
            AgentModuleFactory().create('learningmodule', 'modelbasedlearner')

    def teardown_method(self, _):
        pass


class TestAgent(object):
    def setup_method(self, _):
        pass

    def test_agent_creation(self):
        from mlpy.agents.modelbased import ModelBasedAgent

        ModelBasedAgent('learningmodule', learner_type='qlearner')

        ModelBasedAgent('learningmodule', learner_type='qlearner', alpha=0.7)
        ModelBasedAgent('learningmodule', None, None, None, None, 'qlearner', alpha=0.7)

        from mlpy.mdp.discrete import DiscreteModel
        from mlpy.planners.discrete import ValueIteration
        planner = ValueIteration(DiscreteModel(['out', 'in', 'kick']))

        ModelBasedAgent('learningmodule', learner_type='modelbasedlearner', planner=planner)

    def teardown_method(self, _):
        pass
