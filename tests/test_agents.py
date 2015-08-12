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
        from mlpy.mdp.stateaction import Action
        Action.description = None
        Action.nfeatures = None

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

        from mlpy.mdp.stateaction import Action
        Action.set_description({
            'out': {'value': [-0.004]},
            'in': {'value': [0.004]},
            'kick': {'value': [-1.0]}
        })

        # create `qlearner` learning module
        AgentModuleFactory().create('learningmodule', 'qlearner', max_steps=10)
        AgentModuleFactory().create('learningmodule', 'qlearner', lambda s, a: 1.0, max_steps=10)

        with pytest.raises(ValueError):
            AgentModuleFactory().create('learningmodule', 'qlearner', 1.0, max_steps=10)

        # create `rldtlearner` learner module
        from mlpy.mdp.discrete import DiscreteModel
        from mlpy.planners.discrete import ValueIteration
        planner = ValueIteration(DiscreteModel(['out', 'in', 'kick']))

        AgentModuleFactory().create('learningmodule', 'rldtlearner', None, planner, max_steps=10)
        AgentModuleFactory().create('learningmodule', 'rldtlearner', planner=planner, max_steps=10)

        with pytest.raises(TypeError):
            AgentModuleFactory().create('learningmodule', 'rldtlearner', max_step=10)

    def teardown_method(self, _):
        pass


class TestAgent(object):
    def setup_method(self, _):
        from mlpy.mdp.stateaction import Action
        Action.set_description({
            'out': {'value': [-0.004]},
            'in': {'value': [0.004]},
            'kick': {'value': [-1.0]}
        })

    def test_agent_creation(self):
        from mlpy.agents import Agent
        from mlpy.experiments.task import Task
        from mlpy.mdp.discrete import DiscreteModel
        from mlpy.planners.discrete import ValueIteration

        Agent()

        Agent(module_type='learningmodule', learner_type='qlearner', max_steps=10)
        Agent(None, 'learningmodule', None, 'qlearner', max_steps=10)

        task = Task()
        Agent(None, 'learningmodule', task, 'qlearner', max_steps=10)

        planner = ValueIteration(DiscreteModel(['out', 'in', 'kick']))
        Agent(None, 'learningmodule', None, 'rldtlearner', None, planner, max_steps=10)

    def teardown_method(self, _):
        pass
