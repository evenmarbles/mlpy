"""
====================================
Agent design (:mod:`mlpy.agents`)
====================================

.. currentmodule:: mlpy.agents

This module contains functionality for designing agents
navigating inside an :class:`.Environment`.

Control of the agents is specified by an agent module which
is handled by the :class:`Agent` base class.

An agent class deriving from :class:`Agent` can also make use
of a finite state machine (FSM) to control the agent's behavior
and a world model to maintain a notion of the current state of
the world.


Agents
======

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~modelbased.Bot
   ~modelbased.BotWrapper
   ~modelbased.Agent
   ~modules.AgentModuleFactory
   ~modules.IAgentModule
   ~modules.LearningModule
   ~modules.FollowPolicyModule
   ~modules.UserModule


World Model
===========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~world.WorldObject
   ~world.WorldModel


Finite State Machine
====================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~fsm.Event
   ~fsm.EmptyEvent
   ~fsm.FSMState
   ~fsm.Transition
   ~fsm.OnUpdate
   ~fsm.StateMachine

"""
__all__ = ['modelbased', 'fsm', 'modules', 'world']
