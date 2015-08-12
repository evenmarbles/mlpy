# -*- coding: utf-8 -*-
"""
MLPy: A Machine Learning library for Python
===========================================

``mlpy`` is a collection of modules to support artificial intelligence related
task, it provides among other various machine learning algorithms.
Documentation is available in the docstrings.

Subpackages
-----------
Using any of these subpackages requires an explicit import. For example,
``import mlpy.learners``.

::

agents                          --- Agent base class, finite state machine, and world model
agents.modules                  --- Modules run by the agent for user control,
                                    policy following or learning
agents.fsm                      --- General finite state machine
auxiliary                       --- Various auxiliary functions
auxiliary.array                 --- Functions related to array manipulation
auxiliary.datasets              --- Data sets
auxiliary.datastructs           --- Data structures
auxiliary.io                    --- Function related to input/ouput
auxiliary.misc                  --- Miscellaneous utilities
auxiliary.plotting              --- Plotting auxiliary functions
cluster                         --- Vector Quantization / Kmeans
cluster.vq                      --- Kmeans
constants                       --- Constants
libs                            --- External libraries
environments                    --- Environments
gridworld                       --- General gridworld implementation
experiments                     --- Experiment structure
experiments.tasks               --- Task implementation
knowledgerep                    --- Knowledge representation
knowledgerep.cbr                --- Case based reasoning engine
learners                        --- Learning algorithms
learners.offline                --- Offline learning algorithms
learners.online                 --- Online learning algorithms
mdp                             --- Markov decision process
mdp.continuous                  --- Continuous models
mdp.discrete                    --- Discrete models
mdp.distrib                     --- Empirical probability distribution
mdp.model                       --- Markov decision process model
mdp.stateaction                 --- State and action classes
modules                         --- Various base modules and design patterns
modules.patterns                --- Design pattern implementations
optimize                        --- Optimization tools
optimize.algorithms             --- Optimization algorithms
optimize.utils                  --- Optimization
planners                        --- Planning tools
planners.discrete               --- Discrete planning algorithms
planners.explorers              --- Explorer tools
planners.explorers.discrete     --- Discrete explorer (EGreedy, softmax)
search                          --- Search tools
search.informed                 --- Informed search algorithms (A*)
stats                           --- Statistical functions
stats.dbn                       --- Dynamic Bayesian networks
stats.dbn.hmm                   --- Hidden Markov Models
stats.conditional               --- Conditional probability distributions
stats.discrete                  --- Discrete probability distributions
stats.mixture                   --- Mixture models
stats.multivariate              --- Multivariate distributions
stats.stats                     --- Statistical helper functions
tools                           --- Tools
tools.configuration             --- Configuration manager
tools.log                       --- Logging manager
tools.misc                      --- Miscellaneous tools
"""
from __future__ import division, print_function, absolute_import

__author__ = 'Astrid Jackson'
__email__ = 'ajackson@eecs.ucf.edu'
__version__ = '0.1.0'
