#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_planners
----------------------------------

Tests for `mlpy.planners` module.
"""


class TestExplorers(object):
    def setup_method(self, _):
        pass

    def test_explorer_creation(self):
        from mlpy.planners.explorers import ExplorerFactory

        # create epsilon greedy explorer
        ExplorerFactory().create('egreedyexplorer')
        ExplorerFactory().create('egreedyexplorer', 0.8)

        # create softmax explorer
        ExplorerFactory().create('softmaxexplorer')
        ExplorerFactory().create('softmaxexplorer', tau=3.0, decay=0.4)

    def teardown_method(self, _):
        pass
