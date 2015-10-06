#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mdp
----------------------------------

Tests for `mlpy.mdp` module.
"""
import pytest


class TestMDPModel(object):
    def setup_method(self, _):
        pass

    def test_model_creation(self):
        from mlpy.mdp import MDPModelFactory
        from mlpy.mdp.continuous import CbTData

        # create discrete model
        MDPModelFactory.create('discretemodel')

        # create decision tree model
        MDPModelFactory.create('decisiontreemodel')
        MDPModelFactory.create('decisiontreemodel', explorer_type='unknownbonusexplorer')
        MDPModelFactory.create('decisiontreemodel', explorer_type='leastvisitedbonusexplorer',
                               explorer_params={'rmax': 1.0})

        with pytest.raises(ValueError):
            MDPModelFactory.create('decisiontreemodel', explorer_type='undefined')

        # create CASML model
        case_template = {
            "state": {
                "type": "float",
                "value": "data.state",
                "is_index": True,
                "retrieval_method": "radius-n",
                "retrieval_method_params": 0.01
            },
            "delta_state": {
                "type": "float",
                "value": "data.next_state - data.state",
                "is_index": False,
            }
        }
        MDPModelFactory.create('casml', CbTData(case_template))

    def teardown_method(self, _):
        pass
