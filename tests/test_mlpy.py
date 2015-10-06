#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_mlpy
----------------------------------

Tests for `mlpy` module.
"""

import os
import numpy as np


class TestHMM(object):

    def setup_method(self, _):
        import scipy.io
        mat = scipy.io.loadmat(os.path.join(os.getcwd(), 'tests', 'data/speechDataDigits4And5.mat'))
        self.X = np.hstack([mat['train4'][0], mat['train5'][0]])
        dim = self.X[0].shape[0]
        from mlpy.stats import normal_invwishart
        emission_prior = normal_invwishart(np.ones(dim), dim, dim + 1, 0.1 * np.eye(dim))
        from mlpy.stats.dbn.hmm import GaussianHMM
        self.model = GaussianHMM(ncomponents=2, startprob_prior=[3, 2], emission_prior=emission_prior)
        self.model.fit(self.X, n_init=3)

    def test_score_sample(self):
        obs, hidden = self.model.sample(2, 10)
        self.model.score_samples(obs)

    def test_decode(self):
        obs = np.array([[-0.546801713509048, -0.419653261327029],
                       [-0.935576633566449, -0.657506401748589],
                       [-0.725820776888200, 0.599246387463915],
                       [-0.704569013561647, -0.262774535633877],
                       [0.687557416511721, 0.576826951661595],
                       [-0.192086395176301, 0.115146226815108],
                       [-0.132247659696746, -0.927512789802041],
                       [-0.295411326485649, -0.0143719802889745],
                       [0.927473783664851, 0.0131668616996862],
                       [1.44181054016088, 1.20913420252569],
                       [0.873087639691135, -0.881745308344566],
                       [0.894788321023500, 1.20110834379367],
                       [0.920403967222602, -0.158058390367012]])
        self.model.decode(obs)

    def teardown_method(self, _):
        pass


class TestCaseBase(object):

    def setup_method(self, _):
        from mlpy.mdp.stateaction import MDPState, MDPAction
        MDPState.nfeatures = None
        MDPAction.description = None
        MDPAction.nfeatures = None

        case_template = {
            "state": {
                "type": "float",
                "value": "data.state",
                "is_index": True,
                "retrieval_method": "knn",
                "retrieval_method_params": 5
            },
            "act": {
                "type": "float",
                "value": "data.action",
                "is_index": False,
                "retrieval_method": "cosine",
            },
            "delta_state": {
                "type": "float",
                "value": "data.next_state - data.state",
                "is_index": False,
            }
        }

        from mlpy.knowledgerep.cbr.engine import CaseBase
        self.cb = CaseBase(case_template, retention_method_params={'max_error': 1e-5})

        from mlpy.auxiliary.io import load_from_file
        self.data = load_from_file(os.path.join(os.getcwd(), 'tests', 'data/jointsAndActionsData.pkl'))

    def test_cb_run(self):
        from mlpy.mdp.stateaction import Experience, MDPState, MDPAction

        for i in xrange(len(self.data.itervalues().next())):
            for j in xrange(len(self.data.itervalues().next()[0][i]) - 1):
                # noinspection PyTypeChecker
                experience = Experience(MDPState(self.data["states"][i][:, j]), MDPAction(self.data["actions"][i][:, j]),
                                        MDPState(self.data["states"][i][:, j + 1]))
                self.cb.run(self.cb.case_from_data(experience))

    def teardown_method(self, _):
        pass


class TestCASML(object):

    def setup_method(self, _):
        from mlpy.mdp.stateaction import MDPState, MDPAction
        MDPState.nfeatures = None
        MDPAction.description = None
        MDPAction.nfeatures = None

        case_template = {
            "state": {
                "type": "float",
                "value": "data.state",
                "is_index": True,
                "retrieval_method": "knn",
                "retrieval_method_params": 5
            },
            "act": {
                "type": "float",
                "value": "data.action",
                "is_index": False,
                "retrieval_method": "cosine",
            },
            "delta_state": {
                "type": "float",
                "value": "data.next_state - data.state",
                "is_index": False,
            }
        }

        from mlpy.mdp.continuous.casml import CASML
        from mlpy.mdp.continuous import CbTData
        self.model = CASML(CbTData(case_template, tau=1e-5, sigma=1e-5), ncomponents=2)

        from mlpy.auxiliary.io import load_from_file
        data = load_from_file(os.path.join(os.getcwd(), 'tests', 'data/jointsAndActionsData.pkl'))

        # Extract 10th experience for testing
        self.unseen_state = data["states"][0][:, 10]
        self.unseen_action = data["actions"][0][:, 10]
        self.model.fit(np.delete(data["states"][0], 10, 1), np.delete(data["actions"][0], 10, 1))

    def test_predict_proba(self):
        from mlpy.mdp.stateaction import MDPState, MDPAction
        # noinspection PyTypeChecker
        self.model.predict_proba(MDPState(self.unseen_state), MDPAction(self.unseen_action))

    def teardown_method(self, _):
        pass
