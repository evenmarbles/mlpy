"""
=====================================================
Case base reasoning (:mod:`mlpy.knowledgerep.cbr`)
=====================================================

.. currentmodule:: mlpy.knowledgerep.cbr


Engine
======

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~engine.CaseMatch
   ~engine.Case
   ~engine.CaseBaseEntry
   ~engine.CaseBase


Features
========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~features.FeatureFactory
   ~features.Feature
   ~features.BoolFeature
   ~features.StringFeature
   ~features.IntFeature
   ~features.FloatFeature


Similarity measures
===================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~similarity.Stat
   ~similarity.SimilarityFactory
   ~similarity.ISimilarity
   ~similarity.NeighborSimilarity
   ~similarity.KMeansSimilarity
   ~similarity.ExactMatchSimilarity
   ~similarity.CosineSimilarity


Problem solving methods
=======================

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~methods.CBRMethodFactory
   ~methods.ICBRMethod
   ~methods.IReuseMethod
   ~methods.IRevisionMethod
   ~methods.IRetentionMethod
   ~methods.DefaultReuseMethod
   ~methods.DefaultRevisionMethod
   ~methods.DefaultRetentionMethod

"""
from __future__ import division, print_function, absolute_import

__all__ = ['engine', 'features', 'methods', 'similarity']
