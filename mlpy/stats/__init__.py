"""
============================================
Statistical functions (:mod:`mlpy.stats`)
============================================

.. currentmodule:: mlpy.stats


Discrete distributions
======================

.. toctree::
   :hidden:

   generated/mlpy.stats.nonuniform
   generated/mlpy.stats.gibbs

============================  =================================================================
:data:`.nonuniform`           A non-uniform discrete random variable.
:data:`.gibbs`                A Gibbs distribution discrete random variable.
============================  =================================================================


Conditional distributions
=========================

.. toctree::
   :hidden:

   generated/mlpy.stats.conditional_normal
   generated/mlpy.stats.conditional_student
   generated/mlpy.stats.conditional_mix_normal

===============================  =================================================================
:data:`.conditional_normal`      Conditional Normal random variable.
:data:`.conditional_student`     Conditional Student random variable.
:data:`.conditional_mix_normal`  Conditional Mix-Normal random variable.
===============================  =================================================================


Multivariate distributions
==========================

.. toctree::
   :hidden:

   generated/mlpy.stats.multivariate_normal
   generated/mlpy.stats.multivariate_student
   generated/mlpy.stats.invwishart
   generated/mlpy.stats.normal_invwishart

=============================  =================================================================
:data:`.multivariate_normal`   Multivariate Normal random variable.
:data:`.multivariate_student`  Multivariate Student random variable.
:data:`.invwishart`            Inverse Wishart random variable.
:data:`.normal_invwishart`     Normal-Inverse Wishart random variable.
=============================  =================================================================


Statistical Models
==================

.. toctree::
   :hidden:

   generated/mlpy.stats.models.markov

=============================  =================================================================
:data:`~.models.markov`         Markov model.
=============================  =================================================================


Mixture Models
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~models.mixture.MixtureModel
   ~models.mixture.DiscreteMM
   ~models.mixture.GMM
   ~models.mixture.StudentMM


Statistical functions
=====================

.. toctree::
   :glob:
   :hidden:

   generated/mlpy.stats.canonize_labels
   generated/mlpy.stats.is_posdef
   generated/mlpy.stats.normalize_logspace
   generated/mlpy.stats.partitioned_cov
   generated/mlpy.stats.partitioned_mean
   generated/mlpy.stats.partitioned_sum
   generated/mlpy.stats.randpd
   generated/mlpy.stats.shrink_cov
   generated/mlpy.stats.sq_distance
   generated/mlpy.stats.stacked_randpd

============================  =================================================================
:func:`.is_posdef`            Test if matrix `a` is positive definite.
:func:`.randpd`               Create a random positive definite matrix.
:func:`.stacked_randpd`       Create multiple random positive definite matrices.
:func:`.normalize_logspace`   Normalize in log space while avoiding numerical underflow.
:func:`.sq_distance`          Efficiently compute squared Euclidean distances between stats of vectors.
:func:`.partitioned_cov`      Partition the rows of `x` according to `y` and take the covariance of each group.
:func:`.partitioned_mean`     Groups the rows of x according to the class labels in y and takes the mean of each group.
:func:`.partitioned_sum`      Groups the rows of x according to the class labels in y and sums each group.
:func:`.shrink_cov`           Ledoit-Wolf optimal shrinkage estimator.
:func:`.canonize_labels`      Transform labels to 1:k.
============================  =================================================================

"""
from ._stats import *
from ._discrete import *
from ._multivariate import *
from ._conditional import *

__all__ = [s for s in dir() if not (s.startswith('_') or s.endswith('cython'))]
__all__ += ['dbn', 'models']
