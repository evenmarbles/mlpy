"""
==============================================
Auxiliary functions (:mod:`mlpy.auxiliary`)
==============================================

.. currentmodule:: mlpy.auxiliary

This modules.

Array
=====

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~array.accum
   ~array.normalize
   ~array.nunique


I/O
===

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~io.import_module_from_path
   ~io.load_from_file
   ~io.save_to_file
   ~io.is_pickle
   ~io.txt2pickle


Data structures
===============

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~datastructs.Array
   ~datastructs.Point2D
   ~datastructs.Point3D
   ~datastructs.Vector3D
   ~datastructs.Queue
   ~datastructs.FIFOQueue
   ~datastructs.PriorityQueue


Data sets
=========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~datasets.DataSet


Miscellaneous
=============

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~misc.remove_key
   ~misc.listify
   ~misc.stdout_redirected
   ~misc.columnize


Plotting
========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~plotting.Arrow3D

"""
from __future__ import division, print_function, absolute_import

__all__ = ['array', 'datasets', 'datastructs', 'io', 'misc', 'plotting']
