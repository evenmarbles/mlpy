"""
====================================================
Modules and design patterns (:mod:`mlpy.modules`)
====================================================

.. currentmodule:: mlpy.modules

This module contains various modules and design patterns.

Modules
=======

.. autosummary::
   :toctree: generated/
   :nosignatures:

   UniqueModule
   Module


Patterns
========

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~patterns.Borg
   ~patterns.Observable
   ~patterns.Listener

Meta classes
------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~patterns.Singleton
   ~patterns.RegistryInterface

"""
from __future__ import division, print_function, absolute_import

from itertools import count
from ..auxiliary.io import load_from_file, save_to_file

__all__ = ['patterns']


class UniqueModule(object):
    """Class ensuring each instance has a unique name.

    The unique id can either be passed to the class or if none
    is passed, it will be generated using the module and class name.

    Parameters
    ----------
    mid : str
        The module's unique identifier

    Examples
    --------
    >>> from mlpy.modules import UniqueModule
    >>> class MyClass(UniqueModule):
    >>>
    >>>     def __init__(self, mid=None):
    >>>         super(MyClass, self).__init__(mid)

    This creates a unique model.

    """
    _ids = count(0)

    @property
    def mid(self):
        """The module's unique identifier.

        Returns
        -------
        str :
            The module's unique identifier
        """
        return self._mid

    @mid.setter
    def mid(self, value):
        self._mid = value

    def __init__(self, mid=None):
        self._mid = mid if mid is not None else self._generate()

    def __repr__(self):
        return "<%s '%s'>" % (self.__class__.__name__, self._mid)

    def __setstate__(self, d):
        self._mid = self._generate()

    def _generate(self):
        """Generate unique identifier."""
        return "%s.%s:%i" % (self.__class__.__module__, self.__class__.__name__, next(self._ids))

    @classmethod
    def load(cls, filename):
        """Load the state of the module from file.

        Parameters
        ----------
        filename : str
            The name of the file to load from.

        Notes
        -----
        This is a class method, it can be accessed without instantiation.

        """
        module = load_from_file(filename)

        mod_name = module.__module__ + "." + module.__class__.__name__
        cls_name = cls.__module__ + "." + cls.__name__
        if not mod_name == cls_name:
            raise ValueError("File '%s' does not store valid type '%s': %s" % (filename, cls_name, mod_name))

        return module

    def save(self, filename):
        """Save the current state of the module to file.

        Parameters
        ----------
        filename : str
            The name of the file to save to.

        """
        save_to_file(filename, self)


class Module(UniqueModule):
    """Base module class from which most modules inherit from.

    The base module class handles processing of the program
    loop. A module inherits from the unique module class, thus
    every module has a unique name.

    Parameters
    ----------
    mid : str
        The module's unique identifier

    Examples
    --------
    To create a module handling the program loop, write

    >>> from mlpy.modules import Module
    >>> class MyClass(Module):
    >>>     pass

    """
    def __init__(self, mid=None):
        super(Module, self).__init__(mid)
        self._t = 0.0
        """type: float"""

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def reset(self, t, **kwargs):
        """Reset the module.

        Parameters
        ----------
        t : float
            The current time (sec)
        kwargs : dict
            Additional non-positional parameters.

        """
        self._t = t

    def enter(self, t):
        """Enter the module and perform initialization tasks.

        Parameters
        ----------
        t : float
            The current time (sec)

        """
        self._t = t

    def update(self, dt):
        """Update the module at every delta time step dt.

        Parameters
        ----------
        dt : float
            The elapsed time (sec)

        """
        self._t += dt

    # noinspection PyMethodMayBeStatic
    def exit(self):
        """Exit the module and perform cleanup tasks."""
        pass
