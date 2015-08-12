"""
.. module:: mlpy.auxiliary.misc
   :platform: Unix, Windows
   :synopsis: Utility functions.

.. moduleauthor:: Astrid Jackson <ajackson@eecs.ucf.edu>
"""
from __future__ import division, print_function, absolute_import

import os
import sys
from contextlib import contextmanager


def remove_key(d, key):
    """Safely remove the `key` from the dictionary.

    Safely remove the `key` from the dictionary `d` by first
    making a copy of dictionary. Return the new dictionary together
    with the value stored for the `key`.

    Parameters
    ----------
    d : dict
        The dictionary from which to remove the `key`.
    key :
        The key to remove

    Returns
    -------
    v :
        The value for the key
    r : dict
        The dictionary with the key removed.

    """
    r = dict(d)
    v = r[key]
    del r[key]
    return v, r


def listify(obj):
    """Ensure that the object `obj` is of type list.

    If the object is not of type `list`, the object is
    converted into a list.

    Parameters
    ----------
    obj :
        The object.

    Returns
    -------
    list :
        The object inside a list.

    """
    if obj is None:
        return []

    return obj if isinstance(obj, (list, type(None))) else [obj]


@contextmanager
def stdout_redirected(to=os.devnull):
    """Preventing a C shared library to print on stdout.

    Examples
    --------
    >>> import os
    >>>
    >>> with stdout_redirected(to="filename"):
    >>>    print("from Python")
    >>>    os.system("echo non-Python applications are also supported")

    .. note::
        | Project: Code from `StackOverflow <http://stackoverflow.com/a/17954769>`_.
        | Code author: `J.F. Sebastian <http://stackoverflow.com/users/4279/j-f-sebastian>`_
        | License: `CC-Wiki <http://creativecommons.org/licenses/by-sa/3.0/>`_

    """
    fd = sys.stdout.fileno()

    # noinspection PyShadowingNames
    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)    # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w')     # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as f:
            _redirect_stdout(to=f)
        try:
            yield   # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)     # restore stdout, buffering and flags such as CLOEXEC may be different
