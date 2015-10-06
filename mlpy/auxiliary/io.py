"""
.. module:: mlpy.auxiliary.io
   :platform: Unix, Windows
   :synopsis: I/O utility functions.

.. moduleauthor:: Astrid Jackson <ajackson@eecs.ucf.edu>
"""
from __future__ import division, print_function, absolute_import

import os
import sys
import pickle
import importlib

from .misc import listify


def import_module_from_path(full_path, global_name):
    """Import a module from a file path and return the module object.

    Allows one to import from anywhere, something :func:`__import__()` does not do.
    The module is added to :data:`sys.modules` as `global_name`.

    Parameters
    ----------
    full_path : str
        The absolute path to the module .py file
    global_name : str
        The name assigned to the module in :data:`sys.modules`. To avoid
        confusion, the global_name should be the same as the variable to which
        you're assigning the returned module.

    Examples
    --------
    >>> from mlpy.auxiliary.io import import_module_from_path


    .. note::
        | Project: Code from `Trigger <https://github.com/trigger/trigger>`_.
        | Copyright (c) 2006-2012, AOL Inc.
        | License: `BSD <https://github.com/trigger/trigger/blob/develop/LICENSE.rst>`_

    """
    path, filename = os.path.split(full_path)
    module, ext = os.path.splitext(filename)
    sys.path.append(path)

    try:
        mymodule = __import__(module)
        sys.modules[global_name] = mymodule
    except ImportError:
        raise ImportError('Module could not be imported from %s.' % full_path)
    finally:
        del sys.path[-1]

    return mymodule


def load_from_file(filename, import_modules=None):
    """Load data from file.

    Different formats are supported.

    Parameters
    ----------
    filename : str
        Name of the file to load data from.
    import_modules : str or list
        List of modules that may be required by the data
        that need to be imported.

    Returns
    -------
    dict or list :
        The loaded data. If any errors occur, ``None`` is returned.

    """
    for m in listify(import_modules):
        vars()[m] = importlib.import_module(m)

    try:
        with open(filename, 'rb') as f:
            try:
                data = eval(f.read())
            except TypeError:
                f.seek(0, 0)
                data = pickle.load(f)
            except SyntaxError:
                f.seek(0, 0)
                data = []
                for line in f:
                    data.append(line.strip('\r\n'))
            finally:
                f.seek(0, 0)
        return data
    except TypeError:
        return None


def save_to_file(filename, data):
    """Saves data to file.

    The data can be a dictionary or an object's
    state and is saved in :mod:`pickle` format.

    Parameters
    ----------
    filename : str
        Name of the file to which to save the data to.
    data : dict or object
        The data to be saved.

    """
    if filename is None:
        return

    path = os.path.dirname(filename)
    if path and not os.path.exists(path):
        os.makedirs(path)

    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    except TypeError:
        return


def is_pickle(filename):
    """Check if the file with the given name is :mod:`pickle` encoded.

    Parameters
    ----------
    filename : str
        The name of the file to check.

    Returns
    -------
    bool :
        Whether the file is :mod:`pickle` encoded or not.

    """
    with open(filename, 'rb') as f:
        try:
            pickle.load(f)
            retval = True
        except:
            retval = False
        finally:
            f.seek(0, 0)
    return retval


def txt2pickle(filename, new_filename=None, func=None):
    """Converts a text file into a :mod:`pickle` encoded file.

    Parameters
    ----------
    filename : str
        The name of the file to encode.
    new_filename : str
        New file name to which the encoded data is saved to.
    func : callable
        A data encoding helper function.

    Returns
    -------
    str :
        The name of the file to which the encoded data was saved to.

    """
    data = load_from_file(filename)
    if data is not None:
        if hasattr(func, '__call__'):
            data = func(data)
        path, filename = os.path.split(filename)
        module, ext = os.path.splitext(filename)

        filename = new_filename if new_filename is not None else module + ".pkl"
        filename = path + "/" + filename
        save_to_file(filename, data)
        return filename
    return None
