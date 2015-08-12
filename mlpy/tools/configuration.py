"""
.. module:: mlpy.auxiliary.configuration
   :platform: Unix, Windows
   :synopsis: Manages configurations read in JSON format.

.. moduleauthor:: Astrid Jackson <ajackson@eecs.ucf.edu>
"""
from __future__ import division, print_function, absolute_import

from ..auxiliary.io import load_from_file
from ..auxiliary.misc import listify


def _decode_list(data):
    rv = []
    for item in data:
        if isinstance(item, unicode):
            item = item.encode('utf-8')
        elif isinstance(item, list):
            item = _decode_list(item)
        elif isinstance(item, dict):
            item = _decode_dict(item)
        rv.append(item)
    return rv


def _decode_dict(data):
    rv = {}
    for key, value in data.iteritems():
        if isinstance(key, unicode):
            key = key.encode('utf-8')
        if isinstance(value, unicode):
            value = value.encode('utf-8')
        elif isinstance(value, list):
            value = _decode_list(value)
        elif isinstance(value, dict):
            value = _decode_dict(value)
        rv[key] = value
    return rv


class ConfigMgr(object):
    """The configuration manager.

    The configuration manager provides access to configuration files (usually
    in `JSON <http://json.org/>`_ format) for client applications.

    Parameters
    ----------
    filename : str
        The name of the configuration file.
    import_modules : str or list[str]
        Modules required by the configuration file that must
        be imported first.
    eval_key : bool
        Whether to evaluate the key. If this is `True`, the key will be
        evaluated as a statement by a call to :func:`eval()`.

    Raises
    ------
    TypeError
        If the configuration file is not read in a dictionary

    Examples
    --------
    Assuming there exists a file ``events_map.json`` containing the following
    configuration:

    ::

        {
            "keyboard": {
                "down": {
                    "pygame.K_ESCAPE": "QUIT",
                    "pygame.K_SPACE": [-1.0],
                    "pygame.K_LEFT" : [-0.004],
                    "pygame.K_RIGHT":  [0.004]
                }
            }
        }

    The keys can be mapped to the `PyGame <http://www.pygame.org/>`_ keyboard
    constants, when the file is loaded by the configuration manager as follows:

    >>> cfg = ConfigMgr("events_map.json", "pygame", eval_key=True)

    This allows to retrieve the values for the keys in the configuration file
    by using the `PyGame <http://www.pygame.org/>`_ keyboard constants, which
    are returned in the `key` attribute of the `PyGame <http://www.pygame.org/>`_
    event:

    >>> import pygame
    >>> for event in pygame.event.get():
    >>>     print cfg.get("keyboard.down." + str(event.key))

    """
    def __init__(self, filename, import_modules=None, eval_key=None):
        self._config = load_from_file(filename, import_modules)
        if not isinstance(self._config, dict):
            raise TypeError("The configuration file must be of type `dict`.")

        eval_key = eval_key if eval_key is not None else False
        if eval_key:
            self._eval_key(self._config, import_modules)

    def get(self, key):
        """Return the value for the given key.

        Parameters
        ----------
        key : str
            The key for the configuration. Concatenate keys by dots (`.`)
            to access keys at deeper levels in the configuration.

        Raises
        ------
        KeyError
            If the key does not exist in the configuration

        """
        key_list = key.split(".")
        config = self._config

        for k in key_list:
            if k in config:
                # noinspection PyTypeChecker
                config = config[k]
            else:
                raise KeyError(k)
        return config

    def has_config(self, key):
        """Checks if the given key exists in the configuration.

        Parameters
        ----------
        key : str
            The key for the configuration. Concatenate keys by dots (``.``)
            to access keys at deeper levels in the configuration.

        Returns
        -------
        bool :
            Whether the key exists or not.

        """
        key_list = key.split(".")
        config = self._config

        for k in key_list:
            if k in config:
                # noinspection PyTypeChecker
                config = config[k]
            else:
                return False
        return True

    def _eval_key(self, d, import_modules):
        """Evaluate the key by calling eval().

        Parameters
        ----------
        d : dict
            The configuration dictionary.
        import_modules : str or list[str]
            The modules that must be imported first.

        """
        def items():
            import importlib
            for m in listify(import_modules):
                vars()[m] = importlib.import_module(m)

            for key, value in d.items():
                try:
                    d[eval("str(" + key + ")")] = d.pop(key)
                except NameError:
                    d[key] = value

                if isinstance(value, dict):
                    self._eval_key(value, import_modules)

        items()
