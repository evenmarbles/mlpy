from __future__ import division, print_function, absolute_import

import os
import logging
from datetime import datetime

from ..modules.patterns import Singleton


class SilenceableStreamHandler(logging.StreamHandler):
    def __init__(self, *args, **kwargs):
        super(SilenceableStreamHandler, self).__init__(*args, **kwargs)
        self.silenced = False

    def emit(self, record):
        if not self.silenced:
            super(SilenceableStreamHandler, self).emit(record)


class SilenceableFileHandler(logging.FileHandler):
    def __init__(self, *args, **kwargs):
        super(SilenceableFileHandler, self).__init__(*args, **kwargs)
        self.silenced = False

    def emit(self, record):
        if not self.silenced:
            super(SilenceableFileHandler, self).emit(record)


class LoggingMgr(object):
    """The logging manager :class:`.Singleton` class.

    The logger manager can be included as a member to any class to
    manager logging of information. Each logger is identified by
    the module id (`mid`), with which the logger settings can be
    changed.

    By default a logger with log level LOG_INFO that is output to the stdout
    is created.

    Attributes
    ----------
    LOG_TYPE_STREAM=0
        Log only to output stream (stdout).
    LOG_TYPE_FILE=1
        Log only to an output file.
    LOG_TYPE_ALL=2
        Log to both output stream (stdout) and file.
    LOG_DEBUG=10
        Detailed information, typically of interest only when diagnosing problems.
    LOG_INFO=20
        Confirmation that things are working as expected.
    LOG_WARNING=30
        An indication that something unexpected happened, or indicative
        of some problem in the near future. The software is still working as expected.
    LOG_ERROR=40
        Due to a more serious problem, the software has not been able to perform some
        function.
    LOG_CRITICAL=50
        A serious error, indicating that the problem itself may be unable to continue
        running.

    See Also
    --------
    :mod:`logging`

    Examples
    --------
    >>> from mlpy.tools.log import LoggingMgr
    >>> logger = LoggingMgr().get_logger('my_id')
    >>> logger.info('This is a useful information.')

    This gets a new logger. If a logger with the module id `my_id` already exists
    that logger will be returned, otherwise a logger with the default settings is
    created.

    >>> LoggingMgr().add_handler('my_id', htype=LoggingMgr.LOG_TYPE_FILE)

    This adds a new handler for the logger with module id `my_id` writing the logs
    to a file.

    >>> LoggingMgr().remove_handler('my_id', htype=LoggingMgr.LOG_TYPE_STREAM)

    This removes the stream handler from the logger with module id `my_id`.

    >>> LoggingMgr().change_level('my_id', LoggingMgr.LOG_TYPE_ALL, LoggingMgr.LOG_DEBUG)

    This changes the log level for all attached handlers of the logger identified by
    `my_id` to LOG_DEBUG.

    """
    __metaclass__ = Singleton

    LOG_TYPE_STREAM = 0
    LOG_TYPE_FILE = 1
    LOG_TYPE_ALL = 2

    LOG_DEBUG = logging.DEBUG
    LOG_INFO = logging.INFO
    LOG_WARNING = logging.WARNING
    LOG_ERROR = logging.ERROR
    LOG_CRITICAL = logging.CRITICAL

    def __init__(self):
        self._loggers = {}
        self._verbosity = {}
        self._filename = None

    def get_verbosity(self, mid):
        """ Gets the verbosity.

        The current setting of the verbosity of the logger identified
        by `mid` is returned.

        Parameters
        ----------
        mid : str
            The module id of the logger to change the verbosity of.

        Returns
        -------
        bool :
            Whether to turn the verbosity on or off.

        """
        return self._verbosity[mid]

    def set_verbosity(self, mid, value):
        """Sets the verbosity.

        Turn logging on/off for logger identified by `mid`.

        Parameters
        ----------
        mid : str
            The module id of the logger to change the verbosity of.
        value : bool
            Whether to turn the verbosity on or off.

        """
        handlers = self._loggers[mid].handlers
        for hdl in handlers:
            hdl.silenced = value

    def get_logger(self, mid, level=LOG_INFO, htype=LOG_TYPE_STREAM, fmt=None, verbose=True, filename=None):
        """Get the logger instance with the identified `mid`.

        If a logger with the `mid` does not exist, a new logger will be created with the given settings.
        By default only a stream handler is attached to the logger.

        Parameters
        ----------
        mid : str
            The module id of the logger.
        level : int, optional
            The top level logging level. Default is LOG_INFO.
        htype : int, optional
            The logging type of handler. Default is LOG_TYPE_STREAM.
        fmt : str, optional
            The format in which the information is presented.
            Default is "[%(levelname)-8s ] %(name)s: %(funcName)s: %(message)s"
        verbose : bool, optional
            The verbosity setting of the logger. Default is True
        filename : str, optional
            The name of the file the file handler writes the logs to.
            Default is a generated filename.

        Returns
        -------
        The logging instance.

        """
        if mid not in self._loggers:
            logger = logging.getLogger(mid)
            logger.setLevel(level)
            self._loggers[mid] = logger
            self._verbosity[mid] = verbose if verbose is not None else True
            self.add_handler(mid, htype, level, fmt, filename)
        return self._loggers[mid]

    def add_handler(self, mid, htype=LOG_TYPE_STREAM, hlevel=LOG_INFO, fmt=None, filename=None):
        """Add a handler to the logger.

        Parameters
        ----------
        mid : str
            The module id of the logger
        htype : int, optional
            The logging type to add to the handler. Default is LOG_TYPE_STREAM.
        hlevel : int, optional
            The logging level. Default is LOG_INFO.
        fmt : str, optional
            The format in which the information is presented.
            Default is "[%(levelname)-8s ] %(name)s: %(funcName)s: %(message)s"
        filename : str, optional
            The name of the file the file handler writes the logs to.
            Default is a generated filename.

        """
        if fmt is None:
            fmt = "[%(levelname)-8s ] %(name)s: %(funcName)s: %(message)s"
        formatter = logging.Formatter(fmt)

        if htype == self.LOG_TYPE_STREAM or htype == self.LOG_TYPE_ALL:
            handler = SilenceableStreamHandler()
            self._add_handler(mid, hlevel, handler, formatter)

        if htype == self.LOG_TYPE_FILE or htype == self.LOG_TYPE_ALL:
            if self._filename is None:
                if not os.path.exists("logs"):
                    os.makedirs("logs")
                dt = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                self._filename = "logs\logfile " + dt + ".log"
            filename = filename if filename is not None else self._filename

            handler = SilenceableFileHandler(filename)
            self._add_handler(mid, hlevel, handler, formatter)

    def remove_handler(self, mid, htype):
        """Remove handlers.

        Removes all handlers of the given handler type from the logger.

        Parameters
        ----------
        mid : str
            The module id of the logger
        htype : int
            The logging type to remove from the handler.

        """
        handlers = self._loggers[mid].handlers
        for hdl in handlers:
            if htype == self.LOG_TYPE_FILE and isinstance(hdl, logging.FileHandler):
                self._loggers[mid].removeHandler(hdl)
            elif htype == self.LOG_TYPE_STREAM and isinstance(hdl, logging.StreamHandler):
                self._loggers[mid].removeHandler(hdl)

    def change_level(self, mid, hlevel, htype=LOG_TYPE_ALL):
        """Set the log level for a handler.

        Parameters
        ----------
        mid : str
            The module id of the logger
        hlevel : int
            The logging level.
        htype : int, optional
            The logging type of handler for which to change the
            log level. Default is LOG_TYPE_ALL.

        """
        handlers = self._loggers[mid].handlers
        if hlevel < self._loggers[mid].level:
            self._loggers[mid].level = hlevel

        for hdl in handlers:
            if htype == self.LOG_TYPE_ALL:
                hdl.level = hlevel
            elif htype == self.LOG_TYPE_FILE and isinstance(hdl, logging.FileHandler):
                hdl.level = hlevel
            elif htype == self.LOG_TYPE_STREAM and isinstance(hdl, logging.StreamHandler):
                hdl.level = hlevel

    def _add_handler(self, mid, hlevel, handler, formatter):
        handler.setLevel(hlevel)
        handler.setFormatter(formatter)
        self._loggers[mid].addHandler(handler)
