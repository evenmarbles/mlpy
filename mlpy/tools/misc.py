from __future__ import division, print_function, absolute_import

import sys
import time
import threading
import numpy as np
from mlpy.auxiliary.io import is_pickle, txt2pickle

__all__ = ['Waiting', 'Timer', 'convert_to_policy']


def convert_to_policy(filename, description=None):
    """
    Converts a list of floating point numbers into a list of
    action sequences.

    Loads the file by the given filename containing lists of floating
    point numbers, converts them into a action sequence format, and
    saves to a pickle file.

    Parameters
    ----------
    filename: str
        Name of the file containing the list of floats.
    description: dict, optional
        Description of the action features stored with the policies.
        Default is None.

    Returns
    -------
    str :
        Name of the file to which the action sequences have been
        saved to.

    """
    if not is_pickle(filename):
        def convert(d):
            arr = np.zeros((len(d),), dtype=np.object)
            for i, l in enumerate(d):
                s = np.asarray(eval(l))
                if s.ndim == 1:
                    s.shape = (1, s.shape[0])
                if s.shape[1] == 1:
                    s = np.reshape(s, (-1, s.shape[0]))
                else:
                    s = s.T
                arr[i] = s
            policy = {'act': arr}
            if description is not None:
                policy.update({'act_desc': description})
            return policy

        return txt2pickle(filename, func=convert)
    return filename


class Waiting(threading.Thread):
    """The waiting class.

    The waiting class prints dots (`.`) on stdout to indicate that a process is running.
    The waiting process runs on a different thread to not disturbed the running process.

    Examples
    --------
    >>> def long_process():
    ...     for i in xrange(20):
    ...         pass
    ...
    >>> w = Waiting("processing")
    >>>
    >>> w.start()
    >>> long_process()
    >>> w.stop()
    processing ......

    """
    def __init__(self, text=None):
        super(Waiting, self).__init__()
        self._text = text
        self.event = threading.Event()

    def start(self):
        """Start the process."""
        if not self.isAlive():
            if self._text is not None:
                sys.stdout.write(self._text + ' ')
            super(Waiting, self).start()

    def run(self):
        """Run the process.

        This method is automatically called by the thread.

        """
        while not self.event.wait(0.5):
            sys.stdout.write('.')

    def stop(self):
        """End the process."""
        self.event.set()
        sys.stdout.write('\n')


class Timer(object):
    """Timer class for timing sections of code.

    The timer class follows the context management protocol and
    thus is used with the `with` statement.

    Examples
    --------
    >>> with Timer() as t:
    ...     # code to time here
    >>> print('Request took %.03f sec.' % t.time)

    """

    def __enter__(self):
        self.s0 = time.clock()
        return self

    def __exit__(self, *args):
        self.s1 = time.clock()
        self.time = self.s1 - self.s0
