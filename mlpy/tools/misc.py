from __future__ import division, print_function, absolute_import

import sys
import threading


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
        if self._text is not None:
            sys.stdout.write(self._text + ' ')
        super(Waiting, self).start()

    def run(self):
        """Run the process.

        This method is automatically called by the thead.

        """
        while not self.event.wait(0.5):
            sys.stdout.write('.')

    def stop(self):
        """End the process."""
        self.event.set()
        sys.stdout.write('\n')
