from __future__ import division, print_function, absolute_import

import sys
import time
import socket


kLocalHost = "127.0.0.1"
kDefaultPort = 12345
kRetryTimeout = 2


class WebotsClient(object):
    """Webots client.

    The Webots client works in conjunction with a controller
    specified for a supervisor. A sample controller can be found in
    `webots/controllers/serverc`. This controller listens on port
    `12345` for the following events:

        request reset
            Requests an environment reset from the controller.

        check goal
            Requests from the controller a check whether a goal
            was scored or not. The result of that check is send
            back to the client.

    Notes
    -----
    When requested to reset, the client will request to reset
    the simulated environment in Webots from the controller.
    It is also possible to check if a goal was scored by calling
    the function :meth:`query` with the argument 'check goal'.
    This sends a request to the controller to check if a goal was
    scored.

    """
    RECV_BUFFER = 256

    def __init__(self):
        super(WebotsClient, self).__init__()

        self._sock = None

    def __str__(self):
        return "Webots Pro version 8.0"

    def connect(self, host=kLocalHost, port=kDefaultPort, retry_timeout=kRetryTimeout):
        """Connect to the server (controller).

        Parameters
        ----------
        host : str, optional
            The host the controller listens to. Default is `127.0.0.1`
        port : int, optional
            The port the controller listens to. If using the client
            in conjunction with controller `serverc` the port number is
            `12345`. Default is 12345.
        retry_timeout: int, optional
            The time before retrying to connect in seconds. Default is 2.

        """
        print(str(self))
        print("\tConnecting to " + host + " on port " + str(port) + "...")
        sys.stdout.flush()

        while self._sock is None:
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._sock.connect((host, port))
                self._sock.setblocking(0)
            except socket.error, msg:
                self._sock = None
                time.sleep(retry_timeout)
            else:
                break
        print("\t Connected to Webots")

    def close(self):
        """Close the connection."""
        if self._sock:
            self._sock.close()

    def reset(self, host=kLocalHost, port=kDefaultPort, retry_timeout=kRetryTimeout):
        """Reset the environment and all agents.

        A request is send to the controller to reset the
        environment. Once the environment is reset all
        agents acting in the environment are reset.

        Parameters
        ----------
        host : str, optional
            The host the controller listens to. Default is `127.0.0.1`
        port : int, optional
            The port the controller listens to. If using the client
            in conjunction with controller `serverc` the port number is
            `12345`. Default is 12345.
        retry_timeout: int, optional
            The time before retrying to connect in seconds. Default is 2.

        """
        if self._sock is not None:
            self._sock.send("request reset")

            done = False
            while not done:
                try:
                    data = self._sock.recv(WebotsClient.RECV_BUFFER)
                    self._log(data)
                    if data == "reset requested":
                        time.sleep(5)
                        self._sock = None
                        self.connect(host, port, retry_timeout)
                        done = True
                except socket.error:
                    pass
        else:
            self.connect(host, port, retry_timeout)

    def query(self, msg):
        """Query the server (controller).

        Parameters
        ----------
        msg : str
            The message send to the controller.

        Returns
        -------
        The result returned by the controller.

        Notes
        -----
        The Webots environment works in conjunction with a controller
        specified for a supervisor. When querying the controller, a request
        with the `msg` is send to the controller which extracts the
        information and returns the results.

        """
        self._sock.send(msg)
        while True:
            try:
                return self._sock.recv(WebotsClient.RECV_BUFFER)
            except socket.error:
                pass

    def _log(self, text):
        print(str(self) + ": " + text)
