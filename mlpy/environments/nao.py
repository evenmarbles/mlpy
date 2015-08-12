from __future__ import division, print_function, absolute_import

import sys
import traceback
import time
import socket

from naoqi import ALProxy

from ..environments import Environment

__all__ = ['NaoEnvFactory', 'PhysicalWorld', 'Webots']


class NaoEnvFactory(object):
    """The NAO environment factory.

    An instance of a NAO environment can be created by passing
    the environment type.

    Notes
    -----
    Currently only soccer related events are handled by the Nao worlds. To
    be more specific the environment attempts to detect whether a goal was
    scored. The physical world will prompt the user to reset the experiment
    while the supervisor in the Webots world is responsible for resetting
    the environment for the next experiment.

    Examples
    --------
    >>> from mlpy.environments.nao import NaoEnvFactory
    >>> NaoEnvFactory.create('nao.physicalworld')

    This creates a :class:`.PhysicalWorld` instance controlling
    agents in the real world.

    >>> NaoEnvFactory.create('nao.webots', 12345)

    This creates a :class:`.Webots` instance controlling simulated
    agents in Webots. The port '12345' is the port the controller of
    the supervisor in the Webots world listens to.

    """
    @staticmethod
    def create(_type, *args, **kwargs):
        """Create a Nao environment of the given type.

        Parameters
        ----------
        _type : str
            The Nao environment type. Valid environment types:

            nao.physicalworld
                This controls the robots in the real world. The environment
                interacts with the user to inquire about events happening
                in the real world. A :class:`PhysicalWorld` instance is created.

            nao.webots
                This controls the simulated robots in the Webots simulator.
                The world in the simulator should include a supervisor using
                a controller to handle the required events. A sample controller
                can be found in `environments/webots/controllers/serverc`. A
                :class:`.Webots` instance is created.

        args : tuple, optional
            Positional arguments to pass to the class of the given type for
            initialization.
        kwargs : dict, optional
            Non-positional arguments to pass to the class of the given type
            for initialization.

        Returns
        -------
        Environment
            A Nao environment instance of the given type.

        """
        try:
            return {
                "nao.physicalworld": PhysicalWorld,
                "nao.webots": Webots,
            }[_type](* args, **kwargs)

        except KeyError:
            return None


class PhysicalWorld(Environment):
    """The physical (real) environment.

    Parameters
    ----------
    agents : Agent or list[Agent], optional
        A list of agents that act in the environment.

    Notes
    -----
    The agents are acting in the real world. To capture events happening
    in the real world the user is prompted to provide the information.

    """

    def __init__(self, agents=None):
        super(PhysicalWorld, self).__init__(agents)

    def __str__(self):
        return "Physical World"

    def reset(self, t, **kwargs):
        """Reset the environment and all agents.

        The user is prompted to reset the environment (i.e., experiment).
        Ones the user has reset the environment all agents are reset.

        Parameters
        ----------
        t : float
            The current time (sec)
        kwargs : dict, optional
            Non-positional parameters, optional.

        """
        if raw_input("Please reset your experiment. Press ENTER to continue. >>>"):
            pass

        super(PhysicalWorld, self).reset(t, **kwargs)

    # noinspection PyMethodMayBeStatic
    def check_data(self, value):
        """Request to check for data.

        Parameters
        ----------
        value : str
            The request identifier.

        Returns
        -------
        The result returned by the user.

        Notes
        -----
        When checking for data, the user is prompted to provide the
        information via the console.

        """
        if value == "check goal":
            user_input = None
            while user_input not in ["0", "1"]:
                user_input = raw_input("Goal scored? [\"goal scored\": 1; \"goal missed\": 0] >>> ")
            return "success" if user_input == "1" else "fail"


class Webots(Environment):
    """Simulated environment using the Webots simulator.

    The Webots environment works in conjunction with a controller
    specified for a supervisor. A sample controller can be found in
    `webots/controllers/serverc`. This controller listens on port
    `12345` for the following events:

        request reset
            Requests an environment reset from the controller.

        check goal
            Requests from the controller a check whether a goal
            was scored or not. The result of that check is send
            back to the client.

    Parameters
    ----------
    port : int, optional
        The port the controller listens to. If using the environment
        in conjunction with controller `serverc` the port number is
        `12345`. Default is 12345.
    agents : Agent or list[Agent], optional
        A list of agents that act in the environment.

    Notes
    -----
    When requested to reset, the environment will request to reset
    the simulated environment in Webots from the controller.
    It is also possible to check if a goal was scored by calling
    the function :meth:`check_data` with the argument 'check goal'.
    This sends a request to the controller to check if a goal was
    scored.

    .. attention::
        The Webots environment class requires the `NAOqi <http://doc.aldebaran.com/2-1/index.html>`_
        API from Aldebaran be installed on your machine. A separate license is
        required for the API.

    """
    RECV_BUFFER = 256

    def __init__(self, port=12345, agents=None):
        super(Webots, self).__init__(agents)

        self._modules = ["ALMemory",
                         "ALMotion",
                         "ALRobotPosture",
                         "ALVideoDevice"]

        self._port = port
        self._sock = None

    def __str__(self):
        return "Webots Pro version 8.0"

    def reset(self, t, **kwargs):
        """Reset the environment and all agents.

        A request is send to the controller to reset the
        environment. Once the environment is reset all
        agents acting in the environment are reset.

        Parameters
        ----------
        t : float
            The current time (sec)
        kwargs : dict, optional
            Non-positional parameters, optional.

        """
        if self._sock:
            self._sock.send("request reset")

            done = False
            while not done:
                try:
                    data = self._sock.recv(Webots.RECV_BUFFER)
                    self._log(data)
                    if data == "reset requested":
                        time.sleep(5)
                        self._sock = self._connect()
                        done = True
                except socket.error:
                    pass

        super(Webots, self).reset(t, **kwargs)

    def is_complete(self):
        """Checks if the environment has completed.

        This is dependent on whether the agent(s) have
        completed their task.

        Returns
        -------
        bool
            Whether the environment has reached some end goal.

        """
        is_complete = super(Webots, self).is_complete()
        # TODO: check for pending server requests
        return is_complete

    def enter(self, t):
        """Enter the environment and all agents.

        Parameters
        ----------
        t : float
            The current time (sec).

        """
        self._sock = self._connect()

        super(Webots, self).enter(t)

    def exit(self):
        """Exit the environment and all agents.

        Perform cleanup tasks here.

        """
        super(Webots, self).exit()

        if self._sock:
            self._sock.close()

    def check_data(self, value):
        """Request to check for data.

        Parameters
        ----------
        value : str
            The request identifier.

        Returns
        -------
        The result returned by the controller.

        Notes
        -----
        The Webots environment works in conjunction with a controller
        specified for a supervisor. When checking for data, a request
        with the `value` is send to the controller which extracts the
        information and returns the results.

        """
        self._sock.send(value)
        while True:
            try:
                return self._sock.recv(Webots.RECV_BUFFER)
            except socket.error:
                pass

    def _connect(self):
        """ Connects to the server (controller). """
        self._log("connecting...")
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            sys.exit(1)

        host = socket.gethostname()

        connected = False
        while not connected:
            try:
                s.connect((host, self._port))
                s.setblocking(0)
                while True:
                    ready = True
                    for m in self._modules:
                        # noinspection PyBroadException
                        try:
                            for agent in self._agents.itervalues():
                                ALProxy(m, agent.pip, agent.pport)
                        except:
                            ready = False
                            break
                    if ready:
                        break
                    time.sleep(1)
                connected = True
            except socket.error:
                pass

        self._log("connected")
        return s

    def _log(self, text):
        print(str(self) + ": " + text)
