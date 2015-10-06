from __future__ import division, print_function, absolute_import

import sys
import socket  # Import socket module
import numpy as np

from threading import Timer

# noinspection PyUnresolvedReferences
from controller import Supervisor
# noinspection PyUnresolvedReferences
from controller import Node
# noinspection PyUnresolvedReferences
from controller import Receiver


class Server(Supervisor):
    TIME_STEP = 64

    RECV_BUFFER = 256

    GOAL_X_LIMIT = 4.5
    GAOL_Z_LIMIT = 0.75

    GOAL_INVALID = -1
    GOAL_FAIL = 0
    GOAL_SUCCESS = 1

    def __init__(self, port):
        """
        Initialization of the supervisor.

        :param port: Port number reserved for service
        :type port: int
        """
        Supervisor.__init__(self)

        self._connections = []

        self._prev_t = 0.0
        self._t = 0.0   # simulation time

        ball = self.getFromDef("BALL")
        self._ball_trans = ball.getField('translation')
        self._prev_ball_pos = np.array(self._ball_trans.getSFVec3f())
        self._ball_pos = self._prev_ball_pos

        try:
            self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error, msg:
            print(msg[1])

        try:
            host = '127.0.0.1'
            self._s.bind((host, port))
        except socket.error, msg:
            print(msg[1])

        self._s.listen(10)
        print("Waiting for a connection on port {0}...".format(port))

        # Set the server socket in non-blocking mode
        self._s.setblocking(0)

    def __del__(self):
        Supervisor.__del__(self)
        print("Cleanup")
        for conn in self._connections:
            conn.close()  # Close the connection
        self._s.close()

    def run(self):
        while True:
            # Perform a simulation step of 64 milliseconds
            # and leave the loop when the simulation is over
            if self.step(Server.TIME_STEP) == -1:
                break

            self._prev_t = self._t
            self._t += Server.TIME_STEP / 1000.0

            self._accept_connection()

            reset = False
            for conn in self._connections:
                try:
                    data = conn.recv(Server.RECV_BUFFER)
                    if not data:
                        continue
                    print("Received: ", data)

                    if data == "request reset":
                        reset = True
                    elif data == "check goal":
                        t = Timer(1 / 1000.0, self._check_goal, [conn])
                        t.start()
                except socket.error:
                    pass

            if reset:
                for conn in self._connections:
                    conn.send("reset requested")
                    self.simulationRevert()

            self._prev_ball_pos = self._ball_pos
            self._ball_pos = np.array(self._ball_trans.getSFVec3f())

    def _is_goal(self):
        is_goal = Server.GOAL_INVALID
        if self._ball_pos[0] > Server.GOAL_X_LIMIT:
            if -Server.GAOL_Z_LIMIT < self._ball_pos[2] < Server.GAOL_Z_LIMIT:
                is_goal = Server.GOAL_SUCCESS
            else:
                is_goal = Server.GOAL_FAIL
        return is_goal

    def _check_goal(self, conn):
        is_goal = Server.GOAL_INVALID
        while is_goal == Server.GOAL_INVALID:
            if self._t <= self._prev_t:
                if self._ball_pos[0] == self._prev_ball_pos[0] and self._ball_pos[1] == self._prev_ball_pos[1] \
                        and self._ball_pos[2] == self._prev_ball_pos[2]:
                    is_goal = Server.GOAL_FAIL
            else:
                if self._t - self._prev_t == 0:
                    print("Why is time diff equal to zero?")
                # noinspection PyTypeChecker
                ball_vel = np.divide((self._ball_pos - self._prev_ball_pos), (self._t - self._prev_t))
                ball_speed = np.linalg.norm(ball_vel)

                if ball_vel[0] < 0:     # ball is moving away from goal
                    is_goal = Server.GOAL_FAIL
                elif ball_speed < 0.0005:   # ball movement has slowed below threshold
                    is_goal = self._is_goal()
                else:
                    is_goal = self._is_goal()

            if is_goal != Server.GOAL_INVALID:
                print("is_goal=" + str(is_goal))
                if is_goal:
                    conn.send("success")
                else:
                    conn.send("failure")

    def _accept_connection(self):
        try:
            conn, addr = self._s.accept()
            self._connections.append(conn)
            # noinspection PyStringFormat
            print('Client (%s, %s) connected' % addr)
        except socket.error:
            pass


def main(argv):
    """
    Main entry point
    """
    if len(argv) < 1:
        print(usage())
        sys.exit(0)

    port = int(argv[0])

    controller = Server(port)
    controller.run()


def usage():
    return "Please specify the SUPERVISOR_PORT_NUMBER in the 'controllerArgs' field of the Supervisor robot."


if __name__ == "__main__":
    main(sys.argv[1:])
