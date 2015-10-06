from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.stats import bernoulli

from rlglued.environment.environment import Environment
from rlglued.environment import loader as env_loader
from rlglued.utils.taskspecvrlglue3 import TaskSpec
from rlglued.types import Observation
from rlglued.types import Reward_observation_terminal

from mlpy.auxiliary.datastructs import Point2D as Coord
from mlpy.environments.utils.gridworld import Gridworld


# noinspection PyMethodMayBeStatic
class Taxi(Environment):

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    PUTDOWN = 5

    def __init__(self):
        self._non_markov = False
        self._noisy = False
        self._fickle = False

        self._grid = None
        self._landmarks = []
        self._ns = None
        self._ew = None
        self._pass = None
        self._dest = None

    def init(self):
        self._grid = self._create_map()
        self._landmarks.append(Coord(4.0, 0.0))
        self._landmarks.append(Coord(0.0, 3.0))
        self._landmarks.append(Coord(4.0, 4.0))
        self._landmarks.append(Coord(0.0, 0.0))

        ts = TaskSpec(discount_factor=1.0, reward_range=(-10, 20))
        ts.set_episodic()
        ts.add_int_obs((0, 4), repeat=3)
        ts.add_int_obs((0, 3))
        ts.add_int_act((0, 5))

        descr = "STATEDESCR {'descr':['northsouth','eastwest','passenger','destination']} " \
                "ACTIONDESCR {'north':{'value':[0]},'south':{'value': [1]},'east':{'value':[2]},'west':{'value':[3]}," \
                "'pickup':{'value':[4]},'putdown':{'value':[5]}}"
        ts.set_extra(descr + " COPYRIGHT Taxi (Python) implemented by Astrid Jackson.")
        return ts.to_taskspec()

    def start(self):
        self._ns = np.random.randint(0, self._grid.height)
        self._ew = np.random.randint(0, self._grid.width)
        self._pass = np.random.randint(0, len(self._landmarks))
        self._fickle = False

        while True:
            self._dest = np.random.randint(0, len(self._landmarks))
            if self._dest != self._pass:
                break

        return_obs = Observation()
        return_obs.intArray.append(self._ns)
        return_obs.intArray.append(self._ew)
        return_obs.intArray.append(self._pass)
        return_obs.intArray.append(self._dest)
        return return_obs

    def step(self, action):
        return_ro = Reward_observation_terminal()
        return_ro.r = self._apply(action.intArray[0])
        return_ro.terminal = 1 if self._pass == self._dest else 0

        obs = Observation()
        obs.intArray.append(self._ns)
        obs.intArray.append(self._ew)
        obs.intArray.append(self._pass)
        obs.intArray.append(self._dest)
        return_ro.o = obs

        return return_ro

    def cleanup(self):
        self._landmarks = []

    def message(self, msg):
        # # Message Description
        # # 'set-random-seed X'
        # # Action: Set flag to do fixed starting states (row=X, col=Y)
        # if msg.startswith("set-random-seed"):
        #     split_string = msg.split(" ")
        #     self.start_row = int(split_string[1])
        #     self.start_col = int(split_string[2])
        #     self.fixed_start_state = True
        #     return "Message understood.  Using fixed start state."

        return "Taxi (Python) does not respond to that message."

    def _create_map(self):
        nsv = np.zeros((5, 4), dtype=bool)
        ewv = np.zeros((5, 4), dtype=bool)

        x_coords = [0, 0, 1, 1, 3, 4]
        y_coords = [0, 2, 0, 2, 1, 1]
        for x, y in zip(x_coords, y_coords):
            ewv[x][y] = True
        return Gridworld(5, 5, nsv, ewv)

    def _add_noise(self, action):
        if action == Taxi.NORTH or action == Taxi.SOUTH:
            return action if bernoulli.rvs(0.8) else Taxi.EAST if bernoulli.rvs(0.5) else Taxi.WEST
        if action == Taxi.EAST or action == Taxi.WEST:
            return action if bernoulli.rvs(0.8) else Taxi.NORTH if bernoulli.rvs(0.5) else Taxi.SOUTH
        return action

    def _apply_fickle_passanger(self):
        if self._fickle:
            self._fickle = False
            if bernoulli.rvs(0.3):
                self._dest += np.random.randint(0, len(self._landmarks))
                self._dest %= len(self._landmarks)

    def _apply(self, action):
        effect = self._add_noise(action) if self._noisy else action

        if effect == Taxi.NORTH:
            if not self._grid.wall(self._ns, self._ew, effect):
                self._ns += 1
                self._apply_fickle_passanger()
            return -1
        if effect == Taxi.SOUTH:
            if not self._grid.wall(self._ns, self._ew, effect):
                self._ns -= 1
                self._apply_fickle_passanger()
            return -1
        if effect == Taxi.EAST:
            if not self._grid.wall(self._ns, self._ew, effect):
                self._ew += 1
                self._apply_fickle_passanger()
            return -1
        if effect == Taxi.WEST:
            if not self._grid.wall(self._ns, self._ew, effect):
                self._ew -= 1
                self._apply_fickle_passanger()
            return -1
        if effect == Taxi.PICKUP:
            if self._pass < len(self._landmarks) and Coord(
                    self._ns, self._ew) == self._landmarks[self._dest]:
                self._pass = len(self._landmarks)
                self._fickle = self._non_markov and self._noisy
                return -10
            return -1
        if effect == Taxi.PUTDOWN:
            if self._pass == len(self._landmarks) and Coord(
                    self._ns, self._ew) == self._landmarks[self._dest]:
                self._pass = self._dest
                return 20
            return -10

        print("Unreachable point reached in Taxi.apply")
        return 0


if __name__ == "__main__":
    env_loader.load_environment(Taxi())
