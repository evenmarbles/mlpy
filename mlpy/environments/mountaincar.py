import numpy as np
import matplotlib.pyplot as plt

from rlglued.environment.environment import Environment
from rlglued.environment import loader as env_loader
from rlglued.utils.taskspecvrlglue3 import TaskSpec
from rlglued.types import Observation
from rlglued.types import Reward_observation_terminal


class MountainCar(Environment):

    def __init__(self, noise=0.0, reward_noise=0.0, random_start=False, visualize=False):
        self._noise = noise
        self._reward_noise = reward_noise
        self._random_start = random_start

        self._visualize = visualize
        self._fig = None
        self._ax = None
        self._t = np.arange(-1.2, .6, .01)

        self._sensors = np.zeros(2)
        self._limits = np.array([[-1.2, .6], [-.07, .07]])
        self._goal_pos = 0.5
        self._accel = .001
        self._gravity = -.0025
        self._hill_freq = 3.
        self._dt = 1.

    def init(self):
        ts = TaskSpec(discount_factor=1.0, reward_range=(-1.0, 0.0))
        ts.set_episodic()

        for i, (min_, max_) in enumerate(self._limits):
            if min_ == -np.inf:
                min_ = 'NEGINF'
            if max_ == np.inf:
                max_ = 'POSINF'
            ts.add_double_obs((min_, max_))

        ts.add_int_act((0, 2))

        extra = " COPYRIGHT Mountain Car (Python) implemented by Astrid Jackson."
        state_descr = "STATEDESCR {'descr':['car position','car velocity']}"
        action_descr = "ACTIONDESCR {'forward':{'value':[1.]},'backward':{'value':[-1.]},'no throttle':{'value':[0.]}}"

        ts.set_extra(state_descr + " " + action_descr + extra)
        return ts.to_taskspec()

    def start(self):
        if self._random_start:
            self._sensors = np.random.random(self._sensors.shape)
            self._sensors *= (self._limits[:, 1] - self._limits[:, 0])
            self._sensors += self._limits[:, 0]
        else:
            self._sensors = np.zeros(self._sensors.shape)
            self._sensors[0] = -0.5

        self._render(self._sensors[0])

        return_obs = Observation()
        return_obs.doubleArray = self._sensors.tolist()
        return return_obs

    def step(self, action):
        return_ro = Reward_observation_terminal()
        self._apply(action)
        self._render(self._sensors[0])

        return_ro.terminal = self._is_terminal()

        return_ro.r = -1.
        if return_ro.terminal:
            return_ro.r = .0

        if self._reward_noise > 0:
            return_ro.r += np.random.normal(scale=self._reward_noise)

        obs = Observation()
        obs.doubleArray = self._sensors.tolist()
        return_ro.o = obs

        return return_ro

    def cleanup(self):
        pass

    def message(self, msg):
        return "I don't know how to respond to your message"

    def _apply(self, action):
        direction = action.intArray[0]
        if direction == 0:
            direction = -1
        elif direction == 1:
            direction = 0
        else:
            direction = 1
        direction += self._accel * np.random.normal(scale=self._noise) if self._noise > 0 else 0.0

        self._sensors[1] += (self._accel * direction) + (self._gravity * np.cos(self._hill_freq * self._sensors[0]))
        self._sensors[1] = self._sensors[1].clip(min=self._limits[1, 0], max=self._limits[1, 1])
        self._sensors[0] += self._dt * self._sensors[1]
        self._sensors[0] = self._sensors[0].clip(min=self._limits[0, 0], max=self._limits[0, 1])

    def _is_terminal(self):
        return self._sensors[0] >= self._goal_pos

    def _render(self, pos):
        if self._visualize:
            if self._fig is None or not plt.fignum_exists(self._fig.number):
                self._fig = plt.figure()
                plt.rcParams['legend.fontsize'] = 10
                self._ax = self._fig.add_subplot(1, 1, 1)
                self._fig.show()

            self._ax.cla()
            self._ax.plot(self._t, np.sin(3 * self._t))

            car = plt.Circle((pos, np.sin(3 * pos)), radius=0.02, fc='r')
            plt.gca().add_patch(car)

            self._fig.canvas.draw()


if __name__ == "__main__":
    env_loader.load_environment(MountainCar())
