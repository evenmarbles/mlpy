from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import math
import numpy as np

from rlglued.environment.environment import Environment
from rlglued.environment import loader as environment_loader
from rlglued.utils.taskspecvrlglue3 import TaskSpec
from rlglued.types import Observation
from rlglued.types import Reward_observation_terminal


# /**
#  *  This is a very simple environment with discrete observations corresponding to states labeled {0,1,...,19,20}
#     The starting state is 10.
# 
#     There are 2 actions = {0,1}.  0 decrements the state, 1 increments the state.
# 
#     The problem is episodic, ending when state 0 or 20 is reached, giving reward -1 or +1, respectively.
#     The reward is 0 on all other steps.
#  * @author Brian Tanner
#  */

class CartPole(Environment):
    def __init__(self, mode='easy', pole_len=None, pole_mass=None, max_force=10, noise=0.0, reward_noise=0.0,
                 discrete_state=False, discrete_action=False, random_start=True, cart_loc=None, cart_vel=None,
                 pole_angle=None, pole_vel=None, mu_c=None, mu_p=None, simsteps=10, discount_factor=0.999,
                 dim_size=None, include_friction=True):
        if mode not in ['easy', 'hard', 'swingup', 'custom']:
            raise ValueError("Unsupported mode '{0}'.".format(mode))

        self._noise = noise
        self._reward_noise = reward_noise
        self._random_start = random_start
        self._discrete_state = discrete_state
        self._discrete_action = discrete_action

        self._dim_size = dim_size

        self._include_friction = include_friction

        self._gravity = -9.81           # Gravity in m/s^2
        self._cart_mass = 1.            # Weight of the cart in kg
        self._max_force = max_force     # Force in N

        # Length of the pole(s) in m
        self._pole_len = np.asarray(pole_len, dtype=np.float) if pole_len is not None else np.asarray([0.5])
        # Weight of the pole(s) in kg
        self._pole_mass = np.asarray(pole_mass, dtype=np.float) if pole_mass is not None else np.asarray([0.1])

        if self._pole_len.size == 1:
            self._pole_len.shape = (1,)
        if self._pole_mass.size == 1:
            self._pole_mass.shape = (1,)
        if self._pole_len.shape[0] != self._pole_mass.shape[0]:
            msg = ("Dimension mismatch: Array 'pole_len' is a vector of length %d,"
                   " but array 'pole_mass' is a vector of length %d.")
            msg = msg % (self._pole_len.shape[0], self._pole_mass.shape[0])
            raise ValueError(msg)

        if self._pole_len.shape[0] > 1 and not self._include_friction:
            raise ValueError("Multiple poles without friction not implemented.")

        self._cart_loc = 0.0
        self._cart_vel = 0.0
        self._pole_angle = np.zeros(self._pole_len.shape)
        """:type: ndarray"""
        self._pole_vel = np.zeros(self._pole_len.shape)
        """:type: ndarray"""

        self._mode = mode
        if mode == 'custom':
            self._init(cart_loc, cart_vel, np.pi/180 * np.asarray(pole_angle), pole_vel, mu_c, mu_p, simsteps,
                       discount_factor)
        elif mode == 'hard':
            self._init([-3., 3.], [-5., 5.], [-np.pi * 45 / 180, np.pi * 45 / 180], [-2.5 * np.pi, 2.5 * np.pi],
                       0.0005, 0.000002, 10, 0.999)
        elif mode == 'swingup':
            self._init([-3., 3.], [-5., 5.], [-np.pi * 45 / 180, np.pi * 45 / 180], [-2.5 * np.pi, 2.5 * np.pi],
                       0.0005, 0.000002, 10, 1.)
        else:
            if mode != 'easy':
                print("Error: CartPole does not recognize mode", mode)
                print("Defaulting to mode 'easy'")
            self._init([-2.4, 2.4], [-6., 6.], [-np.pi * 12 / 180, np.pi * 12 / 180], [-6, 6],
                       0., 0., 1, 0.999)

        self._dt = 0.02

    def _init(self, cart_loc, cart_vel, pole_angle, pole_vel, mu_c, mu_p, simsteps, discount_factor):
        self._feature_limits = np.asarray([  # Admissible state space
                                             cart_loc,  # Position of the cart [-2.4, 2.4]
                                             cart_vel,  # Velocity of cart, [-1., 1.]
                                             pole_angle,  # Angular position of poles, [-36, 36]
                                             pole_vel  # Angular velocity of poles, [-.5, .5]
                                             ], dtype=np.float)

        self._mu_c = mu_c  # Friction for the cart in N s/m
        self._mu_p = mu_p  # Friction of the poles in N s/m
        self._simsteps = simsteps
        self._discount_factor = discount_factor

    def init(self):
        reward_range = (-1000., float(self._pole_len.shape[0])) if self._mode == 'swingup' else (-1., 1.)
        ts = TaskSpec(discount_factor=self._discount_factor, reward_range=reward_range)
        ts.set_episodic()

        num_poles = self._pole_angle.shape[0]
        for i, (min_, max_) in enumerate(self._feature_limits):
            if min_ == -np.inf:
                min_ = 'NEGINF'
            elif self._discrete_state:
                min_ = math.floor(min_)
            if max_ == np.inf:
                max_ = 'POSINF'
            elif self._discrete_state:
                max_ = math.ceil(max_)
            limits = (min_, max_)

            repeat = 1
            if i >= 2 and num_poles > 1:
                repeat = num_poles

            # if self._discrete_state:
            #     ts.add_int_obs(limits, repeat=repeat)
            # else:
            ts.add_double_obs(limits, repeat=repeat)

        extra = " COPYRIGHT Cart Pole (Python) implemented by Astrid Jackson."

        pole_angle_descr = "'pole angle',"
        pole_vel_descr = "'pole velocity'"
        if num_poles > 1:
            pole_angle_descr = "'pole angle %d'," * num_poles
            pole_angle_descr = pole_angle_descr % tuple(np.arange(1, num_poles + 1))
            pole_angle_descr = pole_angle_descr.strip()
            pole_vel_descr = "'pole velocity %d'," * num_poles
            pole_vel_descr = pole_vel_descr % tuple(np.arange(1, num_poles + 1))
            pole_vel_descr = pole_vel_descr.strip(',')
        state_descr = "STATEDESCR {'descr':['cart location','cart velocity'," + pole_angle_descr + pole_vel_descr + "]}"

        if self._discrete_action:
            ts.add_int_act((-self._max_force, self._max_force))
            action_descr = "'move %d':{'value':[%d]}," * 21
            step = (self._max_force + self._max_force) / 20
            action_descr = action_descr % tuple(np.repeat(np.arange(-self._max_force, self._max_force + 1, step), 2))
            action_descr = action_descr.strip(',')
            action_descr = "ACTIONDESCR {%s}" % action_descr
            actions_per_dim = " ACTIONS_PER_DIM (20)"
            extra = actions_per_dim + extra
        else:
            ts.add_double_act((-self._max_force, self._max_force))
            action_descr = "ACTIONDESCR {'force':{'value':'*','descr':{'cart':[0]}}}"

        if self._discrete_state:
            states_per_dim_descr = " STATES_PER_DIM ("
            for i, ((min_, max_), s) in enumerate(zip(self._feature_limits, self._dim_size)):
                states_per_dim = int((math.ceil(max_) - math.floor(min_)) / s)
                states_per_dim_descr += "%d," % states_per_dim

                if i >= 2 and num_poles > 1:
                    states_per_dim_descr += "%d," % states_per_dim
            states_per_dim_descr = states_per_dim_descr.strip(',')
            states_per_dim_descr += ")"
            extra = states_per_dim_descr + extra

        ts.set_extra(state_descr + " " + action_descr + extra)
        return ts.to_taskspec()

    def start(self):
        self._cart_loc = 0.
        self._cart_vel = 0.
        self._pole_angle.fill(0.0)        # = np.asarray([np.pi * 1. / 180., 0.])
        self._pole_vel.fill(0.)
        if self._random_start:
            self._pole_angle = (np.random.random(self._pole_angle.shape) - 0.5) / 5.

        return_obs = Observation()
        return_obs.doubleArray = self._get_observation()
        return return_obs

    def step(self, action):
        return_ro = Reward_observation_terminal()
        return_ro.r = self._apply(action)
        if self._reward_noise > 0:
            return_ro.r += np.random.normal(scale=self._reward_noise)
        return_ro.terminal = self._is_terminal()

        obs = Observation()
        obs.doubleArray = self._get_observation()
        return_ro.o = obs

        return return_ro

    def cleanup(self):
        pass

    def message(self, msg):
        return "CartPole (Python) does not respond to that message."

    def _get_observation(self):
        return [self._cart_loc, self._cart_vel] + self._pole_angle.tolist() + self._pole_vel.tolist()

    def _apply(self, action):
        try:
            force = min(max(action.intArray[0], -self._max_force), self._max_force)
        except IndexError:
            force = min(max(action.doubleArray[0], -self._max_force), self._max_force)
            # if np.fabs(force) < 10. / 256.:
            #     force = 10. / 256. if force > 0 else -10. / 256.
        force += self._max_force * np.random.normal(scale=self._noise) if self._noise > 0 else 0.0

        df = self._dt / float(self._simsteps)
        for step in range(self._simsteps):
            if self._include_friction:
                cart_accel = force - self._mu_c * np.sign(self._cart_vel) + self._effective_force()
                cart_accel /= (self._cart_mass + self._effective_mass())
                pole_accel = (-.75 / self._pole_len) * (cart_accel * np.cos(self._pole_angle) + self._gravity_on_pole())

            else:
                cos_pole_angle = np.cos(self._pole_angle)
                sin_pole_angle = np.sin(self._pole_angle)
                pole_vel_sq = self._pole_vel * self._pole_vel
                cos_pole_angle_sq = cos_pole_angle * cos_pole_angle

                totalM = self._cart_mass + self._pole_mass
                ml = self._pole_mass * self._pole_len

                pole_accel = force * cos_pole_angle - totalM * self._gravity * sin_pole_angle + ml * (
                    cos_pole_angle * sin_pole_angle) * pole_vel_sq
                pole_accel /= ml * cos_pole_angle_sq - totalM * self._pole_len

                cart_accel = force + ml * sin_pole_angle * pole_vel_sq - self._pole_mass * self._gravity * cos_pole_angle * sin_pole_angle
                cart_accel /= totalM - self._pole_mass * cos_pole_angle_sq

            self._cart_loc += df * self._cart_vel
            self._cart_vel += df * cart_accel
            self._pole_angle += df * self._pole_vel
            self._pole_vel += df * pole_accel

        # If theta (state[2]) has gone past the conceptual limits of [-pi,pi]
        # map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
        for i in range(self._pole_angle.shape[0]):
            while self._pole_angle[i] < -np.pi:
                self._pole_angle[i] += 2. * np.pi

            while self._pole_angle[i] > np.pi:
                self._pole_angle[i] -= 2. * np.pi

        if self._mode == 'swingup':
            return np.cos(np.abs(self._pole_angle)).sum()
        else:
            return 0. if self._is_terminal() else 1.

    def _is_terminal(self):
        return np.abs(self._cart_loc) > self._feature_limits[0, 1] or \
               (np.abs(self._pole_angle) > self._feature_limits[2, 1]).any()

    def _effective_force(self):
        f = self._pole_mass * self._pole_len * (self._pole_vel ** 2) * np.sin(self._pole_angle)
        f += .75 * self._pole_mass * np.cos(self._pole_angle) * self._gravity_on_pole()
        return f.sum()

    def _effective_mass(self):
        return (self._pole_mass * (1. - .75 * np.cos(self._pole_angle) ** 2)).sum()

    def _gravity_on_pole(self):
        pull = self._mu_p * self._pole_vel / (self._pole_mass * self._pole_len)
        pull += self._gravity * np.sin(self._pole_angle)
        return pull


if __name__ == "__main__":
    environment_loader.load_environment(CartPole())
