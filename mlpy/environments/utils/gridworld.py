from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import math
import numpy as np
from scipy.stats import bernoulli


class Gridworld(object):
    @property
    def width(self):
        return self._w

    @property
    def height(self):
        return self._h

    def __init__(self, height, width, northsouth=None, eastwest=None):
        self._w = width
        self._h = height

        self._ns = northsouth
        self._ew = eastwest

        if self._ns is None or self._ew is None:
            self._ns = np.zeros((width, height-1), dtype=bool)
            self._ew = np.zeros((height, width - 1), dtype=bool)

            n = int(math.sqrt(self._w * self._h))
            for i in range(n):
                self._add_obstacle()

    def __str__(self):
        out = ""
        for i in range(self._ns.shape[0]):
            out += " -"
        out += "\n"

        for h in range(self._ew.shape[0] - 1, 0, -1):
            out += "| "
            for w in range(self._ew.shape[1]):
                out += "| " if self._ew[h][w] else "  "
            out += "|\n"

            for w in range(self._ns.shape[0]):
                out += " -" if self._ns[w][h-1] else "  "
            out += " \n"

        out += "| "
        for w in range(self._ew.shape[1]):
            out += "| " if self._ew[0][w] else "  "
        out += "|\n"

        for i in range(self._ns.shape[0]):
            out += " -"
        out += " \n"
        return out

    def __repr__(self):
        return "width=%d height=%d" % (self._w, self._h)

    def wall(self, ns_coord, ew_coord, direction):
        is_ns = 0 == direction / 2.0
        is_incr = 0 == direction % 2.0

        walls = self._ns if is_ns else self._ew

        major = ew_coord if is_ns else ns_coord
        minor = ns_coord if is_ns else ew_coord

        if not is_incr:
            if minor == 0:
                return True
            minor -= 1

        if minor >= walls[major].shape[0]:
            return True
        return walls[major][minor]

    def _add_obstacle(self):
        direction = bernoulli.rvs(0.5)
        parallel = self._ns if direction else self._ew
        perpendicular = self._ew if direction else self._ns

        seedi = np.random.randint(0, parallel.shape[0])
        seedj = np.random.randint(0, parallel[seedi].shape[0])

        first = seedi + 1
        while self._is_clear(first - 1, seedj, parallel, perpendicular):
            first -= 1
        last = seedi
        while self._is_clear(last + 1, seedj, parallel, perpendicular):
            last += 1

        self._choose_segment(first, last, seedj, parallel)

    def _choose_segment(self, first, last, j, parallel):
        if last <= first:
            return

        max_length = last - first
        if max_length >= parallel.shape[0]:
            max_length = parallel.shape[0]

        length = np.random.random_integers(1, max_length) if max_length > 1 else 1
        direction = 1 - 2 * np.random.random_integers(0, 1)
        start = first if direction > 0 else last - 1

        for i in range(length):
            parallel[start + i * direction][j] = True

    def _is_clear(self, i, j, parallel, perpendicular):
        if i > parallel.shape[0]:
            return False
        if i < parallel.shape[0] and parallel[i][j]:
            return False
        if i > 0 and parallel[i - 1][j]:
            return False
        if 0 < i <= perpendicular[j].shape[0] and perpendicular[j][i - 1]:
            return False
        if 0 < i <= perpendicular[j + 1].shape[0] and perpendicular[j + 1][i - 1]:
            return False
        return True
