from __future__ import division, print_function, absolute_import

import numpy as np

from ..auxiliary.datastructs import PriorityQueue, FIFOQueue
from . import ISearch, Node


class AStar(ISearch):
    """A* algorithm.

    This class implements the A* algorithm

    Parameters
    ----------
    task : SearchTask
        The search task to perform

    """

    def __init__(self, task):
        super(AStar, self).__init__(task)

        def f(n):
            return max(getattr(n, 'f', -np.inf), n.g + self._task.h(n))

        self._frontier = PriorityQueue(f)
        self._frontier.push(Node(self._task.initial_state))
        self._explored = FIFOQueue()

    def get_results(self):
        """Display the search results visually."""
        results = {}
        for node in self._explored:
            info = {}
            if self._task.is_terminal:
                info['terminal'] = True
            if node.state == self._task.initial_state:
                info['initial'] = True
            info['g'] = "{0:.2f}".format(node.g)
            info['h'] = "{0:.2f}".format(node.h)
            info['f'] = "{0:.2f}".format(node.g + node.h)
            results[node.state] = info
        return results

    def search(self):
        """Perform the search.

        Performs the actual search for the optimal path using
        the A* algorithm

        """
        while not self._frontier.empty():
            current = self._frontier.pop()
            if self._task.is_terminal:
                self._explored.push(current)
                self._endnode = current
                return
            if current not in self._explored:
                self._explored.push(current)

                neighbors = current.expand(self._task)
                for neighbor in neighbors:
                    node = self._explored.get(neighbor)
                    if node:
                        if node.g <= neighbor.g:
                            continue
                        self._explored.remove(node)

                    else:
                        node = self._frontier.get(neighbor)
                        if node:
                            if node.g <= neighbor.g:
                                continue
                            self._frontier.remove(node)

                    self._frontier.push(neighbor)
