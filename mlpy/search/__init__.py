"""
====================================
Search tools (:mod:`mlpy.search`)
====================================

.. currentmodule:: mlpy.search


.. autosummary::
   :toctree: generated/
   :nosignatures:

   Node
   ISearch


Informed Search
===============

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ~informed.AStar

"""
from __future__ import division, print_function, absolute_import

from abc import ABCMeta, abstractmethod
from ..modules import UniqueModule

__all__ = ['informed']


class Node(object):
    """The node class.

    The node within the graph that is being built while searching
    for the optimal path

    Parameters
    ----------
    state : int or tuple[int]
        The state.
    parent : Node
        The parent node.
    action : str
        The action performed in the state.
    cost : float
        The path cost to the state.

    """
    @property
    def state(self):
        """The state associated the node.

        Returns
        -------
        int or tuple[int] :
            The state.

        """
        return self._state

    @property
    def parent(self):
        """The node's parent.

        Returns
        -------
        Node :
            The parent node.

        """
        return self._parent

    @property
    def g(self):
        """The path cost.

        Returns
        -------
        float :
            The cost.

        """
        return self._g

    @property
    def depth(self):
        """The depth of the node.

        The number of steps between the start node
        and this node.

        Returns
        -------
        int :
            The depth.

        """
        return self._depth

    def __init__(self, state, parent=None, action=None, cost=0):
        self.h = 0
        self._state = state
        self._parent = parent
        self._action = action
        self._g = cost

        self._depth = 0
        if parent:
            self._depth = parent.depth + 1

    def __cmp__(self, other):
        """Comparison between two nodes.

        The state of the two nodes are compared.

        Parameters
        ----------
        other : Node
            Another node.

        Returns
        -------
        int : {0, 1, -1}
            Returns `-1` if the state of this node is less than the state of the other node,
            `1` if the state of this node is greater than the state of the other node and
            `0` if the two nodes are equal.

        """
        return cmp(self.state, other.state)

    def expand(self, task):
        """Expands a node's neighbors.

        Parameters
        ----------
        task : SearchTask
            A search task instance.

        """
        # noinspection PyArgumentList
        return [Node(next_node, self, action, task.get_path_cost(self._g, self.state, action, next_node)) for
                action, next_node in task.get_successor(self._state)]


class ISearch(UniqueModule):
    """The search class interface.

    """
    __metaclass__ = ABCMeta

    def __init__(self, task):
        super(ISearch, self).__init__()

        self._task = task
        self._endnode = None

    # noinspection PyMethodMayBeStatic
    def save_path(self, path, filename):
        """
        Save the path to file.

        Parameters
        ----------
        path : list[int or tuple[int]]
            The found path.
        filename : str
            The filename to save the path to.

        """
        if not path:
            print('The path is empty. No path saved.')
            return

        with open(filename, 'wb') as f:
            f.write(','.join('(%s, %s)' % x for x in path))

    def get_path(self):
        """Return the optimal path from the start to the goal node.

        Returns
        -------
        list[int or tuple[int]] :
            The optimal path.

        """
        if self._endnode is None:
            raise ReferenceError('You must run \'search\' to find the path first')

        path = []
        if self._endnode:
            node = self._endnode
            while node.state != self._task.initial_state:
                path.append(node.state)
                node = node.parent
            path.append(node.state)

            return path[::-1]
        return path

    @abstractmethod
    def search(self):
        """Search for the optimal path."""
        raise NotImplementedError
