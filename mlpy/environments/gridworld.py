from __future__ import division, print_function, absolute_import
# noinspection PyUnresolvedReferences
from six.moves import range

import os
import sys
import random
from abc import ABCMeta, abstractmethod

from ..auxiliary.datastructs import Point2D
from ..tools.configuration import ConfigMgr
from . import Environment

__all__ = ['Cell', 'GridWorld']


class Cell(object):
    """The abstract cell module.

    A cell is a base unit in a 2d-grid. The :class:`GridWorld` is composed
    of cells.

    Parameters
    ----------
    x : int
        The x-position of the cell.
    y : int
        The y-position of the cell.
    func : callable
        A callback function to find the neighboring cells.

    Notes
    -----
    Every class inheriting from Cell must implement :meth:`is_occupied`.

    """
    __metaclass__ = ABCMeta

    @property
    def x(self):
        """The x-position of the cell.

        Returns
        -------
        int :
            The x-position.

        """
        return self._x

    @property
    def y(self):
        """The y-position of the cell.

        Returns
        -------
        int :
            The y-position.

        """
        return self._y

    @property
    def neighbors(self):
        """The cell's neighbors.

        Returns
        -------
        list[Point2D] :
            A list of neighbors.

        """
        return self._neighbors

    def __init__(self, x, y, func):
        self.data = None

        self._x = x
        self._y = y

        self._find_neighbors(func)

    @abstractmethod
    def is_occupied(self):
        """Determines if the cell is occupied.

        Returns
        -------
        bool :
            Whether the cell is occupied.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    def _find_neighbors(self, func):
        """Find the neighboring cells"""
        neighbors = [func(self, d) for d in range(4)]
        self._neighbors = [coord for coord in neighbors if coord is not None]


class GridWorld(Environment):
    """A gridworld consisting of a 2d-grid.

    A gridworld's basic unit is a cell. Each cell has four
    neighbors corresponding to the actions the agent can take
    in the four compass directions (N, S, E, W).

    Parameters
    ----------
    width : int
        The number of cells in the x-direction.
    height : int
        The number of cells in the y-direction.
    agents : Agent or list[Agent]
        A list of agents that act in the gridworld.
    filename : str
        The name of the file containing the configuration of
        the gridworld.

    Notes
    -----
    Within the gridworld, the agent's location is denoted by `o`.

    """
    @property
    def width(self):
        """The number of cells in the x-direction.

        Returns
        -------
        int :
            The width.
        """
        return self._width

    @property
    def height(self):
        """The number of cells in the y-direction.

        Returns
        -------
        int :
            The height.
        """
        return self._height

    def __init__(self, width=20, height=20, agents=None, filename=None):
        super(GridWorld, self).__init__(agents)

        self._width = width
        self._height = height

        self._grid = []
        """:type: list[list[Cell]]"""

        self._config_mgr = None
        """:type: ConfigMgr"""

        if filename is not None:
            self.load(filename)

    def __str__(self):
        column = 3
        result = ""
        spaces = sum([column + 1 for _ in range(len(self._grid[0]))])
        result += '-' * spaces
        result += '\n'

        for i, item in enumerate(self._grid):
            result += '%s%s%s%s' % ('|'.join([self._columnize(obj.data, column, 'Center')
                                              for obj in item]), '|\n', '-' * spaces, '\n')
        return result

    def reset(self, t, **kwargs):
        """Reset the agent's state.

        Parameters
        ----------
        t : float
            The current time (sec).
        kwargs : dict, optional
            Non-positional parameters.

        """
        super(GridWorld, self).reset(t, **kwargs)
        self._initialize()

    def update(self, dt):
        """Update the agents.

        Parameters
        ----------
        dt : float
            The elapsed time (sec)

        """
        super(GridWorld, self).update(dt)

        for i, agent in enumerate(self._agents):
            # prevLoc = agent.loc
            agent.update(dt)
            # if prevLoc != agent.loc:
            # self.display.redrawCell(prevLoc.x, prevLoc.y)
            # self.display.redrawCell(agent.loc.x, agent.loc.y)
            # self.display.update()

    def make_cell(self, x, y):
        """Create the new cell.

        x : int
            The x-coordinate within the gridworld.
        y : int
            The y-coordinate within the gridworld.

        Returns
        -------
        Cell :
            The created cell.

        """
        return Cell(x, y, self.move_coords)

    def load(self, f):
        """Loads the world from file.

        If a `*.cfg` with the same name exists, the configurations
        are being loaded as well.

        Parameters
        ----------
        f : str or file
            The file instance or the filename.

        """
        assert(f is file or isinstance(f, basestring))

        fh = f
        if isinstance(f, basestring):
            fh = file(f)
        data = fh.readlines()
        fh.close()

        # Load config file if it exists
        if isinstance(f, basestring):
            if os.path.isfile(f):
                base = os.path.splitext(f)[0]
                cf = base + ".config"
                self._config_mgr = ConfigMgr(cf)

        data = [line.rstrip('\n') for line in data]
        self._height = len(data)
        self._width = max([len(x) for x in data])

        self._initialize()
        for y in range(self._height):
            for x in range(min(self._width, len(data[y]))):
                self._grid[y][x].data = data[y][x]

    def save(self, f=None):
        """Save the world to file.

        Parameters
        ----------
        f : str or file
            The file instance or the filename.

        """
        assert(f is file or isinstance(f, basestring))

        total = ""
        for y in range(self._height):
            line = ""
            for x in range(self.width):
                line += self._grid[y][x].data
            total += "%s\n" % line

        fh = f
        if isinstance(f, basestring):
            fh = file(f, "w")
        fh.write(total)
        fh.close()

    def get_cell(self, loc):
        """Return the cell based on its x/y-coordinates.

        Parameters
        ----------
        loc : Point2D
            The x-/y-coordinates of the cell.

        Returns
        -------
        Cell :
            The cell at the specified location.

        """
        try:
            value = self._grid[loc.y][loc.x]
        except Exception as e:
            sys.exit(e)
        return value

    def find_cells(self, data):
        """Find the cells containing given data.

        Parameters
        ----------
        data : str
            The data to match the cell to.

        Returns
        -------
        list[Cell] :
            All cells that contain the specified data.
        """
        return [c for x in self._grid for c in x if c.data == data]

    def move_coords(self, cell, move):
        """
        Returns the coordinates of the neighboring cell the agent transitions
        to following the given move.

        Parameters
        ----------
        cell : Cell
            The current cell.
        move : int
            The action performed.

        Returns
        -------
        Point2D :
            The x-/y-coordinates of the resulting cell.

        """
        dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][move]
        coords = Point2D(cell.x + dx, cell.y + dy)

        if coords.x < 0 or coords.x >= self._width or coords.y < 0 or coords.y >= self._height:
            coords = None
        return coords

    def random_location(self):
        """
        Find a random unoccupied location within the grid.

        Returns
        -------
        Point2D :
            The random x-/y-coordinates.

        """
        while True:
            x = random.randrange(self._width)
            y = random.randrange(self._height)
            loc = Point2D(x, y)
            cell = self.get_cell(loc)
            if not cell.is_occupied():
                return loc

    def set_start_loc(self, loc):
        """
        Set the agent's starting location.

        Parameters
        ----------
        loc : Point2D
            The x-/y-coordinates the agent starts out in.

        """
        cells = self.find_cells('o')
        for c in cells:
            c.data = ''
        self._grid[loc.y][loc.x].data = 'o'

    def _initialize(self):
        self._grid = [[self.make_cell(i, j) for i in range(self._width)] for j in range(self._height)]

    # noinspection PyMethodMayBeStatic
    def _columnize(self, word, width, align='Left'):
        """
        Create a column from a string

        Parameters
        ----------
        word: str
            The string to be processed.
        width: int
            The width of the column.
        align: {'Left', 'Right'}
            The column alignment.

        Returns
        -------
        str :
            Columnized string.

        """
        nspaces = width - len(word)
        if nspaces < 0:
            nspaces = 0
        if align == 'Left':
            return word + (" " * nspaces)
        if align == 'Right':
            return (" " * nspaces) + word
        return (" " * (nspaces / 2)) + word + (" " * (nspaces - nspaces / 2))
