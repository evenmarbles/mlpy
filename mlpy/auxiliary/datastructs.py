"""
.. module:: mlpy.auxiliary.datastructs
   :platform: Unix, Windows
   :synopsis: Provides data structure implementations.

.. moduleauthor:: Astrid Jackson <ajackson@eecs.ucf.edu>
"""
from __future__ import division, print_function, absolute_import

import heapq
import numpy as np

from abc import ABCMeta, abstractmethod


class Array(object):
    """The managed array class.

    The managed array class pre-allocates memory to the given size
    automatically resizing as needed.

    Parameters
    ----------
    size : int
        The size of the array.

    Examples
    --------
    >>> a = Array(5)
    >>> a[0] = 3
    >>> a[1] = 6

    Retrieving an elements:

    >>> a[0]
    3
    >>> a[2]
    0

    Finding the length of the array:

    >>> len(a)
    2

    """
    def __init__(self, size):
        self._data = np.zeros((size,))
        self._capacity = size
        self._size = 0

    def __setitem__(self, index, value):
        """Set the the array at the index to the given value.

        Parameters
        ----------
        index : int
            The index into the array.
        value :
            The value to set the array to.

        """
        if index >= self._size:
            if self._size == self._capacity:
                self._capacity *= 2
                new_data = np.zeros((self._capacity,))
                new_data[:self._size] = self._data
                self._data = new_data

            self._size += 1

        self._data[index] = value

    def __getitem__(self, index):
        """Get the value at the given index.

        Parameters
        ----------
        index : int
            The index into the array.

        """
        return self._data[index]

    def __len__(self):
        """The length of the array.

        Returns
        -------
        int :
            The size of the array

        """
        return self._size


class Point2D(object):
    """The 2d-point class.

    The 2d-point class is a container for positions
    in a 2d-coordinate system.

    Parameters
    ----------
    x : float, optional
        The x-position in a 2d-coordinate system. Default is 0.0.
    y : float, optional
        The y-position in a 2d-coordinate system. Default is 0.0.

    Attributes
    ----------
    x : float
        The x-position in a 2d-coordinate system.
    y : float
        The y-position in a 2d-coordinate system.

    """
    __slots__ = ['x', 'y']

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class Point3D(object):
    """
    The 3d-point class.

    The 3d-point class is a container for positions
    in a 3d-coordinate system.

    Parameters
    ----------
    x : float, optional
        The x-position in a 2d-coordinate system. Default is 0.0.
    y : float, optional
        The y-position in a 2d-coordinate system. Default is 0.0.
    z : float, optional
        The z-position in a 3d-coordinate system. Default is 0.0.

    Attributes
    ----------
    x : float
        The x-position in a 2d-coordinate system.
    y : float
        The y-position in a 2d-coordinate system.
    z : float
        The z-position in a 3d-coordinate system.

    """
    __slots__ = ['x', 'y', 'z']

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class Vector3D(Point3D):
    """The 3d-vector class.

    .. todo::
        Implement vector functionality.

    Parameters
    ----------
    x : float, optional
        The x-position in a 2d-coordinate system. Default is 0.0.
    y : float, optional
        The y-position in a 2d-coordinate system. Default is 0.0.
    z : float, optional
        The z-position in a 3d-coordinate system. Default is 0.0.

    Attributes
    ----------
    x : float
        The x-position in a 2d-coordinate system.
    y : float
        The y-position in a 2d-coordinate system.
    z : float
        The z-position in a 3d-coordinate system.

    """

    def __init__(self, x=0.0, y=0.0, z=0.0):
        super(Vector3D, self).__init__(x, y, z)


class Queue(object):
    """The abstract queue base class.

    The queue class handles core functionality common for
    any type of queue. All queues inherit from the queue
    base class.

    See Also
    --------
    :class:`FIFOQueue`, :class:`PriorityQueue`

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._queue = []

    def __len__(self):
        return len(self._queue)

    def __contains__(self, item):
        try:
            self._queue.index(item)
            return True
        except Exception:
            return False

    def __iter__(self):
        return iter(self._queue)

    def __str__(self):
        return '[' + ', '.join('{}'.format(el) for el in self._queue) + ']'

    def __repr__(self):
        return ', '.join('{}'.format(el) for el in self._queue)

    @abstractmethod
    def push(self, item):
        """Push a new element on the queue

        Parameters
        ----------
        item :
            The element to push on the queue

        """
        raise NotImplementedError

    @abstractmethod
    def pop(self):
        """Pop an element from the queue."""
        raise NotImplementedError

    def empty(self):
        """Check if the queue is empty.

        Returns
        -------
        bool :
            Whether the queue is empty.

        """
        return len(self._queue) <= 0

    def extend(self, items):
        """Extend the queue by a number of elements.

        Parameters
        ----------
        items : list
            A list of items.

        """
        for item in items:
            self.push(item)

    def get(self, item):
        """Return the element in the queue identical to `item`.

        Parameters
        ----------
        item :
            The element to search for.

        Returns
        -------
        The element in the queue identical to `item`. If the element
        was not found, None is returned.

        """
        try:
            index = self._queue.index(item)
            return self._queue[index]
        except Exception:
            return None

    def remove(self, item):
        """Remove an element from the queue.

        Parameters
        ----------
        item :
            The element to remove.

        """
        self._queue.remove(item)


class FIFOQueue(Queue):
    """The first-in-first-out (FIFO) queue.

    In a FIFO queue the first element added to the queue
    is the first element to be removed.

    Examples
    --------
    >>> q = FIFOQueue()
    >>> q.push(5)
    >>> q.extend([1, 3, 7])
    >>> print q
    [5, 1, 3, 7]

    Retrieving an element:

    >>> q.pop()
    5

    Removing an element:

    >>> q.remove(3)
    >>> print q
    [1, 7]

    Get the element in the queue identical to the given item:

    >>> q.get(7)
    7

    Check if the queue is empty:

    >>> q.empty()
    False

    Loop over the elements in the queue:

    >>> for x in q:
    >>>     print x
    1
    7

    Check if an element is in the queue:

    >>> if 7 in q:
    >>>     print "yes"
    yes

    See Also
    --------
    :class:`PriorityQueue`

    """
    def __init__(self):
        super(FIFOQueue, self).__init__()

    def push(self, item):
        """Push an element to the end of the queue.

        Parameters
        ----------
        item :
            The element to append.

        """
        self._queue.append(item)

    def pop(self):
        """Return the element at the front of the queue.

        Returns
        -------
        The first element in the queue.

        """
        return self._queue.pop(0)

    def extend(self, items):
        """Append a list of elements at the end of the queue.

        Parameters
        ----------
        items : list
            List of elements.

        """
        self._queue.extend(items)


class PriorityQueue(Queue):
    """
    The priority queue.

    In a priority queue each element has a priority associated with it. An element
    with high priority (i.e., smallest value) is served before an element with low priority
    (i.e., largest value). The priority queue is implemented with a heap.

    Parameters
    ----------
    func : callable
        A callback function handling the priority. By default the priority
        is the value of the element.

    Examples
    --------
    >>> q = PriorityQueue()
    >>> q.push(5)
    >>> q.extend([1, 3, 7])
    >>> print q
    [(1,1), (5,5), (3,3), (7,7)]

    Retrieving the element with highest priority:

    >>> q.pop()
    1

    Removing an element:

    >>> q.remove((3, 3))
    >>> print q
    [(5,5), (7,7)]

    Get the element in the queue identical to the given item:

    >>> q.get(7)
    7

    Check if the queue is empty:

    >>> q.empty()
    False

    Loop over the elements in the queue:

    >>> for x in q:
    >>>     print x
    (5, 5)
    (7, 7)

    Check if an element is in the queue:

    >>> if 7 in q:
    >>>     print "yes"
    yes

    See Also
    --------
    :class:`FIFOQueue`

    """
    def __init__(self, func=lambda x: x):
        super(PriorityQueue, self).__init__()

        self.func = func

    def __contains__(self, item):
        for _, element in self._queue:
            if item == element:
                return True
        return False

    def __str__(self):
        return '[' + ', '.join('({},{})'.format(*el) for el in self._queue) + ']'

    def push(self, item):
        """Push an element on the priority queue.

        The element is pushed on the priority queue according
        to its priority.

        Parameters
        ----------
        item :
            The element to push on the queue.

        """
        heapq.heappush(self._queue, (self.func(item), item))

    def pop(self):
        """Get the element with the highest priority.

        Get the element with the highest priority (i.e., smallest value).

        Returns
        -------
        The element with the highest priority.

        """
        return heapq.heappop(self._queue)[1]

    def get(self, item):
        """Return the element in the queue identical to `item`.

        Parameters
        ----------
        item :
            The element to search for.

        Returns
        -------
        The element in the queue identical to `item`. If the element
        was not found, None is returned.

        """
        for _, element in self._queue:
            if item == element:
                return element
        return None

    def remove(self, item):
        """Remove an element from the queue.

        Parameters
        ----------
        item :
            The element to remove.

        """
        super(PriorityQueue, self).remove(item)
        heapq.heapify(self._queue)
