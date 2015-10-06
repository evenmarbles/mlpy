from __future__ import division, print_function, absolute_import

from ..auxiliary.datastructs import Point3D
from ..modules import Module
from ..modules.patterns import Singleton


class WorldObject(Module):
    """The world object base class.

    The world object base class keeps track of the location of the
    object and its level of confidence for the information based on
    when the object was last seen.

    Attributes
    ----------
    location : Points3D
        The objects current location.
    timestamp : float
        The timestamp the object was last seen (the image was captured).
    confidence : float
        The level of confidence for the object's information based on
        when the object was last seen.

    Notes
    -----
    All world objects should derive from this class.

    .. todo::
        Update the location based on localization.

    """
    __slots__ = ('location', 'timestamp', 'confidence')

    CONFIDENCE_VALID = 0
    CONFIDENCE_SUSPICIOUS = 1
    CONFIDENCE_INVALID = 2

    THRESHOLD_SUSPICIOUS = 3
    THRESHOLD_INVALID = 5

    @property
    def confidence(self):
        """The level of confidence of the object's information based on
        when the object was last seen.

        Returns
        -------
        float :
            The level of confidence.

        """
        return self._confidence

    def __init__(self):
        super(WorldObject, self).__init__()

        self.location = Point3D()
        self.timestamp = -1.0

        self._confidence_state = WorldObject.CONFIDENCE_INVALID
        self._confidence = 0.0

    def __getstate__(self):
        d = dict(self.__dict__)
        for name in self.__slots__:
            if not name == "confidence":
                d[name] = getattr(self, name)
        return d

    def __setstate__(self, d):
        for name, value in d.iteritems():
            setattr(self, name, value)

    def enter(self, t):
        """Enter the world object.

        Parameters
        ----------
        t : float
            The current time (sec)

        """
        super(WorldObject, self).enter(t)

    def update(self, dt):
        """Update the world object based on the elapsed time.

        Parameters
        ----------
        dt : float
            The elapsed time (sec)

        """
        super(WorldObject, self).update(dt)

        self.update_confidence()

        # TODO: Update the location based on localization.

    # noinspection PyUnusedLocal
    def update_confidence(self):
        """Update the level of confidence.

        Based on when the object was last seen, the level of confidence
        for the information available on the object is update.

        """
        # TODO: calculation of confidence should be a little more complicated
        if self.timestamp > 0.0:
            dt = self._t - self.timestamp
            if dt <= WorldObject.THRESHOLD_SUSPICIOUS:
                self._confidence_state = WorldObject.CONFIDENCE_VALID
                self._confidence = 1.0
            elif dt <= WorldObject.THRESHOLD_INVALID:
                self._confidence_state = WorldObject.CONFIDENCE_SUSPICIOUS
                self._confidence = 0.5
            else:
                self._confidence_state = WorldObject.CONFIDENCE_INVALID
                self._confidence = 0.0


class WorldModel(Module):
    """The world model.

    The world model manages the world objects of type :class:`WorldObject` by ensuring
    that the objects are updated with the latest information at every time step of the
    program loop. Furthermore, information of the world objects can be accessed from
    the world model.

    Examples
    --------
    >>> from mlpy.agents.world import WorldModel, WorldObject
    >>> WorldModel().add_object("ball", WorldObject)

    >>> from mlpy.agents.world import WorldModel
    >>> ball = WorldModel().get_object("ball")

    Notes
    -----
    The world module follows the singleton design pattern (:class:`~mlpy.modules.patterns.Singleton`),
    ensuring only one instances of the world module exist. This allows for accessing the information of
    the world model from anywhere in the program.

    """
    __metaclass__ = Singleton

    def __init__(self):
        super(WorldModel, self).__init__()

        self._objects = {}

    def add_object(self, name, obj):
        """Add a world object.

        Parameters
        ----------
        name : str
            The identifier of the world object.
        obj : WorldObject
            The world object instance.

        Raises
        ------
        AttributeError
            If an world object with the given name has already been registered.

        """
        if name in self._objects:
            raise AttributeError("A world object with the name `%s` already exists." % name)

        self._objects[name] = obj

    def get_object(self, name):
        """Returns the object with the given name.

        Parameters
        ----------
        name : str
            The identifier of the world object.

        Returns
        -------
        WorldObject :
            The world object

        """
        return self._objects[name]

    def enter(self, t):
        """Enter the world model.

        Parameters
        ----------
        t : float
            The current time (sec)
        """
        super(WorldModel, self).enter(t)

        for key, obj in self._objects.iteritems():
            obj.enter(t)

    def update(self, dt):
        """Update all world objects.

        The world objects are updated at each time step of the program loop.

        Parameters
        ----------
        dt : float
            The elapsed time (sec)

        """
        super(WorldModel, self).update(dt)

        for key, obj in self._objects.iteritems():
            obj.update(dt)
