from __future__ import division, print_function, absolute_import

import sys
import traceback
from abc import ABCMeta, abstractmethod

from . import UniqueModule


class Singleton(type):
    """
    Metaclass ensuring only one instance of the class exists.

    The singleton pattern ensures that a class has only one instance
    and provides a global point of access to that instance.

    Methods
    -------
    __call__

    Notes
    -----
    To define a class as a singleton include the :data:`__metaclass__`
    directive.

    See Also
    --------
    :class:`Borg`

    Examples
    --------
    Define a singleton class:

    >>> from mlpy.modules.patterns import Singleton
    >>> class MyClass(object):
    >>>     __metaclass__ = Singleton

    .. note::
        | Project: Code from `StackOverflow <http://stackoverflow.com/q/6760685>`_.
        | Code author: `theheadofabroom <http://stackoverflow.com/users/655372/theheadofabroom>`_
        | License: `CC-Wiki <http://creativecommons.org/licenses/by-sa/3.0/>`_

    """
    _instance = {}

    def __call__(cls, *args, **kwargs):
        """Returns instance to object."""
        if cls not in cls._instance:
            # noinspection PyArgumentList
            cls._instance[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance[cls]


class Borg(object):
    """Class ensuring that all instances share the same state.

    The borg design pattern ensures that all instances of a class share
    the same state  and provides a global point of access to the shared state.

    Rather than enforcing that only ever one instance of a class exists,
    the borg design pattern ensures that all instances share the same state.
    That means every the values of the member variables are the same for every
    instance of the borg class.

    The member variables which are to be shared among all instances must be
    declared as class variables.

    See Also
    --------
    :class:`Singleton`

    Notes
    -----
    One side effect is that if you subclass a borg, the objects all have the
    same state, whereas subclass objects of a singleton have different states.

    Examples
    --------
    Create a borg class:

    >>> from mlpy.modules.patterns import Borg
    >>> class MyClass(Borg):
    >>>     shared_variable = None

    .. note::
        | Project: Code from `ActiveState <http://code.activestate.com/recipes/66531-singleton-we-dont-need-no-stinkin-singleton-the-bo/>`_.
        | Code author: `Alex Naanou <http://code.activestate.com/recipes/users/104183/>`_
        | License: `CC-Wiki <http://creativecommons.org/licenses/by-sa/3.0/>`_

    """
    _shared_state = {}

    def __new__(cls, *p, **k):
        # noinspection PyArgumentList
        self = object.__new__(cls, *p, **k)
        self.__dict__ = cls._shared_state
        return self


class RegistryInterface(type):
    """Metaclass registering all subclasses derived from a given class.

    The registry interface adds every class derived from a given class
    to its registry dictionary. The `registry` attribute is a class
    variable and can be accessed anywhere. Therefore, this interface can
    be used to find all subclass of a given class.

    One use case are factory classes.

    Attributes
    ----------
    registry : list
        List of all classes deriving from a registry class.

    Methods
    -------
    __init__

    Examples
    --------
    Create a registry class:

    >>> from mlpy.modules.patterns import RegistryInterface
    >>> class MyRegistryClass(object):
    ...     __metaclass__ = RegistryInterface

    .. note::
        | Project: Code from `A Primer on Python Metaclasses <https://jakevdp.github.io/blog/2012/12/01/a-primer-on-python-metaclasses/>`_.
        | Code author: `Jake Vanderplas <http://www.astro.washington.edu/users/vanderplas/>`_
        | License: `CC-Wiki <http://creativecommons.org/licenses/by-sa/3.0/>`_

    """
    __metaclass__ = ABCMeta

    def __init__(cls, name, bases, dct):
        """Register the deriving class on instantiation."""
        if not hasattr(cls, 'registry'):
            cls.registry = {}
        else:
            cls.registry[name.lower()] = cls

        super(RegistryInterface, cls).__init__(name, bases, dct)


class Observable(UniqueModule):
    """The observable base class.

    The observable keeps a record of all listeners and notifies them
    of the events they have subscribed to by calling :meth:`Listener.notify`.

    The listeners are notified by calling :meth:`dispatch`. Listeners are notified
    if either the event that is being dispatched is ``None`` or the listener has
    subscribed to a ``None`` event, or the name of the event the listener has subscribed
    to is equal to the name of the dispatching event.

    An event is an object consisting of the `source`; i.e. the observable, the event
    `name`, and the event `data` to be passed to the listener.

    Parameters
    ----------
    mid : str
        The module's unique identifier

    Methods
    -------
    dispatch
    load
    save
    subscribe
    unsubscribe

    Examples
    --------
    >>> from mlpy.modules.patterns import Observable
    >>>
    >>> class MyObservable(Observable):
    >>>     pass
    >>>
    >>> o = MyObservable()

    This defines the observable `MyObservable` and creates
    an instance of it.

    >>> from mlpy.modules.patterns import Listener
    >>>
    >>> class MyListener(Listener):
    >>>
    >>>     def notify(self, event):
    >>>         print "I have been notified!"
    >>>
    >>> l = MyListener(o, "test")

    This defines the listener `MyListener` that when notified will print
    the same text to the console regardless of which event has been thrown
    (as long as the listener has subscribed to the event). Then an instance
    of MyListener is created that subscribes to the event `test` of `MyObservable`.

    When the event `test` is dispatched by the observable, the listener is notified
    and the text is printed on the stdout:

    >>> o.dispatch("test", **{})
    I have been notified!

    """
    class Event(object):
        """Event being dispatched by the observable.

        Parameters
        ----------
        source : Observable
            The observable instance.
        name : str
            The name of the event.
        data : dict
            The information to be send.

        """
        def __init__(self, source, name, data=None):
            self.source = source
            self.name = name
            self.data = data if data is not None else {}

    def __init__(self, mid=None):
        super(Observable, self).__init__(mid)

        self._listeners = {}

    def subscribe(self, listener, events=None):
        """Subscribe to the observable.

        Parameters
        ----------
        listener : Listener
            The listener instance.
        events : str or list[str] or tuple[str] or None
            The event names the listener wants to be notified about.

        """
        if events is not None and not isinstance(events, (list, tuple)):
            events = (events,)

        self._listeners[listener] = events

    def unsubscribe(self, listener):
        """Unsubscribe from the observable.

        The listener is removed from the list of listeners.

        Parameters
        ----------
        listener : Listener
            The listener instance.

        """
        del self._listeners[listener]

    def dispatch(self, name, **attrs):
        """Dispatch the event to all listeners.

        Parameters
        ----------
        name : str
            The name of the event to dispatch.
        attrs : dict
            The information send to the listeners.

        """
        # Create the event to send
        e = Observable.Event(self, name, {k: v for k, v in attrs.iteritems()})

        # Notify all listeners of this event
        for listener, events in self._listeners.items():
            if events is None or name is None or name in events:
                try:
                    listener.notify(e)
                except Exception:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    traceback.print_exception(exc_type, exc_value, exc_traceback)
                    sys.exit(1)


class Listener(object):
    """The listener interface.

    A listener subscribes to an observable identifying the events the listener is
    interested in. The observable calls :meth:`notify` to send relevant event information.

    Parameters
    ----------
    o : Observable, optional
        The observable instance.
    events : str or list[str], optional
        The event names the listener wants to be notified about.

    Notes
    -----
    Every class inheriting from Listener must implement :meth:`notify`, which
    defines what to do with the information send by the observable.

    Examples
    --------
    >>> from mlpy.modules.patterns import Observable
    >>>
    >>> class MyObservable(Observable):
    >>>     pass
    >>>
    >>> o = MyObservable()

    This defines the observable `MyObservable` and creates
    an instance of it.

    >>> from mlpy.modules.patterns import Listener
    >>>
    >>> class MyListener(Listener):
    >>>
    >>>     def notify(self, event):
    >>>         print "I have been notified!"
    >>>
    >>> l = MyListener(o, "test")

    This defines the listener `MyListener` that when notified will print
    the same text to the console regardless of which event has been thrown
    (as long as the listener has subscribed to the event). Then an instance
    of MyListener is created that subscribes to the event `test` of `MyObservable`.

    When the event `test` is dispatched by the observable, the listener is notified
    and the text is printed on the stdout:

    >>> o.dispatch("test", **{})
    I have been notified!

    """
    __metaclass__ = ABCMeta

    def __init__(self, o=None, events=None):
        if o is not None:
            o.subscribe(self, events)

    @abstractmethod
    def notify(self, event):
        """Notification from the observable.

        Parameters
        ----------
        event : Observable.Event
            The event object dispatched by the observable consisting of `source`;
            i.e. the observable, the event `name`, and the event `data`.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        Notes
        -----
        This is an abstract method and *must* be implemented by its deriving class.

        """
        raise NotImplementedError
