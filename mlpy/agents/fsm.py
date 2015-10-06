"""
.. module:: mlpy.agents.fsm
   :platform: Unix, Windows
   :synopsis: Implementation of a generic finite state machine.

.. moduleauthor:: Astrid Jackson <ajackson@eecs.ucf.edu>
"""
from __future__ import division, print_function, absolute_import

import sys
import traceback
import importlib
import time

from ..auxiliary.io import import_module_from_path
from ..auxiliary.misc import listify
from ..tools.log import LoggingMgr
from ..tools.configuration import ConfigMgr
from ..modules import Module


class Event(object):
    """Transition event definition.

    When transitioning from a source state to a destination state
    a transition event is fired.

    Parameters
    ----------
    name : str
        Name of the event.
    state : FSMState, optional
        The current state.
    machine : StateMachine, optional
        Reference to the state machine.
    delay : int, optional
        The amount of time (milliseconds) by which checking
        this event is delayed. Default is 0.
    args : tuple, optional
        Positional parameters passed to the next state.
    kwargs : dict, optional
        Non-positional parameters passed to the next state.

    """
    def __init__(self, name, state=None, machine=None, delay=None, *args, **kwargs):
        self._create_time = time.time() * 1000

        self.trigger_time = None
        self.name = name
        self.state = state
        self.machine = machine
        self.delay = delay if delay is not None else 0
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.name

    def ready(self):
        """Check if the event is ready.

        Check if the event has waited the requested amount of time.
        If so, the event fires.
        """
        current_time = time.time() * 1000
        return current_time - self._create_time >= self.delay


class EmptyEvent(Event):
    """A no-op transition event.

    A no-op transition event does nothing when it is fired; i.e. it stays in
    the same state without transitioning.

    Parameters
    ----------
    state : FSMState, optional
        The current state.
    machine : StateMachine, optional
        Reference to the state machine.
    delay : int, optional
        The amount of time (milliseconds) by which checking
        this event is delayed. Default is 0.
    args : tuple, optional
        Positional parameters passed to the next state.
    kwargs : dict, optional
        Non-positional parameters passed to the next state.

    """
    def __init__(self, state=None, machine=None, delay=None, *args, **kwargs):
        super(EmptyEvent, self).__init__("no-op", state, machine, delay, *args, **kwargs)


class FSMState(Module):
    """State base class.

    A state of the finite state machine.

    """
    @property
    def name(self):
        """ Name of the state.

        Returns
        -------
        str :
            The state's name.
        """
        return self.__class__.__name__

    def __init__(self):
        super(FSMState, self).__init__()
        self._logger = LoggingMgr().get_logger(self._mid)

    def __getstate__(self):
        d = dict(self.__dict__)
        remove_list = ['_logger']
        for key in remove_list:
            del d[key]
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self._logger = LoggingMgr().get_logger(self._mid)

    def enter(self, t, *args, **kwargs):
        """MDPState initialization.

        Parameters
        ----------
        t : float
            The current time (sec)

        """
        self._logger.debug("Enter")
        super(FSMState, self).enter(t)

    def update(self, dt):
        """Update the state.

        Update the state and handle state transitions based on events.

        Parameters
        ----------
        dt : float
            The elapsed time (sec)

        Returns
        -------
        Event :
            The transition event.

        """
        super(FSMState, self).update(dt)
        return None

    def exit(self):
        """Perform cleanup tasks."""
        super(FSMState, self).exit()
        self._logger.debug("Exit")


class Transition(object):
    """Transition class.

    Each transition contains a source and a destination state.
    Furthermore, conditions for transitioning and callbacks before
    and after transitioning can be specified.

    Parameters
    ----------
    source : str
        The source state.
    dest : str
        The destination state.
    conditions : list[callable]
        The transition is only executed once the condition(s)
        have been met.
    before : callable
        Callback function to be called before exiting the
        source state.
    after : callable
        Callback function to be called after entering the
        destination state.

    """
    class _Condition(object):
        def __init__(self, func):
            self.func = func

        def check(self, current_state):
            return self.func(current_state)

    def __init__(self, source, dest, conditions=None, before=None, after=None):
        self._source = source
        self._dest = dest

        self._conditions = []
        if conditions is not None:
            for c in listify(conditions):
                self._conditions.append(self._Condition(c))

        self._before = listify(before) if before is not None else []
        self._after = listify(after) if after is not None else []

    def execute(self, event):
        """Execute the transition.

        The transition is only executed, if all conditions are met.

        Parameters
        ----------
        event : Event
            The transition event.

        Returns
        -------
        bool :
            Whether the transition was executed or not.
        """
        machine = event.machine

        for c in self._conditions:
            if not c.check(event.state):
                return False

        for func in self._before:
            if isinstance(func, dict):
                func = getattr(func["model"], func["func"])
            func()

        if not isinstance(event, EmptyEvent):
            machine.get_state(self._source).exit()
            machine.set_state(self._dest)
            machine.get_state(self._dest).enter(event.trigger_time, *event.args, **event.kwargs)

        machine.clear_events(event.state.name)

        for func in self._after:
            if isinstance(func, dict):
                func = getattr(func["model"], func["func"])
            func()
        return True


class OnUpdate(object):
    """OnUpdate class.

    On update of the current state, a callback can be specified
    which will be called if the conditions have been met.

    Parameters
    ----------
    source : str
        The source state.
    onupdate: callable
        The callback function to be called
    conditions : list[callable]
        The condition(s) which have to be met in order for the
        callback to be called.

    """
    class _Condition(object):
        def __init__(self, func):
            self.func = func

        def check(self, current_state):
            return self.func(current_state)

    def __init__(self, source, onupdate=None, conditions=None):
        self._source = source

        self._conditions = []
        if conditions is not None:
            for c in listify(conditions):
                self._conditions.append(self._Condition(c))

        self._cb_onupdate = listify(onupdate) if onupdate is not None else []

    def execute(self, machine):
        """Execute the callback.

        The callbacks are only called if all conditions are met.

        Parameters
        ----------
        machine : StateMachine
            Reference to the state machine.

        """
        for c in self._conditions:
            if not c.check(machine.get_state(self._source)):
                return False

        for func in self._cb_onupdate:
            if isinstance(func, dict):
                func = getattr(func["model"], func["func"])
            func()


class StateMachine(Module):
    """The finite state machine.

    The finite state machine handles state transitions,
    by triggering events. Events can also be fired from
    outside the state machine to force a transition.

    Parameters
    ----------
    states : FSMState | list[FSMState], optional
        A list of states.
    initial : str, optional
        The initial state.
    transitions : list[dict] | list[list], optional
        Transition information.
    onupdate : list[dict] | list[list], optional
        Callback information to be executed on update.

    """
    @property
    def current_state(self):
        """The current event.

        Returns
        -------
        FSMState :
            the current state.

        """
        return self._current_state

    def __init__(self, states=None, initial=None, transitions=None, onupdate=None):
        super(StateMachine, self).__init__()
        self._logger = LoggingMgr().get_logger(self._mid)

        self._states = {}
        """:type : dict[str, FSMState]"""
        self._transitions = {}
        """:type : dict[str, dict[str, Transition]]"""
        self._onupdate = {}
        """:type: dict[str, OnUpdate]"""
        self._events = []
        """:type : list[TEvent]"""
        self._current_state = None
        """:type: FSMState"""

        if states is not None:
            self.add_states(states)

        if initial is not None and isinstance(initial, FSMState):
            if initial.name not in self._states:
                self._states[initial.name] = initial

        self._initial = initial
        """:type : str"""

        if transitions is not None:
            for t in listify(transitions):
                self.add_transition(**t)

        if onupdate is not None:
            for u in listify(onupdate):
                self.add_onupdate(**u)

    def __getstate__(self):
        d = super(StateMachine, self).__getstate__()
        remove_list = ['_logger', '_states', '_transitions', '_onupdate', '_events', '_current_state']
        for key in remove_list:
            del d[key]
        return d

    def __setstate__(self, d):
        # TODO: currently the only way to solve the serialization problem
        #   is to reload the variables from file due to the need of function serialization
        super(StateMachine, self).__setstate__(d)
        self._logger = LoggingMgr().get_logger(self._mid)

        self._states = {}
        self._transitions = {}
        self._onupdate = {}
        self._events = []
        self._current_state = None

    def load_from_file(self, owner, filename, **kwargs):
        """Load the FSM from file.

        Read the information of the state machine from
        file. The file contains information of the states,
        transitions and callback function.

        Parameters
        ----------
        owner : object
            Reference to the object owning the FSM.
        filename : str
            The name of the file containing the FSM
            configuration information.
        kwargs : dict
            Non-positional arguments match with configuration
            parameters during state creation.

        Notes
        -----
        The FSM setup can be specified via a configuration file in ``.json``
        format. The configuration file must follow a specific format.

        The configuration file must contain the absolute path to the module
        containing the implementation of each state. Additionally, the configuration
        file must contain the name of the initial state, a list of states, their
        transitions, and information of the `onupdate` callback functions.

        :Example:

            A skeleton configuration with two states named "<initial>" and "<next>" and one
            simple transition between them named "<event>". The implementation of the states
            are defined in the file specified in "Module".

            ::

                {
                    "Module": "absolute/path/to/the/fsm/states.py",
                    "States": [
                        "<initial>",
                        "<next>"
                    ],
                    "Initial": "<initial>",
                    "Transitions": [
                        {"source": "<initial>", "event": "<event>", "dest": "<next>"},
                        {"source": "<next>", "event": "<event2>", "dest": "<initial>"}
                    ],
                    "OnUpdate": [
                    ]
                }

            If the states have initialization parameters these can be specified as follows:

            ::

                {
                    "States": [
                        {"<initial>": {
                            "args": "motion",
                            "kwargs": {"support_leg": "right"}
                        }}
                    ]
                }

            This lets the FSM know that the state "<initial>" has two parameters. The positional
            arguments (specified in `args`) are compared to non-positional arguments in kwargs passed to
            `load_from_file`. If a match exists, the value of the match is passed as argument. If no
            match exist the value in "args" is send directly to the state. Multiple positional arguments
            can be specified by adding them in a list: ["arg1", "arg2", ...]. The non-positional arguments
            are send as is to the state.

            To specify callback functions before and after transitioning from a source to the destination
            the following formats are available:

            ::

                {
                    "Transitions": [
                        {"source": "<initial>", "event": "<event>", "dest": "<next>",
                            "before": {"model": "<ClassName>", "func": "<FuncName>"},
                            "after": {"model": "<ClassName>", "func": "<FuncName>"}},
                        {"source": "<next>", "event": "<event2>", "dest": "<initial>",
                            "before": "<FuncName>",
                            "after": "<FuncName>"}
                    ]
                }

            It is also possible to define conditions on the transitions, such that a transition
            between states is only performed when the condition(s) are met:

            ::

                {
                    "Transitions": [
                        {"source": "<initial>", "event": "<event>", "dest": "<next>",
                            "conditions": ["lambda x: not x._motion.is_running",
                                           "FuncName"]
                            "before": {"model": "<ClassName>", "func": "<FuncName>"},
                            "after": {"model": "<ClassName>", "func": "<FuncName>"}},
                        {"source": "<next>", "event": "<event2>", "dest": "<initial>",
                            "conditions": "FuncName"
                            "before": "<FuncName>",
                            "after": "<FuncName>"}
                    ]
                }

            It is also possible to add a transition to every state by using ``*``:

            ::

                {
                    "Transitions": [
                        {"source": "*", "event": "<event>", "dest": "*"},
                    ]
                }

            This statement means that "<event>" is a valid event from every state to every
            other state.

            To identify the `onupdate` callback functions use the following format:

            ::

                {
                    "OnUpdate": [
                        {"source": "<initial>", "conditions": ["lambda x: not x._motion.is_running",
                                                               "FuncName"]
                            "onupdate": {"model": "<ClassName>", "func": "<FuncName>"}},
                    ]
                }

            This lets the FSM know to call the function specified in "onupdate" when in state
            "<initial>" when the conditions are met. The conditions are optional. Also, instead
            of calling a class function a lambda or other function can be called here.

        """
        try:
            cfg = ConfigMgr(filename)
            module = import_module_from_path(cfg.get("Module"), "task_fsm")

            self._initial = cfg.get("Initial")

            for state in cfg.get("States"):
                pargs = ()
                npargs = {}
                if isinstance(state, dict):
                    npargs = state.itervalues().next()
                    state = state.iterkeys().next()

                    if "args" in npargs:
                        for arg in listify(npargs["args"]):
                            if arg in kwargs:
                                pargs += (kwargs[arg],)
                            else:
                                pargs += (arg,)
                        del npargs["args"]
                    if "kwargs" in npargs:
                        npargs = npargs["kwargs"]
                state_c = getattr(module, state)(*pargs, **npargs)

                self.add_states(state_c)

            for t in cfg.get("Transitions"):
                for key in t.iterkeys():
                    c = None
                    if key == "before":
                        c = "before"
                    if key == "after":
                        c = "after"
                    if key == "conditions":
                        c = "conditions"

                    if c is not None:
                        if c in ["before", "after"]:
                            if t[c]["model"] == owner.__class__.__name__:
                                t[c] = getattr(owner, t[c]["func"])
                            else:
                                k = t[c]["model"].rfind(".")
                                n = t[c]["model"][:k]
                                m = t[c]["model"][k+1:]
                                # noinspection PyUnusedLocal
                                module = importlib.import_module(n)
                                t[c]["model"] = eval("module." + m)
                        if c == "conditions":
                            for i, cond in enumerate(t[c]):
                                t[c][i] = eval(t[c][i])

                self.add_transition(t["event"], t["source"], t["dest"],
                                    conditions=t["conditions"] if "conditions" in t else None,
                                    before=t["before"] if "before" in t else None,
                                    after=t["after"] if "after" in t else None)

            for u in cfg.get("OnUpdate"):
                for key in u.iterkeys():
                    c = None
                    if key == "onupdate":
                        c = "onupdate"
                    if key == "conditions":
                        c = "conditions"

                    if c is not None:
                        if c in "onupdate":
                            if u[c]["model"] == owner.__class__.__name__:
                                u[c] = getattr(owner, u[c]["func"])
                            else:
                                k = u[c]["model"].rfind(".")
                                n = u[c]["model"][:k]
                                m = u[c]["model"][k+1:]
                                # noinspection PyUnusedLocal
                                module = importlib.import_module(n)
                                u[c]["model"] = eval("module." + m)
                        if c == "conditions":
                            for i, cond in enumerate(u[c]):
                                u[c][i] = eval(u[c][i])

                self.add_onupdate(u["source"], onupdate=u["onupdate"] if "onupdate" in u else None,
                                  conditions=u["conditions"] if "conditions" in u else None)
        except KeyError:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            sys.exit(1)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            sys.exit(1)

    def get_state(self, state):
        """Return the FSMState instance with the given name.

        Parameters
        ----------
        state : str
            The name of the state to retrieve.

        Returns
        -------
        FSMState :
            The state.

        Raises
        ------
        ValueError
            If the state is not a registered state.

        """
        if state not in self._states:
            raise ValueError("State '%s' is not a registered state." % state)
        return self._states[state]

    def set_state(self, state):
        """Set the current state.

        Parameters
        ----------
        state : str or FSMState
            The (name of the) state.

        Raises
        ------
        ValueError
            If the state is not a string or an instance of FSMState.

        """
        if isinstance(state, basestring):
            state = self.get_state(state)
        if not isinstance(state, FSMState):
            raise ValueError("State '%s' is not a valid FSMState instance." % state)

        self._current_state = state

    def add_states(self, states):
        """Add new state(s) to the managed states.

        Parameters
        ----------
        states : FSMState | list[FSMState]
            The state(s) to be added.

        """
        for state in listify(states):
            if state.name not in self._states:
                self._states[state.name] = state

    def add_transition(self, event, source, dest, conditions=None, before=None, after=None):
        """Add a transition from a source state to a destination state.

        Parameters
        ----------
        event : str
            The event driving the transition.
        source : str
            The source state.
        dest : str
            The destination state.
        conditions : list[callable]
            The conditions that must be met for transition
            to execute.
        before : callable
            Callback function to be called before exiting the
            source state.
        after : callable
            Callback function to be called after entering the
            source state.

        Raises
        ------
        ValueError
            If the source or the destination is not a registered state.

        Notes
        -----
        By setting source to `*`, the callbacks will be called for all states.

        """
        if not source == "*" or not dest == "*":
            if source not in self._states:
                raise ValueError("State '%s' is not a registered state." % source)
            if dest not in self._states:
                raise ValueError("State '%s' is not a registered state." % dest)

        if event not in self._transitions:
            self._transitions[event] = {}

        self._transitions[event][source] = Transition(source, dest, conditions, before, after)

    def add_onupdate(self, source, onupdate=None, conditions=None):
        """Add a callback to be called on update.

        Parameters
        ----------
        source : str
            The state for which the callback will be triggered.
        onupdate : callable
            The callback function.
        conditions : list[callable]
            The conditions that must be met in order to execute the callback.

        Raises
        ------
        ValueError
            If the source is not a registered state.

        Notes
        -----
        By setting source to `*`, the callbacks will be called for all states.

        """
        if not source == "*":
            if source not in self._states:
                raise ValueError("State '%s' is not a registered state." % source)

        self._onupdate[source] = OnUpdate(source, onupdate, conditions)

    def post_event(self, e, *args, **kwargs):
        """An event is added to the events list.

        The first event that meets all the conditions will be executed.

        Parameters
        ----------
        e : str | Event
            The event
        args : tuple, optional
            Positional parameters send to the next state.
        kwargs : dict, optional
            Non-positional parameters send to the next state.

        Raises
        ------
        ValueError
            If the event `e` is not a string or an instance of Event.
        ValueError
            If event e is not a registered transition event or the event
            is not registered for the current state.

        """
        if e is None:
            return

        if isinstance(e, basestring):
            if e == "no-op":
                e = EmptyEvent(self._current_state, self, *args, **kwargs)
            else:
                e = Event(e, self._current_state, self, *args, **kwargs)

        elif isinstance(e, Event):
            if e.state is None:
                e.state = self._current_state
            if e.machine is None:
                e.machine = self
            e.args += args
            e.kwargs.update(kwargs)

        else:
            raise ValueError("'(%s: %s)' is not a proper event type" % (e, type(e)))

        if e.name not in self._transitions:
            raise ValueError("No events with name '%s' registered" % e.name)
        if "*" not in self._transitions[e.name] and self._current_state.name not in self._transitions[e.name]:
            raise ValueError("Event '%s' is not registered for state '%s'" % (e.name, self._current_state))
        self._events.append(e)

    def clear_events(self, state_name=None):
        """Clear all events.

        If state_name is given, the events are only cleared for
        the given state.

        Parameters
        ----------
        state_name : str
            The name of the state for which to clear all events.
            If `None` all events are removed. Default is None.

        """
        self._events = [x for x in self._events if not x.state.name == state_name] if state_name is not None else []

    def enter(self, t):
        """Enter the current state.

        Parameters
        ----------
        t : float
            The current time (sec)

        """
        super(StateMachine, self).enter(t)

        self.set_state(self._initial)
        self._current_state.enter(t)

    def update(self, dt):
        """Update state and handle event transitions.

        Parameters
        ----------
        dt : float
            The elapsed time (sec)

        """
        super(StateMachine, self).update(dt)

        did_transitioned = False
        if self._events:
            did_transitioned = self._check_transition()

        if not did_transitioned:
            try:
                # noinspection PyTypeChecker
                self.post_event(self._current_state.update(dt))
                if self._current_state.name in self._onupdate:
                    self._onupdate[self._current_state.name].execute(self)
            except Exception as e:
                self._logger.exception(e.message)

            if self._events:
                self._check_transition()

    def exit(self):
        """Exit the finite state machine."""
        super(StateMachine, self).exit()

        self._current_state.exit()

    def _check_transition(self):
        """Check transitions.

        Transition on the first event found, which meets all conditions.

        Returns
        -------
        bool :
            Whether a transition fired or not.

        """
        state_name = self._current_state.name
        for e in self._events:
            if e.ready():
                t = self._transitions[e.name]["*"] if "*" in self._transitions[e.name] else self._transitions[e.name][
                    state_name]

                e.trigger_time = self._t
                if t.execute(e):
                    return True

        return False
