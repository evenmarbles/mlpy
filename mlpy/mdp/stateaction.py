from __future__ import division, print_function, absolute_import

import copy
import math
import numpy as np

from .distrib import ProbabilityDistribution
from ..stats import random_floats


class Experience(object):
    """Experience base class.

    Representation of an experience occurring from acting in the environment.

    Parameters
    ----------
    state : MDPState
        The representation of the current state.
    action : MDPAction
        The executed action.
    next_state : MDPState
        The representation of the state following from acting
        with `action` in state `state`.
    reward : int or float
        The reward awarded by the environment for the state-action
        pair.

    Attributes
    ----------
    state : MDPState
        The experienced state
    action : MDPAction
        The experienced action.
    next_state : MDPState
        The experienced next state.
    reward : float
        The experienced reward.

    """
    __slots__ = ('state', 'action', 'next_state', 'reward')

    def __init__(self, state, action, next_state, reward=None):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    def __str__(self):
        s = "state={0} act={1} reward={2:.2f} next_state={3}".format(self.state, self.action, self.reward,
                                                                     self.next_state) if self.reward else \
            "state={0} act={1} next_state={2}".format(self.state, self.action, self.next_state)
        return s

    def __repr__(self):
        s = "state={0} act={1} next_state={2}".format(self.state, self.action, self.next_state) if self.reward else \
            "state={0} act={1} reward={2:.2f} next_state={3}".format(
                self.state, self.action, self.reward, self.next_state)
        return s


class RewardFunction(object):
    """The reward function.

    The reward function is responsible for calculating the proper value
    of the reward. Callback functions can be specified for custom calculation
    of the reward value.

    Attributes
    ----------
    cb_get : callable
        Callback function to retrieve the reward value.
    cb_set : callable
        Callback function to set the reward value.
    reward : float
        The reward value.
    bonus
    rmax : float
        The maximum possible reward.
    activate_bonus : bool
        Flag activating/deactivating the bonus.

    Notes
    -----
    To ensure that the correct value of the reward is being accessed,
    the user should not access the class variables directly but instead
    use the methods :meth:`set` and :meth:`get` to set and get the reward
    respectively.

    Examples
    --------
    >>> RewardFunction.cb_get = staticmethod(lambda r, s: np.dot(s, RewardFunction.reward))

    In this cas the reward function is calculated by taking the dot product
    of the stored reward and a passed in value.

    >>> RewardFunction.reward = [0.1, 0.9. 1.0, 0.0]

    This sets the reward for all instances of the reward function.

    >>> reward_func = RewardFunction()
    >>> print reward_func.get([0.9, 0.5, 0.0, 1.0])
    0.54

    This calculates the reward `r` according to previously defined the
    callback function.

    """
    cb_get = None
    cb_set = None

    reward = 0.0
    rmax = 0.0
    activate_bonus = False

    @property
    def bonus(self):
        """The bonus added to the reward to encourage exploration.

        Returns
        -------
        float :
            The bonus added to the reward.

        """
        return self._bonus

    @bonus.setter
    def bonus(self, value):
        self._bonus = value

    def __init__(self):
        self._bonus = 0.0
        """:type: float"""

    def __getstate__(self):
        return {
            'reward': self.reward,
            'rmax': self.rmax,
            'bonus': self.bonus,
            'activate_bonus': self.activate_bonus
        }

    def __setstate__(self, d):
        for name, value in d.iteritems():
            if not name == 'bonus':
                setattr(type(self), name, value)
            else:
                setattr(self, name, value)

    def set(self, value, *args, **kwargs):
        """Set the reward value.

        If :meth:`cb_set` is set, the callback will be called
        to set the value.

        Parameters
        ----------
        args : tuple
            Positional arguments passed to the callback.
        kwargs : dict
            Non-positional arguments passed to the callback.

        """
        if self.cb_set is not None:
            self.reward = self.cb_set(*args, **kwargs)
            return
        self.reward = value

    def get(self, *args, **kwargs):
        """Retrieve the reward value.

        If :meth:`cb_get` is set, the callback will be called
        to retrieve the value.

        Parameters
        ----------
        args : tuple
            Positional arguments passed to the callback.
        kwargs : dict
            Non-positional arguments passed to the callback.

        Returns
        -------
        float :
            The (calculated) reward value.

        """
        reward = self.reward
        if self.cb_get is not None:
            reward = self.cb_get(self.reward, *args, **kwargs)

        if self.activate_bonus:
            reward = max(self.reward + self.bonus, self.rmax)
        return reward


class MDPStateActionInfo(object):
    """The models interface.

    Contains all relevant information predicted by a model for a
    given state-action pair. This includes the (predicted) reward and
    transition probabilities to possible next states.

    Attributes
    ----------
    transition_proba : ProbabilityDistribution
        The transition probability distribution.
    reward_func : RewardFunction
        The reward function.
    visits : int
        The number of times the state-action pair has been visited.
    known : bool
        Flag indicating whether a reward value is known or not.

    """
    __slots__ = ('transition_proba', 'reward_func', 'visits', 'known')

    def __init__(self):
        self.transition_proba = ProbabilityDistribution()
        self.reward_func = RewardFunction()

        self.visits = 0
        self.known = False

    def __getstate__(self):
        data = {}
        for name in self.__slots__:
            data[name] = getattr(self, name)
        return data

    def __setstate__(self, d):
        for name, value in d.iteritems():
            setattr(self, name, value)


class MDPStateData(object):
    """State information interface.

    Information about the state can be accessed here.

    Parameters
    ----------
    state_id : int
        The unique id of the state
    actions : list[MDPAction]
        List of actions that can be taken in this state.

    Attributes
    ----------
    id : int
        The unique id of the state.
    models : dict
        The reward and transition models for each action.
    q : dict
        The q-table, containing a q-value for each action.
    steps_away : int
        The number of steps the state is away from its closest neighbor.

    """
    __slots__ = ('id', 'models', 'q', 'v', 'steps_away')

    def __init__(self, state_id, actions):
        self.id = state_id
        """:type: int"""
        self.models = {a: MDPStateActionInfo() for a in actions}
        """:type: dict[MDPAction, MDPStateActionInfo]"""
        # Randomizing the initial q-values impedes performance
        # self.q = {a: ((0.01 - 0.0) * np.random.random() + 0.0) for a in actions}
        self.q = {a: 0.0 for a in actions}
        """:type: dict[MDPAction, float]"""
        self.v = 0.0
        """:type: float"""
        self.steps_away = 100000
        """:type: int"""

    def __getstate__(self):
        data = {}
        for name in self.__slots__:
            data[name] = getattr(self, name)
        return data

    def __setstate__(self, d):
        for name, value in d.iteritems():
            setattr(self, name, value)


class MDPPrimitive(object):
    """A Markov decision process primitive.

    The base class for :class:`MDPState` and :class:`MDPAction`. Primitives
    are represented by a list of features. They optionally can have a `name`.

    Parameters
    ----------
    features : array_like, shape (`nfeatures`,)
        List of features, where `nfeatures` is the number of features
        identifying the primitive.
    name : str, optional
        The name of the primitive. Default is "".

    Attributes
    ----------
    name
    dtype : {DTYPE_FLOAT, DTYPE_INT, DTYPE_OBJECT}
        The type of the features.
    nfeatures : int
        The number of features.
    discretized : bool
        Flag indicating whether the features are discretized or not.
    min_features : list
        The minimum value for each feature.
    max_features : list
        The minimum value for each feature.
    states_per_dim : list
        The number of states per dimension.
    description : dict
        A description of the features.

    Raises
    ------
    ValueError
        If the feature array is not one-dimensional.

    Notes
    -----
    Use the `description` to encode action information. The information
    should contain the list of all available feature combinations, the
    name of each feature.

    :Examples:

        A description of an action with three possible discrete actions:

        ::

            {
                "out": {"value": [-0.004]},
                "in": {"value": [0.004]},
                "kick": {"value": [-1.0]}
            }

        A description of an action with one possible continuous action with
        name `move`, a value of `*` allows to find the action for every
        feature array. Additional information encodes the feature name together
        with its index into the feature array are given for each higher level
        element of feature array:

        ::

            {
                "move": {
                    "value": "*",
                    "descr": {
                        "LArm": {"dx": 0, "dy": 1, "dz": 2},
                        "RArm": {"dx": 3, "dy": 4, "dz": 5},
                        "LLeg": {"dx": 6, "dy": 7, "dz": 8},
                        "RLeg": {"dx": 9, "dy": 10, "dz": 11},
                        "Torso": {"dx": 12, "dy": 13, "dz": 14}
                    }
                }
            }

        Similarly, a continuous state can be encoded as follows, which identifies
        the name of each feature together with its index into the feature array:

        ::

            {
                "LArm": {"x": 0, "y": 1, "z": 2},
                "RArm": {"x": 3, "y": 4, "z": 5},
                "LLeg": {"x": 6, "y": 7, "z": 8},
                "RLeg": {"x": 9, "y": 10, "z": 11},
                "Torso": {"x": 12, "y": 13, "z": 14}
            }

        A discrete state can be encoded by identifying the position of each feature:

        ::

            {
                "image x-position": 0,
                "displacement (mm)": 1
            }

        Alternatively, the feature can be identified by a list of features, giving he
        positional description:

        ::

            ["image x-position", "displacement (mm)"]

    Rather then setting the attributes directly, use the methods :meth:`set_nfeatures`,
    :meth:`set_dtype`, :meth:`set_description`, :meth:`set_discretized`, :meth:`set_minmax_features`,
    and :meth:`set_states_per_dim` in order to enforce type checking.

    """
    __slots__ = ('dtype', 'nfeatures', 'description', 'discretized', 'min_features', 'max_features',
                 'states_per_dim', '_features', '_name', 'ix')

    DTYPE_OBJECT = np.object
    DTYPE_FLOAT = np.float64
    DTYPE_INT = np.int32

    dtype = DTYPE_FLOAT
    nfeatures = None
    description = None

    discretized = False
    min_features = None
    max_features = None
    states_per_dim = None

    @property
    def name(self):
        """The name of the MDP primitive.

        Returns
        -------
        str :
            The name of the primitive.

        """
        return self._name

    @classmethod
    def set_nfeatures(cls, n):
        """Set the number of features.

        Parameters
        ----------
        n : int
            The number of features.

        Raises
        ------
        ValueError
            If `n` is not of type integer.

        """
        if not isinstance(n, int):
            raise ValueError("Attribute 'nfeatures' must be of <type 'int'>, got %s" % str(type(n)))
        cls.nfeatures = n

    @classmethod
    def set_dtype(cls, value=DTYPE_FLOAT):
        """Set the feature's data type.

        Parameters
        ----------
        value : {DTYPE_FLOAT, DTYPE_INT, DTYPE_OBJECT}
            The data type.

        Raises
        ------
        ValueError
            If the data type is not one of the allowed types.

        """
        if value not in [np.float64, np.int32, np.object]:
            raise ValueError("Attribute 'dtype' must be one of the allowed types, got %s" % str(type(value)))
        cls.dtype = value

    @classmethod
    def set_description(cls, descr):
        """Set the feature description.

        This extracts the number of features from the description and checks
        that it matches with the `nfeatures`. If `nfeatures` is None, `nfeatures`
        is set to the extracted value.

        Parameters
        ----------
        descr : dict
            The feature description.

        Raises
        ------
        ValueError
            If the number of features extracted from the description does not
            match `nfeatures` or if `name` isn't of type string.

        Notes
        -----
        Use the `description` to encode action information. The information
        should contain the list of all available feature combinations, the
        name of each feature.

        Examples
        --------

            A description of an action with three possible discrete actions:

            ::

                {
                    "out": {"value": [-0.004]},
                    "in": {"value": [0.004]},
                    "kick": {"value": [-1.0]}
                }

            A description of an action with one possible continuous action with
            name `move`, a value of `*` allows to find the action for every
            feature array. Additional information encodes the feature name together
            with its index into the feature array are given for each higher level
            element of feature array:

            ::

                {
                    "move": {
                        "value": "*",
                        "descr": {
                            "LArm": {"dx": 0, "dy": 1, "dz": 2},
                            "RArm": {"dx": 3, "dy": 4, "dz": 5},
                            "LLeg": {"dx": 6, "dy": 7, "dz": 8},
                            "RLeg": {"dx": 9, "dy": 10, "dz": 11},
                            "Torso": {"dx": 12, "dy": 13, "dz": 14}
                        }
                    }
                }

            Similarly, a continuous state can be encoded as follows, which identifies
            the name of each feature together with its index into the feature array:

            ::

                {
                    "LArm": {"x": 0, "y": 1, "z": 2},
                    "RArm": {"x": 3, "y": 4, "z": 5},
                    "LLeg": {"x": 6, "y": 7, "z": 8},
                    "RLeg": {"x": 9, "y": 10, "z": 11},
                    "Torso": {"x": 12, "y": 13, "z": 14}
                }

            A discrete state can be encoded by identifying the position of each feature:

            ::

                "descr": {
                    "image x-position": 0,
                    "displacement (mm)": 1
                }

            Alternatively, the feature can be identified by a list of features, giving he
            positional description:

            ::

                ["image x-position", "displacement (mm)"]

        """
        nfeatures = None
        if isinstance(descr, dict):
            config = descr.itervalues().next()
            if 'descr' in config:
                nfeatures = sum(len(v) for v in config['descr'].itervalues())
                if cls.nfeatures is not None and not cls.nfeatures == nfeatures:
                    raise ValueError("Dimension mismatch: array described by 'descr' is a vector of length %d,"
                                     " but attribute cls.nfeatures = %d" % (nfeatures, cls.nfeatures))
            elif 'value' in config and not config['value'] == '*':
                nfeatures = len(config['value'])
                if cls.nfeatures is not None and not cls.nfeatures == nfeatures:
                    raise ValueError("Dimension mismatch: array described by 'value' is a vector of length %d,"
                                     " but attribute cls.nfeatures = %d" % (nfeatures, cls.nfeatures))
            else:
                nfeatures = sum(len(v) for v in descr.itervalues())
                if cls.nfeatures is not None and not cls.nfeatures == nfeatures:
                    raise ValueError("Dimension mismatch: 'descr' is a vector of length %d,"
                                     " but attribute cls.nfeatures = %d" % (nfeatures, cls.nfeatures))

        elif isinstance(descr, list):
            nfeatures = len(descr)
            if cls.nfeatures is not None and not cls.nfeatures == nfeatures:
                raise ValueError("Dimension mismatch: 'descr' is a vector of length %d,"
                                 " but attribute cls.nfeatures = %d" % (nfeatures, cls.nfeatures))

        if cls.nfeatures is None:
            cls.nfeatures = nfeatures
        cls.description = descr

    @classmethod
    def set_discretized(cls, val=False):
        """Sets the `discretized` flag.

        Parameters
        ----------
        val : bool
            Flag identifying whether the features are discretized or not.
            Default is False.

        Raises
        ------
        ValueError
            If `val` is not boolean type.

        """
        if not isinstance(val, bool):
            raise ValueError("Attribute 'nfeatures' must be of <type 'bool'>, got %s" % str(type(val)))
        cls.discretized = val

    @classmethod
    def set_minmax_features(cls, minmax, *args):
        """Sets the minimum and maximum value for each feature.

        This extracts the number of features from the `_min` and `_max`
        values and ensures that it matches with `nfeatures`. If `nfeatures`
        is None, the `nfeatures` attribute is set to the extracted value.

        Parameters
        ----------
        minmax : array_like, shape(2,)
            The minimum and maximum value for the first feature
        args : tuple
            Min-max arrays for additional features

        Raises
        ------
        ValueError
            If the arrays are not one-dimensional vectors, the shapes of the
            arrays don't match, or the number of features does not agree with
            the attribute `nfeatures`.

        """
        minmax = np.asarray(minmax, dtype=cls.dtype)
        dim = minmax.size
        if dim == 1:
            minmax.shape = (1,)

        if minmax.shape[0] != 2:
            raise ValueError("Each array must identifying min and max value for the feature")

        if args is not None:
            args = list(args)
            args.insert(0, minmax)

        _minmax = []
        for val in zip(*args):
            _minmax.append(np.asarray(val, dtype=cls.dtype))

        if cls.nfeatures is None:
            cls.nfeatures = _minmax[0].shape[0]

        if _minmax[0].shape[0] != cls.nfeatures or _minmax[1].shape[0] != cls.nfeatures:
            raise ValueError("No more than %d minmax arrays can be given." % cls.nfeatures)

        cls.min_features = _minmax[0]
        cls.max_features = _minmax[1]

    @classmethod
    def set_states_per_dim(cls, nstates):
        """Sets the number of states per feature.

        This extracts the number of features from `nstates` and compares
        it to the attribute `nfeatures`. If it doesn't match, an exception
        is thrown. If the `nfeatures` attribute is None, `nfeatures` is set
        to the extracted value.

        Parameters
        ----------
        nstates : array_like, shape (`nfeatures`,)
            The number of states per features

        Raises
        ------
        ValueError
            If the array is not a vector of length `nfeatures`.

        """
        nstates = np.asarray(nstates, dtype=cls.dtype)
        dim = nstates.size
        if dim == 1:
            nstates.shape = (1,)

        if cls.nfeatures is None:
            cls.nfeatures = nstates.shape[0]

        if nstates.ndim != 1 or nstates.shape[0] != cls.nfeatures:
            raise ValueError("Array 'nstates' must be a vector of length %d." % cls.nfeatures)

        cls.states_per_dim = nstates
        cls.discretized = True

    def __init__(self, features, name=None):
        if type(self).dtype is None:
            type(self).dtype = MDPPrimitive.DTYPE_FLOAT

        self._features = np.asarray(features, dtype=self.dtype)
        if self._features.ndim != 1:
            raise ValueError("Array 'features' must be one-dimensional,"
                             " but features.ndim = %d" % self._features.ndim)

        self._name = name if name is not None else ""
        if not isinstance(self._name, basestring):
            raise ValueError("'name' must be a string, but got %s" % str(type(self._name)))

        if type(self).nfeatures is None:
            type(self).nfeatures = self._features.shape[0]
        elif not self._features.shape[0] == type(self).nfeatures:
            raise ValueError("Dimension mismatch: array 'features' is a vector of length %d, but"
                             " attribute cls.nfeatures = %d" % (self._features.shape[0], type(self).nfeatures))

        if type(self).discretized and type(self).states_per_dim is not None:
            self.discretize()

    # noinspection PyUnusedLocal
    def __get__(self, instance, owner):
        return self._features

    def __getitem__(self, index):
        checker = np.vectorize(lambda x: isinstance(x, slice))
        if index > len(self) and not np.any(checker(index)):
            raise IndexError("Assignment index out of range")
        return self._features[index]

    def __setitem__(self, index, value):
        if index > len(self):
            raise IndexError("Assignment index out of range")
        self._features[index] = value

    def __len__(self):
        return len(self._features)

    def __contains__(self, item):
        return item in self._features

    def __hash__(self):
        return hash(tuple(self._features)) if self._features is not None else None

    def __eq__(self, other):
        return np.array_equal(other.get(), self._features)

    def __sub__(self, other):
        return self._features - other

    def __mul__(self, other):
        return self._features * other

    def __rmul__(self, other):
        return other * self._features

    def __imul__(self, other):
        self._features *= other
        return self

    def __iter__(self):
        self.ix = 0
        return self

    def __str__(self):
        features = np.array_str(self.encode())
        return "\'" + self._name + "\':\t" + features if self._name else features

    def __repr__(self):
        features = np.array_str(self.encode())
        return "\'" + self._name + "\':\t" + features if self._name else features

    def next(self):
        if self.ix == len(self):
            raise StopIteration
        item = self._features[self.ix]
        self.ix += 1
        return item

    def __copy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k in self.__slots__:
            try:
                setattr(result, k, copy.copy(getattr(self, k)))
            except AttributeError:
                pass
        return result

    def __getstate__(self):
        data = {}
        for name in self.__slots__:
            if not name == 'ix':
                data[name] = getattr(self, name)
        return data

    def __setstate__(self, d):
        for name, value in d.iteritems():
            if name not in ['nfeatures', 'dtype', 'description', 'discretized',
                            'min_features', 'max_features', 'states_per_dim']:
                setattr(self, name, value)

        type(self).nfeatures = self._features.shape[0]

    def get(self):
        """Return the feature array.

        Returns
        -------
        ndarray :
            The feature array.

        """
        return self._features

    def tolist(self):
        """Returns the feature array as a list.

        Returns
        -------
        list :
            The features list.

        """
        return self._features.tolist()

    def set(self, features):
        """Sets the feature array to the given array.

        Parameters
        ----------
        features : array_like, shape (`nfeatures`,)
            The new feature values.

        """
        features = np.asarray(features, dtype=type(self).dtype)
        if features.ndim != 1 or features.shape[0] != type(self).nfeatures:
            raise ValueError("Array 'features' must be a vector of length %d." % type(self).nfeatures)

        self._features = np.asarray(features)

    def discretize(self):
        """Discretizes the state.

        Discretize the state using the information from the minimum and
        maximum values for each feature and the number of states attributed
        to each feature.
        """
        if not self.discretized:
            return

        nfeatures = type(self).nfeatures
        min_features = type(self).min_features
        max_features = type(self).max_features
        states_per_dim = type(self).states_per_dim

        if min_features is None or min_features.shape[0] != nfeatures:
            raise ValueError("Attribute 'min_features' must be a vectors of length %d." % nfeatures)
        if max_features is None or max_features.shape[0] != nfeatures:
            raise ValueError("Attribute 'max_features' must be a vectors of length %d." % nfeatures)
        if states_per_dim is None or states_per_dim.shape[0] != nfeatures:
            raise ValueError("Attribute 'states_per_dim' must be a vectors of length %d." % nfeatures)

        ds = []
        for i, feat in enumerate(self):
            factor = (max_features[i] - min_features[i]) / states_per_dim[i]
            if feat > 0:
                bin_num = int((feat + factor / 2) / factor)
            else:
                bin_num = int((feat - factor / 2) / factor)

            ds.append(bin_num * factor)

        self._features = np.asarray(ds)

    def encode(self):
        # noinspection PyUnresolvedReferences,PyUnusedLocal
        """Encodes the state into a human readable representation.

        Returns
        -------
        ndarray :
            The encoded state.

        Notes
        -----
        Optionally this method can be overwritten at runtime.

        Examples
        --------
        >>> def my_encode(self)
        ...     pass
        ...
        >>> MDPPrimitive.encode = my_encode

        """
        return self._features

    @classmethod
    def decode(cls, _repr):
        # noinspection PyUnresolvedReferences,PyUnusedLocal
        """Decodes the state into its original representation.

        Parameters
        ----------
        _repr : tuple
            The readable representation of the primitive.

        Returns
        -------
        MDPState :
            The decoded state.

        Notes
        -----
        Optionally this method can be overwritten at runtime.

        Examples
        --------
        >>> def my_decode(cls, _repr)
        ...     pass
        ...
        >>> MDPPrimitive.decode = classmethod(my_decode)

        """
        return cls(_repr)

    @staticmethod
    def key_to_index(key):
        # noinspection PyUnresolvedReferences,PyUnusedLocal
        """Maps internal name to group index.

        Maps the internal name of a feature to the index of the corresponding
        feature grouping. For example for a feature vector consisting of the
        x-y-z position of the left and the right arm, the features for the left
        and the right arm can be extracted separately as a group, effectively
        splitting the feature vector into two vectors with x, y, and z at the
        positions specified by the the mapping of this function.

        Parameters
        ----------
        key : str
            The key into the mapping

        Returns
        -------
        int :
            The index in the feature array.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        Notes
        -----
        Optionally this method can be overwritten at runtime.

        Examples
        --------
        >>> def my_key_to_index(key)
        ...     return {
        ...         "x": 0,
        ...         "y": 1,
        ...         "z": 2
        ...     }[key]
        ...
        >>> MDPState.description = {'LArm': {'x': 0, 'y': 1, 'z': 2}
        ...                      'RArm': {'x': 3, 'y': 4, 'z': 5}}
        >>> MDPState.key_to_index = staticmethod(my_key_to_index)

        This specifies the mapping in both direction.

        >>> state = [0.1, 0.4, 0.3. 4.6. 2.5. 0.9]
        >>>
        >>> mapping = MDPState.description['LArm']
        >>>
        >>> larm = np.zeros[len(mapping.keys())]
        >>> for key, axis in mapping.iteritems():
        ...     larm[MDPState.key_to_index(key)] = state[axis]
        ...
        >>> print larm
        [0.1, 0.4, 0.3]

        This extracts the features for the left arm from the `state` vector.

        """
        raise NotImplementedError


# noinspection PyAbstractClass,PyUnresolvedReferences
class MDPState(MDPPrimitive):
    """Representation of the state.

    States are represented by an array of features.

    Parameters
    ----------
    features : array_like, shape (`nfeatures`,)
        List of features, where `nfeatures` is the number of features
        identifying the primitive.
    name : str, optional
        The name of the primitive. Default is ''.

    Attributes
    ----------
    name
    dtype : {DTYPE_FLOAT, DTYPE_INT, DTYPE_OBJECT}
        The type of the features.
    nfeatures : int
        The number of features.
    discretized : bool
        Flag indicating whether the features are discretized or not.
    min_features : list
        The minimum value for each feature.
    max_features : list
        The minimum value for each feature.
    states_per_dim : list
        The number of states per dimension.
    description : dict
        A description of the features.
    initial_states : list
        List of initial states.
    terminal_states : list
        List of terminal states.

    Notes
    -----
    Use the `description` to encode action information. The information
    should contain the list of all available feature combinations, the
    name of each feature.

    :Examples:

        A description of an action with three possible discrete actions:

        ::

            {
                "out": {"value": [-0.004]},
                "in": {"value": [0.004]},
                "kick": {"value": [-1.0]}
            }

        A description of an action with one possible continuous action with
        name `move`, a value of `*` allows to find the action for every
        feature array. Additional information encodes the feature name together
        with its index into the feature array are given for each higher level
        element of feature array:

        ::

            {
                "move": {
                    "value": "*",
                    "descr": {
                        "LArm": {"dx": 0, "dy": 1, "dz": 2},
                        "RArm": {"dx": 3, "dy": 4, "dz": 5},
                        "LLeg": {"dx": 6, "dy": 7, "dz": 8},
                        "RLeg": {"dx": 9, "dy": 10, "dz": 11},
                        "Torso": {"dx": 12, "dy": 13, "dz": 14}
                    }
                }
            }

        Similarly, a continuous state can be encoded as follows, which identifies
        the name of each feature together with its index into the feature array:

        ::

            {
                "LArm": {"x": 0, "y": 1, "z": 2},
                "RArm": {"x": 3, "y": 4, "z": 5},
                "LLeg": {"x": 6, "y": 7, "z": 8},
                "RLeg": {"x": 9, "y": 10, "z": 11},
                "Torso": {"x": 12, "y": 13, "z": 14}
            }

        A discrete state can be encoded by identifying the position of each feature:

        ::

            {
                "image x-position": 0,
                "displacement (mm)": 1
            }

        Alternatively, the feature can be identified by a list of features, giving he
        positional description:

        ::

            ["image x-position", "displacement (mm)"]

    Rather then setting the attributes directly, use the methods :meth:`set_nfeatures`,
    :meth:`set_dtype`, :meth:`set_description`, :meth:`set_discretized`, :meth:`set_minmax_features`,
    :meth:`set_states_per_dim`, :meth:`set_initial_states`, and :meth:`set_terminal_states` in order
    to enforce type checking.

    Examples
    --------
    >>> MDPState.set_description({'LArm': {'x': 0, 'y': 1, 'z': 2}
    ...                      'RArm': {'x': 3, 'y': 4, 'z': 5}})

    This description identifies the features to be the x-y-z-position of
    the left and the right arm. The position into the feature array is given
    by the integer numbers.

    >>> def my_key_to_index(key)
    ...     return {
    ...         "x": 0,
    ...         "y": 1,
    ...         "z": 2
    ...     }[key]
    ...
    >>> MDPState.key_to_index = staticmethod(my_key_to_index)

    This defines a mapping for each key.

    >>> state = [0.1, 0.4, 0.3. 4.6. 2.5. 0.9]
    >>>
    >>> mapping = MDPState.description['LArm']
    >>>
    >>> larm = np.zeros[len(mapping.keys())]
    >>> for key, axis in mapping.iteritems():
    ...     larm[MDPState.key_to_index(key)] = state[axis]
    ...
    >>> print larm
    [0.1, 0.4, 0.3]

    This extracts the features for the left arm from the `state` vector.

    >>> s1 = MDPState([0.1, 0.4, 0.2])
    >>> s2 = MDPState([0.5, 0.3, 0.5])
    >>> print s1 - s2
    [-0.4, 0.1, -0.3]

    Subtract states from each other.

    >>> print s1 * s2
    [0.05, 0.12, 0.1]

    Multiplies two states with each other.

    >>> s1 *= s2
    >>> print s1
    [0.05, 0.12, 0.1]

    Multiplies two states in place.

    """

    class _S(object):
        def __init__(self, features, name=None):
            self.name = name if name is not None else '*'
            self.features = features

        def __str__(self):
            features = np.array_str(self.features)
            return "\'" + self.name + "\':\t" + features if self.name else features

        def __repr__(self):
            features = np.array_str(self.features)
            return "\'" + self.name + "\':\t" + features if self.name else features

        def __len__(self):
            return len(self.features)

    initial_states = None
    """:type: list[_S]"""
    terminal_states = None
    """:type: list[_S]"""

    @classmethod
    def set_nfeatures(cls, n):
        """Set the number of features.

        Parameters
        ----------
        n : int
            The number of features.

        Raises
        ------
        ValueError
            If `n` is not of type integer.

        """
        super(MDPState, cls).set_nfeatures(n)
        cls._update_initial_and_terminal_states()

    @classmethod
    def set_description(cls, descr):
        """Set the feature description.

        This extracts the number of features from the description and checks
        that it matches with the `nfeatures`. If `nfeatures` is None, `nfeatures`
        is set to the extracted value.

        Parameters
        ----------
        descr : dict
            The feature description.

        Raises
        ------
        ValueError
            If the number of features extracted from the description does not
            match `nfeatures` or if `name` isn't of type string.

        Notes
        -----
        Use the `description` to encode action information. The information
        should contain the list of all available feature combinations, the
        name of each feature.

        Examples
        --------

            A description of an action with three possible discrete actions:

            ::

                {
                    "out": {"value": [-0.004]},
                    "in": {"value": [0.004]},
                    "kick": {"value": [-1.0]}
                }

            A description of an action with one possible continuous action with
            name `move`, a value of `*` allows to find the action for every
            feature array. Additional information encodes the feature name together
            with its index into the feature array are given for each higher level
            element of feature array:

            ::

                {
                    "move": {
                        "value": "*",
                        "descr": {
                            "LArm": {"dx": 0, "dy": 1, "dz": 2},
                            "RArm": {"dx": 3, "dy": 4, "dz": 5},
                            "LLeg": {"dx": 6, "dy": 7, "dz": 8},
                            "RLeg": {"dx": 9, "dy": 10, "dz": 11},
                            "Torso": {"dx": 12, "dy": 13, "dz": 14}
                        }
                    }
                }

            Similarly, a continuous state can be encoded as follows, which identifies
            the name of each feature together with its index into the feature array:

            ::

                {
                    "LArm": {"x": 0, "y": 1, "z": 2},
                    "RArm": {"x": 3, "y": 4, "z": 5},
                    "LLeg": {"x": 6, "y": 7, "z": 8},
                    "RLeg": {"x": 9, "y": 10, "z": 11},
                    "Torso": {"x": 12, "y": 13, "z": 14}
                }

            A discrete state can be encoded by identifying the position of each feature:

            ::

                "descr": {
                    "image x-position": 0,
                    "displacement (mm)": 1
                }

            Alternatively, the feature can be identified by a list of features, giving he
            positional description:

            ::

                ["image x-position", "displacement (mm)"]

        """
        super(MDPState, cls).set_description(descr)
        cls._update_initial_and_terminal_states()

    @classmethod
    def set_minmax_features(cls, minmax, *args):
        """Sets the minimum and maximum value for each feature.

        This extracts the number of features from the `_min` and `_max`
        values and ensures that it matches with `nfeatures`. If `nfeatures`
        is None, the `nfeatures` attribute is set to the extracted value.

        Parameters
        ----------
        _min : array_like, shape(`nfeatures`,)
            The minimum value for each feature
        _max : array_like, shape(`nfeatures`,)
            The maximum value for each feature

        Raises
        ------
        ValueError
            If the arrays are not one-dimensional vectors, the shapes of the
            arrays don't match, or the number of features does not agree with
            the attribute `nfeatures`.

        """
        super(MDPState, cls).set_minmax_features(minmax, *args)
        cls._update_initial_and_terminal_states()

    @classmethod
    def set_states_per_dim(cls, nstates):
        """Sets the number of states per feature.

        This extracts the number of features from `nstates` and compares
        it to the attribute `nfeatures`. If it doesn't match, an exception
        is thrown. If the `nfeatures` attribute is None, `nfeatures` is set
        to the extracted value.

        Parameters
        ----------
        nstates : array_like, shape (`nfeatures`,)
            The number of states per features

        Raises
        ------
        ValueError
            If the array is not a vector of length `nfeatures`.

        """
        super(MDPState, cls).set_states_per_dim(nstates)
        cls._update_initial_and_terminal_states()

    @classmethod
    def set_initial_states(cls, states):
        """Set the initial states.

        Parameters
        ----------
        states : str or MDPState or array_like or list
            The initial state(s).

        Raises
        ------
        ValueError
            If both `name` and `features` are unspecified.

        """
        cls.initial_states = cls._set_initial_and_terminal_states(states)

    @classmethod
    def set_terminal_states(cls, states):
        """Set the terminal states.

        Parameters
        ----------
        states : str or ndarray or MDPState or list[str|ndarray|MDPState]
            The initial state(s).

        Raises
        ------
        ValueError
            If both `name` and `features` are unspecified.

        """
        cls.terminal_states = cls._set_initial_and_terminal_states(states)

    def __init__(self, features, name=None):
        super(MDPState, self).__init__(features, name)
        self._update_initial_and_terminal_states()

    @classmethod
    def random_initial_state(cls):
        """Return a random initial state.

        Returns
        -------
        str or MDPState :
            A random initial state.

        """
        s = None
        if cls.initial_states is not None:
            s = np.random.choice(cls.initial_states)

            if np.result_type(s.features) == np.object:
                if s.features.ndim > 1 or np.any([isinstance(f, tuple) for f in s.features]):
                    features = []
                    for f in s.features:
                        if isinstance(f, tuple) or isinstance(f, np.ndarray):
                            features.append(random_floats(f[0], f[1]))
                        else:
                            features.append(f)
                    s = cls(features, s.name)
            else:
                s = cls(s.features, s.name)
        return s

    def is_initial(self):
        """Checks if the state is an initial state.

        Returns
        -------
        bool :
            Whether the state is an initial state or not.

        """
        if MDPState.initial_states is None:
            return False

        for s in MDPState.initial_states:
            value = self._is_equal(s)
            if value:
                return value
        return False

    def is_terminal(self):
        """Checks if the state is a terminal state.

        Returns
        -------
        bool :
            Whether the state is a terminal state or not.

        """
        if MDPState.terminal_states is None:
            return False

        for s in MDPState.terminal_states:
            value = self._is_equal(s)
            if value:
                return value
        return False

    # noinspection PyMethodMayBeStatic
    def is_valid(self):
        # noinspection PyUnresolvedReferences,PyUnusedLocal
        """Check if this state is a valid state.

        Returns
        -------
        bool :
            Whether the state is valid or not.

        Notes
        -----
        Optionally this method can be overwritten at runtime.

        Examples
        --------
        >>> def my_is_valid(self)
        ...     pass
        ...
        >>> MDPPrimitive.is_valid = my_is_valid

        """
        return True

    @classmethod
    def _set_initial_and_terminal_states(cls, states):
        if isinstance(states, list):
            state_list = []
            for state in states:
                state_list.append(cls._format_state(state))
        else:
            state_list = [cls._format_state(states)]
        return state_list

    @classmethod
    def _format_state(cls, state):
        nfeatures = cls.nfeatures if cls.nfeatures is not None else 1
        if isinstance(state, basestring):
            features = np.empty(nfeatures)
            features[:] = np.NaN
            # noinspection PyProtectedMember
            state = cls._S(features, state)
        elif isinstance(state, (list, np.ndarray)):
            if np.any([isinstance(f, (list, tuple)) for f in state]):
                state = np.asarray(state, dtype=np.object)
            else:
                state = np.asarray(state, dtype=cls.dtype)
            dim = state.size
            if dim == 1:
                state.shape = (1,)
            # noinspection PyProtectedMember
            state = cls._S(state)
        if state.name == '*' and np.result_type(state.features) != np.object and np.all(np.isnan(state.features)):
            raise ValueError('Initial states must identify `name`, `features` or both.')
        return state

    @classmethod
    def _update_initial_and_terminal_states(cls):
        if cls.initial_states is not None and len(cls.initial_states[0]) != cls.nfeatures:
            for s in cls.initial_states:
                s.features = np.tile(s.features, cls.nfeatures)
        if cls.terminal_states is not None and len(cls.terminal_states[0]) != cls.nfeatures:
            for s in cls.terminal_states:
                s.features = np.tile(s.features, cls.nfeatures)

    def _is_equal(self, state):
        value = True
        if state.name != '*':
            value = value and self.name == state.name
        if np.result_type(state.features) == np.object:
            if state.features.ndim > 1 or np.any([isinstance(f, tuple) for f in state.features]):
                for f1, f2 in zip(self._features, state.features):
                    if isinstance(f2, tuple) or isinstance(f2, np.ndarray):
                        value = value and f2[0] <= f1 <= f2[1]
                    else:
                        value = value and f1 == f2
            else:
                value = value and self == state
        elif not np.all(np.isnan(state.features)):
            value = value and self == state
        return value


# noinspection PyAbstractClass,PyUnresolvedReferences
class MDPAction(MDPPrimitive):
    """Representation of an action.

    Actions are represented by an array of features.

    Parameters
    ----------
    features : array_like, shape (`nfeatures`,)
        List of features, where `nfeatures` is the number of features
        identifying the primitive.
    name : str, optional
        The name of the primitive. Default is ''.

    Attributes
    ----------
    name
    dtype : {DTYPE_FLOAT, DTYPE_INT, DTYPE_OBJECT}
        The type of the features.
    nfeatures : int
        The number of features.
    discretized : bool
        Flag indicating whether the features are discretized or not.
    min_features : list
        The minimum value for each feature.
    max_features : list
        The minimum value for each feature.
    states_per_dim : list
        The number of states per dimension.
    description : dict
        A description of the features.

    Notes
    -----
    Use the `description` to encode action information. The information
    should contain the list of all available feature combinations, the
    name of each feature.

    :Examples:

        A description of an action with three possible discrete actions:

        ::

            {
                "out": {"value": [-0.004]},
                "in": {"value": [0.004]},
                "kick": {"value": [-1.0]}
            }

        A description of an action with one possible continuous action with
        name `move`, a value of `*` allows to find the action for every
        feature array. Additional information encodes the feature name together
        with its index into the feature array are given for each higher level
        element of feature array:

        ::

            {
                "move": {
                    "value": "*",
                    "descr": {
                        "LArm": {"dx": 0, "dy": 1, "dz": 2},
                        "RArm": {"dx": 3, "dy": 4, "dz": 5},
                        "LLeg": {"dx": 6, "dy": 7, "dz": 8},
                        "RLeg": {"dx": 9, "dy": 10, "dz": 11},
                        "Torso": {"dx": 12, "dy": 13, "dz": 14}
                    }
                }
            }

        Similarly, a continuous state can be encoded as follows, which identifies
        the name of each feature together with its index into the feature array:

        ::

            {
                "LArm": {"x": 0, "y": 1, "z": 2},
                "RArm": {"x": 3, "y": 4, "z": 5},
                "LLeg": {"x": 6, "y": 7, "z": 8},
                "RLeg": {"x": 9, "y": 10, "z": 11},
                "Torso": {"x": 12, "y": 13, "z": 14}
            }

        A discrete state can be encoded by identifying the position of each feature:

        ::

            {
                "image x-position": 0,
                "displacement (mm)": 1
            }

        Alternatively, the feature can be identified by a list of features, giving he
        positional description:

        ::

            ["image x-position", "displacement (mm)"]

    Rather then setting the attributes directly, use the methods :meth:`set_nfeatures`,
    :meth:`set_dtype`, :meth:`set_description`, :meth:`set_discretized`, :meth:`set_minmax_features`,
    and :meth:`set_states_per_dim` in order to enforce type checking.

    Examples
    --------
    >>> MDPAction.set_description({'LArm': {'dx': 0, 'dy': 1, 'dz': 2}
    ...                         'RArm': {'dx': 3, 'dy': 4, 'dz': 5}})

    This description identifies the features to be the delta x-y-z-position of
    the left and the right arm. The position into the feature array is given
    by the integer numbers.

    >>> def my_key_to_index(key)
    ...     return {
    ...         "dx": 0,
    ...         "dy": 1,
    ...         "dz": 2
    ...     }[key]
    ...
    >>> MDPAction.key_to_index = staticmethod(my_key_to_index)

    This defines a mapping for each key.

    >>> action = [0.1, 0.4, 0.3. 4.6. 2.5. 0.9]
    >>>
    >>> mapping = MDPAction.description['LArm']
    >>>
    >>> larm = np.zeros[len(mapping.keys())]
    >>> for key, axis in mapping.iteritems():
    ...     larm[MDPAction.key_to_index(key)] = action[axis]
    ...
    >>> print larm
    [0.1, 0.4, 0.3]

    This extracts the features for the left arm from the `action` vector.

    >>> a1 = MDPAction([0.1, 0.4, 0.2])
    >>> a2 = MDPAction([0.5, 0.3, 0.5])
    >>> print a1 - a2
    [-0.4, 0.1, -0.3]

    Subtract actions from each other.

    >>> print a1 * a2
    [0.05, 0.12, 0.1]

    Multiplies two actions with each other.

    >>> a1 *= a2
    >>> print a1
    [0.05, 0.12, 0.1]

    Multiplies two actions in place.

    """

    def __init__(self, features, name=None):
        super(MDPAction, self).__init__(features, name)

        self._name = name if name is not None else MDPAction.get_name(self._features)

    @classmethod
    def get_name(cls, features):
        """Retrieves the name of the action.

        Retrieve the name of the action using the action's description. In the case
        that all features are zero the action is considered a `no-op` action.

        Parameters
        ----------
        features : ndarray
            A feature array.

        Returns
        -------
        str :
            The name of the action.

        """
        features = np.asarray(features, dtype=cls.dtype)

        if cls.description is not None:
            for e, config in cls.description.iteritems():
                if config["value"] == features:
                    if np.asarray(config["value"]).shape != features.shape:
                        ValueError("Dimension mismatch: array 'config['value']' is vector of length %d,"
                                   " but 'features' is a vector of length %d." % (np.asarray(config["value"]).shape[0],
                                                                                  features.shape[0]))
                if config["value"] == features or config["value"] == "*":
                    return e

        if not features.any():
            return "no-op"

        return ""

    @classmethod
    def get_noop_action(cls):
        """Creates a `no-op` action.

        A `no-op` action does not have any effect.

        Returns
        -------
        MDPAction :
            A `no-op` action.

        """
        if not isinstance(cls.nfeatures, int):
            raise ValueError("Attribute 'nfeatures' must be of <type 'int'>, got %s" % str(type(cls.nfeatures)))

        return cls(np.zeros(cls.nfeatures), "no-op")
