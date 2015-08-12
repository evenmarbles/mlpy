from __future__ import division, print_function, absolute_import

import math
import numpy as np

from abc import ABCMeta, abstractmethod

from ...auxiliary.misc import listify


class FeatureFactory(object):
    """The feature factory.

    An instance of a feature can be created by passing
    the feature type.

    Examples
    --------
    >>> from mlpy.knowledgerep.cbr.features import FeatureFactory
    >>> FeatureFactory.create('float', **{})

    """
    @staticmethod
    def create(_type, name, value, **kwargs):
        """Create a feature of the given type.

        Parameters
        ----------
        _type: str
            The feature type. Valid feature types are:

            bool
                The feature values are boolean types (:class:`.BoolFeature`).

            string
                The feature values are of types sting (:class:`.StringFeature`).

            int
                The feature values are of type integer (:class:`.IntFeature`).

            float
                The feature values are of type float (:class:`.FloatFeature`).

        kwargs : dict, optional
            Non-positional arguments to pass to the class of the given type
            for initialization.

        Returns
        -------
        Feature :
            A feature instance of the given type.

        """
        try:
            return {
                "bool": BoolFeature,
                "string": StringFeature,
                "int": IntFeature,
                "float": FloatFeature,
            }[_type](name, value, **kwargs)

        except KeyError:
            return None


class Feature(object):
    """The abstract feature class.

    A feature consists of one or more feature values.

    Parameters
    ----------
    name : str
        The name of the feature (this also serves as the features identifying key).
    value : bool or string or int or float or list
        The feature value.

    Other Parameters
    ----------------
    weight : float or list[float]
        The weights given to each feature value.
    is_index : bool
        Flag indicating whether this feature is an index.
    retrieval_method : str
        The similarity model used for retrieval. Refer to
        :attr:`.Feature.retrieval_method` for valid methods.
    retrieval_method_params : dict
        Parameters relevant to the selected retrieval method.
    retrieval_algorithm : str
        The internal indexing structure of the training data. Refer
        to :attr:`.Feature.retrieval_method` for valid algorithms.
    retrieval_metric : str
        The metric used to compute the distances between pairs of points.
        Refer to :class:`sklearn.neighbors.DistanceMetric` for valid identifiers.
    retrieval_metric_params : dict
        Parameters relevant to the specified metric.

    """
    __metaclass__ = ABCMeta

    __slots__ = (
        '_name', '_value', '_weight', '_is_index', '_retrieval_method', '_retrieval_method_params',
        '_retrieval_algorithm', '_retrieval_metric', '_retrieval_metric_params')

    @property
    def name(self):
        """The name of the feature (this also serves as the features identifying key).

        Returns
        -------
        str :
            The name of the feature.

        """
        return self._name

    @property
    def value(self):
        """The feature value.

        Returns
        -------
        bool or string or int or float :
            The feature value.

        """
        return self._value

    @property
    def weight(self):
        """The weights given to each feature value

        Returns
        -------
        float or list[float] :
            The feature weights.

        """
        return self._weight

    @property
    def is_index(self):
        """Flag indicating whether this feature is an index.

        Returns
        -------
        bool :
            Whether the feature is an index.

        """
        return self._is_index

    @property
    def retrieval_method(self):
        """The similarity model used during retrieval.

        Valid models are:

            knn
                A k-nearest-neighbor algorithm is used to determine similarity
                between cases (:class:`NeighborSimilarity`). The value `k` must
                be specified.

            radius-n
                Similarity between cases is determined by the nearest neighbors
                within a radius (:class:`NeighborSimilarity`). The radius `n`
                must be specified.

            kmeans
                Similarity is determined by a kmeans clustering algorithm
                (:class:`KMeansSimilarity`).

            exact-match
                Only exact matches are considered similar (:class:`ExactMatchSimilarity`).

            cosine
                A cosine similarity measure is used to determine similarity between
                cases (:class:`CosineSimilarity`).

        Returns
        -------
        str :
            The retrieval method.

        """
        return self._retrieval_method

    @property
    def retrieval_method_params(self):
        """Parameters relevant to the specified retrieval method.

        Returns
        -------
        dict :
            Retrieval parameters.

        """
        return self._retrieval_method_params

    @property
    def retrieval_algorithm(self):
        """The internal indexing structure of the training data.

        The retrieval algorithm is only relevant for :class:`NeighborSimilarity`.
        Valid algorithms are:

            ball_tree
                A ball tree data structure is used for computational efficiency of
                the calculation of the distances between pairs of points.

            kd_tree
                A K-D Tree data structure is used for computational efficiency of
                the calculation of the distances between pairs of points.

            brute
                The nearest neighbors are determined by brute-force computation of
                distances between all pairs of points in the dataset.

            auto
                When ``auto`` is passed, the algorithm attempts to determine the best
                approach from the training data.

        Returns
        -------
        str :
            The retrieval algorithm.

        """
        return self._retrieval_algorithm

    @property
    def retrieval_metric(self):
        """The metric used to compute the distances between pairs of points.

        The retrieval metric is only relevant for :class:`NeighborSimilarity`.
        Refer to :class:`sklearn.neighbors.DistanceMetric` for valid metric
        identifiers.

        Returns
        -------
        str :
            The retrieval metric.
        """
        return self._retrieval_metric

    @property
    def retrieval_metric_params(self):
        """Parameters relevant to the specified metric.

        Returns
        -------
        dict :
            The retrieval metric parameters.

        """
        return self._retrieval_metric_params

    def __init__(self, name, value, **kwargs):
        self._name = name
        self._value = value

        self._weight = kwargs["weight"] if "weight" in kwargs else 1.0
        """:ivar: float"""
        self._is_index = kwargs["is_index"] if "is_index" in kwargs else True
        """:ivar: bool"""

        if "retrieval_method" in kwargs:
            if kwargs["retrieval_method"] not in ["knn", "radius-n", "kmeans", "exact-match", "cosine"]:
                raise ValueError("%s is not a valid retrieval method" % kwargs["retrieval_method"])
            self._retrieval_method = kwargs["retrieval_method"]
        else:
            self._retrieval_method = "knn"

        self._retrieval_method_params = kwargs[
            "retrieval_method_params"] if "retrieval_method_params" in kwargs else None

        if "retrieval_algorithm" in kwargs:
            if kwargs["retrieval_algorithm"] not in ["ball_tree", "kd_tree", "brute", "auto"]:
                raise ValueError("%s is not a valid retrieval algorithm" % kwargs["retrieval_algorithm"])
            self._retrieval_algorithm = kwargs["retrieval_algorithm"]
        else:
            self._retrieval_algorithm = "kd_tree"

        if "retrieval_metric" in kwargs:
            if kwargs["retrieval_metric"] not in ["euclidean", "minkowski", "manhattan", "chebyshev", "wminkowski"
                                                                                                      "seucliden",
                                                  "mahalanobis"]:
                raise ValueError("%s is not a valid retrieval metric" % kwargs["retrieval_metric"])
            self._retrieval_metric = kwargs["retrieval_metric"]
        else:
            self._retrieval_metric = "minkowski"

        self._retrieval_metric_params = kwargs["retrieval_metric_params"] if "retrieval_metric_params" in kwargs else 2

    @abstractmethod
    def compare(self, other):
        """Compare this feature to another feature.

        Parameters
        ----------
        other : Feature
            The other feature to compare this feature to.

        Returns
        -------
        float :
            The similarity metric.

        Raises
        ------
        NotImplementedError:
            If the child class does not implement this function.

        """
        raise NotImplementedError


class BoolFeature(Feature):
    """The boolean feature.

    The boolean feature is represented by a scalar.

    Parameters
    ----------
    name : str
        The name of the feature (this also serves as the features identifying key).
    value : bool or string or int or float or list
        The feature value.

    Other Parameters
    ----------------
    weight : float or list[float]
        The weights given to each feature value.
    is_index : bool
        Flag indicating whether this feature is an index.
    retrieval_method : str
        The similarity model used for retrieval. Refer to
        :attr:`.Feature.retrieval_method` for valid methods.
    retrieval_method_params : dict
        Parameters relevant to the selected retrieval method.
    retrieval_algorithm : str
        The internal indexing structure of the training data. Refer
        to :attr:`.Feature.retrieval_method` for valid algorithms.
    retrieval_metric : str
        The metric used to compute the distances between pairs of points.
        Refer to :class:`sklearn.neighbors.DistanceMetric` for valid identifiers.
    retrieval_metric_params : dict
        Parameters relevant to the specified metric.

    Raises
    ------
    ValueError :
        If the feature values is not of type `boolean`.

    """
    def __init__(self, name, value, **kwargs):
        super(BoolFeature, self).__init__(name, value, **kwargs)

        if not isinstance(value, bool):
            raise ValueError("The feature value is not of type `bool`.")

    def compare(self, other):
        """Compare this feature to another feature.

        The strings are compared directly and receive a similarity measure
        of `1` if they are the same, `0` otherwise.

        Parameters
        ----------
        other : Feature
            The other feature to compare this feature to.

        Returns
        -------
        float :
            The similarity metric.

        """
        if self.value == other.value:
            return 1.0
        return 0.0


class StringFeature(Feature):
    """The string feature.

    The string feature is represented by a scalar.

    Parameters
    ----------
    name : str
        The name of the feature (this also serves as the features identifying key).
    value : bool or string or int or float or list
        The feature value.

    Other Parameters
    ----------------
    weight : float or list[float]
        The weights given to each feature value.
    is_index : bool
        Flag indicating whether this feature is an index.
    retrieval_method : str
        The similarity model used for retrieval. Refer to
        :attr:`.Feature.retrieval_method` for valid methods.
    retrieval_method_params : dict
        Parameters relevant to the selected retrieval method.
    retrieval_algorithm : str
        The internal indexing structure of the training data. Refer
        to :attr:`.Feature.retrieval_method` for valid algorithms.
    retrieval_metric : str
        The metric used to compute the distances between pairs of points.
        Refer to :class:`sklearn.neighbors.DistanceMetric` for valid identifiers.
    retrieval_metric_params : dict
        Parameters relevant to the specified metric.

    Raises
    ------
    ValueError
        If the feature values is not of type `string`.

    """
    def __init__(self, name, value, **kwargs):
        super(StringFeature, self).__init__(name, value, **kwargs)

        if not isinstance(value, basestring):
            raise ValueError("The feature value is not of type `string`.")

    def compare(self, other):
        """Compare this feature to another feature.

        The strings are compared directly and receive a similarity measure
        of `1` if they are the same, `0` otherwise.

        Parameters
        ----------
        other : Feature
            The other feature to compare this feature to.

        Returns
        -------
        float :
            The similarity metric.

        """
        if self.value == other.value:
            return 1.0
        return 0.0


class IntFeature(Feature):
    """The integer feature.

    The integer feature is either represented by a scalar
    or by a list or values.

    Parameters
    ----------
    name : str
        The name of the feature (this also serves as the features identifying key).
    value : bool or string or int or float or list
        The feature value.

    Other Parameters
    ----------------
    weight : float or list[float]
        The weights given to each feature value.
    is_index : bool
        Flag indicating whether this feature is an index.
    retrieval_method : str
        The similarity model used for retrieval. Refer to
        :attr:`.Feature.retrieval_method` for valid methods.
    retrieval_method_params : dict
        Parameters relevant to the selected retrieval method.
    retrieval_algorithm : str
        The internal indexing structure of the training data. Refer
        to :attr:`.Feature.retrieval_method` for valid algorithms.
    retrieval_metric : str
        The metric used to compute the distances between pairs of points.
        Refer to :class:`sklearn.neighbors.DistanceMetric` for valid identifiers.
    retrieval_metric_params : dict
        Parameters relevant to the specified metric.

    Raises
    ------
    ValueError
        If not all feature values are of type `integer`.

    """
    def __init__(self, name, value, **kwargs):
        super(IntFeature, self).__init__(name, value, **kwargs)

        if isinstance(value, (np.ndarray, list)):
            for v in value:
                if not isinstance(v, (int, long)):
                    raise ValueError("The feature value is not of type `integer`.")

    def compare(self, other):
        """Compare this feature to another feature.

        If the feature is represented by a list the similarity
        between the two features is determined by the Euclidean
        distance of the feature values.

        Parameters
        ----------
        other : Feature
            The other feature to compare this feature to.

        Returns
        -------
        float :
            The similarity metric.

        """
        if isinstance(self._value, (int, long)):
            return self.value - other.value

        assert len(self._value) == len(other.value), "Features don't match"

        total_similarity = 0.0

        for i, val in enumerate(listify(self._value)):
            total_similarity += math.pow(val - other.value[i], 2)

        return math.sqrt(total_similarity)


class FloatFeature(Feature):
    """The float feature.

    The float feature is either represented by a scalar
    or by a list or values.

    Parameters
    ----------
    name : str
        The name of the feature (this also serves as the features identifying key).
    value : bool or string or int or float or list
        The feature value.

    Other Parameters
    ----------------
    weight : float or list[float]
        The weights given to each feature value.
    is_index : bool
        Flag indicating whether this feature is an index.
    retrieval_method : str
        The similarity model used for retrieval. Refer to
        :attr:`.Feature.retrieval_method` for valid methods.
    retrieval_method_params : dict
        Parameters relevant to the selected retrieval method.
    retrieval_algorithm : str
        The internal indexing structure of the training data. Refer
        to :attr:`.Feature.retrieval_method` for valid algorithms.
    retrieval_metric : str
        The metric used to compute the distances between pairs of points.
        Refer to :class:`sklearn.neighbors.DistanceMetric` for valid identifiers.
    retrieval_metric_params : dict
        Parameters relevant to the specified metric.

    Raises
    ------
    ValueError
        If not all feature values are of type `float`.

    """
    def __init__(self, name, value, **kwargs):
        super(FloatFeature, self).__init__(name, value, **kwargs)

        if isinstance(value, (np.ndarray, list)):
            for v in value:
                if not isinstance(v, float):
                    raise ValueError("The feature value is not of type `float`.")

    def compare(self, other):
        """Compare this feature to another feature.

        If the feature is represented by a list the similarity
        between the two features is determined by the Euclidean
        distance of the feature values.

        Parameters
        ----------
        other : Feature
            The other feature to compare this feature to.

        Returns
        -------
        float :
            The similarity metric.

        """
        if isinstance(self._value, float):
            return self.value - other.value

        assert len(self._value) == len(other.value), "Features don't match"

        total_similarity = 0.0

        for i, val in enumerate(listify(self._value)):
            total_similarity += math.pow(val - other.value[i], 2)

        return math.sqrt(total_similarity)
