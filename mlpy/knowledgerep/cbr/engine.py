from __future__ import division, print_function, absolute_import

import math
import copy

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from ...auxiliary.misc import remove_key, listify
from ...knowledgerep.cbr.methods import CBRMethodFactory
from .features import FeatureFactory
from .similarity import SimilarityFactory


class CaseMatch(object):
    """Case match information.

    Parameters
    ----------
    case : Case
        The matching case.
    similarity :
        A measure for the similarity to the query case.

    Attributes
    ----------
    is_solution : bool
        Whether this case match is a solution to the query case or not.
    error : float
        The error of the prediction.
    predicted : bool
        Whether the query case could be correctly predicted using this
        case match.

    """
    __slots__ = ('_case', '_similarity', 'is_solution', 'error', 'predicted')

    @property
    def case(self):
        """The case that matches the query case.

        Returns
        -------
        Case :
            The case matching the query.
        """
        return self._case

    # noinspection PyShadowingNames
    def __init__(self, case, key, similarity=None):
        self._case = case

        self._similarity = {
            key: similarity
        }
        self.is_solution = False
        self.error = np.inf
        self.predicted = False

    def get_similarity(self, key):
        """Retrieve the similarity measure for the feature identified by the key.

        Returns
        -------
        float :
            The similarity measure.
        """
        return self._similarity[key]

    def set_similarity(self, key, value):
        """Set the similarity measure for the feature identified by the key."""
        self._similarity[key] = value


class Case(object):
    """The representation of a case in the case base.

    A case is composed of one or more :class:`Feature`.

    Parameters
    ----------
    cid : int
        The case's unique identifier.
    name : str
        The name for the case.
    description : str
        Text describing the case, optional.
    features : dict
        A list of features describing the case.

    """
    __slots__ = ('_id', '_name', '_description', '_features', 'ix')

    @property
    def id(self):
        """
        The case's unique identifier.

        :rtype: int
        """
        return self._id

    def __init__(self, cid, name=None, description=None, features=None):
        self._id = cid
        self._name = "" if name is None else name
        self._description = "" if description is None else description
        self._features = {} if features is None else copy.copy(features)

    def __getitem__(self, key):
        return self._features[key].value

    def __setitem__(self, key, value):
        self._features[key].value = value

    def __len__(self):
        return len(self._features)

    def __iter__(self):
        self.ix = 0
        return self

    def next(self):
        if self.ix == len(self._features):
            raise StopIteration
        item = self._features[self.ix][1]
        self.ix += 1
        return item

    def add_feature(self, name, _type, value, **kwargs):
        """Add a new feature.

        Parameters
        ----------
        name : str
            The name of the feature (this also serves as the features identifying key).
        _type : str
            The type of the feature. Valid feature types are:

            bool
                The feature values are boolean types (:class:`.BoolFeature`).

            string
                The feature values are of types sting (:class:`.StringFeature`).

            int
                The feature values are of type integer (:class:`.IntFeature`).

            float
                The feature values are of type float (:class:`.FloatFeature`).

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
        feature = FeatureFactory.create(_type, name, value, **kwargs)
        self._features[name] = feature

    def get_retrieval_method(self, names):
        """Returns the retrieval method for the given features.

        Parameters
        ----------
        names : str or list[str]
            The name(s) of the feature for which to retrieve the retrieval method.

        Returns
        -------
        str :
            The retrieval method for all feature. Features grouped together
            for retrieval must use the same retrieval method.

        Raises
        ------
        UserWarning
            If not all features use the same retrieval method.

        """
        method = None

        for n in listify(names):
            feat = self._features[n]

            if method is None:
                method = feat.retrieval_method
            elif not method == feat.retrieval_method:
                raise UserWarning("All features grouped for retrieval must use the same retrieval method.")

        return method

    def get_retrieval_params(self, names):
        """Return the retrieval parameters for the given features.

        Parameters
        ----------
        names : str or list[str]
            The name(s) of the feature for which to retrieve the retrieval parameters.

        Returns
        -------
        dict :
            The retrieval parameters for the feature(s). Features grouped together
            for retrieval must use the same retrieval parameters.

        Raises
        ------
        UserWarning
            If not all features use the same retrieval parameters.

        """
        params = {}

        for n in listify(names):
            feat = self._features[n]

            if not params:
                params["method_params"] = feat.retrieval_method_params
                params["algorithm"] = feat.retrieval_algorithm
                params["metric"] = feat.retrieval_metric
                params["metric_params"] = feat.retrieval_metric_params
            elif not (params["method_params"] == feat.retrieval_method_params and
                      params["algorithm"] == feat.retrieval_algorithm and
                      params["metric"] == feat.retrieval_metric and
                      params["metric_params"] == feat.retrieval_metric_param):
                raise UserWarning("All features grouped for retrieval must have the same retrieval parameters")

        return params

    # noinspection PyShadowingNames
    def get_indexed(self):
        """Return sorted collection of all indexed features.

        Returns
        -------
        list :
            The names of the indexed features in ascending order.

        """
        names = [x[1].name for x in self._features.items() if x[1].is_index]
        if len(names) == 1:
            return names[0]
        return names

    # noinspection PyShadowingNames
    def get_features(self, names):
        """Return sorted collection of features with the specified name.

        Parameters
        ----------
        names : str or list[str]
            The name(s) of the features to retrieve.

        Returns
        -------
        list or int or str or bool or float :
            List of features with the specified names(s)

        """
        if isinstance(names, list):
            return [x[1].value for x in self._features.items() if x[1].name in names]

        return self._features[names].value

    def compute_similarity(self, other):
        """Computes how similar two cases are.

        Parameters
        ----------
        other : Case
            The other case this case is compared to.

        Returns
        -------
        float :
            The similarity measure between the two cases.

        """
        total_similarity = 0.0

        for key, sfeature in self._features.iteritems():
            ofeature = other[key]

            if sfeature.is_index and ofeature.is_index:
                weight = sfeature.weight * ofeature.weight
                total_similarity += weight * math.pow(sfeature.compare(ofeature), 2)

        return math.sqrt(total_similarity)


class CaseBaseEntry(object):
    """The case base entry class.

    The entry maintains a similarity model from which similar cases can
    be derived.

    Internally the similarity model maintains an indexing structure
    dependent on the similarity model type for efficient computation
    of the similarity between cases. The case base is responsible for
    updating the indexing structure as cases are added and removed.

    Parameters
    ----------
    model : ISimilarity
        The similarity model.
    validity_check : bool
        This flag controls whether the dirty flag is being checked before
        determining whether to rebuild the model or not.

    Attributes
    ----------
    dirty : bool
        A flag which identifies whether the model needs to be rebuild.

        The indexing structure of the similarity model is always rebuild
        unless a validity check is required. If a validity check is required
        the indexing structure is only rebuild if the entry is considered dirty.

    """
    __slots__ = ('dirty', '_similarity', '_validity_check')

    def __init__(self, model, validity_check=True):
        self.dirty = True

        self._similarity = model
        self._validity_check = validity_check

    def compute_similarity(self, data_point, **kwargs):
        """Computes the similarity.

        Computes the similarity between the data point and each
        entry in the similarity model's indexing structure.

        Parameters
        ----------
        data_point : list[float]
            The data point to each entry in the similarity model is compared to.

        Returns
        -------
        list[Stat] :
            The similarity statistics of all entries in the model's indexing structure.

        Other Parameters
        ----------------
        cases : dict[int, Case]
            The complete case base from which to build the indexing structure used
            by the similarity model.
        data : ndarray[ndarray[float]]
            If this keyword is set, the cases in the case base are ignored and the
            data entries specified in this variable are used to build the indexing
            structure.
        names : str or list[str]
            The feature name(s) relevant for the similarity computation. This field
            is only required if the cases for building the indexing structure comes
            from the `cases` field.
        id_map : dict[int, int]
            The mapping from the data stored in the `data` field to their case ids.
            This field is only required if the data for building the indexing structure
            comes from the `data` field.

        """
        self._build_indexing_structure(**kwargs)

        return self._similarity.compute_similarity(data_point)

    def _build_indexing_structure(self, **kwargs):
        """Build the indexing structure.

        Build the indexing structure of the similarity model for specific feature names
        or alternatively for the cases provided in the `data` field.

        Parameters
        ----------
        cases : dict[int, Case]
            The complete case base from which to build the indexing structure used
            by the similarity model.
        data : ndarray[ndarray[float]]
            If this keyword is set, the cases in the case base are ignored and the
            data entries specified in this variable are used to build the indexing
            structure.
        names : str or list[str]
            The feature name(s) relevant for the similarity computation. This field
            is only required if the cases for building the indexing structure comes
            from the `cases` field.
        id_map : dict[int, int]
            The mapping from the data stored in the `data` field to their case ids.
            This field is only required if the data for building the indexing structure
            comes from the `data` field.

        """
        if not self._validity_check or self.dirty:
            try:
                data = kwargs["data"]
                id_map = kwargs["id_map"]
            except KeyError:
                data = None
                id_map = {}

                for i, c in enumerate(kwargs["cases"].itervalues()):
                    feature_list = c.get_features(kwargs["names"])

                    if data is None:
                        data = np.empty((0, len(feature_list)), dtype=np.float64)
                    data = np.vstack([data, feature_list])
                    id_map[i] = c.id

            self._similarity.build_indexing_structure(data, id_map)
            self.dirty = False


class CaseBase(object):
    # noinspection PyTypeChecker
    """The case base engine.

    The case base engine maintains the a database of all cases entered
    into the case base. It manages retrieval, revision, reuse, and retention
    of cases.

    Parameters
    ----------
    case_template: dict
        The template from which to create a new case.

        :Example:

            An example template for a feature named ``state`` with the
            specified feature parameters. ``data`` is the data from which
            to extract the case from. In this example it is expected that
            ``data`` has a member variable ``state``.

            ::

                {
                    "state": {
                        "type": "float",
                        "value": "data.state",
                        "is_index": True,
                        "retrieval_method": "radius-n",
                        "retrieval_method_params": 0.01
                    },
                    "delta_state": {
                        "type": "float",
                        "value": "data.next_state - data.state",
                        "is_index": False,
                    }
                }

    reuse_method : str
        The reuse method name to be used during the reuse step. Default is
        `defaultreusemethod`.
    reuse_method_params : dict
        Non-positional initialization parameters for the reuse method instantiation.
    revision_method : str
        The revision method name to be used during the revision step. Default is
        `defaultrevisionmethod`.
    revision_method_params : dict
        Non-positional initialization parameters for the revision method instantiation.
    retention_method : str
        The retention method name to be used during the retention step. Default is
        `defaultretentionmethod`.
    retention_method_params : dict
        Non-positional initialization parameters for the retention method instantiation.
    plot_retrieval : bool
        Whether to plot the result or not. Default is False.
    plot_retrieval_names : str or list[str]
        The names of the feature which to plot.

    Examples
    --------
    Create a case base:

    >>> from mlpy.auxiliary.io import load_from_file
    >>>
    >>> template = {}
    >>> cb = CaseBase(template)

    Fill case base with data read from file:

    >>> from mlpy.mdp.stateaction import Experience, MDPState, MDPAction
    >>>
    >>> data = load_from_file("data/jointsAndActionsData.pkl")
    >>> for i in xrange(len(data.itervalues().next())):
    ...     for j in xrange(len(data.itervalues().next()[0][i]) - 1):
    ...         if not j == 10:  # exclude one experience as test case
    ...             experience = Experience(MDPState(data["states"][i][:, j]),
    ...                                     MDPAction(data["actions"][i][:, j]),
    ...                                     MDPState(data["states"][i][:, j + 1]))
    ...             cb.run(cb.case_from_data(experience))


    Loop over all cases in the case base:

    >>> for i in len(cb):
    ...     pass

    Retrieve case with ``id=0``:

    >>> case = cb[0]

    """
    @property
    def counter(self):
        """The case counter.

        The counter is increased with every case added to the case base.

        Returns
        -------
        int :
            The current count.

        """
        return self._counter

    @property
    def cases(self):
        """The unadulterated cases.

        Returns
        -------
        dict :
            The cases in the case base.
        """
        return self._cases

    def __init__(self, case_template, reuse_method=None, reuse_method_params=None, revision_method=None,
                 revision_method_params=None, retention_method=None, retention_method_params=None,
                 plot_retrieval=None, plot_retrieval_names=None):
        #: Collection of the unadulterated cases.
        self._cases = {}
        """:type: dict[int, Case]"""

        #: The cases database keeping a similarity model for a
        #: collection of cases specified by their feature names
        self._cb = {}
        """:type: dict[str|tuple[str], CaseBaseEntry]"""

        self._counter = 0
        """:type: int"""

        self._case_template = case_template
        """:type: dict"""

        try:
            reuse_method_params = reuse_method_params if reuse_method_params is not None else {}
            self._reuse_method = CBRMethodFactory.create(reuse_method, self, **reuse_method_params)
        except:
            self._reuse_method = None
        """:type: IReuseMethod"""

        try:
            revision_method_params = revision_method_params if revision_method_params is not None else {}
            self._revision_method = CBRMethodFactory.create(revision_method, self, **revision_method_params)
        except:
            self._revision_method = None
        """:type: IRevisionMethod"""

        try:
            retention_method = retention_method if retention_method is not None else 'defaultretentionmethod'
            retention_method_params = retention_method_params if retention_method_params is not None else {}
            self._retention_method = CBRMethodFactory.create(retention_method, self, **retention_method_params)
            """:type: IRetentionMethod"""
        except:
            raise ValueError("%s is not a valid retention method" % retention_method)

        self._plot_retrieval = plot_retrieval if plot_retrieval is not None else False
        """:type: bool"""
        if self._plot_retrieval:
            self._plot_retrieval_names = plot_retrieval_names if plot_retrieval_names is not None else None
            """:type: str | list[str]"""

            self._fig = None
            self._ax = None

    def __getitem__(self, key):
        return self._cases[key]

    def __len__(self):
        return len(self._cases)

    def __iter__(self):
        self.ix = 0
        return self

    def next(self):
        if self.ix == len(self._cases):
            raise StopIteration
        item = self._cases[self.ix]
        self.ix += 1
        return item

    def load(self, filename):
        pass

    def save(self, filename):
        pass

    def get_new_id(self):
        """Return an unused case id.

        Returns
        -------
        int :
            Unused case ID.

        """
        return self.counter

    def add(self, case):
        """Add a new case without any checks.

        Parameters
        ----------
        case : Case
            The case to add to the case base.

        """
        self._cases[self._counter] = case
        self._counter += 1
        for c in self._cb.itervalues():
            c.dirty = True

    def run(self, case):
        """Run the case base.

        Run the case base using the CBR methods retrieve, reuse,
        revision and retention.

        Parameters
        ----------
        case : Case
            The query case

        Returns
        -------
        dict[int, CaseMatch] :
            The solution to the problem-solving experience

        """
        case_matches = self.retrieve(case)
        solution = self.reuse(case, case_matches)
        solution = self.revision(case, solution)
        self.retain(case, solution)

        return solution

    def retrieve(self, case, names=None, validity_check=True, **kwargs):
        """Retrieve cases similar to the query case.

        Parameters
        ----------
        case : Case
            The query case.
        names : str or list[str]
            The name(s) of the features for which to retrieve similar cases.
        validity_check : bool
            This flag controls whether the dirty flag is being checked before
            determining whether to rebuild the indexing structure or not.

        Returns
        -------
        dict[int, CaseMatch] :
            The solution to the problem-solving experience.

        Other Parameters
        ----------------
        cases : dict[int, Case]
            The complete case base from which to build the indexing structure used
            by the similarity model.
        data : ndarray[ndarray[float]]
            If this keyword is set, the cases in the case base are ignored and the
            data entries specified in this variable are used to build the indexing
            structure.
        names : str or list[str]
            The feature name(s) relevant for the similarity computation. This field
            is only required if the cases for building the indexing structure comes
            from the `cases` field.
        id_map : dict[int, int]
            The mapping from the data stored in the `data` field to their case ids.
            This field is only required if the data for building the indexing structure
            comes from the `data` field.

        """
        if len(self._cases) == 0:
            return {}

        if names is None:
            names = case.get_indexed()
        key = tuple(names) if isinstance(names, list) else names

        # Update the similarity model
        if key not in self._cb:
            self._cb[key] = CaseBaseEntry(
                SimilarityFactory.create(case.get_retrieval_method(names),
                                         **case.get_retrieval_params(names)), validity_check)

        if "data" not in kwargs and "id_map" not in kwargs:
            kwargs["cases"] = self._cases
            kwargs["names"] = names

        stats = self._cb[key].compute_similarity(case.get_features(names), **kwargs)

        if self._plot_retrieval and names == self._plot_retrieval_names:
            self.plot_retrieval(case, [s.case_id for s in stats], names)

        return {s.case_id: CaseMatch(self._cases[s.case_id], names, s.similarity) for s in stats}

    def reuse(self, case, case_matches):
        """Performs the reuse step

        Performs new generalizations and specializations as a consequence of
        the solution transformation.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution to the problem-solving experience.

        Returns
        -------
        dict[int, CaseMatch] :
            The revised solution to the problem-solving experience.

        """
        if not case_matches:
            return {}

        return self._reuse_method.execute(case, case_matches) if self._reuse_method is not None else case_matches

    def revision(self, case, case_matches):
        """Evaluate solution provided by problem-solving experience.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The revised solution to the problem-solving experience.

        Returns
        -------
        dict[int, CaseMatch] :
            The corrected solution.

        """
        if not case_matches:
            return {}

        return self._revision_method.execute(case, case_matches) if self._revision_method is not None else case_matches

    def retain(self, case, case_matches):
        """Retain new case.

        Retain new case depending on the revise outcomes and the
        CBR policy regarding case retention.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The corrected solution

        """
        self._retention_method.execute(case, case_matches)

    def plot_retrieval(self, case, case_id_list, names=None):
        """Plot the retrieved data.

        Parameters
        ----------
        case : Case
            The query case.
        case_id_list : list[int]
            The ids of the cases identified to be similar.
        names : str or list[str]
            The name(s) of the features for which similar cases were retrieve.

        """
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            self._fig = plt.figure()
            plt.rcParams['legend.fontsize'] = 10
            self._fig.suptitle('Similarity: {0}'.format(names))
            self._ax = self._fig.add_subplot(111, projection='3d')
            self._fig.show()

        self._ax.cla()

        [x, y, z] = case.get_features(names)
        self._ax.scatter(x, y, z, edgecolors='r', c='r', marker='o')

        for c in self._cases.itervalues():
            [xs, ys, zs] = c.get_features(names)
            if c.id in case_id_list:
                self._ax.scatter(xs, ys, zs, edgecolors='g', c='g', marker='^')
            else:
                self._ax.scatter(xs, ys, zs, c='k', marker='o')

        scatter1_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='r', c='r', marker='o')
        scatter2_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='k', c='k', marker='o')
        scatter3_proxy = matplotlib.lines.Line2D([0], [0], linestyle="none", markeredgecolor='g', c='g', marker='^')
        self._ax.legend([scatter1_proxy, scatter2_proxy, scatter3_proxy],
                        ['query case', 'cases', 'similar'], numpoints=1)

        self._ax.set_xlabel('X position')
        self._ax.set_ylabel('Y position')
        self._ax.set_zlabel('Z position')

        self._fig.canvas.draw()

    def plot_reuse(self, case, case_matches, revised_matches):
        """Plot the reuse result.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution to the problem-solving experience.
        revised_matches : dict[int, CaseMatch]
            The revised solution to the problem-solving experience.

        """
        self._reuse_method.plot_data(case, case_matches, revised_matches)

    def plot_revision(self, case, case_matches):
        """Plot revision results.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The revised solution to the problem-solving experience.

        """
        self._revision_method.plot_data(case, case_matches)

    def plot_retention(self, case, case_matches):
        """Plot the retention result.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The corrected solution

        """
        self._retention_method.plot_data(case, case_matches)

    # noinspection PyUnusedLocal
    def case_from_data(self, data):
        """Convert data into a case using the case template.

        Parameters
        ----------
        data :
            The data from which to extract the case.

        Returns
        -------
        Case :
            The case extracted from the data.
        """
        feature_list = {}
        for key, t in self._case_template.iteritems():
            type_, params = remove_key(t, "type")
            value, params = remove_key(params, "value")
            feature_list[key] = FeatureFactory.create(type_, key, eval(value), **params)

        return Case(self.get_new_id(), features=feature_list)
