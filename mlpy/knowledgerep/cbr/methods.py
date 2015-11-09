from __future__ import division, print_function, absolute_import

from abc import abstractmethod
from ...modules.patterns import RegistryInterface


class CBRMethodFactory(object):
    """The case base reasoning factory.

    An instance of a case base reasoning method can be created by passing
    the case base reasoning method type. Case base reasoning methods are
    used to implement task specific reuse, revision, and retention.

    Examples
    --------
    >>> from mlpy.knowledgerep.cbr.methods import CBRMethodFactory
    >>> CBRMethodFactory.create('defaultreusemethod', **{})

    """
    @staticmethod
    def create(_type, *args, **kwargs):
        """Create an case base reasoning method of the given type.

        _type : str
            The case base reasoning method type. The method type should be
            equal to the class name of the method.
        args : tuple, optional
            Positional arguments passed to the class of the given type for
            initialization.
        kwargs : dict, optional
            Non-positional arguments passed to the class of the given type
            for initialization.

        Returns
        -------
        ICBRMethod :
            A case base method instance of the given type.

        """
        # noinspection PyUnresolvedReferences
        return ICBRMethod.registry[_type.lower()](*args, **kwargs)


class ICBRMethod(object):
    """The method interface.

    This is the interface for reuse, revision, and retention methods and handles
    registration of all subclasses.

    It uses the :class:`.RegistryInterface` ensuring that all subclasses are
    registered. Therefore, new specialty case base reasoning method classes can
    be defined and are recognized by the case base reasoning engine.

    Parameters
    ----------
    owner : CaseBase
        A pointer to the owning case base.

    Notes
    -----
    All case base reasoning method must inherit from this class.

    """
    __metaclass__ = RegistryInterface

    def __init__(self, owner):
        self._owner = owner

    @abstractmethod
    def execute(self, case, case_matches):
        """Execute reuse step.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def plot_data(self, case, case_matches):
        """Plot the data.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        """
        pass


class IReuseMethod(ICBRMethod):
    """The reuse method interface.

    The solutions of the best (or set of best) retrieved cases are used to construct
    the solution for the query case; new generalizations and specializations may occur
    as a consequence of the solution transformation.

    Parameters
    ----------
    owner : CaseBase
        A pointer to the owning case base.

    Notes
    -----
    All reuse method implementations must inherit from this class.

    """
    @abstractmethod
    def execute(self, case, case_matches):
        """Execute reuse step.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        Returns
        -------
        dict[int, CaseMatch] :
            The revised solution.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def plot_data(self, case, case_matches):
        """Plot the data.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        """
        pass


class IRevisionMethod(ICBRMethod):
    """The revision method interface.

    The solutions provided by the query case is evaluated and information about whether the solution
    has or has not provided a desired outcome is gathered.

    Parameters
    ----------
    owner : CaseBase
        A pointer to the owning case base.

    Notes
    -----
    All revision method implementations must inherit from this class.

    """
    @abstractmethod
    def execute(self, case, case_matches):
        """Execute the revision step.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        Returns
        -------
        dict[int, CaseMatch] :
            The corrected solution.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def plot_data(self, case, case_matches):
        """Plot the data.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        """
        pass


class IRetentionMethod(ICBRMethod):
    """The retention method interface.

    When the new problem-solving experience can be stored or not stored in memory, depending on
    the revision outcomes and the case base reasoning policy regarding case retention.

    Parameters
    ----------
    owner : CaseBase
        A pointer to the owning case base.

    Notes
    -----
    All retention method implementations must inherit from this class.

    """
    @abstractmethod
    def execute(self, case, case_matches):
        """Execute retention step.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        Raises
        ------
        NotImplementedError
            If the child class does not implement this function.

        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def plot_data(self, case, case_matches):
        """Plot the data.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        """
        pass


class DefaultReuseMethod(IReuseMethod):
    """The default reuse method implementation.

    The solutions of the best (or set of best) retrieved cases are used to construct
    the solution for the query case; new generalizations and specializations may occur
    as a consequence of the solution transformation.

    Parameters
    ----------
    owner : CaseBase
        A pointer to the owning case base.

    Notes
    -----
    The default reuse method does not perform any solution transformations.

    """
    # noinspection PyUnusedLocal
    def __init__(self, owner):
        super(DefaultReuseMethod, self).__init__(owner)

    def execute(self, case, case_matches):
        """
        Execute reuse step.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        Returns
        -------
        dict[int, CaseMatch] :
            The revised solution.

        """
        return case_matches


class DefaultRevisionMethod(IRevisionMethod):
    """
    The default revision method implementation called from the case base.

    The solutions provided by the query case is evaluated and information about whether the solution
    has or has not provided a desired outcome is gathered.

    Parameters
    ----------
    owner : CaseBase
        A pointer to the owning case base.

    Notes
    -----
    The default revision method returns the original solution without making any modifications.
    """

    # noinspection PyUnusedLocal
    def __init__(self, owner):
        super(DefaultRevisionMethod, self).__init__(owner)

    def execute(self, case, case_matches):
        """Execute the revision step.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        Returns
        -------
        dict[int, CaseMatch] :
            The corrected solution.

        """
        return case_matches


class DefaultRetentionMethod(IRetentionMethod):
    """
    The default retention method implementation called from the case base.

    When the new problem-solving experience can be stored or not stored in memory, depending on
    the revision outcomes and the case base reasoning policy regarding case retention.

    Parameters
    ----------
    owner : CaseBase
        A pointer to the owning case base.
    max_error : float
        The maximum permitted error.

    Notes
    -----
    The default retention method adds the new experience only if the query case is within
    the maximum permitted error of the most similar solution case:

    .. math::
        d(\\text{case}, \\text{solution}[0]) < \\text{max\_error}.

    """
    def __init__(self, owner, max_error=None):
        super(DefaultRetentionMethod, self).__init__(owner)

        self._max_error = max_error if max_error is not None else 1e-06

    def execute(self, case, case_matches):
        """Execute retention step.

        Parameters
        ----------
        case : Case
            The query case.
        case_matches : dict[int, CaseMatch]
            The solution identified through the similarity measure.

        """
        # noinspection PyUnresolvedReferences
        key = case.get_indexed()
        if not case_matches or case_matches[
           min(case_matches, key=lambda x: case_matches[x].get_similarity(key))].get_similarity(key) < self._max_error:
            self._owner.add(case)
