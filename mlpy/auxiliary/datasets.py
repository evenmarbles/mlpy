"""
.. module:: mlpy.auxiliary.dataset
   :platform: Unix, Windows
   :synopsis: Manages recording of data.

.. moduleauthor:: Astrid Jackson <ajackson@eecs.ucf.edu>
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from .io import load_from_file, save_to_file


class DataSet(object):
    """The data set.

    The data set class a container for tracked data. Data can be tracked by
    adding a ``field`` for the data of interest. A :class:`numpy.ndarray` is created
    for every field that is added for recording. Optionally a `description`
    and a :class:`numpy.dtype` can be associated with the field.

    Parameters
    ----------
    capacity : int
        The initial capacity of the record. Defaults to 10.
    filename : str
        The name of the file to load from/save to the record.
    append : bool
        Whether to append to the existing records loaded from file
        or to overwrite data. Defaults to ``False``.

    Examples
    --------
    Creating a new dataset that stores its records in ``my_history.pkl``:

    >>> history = DataSet(capacity=2, filename="my_history.pkl")

    Adding a new field:

    >>> history.add_field("state", 3, dtype=DataSet.DTYPE_FLOAT)
    >>> print history
    state: dim(2,)
    []

    Adding a new data record:

    >>> import numpy as np
    >>> history.append("state", np.ones(3))

    Add a new sequence:

    >>> history.new_sequence()

    Save the dataset to file:

    >>> history.save()

    """
    DTYPE_OBJECT = np.object
    DTYPE_FLOAT = np.float64
    DTYPE_INT = np.int32

    def __init__(self, capacity=None, filename=None, append=None):
        self._sequence_index = -1

        self._description = {}
        self._data = {}
        self._endmarker = {}
        self._dtype = {}

        self._capacity = capacity if capacity is not None else 10
        self._append = append if append is not None else False
        self._filename = filename

    def load(self, filename=None):
        """Load the records from file.

        If filename is ``None``, the record is loaded from the
        class variable filename.

        Parameters
        ----------
        filename : str
            The name of the file.

        Raises
        ------
        ValueError
            If no filename is passed to the function and the member
            variable filename is `None`
        IOError
            If a file with the name does not exist.

        """
        if self._append is False:
            return

        if filename is None:
            filename = self._filename
        if filename is None:
            raise ValueError("No filename specified.")

        try:
            data = load_from_file(filename)

            check_capacity = True
            for name in data:
                idx = name.find("_descr")
                if idx == -1:
                    self._data[name] = data[name]
                    self._dtype[name] = self._get_record_info(data[name][self._sequence_index][0])[1]
                    self._endmarker[name] = self._data[name][self._sequence_index].shape[1]

                    if check_capacity:
                        self._capacity = data[name].shape[0]
                        check_capacity = False
                    continue

                self._description[name[:idx]] = data[name]

            self._sequence_index = self._capacity - 1
        except IOError:
            pass

    def save(self, filename=None):
        """Save the record to file.

        If filename is `None`, the record is saved to the class
        variable filename.

        Parameters
        ----------
        filename : str
            The name of the file

        Raises
        ------
        ValueError
            If no filename is passed to the function and the member variable
            filename is `None`.

        Notes
        -----
        If an error occurred during saving, the function fails silently.

        """
        if filename is None:
            filename = self._filename
        if filename is None:
            raise ValueError("No filename specified.")

        data = {}
        description = {}
        for name in self.get_field_names():
            data[name] = self._reduce(self._data[name], self._sequence_index + 1)
            data[name][self._sequence_index] = self._reduce(self._data[name][self._sequence_index],
                                                            self._endmarker[name], axis=1)
            if name in self._description:
                description[name + "_descr"] = self._description[name]

        if self._description:
            data.update(description)

        save_to_file(filename, data)

    def __str__(self):
        s = ""
        for name in self._data:
            s = s + name + ": dim" + str(self._data[name].shape) + "\n" + str(
                self._data[name][:self._endmarker[name]]) + "\n\n"
        return s

    def __getitem__(self, name):
        return self.get_field(name)

    def get_field_names(self):
        """Returns all field names.

        Returns
        -------
        tuple[str] :
            A list of field names.

        """
        return tuple(self._data.keys())

    def get_field(self, name):
        """Returns the field with the given name.

        Parameters
        ----------
        name : str
            The name of the field.

        Returns
        -------
        ndarray :
            If a field with that name exists, the field data is returned.

        """
        if self.has_field(name):
            return self._data[name]

    def has_field(self, name):
        """Checks if a field with that name exists.

        Parameters
        ----------
        name : str
            The name of the field.

        Returns
        -------
        bool :
            Whether a field with that name exists.

        """
        return name in self._data

    def add_field(self, name, dim, dtype=None, description=None):
        """Add a field with given the specifications.

        Parameters
        ----------
        name : str
            The name of the field.
        dim : int
            The dimensions of the field
        dtype : dtype
            The :class:`numpy.dtype` for the underlying :class:`numpy.ndarray`.
        description : str
            An optional description of the field.

        """
        if self._sequence_index < 0:
            self._sequence_index += 1

        if name not in self._data:
            self._data[name] = np.zeros((self._capacity,), dtype=np.object)

            if description is not None:
                self._description[name] = description

            dtype = dtype if dtype is not None else np.float64
            self._data[name][self._sequence_index] = np.zeros((dim, 50), dtype=dtype)
            self._dtype[name] = dtype
            self._endmarker[name] = 0

    def append(self, name, data):
        """Append a new data record.

        Append a new data record to the current sequence of samples of
        the field with the given `name`.

        Parameters
        ----------
        name : str
            The name of the field.
        data : str or int or float or ndarray
            The data record.

        """
        if name not in self._data:
            raise KeyError("Field '%s' not registered." % name)

        dim, size = self._data[name][self._sequence_index].shape
        if size <= self._endmarker[name]:
            self._data[name][self._sequence_index] = self._resize(self._data[name][self._sequence_index],
                                                                  shape=(dim, size * 2), axis=1,
                                                                  dtype=self._dtype[name])
        if dim > 1:
            data = np.asarray(data, dtype=self._dtype[name])
        self._data[name][self._sequence_index][:, self._endmarker[name]] = data
        self._endmarker[name] += 1

    def new_sequence(self):
        """Adds a new sequence.

        Adds a new sequence of samples for all fields and
        increments the sequence counter.

        """
        self._sequence_index += 1

        resize = False
        if self._capacity <= self._sequence_index:
            self._capacity *= 2
            resize = True

        for name in self.get_field_names():
            if resize:
                self._data[name] = self._resize(self._data[name], shape=(self._capacity,), dtype=np.object)

            self._data[name][self._sequence_index - 1] = self._reduce(self._data[name][self._sequence_index - 1],
                                                                      self._endmarker[name], axis=1)

            dim = self._data[name][self._sequence_index - 1].shape[0]
            self._data[name][self._sequence_index] = np.zeros((dim, 50), dtype=self._dtype[name])
            self._endmarker[name] = 0

    # noinspection PyMethodMayBeStatic
    def _resize(self, a, shape, axis=0, dtype=None):
        dtype = dtype if dtype is not None else np.float64

        size = a.shape[axis]
        data = np.zeros(shape, dtype=dtype)
        if axis == 0:
            data[:size] = a
        elif axis == 1:
            data[:, :size] = a
        a = data
        return a

    # noinspection PyMethodMayBeStatic
    def _reduce(self, a, endmarker, axis=0):
        if axis == 0:
            return a[:endmarker]

        return a[:, :endmarker]

    # noinspection PyMethodMayBeStatic
    def _get_record_info(self, record):
        dim = 1
        dtype = np.float64

        try:
            r = record.tolist()
        except:
            r = record

        if isinstance(r, list):
            dim = len(r)
            r = r[0]
        if isinstance(r, basestring):
            dtype = np.object
        if isinstance(r, int):
            dtype = np.int32
        return dim, dtype
