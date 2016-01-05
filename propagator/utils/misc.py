""" Miscellaneous helper functions and classes.

This contains basic file I/O, coversion, and spatial analysis functions
to support the python-propagator library.

(c) Geosyntec Consultants, 2015.

Released under the BSD 3-clause license (see LICENSE file for more info)

Written by Paul Hobson (phobson@geosyntec.com)

"""


import os
import datetime
import itertools
from functools import wraps
from contextlib import contextmanager
from collections import namedtuple

import numpy


Statistic = namedtuple("Statistic", ("srccol", "aggfxn", "rescol"))


def _status(msg, verbose=False, asMessage=False, addTab=False): # pragma: no cover
    import arcpy
    if verbose:
        if addTab:
            msg = '\t' + msg
        if asMessage:
            arcpy.AddMessage(msg)
        else:
            print(msg)


def update_status(): # pragma: no cover
    """ Decorator to allow a function to take a additional keyword
    arguments related to printing status messages to stdin or as arcpy
    messages.

    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = kwargs.pop("msg", None)
            verbose = kwargs.pop("verbose", False)
            asMessage = kwargs.pop("asMessage", False)
            addTab = kwargs.pop("addTab", False)
            _status(msg, verbose=verbose, asMessage=asMessage, addTab=addTab)

            return func(*args, **kwargs)
        return wrapper
    return decorate


def find_row_in_array(array, column, value):
    """
    Find a single row in a record array.

    Parameters
    ----------
    array : numpy.recarray
        The record array to be searched.
    column : str
        The name of the column of the array to search.
    value : int, str, or float
        The value sought in ``column``

    Raises
    ------
    ValueError
        An error is raised if more than one row is found.

    Returns
    -------
    row : numpy.recarray row
        The found row from ``array``.

    Examples
    --------
    >>> from propagator import utils
    >>> import numpy
    >>> x = numpy.array(
            [
                ('A1', 'Ocean', 'A1_x'), ('A2', 'Ocean', 'A2_x'),
                ('B1', 'A1', 'None'), ('B2', 'A1', 'B2_x'),
            ], dtype=[('ID', '<U5'), ('DS_ID', '<U5'), ('Cu', '<U5'),]
        )
    >>> utils.find_row_in_array(x, 'ID', 'A1')
    ('A1', 'Ocean', 'A1_x', 'A1_y')

    """

    rows = filter(lambda x: x[column] == value, array)
    if len(rows) == 0:
        row = None
    elif len(rows) == 1:
        row = rows[0]
    else:
        raise ValueError("more than one row where {} == {}".format(column, value))

    return row


def rec_groupby(array, group_cols, *stats):
    """
    Perform a groupby-apply operation on a numpy record array.

    Returned record array has *dtype* names for each attribute name in
    the *groupby* argument, with the associated group values, and
    for each outname name in the *stats* argument, with the associated
    stat summary output. Adapted from https://goo.gl/NgwOID.

    Parameters
    ----------
    array : numpy.recarray
        The data to be grouped and aggregated.
    group_cols : str or sequence of str
        The columns that identify each group
    *stats : namedtuples or object
        Any number of namedtuples or objects specifying which columns
        should be aggregated, how they should be aggregated, and what
        the resulting column name should be. The keys/attributes for
        these tuples/objects must be: "srccol", "aggfxn", and "rescol".

    Returns
    -------
    aggregated : numpy.recarray

    See also
    --------
    Statistic

    Examples
    --------
    >>> from collections import namedtuple
    >>> from propagator import utils
    >>> import numpy
    >>> Statistic = namedtuple("Statistic", ("srccol", "aggfxn", "rescol"))
    >>> data = data = numpy.array([
            (u'050SC', 88.3, 0.0),  (u'050SC', 0.0, 0.1),
            (u'045SC', 49.2, 0.04), (u'045SC', 0.0, 0.08),
        ], dtype=[('ID', '<U10'), ('Cu', '<f8'), ('Pb', '<f8'),])
    >>> stats = [
            Statistic('Cu', numpy.max, 'MaxCu'),
            Statistic('Pb', numpy.min, 'MinPb')
        ]
    >>> utils.rec_groupby(data, ['ID'], *stats)
    rec.array(
        [(u'045SC', 49.2, 0.04),
         (u'050SC', 88.3, 0.0)],
        dtype=[('ID', '<U5'), ('MaxCu', '<f8'), ('MinPb', '<f8')]
    )

    """
    if numpy.isscalar(group_cols):
        group_cols = [group_cols]

    # build a dictionary from group_cols keys -> list of indices into
    # array with  those keys
    row_dict = dict()
    for i, row in enumerate(array):
        key = tuple([row[attr] for attr in group_cols])
        row_dict.setdefault(key, []).append(i)

    # sort the output by group_cols keys
    keys = list(row_dict.keys())
    keys.sort()

    output_rows = []
    for key in keys:
        row = list(key)

        # get the indices for this group_cols key
        index = row_dict[key]
        this_row = array[index]

        # call each aggregating function for this group_cols slice
        row.extend([stat.aggfxn(this_row[stat.srccol]) for stat in stats])
        output_rows.append(row)

    # build the output record array with group_cols and outname attributes
    outnames = [stat.rescol for stat in stats]
    names = list(group_cols)
    names.extend(outnames)
    record_array = numpy.rec.fromrecords(output_rows, names=names)
    return record_array


def stats_with_ignored_values(array, statfxn, ignored_value=None):
    """
    Compute statistics on arrays while ignoring certain values

    Parameters
    ----------
    array : numyp.array (of floats)
        The values to be summarized
    statfxn : callable
        Function, lambda, or classmethod that can be called with
        ``array`` as the only input and returns a scalar value.
    ignored_value : float, optional
        The values in ``array`` that should be ignored.

    Returns
    -------
    res : float
        Scalar result of ``statfxn``. In that case that all values in
        ``array`` are ignored, ``ignored_value`` is returned.

    Examples
    --------
    >>> import numpy
    >>> from propagator import utils
    >>> x = [1., 2., 3., 4., 5.]
    >>> utils.stats_with_ignored_values(x, numpy.mean, ignored_value=5)
    2.5

    """

    # ensure that we're working with an array
    array = numpy.asarray(array)

    # drop ignored values if we know what to ignore
    if ignored_value is not None:
        array = array[numpy.nonzero(array != ignored_value)]

    # if empty, return the ignored value
    if len(array) == 0:
        res = ignored_value
    else:
        res = statfxn(array)
    return res
