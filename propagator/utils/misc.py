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

import numpy


def _status(msg, verbose=False, asMessage=False, addTab=False): # pragma: no cover
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


@update_status() # array
def flood_zones(zones_array, topo_array, elevation):
    """ Mask out non-flooded portions of arrays.

    Parameters
    ----------
    zones_array : numpy.array
        Array of zone IDs from each zone of influence.
    topo_array : numpy.array
        Digital elevation model (as an array) of the areas.
    elevation : float
        The flood elevation *above* which everything will be masked.

    Returns
    -------
    flooded_array : numpy.array
        Array of zone IDs only where there is flooding.

    """

    # compute mask of non-zoned areas of topo
    nonzone_mask = zones_array <= 0

    invalid_mask = numpy.ma.masked_invalid(topo_array).mask
    topo_array[invalid_mask] = -999

    # mask out zoned areas above the flood elevation
    unflooded_mask = topo_array > elevation

    # apply the mask to the zone array
    final_mask = nonzone_mask | unflooded_mask
    flooded_array = zones_array.copy()
    flooded_array[final_mask] = 0

    return flooded_array


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
