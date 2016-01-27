""" Simple validation functions for ``propagator``.

This contains the main functions used to propagate and accumulate
subcatchment properties within a larger watershed.

(c) Geosyntec Consultants, 2015.

Released under the BSD 3-clause license (see LICENSE file for more info)

Written by Paul Hobson (phobson@geosyntec.com)

"""

from copy import copy

import numpy


def flow_direction(up_or_down):
    """
    Validates the direction of propation or accumulation

    Parameters
    ----------
    up_or_down : str
        The direction in which things will move. Valid values are either
        "upstream" or "downstream".

    Raises
    ------
    ValueError
        If the value of ``up_ot_down``.

    Returns
    -------
    validated : str
        Validated version of ``up_or_down``.

    """

    valid_directions = ['upstream', 'downstream']
    if up_or_down.lower() not in valid_directions:
        raise ValueError("{} is not one of {}".format(up_or_down, valid_directions))
    else:
        return up_or_down.lower()


def non_empty_list(list_obj, msg=None, on_fail='error'):
    """
    Validates a list as having at least one element.

    Parameters
    ----------
    list_obj : list or scalar
        The object that needs to be validated as a non-empty list. If
        a scalar value is provided, that value is placed inside a new
        list.
    msg : str, optional
        Custom error message to be raised.
    on_fail : str, optional
        Desired behavior when ``list_obj`` cannot be validated. Valid
        values are `"error"` and `"create"`. The former raises an error.
        The later returns an empty string.

    Raises
    ------
    ValueError
        A `ValueError` is raised when `list_obj` is an empty list or
        `None` and on_fail is set to `'error'`, which is the default.

    Returns
    -------
    validated : list
        The validated list object.

    Examples
    --------
    >>> from propagator import validate
    >>> validate.non_empty_list([1, 2, 3])
    [1, 2, 3]

    >>> try:
    ...     validate.non_empty_list([])
    ... except:
    ...     print("List was empty")
    List was empty

    >>> validate.non_empty_list([], on_fail='create')
    []

    >>> validate.non_empty_list(2)
    [2]

    """

    if msg is None:
        msg = "list cannot be empty or None"

    if numpy.isscalar(list_obj):
        list_obj = [list_obj]

    if list_obj is None or len(list_obj) == 0:
        if on_fail in ('error', 'raise'):
            raise ValueError(msg)
        elif on_fail in ('empty', 'create'):
            list_obj = []

    return list_obj


def value_column_stats(value_columns, default_second_value):
    """
    Validates a list's elements as being at least two-tuples.

    Parameters
    ----------
    value_columns : list of str or two-tuples
        List of columns to be aggregated. If a tuple, the first element
        specifies the column, the second specifies the statistic used
        to aggregate values in that column. If there is not second
        element, ``default_second_value`` is used.
    default_second_value
        The default aggregation method to be used when one is not
        provided.

    Returns
    -------
    validated : list
        List of two-tuples (<colname>, <stat. fxn>).

    """

    value_columns = non_empty_list(value_columns, on_fail='error')

    validated = copy(value_columns)
    for n, vc in enumerate(value_columns):
        if len(vc) == 1:
            vc = vc[0]
        if numpy.isscalar(vc):
            validated[n] = (vc, default_second_value)
        elif len(vc) == 0:
            raise ValueError("value_columns` cannot contain empty elements.")
    return validated
