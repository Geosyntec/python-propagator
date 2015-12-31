""" Simple validation functions for ``propagator``.

This contains main functions use propagate and accumlate catchment
properties in a larger watershed.

(c) Geosyntec Consultants, 2015.

Released under the BSD 3-clause license (see LICENSE file for more info)

Written by Paul Hobson (phobson@geosyntec.com)

"""

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


def non_empty_list(list_obj, msg=None):
    if msg is None:
        msg = "list cannot be empty or None"

    if numpy.isscalar(list_obj):
        list_obj = [list_obj]

    if list_obj is None or len(list_obj) == 0:
        raise ValueError(msg)

    return list_obj

