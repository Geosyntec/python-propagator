""" Top-level functions for ``propagator``.

This contains main functions use propagate and accumlate catchment
properties in a larger watershed.

(c) Geosyntec Consultants, 2015.

Released under the BSD 3-clause license (see LICENSE file for more info)

Written by Paul Hobson (phobson@geosyntec.com)

"""


import os
import sys
import glob
import datetime

import numpy

import arcpy

from . import utils


def trace_upstream(subcatchment_array, subcatchment_ID, id_col='ID',
                   ds_col='DS_ID', downstream=None):
    """
    Recursively traces an upstream path of subcatchments through a
    watetershed.

    Parameters
    ----------
    subcatchment_array : numpy.recarry
        A record array of all of the subcatchments in the watershed.
        This array must have a "downstrea ID" column in which each
        subcatchment identifies as single, downstream neighbor.
    subcatchment_ID : str
        The ID of the downstream catchment from which the trace
        originates.
    id_col : str, optional
        The name of the column that specifies the current subcatchment.
    ds_col : str, optional
        The name of the column that identifies the downstream
        subcatchment.
    downstream : list, optional
        A list of already known downstream catchments in the trace.
        .. note ::
           This is *only* used in the recursive calls to this function.
           You should never provide this value.

    Returns
    -------
    upstream : numpy.recarry
        A record array of all of the upstream subcatchments. This will
        have the same schema as ``subcatchment_array``

    """

    if downstream is None:
        downstream = []

    _neighbors = filter(lambda row: row[ds_col] == subcatchment_ID, subcatchment_array)

    for n in _neighbors:
        downstream.append(n)
        trace_upstream(subcatchment_array, n[id_col], downstream=downstream)

    return numpy.array(downstream, dtype=subcatchment_array.dtype)


def find_bottoms(subcatchment_array, bottom_ID='ocean', ds_col='DS_ID'):
    """
    Finds the lowest, non-ocean subcatchments in a watershed.

    Parameters
    ----------
    subcatchment_array : numpy.recarry
        A record array of all of the subcatchments in the watershed.
        This array must have a "downstrea ID" column in which each
        subcatchment identifies as single, downstream neighbor.
    bottom_ID : str, optional
        The subcatchment ID of the pseudo-catchments in the Ocean.
    ds_col : str, optional
        The name of the column that identifies the downstream
        subcatchment.

    Returns
    -------
    bottoms : numpy.recarry
        A record array of all of subcatchments that drain into the
        ocean.

    """

    bottoms = filter(lambda row: row[ds_col] == bottom_ID, subcatchment_array)
    return numpy.array(list(bottoms), dtype=subcatchment_array.dtype)


def find_tops(subcatchment_array, id_col='ID', ds_col='DS_ID'):
    """
    Finds the the subcatchments in a watershed that do not accept
    any upstrea tributary flow.

    Parameters
    ----------
    subcatchment_array : numpy.recarry
        A record array of all of the subcatchments in the watershed.
        This array must have a "downstrea ID" column in which each
        subcatchment identifies as single, downstream neighbor.
    subcatchment_ID : str
        The ID of the downstream catchment from which the trace
        originates.
    id_col : str, optional
        The name of the column that specifies the current subcatchment.
    ds_col : str, optional
        The name of the column that identifies the downstream
        subcatchment.

    Returns
    -------
    top : numpy.recarry
        A record array of all of the upper most subcatchments.

    """

    tops = filter(lambda r: r[id_col] not in subcatchment_array[ds_col], subcatchment_array)
    return numpy.array(list(tops), dtype=subcatchment_array.dtype)


def propagate_scores(subcatchment_array, testcol, null_value='None',
                     id_col='ID', ds_col='DS_ID', bottom_ID='Ocean'):
    """
    Propagate values into upstream subcatchments through a watershed.

    Parameters
    ----------
    subcatchment_array : numpy.recarry
        A record array of all of the subcatchments in the watershed.
        This array must have a "downstrea ID" column in which each
        subcatchment identifies as single, downstream neighbor.
    testcol : str
        Name of a representative water quality column that can be used
        to indicate if the given subcatchment has or has not been
        populated with water quality data.
    null_value : str, optional
        The string representing unpopulated values in the array of
        subcatchment and water quality data.
    id_col : str, optional
        The name of the column that specifies the current subcatchment.
    ds_col : str, optional
        The name of the column that identifies the downstream
        subcatchment.
    bottom_ID : str, optional
        The subcatchment ID of the pseudo-catchments in the Ocean.

    Returns
    -------
    propagated : numpy.recarry
        A copy of ``subcatchment_array`` with all of the water quality
        records populated.

    """

    propagated = subcatchment_array.copy()
    for n, row in enumerate(propagated):
        is_bottom = row[ds_col] == bottom_ID
        if row[testcol] == null_value and not is_bottom:
            vals = find_downstream_scores(propagated, row[ds_col], testcol)
            vals[id_col] = row[id_col]
            vals[ds_col] = row[ds_col]
            propagated[n] = vals

    return propagated


def find_downstream_scores(subcatchment_array, subcatchment_ID, testcol,
                           null_value='None', id_col='ID', ds_col='DS_ID'):
    """
    Recursively look for populated water quality score in downstream
    subcatchments.

    Parameters
    ----------
    subcatchment_array : numpy.recarry
        A record array of all of the subcatchments in the watershed.
        This array must have a "downstrea ID" column in which each
        subcatchment identifies as single, downstream neighbor.
    subcatchment_ID : str
        ID of the subcatchment whose water quality scores need to be
        populated.
    testcol : str
        Name of a representative water quality column that can be used
        to indicate if the given subcatchment has or has not been
        populated with water quality data.
    null_value : str, optional
        The string representing unpopulated values in the array of
        subcatchment and water quality data.
    id_col : str, optional
        The name of the column that specifies the current subcatchment.
    ds_col : str, optional
        The name of the column that identifies the downstream
        subcatchment.

    Returns
    -------
    vals : numpy.recarray
        Row of water quality score to be used for the subcatchment.

    """

    vals = filter(lambda r: r[id_col] == subcatchment_ID, subcatchment_array)[0]
    if vals[testcol] == null_value:
        return find_downstream_scores(subcatchment_array, vals[ds_col])
    else:
        return vals.copy()


def split_streams(stream_layer, subcatchment_layer):
    raise NotImplementedError
