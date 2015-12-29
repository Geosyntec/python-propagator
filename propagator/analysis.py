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


def propagate_scores(subcatchment_array, wq_column, null_value='None',
                     id_col='ID', ds_col='DS_ID', bottom_ID='Ocean'):
    """
    Propagate values into upstream subcatchments through a watershed.

    Parameters
    ----------
    subcatchment_array : numpy.recarry
        A record array of all of the subcatchments in the watershed.
        This array must have a "downstrea ID" column in which each
        subcatchment identifies as single, downstream neighbor.
    wq_column : str
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

    # copy the input array so that we always have the
    # original to compare to.
    propagated = subcatchment_array.copy()

    # loop through the array
    for n, row in enumerate(propagated):

        # check to see if we're at the bottom of the watershed
        is_bottom = row[ds_col] == bottom_ID

        # look for a downstream value if there is not value
        # and we're not already at the bottom
        if row[wq_column] == null_value and not is_bottom:
            # find the downstream value
            ds_values = find_downstream_scores(propagated, row[ds_col], wq_column)

            # assign the downstream value to the current (empty) value
            propagated[wq_column][n] = ds_values[wq_column]

    return propagated


def find_downstream_scores(subcatchment_array, subcatchment_ID, wq_column,
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
    wq_column : str
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

    vals = utils.find_row_in_array(subcatchment_array, id_col, subcatchment_ID)
    if vals[wq_column] == null_value:
        return find_downstream_scores(subcatchment_array, vals[ds_col], wq_column,
                                      null_value=null_value, id_col=id_col,
                                      ds_col=ds_col)
    else:
        return vals.copy()


def update_water_quality_layer(layer, water_quality):
    raise NotImplementedError


def split_streams(stream_layer, subcatchment_layer):
    raise NotImplementedError
