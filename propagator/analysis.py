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


@utils.update_status()
def prepare_data(mon_locations, monloc_id_col, subcatchments,
                 subcatch_id_col, final_fields, outputfile,
                 **verbose_options):
    """
    Assigns water quality ranking from monitoring locations to the
    subcatchments. If multiple monitoring locations are present in
    one subcatchment, water quality rankings are aggregated prior to
    assignment.

    Parameters
    ----------
    mon_locations : str
        File path to the monitoring locations feature class.
    monloc_id_col : str
        Field name of the monitoring location ID column in
        ``mon_locations``.
    subcatchments : str
        File path to the subcatchments feature class.
    subcatch_id_col : str
        Field name of the subcatchment ID column in ``subcatchments``.
    final_fields : list
        List of field names to be retained in the output feature class.
    outputfile : str
        Filename where the output should be saved.

    Returns
    -------
    subcatchment_wq : arcpy.Layer
        A subcatchment class that inherited water quality rankings
        from respective monitoring locations.

    """

    # Step 0. Load input and create temporary shapes
    raw_ml = utils.load_data(
        datapath=mon_locations,
        datatype="shape",
        msg='Loading Monitoring Location {}'.format(mon_locations),
        **verbose_options
    )

    raw_subcatchments = utils.load_data(
        datapath=subcatchments,
        datatype="shape",
        msg='Loading Monitoring Location {}'.format(subcatchments),
        **verbose_options
    )

    _reduced_ml = utils.create_temp_filename("reML", filetype='shape')
    _cat_ml_int = utils.create_temp_filename("Cat_ML_int", filetype='shape')
    _ml = utils.create_temp_filename("ml", filetype='shape')
    _out_ml = utils.create_temp_filename("out_ml", filetype='shape')
    subcatchment_wq = utils.create_temp_filename(outputfile, filetype='shape')

    # Step 1. Intersect ml and cat to generate a point shapefile
    # that contains an additional catid field showing where the ML
    # is located at.
    arcpy.analysis.Intersect([raw_subcatchments, raw_ml], _cat_ml_int, "ALL", "", "INPUT")

    # Step 2. Create a new point file (_reduced_ml) that pair only one
    # set of ranking data to each catchment.

    # extract all subcatchment ID that has at least one
    # monitoring location
    arr = utils.load_attribute_table(_cat_ml_int, subcatch_id_col, monloc_id_col)
    catid = numpy.unique(arr[subcatch_id_col])

    for lc, ucat in enumeriate(catid):
        sqlexp = '"{}" = \'{}\''.format(subcatch_id_col, ucat)
        arcpy.analysis.Select(_cat_ml_int, _ml, sqlexp)

        # if there is only one monitoring location  in the subcatchment,
        # this monitoring location is direclty copied to a new point
        # feature class.
        if lc == 0:
            fxn = arcpy.management.Copy
        # otherwise, aggregates monitoring locations to one single entry
        else:
            fxn = arcpy.management.Append

        # count number of MLs within each catchment
        count = arcpy.management.GetCount(_ml).getOutput(0)

        if count == 1:
            fxn(_ml, _reduced_ml)
        else:
            fxn(_reduce(_ml, _out_ml), _reduced_ml)

    # Step 3. spatial join _reduced_ml with raw_subcatchments, so that
    # the new cathcment shapefile will inherit water quality data from
    # the only monitoring location in it.
    arcpy.analysis.SpatialJoin(raw_subcatchments, _reduced_ml, subcatchment_wq)

    # removed extraneous columns
    fields_to_remove = filter(
        lambda name: name not in final_fields,
        [f.name for f in arcpy.ListFields(subcatchment_wq)]
    )
    utils.delete_columns(subcatchment_wq, *fields_to_remove)

    # clean up intermediate files
    utils.cleanup_temp_results(
        _reduced_ml,
        _cat_ml_int,
        _ml,
        _out_ml,
    )

    return subcatchment_wq


def _reduce(_ml, _out_ml):
    """
    Aggregates water quality ranksing from all monitoring locations
    in the same subcatchment.

    Parameters
    ----------
    _ml : str
        File path to the temporary monitoring location feature class
    _out_ml : str
        File path to an empty temporary feature class that will be
        filled in this function.

    Returns
    -------
    _out_ml : str

    """

    #load water quality data - place holder for now
    _arr = utils.load_attribute_table(_ml, 'FID_ml', 'Catch_ID_a')

    # for testing purpose at the moment, export the feature with the
    # smallest FID as output
    _sqlexp = '"FID_ml" = ' + "%i" % numpy.min(_arr['FID_ml'])
    arcpy.analysis.Select(_ml, _out_ml, _sqlexp)

    #process the WQ data - [e.g. out_ml.wq = average(temp_ml.wq)]
    return _out_ml

