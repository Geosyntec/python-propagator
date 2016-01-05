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
import itertools
from functools import partial

import numpy

import arcpy

from . import utils
from . import validate


@utils.update_status()
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

        .. warning ::
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
        trace_upstream(subcatchment_array, n[id_col],
                      id_col=id_col, ds_col=ds_col,
                      downstream=downstream)

    return numpy.array(downstream, dtype=subcatchment_array.dtype)


@utils.update_status()
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


@utils.update_status()
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


@utils.update_status()
def propagate_scores(subcatchment_array, id_col, ds_col, value_column,
                     ignored_value=0, bottom_ID='Ocean'):
    """
    Propagate values into upstream subcatchments through a watershed.

    Parameters
    ----------
    subcatchment_array : numpy.recarry
        A record array of all of the subcatchments in the watershed.
        This array must have a "downstrea ID" column in which each
        subcatchment identifies as single, downstream neighbor.
    id_col : str, optional
        The name of the column that specifies the current subcatchment.
    ds_col : str, optional
        The name of the column that identifies the downstream
        subcatchment.
    value_column : str
        Name of a representative water quality column that can be used
        to indicate if the given subcatchment has or has not been
        populated with water quality data.
    ignored_value : float, optional
        The values representing unpopulated records in the array of
        subcatchment and water quality data.
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
        is_bottom = row[ds_col].lower() == bottom_ID.lower()

        # look for a downstream value if there is not value
        # and we're not already at the bottom
        if row[value_column] == ignored_value and not is_bottom:
            # find the downstream value
            ds_values = _find_downstream_scores(
                subcatchment_array=propagated,
                subcatchment_ID=row[ds_col],
                value_column=value_column,
                ignored_value=ignored_value,
                id_col=id_col,
                ds_col=ds_col,
            )

            # assign the downstream value to the current (empty) value
            propagated[value_column][n] = ds_values[value_column]

    return propagated


@utils.update_status()
def _find_downstream_scores(subcatchment_array, subcatchment_ID, value_column,
                            ignored_value='None', id_col='ID', ds_col='DS_ID',
                            bottom_ID='Ocean'):
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
    value_column : str
        Name of a representative water quality column that can be used
        to indicate if the given subcatchment has or has not been
        populated with water quality data.
    ignored_value : float, optional
        The values representing unpopulated records in the array of
        subcatchment and water quality data.
    id_col : str, optional
        The name of the column that specifies the current subcatchment.
    ds_col : str, optional
        The name of the column that identifies the downstream
        subcatchment.

    Returns
    -------
    scores : numpy record
        Row of water quality score to be used for the subcatchment.

    """

    row = utils.find_row_in_array(subcatchment_array, id_col, subcatchment_ID)

    # check to see if we're at the bottom of the watershed
    is_bottom = row[ds_col].lower() == bottom_ID.lower()
    if row[value_column] == ignored_value and not is_bottom:
        scores = _find_downstream_scores(
            subcatchment_array=subcatchment_array,
            subcatchment_ID=row[ds_col],
            value_column=value_column,
            ignored_value=ignored_value,
            id_col=id_col,
            ds_col=ds_col
        )
    else:
        scores = row.copy()

    return scores


@utils.update_status()
def preprocess_wq(monitoring_locations, subcatchments, id_col, ds_col,
                  output_path, value_columns=None, aggfxn=numpy.mean,
                  ignored_value=0, cleanup=True):
    """
    Preprocess the water quality data to have to averaged score for
    each subcatchment.

    Parameters
    ----------
    monitoring_locations : str
        Path to the feature class containing the monitoring locations
        and their water quality scores.
    subcatchments : str
        Path to the feature class containing the subcatchment
        boundaries.
    id_col : str, ds_col
        Name of the column in ``subcatchments`` that contains the
        (ds = downstream) subcatchment IDs.
    output_path : str
        Path of the new feature class where the preprocessed data
        should be saved.
    value_columns : list of str
        A list of the names of the fields containing water quality
        scores that need to be analyzed.
    aggfxn : callable, optional
        A function, lambda, or class method that reduces arrays into
        scalar values. By default, this is ``numpy.mean``.
    ignored_value : int, optional
        The values in ``monitoring_locations`` that should be ignored.
        Given the default input datasets, zero has been chosen to
        signal that a value is missing.
    cleanup : bool, optional
        Toggles the deletion of temporary files.

    Returns
    -------
    array : numpy.recarray
        A numpy record array of the subcatchments with their aggregated
        water quality scores.

    """

    # validate value_columns
    value_columns = validate.non_empty_list(value_columns, msg="you must provide `value_columns` to aggregate")

    # create the output feature class as a copy of the `subcatchments`
    output_path = utils.copy_layer(subcatchments, output_path)

    # associate subcatchment IDs with all of the monitoring locations
    joined = utils.intersect_layers(
        input_paths=[monitoring_locations, subcatchments],
        output_path=utils.create_temp_filename("joined_ml_sc", filetype='shape'),
        how="ALL",
    )

    # define the Statistic objects that will be passed to `rec_groupby`
    statfxn = partial(
        utils.stats_with_ignored_values,
        statfxn=aggfxn,
        ignored_value=ignored_value
    )

    res_columns = ['avg{}'.format(col) for col in value_columns]
    statistics = [
        utils.Statistic(srccol, statfxn, rescol)
        for srccol, rescol in zip(value_columns, res_columns)
    ]

    # compile the original fields to read in from the joined table
    orig_fields = [id_col, ds_col]
    orig_fields.extend([stat.srccol for stat in statistics])
    array = utils.load_attribute_table(joined, *orig_fields)

    # compile the final results (aggregated) fields for the output
    final_fields = [id_col, ds_col]
    final_fields.extend([stat.rescol for stat in statistics])

    # aggregate the data within each subcatchment
    aggregated = utils.rec_groupby(array, orig_fields[:2], *statistics)

    # add the new columns for the aggregated data to the output
    for rescol in res_columns:
        utils.add_field_with_value(
            table=output_path,
            field_name=rescol,
            field_value=float(ignored_value),
            overwrite=True
        )

    # update the output's attribute table with the aggegated data
    output_path = utils.update_attribute_table(
        output_path, aggregated, id_col,
        *[s.rescol for s in statistics]
    )

    # remove the temporary data
    if cleanup:
        utils.cleanup_temp_results(joined)

    return utils.load_attribute_table(output_path), res_columns


@utils.update_status()
def split_streams(stream_layer, subcatchment_layer):
    raise NotImplementedError


@utils.update_status()
def prepare_data(mon_locations, subcatchments, subcatch_id_col,
                 sort_id, header_fields, wq_fields, outputfile,
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
    subcatchments : str
        File path to the subcatchments feature class.
    subcatch_id_col : str
        Field name of the subcatchment ID column in ``subcatchments``.
    sort_id: str
        Arbituary field name for sorting purpose. Default value is 'FID'
    header_fields : list
        List of header field names to be retained in the output feature class.
    wq_fields : list
        List of water quality field names to be analyzed
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
        msg='Loading Subcatchments {}'.format(subcatchments),
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
    utils.intersect_layers(
        input_paths=[raw_subcatchments, raw_ml],
        output_path=_cat_ml_int,
        how="ALL",
    )

    # Step 2. Create a new point file (_reduced_ml) that pairs only one
    # set of ranking data to each catchment.

    # extract all subcatchment ID that has at least one
    # monitoring location
    arr = utils.load_attribute_table(_cat_ml_int, subcatch_id_col)
    catid = numpy.unique(arr[subcatch_id_col])

    for lc, ucat in enumerate(catid):
        sqlexp = '"{}" = \'{}\''.format(subcatch_id_col, ucat)
        _ml = utils.query_layer(_cat_ml_int, _ml, sqlexp)

        # if there is only one monitoring location  in the subcatchment,
        # this monitoring location is direclty copied to a new point
        # feature class.
        if lc == 0:
            fxn = arcpy.management.Copy
        # otherwise, aggregates monitoring locations to one single entry
        else:
            fxn = arcpy.management.Append

        # count number of MLs within each catchment
        fxn(_reduce(_ml, _out_ml, wq_fields, subcatch_id_col, sort_id), _reduced_ml)

    # Step 3. Spatial join _reduced_ml with raw_subcatchments, so that
    # the new cathcment shapefile will inherit water quality data from
    # the only monitoring location in it.
    subcatchment_wq = utils.spatial_join(
        left=raw_subcatchments,
        right=_reduced_ml,
        outputfile=subcatchment_wq
    )

    # Remove extraneous columns
    fields_to_remove = filter(
        lambda name: name not in header_fields and name not in wq_fields,
        [f.name for f in arcpy.ListFields(subcatchment_wq)]
    )
    utils.delete_columns(subcatchment_wq, *fields_to_remove)
    # Delete temporary files.
    utils.cleanup_temp_results(
        _reduced_ml,
        _cat_ml_int,
        _ml,
        _out_ml,
    )

    return subcatchment_wq


def _reduce(_ml, _out_ml, wq_fields, subcatch_id_col, sort_id):
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
    wq_fields : list
        Lisf of water quality parameters to be analyzed
    subcatch_id_col : str
        Field name of the subcatchment ID column in ``subcatchments``.

    Returns
    -------
    _out_ml : str

    """

    # Load water quality data
    _arr = utils.load_attribute_table(_ml, sort_id, *[f for f in wq_fields])

    # Copy and paste the monitoring location with the smallest FID.
    # This is essentially an arbiutary pick. What matters is we need
    # to only output one point back to the upstream function.
    _sqlexp = '"{}" = {}'.format(sort_id, numpy.min(_arr[sort_id]))
    arcpy.analysis.Select(_ml, _out_ml, _sqlexp)

    # Loop through each water quality parameter.
    # If at futre stage it is decided to compute the water quality
    # value with a different methods (i.e. median), either revise
    # the existing _non_zero_means function, or (preferrably) write
    # a new function (i.e. non_zero_median) and assign the new
    # function to the aggfxn varible.
    for wq_par in wq_fields:
        wq_value = utils.groupby_and_aggregate(
            input_path=_ml,
            groupfield=subcatch_id_col,
            valuefield=wq_par,
            aggfxn=_non_zero_means
        )

        # Overwrite field [wq_par] of _out_ml with the computed value.
        utils.populate_field(_out_ml, lambda row: wq_value.values()[0], wq_par)
    return _out_ml


def _non_zero_means(_arr):
    """
    Compute average value of all non-zero values in the
    list.

    Parameters
    ----------
    _arr : object or list
        Contains the list of interested values, which
        are extracted from the parental feature.

    Returns
    -------
    numpy.mean(numlst): float
        Zero-excluded average of the list. If the list
        contains no value, return a value of 0.

    See also
    --------
    utils.groupby_and_aggregate
    """

    if isinstance (_arr, list) or isinstance(_arr, numpy.ndarray):
        _numlst = filter(lambda n: n > 0, _arr)
    else:
        # This hanldes the case when the function is
        # an input for utils.groupby_and_aggregate.
        _numlst = filter(lambda n: n > 0, [r[1] for r in _arr])

    if numpy.isnan(numpy.mean(_numlst)):
        return 0
    else:
        return numpy.mean(_numlst)
