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
import warnings
from copy import copy

import numpy
from numpy.lib import recfunctions

import arcpy

from . import utils
from . import validate


AGG_METHOD_DIC = {
    'average': numpy.mean,
    'ave': numpy.mean,
    'avg': numpy.mean,
    'mean': numpy.mean,
    'median': numpy.median,
    'med': numpy.median,
    'maximum': numpy.max,
    'max': numpy.max,
    'minimum': numpy.min,
    'min': numpy.min,
    '10th': partial(numpy.percentile, q=10),
    '10%': partial(numpy.percentile, q=10),
    '25th': partial(numpy.percentile, q=25),
    '25%': partial(numpy.percentile, q=25),
    '50th': partial(numpy.percentile, q=50),
    '50%': partial(numpy.percentile, q=50),
    '75th': partial(numpy.percentile, q=75),
    '75%': partial(numpy.percentile, q=75),
    '90th': partial(numpy.percentile, q=90),
    '90%': partial(numpy.percentile, q=90),
}


@utils.update_status()
def trace_upstream(subcatchment_array, subcatchment_ID, id_col='ID',
                   ds_col='DS_ID', include_base=False, downstream=None):
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
    include_base : bool, optional
        Toggles the inclusion of target subcatchment itself in upstream
        subcatchment list.

        .. note ::
           This should *always* be False in the *recursive* calls the
           function.

    downstream : list, optional
        A list of already known downstream catchments in the trace.

        .. warning ::
           This is *only* to be used in the recursive calls to this
           function. You should never provide this value.

    Returns
    -------
    upstream : numpy.recarry
        A record array of all of the upstream subcatchments. This will
        have the same schema as ``subcatchment_array``

    """
    if downstream is None:
        downstream = []


    # If needed, add the bottom subcatchment to the output list
    if include_base:
        base_row = utils.find_row_in_array(subcatchment_array, id_col, subcatchment_ID)
        downstream.append(base_row)

    _neighbors = filter(lambda row: row[ds_col] == subcatchment_ID, subcatchment_array)

    for n in _neighbors:
        downstream.append(n)
        trace_upstream(subcatchment_array, n[id_col],
                       id_col=id_col, ds_col=ds_col,
                       include_base=False,
                       downstream=downstream)

    return numpy.array(downstream, dtype=subcatchment_array.dtype)


@utils.update_status()
def find_edges(subcatchment_array, edge_ID='bottom', ds_col='DS_ID'):
    """
    Finds the lowest, non-ocean subcatchments in a watershed.

    Parameters
    ----------
    subcatchment_array : numpy.recarry
        A record array of all of the subcatchments in the watershed.
        This array must have a "downstrea ID" column in which each
        subcatchment identifies as single, downstream neighbor.
    edge_ID : str, optional
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

    bottoms = filter(lambda row: row[ds_col] == edge_ID, subcatchment_array)
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
                     ignored_value=0, edge_ID='bottom'):
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
    edge_ID : str, optional
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
        is_bottom = row[ds_col].lower() == edge_ID.lower()

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
                edge_ID=edge_ID,
            )

            # assign the downstream value to the current (empty) value
            propagated[value_column][n] = ds_values[value_column]

    return propagated


@utils.update_status()
def _find_downstream_scores(subcatchment_array, subcatchment_ID, value_column,
                            ignored_value='None', id_col='ID', ds_col='DS_ID',
                            edge_ID='bottom'):
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
    is_bottom = row[ds_col].lower() == edge_ID.lower()
    if row[value_column] == ignored_value and not is_bottom:
        scores = _find_downstream_scores(
            subcatchment_array=subcatchment_array,
            subcatchment_ID=row[ds_col],
            value_column=value_column,
            ignored_value=ignored_value,
            id_col=id_col,
            ds_col=ds_col,
            edge_ID=edge_ID,
        )
    else:
        scores = row.copy()

    return scores


@utils.update_status()
def mark_edges(subcatchment_array, id_col='ID', ds_col='DS_ID',
               edge_ID='EDGE'):
    """
    Mark of all of the subcatchments on the edges of the study area
    (i.e., flow out of the study area). In this case "mark" means that
    the downstream subcatchment ID is set to a constant value.

    Parameters
    ----------
    subcatchment_array : numpy.recarray
        Record array of subcatchments with at least ``id_col`` and
        ``ds_col`` columns.
    id_col, ds_col : str, optional
        Label of the subcatchment ID and downstream subcatchment ID
        columns, respectively
    edge_ID : str, optional
        The downstream subcatchment ID that will given to the
        subcatchments that flow out of the study area.

    Returns
    -------
    array : numpy.recarray
        An array with the same schema as ``subcatchment_array``, but
        without the orphans.

    """

    subc = subcatchment_array.copy()
    for n, row in enumerate(subcatchment_array):
        if row[ds_col] not in subc[id_col]:
            subc[n][ds_col] = edge_ID

    return subc


@numpy.deprecate
@utils.update_status()
def remove_orphan_subcatchments(subcatchment_array, id_col='ID', ds_col='DS_ID',
                                bottom_ID='Ocean'):
    """
    Recursively removes subcatchments that flow laterally out of the
    project area. Basically, an orpan subcatchment has a downstream
    subcatchment that does not exist in the project area.

    Parameters
    ----------
    subcatchment_array : numpy.recarray
        Record array of subcatchments with at least ``id_col`` and
        ``ds_col`` columns.
    id_col, ds_col : str, optional
        Label of the subcatchment ID and downstream subcatchment ID
        columns, respectively
    bottom_ID : str, optional
        The downstream subcatchment ID given to the bottom-most
        subcatchments.

    Returns
    -------
    array : numpy.recarray
        An array with the same schema as ``subcatchment_array``, but
        without the orphans.

    """

    def keep_it(x):
        ds_exists = x[ds_col] in subcatchment_array[id_col]
        is_bottom = x[ds_col].lower() == bottom_ID.lower()
        return ds_exists or is_bottom

    _subc = [x for x in subcatchment_array if keep_it(x)]
    subc = numpy.rec.fromrecords(_subc, dtype=subcatchment_array.dtype)
    if subc.shape[0] != subcatchment_array.shape[0]:
        subc = remove_orphan_subcatchments(subc, id_col=id_col, ds_col=ds_col, bottom_ID=bottom_ID)
    return subc


@utils.update_status()
def preprocess_wq(monitoring_locations, subcatchments, id_col, ds_col,
                  output_path, value_columns=None, ml_filter=None,
                  ml_filter_cols=None, aggfxn=numpy.mean, ignored_value=0,
                  terminator_value=-99, cleanup=True):
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
    id_col, ds_col : str
        Name of the column in ``subcatchments`` that contains the
        (ds = downstream) subcatchment IDs.
    ml_filter_cols : str, optional
        Name of any additional columns in ``monitoring_locations`` that
        are required to use ``ml_filter``.
    output_path : str
        Path of the new feature class where the preprocessed data
        should be saved.
    value_columns : list of str
        A list of the names of the fields containing water quality
        scores that need to be analyzed.
    ml_filter : callable, optional
        Function used to exclude (remove) monitoring locations from
        from aggregation/propagation.
    aggfxn : callable, optional
        A function, lambda, or class method that reduces arrays into
        scalar values. By default, this is `numpy.mean`.
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

    ml_filter_cols = validate.non_empty_list(ml_filter_cols, on_fail='create')
    if ml_filter is None:
        ml_filter = lambda row: row

    # validate value_columns
    value_columns = validate.non_empty_list(value_columns, msg="you must provide `value_columns` to aggregate")

    # Seperate aggregation method from value_columns input
    split_value_columns = [i.split(";") for i in value_columns]
    split_value_columns = [i.split() for i in split_value_columns[0]]
    value_columns_field = [i[0] for i in split_value_columns]
    value_columns_aggmethod = [i[1] for i in split_value_columns]
    # create the output feature class as a copy of the `subcatchments`
    output_path = utils.copy_layer(subcatchments, output_path)

    # associate subcatchment IDs with all of the monitoring locations
    joined = utils.intersect_layers(
        input_paths=[monitoring_locations, subcatchments],
        output_path=utils.create_temp_filename("joined_ml_sc", filetype='shape'),
        how="ALL",
    )

    # define the Statistic objects that will be passed to `rec_groupby`
    statfxn_columns = []
    for i in value_columns_aggmethod:
        statfxn_columns.append(partial(
            utils.stats_with_ignored_values,
            statfxn=AGG_METHOD_DIC[i.lower()],
            ignored_value=ignored_value
        )
        )
    res_columns = ['{}{}'.format(prefix[0:3].lower(), col) for prefix, col in zip(value_columns_aggmethod, value_columns_field)]

    statistics = [
        utils.Statistic(srccol, statfxn, rescol)
        for srccol, statfxn, rescol in zip(value_columns_field, statfxn_columns, res_columns)
    ]

    # compile the original fields to read in from the joined table
    orig_fields = [id_col, ds_col]
    orig_fields.extend([stat.srccol for stat in statistics])
    orig_fields.extend(ml_filter_cols)

    raw_array = utils.load_attribute_table(joined, *orig_fields)
    # factor this into load_attribute_table
    array = numpy.array(filter(ml_filter, raw_array), dtype=raw_array.dtype)

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

    # update the output's attribute table with the aggregated data
    output_path = utils.update_attribute_table(
        layerpath=output_path,
        attribute_array=aggregated,
        id_column=id_col,
        orig_columns=[s.rescol for s in statistics]
    )

    # remove the temporary data
    if cleanup:
        utils.cleanup_temp_results(joined)

    return utils.load_attribute_table(output_path), res_columns


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

    if isinstance(_arr, list) or isinstance(_arr, numpy.ndarray):
        _numlst = filter(lambda n: n > 0, _arr)
    else:
        # This hanldes the case when the function is
        # an input for utils.groupby_and_aggregate.
        _numlst = filter(lambda n: n > 0, [r[1] for r in _arr])

    if len(_numlst) == 0:
        return 0
    else:
        return numpy.mean(_numlst)


@utils.update_status()
def aggregate_streams_by_subcatchment(stream_layer, subcatchment_layer,
                                      id_col, ds_col, other_cols,
                                      agg_method="first",
                                      output_layer=None,
                                      cleanup=True):
    """
    Split up stream into segments based on subcatchment borders, and
    then aggregates all of the individual segments within each
    subcatchments into a single multi-part geometry

    Parameters
    ----------
    stream_layer, subcatchment_layer : str
        Name of the feature class containing streams and subcatchments,
        respectively.
    id_col, ds_col : str
        Names of the fields in ``subcatchment_layer`` that contain the
        subcatchment ID and downstream subcatchment ID, respectively.
    other_cols : list of str
        Other, non-grouping columns to keep in ``output_layer``.
    agg_method : str, optional
        Method by which `other_cols` will be aggregated. The default
        value is 'FIRST'.

        .. note ::

           The methods available are limited to those supported by
           `arcpy.management.Dissolve`. Those are: "FIRST", "LAST",
           "SUM", "MEAN", "MIN", "MAX", "RANGE", "STD", and "COUNT".

    output_layer : str, optional
        Names of the new layer where the results should be saved.
    cleanup : bool, optional
        Toggles the deletion of intermediate files.

    Returns
    -------
    output_layer : str
        Names of the new layer where the results were successfully
        saved.

    Examples
    --------
    >>> import propagator
    >>> from propagator import utils
    >>> with utils.WorkSpace('C:/SOC/data.gdb'):
    ...     propagator.aggregate_streams_by_subcatchment(
    ...         stream_layer='streams_with_WQ_scores',
    ...         subcatchment_layer='SOC_subbasins',
    ...         id_col='Catch_ID',
    ...         ds_col='DS_Catch_ID',
    ...         other_cols=['Dry_Metals', 'Wet_Metals'],
    ...         agg_method='MAX',
    ...         output_layer='agg_streams'
    ...     )

    See also
    --------
    propagator.utils.intersect_layers
    propagator.utils.aggregate_geoms

    """

    utils.check_fields(subcatchment_layer, id_col, ds_col, *other_cols, should_exist=True)

    intersected = utils.intersect_layers(
        input_paths=[stream_layer, subcatchment_layer],
        output_path=utils.create_temp_filename(output_layer, filetype='shape'),
        how="NO_FID",
    )

    stats_tuples = [(col, agg_method) for col in other_cols]

    final = utils.aggregate_geom(
        layerpath=intersected,
        by_fields=[id_col, ds_col],
        field_stat_tuples=stats_tuples,
        outputpath=output_layer,
        multi_part="MULTI_PART",
        unsplit_lines="DISSOLVE_LINES",
    )

    if cleanup:
        utils.cleanup_temp_results(intersected)

    return final


def collect_upstream_attributes(subcatchments_table, target_subcatchments,
                                id_col, ds_col, preserved_fields):
    """
    Identifies all upstream subcatchment IDs of each target
    subcatchment.

    Parameters
    ----------
    subcatchments_table : numpy.ndarray
        List of all subcatchments
    target_subcatchments : numpy.ndarray
        List of subcatchments whose upstream contributing subcatchments
        will be identified.
    id_col, ds_col : str
        Names of the columns in ``subcatchment_table`` that contain the
        subcatchment ID and downstream subcatchment ID, respectively.
    preserved_fields : list
        List of column IDs that will be kept in output table.

    Returns
    -------
    output : numpy.ndarray
        An array listing all upstream subcatchments for each target
        subcatchment.

    """

    final_cols = list(preserved_fields)
    final_cols.append(id_col)
    template = subcatchments_table[final_cols].dtype

    n = -1
    for row in target_subcatchments:
        n = n+1
        upstream_subcatchments = trace_upstream(
            subcatchments_table, row[id_col],
            id_col=id_col, ds_col=ds_col, include_base=True
        )

        if upstream_subcatchments.shape[0] > 0:
            # factor out the next 5 SLOC
            # add the ID of the "bottom" subcatchment as a column to
            # the array of upstream attributes
            id_array = numpy.array([row[id_col]] * upstream_subcatchments.shape[0])

            # recfunctions.append_fields is not compatible with unicode input; hence
            # all inputs are converted to strings
            upstream_subcatchments = upstream_subcatchments[preserved_fields]
            dname = numpy.array(upstream_subcatchments.dtype.names)
            upstream_subcatchments.dtype.names = [i.encode('ascii', 'ignore') for i in dname]

            # recfunctions.append_fields has a nasty warnings that we don't need to see
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("ignore")
                id_col_str = id_col.encode('ascii','ignore')
                _src_array = recfunctions.append_fields(upstream_subcatchments, [id_col_str], [id_array])
            if n == 0:
                src_array = _src_array.copy().tolist()
            else:
                src_array.extend(_src_array.copy().tolist())
                #src_array = numpy.hstack([src_array, _src_array])
        else:
            n = n-1

    return numpy.array(src_array, dtype=template)
