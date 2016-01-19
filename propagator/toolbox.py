""" ArcGIS python toolboxes for ``propagator``.

This contains Classes compatible with ArcGIS python toolbox
infrastructure.

(c) Geosyntec Consultants, 2015.

Released under the BSD 3-clause license (see LICENSE file for more info)

Written by Paul Hobson (phobson@geosyntec.com)

"""


import os
from textwrap import dedent
from collections import OrderedDict

import arcpy

import numpy

from propagator import analysis
from propagator import validate
from propagator import utils
from propagator import base_tbx


def propagate(subcatchments=None, id_col=None, ds_col=None,
              monitoring_locations=None, ml_filter=None,
              ml_filter_cols=None, value_columns=None, streams=None,
              output_path=None, verbose=False, asMessage=False):
    """
    Propagate water quality scores upstream from the subcatchments of
    a watershed.

    Parameters
    ----------
    subcatchments : str
        Path to the feature class containing the subcatchments.
        Attribute table must contain fields for the subcatchment ID
        and the ID of the downstream subcatchment.
    id_col, ds_col : str
        Names of the fields in the ``subcatchments`` feature class that
        specifies the subcatchment ID and the ID of the downstream
        subcatchment, respectively.
    monitoring_locations : str
        Path to the feature class containing the monitoring locations
        and water quality scores.
    value_columns : list of str
        List of the fields in ``monitoring_locations`` that contains
        water quality score that should be propagated.
    ml_filter : callable, optional
        Function used to exclude (remove) monitoring locations from
        from aggregation/propagation.
    ml_filter_cols : str, optional
        Name of any additional columns in ``monitoring_locations`` that
        are required to use ``ml_filter``.
    streams : str
        Path to the feature class containing the streams.
    output_path : str
        Path to where the the new subcatchments feature class with the
        propagated water quality scores should be saved.

    Returns
    -------
    output_path : str

    Examples
    --------
    >>> import propagator
    >>> from propagator import utils
    >>> with utils.WorkSpace('C:/gis/SOC.gdb'):
    ...     propagator.propagate(
    ...         subcatchments='subbasins',
    ...         id_col='Catch_ID',
    ...         ds_col='DS_ID',
    ...         monitoring_locations='wq_data',
    ...         value_columns=['Dry_Metals', 'Wet_Metals', 'Wet_TSS'],
    ...         ml_filter=lambda row: row['StationType'] != 'Coastal',
    ...         ml_filter_cols=['StationType'],
    ...         streams='SOC_streams',
    ...         output_path='propagated_metals'
    ...     )

    See also
    --------
    propagator.analysis.preprocess_wq
    propagator.analysis.mark_edges
    propagator.analysis.propagate_scores
    propagator.analysis.aggregate_streams_by_subcatchment
    propagator.utils.update_attribute_table

    """

    subcatchment_output = utils.add_suffix_to_filename(output_path, 'subcatchments')
    stream_output = utils.add_suffix_to_filename(output_path, 'streams')

    wq, result_columns = analysis.preprocess_wq(
        monitoring_locations=monitoring_locations,
        ml_filter=ml_filter,
        ml_filter_cols=ml_filter_cols,
        subcatchments=subcatchments,
        value_columns=value_columns,
        id_col=id_col,
        ds_col=ds_col,
        output_path=subcatchment_output,
        verbose=verbose,
        asMessage=asMessage,
        msg="Aggregating water quality data in subcatchments"
    )

    wq = analysis.mark_edges(
        wq,
        id_col=id_col,
        ds_col=ds_col,
        edge_ID='EDGE',
        verbose=verbose,
        asMessage=asMessage,
        msg="Marking all subcatchments that flow out of the watershed"
    )

    for n, res_col in enumerate(result_columns, 1):
        wq = analysis.propagate_scores(
            subcatchment_array=wq,
            id_col=id_col,
            ds_col=ds_col,
            value_column=res_col,
            edge_ID='EDGE',
            verbose=verbose,
            asMessage=asMessage,
            msg="{} of {}: Propagating {} scores".format(n, len(result_columns), res_col)
        )

    utils.update_attribute_table(subcatchment_output, wq, id_col, result_columns)

    stream_output = analysis.aggregate_streams_by_subcatchment(
        stream_layer=streams,
        subcatchment_layer=subcatchment_output,
        id_col=id_col,
        ds_col=ds_col,
        other_cols=result_columns,
        agg_method='first',
        output_layer=stream_output,
        verbose=verbose,
        asMessage=asMessage,
        msg='Aggregating and associating scores with streams.',
    )

    return subcatchment_output, stream_output


def accumulate(subcatchments_layer=None, id_col=None, ds_col=None,
               area_col=None, imp_col=None, streams_layer=None,
               output_layer=None, verbose=False, asMessage=False):
    """
    Accumulate upstream subcatchment properties in each stream segment.

    Parameters
    ----------
    subcatchments_layer, streams_layer : str
        Names of the feature classes containing subcatchments and
        streams, respectively.
    id_col, ds_col : str
        Names of the fields in ``subcatchment_layer`` that contain the
        subcatchment ID and downstream subcatchment ID, respectively.
    sum_cols, avg_cols : list of str
        Names of the fields that will be accumulated by summing (e.g.,
        number of permit violations) and area-weighted averaging (e.g.,
        percent impervious land cover).

        .. note ::
           Do not include a column for subcatchment area in
           ``sum_cols``. Specify that in ``area_col`` instead.

    area_col : str, optional
        Name of a specific field of areas in the attribute table of
        ``subcatchments_layer``. Falls back to computing areas
        on-the-fly if not provided.
    output_layer : str, optional
        Names of the new layer where the results should be saved.

    Returns
    -------
    output_layer : str
        Names of the new layer where the results were successfully
        saved.

    See also
    --------
    propagator.analysis.aggregate_streams_by_subcatchment
    propagator.analysis.collect_upstream_attributes
    propagator.utils.rec_groupby

    """

    area_col = area_col or 'SHAPE@AREA'
    if imp_col is None:
        raise ValueError("imperviousness is required")

    stats = [
        utils.Statistic(area_col, numpy.sum, 'Sum_Area'),
        utils.Statistic(
            [imp_col, area_col],
            lambda x: utils.weighted_average(x, imp_col, area_col),
            'Wt_Avg_Imp'
        ),
    ]

    # create a unique list of columns we need
    # from the subcatchment layer
    target_fields = []
    for s in stats:
        if numpy.isscalar(s.srccol):
            target_fields.append(s.srccol)
        else:
            target_fields.extend(s.srccol)
    target_fields = numpy.unique(target_fields)

    # split the stream at the subcatchment boundaries and then
    # aggregate all of the stream w/i each subcatchment
    # into single geometries/records.
    split_streams_layer = analysis.aggregate_streams_by_subcatchment(
        stream_layer=streams_layer,
        subcatchment_layer=subcatchments_layer,
        id_col=id_col,
        ds_col=ds_col,
        other_cols=target_fields,
        output_layer=output_layer,
        agg_method="first",  # first works b/c all values are equal
    )

    # Add target_field columns back to spilt_stream_layer.
    for i in target_fields:
        arcpy.management.AddField(split_streams_layer, i, "DOUBLE")

    # load the split/aggregated streams' attribute table
    split_streams_table = utils.load_attribute_table(
        split_streams_layer, id_col, ds_col, *target_fields
    )

    # load the subcatchment attribute table
    subcatchments_table = utils.load_attribute_table(
        subcatchments_layer, id_col, ds_col, *target_fields
    )

    upstream_attributes = analysis.collect_upstream_attributes(
        subcatchments_table,
        split_streams_table,
        id_col,
        ds_col,
        target_fields
    )
    aggregated_properties = utils.rec_groupby(upstream_attributes, id_col, *stats)

    # Update output layer with aggregated values.
    final_fields = [stat.rescol for stat in stats]
    utils.update_attribute_table(
        split_streams_layer,
        aggregated_properties,
        id_col,
        target_fields,
        final_fields,
    )

    # Remove extraneous columns
    required_columns = [id_col, ds_col, 'FID', 'Shape', 'Shape_Length', 'Shape_Area', 'OBJECTID']
    fields_to_remove = filter(
        lambda name: name not in required_columns and name not in target_fields,
        [f.name for f in arcpy.ListFields(split_streams_layer)]
    )
    utils.delete_columns(split_streams_layer, *fields_to_remove)

    return split_streams_layer


class Propagator(base_tbx.BaseToolbox_Mixin):
    """
    ArcGIS Python toolbox to propagate water quality metrics upstream
    through subcatchments in a watershed.

    Parameters
    ----------
    None

    See also
    --------
    Accumulator

    """

    def __init__(self):
        """
        Define the tool (tool name is the name of the class).
        """

        # std attributes
        self.label = "1 - Propagate WQ scores to upstream subcatchments"
        self.description = dedent("""
        TDB
        """)

        # lazy properties
        self._workspace = None
        self._subcatchments = None
        self._ID_column = None
        self._downstream_ID_column = None
        self._monitoring_locations = None
        self._value_columns = None
        self._output_layer = None
        self._streams = None
        self._add_output_to_map = None

    @property
    def monitoring_locations(self):
        """ The monitoring location points whose data will be
        propagated to the subcatchments. """

        if self._monitoring_locations is None:
            self._monitoring_locations = arcpy.Parameter(
                displayName="Monitoring Locations",
                name="monitoring_locations",
                datatype="DEFeatureClass",
                parameterType="Required",
                direction="Input",
                multiValue=False
            )
            self._set_parameter_dependency(self._monitoring_locations, self.workspace)
        return self._monitoring_locations

    @property
    def value_columns(self):
        """ The names of the fields to be propagated into upstream
        subcatchments. """
        if self._value_columns is None:
            self._value_columns = arcpy.Parameter(
                displayName="Values to be Propagated",
                name="value_columns",
                datatype="Field",
                parameterType="Required",
                direction="Input",
                multiValue=True
            )
            self._set_parameter_dependency(self._value_columns, self.monitoring_locations)
        return self._value_columns

    def _params_as_list(self):
        params = [
            self.workspace,
            self.subcatchments,
            self.ID_column,
            self.downstream_ID_column,
            self.monitoring_locations,
            self.value_columns,
            self.streams,
            self.output_layer,
            self.add_output_to_map,
        ]
        return params

    def analyze(self, **params):
        """ Propagates water quality scores from monitoring locations
        to upstream subcatchments. Calls directly to :func:`propagate`.
        """

        # analysis options
        ws = params.pop('workspace', '.')
        overwrite = params.pop('overwrite', True)
        add_output_to_map = params.pop('add_output_to_map', False)

        # input parameters
        sc = params.pop('subcatchments', None)
        ID_col = params.pop('ID_column', None)
        downstream_ID_col = params.pop('downstream_ID_column', None)
        ml = params.pop('monitoring_locations', None)
        streams = params.pop('streams', None)
        value_cols = params.pop('value_columns', None)
        output_layer = params.pop('output_layer', None)

        # perform the analysis
        with utils.WorkSpace(ws), utils.OverwriteState(overwrite):
            output_layers = propagate(
                subcatchments=sc,
                id_col=ID_col,
                ds_col=downstream_ID_col,
                monitoring_locations=ml,
                value_columns=value_cols,
                output_path=output_layer,
                streams=streams,
                verbose=True,
                asMessage=True,
            )

            if add_output_to_map:
                for lyr in output_layers:
                    self._add_to_map(lyr)

        return output_layers


class Accumulator(base_tbx.BaseToolbox_Mixin):
    """
    ArcGIS Python toolbox to accumulate subcatchments attributes and
    water quality parameters downstream through a stream.

    Parameters
    ----------
    None

    See also
    --------
    Propagator

    """

    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        # std attributes
        self.label = "2 - Accumulate subcatchment properties to stream"
        self.description = dedent("""
        TDB
        """)

        # lazy properties
        self._workspace = None
        self._subcatchments = None
        self._ID_column = None
        self._downstream_ID_column = None
        self._area_col = None
        self._imp_col = None
        self._streams = None
        self._output_layer = None
        self._add_output_to_map = None

    def _params_as_list(self):
        params = [
            self.workspace,
            self.subcatchments,
            self.ID_column,
            self.downstream_ID_column,
            self.area_col,
            self.imp_col,
            self.streams,
            self.output_layer,
            self.add_output_to_map,
        ]
        return params

    @property
    def area_col(self):
        """ Name of the field in the `subcatchments` layer specifies
        the catchment area. Optional as this can be computed on-the-fly.
        """

        if self._area_col is None:
            self._area_col = arcpy.Parameter(
                displayName="Column with Subcatchment Areas",
                name="area_col",
                datatype="Field",
                parameterType="Optional",
                direction="Input",
                multiValue=False
            )
            self._set_parameter_dependency(self._area_col, self.subcatchments)
        return self._area_col

    @property
    def imp_col(self):
        """ Name of the field in the `subcatchments` layer specifies
        the percent impervious cover. """

        if self._imp_col is None:
            self._imp_col = arcpy.Parameter(
                displayName="Column with Subcatchment Percents impervious",
                name="imp_col",
                datatype="Field",
                parameterType="Required",
                direction="Input",
                multiValue=False
            )
            self._set_parameter_dependency(self._imp_col, self.subcatchments)
        return self._imp_col

    def analyze(self, **params):
        """ Accumulates subcatchments properties from upstream
        subcatchments into stream. Calls directly to :func:`accumulate`.

        """

        # analysis options
        ws = params.pop('workspace', '.')
        overwrite = params.pop('overwrite', True)
        add_output_to_map = params.pop('add_output_to_map', False)

        # input parameters
        sc = params.pop('subcatchments', None)
        ID_col = params.pop('ID_column', None)
        downstream_ID_col = params.pop('downstream_ID_column', None)
        streams = params.pop('streams', None)
        area_col = params.pop('area_col', 'SHAPE@AREA')
        imp_col = params.pop('imp_col', None)
        output_layer = params.pop('output_layer', None)

        with utils.WorkSpace(ws), utils.OverwriteState(overwrite):
            output_layers = accumulate(
                subcatchments_layer=sc,
                id_col=ID_col,
                ds_col=downstream_ID_col,
                area_col=area_col,
                imp_col=imp_col,
                streams_layer=streams,
                output_layer=output_layer,
                verbose=True,
                asMessage=True,
            )

            if add_output_to_map:
                self._add_to_map(output_layers)
                #for lyr in output_layers:

        return output_layers
