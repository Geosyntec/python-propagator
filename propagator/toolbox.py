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
try:
    from tqdm import tqdm
except ImportError: # pragma: no cover
    tqdm = lambda x: x

import propagator
from propagator import utils


def propagate(monitoring_locations=None, subcatchments=None,
              id_col=None, ds_col=None, output_path=None,
              value_columns=None):

    wq, result_columns = propagator.preprocess_wq(
        monitoring_locations=monitoring_locations,
        subcatchments=subcatchments,
        id_col=id_col,
        ds_col=ds_col,
        output_path=output_path,
        value_columns=value_columns,
    )

    for res_col in result_columns:
        wq = propagator.propagate_scores(
            subcatchment_array=wq,
            id_col=id_col,
            ds_col=ds_col,
            value_column=res_col,
        )

    utils.update_attribute_table(output_path, wq, id_col, *result_columns)

    return output_path


def accumulate(**params):
    workspace = params.pop('workspace', '.')
    subcatchments = params.pop('subcatchments', None)
    id_col = params.pop('ID_column', None)
    ds_col = params.pop('downstream_ID_column', None)
    streams = params.pop('streams', None)
    output_layer = params.pop('output_layer', None)
    water_quality_columns = params.pop('water_quality_columns', None)

    with utils.WorkSpace(workspace), utils.OverwriteState(True):
        subcatchment_array = propagator.prep_subcatchments(subcatchments)
        accumulated = propagator.accumulate_watershed_props(subcatchment_array, streams)

        propagator.update_attribute_table(output_layer, accumulated)

    return output_layer


class BaseToolbox_Mixin(object):

    def isLicensed(self):
        """ PART OF THE ESRI BLACK BOX.

        Esri says:

            Set whether tool is licensed to execute.


        So I just make this always true b/c it's an open source project
        with a BSD license -- (c) Geosyntec Consultants -- so who cares?

        """
        return True

    def updateMessages(self, parameters): # pragma: no cover
        """ PART OF THE ESRI BLACK BOX.

        Esri says:

            Modify the messages created by internal validation for each
            parameter of the tool.  This method is called after internal
            validation.


        But I have no idea when or how internal validation is called so
        that's pretty useless information.

        """
        return

    def updateParameters(self, parameters): # pragma: no cover
        """ PART OF THE ESRI BLACK BOX.

        Automatically called when any parameter is updated in the GUI.

        The general flow is like this:

          1. User interacts with GUI, filling out some input element
          2. ``self.getParameterInfo`` is called
          3. Parameteter are fed to this method as a list

        I used to set the parameter dependecies in here, but that didn't
        work. So now this does nothing and dependecies are set when the
        parameters (as class properties) are created (i.e., called for
        the first time).

        """
        return

    def getParameterInfo(self):
        """ PART OF THE ESRI BLACK BOX

        This *must* return a list of all of the parameter definitions.

        Esri recommends that you create all of the parameters in here,
        and always return that list. I instead chose to create the list
        from the class properties I've defined. Accessing things with
        meaningful names is always better, in my opinion.

        """
        return self._params_as_list()

    def execute(self, parameters, messages): # pragma: no cover
        """ PART OF THE ESRI BLACK BOX

        This method is called when the tool is actually executed. It
        gets passed magics lists of parameters and messages that no one
        can actually see.

        Due to this mysterious nature, I do the following:

        1) turn all of the elements of the list into a dictionary
           so that we can access them in a meaningful way. This
           means, instead of doing something like

        .. code-block:: python

           dem = parameters[0].valueAsText
           zones = parameters[1].valueAsText
           # yada yada
           nth_param = parameters[n].valueAsText

        for EVERY. SINGLE. PARAMETER, we can instead do something like:

        .. code-block:: python

           params = self._get_parameter_values(parameters, multivals=['elevation'])
           dem = params['dem']
           zones = params['zones'].
           # yada

        This is much cleaner, in my opinion, and we don't have to
        magically know where in the list of parameters e.g., the
        DEM is found. Take note, Esri.

        2) call :meth:`.analyze`.

        """

        params = self._get_parameter_values(parameters)
        self.analyze(**params)
        return None

    @staticmethod
    def _set_parameter_dependency(downstream, *upstream):
        """ Set the dependecy of a arcpy.Parameter

        Parameters
        ----------
        downstream : arcpy.Parameter
            The Parameter that is reliant on an upstream parameter.
        upstream : acrpy.Parameters
            An arbitraty number of "upstream" parameters on which the
            "downstream" parameter depends.

        Returns
        -------
        None

        See Also
        --------
        http://goo.gl/HcR6WJ

        """

        downstream.parameterDependencies = [u.name for u in upstream]

    @staticmethod
    def _show_header(title, verbose=True):
        """ Creates and shows a little header from a title.

        Parameters
        ----------
        title : str
            The message to be shown
        verbose : bool, optional (True)
            Whether or not the final message should be printed

        Returns
        -------
        header : str
            The formatted title as a header

        Examples
        --------
        >>> Flooder._show_header('Hello, world', verbose=True)
        'Hello, world'
         --------------

        """
        underline = ''.join(['-'] * len(title))
        header = '\n{}\n{}'.format(title, underline)
        utils.misc._status(header, verbose=verbose, asMessage=True, addTab=False)
        return header

    @staticmethod
    def _add_to_map(layerfile, mxd=None):
        """ Adds a layer or raster to the "CURRENT" map.

        Parameters
        ----------
        layerfile : str
            Path to the layer or raster that will be added
        mxd : str, optional
            Path to an ESRI mapdocument.

        Returns
        -------
        ezmd : EasyMapDoc
            The "easy map document" to which ``layerfile`` was added.

        """
        if mxd is None:
            mxd = 'CURRENT'
        ezmd = utils.EasyMapDoc(mxd)
        if ezmd.mapdoc is not None:
            ezmd.add_layer(layerfile)

        return ezmd

    @staticmethod
    def _get_parameter_values(parameters):
        """ Returns a dictionary of the parameters values as passed in from
        the ESRI black box. Keys are the parameter names, values are the
        actual values (as text) of the parameters.

        Parameters
        ----------
        parameters : list of arcpy.Parameter-type thingies
            The list of whatever-the-hell ESRI passes to the
            :meth:`.execute` method of a toolbox.
        multivals : str or list of str, optional
            Parameter names that can take mulitiple values.

        Returns
        -------
        params : dict
            A python dictionary of parameter values mapped to the
            parameter names.

        """

        params = {}
        for p in parameters:
            value = p.valueAsText
            if p.multiValue:
                value = value.split(';')
            params[p.name] = value

        return params

    @property
    def workspace(self):
        """ The directory or geodatabase in which the analysis will
        occur.

        """

        if self._workspace is None:
            self._workspace = arcpy.Parameter(
                displayName="Analysis WorkSpace",
                name='workspace',
                datatype="DEWorkspace",
                parameterType="Required",
                direction="Input",
                multiValue=False
            )
        return self._workspace

    @property
    def subcatchments(self):
        """ The subcatchments polygons to be used in the analysis.

        """

        if self._subcatchments is None:
            self._subcatchments = arcpy.Parameter(
                displayName="Subcatchments",
                name="subcatchments",
                datatype="DEFeatureClass",
                parameterType="Required",
                direction="Input",
                multiValue=False
            )
            self._set_parameter_dependency(self._subcatchments, self.workspace)
        return self._subcatchments

    @property
    def ID_column(self):
        """
        Name of the field in the `subcatchments` layer that uniquely
        identifies each subcatchment.

        """

        if self._ID_column is None:
            self._ID_column = arcpy.Parameter(
                displayName="Column with Subcatchment IDs",
                name="ID_column",
                datatype="Field",
                parameterType="Required",
                direction="Input",
                multiValue=False
            )
            self._set_parameter_dependency(self._ID_column, self.subcatchments)
        return self._ID_column

    @property
    def downstream_ID_column(self):
        """
        Name of the field in the `subcatchments` layer that specifies
        the downstream subcatchment.

        """

        if self._downstream_ID_column is None:
            self._downstream_ID_column = arcpy.Parameter(
                displayName="Column with the Downstream Subcatchment IDs",
                name="downstream_ID_column",
                datatype="Field",
                parameterType="Required",
                direction="Input",
                multiValue=False
            )
            self._set_parameter_dependency(self._downstream_ID_column, self.subcatchments)
        return self._downstream_ID_column

    @property
    def output_layer(self):
        """ Where the propagated/accumulated data will be saved.

        """

        if self._output_layer is None:
            self._output_layer = arcpy.Parameter(
                displayName="Output subcatchments layer/filename",
                name="output_layer",
                datatype="GPString",
                parameterType="Required",
                direction="Input"
            )
        return self._output_layer


class Propagator(BaseToolbox_Mixin):
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
        self.canRunInBackground = False
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

    @property
    def monitoring_locations(self):
        """
        The monitoring location points whose data will be propagated
        to the subcatchments.

        """

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
            self.output_layer,
        ]
        return params

    def analyze(self, **params):
        """ Propagates water quality scores from monitoring locations
        to upstream subcatchments. Calls directly to :func:`propagate`.

        """
        utils.misc._status(params, verbose=True, asMessage=True)
        ws = params.pop('workspace', '.')
        overwrite = params.pop('overwrite', True)
        sc = params.pop('subcatchments', None)
        ID_col = params.pop('ID_column', None)
        downstream_ID_col = params.pop('downstream_ID_column', None)
        ml = params.pop('monitoring_locations', None)
        value_cols = params.pop('value_columns', None)
        output_layer = params.pop('output_layer', None)
        add_to_map = params.pop('add_to_map', True)

        with utils.WorkSpace(ws), utils.OverwriteState(overwrite):
            output_layer = propagate(
                monitoring_locations=ml,
                subcatchments=sc,
                id_col=ID_col,
                ds_col=downstream_ID_col,
                output_path=output_layer,
                value_columns=value_cols,
            )

            if add_to_map:
                self._add_to_map(output_layer)

        return output_layer

class Accumulator(BaseToolbox_Mixin):
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
    direction = 'downstream'

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
        self._value_columns = None
        self._streams = None
        self._output_layer = None

    def _params_as_list(self):
        params = [
            self.workspace,
            self.subcatchments,
            self.ID_column,
            self.downstream_ID_column,
            self.value_columns,
            self.streams,
            self.output_layer,
        ]
        return params

    @property
    def streams(self):
        """ The streams who will accumulate attributes from
        subcatchments.

        """

        if self._streams is None:
            self._streams = arcpy.Parameter(
                displayName="Streams",
                name="streams",
                datatype="DEFeatureClass",
                parameterType="Required",
                direction="Input",
                multiValue=False
            )
            self._set_parameter_dependency(self._streams, self.workspace)
        return self._streams

    @property
    def value_columns(self):
        if self._value_columns is None:
            self._value_columns = arcpy.Parameter(
                displayName="Values to be Accumulated",
                name="value_columns",
                datatype="Field",
                parameterType="Required",
                direction="Input",
                multiValue=True
            )
            self._set_parameter_dependency(self._value_columns, self.subcatchments)
        return self._value_columns

    @staticmethod
    def analyze(self, **params):
        """ Accumulates subcatchments properties from upstream
        subcatchments into stream. Calls directly to :func:`accumulate`.

        """

        return accumulate(**params)
