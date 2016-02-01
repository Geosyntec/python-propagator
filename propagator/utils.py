""" Wrappers around Esri's ``arcpy`` library.

This contains basic file I/O, conversion, and spatial analysis functions
to support the python-propagator library. These functions generally
are simply wrappers around their ``arcpy`` counter parts. This was done
so that in the future, these functions could be replaced with calls to
a different geoprocessing library and eventually ween the code base off
of its ``arcpy`` dependency.

(c) Geosyntec Consultants, 2015.

Released under the BSD 3-clause license (see LICENSE file for more info)

Written by Paul Hobson (phobson@geosyntec.com)

"""


import os
import itertools
from functools import wraps
from contextlib import contextmanager
from collections import namedtuple
from copy import copy
import warnings

import numpy

import arcpy

from propagator import validate
import pdb


# basic named tuple for recarray aggregation
Statistic = namedtuple("Statistic", ("srccol", "aggfxn", "rescol"))


def _status(msg, verbose=False, asMessage=False, addTab=False):  # pragma: no cover
    if verbose:
        if addTab:
            msg = '\t' + msg
        if asMessage:
            arcpy.AddMessage(msg)
        else:
            print(msg)


def update_status():  # pragma: no cover
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


def add_suffix_to_filename(filename, suffix):
    """
    Adds a suffix to a(n output) filename.

    Parameters
    ----------
    filename : str or a list of str
        The filename that needs a suffix.
    suffix : str
        The suffix to be added.

    Returns
    -------
    new_filename : str
        The original filename with followed by an underscore, the
        suffix, and then any extension that the original filename had.

    Examples
    --------
    >>> from propagator import utils
    >>> utils.add_suffix_to_filename('test_shapefile.shp', 'try_2')
    'test_shapefile_try_2.shp'

    >>> utils.add_suffix_to_filename('test_layer', 'try_2')
    'test_layer_try_2'

    >>> utils.add_suffix_to_filename(['streams.shp', 'locations.shp'])

    """
    name, extension = os.path.splitext(filename)
    return '{}_{}{}'.format(name, suffix, extension)


class RasterTemplate(object):
    """ Georeferencing template for Rasters.

    This mimics the attributes of the ``arcpy.Raster`` class enough
    that it can be used as a template to georeference numpy arrays
    when converting to rasters.

    Parameters
    ----------
    cellsize : int or float
        The width of the raster's cells.
    xmin, ymin : float
        The x- and y-coordinates of the raster's lower left (south west)
        corner.

    Attributes
    ----------
    cellsize : int or float
        The width of the raster's cells.
    extent : Extent
        Yet another mock-ish class that ``x`` and ``y`` are stored in
        ``extent.lowerLeft`` as an ``arcpy.Point``.

    See also
    --------
    arcpy.Extent

    """

    def __init__(self, cellsize, xmin, ymin):
        self.meanCellWidth = cellsize
        self.meanCellHeight = cellsize
        self.extent = arcpy.Extent(xmin, ymin, numpy.nan, numpy.nan)

    @classmethod
    def from_raster(cls, raster):
        """ Alternative constructor to generate a RasterTemplate from
        an actual raster.

        Parameters
        ----------
        raster : arcpy.Raster
            The raster whose georeferencing attributes need to be
            replicated.

        Returns
        -------
        template : RasterTemplate

        """
        template = cls(
            raster.meanCellHeight,
            raster.extent.lowerLeft.X,
            raster.extent.lowerLeft.Y,
        )
        return template


class EasyMapDoc(object):
    """ The object-oriented map class Esri should have made.

    Create this the same you would make any other
    `arcpy.mapping.MapDocument`_. But now, you can directly list and
    add layers and dataframes. See the two examples below.

    Has ``layers`` and ``dataframes`` attributes that return all of the
    `arcpy.mapping.Layer`_ and `arcpy.mapping.DataFrame`_ objects in the
    map, respectively.

    .. _arcpy.mapping.MapDocument: http://goo.gl/rf4GBH
    .. _arcpy.mapping.DataFrame: http://goo.gl/ctJu3B
    .. _arcpy.mapping.Layer: http://goo.gl/KfrGNa

    Attributes
    ----------
    mapdoc : arcpy.mapping.MapDocument
        The underlying arcpy MapDocument that serves as the basis for
        this class.

    Examples
    --------
    >>> # Adding a layer with the Esri version:
    >>> import arpcy
    >>> md = arcpy.mapping.MapDocument('CURRENT')
    >>> df = arcpy.mapping.ListDataFrames(md)
    >>> arcpy.mapping.AddLayer(df, myLayer, 'TOP')

    >>> # And now with an ``EasyMapDoc``:
    >>> from propagator import utils
    >>> ezmd = utils.EasyMapDoc('CURRENT')
    >>> ezmd.add_layer(myLayer)

    """

    def __init__(self, *args, **kwargs):
        try:
            self.mapdoc = arcpy.mapping.MapDocument(*args, **kwargs)
        except RuntimeError:
            self.mapdoc = None

    @property
    def layers(self):
        """
        All of the layers in the map.
        """
        return arcpy.mapping.ListLayers(self.mapdoc)

    @property
    def dataframes(self):
        """
        All of the dataframes in the map.
        """
        return arcpy.mapping.ListDataFrames(self.mapdoc)

    def findLayerByName(self, name):
        """ Finds a `layer`_ in the map by searching for an exact match
        of its name.

        .. _layer: http://goo.gl/KfrGNa

        Parameters
        ----------
        name : str
            The name of the layer you want to find.

        Returns
        -------
        lyr : arcpy.mapping.Layer
            The map layer or None if no match is found.

        .. warning:: Group Layers are not returned.

        Examples
        --------
        >>> from propagator import utils
        >>> ezmd = utils.EasyMapDoc('CURRENT')
        >>> wetlands = ezmd.findLayerByName("wetlands")
        >>> if wetlands is not None:
        ...     # do something with `wetlands`

        """

        for lyr in self.layers:
            if not lyr.isGroupLayer and lyr.name == name:
                return lyr

    def add_layer(self, layer, df=None, position='top'):
        """ Simply adds a `layer`_ to a map.

        .. _layer: http://goo.gl/KfrGNa

        Parameters
        ----------
        layer : str or arcpy.mapping.Layer
            The dataset to be added to the map.
        df : arcpy.mapping.DataFrame, optional
            The specific dataframe to which the layer will be added. If
            not provided, the data will be added to the first dataframe
            in the map.
        position : str, optional ('TOP')
            The positional within `df` where the data will be added.
            Valid options are: 'auto_arrange', 'bottom', and 'top'.

        Returns
        -------
        layer : arcpy.mapping.Layer
            The successfully added layer.

        Examples
        --------
        >>> from propagator import utils
        >>> ezmd = utils.EasyMapDoc('CURRENT')
        >>> watersheds = utils.load_data("C:/gis/hydro.gdb/watersheds")
        >>> ezmd.add_layer(watersheds)

        """

        # if no dataframe is provided, select the first
        if df is None:
            df = self.dataframes[0]

        # check that the position is valid
        valid_positions = ['auto_arrange', 'bottom', 'top']
        if position.lower() not in valid_positions:
            raise ValueError('Position: %s is not in %s' % (position.lower, valid_positions))

        # layer can be a path to a file. if so, convert to a Layer object
        layer = load_data(layer, 'layer')

        # add the layer to the map
        arcpy.mapping.AddLayer(df, layer, position.upper())

        # return the layer
        return layer


@contextmanager
def Extension(name):
    """ Context manager to facilitate the use of ArcGIS extensions

    Inside the context manager, the extension will be checked out. Once
    the interpreter leaves the code block by any means (e.g., successful
    execution, raised exception) the extension will be checked back in.

    Examples
    --------
    >>> import propagator, arcpy
    >>> with propagator.utils.Extension("spatial"):
    ...     arcpy.sa.Hillshade("C:/data/dem.tif")

    """

    if arcpy.CheckExtension(name) == u"Available":
        status = arcpy.CheckOutExtension(name)
        yield status
    else:
        raise RuntimeError("%s license isn't available" % name)

    arcpy.CheckInExtension(name)


@contextmanager
def OverwriteState(state):
    """ Context manager to temporarily set the ``overwriteOutput``
    environment variable.

    Inside the context manager, the ``arcpy.env.overwriteOutput`` will
    be set to the given value. Once the interpreter leaves the code
    block by any means (e.g., successful execution, raised exception),
    ``arcpy.env.overwriteOutput`` will reset to its original value.

    Parameters
    ----------
    path : str
        Path to the directory that will be set as the current workspace.

    Examples
    --------
    >>> from propagator import utils
    >>> with utils.OverwriteState(False):
    ...     # some operation that should fail if output already exists

    """

    orig_state = arcpy.env.overwriteOutput
    arcpy.env.overwriteOutput = bool(state)
    yield state
    arcpy.env.overwriteOutput = orig_state


@contextmanager
def WorkSpace(path):
    """ Context manager to temporarily set the ``workspace``
    environment variable.

    Inside the context manager, the `arcpy.env.workspace`_ will
    be set to the given value. Once the interpreter leaves the code
    block by any means (e.g., successful execution, raised exception),
    `arcpy.env.workspace`_ will reset to its original value.

    .. _arcpy.env.workspace: http://goo.gl/0NpeFN

    Parameters
    ----------
    path : str
        Path to the directory that will be set as the current workspace.

    Examples
    --------
    >>> import propagator
    >>> with propagator.utils.OverwriteState(False):
    ...     # some operation that should fail if output already exists

    """

    orig_workspace = arcpy.env.workspace
    arcpy.env.workspace = path
    yield path
    arcpy.env.workspace = orig_workspace


def create_temp_filename(filepath, filetype=None, prefix='_temp_', num=None):
    """ Helper function to create temporary filenames before to be saved
    before the final output has been generated.

    Parameters
    ----------
    filepath : str
        The file path/name of what the final output will eventually be.
    filetype : str, optional
        The type of file to be created. Valid values: "Raster" or
        "Shape".
    prefix : str, optional ('_temp_')
        The prefix that will be applied to ``filepath``.
    num : int, optional
        A file "number" that can be appended to the very end of the
        filename.

    Returns
    -------
    str : temp_filename

    Examples
    --------
    >>> from propagator import utils
    >>> utils.create_temp_filename('path/to/wetlands.shp', filetype='shape')
    path/to/_temp_wetlands.shp

    >>> utils.create_temp_filename('path.gdb/wetlands', filetype='shape')
    path.gbd/_temp_wetlands

    >>> utils.create_temp_filename('path/to/DEM.tif', filetype='raster')
    path/to/_temp_DEM.shp

    >>> utils.create_temp_filename('path.gdb/DEM', filetype='raster')
    path.gbd/_temp_DEM

    """

    file_extensions = {
        'raster': '.tif',
        'shape': '.shp'
    }

    if num is None:
        num = ''
    else:
        num = '_{}'.format(num)

    ws = arcpy.env.workspace or '.'
    filename, _ = os.path.splitext(os.path.basename(filepath))
    folder = os.path.dirname(filepath)
    if folder != '':
        final_workspace = os.path.join(ws, folder)
    else:
        final_workspace = ws

    if os.path.splitext(final_workspace)[1] == '.gdb':
        ext = ''
    else:
        ext = file_extensions[filetype.lower()]


    return os.path.join(ws, folder, prefix + filename + num + ext)


def check_fields(table, *fieldnames, **kwargs):
    """
    Checks that field are (or are not) in a table. The check fails, a
    ``ValueError`` is raised.

    Parameters
    ----------
    table : arcpy.mapping.Layer or similar
        Any table-like that we can pass to `arcpy.ListFields`.
    *fieldnames : str arguments
        optional string arguments that whose existence in `table` will
        be checked.
    should_exist : bool, optional (False)
        Whether we're testing for for absence (False) or existence
        (True) of the provided field names.

    Returns
    -------
    None

    """

    should_exist = kwargs.pop('should_exist', False)

    existing_fields = get_field_names(table)
    bad_names = []
    for name in fieldnames:
        exists = name in existing_fields
        if should_exist != exists and name != 'SHAPE@AREA':
            bad_names.append(name)

    if not should_exist:
        qual = 'already'
    else:
        qual = 'not'

    if len(bad_names) > 0:
        raise ValueError('fields {} are {} in {}'.format(bad_names, qual, table))


def result_to_raster(result):
    """ Gets the actual `arcpy.Raster`_ from an `arcpy.Result`_ object.

    .. _arcpy.Raster: http://goo.gl/AQgFXW
    .. _arcpy.Result: http://goo.gl/xPIbHi

    Parameters
    ----------
    result : arcpy.Result
        The `Result` object returned from some other geoprocessing
        function.

    Returns
    -------
    arcpy.Raster

    See also
    --------
    result_to_layer

    """
    return arcpy.Raster(result.getOutput(0))


def result_to_layer(result):
    """ Gets the actual `arcpy.mapping.Layer`_ from an `arcpy.Result`_
    object.

    .. _arcpy.mapping.Layer: http://goo.gl/KfrGNa
    .. _arcpy.Result: http://goo.gl/xPIbHi

    Parameters
    ----------
    result : arcpy.Result
        The `Result` object returned from some other geoprocessing
        function.

    Returns
    -------
    arcpy.mapping.Layer

    See also
    --------
    result_to_raster

    """

    return arcpy.mapping.Layer(result.getOutput(0))


def load_data(datapath, datatype, greedyRasters=True, **verbosity):
    """ Loads vector and raster data from filepaths.

    Parameters
    ----------
    datapath : str, arcpy.Raster, or arcpy.mapping.Layer
        The (filepath to the) data you want to load.
    datatype : str
        The type of data you are trying to load. Must be either
        "shape" (for polygons) or "raster" (for rasters).
    greedyRasters : bool (default = True)
        Currently, arcpy lets you load raster data as a "Raster" or as a
        "Layer". When ``greedyRasters`` is True, rasters loaded as type
        "Layer" will be forced to type "Raster".

    Returns
    -------
    data : `arcpy.Raster`_ or `arcpy.mapping.Layer`_
        The data loaded as an arcpy object.

    .. _arcpy.Raster: http://goo.gl/AQgFXW
    .. _arcpy.mapping.Layer: http://goo.gl/KfrGNa

    """

    dtype_lookup = {
        'raster': arcpy.Raster,
        'grid': arcpy.Raster,
        'shape': arcpy.mapping.Layer,
        'layer': arcpy.mapping.Layer,
    }

    try:
        objtype = dtype_lookup[datatype.lower()]
    except KeyError:
        msg = "Datatype {} not supported. Must be raster or layer".format(datatype)
        raise ValueError(msg)

    # if the input is already a Raster or Layer, just return it
    if isinstance(datapath, objtype):
        data = datapath
    # otherwise, load it as the datatype
    else:
        try:
            data = objtype(datapath)
        except:
            raise ValueError("could not load {} as a {}".format(datapath, objtype))

    if greedyRasters and isinstance(data, arcpy.mapping.Layer) and data.isRasterLayer:
        data = arcpy.Raster(datapath)

    return data


def add_field_with_value(table, field_name, field_value=None,
                         overwrite=False, **field_opts):
    """ Adds a numeric or text field to an attribute table and sets it
    to a constant value. Operates in-place and therefore does not
    return anything.

    Relies on `arcpy.management.AddField`_.

    .. _arcpy.management.AddField: http://goo.gl/wivgDX

    Parameters
    ----------
    table : Layer, table, or file path
        This is the layer/file that will have a new field created.
    field_name : string
        The name of the field to be created.
    field_value : float or string, optional
        The value of the new field. If provided, it will be used to
        infer the ``field_type`` parameter required by
        `arcpy.management.AddField` if ``field_type`` is itself not
        explicitly provided.
    overwrite : bool, optional (False)
        If True, an existing field will be overwritten. The default
        behavior will raise a `ValueError` if the field already exists.
    **field_opts : keyword options
        Keyword arguments that are passed directly to
        `arcpy.management.AddField`.

    Returns
    -------
    None

    Examples
    --------
    >>> from propagator import utils
    >>> # add a text field to shapefile (text fields need a length spec)
    >>> utils.add_field_with_value("mypolygons.shp", "storm_event",
                                   "100-yr", field_length=10)
    >>> # add a numeric field (doesn't require additional options)
    >>> utils.add_field_with_value("polygons.shp", "flood_level", 3.14)

    """

    # how Esri maps python types to field types
    typemap = {
        int: 'LONG',
        float: 'DOUBLE',
        unicode: 'TEXT',
        str: 'TEXT',
        type(None): None
    }

    # pull the field type from the options if it was specified,
    # otherwise lookup a type based on the `type(field_value)`.
    field_type = field_opts.pop("field_type", typemap[type(field_value)])

    if not overwrite:
        check_fields(table, field_name, should_exist=False)

    if field_value is None and field_type is None:
        raise ValueError("must provide a `field_type` if not providing a value.")

    # see http://goo.gl/66QD8c
    arcpy.management.AddField(
        in_table=table,
        field_name=field_name,
        field_type=field_type,
        **field_opts
    )

    # set the value in all rows
    if field_value is not None:
        populate_field(table, lambda row: field_value, field_name)


def cleanup_temp_results(*results):
    """ Deletes temporary results from the current workspace.

    Relies on `arcpy.management.Delete`_.

    .. _arcpy.management.Delete: http://goo.gl/LW85an

    Parameters
    ----------
    *results : str
        Paths to the temporary results

    Returns
    -------
    None

    """

    for r in results:
        if isinstance(r, basestring):
            path = r
        elif isinstance(r, arcpy.Result):
            path = r.getOutput(0)
        elif isinstance(r, arcpy.mapping.Layer):
            path = r.dataSource
        elif isinstance(r, arcpy.Raster):
            # Esri docs are incorrect here:
            # --> http://goo.gl/67NwDj
            # path doesn't include the name
            path = os.path.join(r.path, r.name)
        else:
            raise ValueError("Input must be paths, Results, Rasters, or Layers")

        fullpath = os.path.join(os.path.abspath(arcpy.env.workspace), path)
        arcpy.management.Delete(fullpath)


def intersect_polygon_layers(destination, *layers, **intersect_options):
    """
    Intersect polygon layers with each other. Basically a thin wrapper
    around `arcpy.analysis.Intersect`_.

    .. _arcpy.analysis.Intersect: http://goo.gl/O9YMY6

    Parameters
    ----------
    destination : str
        Filepath where the intersected output will be saved.
    *layers : str or arcpy.Mapping.Layer
        The polygon layers (or their paths) that will be intersected
        with each other.
    **intersect_options : keyword arguments
        Additional arguments that will be passed directly to
        `arcpy.analysis.Intersect`.

    Returns
    -------
    intersected : arcpy.mapping.Layer
        The arcpy Layer of the intersected polygons.

    Examples
    --------
    >>> from propagator import utils
    >>> blobs = utils.intersect_polygon_layers(
    ...     "flood_damage_intersect.shp"
    ...     "floods.shp",
    ...     "wetlands.shp",
    ...     "buildings.shp"
    ... )

    """

    result = arcpy.analysis.Intersect(
        in_features=layers,
        out_feature_class=destination,
        **intersect_options
    )

    intersected = result_to_layer(result)
    return intersected


def load_attribute_table(input_path, *fields):
    """
    Loads a shapefile's attribute table as a numpy record array.

    Relies on `arcpy.da.TableToNumPyArray`_.

    .. _arcpy.da.TableToNumPyArray: http://goo.gl/NzS6sB

    Parameters
    ----------
    input_path : str
        Fiilepath to the shapefile or feature class whose table needs
        to be read.
    *fields : str
        Names of the fields that should be included in the resulting
        array.

    Returns
    -------
    records : numpy.recarray
        A record array of the selected fields in the attribute table.

    See also
    --------
    groupby_and_aggregate

    Examples
    --------
    >>> from propagator import utils
    >>> path = "data/subcatchment.shp"
    >>> catchements = utils.load_attribute_table(path, 'CatchID',
    ... 'DwnCatchID', 'Watershed')
    >>> catchements[:5]
    array([(u'541', u'571', u'San Juan Creek'),
           (u'754', u'618', u'San Juan Creek'),
           (u'561', u'577', u'San Juan Creek'),
           (u'719', u'770', u'San Juan Creek'),
           (u'766', u'597', u'San Juan Creek')],
          dtype=[('CatchID', '<U20'), ('DwnCatchID', '<U20'),
                 ('Watershed', '<U50')])
    """
    # load the data
    layer = load_data(input_path, "layer")

    if len(fields) == 0:
        fields = get_field_names(input_path)

    # check that fields are valid
    check_fields(layer.dataSource, *fields, should_exist=True)

    array = arcpy.da.FeatureClassToNumPyArray(in_table=input_path, field_names=fields)
    return array


def unique_field_values(input_path, field):
    """
    Get an array of unique values in a table field.

    Parameters
    ----------
    input_path : str
        Fiilepath to the shapefile or feature class whose table needs
        to be read.
    fields : str
        Name of the field whose unique values will be returned

    Returns
    -------
    values : numpy.array
        The unique values of `field`.

    """

    table = load_attribute_table(input_path, field)
    return numpy.unique(table[field])


def groupby_and_aggregate(input_path, groupfield, valuefield,
                          aggfxn=None):
    """
    Counts the number of distinct values of `valuefield` are associated
    with each value of `groupfield` in a data source found at
    `input_path`.

    Parameters
    ----------
    input_path : str
        File path to a shapefile or feature class whose attribute table
        can be loaded with `arcpy.da.TableToNumPyArray`.
    groupfield : str
        The field name that would be used to group all of the records.
    valuefield : str
        The field name whose distinct values will be counted in each
        group defined by `groupfield`.
    aggfxn : callable, optional.
        Function to aggregate the values in each group to a single group.
        This function should accept an `itertools._grouper` as its only
        input. If not provided, unique number of value in the group will
        be returned.

    Returns
    -------
    counts : dict
        A dictionary whose keys are the distinct values of `groupfield`
        and values are the number of distinct records in each group.

    Examples
    --------
    >>> # compute total areas for each 'GeoID'
    >>> wetland_areas = utils.groupby_and_aggregate(
    ...     input_path='wetlands.shp',
    ...     groupfield='GeoID',
    ...     valuefield='SHAPE@AREA',
    ...     aggfxn=lambda group: sum([row[1] for row in group])
    ... )

    >>> # count the number of structures associated with each 'GeoID'
    >>> building_counts = utils.groupby_and_aggregate(
    ...     input_path=buildingsoutput,
    ...     groupfield=ID_column,
    ...     valuefield='STRUCT_ID'
    ... )

    See also
    --------
    itertools.groupby
    populate_field
    load_attribute_table

    """

    if aggfxn is None:
        aggfxn = lambda x: int(numpy.unique(list(x)).shape[0])

    table = load_attribute_table(input_path, groupfield, valuefield)
    table.sort()

    counts = {}
    for groupname, shapes in itertools.groupby(table, lambda row: row[groupfield]):
        counts[groupname] = aggfxn(shapes)

    return counts


def rename_column(table, oldname, newname, newalias=None):  # pragma: no cover
    """
    .. warning: Not yet implemented.
    """
    raise NotImplementedError
    if newalias is None:
        newalias = newname

    oldfield = filter(lambda f: name == oldname, get_field_names(table))[0]

    arcpy.management.AlterField(
        in_table=table,
        field=oldfield,
        new_field_name=newname,
        new_field_alias=newalias
    )


def populate_field(table, value_fxn, valuefield, keyfields=None):
    """
    Loops through the records of a table and populates the value of one
    field (`valuefield`) based on another field (`keyfield`) by passing
    the entire row through a function (`value_fxn`).

    Relies on `arcpy.da.UpdateCursor`_.

    .. _arcpy.da.UpdateCursor: http://goo.gl/sa3mW6

    Parameters
    ----------
    table : Layer, table, or file path
        This is the layer/file that will have a new field created.
    value_fxn : callable
        Any function that accepts a row from an `arcpy.da.SearchCursor`
        and returns a *single* value.
    valuefield : string
        The name of the field to be computed.
    keyfields : list of str, optional
        The other fields that need to be present in the rows of the
        cursor.

    Returns
    -------
    None

    .. note::
       In the row object, the `valuefield` will be the last item.
       In other words, `row[0]` will return the first values in
       `*keyfields` and `row[-1]` will return the existing value of
       `valuefield` in that row.

    Examples
    --------
    >>> # populate field ("Company") with a constant value ("Geosyntec")
    >>> populate_field("wetlands.shp", lambda row: "Geosyntec", "Company")

    """

    fields = validate.non_empty_list(keyfields, on_fail='create')
    fields.append(valuefield)
    check_fields(table, *fields, should_exist=True)

    with arcpy.da.UpdateCursor(table, fields) as cur:
        for row in cur:
            row[-1] = value_fxn(row)
            cur.updateRow(row)


def copy_layer(existing_layer, new_layer):
    """
    Makes copies of features classes, shapefiles, and maybe rasters.

    Parameters
    ----------
    existing_layer : str
        Path to the data to be copied
    new_layer : str
        Path to where ``existing_layer`` should be copied.

    Returns
    -------
    new_layer : str

    """

    arcpy.management.Copy(in_data=existing_layer, out_data=new_layer)
    return new_layer


def concat_results(destination, input_files):
    """ Concatentates (merges) serveral datasets into a single shapefile
    or feature class.

    Relies on `arcpy.management.Merge`_.

    .. _arcpy.management.Merge: http://goo.gl/JD3q0f

    Parameters
    ----------
    destination : str
        Path to where the concatentated dataset should be saved.
    input_files : list of str
        Strings of the paths of the datasets to be merged.

    Returns
    -------
    arcpy.mapping.Layer

    See also
    --------
    join_results_to_baseline

    """

    result = arcpy.management.Merge(input_files, destination)
    return result_to_layer(result)


def update_attribute_table(layerpath, attribute_array, id_column,
                           orig_columns, new_columns=None):
    """
    Update the attribute table of a feature class from a record array.

    Parameters
    ----------
    layerpath : str
        Path to the feature class to be updated.
    attribute_array : numpy.recarray
        A record array that contains the data to be writted into
        ``layerpath``.
    id_column : str
        The name of the column that uniquely identifies each feature in
        both ``layerpath`` and ``attribute_array``.
    *update_columns : str
        Names of the columns in both ``layerpath`` and
        ``attribute_array`` that will be updated.

    Returns
    -------
    None

    """

    if new_columns is None:
        new_columns = copy(orig_columns)

    # place the ID_column and columnes to be updated
    # in a single list
    all_columns = [id_column]
    all_columns.extend(orig_columns)

    # load the existing attributed table, loop through all rows
    with arcpy.da.UpdateCursor(layerpath, all_columns) as cur:
        for oldrow in cur:
            # find the current row in the new array
            newrow = find_row_in_array(attribute_array, id_column, oldrow[0])
            # loop through the value colums, setting them to the new values
            if newrow is not None:
                for n, col in enumerate(new_columns, 1):
                    oldrow[n] = newrow[col]

            # update the row
            cur.updateRow(oldrow)

    return layerpath


def delete_columns(layerpath, *columns):
    """
    Delete unwanted fields from an attribute table of a feature class.

    Parameters
    ----------
    layerpath : str
        Path to the feature class to be updated.
    *columns : str
        Names of the columns in ``layerpath`` that will be deleted

    Returns
    -------
    None

    """
    if len(columns) > 0:
        col_str = ";".join(columns)
        arcpy.management.DeleteField(layerpath, col_str)

    return layerpath


def spatial_join(left, right, outputfile, **kwargs):
    arcpy.analysis.SpatialJoin(
        target_features=left,
        join_features=right,
        out_feature_class=outputfile,
        **kwargs
    )

    return outputfile


def count_features(layer):
    return int(arcpy.management.GetCount(layer).getOutput(0))


def query_layer(inputpath, outputpath, sql):
    arcpy.analysis.Select(
        in_features=inputpath,
        out_feature_class=outputpath,
        where_clause=sql
    )

    return outputpath


def intersect_layers(input_paths, output_path, how='all'):
    """
    Intersect polygon layers with each other. Basically a thin wrapper
    around `arcpy.analysis.Intersect`_.

    .. _arcpy.analysis.Intersect: http://goo.gl/O9YMY6

    Parameters
    ----------
    input_paths : list of str or list of arcpy.Mapping.Layer
        The layers (or their paths) that will be intersected with each
        other.
    output_path : str
        Filepath where the intersected output will be saved.
    how : str
        Method by which the attributes should be joined. Valid values
        are: "all" (all attributes), or "only_fid" (just the feature
        IDs), or  "no_fid" (everything but the feature IDs)

    Returns
    -------
    output_path : arcpy.mapping.Layer
        The path to the layer containing the successfully intersected
        layers.

    Examples
    --------
    >>> from propagator import utils
    >>> blobs = utils.intersect_layers(
    ...     ["floods.shp", wetlands.shp", "buildings.shp"],
    ...     "flood_damage_intersect.shp"
    ... )

    """
    arcpy.analysis.Intersect(
        in_features=input_paths,
        out_feature_class=output_path,
        join_attributes=how.upper(),
        output_type="INPUT"
    )

    return output_path


def get_field_names(layerpath):
    """
    Gets the names of fields/columns in a feature class or table.
    Relies on `arcpy.ListFields`_.

    .. _arcpy.ListFields: http://goo.gl/Siq5y7

    Parameters
    ----------
    layerpath : str, arcpy.Layer, or arcpy.table
        The thing that has fields.

    Returns
    -------
    fieldnames : list of str

    """

    return [f.name for f in arcpy.ListFields(layerpath)]


def aggregate_geom(layerpath, by_fields, field_stat_tuples, outputpath=None, **kwargs):
    """
    Aggregates features class geometries into multipart geometries
    based on columns in attributes table. Basically this is a groupby
    operation on the attribute table, and the geometries are simply
    combined for aggregation. Other fields can but statistically
    aggregated as well.

    Parameters
    ----------
    layerpath : str
        Name of the input feature class.
    by_fields : list of str
        The fields in the attribute table on which the records will be
        aggregated.
    field_stat_tuples : list of tuples of str
        List of two-tuples where the first element element is a field
        in the atrribute and the second element is how that column
        should be aggreated.

        .. note ::

           Statistics that are available are limited to those supported
           by `arcpy.management.Dissolve`. Those are: "FIRST", "LAST",
           "SUM", "MEAN", "MIN", "MAX", "RANGE", "STD", and "COUNT".

    outputpath : str, optional
        Name of the new feature class where the output should be saved.
    **kwargs
        Additional parameters passed to `arcpy.management.Dissolve.

    Returns
    -------
    outputpath : str, optional
        Name of the new feature class where the output was sucessfully
        saved.

    Examples
    --------
    >>> from propagator import utils
    >>> with utils.WorkSpace('C:/SOC/data.gdb'):
    ...     utils.aggregate_geom(
    ...         layerpath='streams_with_WQ_scores',
    ...         by_fields=['Catch_ID', 'DS_Catch_ID'],
    ...         field_stat_tuples=[('Dry_Metals', 'max'), ('Wet_Metals', 'min')],
    ...         outputpath='agg_streams'
    ...     )

    """

    by_fields = validate.non_empty_list(by_fields)
    arcpy.management.Dissolve(
        in_features=layerpath,
        out_feature_class=outputpath,
        dissolve_field=by_fields,
        statistics_fields=field_stat_tuples,
        **kwargs
    )

    return outputpath


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


def rec_groupby(array, group_cols, *stats):
    """
    Perform a groupby-apply operation on a numpy record array.

    Returned record array has *dtype* names for each attribute name in
    the *groupby* argument, with the associated group values, and
    for each outname name in the *stats* argument, with the associated
    stat summary output. Adapted from https://goo.gl/NgwOID.

    Parameters
    ----------
    array : numpy.recarray
        The data to be grouped and aggregated.
    group_cols : str or sequence of str
        The columns that identify each group
    *stats : namedtuples or object
        Any number of namedtuples or objects specifying which columns
        should be aggregated, how they should be aggregated, and what
        the resulting column name should be. The keys/attributes for
        these tuples/objects must be: "srccol", "aggfxn", and "rescol".

    Returns
    -------
    aggregated : numpy.recarray

    See also
    --------
    Statistic

    Examples
    --------
    >>> from collections import namedtuple
    >>> from propagator import utils
    >>> import numpy
    >>> Statistic = namedtuple("Statistic", ("srccol", "aggfxn", "rescol"))
    >>> data = data = numpy.array([
            (u'050SC', 88.3, 0.0),  (u'050SC', 0.0, 0.1),
            (u'045SC', 49.2, 0.04), (u'045SC', 0.0, 0.08),
        ], dtype=[('ID', '<U10'), ('Cu', '<f8'), ('Pb', '<f8'),])
    >>> stats = [
            Statistic('Cu', numpy.max, 'MaxCu'),
            Statistic('Pb', numpy.min, 'MinPb')
        ]
    >>> utils.rec_groupby(data, ['ID'], *stats)
    rec.array(
        [(u'045SC', 49.2, 0.04),
         (u'050SC', 88.3, 0.0)],
        dtype=[('ID', '<U5'), ('MaxCu', '<f8'), ('MinPb', '<f8')]
    )

    """
    if numpy.isscalar(group_cols):
        group_cols = [group_cols]

    # build a dictionary from group_cols keys -> list of indices into
    # array with  those keys
    row_dict = dict()
    for i, row in enumerate(array):
        key = tuple([row[attr] for attr in group_cols])
        row_dict.setdefault(key, []).append(i)

    # sort the output by group_cols keys
    keys = list(row_dict.keys())
    keys.sort()

    output_rows = []
    for key in keys:
        row = list(key)

        # get the indices for this group_cols key
        index = row_dict[key]
        this_row = array[index]

        # call each aggregating function for this group_cols slice
        row.extend([stat.aggfxn(this_row[stat.srccol]) for stat in stats])
        output_rows.append(row)

    # build the output record array with group_cols and outname attributes
    outnames = [stat.rescol for stat in stats]
    names = list(group_cols)
    names.extend(outnames)
    record_array = numpy.rec.fromrecords(output_rows, names=names)
    return record_array


def stats_with_ignored_values(array, statfxn, ignored_value=None,
                              terminator_value=None):
    """
    Compute statistics on arrays while ignoring certain values

    Parameters
    ----------
    array : numyp.array (of floats)
        The values to be summarized
    statfxn : callable
        Function, lambda, or classmethod that can be called with
        ``array`` as the only input and returns a scalar value.
    ignored_value : float, optional
        The values in ``array`` that should be ignored.
    terminator_value : float, optional
        A value that is not propagated unless it is the only
        non-``ignored_value`` in the array.

    Returns
    -------
    res : float
        Scalar result of ``statfxn``. In that case that all values in
        ``array`` are ignored, ``ignored_value`` is returned.

    Examples
    --------
    >>> import numpy
    >>> from propagator import utils
    >>> x = [1., 2., 3., 4., 5.]
    >>> utils.stats_with_ignored_values(x, numpy.mean, ignored_value=5)
    2.5

    >>> y = [-99., 0., 1., 2., 3.]
    >>> utils.stats_with_ignored_values(y, numpy.mean, ignored_value=0
    ...                                 terminator_value=-99)
    2.0

    >>> z = [-99., 0., 0., 0., 0.]
    >>> utils.stats_with_ignored_values(y, numpy.mean, ignored_value=0
    ...                                 terminator_value=-99)
    -99.

    """

    if ignored_value is not None and ignored_value == terminator_value:
        raise ValueError("terminator and ignored values must be different.")

    # ensure that we're working with an array
    array = numpy.asarray(array)

    # drop ignored values if we know what to ignore
    if ignored_value is not None:
        array = array[numpy.nonzero(array != ignored_value)]

    # terminator values if necessary
    if terminator_value is not None:
        res =  stats_with_ignored_values(array, statfxn, ignored_value=terminator_value,
                                               terminator_value=None)
    # if empty, return the ignored value.
    # in a recursed, call, this is actually the terminator value.
    elif len(array) == 0:
        res = ignored_value

    # otherwise compute the stat.
    else:
        res = statfxn(array)
    return res


def weighted_average(arr):
    """
    Computed weighted average from two columns in an array.

    Parameters
    ----------
    arr : array-like
        Contains source values and weighting factors.
    value_col : str or int
        ID of the values column.
    weight_col : str or int
        ID of the weighting factor column.

    Returns
    -------
    output : float
        Weighted average.

    """
    columns = arr.dtype.names
    return numpy.average(arr[columns[0]], weights=arr[columns[1]])


def append_column_to_array(array, new_column, new_values, other_cols=None):
    """
    Adds a new column to a record array

    Parameters
    ----------
    array : numpy record array
    new_column : str
        The name of the new column to be created
    new_values : scalar or sequence
        The value or array of values to be inserted into the new column.
    other_cols : sequence of str, optional
        A subset of exististing columns that will be kept in the final
        array. If not provided, all existing columns are retained.

    Returns
    -------
    new_array : numpy record array
        The new array with the new column.

    """

    from numpy.lib.recfunctions import append_fields

    # validate and convert the new column to a list to work
    # with the numpy API
    new_column = validate.non_empty_list(new_column)

    # validate and select out all of the "other" columns
    if other_cols is not None:
        other_cols = validate.non_empty_list(other_cols)
        if new_column in other_cols:
            msg = "`new_column` can not be the name of an existing column."
            raise ValueError(msg)
        else:
            # this raises a nasty warning in numpy even though this is the
            # way the warning says we should do this:
            with warnings.catch_warnings(record=True) as w:  # pragma: no cover
                warnings.simplefilter("ignore")
                array = array[other_cols].copy()

                # make sure we don't have any unicode column names
                col_names = numpy.array(array.dtype.names)
                array.dtype.names = [cn.encode('ascii', 'ignore') for cn in col_names]

    # convert the new value to an array if necessary
    if numpy.isscalar(new_values):
        new_values = numpy.array([new_values] * array.shape[0])

    # append the new colum
    new_array = append_fields(array, new_column, [new_values])

    return new_array.data
