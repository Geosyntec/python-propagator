import os
from pkg_resources import resource_filename
import time

import arcpy
import numpy

import nose.tools as nt
import numpy.testing as nptest
import propagator.testing as pptest
import mock

import propagator
from propagator import utils


@nt.nottest
class MockResult(object):
    def __init__(self, path):
        self.path = path

    def getOutput(*args, **kwargs):
        return self.path


def test_add_suffix_to_filename():
    nt.assert_equal(utils.add_suffix_to_filename('example.shp', 'test'), 'example_test.shp')
    nt.assert_equal(utils.add_suffix_to_filename('example', 'test'), 'example_test')


def test_RasterTemplate():
    size, x, y = 8, 1, 2
    template = utils.RasterTemplate(size, x, y)
    nt.assert_equal(template.meanCellWidth, size)
    nt.assert_equal(template.meanCellHeight, size)
    nt.assert_equal(template.extent.lowerLeft.X, x)
    nt.assert_equal(template.extent.lowerLeft.Y, y)


def test_RasterTemplate_from_raster():
    _raster = resource_filename('propagator.testing._Template', 'dem.tif')
    raster = utils.load_data(_raster, 'raster')
    template = utils.RasterTemplate.from_raster(raster)
    nt.assert_equal(template.meanCellWidth, raster.meanCellWidth)
    nt.assert_equal(template.meanCellHeight, raster.meanCellHeight)
    nt.assert_equal(template.extent.lowerLeft.X, raster.extent.lowerLeft.X)
    nt.assert_equal(template.extent.lowerLeft.Y, raster.extent.lowerLeft.Y)


class Test_EasyMapDoc(object):
    def setup(self):
        self.mxd = resource_filename("propagator.testing.EasyMapDoc", "test.mxd")
        self.ezmd = utils.EasyMapDoc(self.mxd)

        self.knownlayer_names = ['ZOI', 'wetlands', 'ZOI_first_few', 'wetlands_first_few']
        self.knowndataframe_names = ['Main', 'Subset']
        self.add_layer_path = resource_filename("propagator.testing.EasyMapDoc", "ZOI.shp")

    def test_layers(self):
        nt.assert_true(hasattr(self.ezmd, 'layers'))
        layers_names = [layer.name for layer in self.ezmd.layers]
        nt.assert_list_equal(layers_names, self.knownlayer_names)

    def test_dataframes(self):
        nt.assert_true(hasattr(self.ezmd, 'dataframes'))
        df_names = [df.name for df in self.ezmd.dataframes]
        nt.assert_list_equal(df_names, self.knowndataframe_names)

    def test_findLayerByName(self):
        name = 'ZOI_first_few'
        lyr = self.ezmd.findLayerByName(name)
        nt.assert_true(isinstance(lyr, arcpy.mapping.Layer))
        nt.assert_equal(lyr.name, name)

    def test_add_layer_with_path(self):
        nt.assert_equal(len(self.ezmd.layers), 4)
        self.ezmd.add_layer(self.add_layer_path)
        nt.assert_equal(len(self.ezmd.layers), 5)

    def test_add_layer_with_layer_and_other_options(self):
        layer = arcpy.mapping.Layer(self.add_layer_path)
        nt.assert_equal(len(self.ezmd.layers), 4)
        self.ezmd.add_layer(layer, position='bottom', df=self.ezmd.dataframes[1])
        nt.assert_equal(len(self.ezmd.layers), 5)

    @nt.raises(ValueError)
    def test_bad_layer(self):
        self.ezmd.add_layer(123456)

    @nt.raises(ValueError)
    def test_bad_position(self):
        self.ezmd.add_layer(self.add_layer_path, position='junk')


class Test_Extension(object):
    def setup(self):
        self.known_available = 'spatial'
        self.known_unavailable = 'tracking'

    @nt.raises(RuntimeError)
    def test_unlicensed_extension(self):
        with utils.Extension(self.known_unavailable):
            pass

    def test_licensed_extension(self):
        nt.assert_equal(arcpy.CheckExtension(self.known_available), u'Available')
        with utils.Extension(self.known_available) as ext:
            nt.assert_equal(ext, 'CheckedOut')

        nt.assert_equal(arcpy.CheckExtension(self.known_available), u'Available')

    def teardown(self):
        arcpy.CheckInExtension(self.known_available)


class Test_OverwriteState(object):
    def test_true_true(self):
        arcpy.env.overwriteOutput = True

        nt.assert_true(arcpy.env.overwriteOutput)
        with utils.OverwriteState(True):
            nt.assert_true(arcpy.env.overwriteOutput)

        nt.assert_true(arcpy.env.overwriteOutput)

    def test_false_false(self):
        arcpy.env.overwriteOutput = False

        nt.assert_false(arcpy.env.overwriteOutput)
        with utils.OverwriteState(False):
            nt.assert_false(arcpy.env.overwriteOutput)

        nt.assert_false(arcpy.env.overwriteOutput)

    def test_true_false(self):
        arcpy.env.overwriteOutput = True

        nt.assert_true(arcpy.env.overwriteOutput)
        with utils.OverwriteState(False):
            nt.assert_false(arcpy.env.overwriteOutput)

        nt.assert_true(arcpy.env.overwriteOutput)

    def test_false_true(self):
        arcpy.env.overwriteOutput = False

        nt.assert_false(arcpy.env.overwriteOutput)
        with utils.OverwriteState(True):
            nt.assert_true(arcpy.env.overwriteOutput)

        nt.assert_false(arcpy.env.overwriteOutput)


class Test_WorkSpace(object):
    def setup(self):
        self.baseline = os.getcwd()
        self.new_ws = u'C:/Users'

        arcpy.env.workspace = self.baseline

    def test_workspace(self):
        nt.assert_equal(arcpy.env.workspace, self.baseline)
        with utils.WorkSpace(self.new_ws):
            nt.assert_equal(arcpy.env.workspace, self.new_ws)

        nt.assert_equal(arcpy.env.workspace, self.baseline)


class Test_create_temp_filename():
    def setup(self):
        self.folderworkspace = os.path.join('some', 'other', 'folder')
        self.geodbworkspace = os.path.join('another', 'geodb.gdb')

    def test_folderworkspace_withsubfolder(self):
        with utils.WorkSpace(self.folderworkspace):
            known_raster = os.path.join(self.folderworkspace, 'subfolder', '_temp_test.tif')
            temp_raster = utils.create_temp_filename(os.path.join('subfolder', 'test'), filetype='raster')
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, 'subfolder', '_temp_test.shp')
            temp_shape = utils.create_temp_filename(os.path.join('subfolder','test'), filetype='shape')
            nt.assert_equal(temp_shape, known_shape)

    def test_folderworkspace_withsubfolder_with_num(self):
        with utils.WorkSpace(self.folderworkspace):
            known_raster = os.path.join(self.folderworkspace, 'subfolder', '_temp_test_1.tif')
            temp_raster = utils.create_temp_filename(os.path.join('subfolder', 'test'), filetype='raster', num=1)
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, 'subfolder', '_temp_test_12.shp')
            temp_shape = utils.create_temp_filename(os.path.join('subfolder','test'), filetype='shape', num=12)
            nt.assert_equal(temp_shape, known_shape)

    def test_folderworkspace_barefile(self):
        with utils.WorkSpace(self.folderworkspace):
            known_raster = os.path.join(self.folderworkspace, '_temp_test.tif')
            temp_raster = utils.create_temp_filename('test', filetype='raster')
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, '_temp_test.shp')
            temp_shape = utils.create_temp_filename('test', filetype='shape')
            nt.assert_equal(temp_shape, known_shape)

    def test_folderworkspace_barefile_with_num(self):
        with utils.WorkSpace(self.folderworkspace):
            known_raster = os.path.join(self.folderworkspace, '_temp_test_14.tif')
            temp_raster = utils.create_temp_filename('test', filetype='raster', num=14)
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, '_temp_test_3.shp')
            temp_shape = utils.create_temp_filename('test', filetype='shape', num=3)
            nt.assert_equal(temp_shape, known_shape)

    def test_geodb_barefile(self):
        with utils.WorkSpace(self.geodbworkspace):
            known_raster = os.path.join(self.geodbworkspace, '_temp_test')
            temp_raster = utils.create_temp_filename('test', filetype='raster')
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.geodbworkspace, '_temp_test')
            temp_shape = utils.create_temp_filename('test', filetype='shape')
            nt.assert_equal(temp_shape, known_shape)

    def test_geodb_barefile_with_num(self):
        with utils.WorkSpace(self.geodbworkspace):
            known_raster = os.path.join(self.geodbworkspace, '_temp_test_7')
            temp_raster = utils.create_temp_filename('test', filetype='raster', num=7)
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.geodbworkspace, '_temp_test_22')
            temp_shape = utils.create_temp_filename('test', filetype='shape', num=22)
            nt.assert_equal(temp_shape, known_shape)

    def test_geodb_as_subfolder(self):
        with utils.WorkSpace(self.folderworkspace):
            filename = os.path.join(self.geodbworkspace, 'test')
            known_raster = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test')
            temp_raster = utils.create_temp_filename(filename, filetype='raster')
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test')
            temp_shape = utils.create_temp_filename(filename, filetype='shape')
            nt.assert_equal(temp_shape, known_shape)

    def test_geodb_as_subfolder_with_num(self):
        with utils.WorkSpace(self.folderworkspace):
            filename = os.path.join(self.geodbworkspace, 'test')
            known_raster = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test_5')
            temp_raster = utils.create_temp_filename(filename, filetype='raster', num=5)
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test_99')
            temp_shape = utils.create_temp_filename(filename, filetype='shape', num=99)
            nt.assert_equal(temp_shape, known_shape)

    def test_with_extension_geodb(self):
        with utils.WorkSpace(self.folderworkspace):
            filename = os.path.join(self.geodbworkspace, 'test')
            known_raster = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test')
            temp_raster = utils.create_temp_filename(filename + '.tif', filetype='raster')
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test')
            temp_shape = utils.create_temp_filename(filename + '.tif', filetype='shape')
            nt.assert_equal(temp_shape, known_shape)

    def test_with_extension_geodb_with_num(self):
        with utils.WorkSpace(self.folderworkspace):
            filename = os.path.join(self.geodbworkspace, 'test')
            known_raster = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test_2000')
            temp_raster = utils.create_temp_filename(filename + '.tif', filetype='raster', num=2000)
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test_999')
            temp_shape = utils.create_temp_filename(filename + '.tif', filetype='shape', num=999)
            nt.assert_equal(temp_shape, known_shape)

    def test_with_extension_folder(self):
        with utils.WorkSpace(self.folderworkspace):
            filename = 'test'
            known_raster = os.path.join(self.folderworkspace, '_temp_test.tif')
            temp_raster = utils.create_temp_filename(filename + '.tif', filetype='raster')
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, '_temp_test.shp')
            temp_shape = utils.create_temp_filename(filename + '.shp', filetype='shape')
            nt.assert_equal(temp_shape, known_shape)

    def test_with_extension_folder_with_num(self):
        with utils.WorkSpace(self.folderworkspace):
            filename = 'test'
            known_raster = os.path.join(self.folderworkspace, '_temp_test_4.tif')
            temp_raster = utils.create_temp_filename(filename + '.tif', filetype='raster', num=4)
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, '_temp_test_4.shp')
            temp_shape = utils.create_temp_filename(filename + '.shp', filetype='shape', num=4)
            nt.assert_equal(temp_shape, known_shape)


class Test_check_fields(object):
    table = resource_filename("propagator.testing.check_fields", "test_file.shp")

    def test_should_exist_uni(self):
        utils.check_fields(self.table, "Id", should_exist=True)

    def test_should_exist_multi(self):
        utils.check_fields(self.table, "Id", "existing", should_exist=True)

    def test_should_exist_multi_witharea(self):
        utils.check_fields(self.table, "Id", "existing", "SHAPE@AREA", should_exist=True)

    @nt.raises(ValueError)
    def test_should_exist_bad_vals(self):
        utils.check_fields(self.table, "Id", "existing", "JUNK", "GARBAGE", should_exist=True)

    def test_should_not_exist_uni(self):
        utils.check_fields(self.table, "NEWFIELD", should_exist=False)

    def test_should_not_exist_multi(self):
        utils.check_fields(self.table, "NEWFIELD", "YANFIELD", should_exist=False)

    def test_should_not_exist_multi_witharea(self):
        utils.check_fields(self.table, "NEWFIELD", "YANFIELD", "SHAPE@AREA", should_exist=False)

    @nt.raises(ValueError)
    def test_should_not_exist_bad_vals(self):
        utils.check_fields(self.table, "NEWFIELD", "YANFIELD", "existing", should_exist=False)


def test_result_to_raster():
    mockResult = mock.Mock(spec=arcpy.Result)
    mockRaster = mock.Mock(spec=arcpy.Raster)
    with mock.patch('arcpy.Raster', mockRaster):
        raster = utils.result_to_raster(mockResult)
        mockResult.getOutput.assert_called_once_with(0)


def test_result_to_Layer():
    mockResult = mock.Mock(spec=arcpy.Result)
    mockLayer = mock.Mock(spec=arcpy.mapping.Layer)
    with mock.patch('arcpy.mapping.Layer', mockLayer):
        layer = utils.result_to_layer(mockResult)
        mockResult.getOutput.assert_called_once_with(0)


class Test_load_data(object):
    rasterpath = resource_filename("propagator.testing.load_data", 'test_dem.tif')
    vectorpath = resource_filename("propagator.testing.load_data", 'test_wetlands.shp')

    @nt.raises(ValueError)
    def test_bad_datatype(self):
        utils.load_data(self.rasterpath, 'JUNK')

    @nt.raises(ValueError)
    def test_datapath_doesnt_exist(self):
        utils.load_data('junk.shp', 'grid')

    @nt.raises(ValueError)
    def test_datapath_bad_value(self):
        utils.load_data(12345, 'grid')

    @nt.raises(ValueError)
    def test_vector_as_grid_should_fail(self):
        x = utils.load_data(self.vectorpath, 'grid')

    @nt.raises(ValueError)
    def test_vector_as_raster_should_fail(self):
        x = utils.load_data(self.vectorpath, 'raster')

    def test_raster_as_raster(self):
        x = utils.load_data(self.rasterpath, 'raster')
        nt.assert_true(isinstance(x, arcpy.Raster))

    def test_raster_as_grid_with_caps(self):
        x = utils.load_data(self.rasterpath, 'gRId')
        nt.assert_true(isinstance(x, arcpy.Raster))

    def test_raster_as_layer_not_greedy(self):
        x = utils.load_data(self.rasterpath, 'layer', greedyRasters=False)
        nt.assert_true(isinstance(x, arcpy.mapping.Layer))

    def test_raster_as_layer_greedy(self):
        x = utils.load_data(self.rasterpath, 'layer')
        nt.assert_true(isinstance(x, arcpy.Raster))

    def test_vector_as_shape(self):
        x = utils.load_data(self.vectorpath, 'shape')
        nt.assert_true(isinstance(x, arcpy.mapping.Layer))

    def test_vector_as_layer_with_caps(self):
        x = utils.load_data(self.vectorpath, 'LAyeR')
        nt.assert_true(isinstance(x, arcpy.mapping.Layer))

    def test_already_a_layer(self):
        lyr = arcpy.mapping.Layer(self.vectorpath)
        x = utils.load_data(lyr, 'layer')
        nt.assert_equal(x, lyr)

    def test_already_a_raster(self):
        raster = arcpy.Raster(self.rasterpath)
        x = utils.load_data(raster, 'raster')
        nt.assert_true(isinstance(x, arcpy.Raster))


class Test_add_field_with_value(object):
    def setup(self):
        source = resource_filename("propagator.testing.add_field_with_value", 'field_adder.shp')
        with utils.OverwriteState(True):
            self.testfile = utils.copy_layer(source, source.replace('field_adder', 'test'))
        self.fields_added = ["_text", "_unicode", "_int", "_float", '_no_valstr', '_no_valnum']

    def teardown(self):
        utils.cleanup_temp_results(self.testfile)

    def test_float(self):
        name = "_float"
        utils.add_field_with_value(self.testfile, name,
                                   field_value=5.0)
        nt.assert_true(name in [f.name for f in arcpy.ListFields(self.testfile)])

        newfield = arcpy.ListFields(self.testfile, name)[0]
        nt.assert_equal(newfield.type, u'Double')

    def test_int(self):
        name = "_int"
        utils.add_field_with_value(self.testfile, name,
                                   field_value=5)
        nt.assert_true(name in [f.name for f in arcpy.ListFields(self.testfile)])

        newfield = arcpy.ListFields(self.testfile, name)[0]
        nt.assert_equal(newfield.type, u'Integer')

    def test_string(self):
        name = "_text"
        utils.add_field_with_value(self.testfile, name,
                                   field_value="example_value",
                                   field_length=15)

        nt.assert_true(name in [f.name for f in arcpy.ListFields(self.testfile)])

        newfield = arcpy.ListFields(self.testfile, name)[0]
        nt.assert_equal(newfield.type, u'String')
        nt.assert_true(newfield.length, 15)

    def test_unicode(self):
        name = "_unicode"
        utils.add_field_with_value(self.testfile, name,
                                   field_value=u"example_value",
                                   field_length=15)

        nt.assert_true(name in [f.name for f in arcpy.ListFields(self.testfile)])

        newfield = arcpy.ListFields(self.testfile, name)[0]
        nt.assert_equal(newfield.type, u'String')
        nt.assert_true(newfield.length, 15)

    def test_no_value_string(self):
        name = "_no_valstr"
        utils.add_field_with_value(self.testfile, name,
                                   field_type='TEXT',
                                   field_length=15)

        nt.assert_true(name in [f.name for f in arcpy.ListFields(self.testfile)])

        newfield = arcpy.ListFields(self.testfile, name)[0]
        nt.assert_equal(newfield.type, u'String')
        nt.assert_true(newfield.length, 15)

    def test_no_value_number(self):
        name = "_no_valnum"
        utils.add_field_with_value(self.testfile, name,
                                   field_type='DOUBLE')

        nt.assert_true(name in [f.name for f in arcpy.ListFields(self.testfile)])

        newfield = arcpy.ListFields(self.testfile, name)[0]
        nt.assert_equal(newfield.type, u'Double')

    @nt.raises(ValueError)
    def test_no_value_no_field_type(self):
        utils.add_field_with_value(self.testfile, "_willfail")

    @nt.raises(ValueError)
    def test_overwrite_existing_no(self):
        utils.add_field_with_value(self.testfile, "existing")

    def test_overwrite_existing_yes(self):
        utils.add_field_with_value(self.testfile, "existing",
                                   overwrite=True,
                                   field_type="LONG")


def test_cleanup_temp_results():

    workspace = os.path.abspath(resource_filename('propagator.testing', 'cleanup_temp_results'))
    template_file = 'test_dem.tif'

    name1 = 'temp_1.tif'
    name2 = 'temp_2.tif'

    with utils.WorkSpace(workspace):
        raster1 = utils.copy_layer(template_file, name1)
        raster2 = utils.copy_layer(template_file, name2)

    nt.assert_true(os.path.exists(os.path.join(workspace, 'temp_1.tif')))
    nt.assert_true(os.path.exists(os.path.join(workspace, 'temp_2.tif')))

    with utils.WorkSpace(workspace):
        utils.cleanup_temp_results(name1, name2)

    nt.assert_false(os.path.exists(os.path.join(workspace, 'temp_1.tif')))
    nt.assert_false(os.path.exists(os.path.join(workspace, 'temp_2.tif')))


@nt.raises(ValueError)
def test_cleanup_with_bad_input():
    utils.cleanup_temp_results(1, 2, ['a', 'b', 'c'])


@nptest.dec.skipif(not pptest.has_fiona)
def test_intersect_polygon_layers():
    input1_file = resource_filename("propagator.testing.intersect_polygons", "intersect_input1.shp")
    input2_file = resource_filename("propagator.testing.intersect_polygons", "intersect_input2.shp")
    known_file = resource_filename("propagator.testing.intersect_polygons", "intersect_known.shp")
    output_file = resource_filename("propagator.testing.intersect_polygons", "intersect_output.shp")

    with utils.OverwriteState(True):
        output = utils.intersect_polygon_layers(
            output_file,
            [input1_file, input2_file,]
        )

    nt.assert_true(isinstance(output, arcpy.mapping.Layer))
    pptest.assert_shapefiles_are_close(output_file, known_file)

    utils.cleanup_temp_results(output)


def test_load_attribute_table():
    path = resource_filename('propagator.testing.load_attribute_table', 'subcatchments.shp')
    expected_top_five = numpy.array(
        [
            (u'541', u'571', u'San Juan Creek'),
            (u'754', u'618', u'San Juan Creek'),
            (u'561', u'577', u'San Juan Creek'),
            (u'719', u'770', u'San Juan Creek'),
            (u'766', u'597', u'San Juan Creek'),
        ],
        dtype=[
            ('CatchID', '<U20'),
            ('DwnCatchID', '<U20'),
            ('Watershed', '<U50'),
        ]
    )

    result = utils.load_attribute_table(path, 'CatchID', 'DwnCatchID', 'Watershed')
    nptest.assert_array_equal(result[:5], expected_top_five)


def test_unique_field_values():
    path = resource_filename('propagator.testing.load_attribute_table', 'subcatchments.shp')
    result = utils.unique_field_values(path, 'Watershed')
    nptest.assert_array_equal(result, numpy.array(['San Clemente', 'San Juan Creek']))


class Test_groupby_and_aggregate():
    known_counts = {16.0: 32, 150.0: 2}
    buildings = resource_filename("propagator.testing.groupby_and_aggregate", "flooded_buildings.shp")
    group_col = 'GeoID'
    count_col = 'STRUCT_ID'
    area_op = 'SHAPE@AREA'

    areas = resource_filename("propagator.testing.groupby_and_aggregate", "intersect_input1.shp")
    known_areas = {2: 1327042.1024, 7: 1355433.0192, 12: 1054529.2882}

    def test_defaults(self):
        counts = utils.groupby_and_aggregate(
            self.buildings,
            self.group_col,
            self.count_col,
            aggfxn=None
        )

        nt.assert_dict_equal(counts, self.known_counts)

    def test_area(self):
        areadict = utils.groupby_and_aggregate(
            self.areas,
            self.group_col,
            self.area_op,
            aggfxn=lambda g: sum([row[1] for row in g])
        )
        for key in areadict.keys():
            nt.assert_almost_equal(
                areadict[key],
                self.known_areas[key],
                delta=0.01
            )

    def test_recarry_sort_no_args(self):
        known = numpy.array([
            ('A', 1.), ('A', 2.), ('A', 3.), ('A', 4.),
            ('B', 1.), ('B', 2.), ('B', 3.), ('B', 4.),
            ('C', 1.), ('C', 2.), ('C', 3.), ('C', 4.),
        ], dtype=[('GeoID', 'S4'), ('Area', float)])

        test = numpy.array([
            ('A', 1.), ('B', 1.), ('C', 3.), ('A', 4.),
            ('C', 4.), ('A', 2.), ('C', 1.), ('A', 3.),
            ('B', 2.), ('C', 2.), ('B', 4.), ('B', 3.),
        ], dtype=[('GeoID', 'S4'), ('Area', float)])

        test.sort()
        nptest.assert_array_equal(test, known)

    @nt.raises(ValueError)
    def test_bad_group_col(self):
        counts = utils.groupby_and_aggregate(
            self.buildings,
            "JUNK",
            self.count_col
        )

    @nt.raises(ValueError)
    def test_bad_count_col(self):
        counts = utils.groupby_and_aggregate(
            self.buildings,
            self.group_col,
            "JUNK"
        )


@nt.raises(NotImplementedError)
def test_rename_column():
    layer = resource_filename("propagator.testing.rename_column", "rename_col.dbf")
    oldname = "existing"
    newname = "exists"

    #layer = utils.load_data(inputfile, "layer")

    utils.rename_column(layer, oldname, newname)
    utils.check_fields(layer, newname, should_exist=True)
    utils.check_fields(layer, oldname, should_exist=False)

    utils.rename_column(layer, newname, oldname)
    utils.check_fields(layer, newname, should_exist=False)
    utils.check_fields(layer, oldname, should_exist=True)


class Test_populate_field(object):
    def setup(self):
        source = resource_filename("propagator.testing.populate_field", 'source.shp')
        self.testfile = utils.copy_layer(source, source.replace('source', 'test'))
        self.field_added = "newfield"

    def teardown(self):
        utils.cleanup_temp_results(self.testfile)

    def test_with_dictionary(self):
        value_dict = {n: n for n in range(7)}
        value_fxn = lambda row: value_dict.get(row[0], -1)
        utils.add_field_with_value(self.testfile, self.field_added, field_type="LONG")

        utils.populate_field(
            self.testfile,
            lambda row: value_dict.get(row[0], -1),
            self.field_added,
            ["FID"]
        )

        with arcpy.da.SearchCursor(self.testfile, [self.field_added, "FID"]) as cur:
            for row in cur:
                nt.assert_equal(row[0], row[1])

    def test_with_general_function(self):
        utils.add_field_with_value(self.testfile, self.field_added, field_type="LONG")
        utils.populate_field(
            self.testfile,
            lambda row: row[0]**2,
            self.field_added,
            ["FID"]
        )

        with arcpy.da.SearchCursor(self.testfile, [self.field_added, "FID"]) as cur:
            for row in cur:
                nt.assert_equal(row[0], row[1] ** 2)


def test_copy_layer():
    with mock.patch.object(arcpy.management, 'Copy') as _copy:
        in_data = 'input'
        out_data = 'new_copy'

        result = utils.copy_layer(in_data, out_data)
        _copy.assert_called_once_with(in_data=in_data, out_data=out_data)
        nt.assert_equal(result, out_data)


def test_intersect_layers():
    ws = resource_filename('propagator.testing', 'intersect_layers')
    with utils.OverwriteState(True), utils.WorkSpace(ws):
        utils.intersect_layers(
            ['subcatchments.shp', 'monitoring_locations.shp'],
            'test.shp',
        )

    pptest.assert_shapefiles_are_close(
        os.path.join(ws, 'expected.shp'),
        os.path.join(ws, 'test.shp'),
    )

    utils.cleanup_temp_results(os.path.join(ws, 'test.shp'))


@nptest.dec.skipif(not pptest.has_fiona)
def test_concat_results():
    known = resource_filename('propagator.testing.concat_results', 'known.shp')
    with utils.OverwriteState(True):
        test = utils.concat_results(
            resource_filename('propagator.testing.concat_results', 'result.shp'),
            [resource_filename('propagator.testing.concat_results', 'input1.shp'),
             resource_filename('propagator.testing.concat_results', 'input2.shp')]
        )

    nt.assert_true(isinstance(test, arcpy.mapping.Layer))
    pptest.assert_shapefiles_are_close(test.dataSource, known)

    utils.cleanup_temp_results(test)


@nptest.dec.skipif(not pptest.has_fiona)
def test_spatial_join():
    known = resource_filename('propagator.testing.spatial_join', 'merge_result.shp')
    left = resource_filename('propagator.testing.spatial_join', 'merge_baseline.shp')
    right = resource_filename('propagator.testing.spatial_join', 'merge_join.shp')
    outputfile = resource_filename('propagator.testing.spatial_join', 'merge_result.shp')
    with utils.OverwriteState(True):
        test = utils.spatial_join(left=left, right=right, outputfile=outputfile)

    nt.assert_equal(test, outputfile)
    pptest.assert_shapefiles_are_close(test, known)

    utils.cleanup_temp_results(test)


@nptest.dec.skipif(not pptest.has_fiona)
def test_update_attribute_table():
    ws = resource_filename('propagator.testing', 'update_attribute_table')
    with utils.WorkSpace(ws), utils.OverwriteState(True):
        inputpath = resource_filename("propagator.testing.update_attribute_table", "input.shp")
        testpath = inputpath.replace('input', 'test')
        expected = resource_filename("propagator.testing.update_attribute_table", "expected_output.shp")

        new_attributes = numpy.array(
            [
                (1, 0, u'Cu_1', 'Pb_1'), (2, 0, u'Cu_2', 'Pb_2'),
                (3, 0, u'Cu_3', 'Pb_3'), (4, 0, u'Cu_4', 'Pb_4'),
            ], dtype=[('id', int), ('ds_id', int), ('Cu', '<U5'), ('Pb', '<U5'),]
        )

        arcpy.management.Copy(inputpath, testpath)
        utils.update_attribute_table(testpath, new_attributes, 'id', ['Cu', 'Pb'])

        pptest.assert_shapefiles_are_close(testpath, expected)
        utils.cleanup_temp_results(testpath)


def test_get_field_names():
    expected = [u'FID', u'Shape', u'Station', u'Latitude', u'Longitude']
    layer = resource_filename('propagator.testing.get_field_names', 'input.shp')
    result = utils.get_field_names(layer)
    nt.assert_list_equal(result, expected)


class Test_aggregate_geom(object):
    def setup(self):
        self.workspace = resource_filename('propagator.testing', 'aggregate_geom')
        self.expected_single = 'known_one_group_field.shp'
        self.expected_dual = 'known_two_group_fields.shp'
        self.input_file = 'agg_geom.shp'
        self.output = 'test.shp'
        self.stats = [('WQ_1', 'mean'), ('WQ_2', 'max')]

    def teardown(self):
        with utils.WorkSpace(self.workspace):
            utils.cleanup_temp_results(self.output)

    @nt.nottest
    def check(self, results, expected):
        nt.assert_equal(results, self.output)
        pptest.assert_shapefiles_are_close(
            os.path.join(self.workspace, expected),
            os.path.join(self.workspace, results),
        )

    def test_single_group_col(self):
        with utils.WorkSpace(self.workspace):
            results = utils.aggregate_geom(
                layerpath=self.input_file,
                by_fields='CID',
                field_stat_tuples=self.stats,
                outputpath=self.output,
            )
        self.check(results, self.expected_single)

    def test_dual_group_col(self):
        with utils.WorkSpace(self.workspace):
            results = utils.aggregate_geom(
                layerpath=self.input_file,
                by_fields=['CID', 'DS_CID'],
                field_stat_tuples=self.stats,
                outputpath=self.output,
            )

        self.check(results, self.expected_dual)


def test_count_features():
    layer = resource_filename('propagator.testing.count_features', 'monitoring_locations.shp')
    nt.assert_equal(utils.count_features(layer), 14)


def test_query_layer():
    with mock.patch.object(arcpy.analysis, 'Select') as query:
        in_data = 'input'
        out_data = 'new_copy'
        sql = "fake SQL string"

        result = utils.query_layer(in_data, out_data, sql)
        query.assert_called_once_with(
            in_features=in_data,
            out_feature_class=out_data,
            where_clause=sql,
        )
        nt.assert_equal(result, out_data)


def test_delete_columns():
    with mock.patch.object(arcpy.management, 'DeleteField') as delete:
        in_data = 'input'
        expected_col_string = 'ThisColumn;ThatColumn;AnotherColumn'

        result = utils.delete_columns(in_data, 'ThisColumn', 'ThatColumn', 'AnotherColumn')
        delete.assert_called_once_with(in_data, expected_col_string)
        nt.assert_equal(result, in_data)


def test_delete_columns_no_columns():
    with mock.patch.object(arcpy.management, 'DeleteField') as delete:
        in_data = 'input'
        expected_col_string = 'ThisColumn;ThatColumn;AnotherColumn'

        result = utils.delete_columns(in_data)
        delete.assert_not_called()
        nt.assert_equal(result, in_data)


class Test_find_row_in_array(object):
    def setup(self):
        self.input_array = numpy.array(
            [
                ('A1', 'Ocean', 'A1_x', 'A1_y'), ('A2', 'Ocean', 'A2_x', 'A2_y'),
                ('B1', 'A1', 'None', 'B1_y'), ('B2', 'A1', 'B2_x', 'None'),
                ('B3', 'A2', 'B3_x', 'B3_y'), ('C1', 'B2', 'C1_x', 'None'),
            ], dtype=[('ID', '<U5'), ('DS_ID', '<U5'), ('Cu', '<U5'), ('Pb', '<U5'),]
        )

    def test_no_rows_returned(self):
        row = utils.find_row_in_array(self.input_array, 'ID', 'Junk')
        nt.assert_true(row is None)

    def test_normal_1_row(self):
        row = utils.find_row_in_array(self.input_array, 'ID', 'A1')
        nt.assert_tuple_equal(tuple(row), tuple(self.input_array[0]))

    @nt.raises(ValueError)
    def test_too_man_rows(self):
         row = utils.find_row_in_array(self.input_array, 'DS_ID', 'A1')


def test_Statistic():
    x = utils.Statistic('Cu', numpy.mean, 'MaxCu')
    nt.assert_true(hasattr(x, 'srccol'))
    nt.assert_equal(x.srccol, 'Cu')

    nt.assert_true(hasattr(x, 'aggfxn'))
    nt.assert_equal(x.aggfxn, numpy.mean)

    nt.assert_true(hasattr(x, 'rescol'))
    nt.assert_equal(x.rescol, 'MaxCu')


class Test_rec_groupby(object):
    def setup(self):
        self.data = numpy.array([
            (u'050SC', u'A', 88.3, 0.25), (u'050SC', 'B', 0.0, 0.50),
            (u'050SC', u'A', 98.3, 0.75), (u'050SC', 'B', 1.0, 1.00),
            (u'045SC', u'A', 49.2, 0.04), (u'045SC', 'B', 0.0, 0.08),
            (u'045SC', u'A', 69.2, 0.08), (u'045SC', 'B', 2.0, 0.16),
        ], dtype=[('ID', '<U5'), ('DS_ID', '<U5'), ('Cu', '<f8'), ('Pb', '<f8'),])

        self.expected_one_group_col = numpy.rec.fromrecords([
            (u'050SC', 98.3, 0.625),
            (u'045SC', 69.2, 0.090),
        ], names=['ID', 'MaxCu', 'AvgPb'])

        self.expected_two_group_col = numpy.rec.fromrecords([
            (u'050SC', u'A', 98.3, 0.50),
            (u'050SC', u'B',  1.0, 0.75),
            (u'045SC', u'A', 69.2, 0.06),
            (u'045SC', u'B',  2.0, 0.12),
        ], names=['ID', 'DS_ID', 'MaxCu', 'AvgPb'])

        self.stats = [
            utils.Statistic('Cu', numpy.max, 'MaxCu'),
            utils.Statistic('Pb', numpy.mean, 'AvgPb')
        ]

    def test_one_group_col(self):
        result = utils.rec_groupby(self.data, 'ID', *self.stats)
        result.sort()

        expected = self.expected_one_group_col.copy()
        expected.sort()

        nptest.assert_array_equal(result, expected)

    def test_two_group_col(self):
        result = utils.rec_groupby(self.data, ['ID', 'DS_ID'], *self.stats)
        result.sort()

        expected = self.expected_two_group_col.copy()
        expected.sort()

        nptest.assert_array_equal(result, expected)


class Test_stats_with_ignored_values(object):
    def setup(self):
        self.x1 = [1., 2., 3., 4., 5.]
        self.x2 = [5.] * 5 # just a list of 5's
        self.x3 = [1., 1., 1., 1., 5.]

    def test_defaults(self):
        expected = 3.
        result = utils.stats_with_ignored_values(self.x1, numpy.mean, ignored_value=None)
        nt.assert_equal(result, expected)

    def test_with_ignore(self):
        expected = 2.5
        result = utils.stats_with_ignored_values(self.x1, numpy.mean, ignored_value=5)
        nt.assert_equal(result, expected)

    def test_with_terminator(self):
        expected = 3.5
        result = utils.stats_with_ignored_values(self.x1, numpy.mean, terminator_value=1)
        nt.assert_equal(result, expected)

    def test_nothing_to_ignore(self):
        expected = 3.
        result = utils.stats_with_ignored_values(self.x1, numpy.mean, ignored_value=6)
        nt.assert_equal(result, expected)

    def test_nothing_to_terminate(self):
        expected = 3.
        result = utils.stats_with_ignored_values(self.x1, numpy.mean, terminator_value=6)
        nt.assert_equal(result, expected)

    def test_only_ignore_everthing(self):
        expected = 5.
        result = utils.stats_with_ignored_values(self.x2, numpy.mean, ignored_value=5)
        nt.assert_equal(result, expected)

    def test_only_terminate_everthing(self):
        expected = 5.
        result = utils.stats_with_ignored_values(self.x2, numpy.mean, terminator_value=5)
        nt.assert_equal(result, expected)

    def test_ignored_and_terminated_returns_stat_value(self):
        expected = 2.
        result = utils.stats_with_ignored_values(self.x1, numpy.mean,
                                                 ignored_value=4.,
                                                 terminator_value=5.,)
        nt.assert_equal(result, expected)

    def test_everything_ignored_or_terminated_returns_terminator(self):
        expected = 5.
        result = utils.stats_with_ignored_values(self.x2, numpy.mean,
                                                 ignored_value=1.,
                                                 terminator_value=5.,)
        nt.assert_equal(result, expected)


def test_weighted_average():
    raw_data = numpy.array(
        [
            (20, 45.23,),
            (43.3, 45.23,),
            (0.32, 41,),
            (0.32, 4,),
            (32, 45.23,),
            (1, 45.23,),
        ], dtype=[('value', '<f4'), ('w_factor', '<f4'),]
    )

    expected_result = numpy.array(
        [
            (19.343349,),
        ], dtype=[('value', '<f4'),]
    )

    result = utils.weighted_average(raw_data)
    nt.assert_equal(result, expected_result['value'])


class Test_append_column_to_array():
    def setup(self):
        self.newval = 12.34
        self.newcol = 'newcol'
        self.raw_data = numpy.array(
            [
                (20.0 , 45.23,),
                (43.3 , 45.23,),
                ( 0.32, 41.0 ,),
                ( 0.32,  4.0 ,),
                (32.0 , 45.23,),
                ( 1.0 , 45.23,),
            ], dtype=[('value', '<f4'), ('w_factor', '<f4'),]
        )

        self.expected_all = numpy.array(
            [
                (20.0 , 45.23, self.newval),
                (43.3 , 45.23, self.newval),
                ( 0.32, 41.0 , self.newval),
                ( 0.32,  4.0 , self.newval),
                (32.0 , 45.23, self.newval),
                ( 1.0 , 45.23, self.newval),
            ], dtype=[('value', '<f4'), ('w_factor', '<f4'), (self.newcol, '<f8')]
        )

        self.expected_subset = numpy.array(
            [
                (20.0 , self.newval),
                (43.3 , self.newval),
                ( 0.32, self.newval),
                ( 0.32, self.newval),
                (32.0 , self.newval),
                ( 1.0 , self.newval),
            ], dtype=[('value', '<f4'), (self.newcol, '<f8')]
        )

    def test_basic(self):
        result = utils.append_column_to_array(self.raw_data, self.newcol, self.newval)
        nptest.assert_array_equal(result, self.expected_all)

    def test_subset_scalar_colnames(self):
        result = utils.append_column_to_array(self.raw_data, self.newcol, self.newval,
                                              other_cols='value')
        nptest.assert_array_equal(result, self.expected_subset)

    def test_subset_list_colnames(self):
        result = utils.append_column_to_array(self.raw_data, self.newcol, self.newval,
                                              other_cols=['value'])
        nptest.assert_array_equal(result, self.expected_subset)
