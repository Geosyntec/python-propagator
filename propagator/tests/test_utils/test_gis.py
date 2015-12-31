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
from propagator.utils import gis


@nt.nottest
class MockResult(object):
    def __init__(self, path):
        self.path = path

    def getOutput(*args, **kwargs):
        return self.path


def test_RasterTemplate():
    size, x, y = 8, 1, 2
    template = gis.RasterTemplate(size, x, y)
    nt.assert_equal(template.meanCellWidth, size)
    nt.assert_equal(template.meanCellHeight, size)
    nt.assert_equal(template.extent.lowerLeft.X, x)
    nt.assert_equal(template.extent.lowerLeft.Y, y)


def test_RasterTemplate_from_raster():
    _raster = resource_filename('propagator.testing._Template', 'dem.tif')
    raster = gis.load_data(_raster, 'raster')
    template = gis.RasterTemplate.from_raster(raster)
    nt.assert_equal(template.meanCellWidth, raster.meanCellWidth)
    nt.assert_equal(template.meanCellHeight, raster.meanCellHeight)
    nt.assert_equal(template.extent.lowerLeft.X, raster.extent.lowerLeft.X)
    nt.assert_equal(template.extent.lowerLeft.Y, raster.extent.lowerLeft.Y)


class Test_EasyMapDoc(object):
    def setup(self):
        self.mxd = resource_filename("propagator.testing.EasyMapDoc", "test.mxd")
        self.ezmd = gis.EasyMapDoc(self.mxd)

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
        self.known_unavailable = 'Datareviewer'

    @nt.raises(RuntimeError)
    def test_unlicensed_extension(self):
        with gis.Extension(self.known_unavailable):
            pass

    def test_licensed_extension(self):
        nt.assert_equal(arcpy.CheckExtension(self.known_available), u'Available')
        with gis.Extension(self.known_available) as ext:
            nt.assert_equal(ext, 'CheckedOut')

        nt.assert_equal(arcpy.CheckExtension(self.known_available), u'Available')

    def teardown(self):
        arcpy.CheckExtension(self.known_available)


class Test_OverwriteState(object):
    def test_true_true(self):
        arcpy.env.overwriteOutput = True

        nt.assert_true(arcpy.env.overwriteOutput)
        with gis.OverwriteState(True):
            nt.assert_true(arcpy.env.overwriteOutput)

        nt.assert_true(arcpy.env.overwriteOutput)

    def test_false_false(self):
        arcpy.env.overwriteOutput = False

        nt.assert_false(arcpy.env.overwriteOutput)
        with gis.OverwriteState(False):
            nt.assert_false(arcpy.env.overwriteOutput)

        nt.assert_false(arcpy.env.overwriteOutput)

    def test_true_false(self):
        arcpy.env.overwriteOutput = True

        nt.assert_true(arcpy.env.overwriteOutput)
        with gis.OverwriteState(False):
            nt.assert_false(arcpy.env.overwriteOutput)

        nt.assert_true(arcpy.env.overwriteOutput)

    def test_false_true(self):
        arcpy.env.overwriteOutput = False

        nt.assert_false(arcpy.env.overwriteOutput)
        with gis.OverwriteState(True):
            nt.assert_true(arcpy.env.overwriteOutput)

        nt.assert_false(arcpy.env.overwriteOutput)


class Test_WorkSpace(object):
    def setup(self):
        self.baseline = os.getcwd()
        self.new_ws = u'C:/Users'

        arcpy.env.workspace = self.baseline

    def test_workspace(self):
        nt.assert_equal(arcpy.env.workspace, self.baseline)
        with gis.WorkSpace(self.new_ws):
            nt.assert_equal(arcpy.env.workspace, self.new_ws)

        nt.assert_equal(arcpy.env.workspace, self.baseline)


class Test_create_temp_filename():
    def setup(self):
        self.folderworkspace = os.path.join('some', 'other', 'folder')
        self.geodbworkspace = os.path.join('another', 'geodb.gdb')

    def test_folderworkspace_withsubfolder(self):
        with gis.WorkSpace(self.folderworkspace):
            known_raster = os.path.join(self.folderworkspace, 'subfolder', '_temp_test.tif')
            temp_raster = gis.create_temp_filename(os.path.join('subfolder', 'test'), filetype='raster')
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, 'subfolder', '_temp_test.shp')
            temp_shape = gis.create_temp_filename(os.path.join('subfolder','test'), filetype='shape')
            nt.assert_equal(temp_shape, known_shape)

    def test_folderworkspace_withsubfolder_with_num(self):
        with gis.WorkSpace(self.folderworkspace):
            known_raster = os.path.join(self.folderworkspace, 'subfolder', '_temp_test_1.tif')
            temp_raster = gis.create_temp_filename(os.path.join('subfolder', 'test'), filetype='raster', num=1)
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, 'subfolder', '_temp_test_12.shp')
            temp_shape = gis.create_temp_filename(os.path.join('subfolder','test'), filetype='shape', num=12)
            nt.assert_equal(temp_shape, known_shape)

    def test_folderworkspace_barefile(self):
        with gis.WorkSpace(self.folderworkspace):
            known_raster = os.path.join(self.folderworkspace, '_temp_test.tif')
            temp_raster = gis.create_temp_filename('test', filetype='raster')
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, '_temp_test.shp')
            temp_shape = gis.create_temp_filename('test', filetype='shape')
            nt.assert_equal(temp_shape, known_shape)

    def test_folderworkspace_barefile_with_num(self):
        with gis.WorkSpace(self.folderworkspace):
            known_raster = os.path.join(self.folderworkspace, '_temp_test_14.tif')
            temp_raster = gis.create_temp_filename('test', filetype='raster', num=14)
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, '_temp_test_3.shp')
            temp_shape = gis.create_temp_filename('test', filetype='shape', num=3)
            nt.assert_equal(temp_shape, known_shape)

    def test_geodb_barefile(self):
        with gis.WorkSpace(self.geodbworkspace):
            known_raster = os.path.join(self.geodbworkspace, '_temp_test')
            temp_raster = gis.create_temp_filename('test', filetype='raster')
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.geodbworkspace, '_temp_test')
            temp_shape = gis.create_temp_filename('test', filetype='shape')
            nt.assert_equal(temp_shape, known_shape)

    def test_geodb_barefile_with_num(self):
        with gis.WorkSpace(self.geodbworkspace):
            known_raster = os.path.join(self.geodbworkspace, '_temp_test_7')
            temp_raster = gis.create_temp_filename('test', filetype='raster', num=7)
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.geodbworkspace, '_temp_test_22')
            temp_shape = gis.create_temp_filename('test', filetype='shape', num=22)
            nt.assert_equal(temp_shape, known_shape)

    def test_geodb_as_subfolder(self):
        with gis.WorkSpace(self.folderworkspace):
            filename = os.path.join(self.geodbworkspace, 'test')
            known_raster = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test')
            temp_raster = gis.create_temp_filename(filename, filetype='raster')
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test')
            temp_shape = gis.create_temp_filename(filename, filetype='shape')
            nt.assert_equal(temp_shape, known_shape)

    def test_geodb_as_subfolder_with_num(self):
        with gis.WorkSpace(self.folderworkspace):
            filename = os.path.join(self.geodbworkspace, 'test')
            known_raster = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test_5')
            temp_raster = gis.create_temp_filename(filename, filetype='raster', num=5)
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test_99')
            temp_shape = gis.create_temp_filename(filename, filetype='shape', num=99)
            nt.assert_equal(temp_shape, known_shape)

    def test_with_extension_geodb(self):
        with gis.WorkSpace(self.folderworkspace):
            filename = os.path.join(self.geodbworkspace, 'test')
            known_raster = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test')
            temp_raster = gis.create_temp_filename(filename + '.tif', filetype='raster')
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test')
            temp_shape = gis.create_temp_filename(filename + '.tif', filetype='shape')
            nt.assert_equal(temp_shape, known_shape)

    def test_with_extension_geodb_with_num(self):
        with gis.WorkSpace(self.folderworkspace):
            filename = os.path.join(self.geodbworkspace, 'test')
            known_raster = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test_2000')
            temp_raster = gis.create_temp_filename(filename + '.tif', filetype='raster', num=2000)
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, self.geodbworkspace, '_temp_test_999')
            temp_shape = gis.create_temp_filename(filename + '.tif', filetype='shape', num=999)
            nt.assert_equal(temp_shape, known_shape)

    def test_with_extension_folder(self):
        with gis.WorkSpace(self.folderworkspace):
            filename = 'test'
            known_raster = os.path.join(self.folderworkspace, '_temp_test.tif')
            temp_raster = gis.create_temp_filename(filename + '.tif', filetype='raster')
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, '_temp_test.shp')
            temp_shape = gis.create_temp_filename(filename + '.shp', filetype='shape')
            nt.assert_equal(temp_shape, known_shape)

    def test_with_extension_folder_with_num(self):
        with gis.WorkSpace(self.folderworkspace):
            filename = 'test'
            known_raster = os.path.join(self.folderworkspace, '_temp_test_4.tif')
            temp_raster = gis.create_temp_filename(filename + '.tif', filetype='raster', num=4)
            nt.assert_equal(temp_raster, known_raster)

            known_shape = os.path.join(self.folderworkspace, '_temp_test_4.shp')
            temp_shape = gis.create_temp_filename(filename + '.shp', filetype='shape', num=4)
            nt.assert_equal(temp_shape, known_shape)


class Test_check_fields(object):
    table = resource_filename("propagator.testing.check_fields", "test_file.shp")

    def test_should_exist_uni(self):
        gis.check_fields(self.table, "Id", should_exist=True)

    def test_should_exist_multi(self):
        gis.check_fields(self.table, "Id", "existing", should_exist=True)

    def test_should_exist_multi_witharea(self):
        gis.check_fields(self.table, "Id", "existing", "SHAPE@AREA", should_exist=True)

    @nt.raises(ValueError)
    def test_should_exist_bad_vals(self):
        gis.check_fields(self.table, "Id", "existing", "JUNK", "GARBAGE", should_exist=True)

    def test_should_not_exist_uni(self):
        gis.check_fields(self.table, "NEWFIELD", should_exist=False)

    def test_should_not_exist_multi(self):
        gis.check_fields(self.table, "NEWFIELD", "YANFIELD", should_exist=False)

    def test_should_not_exist_multi_witharea(self):
        gis.check_fields(self.table, "NEWFIELD", "YANFIELD", "SHAPE@AREA", should_exist=False)

    @nt.raises(ValueError)
    def test_should_not_exist_bad_vals(self):
        gis.check_fields(self.table, "NEWFIELD", "YANFIELD", "existing", should_exist=False)


def test_result_to_raster():
    mockResult = mock.Mock(spec=arcpy.Result)
    mockRaster = mock.Mock(spec=arcpy.Raster)
    with mock.patch('arcpy.Raster', mockRaster):
        raster = gis.result_to_raster(mockResult)
        mockResult.getOutput.assert_called_once_with(0)


def test_result_to_Layer():
    mockResult = mock.Mock(spec=arcpy.Result)
    mockLayer = mock.Mock(spec=arcpy.mapping.Layer)
    with mock.patch('arcpy.mapping.Layer', mockLayer):
        layer = gis.result_to_layer(mockResult)
        mockResult.getOutput.assert_called_once_with(0)


class Test_rasters_to_arrays(object):
    def setup(self):
        from numpy import nan
        self.known_array1 = numpy.array([
            [ 0.0,  1.0,  2.0,  3.0,  4.0],
            [ 5.0,  6.0,  7.0,  8.0,  9.0],
            [10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0]
        ])

        self.known_array2 = numpy.array([
            [nan,  10.0,  20.0,  30.0,  40.0],
            [nan,  60.0,  70.0,  80.0,  90.0],
            [nan, 110.0, 120.0, 130.0, 140.0],
            [nan, 160.0, 170.0, 180.0, 190.0]
        ])

        self.known_array3 = numpy.array([
            [  00,  100,  200,  300,  400],
            [ 500,  600,  700,  800,  900],
            [1000, 1100, 1200, 1300, 1400],
            [1500, 1600, 1700, 1800, 1900]
        ])

        self.rasterfile1 = resource_filename("propagator.testing.rasters_to_arrays", 'test_raster1')
        self.rasterfile2 = resource_filename("propagator.testing.rasters_to_arrays", 'test_raster2')
        self.rasterfile3 = resource_filename("propagator.testing.rasters_to_arrays", 'test_raster3')

    def test_one_raster(self):
        array = gis.rasters_to_arrays(self.rasterfile1)
        nt.assert_true(isinstance(array, list))
        nt.assert_equal(len(array), 1)
        nptest.assert_array_almost_equal(array[0], self.known_array1)

    def test_one_raster_squeezed(self):
        array = gis.rasters_to_arrays(self.rasterfile1, squeeze=True)
        nt.assert_true(isinstance(array, numpy.ndarray))
        nptest.assert_array_almost_equal(array, self.known_array1)

    def test_with_missing_values_squeeze(self):
        array = gis.rasters_to_arrays(self.rasterfile2, squeeze=True)
        nt.assert_true(isinstance(array, numpy.ndarray))
        nptest.assert_array_almost_equal(array, self.known_array2)

    def test_int_array(self):
        array = gis.rasters_to_arrays(self.rasterfile3, squeeze=True)
        nt.assert_true(isinstance(array, numpy.ndarray))
        nptest.assert_array_almost_equal(array, self.known_array3)

    def test_multiple_args(self):
        arrays = gis.rasters_to_arrays(
            self.rasterfile1,
            self.rasterfile2,
            self.rasterfile3,
            squeeze=True
        )

        nt.assert_true(isinstance(arrays, list))
        nt.assert_equal(len(arrays), 3)

        for a, kn in zip(arrays, [self.known_array1, self.known_array2, self.known_array3]):
            nt.assert_true(isinstance(a, numpy.ndarray))
            nptest.assert_array_almost_equal(a, kn)


def test_array_to_raster():
    template_file = resource_filename("propagator.testing.array_to_raster", 'test_raster2')
    template = arcpy.Raster(template_file)
    array = numpy.arange(5, 25).reshape(4, 5).astype(float)

    raster = gis.array_to_raster(array, template)
    nt.assert_true(isinstance(raster, arcpy.Raster))
    nt.assert_true(raster.extent.equals(template.extent))
    nt.assert_equal(raster.meanCellWidth, template.meanCellWidth)
    nt.assert_equal(raster.meanCellHeight, template.meanCellHeight)


class Test_load_data(object):
    rasterpath = resource_filename("propagator.testing.load_data", 'test_dem.tif')
    vectorpath = resource_filename("propagator.testing.load_data", 'test_wetlands.shp')

    @nt.raises(ValueError)
    def test_bad_datatype(self):
        gis.load_data(self.rasterpath, 'JUNK')

    @nt.raises(ValueError)
    def test_datapath_doesnt_exist(self):
        gis.load_data('junk.shp', 'grid')

    @nt.raises(ValueError)
    def test_datapath_bad_value(self):
        gis.load_data(12345, 'grid')

    @nt.raises(ValueError)
    def test_vector_as_grid_should_fail(self):
        x = gis.load_data(self.vectorpath, 'grid')

    @nt.raises(ValueError)
    def test_vector_as_raster_should_fail(self):
        x = gis.load_data(self.vectorpath, 'raster')

    def test_raster_as_raster(self):
        x = gis.load_data(self.rasterpath, 'raster')
        nt.assert_true(isinstance(x, arcpy.Raster))

    def test_raster_as_grid_with_caps(self):
        x = gis.load_data(self.rasterpath, 'gRId')
        nt.assert_true(isinstance(x, arcpy.Raster))

    def test_raster_as_layer_not_greedy(self):
        x = gis.load_data(self.rasterpath, 'layer', greedyRasters=False)
        nt.assert_true(isinstance(x, arcpy.mapping.Layer))

    def test_raster_as_layer_greedy(self):
        x = gis.load_data(self.rasterpath, 'layer')
        nt.assert_true(isinstance(x, arcpy.Raster))

    def test_vector_as_shape(self):
        x = gis.load_data(self.vectorpath, 'shape')
        nt.assert_true(isinstance(x, arcpy.mapping.Layer))

    def test_vector_as_layer_with_caps(self):
        x = gis.load_data(self.vectorpath, 'LAyeR')
        nt.assert_true(isinstance(x, arcpy.mapping.Layer))

    def test_already_a_layer(self):
        lyr = arcpy.mapping.Layer(self.vectorpath)
        x = gis.load_data(lyr, 'layer')
        nt.assert_equal(x, lyr)

    def test_already_a_raster(self):
        raster = arcpy.Raster(self.rasterpath)
        x = gis.load_data(raster, 'raster')
        nt.assert_true(isinstance(x, arcpy.Raster))

        nptest.assert_array_almost_equal(*gis.rasters_to_arrays(x, raster))


class _polygons_to_raster_mixin(object):
    testfile = resource_filename("propagator.testing.polygons_to_raster", "test_zones.shp")
    known_values = numpy.array([-999, 16, 150])

    @nptest.dec.skipif(not pptest.has_spatial)
    def test_process(self):
        raster = gis.polygons_to_raster(self.testfile, "GeoID", **self.kwargs)
        nt.assert_true(isinstance(raster, arcpy.Raster))

        array = gis.rasters_to_arrays(raster, squeeze=True)
        arcpy.management.Delete(raster)

        flat_arr = array.flatten()
        bins = numpy.bincount(flat_arr[flat_arr > 0])
        nptest.assert_array_almost_equal(numpy.unique(array), self.known_values)
        nptest.assert_array_almost_equal(bins[bins > 0], self.known_counts)
        nt.assert_tuple_equal(array.shape, self.known_shape)


class Test_polygons_to_raster_default(_polygons_to_raster_mixin):
    def setup(self):
        self.kwargs = {}
        self.known_shape = (854, 661)
        self.known_counts = numpy.array([95274, 36674])


class Test_polygons_to_raster_x02(_polygons_to_raster_mixin):
    def setup(self):
        self.kwargs = {'cellsize': 2}
        self.known_shape = (1709, 1322)
        self.known_counts = numpy.array([381211, 146710])


class Test_polygons_to_raster_x08(_polygons_to_raster_mixin):
    def setup(self):
        self.kwargs = {'cellsize': 8}
        self.known_shape = (427, 330)
        self.known_counts = numpy.array([23828,  9172])

    @nptest.dec.skipif(not pptest.has_spatial)
    def test_actual_arrays(self):
        known_raster_file = resource_filename("propagator.testing.polygons_to_raster", "test_zones_raster.tif")
        known_raster = gis.load_data(known_raster_file, 'raster')
        raster = gis.polygons_to_raster(self.testfile, "GeoID", **self.kwargs)
        arrays = gis.rasters_to_arrays(raster, known_raster)
        arcpy.management.Delete(raster)

        nptest.assert_array_almost_equal(*arrays)


class Test_polygons_to_raster_x16(_polygons_to_raster_mixin):
    def setup(self):
        self.kwargs = {'cellsize': 16}
        self.known_shape = (214, 165)
        self.known_counts = numpy.array([5953, 2288])


def test_clip_dem_to_zones():
    demfile = resource_filename("propagator.testing.clip_dem_to_zones", 'test_dem.tif')
    zonefile = resource_filename("propagator.testing.clip_dem_to_zones", "test_zones_raster_small.tif")
    raster = gis.clip_dem_to_zones(demfile, zonefile)

    zone_r = gis.load_data(zonefile, 'raster')

    arrays = gis.rasters_to_arrays(raster, zone_r)

    dem_a, zone_a = arrays[0], arrays[1]
    arcpy.management.Delete(raster)

    nt.assert_true(isinstance(raster, arcpy.Raster))

    known_shape = (146, 172)
    nt.assert_tuple_equal(dem_a.shape, zone_a.shape)


@nptest.dec.skipif(not pptest.has_fiona)
def test_raster_to_polygons():
    zonefile = resource_filename("propagator.testing.raster_to_polygons", "input_raster_to_polygon.tif")
    knownfile = resource_filename("propagator.testing.raster_to_polygons", "known_polygons_from_raster_1.shp")
    testfile = resource_filename("propagator.testing.raster_to_polygons", "test_polygons_from_raster_1.shp")

    with gis.OverwriteState(True):
        zones = gis.load_data(zonefile, 'raster')
        known = gis.load_data(knownfile, 'layer')
        test = gis.raster_to_polygons(zones, testfile)

    pptest.assert_shapefiles_are_close(test.dataSource, known.dataSource)
    gis.cleanup_temp_results(testfile)


@nptest.dec.skipif(not pptest.has_fiona)
def test_raster_to_polygons_with_new_field():
    zonefile = resource_filename("propagator.testing.raster_to_polygons", "input_raster_to_polygon.tif")
    knownfile = resource_filename("propagator.testing.raster_to_polygons", "known_polygons_from_raster_2.shp")
    testfile = resource_filename("propagator.testing.raster_to_polygons", "test_polygons_from_raster_2.shp")

    with gis.OverwriteState(True):
        zones = gis.load_data(zonefile, 'raster')
        known = gis.load_data(knownfile, 'layer')
        test = gis.raster_to_polygons(zones, testfile, newfield="GeoID")

    pptest.assert_shapefiles_are_close(test.dataSource, known.dataSource)
    gis.cleanup_temp_results(testfile)


@nptest.dec.skipif(not pptest.has_fiona)
def test_aggregate_polygons():
    inputfile = resource_filename("propagator.testing.aggregate_polygons", "input_polygons_from_raster.shp")
    knownfile = resource_filename("propagator.testing.aggregate_polygons", "known_dissolved_polygons.shp")
    testfile = resource_filename("propagator.testing.aggregate_polygons", "test_dissolved_polygons.shp")

    with gis.OverwriteState(True):
        raw = gis.load_data(inputfile, 'layer')
        known = gis.load_data(knownfile, 'layer')
        test = gis.aggregate_polygons(raw, "gridcode", testfile)

    pptest.assert_shapefiles_are_close(test.dataSource, known.dataSource)

    gis.cleanup_temp_results(testfile)


class Test_add_field_with_value(object):
    def setup(self):
        self.shapefile = resource_filename("propagator.testing.add_field_with_value", 'field_adder.shp')
        self.fields_added = ["_text", "_unicode", "_int", "_float", '_no_valstr', '_no_valnum']

    def teardown(self):
        field_names = [f.name for f in arcpy.ListFields(self.shapefile)]
        for field in self.fields_added:
            if field in field_names:
                arcpy.management.DeleteField(self.shapefile, field)

    def test_float(self):
        name = "_float"
        gis.add_field_with_value(self.shapefile, name,
                                   field_value=5.0)
        nt.assert_true(name in [f.name for f in arcpy.ListFields(self.shapefile)])

        newfield = arcpy.ListFields(self.shapefile, name)[0]
        nt.assert_equal(newfield.type, u'Double')

    def test_int(self):
        name = "_int"
        gis.add_field_with_value(self.shapefile, name,
                                   field_value=5)
        nt.assert_true(name in [f.name for f in arcpy.ListFields(self.shapefile)])

        newfield = arcpy.ListFields(self.shapefile, name)[0]
        nt.assert_equal(newfield.type, u'Integer')

    def test_string(self):
        name = "_text"
        gis.add_field_with_value(self.shapefile, name,
                                   field_value="example_value",
                                   field_length=15)

        nt.assert_true(name in [f.name for f in arcpy.ListFields(self.shapefile)])

        newfield = arcpy.ListFields(self.shapefile, name)[0]
        nt.assert_equal(newfield.type, u'String')
        nt.assert_true(newfield.length, 15)

    def test_unicode(self):
        name = "_unicode"
        gis.add_field_with_value(self.shapefile, name,
                                   field_value=u"example_value",
                                   field_length=15)

        nt.assert_true(name in [f.name for f in arcpy.ListFields(self.shapefile)])

        newfield = arcpy.ListFields(self.shapefile, name)[0]
        nt.assert_equal(newfield.type, u'String')
        nt.assert_true(newfield.length, 15)

    def test_no_value_string(self):
        name = "_no_valstr"
        gis.add_field_with_value(self.shapefile, name,
                                   field_type='TEXT',
                                   field_length=15)

        nt.assert_true(name in [f.name for f in arcpy.ListFields(self.shapefile)])

        newfield = arcpy.ListFields(self.shapefile, name)[0]
        nt.assert_equal(newfield.type, u'String')
        nt.assert_true(newfield.length, 15)

    def test_no_value_number(self):
        name = "_no_valnum"
        gis.add_field_with_value(self.shapefile, name,
                                   field_type='DOUBLE')

        nt.assert_true(name in [f.name for f in arcpy.ListFields(self.shapefile)])

        newfield = arcpy.ListFields(self.shapefile, name)[0]
        nt.assert_equal(newfield.type, u'Double')

    @nt.raises(ValueError)
    def test_no_value_no_field_type(self):
        gis.add_field_with_value(self.shapefile, "_willfail")

    @nt.raises(ValueError)
    def test_overwrite_existing_no(self):
        gis.add_field_with_value(self.shapefile, "existing")

    def test_overwrite_existing_yes(self):
        gis.add_field_with_value(self.shapefile, "existing",
                                   overwrite=True,
                                   field_type="LONG")


class Test_cleanup_temp_results(object):
    def setup(self):
        self.workspace = os.path.abspath(resource_filename('propagator.testing', 'cleanup_temp_results'))
        self.template_file = resource_filename('propagator.testing.cleanup_temp_results', 'test_dem.tif')
        self.template = gis.load_data(self.template_file, 'raster')

        raster1 = gis.array_to_raster(numpy.random.normal(size=(30, 30)), self.template)
        raster2 = gis.array_to_raster(numpy.random.normal(size=(60, 60)), self.template)

        self.name1 = 'temp_1.tif'
        self.name2 = 'temp_2.tif'

        self.path1 = os.path.join(self.workspace, self.name1)
        self.path2 = os.path.join(self.workspace, self.name2)

        with gis.OverwriteState(True), gis.WorkSpace(self.workspace):
            raster1.save(self.path1)
            raster2.save(self.path2)

    @nt.nottest
    def check_outcome(self):
        nt.assert_false(os.path.exists(os.path.join(self.workspace, 'temp_1.tif')))
        nt.assert_false(os.path.exists(os.path.join(self.workspace, 'temp_2.tif')))

    def test_with_names_in_a_workspace(self):
        with gis.WorkSpace(self.workspace):
            gis.cleanup_temp_results(self.name1, self.name2)
            self.check_outcome()

    def test_with_paths_absolute(self):
        gis.cleanup_temp_results(self.path1, self.path2)
        self.check_outcome()

    def test_with_rasters(self):
        with gis.WorkSpace(self.workspace):
            raster1 = gis.load_data(self.path1, 'raster')
            raster2 = gis.load_data(self.path2, 'raster')
            gis.cleanup_temp_results(raster1, raster2)
            self.check_outcome()

    def test_with_results(self):
        with gis.WorkSpace(self.workspace):
            res1 = arcpy.Result(toolname='Clip_management')
            res2 = arcpy.Result(toolname='Clip_management')
            with mock.patch.object(res1, 'getOutput', return_value='temp_1.tif'), \
                 mock.patch.object(res2, 'getOutput', return_value='temp_2.tif'):
                gis.cleanup_temp_results(res1, res2)
                self.check_outcome()

    def test_with_layers(self):
        with gis.WorkSpace(self.workspace):
            lyr1 = gis.load_data('temp_1.tif', 'layer', greedyRasters=False)
            lyr2 = gis.load_data('temp_2.tif', 'layer', greedyRasters=False)
            gis.cleanup_temp_results(lyr1, lyr2)
            self.check_outcome()

    @nt.raises(ValueError)
    def test_with_bad_input(self):
        gis.cleanup_temp_results(1, 2, ['a', 'b', 'c'])

    def teardown(self):
        with gis.WorkSpace(self.workspace):
            gis.cleanup_temp_results('temp_1.tif', 'temp_2.tif')


@nptest.dec.skipif(not pptest.has_fiona)
def test_intersect_polygon_layers():
    input1_file = resource_filename("propagator.testing.intersect_polygons", "intersect_input1.shp")
    input2_file = resource_filename("propagator.testing.intersect_polygons", "intersect_input2.shp")
    known_file = resource_filename("propagator.testing.intersect_polygons", "intersect_known.shp")
    output_file = resource_filename("propagator.testing.intersect_polygons", "intersect_output.shp")

    with gis.OverwriteState(True):
        output = gis.intersect_polygon_layers(
            output_file,
            input1_file,
            input2_file,
        )

    nt.assert_true(isinstance(output, arcpy.mapping.Layer))
    pptest.assert_shapefiles_are_close(output_file, known_file)

    gis.cleanup_temp_results(output)


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

    result = gis.load_attribute_table(path, 'CatchID', 'DwnCatchID', 'Watershed')
    nptest.assert_array_equal(result[:5], expected_top_five)


class Test_groupby_and_aggregate():
    known_counts = {16.0: 32, 150.0: 2}
    buildings = resource_filename("propagator.testing.groupby_and_aggregate", "flooded_buildings.shp")
    group_col = 'GeoID'
    count_col = 'STRUCT_ID'
    area_op = 'SHAPE@AREA'

    areas = resource_filename("propagator.testing.groupby_and_aggregate", "intersect_input1.shp")
    known_areas = {2: 1327042.1024, 7: 1355433.0192, 12: 1054529.2882}

    def test_defaults(self):
        counts = gis.groupby_and_aggregate(
            self.buildings,
            self.group_col,
            self.count_col,
            aggfxn=None
        )

        nt.assert_dict_equal(counts, self.known_counts)

    def test_area(self):
        areadict = gis.groupby_and_aggregate(
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
        counts = gis.groupby_and_aggregate(
            self.buildings,
            "JUNK",
            self.count_col
        )

    @nt.raises(ValueError)
    def test_bad_count_col(self):
        counts = gis.groupby_and_aggregate(
            self.buildings,
            self.group_col,
            "JUNK"
        )


@nt.raises(NotImplementedError)
def test_rename_column():
    layer = resource_filename("propagator.testing.rename_column", "rename_col.dbf")
    oldname = "existing"
    newname = "exists"

    #layer = gis.load_data(inputfile, "layer")

    gis.rename_column(layer, oldname, newname)
    gis.check_fields(layer, newname, should_exist=True)
    gis.check_fields(layer, oldname, should_exist=False)

    gis.rename_column(layer, newname, oldname)
    gis.check_fields(layer, newname, should_exist=False)
    gis.check_fields(layer, oldname, should_exist=True)


class Test_populate_field(object):
    def setup(self):
        self.shapefile = resource_filename("propagator.testing.populate_field", 'populate_field.shp')
        self.field_added = "newfield"

    def teardown(self):
        arcpy.management.DeleteField(self.shapefile, self.field_added)

    def test_with_dictionary(self):
        value_dict = {n: n for n in range(7)}
        value_fxn = lambda row: value_dict.get(row[0], -1)
        gis.add_field_with_value(self.shapefile, self.field_added, field_type="LONG")

        gis.populate_field(
            self.shapefile,
            lambda row: value_dict.get(row[0], -1),
            self.field_added,
            "FID"
        )

        with arcpy.da.SearchCursor(self.shapefile, [self.field_added, "FID"]) as cur:
            for row in cur:
                nt.assert_equal(row[0], row[1])

    def test_with_general_function(self):
        gis.add_field_with_value(self.shapefile, self.field_added, field_type="LONG")
        gis.populate_field(
            self.shapefile,
            lambda row: row[0]**2,
            self.field_added,
            "FID"
        )

        with arcpy.da.SearchCursor(self.shapefile, [self.field_added, "FID"]) as cur:
            for row in cur:
                nt.assert_equal(row[0], row[1] ** 2)


def test_copy_layer():
    with mock.patch.object(arcpy.management, 'Copy') as _copy:
        in_data = 'input'
        out_data = 'new_copy'

        result = gis.copy_layer(in_data, out_data)
        _copy.assert_called_once_with(in_data=in_data, out_data=out_data)
        nt.assert_equal(result, out_data)


def test_intersect_layers():
    ws = resource_filename('propagator.testing', 'intersect_layers')
    with gis.OverwriteState(True), gis.WorkSpace(ws):
        gis.intersect_layers(
            ['subcatchments.shp', 'monitoring_locations.shp'],
            'test.shp',
        )

    pptest.assert_shapefiles_are_close(
        os.path.join(ws, 'expected.shp'),
        os.path.join(ws, 'test.shp'),
    )

    gis.cleanup_temp_results(os.path.join(ws, 'test.shp'))


@nptest.dec.skipif(not pptest.has_fiona)
def test_concat_results():
    known = resource_filename('propagator.testing.concat_results', 'known.shp')
    with gis.OverwriteState(True):
        test = gis.concat_results(
            resource_filename('propagator.testing.concat_results', 'result.shp'),
            resource_filename('propagator.testing.concat_results', 'input1.shp'),
            resource_filename('propagator.testing.concat_results', 'input2.shp')
        )

    nt.assert_true(isinstance(test, arcpy.mapping.Layer))
    pptest.assert_shapefiles_are_close(test.dataSource, known)

    gis.cleanup_temp_results(test)


@nptest.dec.skipif(not pptest.has_fiona)
def test_spatial_join():
    known = resource_filename('propagator.testing.spatial_join', 'merge_result.shp')
    left = resource_filename('propagator.testing.spatial_join', 'merge_baseline.shp')
    right = resource_filename('propagator.testing.spatial_join', 'merge_join.shp')
    outputfile = resource_filename('propagator.testing.spatial_join', 'merge_result.shp')
    with gis.OverwriteState(True):
        test = gis.spatial_join(left=left, right=right, outputfile=outputfile)

    nt.assert_equal(test, outputfile)
    pptest.assert_shapefiles_are_close(test, known)

    gis.cleanup_temp_results(test)


@nptest.dec.skipif(not pptest.has_fiona)
def test_update_attribute_table():
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
    gis.update_attribute_table(testpath, new_attributes, 'id', 'Cu', 'Pb')

    pptest.assert_shapefiles_are_close(testpath, expected)
    gis.cleanup_temp_results(testpath)


def test_get_field_names():
    expected = [u'FID', u'Shape', u'Station', u'Latitude', u'Longitude']
    layer = resource_filename('propagator.testing.get_field_names', 'input.shp')
    result = gis.get_field_names(layer)
    nt.assert_list_equal(result, expected)


def test_count_features():
    layer = resource_filename('propagator.testing.count_features', 'monitoring_locations.shp')
    nt.assert_equal(gis.count_features(layer), 14)


def test_query_layer():
    with mock.patch.object(arcpy.analysis, 'Select') as query:
        in_data = 'input'
        out_data = 'new_copy'
        sql = "fake SQL string"

        result = gis.query_layer(in_data, out_data, sql)
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

        result = gis.delete_columns(in_data, 'ThisColumn', 'ThatColumn', 'AnotherColumn')
        query.assert_called_once_with(in_data, expected_col_string )
        nt.assert_equal(result, in_data)

