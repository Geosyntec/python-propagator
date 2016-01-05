import os
from pkg_resources import resource_filename

import arcpy
import numpy

import nose.tools as nt
import numpy.testing as nptest
import propagator.testing as pptest
import mock

import propagator
from propagator import utils, toolbox


@nt.nottest
class MockResult(object):
    @staticmethod
    def getOutput(index):
        if index == 0:
            return resource_filename('propagator.testing.input', 'test_zones.shp')


@nt.nottest
class MockParam(object):
    def __init__(self, name, value, multival):
        self.name = name
        self.valueAsText = value
        self.multiValue = multival


@nt.nottest
def mock_status(*args, **kwargs):
    pass


def test_propagate():
    ws = resource_filename('propagator.testing', 'tbx_propagate')
    columns = ['Dry_B', 'Dry_M', 'Dry_N', 'Wet_B', 'Wet_M', 'Wet_N']
    with utils.WorkSpace(ws), utils.OverwriteState(True):
        output_layer = propagator.toolbox.propagate(
            subcatchments='subcatchments.shp',
            monitoring_locations='monitoring_locations.shp',
            id_col='CID',
            ds_col='DS_CID',
            value_columns=columns,
            output_path='test.shp'
        )

    pptest.assert_shapefiles_are_close(
        os.path.join(ws, 'expected.shp'),
        os.path.join(ws, output_layer),
    )

    utils.cleanup_temp_results(os.path.join(ws, output_layer))

class BaseToolboxChecker_Mixin(object):
    mockMap = mock.Mock(spec=utils.EasyMapDoc)
    mockLayer = mock.Mock(spec=arcpy.mapping.Layer)
    mockUtils = mock.Mock(spec=utils)
    mxd = resource_filename('propagator.testing.toolbox', 'test.mxd')
    simple_shp = resource_filename('propagator.testing.toolbox', 'ZOI.shp')
    outfile = 'output.shp'

    parameters = [
        MockParam('workspace', 'path/to/the/workspace.gdb', False),
        MockParam('ID_column', 'GeoID', False),
        MockParam('downstream_ID_column', 'DS_GeoID', False),
        MockParam('value_columns', 'SHAPE_AREA;SHAPE_LENGTH;NAME', True),
    ]

    parameter_dict = {
        'workspace': 'path/to/the/workspace.gdb',
        'ID_column': 'GeoID',
        'downstream_ID_column': 'DS_GeoID',
        'value_columns': ['SHAPE_AREA', 'SHAPE_LENGTH', 'NAME'],
    }

    def test_isLicensed(self):
        # every toolbox should be always licensed!
        nt.assert_true(self.tbx.isLicensed())

    def test_getParameterInfo(self):
        with mock.patch.object(self.tbx, '_params_as_list') as _pal:
            self.tbx.getParameterInfo()
            _pal.assert_called_once_with()

    def test_execute(self):
        messages = ['message1', 'message2']
        with mock.patch.object(self.tbx, 'analyze') as analyze:
            self.tbx.execute(self.parameters, messages)
            analyze.assert_called_once_with(**self.parameter_dict)

    def test__set_parameter_dependency_single(self):
        self.tbx._set_parameter_dependency(
            self.tbx.ID_column,
            self.tbx.subcatchments
        )

        nt.assert_list_equal(
            self.tbx.ID_column.parameterDependencies,
            [self.tbx.subcatchments.name]
        )

    def test__set_parameter_dependency_many(self):
        self.tbx._set_parameter_dependency(
            self.tbx.ID_column,
            self.tbx.workspace,
            self.tbx.subcatchments,
        )

        nt.assert_list_equal(
            self.tbx.ID_column.parameterDependencies,
            [self.tbx.workspace.name, self.tbx.subcatchments.name]
        )

    def test__show_header(self):
        header = self.tbx._show_header('TEST MESSAGE', verbose=False)
        expected = '\nTEST MESSAGE\n------------'
        nt.assert_equal(header, expected)

    def test__add_to_map(self):
        with mock.patch.object(utils.EasyMapDoc, 'add_layer') as add_layer:
            ezmd = self.tbx._add_to_map(self.simple_shp, mxd=self.mxd)
            nt.assert_true(isinstance(ezmd, utils.EasyMapDoc))
            add_layer.assert_called_once_with(self.simple_shp)

    def test__get_parameter_values(self):
        param_vals = self.tbx._get_parameter_values(self.parameters)
        expected = {
            'workspace': 'path/to/the/workspace.gdb',
            'ID_column': 'GeoID',
            'downstream_ID_column': 'DS_GeoID',
            'value_columns': ['SHAPE_AREA', 'SHAPE_LENGTH', 'NAME'],        }
        nt.assert_dict_equal(param_vals, expected)

    def test_workspace(self):
        nt.assert_true(hasattr(self.tbx, 'workspace'))
        nt.assert_true(isinstance(self.tbx.workspace, arcpy.Parameter))
        nt.assert_equal(self.tbx.workspace.parameterType, 'Required')
        nt.assert_equal(self.tbx.workspace.direction, 'Input')
        nt.assert_equal(self.tbx.workspace.datatype, 'Workspace')
        nt.assert_equal(self.tbx.workspace.name, 'workspace')
        nt.assert_list_equal(self.tbx.workspace.parameterDependencies, [])

    def test_subcatchment(self):
        nt.assert_true(hasattr(self.tbx, 'subcatchments'))
        nt.assert_true(isinstance(self.tbx.subcatchments, arcpy.Parameter))
        nt.assert_equal(self.tbx.subcatchments.parameterType, 'Required')
        nt.assert_equal(self.tbx.subcatchments.direction, 'Input')
        nt.assert_equal(self.tbx.subcatchments.datatype, 'Feature Class')
        nt.assert_equal(self.tbx.subcatchments.name, 'subcatchments')
        nt.assert_list_equal(self.tbx.subcatchments.parameterDependencies, ['workspace'])
        nt.assert_false(self.tbx.subcatchments.multiValue)

    def test_ID_column(self):
        nt.assert_true(hasattr(self.tbx, 'ID_column'))
        nt.assert_true(isinstance(self.tbx.ID_column, arcpy.Parameter))
        nt.assert_equal(self.tbx.ID_column.parameterType, 'Required')
        nt.assert_equal(self.tbx.ID_column.direction, 'Input')
        nt.assert_equal(self.tbx.ID_column.datatype, 'Field')
        nt.assert_equal(self.tbx.ID_column.name, 'ID_column')
        nt.assert_list_equal(self.tbx.ID_column.parameterDependencies, ['subcatchments'])
        nt.assert_false(self.tbx.ID_column.multiValue)

    def test_downstream_ID_column(self):
        nt.assert_true(hasattr(self.tbx, 'downstream_ID_column'))
        nt.assert_true(isinstance(self.tbx.downstream_ID_column, arcpy.Parameter))
        nt.assert_equal(self.tbx.downstream_ID_column.parameterType, 'Required')
        nt.assert_equal(self.tbx.downstream_ID_column.direction, 'Input')
        nt.assert_equal(self.tbx.downstream_ID_column.datatype, 'Field')
        nt.assert_equal(self.tbx.downstream_ID_column.name, 'downstream_ID_column')
        nt.assert_list_equal(self.tbx.downstream_ID_column.parameterDependencies, ['subcatchments'])
        nt.assert_false(self.tbx.downstream_ID_column.multiValue)

    def test_value_columns(self):
        nt.assert_true(hasattr(self.tbx, 'value_columns'))
        nt.assert_true(isinstance(self.tbx.value_columns, arcpy.Parameter))
        nt.assert_equal(self.tbx.value_columns.parameterType, 'Required')
        nt.assert_equal(self.tbx.value_columns.direction, 'Input')
        nt.assert_equal(self.tbx.value_columns.datatype, 'Field')
        nt.assert_equal(self.tbx.value_columns.name, 'value_columns')
        nt.assert_list_equal(self.tbx.value_columns.parameterDependencies, [self.value_col_dependency])
        nt.assert_true(self.tbx.value_columns.multiValue)

    def test_output_layer(self):
        nt.assert_true(hasattr(self.tbx, 'output_layer'))
        nt.assert_true(isinstance(self.tbx.output_layer, arcpy.Parameter))
        nt.assert_equal(self.tbx.output_layer.parameterType, 'Required')
        nt.assert_equal(self.tbx.output_layer.direction, 'Input')
        nt.assert_equal(self.tbx.output_layer.datatype, 'String')
        nt.assert_equal(self.tbx.output_layer.name, 'output_layer')
        nt.assert_list_equal(self.tbx.output_layer.parameterDependencies, [])
        nt.assert_false(self.tbx.output_layer.multiValue)

    def test_add_output_to_map(self):
        nt.assert_true(hasattr(self.tbx, 'add_output_to_map'))
        nt.assert_true(isinstance(self.tbx.add_output_to_map, arcpy.Parameter))
        nt.assert_equal(self.tbx.add_output_to_map.parameterType, 'Required')
        nt.assert_equal(self.tbx.add_output_to_map.direction, 'Input')
        nt.assert_equal(self.tbx.add_output_to_map.datatype, 'Boolean')
        nt.assert_equal(self.tbx.add_output_to_map.name, 'add_output_to_map')
        nt.assert_list_equal(self.tbx.add_output_to_map.parameterDependencies, [])
        nt.assert_false(self.tbx.add_output_to_map.multiValue)


@mock.patch('propagator.utils.misc._status', mock_status)
class Test_Propagator(BaseToolboxChecker_Mixin):
    def setup(self):
        self.tbx = toolbox.Propagator()
        self.main_execute_dir = 'propagator.testing.Propagator'
        self.main_execute_ws = resource_filename('propagator.testing', 'Propagator')
        self.value_col_dependency = 'monitoring_locations'

    def test_monitoring_locations(self):
        nt.assert_true(hasattr(self.tbx, 'monitoring_locations'))
        nt.assert_true(isinstance(self.tbx.monitoring_locations, arcpy.Parameter))
        nt.assert_equal(self.tbx.monitoring_locations.parameterType, 'Required')
        nt.assert_equal(self.tbx.monitoring_locations.direction, 'Input')
        nt.assert_equal(self.tbx.monitoring_locations.datatype, 'Feature Class')
        nt.assert_equal(self.tbx.monitoring_locations.name, 'monitoring_locations')
        nt.assert_list_equal(self.tbx.monitoring_locations.parameterDependencies, ['workspace'])
        nt.assert_false(self.tbx.monitoring_locations.multiValue)

    def test_params_as_list(self):
        params = self.tbx._params_as_list()
        names = [str(p.name) for p in params]
        known_names = [
            'workspace',
            'subcatchments',
            'ID_column',
            'downstream_ID_column',
            'monitoring_locations',
            'value_columns',
            'output_layer',
            'add_output_to_map',
        ]
        nt.assert_list_equal(names, known_names)

    @nptest.dec.skipif(not pptest.has_fiona)
    def test_analyze(self):
        tbx = toolbox.Propagator()
        ws = resource_filename('propagator.testing', 'tbx_propagate')
        columns = ['Dry_B', 'Dry_M', 'Dry_N', 'Wet_B', 'Wet_M', 'Wet_N']
        with mock.patch.object(toolbox.Propagator, '_add_to_map') as atm:
            output_layer = tbx.analyze(
                workspace=ws,
                overwrite=True,
                subcatchments='subcatchments.shp',
                ID_column='CID',
                downstream_ID_column='DS_CID',
                monitoring_locations='monitoring_locations.shp',
                value_columns=columns,
                output_layer='test.shp',
                add_output_to_map=True
            )

            pptest.assert_shapefiles_are_close(
                os.path.join(ws, 'expected.shp'),
                os.path.join(ws, output_layer),
            )

            utils.cleanup_temp_results(os.path.join(ws, output_layer))
            atm.assert_called_once_with(output_layer)


@mock.patch('propagator.utils.misc._status', mock_status)
class Test_Accumulator(BaseToolboxChecker_Mixin):
    def setup(self):
        self.tbx = toolbox.Accumulator()
        self.main_execute_dir = 'propagator.testing.Accumulator'
        self.main_execute_ws = resource_filename('propagator.testing', 'Accumulator')
        self.value_col_dependency = 'subcatchments'

    def test_streams(self):
        nt.assert_true(hasattr(self.tbx, 'streams'))
        nt.assert_true(isinstance(self.tbx.streams, arcpy.Parameter))
        nt.assert_equal(self.tbx.streams.parameterType, 'Required')
        nt.assert_equal(self.tbx.streams.direction, 'Input')
        nt.assert_equal(self.tbx.streams.datatype, 'Feature Class')
        nt.assert_equal(self.tbx.streams.name, 'streams')
        nt.assert_list_equal(self.tbx.streams.parameterDependencies, ['workspace'])
        nt.assert_false(self.tbx.streams.multiValue)

    def test_params_as_list(self):
        params = self.tbx._params_as_list()
        names = [str(p.name) for p in params]
        known_names = [
            'workspace',
            'subcatchments',
            'ID_column',
            'downstream_ID_column',
            'value_columns',
            'streams',
            'output_layer',
            'add_output_to_map',
        ]
        nt.assert_list_equal(names, known_names)


