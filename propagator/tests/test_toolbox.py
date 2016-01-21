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


class Test_propagate(object):
    def setup(self):
        self.ws = resource_filename('propagator.testing', 'tbx_propagate')
        self.columns = [
            ['Dry_B', 'Average'],
            ['Dry_M', 'Median'],
            ['Dry_N', 'Min'],
            ['Wet_B', 'Max'],
            ['Wet_M', 'Average'],
            ['Wet_N', 'Median'],
        ]
        self.subc_res = 'test_subcatchments.shp'
        self.stream_res = 'test_streams.shp'

    @nt.nottest
    def check(self, subc_res, stream_res, subc_exp, stream_exp):
        nt.assert_equal(subc_res, self.subc_res)
        nt.assert_equal(stream_res, self.stream_res)

        pptest.assert_shapefiles_are_close(
            os.path.join(self.ws, subc_exp),
            os.path.join(self.ws, self.subc_res),
        )

        pptest.assert_shapefiles_are_close(
            os.path.join(self.ws, stream_exp),
            os.path.join(self.ws, self.stream_res),
        )

    def teardown(self):
        utils.cleanup_temp_results(
            os.path.join(self.ws, self.subc_res),
            os.path.join(self.ws, self.stream_res)
        )

    @nptest.dec.skipif(not pptest.has_fiona)
    def test_baseline(self):
        with utils.WorkSpace(self.ws), utils.OverwriteState(True):
            subc_layer, stream_layer = propagator.toolbox.propagate(
                subcatchments='subcatchments.shp',
                monitoring_locations='monitoring_locations.shp',
                id_col='CID',
                ds_col='DS_CID',
                value_columns=self.columns,
                streams='streams.shp',
                output_path='test.shp'
            )

        self.check(subc_layer, stream_layer, 'expected_subc.shp', 'expected_streams.shp')

    @nptest.dec.skipif(not pptest.has_fiona)
    def test_filtered(self):
        stacol = 'StationTyp'
        with utils.WorkSpace(self.ws), utils.OverwriteState(True):
            subc_layer, stream_layer = propagator.toolbox.propagate(
                subcatchments='subcatchments.shp',
                id_col='CID',
                ds_col='DS_CID',
                monitoring_locations='monitoring_locations.shp',
                ml_filter=lambda row: row[stacol] in ['Channel', 'Outfall', 'Outfall, Coastal'],
                ml_filter_cols=stacol,
                value_columns=self.columns,
                streams='streams.shp',
                output_path='test.shp'
            )

        self.check(subc_layer, stream_layer, 'expected_filtered_subc.shp', 'expected_filtered_streams.shp')
        nt.assert_equal(subc_layer, 'test_subcatchments.shp')
        nt.assert_equal(stream_layer, 'test_streams.shp')


def test_accumulate():
    ws = resource_filename('propagator.testing', 'score_accumulator')
    with utils.WorkSpace(ws), utils.OverwriteState(True):
        results = toolbox.accumulate(
            subcatchments_layer='subcatchment_wq.shp',
            id_col='Catch_ID_a',
            ds_col='Dwn_Catch_',
            area_col='Area',
            imp_col='Imp',
            streams_layer='streams.shp',
            output_layer='output.shp',
        )

        pptest.assert_shapefiles_are_close(os.path.join(ws, 'expected_results.shp'),
                                           os.path.join(ws, results))

        utils.cleanup_temp_results(os.path.join(ws, results))


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

    parameter_value_dict = {
        'workspace': 'path/to/the/workspace.gdb',
        'ID_column': 'GeoID',
        'downstream_ID_column': 'DS_GeoID',
        'value_columns': ['SHAPE_AREA', 'SHAPE_LENGTH', 'NAME'],
    }

    parameter_dict = {
        'workspace': parameters[0],
        'ID_column': parameters[1],
        'downstream_ID_column': parameters[2],
        'value_columns': parameters[3],
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
            analyze.assert_called_once_with(**self.parameter_value_dict)

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

    def test__get_parameter_dict(self):
        param_dict = self.tbx._get_parameter_dict(self.parameters)
        nt.assert_dict_equal(param_dict, self.parameter_dict)

    def test__get_parameter_values(self):
        param_vals = self.tbx._get_parameter_values(self.parameters)
        expected = {
            'workspace': 'path/to/the/workspace.gdb',
            'ID_column': 'GeoID',
            'downstream_ID_column': 'DS_GeoID',
            'value_columns': ['SHAPE_AREA', 'SHAPE_LENGTH', 'NAME'],
        }
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

    def test_streams(self):
        nt.assert_true(hasattr(self.tbx, 'streams'))
        nt.assert_true(isinstance(self.tbx.streams, arcpy.Parameter))
        nt.assert_equal(self.tbx.streams.parameterType, 'Required')
        nt.assert_equal(self.tbx.streams.direction, 'Input')
        nt.assert_equal(self.tbx.streams.datatype, 'Feature Class')
        nt.assert_equal(self.tbx.streams.name, 'streams')
        nt.assert_list_equal(self.tbx.streams.parameterDependencies, ['workspace'])
        nt.assert_false(self.tbx.streams.multiValue)

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
        nt.assert_equal(self.tbx.add_output_to_map.parameterType, 'Optional')
        nt.assert_equal(self.tbx.add_output_to_map.direction, 'Input')
        nt.assert_equal(self.tbx.add_output_to_map.datatype, 'Boolean')
        nt.assert_equal(self.tbx.add_output_to_map.name, 'add_output_to_map')
        nt.assert_list_equal(self.tbx.add_output_to_map.parameterDependencies, [])
        nt.assert_false(self.tbx.add_output_to_map.multiValue)


@mock.patch('propagator.utils._status', mock_status)
class Test_Propagator(BaseToolboxChecker_Mixin):
    def setup(self):
        self.tbx = toolbox.Propagator()
        self.main_execute_dir = 'propagator.testing.Propagator'
        self.main_execute_ws = resource_filename('propagator.testing', 'Propagator')
        self.value_col_dependency = 'monitoring_locations'

    def test_value_columns(self):
        nt.assert_true(hasattr(self.tbx, 'value_columns'))
        nt.assert_true(isinstance(self.tbx.value_columns, arcpy.Parameter))
        nt.assert_equal(self.tbx.value_columns.parameterType, 'Required')
        nt.assert_equal(self.tbx.value_columns.direction, 'Input')
        nt.assert_equal(self.tbx.value_columns.datatype, 'Value Table')
        nt.assert_equal(self.tbx.value_columns.name, 'value_columns')
        nt.assert_list_equal(self.tbx.value_columns.parameterDependencies, [self.value_col_dependency])
        # see explanation in toolbox.value_columns.
        nt.assert_false(self.tbx.value_columns.multiValue)

    def test_monitoring_locations(self):
        nt.assert_true(hasattr(self.tbx, 'monitoring_locations'))
        nt.assert_true(isinstance(self.tbx.monitoring_locations, arcpy.Parameter))
        nt.assert_equal(self.tbx.monitoring_locations.parameterType, 'Required')
        nt.assert_equal(self.tbx.monitoring_locations.direction, 'Input')
        nt.assert_equal(self.tbx.monitoring_locations.datatype, 'Feature Class')
        nt.assert_equal(self.tbx.monitoring_locations.name, 'monitoring_locations')
        nt.assert_list_equal(self.tbx.monitoring_locations.parameterDependencies, ['workspace'])
        nt.assert_false(self.tbx.monitoring_locations.multiValue)

    def test_ml_type_col(self):
        nt.assert_true(hasattr(self.tbx, 'ml_type_col'))
        nt.assert_true(isinstance(self.tbx.ml_type_col, arcpy.Parameter))
        nt.assert_equal(self.tbx.ml_type_col.parameterType, 'Required')
        nt.assert_equal(self.tbx.ml_type_col.direction, 'Input')
        nt.assert_equal(self.tbx.ml_type_col.datatype, 'Field')
        nt.assert_equal(self.tbx.ml_type_col.name, 'ml_type_col')
        nt.assert_list_equal(self.tbx.ml_type_col.parameterDependencies, ['monitoring_locations'])
        nt.assert_false(self.tbx.ml_type_col.multiValue)

    def test_included_ml_types(self):
        nt.assert_true(hasattr(self.tbx, 'included_ml_types'))
        nt.assert_true(isinstance(self.tbx.included_ml_types, arcpy.Parameter))
        nt.assert_equal(self.tbx.included_ml_types.parameterType, 'Required')
        nt.assert_equal(self.tbx.included_ml_types.direction, 'Input')
        nt.assert_equal(self.tbx.included_ml_types.datatype, 'String')
        nt.assert_equal(self.tbx.included_ml_types.name, 'included_ml_types')
        nt.assert_list_equal(self.tbx.included_ml_types.parameterDependencies, [])
        nt.assert_true(self.tbx.included_ml_types.multiValue)
        nt.assert_equal(self.tbx.included_ml_types.filter.type, "ValueList")

    def test_params_as_list(self):
        params = self.tbx._params_as_list()
        names = [str(p.name) for p in params]
        known_names = [
            'workspace',
            'subcatchments',
            'ID_column',
            'downstream_ID_column',
            'monitoring_locations',
            'ml_type_col',
            'included_ml_types',
            'value_columns',
            'streams',
            'output_layer',
            'add_output_to_map',
        ]
        nt.assert_list_equal(names, known_names)

    @nptest.dec.skipif(not pptest.has_fiona)
    def test_analyze(self):
        tbx = toolbox.Propagator()
        ws = resource_filename('propagator.testing', 'tbx_propagate')
        columns = 'Dry_B Average;Dry_M Median;Dry_N Min;Wet_B Max;Wet_M #;Wet_N Median'
        with mock.patch.object(toolbox.Propagator, '_add_to_map') as atm:
            subc_layer, stream_layer = tbx.analyze(
                workspace=ws,
                overwrite=True,
                subcatchments='subcatchments.shp',
                ID_column='CID',
                downstream_ID_column='DS_CID',
                monitoring_locations='monitoring_locations.shp',
                value_columns=columns,
                output_layer='test.shp',
                streams='streams.shp',
                add_output_to_map=True
            )

            nt.assert_equal(subc_layer, 'test_subcatchments.shp')
            nt.assert_equal(stream_layer, 'test_streams.shp')

            pptest.assert_shapefiles_are_close(
                os.path.join(ws, 'expected_subc.shp'),
                os.path.join(ws, subc_layer),
            )

            pptest.assert_shapefiles_are_close(
                os.path.join(ws, 'expected_streams.shp'),
                os.path.join(ws, stream_layer),
            )

            utils.cleanup_temp_results(
                os.path.join(ws, subc_layer),
                os.path.join(ws, stream_layer)
            )
            atm.assert_has_calls([mock.call(subc_layer), mock.call(stream_layer)])

    @nptest.dec.skipif(not pptest.has_fiona)
    def test_analyze_filter(self):
        tbx = toolbox.Propagator()
        ws = resource_filename('propagator.testing', 'tbx_propagate')
        columns = 'Dry_B #;Dry_M Median;Dry_N Min;Wet_B Max;Wet_M Average;Wet_N Median'
        stacol = 'StationTyp'
        with mock.patch.object(toolbox.Propagator, '_add_to_map') as atm:
            subc_layer, stream_layer = tbx.analyze(
                workspace=ws,
                overwrite=True,
                subcatchments='subcatchments.shp',
                ID_column='CID',
                downstream_ID_column='DS_CID',
                monitoring_locations='monitoring_locations.shp',
                ml_type_col=stacol,
                included_ml_types=['Channel', 'Outfall', 'Outfall, Coastal'],
                value_columns=columns,
                output_layer='test_filtered.shp',
                streams='streams.shp',
                add_output_to_map=True
            )

            nt.assert_equal(subc_layer, 'test_filtered_subcatchments.shp')
            nt.assert_equal(stream_layer, 'test_filtered_streams.shp')

            pptest.assert_shapefiles_are_close(
                os.path.join(ws, 'expected_filtered_subc.shp'),
                os.path.join(ws, subc_layer),
            )

            pptest.assert_shapefiles_are_close(
                os.path.join(ws, 'expected_filtered_streams.shp'),
                os.path.join(ws, stream_layer),
            )

            utils.cleanup_temp_results(
                os.path.join(ws, subc_layer),
                os.path.join(ws, stream_layer)
            )

            atm.assert_has_calls([mock.call(subc_layer), mock.call(stream_layer)])


@mock.patch('propagator.utils._status', mock_status)
class Test_Accumulator(BaseToolboxChecker_Mixin):
    def setup(self):
        self.tbx = toolbox.Accumulator()
        self.main_execute_dir = 'propagator.testing.Accumulator'
        self.main_execute_ws = resource_filename('propagator.testing', 'Accumulator')
        self.value_col_dependency = 'subcatchments'

    def test_params_as_list(self):
        params = self.tbx._params_as_list()
        names = [str(p.name) for p in params]
        known_names = [
            'workspace',
            'subcatchments',
            'ID_column',
            'downstream_ID_column',
            'area_col',
            'imp_col',
            'streams',
            'output_layer',
            'add_output_to_map',
        ]
        nt.assert_list_equal(names, known_names)

    def test_area_col(self):
        nt.assert_true(hasattr(self.tbx, 'area_col'))
        nt.assert_true(isinstance(self.tbx.area_col, arcpy.Parameter))
        nt.assert_equal(self.tbx.area_col.parameterType, 'Optional')
        nt.assert_equal(self.tbx.area_col.direction, 'Input')
        nt.assert_equal(self.tbx.area_col.datatype, 'Field')
        nt.assert_equal(self.tbx.area_col.name, 'area_col')
        nt.assert_list_equal(self.tbx.area_col.parameterDependencies, ['subcatchments'])
        nt.assert_false(self.tbx.area_col.multiValue)

    def test_imp_col(self):
        nt.assert_true(hasattr(self.tbx, 'imp_col'))
        nt.assert_true(isinstance(self.tbx.imp_col, arcpy.Parameter))
        nt.assert_equal(self.tbx.imp_col.parameterType, 'Required')
        nt.assert_equal(self.tbx.imp_col.direction, 'Input')
        nt.assert_equal(self.tbx.imp_col.datatype, 'Field')
        nt.assert_equal(self.tbx.imp_col.name, 'imp_col')
        nt.assert_list_equal(self.tbx.imp_col.parameterDependencies, ['subcatchments'])
        nt.assert_false(self.tbx.imp_col.multiValue)

    @nptest.dec.skipif(not pptest.has_fiona)
    def test_analyze(self):
        tbx = toolbox.Accumulator()
        ws = resource_filename('propagator.testing', 'score_accumulator')
        columns = ['Dry_B', 'Dry_M', 'Dry_N', 'Wet_B', 'Wet_M', 'Wet_N']
        with mock.patch.object(tbx, '_add_to_map') as atm:
            stream_layer = tbx.analyze(
                workspace=ws,
                overwrite=True,
                subcatchments='subcatchment_wq.shp',
                ID_column='Catch_ID_a',
                downstream_ID_column='Dwn_Catch_',
                area_col='Area',
                imp_col='Imp',
                streams='streams.shp',
                output_layer='output.shp',
                add_output_to_map=True
            )

            nt.assert_equal(stream_layer, 'output.shp')

            pptest.assert_shapefiles_are_close(
                os.path.join(ws, 'expected_results.shp'),
                os.path.join(ws, stream_layer),
            )

            utils.cleanup_temp_results(
                os.path.join(ws, stream_layer)
            )
            atm.assert_called_once_with(stream_layer)

            utils.cleanup_temp_results(stream_layer)
