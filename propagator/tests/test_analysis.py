import os
from pkg_resources import resource_filename

import numpy

import nose.tools as nt
import numpy.testing as nptest
import propagator.testing as pptest

from propagator import analysis
from propagator import utils


SIMPLE_SUBCATCHMENTS = numpy.array(
    [
        ('A1', 'Ocean', 'A1_x', 'A1_y'), ('A2', 'Ocean', 'A2_x', 'A2_y'),
        ('B1', 'A1', 'None', 'B1_y'), ('B2', 'A1', 'B2_x', 'None'),
        ('B3', 'A2', 'B3_x', 'B3_y'), ('C1', 'B2', 'C1_x', 'None'),
        ('C2', 'B3', 'None', 'None'), ('C3', 'B3', 'None', 'None'),
        ('D1', 'C1', 'None', 'None'), ('D2', 'C3', 'None', 'D2_y'),
        ('E1', 'D1', 'None', 'E1_y'), ('E2', 'D2', 'None', 'None'),
        ('F1', 'E1', 'F1_x', 'None'), ('F2', 'E1', 'None', 'None'),
        ('F3', 'E1', 'F3_x', 'None'), ('G1', 'F1', 'None', 'None'),
        ('G2', 'F3', 'None', 'None'), ('H1', 'G2', 'None', 'None'),
    ], dtype=[('ID', '<U5'), ('DS_ID', '<U5'), ('Cu', '<U5'), ('Pb', '<U5'),]
)

COMPLEX_SUBCATCHMENTS = numpy.array(
    [
        (u'A1', u'Ocean', u'A1Cu'), (u'A2', u'Ocean', u'A2Cu'), (u'B1', u'A1', u'None'),
        (u'B2', u'A1', u'None'), (u'B3', u'A2', u'None'), (u'C1', u'B2', u'None'),
        (u'D1', u'C1', u'D1Cu'), (u'C2', u'B3', u'None'), (u'C3', u'B3', u'C3Cu'),
        (u'D2', u'C3', u'None'), (u'E2', u'D2', u'None'), (u'E1', u'D1', u'None'),
        (u'F1', u'E1', u'None'), (u'F2', u'E1', u'F2Cu'), (u'F3', u'E1', u'None'),
        (u'G1', u'F1', u'None'), (u'G2', u'F3', u'None'), (u'H1', u'F3', u'H1Cu'),
        (u'I1', u'H1', u'None'), (u'J1', u'I1', u'None'), (u'J2', u'I1', u'J2Cu'),
        (u'K2', u'J2', u'None'), (u'K1', u'J2', u'None'), (u'L1', u'K1', u'None'),
    ], dtype=[('ID', '<U5'), ('DS_ID', '<U5'), ('Cu', '<U5'),]
)


class Test_trace_upstream(object):
    def setup(self):
        self.subcatchments = SIMPLE_SUBCATCHMENTS.copy()

        self.expected_left_no_base = numpy.array(
            [
                (u'B1', u'A1', u'None', u'B1_y'), (u'B2', u'A1', u'B2_x', u'None'),
                (u'C1', u'B2', u'C1_x', u'None'), (u'D1', u'C1', u'None', u'None'),
                (u'E1', u'D1', u'None', u'E1_y'), (u'F1', u'E1', u'F1_x', u'None'),
                (u'G1', u'F1', u'None', u'None'), (u'F2', u'E1', u'None', u'None'),
                (u'F3', u'E1', u'F3_x', u'None'), (u'G2', u'F3', u'None', u'None'),
                (u'H1', u'G2', u'None', u'None')
            ], dtype=self.subcatchments.dtype
        )

        self.expected_right_no_base = numpy.array(
            [
                (u'B3', u'A2', u'B3_x', u'B3_y'), (u'C2', u'B3', u'None', u'None'),
                (u'C3', u'B3', u'None', u'None'), (u'D2', u'C3', u'None', u'D2_y'),
                (u'E2', u'D2', u'None', u'None')
            ], dtype=self.subcatchments.dtype
        )

        self.expected_left_with_base = numpy.array(
            [
                ('A1', 'Ocean', 'A1_x', 'A1_y'),
                (u'B1', u'A1', u'None', u'B1_y'), (u'B2', u'A1', u'B2_x', u'None'),
                (u'C1', u'B2', u'C1_x', u'None'), (u'D1', u'C1', u'None', u'None'),
                (u'E1', u'D1', u'None', u'E1_y'), (u'F1', u'E1', u'F1_x', u'None'),
                (u'G1', u'F1', u'None', u'None'), (u'F2', u'E1', u'None', u'None'),
                (u'F3', u'E1', u'F3_x', u'None'), (u'G2', u'F3', u'None', u'None'),
                (u'H1', u'G2', u'None', u'None')
            ], dtype=self.subcatchments.dtype
        )

        self.expected_right_with_base = numpy.array(
            [
                ('A2', 'Ocean', 'A2_x', 'A2_y'),
                (u'B3', u'A2', u'B3_x', u'B3_y'), (u'C2', u'B3', u'None', u'None'),
                (u'C3', u'B3', u'None', u'None'), (u'D2', u'C3', u'None', u'D2_y'),
                (u'E2', u'D2', u'None', u'None')
            ], dtype=self.subcatchments.dtype
        )

    def test_left_fork_no_base(self):
        upstream = analysis.trace_upstream(self.subcatchments, 'A1')
        nptest.assert_array_equal(upstream, self.expected_left_no_base)

    def test_right_fork_no_base(self):
        upstream = analysis.trace_upstream(self.subcatchments, 'A2')
        nptest.assert_array_equal(upstream, self.expected_right_no_base)

    def test_left_fork_with_base(self):
        upstream = analysis.trace_upstream(self.subcatchments, 'A1', include_base=True)
        nptest.assert_array_equal(upstream, self.expected_left_with_base)

    def test_right_fork_with_base(self):
        upstream = analysis.trace_upstream(self.subcatchments, 'A2', include_base=True)
        nptest.assert_array_equal(upstream, self.expected_right_with_base)


def test_find_edges():
    subcatchments = SIMPLE_SUBCATCHMENTS.copy()
    expected = numpy.array(
        [(u'A1', u'Ocean', u'A1_x', u'A1_y'), (u'A2', u'Ocean', u'A2_x', u'A2_y')],
        dtype=subcatchments.dtype
    )
    result = analysis.find_edges(subcatchments, 'Ocean')
    nptest.assert_array_equal(result, expected)


def test_find_tops():
    subcatchments = SIMPLE_SUBCATCHMENTS.copy()
    expected = numpy.array(
        [
            (u'B1', u'A1', u'None', u'B1_y'),
            (u'C2', u'B3', u'None', u'None'),
            (u'E2', u'D2', u'None', u'None'),
            (u'F2', u'E1', u'None', u'None'),
            (u'G1', u'F1', u'None', u'None'),
            (u'H1', u'G2', u'None', u'None'),
        ],
      dtype=subcatchments.dtype
    )
    result = analysis.find_tops(subcatchments)
    nptest.assert_array_equal(result, expected)


def test_propagate_scores_complex_1_columns():
    subcatchments = COMPLEX_SUBCATCHMENTS.copy()
    expected = numpy.array(
        [
            (u'A1', u'Ocean', u'A1Cu'), (u'A2', u'Ocean', u'A2Cu'),
            (u'B1', u'A1', u'A1Cu'), (u'B2', u'A1', u'A1Cu'),
            (u'B3', u'A2', u'A2Cu'), (u'C1', u'B2', u'A1Cu'),
            (u'D1', u'C1', u'D1Cu'), (u'C2', u'B3', u'A2Cu'),
            (u'C3', u'B3', u'C3Cu'), (u'D2', u'C3', u'C3Cu'),
            (u'E2', u'D2', u'C3Cu'), (u'E1', u'D1', u'D1Cu'),
            (u'F1', u'E1', u'D1Cu'), (u'F2', u'E1', u'F2Cu'),
            (u'F3', u'E1', u'D1Cu'), (u'G1', u'F1', u'D1Cu'),
            (u'G2', u'F3', u'D1Cu'), (u'H1', u'F3', u'H1Cu'),
            (u'I1', u'H1', u'H1Cu'), (u'J1', u'I1', u'H1Cu'),
            (u'J2', u'I1', u'J2Cu'), (u'K2', u'J2', u'J2Cu'),
            (u'K1', u'J2', u'J2Cu'), (u'L1', u'K1', u'J2Cu')
        ], dtype=subcatchments.dtype
    )
    result = analysis.propagate_scores(subcatchments, 'ID', 'DS_ID', 'Cu',
                                       ignored_value='None')
    nptest.assert_array_equal(result, expected)


def test_propagate_scores_simple_2_columns():
    subcatchments = SIMPLE_SUBCATCHMENTS.copy()
    expected = numpy.array(
        [
            ('A1', 'Ocean', 'A1_x', 'A1_y'), ('A2', 'Ocean', 'A2_x', 'A2_y'),
            ('B1', 'A1', 'A1_x', 'B1_y'), ('B2', 'A1', 'B2_x', 'A1_y'),
            ('B3', 'A2', 'B3_x', 'B3_y'), ('C1', 'B2', 'C1_x', 'A1_y'),
            ('C2', 'B3', 'B3_x', 'B3_y'), ('C3', 'B3', 'B3_x', 'B3_y'),
            ('D1', 'C1', 'C1_x', 'A1_y'), ('D2', 'C3', 'B3_x', 'D2_y'),
            ('E1', 'D1', 'C1_x', 'E1_y'), ('E2', 'D2', 'B3_x', 'D2_y'),
            ('F1', 'E1', 'F1_x', 'E1_y'), ('F2', 'E1', 'C1_x', 'E1_y'),
            ('F3', 'E1', 'F3_x', 'E1_y'), ('G1', 'F1', 'F1_x', 'E1_y'),
            ('G2', 'F3', 'F3_x', 'E1_y'), ('H1', 'G2', 'F3_x', 'E1_y'),
        ], dtype=[('ID', '<U5'), ('DS_ID', '<U5'), ('Cu', '<U5'), ('Pb', '<U5'),]
    )

    result = analysis.propagate_scores(subcatchments, 'ID', 'DS_ID', 'Pb',
                                       ignored_value='None')
    result = analysis.propagate_scores(result, 'ID', 'DS_ID', 'Cu',
                                       ignored_value='None')

    nptest.assert_array_equal(result, expected)


def test__find_downstream_scores():
    subcatchments = SIMPLE_SUBCATCHMENTS.copy()
    expected = ('E1', 'D1', 'None', 'E1_y')
    value = analysis._find_downstream_scores(subcatchments, 'G1', 'Pb')
    nt.assert_tuple_equal(tuple(value), expected)


class Test_preprocess_wq(object):
    def setup(self):
        self.ws = resource_filename('propagator.testing', 'preprocess_wq')
        self.ml = 'monitoring_locations.shp'
        self.sc = 'subcatchments.shp'
        self.expected = 'expected.shp'
        self.results = 'test.shp'
        self.wq_cols = [
            ('Dry_B', 'medIAN'),
            ('Dry_M', 'Median'),
            ('Dry_N', 'minIMUM'),
            ('Wet_B', 'minIMUM'),
            ('Wet_M',),
            ('Wet_N', 'MAXIMUM'),
        ]

        self.expected_cols = [
            'medDry_B', 'medDry_M', 'minDry_N',
            'minWet_B', 'aveWet_M', 'maxWet_N',
        ]

    def test_baseline(self):
        with utils.OverwriteState(True), utils.WorkSpace(self.ws):
            wq, cols = analysis.preprocess_wq(
                monitoring_locations=self.ml,
                subcatchments=self.sc,
                id_col='CID',
                ds_col='DS_CID',
                output_path=self.results,
                value_columns=self.wq_cols
            )
        expected = 'expected.shp'
        pptest.assert_shapefiles_are_close(
            os.path.join(self.ws, expected),
            os.path.join(self.ws, self.results),
        )
        nt.assert_true(isinstance(wq, numpy.ndarray))
        nt.assert_list_equal(cols, self.expected_cols)

    def test_with_filter(self):
        with utils.OverwriteState(True), utils.WorkSpace(self.ws):
            wq, cols = analysis.preprocess_wq(
                monitoring_locations=self.ml,
                subcatchments=self.sc,
                id_col='CID',
                ds_col='DS_CID',
                ml_filter_cols='StationTyp',
                ml_filter=lambda row: row['StationTyp'] != 'Outfall',
                output_path=self.results,
                value_columns=self.wq_cols
            )
        expected = 'expected_filtered.shp'
        pptest.assert_shapefiles_are_close(
            os.path.join(self.ws, expected),
            os.path.join(self.ws, self.results),
        )
        nt.assert_true(isinstance(wq, numpy.ndarray))
        nt.assert_list_equal(cols, self.expected_cols)

    @nt.raises(ValueError)
    def test_no_wq_col_error(self):
        with utils.OverwriteState(True), utils.WorkSpace(self.ws):
            wq, cols = analysis.preprocess_wq(
                monitoring_locations=self.ml,
                subcatchments=self.sc,
                id_col='CID',
                ds_col='DS_CID',
                output_path=self.results,
            )

    def teardown(self):
        utils.cleanup_temp_results(os.path.join(self.ws, self.results))


@nt.nottest
def doctor_subcatchments(array, to_remove):
    sub_array = numpy.rec.fromrecords(
        filter(lambda x: x['ID'] not in to_remove, array.copy()),
        dtype=array.dtype
    )

    return sub_array


def test_remove_orphan_subcatchments():
    to_remove = ['E1', 'C3']
    input_array = doctor_subcatchments(SIMPLE_SUBCATCHMENTS, to_remove)
    expected = numpy.array(
        [
            ('A1', 'Ocean', 'A1_x', 'A1_y'), ('A2', 'Ocean', 'A2_x', 'A2_y'),
            ('B1', 'A1', 'None', 'B1_y'), ('B2', 'A1', 'B2_x', 'None'),
            ('B3', 'A2', 'B3_x', 'B3_y'), ('C1', 'B2', 'C1_x', 'None'),
            ('C2', 'B3', 'None', 'None'), ('D1', 'C1', 'None', 'None'),
        ], dtype=[('ID', '<U5'), ('DS_ID', '<U5'), ('Cu', '<U5'), ('Pb', '<U5'),]
    )
    result = analysis.remove_orphan_subcatchments(input_array, id_col='ID', ds_col='DS_ID',
                                                  bottom_ID='Ocean')
    nptest.assert_array_equal(result, expected)


def test_mark_edges():
    to_remove = ['E1', 'C3']
    input_array = doctor_subcatchments(SIMPLE_SUBCATCHMENTS, to_remove)
    expected = numpy.array(
        [
            ('A1', 'EDGE', 'A1_x', 'A1_y'), ('A2', 'EDGE', 'A2_x', 'A2_y'),
            ('B1', 'A1', 'None', 'B1_y'), ('B2', 'A1', 'B2_x', 'None'),
            ('B3', 'A2', 'B3_x', 'B3_y'), ('C1', 'B2', 'C1_x', 'None'),
            ('C2', 'B3', 'None', 'None'), ('D1', 'C1', 'None', 'None'),
            ('D2', 'EDGE', 'None', 'D2_y'), ('E2', 'D2', 'None', 'None'),
            ('F1', 'EDGE', 'F1_x', 'None'), ('F2', 'EDGE', 'None', 'None'),
            ('F3', 'EDGE', 'F3_x', 'None'), ('G1', 'F1', 'None', 'None'),
            ('G2', 'F3', 'None', 'None'), ('H1', 'G2', 'None', 'None'),
        ], dtype=[('ID', '<U5'), ('DS_ID', '<U5'), ('Cu', '<U5'), ('Pb', '<U5'),]
    )
    results = analysis.mark_edges(input_array, id_col='ID', ds_col='DS_ID', edge_ID='EDGE')
    nptest.assert_array_equal(results, expected)


def test__get_wq_fields():
    ws = resource_filename('propagator.testing', 'get_wq_fields')
    with utils.WorkSpace(ws):
        results = analysis._get_wq_fields('monitoring_locations.shp', ['dry', 'wet'])

    expected = [u'Dry_B', u'Dry_M', u'Dry_N', u'Wet_B', u'Wet_M', u'Wet_N']

    nt.assert_list_equal(results, expected)


def test_reduce():
    ws = resource_filename("propagator.testing", "_reduce")
    with utils.OverwriteState(True), utils.WorkSpace(ws):
        mon_locations = resource_filename("propagator.testing._reduce", "point.shp")
        expected_reduced_mon_locations = resource_filename("propagator.testing._reduce", "reduced_point.shp")
        # Create a placeholder for output first, since the function takes the output file as an input.

        reduced_mon_locations = utils.create_temp_filename("reduced_point", filetype='shape')
        reduced_mon_locations = analysis._reduce(mon_locations, reduced_mon_locations, ["WQ1","WQ2","WQ3"],'ID','FID')
        pptest.assert_shapefiles_are_close(reduced_mon_locations, expected_reduced_mon_locations)
        utils.cleanup_temp_results(reduced_mon_locations)


def test_non_zero_means():
    num_lst = [1, 2, 3, 0 ]
    num_lst2 = [0, 0 ,0 ,0]
    expected_lst_mean = 2
    expected_lst2_mean = 0
    lst_mean = analysis._non_zero_means(num_lst)
    lst2_mean = analysis._non_zero_means(num_lst2)
    nt.assert_equal(lst_mean, expected_lst_mean)
    nt.assert_equal(lst2_mean, expected_lst2_mean)


def test_aggregate_streams_by_subcatchment():
    ws = resource_filename('propagator.testing', 'agg_stream_in_subc')
    with utils.WorkSpace(ws), utils.OverwriteState(True):
        results = analysis.aggregate_streams_by_subcatchment(
            stream_layer='streams.shp',
            subcatchment_layer='subc.shp',
            id_col='CID',
            ds_col='DS_CID',
            other_cols=['WQ_1', 'WQ_2'],
            output_layer='test.shp'
        )

    nt.assert_equal(results, 'test.shp')
    pptest.assert_shapefiles_are_close(
        os.path.join(ws, results),
        os.path.join(ws, 'expected.shp'),
        ngeom=4
    )

    utils.cleanup_temp_results(os.path.join(ws, results),)


def test_collect_upstream_attributes():
    subcatchments_table = numpy.array(
        [
            ('A1', 'Ocean', 20, 45.23), ('A2', 'Ocean', 0.64, 42),
            ('B1', 'A1', 43.3, 45.23), ('B2', 'A1', 0.32, 41),
            ('B3', 'A2', 91, 15.23), ('C1', 'B2', 0.32, 4),
            ('C2', 'B3', 50.3, 45.23), ('C3', 'B3', 0.32, 41),
            ('D1', 'C1', 32, 45.23), ('D2', 'C3', 0.32, 41),
            ('E1', 'D1', 1, 45.23), ('E2', 'D2', 0.32, 100),
            ('F1', 'E1', 42, 35.3), ('F2', 'E1', 0.32, 315),
            ('F3', 'E1', 5, 45.23), ('G1', 'F1', 0.32, 123),
            ('G2', 'F3', 8, 45.23), ('H1', 'G2', 0.32, 41),
        ], dtype=[('ID', '<U5'), ('DS_ID', '<U5'), ('Imp', '<f8'), ('Area', '<f8'),]
    )

    split_streams_table1 = numpy.array(
        [('C2',), ('A1',), ('E2',), ('A2',)],
        dtype=[('ID', '<U2')]
    )

    result = analysis.collect_upstream_attributes(
        subcatchments_table=subcatchments_table,
        target_subcatchments=split_streams_table1,
        id_col='ID',
        ds_col='DS_ID',
        preserved_fields=['Imp', 'Area']
    )

    expected = numpy.array(
        [
            (20, 45.23, 'A1',),
            (43.3, 45.23, 'A1',),
            (0.32, 41, 'A1',),
            (0.32, 4, 'A1',),
            (32, 45.23, 'A1',),
            (1, 45.23, 'A1',),
            (42, 35.3, 'A1',),
            (0.32, 315, 'A1',),
            (5, 45.23, 'A1',),
            (0.32, 123, 'A1',),
            (8, 45.23, 'A1',),
            (0.32, 41, 'A1',),
            (91, 15.23, 'A2',),
            (50.3, 45.23, 'A2',),
            (0.64, 42, 'A2',),
            (0.32, 41, 'A2',),
            (0.32, 41, 'A2',),
            (0.32, 100, 'A2',),
            (50.3, 45.23, 'C2',),
            (0.32, 100, 'E2',),
        ], dtype=[('Imp', '<f8'), ('Area', '<f8'), ('ID','<U5'),]
    )

    result.sort()
    expected.sort()
    nptest.assert_array_equal(result, expected)
