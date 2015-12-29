from pkg_resources import resource_filename
import os

import arcpy
import numpy

import nose.tools as nt
import numpy.testing as nptest
import propagator.testing as pgtest

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

        self.expected_left = numpy.array(
            [
                (u'B1', u'A1', u'None', u'B1_y'), (u'B2', u'A1', u'B2_x', u'None'),
                (u'C1', u'B2', u'C1_x', u'None'), (u'D1', u'C1', u'None', u'None'),
                (u'E1', u'D1', u'None', u'E1_y'), (u'F1', u'E1', u'F1_x', u'None'),
                (u'G1', u'F1', u'None', u'None'), (u'F2', u'E1', u'None', u'None'),
                (u'F3', u'E1', u'F3_x', u'None'), (u'G2', u'F3', u'None', u'None'),
                (u'H1', u'G2', u'None', u'None')
            ], dtype=self.subcatchments.dtype
        )

        self.expected_right = numpy.array(
            [
                (u'B3', u'A2', u'B3_x', u'B3_y'), (u'C2', u'B3', u'None', u'None'),
                (u'C3', u'B3', u'None', u'None'), (u'D2', u'C3', u'None', u'D2_y'),
                (u'E2', u'D2', u'None', u'None')
            ], dtype=self.subcatchments.dtype
        )

    def test_left_fork(self):
        upstream = analysis.trace_upstream(self.subcatchments, 'A1')
        nptest.assert_array_equal(upstream, self.expected_left)

    def test_right_fork(self):
        upstream = analysis.trace_upstream(self.subcatchments, 'A2')
        nptest.assert_array_equal(upstream, self.expected_right)


def test_find_bottoms():
    subcatchments = SIMPLE_SUBCATCHMENTS.copy()
    expected = numpy.array(
        [(u'A1', u'Ocean', u'A1_x', u'A1_y'), (u'A2', u'Ocean', u'A2_x', u'A2_y')],
        dtype=subcatchments.dtype
    )
    result = analysis.find_bottoms(subcatchments, 'Ocean')
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
    result = analysis.propagate_scores(subcatchments, 'Cu')
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

    result = analysis.propagate_scores(subcatchments, 'Pb')
    result = analysis.propagate_scores(result, 'Cu')


def test_find_downstream_scores():
    subcatchments = SIMPLE_SUBCATCHMENTS.copy()
    expected = ('E1', 'D1', 'None', 'E1_y')
    value = analysis.find_downstream_scores(subcatchments, 'G1', 'Pb')
    nt.assert_tuple_equal(tuple(value), expected)


def test_prepdata():
    ws = resource_filename("propagator.testing", "prepare_data")
    with utils.OverwriteState(True), utils.WorkSpace(ws):
        cat = resource_filename("propagator.testing.prepare_data", "cat.shp") 
        ml = resource_filename("propagator.testing.prepare_data", "ml.shp")
        expected_cat_wq = resource_filename("propagator.testing.prepare_data", "cat_wq.shp")
        final_field = [
            "FID",
            "Shape",
            "Catch_ID_a", 
            "Dwn_Catch_", 
            "Watershed", 
            "Station", 
            "Latitude",
            "Longitude"
        ]
        cat_wq = analysis.prepare_data(ml, cat, final_field)
        pgtest.assert_shapefiles_are_close(cat_wq, expected_cat_wq)
        utils.cleanup_temp_results(cat_wq)
        
