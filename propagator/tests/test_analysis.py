import numpy

import nose.tools as nt
import numpy.testing as nptest

from propagator import analysis

SUBCATCHMENTS = numpy.array(
    [
        ('A1', 'Ocean', 'A1_x'), ('A2', 'Ocean', 'A2_x'), ('B1', 'A1', 'None'),
        ('B2', 'A1', 'B2_x'), ('B3', 'A2', 'B3_x'), ('C1', 'B2', 'C1_x'),
        ('C2', 'B3', 'None'), ('C3', 'B3', 'None'), ('D1', 'C1', 'None'),
        ('D2', 'C3', 'None'), ('E1', 'D1', 'None'), ('E2', 'D2', 'None'),
        ('F1', 'E1', 'F1_x'), ('F2', 'E1', 'None'), ('F3', 'E1', 'F3_x'),
        ('G1', 'F1', 'None'), ('G2', 'F3', 'None'), ('H1', 'G2', 'None'),
    ], dtype=[('ID', '<U5'), ('DS_ID', '<U5'), ('Cu', '<U5'),]
)


class Test_trace_upstream(object):
    def setup(self):
        self.subcatchments = SUBCATCHMENTS.copy()

        self.expected_left = numpy.array(
            [
                (u'B1', u'A1', u'None'), (u'B2', u'A1', u'B2_x'),
                (u'C1', u'B2', u'C1_x'), (u'D1', u'C1', u'None'),
                (u'E1', u'D1', u'None'), (u'F1', u'E1', u'F1_x'),
                (u'G1', u'F1', u'None'), (u'F2', u'E1', u'None'),
                (u'F3', u'E1', u'F3_x'), (u'G2', u'F3', u'None'),
                (u'H1', u'G2', u'None')
            ], dtype=self.subcatchments.dtype
        )

        self.expected_right = numpy.array(
            [
                (u'B3', u'A2', u'B3_x'), (u'C2', u'B3', u'None'),
                (u'C3', u'B3', u'None'), (u'D2', u'C3', u'None'),
                (u'E2', u'D2', u'None')
            ], dtype=self.subcatchments.dtype
        )

    def test_left_fork(self):
        upstream = analysis.trace_upstream(self.subcatchments, 'A1')
        nptest.assert_array_equal(upstream, self.expected_left)

    def test_right_fork(self):
        upstream = analysis.trace_upstream(self.subcatchments, 'A2')
        nptest.assert_array_equal(upstream, self.expected_right)


def test_find_bottoms():
    subcatchments = SUBCATCHMENTS.copy()
    expected = numpy.array(
        [(u'A1', u'Ocean', u'A1_x'), (u'A2', u'Ocean', u'A2_x')],
        dtype=subcatchments.dtype
    )
    result = analysis.find_bottoms(subcatchments, 'Ocean')
    nptest.assert_array_equal(result, expected)


def test_find_tops():
    subcatchments = SUBCATCHMENTS.copy()
    expected = numpy.array(
        [
            (u'B1', u'A1', u'None'),
            (u'C2', u'B3', u'None'),
            (u'E2', u'D2', u'None'),
            (u'F2', u'E1', u'None'),
            (u'G1', u'F1', u'None'),
            (u'H1', u'G2', u'None'),
        ],
      dtype=subcatchments.dtype
    )
    result = analysis.find_tops(subcatchments)
    nptest.assert_array_equal(result, expected)


def test_propagate_scores():
    pass
