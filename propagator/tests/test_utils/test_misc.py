import numpy

import nose.tools as nt
import numpy.testing as nptest

from propagator.utils import misc


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
        row = misc.find_row_in_array(self.input_array, 'ID', 'Junk')
        nt.assert_true(row is None)

    def test_normal_1_row(self):
        row = misc.find_row_in_array(self.input_array, 'ID', 'A1')
        nt.assert_tuple_equal(tuple(row), tuple(self.input_array[0]))

    @nt.raises(ValueError)
    def test_too_man_rows(self):
         row = misc.find_row_in_array(self.input_array, 'DS_ID', 'A1')


def test_Statistic():
    x = misc.Statistic('Cu', numpy.mean, 'MaxCu')
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
            misc.Statistic('Cu', numpy.max, 'MaxCu'),
            misc.Statistic('Pb', numpy.mean, 'AvgPb')
        ]

    def test_one_group_col(self):
        result = misc.rec_groupby(self.data, 'ID', *self.stats)
        result.sort()

        expected = self.expected_one_group_col.copy()
        expected.sort()

        nptest.assert_array_equal(result, expected)

    def test_two_group_col(self):
        result = misc.rec_groupby(self.data, ['ID', 'DS_ID'], *self.stats)
        result.sort()

        expected = self.expected_two_group_col.copy()
        expected.sort()

        nptest.assert_array_equal(result, expected)



class Test_stats_with_ignored_values(object):
    def setup(self):
        self.x1 = [1., 2., 3., 4., 5.]
        self.x2 = [5.] * 5 # just a list of 5's

    def test_normal(self):
        expected = 2.5
        result = misc.stats_with_ignored_values(self.x1, numpy.mean, ignored_value=5)
        nt.assert_equal(result, expected)

    def test_nothing_ignored(self):
        expected = 3.
        result = misc.stats_with_ignored_values(self.x1, numpy.mean, ignored_value=6)
        nt.assert_equal(result, expected)

    def test_nothing_to_ignore(self):
        expected = 3.
        result = misc.stats_with_ignored_values(self.x1, numpy.mean, ignored_value=None)
        nt.assert_equal(result, expected)

    def test_ignore_everthing(self):
        expected = 5
        result = misc.stats_with_ignored_values(self.x2, numpy.mean, ignored_value=5)
        nt.assert_equal(result, expected)
