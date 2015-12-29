import numpy

import nose.tools as nt
import numpy.testing as nptest

from propagator.utils import misc


def test_mask_array_with_flood():
    zones = numpy.array([
        [  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
        [  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   0],
        [  1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0],
        [  1,   1,   1,   1,   2,   2,   2,   2,   0,   0,   0],
        [  0,   0,   0,   2,   2,   2,   2,   0,   0,   0,   0],
        [  2,   2,   2,   2,   2,   2,   2,   0,   0,   0,   0],
        [  2,   2,   2,   2,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    ])

    topo = numpy.array([
        [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.],
        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.],
        [ 2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12.],
        [ 3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13.],
        [ 4.,  5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14.],
        [ 5.,  6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15.],
        [ 6.,  7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16.],
        [ 7.,  8.,  9., 10., 11., 12., 13., 14., 15., 16., 17.],
        [ 8.,  9., 10., 11., 12., 13., 14., 15., 16., 17., 18.],
        [ 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.],
        [10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.],
    ])

    known_flooded = numpy.array([
        [  1,   1,   1,   1,   1,   1,   1,   0,   0,   0,   0],
        [  1,   1,   1,   1,   1,   1,   0,   0,   0,   0,   0],
        [  1,   1,   1,   1,   1,   0,   0,   0,   0,   0,   0],
        [  1,   1,   1,   1,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  2,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    ])

    flooded = misc.flood_zones(zones, topo, 6.0)
    nptest.assert_array_almost_equal(flooded, known_flooded)



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

