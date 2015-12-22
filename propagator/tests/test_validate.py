import os
from pkg_resources import resource_filename
import time

import arcpy
import numpy

import nose.tools as nt
import numpy.testing as nptest
import propagator.testing as tgtest
import mock

from propagator import validate


def test_flow_direction_good():
    nt.assert_equal(
        validate.flow_direction("uPSTReam"),
        "upstream"
    )

    nt.assert_equal(
        validate.flow_direction("downSTReam"),
        "downstream"
    )

@nt.raises(ValueError)
def test_flow_direction_bad():
    validate.flow_direction("sideways")
