""" Top-level functions for ``propagator``.

This contains main functions use propagate and accumlate catchment
properties in a larger watershed.

(c) Geosyntec Consultants, 2015.

Released under the BSD 3-clause license (see LICENSE file for more info)

Written by Paul Hobson (phobson@geosyntec.com)

"""


import os
import sys
import glob
import datetime

import numpy

import arcpy

from . import utils


def create_relationship_array(subcatchment_layer):
    """
    Creates a numpy record array of all of the upstream/downstream
    relationships in a collection of SBPAT-ish catchments

    Parameters
    ----------
    catchment_layer : arcpy.Layer
        A layer object of all of the catchments. There much be a field
        listing the ID the dowmstream catchments in the attribute table.

    Returns
    -------
    relationships : numpy record array
        Labeled array with "upstream" and "downstream" columns for every
        relationship among the catchments.

    """
    pass


def propagate_values(point_layer, subcatchment_layer, direction='upstream'):
    direction = validate.direction(direction)
    pass


def accumulate_values(stream_layer, subcatchment_layer, direction='downstream'):
    direction = validate.direction(direction)
    pass


def _split_streams(stream_layer, subcatchment_layer):
    pass


def _find_closest_catchment(point, subcatchment_layer):
    pass


def _assign_values(point, catchment):
    pass


