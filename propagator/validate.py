""" Simple validation functions for ``propagator``.

This contains main functions use propagate and accumlate catchment
properties in a larger watershed.

(c) Geosyntec Consultants, 2015.

Released under the BSD 3-clause license (see LICENSE file for more info)

Written by Paul Hobson (phobson@geosyntec.com)

"""


def flow_direction(up_or_down):
    valid_directions = ['upstream', 'downstream']
    if up_or_down.lower() not in valid_directions:
        raise ValueError("{} is not one of {}".format(up_or_down, valid_directions))
    else:
        return up_or_down.lower()
