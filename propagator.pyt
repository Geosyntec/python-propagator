import arcpy

from propagator.toolbox import Propagator, Accumulator


class Toolbox(object):
    def __init__(self):
        """ Propagator: Push monitoring location attributes upstream and
        downstream through sub-catchments.

        """

        self.label = "Propagator"
        self.alias = "Propagator"

        # List of tool classes associated with this toolbox
        self.tools = [Propagator, Accumulator]
