.. _python:

Using **python-propagator** from python
======================================

You don't have to start an ArcGIS session to use **python-propagator**.
The analytical guts of the library are completely isolated from the ArcGIS interface portions.
This allows users to directly use the analytical capabilities from a python script or an interactive python session.

There are two interfaces to the analytic capabilities:

   1. :ref:`toolbox_guide`, which is easier to use and parallels the toolboxes.
   2. :ref:`analysis_guide`, which is more powerful but requires precise coding.


Common elements of examples
---------------------------
The following import statements are prerequisites for all of the code snippets below.

.. code-block:: python

   import numpy
   import arcpy
   import propagator
   from propagator import utils


.. _toolbox_guide:

The ``toolbox`` API
--------------------------------


(see :ref:`install instructions <install>`)


For a full description of the API, see the :ref:`reference guide <toolbox_auto>`

The :mod:`propagator.toolbox` interface provides a very high-level interface to **python-propagator** that very closely mimics the tooboxes.
Just like how there are two forms in the ArcGIS toolbox, there are two analagous classes available in the :mod:`propagator.toolbox` API.

The :func:`propagator.toolbox.propagate` function allows the user to progagate water quality scores from monitoring locations to upstream subcatchments.

The :func:`propagator.toolbox.accumulate` function automatically accumulate upstream subcatchment attributes into stream features.

Common input parameters
~~~~~~~~~~~~~~~~~~~~~~~

The following are the parameters shared by both toolboxes.
All parameters are required except where noted.

Analysis Workspace (``workspace``)
    This is the folder or geodatabase that contains all of the input for the analysis.

    .. note:: All of the input for the analysis (see below) *must* be in this workspace.

TBD

Code examples
~~~~~~~~~~~~~

Below is an example of using the :func:`propagator.toolbox.propagate` class to evaluate custom flood elevations.

.. code-block:: python

    # define the workspace as a geodatabase
    workspace = r'F:\phobson\propagator\MB_Small.gdb'

    # TBD


Below is an example of using the :func:`propagator.toolbox.accumulate` class to evaluate custom flood elevations.

.. code-block:: python

    # define the workspace as a geodatabase
    workspace = r'F:\phobson\propagator\MB_Small.gdb'

    # TBD


.. _analysis_guide:

The ``analysis`` API
--------------------

For a full description of the API, see the :mod:`propagator.analysis`.

The ``analysis`` API can be used to taylor a more nuanced, custom analysis of the impacts resulting from a flood event.
Where the ``toolbox`` API effectively limits the user to computing total area and counts of one asset each, the functions below can be used by a python programmer to assess the impact to any number of assets.

General descriptions
~~~~~~~~~~~~~~~~~~~~

The :mod:`propagator.analysis` submodule contains five functions:

:func:`propagator.analysis.trace_upstream`
    TBD


Code examples
~~~~~~~~~~~~~

The classes in :mod:`propagator.toolbox` rely on the functions in :mod:`propagator.analysis` to determine
   - TBD
   = TBD

The sample script below does TDB by using :mod:`propagator.analysis` directly.


.. code-block:: python

    # common parameters
    workspace = r'F:\phobson\propagator\MB_Small.gdb'
    # TBD
