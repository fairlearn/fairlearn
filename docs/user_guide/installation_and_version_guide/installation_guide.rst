.. _installation_guide:

Installation Guide
==================

Installation
------------

Fairlearn can be installed with :code:`pip` from
`PyPI <https://pypi.org/project/fairlearn>`_ as follows:

.. code-block:: bash

    pip install fairlearn

Fairlearn is also available on
`conda-forge <https://anaconda.org/conda-forge/fairlearn>`_:

.. code-block:: bash

    conda install -c conda-forge fairlearn

Some modules of Fairlearn have *optional* dependencies (see
:ref:`optional-dependencies`), which are not installed by default using
the basic installation.

These dependencies are grouped in **extras**, which can be installed
like so (by the example of the ``customplot`` extra):

.. code-block:: bash

    pip install fairlearn[customplot]

.. _optional-dependencies:

Dependencies
------------

Fairlearn has the following optional dependencies which can be 
installed via the corresponding **extra**, ordered by dependent module:

.. list-table:: 
    :widths: 50 25 25
    :header-rows: 1

    * - Dependent Module
      - Dependency
      - Extra
    * - ``fairlearn/postprocessing/_plotting.py``
      - ``matplotlib``
      - ``customplot``