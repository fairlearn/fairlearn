Creating new releases
---------------------

First add a description of the changes introduced in the package version you
want to release to `CHANGES.md <https://github.com/fairlearn/fairlearn/CHANGES.md>`_.

It is also best to verify that the Dashboard loads correctly. This is slightly
involved:

#. Create a wheel by running :code:`python setup.py sdist bdist_wheel` from
   the repository root. This will create a :code:`dist` directory which
   contains a :code:`.whl` file.
#. Create a new conda environment for the test
#. In this new environment, install this wheel by running
   :code:`pip install dist/<FILENAME>.whl`
#. Install any pip packages required for the notebooks
#. Check that the dashboard loads in the notebooks

We have an
`Azure DevOps Pipeline <https://dev.azure.com/responsibleai/fairlearn/_build?definitionId=48&_a=summary>`_
which takes care of building wheels and pushing to PyPI. Validations are also
performed prior to any deployments, and also following the uploads to Test-PyPI
and PyPI. To use it:

#. Ensure that :code:`_base_version` in :code:`fairlearn/__init__.py` is set
   correctly for
   PyPI.
#. Put down a tag corresponding to this :code:`_base_version` but preprended
   with :code:`v`. For example, version :code:`0.5.0` should be tagged with
   :code:`v0.5.0`.
#. Queue the pipeline at this tag, with a variable :code:`DEV_VERSION` set to
   zero. When the package is uploaded to Test-PyPI, this number will be appended to
   the version as a :code:`dev[n]` suffix

The pipeline requires sign offs immediately prior to the deployments to
Test-PyPI and PyPI. If there is an issue found, then after applying the fix,
update the location of the tag, and queue a new release pipeline with the value
of :code:`DEV_VERSION` increased by one.

The :code:`DEV_VERSION` variable is to work around the PyPI behaviour where
uploads are immutable and published immediately. This means that each upload
to PyPI 'uses up' that particular version (on that particular PyPI instance).
Since we wish to deploy exactly the same bits to both Test-PyPI and PyPI,
without this workaround the version released to PyPI would depend on the
number of issues discovered when uploading to Test-PyPI. If PyPI updates their
release process to separate the 'upload' and 'publish' steps (i.e. until a
package is published, it remains hidden and mutable) then the code associated
with :code:`DEV_VERSION` should be removed.

As part of the release process, the :code:`build_wheels.py` script uses
:code:`process_readme.py` to turn all the relative links in the ReadMe file
into absolute ones (this is the reason why the applied tag has be of the form
:code:`v[_base_version]`). The :code:`process_readme.py` script is slightly
fragile with respect to the contents of the ReadMe, so after significant
changes its output should be verified.
