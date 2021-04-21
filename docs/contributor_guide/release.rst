.. release_guide

Release Process
---------------

This page outlines the process for creating a new Fairlearn release.
The following steps must be performed on a clone of Fairlearn, not
a fork.

#. Create an annotated tag using the command at the branch point:

    :code:`git tag -a v<x.y.z> -m "Branch point for v<x.y.z> release"`

#. Push the tag to GitHub:

    :code:`git push origin v<x.y.z>`

#. Create the branch for the release:

    :code:`git checkout -b release/v<x.y.z>`

#. On the release branch:

    #. Update the version in `__init__.py` to `x.y.z`
    #. Update the version in the ReadMe

#. On the `main` branch:

    #. Update the version in `__init__.py` to `xy.z+1.dev0`
    #. Update the ReadMe to link to `v<x.y.z>`
    #. Check `docs/conf.py` to make sure that `smv_tag_whitelist` will pick up the
       new release
    #. Update `docs/static_landing_page/` so that all the links point to the new release

#. Run the `release pipeline <https://dev.azure.com/responsibleai/fairlearn/_build?definitionId=60>`_

    #. Ensure that you have selected the correct release branch
    #. Run first on 'Test' which will upload to <https://test.pypi.org>
    #. Finally, run the release pipeline set to 'Production' which will upload to <https://pypi.org/>
