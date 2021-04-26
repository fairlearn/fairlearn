.. release_guide

Release Process
---------------

This page outlines the process for creating a new Fairlearn release.
The following steps assume git remote's `origin` points to
`fairlearn/fairlearn`.

#. Create the branch for the release:

    :code:`git checkout -b release/v<x.y.z>`

#. On the release branch:

    #. Update the version in `__init__.py` to `x.y.z`
    #. Update the version in the ReadMe
    #. Publish the branch to GitHub

#. On the `main` branch, create a PR to:

    #. Update the version in `__init__.py` to `xy.z+1.dev0`
    #. In the ReadMe, update the 'current stable release' sentence to link to `v<x.y.z>`
    #. Check `docs/conf.py` to make sure that `smv_tag_whitelist` will pick up the
       new release
    #. Update `docs/static_landing_page/` so that all the links point to the new release

#. Run the `release pipeline <https://dev.azure.com/responsibleai/fairlearn/_build?definitionId=60>`_

    #. Ensure that you have selected the correct release branch
    #. Run first on 'Test' which will upload to <https://test.pypi.org>
    #. Finally, run the release pipeline set to 'Production' which will upload to <https://pypi.org/>

#. On the release branch, place an annotated tag:

    :code:`git tag -a v<x.y.z> -m "v<x.y.z> release"`

#. Push the tag to GitHub:

    :code:`git push origin v<x.y.z>`
