.. release_guide

Release Process
---------------

This page outlines the process for creating a new Fairlearn release.
The following steps must be performed on a clone of Fairlearn, not
a fork.

1. Create an annotated tag using the command at the branch point:

    :code:`git tag -a v<x.y.z> -m "Branch point for v<x.y.z> release"`

1. Push the tag to GitHub:

    :code:`git push origin v<x.y.z>`

1. Create the branch for the release:

    :code:`git checkout -b release/v<x.y.z>`

1. On the release branch:

    1. Update the version in `__init__.py` to `x.y.z`
    1. Update the version in the ReadMe

1. On the `main` branch:

    1. Update the version in `__init__.py` to `xy.z+1.dev0`
    1. Update the ReadMe to link to `v<x.y.z>`
    1. Check `docs/conf.py` to make sure that `smv_tag_whitelist` will pick up the
       new release
    1. Update `docs/static_landing_page/` so that all the links point to the new release

1. Run the `release pipeline <https://dev.azure.com/responsibleai/fairlearn/_build?definitionId=60>_`

    1. Ensure that you have selected the correct release branch
    1. Run first on 'Test' which will upload to <https://test.pypi.org>