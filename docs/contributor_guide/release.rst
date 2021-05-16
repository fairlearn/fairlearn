.. release_guide

Release Process
---------------

This page outlines the process for creating a new Fairlearn release.
The following steps assume git remote's `origin` points to
`fairlearn/fairlearn` (in practical terms, that the work is being
done on a clone of `fairlearn/fairlearn` and not on a fork).

#. Create the branch for the release:

    :code:`git checkout -b release/v<x.y.z>`

#. Push the branch to GitHub:

    :code:`git push -u origin release/v<x.y.z>`

#. On the release branch, create a PR to:

    #. Update the version in `__init__.py` to `x.y.z`
    #. Update the version in the ReadMe

#. Merge that PR.

#. Run the `release pipeline <https://dev.azure.com/responsibleai/fairlearn/_build?definitionId=60>`_

    #. Ensure that you have selected the correct release branch
    #. Run first on 'Test' which will upload to <https://test.pypi.org>
    #. Finally, run the release pipeline set to 'Production' which will upload to <https://pypi.org/>

#. On the release branch, place an annotated tag:

    :code:`git tag -a v<x.y.z> -m "v<x.y.z> release"`

#. Push the tag to GitHub:

    :code:`git push origin v<x.y.z>`

#. On `GitHub's release page <https://github.com/fairlearn/fairlearn/releases>`_
   there should be a new release named `v<x.y.z>`.
   Open it and post the changes from CHANGES.md into the description, then hit "publish".

#. On the `main` branch, create a PR to:

    #. Update the version in `__init__.py` to `x.y.z+1.dev0`
    #. Update the 'current stable release' sentence in the ReadMe to link to `v<x.y.z>`
    #. Ensure that `smv_tag_whitelist` in `docs/conf.py` will pick up the
       new release
    #. Update `docs/static_landing_page/` so that all the links point to the new release
   
.. note::
    Make sure to add a note to this second PR:
    "Do not merge until the release is completed. Otherwise a new website will
    be published that points to the new version which doesn't exist yet." 

#. Merge that PR.
