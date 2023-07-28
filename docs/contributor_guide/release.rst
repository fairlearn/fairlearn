.. release_guide

Release Process
---------------

This page outlines the process for creating a new Fairlearn release.
The following steps assume git remote's `origin` points to
`fairlearn/fairlearn`.

#. Ensure the maintainers listed in `scripts/generate_maintainers_table.py`
   are up to date. Run `python scripts/generate_maintainers_table.py` from the
   repository root directory. If the generated file
   `docs/about/maintainers.rst` does not change you can proceed with the next
   step. Otherwise, create a PR to update the generated maintainers file on
   the `main` branch. Proceed only when the PR is merged.

#. Check the `docs/user_guide/installation_and_version_guide` for a file
   related to the release. Make sure formatting and contents are correct and
   create a summary of the highlights at the top of the file. Create a PR
   for this and merge it before proceeding.

#. If this is a non-patch release:

    #. Create a new branch:

        :code:`git checkout -b release/v<x.y>.X`

    #. Push the branch to GitHub:

        :code:`git push -u origin release/v<x.y>.X`

       You may need to temporarily add an exception to the 
       `branch protection rules <https://github.com/fairlearn/fairlearn/settings/branches>`_ 
       by adding a new branch protection rule for `release/v<x.y>.X`.

#. On the release branch, create a PR to update the version in `__init__.py`
   to `x.y.z` (where `z=0` for the first release from a branch)

#. Merge that PR.

#. Run the `Release Wheel workflow on GitHub <https://github.com/fairlearn/fairlearn/actions/workflows/release-wheel.yml>`_

.. note::
    Ensure that you have selected the correct release branch

#. On the release branch, place an annotated tag:

    :code:`git tag -a v<x.y.z> -m "v<x.y.z> release"`

#. Push the tag to GitHub:

    :code:`git push origin v<x.y.z>`

#. On `GitHub's release page <https://github.com/fairlearn/fairlearn/releases>`_,
   draft a new release. Choose the new tag, title the release `v<x.y.z>`, 
   and post the changes from the release file within `docs/user_guide/installation_and_version_guide` 
   into the description, then hit "publish".

#. On the `main` branch, create a PR to:

    #. Update the version in `__init__.py` to `x.y+1.z.dev0`
    #. Update the version in `docs/static_landing_page/js/landing_page.js`
       so that all the links point to the new release
    #. Update the `docs/_static/versions.json` file
    #. Create a new file `vx.y+1.z.rst` in `docs/user_guide/installation_and_version_guide`
   
.. note::
    Make sure to add a note to this PR:
    "Do not merge until the release is completed. Otherwise a new website will
    be published that points to the new version which doesn't exist yet." 

#. Merge that PR.
