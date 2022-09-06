.. _contributing_documentation:

Contributing documentation
--------------------------

Documentation is formatted in restructured text (ReST) and the website is
built using `Sphinx <https://www.sphinx-doc.org/en/master/>`_ and some of its
extensions. Specifically, the website is available for all our releases to
allow users to check the documentation of the version of the package that they
are using.

To contribute, make sure to install sphinx and its
add-ons by running

.. code-block::

    python scripts/install_requirements.py --pinned False

in the repository root directory.
You may also need to `install Pandoc <https://pandoc.org/installing.html>`_.
You can contribute updates to existing documentation by navigating to the
relevant part of the repository (typically in the `docs` directory), and
editing the restructured text files (`.rst`) corresponding to your updates.

To build the webpage run the following command from the repository root
directory:

.. code-block::

    python -m sphinx -v -b html -n -j auto docs docs/_build/html

or use the shortcut

.. code-block::

        make doc

This will generate the website in the directory mentioned at the end of the
command. Rerunning this after making changes to individual files only
rebuilds the changed pages, so the build time should be a lot shorter.

You can check that the document(s) render properly by inspecting the HTML with
the following commands:

.. code-block::

    start docs/_build/html/index.html
    start docs/_build/html/quickstart.html
    ...
    start docs/_build/html/auto_examples/plot_*.html

The above code block works for Windows users.
For MacOS users, use :code: `open docs/_build/html/index.html`
For Linux users, use :code: `xdg-open docs/_build/html/index.html`

.. note::

    The rendered HTML files can be explored in any file explorer and then opened
    using a browser (e.g., Chrome/Firefox/Safari).

:code:`plot_*` can be replaced with any of the notebooks in the
:code:`auto_examples` folder. To view your changes, simply navigate to the
relevant part of the website and check that your updates render properly
and links work as expected.

Note that some changes to documentation may involve modifying several files
(i.e: index files, other documents in which the current one should be linked).
Be sure to modify all of the relevant documents and use the commands above to
ensure that they all render properly.

.. note::

    If you encounter problems with the documentation build (locally or in your
    pull request) and need help, simply @mention
    :code:`@fairlearn/fairlearn-maintainers` to get help.

Building the website for all versions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our documentation build runs for each pull request and upon merging pull
requests. There should not be a need to run this locally except for very rare
cases.

To fully build the website for all versions use the following script:

.. code-block::

    make doc

The comprehensive set of commands to build the website is in our CircleCI
configuration file in the `.circleci` directory of the repository.
