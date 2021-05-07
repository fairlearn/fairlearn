.. _contributing_documentation:

Contributing documentation
------------------------------

Documentation is formatted in restructured text (ReST). To contribute, make sure to install sphinx and its
add-ons by running :code:`python scripts/install_requirements.py --pinned False` in the repository
root directory. You may also need to `install Pandoc <https://pandoc.org/installing.html>`_. You can contribute
updates to existing documentation by navigating to the relevant part of the repository, and editing the restructured
text files corresponding to your updates.

To build the webpage run the following command from the repository root
directory:

.. code::

    python -m sphinx -v -b html -n -j auto docs docs/_build/html

Rerunning this after making changes to individual files only rebuilds the
changed pages, so the build time should be a lot shorter.

You can check that the document(s) render properly 
by launching the HTML with the following command: 

.. code::

    start docs/_build/html/auto_examples/plot_*.html

plot_* can be replaced with any of the notebooks in the auto_examples folder. To view the 
change you've made to the documentation, simply navigate to the relevant part of the website and
check that the updates you've made render properly. 

Note that some changes to documentation may involve modifying several files (i.e: index files, other documents
in which the current one should be linked). Be sure to modify all of the relevant documents and use the commands 
above to ensure that they all render properly. 
