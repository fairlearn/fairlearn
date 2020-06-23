.. _contributing_example_notebooks:

Contributing example notebooks
------------------------------

To contribute new example notebooks please submit a pull request. We have
certain requirements that need to be satisfied to contribute a notebook. Those
include:

* Clear structure
* Appropriate framing of the problem while respecting that fairness is a
  fundamentally sociotechnical challenge
* Usage of at least some part of Fairlearn

Good examples of existing notebooks that abide by these requirements are:

* `Mitigating Disparities in Ranking from Binary Data <https://github.com/fairlearn/fairlearn/blob/master/notebooks/Mitigating%20Disparities%20in%20Ranking%20from%20Binary%20Data.ipynb>`_
* `Binary Classification with the UCI Credit-card Default Dataset <https://github.com/fairlearn/fairlearn/blob/master/notebooks/Binary%20Classification%20with%20the%20UCI%20Credit-card%20Default%20Dataset.ipynb>`_

Working with example notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example notebooks in Fairlearn are surfaced through the website's
`example notebook page <https://fairlearn.github.io/auto_examples/notebooks/index.html>`_.
This allows them to be rendered properly with output from all cells.

.. note:

    Rendering the Fairlearn dashboard is still an outstanding issue.

These notebooks are generated based on `.py` files in the
`examples/notebooks directory <https://github.com/fairlearn/fairlearn/tree/master/examples/notebooks>`_
of the repository. To do this yourself make sure to install sphinx and its
add-ons by running :code:`pip install -r requirements.txt` in the repository
root directory.

When working on the notebooks it can be tedious to make changes in the
notebook and then make the same change in the `.py` file. To make the
development process smoother we recommend using 
`Visual Studio Code <https://code.visualstudio.com/docs/python/jupyter-support>`_.
VS Code recognizes the lines starting with :code:`$ %%` as new cells.
Each cell can be executed individually by clicking on *Run Cell*, and VS Code
opens a *Python Interactive* tab to show the output.

.. image:: ../_static/images/vscode-jupyter.png

Note that the text portion of these notebooks should be written in
restructured text (ReST), not markdown, so that the sphinx documentation build
can render it nicely for the website. However, when downloading the `.ipynb`
file through the website certain parts of the ReST will look odd if viewed in
tools like Jupyter since it is meant to render markdown. For that reason try
to limit use of ReST directives (e.g., `.. note:`), internal links
(e.g., `:ref:`), and other functionality that won't render well in markdown.
