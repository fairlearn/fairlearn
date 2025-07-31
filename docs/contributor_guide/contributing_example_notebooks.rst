.. _contributing_example_notebooks:

Contributing example notebooks
------------------------------

We'd love to collaborate with anyone interested in describing scenarios for
using Fairlearn!

A good example notebook exhibits the following attributes:

1. **Deployment context**: Describes a real deployment context, not just a
   dataset.
2. **Real harms**: Focuses on real harms to real people :footcite:`blodgett2020languagepower`.
3. **Sociotechnical**: Models the Fairlearn team's value that fairness is a
   sociotechnical challenge.
   Avoids abstraction traps :footcite:`selbst2019fairness`.
4. **Substantiated**: Discusses trade-offs and compares alternatives.
   Describes why using particular Fairlearn functionalities makes sense.
5. **For developers**: Speaks the language of developers and data scientists.
   Considers real practitioner needs.
   Fits within the lifecycle of real practitioner work :footcite:`holstein2019improving`, :footcite:`madaio2020codesigning`.

Please keep these in mind when creating, discussing, and critiquing examples.

If you'd like to collaborate, please :ref:`reach out <communication>`.

Formatting of example notebooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example notebooks in Fairlearn are surfaced through the website's
:ref:`example notebook page <examples>`.
This allows them to be rendered properly with output from all cells.

These notebooks are generated based on `plot_*.py` files in
`percent format <https://jupytext.readthedocs.io/en/latest/formats.html#the-percent-format>`_
in the
`examples/notebooks directory <https://github.com/fairlearn/fairlearn/tree/main/examples/notebooks>`_
of the repository.
The filename *must* begin with `plot_` for the cell output to be rendered as a webpage.

The instructions on how to build the webpage are in the
:ref:`contributing_documentation` section.

To edit the notebook we recommend using
`Visual Studio Code <https://code.visualstudio.com/docs/python/jupyter-support>`_.
VS Code recognizes the lines starting with :code:`# %%` as new cells.
Each cell can be executed individually by clicking on *Run Cell*, and VS Code
opens a *Python Interactive* tab to show the output.

.. image:: ../_static/images/vscode-jupyter.png

If you prefer working with Jupyter simply open the `.py` file with Jupyter.
Changes made in Jupyter automatically show up in the `.py` file.

Note that the text portion of these notebooks should be written in
restructured text (ReST), not markdown, so that the sphinx documentation build
can render it nicely for the website. When downloading the `.ipynb` file through
the website the text portions will be in markdown due to automatic conversion from
ReST to markdown by sphinx-gallery. This currently only works for basic ReST
functionality, so try to limit use of ReST directives (e.g., `.. note::`),
internal links (e.g., `:ref:`), and other functionality that won't render well
in markdown.

.. topic:: References

   .. footbibliography::
