.. _contributing_example_notebooks:

Contributing example notebooks
------------------------------

We'd love to collaborate with anyone interested in describing scenarios for
using Fairlearn!

A good example notebook exhibits the following attributes:

1. **Deployment context**: Describes a real deployment context, not just a
   dataset.
2. **Real harms**: Focuses on real harms to real people.
   See `Blodget et al. (2020) <https://arxiv.org/abs/2005.14050>`_.
3. **Sociotechnical**: Models the Fairlearn team's value that fairness is a
   sociotechnical challenge.
   Avoids abstraction traps.
   See `Selbst et al. (2020) <https://andrewselbst.files.wordpress.com/2019/10/selbst-et-al-fairness-and-abstraction-in-sociotechnical-systems.pdf>`_.
4. **Substantiated**: Discusses trade-offs and compares alternatives.
   Describes why using particular Fairlearn functionalities makes sense if
   Fairlearn is used.
5. **For developers**: Speaks the language of developers and data scientists.
   Considers real practitioner needs.
   Fits within the lifecycle of real practitioner work.
   See `Holstein et al (2019) <https://arxiv.org/pdf/1812.05239.pdf>`_,
   `Madaio et al. (2020) <http://www.jennwv.com/papers/checklists.pdf>`_.

Please keep these in mind when creating, discussing, and critiquing examples.

Process
^^^^^^^

All current efforts are tracked through items in the corresponding
`Project Board <https://github.com/fairlearn/fairlearn/projects/3>`_.
If you'd like to suggest a different kind of use case, please
`open a new issue <https://github.com/fairlearn/fairlearn/issues/new/choose>`_
if you cannot find a similar one that is already tracked in the
`issue tracker <https://github.com/fairlearn/fairlearn/issues>`_.
If you'd like to collaborate with others from the community, please
:ref:`reach out <communication>` to share your idea.

In the issue, make sure to outline your goals, describe your audience, and
provide a high-level overview of what you would like to show.
Note that the examples do not need to contain code.
In some application contexts it may be preferable not to build a
technical system, so a textual explanation is perfectly acceptable.
All contributions should abide by the guidelines outlined above, though.

Once the issue is on GitHub, members from the community can respond with
questions, comments, and perhaps express their interest in fixing the
issue. Once the preliminary questions are sorted out, somebody can
`open a pull request <https://github.com/fairlearn/fairlearn/compare>`_.
Everyone from the community is encouraged to review the pull request and
provide feedback. The acceptance process is identical to all other
contributions as outlined :ref:`here <development_process>`, except that
the additional criteria at the top of this page apply additionally.

Whether and how project members communicate is up to the members themselves.
We especially encourage asynchronous means of communication to be inclusive
of people in all time zones.
This could include any of the following:

- a GitHub `discussion <https://github.com/fairlearn/fairlearn/discussions>`_
  or `issue <https://github.com/fairlearn/fairlearn/issues>`_
- a dedicated channel on [Gitter](https://gitter.im/fairlearn) in the
  Fairlearn community
- email
- recurring or ad-hoc meetings

If the project group would like to discuss their project with the community
we can use one of our
:ref:`community calls <community_calls>` for this purpose.

If you think we can improve any parts of this process or for feedback please
:ref:`reach out <communication>`.

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
functionality, so try to limit use of ReST directives (e.g., `.. note:`),
internal links (e.g., `:ref:`), and other functionality that won't render well
in markdown.
