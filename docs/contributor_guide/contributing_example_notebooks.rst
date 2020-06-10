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
development process smoother we recommend using the
`jupytext <https://jupytext.readthedocs.io/>`_ package. Jupytext provides
the functionality to keep `.py` and `.ipynb` files in sync.
The synchronization is established automatically due to the jupytext.yml
configuration file in the notebooks directory. All you need to do is run

```bash
> jupytext --to notebook .\examples\notebooks\plot_binary_classification_uci_credit_card_default.py
[jupytext] Reading .\examples\notebooks\plot_binary_classification_uci_credit_card_default.py
[jupytext] Writing .\examples\notebooks\plot_binary_classification_uci_credit_card_default.ipynb
```
followed by `jupyter notebook`. Any changes you make to the `.ipynb` file
should now automatically be synced with the `.py` file.

.. note::

    Typical markdown cells in a jupyter notebook are written in ReST format
    to render properly on the webpage. That means the generated `.ipynb` file
    likely has strangely rendered markdown cells because it's trying to show
    ReSt-formatted content in markdown.

The generated `.ipynb` files are automatically ignored by git
(version control), so there's no danger of checking them in by accident.

You can also make changes to the `.py` file. The synchronization is goes both
ways.
