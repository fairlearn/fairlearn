.. _about:

About
=====

.. _mission:

Mission
-------

Fairlearn is an open source project that seeks to empower data scientists and
developers to assess and improve fairness of AI systems.
The project aspires to include:

- A Python library of fairness metrics and mitigation algorithms
- Tools for (interactive) fairness assessment (e.g., plotting, dashboards, etc.)
- Educational materials (comprehensive user guide, detailed case studies,
  Jupyter notebooks, white papers, etc.)

Development of Fairlearn is firmly grounded in the understanding that fairness
in AI systems is a sociotechnical challenge.
Because there are many complex sources of unfairness&mdash;some societal and
some technical&mdash;it is not possible to fully "debias" a system or to
guarantee fairness.
Our goal is to enable humans to assess fairness-related harms, review the
impacts of different mitigation strategies and then make trade-offs
appropriate to their scenario.

Fairlearn is currently driven by Microsoft Research and Azure with a substantial
involvement of external contributors and an intention to eventually transition to a
community-driven open source project under a neutral organization.
The development and growth of Fairlearn are guided by the belief that meaningful progress
toward fairer AI systems requires input from a breadth of perspectives,
ranging from data scientists, developers, and business decision makers to the
people whose lives may be affected by the predictions of AI systems. 

.. _code_of_conduct:

Code of conduct
---------------

This project follows the
`GitHub community guidelines <https://help.github.com/en/github/site-policy/github-community-guidelines>`_.

.. _history:

Project history
---------------

Fairlearn was started in 2018 by Miro Dudik from Microsoft Research as a
Python package to accompany the research paper,
`A Reductions Approach to Fair Classification <http://proceedings.mlr.press/v80/agarwal18a/agarwal18a.pdf>`_.
The package provided a reduction algorithm for mitigating unfairness in binary
classification models&mdash;a setting that was commonly studied in the
machine learning community.
The paper and the Python package were well received, so Miro Dudik and Hanna
Wallach with their collaborators sought to translate the research into an industry context.
However, they discovered that practitioners typically need to address more
fundamental fairness issues before applying specific algorithms, and that
mitigating unfairness in binary classification models is a relatively rare use
case.
They also discovered that fairness assessment is a common need, along with
access to domain-specific guides to fairness metrics and unfairness mitigation
algorithms.
Additionally, many use cases take the form of regression or ranking, rather
than classification.
As a result of these insights, fairness assessment and use-case notebooks
became key components of Fairlearn.
Fairlearn also focuses on machine learning tasks beyond binary classification.

The project was greatly expanded in the second half of 2019 thanks to the
involvement of many contributors from Azure ML and Microsoft Research.
At that time, the project started to have regular releases.

.. _roadmap:

Project focus areas
-------------------

As an open-source project, Fairlearn strives to incorporate the best of
research and practice.
AI is a rapidly evolving field, and fairness in AI is all the more so.
We therefore encourage researchers, practitioners, and other stakeholders to
contribute fairness metrics and assessment tools, unfairness mitigation algorithms,
case studies and other educational materials to Fairlearn as we experiment,
learn, and evolve the project together.

Below we list the key areas that we prioritize in the short
and medium term, but we are happy to consider other directions
if they are aligned with the mission of Fairlearn and there is enough commitment
from the contributors. If you want to get involved, please
:ref:`reach out <communication>`. For concrete opportunities and
work in progress please review our `project boards <https://github.com/fairlearn/fairlearn/projects>`_.

Short and medium-term focus areas
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Last update: August 12, 2020*

#. *Decrease adoption barriers for the current assessment and mitigation tools in Fairlearn*

   - **Improve existing notebooks and create new notebooks** that
     `make fairness issues concrete <https://fairlearn.github.io/contributor_guide/contributing_example_notebooks.html>`_
     while demonstrating how to apply Fairlearn. To accelerate this process, we are experimenting with
     :ref:`weekly sessions <notebook_deep_dive>` where people can discuss
     individual example notebooks in detail.
     
   - **Improve the project documentation**: critique the current content, expand user guides,
     improve writing and examples, port the current Python documentation to the numpydoc format. 
     Convert existing example notebooks from `.ipynb` into `.py` files that
     `render properly on the website <https://fairlearn.github.io/auto_examples/notebooks/index.html>`_,
     while still being downloadable as `.py` or `.ipynb` files.

   - **Improve the usability, relevance, and look of Fairlearn website** with the audience of practitioners in mind.
     Engage by participanting in the discussions on developer calls or in the issues on the `project board about the Fairlearn
     website design <https://github.com/fairlearn/fairlearn/projects/9>`_. You can also give us feedback by filing
     a new issue.

   - **Improve the usability and relevance of fairness metrics** by
     `critiquing and improving the current metrics API <https://github.com/fairlearn/fairlearn-proposals/issues/12>`_,
     suggesting new metrics motivated by concrete use cases, and implementing the new metrics.
   
   - **Move towards compatibility with scikit-learn**: identify non-compatible aspects, improve code
     towards compatibility.
     While we aim for compatibility there may be aspects that are too
     restricting for Fairlearn, so this may need to be evaluated on a
     case-by-case basis.

#. *Grow and nurture a diverse community of contributors*
   
   - **Reach out** with feedback on what is working and what
     is not working; suggest how to improve things; point out where
     documentation, our processes or any other aspect of the projects create
     barriers of entry.

   - **Participate** in our :ref:`monthly developer calls <developer_calls>`.
     We are experimenting with adding :ref:`deep dive sessions <notebook_deep_dive>` around sociotechnical
     aspects of notebooks.
     We also work with universities to engage student contributors
     through course projects and are also open to other forms of collaboration&mdash;let us know if you
     are interested.

   - **Improve the Fairlearn website and documentation** with the audience of contributors in mind.
   
   - **Add tests and improve testing infrastructure.**
     
#. *Create metrics, assessment tools, and algorithms to cover more complex ML tasks*

   - **Create notebooks and use cases** that deal with
     `concrete fairness issues <https://fairlearn.github.io/contributor_guide/contributing_example_notebooks.html>`_
     in complex ML tasks including
     ranking, counterfactual estimation, text, computer vision, and speech.
   
   - **Lead and participate in contribution efforts** around under-researched, but practically relevant
     ML areas in ranking, counterfactual estimation, text, computer vision, and speech. These are likely
     to be mixed research / practice efforts and we expect substantial contributor commitment before
     embarking on these.

.. _governance:

Governance
----------
