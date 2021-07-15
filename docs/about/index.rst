.. _about:

About Us
========

.. _mission:

Mission
-------

Fairlearn is an open-source, community-driven project to help data scientists
improve fairness of AI systems.

The project aspires to include:

- A Python library for fairness assessment and improvement (fairness metrics, 
  mitigation algorithms, plotting, etc.)
- Educational resources covering organizational and technical processes for
  unfairness mitigation (comprehensive user guide, detailed case studies, 
  Jupyter notebooks, white papers, etc.)

Development of Fairlearn is firmly grounded in the understanding that fairness
in AI systems is a sociotechnical challenge.
Because there are many complex sources of unfairness --- some societal and
some technical --- it is not possible to fully "debias" a system or to
guarantee fairness.
Our goal is to enable humans to assess fairness-related harms, review the
impacts of different mitigation strategies and then make trade-offs
appropriate to their scenario.

Fairlearn is a community-driven open source project!
The development and growth of Fairlearn are guided by the belief that
meaningful progress toward fairer AI systems requires input from a breadth
of perspectives, ranging from data scientists, developers, and business
decision makers to the people whose lives may be affected by the predictions
of AI systems. 

.. _code_of_conduct:

Code of conduct
---------------

Fairlearn follows the
`Fairlearn Organization's Code of Conduct <https://github.com/fairlearn/governance/blob/main/code-of-conduct.md>`_.

.. _roadmap:

Project focus areas
-------------------

*Last update: May 16, 2021*

As an open-source project, Fairlearn strives to incorporate the best of
research and practice.
AI is a rapidly evolving field, and fairness in AI is all the more so.
We therefore encourage researchers, practitioners, and other stakeholders to
contribute fairness metrics and assessment tools, unfairness mitigation
algorithms, case studies and other educational materials to Fairlearn as we
experiment, learn, and evolve the project together.

Below we list the key areas that we prioritize in the short
and medium term, but we are happy to consider other directions
if they are aligned with the mission of Fairlearn and there is enough commitment
from the contributors. If you want to get involved, please
:ref:`reach out <communication>`. For concrete opportunities and
work in progress please review our
`issues <https://github.com/fairlearn/fairlearn/issues>`_.

#. *Decrease adoption barriers for the current assessment and mitigation tools in Fairlearn*

   - **Improve existing use cases and create new ones** that
     `make fairness issues concrete <https://fairlearn.github.io/contributor_guide/contributing_example_notebooks.html>`_.
     These use cases may or may not use the Fairlearn package.
     To accelerate this process, we are experimenting with
     :ref:`weekly sessions <community_calls>` where people can discuss ideas,
     ongoing projects, and individual example notebooks in detail.
     
   - **Improve the project documentation**: critique the current content,
     expand user guides, improve writing and examples, port the current Python
     documentation to the numpydoc format. 
     Convert existing example notebooks from `.ipynb` into `.py` files that
     `render properly on the website <https://fairlearn.github.io/auto_examples/notebooks/index.html>`_,
     while still being downloadable as `.py` or `.ipynb` files.

   - **Improve the usability, relevance, and look of Fairlearn website**
     with the audience of practitioners in mind.
     Engage by participating in the discussions on
     `community calls <community_calls>`_ or in the corresponding
     `discussions <https://github.com/fairlearn/fairlearn/discussions>`_.
     You can also give us feedback by filing a new issue.

   - **Improve the usability and relevance of fairness metrics** by
     critiquing and improving the current metrics API, suggesting new metrics
     motivated by concrete use cases, and implementing the new metrics.
     The issues page
     `contains several metrics tasks <https://github.com/fairlearn/fairlearn/issues?q=is%3Aissue+is%3Aopen+metric>`_. 
   
   - **Move towards compatibility with scikit-learn**:
     identify incompatible aspects, improve code towards compatibility.
     While we aim for compatibility there may be aspects that are too
     restricting for Fairlearn, so this may need to be evaluated on a
     case-by-case basis.

#. *Grow and nurture a diverse community of contributors*
   
   - **Reach out** with feedback on what is working and what
     is not working; suggest how to improve things; point out where
     documentation, our processes or any other aspect of the projects create
     barriers of entry.

   - **Participate** in our :ref:`weekly community calls <community_calls>`.
     We also work with universities to engage student contributors
     through course projects and are also open to other forms of
     collaboration --- let us know if you are interested.

   - **Improve the Fairlearn website and documentation**.
     See the contributor guide on
     :ref:`how to contribute to our docs <contributing_documentation>`.
   
   - **Add tests and improve testing infrastructure.**
     
#. *Create metrics, assessment tools, and algorithms to cover more complex ML tasks*

   - **Create notebooks and use cases** that deal with
     :ref:`concrete fairness issues <contributing_example_notebooks>`
     in complex ML tasks including ranking, counterfactual estimation, text,
     computer vision, speech, etc.
   
   - **Lead and participate in contribution efforts**
     around under-researched, but practically relevant ML areas in ranking,
     counterfactual estimation, text, computer vision, speech, etc.
     These are likely to be mixed research / practice efforts and we expect
     substantial contributor commitment before embarking on these.

.. _governance:

Governance
----------

Fairlearn is a project of the
`Fairlearn Organization <https://github.com/fairlearn/governance/blob/main/ORG-GOVERNANCE.md>`_
and follows the
`Fairlearn Organization's Project Governance <https://github.com/fairlearn/governance/blob/main/PROJECT-GOVERNANCE.md>`_.

.. _maintainers:

Maintainers
^^^^^^^^^^^

The maintainers of the Fairlearn project are

- `Adrin Jalali <https://github.com/adrinjalali>`_
- `Hilde Weerts <https://github.com/hildeweerts>`_
- `Michael Madaio <https://github.com/mmadaio>`_
- `Miro Dudik <https://github.com/MiroDudik>`_
- `Richard Edgar <https://github.com/riedgar-ms>`_
- `Roman Lutz <https://github.com/romanlutz>`_

.. _history:

Project history
---------------

Fairlearn was started in 2018 by Miro Dudik from Microsoft Research as a
Python package to accompany the research paper,
`A Reductions Approach to Fair Classification <http://proceedings.mlr.press/v80/agarwal18a/agarwal18a.pdf>`_.
The package provided a reduction algorithm for mitigating unfairness in binary
classification models --- a setting that was commonly studied in the
machine learning community.
The paper and the Python package were well received, so Miro Dudik and Hanna
Wallach with their collaborators sought to translate the research into an
industry context.
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

In 2021 Fairlearn adopted
`neutral governance <https://github.com/fairlearn/governance>`_
and since then the project is completely community-driven.

Citing Fairlearn
----------------

If you wish to cite Fairlearn in your work, please use the following:

.. code ::

    @techreport{bird2020fairlearn,
        author = {Bird, Sarah and Dud{\'i}k, Miro and Edgar, Richard and Horn, Brandon and Lutz, Roman and Milan, Vanessa and Sameki, Mehrnoosh and Wallach, Hanna and Walker, Kathleen},
        title = {Fairlearn: A toolkit for assessing and improving fairness in {AI}},
        institution = {Microsoft},
        year = {2020},
        month = {May},
        url = "https://www.microsoft.com/en-us/research/publication/fairlearn-a-toolkit-for-assessing-and-improving-fairness-in-ai/",
        number = {MSR-TR-2020-32},
    }

Frequently asked questions
--------------------------

See our :ref:`faq` page.
