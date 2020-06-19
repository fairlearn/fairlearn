.. _about:

About
=====

.. _mission:

Mission
-------

Fairlearn is an open source toolkit that seeks to empower data scientists and
developers to assess and improve fairness of their AI systems.
The toolkit aspires to include:

- Fairness metrics and unfairness mitigation algorithms
- Interactive dashboard for fairness assessment
- Educational materials (comprehensive user guide, detailed case studies,
  white papers, etc.)

Development of Fairlearn is firmly grounded in the understanding that fairness
in AI systems is a sociotechnical challenge.  Because there are many complex
sources of unfairness&mdash;some societal and some technical&mdash;it is not
possible to fully "debias" a system or to guarantee fairness.
Our goal is to enable humans to assess fairness-related harms, review the
impacts of different mitigation strategies and then make trade-offs
appropriate to their scenario.

Fairlearn is a community-driven open source project, to be shaped through
stakeholder engagement.  Its development and growth are guided by the belief
that meaningful progress toward fairer AI systems requires input from a
breadth of perspectives, ranging from data scientists, developers, and
business decision makers to the people whose lives may be affected by the
predictions of AI systems. 

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
classification models&mdash;a setting that was commonly studied in the machine
learning community.  The paper and the Python package were well received, so
Miro Dudik and Hanna Wallach wanted to translate the research into an industry
context.  However, they discovered that practitioners typically need to
address more fundamental fairness issues before they can use specific
algorithms, and that mitigating unfairness in binary classification models is
a relatively rare use case. They also discovered that fairness assessment is a
common need, along with access to domain-specific guides to fairness metrics
and unfairness mitigation algorithms. Additionally, many use cases take the
form of regression or ranking, rather than classification. As a result of
these insights, fairness assessment and use-case notebooks became key
components of Fairlearn. Fairlearn also focuses on machine learning tasks
beyond binary classification.

The project was greatly expanded in the second half of 2019 thanks to the
involvement of many contributors from Azure ML and Microsoft Research.
At that time, the project started to have regular releases.

.. _roadmap:

Project roadmap
---------------

As an open-source project, Fairlearn strives to incorporate the best of
research and practice.  AI is a rapidly evolving field, and fairness in AI is
all the more so. We therefore encourage researchers, practitioners, and other
stakeholders to contribute fairness metrics, unfairness mitigation algorithms,
and visualization capabilities to Fairlearn as we experiment, learn, and
evolve the project together.

There are many areas for future enhancement and growth. For example, Fairlearn
currently supports only *group fairness*&mdash;i.e., fairness with respect to
groups of people, such as those defined in terms of race, sex, age, or
disability status&mdash;and not other conceptualizations of fairness, such as
*individual fairness* or *counterfactual fairness.* Fairlearn also currently
includes only a limited set of unfairness mitigation algorithms, although we
note that these algorithms are not restricted to classification tasks. Besides
adding fairness metrics, conceptualizations of fairness, and unfairness
mitigation algorithms, we also hope that Fairlearn will expand to cover more
complex machine learning tasks in areas like counterfactual reasoning,
computer vision, and natural language processing.
We also anticipate integrating Fairlearn with interpretability tools, such as
`InterpretML <https://github.com/interpretml>`_.

Ultimately, we hope that Fairlearn will become more than a software
tool&mdash;a vibrant community and resource center that provides not only
code, but also resources like domain-specific guides for when and when not to
use different fairness metrics and unfairness mitigation algorithms.

Short-term roadmap
^^^^^^^^^^^^^^^^^^

*Last update: June 19, 2020*

Here we list some specific areas we would like to work on in the next few
weeks and months. If you want to get involved, please
:ref:`reach out <communication>`.  The inclusion of tasks in the list below
does not mean that we are committing to deliver them by a specific date.
It merely represents a current list of tasks that members of the community
work on. If you are interested to work on a topic that is not covered, please
reach out.

Tasks
~~~~~

#. *Improve the `fairlearn.metrics` API*
   More information in the
   `fairlearn-proposals <https://github.com/fairlearn/fairlearn-proposals/blob/master/api/METRICS.md>`_
   repository.

   **How you can help:** critique the proposal, suggest new metrics, work on
   implementing new metrics.

#. *Improve the project documentation*
   The Fairlearn website is still very new and raw with lots of gaps.
   `This proposal <https://github.com/fairlearn/fairlearn-proposals/pull/8>`_
   outlines the planned efforts, some of which have been completed. There are
   still massive gaps, particularly in the user guides.

   **How you can help:** critique the current content, improve writing and
   examples, port the current documentation to the numpydoc format.

#. *Improve example notebooks and add new ones*
   The repository started out with all notebooks in the `notebooks` directory
   as `.ipynb` files. These notebooks are now being
   converted into `.py` files that
   `render properly on the website <https://fairlearn.github.io/auto_examples/notebooks/index.html>`_,
   while still being downloadable as `.py` or `.ipynb` files.
   
   - Some of the notebooks may be better off as parts of the user guide.
   - As pointed out in various issues the existing notebooks have plenty of
     potential for improvement to better reflect the sociotechnical focus of Fairlearn.
     To accelerate this process, we experiment with weekly sessions
     where people can discuss individual example notebooks in detail.
   - There are only very few existing example notebooks. To cover a
     wider range of applications we need to work with stakeholders to develop
     more guides.

   **How you can help:** critique and improve existing notebooks, join the
   discussion sessions, create new notebooks, move the notebooks to the user
   guide.

#. *Nurture a diverse community of contributors*
   We would like to ensure that all feel welcome and can contribute in a way
   of their choosing. So far we have monthly developer calls.  We are
   experimenting with adding deep dive sessions around sociotechnical aspects
   of notebooks.  We also work with universities on engaging with student
   contributors through course projects.  Finally, we are working to improve
   documentation.
   
   **How you can help:** reach out with feedback on what is working and what
   is not working; suggest how to improve things; point out where
   documentation, our processes or any other aspect of the projects create
   barriers of entry.

#. *Move towards compatibility with scikit-learn*
   
   **How can you help:** help identify non-compatible aspects, improve code
   towards compatibility.

#. *Add regression to the exponentiated gradient algorithm*
   The exponentiated gradient algorithm originally worked only for binary
   classification.  There is a pull request that implements the extension to
   regression (and the bounded group loss fairness criterion).
   We intend to add the corresponding documentation before completing it.

   **How you can help:** review and improve the documentation once it's added
   to the repository

#. *Document and modularize the Fairlearn dashboard*
   The Fairlearn dashboard is currently not ready for contributions by
   the community.  Over the next few months, the Microsoft Research and Azure
   ML teams are working to properly open it for contributions similarly to all
   other parts of this project.  In the meantime, we have set up a working
   group for those that would like to work on UX design and HCI research
   within Fairlearn.

   **How you can help:** join the working group if you are interested in
   questions around UX design

.. _governance:

Governance
----------
