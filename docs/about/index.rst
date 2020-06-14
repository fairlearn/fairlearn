.. _about:

About
=====

.. _mission:

Mission
-------


.. _code_of_conduct:

Code of conduct
---------------

This project follows the
`GitHub community guidelines <https://help.github.com/en/github/site-policy/github-community-guidelines>`_.

.. _roadmap:

Project roadmap
---------------

*Last update: June 14, 2020*

From a high-level perspective all tasks in the Fairlearn project are centered
around building a diverse community of contributors and users and providing
them with with tools and educational material to help them build fair machine
learning systems.

First, due to the sociotechnical nature of fairness in machine learning it is
crucial that the community is diverse. To grow and nurture this community
there are regular :ref:`developer_calls` and options for :ref:`communication`.
As an open source project we rely on the feedback from users to indicate
whether the educational materials are helpful and whether the toolkit is
useful.

Second, fairness is highly specific to the application context. This needs to
be reflected in the project documentation through appropriate examples.

Third, Fairlearn is currently restricted to standard classification and
regression scenarios. In the future we plan to expand to more complex machine
learning tasks such as ranking, text, counterfactual inference, computer
vision, and speech.

Fourth, the Fairlearn dashboard is still in an early stage. We hope to evolve
the user experience based on existing research and by conducting user studies.
Furthermore, bridging the related topics of fairness and interpretability is
a frequent request from users. We will explore potential collaborations with
`InterpretML <https://github.com/interpretml>`_.

Short-term roadmap
^^^^^^^^^^^^^^^^^^

While the above roadmap describes the high-level outline it can be useful to
have an overview over the ongoing and upcoming efforts from a more immediate
or short-term perspective. These tasks are planned to be addressed within the
next few weeks and months. If there is a particular area you would like to get
involved in please :ref:`reach out <communication>`. Similarly, if topics you
care about are missing from the list below please let us know and get
involved. The inclusion of tasks in the list below does not mean that we
commit to deliver them by a specific date. It merely represents a current list
of tasks that members of the community work on. 

#. Improving the `fairlearn.metrics` API
   More information in the
   `fairlearn-proposals <https://github.com/fairlearn/fairlearn-proposals/blob/master/api/METRICS.md>`_
   repository.
#. Improve project documentation
   The Fairlearn website is still very new and raw with lots of gaps.
   `This proposal <https://github.com/fairlearn/fairlearn-proposals/pull/8>`_
   outlines the planned efforts, some of which have been completed. There are
   still massive gaps, particularly in the user guides.
#. Adding and Improving example notebooks
   The repository started out with all notebooks in the `notebooks` directory
   as `.ipynb` files. As we transition to the website these notebooks are
   converted into `.py` files that
   `render properly on the website <https://fairlearn.github.io/auto_examples/notebooks/index.html>`_
   while still being downloadable as `.py` or `.ipynb` files.
   Some of the notebooks may be better off as user guides, so this decision
   needs to be made on a case-by-case basis.
   As pointed out in various issues the existing notebooks have plenty of
   potential for improvement to better reflect the focus on the sociotechnical
   context. To accelerate this process we will experiment with weekly sessions
   where people can discuss an individual example notebook in detail.
   Finally, there are only very few existing example notebooks. To cover a
   wider range of applications we need to work with stakeholders to develop
   more guides.
#. Nurturing a diverse community
   This includes various activities to make sure everyone is welcome and can
   contribute in a way of their choosing. So far we have monthly developer
   calls. We are experimenting with adding deep dive sessions to examine
   specific application examples as mentioned above. Beyond that, we worked
   with universities in the last few months to add functionality through
   projects over the course of a semester. Through these collaborations we
   added weekly office hours that the students found helpful. Perhaps this
   format could be useful for the project as a whole, either to discuss
   particular issues or code, or even just to connect with others in the
   community.
   Other than real-time interactions we need to remove obstacles for
   contributors to get involved. To some extent this ties into the
   documentation improvements mentioned above. But it also goes further in
   that we need to build the culture where everyone feels comfortable to
   report problems, and better yet empowered to fix them right away.
#. Compatibility with scikit-learn
   This has been a goal for some time now. We have moved closer to becoming
   compatible by using scikit-learn's input validation and other checks,
   conforming to naming conventions (both for `fit` and `predict` methods and
   naming of member variables with leading/trailing underscores).
   To ensure we don't break compatibility after getting there we need to add
   tests.
   To a certain extent this will always remain a focus of Fairlearn, since
   scikit-learn is evolving itself.
#. Adding regression to Exponentiated Gradient
   Exponentiated Gradient originally worked only for binary classification.
   The only missing piece to enable regression is already in a pull request.
   We intend to add the corresponding documentation before completing it.
#. Evolving the Fairlearn dashboard
   The existing Fairlearn dashboard is still in its original version.
   It lacks the capability to represent some of the metrics produced through
   `fairlearn.metrics`. This is partially addressed through a current pull
   request. However, there are many kinds of things that could be done with
   the dashboard beyond small incremental changes. For this reason a working
   group is set to think through, design, build, and research what the user
   experience for Fairlearn will look like.

.. _governance:

Governance
----------

