|MIT license| |PyPI| |Discord| |StackOverflow|

Fairlearn
=========

Fairlearn is a Python package that empowers developers of artificial
intelligence (AI) systems to assess their system's fairness and mitigate
any observed unfairness issues. Fairlearn contains mitigation algorithms
as well as metrics for model assessment. Besides the source code, this
repository also contains Jupyter notebooks with examples of Fairlearn
usage.

Website: https://fairlearn.org/

-  `Current release <#current-release>`__
-  `What we mean by *fairness* <#what-we-mean-by-fairness>`__
-  `Overview of Fairlearn <#overview-of-fairlearn>`__
-  `Fairlearn metrics <#fairlearn-metrics>`__
-  `Fairlearn algorithms <#fairlearn-algorithms>`__
-  `Install Fairlearn <#install-fairlearn>`__
-  `Usage <#usage>`__
-  `Contributing <#contributing>`__
-  `Maintainers <#maintainers>`__
-  `Issues <#issues>`__

Current release
---------------

-  The current stable release is available on
   `PyPI <https://pypi.org/project/fairlearn/>`__.

-  Our current version may differ substantially from earlier versions.
   Users of earlier versions should visit our
   `version guide <https://fairlearn.org/main/user_guide/installation_and_version_guide/version_guide.html>`_
   to navigate significant changes and find information on how to migrate.

What we mean by *fairness*
--------------------------

An AI system can behave unfairly for a variety of reasons. In Fairlearn,
we define whether an AI system is behaving unfairly in terms of its
impact on people – i.e., in terms of harms. We focus on two kinds of
harms:

-  *Allocation harms.* These harms can occur when AI systems extend or
   withhold opportunities, resources, or information. Some of the key
   applications are in hiring, school admissions, and lending.

-  *Quality-of-service harms.* Quality of service refers to whether a
   system works as well for one person as it does for another, even if
   no opportunities, resources, or information are extended or withheld.

We follow the approach known as **group fairness**, which asks: *Which
groups of individuals are at risk for experiencing harms?* The relevant
groups need to be specified by the data scientist and are application
specific.

Group fairness is formalized by a set of constraints, which require that
some aspect (or aspects) of the AI system's behavior be comparable
across the groups. The Fairlearn package enables assessment and
mitigation of unfairness under several common definitions. To learn more
about our definitions of fairness, please visit our
`user guide on Fairness of AI Systems <https://fairlearn.org/main/user_guide/fairness_in_machine_learning.html#fairness-of-ai-systems>`__.

    *Note*: Fairness is fundamentally a sociotechnical challenge. Many
    aspects of fairness, such as justice and due process, are not
    captured by quantitative fairness metrics. Furthermore, there are
    many quantitative fairness metrics which cannot all be satisfied
    simultaneously. Our goal is to enable humans to assess different
    mitigation strategies and then make trade-offs appropriate to their
    scenario.

Overview of Fairlearn
---------------------

The Fairlearn Python package has two components:

-  *Metrics* for assessing which groups are negatively impacted by a
   model, and for comparing multiple models in terms of various fairness
   and accuracy metrics.

-  *Algorithms* for mitigating unfairness in a variety of AI tasks and
   along a variety of fairness definitions.

Fairlearn metrics
~~~~~~~~~~~~~~~~~

Check out our in-depth `guide on the Fairlearn
metrics <https://fairlearn.org/main/user_guide/assessment>`__.

Fairlearn algorithms
~~~~~~~~~~~~~~~~~~~~

For an overview of our algorithms please refer to our
`website <https://fairlearn.org/main/user_guide/mitigation/index.html>`__.

Install Fairlearn
-----------------

For instructions on how to install Fairlearn check out our `Quickstart
guide <https://fairlearn.org/main/quickstart.html>`__.

Usage
-----

For common usage refer to the `Jupyter notebooks <https://fairlearn.org/main/auto_examples/index.html>`__ and
our `user guide <https://fairlearn.org/main/user_guide/index.html>`__.
Please note that our APIs are subject to change, so notebooks downloaded
from ``main`` may not be compatible with Fairlearn installed with
``pip``. In this case, please navigate the tags in the repository (e.g.
`v0.7.0 <https://github.com/fairlearn/fairlearn/tree/v0.7.0>`__) to
locate the appropriate version of the notebook.

Contributing
------------

To contribute please check our `contributor
guide <https://fairlearn.org/main/contributor_guide/index.html>`__.

Maintainers
-----------

A list of current maintainers is
`on our website <https://fairlearn.org/main/about/index.html>`__.

Issues
------

Usage Questions
~~~~~~~~~~~~~~~

Pose questions and help answer them on `Stack
Overflow <https://stackoverflow.com/questions/tagged/fairlearn>`__ with
the tag ``fairlearn`` or on
`Discord <https://discord.gg/R22yCfgsRn>`__.

Regular (non-security) issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Issues are meant for bugs, feature requests, and documentation
improvements. Please submit a report through
`GitHub issues <https://github.com/fairlearn/fairlearn/issues>`__.
A maintainer will respond promptly as appropriate.

Maintainers will try to link duplicate issues when possible.

Reporting security issues
~~~~~~~~~~~~~~~~~~~~~~~~~

To report security issues please send an email to
``fairlearn-internal@python.org``.

.. |MIT license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/fairlearn/fairlearn/blob/main/LICENSE
.. |PyPI| image:: https://img.shields.io/pypi/v/fairlearn?color=blue
   :target: https://pypi.org/project/fairlearn/
.. |Discord| image:: https://img.shields.io/discord/840099830160031744
   :target: https://discord.gg/R22yCfgsRn
.. |StackOverflow| image:: https://img.shields.io/badge/StackOverflow-questions-blueviolet
   :target: https://stackoverflow.com/questions/tagged/fairlearn
