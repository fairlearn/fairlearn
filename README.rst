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

-  `What we mean by *fairness* <#what-we-mean-by-fairness>`__
-  `Overview of Fairlearn <#overview-of-fairlearn>`__
-  `Getting started with fairlearn <#getting-started-with-fairlearn>`__
-  `Maintainers <#maintainers>`__
-  `Code of conduct <#code-of-conduct>`__
-  `Issues <#issues>`__

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
~~~~~~~~~~~~~~~~

Check out our in-depth `guide on the Fairlearn metrics <https://fairlearn.org/main/user_guide/assessment>`__.

Fairlearn algorithms
~~~~~~~~~~~~~~~~~~~

For an overview of our algorithms please refer to our
`website <https://fairlearn.org/main/user_guide/mitigation/index.html>`__.

Getting Started with Fairlearn
-----------------------------

First steps
~~~~~~~~~~~

- Install via pip: ``pip install fairlearn``

- Visit the `Quickstart guide <https://fairlearn.org/main/quickstart.html>`__.

- **Learning Resources**:

  - Read the comprehensive `user guide <https://fairlearn.org/main/user_guide/index.html>`__.

  - Look through the `example notebooks <https://fairlearn.org/main/auto_examples/index.html>`__.


For Users & Practitioners
~~~~~~~~~~~~~~~~~~~~~~~~

- Browse the `example gallery <https://fairlearn.org/main/auto_examples/index.html>`__. 
Please note that notebooks downloaded from `main` may not be compatible with pip-installed versions.

- Check the `API reference <https://fairlearn.org/main/api_reference/index.html>`__.

- **Get Help**:

  - Ask questions on `Stack Overflow <https://stackoverflow.com/questions/tagged/fairlearn>`__ with tag ``fairlearn``.

  - Join the `Discord community <https://discord.gg/R22yCfgsRn>`__ for discussions.

For Contributors
~~~~~~~~~~~~~~~

- Read the `contributor guide <https://fairlearn.org/main/contributor_guide/index.html>`__.

- Check out the `good first issues <https://github.com/fairlearn/fairlearn/labels/good%20first%20issue>`__.

- Follow the `development process <https://fairlearn.org/main/contributor_guide/development_process.html>`__.

- Join the `Discord <https://discord.gg/R22yCfgsRn>`__ for contributor discussions. Please use the ``#development`` channel.


Maintainers
-----------

A list of current maintainers is
`on our website <https://fairlearn.org/main/about/index.html>`__.

Code of conduct
---------------
Fairlearn follows the `Fairlearn Organization's Code of Conduct <https://github.com/fairlearn/governance/blob/main/code-of-conduct.md>`__.

Issues
------

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
