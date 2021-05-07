Development process
-------------------

Development happens against the :code:`main` branch following the
`GitHub flow model <https://guides.github.com/introduction/flow/>`_.
Contributors should use their own forks of the repository. In their fork, they
create feature branches off of :code:`main`, and their pull requests should
target the :code:`main` branch. Maintainers are responsible for prompt
review of pull requests.

Pull requests against :code:`main` trigger automated tests that are run
through Azure DevOps, GitHub Actions, and CircleCI. Additional test suites are
run periodically. When adding new code paths or features, tests are a
requirement to complete a pull request. They should be added in the
:code:`test` directory.

Documentation should be provided with pull requests that add or change
functionality. This includes comments in the code itself, docstrings, and user
guides. For exceptions to this rule the pull request author should coordinate
with a maintainer. For changes that fix bugs, add new features, change APIs,
etc., i.e., for changes that are relevant to developers and/or users please
also add an entry in CHANGES.md in the section corresponding to the *next*
release, since that's where your change will be included.
If you're a new contributor please also add yourself to AUTHORS.md.

Docstrings should follow
`numpydoc format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
This is a `recent decision by the community <https://github.com/fairlearn/fairlearn/issues/314>`_.
The new policy is to update docstrings that a PR touches, as opposed to
changing all the docstrings in one PR.

Advanced installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While working on Fairlearn itself you may want to install it in editable mode.
This allows you to test the changed functionality. First, clone the repository
locally via

.. code-block::

    git clone git@github.com:fairlearn/fairlearn.git

To install in editable mode using :code:`pip` run 

.. code-block::

    pip install -e .

from the repository root path.

To verify that the code works as expected run

.. code-block::

    python ./scripts/install_requirements.py --pinned False
    python -m pytest -s ./test/unit

Fairlearn currently includes plotting functionality that requires the
:code:`matplotlib` package to be installed. Since this is for a niche use case
Fairlearn comes without :code:`matplotlib` by default. To install Fairlearn
with its full feature set simply append :code:`customplots` to the install
command

.. code-block::

    pip install -e .[customplots]

Note that the Fairlearn dashboard is built using nodejs and requires
additional steps. To build the Fairlearn dashboard after making changes to it,
`install Yarn <https://yarnpkg.com/lang/en/docs/install>`_, and then run the
`widget build script <https://github.com/fairlearn/fairlearn/tree/main/scripts/build_widget.py>`_.

The Requirements Files
""""""""""""""""""""""

The prerequisites for Fairlearn are split between three separate files:

    -  `requirements.txt <https://github.com/fairlearn/fairlearn/blob/main/requirements.txt>`_
       contains the prerequisites for the core Fairlearn package

    -  `requirements-customplots.txt <https://github.com/fairlearn/fairlearn/blob/main/requirements-customplots.txt>`_
       contains additional prerequisites for the :code:`[customplots]` extension for Fairlearn

    -  `requirements-dev.txt <https://github.com/fairlearn/fairlearn/blob/main/requirements-dev.txt>`_ contains
       the prerequisites for Fairlearn development (such as flake8 and pytest)

The `requirements.txt <https://github.com/fairlearn/fairlearn/blob/main/requirements.txt>`_
and
`requirements-customplots.txt <https://github.com/fairlearn/fairlearn/blob/main/requirements-customplots.txt>`_
files are consumed
by `setup.py <https://github.com/fairlearn/fairlearn/blob/main/setup.py>`_ to specify the dependencies to be
documented in the wheel files.
To help simplify installation of the prerequisites, we have the
`install_requirements.py <https://github.com/fairlearn/fairlearn/blob/main/scripts/install_requirements.py>`_
script which runs :code:`pip install` on all three of the above files.
This script will also optionally pin the requirements to any lower bound specified (by changing any
occurrences of :code:`>=` to :code:`==` in each file).

.. _onboarding-guide:

.. raw:: html

    <details id="onboarding-guide">
    <summary>
    <strong>
    <em>

Onboarding guide for users of version 0.2 or earlier

.. raw:: html

    </em>
    </strong>
    </summary>

Up to version 0.2, Fairlearn contained only the exponentiated gradient method.
The Fairlearn repository now has a more comprehensive scope and aims to
incorporate other methods. The same exponentiated gradient technique is now
the class :code:`fairlearn.reductions.ExponentiatedGradient`. While in the past
exponentiated gradient was invoked via

.. code-block::

    import numpy as np
    from fairlearn.classred import expgrad
    from fairlearn.moments import DP

    estimator = LogisticRegression()  # or any other estimator
    exponentiated_gradient_result = expgrad(X, sensitive_features, y, estimator, constraints=DP())
    positive_probabilities = exponentiated_gradient_result.best_classifier(X)
    randomized_predictions = (positive_probabilities >= np.random.rand(len(positive_probabilities))) * 1

the equivalent operation is now

.. code-block::

    from fairlearn.reductions import ExponentiatedGradient, DemographicParity

    estimator = LogisticRegression()  # or any other estimator
    exponentiated_gradient = ExponentiatedGradient(estimator, constraints=DemographicParity())
    exponentiated_gradient.fit(X, y, sensitive_features=sensitive_features)
    randomized_predictions = exponentiated_gradient.predict(X)


Please open a `new issue <https://github.com/fairlearn/fairlearn/issues>`_ if
you encounter any problems.

.. raw:: html

    </details>

Investigating automated test failures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For every pull request to :code:`main` with automated tests, you can check
the logs of the tests to find the root cause of failures. Our tests currently
run through Azure Pipelines with steps for setup, testing, and teardown. The
:code:`Checks` tab of a pull request contains a link to the
`Azure Pipelines page <dev.azure.com/responsibleai/fairlearn/_build/results>`_),
where you can review the logs by clicking on a specific step in the automated
test sequence. If you encounter problems with this workflow, please reach out
through `GitHub issues <https://github.com/fairlearn/fairlearn/issues>`_.

To run the same tests locally, find the corresponding pipeline definition (a
:code:`yml` file) in the :code:`devops` directory. It either directly contains
the command to execute the tests (usually starting with
:code:`python -m pytest`) or it refers to a template file with the command.

Building the website
^^^^^^^^^^^^^^^^^^^^

The website is built using `Sphinx <https://www.sphinx-doc.org/en/master/>`_
and some of its extensions. Specifically, the website is available for all our
releases to allow users to check the documentation of the version of the
package that they are using.

To be able to build the documentation you need to install all the
requirements using :code:`pip install -r requirements-dev.txt`.

When making changes to the documentation at least run the following command
to build the website using your changes:

.. code-block::

    python -m sphinx -v -b html -n -j auto docs docs/_build/html

or use the shortcut

.. code-block::

    make doc

This will generate the website in the directory mentioned at the end of the
command. Navigate to that directory and find the corresponding files where
you made changes, open them in the browser and verify that your changes
render properly and links are working as expected.

To fully build the website for all versions use the following script:

.. code-block::

    python scripts/build_documentation.py --documentation-path=docs --output-path=docs/_build/html

or the shortcut

.. code-block::

    make doc-multiversion

The comprehensive set of commands to build the website is in our CircleCI
configuration file in the `.circleci` directory of the repository.
