Development process
-------------------

Development happens against the :code:`master` branch following the
`GitHub flow model <https://guides.github.com/introduction/flow/>`_.
Contributors should use their own forks of the repository. In their fork, they
create feature branches off of :code:`master`, and their pull requests should
target the :code:`master` branch. Maintainers are responsible for prompt
review of pull requests.

Pull requests against :code:`master` trigger automated tests that are run
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

Developer certificate of origin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All contributions require you to sign the *developer certificate of origin
(DCO)*. This is a developer's certification in which you declare that you have
the right to, and actually do, grant us the rights to use your contribution.
We use the exact same one created and used by the Linux kernel developers. You
can read it at https://developercertificate.org.

You sign the DCO by *signing off* every commit comment with your name and email
address: *Signed-off-by: Your Name <your.email@example.com>*

When you submit a pull request, a DCO-bot will automatically determine whether
you need to provide a DCO and indicate how you can decorate the PR
appropriately (e.g., label, comment).

Manually
""""""""

You can manually sign-off by adding a separate paragraph to your commit
message:

.. code-block::

    git commit -m “Your message
    Signed-off-by: Your Name <your.email@example.com>

or

.. code-block::

    git commit -m “Your message" -m “Signed-off-by: Your Name <your.email@example.com>”

If this feels like a lot of typing, you can configure your name and e-mail in
git to sign-off:

.. code-block::

    git config --global user.name “Your Name”
    git config --global user.email “your.email@example.com”


Now, you can sign off using :code:`-s` or :code:`--signoff`:

.. code-block::

    git commit -s -m "Your message"

If you find :code:`-s` too much typing as well, you can also add an alias:

.. code-block::

    git config --global alias.c "commit --signoff"


Which allows you to commit including a signoff as :code:`git c -m "Your
Message"`.

These instructions were adapted from `this blog post <https://kauri.io/dco-signoff-commiting-code-to-hyperledger-besu/f58190e5e3bc4b1a9ed902bfccfe58b9/a>`_.

Automatically
"""""""""""""

You can also fully automate signing off using git hooks, by following the
instructions of `this stack overflow post <https://stackoverflow.com/questions/15015894/git-add-signed-off-by-line-using-format-signoff-not-working/46536244#46536244>`_.

.. _advanced_install:

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

    pip install -r requirements.txt
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
`widget build script <https://github.com/fairlearn/fairlearn/tree/master/scripts/build_widget.py>`_.

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

For every pull request to :code:`master` with automated tests, you can check
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

Creating new releases
^^^^^^^^^^^^^^^^^^^^^

First add a description of the changes introduced in the package version you
want to release to `CHANGES.md <https://github.com/fairlearn/fairlearn/CHANGES.md>`_.

It is also best to verify that the Fairlearn dashboard loads correctly. This
is slightly involved:

#. Install the :code:`wheel` package by running :code:`pip install wheel`
#. Create a wheel by running :code:`python setup.py sdist bdist_wheel` from
   the repository root. This will create a :code:`dist` directory which
   contains a :code:`.whl` file.
#. Create a new conda environment for the test
#. In this new environment, install this wheel by running
   :code:`pip install dist/<FILENAME>.whl`
#. Install any pip packages required for the notebooks using
   :code:`pip install -r requirements.txt`
#. Check that the dashboard loads in the notebooks

We have an
`Azure DevOps Pipeline <https://dev.azure.com/responsibleai/fairlearn/_build?definitionId=60&_a=summary>`_
which takes care of building wheels and pushing to PyPI. Validations are also
performed prior to any deployments, and also following the uploads to Test-PyPI
and PyPI. To use it:

#. Ensure that `fairlearn/__init__.py` has the correct version set.
#. Put down a tag corresponding to this version but preprended with :code:`v`.
   For example, version :code:`0.5.0` should be tagged with :code:`v0.5.0`.

At queue time, select Test or Production PyPI as appropriate.

As part of the release process, the :code:`build_wheels.py` script uses
:code:`process_readme.py` to turn all the relative links in the ReadMe file
into absolute ones (this is the reason why the applied tag has be of the form
:code:`v[__version__]`). The :code:`process_readme.py` script is slightly
fragile with respect to the contents of the ReadMe, so after significant
changes its output should be verified.
