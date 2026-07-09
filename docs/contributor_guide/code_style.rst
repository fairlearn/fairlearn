.. _code_style:

Code Style
==========

Github conventions
------------------

Because the Fairlearn team squash merges pull requests, you do not need to put
much effort into the commit messages you submit when pushing code.

Titles should be descriptive and include one of the following prefixes:
* DOC: any documentation-related PRs (including the user guide, API reference, and other
website-related PRs).
* MNT: code maintenance (refactoring, improve efficiency, etc.).
* CI: anything related to our automated tests (nightly builds, CircleCI, releases, etc.).
* FIX: bug fixes.
* FEAT/ENH: adds a new feature or removes a feature in the codebase.

Test coverage is checked with ``codecov`` and line(s) missing tests will show up in CI
as a failure. Therefore, we recommend contributors ensure all new content
they introduce has adequate test coverage.


Code conventions
----------------

Linting
^^^^^^^

We recommend using a linter to check your code before you submit a PR.
We use ``ruff`` for both linting (checking for PEP8 compatibility issues) and
code formatting. You can check your code with ``ruff check`` and automatically
format it with ``ruff format``; the two together satisfy our formatting
requirements. You can also configure your IDE to run ``ruff`` on save. Please
refer to your IDE's instructions for further details.
Attaining the project compatible linting is also possible without changing your local setup.
You can enable the pre-commit hooks, which will run the linters described above with the settings
defined in the `pyproject.toml <https://github.com/fairlearn/fairlearn/blob/main/pyproject.toml>`_.
The installation instructions are described in step 6
of `"Contributing a Pull Request" <https://fairlearn.org/main/contributor_guide/development_process.html#contributing-a-pull-request>`_ .

Considerations for new methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are introducing new estimators to the Fairlearn, you must ensure the
estimator is fully compatible with scikit-learn
(defined `here <https://scikit-learn.org/stable/developers/develop.html>`_).
For more resources on how to develop scikit-learn estimators, review this
`post <https://tamaraatanasoska.github.io/learning/2025/01/15/week-2-2024.html>`_
by one of the Fairlearn maintainers.

The Fairlearn team is in the process of swapping out the ``pandas`` library for
``narwhals`` for data manipulation tasks. If you are contributing code that
includes Pandas, we recommend you use Narwhals instead to stay ahead of this effort.

Because there are many complex sources of unfairness — some societal and some technical — it is not
possible to fully “debias” a system or to guarantee fairness. In Fairlearn, we therefore try to
avoid naming mitigation techniques in a way that could suggest they offer a simple fix towards a
"fair" model. Instead, we opt for descriptive names, such as "ThresholdOptimizer" (rather than
e.g. "FairThresholder") and "CorrelationRemover" (instead of e.g. "BiasRemover").

Test coverage
^^^^^^^^^^^^^
For more information about Fairlean's default coverage settings
check the `codecov documentation <https://docs.codecov.com/docs/coverage-
configuration#:~:text=Codecov%20will%20round%20coverage%20down,45.15313%25%20would%20become%2045.15%25>`_.
