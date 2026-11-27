.. _development_process:

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

The alternative-ML GitHub Actions jobs use :code:`ubuntu-latest` and explicitly
install Miniconda with :code:`conda-incubator/setup-miniconda`. The action creates
the environment from :code:`test_othermlpackages/conda-<package>.yaml`, preserving
each package's Python constraints. These jobs use login Bash shells so the
environment is activated for every test step, without depending on the runner
image's preinstalled software or a fixed Miniconda installation path.

Documentation should be provided with pull requests that add or change
functionality. This includes comments in the code itself, docstrings, and user
guides. For exceptions to this rule the pull request author should coordinate
with a maintainer. For changes that fix bugs, add new features, change APIs,
etc., i.e., for changes that are relevant to developers and/or users please
also add an entry in :ref:`version_guide` in the section corresponding to the
*next* release, since that's where your change will be included.

Docstrings should follow
`numpydoc format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.
This is a `recent decision by the community <https://github.com/fairlearn/fairlearn/issues/314>`_.
The new policy is to update docstrings that a PR touches, as opposed to
changing all the docstrings in one PR.


Installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You will need a GitHub account to contribute to this project. Authenticating with Github
using an ssh connection is highly recommended. You can learn more about it
`here <https://docs.github.com/en/get-started/onboarding/getting-started-with-your-github-account>`_.

First, clone the repository locally via:

.. prompt:: bash

   git clone git@github.com:fairlearn/fairlearn.git


Install `uv <https://docs.astral.sh/uv/getting-started/installation/>`_, then run
from the repository root:

.. prompt:: bash

   uv sync --locked

This creates a project-local :code:`.venv`, installs Fairlearn in editable mode, and
installs all development dependencies, including documentation dependencies. The
checked-in :code:`.python-version` selects Python 3.13 by default; use
:code:`--python 3.11` (or another supported version) on both :code:`uv sync` and
:code:`uv run` to override it. Alternatively, set :code:`UV_PYTHON=3.11` for both commands.
Each checkout or worktree has its own environment. No manual activation is required:
:code:`uv run` executes commands in that checkout's environment.

To verify that the code works as expected, run:

.. prompt:: bash

   uv run --locked python -m pytest -s ./test/unit

Matplotlib and the adversarial ML backends remain optional for end users. Development
groups install Matplotlib and Torch for tests and examples; they are not added to
Fairlearn's published runtime requirements.

Dependencies and the development lockfile
""""""""""""""""""""""""""""""""""""""""""""

Fairlearn's runtime dependencies (the packages required to actually use Fairlearn)
are declared in the ``[project] dependencies`` table of
`pyproject.toml <https://github.com/fairlearn/fairlearn/blob/main/pyproject.toml>`_ and
are installed automatically by :code:`uv sync` or a regular package installation.

The same file declares development-only dependency groups:

* :code:`tests` contains pytest, plotting, and dataframe-backend test dependencies.
* :code:`docs` contains the documentation toolchain and includes the test dependencies
  needed by examples.
* :code:`tooling` contains Ruff, the build frontend, and pre-commit.
* The default :code:`dev` group includes all three.

For a smaller environment, select the groups explicitly in both installation and
execution commands. For example:

.. prompt:: bash

   uv sync --locked --no-default-groups --group tests
   uv run --locked --no-default-groups --group tests python -m pytest ./test/unit

Omitting the group-selection flags from :code:`uv run` restores the default development
groups. The committed :code:`uv.lock` resolves all groups together, even when only some
are installed. :code:`--locked` fails if the manifest and lockfile disagree instead of
silently changing the lock.

When changing dependencies, edit the appropriate group in :code:`pyproject.toml`, run
:code:`uv lock`, and commit both files. To intentionally update locked dependencies:

.. prompt:: bash

   uv lock --upgrade
   uv lock --upgrade-package PACKAGE_NAME

Use the second command for a targeted update, rather than running both. The development
lockfile is not a runtime constraint for package users or compatibility tests.

Compatibility environments
""""""""""""""""""""""""""

Latest, minimum, minimal-installation, and scikit-learn nightly CI environments resolve
independently of :code:`uv.lock`. Use a separate environment and :code:`uv pip` to reproduce
them; do not use project-mode :code:`uv run`, which can restore locked dependencies.

.. prompt:: bash

   uv venv --python 3.11 .venv-compat
   uv pip install --python .venv-compat . --group tests -r requirements-min.txt
   uv run --no-project --python .venv-compat python -m pytest -m narwhals ./test/unit

:code:`requirements-min.txt` pins all runtime requirements and selected development
requirements. The minimum matrix uses Python 3.11, not newer Python versions whose
wheels may require newer dependencies. Optional-backend compatibility bounds in that
file are not claims about the oldest supported backend versions. Latest tests omit
:code:`-r requirements-min.txt`; minimal-installation tests install only Fairlearn,
pytest, and pytest-cov, without the :code:`tests` group.

.. _contributing_pull_requests:

Contributing a pull request
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the steps below to create a pull request.

#. Get a `GitHub account <https://github.com/>`_.

#. Install `GIT <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_.

#. Look at Fairlearn's issues on GitHub, specifically the ones marked
   `"help wanted" <https://github.com/fairlearn/fairlearn/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`_.
   Within this category we've marked issues with labels:

   * `"good first issue" <https://github.com/fairlearn/fairlearn/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22+label%3A%22good+first+issue%22>`_:
     Issues suitable for first time contributors, including people with no prior experience with coding or GitHub.
     This is an excellent way to get started!

   * `"easy" <https://github.com/fairlearn/fairlearn/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22+label%3A%22easy%22>`_:
     Issues suitable for folks with at least a bit of experience and/or able to allocate some time to look for a solution.

   *  Neither of the two above:
      issues that are demanding or awaiting scope. Likely to take more than a day or two.
      If you think this is something for you, please:

      * Identify an issue that you would like to work on.
      * Leave a comment on the issue indicating interest and outlining possible questions.
      * Once we know you are working on it, we will support you on your contribution journey!

.. note::

   If you claim an issue, please try to keep it updated each week, either by continuing a discussion in the issue itself or in a pull request.
   Issues which are not receiving updates may be claimed by someone else.

#. The communication channels are outlined here: :ref:`communication`.

#. Fork the `project repository
   <https://github.com/fairlearn/fairlearn.git>`__ by clicking on the 'Fork'
   button near the top of the page. This creates a copy of the code on your GitHub user account.
   For more details on how to fork a
   repository see `this guide <https://help.github.com/articles/fork-a-repo/>`_.

#. Clone your fork of the fairlern repo from your GitHub account to your
   local machine:

   .. prompt:: bash

      git clone git@github.com:YourLogin/fairlearn.git  # add --depth 1 if your connection is slow
      cd fairlearn

#. Add the ``upstream`` remote. This saves a reference to the main
   fairlearn repository, which you can use to keep your repository
   synchronized with the latest changes:

   .. prompt:: bash

      $ git remote add upstream git@github.com:fairlearn/fairlearn.git

#. Check that the :code:`upstream` and :code:`origin` remote aliases are configured correctly
   by running

   :code:`git remote -v` which should display:

   .. code-block:: text

        origin	git@github.com:YourLogin/fairlearn.git (fetch)
        origin	git@github.com:YourLogin/fairlearn.git (push)
        upstream	git@github.com:fairlearn/fairlearn.git (fetch)
        upstream	git@github.com:fairlearn/fairlearn.git (push)


#. (Optional) Install `pre-commit <https://pre-commit.com/#install>`_ to run code style checks before each commit:

   .. prompt:: bash

      uv run --locked pre-commit install

   Pre-commit checks can be disabled for a particular commit with :code:`git commit -n`.

#. To contribute, you will need to create a branch on your forked repository and make a pull request to the original fairlearn repository.
   Detailed description of this process you can find here:

   * `Create a branch <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project#creating-a-branch-to-work-on>`_.
   * `Commit and push changes <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project#making-and-pushing-changes>`_.
   * `Open a pull request <https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project#making-a-pull-request>`_.


      * Build the website following the guidelines in :ref:`contributing_documentation` and run the tests if necessary.
      * Opening a pull request comes with filling up an already provided description template.
        Please fill it up! If you created the pull request in response to an issue add :code:`#<issue-number>` for reference.

#. Celebration time! We would like to encourage you to become a part of our Fairlearn community. To do so, join our communication channels: :ref:`communication`. Please respect our `Code of Conduct <https://github.com/fairlearn/governance/blob/main/code-of-conduct.md>`_.

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
