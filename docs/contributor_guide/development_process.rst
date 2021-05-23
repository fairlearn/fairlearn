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
