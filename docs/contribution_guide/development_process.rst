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

Developer certificate of origin
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contributions require you to sign a *developer certificate of origin* (DCO)
declaring that you have the right to, and actually do, grant us the rights to
use your contribution. For details, visit https://developercertificate.org/.

When you submit a pull request, a DCO-bot will automatically determine whether
you need to provide a DCO and decorate the PR appropriately (e.g., label,
comment).

Signing off means you need to have your name and email address attached as a
commit comment, which you can automate using git hooks as shown
`here <https://stackoverflow.com/questions/15015894/git-add-signed-off-by-line-using-format-signoff-not-working/46536244#46536244>`_.s

.. _advanced_install:

Advanced installation instructions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While working on Fairlearn itself you may want to install it in editable mode.
This allows you to test the changed functionality. To install in editable mode
using :code:`pip` run :code:`pip install -e .` from the repository root path.
Note that the dashboard is built using nodejs and requires additional steps.
To build the Fairlearn dashboard after making changes to it,
`install Yarn <https://yarnpkg.com/lang/en/docs/install>`_, and then run the
`widget build script <https://github.com/fairlearn/fairlearn/tree/master/scripts/build_widget.py>`_.

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
