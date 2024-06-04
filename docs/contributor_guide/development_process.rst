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
Fairlearn comes without :code:`matplotlib` by default. To get Fairlearn's
plotting capabilities simply run `pip install matplotlib` after installing
Fairlearn.

.. code-block::

    pip install -e .
    pip install matplotlib

The Requirements Files
""""""""""""""""""""""

The prerequisites for Fairlearn are split between three separate files:

    -  `requirements.txt <https://github.com/fairlearn/fairlearn/blob/main/requirements.txt>`_
       contains the prerequisites for the core Fairlearn package

    -  `requirements-dev.txt <https://github.com/fairlearn/fairlearn/blob/main/requirements-dev.txt>`_ contains
       the prerequisites for Fairlearn development (such as ruff and pytest)

The `requirements.txt <https://github.com/fairlearn/fairlearn/blob/main/requirements.txt>`_
file is consumed
by `setup.py <https://github.com/fairlearn/fairlearn/blob/main/setup.py>`_ to specify the dependencies to be
documented in the wheel files.
To help simplify installation of the prerequisites, we have the
`install_requirements.py <https://github.com/fairlearn/fairlearn/blob/main/scripts/install_requirements.py>`_
script which runs :code:`pip install` on both the above files.
This script will also optionally pin the requirements to any lower bound specified (by changing any
occurrences of :code:`>=` to :code:`==` in each file).

.. _contributing_pull_requests:

Contributing a pull request
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the steps below to create a pull request.

1. Get a `GitHub account <https://github.com/>`_

2. Install `git <https://git-scm.com/book/en/v2/Getting-Started-Installing-Git>`_

3. Look at Fairlearn's issues on GitHub, specifically the ones marked `"help wanted" <https://github.com/fairlearn/fairlearn/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22>`_. Within this category we've marked issues with labels:

   - `"good first issue" <https://github.com/fairlearn/fairlearn/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22+label%3A%22good+first+issue%22>`_ means issues are a good fit for first time contributors including people with no prior experience with coding or GitHub. This is an excellent way to get started!

   - `"easy" <https://github.com/fairlearn/fairlearn/issues?q=is%3Aopen+is%3Aissue+label%3A%22help+wanted%22+label%3A%22easy%22+>`_ means the issue is somewhat harder than a "good first issue", but still quite doable if you have prior experience or you're willing to spend a little bit of time.

   - Neither of the two above: this means issues haven't been scoped out properly yet or are more difficult and likely won't be doable within a day or two. If you're still interested just let us know and we'll figure out a way! Once you find an issue you're interested in comment on the issue with any questions you may have and let us know that you'd like to do it. This helps us avoid duplication and we can help you quicker.

4. Whenever questions come up don't hesitate to reach out (see :ref:`communication`). We're here to help!

5. Clone the Fairlearn repository onto your machine using

.. code-block::

 git clone https://github.com/fairlearn/fairlearn.git

6. Use the "Fork" button to create your own copy of the repository. Run the
command below, replacing :code:`<your-alias>` with your own GitHub alias:

.. code-block::

 git remote add <your-alias> https://github.com/<your-alias>/fairlearn.git

If the execution was successful, running :code:`git remote -v` will show both
:code:`origin` and :code:`<your-alias>`, the first poiting to the original repo
and the second to your fork. Now you can create a new branch and start changing the world!

7.(Optional) Install `pre-commit <https://pre-commit.com/#install>`_ to run code style checks before each commit:

.. code-block::

 pip install pre-commit
 pre-commit install

Pre-commit checks can be disabled for a particular commit with :code:`git commit -n`.

8. To check your branch run :code:`git status`. Initially it will point to :code:`main` which is the default. Create a new branch for yourself by running :code:`git checkout -b <branch-name>`. :code:`git checkout` is your way of switching branches, while :code:`-b` creates a new branch and should only be added the first time you check out a (new) branch. Whenever you are ready to commit your changes run :code:`git add --all` and :code:`git commit --all` or use the version control functionality of your IDE (e.g., Visual Studio Code). To push the changes to your fork run :code:`git push <your-alias>`. Note that you cannot push to :code:`origin` (the main fairlean repository) because it is access-restricted.

9. Build the website following :ref:`contributing_documentation`

10. To create a pull request go to the `Fairlearn repo <https://github.com/fairlearn/fairlearn/pulls>`_ and select "New Pull Request". Click "compare across forks" and subsequently configure the "compare" branch to be the one you pushed your changes to. Briefly check the file changes in the resulting view and click "create pull request" when you're confident about your changes. The following view will ask you to add a pull request title and description, and if you created the pull request in response to an issue add :code:`#<issue-number>` for reference.

11. Celebrate! You did great by participating. If you would like to be a part of the Fairlearn community we'd be thrilled to discuss ways for you to get involved! Check out our communication channels, :ref:`communication`, for more information.

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
