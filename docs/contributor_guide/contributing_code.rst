Contributing code
=================

General advice
--------------

The field of ML fairness is nascent and developing, and while there are many
emerging methods in the fairness literature, the Fairlearn team is discerning
when it comes to adding new methods to the library. We often get requests to
add emerging methods as features, but if you want to suggest including
new features in the future, please keep in mind the guidance in this section.
For algorithms, we require all methods to be described in a peer-reviewed
paper; the Fairlean team specifies this requirement as a quality check, so
we do not need to complete the peer reviewing ourselves. We have a preference
for algorithms that are intuitive, easy to understand, and make explicit
the underlying empirical and normative assumptions (for example, algorithms
that are designed to address a specific type of measurement bias).
For metrics, Fairlearn currently only supports disaggregated methods, so
any proposed metrics that do not fall into the group fairness metric
paradigm would first require thorough discussion with maintainers.

Code Style
----------
View the :ref:`code_style` section for an overview of the style we expect for contributed code.

Coverage reporting
------------------

The ``build-test`` workflow collects line coverage from the selected Python
versions on Linux, Windows, and macOS, including the minimum-version,
install-only, and other-ML suites. The ``coverage`` job requires the complete
set of raw coverage artifacts before combining them and uploading one report
to Codecov under the ``unittests`` flag. A missing artifact, failed test suite,
or upload error prevents a successful report. Each artifact identifies its
suite, operating system, and Python version; the combined report also includes
JSON totals. Reruns replace their artifacts, and reports are not skipped based
on the age of the commit. Nightly runs retain artifacts without uploading to
the canonical Codecov report.

The initial Codecov policy is informational, not a required merge check:

* ``codecov/project/line-floor`` has an absolute 95% line-coverage target.
* ``codecov/project/non-regression`` compares coverage against the base commit
  with no allowed decrease. The absolute floor alone would permit regressions
  from higher coverage.
* ``codecov/patch`` has a 90% target for changed executable lines.

All targets have zero tolerance. Informational statuses do not block merges,
even when a target is missed. Missing coverage is not carried forward from
earlier commits. The initial main baseline was 95.02% (3,172 of 3,338 lines);
the new aggregate must be verified against a comparable main report before
these checks can become enforcing. Codecov may merge uploads from repeated
runs on the same commit, so use the per-run artifacts to diagnose freshness.
Fork PR uploads, old-commit reruns, comparison bases (including release
branches), and final notification timing must also be verified before making
any external coverage status required. Upload acceptance is not proof that
Codecov has finished processing the report.

For local line coverage, install the development test requirements and use
the non-interactive ``Agg`` backend (``MPLBACKEND=Agg``), as in CI. Run the
existing test selection with pytest-cov:

.. code-block:: console

   uv run --with pytest-cov pytest --mpl --mpl-baseline-path=test/unit/plot_snapshots --ignore=test/install -m "not openml" --cov=fairlearn --cov-report=term-missing test/

To measure a branch baseline, manually dispatch ``build-test`` with the
``branch_coverage`` input enabled. The extra Ubuntu/Python 3.13 full-suite job
adds ``--cov-branch`` and retains a separate ``coverage-branch-full-*``
artifact containing raw data, XML, and JSON. It does not upload branch results
to Codecov or enforce a coverage percentage. The usual line report is
unchanged. The JSON report records covered and missing lines and branches;
its overall percentage includes both kinds of execution opportunities.

This single-suite branch measurement is diagnostic, not a full-matrix
baseline. Before migrating the aggregate to branch coverage, all contributing
suites must collect branches consistently, and a new main baseline and target
must be established. The 95% line floor must not be reused for a different
metric. In particular, the small ``test/install`` suite contributes to the
aggregate but has no independent full-library floor. Do not add a global
``--cov-fail-under`` to shared pytest settings.


API conventions
---------------

This section relies on the definitions from our :ref:`terminology` guide,
including the definitions of "estimator", "reduction", "sensitive features",
"moment", and "parity".

Unfairness mitigation algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unfairness mitigation algorithms take form of scikit-learn-style estimators.
Any algorithm-specific parameters are passed to the constructor. The resulting
instance of the algorithm should support methods :code:`fit` and
:code:`predict` with APIs resembling those of scikit-learn as much as
possible. Any deviations are noted below.

Reductions
""""""""""

Reduction constructors require a parameter corresponding to an estimator that
implements the :code:`fit` method with the :code:`sample_weight` argument.
Parity constraints for reductions are expressed via instances of various
subclasses of the class :class:`fairlearn.reductions.Moment`. Formally,
instances of the class :class:`.Moment` implement vector-valued random variables
whose sample averages over the data are required to be bounded (above and/or
below).

.. code-block:: python

    constraints = Moment()
    reduction = Reduction(estimator, constraints)

Reductions provide :code:`fit` and :code:`predict` methods with the following
signatures:

.. code-block:: python

    reduction.fit(X, y, **kwargs)
    reduction.predict(X)

All of the currently supported parity constraints (subclasses of
:class:`.Moment`) are based on sensitive features that need to be provided to
:code:`fit` as a keyword argument :code:`sensitive_features`. In the future,
it will also be possible to provide sensitive features as columns of
:code:`X`.

Postprocessing algorithms
""""""""""""""""""""""""""

The constructors of postprocessing algorithms require either an already
trained predictor or an estimator (which is trained on the data when executing
:code:`fit`). For postprocessing algorithms, the :code:`constraints` argument
is provided as a string.

.. code-block:: python

    postprocessor = PostProcessing(estimator=estimator, constraints=constraints)

Post-processing algorithms (such as the ones under
:py:mod:`fairlearn.postprocessing`) provide the same functions as the reductions
above albeit with :code:`sensitive_features` as a required argument for
:code:`predict`. In the future, we will make :code:`sensitive_features`
optional if the sensitive features are already provided through :code:`X`.

.. code-block:: python

    postprocessor.fit(X, y, sensitive_features=sensitive_features)
    postprocessor.predict(X, sensitive_features=sensitive_features)
