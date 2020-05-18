Contributing code
=================

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
subclasses of the class :code:`fairlearn.reductions.Moment`. Formally,
instances of the class :code:`Moment` implement vector-valued random variables
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
:code:`Moment`) are based on sensitive features that need to be provided to
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
:code:`fairlearn.postprocessing`) provide the same functions as the reductions
above albeit with :code:`sensitive_features` as a required argument for
:code:`predict`. In the future, we will make :code:`sensitive_features`
optional if the sensitive features are already provided through :code:`X`.

.. code-block:: python

    postprocessor.fit(X, y, sensitive_features=sensitive_features)
    postprocessor.predict(X, sensitive_features=sensitive_features)
