.. _migrating_to_v0_3_0:

Migrating to v0.3 from v0.2
===========================

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


