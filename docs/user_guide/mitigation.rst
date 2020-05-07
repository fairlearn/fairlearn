.. _mitigation:

Mitigation
==========

.. toctree::
   :maxdepth: 2

   postprocessing
   reductions


The Fairlearn package largely follows the
`terminology established by scikit-learn <https://scikit-learn.org/stable/developers/contributing.html#different-objects>`_,
specifically:

* *Estimators* implement a :code:`fit` method.
* *Predictors* implement a :code:`predict` method.


**Randomization.** In contrast with 
`scikit-learn <https://scikit-learn.org/stable/glossary.html#term-estimator>`_,
estimators in Fairlearn can produce randomized predictors. Randomization of
predictions is required to satisfy many definitions of fairness. Because of
randomization, it is possible to get different outputs from the predictor's
:code:`predict` method on identical data. For each of our methods, we provide
explicit access to the probability distribution used for randomization.