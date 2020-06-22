.. _mitigation:

Mitigation
==========

Fairlearn contains the following algorithms for mitigating unfairness:

.. list-table:: Unfairness mitigation algorithms
   :header-rows: 1
   :widths: 5 20 5 5 8
   :stub-columns: 1

   *  - algorithm
      - description
      - binary classification
      - regression
      - supported fairness definitions
   *  - :code:`fairlearn.` :code:`reductions.` :code:`ExponentiatedGradient`
      - Black-box approach to fair classification described in *A Reductions*
        *Approach to Fair Classification* [#1]_.
      - ✔
      - ✘ :superscript:`*`
      - DP, EO, TPRP, ERR
   *  - :code:`fairlearn.` :code:`reductions.` :code:`GridSearch`
      - Black-box approach described in Section 3.4 of *A Reductions*
        *Approach to Fair Classification* [#1]_. For regression it acts as a
        grid-search variant of the algorithm described in Section 5 of
        *Fair Regression: Quantitative Definitions and Reduction-based*
        *Algorithms* [#2]_.
      - ✔
      - ✔
      - DP, EO, TPRP, ERR, BGL
   *  - :code:`fairlearn.` :code:`postprocessing.` :code:`ThresholdOptimizer`
      - Postprocessing algorithm based on the paper *Equality of Opportunity*
        *in Supervised Learning* [#3]_. This technique takes as input an
        existing classifier and the sensitive feature, and derives a monotone
        transformation of the classifier's prediction to enforce the specified
        parity constraints.
      - ✔
      - ✘
      - DP, EO

.. [*] coming soon!

DP refers to demographic parity, EO to equalized odds, TPRD to true positive
rate difference, ERR to error rate ratio, and BGL to bounded group loss. For
more information on the definitions refer to
:ref:`fairness_in_machine_learning`. To request additional algorithms or
fairness definitions, please open a
`new issue <https://github.com/fairlearn/fairlearn/issues>`_ on GitHub.

.. note: Randomized predictors

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

.. _postprocessing:

Postprocessing
--------------

.. currentmodule:: fairlearn.postprocessing

.. _reductions:

Reductions
----------

.. currentmodule:: fairlearn.reductions

Exponentiated Gradient
^^^^^^^^^^^^^^^^^^^^^^

Grid Search
^^^^^^^^^^^

.. topic:: References:

   .. [#1] Agarwal, Dudik, Wu `"Fair Regression: Quantitative Definitions and
      Reduction-based Algorithms" <https://arxiv.org/pdf/1905.12843.pdf>`_,
      ICML, 2019.
   
   .. [#2] Agarwal, Beygelzimer, Dudik, Langford, Wallach `"A Reductions
      Approach to Fair Classification"
      <https://arxiv.org/pdf/1803.02453.pdf>`_, ICML, 2018.
   
   .. [#3] Hardt, Price, Srebro `"Equality of Opportunity in Supervised
      Learning"
      <https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning.pdf>`_,
      NIPS, 2016.
