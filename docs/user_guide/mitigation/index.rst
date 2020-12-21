.. _mitigation:

Mitigation
==========

Fairlearn contains the following algorithms for mitigating unfairness:

.. list-table::
   :header-rows: 1
   :widths: 5 20 5 5 8
   :stub-columns: 1

   *  - algorithm
      - description
      - binary classification
      - regression
      - supported fairness definitions
   *  - :code:`fairlearn.` :code:`reductions.` :code:`ExponentiatedGradient`
      - A wrapper (reduction) approach to fair classification described in *A Reductions*
        *Approach to Fair Classification* [#2]_.
      - ✔
      - ✔
      - DP, EO, TPRP, FPRP, ERP, BGL
   *  - :code:`fairlearn.` :code:`reductions.` :code:`GridSearch`
      - A wrapper (reduction) approach described in Section 3.4 of *A Reductions*
        *Approach to Fair Classification* [#2]_. For regression it acts as a
        grid-search variant of the algorithm described in Section 5 of
        *Fair Regression: Quantitative Definitions and Reduction-based*
        *Algorithms* [#1]_.
      - ✔
      - ✔
      - DP, EO, TPRP, FPRP, ERP, BGL
   *  - :code:`fairlearn.` :code:`postprocessing.` :code:`ThresholdOptimizer`
      - Postprocessing algorithm based on the paper *Equality of Opportunity*
        *in Supervised Learning* [#3]_. This technique takes as input an
        existing classifier and the sensitive feature, and derives a monotone
        transformation of the classifier's prediction to enforce the specified
        parity constraints.
      - ✔
      - ✘
      - DP, EO, TPRP, FPRP

DP refers to *demographic parity*, EO to *equalized odds*, TPRP to *true positive
rate parity*, FPRP to *false positive rate parity*, ERP to *error rate parity*, and
BGL to *bounded group loss*. For
more information on the definitions refer to
:ref:`fairness_in_machine_learning`. To request additional algorithms or
fairness definitions, please open a
`new issue <https://github.com/fairlearn/fairlearn/issues>`_ on GitHub.

.. note::

   Fairlearn mitigation algorithms largely follow the
   `conventions of scikit-learn <https://scikit-learn.org/stable/developers/contributing.html#different-objects>`_,
   meaning that they implement the :code:`fit` method to train a model and the :code:`predict` method
   to make predictions. However, in contrast with 
   `scikit-learn <https://scikit-learn.org/stable/glossary.html#term-estimator>`_,
   Fairlearn algorithms can produce randomized predictors. Randomization of
   predictions is required to satisfy many definitions of fairness. Because of
   randomization, it is possible to get different outputs from the predictor's
   :code:`predict` method on identical data. For each of our algorithms, we provide
   explicit access to the probability distribution used for randomization.

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
      NeurIPS, 2016.
