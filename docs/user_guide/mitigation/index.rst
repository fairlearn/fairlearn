.. _mitigation:

Mitigations
===========

In this section, we discuss the various mitigation techniques implemented
in Fairlearn.
One thing which should always be remembered: all the algorithms herein
will provide mathematical guarantees as to how close they can drive some
unfairness metric to zero.
**However**, this does not mean that the results are *fair*.

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
   *  - :class:`~fairlearn.reductions.ExponentiatedGradient`
      - A wrapper (reduction) approach to fair classification described in *A Reductions*
        *Approach to Fair Classification* :footcite:`agarwal2018reductions`.
      - ✔
      - ✔
      - DP, EO, TPRP, FPRP, ERP, BGL
   *  - :class:`~fairlearn.reductions.GridSearch`
      - A wrapper (reduction) approach described in Section 3.4 of *A Reductions*
        *Approach to Fair Classification* :footcite:`agarwal2018reductions`. For regression it acts as a
        grid-search variant of the algorithm described in Section 5 of
        *Fair Regression: Quantitative Definitions and Reduction-based*
        *Algorithms* :footcite:`agarwal2019fair`.
      - ✔
      - ✔
      - DP, EO, TPRP, FPRP, ERP, BGL
   *  - :class:`~fairlearn.postprocessing.ThresholdOptimizer`
      - Postprocessing algorithm based on the paper *Equality of Opportunity*
        *in Supervised Learning* :footcite:`hardt2016equality`. This technique takes as input an
        existing classifier and the sensitive feature, and derives a monotone
        transformation of the classifier's prediction to enforce the specified
        parity constraints.
      - ✔
      - ✘
      - DP, EO, TPRP, FPRP
   *  - :class:`~fairlearn.preprocessing.CorrelationRemover`
      - Preprocessing algorithm that removes correlation between sensitive
        features and non-sensitive features through linear transformations.
      - ✔
      - ✔
      - ✘
   *  - :class:`~fairlearn.adversarial.AdversarialFairnessClassifier`
      - An optimization algorithm based on the paper *Mitigating Unwanted Biases*
        *with Adversarial Learning* :footcite:`zhang2018mitigating`. This method trains a neural
        network classifier that minimizes training error while
        preventing an adversarial network from inferring sensitive features.
        The neural networks can be defined either as a `PyTorch module
        <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ or
        `TensorFlow2 model 
        <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_.
      - ✔
      - ✘
      - DP, EO
   *  - :class:`~fairlearn.adversarial.AdversarialFairnessRegressor`
      - The regressor variant of the above :class:`~fairlearn.adversarial.AdversarialFairnessClassifier`.
        Useful to train a neural network with continuous valued output(s).
      - ✘
      - ✔
      - DP, EO

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


.. toctree::
   :maxdepth: 2

   preprocessing
   postprocessing
   reductions
   adversarial