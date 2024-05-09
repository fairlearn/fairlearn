.. _api_reference:

API Docs
========

fairlearn.datasets
==================

.. automodule:: fairlearn.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: fairlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

   datasets.fetch_acs_income
   datasets.fetch_adult
   datasets.fetch_bank_marketing
   datasets.fetch_boston
   datasets.fetch_credit_card
   datasets.fetch_diabetes_hospital

fairlearn.metrics
=================

.. automodule:: fairlearn.metrics
    :no-members:
    :no-inherited-members:

.. currentmodule:: fairlearn

.. autosummary:: 
   :nosignatures:
   :toctree: generated/
   :template: function.rst

    metrics.count
    metrics.demographic_parity_difference
    metrics.demographic_parity_ratio
    metrics.equalized_odds_difference
    metrics.equalized_odds_ratio
    metrics.equal_opportunity_difference
    metrics.equal_opportunity_ratio
    metrics.false_negative_rate
    metrics.false_positive_rate
    metrics.make_derived_metric
    metrics.mean_prediction
    metrics.plot_model_comparison
    metrics.selection_rate
    metrics.true_negative_rate
    metrics.true_positive_rate


.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   metrics.MetricFrame

fairlearn.postprocessing
========================

.. automodule:: fairlearn.postprocessing
    :no-members:
    :no-inherited-members:

.. currentmodule:: fairlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   postprocessing.ThresholdOptimizer

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst

   postprocessing.plot_threshold_optimizer

fairlearn.preprocessing
=======================

.. automodule:: fairlearn.preprocessing
    :no-members:
    :no-inherited-members:

.. currentmodule:: fairlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   preprocessing.CorrelationRemover

fairlearn.reductions
====================

.. automodule:: fairlearn.reductions
    :no-members:
    :no-inherited-members:

.. currentmodule:: fairlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst

   reductions.AbsoluteLoss
   reductions.BoundedGroupLoss
   reductions.ClassificationMoment
   reductions.DemographicParity
   reductions.EqualizedOdds
   reductions.ErrorRate
   reductions.ErrorRateParity
   reductions.ExponentiatedGradient
   reductions.TruePositiveRateParity
   reductions.FalsePositiveRateParity
   reductions.UtilityParity
   reductions.GridSearch
   reductions.LossMoment
   reductions.Moment
   reductions.BoundedGroupLoss
   reductions.SquareLoss
   reductions.ZeroOneLoss

fairlearn.adversarial
=====================

.. automodule:: fairlearn.adversarial
    :no-members:
    :no-inherited-members:

.. currentmodule:: fairlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: class.rst
   
   adversarial.AdversarialFairnessClassifier
   adversarial.AdversarialFairnessRegressor

fairlearn.experimental
======================

.. automodule:: fairlearn.experimental
    :no-members:
    :no-inherited-members:

.. currentmodule:: fairlearn

.. autosummary::
   :nosignatures:
   :toctree: generated/
   :template: function.rst
   
   metrics._plotter.plot_metric_frame
