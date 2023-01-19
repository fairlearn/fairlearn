.. _api_reference:

API Docs
========

.. toctree::
   :maxdepth: 1

   fairlearn.datasets
   fairlearn.metrics
   fairlearn.postprocessing
   fairlearn.preprocessing
   fairlearn.reductions
   fairlearn.adversarial
   fairlearn.experimental

.. currentmodule:: fairlearn

fairlearn.datasets
==================
.. autosummary::
   :nosignatures:
   :template: function.rst

   datasets.fetch_acs_income
   datasets.fetch_adult
   datasets.fetch_bank_marketing
   datasets.fetch_boston
   datasets.fetch_credit_card
   datasets.fetch_diabetes_hospital

fairlearn.metrics
=================
.. autosummary:: 
   :nosignatures:
   :template: function.rst

    metrics.count
    metrics.demographic_parity_difference
    metrics.demographic_parity_ratio
    metrics.equalized_odds_difference
    metrics.equalized_odds_ratio
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
   :template: classes.rst
   
   metrics.MetricFrame

fairlearn.postprocessing
========================
.. autosummary::
   :nosignatures:
   :template: classes.rst

   postprocessing.ThresholdOptimizer

.. autosummary::
   :nosignatures:
   :template: function.rst

   postprocessing.plot_threshold_optimizer

fairlearn.preprocessing
=======================
.. autosummary::
   :nosignatures:
   :template: classes.rst

   preprocessing.CorrelationRemover

fairlearn.reductions
====================
.. autosummary::
   :nosignatures:
   :template: classes.rst

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
.. autosummary::
   :nosignatures:
   :template: classes.rst
   
   adversarial.AdversarialFairnessClassifier
   adversarial.AdversarialFairnessRegressor

fairlearn.experimental
======================
.. autosummary::    
   :nosignatures:
   :template: function.rst
   
   metrics._plotter.plot_metric_frame
