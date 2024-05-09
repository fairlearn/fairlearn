.. _assessment:

Assessment
==========

.. currentmodule:: fairlearn.metrics

In this section, we will describe the steps involved in performing a fairness
assessment, and introduce some widely (if occasionally incautiously) used
fairness metrics, such as demographic parity and equalized odds.
We will show how :class:`MetricFrame` can be used to evaluate the metrics
identified during the course of a fairness assessment.

Fairlean provides two primary ways of assessing fairness: :class:`MetricFrame`,
which can be used to perform disaggregated analysis of a particular performance 
metric (such as accuracy, false positive rate, etc.) across sensitive 
groups, and a set of predefined fairness metrics that use :class:`MetricFrame` 
internally to output an aggregate value.
:class:`MetricFrame` can also be used to output an aggregate value, 
but the predefined fairness metrics can be used when direct by-group 
comparison is not necessary.
In the :ref:`perform_fairness_assessment`, we will dive further into
each of these types of fairness assessments. 

In the mathematical definitions below, :math:`X` denotes a feature vector 
used for predictions, :math:`A` will be a single sensitive feature (such as age 
or race), and :math:`Y` will be the true label.
Fairness metrics are phrased in terms of expectations with respect to the
distribution over :math:`(X,A,Y)`.
Note that :math:`X` and :math:`A` may or may not share columns, dependent on
whether the model is allowed to 'see' the sensitive features.
When we need to refer to particular values, we will use lower case letters;
since we are going to be comparing between groups identified by the
sensitive feature, :math:`\forall a \in A` will be appearing regularly to
indicate that a property holds for all identified groups.

.. toctree::
   :maxdepth: 2

   perform_fairness_assessment
   common_fairness_metrics
   custom_fairness_metrics
   intersecting_groups
   confidence_interval_estimation
   advanced_metricframe
   plotting
   saving_loading_metricframe


.. _dashboard:

Fairlearn dashboard
-------------------

The Fairlearn dashboard was a Jupyter notebook widget for assessing how a
model's predictions impact different groups (e.g., different ethnicities), and
also for comparing multiple models along different fairness and performance
metrics.

.. note::

    The :code:`FairlearnDashboard` is no longer being developed as
    part of Fairlearn.
    For more information on how to use it refer to
    `https://github.com/microsoft/responsible-ai-toolbox <https://github.com/microsoft/responsible-ai-toolbox>`_.
    Fairlearn provides some of the existing functionality through
    :code:`matplotlib`-based visualizations. Refer to the :ref:`plot_metricframe` section.
