.. _assessment:

Assessment
==========

.. currentmodule:: fairlearn.metrics

In this section, we will describe the steps involved in performing a fairness
assessment, and introduce some widely (if occasionally incautiously) used
fairness metrics, such as demographic parity and equalized odds.
We will show how :class:`MetricFrame` can be used to evaluate the metrics
identified during the course of a fairness assessment.

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