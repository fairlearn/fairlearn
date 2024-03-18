.. _quickstart:

Get Started
===========

Installation
------------

Fairlearn can be installed with :code:`pip` from
`PyPI <https://pypi.org/project/fairlearn>`_ as follows:

.. code-block:: bash

   pip install fairlearn

Fairlearn is also available on
`conda-forge <https://anaconda.org/conda-forge/fairlearn>`_:

.. code-block:: bash

    conda install -c conda-forge fairlearn

For further information on how to install Fairlearn and its optional
dependencies, please check out the :ref:`installation_guide`.

If you are updating from a previous version of Fairlearn, please
see :ref:`version_guide`.

.. note::

    The Fairlearn API is still evolving, so example code in
    this documentation may not work with every version of Fairlearn.
    Please use the version selector to get to the instructions for
    the appropriate version. The instructions for the :code:`main`
    branch require Fairlearn to be installed from a clone of the
    repository.

Overview of Fairlearn
---------------------

The Fairlearn package has two components:

- *Metrics* for assessing which groups are negatively impacted by a model,
  and for comparing multiple models in terms of various fairness and accuracy
  metrics.

- *Algorithms* for mitigating unfairness in a variety of AI tasks and along a
  variety of fairness definitions.

Fairlearn in 10 minutes
-----------------------

The Fairlearn toolkit can assist in assessing and mitigation unfairness in
Machine Learning models. It's impossible to provide a sufficient overview of
fairness in ML in this Quickstart tutorial, so we highly recommend starting
with our :ref:`user_guide`. Fairness is a fundamentally sociotechnical
challenge and cannot be solved with technical tools alone. They may be helpful
for certain tasks such as assessing unfairness through various metrics, or to
mitigate observed unfairness when training a model. Additionally, fairness has
different definitions in different contexts and it may not be possible to
represent it quantitatively at all.

Given these considerations this Quickstart tutorial merely provides short
code snippet examples of how to use basic Fairlearn functionality for those
who are already intimately familiar with fairness in ML. The example below
is about binary classification, but we similarly support regression.

Prerequisites
^^^^^^^^^^^^^

In order to run the code samples in the Quickstart tutorial, you need to 
install the following dependencies:

.. code-block:: bash

    pip install fairlearn matplotlib

Loading the dataset
^^^^^^^^^^^^^^^^^^^

For this example, we use a `clincial dataset <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>`_ 
of hospital re-admissions over a ten-year period (1998-2008) for 
diabetic patients across 130 different hospitals in the U.S. This scenario 
builds upon prior research on how racial disparities impact health care 
resource allocation in the U.S. For an in-depth analysis of this dataset,
review the `Scipy tutorial <https://github.com/fairlearn/talks/tree/main/2021_scipy_tutorial>`_
that the Fairlearn team presented in 2021.

We will use machine learning to predict whether an individual in the dataset 
is readmnitted to the hospital within 30 days of hospital release. 
A hospital readmission within 30 days can be viewed as a proxy that the 
patients needed more assistance at the release time.
In the next section, we build a classification model to accomplish the 
prediction task.

.. doctest:: quickstart

    >>> import numpy as np
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> from fairlearn.datasets import fetch_diabetes_hospital
    >>> data = fetch_diabetes_hospital(as_frame=True)
    >>> X = data.data
    >>> X.drop(columns=["readmitted", "readmit_binary"], inplace=True)
    >>> y_true = data.target
    >>> 
    >>> X_ohe = pd.get_dummies(X)
    >>> race = X['race']
    >>> race.value_counts()
    Caucasian          76099
    AfricanAmerican    19210
    Unknown             2273
    Hispanic            2037
    Other               1506
    Asian                641
    Name: race, dtype: int64

.. figure:: auto_examples/images/sphx_glr_plot_quickstart_selection_rate_001.png
    :target: auto_examples/plot_quickstart_selection_rate.html
    :align: center


Evaluating fairness-related metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, Fairlearn provides fairness-related metrics that can be compared
between groups and for the overall population. Using existing metric
definitions from
`scikit-learn <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
we can evaluate metrics for subgroups within the data as below:

.. doctest:: quickstart
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import MetricFrame
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X_train, X_test, y_train, y_test, \
    A_train, A_test = train_test_split(X_ohe, y_true, race, random_state=123)
    >>>
    >>> classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
    >>> classifier.fit(X_train, y_train)
    DecisionTreeClassifier(...)
    >>> y_pred = (classifier.predict_proba(X_test)[:,1] >= 0.1)
    >>> mf = MetricFrame(metrics=accuracy_score, y_true=y_test, y_pred=y_pred, \
    sensitive_features=A_test)
    >>> mf.overall
    0.514...
    >>> mf.by_group
    race
    AfricanAmerican    0.530935
    Asian              0.658683
    Caucasian          0.503535
    Hispanic           0.612524
    Other              0.591549
    Unknown            0.574576
    Name: accuracy_score, dtype: float64

Note that above, because the outcome variable exhibits label imbalance, we set 
the threshold for triggering a positive prediction lower 
(to a probability of 0.1 or greater).
Fairlearn has many standard metrics built-in, such as
selection rate, i.e., the percentage of the population which have '1' as
their label:

.. doctest:: quickstart
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.metrics import selection_rate
    >>> sr = MetricFrame(metrics=selection_rate, y_true=y_test, y_pred=y_pred, \
    sensitive_features=A_test)
    >>> sr.overall
    0.527...
    >>> sr.by_group
    race
    AfricanAmerican    0.514080
    Asian              0.341317
    Caucasian          0.539675
    Hispanic           0.426614
    Other              0.445070
    Unknown            0.438983
    Name: selection_rate, dtype: float64

Fairlearn also allows us to quickly plot these metrics from the
:class:`fairlearn.metrics.MetricFrame`

.. literalinclude:: auto_examples/plot_quickstart.py
    :language: python
    :start-after: # Analyze metrics using MetricFrame
    :end-before: # Customize plots with ylim

.. figure:: auto_examples/images/sphx_glr_plot_quickstart_diabetes_001.png
    :target: auto_examples/plot_quickstart.html
    :align: center


Mitigating disparity
^^^^^^^^^^^^^^^^^^^^

If we observe disparities between groups we may want to create a new model
while specifying an appropriate fairness constraint. Note that the choice of
fairness constraints is crucial for the resulting model, and varies based on
application context. If selection rate is highly relevant for fairness in this
contrived example, we can attempt to mitigate the observed disparity using the
corresponding fairness constraint called Demographic Parity. In real world
applications we need to be mindful of the sociotechnical context when making
such decisions. The Exponentiated Gradient mitigation technique used fits the
provided classifier using Demographic Parity as the constraint, leading to
a vastly reduced difference in selection rate:

.. doctest:: quickstart
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.reductions import ErrorRateParity, DemographicParity, ExponentiatedGradient
    >>> np.random.seed(0)  # set seed for consistent results with ExponentiatedGradient
    >>> objective = ErrorRate(costs={'fp': 0.5, 'fn': 0.5})
    >>> constraint = DemographicParity()
    >>> classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
    >>> mitigator = ExponentiatedGradient(classifier, constraint, objective = objective)
    >>> mitigator.fit(X_train, y_train, sensitive_features=A_train)
    ExponentiatedGradient(...)
    >>> y_pred_mitigated = mitigator.predict(X_test)
    >>>
    >>> sr_mitigated = MetricFrame(metrics=selection_rate, y_true=y_test, \
    y_pred=y_pred_mitigated, sensitive_features=A_test)
    >>> sr_mitigated.overall
    0.0004...
    >>> sr_mitigated.by_group
    race
    AfricanAmerican    0.000000
    Asian              0.000000
    Caucasian          0.000475
    Hispanic           0.001957
    Other              0.000000
    Unknown            0.000000
    Name: selection_rate, dtype: float64

Note that because :class:`ExponentiatedGradient` does not have a `predict_proba`
method, we cannot set a lower threshold for a positive prediction to deal with the 
label imbalance in the outcome variable.


What's next?
------------

Please refer to our :ref:`user_guide` for a comprehensive view on Fairness in
Machine Learning and how Fairlearn fits in, as well as an exhaustive guide on
all parts of the toolkit. For concrete examples check out the
:ref:`examples` section. Finally, we also have a collection
of :ref:`faq`.
