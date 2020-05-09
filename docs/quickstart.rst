.. _quickstart:

Quickstart
==========

Installation
------------

.. raw:: html

    <div class="container">
        <div class="row">
            <div class="col-lg-6 col-md-6 col-sm-12 col-xs-12 d-flex install-block">
                <div class="card install-card shadow w-100">
                <div class="card-header">
                    Working with conda?
                </div>
                <div class="card-body">
                    <p class="card-text">

Fairlearn is on `conda-forge <https://anaconda.org/conda-forge/fairlearn>`_:

.. raw:: html

                    </p>
                </div>
                <div class="card-footer text-muted">

.. code-block:: bash

    conda install -c conda-forge fairlearn 

.. raw:: html

                </div>
                </div>
            </div>
            <div class="col-lg-6 col-md-6 col-sm-12 col-xs-12 d-flex install-block">
                <div class="card install-card shadow w-100">
                <div class="card-header">
                    Prefer pip?
                </div>
                <div class="card-body">
                    <p class="card-text">

Fairlearn can be installed via pip from `PyPI <https://pypi.org/project/fairlearn>`__.

.. raw:: html

                    </p>
                </div>
                <div class="card-footer text-muted">

.. code-block:: bash

   pip install fairlearn

.. raw:: html

                </div>
                </div>
            </div>
            <div class="col-12 d-flex install-block">
                <div class="card install-card shadow w-100">
                <div class="card-header">
                    In-depth instructions?
                </div>
                <div class="card-body">
                    <p class="card-text">Installing a specific version?
                      Installing from source?
                      Check the advanced installation page.</p>

.. container:: custom-button

    :ref:`Learn more <advanced_install>`

.. raw:: html

                </div>
                </div>
            </div>
        </div>
    </div>


Fairlearn in 10 minutes
-----------------------

The Fairlearn tookit can assist in assessing and mitigation unfairness in
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

Loading the dataset
^^^^^^^^^^^^^^^^^^^

In this toy example we create a simple dataset with 200 samples. The first 100
belong to group A, the second 100 belong to group B. The labels and the two
features are randomly generated. Group B's labels are more likely to have
label 1, and its samples are slightly skewed in the second component to allow
the classifier to pick up the difference as opposed to purely working on the
generated noise. This is designed to lead to a classifier that favors group B
with respect to label 1.

    >>> import numpy as np 
    >>> import matplotlib.pyplot as plt 
    >>> np.random.seed(0)  # set seed for consistent results
    >>> X1 = np.random.normal(0.5, 0.1, (200, 1))
    >>> X2 = np.random.normal(0.5, 0.1, (200, 1))
    >>> X2[100:] += 0.15  # skew group B
    >>> X = np.stack((X1, X2), axis=1).squeeze()
    >>> y_groupA = np.random.choice([0,1], (100, 1), p=(0.5, 0.5))
    >>> y_groupB = np.random.choice([0,1], (100, 1), p=(0.3, 0.7))
    >>> y = np.vstack((y_groupA, y_groupB)).squeeze()
    >>> plt.bar([0, 1], [np.sum(y_groupA), np.sum(y_groupB)], color = 'g')
    >>> plt.xticks([0, 1], ["group A", "group B"])
    >>> plt.show() 

Firstly, Fairlearn provides fairness-related metrics that can be compared
between groups and for the overall population. Using existing metric
definitions from
`scikit-learn <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_
we can evaluate metrics to get a group summary as below:

    >>> from fairlearn.metrics import group_summary
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.svm import SVC
    >>> classifier = SVC()
    >>> classifier.fit(X, y)
    >>> y_true = y
    >>> y_pred = classifier.predict(X)
    >>> group = ["A"] * 100 + ["B"] * 100
    >>> group_summary(accuracy_score, y_true, y_pred, sensitive_features=group)
    >>> group_summary(accuracy_score, y_true, y_pred, sensitive_features=gender)
    {'overall': 0.585, 'by_group': {'A': 0.48, 'B': 0.69}}

Additionally, Fairlearn has lots of other standard metrics built-in, such as
selection rate, i.e., the percentage of the population with label 1:

    >>> from fairlearn.metrics import selection_rate_group_summary
    >>> selection_rate_group_summary(y_true, y_pred, sensitive_features=group)
    {'overall': 0.89, 'by_group': {'A': 0.79, 'B': 0.99}}

For a visual representation of the metrics try out the Fairlearn dashboard.
While this page shows only screenshots, the actual dashboard is interactive.

    >>> from fairlearn.widget import FairlearnDashboard
    >>> FairlearnDashboard(sensitive_features=group,
                           sensitive_feature_names=['group'],
                           y_true=y_true,
                           y_pred={"initial model": y_pred})

.. image:: ../img/fairlearn-dashboard-start.png

.. image:: ../img/fairlearn-dashboard-sensitive-features.png

.. image:: ../img/fairlearn-dashboard-performance.png

.. image:: ../img/fairlearn-dashboard-disparity-performance.png

.. image:: ../img/fairlearn-dashboard-disparity-predictions.png

If we observe disparities between groups we may want to create a new model
while specifying an appropriate fairness constraint. Note that the choice of
fairness constraints is crucial for the resulting model, and varies based on
application context. If selection rate is highly relevant for fairness in this
contrived example, we can attempt to mitigate the observed disparity using the
corresponding fairness constraint called Demographic Parity. In real world
applications we need to be mindful of the sociotechnical context when making
such decisions. The Exponentiated Gradient mitigation technique used fits the
provided classifier using Demographic Parity as the objective, leading to
a vastly reduced difference in selection rate:

    >>> from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    >>> constraint = DemographicParity()
    >>> classifier = SVC()
    >>> mitigator = ExponentiatedGradient(classifier, constraint)
    >>> mitigator.fit(X, y_true, sensitive_features=group)
    >>> y_pred_mitigated = mitigator.predict(X)
    >>> selection_rate_group_summary(y_true, y_pred_mitigated, sensitive_features=group)
    {'overall': 0.98, 'by_group': {'A': 0.97, 'B': 0.99}}

Similarly, we can explore the difference between the initial model and the
mitigated model with respect to selection rate and accuracy in the dashboard:

    >>> FairlearnDashboard(sensitive_features=group,
                           sensitive_feature_names=['group'],
                           y_true=y_true,
                           y_pred={"initial model": y_pred, "mitigated model": y_pred_mitigated})

.. image:: ../img/fairlearn-dashboard-comparison.png


What's next?
------------

Please refer to our :ref:`user_guide` for a comprehensive view on Fairness in
Machine Learning and how Fairlearn fits in, as well as an exhaustive guide on
all parts of the toolkit. For concrete examples check out the
:ref:`sphx_glr_auto_examples` section.
