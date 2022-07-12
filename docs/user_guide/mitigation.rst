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
   *  - :class:`~fairlearn.postprocessing.Thresholder`
      - Postprocessing algorithm that outputs a binary prediction and allows
        the user to manually specify group-specific decision thresholds.
      - ✔
      - ✘
      - N/A
   *  - :class:`~fairlearn.postprocessing.RejectOptionClassifier`
      - Postprocessing algorithm based on the paper *Decision Theory for
        Discrimination-aware Classification* :footcite:`kamiran2012rejectoptionclassifier`.
        This algorithm alters decisions made with low certainty in favor of the group to upselect.
      - ✔
      - ✘
      - N/A
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

.. _preprocessing:

Preprocessing
--------------
   
.. currentmodule:: fairlearn.preprocessing

Preprocessing algorithms transform the dataset to mitigate possible unfairness
present in the data.
Preprocessing algorithms in Fairlearn follow the :class:`sklearn.base.TransformerMixin`
class, meaning that they can :code:`fit` to the dataset and :code:`transform` it
(or :code:`fit_transform` to fit and transform in one go).

.. _correlation_remover:

Correlation Remover
~~~~~~~~~~~~~~~~~~~
Sensitive features can be correlated with non-sensitive features in the dataset.
By applying the :code:`CorrelationRemover`, these correlations are projected away
while details from the original data are retained as much as possible (as measured
by the least-squares error). The user can control the level of projection via the
:code:`alpha` parameter. In mathematical terms, assume we have the original dataset
:math:`X` which contains a set of sensitive attributes :math:`S` and a set of
non-sensitive attributes :math:`Z`. The removal of correlation is then
described as:

.. math::
        \min _{\mathbf{z}_{1}, \ldots, \mathbf{z}_{n}} \sum_{i=1}^{n}\left\|\mathbf{z}_{i}
        -\mathbf{x}_{i}\right\|^{2} \\
        \text{subject to} \\
        \frac{1}{n} \sum_{i=1}^{n} \mathbf{z}_{i}\left(\mathbf{s}_{i}-\overline{\mathbf{s}}
        \right)^{T}=\mathbf{0}

The solution to this problem is found by centering sensitive features, fitting a
linear regression model to the non-sensitive features and reporting the residual.
The columns in :math:`S` will be dropped from the dataset :math:`X`.
The amount of correlation that is removed can be controlled using the
:code:`alpha` parameter. This is described as follows:

.. math::
        X_{\text{tfm}} = \alpha X_{\text{filtered}} + (1-\alpha) X_{\text{orig}}

Note that the lack of correlation does not imply anything about statistical dependence.
In particular, since correlation measures linear relationships, it might still be
possible that non-linear relationships exist in the data. Therefore, we expect this
to be most appropriate as a preprocessing step for (generalized) linear models.

In the example below, the `Diabetes 130-Hospitals <https://www.openml.org/d/43874>`_
is loaded and the correlation between the African American race and
the non-sensitive features is removed. This dataset contains more races,
but in example we will only focus on the African American race.
The :code:`CorrelationRemover` will drop the sensitive features from the dataset.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.preprocessing import CorrelationRemover
    >>> import pandas as pd
    >>> from sklearn.datasets import fetch_openml
    >>> data = fetch_openml(data_id=43874, as_frame=True)
    >>> X = data.data[["race", "time_in_hospital", "had_inpatient_days", "medicare"]]
    >>> X = pd.get_dummies(X)
    >>> X = X.drop(["race_Asian",
    ...                     "race_Caucasian",
    ...                     "race_Hispanic",
    ...                     "race_Other",
    ...                     "race_Unknown",
    ...                     "had_inpatient_days_False",
    ...                     "medicare_False"], axis=1)
    >>> cr = CorrelationRemover(sensitive_feature_ids=['race_AfricanAmerican'])
    >>> cr.fit(X)
    CorrelationRemover(sensitive_feature_ids=['race_AfricanAmerican'])
    >>> X_transform = cr.transform(X)

In the visualization below, we see the correlation values in the
original dataset. We are particularly interested in the correlations
between the 'race_AfricanAmerican' column and the three non-sensitive attributes
'time_in_hospital', 'had_inpatient_days' and 'medicare_True'. The target
variable is also included in these visualization for completeness, and it is
defined as a binary feature which indicated whether the readmission of a patient
occurred within 30 days of the release. We see that 'race_AfricanAmerican' is
not highly correlated with the three mentioned attributes, but we want to remove
these correlations nonetheless. The code for generating the correlation matrix
can be found in
`this example notebook
<../auto_examples/plot_correlationremover_before_after.html>`_.

.. figure:: ../auto_examples/images/sphx_glr_plot_correlationremover_before_after_001.png
    :align: center
    :target: ../auto_examples/plot_correlationremover_before_after.html

In order to see the effect of :class:`CorrelationRemover`, we visualize
how the correlation matrix has changed after the transformation of the
dataset. Due to rounding, some of the 0.0 values appear as -0.0. Either
way, the :code:`CorrelationRemover` successfully removed all correlation
between 'race_AfricanAmerican' and the other columns while retaining
the correlation between the other features.

.. figure:: ../auto_examples/images/sphx_glr_plot_correlationremover_before_after_002.png
    :align: center
    :target: ../auto_examples/plot_correlationremover_before_after.html

We can also use the :code:`alpha` parameter with for instance :math:`\alpha=0.5`
to control the level of filtering between the sensitive and non-sensitive features.

.. doctest:: mitigation

    >>> cr = CorrelationRemover(sensitive_feature_ids=['race_AfricanAmerican'], alpha=0.5)
    >>> cr.fit(X)
    CorrelationRemover(alpha=0.5, sensitive_feature_ids=['race_AfricanAmerican'])
    >>> X_transform = cr.transform(X)

As we can see in the visulization below, not all correlation between
'race_AfricanAmerican' and the other columns was removed. This is exactly what
we would expect with :math:`\alpha=0.5`.

.. figure:: ../auto_examples/images/sphx_glr_plot_correlationremover_before_after_003.png
    :align: center
    :target: ../auto_examples/plot_correlationremover_before_after.html

.. _postprocessing:

Postprocessing
--------------

.. currentmodule:: fairlearn.postprocessing

What is a decision threshold?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Many postprocessing algorithms make decisions based on thresholds. But what
is a decision threshold?

In binary classification, the decision threshold is a number which represents the border between
positive and negative predictions. In many implementations of classification algorithms, the
default decision threshold is 0.5, which means that a sample with a probability to get 1
(obtained by calling :code:`estimator.predict_proba()[:,1]`) of 0.6 will be classified
as 1, and a sample with probability to get 1 of 0.4 will be classified as 0.
If we change this threshold to 0.7, both samples will be classified
as 0.

Thresholder
~~~~~~~~~~~~

What it does
^^^^^^^^^^^^
:class:`fairlearn.postprocessing.Thresholder` is a post processing technique
which outputs a binary prediction based on group-specific thresholds. The
specified thresholds can work in two directions
(:ref:`positive<pos_thresholds>`/:ref:`negative<negative_thresholds>`), and can
be applied to the output of the :code:`predict()` function of a regressor,
the second element of each prediction obtained from :code:`predict_proba()` of a classifier, or on
:code:`decision_function()` of, for example, :class:`sklearn.svm.SVC` or
:class:`sklearn.linear_model.LogisticRegression`. Examples of these different applications
are presented below.

Why use it
^^^^^^^^^^^^
:class:`Thresholder` is an easy to interpret tool, that can help explore
trade-offs between fairness metrics and predictive performance. Where an algorithm like
:class:`ThresholdOptimizer` “blindly” satisfies
a specific constraint, :class:`Thresholder` aids the user in manually exploring
different trade-offs, which can help gain more insight into the problem. Also,
since :class:`Thresholder` is a post-processing technique, it is not necessary to alter
an already trained estimator, which might be the only feasible unfairness
mitigation approach when developers cannot influence training, due to
practical reasons or security or privacy.

In the next section, we will explain the different ways in which thresholds can be specified, and
show how to investigate the effect of different thresholds. 

How to use it
^^^^^^^^^^^^^^

The group-specific thresholds are specified in :code:`threshold_dict`. This is
a dictionary where the keys are the sensitive feature value of the sub-group,
and the values corresond to the threshold. Below, we will show all different
kind of ways to specify thresholds.

Furthermore, we will plot some information about the obtained predictions,
in order to help gain an insight in the way :class:`Thresholder` works. Throughout
this example, unless specified otherwise, we will be using the
:ref:`clinical hospital readmissions dataset<hospital_readmissions_dataset>`.

.. _default_predictions:

Classifier: Default predictions
********************************
The classifier we will be using throughout this example is a
:class:`sklearn.ensemble.RandomForestClassifier`.
We first investigate the output when we call predict
on this estimator, without using :class:`Thresholder`. This means all groups
are thresholded at 0.5 (this is the default threshold used in most implementations of
classification algorithms in scikit-learn).

Considering the nature of the problem at hand, important aspects of our model
to investigate are:
the selection rate (due to capacity constraints not all patients can be selected), and
the false negative rate (we don't want people that need extra care not receiving it).

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> #Import dataset, do train_test_split, balance positive/negative samples
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import fetch_openml
    
    >>> #Get dataset, drop columns with unknown sf values
    >>> data = fetch_openml(data_id=43874)

    >>> df = data.data
    >>> Y = data.target

    >>> df = df[df['gender'] != 'Unknown/Invalid']
    >>> df = df[df['race'] != 'Unknown']

    >>> Y = Y[df.index]

    >>> sensitive_features =  df.loc[:, 'race']

    >>> #Create dummy's, drop sf from X, drop columns that correlate with Y 
    >>> X = pd.get_dummies(df.drop(columns=["race", 
    ...                                     "discharge_disposition_id",
    ...                                     "readmitted",
    ...                                     "readmit_binary"]))

    >>> X_train, X_test, Y_train, Y_test, sf_train, sf_test, df_train, df_test = train_test_split(
    ...        X,
    ...        Y,
    ...        sensitive_features,
    ...        df,
    ...        test_size=0.50,
    ...        stratify=Y,
    ...        random_state=445)

    >>> # Resample dataset such that training data has same number of positive and negative samples
    >>> # Done to prevent overfitting towards the overrepresented class
    
    >>> def resample_dataset(X_train, Y_train, sf_train):
    ...   negative_ids = Y_train[Y_train == 0].index
    ...   positive_ids = Y_train[Y_train == 1].index
    ...   np.random.seed(1)
    ...   balanced_ids = positive_ids.union(np.random.choice(a=negative_ids, size=len(positive_ids)))
    ...
    ...   X_train = X_train.loc[balanced_ids, :]
    ...   Y_train = Y_train.loc[balanced_ids]
    ...   sf_train = sf_train.loc[balanced_ids]
    ...
    ...   return X_train, Y_train, sf_train

    >>> X_train, Y_train, sf_train = resample_dataset(X_train, Y_train, sf_train)
   

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> #Train classifier, make and inspect default predictions

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from fairlearn.metrics import MetricFrame, false_negative_rate
    >>> from fairlearn.postprocessing import plot_positive_predictions, plot_proba_distribution

    >>> classifier = RandomForestClassifier(random_state=1).fit(X_train,Y_train) 
    >>> Y_pred_proba_clf = classifier.predict_proba(X_test)[:, 1]
    >>> Y_pred_clf = classifier.predict(X_test)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_clf, 
    ...             sensitive_features=sf_test).by_group
    race
    AfricanAmerican    0.383205
    Asian              0.414634
    Caucasian          0.401478
    Hispanic           0.391304
    Other              0.359375
    Name: false_negative_rate, dtype: float64

    >>> plot_positive_predictions(sf_test,Y_pred_clf)    

.. figure:: ../auto_examples/images/user_guide_thresholder_no_thresholds.png
    :align: center

It can be useful to also plot the distribution of the probabilities obtained
by calling :code:`predict_proba()[:,1]`. This can give extra insights it what
effect choosing a different threshold will have on the amount of positive
predictions. 

.. doctest:: mitigation
    :options: +NORMALIZE_WHITESPACE

    >>> plot_proba_distribution(sf_test,Y_pred_proba_clf)

.. figure:: ../auto_examples/images/user_guide_thresholder_proba_distr.png
    :align: center

.. _pos_thresholds:

Two ways of specifying positive thresholds
*******************************************
Say, you want to change the threshold of a specific group to 0.4 (all 
probabilities greater than 0.4 will be predicted as 1). By default,
:class:`Thresholder` assumes a provided threshold to be positive, which
means that there are two ways to specify this threshold in the
:code:`threshold_dict`: 

    - 0.4
    - ('>', 0.4)

Changing this threshold for groups 'Asian' and 'Other' shows the impact a
change in threshold of just 0.1 can have on the false negative rates and the
amount of positive predictions. Note that the purpose of this user guide is
solely to show functionality, and that it is very much debatable whether these
are the most appropriate changes to make given the problem at hand.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.postprocessing import Thresholder

    >>> threshold_dict = {'Asian': 0.4, 'Other': ('>' , 0.4)}

    >>> thresholder = Thresholder(estimator=classifier,
    ...                           threshold_dict=threshold_dict,
    ...                           prefit=True,
    ...                           predict_method='predict_proba')\
    ...                           .fit(X_train,Y_train,sensitive_features=sf_train)
     
    >>> Y_pred_thresholded = thresholder.predict(X_test,sensitive_features=sf_test)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_thresholded, 
    ...             sensitive_features=sf_test).by_group
    race
    AfricanAmerican    0.383205
    Asian              0.097561
    Caucasian          0.401478
    Hispanic           0.391304
    Other              0.140625
    Name: false_negative_rate, dtype: float64

    >>> plot_positive_predictions(sf_test,Y_pred_thresholded)

.. figure:: ../auto_examples/images/user_guide_thresholder_two_ways_pos_threshold.png
    :align: center

.. _negative_thresholds:

Specifying a negative threshold
********************************
It is also possible to specify a negative threshold. This means that all
instances with a probability *less than* the specified threshold will be
predicted as 1. This is done by adjusting the sign in the threshold. 
Again, the purpose of this example is simply to show the different
functionalities of :class:`Thresholder`, and is not necessarily the right course
of action for this particular model/dataset. 

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> threshold_dict = {'Asian': ('<' , 0.4), 'Other': ('<' , 0.4)}

    >>> thresholder = Thresholder(estimator=classifier,
    ...                           threshold_dict=threshold_dict,
    ...                           prefit=True,
    ...                           predict_method='predict_proba')\
    ...                           .fit(X_train,Y_train,sensitive_features=sf_train)
    
    >>> Y_pred_switched_threshold = thresholder.predict(X_test,sensitive_features=sf_test)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_switched_threshold, 
    ...             sensitive_features=sf_test).by_group
    race
    AfricanAmerican    0.383205
    Asian              0.926829
    Caucasian          0.401478
    Hispanic           0.391304
    Other              0.875000
    Name: false_negative_rate, dtype: float64

    >>> plot_positive_predictions(sf_test,Y_pred_switched_threshold)

.. figure:: ../auto_examples/images/user_guide_thresholder_switch_predictions.png
    :align: center

Specifying a default threshold
********************************
In the previous examples, the threshold used for the groups that were not
specified in the :code:`threshold_dict` was the default threshold of 0.5 (if
:code:`predict_method` is either :code:`predict_proba` or :code:`predict`). It
is possible to specify a different default threshold using the
:code:`default_threshold` parameter. Similar as before, the default threshold
can be specified either as a float or tuple.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> threshold_dict = {'Asian': 0.4, 'Other': ('>', 0.4)}
    >>> thresholder = Thresholder(estimator=classifier,
    ...                           threshold_dict=threshold_dict,
    ...                           prefit=True,
    ...                           predict_method='predict_proba',
    ...                           default_threshold=0.6)\
    ...                           .fit(X_train,Y_train,sensitive_features=sf_train)
    
    >>> default_thresholded_pred = thresholder.predict(X_test,sensitive_features=sf_test)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=default_thresholded_pred, 
    ...             sensitive_features=sf_test).by_group
    race
    AfricanAmerican    0.637066
    Asian              0.097561
    Caucasian          0.664588
    Hispanic           0.643478
    Other              0.140625
    Name: false_negative_rate, dtype: float64

    >>> plot_positive_predictions(sf_test,default_thresholded_pred)

.. figure:: ../auto_examples/images/user_guide_thresholder_default_threshold_0_6.png
    :align: center

.. _regressor:

Specify thresholds for a regressor
***********************************
It is also possible to threshold the predicted output of a regressor.
This can be useful if it is desired to transform the continuous output
of a regressor into a binary prediction. To illustrate how this could work,
consider a simple :class:`sklearn.linear_model.LinearRegression` example on the
:ref:`boston housing dataset<boston_dataset>`, where the sensitive feature
is the LSTAT variable categorized into four groups. 

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.datasets import fetch_boston
    >>> from sklearn.linear_model import LinearRegression

    >>> X_boston, Y_boston = fetch_boston(as_frame=True, return_X_y=True)

    >>> sf_boston = pd.cut(X_boston.LSTAT, bins=[0,10,20,30,40],labels=['0-10','10-20','20-30','30-40'])

    >>> X_boston = pd.get_dummies(X_boston)

    >>> X_boston_train, X_boston_test, Y_boston_train, Y_boston_test, sf_boston_train, sf_boston_test = train_test_split(
    ...     X_boston,
    ...     Y_boston,
    ...     sf_boston,
    ...     test_size=0.3,
    ...     random_state=2)

    >>> regressor = LinearRegression().fit(X_boston_train,Y_boston_train)

    >>> Y_pred_regressor = regressor.predict(X_boston_test)

    >>> print('First, the output is continuous: \n{}'.format(Y_pred_regressor[:5]))
    First, the output is continuous: 
    [23.50867458 22.94718558 33.00839918 33.9504757   3.22390622]

    >>> threshold_dict = {'0-10': 33.5}

    >>> thresholder_regression = Thresholder(estimator=regressor,
    ...                                      threshold_dict=threshold_dict,
    ...                                      prefit=True,
    ...                                      predict_method='predict')\
    ...                                      .fit(X_boston_train, Y_boston_train, sensitive_features=sf_boston_train)

    >>> Y_pred_thresholded_regressor = thresholder_regression.predict(X_boston_test, sensitive_features=sf_boston_test)

    >>> print('Now the output is a binary prediction: \n{}'.format(Y_pred_thresholded_regressor[:5]))
    Now the output is a binary prediction: 
    0    1.0
    1    1.0
    2    0.0
    3    1.0
    4    1.0
    dtype: float64

Specify threshold for predict_method = 'decision_function'
***********************************************************
If the classifier you are using is either
:py:class:`sklearn.linear_model.LogisticRegression` or from
:py:mod:`sklearn.svm`, it is also possible to threshold the predictions
on the output of the decision function. Syntax-wise, this works exactly the
same as specifying a threshold as in the previous examples. Note that
if :code:`predict_method` is set to :code:`decision_function`, the default
threshold will automatically be set to 0, as that is the value used by
classifiers that predict using :code:`decision_function`.


.. doctest:: mitigation
    :options: +NORMALIZE_WHITESPACE

    >>> from sklearn.linear_model import LogisticRegression
    >>> from fairlearn.postprocessing import plot_histograms_per_group
    
    >>> classifier_log_reg = LogisticRegression().fit(X_train,Y_train) 
    >>> Y_pred_decision_func = classifier_log_reg.decision_function(X_test)

    >>> #Check distribution of decision function output
    >>> plot_histograms_per_group(sf_test,Y_pred_decision_func)

.. figure:: ../auto_examples/images/user_guide_thresholder_decision_func_distr.png
    :align: center

.. doctest:: mitigation
    :options: +NORMALIZE_WHITESPACE

    >>> #Check unthresholded predictions
    >>> Y_pred = classifier_log_reg.predict(X_test)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred, 
    ...             sensitive_features=sf_test).by_group
    race
    AfricanAmerican    0.425676
    Asian              0.487805
    Caucasian          0.453915
    Hispanic           0.426087
    Other              0.437500
    Name: false_negative_rate, dtype: float64

    >>> plot_positive_predictions(sf_test,Y_pred)

.. figure:: ../auto_examples/images/user_guide_thresholder_decision_func_positive_pred_no_threshold.png
    :align: center

.. doctest:: mitigation
    :options: +NORMALIZE_WHITESPACE    

    >>> #Specify threshold, and inspect the difference
    >>> threshold_dict = {'Asian': -0.5}

    >>> thresholder = Thresholder(estimator=classifier_log_reg,
    ...                           threshold_dict=threshold_dict,
    ...                           prefit=True,
    ...                           predict_method='decision_function')\
    ...                           .fit(X_train,Y_train,sensitive_features=sf_train)

    >>> Y_pred_thresholded = thresholder.predict(X_test,sensitive_features=sf_test)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_thresholded, 
    ...             sensitive_features=sf_test).by_group
    race
    AfricanAmerican    0.425676
    Asian              0.048780
    Caucasian          0.453915
    Hispanic           0.426087
    Other              0.437500
    Name: false_negative_rate, dtype: float64

    >>> plot_positive_predictions(sf_test,Y_pred_thresholded)

.. figure:: ../auto_examples/images/user_guide_thresholder_decision_func_positive_pred_yes_threshold.png
    :align: center    




Specify threshold for multiple sensitive features
**************************************************
All examples considered above assume that the data at hand has just a single
sensitive feature. It is also possible that you have a dataset with multiple
sensitive features. Fortunately, :class:`Thresholder` can also deal with this.
In order to specify thresholds for a group identified by multiple sensitive
features, specify the group by listing all sensitive feature values
corresponding to that group in a single tuple. In the example below, instead
of only considering 'race' as a sensitive feature (as was done in the 
classification examples above), we will now also include 'gender'. This means
that when we want to specify the threshold for a specific sub-group, we will
have to specify both the race and gender. For example, the subgroup of Asian
women will be specified as :code:`('Asian','Female')`.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> #Include gender as sensitive feature as well
    >>> A_multiple = df.loc[:, ['race','gender']]
    >>> A_multiple_train = A_multiple.loc[sf_train.index, :]
    >>> A_multiple_test = A_multiple.loc[sf_test.index, :]

    >>> #Remove gender from the X data, as it is now a sensitive feature
    >>> X_train_multiple_sf = X_train.drop(columns=['gender_Female','gender_Male'])
    >>> X_test_multiple_sf = X_test.drop(columns=['gender_Female','gender_Male'])

    >>> #Train a classifier on the new data and predict
    >>> classifier_multiple = RandomForestClassifier(random_state=1).fit(X_train_multiple_sf,Y_train)
    
    >>> Y_pred_clf_multiple = classifier_multiple.predict(X_test_multiple_sf)

    >>> #Inspect disaggregated metrics
    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_clf_multiple, 
    ...             sensitive_features=A_multiple_test).by_group
    race             gender
    AfricanAmerican  Female    0.404459
                    Male      0.370098
    Asian            Female    0.384615
                    Male      0.464286
    Caucasian        Female    0.389956
                    Male      0.412948
    Hispanic         Female    0.483333
                    Male      0.363636
    Other            Female    0.451613
                    Male      0.303030
    Name: false_negative_rate, dtype: float64

    >>> plot_positive_predictions(A_multiple_test,Y_pred_clf_multiple)   

.. figure:: ../auto_examples/images/user_guide_thresholder_multiple_sf_no_threshold.png
    :align: center

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> threshold_dict = {('Asian','Female'): 0.45}
    >>> thresholder = Thresholder(estimator=classifier_multiple,
    ...                           threshold_dict=threshold_dict,
    ...                           prefit=True,
    ...                           predict_method='predict_proba')\
    ...                           .fit(X_train_multiple_sf,Y_train,sensitive_features=A_multiple_train)

    >>> Y_pred_multiple_sf_thresholded = thresholder.predict(X_test_multiple_sf,sensitive_features=A_multiple_test)
  
    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_multiple_sf_thresholded, 
    ...             sensitive_features=A_multiple_test).by_group
    race             gender
    AfricanAmerican  Female    0.404459
                     Male      0.370098
    Asian            Female    0.384615
                     Male      0.464286
    Caucasian        Female    0.389956
                     Male      0.412948
    Hispanic         Female    0.483333
                     Male      0.363636
    Other            Female    0.451613
                     Male      0.303030
    Name: false_negative_rate, dtype: float64

    >>> plot_positive_predictions(A_multiple_test,Y_pred_multiple_sf_thresholded)

.. figure:: ../auto_examples/images/user_guide_thresholder_multiple_sf_with_threshold.png
    :align: center

RejectOptionClassifier
~~~~~~~~~~~~~~~~~~~~~~~

What it does
^^^^^^^^^^^^^^
:class:`RejectOptionClassifier` is a postprocessing technique that is
based on the hypothesis that "biased" decisions are more likely to be made close
to the decision boundary. Translating this hypothesis to a relatable context: a decision you have
made is more likely to be wrong if you are not very certain that this decision is the correct one.
:class:`RejectOptionClassifier` implements this idea by splitting all predictions into two
categories: predictions made with sufficient certainty, and predictions made with insufficient
certainty. The latter category is also called the **critical region**. It is hypothesized that
in this area of low certainty, one group will, unfairly, be selected more often than another
group. In our hospital example, this refers to being selected for the primary care program
(=receive label 1). :class:`RejectOptionClassifier` tries to counter this by labeling all
instances inside the critical region based on the group to which they belong. All instances from
the group hypothesized to be selected less (set with the parameter :code:`group_to_upselect`)
will be selected, whilst all instances from the other group
(set with the parameter :code:`group_to_downselect`) will not be selected. Outside the critical
region, the result of the classification remains unchanged. Note that the sensitive feature is
expected to be binary.

The size of the critical region is regulated by the parameter :code:`critical_width`
(0 <= :code:`critical_width` <= 1); it contains all instances that have a probability to get 1
in the range [0.5 - :code:`critical_width`/2, 0.5 + :code:`critical_width`/2].

Why use it
^^^^^^^^^^
Like :class:`Thresholder`, :class:`RejectOptionClassifier` is an easy to interpret tool, which can
be used to explore different trade-offs, without blindly satisfying a constraint, where
:class:`RejectOptionClassifier` specifically focusses on decisions made with low certainty.

How to use it (and deal with multiple sensitive features)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using :class:`RejectOptionClassifier` requires manually specifying the critical region, the
group to (up/down) select, and the selection label.

As mentioned above, :class:`RejectOptionClassifier` assumes that the sensitive feature is binary.
Of course, it can happen that there are multiple sensitive features, or that the sensitive feature
is not binary (just like in our previous example). In this case it is easiest to create a new
sensitive feature column which is binary. To see how this, and using
:class:`RejectOptionClassifier` itself, works, consider the same 
:class:`sklearn.ensemble.RandomForestClassifier` and :ref:`data <hospital_readmissions_dataset>`
as in the section on :ref:`Thresholder<default_predictions>`.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> #create binary sensitive feature vector 
    >>> sf_train_reject_clf = sf_train.copy()\
    ...                        .replace(['Asian','Other'], 'upselect')\
    ...                        .replace(['Caucasian','AfricanAmerican','Hispanic'], 'downselect')

    >>> sf_test_reject_clf = sf_test.copy()\
    ...                        .replace(['Asian','Other'], 'upselect')\
    ...                        .replace(['Caucasian','AfricanAmerican','Hispanic'], 'downselect')

    >>> #Investigate predictions before using RejectOptionClassifier
    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_clf, 
    ...             sensitive_features=sf_test_reject_clf).by_group 
    race
    downselect    0.397810
    upselect      0.380952
    Name: false_negative_rate, dtype: float64

    >>> plot_positive_predictions(sf_test_reject_clf,Y_pred_clf)

.. figure:: ../auto_examples/images/user_guide_reject_option_clf_binary_output_before.png
    :align: center


.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> #import and use RejectOptionClassifier for predictions
    >>> from fairlearn.postprocessing import RejectOptionClassifier

    >>> reject_clf = RejectOptionClassifier(estimator=classifier,
    ...                                     critical_width=0.2,
    ...                                     group_to_upselect='upselect',
    ...                                     group_to_downselect='downselect',
    ...                                     selection_label=1,
    ...                                     prefit=True)\
    ...                                     .fit(X_train,Y_train,sensitive_features=sf_train_reject_clf)

    >>> Y_pred_reject_clf = reject_clf.predict(X_test,sensitive_features=sf_test_reject_clf)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_reject_clf, 
    ...             sensitive_features=sf_test_reject_clf).by_group
    race
    downselect    0.658942
    upselect      0.123810
    Name: false_negative_rate, dtype: float64

    >>> plot_positive_predictions(sf_test_reject_clf, Y_pred_reject_clf)

.. figure:: ../auto_examples/images/user_guide_reject_option_clf_binary_output.png
    :align: center

Changing the selection label
*********************************
By default, the selection label is assumed to be 1. It can happen of course, that the situation is
such that being selected corresponds to receiving the label 0. If this is the case, setting the
parameter :code:`selection_label` to 0 is the only thing that needs to be changed in the
configuration of :class:`RejectOptionClassifier`. The critical region itself remains the same,
whilst the thing that changes is the label each instance inside it receives; the
:code:`group_to_upselect` receives the label 0, while the :code:`group_to_downselect` receives 1.
Outside of the critical region, the labels do not change.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> reject_clf = RejectOptionClassifier(estimator=classifier,
    ...                                     critical_width=.2,
    ...                                     group_to_upselect='upselect',
    ...                                     group_to_downselect='downselect',
    ...                                     selection_label=0,
    ...                                     prefit=True)\
    ...                                     .fit(X_train,Y_train,sensitive_features=sf_train_reject_clf)

    >>> Y_pred_reject_clf = reject_clf.predict(X_test,sensitive_features=sf_test_reject_clf)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_reject_clf, 
    ...             sensitive_features=sf_test_reject_clf).by_group 
    race
    downselect    0.188686
    upselect      0.685714
    Name: false_negative_rate, dtype: float64

    >>> plot_positive_predictions(sf_test_reject_clf, Y_pred_reject_clf)

.. figure:: ../auto_examples/images/user_guide_reject_option_clf_binary_output_des_label_0.png
    :align: center

An important note
~~~~~~~~~~~~~~~~~~~~~~~
The examples shown are merely a way for the reader to get a feeling for the workings of
:class:`Thresholder` and :class:`RejectOptionClassifier`. It is not necessarily the case that the
choices made for the different parameters are the right ones, and anyone using the algorithms
is urged to think critically about how set these parameters.

.. _reductions:

Reductions
----------

.. currentmodule:: fairlearn.reductions

On a high level, the reduction algorithms within Fairlearn
enable unfairness mitigation for an arbitrary machine learning model with
respect to user-provided fairness constraints. All of the constraints currently supported
by reduction algorithms are group-fairness constraints. For more information on the
supported fairness constraints refer to :ref:`constraints_binary_classification`
and :ref:`constraints_regression`.

.. note::

   The choice of a fairness metric and fairness constraints is a crucial
   step in the AI development and deployment, and
   choosing an unsuitable constraint can lead to more harms.
   For a broader discussion of fairness as a
   sociotechnical challenge and how to view Fairlearn in this context refer to
   :ref:`fairness_in_machine_learning`.

The reductions approach for classification seeks to reduce binary
classification subject to fairness constraints to a sequence of weighted
classification problems (see :footcite:`agarwal2018reductions`), and similarly for regression (see :footcite:`agarwal2019fair`).
As a result, the reduction algorithms
in Fairlearn only require a wrapper access to any "base" learning algorithm.
By this we mean that the "base" algorithm only needs to implement :code:`fit` and
:code:`predict` methods, as any standard scikit-learn estimator, but it
does not need to have any knowledge of the desired fairness constraints or sensitive features.

From an API perspective this looks as follows in all situations

>>> reduction = Reduction(base_estimator, constraints, **kwargs)  # doctest: +SKIP
>>> reduction.fit(X_train, y_train, sensitive_features=sensitive_features)  # doctest: +SKIP
>>> reduction.predict(X_test)  # doctest: +SKIP

Fairlearn doesn't impose restrictions on the referenced :code:`base_estimator`
other than the existence of :code:`fit` and :code:`predict` methods.
At the moment, the :code:`base_estimator`'s :code:`fit` method also needs to
provide a :code:`sample_weight` argument which the reductions techniques use
to reweight samples.
In the future Fairlearn will provide functionality to handle this even
without a :code:`sample_weight` argument.

Before looking more into reduction algorithms, this section
reviews the supported fairness constraints. All of them
are expressed as objects inheriting from the base class :code:`Moment`.
:code:`Moment`'s main purpose is to calculate the constraint violation of a
current set of predictions through its :code:`gamma` function as well as to
provide :code:`signed_weights` that are used to relabel and reweight samples.

.. _constraints_binary_classification:

Fairness constraints for binary classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All supported fairness constraints for binary classification inherit from
:code:`UtilityParity`. They are based on some underlying metric called
*utility*, which can be evaluated on individual data points and is averaged
over various groups of data points to form the *utility parity* constraint
of the form

.. math::

    \text{utility}_{a,e} = \text{utility}_e \quad \forall a, e

where :math:`a` is a sensitive feature value and :math:`e` is an *event*
identifier. Each data point has only one value of a sensitive feature,
and belongs to at most one event. In many examples, there is only
a single event :math:`*`, which includes all the data points. Other
examples of events include :math:`Y=0` and :math:`Y=1`. The utility
parity requires that the mean utility within each event equals
the mean utility of each group whose sensitive feature is :math:`a`
within that event.

The class :code:`UtilityParity` implements constraints that allow
some amount of violation of the utility parity constraints, where
the maximum allowed violation is specified either as a difference
or a ratio.

The *difference-based relaxation* starts out by representing
the utility parity constraints as pairs of
inequalities

.. math::

    \text{utility}_{a,e} - \text{utility}_{e} \leq 0 \quad \forall a, e\\
    -\text{utility}_{a,e} + \text{utility}_{e} \leq 0 \quad \forall a, e

and then replaces zero on the right-hand side
with a value specified as :code:`difference_bound`. The resulting
constraints are instantiated as

    >>> UtilityParity(difference_bound=0.01)  # doctest: +SKIP

Note that satisfying these constraints does not mean
that the difference between the groups with the highest and
smallest utility in each event is bounded by :code:`difference_bound`.
The value of :code:`difference_bound` instead bounds
the difference between the utility of each group and the overall mean
utility within each event. This, however,
implies that the difference between groups in each event is
at most twice the value of :code:`difference_bound`.

The *ratio-based relaxation* relaxes the parity
constraint as

.. math::

    r \leq \dfrac{\text{utility}_{a,e}}{\text{utility}_e} \leq \dfrac{1}{r} \quad \forall a, e

for some value of :math:`r` in (0,1]. For example, if :math:`r=0.9`, this means
that within each event
:math:`0.9 \cdot \text{utility}_{a,e} \leq \text{utility}_e`, i.e., the utility for
each group needs to be at least 90% of the overall utility for the event, and
:math:`0.9 \cdot \text{utility}_e \leq \text{utility}_{a,e}`, i.e., the overall utility
for the event needs to be at least 90% of each group's utility.

The two ratio constraints can be rewritten as

.. math::

   - \text{utility}_{a,e} + r \cdot \text{utility}_e \leq 0 \quad \forall a, e \\
   r \cdot \text{utility}_{a,e} - \text{utility}_e \leq 0 \quad \forall a, e

When instantiating the ratio constraints, we use :code:`ratio_bound` for :math:`r`,
and also allow further relaxation by replacing the zeros on the right hand side
by some non-negative :code:`ratio_bound_slack`. The resulting instantiation
looks as

    >>> UtilityParity(ratio_bound=0.9, ratio_bound_slack=0.01)  # doctest: +SKIP

Similarly to the difference constraints, the ratio constraints do not directly
bound the ratio between the pairs of groups, but such a bound is implied.

.. note::

    It is not possible to specify both :code:`difference_bound` *and*
    :code:`ratio_bound` for the same constraint object.

.. _demographic_parity:

Demographic Parity
^^^^^^^^^^^^^^^^^^

A binary classifier :math:`h(X)` satisfies *demographic parity* if

.. math::
    
    \P[h(X) = 1 \given A = a] = \P[h(X) = 1] \quad \forall a
 
In other words, the selection rate or percentage of samples with label 1
should be equal across all groups. Implicitly this means the percentage
with label 0 is equal as well. In this case, the utility function
is equal to :math:`h(X)` and there is only a single event :math:`*`.

In the example below group :code:`"a"` has a selection rate of 60%,
:code:`"b"` has a selection rate of 20%. The overall selection rate is 40%,
so :code:`"a"` is `0.2` above the overall selection rate, and :code:`"b"` is
`0.2` below. Invoking the method :code:`gamma` shows the values
of the left-hand sides of the constraints described
in :ref:`constraints_binary_classification`, which is independent
of the provided :code:`difference_bound`. Note that the left-hand sides
corresponding to different values of :code:`sign` are just negatives
of each other.
The value of :code:`y_true` is in this example irrelevant to the calculations,
because the underlying utility in demographic parity, selection rate, does not
consider performance relative to the true labels, but rather proportions in
the predicted labels.

.. note::

    When providing :code:`DemographicParity` to mitigation algorithms, only use
    the constructor and the mitigation algorithm itself then invokes :code:`load_data`.
    The example below uses :code:`load_data` to illustrate how :code:`DemographicParity`
    instantiates inequalities from :ref:`constraints_binary_classification`.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.reductions import DemographicParity
    >>> from fairlearn.metrics import MetricFrame, selection_rate
    >>> import numpy as np
    >>> import pandas as pd
    >>> dp = DemographicParity(difference_bound=0.01)
    >>> X                  = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    >>> y_true             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  0 ,  0 ,  0 ])
    >>> y_pred             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  0 ,  0 ,  0 ])
    >>> sensitive_features = np.array(["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"])
    >>> selection_rate_summary = MetricFrame(metrics=selection_rate,
    ...                                      y_true=y_true,
    ...                                      y_pred=y_pred,
    ...                                      sensitive_features=pd.Series(sensitive_features, name="SF 0"))
    >>> selection_rate_summary.overall
        0.4
    >>> selection_rate_summary.by_group
    SF 0
    a    0.6
    b    0.2
    Name: selection_rate, dtype: float64
    >>> dp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> dp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a           0.2
                 b          -0.2
    -     all    a          -0.2
                 b           0.2
    dtype: float64
 
The ratio constraints for the demographic parity with :code:`ratio_bound`
:math:`r` (and :code:`ratio_bound_slack=0`) take form

.. math::

    r \leq \dfrac{\P[h(X) = 1 \given A = a]}{\P[h(X) = 1]} \leq \dfrac{1}{r} \quad \forall a

Revisiting the same example as above we get

.. doctest:: mitigation

    >>> dp = DemographicParity(ratio_bound=0.9, ratio_bound_slack=0.01)
    >>> dp.load_data(X, y_pred, sensitive_features=sensitive_features)
    >>> dp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a           0.14
                 b          -0.22
    -     all    a          -0.24
                 b           0.16
    dtype: float64

Following the expressions for the left-hand sides
of the constraints, we obtain

.. math::

    r \cdot \text{utility}_{a,*} - \text{utility}_* = 0.9 \times 0.6 - 0.4 = 0.14 \\
    r \cdot \text{utility}_{b,*} - \text{utility}_* = 0.9 \times 0.2 - 0.4 = -0.22 \\
    - \text{utility}_{a,*} + r \cdot \text{utility}_* = - 0.6 + 0.9 \times 0.4 = -0.24 \\
    - \text{utility}_{b,*} + r \cdot \text{utility}_* = - 0.2 + 0.9 \times 0.4 = 0.16 \\

.. _true_positive_rate_parity:
.. _false_positive_rate_parity:

True Positive Rate Parity and False Positive Rate Parity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A binary classifier :math:`h(X)` satisfies *true positive rate parity* if

.. math::

    \P[h(X) = 1 \given A = a, Y = 1] = \P[h(X) = 1 \given Y = 1] \quad \forall a

and *false positive rate parity* if

.. math::

    \P[h(X) = 1 \given A = a, Y = 0] = \P[h(X) = 1 \given Y = 0] \quad \forall a

In first case, we only have one event :math:`Y=1` and
ignore the samples with :math:`Y=0`, and in the second case vice versa.
Refer to :ref:`equalized_odds` for the fairness constraint type that simultaneously
enforce both true positive rate parity and false positive rate parity
by considering both events :math:`Y=0` and :math:`Y=1`.

In practice this can be used in a difference-based relaxation as follows:

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.reductions import TruePositiveRateParity
    >>> from fairlearn.metrics import true_positive_rate
    >>> import numpy as np
    >>> tprp = TruePositiveRateParity(difference_bound=0.01)
    >>> X                  = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    >>> y_true             = np.array([ 1 ,  1 ,  1 ,  1 ,  1,   1 ,  1 ,  0 ,  0 ,  0 ])
    >>> y_pred             = np.array([ 1 ,  1 ,  1 ,  1 ,  0,   0 ,  0 ,  1 ,  0 ,  0 ])
    >>> sensitive_features = np.array(["a", "b", "a", "a", "b", "a", "b", "b", "a", "b"])
    >>> tpr_summary = MetricFrame(metrics=true_positive_rate,
    ...                           y_true=y_true,
    ...                           y_pred=y_pred,
    ...                           sensitive_features=sensitive_features)
    >>> tpr_summary.overall
    0.5714285714285714
    >>> tpr_summary.by_group
    sensitive_feature_0
    a    0.75...
    b    0.33...
    Name: true_positive_rate, dtype: float64
    >>> tprp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> tprp.gamma(lambda X: y_pred)
    sign  event    group_id
    +     label=1  a           0.1785...
                   b          -0.2380...
    -     label=1  a          -0.1785...
                   b           0.2380...
    dtype: float64

.. note::

    When providing :code:`TruePositiveRateParity` or :code:`FalsePositiveRateParity`
    to mitigation algorithms, only use
    the constructor. The mitigation algorithm itself then invokes :code:`load_data`.
    The example uses :code:`load_data` to illustrate how :code:`TruePositiveRateParity`
    instantiates inequalities from :ref:`constraints_binary_classification`.

Alternatively, a ratio-based relaxation is also available:

.. doctest:: mitigation

    >>> tprp = TruePositiveRateParity(ratio_bound=0.9, ratio_bound_slack=0.01)
    >>> tprp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> tprp.gamma(lambda X: y_pred)
    sign  event    group_id
    +     label=1  a           0.1035...
                   b          -0.2714...
    -     label=1  a          -0.2357...
                   b           0.1809...
    dtype: float64

.. _equalized_odds:
    
Equalized Odds
^^^^^^^^^^^^^^

A binary classifier :math:`h(X)` satisfies *equalized odds* if it satisfies both
*true positive rate parity* and *false positive rate parity*, i.e.,

.. math::

    \P[h(X) = 1 \given A = a, Y = y] = \P[h(X) = 1 \given Y = y] \quad \forall a, y

The constraints represent the union of constraints for true positive rate
and false positive rate.

.. doctest:: mitigation

    >>> from fairlearn.reductions import EqualizedOdds
    >>> eo = EqualizedOdds(difference_bound=0.01)
    >>> eo.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> eo.gamma(lambda X: y_pred)
    sign  event    group_id
    +     label=0  a          -0.3333...
                   b           0.1666...
          label=1  a           0.1785...
                   b          -0.2380...
    -     label=0  a           0.3333...
                   b          -0.1666...
          label=1  a          -0.1785...
                   b           0.2380...
    dtype: float64

.. _error_rate_parity:

Error Rate Parity
^^^^^^^^^^^^^^^^^

The *error rate parity* requires that the error rates should be
the same across all groups. For a classifier :math:`h(X)`
this means that

.. math::

   \P[h(X) \ne Y \given A = a] = \P[h(X) \ne Y] \quad \forall a

In this case, the utility is equal to 1 if :math:`h(X)\ne Y` and equal to
0 if :math:`h(X)=Y`, and so large value of utility here actually correspond
to poor outcomes. The difference-based relaxation specifies that
the error rate of any given group should not deviate from
the overall error rate by more than the value of :code:`difference_bound`.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.reductions import ErrorRateParity
    >>> from sklearn.metrics import accuracy_score
    >>> accuracy_summary = MetricFrame(metrics=accuracy_score,
    ...                                y_true=y_true,
    ...                                y_pred=y_pred,
    ...                                sensitive_features=sensitive_features)
    >>> accuracy_summary.overall
    0.6
    >>> accuracy_summary.by_group
    sensitive_feature_0
    a    0.8
    b    0.4
    Name: accuracy_score, dtype: float64
    >>> erp = ErrorRateParity(difference_bound=0.01)
    >>> erp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> erp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a          -0.2
                 b           0.2
    -     all    a           0.2
                 b          -0.2
    dtype: float64

.. note::

    When providing :code:`ErrorRateParity` to mitigation algorithms, only use
    the constructor. The mitigation algorithm itself then invokes :code:`load_data`.
    The example uses :code:`load_data` to illustrate how :code:`ErrorRateParity`
    instantiates inequalities from :ref:`constraints_binary_classification`.

Alternatively, error rate parity can be relaxed via ratio constraints as

.. math::

   r \leq \dfrac{\P[h(X) \ne Y \given A = a]}{\P[h(X) \ne Y]} \leq \dfrac{1}{r} \quad \forall a

with a :code:`ratio_bound` :math:`r`. The usage is identical with other
constraints:

.. doctest:: mitigation

    >>> from fairlearn.reductions import ErrorRateParity
    >>> erp = ErrorRateParity(ratio_bound=0.9, ratio_bound_slack=0.01)
    >>> erp.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> erp.gamma(lambda X: y_pred)
    sign  event  group_id
    +     all    a          -0.22
                 b           0.14
    -     all    a           0.16
                 b          -0.24
    dtype: float64


Control features
^^^^^^^^^^^^^^^^

The above examples of :class:`Moment` (:ref:`demographic_parity`,
:ref:`True and False Positive Rate Parity <true_positive_rate_parity>`,
:ref:`equalized_odds` and :ref:`error_rate_parity`) all support the concept
of *control features* when applying their fairness constraints.
A control feature stratifies the dataset, and applies the fairness constraint
within each stratum, but not between strata.
One case this might be useful is a loan scenario, where we might want
to apply a mitigation for the sensitive features while controlling for some
other feature(s).
This should be done with caution, since the control features may have a
correlation with the sensitive features due to historical biases.
In the loan scenario, we might choose to control for income level, on the
grounds that higher income individuals are more likely to be able to repay
a loan.
However, due to historical bias, there is a correlation between the income level
of individuals and their race and gender.


Control features modify the above equations.
Consider a control feature value, drawn from a set of valid values
(that is, :math:`c \in \mathcal{C}`).
The equation given above for Demographic Parity will become:


.. math::
    P[h(X) = 1 | A = a, C = c] = P[h(X) = 1 | C = c] \; \forall a, c

The other constraints acquire similar modifications.


.. _constraints_multi_class_classification:

Fairness constraints for multiclass classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reductions approaches do not support multiclass classification yet at this
point. If this is an important scenario for you please let us know!

.. _constraints_regression:

Fairness constraints for regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The performance objective in the regression scenario is to minimize the
loss of our regressor :math:`h`. The loss can be expressed as
:class:`SquareLoss` or :class:`AbsoluteLoss`. Both take constructor arguments
:code:`min_val` and :code:`max_val` that define the value range within which
the loss is evaluated. Values outside of the value range get clipped.

.. doctest:: mitigation

    >>> from fairlearn.reductions import SquareLoss, AbsoluteLoss, ZeroOneLoss
    >>> y_true = [0,   0.3, 1,   0.9]
    >>> y_pred = [0.1, 0.2, 0.9, 1.3]
    >>> SquareLoss(0, 2).eval(y_true, y_pred)
    array([0.01, 0.01, 0.01, 0.16])
    >>> # clipping at 1 reduces the error for the fourth entry
    >>> SquareLoss(0, 1).eval(y_true, y_pred)
    array([0.01, 0.01, 0.01, 0.01])
    >>> AbsoluteLoss(0, 2).eval(y_true, y_pred)
    array([0.1, 0.1, 0.1, 0.4])
    >>> AbsoluteLoss(0, 1).eval(y_true, y_pred)
    array([0.1, 0.1, 0.1, 0.1])
    >>> # ZeroOneLoss is identical to AbsoluteLoss(0, 1)
    >>> ZeroOneLoss().eval(y_true, y_pred)
    array([0.1, 0.1, 0.1, 0.1])

When using Fairlearn's reduction techniques for regression it's required to
specify the type of loss by passing the corresponding loss object when
instantiating the object that represents our fairness constraint. The only
supported type of constraint at this point is :class:`BoundedGroupLoss`.

.. _bounded_group_loss:

Bounded Group Loss
^^^^^^^^^^^^^^^^^^

*Bounded group loss* requires the loss of each group to be below a
user-specified amount :math:`\zeta`. If :math:`\zeta` is chosen reasonably
small the losses of all groups are very similar.
Formally, a predictor :math:`h` satisfies bounded group loss at level
:math:`\zeta` under a distribution over :math:`(X, A, Y)` if

.. math::

    \E[loss(Y, h(X)) \given A=a] \leq \zeta \quad \forall a

In the example below we use :class:`BoundedGroupLoss` with
:class:`ZeroOneLoss` on two groups :code:`"a"` and :code:`"b"`.
Group :code:`"a"` has an average loss of :math:`0.05`, while group
:code:`"b"`'s average loss is :math:`0.5`.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.reductions import BoundedGroupLoss, ZeroOneLoss
    >>> from sklearn.metrics import mean_absolute_error
    >>> bgl = BoundedGroupLoss(ZeroOneLoss(), upper_bound=0.1)
    >>> X                  = np.array([[0], [1], [2], [3]])
    >>> y_true             = np.array([0.3, 0.5, 0.1, 1.0])
    >>> y_pred             = np.array([0.3, 0.6, 0.6, 0.5])
    >>> sensitive_features = np.array(["a", "a", "b", "b"])
    >>> mae_frame = MetricFrame(metrics=mean_absolute_error,
    ...                         y_true=y_true,
    ...                         y_pred=y_pred,
    ...                         sensitive_features=pd.Series(sensitive_features, name="SF 0"))
    >>> mae_frame.overall
    0.275
    >>> mae_frame.by_group
    SF 0
    a    0.05
    b    0.50
    Name: mean_absolute_error, dtype: float64
    >>> bgl.load_data(X, y_true, sensitive_features=sensitive_features)
    >>> bgl.gamma(lambda X: y_pred)
    group_id
    a    0.05
    b    0.50
    Name: loss, dtype: float64

.. note::

    In the example above the :code:`BoundedGroupLoss` object does not use the
    :code:`upper_bound` argument. It is only used by reductions techniques
    during the unfairness mitigation. As a result the constraint violation
    detected by :code:`gamma` is identical to the mean absolute error.

.. _exponentiated_gradient:

Exponentiated Gradient
~~~~~~~~~~~~~~~~~~~~~~

.. _grid_search:

Grid Search
~~~~~~~~~~~

.. _references:

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

References
~~~~~~~~~~

.. footbibliography::
