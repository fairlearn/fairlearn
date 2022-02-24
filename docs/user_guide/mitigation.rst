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
        *Approach to Fair Classification* [#2]_.
      - ✔
      - ✔
      - DP, EO, TPRP, FPRP, ERP, BGL
   *  - :class:`~fairlearn.reductions.GridSearch`
      - A wrapper (reduction) approach described in Section 3.4 of *A Reductions*
        *Approach to Fair Classification* [#2]_. For regression it acts as a
        grid-search variant of the algorithm described in Section 5 of
        *Fair Regression: Quantitative Definitions and Reduction-based*
        *Algorithms* [#1]_.
      - ✔
      - ✔
      - DP, EO, TPRP, FPRP, ERP, BGL
   *  - :class:`~fairlearn.postprocessing.Thresholder`
      - Postprocessing algorithm that outputs a binary prediction and allows
        the user to manually specify group-specific thresholds.
      - ✔
      - ✘
      - ✘
   *  - :class:`~fairlearn.postprocessing.RejectOptionClassifier`
      - Postprocessing algorithm based on the paper *Decision Theory for
        Discrimination-aware Classification* [#4]_. This algorithm 
        alters decisions made with low certainty in favor of the deprived group.
      - ✔
      - ✘
      - ✘
   *  - :class:`~fairlearn.postprocessing.ThresholdOptimizer`
      - Postprocessing algorithm based on the paper *Equality of Opportunity*
        *in Supervised Learning* [#3]_. This technique takes as input an
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

.. _postprocessing:

Postprocessing
--------------

.. currentmodule:: fairlearn.postprocessing

What is a threshold?
~~~~~~~~~~~~~~~~~~~~~
Many postprocessing algorithms make decisions based on thresholds. But what
is a threshold? 

A threshold is a number which represents the border between positive and
negative predictions. For most classification algorithms, the default
threshold is 0.5, which means that a sample with a probability to get 1
(obtained by calling :code:`estimator.predict_proba()[:,1]`) of 0.6 will be classified
as 1, and a sample with probability to get 1 of 0.4 will be classified as 0.
If this threshold would be changed to 0.7, both samples would be classified
as 0.

Thresholder
~~~~~~~~~~~~

What it does
^^^^^^^^^^^^
:class:`fairlearn.postprocessing.Thresholder` is a post processing technique
which outputs a binary prediction based on group-specific thresholds. The
specified thresholds can work in two directions (positive/negative), and can
be applied to the output of the :code:`predict()` function of a regressor,
:code:`predict_proba()[:,1]` of a classifier, or on
:code:`decision_function()` of SVC/LogisticRegression. Examples of these
different applications can be found below. 

Why use it
^^^^^^^^^^^^
:class:`Thresholder` is an easy to interpret tool, that can help explore different
trade-offs. Where an algorithm like 
:class:`fairlearn.postprocessing.ThresholdOptimizer` “blindly” satisfies
a specific constraint, :class:`Thresholder` aids the user in manually exploring
different trade-offs, which can help gain more insight into the data. Also,
since :class:`Thresholder` is a post-processing technique, it is not necessary to alter
an already trained estimator, which might be the only feasible unfairness
mitigation approach when developers cannot influence training, due to
practical reasons or security or privacy.

In the next section, we will consider all ways of specifying thresholds, and
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
`RandomForestClassifier <https://
scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_. 
We first investigate the output when we call predict
on this estimator, without using :class:`Thresholder`. This means all groups
are thresholded at 0.5 (this is the default threshold used by most
classification algorithms). 

Considering the nature of the problem at hand, important aspects of our model
to investigate are:

- How many patients are recommended for care (selection rate = what % of predictions are positive) 
- Reducing false negative rate (we don't want people that need extra care not receiving it)

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> #Import dataset, do train_test_split, balance positive/negative samples
    
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    
    >>> #Get dataset, drop columns with unknown sf values
    >>> df = pd.read_csv("https://raw.githubusercontent.com/fairlearn/talks/main/2021_scipy_tutorial/data/diabetic_preprocessed.csv")

    >>> df = df[df['gender'] != 'Unknown/Invalid']
    >>> df = df[df['race'] != 'Unknown']

    >>> Y, sensitive_features = df.loc[:, 'readmit_30_days'], df.loc[:, 'race']

    >>> #Create dummy's, drop A from X, drop columns that correlate with Y 
    >>> X = pd.get_dummies(df.drop(columns=["race", "discharge_disposition_id",
    ...                                     "readmitted",
    ...                                     "readmit_binary",
    ...                                     "readmit_30_days"]))

    >>> X_train, X_test, Y_train, Y_test, sf_train, sf_test, df_train, df_test = train_test_split(
    ...        X,
    ...        Y,
    ...        sensitive_features,
    ...        df,
    ...        test_size=0.50,
    ...        stratify=Y,
    ...        random_state=445)

    >>> # Resample dataset such that training data has same number of positive and negative samples
    
    >>> def resample_dataset(X_train, Y_train, sf_train):
    >>>   negative_ids = Y_train[Y_train == 0].index
    >>>   positive_ids = Y_train[Y_train == 1].index
    >>>   np.random.seed(1)
    >>>   balanced_ids = positive_ids.union(np.random.choice(a=negative_ids, size=len(positive_ids)))

    >>>   X_train = X_train.loc[balanced_ids, :]
    >>>   Y_train = Y_train.loc[balanced_ids]
    >>>   sf_train = sf_train.loc[balanced_ids]
    
    >>>   return X_train, Y_train, sf_train

    >>> X_train, Y_train, sf_train = resample_dataset(X_train, Y_train, sf_train)
   

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> #Train classifier, make and inspect default predictions

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from fairlearn.metrics import MetricFrame, false_negative_rate
    >>> from fairlearn.postprocessing import plot_positive_predictions, plot_proba_distribution

    >>> classifier = RandomForestClassifier(random_state=1)
    >>> classifier.fit(X_train,Y_train)
    >>> Y_pred_proba_clf = classifier.predict_proba(X_test)[:, 1]
    >>> Y_pred_clf = classifier.predict(X_test)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_clf, 
    ...             sensitive_features=sf_test).by_group
    race
    AfricanAmerican    0.381274
    Asian              0.439024
    Caucasian          0.399399
    Hispanic           0.365217
    Other              0.421875
    Name: false_negative_rate, dtype: object

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


Two ways of specifying positive thresholds
*******************************************
Say, you want to change the threshold of a specific group to 0.4 (all 
probabilities above 0.4 will be predicted as 1). There are two ways to
specify this threshold in the :code:`threshold_dict`: 

    - 0.4
    - ('>', 0.4)

Changing this threshold for groups 'Asian' and 'Other' shows the impact a
change in threshold of just 0.1 can have on the false negative rates and the
amount of positive predictions.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> from fairlearn.postprocessing import Thresholder

    >>> threshold_dict = {'Asian': 0.4, 'Other': ('>' , 0.4)}

    >>> thresholder = Thresholder(estimator=classifier,
    ...                           threshold_dict=threshold_dict,
    ...                           prefit=True,
    ...                           predict_method='predict_proba')
    
    >>> thresholder.fit(X_train,Y_train,sensitive_features=sf_train)
    >>> Y_pred_thresholded = thresholder.predict(X_test,sensitive_features=sf_test)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_thresholded, 
    ...             sensitive_features=sf_test).by_group
    race
    AfricanAmerican    0.381274
    Asian              0.121951
    Caucasian          0.399399
    Hispanic           0.365217
    Other                 0.125
    Name: false_negative_rate, dtype: object

    >>> plot_positive_predictions(sf_test,Y_pred_thresholded)

.. figure:: ../auto_examples/images/user_guide_thresholder_two_ways_pos_threshold.png
    :align: center

Specifying a negative threshold
********************************
It is also possible to specify a negative threshold. This means that all
instances with a probability less than the specified threshold will be
predicted as 1. This is done by switching the sign in the threshold. Doing
this for 'Asian' and 'Other' shows that this flips the false negative rate and
percentage of positive predictions compared to the previous example. Note that
we have to take a threshold just above 0.4 to make sure all the predictions
are switched, as there might be instances with a probability of exaclty 0.4. 

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> threshold_dict = {'Asian': ('<' , 0.400001), 'Other': ('<' , 0.400001)}

    >>> thresholder = Thresholder(estimator=classifier,
    ...                           threshold_dict=threshold_dict,
    ...                           prefit=True,
    ...                           predict_method='predict_proba')
    
    >>> thresholder.fit(X_train,Y_train,sensitive_features=sf_train)
    >>> Y_pred_switched_threshold = thresholder.predict(X_test,sensitive_features=sf_test)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_switched_threshold, 
    ...             sensitive_features=sf_test).by_group
    race
    AfricanAmerican    0.381274
    Asian              0.878049
    Caucasian          0.399399
    Hispanic           0.365217
    Other                 0.875
    Name: false_negative_rate, dtype: object

    >>> plot_positive_predictions(sf_test,Y_pred_switched_threshold)

.. figure:: ../auto_examples/images/user_guide_thresholder_switch_predictions.png
    :align: center

Specifying a default threshold
********************************
In the previous examples, the threshold used for the groups that were not
specified in the :code:`threshold_dict` was the default threshold of 0.5. It
is possible to specify a different default threshold using the
:code:`default_threshold` parameter. This threshold can be specified using all
ways shown above.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> threshold_dict = {'Asian': 0.4, 'Other': ('>', 0.4)}
    >>> thresholder = Thresholder(estimator=classifier,
    ...                           threshold_dict=threshold_dict,
    ...                           prefit=True,
    ...                           predict_method='predict_proba',
    ...                           default_threshold=0.6)

    >>> thresholder.fit(X_train,Y_train,sensitive_features=sf_train)
    >>> default_thresholded_pred = thresholder.predict(X_test,sensitive_features=sf_test)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=default_thresholded_pred, 
    ...             sensitive_features=sf_test).by_group
    race
    AfricanAmerican     0.67278
    Asian              0.121951
    Caucasian          0.673135
    Hispanic           0.704348
    Other                 0.125
    Name: false_negative_rate, dtype: object

    >>> plot_positive_predictions(sf_test,default_thresholded_pred)

.. figure:: ../auto_examples/images/user_guide_thresholder_default_threshold_0_6.png
    :align: center

Dealing with the Prefit parameter
**********************************
In all examples above, the :code:`prefit` parameter was set to :code:`True`.
This parameter indicates whether or not fit has been called on the supplied
estimator. If :code:`Prefit==False`, the :code:`Thresholder.fit()` function
will fit the estimator. For the final predictions it does not
matter whether or the estimator is fit beforehand or in the 
:code:`Thresholder.fit()` call:

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> threshold_dict = {'Asian': 0.4, 'Other': ('>', 0.4)}

    >>> #Prefit = False
    >>> classifier_no_prefit = RandomForestClassifier(random_state=1)

    >>> thresholder_no_prefit = Thresholder(estimator=classifier_no_prefit,
    ...                                     threshold_dict=threshold_dict,
    ...                                     prefit=False,
    ...                                     predict_method='predict_proba')
    
    >>> thresholder_no_prefit.fit(X_train,Y_train,sensitive_features=sf_train)
    >>> Y_pred_no_prefit = thresholder_no_prefit.predict(X_test,sensitive_features=sf_test)

    >>> #Prefit = True
    >>> classifier_prefit = RandomForestClassifier(random_state=1)
    >>> classifier_prefit.fit(X_train,Y_train)

    >>> thresholder_prefit = Thresholder(estimator=classifier_prefit,
    ...                                  threshold_dict=threshold_dict,
    ...                                  prefit=True,
    ...                                  predict_method='predict_proba')
    
    >>> thresholder_prefit.fit(X_train,Y_train,sensitive_features=sf_train)
    >>> Y_pred_prefit = thresholder_prefit.predict(X_test,sensitive_features=sf_test)

    >>> #Results are the same
    >>> print(np.array_equal(Y_pred_no_prefit,Y_pred_prefit))    
    True

Specify thresholds for a regressor
***********************************
It is also possible to threshold the predicted output of a regressor.
This can be useful if it is desired to transform the continuous output
of a regressor into a binary prediction. To illustrate how this could work,
consider a simple LinearRegression example on the
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

    >>> regressor = LinearRegression()
    >>> regressor.fit(X_boston_train,Y_boston_train)

    >>> Y_pred_regressor = regressor.predict(X_boston_test)

    >>> print('First, the output is continuous: \n{}'.format(Y_pred_regressor[:5]))
    First, the output is continuous: 
    [23.50867458 22.94718558 33.00839918 33.9504757 3.22390622].

    >>> threshold_dict = {'0-10': 33.5}

    >>> thresholder_regression = Thresholder(estimator=regressor,
                                        threshold_dict=threshold_dict,
                                        prefit=True,
                                        predict_method='predict')

    >>> thresholder_regression.fit(X_boston_train, Y_boston_train, sensitive_features=sf_boston_train)
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
same as specifying a threshold as in the previous examples.


.. doctest:: mitigation
    :options: +NORMALIZE_WHITESPACE

    >>> from sklearn.linear_model import LogisticRegression
    >>> from fairlearn.postprocessing import plot_histograms_per_group
    
    >>> classifier_log_reg = LogisticRegression()
    >>> classifier_log_reg.fit(X_train,Y_train)
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
    AfricanAmerican    0.423745
    Asian              0.463415
    Caucasian          0.452529
    Hispanic           0.434783
    Other                0.4375
    Name: false_negative_rate, dtype: object

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
    ...                           predict_method='decision_function',
    ...                           default_threshold=0.0)

    >>> thresholder.fit(X_train,Y_train,sensitive_features=sf_train)
    >>> Y_pred_thresholded = thresholder.predict(X_test,sensitive_features=sf_test)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_thresholded, 
    ...             sensitive_features=sf_test).by_group
    race
    AfricanAmerican    0.423745
    Asian               0.04878
    Caucasian          0.452529
    Hispanic           0.434783
    Other                0.4375
    Name: false_negative_rate, dtype: object

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
women will be specified as ('Asian','Female').

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
    >>> classifier_multiple = RandomForestClassifier(random_state=1)
    >>> classifier_multiple.fit(X_train_multiple_sf,Y_train)
    >>> Y_pred_clf_multiple = classifier_multiple.predict(X_test_multiple_sf)

    >>> #Inspect information about predictions
    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_clf_multiple, 
    ...             sensitive_features=A_multiple_test).by_group
    race             gender
    AfricanAmerican  Female    0.409236
                     Male      0.372549
    Asian            Female    0.461538
                     Male      0.392857
    Caucasian        Female    0.382533
                     Male      0.411967
    Hispanic         Female    0.366667
                     Male      0.381818
    Other            Female    0.483871
                     Male      0.272727
    Name: false_negative_rate, dtype: object

    >>> plot_positive_predictions(A_multiple_test,Y_pred_clf_multiple)   

.. figure:: ../auto_examples/images/user_guide_thresholder_multiple_sf_no_threshold.png
    :align: center

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> threshold_dict = {('Asian','Female'): 0.45}
    >>> thresholder = Thresholder(estimator=classifier_multiple,
    ...                           threshold_dict=threshold_dict,
    ...                           prefit=True,
    ...                           predict_method='predict_proba')

    >>> thresholder.fit(X_train_multiple_sf,Y_train,sensitive_features=A_multiple_train)
    >>> Y_pred_multiple_sf_thresholded = thresholder.predict(X_test_multiple_sf,sensitive_features=A_multiple_test)
  
    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_multiple_sf_thresholded, 
    ...             sensitive_features=A_multiple_test).by_group

    race             gender
    AfricanAmerican  Female    0.409236
                     Male      0.372549
    Asian            Female    0.384615
                     Male      0.392857
    Caucasian        Female    0.382533
                     Male      0.411967
    Hispanic         Female    0.366667
                     Male      0.381818
    Other            Female    0.483871
                     Male      0.272727
    Name: false_negative_rate, dtype: object

    >>> plot_positive_predictions(A_multiple_test,Y_pred_multiple_sf_thresholded)

.. figure:: ../auto_examples/images/user_guide_thresholder_multiple_sf_with_threshold.png
    :align: center

RejectOptionClassifier
~~~~~~~~~~~~~~~~~~~~~~~

What it does 
^^^^^^^^^^^^^^
:class:`RejectOptionClassifier` is a postprocessing technique that is
based on the intuitive hypothesis that wrong decisions are more likely to be made close
to the decision boundary. Translating this hypothesis to a relatable context: a decision you have
made is more likely to be wrong if you are not very certain that this decision is the correct one.
:class:`RejectOptionClassifier` splits all predictions into two groups: predictions made with
sufficient certainty, and predictions made with insuffiecient certainty. The latter category is
also called the **critical region**. Inside the critical region all instances from the deprived
group receive the desirable label, while all instances from the favored group receive the
undesirable label. Because of this split, the sensitive feature is expected to be binary.

| **Their way**
| The size of the critical region is regulated by the parameter :code:`theta` (0.5 < :code:`theta`
  < 1); it contains all instances that satisfy the constraint:
  :math:`\max\{p(Y=1|X), 1 - p(Y=1|X)\} \leq` :code:`theta`.

| **My way**
| The size of the critical region is regulated by the parameter :code:`theta` (0 < :code:`theta`
   < 0.5); it contains all instances that have a probability to get 1 in the range [0.5 - 
   :code:`theta`, 0.5 + :code:`theta`].

Why use it
^^^^^^^^^^
Like :class:`Thresholder`, :class:`RejectOptionClassifier` is an easy to interpret tool, which can
be used to explore different trade-offs, without blindly satisfying a constraint, where
:class:`RejectOptionClassifier` specifically focusses on decisions made with low certainty.

How to use it
^^^^^^^^^^^^^^
Using :class:`RejectOptionClassifier` requires manually specifying the critical region, the
deprived and favored group, and the label considered to be the desired label.

As mentioned above, :class:`RejectOptionClassifier` assumes that the sensitive feature is binary.
Of course, it can happen that there are multiple sensitive features, or that the sensitive feature
is not binary (just like in our previous example). In this case it is easiest to create a new
sensitive feature column which is binary. To see how this, and using
:class:`RejectOptionClassifier` itself, works, consider the same `classifier <https://
scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
and :ref:`data <hospital_readmissions_dataset>` as in the section on
:ref:`Thresholder<default_predictions>`.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> #create binary sensitive feature vector 
    >>> sf_train_reject_clf = sf_train.copy()\
    ...                        .replace(['Asian','Other'], 'deprived')\
    ...                        .replace(['Caucasian','AfricanAmerican','Hispanic'], 'favored')

    >>> sf_test_reject_clf = sf_test.copy()\
    ...                        .replace(['Asian','Other'], 'deprived')\
    ...                        .replace(['Caucasian','AfricanAmerican','Hispanic'], 'favored')

    >>> #Investigate predictions before using RejectOptionClassifier
    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_clf, 
    ...             sensitive_features=sf_test_reject_clf).by_group 
    race
    deprived    0.428571
    favored     0.395255
    Name: false_negative_rate, dtype: object

    >>> plot_positive_predictions(sf_test_reject_clf,Y_pred_clf)

.. figure:: ../auto_examples/images/user_guide_reject_option_clf_binary_output_before.png
    :align: center


.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> #import and use RejectOptionClassifier for predictions
    >>> from fairlearn.postprocessing import RejectOptionClassifier

    >>> reject_clf = RejectOptionClassifier(estimator=classifier,
    ...                                     theta=0.6,
    ...                                     deprived_group='deprived',
    ...                                     favored_group='favored',
    ...                                     desired_label=1,
    ...                                     prefit=True,
    ...                                     predict_method='predict_proba')

    >>> reject_clf.fit(X_train,Y_train,sensitive_features=sf_train_reject_clf)
    >>> Y_pred_reject_clf = reject_clf.predict(X_test,sensitive_features=sf_test_reject_clf)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_reject_clf, 
    ...             sensitive_features=sf_test_reject_clf).by_group
    race
    deprived     0.12381
    favored     0.673723
    Name: false_negative_rate, dtype: object

    >>> plot_positive_predictions(sf_test_reject_clf, Y_pred_reject_clf)

.. figure:: ../auto_examples/images/user_guide_reject_option_clf_binary_output.png
    :align: center

Changing the desired label
*********************************
By default, the desired label is assumed to be 1. It can happen of course, that the situation is
such that it is fact desirable to get label 0. If this is the case, setting the parameter
:code:`desired_label` to 0 is the only thing that needs to be changed in the configuration of
:class:`RejectOptionClassifier`. The critical region itself remains the same, whilst the thing
that changes is the label each instance inside it receives; the deprived group receives the label
0, while the favored group receives 1.

.. doctest:: mitigation
    :options:  +NORMALIZE_WHITESPACE

    >>> reject_clf = RejectOptionClassifier(estimator=classifier,
    ...                                     theta=.6,
    ...                                     deprived_group='deprived',
    ...                                     favored_group='favored',
    ...                                     desired_label=0,
    ...                                     prefit=True,
    ...                                     predict_method='predict_proba')

    >>> reject_clf.fit(X_train,Y_train,sensitive_features=sf_train_reject_clf)
    >>> Y_pred_reject_clf = reject_clf.predict(X_test,sensitive_features=sf_test_reject_clf)

    >>> MetricFrame(metrics=false_negative_rate, 
    ...             y_true=Y_test,
    ...             y_pred=Y_pred_reject_clf, 
    ...             sensitive_features=sf_test_reject_clf).by_group 
    race
    deprived    0.647619
    favored     0.183759
    Name: false_negative_rate, dtype: object

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
classification problems (see [#2]_), and similarly for regression (see [#1]_).
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
    Name: selection_rate, dtype: object
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
    Name: true_positive_rate, dtype: object
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
    Name: accuracy_score, dtype: object
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
    b    0.5
    Name: mean_absolute_error, dtype: object
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

   .. [#4] Kamiran, Karim, Zhang `"Decision Theory for Discrimination-aware Classification"
      <https://ieeexplore.ieee.org/document/6413831>`_,
      IEEE ICDM, 2012