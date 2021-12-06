Datasets
==========

Revisiting the Boston Housing Dataset
-------------------------------------

Introduction
^^^^^^^^^^^^^^^^^

The Boston Housing dataset is one of the datasets currently callable in :mod:`fairlearn.datasets` module.
In the past, it has commonly been used for benchmarking in popular machine learning libraries, 
including `scikit-learn <https://scikit-learn.org/>`_ and `OpenML <https://www.openml.org/>`_. 
However, as the machine learning community has developed awareness about fairness issues with 
popular benchmarking datasets, the Boston Housing data has been phased out of many libraries. 
We migrated the dataset to Fairlearn after it was phased out of scikit-learn in June 2020. 
The dataset remains in Fairlearn as an example of how systemic racism can occur in data and to 
show the effect of Fairlearn's unfairness assessment and mitigation tools on real, problematic data. 

We also think this dataset provides an interesting case study of how fairness is fundamentally a
socio-technical issue by exploring how societal biases manifest in data in ways that can't
simply be fixed with technical mitigation approaches (although the harms they engender may be mitigated).
This article has the following goals:

* Educate users about the history of the dataset and how the variables were constructed
* Show users how socioeconomic inequities are reflected in the data in ways that
  can potentially lead to fairness-related harms in downstream modelling tasks
* Suggest alternative benchmarking datasets


.. _boston_dataset_origin:

Dataset Origin and Use
^^^^^^^^^^^^^^^^^^^^^^

Harrison and Rubenfield [#1]_ 
developed the dataset to illustrate the issues with using housing market data 
to measure consumer willingness to pay for clean air. 
The authors use a hedonic pricing [#9]_ 
approach, which assumes that the price of a good or service can be modeled as a 
function of features both internal and external to the good or service. 
The input to this model was a dataset comprising the Boston Standard Metropolitan 
Statistical Area [#10]_, with the nitric oxides concentration (*NOX*) 
serving as a proxy for air quality.

The paper sought to estimate the median value of owner-occupied homes (now 
*MEDV*), and included the remaining variables to capture other neighborhood 
characteristics.
Further, the authors took the derivative of their housing 
value equation with respect to nitric oxides concentration 
to measure the "amount of money households were willing to pay  
when purchasing a home with respect to air pollution levels in their census tracts." 
The variables in the dataset were collected in the early 1970s 
and come from a mixture of surveys, administrative records, and other research
papers. 
While the paper does define each variable and suggest its impact on 
the housing value equation, it lacks reasoning for including that particular
set of variables.

Modern machine learning practitioners have used the Boston Housing dataset as 
a benchmark to assess the performance of emerging supervised learning 
techniques. 
It's featured in `scipy lectures <https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_boston_prediction.html>`_, 
indexed in the `University of California-Irvine Machine Learning Repository 
<https://archive.ics.uci.edu/ml/machine-learning-databases/housing/>`_ and in 
Carnegie Mellon University's `StatLib <http://lib.stat.cmu.edu/datasets/boston>`_, 
and for a time was included as one of scikit-learn's and tensorflow's standard toy datasets
(see :mod:`tf.keras.datasets.boston_housing`).
It has also been the benchmark of choice for many machine learning 
`papers <https://arxiv.org/search/?query=boston+housing&searchtype=all>`_ [#2]_ [#3]_ [#4]_.
In 2020, users brought the dataset's fairness issues to the scikit-learn development team 
(see scikit-learn issue `#16155 <https://github.com/scikit-learn/scikit-learn/issues/16155>`_), after which the team decided to remove the dataset in scikit-learn version 1.2.
In scikit-learn version 1.2, the dataset will be removed.

The dataset contains the following columns:

.. list-table::
   :header-rows: 1
   :widths: 7 30
   :stub-columns: 1

   *  - Column name
      - Description
   *  - CRIM
      - per capita crime rate by town
   *  - ZN
      - proportion of residential land zoned for lots over 25,000 sq.ft.
   *  - INDUS
      - proportion of non-retail business acres per town
   *  - CHAS
      - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
   *  - NOX
      - nitric oxides concentration (parts per 10 million)
   *  - RM
      - average number of rooms per dwelling
   *  - AGE
      - proportion of owner-occupied units built prior to 1940
   *  - DIS
      - weighted distances to five Boston employment centers
   *  - RAD
      - index of accessibility to radial highways
   *  - TAX
      - full-value property-tax rate per $10,000
   *  - PTRATIO
      - pupil-teacher ratio by town
   *  - B
      - 1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town
   *  - LSTAT
      - % lower status of the population
   *  - MEDV
      - Median value of owner-occupied homes in $1000’s

The cells below show basic summary statistics about the data, the data types of the 
columns, and the number of missing values.

Note that the :func:`fairlearn.datasets.fetch_boston` function warns users by 
default that the dataset contains fairness issues. 

Setting :code:`warn=False` will turn the warning off. 

To return the dataset as a :class:`pandas.DataFrame`, pass 
:code:`as_frame=True` and call the *data* attribute.


For more information about how to use the :code:`fetch_boston` function, 
visit :mod:`fairlearn.datasets`. 

.. doctest:: datasets
    :options:  +NORMALIZE_WHITESPACE

    >>> import warnings
    >>> warnings.filterwarnings('ignore')
    >>> from fairlearn.datasets import fetch_boston
    >>> import pandas as pd
    >>> pd.set_option('display.max_columns', 20)
    >>> pd.set_option('display.width', 80)
    >>> X, y = fetch_boston(as_frame=True, return_X_y=True)
    >>> boston_housing=pd.concat([X, y], axis=1)
    >>> with pd.option_context('expand_frame_repr', False):
    ...    boston_housing.head()
          CRIM    ZN  INDUS CHAS    NOX     RM   AGE     DIS RAD    TAX  PTRATIO       B  LSTAT  MEDV
    0  0.00632  18.0   2.31    0  0.538  6.575  65.2  4.0900   1  296.0     15.3   396.90   4.98  24.0
    1  0.02731   0.0   7.07    0  0.469  6.421  78.9  4.9671   2  242.0     17.8   396.90   9.14  21.6
    2  0.02729   0.0   7.07    0  0.469  7.185  61.1  4.9671   2  242.0     17.8   392.83   4.03  34.7
    3  0.03237   0.0   2.18    0  0.458  6.998  45.8  6.0622   3  222.0     18.7   394.63   2.94  33.4
    4  0.06905   0.0   2.18    0  0.458  7.147  54.2  6.0622   3  222.0     18.7   396.90   5.33  36.2

.. _boston_dataset_issues:

Dataset Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the dataset is widely used, it has significant ethical issues.

As explained in :func:`sklearn.datasets.load_boston`, 
Harrison and Rubenfield developed the feature *B* (result of the formula *1000(B_k - 0.63)^2k*) 
under the assumption that racial self-segregation had a positive impact on house prices. 
*B* then encodes systemic racism as a factor in house pricing. 
Thus, any models trained using this data that do not take special care to process *B* 
will learn to use mathematically encoded racism as a factor in house price prediction. 

Harrison and Rubenfield describe their projected impact of the *B* and *LSTAT* 
variables as follows (note that these descriptions 
are verbatim from their paper). However, many of the authors' assumptions 
have later been found to be unsubstantiated.

* *LSTAT*: "Proportion of population that is lower status = 0.5 * 
  (proportion of adults without some high school education and proportion of
  male workers classified as laborers). The logarithmic specification implies
  that socioeconomic status distinctions mean more in the upper brackets of
  society than in the lower classes."

* *B*: "Proportion of population that is Black. At low to moderate levels of B, 
  an increase in B should have a negative influence on housing value
  if Black people are regarded as undesirable neighbors by White people. However, market
  discrimination means that housing values are higher at very high levels of B.
  One expects, therefore, a parabolic relationship between proportion Black in
  a neighborhood and housing values."

To describe the reasoning behind *B* further, the authors assume that 
self-segregation correlates to higher home values. However, other 
researchers (see [#5]_) did not find evidence that supports this hypothesis. 

Additionally, though the authors specify a parabolic transformation 
for *B*, they do not provide evidence that the relationship between *B* and *MEDV* 
is parabolic. 
Harrison and Rubenfield set a threshold of 63% as the point in which median house 
prices flip from declining to increasing, but do not provide the basis for 
this threshold. 
An analysis of the dataset [#8]_ 
by M. Carlisle further shows that the Boston Housing dataset suffers from serious
quality and incompleteness issues, as Carlisle was unable to recover the 
original Census data mapping for all the points in the *B* variable. 


The definition of the *LSTAT* variable is also suspect. 
Harrison and Rubenfield define lower status as a function of the proportion
of adults without some high school education and the proportion of male workers 
classified as laborers. They apply a logarithmic transformation to the variable 
with the assumption that resulting variable distribution reflects their understanding of
socioeconomic distinctions.
However, the categorization of a certain level of 
education and job category as indicative of "lower status" is reflective of
social constructs of class and not objective fact.
Again, the authors provide no evidence of a proposed relationship between
*LSTAT* and *MEDV* and do not sufficiently justify its inclusion 
in the hedonic pricing model.

Construct validity (:ref:`construct_validity`) provides a useful lens through 
which to analyze the construction of this dataset.
Construct validity refers to the extent to which a given measurement model
measures the intended construct in way that is meaningful and useful. 
In Harrison and Rubenfield's analysis, the measurement model involves 
constructing the assumed point at which prejudice against Black people occurs 
and the effect that prejudice has on house values. 
Likewise, another measurement model also constructs membership in
lower-status classes based on educational attainment
and labor category. 
It is useful to ask whether the way the authors chose to create 
the measurements accurately represents phenomenon they 
sought to measure. 
As is discussed above, the authors do not provide justification for their 
variable construction choices beyond the projected impacts described 
in the variable definitions.
Both measurements fail the test of content validity, a subcategory of
construct validity, as the variable definitions are subjective and thus
open to being contested.
The authors also do not establish convergent validity, another subcategory 
of construct validity, in that they do not show their measurements correlate
with measurements from measurement models in which construct validity has 
been established. 
However, given the time period in which the paper 
was published there may have been a dearth of related measurement models.

Intersectionality also requires consideration. 
Intersectionality is defined as the interesection between multiple demographic groups.[#11]_ 
The impacts of a technical system on intersectional groups may be different 
than the impacts experienced by the individual demographic groups (e.g., Black
people in aggregate and women in aggregate may experience a technical system 
differently than Black women).

Due to the effects of discriminatory socioeconomic policies, 
including housing policies, in effect at the time the article was written, 
Black people may have been more likely to be categorized as "lower status" 
by the authors' definition.
Harrison and Rubenfield do not consider this intersectionality in their analysis.
When using a linear model,
intersectionality could be captured via an interaction variable, which combines 
the two fields. 
In the machine learning context, considering each group separately (i.e., 
considering impacts on *B* and *LSTAT* separately) may obscure harms. 
Additionally, including only one of these variables in the analysis is not
sufficient in removing the signals encoded in the removed variable from the dataset.
Because these columns are related, one likely can serve as a proxy for the other.
Thus, we recommend great care be taken to account for intersectionality in data.

The inclusion of these columns might make sense for an econometric analysis, 
which seeks to understand the causal impact of various factors on a dependent 
variable, but these columns are problematic in the context of a predictive
analysis. 
Predictive models will learn the patterns of systemic racism and classism 
encoded in the data and will reproduce those patterns in their predictions.
It's also important to note that merely excluding these variables from the dataset
is not sufficient to mitigate these issues.
However, through careful assessment, the negative effects of these variables
can be mitigated.

The next section describes the potential risk in using this dataset in a 
typical machine learning prediction pipeline.


.. _boston_harms_assessment:

Fairness-related harms assessment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As explained above, machine learning models that use the Boston Housing dataset 
are at risk of generating fairness-related harms. 
How does that look in a typical machine learning pipeline? 

We apply a column transformation to the target feature 
to turn this problem into a classification problem.
The code below maps *LSTAT* and *MEDV* to binary values where values greater 
than the median of the column map to TRUE, and otherwise the values are FALSE. 
This methodology follows scikit-lego's [#7]_ exploration
of the Boston Housing data. We also transform *B* into a binary variable where 
TRUE values are above the value 136.9. Observations below this point correspond to 
the "true" proportion of Black people above 63%, at which point the authors
assumed that house prices would begin to be affected by the racism of 
others in the community.

.. doctest:: datasets
    :options:  +NORMALIZE_WHITESPACE

    >>> import sklearn.metrics as skm
    >>> import fairlearn.metrics as fm
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.model_selection import train_test_split
    >>> import numpy as np
    >>> X_clf = X.assign(B=lambda d: d['B'] > 136.9, 
    ...                  LSTAT=lambda d: d['LSTAT'] > np.median(d['LSTAT']))
    >>> y_clf = y > np.median(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf)
    >>> model = LogisticRegression(random_state=123, solver = 'liblinear')
    >>> model.fit(X_train, y_train)
    LogisticRegression(random_state=123, solver='liblinear')
    >>> predicted = model.predict(X_test)
    >>> DP_B = fm.demographic_parity_difference(y_true = y_test, 
    ...                                      y_pred = predicted, 
    ...                                      sensitive_features = X_test["B"])
    >>> DP_LSTAT = fm.demographic_parity_difference(y_true = y_test, 
    ...                                          y_pred = predicted, 
    ...                                          sensitive_features = X_test["LSTAT"])
    >>> print(f"Demographic parity difference:\nB: {DP_B}\nLSTAT: {DP_LSTAT}")  # doctest: +SKIP
    Demographic parity difference for:
    B: 0.5470085470085471
    LSTAT: 0.8583829365079365
    >>> metrics = {'accuracy': skm.accuracy_score,
    ...            'precision': skm.precision_score,
    ...            'recall': skm.recall_score,
    ...            'false positive rate': fm.false_positive_rate,
    ...            'true positive rate': fm.true_positive_rate,
    ...            'selection rate': fm.selection_rate, 
    ...            'count': fm.count}
    >>> grouped_metric = fm.MetricFrame(metrics=metrics,
    ...                                 y_true=y_test, 
    ...                                 y_pred=predicted,
    ...                                 sensitive_features=X_test["B"])
    >>> print(grouped_metric.by_group)  # doctest: +SKIP
        accuracy     precision       recall   false positive rate   true positive rate    selection rate  count
    B                                                                                                             
    False       1.0           0.0          0.0                   0.0                  0.0               0.0     10
    True   0.837607       0.84375     0.857143              0.185185             0.857143          0.547009    117
    <BLANKLINE>
    [2 rows x 7 columns]
    >>> grouped_metric = fm.MetricFrame(metrics=metrics,
    ...                                 y_true=y_test,
    ...                                 y_pred=predicted,
    ...                                 sensitive_features=X_test["LSTAT"])
    >>> print(grouped_metric.by_group)  # doctest: +SKIP
        accuracy     precision       recall   false positive rate   true positive rate    selection rate  count
    LSTAT                                                                                                         
    False   0.84127      0.864407     0.962264                   0.8             0.962264          0.936508     63
    True   0.859375           0.6          0.3              0.037037                  0.3          0.078125     64    
    <BLANKLINE>
    [2 rows x 7 columns]

The demographic parity differences shows that neither variable has a 
demographic parity at zero, which implies different 
selection rates across groups. 
The vast majority of observations of *B* fall above the cutoff.
For the *B* variable, observations below the cutoff have zero precision 
and recall, but the model has a higher accuracy for 
this group than records where *B* > 136.9.
The precision, recall, and selection rate when *LSTAT* is `True` all are 
lower than when *LSTAT* is `False`. 
These results indicate that our simple model is worse at predicting 
an outcome for individuals in the "lower status" category.
    

.. _discussion:

Discussion
^^^^^^^^^^^^^^^^^^^^^^^^

The Boston housing dataset presents many ethical issues, and in general, we 
strongly discourage using it in predictive modelling analyses. 
We've kept it in Fairlearn because of its potential as a teaching tool 
for how to deal with ethical issues in a dataset. 
There are ways to remove correlations between sensitive features and the remaining columns [#7]_, 
but that is by no means a guarantee that fairness-related harms won't occur. 
Besides, other benchmark datasets exist that do not present these issues.


It's important to keep the differences between the way Harrison and Rubenfield 
used the dataset and the way modern machine learning practicioners have used 
it in focus. 
Harrison and Rubenfield conducted an empirical study,
the goal of which was to determine the causal impacts of these variables on 
median home value. 
Interpretation of causal models involves looking at model
coefficients to ascertain the effect of one variable on the dependent variable,
holding all other factors constant. 
This use case is different than the typical 
supervised learning analysis. 
A machine learning model will pick up on the 
patterns encoded in the data and use that to predict an outcome.
In the Boston housing dataset, the patterns the authors encoded through
the *B* and *LSTAT* variables include systemic racism and class inequalities, 
respectively. 
Using the Boston housing dataset as a benchmark for a new 
supervised learning model means that the model's performance is in part due to
how well it learns and replicates the patterns in this dataset.

The Boston Housing dataset raises the more general issue of whether it's valid to 
port datasets constructed for one specific use case to different use cases (see :ref:`portability_trap`).
Using a dataset without considering the context and purposes for which it 
was created can be risky even if the dataset does not carry the possibility of
generating fairness-related harms. 
Any machine learning model 
developed using a dataset with an opaque data-generating process runs the 
risk of generating spurious or non-meaningful results. 
Construct validity is also relevant here; 
a dataset may not maintain construct validity across
different types of statistical analyses and different predicted outcomes.

If you are searching for a house pricing dataset to use for benchmarking 
purposes or to create a hedonic pricing model, scikit-learn recommends the 
California housing dataset (:func:`sklearn.datasets.fetch_california_housing`)
or the Ames housing dataset [#6]_ 
in place of the Boston housing dataset, as using these datasets should not
generate the same fairness-related harms. 
We strongly discourage using the Boston Housing dataset for machine learning 
benchmarking purposes, and hope this article gives 
you pause about using it in the future.

.. topic:: References:

  .. [#1] David Harrison, Daniel Rubenfield, `"Hedonic Housing Prices and the Demand for Clean Air" <https://deepblue.lib.umich.edu/bitstream/handle/2027.42/22636/0000186.pdf?sequence=1&isAllowed=y>`_,
      Journal of Environmental Economics and Management, 1978.
      

  .. [#2] Ali Al Bataineh, Devinder Kaur, `"A Comparative Study of Different Curve Fitting Algorithms in Artificial Neural Network using Housing Dataset" <https://ieeexplore.ieee.org/abstract/document/8556738>`_,
      IEEE, 2018.
 

  .. [#3] Mohsen Shahhosseini, Guiping Hu, Hieu Pham, `"Optimizing Ensemble Weights for Machine Learning Models: A Case Study for Housing Price Prediction" <https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=1187&context=imse_conf>`_,
      Industrial and Manufacturing Systems Engineering Conference Proceedings and Posters, 2019.   


  .. [#4] Michael E. Tipping , `"The Relevance Vector Machine" <https://proceedings.neurips.cc/paper/1999/file/f3144cefe89a60d6a1afaf7859c5076b-Paper.pdf>`_,
      1999.
  
  .. [#5] John F. Kain, John M. Quigley, `"Housing Markets and Racial Discrimination: A Microeconomic Analysis" <https://www.nber.org/books/kain75-1>`_, 
      National Bureau of Economic Research (NBER), 1975.

  .. [#6] Scikit-Learn, `"The Ames housing dataset" <https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_ames_housing.html>_`,
      2021.
   
  .. [#7] Scikit-Lego, `"Fairness" <https://scikit-lego.netlify.app/fairness.html>`_,
      2019.
   
  .. [#8] M Carlisle, `"racist data destruction?" <https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8>`_,
      Medium, 2019.

  .. [#9] Marshall Hargrave, `"Hedonic Pricing" <https://www.investopedia.com/terms/h/hedonicpricing.asp>`_,
      Investopedia, 2021.
  
  .. [#10] `"Metropolitan Areas", <https://www.census.gov/history/www/programs/geography/metropolitan_areas.html>`_,
        United States Census Bureau.
  
  .. [#11] Kinmberlé Crenshaw, Mapping the margins: Intersectionality, identity politics, and violence against women of color, 
      Stanford Law Review, 1993, 43(6), 1241-1299.
