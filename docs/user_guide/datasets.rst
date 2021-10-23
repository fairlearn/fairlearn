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
socio-technical issue. 
This article has the following goals:
  * Educate users about the history of the dataset and its fairness-related harms
  * Show users how fairness-related harms manifest in the data and in downstream modelling tasks
  * Suggest best practices for dealing with the Boston Housing data and other benchmarking datasets with similar issues


.. _boston_dataset_origin:

Dataset Origin and Use
^^^^^^^^^^^^^^^^^^^^^^

Contrary to expectation, the Boston Housing dataset was not developed for economic purposes.
Harrison and Rubenfield (1978)_ 
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
with respect to air pollution levels in their census tracts." 
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
(see :func:`tf.keras.datasets.boston_housing`). 
It has also been the benchmark of choice for many machine learning 
`papers <https://arxiv.org/search/?query=boston+housing&searchtype=all>`_ [#2]_ [#3]_ [#4]_.
In scikit-learn version 1.2, the dataset will be removed.

The dataset contains the following columns:

============ ==========================================================================
Column Name   Description                                                              
============ ==========================================================================
CRIM         per capita crime rate by town                                         
ZN           proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS        proportion of non-retail business acres per town
CHAS         Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX          nitric oxides concentration (parts per 10 million)
RM           average number of rooms per dwelling
AGE          proportion of owner-occupied units built prior to 1940
DIS          weighted distances to five Boston employment centers
RAD          index of accessibility to radial highways
TAX          full-value property-tax rate per $10,000
PTRATIO      pupil-teacher ratio by town
B            1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT        % lower status of the population
MEDV         Median value of owner-occupied homes in $1000’s
============ ============================================================================

The cells below show basic summary statistics about the data, the data types of the 
columns, and the number of missing values.

Note that the :func:`fairlearn.datasets.fetch_boston` function warns user by 
default that the dataset contains fairness issues. 

Setting :code:`warn=False` will turn the warning off. 

To return the dataset as a :class:`pandas.DataFrame`, pass 
:code:`as_frame=True` and call the *data* attribute.


For more information about how to use the :code:`fetch_boston` function, 
visit :mod:`fairlearn.datasets`. 

.. doctest:: datasets

    >>> from fairlearn.datasets import fetch_boston
    >>> import pandas as pd

    >>> X, y = fetch_boston(as_frame=True, return_X_y=True)
    >>> boston_housing=pd.concat([X, y], axis=1)
    >>> boston_housing.head()
      CRIM	    ZN	   INDUS	 CHAS	 NOX	   RM	    AGE	  DIS	    RAD	 TAX	   PTRATIO	 B	      LSTAT	 MEDV
    0	0.00632	 18.0	 2.31	  0	    0.538	 6.575	 65.2	 4.0900	 1	   296.0	 15.3	    396.90	 4.98	  24.0
    1	0.02731	 0.0	  7.07	  0	    0.469	 6.421	 78.9	 4.9671	 2	   242.0	 17.8	    396.90	 9.14	  21.6
    2	0.02729	 0.0	  7.07	  0	    0.469	 7.185	 61.1	 4.9671	 2	   242.0	 17.8	    392.83	 4.03	  34.7
    3	0.03237	 0.0	  2.18	  0	    0.458	 6.998	 45.8	 6.0622	 3	   222.0	 18.7	    394.63	 2.94	  33.4
    4	0.06905	 0.0	  2.18	  0	    0.458	 7.147	 54.2	 6.0622	 3	   222.0	 18.7	    396.90	 5.33	  36.2

    >>> boston_housing.describe()
          CRIM	      ZN	        INDUS	     NOX	       RM	        AGE	       DIS	       TAX	       PTRATIO	   B	         LSTAT	     MEDV
    count	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000
    mean	 3.613524	  11.363636	 11.136779	 0.554695	  6.284634	  68.574901	 3.795043	  408.237154	18.455534	 356.674032	12.653063	 22.532806
    std	  8.601545	  23.322453	 6.860353	  0.115878	  0.702617	  28.148861	 2.105710	  168.537116	2.164946	  91.294864	 7.141062	  9.197104
    min	  0.006320	  0.000000	  0.460000	  0.385000	  3.561000	  2.900000	  1.129600	  187.000000	12.600000	 0.320000	  1.730000	  5.000000
    25%	  0.082045	  0.000000	  5.190000	  0.449000	  5.885500	  45.025000	 2.100175	  279.000000	17.400000	 375.377500	6.950000	  17.025000
    50%	  0.256510	  0.000000	  9.690000	  0.538000	  6.208500	  77.500000	 3.207450	  330.000000	19.050000	 391.440000	11.360000	 21.200000
    75%	  3.677083	  12.500000	 18.100000	 0.624000	  6.623500	  94.075000	 5.188425	  666.000000	20.200000	 396.225000	16.955000	 25.000000
    max	  88.976200	 100.000000	27.740000	 0.871000	  8.780000	  100.000000	12.126500	 711.000000	22.000000	 396.900000	37.970000	 50.000000    

.. _boston_dataset_issues:

Dataset Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the dataset is widely used, it has significant ethical issues.

As explained in :func:`sklearn.datasets.load_boston`, 
Harrison and Rubenfield developed the feature *B* (proportion of population that is Black) 
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

Construct validity provides a useful lens through which to analyze the 
construction of this dataset.
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
For more information on construct validity, refer to :ref:`construct_validity`.

Intersectionality also requires consideration. 
Intersectionality is defined as the interesection between multiple demographic groups. 
The impacts of a technical system on intersectional groups may be different 
than the impacts experienced by the individual demographic groups (e.g., Black
people in aggregate and women in aggregate may experience a technical system 
differently than Black women).

Due to systemic racism present in the data at the time it was collected,
Black people may have been more likely to be categorized as "lower status" by the authors' 
definition.
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
Because both the sensitive and target features are continuous, to leverage 
Fairlearn's assessment capabilities, we need to apply column transformations 
to turn this problem into a classification problem. 
The code below maps *LSTAT*, *B*, and *MEDV* to binary values 
where values greater than the median of the column map to 1, 
and otherwise the values are 0. 

Note that this methodology follows scikit-lego's [#7]_ of the Boston Housing data.

.. doctest:: datasets
    :options:  +NORMALIZE_WHITESPACE

    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.model_selection import train_test_split
    >>> import numpy as np

    >>> X_clf = X.assign(B=lambda d: d['B'] > np.median(d['B']), 
    ... LSTAT=lambda d: d['LSTAT'] > np.median(d['LSTAT']))
    >>> y_clf = y > np.median(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf)

    >>> pipe = Pipeline( [("scale", StandardScaler()), 
    ... ("predict", LogisticRegression())] )
    >>> pipe.fit(X_train, y_train)
    >>> predicted = pipe.predict(X_test)

    >>> import sklearn.metrics as skm
    >>> from fairlearn.metrics import demographic_parity_difference,
    ... MetricFrame,
    ... false_positive_rate,
    ... true_positive_rate,
    ... selection_rate,
    ... count

    >>> DP_B = demographic_parity_difference(y_true = y_test, 
    ... y_pred = predicted, 
    ... sensitive_features = X_test["B"])
    >>> DP_LSTAT = demographic_parity_difference(y_true = y_test, 
    ... y_pred = predicted, 
    ... sensitive_features = X_test["LSTAT"])

    >>> print(f"Demographic parity difference:\nB: {DP_B}\nLSTAT: {DP_LSTAT}")
    Demographic parity difference for:
    B: 0.0901639344262295
    LSTAT: 0.8877297565822156

Checking the demographic parity differences shows that neither variable has a 
demographic parity at zero, implying a different selection rate across groups. 
The next series of tables further breaks down evaluation metrics by
group. 

The proportion of Black people higher than the median is associated with a 
higher false positve rate. 
*B* == True is also associated with a slightly lower precision. 

The accuracy, recall, and selection rate when *LSTAT* is `True` all are lower than when *LSTAT* is `False`. 
These results indicate that our simple model is worse at predicting 
an outcome for individuals in the "lower status" category.

    >>> metrics = {
    ... 'accuracy': skm.accuracy_score,
    ... 'precision': skm.precision_score,
    ... 'recall': skm.recall_score,
    ... 'false positive rate': false_positive_rate,
    ... 'true positive rate': true_positive_rate,
    ... 'selection rate': selection_rate, 
    ... 'count': count}
    >>> grouped_metric = MetricFrame(metrics=metrics,
    ... y_true=y_test, 
    ... y_pred=predicted,
    ... sensitive_features=X_test["B"])
    >>> print(grouped_metric.by_group)
           accuracy     precision       recall   false positive rate   true positive rate    selection rate  count
    B                                                                                    
    False  0.852459          0.92     0.766667              0.064516             0.766667          0.409836     61   
    True   0.863636      0.909091     0.833333                   0.1             0.833333               0.5     66     

    >>> grouped_metric = MetricFrame(metrics=metrics,
    ... y_true=y_test,
    ... y_pred=predicted,
    ... sensitive_features=X_test["LSTAT"])
    >>> print(grouped_metric.by_group)
           accuracy     precision       recall   false positive rate   true positive rate    selection rate  count
    LSTAT                                                                        
    False  0.901639      0.910714     0.980769              0.555556             0.980769          0.918033     61 
    True   0.818182           1.0     0.142857                   0.0             0.142857          0.030303     66 
    

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
Harrison and Rubenfield conducted an empirical econometric study,
the goal of which was to determine the causal impacts of these variables on 
median home value. 
Interpretation of causal models involves looking at model
coefficients to ascertain the effect of one variable on the dependent variable,
holding all other factors constant. This use case is different than the typical 
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
port datasets constructed for one specific use case to different use cases.
Using a dataset without considering the context and purposes for which it 
was created can be risky even if the dataset does not carry the possibility of
generating fairness-related harms. Any machine learning model 
developed using a dataset with an opaque data-generating process runs the 
risk of generating spurious or non-meaningful results. Construct validity is
also relevant here; a dataset may not maintain construct validity across
different types of statistical analyses and different predicted outcomes.

If you are searching for a house pricing dataset to use for benchmarking 
purposes or to create a hedonic pricing model, scikit-learn recommends the 
California housing dataset (:func:`sklearn.datasets.fetch_california_housing`)
or the Ames housing dataset [#6]_ 
in place of the Boston housing dataset, as using these datasets should not
cause the same fairness-related harms. 
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
         
