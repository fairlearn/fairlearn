Datasets
==========

Revisiting the Boston Housing Dataset
-------------------------------------

Introduction
^^^^^^^^^^^^^^^^^

The Boston Housing dataset is one of the datasets currently callable in 
:mod:`fairlearn.datasets` module. In the past, it has commonly been used for 
benchmarking in popular machine learning libraries, including scikit-learn and 
open-ml. However, as the machine learning community has developed awareness 
about fairness issues with popular benchmarking datasets, the Boston Housing 
data has been phased out of many libraries. We migrated the dataset to  
fairlearn after it was phased out of scikit-learn in June 2020. The dataset  
remains in fairlearn as an example of how system racism can occur in data  
and to show the effect of fairlearn's assessment and pre-/in-/post-processing  
tools on real, problematic data. 

**We wrote this blog post to achieve the following goals:**
  * Educate users about the history of the dataset and its bias
  * Show users how bias manifests in the data and in downstream modelling tasks
  * Suggest best practices for dealing with the Boston Housing data and 
  alternative benchmarking datasets

.. _dataset_origin:

Dataset Origin and Use
^^^^^^^^^^^^^^^^^^^^^^

Contrary to expectation, the Boston Housing dataset was not developed for 
economic purposes. `Harrison and Rubenfield (1978) <https://deepblue.lib.umich.edu/bitstream/handle/2027.42/22636/0000186.pdf?sequence=1&isAllowed=y>` 
developed the dataset to illustrate the issues with using housing market data 
to measure consumer willingness to pay for clean air. The authors use a 
`hedonic pricing <https://www.investopedia.com/terms/h/hedonicpricing.asp>` 
approach, which assumes that the price of a good or service can be modeled as a 
function of features both internal and external to the good or service. The 
input to this model was a dataset comprising the Boston Standard Metropolitan 
Statistical Area, with the *NOX* variable serving as a proxy for air quality. 
The paper sought to estimate the median value of owner-occupied homes (now 
*MEDV*), and included the remaining variables to measure other neighborhood 
characteristics. Further, the authors took the derivative of their housing 
value equation with respect to *NOX* to measure the "amount of money households
were willing to pay with respect to air polution levels in their census 
tracts". The variables in the dataset are representative of the early 1970s 
and come from a mixture of surveys, administrative records, and other research
papers. While the paper does define each variable and suggest its impact on 
the housing value equation, it lacks reasoning for including that particular
set of variables.

Modern machine learning practicioners have used the Boston Housing dataset as 
a benchmark to assess the performance of emerging supervised learning 
techniques. It's featured in `scipy lectures <https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_boston_prediction.html>`, 
indexed in the `University of California-Irvine Machine Learning Repository 
<https://archive.ics.uci.edu/ml/machine-learning-databases/housing/>` and in 
Carnegie Mellon University's `StatLib <http://lib.stat.cmu.edu/datasets/boston>`, 
and for a time was included as one of `scikit-learn's <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html>`
 and tensorflow's standard toy datasets. It has also been the benchmark of 
choice for `many <https://ieeexplore.ieee.org/abstract/document/8556738/>` 
`machine <https://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=1187&context=imse_conf>` 
`learning <https://proceedings.neurips.cc/paper/1999/file/f3144cefe89a60d6a1afaf7859c5076b-Paper.pdf>`
`papers <https://arxiv.org/search/?query=boston+housing&searchtype=all&source=header>`.
As of scikit-learn version 1.2, the dataset has been removed.

The dataset contains the following columns:

============ ==================
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

The cells below show basic summary statistics about the data, the dtypes of the 
columns, and the number of missing values. Note that the 
:func:`fairlearn.datasets.fetch_boston` function warns user by default that the
dataset contains fairness issues. Setting :code:`warn=False` will turn the
warning off. To return the dataset as a data frame, pass 
:code:`as_frame=True` and call the *data* attribute.

For more information about how to use the :code:`fetch_boston` function, 
visit the `fairlearn datasets documentation 
<https://fairlearn.org/v0.7.0/api_reference/fairlearn.datasets.html#id15>`. 

.. note::

    Calling the :func:`fairlearn.datasets.fetch_boston` function will raise a 
    :code:`FairnessWarning`.
    For more information on this warning refer to
    `https://fairlearn.org/v0.7.0/api_reference/fairlearn.datasets.html 
    <https://fairlearn.org/v0.7.0/api_reference/fairlearn.datasets.html>`_.

.. doctest:: datasets

    >>> from fairlearn.datasets import fetch_boston
    >>> import pandas as pd

    >>> X, y = fetch_boston(as_frame = True, return_X_y= True)
    >>> boston_housing = pd.concat([X, y], axis = 1)
    >>> boston_housing.head()
        CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT	MEDV
    0	0.00632	18.0	2.31	0	0.538	6.575	65.2	4.0900	1	296.0	15.3	396.90	4.98	24.0
    1	0.02731	0.0	7.07	0	0.469	6.421	78.9	4.9671	2	242.0	17.8	396.90	9.14	21.6
    2	0.02729	0.0	7.07	0	0.469	7.185	61.1	4.9671	2	242.0	17.8	392.83	4.03	34.7
    3	0.03237	0.0	2.18	0	0.458	6.998	45.8	6.0622	3	222.0	18.7	394.63	2.94	33.4
    4	0.06905	0.0	2.18	0	0.458	7.147	54.2	6.0622	3	222.0	18.7	396.90	5.33	36.2

    >>> boston_housing.describe()
        CRIM	ZN	INDUS	NOX	RM	AGE	DIS	TAX	PTRATIO	B	LSTAT	MEDV
    count	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000	506.000000
    mean	3.613524	11.363636	11.136779	0.554695	6.284634	68.574901	3.795043	408.237154	18.455534	356.674032	12.653063	22.532806
    std	8.601545	23.322453	6.860353	0.115878	0.702617	28.148861	2.105710	168.537116	2.164946	91.294864	7.141062	9.197104
    min	0.006320	0.000000	0.460000	0.385000	3.561000	2.900000	1.129600	187.000000	12.600000	0.320000	1.730000	5.000000
    25%	0.082045	0.000000	5.190000	0.449000	5.885500	45.025000	2.100175	279.000000	17.400000	375.377500	6.950000	17.025000
    50%	0.256510	0.000000	9.690000	0.538000	6.208500	77.500000	3.207450	330.000000	19.050000	391.440000	11.360000	21.200000
    75%	3.677083	12.500000	18.100000	0.624000	6.623500	94.075000	5.188425	666.000000	20.200000	396.225000	16.955000	25.000000
    max	88.976200	100.000000	27.740000	0.871000	8.780000	100.000000	12.126500	711.000000	22.000000	396.900000	37.970000	50.000000    

.. _dataset_issues:

Dataset Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the dataset is widely used, it has significant ethical issues. As 
explained in scikit-learn's `documentation <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html#rec2f484fdebe-2>`, 
Harrison and Rubenfield developed the *B* under the assumption that racial 
self-segregation had a positive impact on house prices. *B* then is a measure 
of systemic racism, as it encodes racism as a factor in house pricing. Thus, 
any models trained using this data that do not take special care to process *B* 
will learn to use mathematically encoded racism as a factor in house price
 prediction.

Harrison and Rubenfield describe their projected impact of the problematic 
variables as follows. Both variables come from the 1970 US Census. 
- *LSTAT*: "Proportion of population that is lower status = 0.5 * 
(**proportion of adults without some high school education and proportion of 
male workers classified as laborers**). The logarithmic specification implies 
that socioeconomic status distinctions mean more in the upper brackets of 
society than in the lower classes."
- *B*: "Black proportion of population. At low to moderate levels of B, 
an **increase in B** should have a **negative influence on housing value** 
if Blacks are regarded as undesirable neighbors by Whites. However, market 
discrimination means that housing values are higher at very high levels of B. 
One expects, therefore, a parabolic relationship between proportion Black in 
a neighborhood and housing values.

To break down the *B* reasoning further, the authors assume that 
self-segregation correlates to higher home values, though subsequent authors 
contend that this hypothesis is impossible to prove with evidence (see `Kain 
and Quigley, 1975 <https://www.nber.org/books/kain75-1>`). Additionally, though
the authors specify a parabolic transformation for *B*, they do not provide 
evidence that the relationship between *B* and *MEDV* is parabolic. Harrison 
and Rubenfield set a threshold of 63% as the point in which median house 
prices flip from declining to increasing, but do not provide the basis for 
this threshold. An `analysis of the dataset 
<https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8>` by 
M. Carlisle further shows that the Boston Housing dataset suffers from serious
quality and incompleteness issues, as Carlisle was unable to recover the 
original Census data mapping for all the points in the *B* variable. 

The definition of the *LSTAT* variable is also suspect. Harrison and 
Rubenfield define lower status as a function of the proportion of adults 
without some high school education and the proportion of male workers 
classified as laborers. However, the categorization of a certain level of 
education and job category as indicative of "lower status" is reflective of
social constructs of class and not objective fact. 
Again, the authors provide no evidence of a proposed relationship between
*LSTAT* and *MEDV* and do not sufficiently justify its inclusion 
in the hedonic pricing model.

The inclusion of these columns might make sense for an econometric analysis, 
which seeks to understand the causal impact of various factors on a dependent 
variable, but these columns are problematic in the context of a predictive
analysis. Predictive models will learn the patterns of systemic bias 
encoded in the data and will reproduce that bias in their predictions.
The next section describes the potential risk in using this dataset in a 
typical machine learning prediction pipeline.


.. _bias_assessment:

Bias Assessment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As explained above, machine learning models that use the Boston Housing dataset 
are at risk of replicating the systemic bias encoded in the variables. 
How does that look in a typical machine learning pipeline? 
Because both the sensitive and target feaures are continuous, to leverage 
fairlearn's assessment capabilities, we need to apply column transformations 
to turn this problem into a classification problem. The code below maps 
*LSTAT*, *B*, and *MEDV* to binary values where values greater than the median 
of the column map to 1, and otherwise the values are 0. Note that this 
methodology follows scikit-lego's `exploration 
<https://scikit-lego.netlify.app/fairness.html>` of the Boston Housing data.

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
The next series of tables dives deeper into the breakdown of various metrics by
group. The proportion of Blacks higher than the median is associated with a 
higher false positve rate. *B* == True is also associated with a slightly 
lower precision. The accuracy, recall, and selection rate when *LSTAT* == 
True all are lower than when *LSTAT* is False. These results indicate that 
our simple model is worse at predicting an outcome for individuals in our 
constructed "lower status" category.

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
        accuracy precision    recall false positive rate true positive rate  \
    B                                                                            
    False  0.852459      0.92  0.766667            0.064516           0.766667   
    True   0.863636  0.909091  0.833333                 0.1           0.833333   

        selection rate count  
    B                           
    False       0.409836    61  
    True             0.5    66  

    >>> grouped_metric = MetricFrame(metrics=metrics,
    ... y_true=y_test,
    ... y_pred=predicted,
    ... sensitive_features=X_test["LSTAT"])
    >>> print(grouped_metric.by_group)
        accuracy precision    recall false positive rate true positive rate  \
    LSTAT                                                                        
    False  0.901639  0.910714  0.980769            0.555556           0.980769   
    True   0.818182       1.0  0.142857                 0.0           0.142857   

        selection rate count  
    LSTAT                       
    False       0.918033    61  
    True        0.030303    66  


.. _discussion:

Discussion
^^^^^^^^^^^^^^^^^^^^^^^^

The Boston housing dataset is rife with ethical issues, and in general, we 
strongly discourage using it in predictive modelling analyses. We've kept it 
in fairlearn because of its potential as a teaching tool for how to deal with 
ethical issues in a dataset. There are ways to `remove correlations between 
sensitive features and the remaining columns 
<https://scikit-lego.netlify.app/fairness.html>`, but other benchmark datasets
exist that do not present these issues.

It's important to keep the differences between the way Harrison and Rubenfield 
used the dataset and the way modern machine learning practicioners have used 
it in focus. Harrison and Rubenfield conducted an empirical econometric study,
the goal of which was to determine the causal impacts of these variables on 
median home value. Interpretation of causal models involves looking at model
coefficients to ascertain the effect of one variable on the depedent variable,
holding all other factors constant. This use case is different than the typical 
supervised learning analysis. A machine learning model will pick up on the 
patterns encoded in the data and use that to predict an outcome.
In the Boston housing dataset, the patterns the authors encoded through
the *B* and *LSTAT* variables include systemic racism and class inequalities, 
respectively. A predictive model will learn to use those biases to make
a prediction. Using the Boston housing dataset as a benchmark for a new 
supervised learning model means that the model's performance is in part due to
how well it learns and replicates the biases in this dataset.

If you are searching for a house pricing dataset to use for benchmarking 
purposes or to create a hedonic pricing model, scikit-learn recommends the 
`California housing dataset <https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset>` 
or the `Ames dataset <https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_ames_housing.html>` 
in place of the Boston housing dataset, as these datasets do not suffer from 
bias issues. We recommend you proceed with extreme caution when calling the 
Boston housing data from fairlearn, and hope this article gives you pause 
about using it in the future.
