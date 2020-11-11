# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
================================
Predictive Policing Case Study
================================
"""
# %%
#
#
#Contents
#========
#
#1. Overview
#2. Background
#3. Fairlearn Results
#4. Problem area & Sociotechnical context
#5. Conclusion
#
#Overview
#--------
#
#In this document we outline a sample sociotechnical case study - using a
#hypothetical predictive policing scenario - based on data found in the
#`Boston housing dataset <https://www.kaggle.com/c/boston-housing>`_
#which we use in our scenario to predict per capita crime rate.
#Discussion around the known fairness issues associated with this dataset
#is included `here
#<https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8>`_.
#For further questions about this document please contact Bruke Kifle
#(@brkifle) or Michael Amoako (@miamoako). The full notebook containing
#the code below is available upon request.
#
#Background
#----------
#
#You are a contractor working with the Boston Police Department and
#advising them in their effort to add predictive analytics and
#intelligence to their existing policing practices. One idea the Boston
#Police Department has is to adopt an algorithmic decision-making tool
#that would allow them to determine how heavily different regions in
#Boston should be policed, based on the model’s predicted crime rate
#classification. Such practice is known as ‘predictive policing’ – the
#use of analytical techniques in law enforcement to predict ‘potential’
#criminal activity and take proactive measures to pre-empt crime. In this
#example, assume that if the model’s output is above the defined
#threshold, the described area is more heavily policed. Some key
#characteristics of the scenario:
#
#
#* Stakeholders: police officers, members of the community (especially
#  people who may be more affected by predictive policing than others),
#  contractor / data scientist
#
#* Harms: overpolicing of neighborhoods can
#  lead to disproportionate effect on communities in these neighborhoods
#  (perhaps exacerbated by feedback loop)
#
#* Notes: Feedback loops!
#  Prediction on behavior based on circumstances; perhaps useful for
#  aggregate observations about behavior but less so for individual
#  predictions
#
#* Dangerous to assume that ‘crime’ is a single phenomenon.
#  There are different kinds of crime, so also need to make sure that data
#  match the crime being predicted.
#
#For this case study, we are utilizing the Boston housing value dataset
#(described here). Originally, this dataset includes the following variables:
#
#.. list-table::
#   :header-rows: 1
#   :widths: 5 20
#   :stub-columns: 1
#
#   *  - Variable
#      - Description
#   *  - CRIM (BLUE)
#      - per capita crime rate by town
#   *  - ZN
#      - proportion of residential land zoned for lots over 25,000 sq.ft.
#   *  - INDUS
#      - proportion of non-retail business acres per town
#   *  - CHAS
#      - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#   *  - NOX
#      - nitric oxides concentration (parts per 10 million)
#   *  - RM
#      - average number of rooms per dwelling
#   *  - AGE
#      - proportion of owner-occupied units built prior to 1940
#   *  - DIS
#      - weighted distances to five Boston employment centres
#   *  - RAD	
#      - index of accessibility to radial highways
#   *  - TAX	
#      - full-value property-tax rate per $10,000
#   *  - PTRATIO
#      - pupil-teacher ratio by town
#   *  - B
#      - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#   *  - LSTAT (BLUE)
#      - % lower status of the population
#   *  - MEDV (GREEN)
#      - Median value of owner-occupied homes in $1000's
#*Figure A: Dataset variables*
#
#Where MEDV (colored green in Figure A above) is the target variable
#(that a model built on this dataset is intended to predict). We make
#modifications below with the understanding that LSTAT is a parameter of
#concern. 
#
#**Import and install relevant packages**
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from fairlearn.metrics import *
from fairlearn.postprocessing import * 
from fairlearn.reductions import *
from fairlearn.datasets import *
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from statistics import *
import matplotlib.pyplot as plt

#%%
# 
#**1. We devise a new target column based on CRIM (colored blue in Figure A above), indicating whether a certain per capita crime rate threshold is met.**
#
#   While we calculate and print Mean, Median, Mode, Max, and Min crime
#   rates below, the analysis to follow is specifically for the mean crime
#   rate as the chosen threshold. Similar results are observed for median
#   crime rate as the chosen threshold as well (which one can observe by
#   changing the index of threshold_crime_rates to 1).
#
bostonDS = fetch_boston(as_frame=True)
crimerate_data = bostonDS.data['CRIM']

mean_crimerate = mean(crimerate_data)
median_crimerate = median(crimerate_data)
threshold_crime_rates = [mean_crimerate, median_crimerate]
mode_crimerate = mode(crimerate_data)
min_crimerate = min(crimerate_data)
max_crimerate = max(crimerate_data)
"""
[0] = Mean
[1] = Median 
"""

y_true = (crimerate_data > threshold_crime_rates[0]) * 1

rawData = bostonDS.data
print(rawData)
print('Min crime rate', min_crimerate)
print('Max crime rate', max_crimerate)
print('Mean crime rate', mean_crimerate)
print('Median crime rate', median_crimerate)
print('Mode crime rate', mode_crimerate)
#%%
#
#**2. We build a classifier using CRIM as the target value that the model ultimately predicts, and also removing CRIM in the data used to train the model.**
#
classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)

data_without_crimerate = rawData.drop('CRIM',1)

print(data_without_crimerate)

#%%
#

classifier.fit(data_without_crimerate, y_true)

y_pred = classifier.predict(data_without_crimerate)
#%%
#
#**3. We create bins for LSTAT (colored blue in Figure A above), to understand the models performance on different LSTAT value intervals.**
#
#  While we calculate and print Mean, Median, Mode, Max, and Min LSTAT
#  values, the chosen bins are based on the observe Min and Max LSTAT
#  values.
#
lstat = bostonDS.data['LSTAT']
mean_lstat = mean(lstat)
median_lstat = median(lstat)
mode_lstat = mode(lstat)
max_lstat = max(lstat)
min_lstat = min(lstat)

print('Mean LSTAT ', mean_lstat)
print('Median LSTAT ', median_lstat)
print('Mode LSTAT ', mode_lstat)
print('Max LSTAT ', max_lstat)
print('Min LSTAT ', min_lstat)

bins = [0, 10,20,30,40]
print(bins)
labels =[1,2,3,4]

lstat_binned = pd.cut(lstat, bins,labels=labels)
#%%
# 
#**4. We treat lstat_binned (defined in step 3 above) as the sensitive feature on which to assess fairness metrics.**
#
accuracy_frame = MetricFrame(accuracy_score, y_true, y_pred, sensitive_features = lstat_binned)
accuracy_by_group = accuracy_frame.by_group

selection_frame = MetricFrame(selection_rate, y_true, y_pred, sensitive_features = lstat_binned)
selection_by_group = selection_frame.by_group

print('Selection rate statistics')
print(selection_frame.overall)
print(selection_by_group)

print('Accuracy rate statistics')
print(accuracy_frame.overall)
print(accuracy_by_group)
#%%
#
from fairlearn.widget import FairlearnDashboard
FairlearnDashboard(sensitive_features=lstat_binned, sensitive_feature_names=['lstat'],y_true=y_true,y_pred={"initial model": y_pred}) 
#%%
#
#Fairlearn Results
#-----------------
#
#Upon running Fairlearn with lstat_binned (described in Background) as
#our sensitive feature, we observe the following disparities in accuracy
#across bins, and selection rates.
#
#.. image:: https://i.imgur.com/x7zJlcU.png
#   :target: https://i.imgur.com/x7zJlcU.png
#   :alt: 
#
#*Figure B: Disparity in accuracy across lstat bins*
#
#
#.. image:: https://i.imgur.com/6Ar1pyS.png
#   :target: https://i.imgur.com/6Ar1pyS.png
#   :alt: 
#
#*Figure C: Disparity in selection rate across lstat bins*
#
#Here are some key questions to think about given the results above: 
#
#
#* Using mean or median crime rate as the threshold value, the overall
#  accuracy of the model is 95%. Does that mean the model is good to go?
#  Can there still be issues despite what seems like a properly working
#  model? Tradeoffs between accuracy and fairness?
#
#* What does it mean that the last bin has the highest selection rate?
#  Given the sociotechnical context, what would this imply in terms of
#  the societal impact and outcome?
#
#* What would a ‘mitigation’ mean in this scenario? Assuming no
#  constraints on the mitigations offered in Fairlearn modules, what type
#  of disparity would we be looking to reduce? (Explored in the section
#  to follow)
#
#Another important question to consider: should the described problem
#(policing) be framed as an ML task at all?
#
#**Exploration: What if we attempt to apply Fairlearn's mitigation functionality?**
#
# 
#*Quickstart - Mitigating Disparity*
#
#
#If we observe disparities between groups we may want to create a new
#model while specifying an appropriate fairness constraint. Note that the
#choice of fairness constraints is crucial for the resulting model, and
#varies based on application context. If selection rate is highly
#relevant for fairness in this contrived example, we can attempt to
#mitigate the observed disparity using the corresponding fairness
#constraint called Demographic Parity. In real world applications we need
#to be mindful of the sociotechnical context when making such decisions.
#The Exponentiated Gradient mitigation technique used fits the provided
#classifier using Demographic Parity as the objective, leading to a
#vastly reduced difference in selection rate.
#
#*Hypothesis*
#
#
#In this sociotechnical context, the disparity is inherent in the data
#itself, and thus an attempt at mitigation will fail.
#
np.random.seed(0)  # set seed for consistent results with ExponentiatedGradient
constraint = DemographicParity()
classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
mitigator = ExponentiatedGradient(classifier, constraint)
mitigator.fit(data_without_crimerate, y_true, sensitive_features=lstat_binned)
y_pred_mitigated = mitigator.predict(data_without_crimerate)

sr_mitigated = MetricFrame(selection_rate, y_true, y_pred_mitigated, sensitive_features=lstat_binned)
print(sr_mitigated.overall)
print(sr_mitigated.by_group)

as_mitigated = MetricFrame(accuracy_score, y_true, y_pred_mitigated, sensitive_features=lstat_binned)
print(as_mitigated.overall)
print(as_mitigated.by_group)
#%%
#
#By constraining selection rate disparity across LSTAT groups, we notice
#a significant difference in the accuracy for the 3rd and 4th bins:
#
#**Before Mitigation - 95%+ accuracy on both bins**
#
#
#**After Mitigation - 74% and 42% respectively**
#
#
#The significant drop in accuracy indicates that the underlying issue may
#not be at the model level but rather a reflection of the data (and
#sociotechnical context itself). In the given data, LSTAT is a reliable
#predictor of the crime rate threshold condition, which is a reflection
#of systemic biases leading to this observed trend. This is why accuracy
#for the higher bins falls off drastically if we try to achieve similar
#selection rates across LSTAT bins.
#
#Problem Area & Sociotechnical Context
#-------------------------------------
#
#The case study above highlights the importance of considering the
#sociotechnical aspects of tasks traditionally deemed to be solely ML
#problems. In this scenario, we’ve seen the grave consequences of failing
#to acknowledge and ensure such considerations are a part of the design
#and deployment of such ML systems. The result? A feedback loop of
#discrimination and injustice due to lack of transparency and biased
#training data. Some key takeaways for this case include:
#
#
#* Algorithms / Algorithmic decision-making systems are not ‘objective’
#* Lack of transparency and biased historical data leads to discriminatory models 
#* Biased data -> Biased predictions = Feedback loop of discrimination & injustice 
#* **Not all problems/tasks should be framed and approached as an ML task**
#
#For more background, check out this resource: `Predictive Policing
#algorithms
#<https://www.technologyreview.com/2020/07/17/1005396/predictive-policing-algorithms-racist-dismantled-machine-learning-bias-criminal-justice/>`_
#are racist (MIT Technology Review)
#
#Conclusion
#----------
#
#The case study on predictive policing and applications of machine
#learning techniques to making decision-making in the criminal justice
#system more ‘objective’ reveals that this socio-technical problem
#requires extensive thoughtful consideration. The findings show how
#biased data and lack of transparency in models can result in ‘unfair’
#models that disproportionately oppress marginalized communities -
#including low income communities as demonstrated in this case.
#
#This case study is also an interesting illustration of the general harms
#of ‘algorithmic decision-making’ systems in various socio-technical
#scenarios, including job hiring, college admissions, loan services, and
#risk-based sentencing, and how Fairlearn could play a role in uncovering
#these harms. While such technological systems are generally thought to
#provide “objectiveness” to the task of decision-making, this case study
#illustrates the importance of understanding and acknowledging the role
#of data and models in perpetuating injustice and inequality. By framing
#the underlying issue and task as ‘sociotechnical’ in nature, we must
#think more critically about the implications of such systems in
#practice, working to mitigate disparity and ensure fairness, and even
#questioning the underlying appropriateness of framing such problems as
#ML tasks.
