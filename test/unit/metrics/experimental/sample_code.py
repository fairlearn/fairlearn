# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from sklearn.model_selection import train_test_split
import sklearn.metrics as skm

import fairlearn.metrics.experimental as metrics

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

import numpy as np
import pandas as pd
import shap


X_raw, Y = shap.datasets.adult()
X_raw["Race"].value_counts().to_dict()

A = X_raw[['Sex', 'Race']]
X = X_raw.drop(labels=['Sex', 'Race'], axis=1)
X = pd.get_dummies(X)


le = LabelEncoder()
Y = le.fit_transform(Y)

X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X_raw,
                                                                     Y,
                                                                     A,
                                                                     test_size=0.2,
                                                                     random_state=0,
                                                                     stratify=Y)

# Work around indexing issue
X_train = X_train.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)

# Improve labels
A_test.Sex.loc[(A_test['Sex'] == 0)] = 'female'
A_test.Sex.loc[(A_test['Sex'] == 1)] = 'male'


A_test.Race.loc[(A_test['Race'] == 0)] = 'Amer-Indian-Eskimo'
A_test.Race.loc[(A_test['Race'] == 1)] = 'Asian-Pac-Islander'
A_test.Race.loc[(A_test['Race'] == 2)] = 'Black'
A_test.Race.loc[(A_test['Race'] == 3)] = 'Other'
A_test.Race.loc[(A_test['Race'] == 4)] = 'White'

unmitigated_predictor = LogisticRegression(solver='liblinear', fit_intercept=True)

unmitigated_predictor.fit(X_train, Y_train)

Y_pred = unmitigated_predictor.predict(X_test)
wgts = np.random.random(size=len(Y_test))

fns = [skm.recall_score, skm.precision_score, skm.fbeta_score]
spn = [['sample_weight'], ['sample_weight'], ['sample_weight']]
params = [{'sample_weight': wgts, 'zero_division': 0},
          {'sample_weight': wgts},
          {'sample_weight': wgts, 'beta': 0.75}]


result = metrics.GroupedMetric(fns, Y_test, Y_pred,
                               sensitive_features=[A_test['Race'], A_test['Sex']],
                               conditional_features=[X_test['Workclass']],
                               sample_param_names=spn,
                               params=params)

print()
print("Result")
print("Overall =================")
print(result.overall)
print()
print("By group =============")
print(result.by_group)


result2 = metrics.GroupedMetric(fns, Y_test, Y_pred,
                                sensitive_features=[A_test['Race'], A_test['Sex']],
                                sample_param_names=spn,
                                params=params)

print()
print("Result 2")
print("Overall =================")
print(result2.overall)
print()
print("By group =============")
print(result2.by_group)


fbeta_difference = metrics.make_derived_metric('difference',
                                               skm.fbeta_score,
                                               sample_param_names=['sample_weights'])

print("fbeta_difference (Race):", fbeta_difference(Y_test, Y_pred,
                                                   beta=0.7,
                                                   sensitive_features=A_test['Race']))

print("fbeta_difference (Sex):", fbeta_difference(Y_test, Y_pred,
                                                  beta=0.7,
                                                  sensitive_features=A_test['Sex']))

print("fbeta_difference (Race, Sex):", fbeta_difference(Y_test, Y_pred,
                                                        beta=0.7,
                                                        sensitive_features=A_test))

result3 = metrics.GroupedMetric(skm.fbeta_score,
                                Y_test, Y_pred,
                                sensitive_features=A_test,
                                sample_param_names=['sample_weights'],
                                params={'beta': 0.7})
print("\n\nresult3.by_group\n")
print(result3.by_group)
