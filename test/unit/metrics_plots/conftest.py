# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.


import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.tree import DecisionTreeClassifier


# copied verbatim from quickstart
def quickstart_setup():
    data = fetch_openml(data_id=1590, as_frame=True)
    X = pd.get_dummies(data.data)
    y_true = (data.target == '>50K') * 1
    sex = data.data['sex']

    classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
    classifier.fit(X, y_true)
    y_pred = classifier.predict(X)
    return (y_true, y_pred, sex)
