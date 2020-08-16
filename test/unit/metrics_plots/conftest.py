# copied verbatim from quickstart
def quickstart_setup():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_openml
    data = fetch_openml(data_id=1590, as_frame=True)
    X = pd.get_dummies(data.data)
    y_true = (data.target == '>50K') * 1
    sex = data.data['sex']

    from fairlearn.metrics import group_summary
    from sklearn.metrics import accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
    classifier.fit(X, y_true)
    y_pred = classifier.predict(X)
    return (y_true, y_pred, sex)
