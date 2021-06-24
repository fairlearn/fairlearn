import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, precision_score, recall_score
import scikeras
import warnings
from tensorflow import get_logger
import numpy as np
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skm
%matplotlib inline
import fairlearn
from fairlearn.metrics import selection_rate

from fairlearn.metrics import (
    MetricFrame,
    false_positive_rate,
    true_positive_rate,
    selection_rate
)

get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore", message="Setting the random state for TF")


data = fetch_openml(data_id=1590, as_frame=True)
X_raw = data.data
Y = (data.target == '>50K') * 1
sex = data.data["sex"]

X = X_raw.drop(labels=['sex'], axis=1)
X = pd.get_dummies(X)

def get_clf(meta, hidden_layer_sizes, dropout):
    n_features_in_ = meta["n_features_in_"]
    n_classes_ = meta["n_classes_"]
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(n_features_in_,)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(keras.layers.Dense(hidden_layer_size, activation="relu"))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    return model


clf = KerasClassifier(
    model=get_clf,
    loss="binary_crossentropy",
    hidden_layer_sizes=(100,),
    dropout=0.5,
)

clf.fit(X, Y);

y_pred = clf.predict(X)

metrics = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'false positive rate': false_positive_rate,
    'true positive rate': true_positive_rate,
    'selection rate': selection_rate,
    'count': lambda Y, y_pred: len(Y),
    'average precision':skm.average_precision_score,
    'roc-auc': skm.roc_auc_score,
}

metric_frame = MetricFrame(metrics,
                           y_true=Y,
                           y_pred=y_pred,
                           sensitive_features=sex)

metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics",
)


# Customize plots with ylim
    metric_frame.by_group.plot(
    kind="bar",
    ylim=[0, 1],
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics with assigned y-axis range",
)

# Customize plots with colormap
metric_frame.by_group.plot(
    kind="bar",
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    colormap="Accent",
    title="Show all metrics in Accent colormap",
)

# Customize plots with kind
metric_frame.by_group.plot(
    kind="pie",
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics in pie",
)

fairlearn.metrics.false_negative_rate(Y, y_pred, sample_weight=None)
fairlearn.metrics.false_positive_rate(Y, y_pred, sample_weight=None)
fairlearn.metrics.demographic_parity_difference(Y, y_pred, sensitive_features=sex, sample_weight=None)
sr = MetricFrame(selection_rate, Y, y_pred, sensitive_features=sex)
sr.overall
sr.by_group

