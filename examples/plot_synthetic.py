# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""
============================================
Synthetic dataset example with visualization
============================================
"""

# %%
from fairlearn.datasets._synthetic import (
    SensitiveDatasetMaker,
    SensitiveFeature,
)
from fairlearn.metrics import (
    MetricFrame,
    false_positive_rate,
    true_positive_rate,
    selection_rate,
    count
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

rng = np.random.RandomState(42)

gender_feature = SensitiveFeature('Gender', ['Man', 'Other', 'Unspecified', 'Woman'])
age_feature = SensitiveFeature('Age', ['<21', '21-64', '>64'])

dataset = SensitiveDatasetMaker(sensitive_features=[gender_feature, age_feature], random_state=rng)

# need a convenience method here
dataset.configured_groups[('Man', '<21')].classification_kwargs['n_samples'] = 100
X, y, features = dataset.make_sensitive_classification(n_samples=300)

X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X, y, features, test_size=0.3, random_state=rng
)

classifier = DecisionTreeClassifier(min_samples_leaf=10, max_depth=4)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Analyze metrics using MetricFrame
metrics = {
    'accuracy': accuracy_score,
    'precision': precision_score,
    'recall': recall_score,
    'false positive rate': false_positive_rate,
    'true positive rate': true_positive_rate,
    'selection rate': selection_rate,
    'count': count}
metric_frame = MetricFrame(metrics=metrics,
                           y_true=y_test,
                           y_pred=y_pred,
                           sensitive_features=sensitive_test)
metric_frame.by_group.plot.bar(
    subplots=True,
    layout=[3, 3],
    legend=False,
    figsize=[12, 8],
    title="Show all metrics",
)

# %%