# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from sklearn.naive_bayes import BernoulliNB, CategoricalNB
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import pandas as pd

from fairlearn.preprocessing import Reweighing
from fairlearn.metrics import MetricFrame


def main_test(est, X, Y, A, pass_sf=False, assert_improvement=False):
    est.fit(X, Y)
    pred_original = est.predict(X)
    diff_original = MetricFrame(
        metrics=accuracy_score,
        y_true=Y,
        y_pred=pred_original,
        sensitive_features=A,
    ).difference()

    metaest = Reweighing(est, passthrough_sensitive_features=pass_sf)
    metaest.fit(X, Y, sensitive_features=A)

    pred_mitigated = est.predict(X)
    diff_mitigated = MetricFrame(
        metrics=accuracy_score,
        y_true=Y,
        y_pred=pred_mitigated,
        sensitive_features=A,
    ).difference()

    # FIXME: There is nothing to test right? No errors occured. Don't know if
    # asserting an improvement in difference makes sense, I would guess not...
    if assert_improvement:
        assert diff_mitigated < diff_original


def test_single_sensitive_feature():
    data = fetch_openml(data_id=1590, as_frame=True)
    X = pd.get_dummies(data.data)
    Y = (data.target == ">50K") * 1
    A = (data.data["sex"] == "Male") * 1

    est = BernoulliNB()
    main_test(est, X, Y, A, assert_improvement=True)


def test_categorical():
    data = fetch_openml(data_id=469, as_frame=True)
    X = pd.get_dummies(data.data)
    Y = data.target
    A = data.data["Ethnic"]

    est = CategoricalNB()
    main_test(est, X, Y, A)
