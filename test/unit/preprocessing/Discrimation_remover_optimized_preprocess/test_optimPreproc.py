import warnings

import numpy as np
import pandas as pd

from fairlearn.datasets import fetch_adult
from fairlearn.preprocessing import OptimizedPreprocessor
from fairlearn.utils._fixes import parametrize_with_checks


def get_distortion_adult_dataframe(vold, vnew):
    """Distortion function for the adult dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """

    # Define local functions to adjust education and age
    def adjustEdu(v):
        if v == ">12":
            return 13
        elif v == "<6":
            return 5
        else:
            return int(v)

    def adjustAge(a):
        if a == ">=70":
            return 70.0
        else:
            return float(a)

    def adjustInc(a):
        if a == "<=50K":
            return 0
        elif a == ">50K":
            return 1
        else:
            return int(a)

    # value that will be returned for events that should not occur
    bad_val = 3.0

    # Adjust education years
    eOld = adjustEdu(vold["Education Years"])
    eNew = adjustEdu(vnew["Education Years"])

    # Education cannot be lowered or increased in more than 1 year
    if (eNew < eOld) | (eNew > eOld + 1):
        return bad_val

    # adjust age
    aOld = adjustAge(vold["Age (decade)"])
    aNew = adjustAge(vnew["Age (decade)"])

    # Age cannot be increased or decreased in more than a decade
    if np.abs(aOld - aNew) > 10.0:
        return bad_val

    # Penalty of 2 if age is decreased or increased
    if np.abs(aOld - aNew) > 0:
        return 2.0

    # Adjust income
    incOld = adjustInc(vold["Income Binary"])
    incNew = adjustInc(vnew["Income Binary"])

    # final penalty according to income
    if incOld > incNew:
        return 1.0
    else:
        return 0.0


def get_distortion_adult_numpy(vold, vnew):
    """Distortion function for the adult dataset. We set the distortion
    metric here. See section 4.3 in supplementary material of
    http://papers.nips.cc/paper/6988-optimized-pre-processing-for-discrimination-prevention
    for an example

    Note:
        Users can use this as templates to create other distortion functions.

    Args:
        vold (dict) : {attr:value} with old values
        vnew (dict) : dictionary of the form {attr:value} with new values

    Returns:
        d (value) : distortion value
    """

    # Define local functions to adjust education and age
    def adjustEdu(v):
        if v == ">12":
            return 13
        elif v == "<6":
            return 5
        else:
            return int(v)

    def adjustAge(a):
        if a == ">=70":
            return 70.0
        else:
            return float(a)

    def adjustInc(a):
        if a == "<=50K":
            return 0
        elif a == ">50K":
            return 1
        else:
            return int(a)

    # value that will be returned for events that should not occur
    bad_val = 3.0

    # Adjust education years
    eOld = adjustEdu(vold["column_2"])
    eNew = adjustEdu(vnew["column_2"])

    # Education cannot be lowered or increased in more than 1 year
    if (eNew < eOld) | (eNew > eOld + 1):
        return bad_val

    # adjust age
    aOld = adjustAge(vold["column_1"])
    aNew = adjustAge(vnew["column_1"])

    # Age cannot be increased or decreased in more than a decade
    if np.abs(aOld - aNew) > 10.0:
        return bad_val

    # Penalty of 2 if age is decreased or increased
    if np.abs(aOld - aNew) > 0:
        return 2.0

    # Adjust income
    incOld = adjustInc(vold["column_4"])
    incNew = adjustInc(vnew["column_4"])

    # final penalty according to income
    if incOld > incNew:
        return 1.0
    else:
        return 0.0


def preprocessed_adult_data():
    df = pd.merge(
        fetch_adult()["data"],
        fetch_adult()["target"],
        left_index=True,
        right_index=True,
    )
    df["Age (decade)"] = df["age"].apply(lambda x: np.floor(x / 10.0) * 10.0)
    df["Age (decade)"] = df["age"].apply(lambda x: np.floor(x / 10.0) * 10.0)

    def group_edu(x):
        if x <= 5:
            return "<6"
        elif x >= 13:
            return ">12"
        else:
            return x

    def age_cut(x):
        if x >= 70:
            return ">=70"
        else:
            return x

    # Limit education range
    df["Education Years"] = df["education-num"].apply(lambda x: group_edu(x))
    df["Education Years"] = df["Education Years"].astype("str")
    # Limit age range
    df["Age (decade)"] = df["Age (decade)"].apply(lambda x: age_cut(x))
    df["Age (decade)"] = df["Age (decade)"].astype("str")
    # Transform all that is non-white into 'minority'
    df["Race"] = df["race"].apply(lambda x: x if x == " White" else "Minority")

    # Add binary income variable
    df["Income Binary"] = df["class"].apply(lambda x: 1 if x == ">50K" else 0)
    features = ["Age (decade)", "Education Years", "sex", "Income Binary"]

    # keep only the features we will use
    df = df[features]
    return df


def test_optimized_preprocessor_with_dataframe():
    """Test OptimizedPreprocessor with pandas DataFrame input."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        warnings.filterwarnings("ignore", message=".*dropping on a non-lexsorted multi-index.*")

        df = preprocessed_adult_data()
        sensitive_features = ["sex"]

        preprocessor = OptimizedPreprocessor(
            sensitive_features,
            distortion_function=get_distortion_adult_dataframe,
        )

        X = df[sensitive_features + ["Age (decade)", "Education Years"]]
        y = df[["Income Binary"]]

        preprocessor.fit(X, y)
        df_trans = preprocessor.transform(X, y)

        assert df_trans.shape == df.shape


def test_optimized_preprocessor_with_numpy():
    """Test OptimizedPreprocessor with NumPy array input."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        warnings.filterwarnings("ignore", message=".*dropping on a non-lexsorted multi-index.*")

        df = preprocessed_adult_data()
        sensitive_features = ["sex"]

        preprocessor = OptimizedPreprocessor(
            [df.columns.get_loc(sensitive_features[0])],
            distortion_function=get_distortion_adult_numpy,
        )

        X = df[["Age (decade)", "Education Years"] + sensitive_features].to_numpy()
        y = df[["Income Binary"]].to_numpy()

        preprocessor.fit(X, y)
        X_trans = preprocessor.transform(X, y)

        assert X_trans.shape == df.shape


EXPECTED_FAILED_CHECKS = {
    "OptimizedPreprocessor": {},
}


@parametrize_with_checks(
    [OptimizedPreprocessor([2], distortion_function=get_distortion_adult_numpy)],
    expected_failed_checks=lambda x: EXPECTED_FAILED_CHECKS.get(x.__class__.__name__, {}),
)
def test_estimators(estimator, check):
    """Check the compatibility with scikit-learn API."""
    check(estimator)
