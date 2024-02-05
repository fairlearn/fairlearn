import pandas as pd
import numpy as np

from fairlearn.datasets import fetch_adult
from fairlearn.preprocessing import OptimizedPreprocessor
from fairlearn.preprocessing._optimPreproc_helper import DTools


def get_distortion_adult(vold, vnew):
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


df = preprocessed_adult_data()
D_features = ["sex"]
Y_features = ["Income Binary"]
X_features = ["Age (decade)", "Education Years"]
optim_options = {
    "distortion_fun": get_distortion_adult,
    "epsilon": 0.05,
    "clist": [0.99, 1.99, 2.99],
    "dlist": [0.1, 0.05, 0],
}
opt = OptimizedPreprocessor(distortion_function=get_distortion_adult)
opt.fit(df=df, D_features=D_features, X_features=X_features, Y_features=Y_features)
df_transformed = opt.transform(
    df=df, D_features=D_features, X_features=X_features, Y_features=Y_features
)
assert df_transformed.shape == df.shape
print("Test Passed!")
