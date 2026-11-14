"""Load the three datasets.

Each loader gives back X (raw features, no encoding yet), y (1 = favorable
outcome) and s (1 = the disadvantaged group).
"""

import pandas as pd

DATA_DIR = "data"


def load_adult():
    cols = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income",
    ]
    # the .test file starts with a line of column names, skip it
    train = pd.read_csv(f"{DATA_DIR}/adult.data", header=None, names=cols,
                        skipinitialspace=True)
    test = pd.read_csv(f"{DATA_DIR}/adult.test", header=None, names=cols,
                       skipinitialspace=True, skiprows=1)
    df = pd.concat([train, test], ignore_index=True)

    # labels in .test carry a trailing dot, and everything needs trimming
    df = df.apply(lambda c: c.str.strip() if c.dtype == object else c)
    df["income"] = df["income"].str.rstrip(".")

    # "?" only ever shows up in native-country, which gets dropped below, so I
    # keep every row. The LFR paper uses the old 45,222-row collection; the
    # current UCI files have 48,842 and no cleaning gets you back to that.
    df = df.replace("?", "Unknown")

    y = (df["income"] == ">50K").astype(int)
    s = (df["sex"] == "Female").astype(int)
    # fnlwgt is a survey weight and native-country is a 42-level id-ish column;
    # standard UCI setups drop both, so I do too
    X = df.drop(columns=["income", "fnlwgt", "native-country"])
    return X, y, s


def load_german():
    # 20 attribute columns + the label, space separated; names per German.names
    cols = [f"feature_{i:02d}" for i in range(20)] + ["credit-risk"]
    df = pd.read_csv(f"{DATA_DIR}/german.data", header=None, names=cols, sep=" ")

    y = (df["credit-risk"] == 1).astype(int)  # 1 = credit granted
    # fairness papers usually split this dataset on age, feature_12 in this file
    s = (df["feature_12"] < 25).astype(int)
    X = df.drop(columns=["credit-risk"])
    return X, y, s


def load_compas():
    df = pd.read_csv(f"{DATA_DIR}/compas-two-years.csv")
    # same filter as the fairlearn tutorial: binary sex, age 20-45
    keep = df["sex"].isin(["Male", "Female"]) & df["age"].between(20, 45)
    df = df[keep].reset_index(drop=True)

    y = (df["two_year_recid"] == 1).astype(int)
    s = (df["sex"] == "Female").astype(int)
    X = df[["sex", "age", "race", "juv_fel_count", "juv_misd_count",
            "juv_other_count", "priors_count", "c_charge_degree"]]
    return X, y, s
