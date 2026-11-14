"""Dataset loading helpers.

Each loader returns (X, y, sensitive):
  X          DataFrame of raw features (mixed numeric + string, no encoding yet)
  y          0/1 label, 1 = favorable outcome
  sensitive  0/1 protected attribute, 1 = unprivileged group
"""

import pandas as pd

DATA_DIR = "data"

ADULT_COLS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]


def load_adult():
    train = pd.read_csv(f"{DATA_DIR}/adult.data", header=None, names=ADULT_COLS,
                        skipinitialspace=True)
    test = pd.read_csv(f"{DATA_DIR}/adult.test", header=None, names=ADULT_COLS,
                       skipinitialspace=True, skiprows=1)
    df = pd.concat([train, test], ignore_index=True)

    # the labels in the test file carry a trailing dot, strip it everywhere
    df["income"] = df["income"].str.strip().str.rstrip(".")
    df = df.apply(lambda c: c.str.strip() if c.dtype == object else c)
    # the UCI files use "?" for missing values; the records themselves are
    # complete (every "?" sits in native-country, which we drop below), so I
    # keep all 48,842 rows and relabel the placeholders. Note: the LFR paper
    # cites the old 45,222-row collection, which the current UCI files no
    # longer reproduce; no cleaning on today's files gets you that number.
    df = df.replace("?", "Unknown")

    y = (df["income"] == ">50K").astype(int)

    # sex is the protected attribute; women are the disadvantaged group here
    sensitive = (df["sex"] == "Female").astype(int)

    # fnlwgt is a survey weight, native-country is a 42-level id-like column;
    # both hurt the model more than they help, so standard UCI setups drop them
    X = df.drop(columns=["income", "fnlwgt", "native-country"])
    return X, y, sensitive


GERMAN_COLS = [f"feature_{i:02d}" for i in range(20)] + ["credit-risk"]


def load_german():
    df = pd.read_csv(f"{DATA_DIR}/german.data", header=None, names=GERMAN_COLS,
                     sep=" ")
    # column order follows data/german.data as published by UCI, where the
    # 20 attribute names are documented in the German.names file
    # label 1 = credit granted (good risk), 2 = default (bad risk)
    y = (df["credit-risk"] == 1).astype(int)

    # common setup in the fairness literature: age >= 25 is the privileged
    # group; feature_12 is the age column in this file's layout
    sensitive = (df["feature_12"] < 25).astype(int)

    X = df.drop(columns=["credit-risk"])
    return X, y, sensitive


def load_compas():
    df = pd.read_csv(f"{DATA_DIR}/compas-scores-two-years.csv")

    # standard cleaning, same as the fairlearn tutorial: keep binary sex,
    # keep young/middle adults, target is two year recidivism
    keep = df["sex"].isin(["Male", "Female"]) & df["age"].between(20, 45)
    df = df[keep].reset_index(drop=True)

    y = (df["two_year_recid"] == 1).astype(int)

    # men are the over-criticised group in this dataset, women are disadvantaged
    sensitive = (df["sex"] == "Female").astype(int)

    features = ["sex", "age", "race", "juv_fel_count", "juv_misd_count",
                "juv_other_count", "priors_count", "c_charge_degree"]
    X = df[features]
    return X, y, sensitive
