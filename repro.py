import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from fairlearn.postprocessing import ThresholdOptimizer

from fairlearn.metrics import (
    MetricFrame,
    false_positive_rate,
    false_negative_rate
)

# Set global 

sensitive = ["race", "gender"]
#sensitive = ["combined"]

random_seed = 445

race_map = {
    "Caucasian": "Caucasian",
    "AfricanAmerican": "AfricanAmerican",
    "?": "Other",
    "Hispanic": "Hispanic",
    "Other": "Other",
    "Asian": "Asian"
}

fairness_metrics = {
    "false_positive_rate": false_positive_rate,
    "positive_count": lambda true, pred: np.sum(true),
    "false_negative_rate": false_negative_rate,
    "negative_count": lambda true, pred: np.sum(1-true),
    "balanced_accuracy": balanced_accuracy_score
}

# Prepare Dataset

df = (pd.read_csv("diabetic_data_prepared.csv")
        .replace({"race": race_map}))
df = df.query("gender != 'Unknown/Invalid'")

categorical_features = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "admission_source_id",
    "max_glu_serum",
    "A1Cresult"
]

for col_name in categorical_features:
    df.loc[:, col_name] = df[col_name].astype("category")

Y, A = df.loc[:, "readmit_binary"], df.loc[:, ["race", "gender"]]
A.loc[:, "combined"] = A.apply(lambda x: "{0}_{1}".format(x.race, x.gender), axis=1)

X = df.drop(columns = [
    "race",
    "gender",
    "payer_code",
    "medical_specialty",
    "readmitted",
    "readmit_binary"
])

# Train fairness-unaware model

X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(
    X,
    Y,
    A,
    test_size=0.25,
    stratify=Y,
    random_state=random_seed
)

column_transformer = ColumnTransformer([
    ("numeric", StandardScaler(), make_column_selector(dtype_exclude="category")),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_include="category"))
])

X_train = column_transformer.fit_transform(X_train).toarray()
X_test = column_transformer.transform(X_test).toarray()

estimator = LogisticRegression(max_iter=1000)
estimator.fit(X_train, Y_train)


# Create Post-processing ThresholdOptimizer

postprocess_est = ThresholdOptimizer(
    estimator=estimator,
    constraints="equalized_odds",
    objective="balanced_accuracy_score"
)

postprocess_est.fit(X_train,
Y_train,
sensitive_features=A_train.loc[:, sensitive]
)

Y_pred_postprocess = postprocess_est.predict(X_test, sensitive_features=A_test.loc[:, sensitive])

metricframe_postprocess = MetricFrame(
    fairness_metrics,
    Y_test,
    Y_pred_postprocess,
    sensitive_features=A_test.loc[:, sensitive]
)

print(metricframe_postprocess.by_group)

print(postprocess_est.interpolated_thresholder_.interpolation_dict)