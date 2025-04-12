"""
.. _adversarial_Example_2:

================================================
`AdversarialFairness` Fine Tuning
================================================

Adversarial learning is inherently difficult because of various issues,
such as mode collapse, divergence, and diminishing gradients.
In particular, mode collapse seems a real problem on this dataset: the
predictor and adversary trap themselves in a local minimum by favoring one
class (mode). Problems with diverging parameters may also occur, which
may be an indication of a bad choice of hyperparameters, such as a
learning rate that is too large. The problems that a user may encounter are
of course case specific, but general good practices when training
such models are: train slowly, ensuring the
losses remain balanced, and keep track of validation accuracies.
Additionally, we found that single hidden layer neural
networks work best for this use case.

In this example, we demonstrate some of these good practices.
We start by defining our
predictor neural network explicitly so that it is more apparent.
We will be using PyTorch, but the same can be achieved using Tensorflow::
"""

# %%
# First, we cover a most basic application of adversarial mitigation.
# We start by loading and preprocessing the dataset:

from fairlearn.datasets import fetch_adult

X, y = fetch_adult(return_X_y=True)
pos_label = y[0]

z = X["sex"]  # In this example, we consider 'sex' the sensitive feature.

# %%
# As with other machine learning methods, it is wise to take a train-test split
# of the data in order to validate the model on unseen data:

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, y_test, Z_train, Z_test = train_test_split(
    X, y, z, test_size=0.2, random_state=12345, stratify=y
)

# %%
# The UCI adult dataset cannot be fed into a neural network (yet),
# as we have many columns that are not numerical in nature. To resolve this
# issue, we could for instance use one-hot encodings to preprocess categorical
# columns. Additionally, let's preprocess the numeric columns to a
# standardized range. For these tasks, we can use functionality from
# scikit-learn (:py:mod:`sklearn.preprocessing`). We also use an imputer
# to get rid of NaN's:

import sklearn
from numpy import number
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sklearn.set_config(enable_metadata_routing=True)

ct = make_column_transformer(
    (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("normalizer", StandardScaler()),
            ]
        ),
        make_column_selector(dtype_include=number),
    ),
    (
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(drop="if_binary", sparse_output=False)),
            ]
        ),
        make_column_selector(dtype_include="category"),
    ),
)

# %%
# Now we define the PyTorch model to be used in the adversarial fairness
# classifier.
import torch


class PredictorModel(torch.nn.Module):
    def __init__(self):
        super(PredictorModel, self).__init__()
        self.layers = torch.nn.Sequential(
            # in_features is the number of features coming out of the above
            # ColumnTransformer
            torch.nn.Linear(in_features=104, out_features=200),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=200, out_features=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


predictor_model = PredictorModel()

# %%
# We also take a look at some validation
# metrics. Most importantly, we chose the demographic parity difference
# to check to what
# extent the constraint (demographic parity in this case) is satisfied.
# We also look at the selection rate to observe whether our model is
# suffering from mode collapse, and we also calculate the accuracy on the
# validation set as well.
# We will pass this validation step to our model later::

from numpy import mean

from fairlearn.metrics import demographic_parity_difference


def validate(pipeline, X, y, z, pos_label):
    predictions = pipeline.predict(X)
    dp_diff = demographic_parity_difference(
        y == pos_label,
        predictions == pos_label,
        sensitive_features=z,
    )
    accuracy = mean(predictions == y.values)
    selection_rate = mean(predictions == pos_label)
    print(
        "DP diff: {:.4f}, accuracy: {:.4f}, selection_rate: {:.4f}".format(
            dp_diff, accuracy, selection_rate
        )
    )
    return dp_diff, accuracy, selection_rate


# %%
# We may define the optimizers however we like. In this case, we use the
# suggestion from the paper to set the hyperparameters :math:`\alpha` and learning
# rate :math:`\eta` to depend on the timestep such that :math:`\alpha \eta
# \rightarrow 0` as the timestep grows:


schedulers = []


def optimizer_constructor(model):
    global schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    schedulers.append(torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995))
    return optimizer


# %%
# We make use of a callback function to both update the hyperparameters and to
# validate the model. We update these hyperparameters at every 10 steps, and we
# validate every 100 steps. Additionally, we can implement early stopping
# easily by calling :code:`return True` in a callback function:

from math import sqrt


def callbacks(model, step, X, y, z, pos_label):
    global schedulers
    # Update hyperparameters
    model.alpha = 0.3 * sqrt(step // 1)
    for scheduler in schedulers:
        scheduler.step()
    # Validate (and early stopping) every 50 steps
    if step % 50 == 0:
        dp_diff, accuracy, selection_rate = validate(model, X, y, z, pos_label)
        # Early stopping condition:
        # Good accuracy + low dp_diff + no mode collapse
        if dp_diff < 0.03 and accuracy > 0.8 and selection_rate > 0.01 and selection_rate < 0.99:
            return True


# %%
# Then, the instance itself. Notice that we do not explicitly define loss
# functions, because adversarial fairness is able to infer the loss function
# on its own in this example:
from fairlearn.adversarial import AdversarialFairnessClassifier

mitigator = AdversarialFairnessClassifier(
    predictor_model=predictor_model,
    adversary_model=[3, "leaky_relu"],
    predictor_optimizer=optimizer_constructor,
    adversary_optimizer=optimizer_constructor,
    epochs=10,
    batch_size=2**7,
    shuffle=True,
    callbacks=callbacks,
    random_state=123,
)

# %%
# We now put the above model in a ``Pipeline`` with the transformation step. Note
# that we use ``scikit-learn``'s metadata routing to pass the sensitive feature::

from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(ct, mitigator.set_fit_request(sensitive_features=True))

# %%
# Then, we fit the model:

pipeline.fit(X_train, Y_train, sensitive_features=Z_train)

from sklearn.metrics import accuracy_score

# %%
# Finally, we validate as before, and take a look at the results:
from fairlearn.metrics import MetricFrame, selection_rate

# to see DP difference, accuracy, and selection_rate
print(validate(pipeline, X_test, y_test, z=Z_test, pos_label=pos_label))
predictions = pipeline.predict(X_test)
mf = MetricFrame(
    metrics={"accuracy": accuracy_score, "selection_rate": selection_rate},
    y_true=y_test == pos_label,
    y_pred=predictions == pos_label,
    sensitive_features=Z_test,
)
print(mf.by_group)

# Notice we achieve a much lower demographic parity
# difference than in Exercise 1! This may come at the cost of some accuracy,
# but such a tradeoff is to be expected as we are purposely mitigating
# the unfairness that was present in the data.

sklearn.set_config(enable_metadata_routing=False)
