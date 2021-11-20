# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

"""
================================================
Mitigating Fairness using Adversarial Mitigation
================================================
"""
# %%
# This notebook demonstrates our implementation of the technique *Mitigating*
# *Unwanted Biases with Adversarial Learning* as proposed by
# `Zhang et al. 2018 <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_.
#
# In short, the authors take classic supervised learning setting in which
# a predictor neural network is trained, and extend it with an adversarial
# network that aims to predict the sensitive feature. Then, they train the
# predictor not only to minimize its own loss, but also minimize the predictive
# ability of the adversarial.
#
# In short, we provide an implementation that supports:
#
# - Any predictor neural network implemented in either PyTorch or Tensorflow2
# - Classification or regression
# - Multiple sensitive features
# - Two fairness objectives: Demographic parity or Equalized Odds
#
# This implementation follows closely the API of an `Estimator` in :class:`sklearn`

# %%
# Example 1: Simple use case with UCI Adult Dataset
# =================================================
# Firstly, we cover the most basic application of adversarial mitigation.
# We start by loading and preprocessing the dataset.
#
# For this example we choose the feature 'sex' as the sensitive feature.


# %%
# Imports used by the rest of the script
from math import sqrt
from fairlearn.metrics import MetricFrame, selection_rate, \
    demographic_parity_difference
from sklearn.metrics import accuracy_score
from fairlearn.adversarial import AdversarialFairnessClassifier, \
    AdversarialFairness
from pandas import Series
from numpy import number, random, mean
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import torch
random.seed(123)
torch.manual_seed(123)

X, y = fetch_openml(data_id=1590, as_frame=True, return_X_y=True)

# Remove rows with NaNs. In many cases, dropping rows that contain missing values is not the best approach for dealing with missing values,
# but for this example we ignore that.
non_NaN_rows = ~X.isna().any(axis=1)

X = X[non_NaN_rows]
y = y[non_NaN_rows]

# Choose sensitive feature
sensitive_feature = X['sex']

# %%
# The UCI adult dataset can not be fed into a neural network (yet),
# as we have many columns that are not numerical in nature. To resolve this
# issue, we could for instance use one-hot-encodings to preprocess categorical
# columns. Additionally, let's preprocess the columns of number to a
# standardized range. For these tasks, we can use functionality from
# `sklearn.preprocessor`.


def transform(X):
    if isinstance(X, Series):  # make_column_transformer works with DataFrames
        X = X.to_frame()
    ct = make_column_transformer(
        (StandardScaler(),
         make_column_selector(dtype_include=number)),
        (OneHotEncoder(drop='if_binary', sparse=False),
         make_column_selector(dtype_include="category")))
    return ct.fit_transform(X)


X = transform(X)
y = transform(y)
sensitive_feature = transform(sensitive_feature)

X_train, X_test, Y_train, Y_test, Z_train, Z_test = \
        train_test_split(X, y, sensitive_feature,
                         test_size=0.2,
                         random_state=12345,
                         stratify=y)

# %%
# Now, we can use :class:`fairlearn.adversarial.AdversarialFairnessClassifier` to train on the
# UCI Adult dataset. As our predictor and adversary models, we use for
# simplicity the default constructors for fully connected neural
# networks with sigmoid activations implemented in Fairlearn. We initialize neural network constructors
# by passing a list :math:`h_1, h_2, \dots` that indicate the number of nodes
# :math:`h_i` per hidden layer :math:`i`.
#
# The specific fairness
# objective that we choose for this example is demographic parity, so we also
# set :code:`objective = "demographic_parity"`.


mitigator = AdversarialFairnessClassifier(
    predictor_model=[50, 20],
    adversary_model=[6, 6],
    constraints="demographic_parity",
    learning_rate=0.0001,
    epochs=100,
    batch_size=2**9,
    shuffle=True,
    progress_updates=5
)

# %%
# This definition of the mitigator is equivalent to the following

mitigator = AdversarialFairness(
    backend="torch",
    predictor_model=[50, 20],
    adversary_model=[6, 6],
    predictor_loss=torch.nn.BCEWithLogitsLoss(reduction='mean'),
    adversary_loss=torch.nn.BCEWithLogitsLoss(reduction='mean'),
    predictor_function=lambda pred: (pred >= 0.).astype(float),
    constraints="demographic_parity",
    optimizer="Adam",
    learning_rate=0.0001,
    alpha=1.0,
    cuda=False,
    epochs=100,
    batch_size=2**9,
    shuffle=True,
    progress_updates=5
)

# %%
# Then, we can fit the data to our model. We generally follow sklearn API,
# but in this case we require some extra kwargs. In particular, we should
# specify the number of epochs, batch size, whether to shuffle the rows of data
# after every epoch, and optionally after how many seconds to show a progress
# update.

mitigator.fit(
    X_train,
    Y_train,
    sensitive_features=Z_train
)

# %%
# Predict and evaluate. In particular, we trained the predictor for demographic
# parity, so we are not only interested in the accuracy, but also in the selection
# rate.

predictions = mitigator.predict(X_test)

mf = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'selection_rate': selection_rate},
    y_true=Y_test,
    y_pred=predictions,
    sensitive_features=Z_test)

print(mf.by_group)

# %%
# We see that the results are not great. The accuracy is not optimal, and there
# remains demographic disparity. That is because there is no known out-of-the-box
# solution that is able to train an adversarial network succesfully. There are many
# variables at play, and we will improve our model in the next section.

# %%
# Example 2: More advanced models
# ===============================
# Below we experiment with our models in order to achieve better results than above.
# Adversarial Learning is inherently difficult because models can diverge quickly.
# Intuitively, you should imagine that there are "very easy" local minima that the
# models may converge to. For instance, if the predictor always outputs class=0,
# then the adversary's objective is much easier, namely it only has to output the
# most correlated sensitive feature to class=0. Such minima are very easily reached
# unfortunately. #TODO this description is very bad at the moment, working on
# understanding it:).
#
# So, to finetune our model and the training thereof, we start by defining our
# neural networks in the way we'd like them and in the way we can easily tweak them.
# We will be using PyTorch, but the same can be achieved using Tensorflow!
random.seed(123)
torch.manual_seed(123)

X = X_train
Y = Y_train
Z = Z_train


class PredictorModel(torch.nn.Module):
    def __init__(self):
        super(PredictorModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 50),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(50, 5),
            torch.nn.Sigmoid(),
            torch.nn.Linear(5, 1)
        )  # NO final activation function!

    def forward(self, x):
        return self.layers(x)


class AdversaryModel(torch.nn.Module):
    def __init__(self):
        super(AdversaryModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 6),
            torch.nn.Sigmoid(),
            torch.nn.Linear(6, 6),
            torch.nn.Sigmoid(),
            torch.nn.Linear(6, 1),
        )  # NO final activation function!

    def forward(self, x):
        return self.layers(x)


predictor_model = PredictorModel()
adversary_model = AdversaryModel()
# %%
# Xavier initialization is more popular than PyTorch's default initialization, so
# let's put that to the test. Note that I also initialize the biases, but this is
# less common in practice. Intuitively, it seems wise to initialize small weights,
# so we set the gain low.


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data, gain=0.1)
        torch.nn.init.xavier_normal_(m.bias.data, gain=0.1)


predictor_model.apply(weights_init)
adversary_model.apply(weights_init)

# %%
# We may define the optimizers however we like. In this case, let's use a
# predictor optimizer with SGD with decaying learning rate, and Adam for the
# adversary.
predictor_optimizer = torch.optim.SGD(predictor_model.parameters(), lr=0.5, momentum=0.9, nesterov=True)
adversary_optimizer = torch.optim.Adam(adversary_model.parameters(), lr=0.1)
scheduler1 = torch.optim.lr_scheduler.LambdaLR(predictor_optimizer, lr_lambda=lambda epoch: 1/(3*epoch+2))

# %%
# Then, the instance itself. We Will define the loss function with
# weights because we are dealing with an imbalanced dataset.

mitigator = AdversarialFairnessClassifier(
    predictor_model=predictor_model,
    adversary_model=adversary_model,
    predictor_loss=torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.ones([1])*mean(Y == 1.)),
    adversary_loss=torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.ones([1])*mean(Z == 1.)),
    predictor_function="binary",
    optimizer=None,
    predictor_optimizer=predictor_optimizer,
    adversary_optimizer=adversary_optimizer,
    alpha=0.1,
    constraints='demographic_parity',
    epochs=13,
    batch_size=2**8,
    shuffle=True,
    progress_updates=None,
    callback_fn=callback_fn,
    warm_start=True
)

# %%
# Instead of only looking at training loss, we also take a look at some validation
# metrics. For this, we chose the demographic parity difference to check to what
# extent the constraint (demographic parity in this case) is satisfied.


def validate():
    predictions = mitigator.predict(X_test)
    dp_diff = demographic_parity_difference(Y_test,
                                            predictions,
                                            sensitive_features=Z_test)
    accuracy = mean(predictions == Y_test)
    selection_rate = mean(predictions == 1.)
    print("DP diff: {:.4f}, accuracy: {:.4f}, selection_rate: {:.4f}".format(
        dp_diff, accuracy, selection_rate))

# %%
# We make use of a callback function to incorporate the validation into the training
# loop. Additionally, we handle the prediction_optimizer updates here.


epoch_count = 1


def callback_fn():
    global epoch_count
    mitigator.alpha = 0.1 * sqrt(epoch_count)
    epoch_count += 1
    scheduler1.step()

    validate()

# %%
# Finally, we fit the model


mitigator.fit(
    X,
    Y,
    sensitive_features=Z)


# %%
# Then, we more carefully improve this by changing the model. That is, we do not
# change the predictor/adversary model parameters, but we do change the learning
# parameters. In particular, we set very small learning rates and alpha a bit higher.
# As mentioned earlier, training adversarially is tricky, and ideally
# you'd train this in a controlled way.


predictor_optimizer = torch.optim.Adam(predictor_model.parameters(), lr=0.0001)
adversary_optimizer = torch.optim.Adam(adversary_model.parameters(), lr=0.0001)
mitigator.alpha = 5


def callback_fn():
    validate()


mitigator.fit(
    X,
    Y,
    sensitive_features=Z)
# %%
# We take a look at the results. Notice we achieve a much lower demographic parity
# difference than in Exercise 1! This does come at the cost of some accuracy, but
# such a tradeof is to be expected.

predictions = mitigator.predict(X_test)

mf = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'selection_rate': selection_rate},
    y_true=Y_test,
    y_pred=predictions,
    sensitive_features=Z_test)

print(mf.by_group)
