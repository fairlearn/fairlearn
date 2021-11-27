# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import torch
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score
from fairlearn.adversarial import AdversarialFairnessClassifier


class PredictorModel(torch.nn.Module):
    def __init__(self):
        super(PredictorModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(10, 10),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(10, 1),
            torch.nn.Sigmoid(),
        )  # NO final activation function!

    def forward(self, x):
        return self.layers(x)


class AdversaryModel(torch.nn.Module):
    def __init__(self):
        super(AdversaryModel, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 3),
            torch.nn.LeakyReLU(),
            # torch.nn.Linear(3, 3),
            # torch.nn.LeakyReLU(),
            torch.nn.Linear(3, 1),
            torch.nn.Sigmoid(),
        )  # NO final activation function!

    def forward(self, x):
        return self.layers(x)


predictor_model = PredictorModel()
adversary_model = AdversaryModel()

gain = 0.01  # torch.nn.init.calculate_gain('leaky_relu', param=None)

torch.manual_seed(123)


def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0.0)
        # torch.nn.init.xavier_normal_(m.bias.data, gain=gain)


predictor_model.apply(weights_init)
adversary_model.apply(weights_init)

predictor_optimizer = torch.optim.Adam(
    predictor_model.parameters(), lr=0.001, weight_decay=0.1
)

adversary_optimizer = torch.optim.Adam(
    adversary_model.parameters(), lr=0.001, weight_decay=0.1
)

# Toy example
n = 1000000
r = np.random.choice([0, 1], size=n, replace=True)
v = np.random.normal(loc=r, scale=1, size=n)
u = np.random.normal(loc=v, scale=1, size=n)
w = np.random.normal(loc=v, scale=1, size=n)

y_ = 1.0 * (w > 0)

X = np.column_stack((r, u))
y_ = y_.reshape(-1, 1)
sensitive_feature = r.reshape(-1, 1)


def evaluate(pred):
    mf = MetricFrame(
        # , "demographic parity": demographic_parity_difference},
        metrics={"accuracy": accuracy_score, "selection rate": selection_rate},
        y_true=y_.copy(),
        y_pred=pred,
        sensitive_features=sensitive_feature,
    )
    print(str(sum(pred == 1)) + "/" + str(sum(pred == 0)))
    print(mf.overall)
    print(mf.by_group)
    return mf


predictor_criterion = torch.nn.BCELoss()
adversary_criterion = torch.nn.BCELoss()

predictor_optimizer = torch.optim.Adam(predictor_model.parameters(), lr=0.001)
adversary_optimizer = torch.optim.Adam(adversary_model.parameters(), lr=0.001)

scheduler1 = torch.optim.lr_scheduler.ExponentialLR(
    predictor_optimizer, gamma=0.999
)
scheduler2 = torch.optim.lr_scheduler.ExponentialLR(
    adversary_optimizer, gamma=0.999
)

step = 1


def callbackfn(model, epoch, batch):
    global step
    step += 1
    if step % 2 == 0:
        model.alpha = sqrt(step // 2)
        scheduler1.step()
        scheduler2.step()
    if step % 100 == 0:
        evaluate(model.predict(X))


model = AdversarialFairnessClassifier(
    predictor_model=predictor_model,
    adversary_model=adversary_model,
    predictor_loss=predictor_criterion,
    adversary_loss=adversary_criterion,
    predictor_optimizer=predictor_optimizer,
    adversary_optimizer=adversary_optimizer,
    epochs=1,
    batch_size=2 ** 7,
    alpha=1,
    progress_updates=1,
    callback_fn=callbackfn,
)

predictor_model.train()

x, y, z = model._validate_input(X, y_, sensitive_feature)


from math import ceil, sqrt

# print("manual training predictor")
# batch_size = 2**6
# epochs = 15
# if batch_size == -1:
#     batch_size = x.shape[0]
# batches = ceil(x.shape[0] / batch_size)

# predictor_optimizer.zero_grad()
# for epoch in range(epochs):
#     for batch in range(batches):

#         batch_slice = slice(batch * batch_size, min((batch + 1) * batch_size, x.shape[0]))


#         Y_hat = predictor_model(x[batch_slice])
#         LP = torch.nn.BCELoss()(Y_hat, y[batch_slice])
#         LP.backward()  # Check what this does at some point in time

#         predictor_optimizer.step()


#     if True and epoch != epochs - 1:
#         x, y, z = model.backendEngine_.shuffle(x, y, z)

# evaluate(model.predict(X))

model.fit(X, y, sensitive_features=sensitive_feature)

evaluate(model.predict(X))
