# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

from ._constants import _IMPORT_ERROR_MESSAGE, _KWARG_ERROR_MESSAGE
from copy import deepcopy

class AdversarialDebiasing():
    def __init__(self, *, 
            environment = 'torch', 
            predictor_model,
            predictor_criterion, 
            adversary_model,
            adversary_criterion,
    ):
        self.torch = False
        self.tensorflow = False
        if environment == 'torch':
            try:
                import torch
            except ImportError:
                raise RuntimeError(_IMPORT_ERROR_MESSAGE.format('torch'))
            self.torch = True
        
        elif environment == 'tensorflow':
            try:
                import torch
            except ImportError:
                raise RuntimeError(_IMPORT_ERROR_MESSAGE.format('tensorflow'))
            
            self.tensorflow = True

        else:
            raise ValueError(_KWARG_ERROR_MESSAGE.format("environment", "one of \[\'torch\',\'tensorflow\'\]"))

        self.predictor_model = predictor_model
        self.predictor_criterion = predictor_criterion
        self.adversary_model = adversary_model
        self.adversary_criterion = adversary_criterion
        # TODO check validity

    def train_initializer(self, learning_rate = 0.01, alpha = 0.1):
        self.alpha = 0.1
        if self.torch:
            self.predictor_optimizer = torch.optim.Adam(self.predictor_model.parameters(), lr=learning_rate)
            self.adversary_optimizer = torch.optim.Adam(self.adversary_model.parameters(), lr=learning_rate)

    def train(self, X, Y, Z, 
            epochs, 
            batch_size,
            learning_rate = 0.01,
            alpha = 0.1
    ):
        self.train_initializer()
        for epoch in epochs:
            self._train_step_torch()

    def _train_step_torch(self, X, Y, Z):
        self.predictor_model.train()
        self.adversary_model.train()

        # # Totally unnecessary for this small problem but pc feels less wasteful then
        # if torch.cuda.is_available():
        #     model.cuda()

        for epoch in range(epochs):
            # Clear gradient
            self.predictor_optimizer.zero_grad()
            self.adversary_optimizer.zero_grad()

            Y_hat = self.predictor_model(X)
            LP = self.predictor_criterion(Y_hat, Y)
            LP.backward(retain_graph=True) # Check what this does at some point in time
        
            dW_LP = [deepcopy(p.grad) for p in self.predictor_model.parameters()]

            self.predictor_optimizer.zero_grad()
            self.adversary_optimizer.zero_grad()
            Z_hat = self.adversary_model(Y_hat)
            LA = self.adversary_criterion(Z_hat, Z)
            LA.backward()

            dW_LA = [p.grad for p in predictor.parameters()]

            for i, p in enumerate(predictor.parameters()):
                # Normalize dW_LA
                unit_dW_LA = dW_LA[i] / (torch.norm(dW_LA[i]) + torch.finfo(float).tiny)
                # Project
                proj = torch.sum(torch.inner(dW_LP[i], unit_dW_LA))
                # Calculate dW
                p.grad = dW_LP[i] - (proj * unit_dW_LA) - (alpha * dW_LA[i])

            predictor_optimizer.step()
            adversary_optimizer.step()

class regressor_softmax(torch.nn.Module):
    def __init__(self, N_in, N_out):
        super(regressor_softmax, self).__init__()
        self.layers = torch.nn.Sequential(        
            torch.nn.Linear(N_in, N_out)
        )
        self.act = torch.nn.Softmax(dim=1) # Or softmax
    
    def forward(self, x):
        x = self.layers(x)
        return self.act(x)

predictor = regressor_softmax(2, 2) # Output \hat y as softmax probability
adversary = regressor_softmax(2, 2) # Take in \hat y as softmax probability

predictor_criterion = torch.nn.NLLLoss()
adversary_criterion = torch.nn.NLLLoss()

test = AdversarialDebiasing(
        environment='torch', 
        predictor_model = predictor, 
        predictor_criterion = predictor_criterion,
        adversary_model = adversary,
        adversary_criterion = adversary_criterion)


X_train = torch.FloatTensor(df[["x1", "x2"]].values)
y_train = torch.LongTensor(df["y"].values)
z_train = torch.LongTensor(df["x1"].values)

test.train(X_train, Y_train, )

predictor.eval()
pred = predictor(X_train)
pred = torch.argmax(pred, dim=1).detach().numpy()
evaluate(pred)