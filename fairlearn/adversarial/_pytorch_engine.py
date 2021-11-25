# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ._backend_engine import BackendEngine
from ._util import Keyword
from ._constants import _KWARG_ERROR_MESSAGE

# dynamic import.
torch = None


class PytorchEngine(BackendEngine):
    """Adds PyTorch specific functions."""

    def __init__(self, base, X, Y, Z):
        """
        Initialize the (Pytorch specific parts) of the backend engine.

        The Pytorch-specifics include setting module class and handling Cuda.
        Also set up the optimizers after the init!
        """
        global torch
        import torch

        self.model_class = torch.nn.Module
        super(PytorchEngine, self).__init__(base, X, Y, Z)

        # Setup cuda - Recommended to do this before setting up optimizers!
        if not base.cuda:
            self.cuda = False
        elif base.cuda:
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available")
            self.cuda = True
            self.device = torch.device(base.cuda)

        if self.cuda:
            base.adversary_model_ = base.adversary_model_.to(self.device)
            base.predictor_model_ = base.predictor_model_.to(self.device)

        self.setup_optimizer()

    def shuffle(self, X, Y, Z):
        """Override base's shuffle to work with `torch.FloatTensor`."""
        idx = torch.randperm(X.shape[0])
        X = X[idx].view(X.size())
        Y = Y[idx].view(Y.size())
        Z = Z[idx].view(Z.size())
        return X, Y, Z

    def evaluate(self, X):
        """
        Evaluate the model given input `X`.

        Feed 2d `numpy.ndarray` through model and receive output as
        2d `numpy.ndarray`.
        """
        self.predictor_model.eval()
        X = torch.from_numpy(X).float()
        if self.cuda:
            X = X.to(self.device)
        with torch.no_grad():
            Y_pred = self.predictor_model(X)
        if self.cuda:
            Y_pred = Y_pred.detach().cpu().numpy()
        else:
            Y_pred = Y_pred.numpy()
        return Y_pred

    def train_step(self, X, Y, Z):
        """
        Perform one training step over data in PyTorch models.

        Returns
        -------
        (LP, LA) : tuple of (float, float)
            predictor loss and adversary loss.
        """
        self.predictor_model.train()
        self.adversary_model.train()

        # Clear gradient
        self.predictor_optimizer.zero_grad()
        self.adversary_optimizer.zero_grad()

        Y_hat = self.predictor_model(X)
        LP = self.predictor_loss(Y_hat, Y)
        LP.backward(
            retain_graph=True
        )  # Check what this does at some point in time

        dW_LP = [
            torch.clone(p.grad.detach())
            for p in self.predictor_model.parameters()
        ]

        self.predictor_optimizer.zero_grad()
        self.adversary_optimizer.zero_grad()

        # For equalized odds
        if self.base.pass_y_:
            Y_hat = torch.cat((Y_hat, Y), dim=1)

        Z_hat = self.adversary_model(Y_hat)
        LA = self.adversary_loss(Z_hat, Z)
        LA.backward()

        dW_LA = [
            torch.clone(p.grad.detach())
            for p in self.predictor_model.parameters()
        ]

        for i, p in enumerate(self.predictor_model.parameters()):
            # Normalize dW_LA
            unit_dW_LA = dW_LA[i] / (
                torch.norm(dW_LA[i]) + torch.finfo(float).tiny
            )
            # Project
            proj = torch.sum(torch.inner(unit_dW_LA, dW_LP[i]))
            # Calculate dW
            p.grad = (
                dW_LP[i] - (proj * unit_dW_LA) - (self.base.alpha * dW_LA[i])
            )

        self.predictor_optimizer.step()
        self.adversary_optimizer.step()

        # EXTRA STEPS

        # self.predictor_model.eval()
        # Y_hat = self.predictor_model(X).detach()

        # if self.base.pass_y_:
        #     Y_hat = torch.cat((Y_hat, Y), dim=1)

        # for i in range(30):
        #     self.adversary_optimizer.zero_grad()

        #     Z_hat = self.adversary_model(Y_hat)
        #     LA = self.adversary_loss(Z_hat, Z)
        #     LA.backward()
        #     self.adversary_optimizer.step()

        return (LP.item(), LA.item())

    def setup_optimizer(self):
        """
        Create the optimizers for PyTorch.

        Setup predictor_optimizer and adversary_optimizer using the
        base.predictor_optimizer and base.adversary_optimizer given by the user.
        If the parameters given by the users are strings, we use get_optimizer
        to get the optimizer base class and initialize it with the lr parameter.
        If the parameter given by the user is not a string, assume it is an
        already initialized optimizer.
        """
        if isinstance(self.base.predictor_optimizer, str):
            optim = self.get_optimizer(
                self.base.predictor_optimizer, "predictor_optimizer"
            )
            self.predictor_optimizer = optim(
                self.predictor_model.parameters(), lr=self.base.learning_rate
            )
        else:
            self.predictor_optimizer = self.base.predictor_optimizer

        if isinstance(self.base.adversary_optimizer, str):
            optim = self.get_optimizer(
                self.base.adversary_optimizer, "adversary_optimizer"
            )
            self.adversary_optimizer = optim(
                self.adversary_model.parameters(), lr=self.base.learning_rate
            )
        else:
            self.adversary_optimizer = self.base.adversary_optimizer

    def get_optimizer(self, optimizer, keyword_name):
        """
        Get the optimizer base class corresponding to the string name.

        The parameter `optimizer` should be a string that tells us which optimizer
        to use.
        """
        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                return torch.optim.Adam
            elif optimizer.lower() == "sgd":
                return torch.optim.SGD
        raise ValueError(
            _KWARG_ERROR_MESSAGE.format(
                keyword_name,
                '"Adam" or "SGD" or an (!)initialized(!) optimizer',
            )
        )

    def get_loss(self, keyword):
        """Get loss function corresponding to the keyword."""
        if callable(keyword):
            return keyword
        if keyword == Keyword.BINARY:
            # Use sigmoid as last layer
            return torch.nn.BCELoss(reduction="mean")
        elif keyword == Keyword.CATEGORY:
            # Use logsoftmax as last layer
            return torch.nn.NLLLoss(reduction="mean")
        elif keyword == Keyword.CONTINUOUS:
            return torch.nn.MSELoss(reduction="mean")

    def get_model(self, list_nodes):
        """Get the model."""  # TODO move to this class.
        from ._models import getTorchModel as getModel

        return getModel(list_nodes=list_nodes)

    def validate_input(self, X, Y, Z):
        """Extend the base `_validate_input` to send data to GPU when required."""
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        Z = torch.from_numpy(Z).float()

        if self.cuda:
            X = X.to(self.device)
            Y = Y.to(self.device)
            Z = Z.to(self.device)

        return X, Y, Z
