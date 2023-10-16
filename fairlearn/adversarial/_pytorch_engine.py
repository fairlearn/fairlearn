# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ._backend_engine import BackendEngine
from ._constants import (
    _MODEL_UNRECOGNIZED_STR,
    _MODEL_UNRECOGNIZED_ITEM,
)

# dynamic import.
torch = None


class PytorchEngine(BackendEngine):
    """Adds PyTorch specific functions."""

    def __init__(self, base, X, Y, A):
        """
        Initialize the (Pytorch specific parts) of the backend engine.

        The Pytorch-specifics include setting module class and handling Cuda.
        Also set up the optimizers after the init!
        """
        global torch
        import torch

        torch.manual_seed(base.random_state_.random())

        self.model_class = torch.nn.Module
        self.optim_class = torch.optim.Optimizer
        super(PytorchEngine, self).__init__(base, X, Y, A)

    def __move_model__(self):
        """Move model to CUDA."""
        # Setup cuda - do this before setting up optimizers!
        if not self.base.cuda:
            self.cuda = False
        elif self.base.cuda:
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available")
            self.cuda = True
            self.device = torch.device(self.base.cuda)

        if self.cuda:
            self.adversary_model = self.adversary_model.to(self.device)
            self.predictor_model = self.predictor_model.to(self.device)

    def shuffle(self, X, Y, A):
        """Override base's shuffle to work with `torch.FloatTensor`."""
        idx = torch.randperm(X.shape[0])
        X = X[idx].view(X.size())
        Y = Y[idx].view(Y.size())
        A = A[idx].view(A.size())
        return X, Y, A

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

    def train_step(self, X, Y, A):
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
        LP.backward(retain_graph=True)  # Check what this does at some point in time

        dW_LP = [
            torch.clone(p.grad.detach()) for p in self.predictor_model.parameters()
        ]

        self.predictor_optimizer.zero_grad()
        self.adversary_optimizer.zero_grad()

        # For equalized odds
        if self.base.pass_y_:
            Y_hat = torch.cat((Y_hat, Y), dim=1)

        A_hat = self.adversary_model(Y_hat)
        LA = self.adversary_loss(A_hat, A)
        LA.backward()

        dW_LA = [
            torch.clone(p.grad.detach()) for p in self.predictor_model.parameters()
        ]

        for i, p in enumerate(self.predictor_model.parameters()):
            # Normalize dW_LA
            unit_dW_LA = dW_LA[i] / (torch.norm(dW_LA[i]) + torch.finfo(float).tiny)
            # Project
            proj = torch.sum(torch.inner(unit_dW_LA, dW_LP[i]))
            # Calculate dW
            p.grad = dW_LP[i] - (proj * unit_dW_LA) - (self.base.alpha * dW_LA[i])

        self.predictor_optimizer.step()
        self.adversary_optimizer.step()

        return (LP.item(), LA.item())

    def get_optimizer(self, optim_param, model):
        """Get an optimizer instance corresponding to the string name."""
        optim = None
        if isinstance(optim_param, str):
            if optim_param.lower() == "adam":
                optim = torch.optim.Adam
            elif optim_param.lower() == "sgd":
                optim = torch.optim.SGD
        if optim is not None:
            return optim(model.parameters(), lr=self.base.learning_rate)

    def get_loss(self, dist_type):
        """Get loss function corresponding to the keyword."""
        if dist_type == "binary":
            # Use sigmoid as last layer
            return torch.nn.BCELoss(reduction="mean")
        elif dist_type == "category":
            # Use logsoftmax as last layer
            return torch.nn.CrossEntropyLoss(reduction="mean")
        elif dist_type == "continuous":
            return torch.nn.MSELoss(reduction="mean")
        super(PytorchEngine, self).get_loss(dist_type)

    def get_model(self, list_nodes):
        """
        Build a model from a list of keywords.

        A BackendEngine should implement get_model in order to
        simplify the user's work. In particular, we will adhere
        to the following API where list_nodes is a list of neural network
        layers.

        Parameters
        ----------
        list_nodes: list
            list of keywords. Integer keywords indicate a layer with
            a number of nodes.
            Callable keywords are added to the model as a layer directly,
            which is useful for activation functions. String keywords are
            not supported in the Pytorch backend (try tensorflow instead).

        Returns
        -------
        model : torch.nn.Module
            initialized model with layers as specified.
        """

        class FullyConnected(torch.nn.Module):
            """Neural network class."""

            def __init__(self):
                """Initialize the layers of the NN."""
                super(FullyConnected, self).__init__()
                layers = []
                nodes = None
                for i, item in enumerate(list_nodes):
                    if isinstance(item, int):
                        if nodes:
                            layers.append(torch.nn.Linear(nodes, list_nodes[i]))
                        nodes = item
                    elif callable(item):
                        layers.append(item)
                    elif isinstance(item, str):
                        if item.lower() == "sigmoid":
                            layers.append(torch.nn.Sigmoid())
                        elif item.lower() == "softmax":
                            layers.append(torch.nn.Softmax())
                        elif item.lower() == "relu":
                            layers.append(torch.nn.ReLU())
                        elif item.lower() == "leaky_relu":
                            layers.append(torch.nn.LeakyReLU())
                        else:
                            raise ValueError(_MODEL_UNRECOGNIZED_STR.format(item))
                        # TODO support more strings? Or better option?
                        # possibly gather all activation classes, get __name__,
                        # and do pattern matching.
                    else:
                        raise ValueError(_MODEL_UNRECOGNIZED_ITEM.format(item))
                self.layers_ = torch.nn.ModuleList(layers)

            def forward(self, x):
                """Propagate x through the network."""
                for layer in self.layers_:
                    x = layer(x)
                return x

        model = FullyConnected()

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        model.apply(init_weights)

        return model

    def validate_input(self, X, Y, A):
        """Extend the base `_validate_input` to send data to GPU when required."""
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        A = torch.from_numpy(A).float()

        if self.cuda:
            X = X.to(self.device)
            Y = Y.to(self.device)
            A = A.to(self.device)

        return X, Y, A
