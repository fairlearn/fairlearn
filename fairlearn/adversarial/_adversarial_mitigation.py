# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ._constants import _IMPORT_ERROR_MESSAGE, _KWARG_ERROR_MESSAGE
from copy import deepcopy
from numpy import finfo, float32, ndarray
from sklearn.utils import shuffle as sklearn_shuffle
from math import ceil

torch = None
tf = None


class AdversarialMitigation():
    r"""Inprocessing algorithm to mitigate biases using PyTorch or Tensorflow.

    This algorithm is our implementation of work in `"Mitigating Unwanted Biases with
    Adversarial Learning" <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_.
    This algorithm takes as input two
    models, a predictor and an adversarial, defined either as a `PyTorch module
    <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ or
    `Tensorflow2 model
    <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_. You train this
    predictor using an API that is similar to estimators in `sklearn`.


    Parameters
    ----------
    environment : str, default = \"torch\"
        The machine learning library to use. Value must be either
        \"torch\" or \"tensorflow\" which indicate PyTorch or Tensorflow 2
        respectively. If none is specified, default is torch, else tensorflow,
        depending on which is installed.

    predictor_model : torch.nn.Module, tensorflow.keras.Model
        The predictor model to train. If :code:`environment = \"torch\"`, we
        expect a `torch.nn.Module`. If :code:`environment = \"tensorflow\"`, we
        expect a `tensorflow.keras.Model`

    constraints : str, default = \"demographic_parity\"
        The fairness measure to optimize for. Must be either \"demographic_parity\"
        (Demographic Parity) or \"equalized_odds\" (Equalized Odds).

    learning_rate : float, default = 0.01
        A small number greater than zero to set as initial learning rate

    alpha : float, default = 0.1
        A small number $\alpha$ as specified in the paper.

    cuda : bool, default = False
        A boolean to indicate whether we can use cuda:0 (first GPU) when training
        a PyTorch model.

    """

    # TODO binary/continuous?
    # TODO more variables (also specify dimensionality!)

    def __init__(self, *,
                 environment='any',
                 predictor_model,
                 constraints='DP',
                 learning_rate=0.01,
                 alpha=0.1,
                 cuda=False
                 ):
        """Initialize Adversarial Mitigation."""
        self._setup_environment(environment)
        self._setup_predictor_model(predictor_model)
        self._setup_constraints(constraints)
        self._setup(learning_rate)
        self.alpha = alpha
        self._setup_cuda(cuda)

    def fit(self, X, y, *, sensitive_feature,
            epochs=1000,
            batch_size=-1,
            shuffle=False):
        """
        Fit the model based on the given training data and sensitive features.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional numpy array containing training data

        y : numpy.ndarray
            One-dimensional numpy array containing training targets

        sensitive_feature : numpy.ndarray
            One-dimensional numpy array containing the sensitive feature of the
            training data.

        epochs : int, default = 1000
            Maximum number of epochs to train for. Default is 1000. # TODO

        batch_size : int, default = -1
            Batch size. For no batching, set this to -1.

        shuffle : bool, default = False
            Iff True, shuffle the data after every iteration. Default is False
        """
        X, Y, Z = self._validate_input(X, y, sensitive_feature)
        # TODO decreasing learning rate: not really necessary with adam
        # TODO stopping condition!? If |grad| < eps
        if batch_size == -1:
            batch_size = X.shape[0]
        batches = ceil(X.shape[0] / batch_size)
        for epoch in range(epochs):
            for batch in range(batches):
                batch_slice = slice(batch * batch_size, min((batch + 1) * batch_size, X.shape[0]))
                self._train_step(X[batch_slice],
                                 Y[batch_slice],
                                 Z[batch_slice])
            if shuffle and epoch != epochs - 1:
                X, Y, Z = self._shuffle(X, Y, Z)

    def _shuffle(self, X, Y, Z):
        """Shuffle the rows of X, Y, Z."""
        if self.torch:
            idx = torch.randperm(X.shape[0])
            X = X[idx].view(X.size())
            Y = Y[idx].view(Y.size())
            Z = Z[idx].view(Z.size())
        elif self.tensorflow:
            X, Y, Z = sklearn_shuffle(X, Y, Z)
        return X, Y, Z

    def partial_fit(self, X, y, *, sensitive_feature):
        """
        Perform one epoch on given samples and update model.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional numpy array containing training data

        y : numpy.ndarray
            One-dimensional numpy array containing training targets

        sensitive_feature : numpy.ndarray
            One-dimensional numpy array containing the sensitive feature of the
            training data.
        """
        X, Y, Z = self._validate_input(X, y, sensitive_feature)
        self._train_step(X, Y, Z)

    def predict(self, X):
        """
        Gather predictions for given test data.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional numpy array containing test data

        Returns
        -------
        y_pred : numpy.ndarray
            One-dimensional numpy array containing discrete predictions for
            given X
        """
        if (not isinstance(X, ndarray)):
            raise ValueError(_KWARG_ERROR_MESSAGE.format("X", "a numpy array"))

        # Check dimensionality
        if (not len(X.shape) == 2):
            raise ValueError(_KWARG_ERROR_MESSAGE.format("X", "2-dimensional"))

        if self.torch:
            self.predictor_model.eval()
            X = torch.from_numpy(X).float()
            if self.cuda:
                X = X.to(self.device)
            with torch.no_grad():
                y_pred = self.predictor_model(X)
            if self.cuda:
                y_pred = y_pred.detach().cpu().numpy()
            else:
                y_pred = y_pred.numpy()
        elif self.tensorflow:
            X = X.astype(float32)
            # TODO

        y_pred = y_pred.reshape(-1)
        y_pred = (y_pred >= 0).astype(float)
        return y_pred

    def _train_step(self, X, Y, Z):
        """Call the implementation-specific train_step."""
        if self.torch:
            self._train_step_torch(X, Y, Z)
        elif self.tensorflow:
            self._train_step_tensorflow(X, Y, Z)

    def _train_step_torch(self, X, Y, Z):
        """Perform one training step over data in pytorch model."""
        self.predictor_model.train()
        self.adversary_model.train()

        # Clear gradient
        self.predictor_optimizer.zero_grad()
        self.adversary_optimizer.zero_grad()

        Y_hat = self.predictor_model(X)
        LP = self.predictor_criterion(Y_hat, Y)
        LP.backward(retain_graph=True)  # Check what this does at some point in time

        dW_LP = [deepcopy(p.grad) for p in self.predictor_model.parameters()]

        self.predictor_optimizer.zero_grad()
        self.adversary_optimizer.zero_grad()

        # For equalized odds
        if self.pass_y:
            Y_hat = torch.cat((Y_hat, Y), dim=1)

        Z_hat = self.adversary_model(Y_hat)
        LA = self.adversary_criterion(Z_hat, Z)
        LA.backward()

        dW_LA = [p.grad for p in self.predictor_model.parameters()]

        for i, p in enumerate(self.predictor_model.parameters()):
            # Normalize dW_LA
            unit_dW_LA = dW_LA[i] / (torch.norm(dW_LA[i]) + torch.finfo(float).tiny)
            # Project
            proj = torch.sum(torch.inner(dW_LP[i], unit_dW_LA))
            # Calculate dW
            p.grad = dW_LP[i] - (proj * unit_dW_LA) - (self.alpha * dW_LA[i])

        self.predictor_optimizer.step()
        self.adversary_optimizer.step()

    def _train_step_tensorflow(self, X, Y, Z):
        """Perform one training step over data in tensorflow model."""
        with tf.GradientTape(persistent=True) as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            Y_hat = self.predictor(X, training=True)
            LP = tf.reduce_mean(self.predictor_criterion(Y_hat, Y))

            # For equalized odds
            if self.pass_y:
                Y_hat = tf.concat((Y_hat, Y), axis=1)

            Z_hat = self.adversary(Y_hat)
            LA = tf.reduce_mean(self.adversary_criterion(Z_hat, Z))

        dW_LP = tape.gradient(LP, self.predictor_model.trainable_variables)
        dU_LA = tape.gradient(LA, self.adversary_model.trainable_variables)
        dW_LA = tape.gradient(LA, self.predictor_model.trainable_variables)

        del tape  # Because persistent=True !

        for i in range(len(dW_LP)):
            # Normalize dW_LA
            unit_dW_LA = dW_LA[i] / (tf.norm(dW_LA[i]) + finfo(float32).tiny)
            # Project
            proj = tf.reduce_sum(tf.multiply(dW_LP[i], unit_dW_LA))
            # Calculate dW
            dW_LP[i] = dW_LP[i] - (proj * unit_dW_LA) - (self.alpha * dW_LA[i])

        self.predictor_optimizer.apply_gradients(
            zip(dW_LP, self.predictor_model.trainable_variables))
        self.adversary_optimizer.apply_gradients(
            zip(dU_LA, self.adversary_model.trainable_variables))

    def _validate_input(self, X, Y, Z):
        """Validate the input data."""
        # Check that data are numpy arrays
        for var, name in [(X, "X"), (Y, "y"), (Z, "sensitive_feature")]:
            if (not isinstance(var, ndarray)):
                raise ValueError(_KWARG_ERROR_MESSAGE.format(name, "a numpy array"))

        # Check for equal number of samples
        if not (X.shape[0] == Y.shape[0] and X.shape[0] == Z.shape[0]):
            raise ValueError("Input data has an ambiguous number of rows")

        # Check dimensionality
        for var, name, dim in [(X, "X", 2), (Y, "y", 1), (Z, "sensitive_feature", 1)]:
            if (not len(var.shape) == dim):
                raise ValueError(_KWARG_ERROR_MESSAGE.format(name, str(dim) + "-dimensional"))

        # Reshape to 2-D
        Y = Y.reshape(-1, 1)
        Z = Z.reshape(-1, 1)

        if self.torch:
            X = torch.from_numpy(X).float()
            Y = torch.from_numpy(Y).float()
            Z = torch.from_numpy(Z).float()
            if self.cuda:
                X = X.to(self.device)
                Y = Y.to(self.device)
                Z = Z.to(self.device)
        elif self.tensorflow:
            # TODO also y?
            X = X.astype(float32)
            Y = Y.astype(float32)
            Z = Z.astype(float32)

        # TODO Validate Z is binary? Possibly Y continuous?
        return X, Y, Z

    def _setup_environment(self, environment):
        """Import either PyTorch or Tensorflow."""
        self.torch = False
        self.tensorflow = False
        # If no environment is passed, try to select torch or tensorflow
        global torch
        global tf
        if environment == "any":
            try:
                import torch
                environment = "torch"
                self.torch = True
            except ImportError:
                pass
            if not (environment == 'torch'):
                try:
                    import tensorflow as tf
                    environment = "tensorflow"
                    self.tensorflow = True
                except ImportError:
                    pass
        elif environment == 'torch':
            try:
                import torch
            except ImportError:
                raise RuntimeError(_IMPORT_ERROR_MESSAGE.format('torch'))
            self.torch = True
        elif environment == 'tensorflow':
            try:
                import tensorflow as tf
            except ImportError:
                raise RuntimeError(_IMPORT_ERROR_MESSAGE.format('tensorflow'))
            self.tensorflow = True

        if not (self.torch or self.tensorflow):
            raise ValueError(_KWARG_ERROR_MESSAGE.format(
                "environment", "one of \\[\'torch\',\'tensorflow\'\\]"))

    def _setup_predictor_model(self, predictor_model):
        """Verify the type of predictor matches the environment."""
        if self.torch:
            if not isinstance(predictor_model, torch.nn.Module):
                raise ValueError(_KWARG_ERROR_MESSAGE.format(
                    "predictor_model", "a `torch.nn.Module`"
                ))
        else:
            if not isinstance(predictor_model, tf.keras.Model):
                raise ValueError(_KWARG_ERROR_MESSAGE.format(
                    "predictor_model", "a `tensorflow.keras.Model`"
                ))
        self.predictor_model = predictor_model

    def _setup(self, learning_rate):
        """Initialize the optimizers depending on which environment is used."""
        if self.torch:
            self.predictor_optimizer = torch.optim.Adam(
                self.predictor_model.parameters(), lr=learning_rate)
            self.adversary_optimizer = torch.optim.Adam(
                self.adversary_model.parameters(), lr=learning_rate)
        elif self.tensorflow:
            self.predictor_optimizer = tf.train.AdamOptimizer(learning_rate)
            self.adversary_optimizer = tf.train.AdamOptimizer(learning_rate)

    def _setup_constraints(self, constraints):
        """Verify the constraints and set up the corresponding network structure."""
        if (constraints == "demographic_parity"):
            self.pass_y = False
        elif (constraints == "equalized_odds"):
            self.pass_y = True
        else:
            raise ValueError(_KWARG_ERROR_MESSAGE.format(
                "constraints", "one of \\[\'DP\',\'EO\'\\]"))

        y_nodes = 1  # y always 1!
        adversarial_in = y_nodes * (2 if self.pass_y else 1)
        z_nodes = 1  # TODO think about multiples sensitive values

        y_binary = True  # TODO continuous case... with MSE loss?
        z_binary = True

        if self.torch:
            from ._pytorch_models import FullyConnected
            self.adversary_model = FullyConnected(adversarial_in, [y_nodes, z_nodes])
            if y_binary:
                self.predictor_criterion = torch.nn.BCEWithLogitsLoss()
            else:
                pass  # TODO continuous case
            if z_binary:
                self.adversary_criterion = torch.nn.BCEWithLogitsLoss()
            else:
                pass  # TODO
        elif self.tensorflow:
            from ._tensorflow_models import FullyConnected
            self.adversary_model = FullyConnected(adversarial_in, [y_nodes, z_nodes])
            if y_binary:
                self.predictor_criterion = tf.losses.BinaryCrossentropy(from_logits=True)
            if z_binary:
                self.adversary_criterion = tf.losses.BinaryCrossentropy(from_logits=True)

    def _setup_cuda(self, cuda):
        """Verify whether we can use the GPU and move pytorch model to it."""
        if (not cuda) or (not self.torch):
            self.cuda = False
        elif (cuda):
            if (not self.torch):
                raise ValueError("Can't use cuda with tensorflow")
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available")
            self.cuda = True
            self.device = torch.device("cuda:0")
            self.adversary_model = self.adversary_model.to(self.device)
            self.predictor_model = self.predictor_model.to(self.device)
