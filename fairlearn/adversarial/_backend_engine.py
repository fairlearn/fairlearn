# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ._constants import (
    _KWARG_ERROR_MESSAGE,
    _NOT_IMPLEMENTED,
    _DIST_TYPE_NOT_IMPLEMENTED,
    _NO_LOSS,
)
from sklearn.utils import shuffle
from numpy import ndarray


class BackendEngine:
    """The interface of a mixin class."""

    model_class = None

    def __init__(self, base, X, Y, Z):
        """
        Initialize the generic parts of the backend engine.

        Every subclass of BackendEngine needs to import its requirements, set
        self.model_class to the type that predictor_model and adversary_model
        should be, and set up the optimizers after the init. The generic
        steps that are performed in this method is to type-check the models
        and set up the losses.
        """
        self.base = base

        n_X_features = X.shape[1]
        n_Y_features = base.y_transform_.n_features_out_
        n_Z_features = base.z_transform_.n_features_out_
        # Set up predictor_model
        if isinstance(base.predictor_model, list):
            predictor_list_nodes = (
                [n_X_features] + base.predictor_model + [n_Y_features]
            )
            self.predictor_model = self.get_model(
                list_nodes=predictor_list_nodes
            )
        elif issubclass(type(base.predictor_model), self.model_class):
            self.predictor_model = base.predictor_model
        else:
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format(
                    "predictor_model",
                    "a list or a '{}'".format(self.model_class),
                )
            )

        # Set up adversary_model
        if isinstance(base.adversary_model, list):
            adversarial_in = n_Y_features * (2 if base.pass_y_ else 1)
            adversary_list_nodes = (
                [adversarial_in] + base.adversary_model + [n_Z_features]
            )
            self.adversary_model = self.get_model(
                list_nodes=adversary_list_nodes
            )
        elif issubclass(type(base.adversary_model), self.model_class):
            self.adversary_model = base.adversary_model
        else:
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format(
                    "adversary_model",
                    "a list or a '{}'".format(self.model_class),
                )
            )

        # Set up losses
        if callable(base.predictor_loss):
            self.predictor_loss = base.predictor_loss
        elif hasattr(base.y_transform_, "dist_type_"):
            self.predictor_loss = self.get_loss(base.y_transform_.dist_type_)
        else:
            raise ValueError(_NO_LOSS.format("predictor_loss"))

        if callable(base.adversary_loss):
            self.adversary_loss = base.adversary_loss
        elif hasattr(base.z_transform_, "dist_type_"):
            self.adversary_loss = self.get_loss(base.z_transform_.dist_type_)
        else:
            raise ValueError(_NO_LOSS.format("adversary_loss"))

    def shuffle(self, X, Y, Z):
        """
        Shuffle the rows of X, Y, Z.

        Needs to be overriden by backends that are non-compatible such as torch.
        """
        X, Y, Z = shuffle(X, Y, Z, random_state=self.base.random_state_)
        return X, Y, Z

    def evaluate(self, X: ndarray) -> ndarray:
        """
        Evaluate the model given input `X`.

        Feed 2d `numpy.ndarray` through model and receive output as
        2d `numpy.ndarray`.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def train_step(self, X: ndarray, Y: ndarray, Z: ndarray):
        """
        Perform one training step over data.

        Returns
        -------
        (LP, LA) : tuple of (float, float)
            predictor loss and adversary loss.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def setup_optimizer(self):
        """Create the predictor_optimizer and adversary_optimizer here."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def get_loss(self, dist_type):
        """
        Infer loss from keyword or callable.

        This is the base function which can be used as fall-back for subclass
        methods.
        """
        raise ValueError(
            _DIST_TYPE_NOT_IMPLEMENTED.format(
                self.__class__.__name__, dist_type
            )
        )

    def get_model(self, list_nodes):
        """Get the model."""  # TODO specify what kind of models these should be
        raise NotImplementedError(_NOT_IMPLEMENTED)
