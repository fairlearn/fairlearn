# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ._constants import (
    _KWARG_ERROR_MESSAGE,
    _NOT_IMPLEMENTED,
    _DIST_TYPE_NOT_IMPLEMENTED,
    _NO_LOSS,
    _LIST_MODEL_UNSUPPORTED,
)
from sklearn.utils import shuffle
from numpy import ndarray


class BackendEngine:
    """The interface of a mixin class."""

    # NOTE: to validate objects passed.
    # We don't validate loss yet.
    model_class = None
    optim_class = None

    def __init__(self, base, X, Y, A):
        """
        Initialize the generic parts of the backend engine.

        Every subclass of BackendEngine needs to import its requirements, set
        self.model_class to the type that predictor_model and adversary_model
        should be, and set up the optimizers after the init. The generic
        steps that are performed in this method is to type-check the models
        and set up the losses.
        """
        self.base = base

        n_X_features = X.shape[1]  # FIXME: what if X.ndim > 2?
        n_Y_features = base._y_transform.n_features_out_
        n_A_features = base._sf_transform.n_features_out_

        # Set up models
        if base.warm_start and hasattr(base, "backendEngine_"):
            self.predictor_model = base.backendEngine_.predictor_model
            self.adversary_model = base.backendEngine_.adversary_model
        else:
            self.predictor_model = self.__init_model__(
                base.predictor_model,
                base.predictor_loss_,
                n_X_features,
                n_Y_features,
                "predictor",
            )
            self.adversary_model = self.__init_model__(
                base.adversary_model,
                base.adversary_loss_,
                n_Y_features * (2 if base.pass_y_ else 1),
                n_A_features,
                "adversary",
            )

        if hasattr(self, "__move_model__"):
            self.__move_model__()

        # Set up losses
        self.predictor_loss = self.__init_loss__(base.predictor_loss_, "predictor")
        self.adversary_loss = self.__init_loss__(base.adversary_loss_, "adversary")

        # Set up optimizers
        self.predictor_optimizer = self.__init_optimizers__(
            base.predictor_optimizer, self.predictor_model, "predictor"
        )
        self.adversary_optimizer = self.__init_optimizers__(
            base.adversary_optimizer, self.adversary_model, "adversary"
        )

    def __init_model__(self, model_param, loss_param, X_features, y_features, name):
        """
        Get an initialized model.

        Parameters
        ----------
        model_param : list, self.model_class
            parameter that specifies the model
        loss_param
            loss function. If this is a str then it is auto-inferred and we
            can build a model here. If it is not a str then we cannot infer,
            and we will raise an error if model_param is a list
        X_features : int
            number of features in input variable
        y_features : int
            number of features in target variable
        name : str
            name of model, either "predictor" or "adversary"
        """
        if isinstance(model_param, list):
            if not isinstance(loss_param, str):
                # Can not parse model as list when loss is undefined
                raise ValueError(_LIST_MODEL_UNSUPPORTED.format(name, name))
            predictor_list_nodes = [X_features] + model_param + [y_features]
            if loss_param == "binary":
                predictor_list_nodes.append("sigmoid")
            elif loss_param == "category":
                predictor_list_nodes.append("softmax")

            return self.get_model(list_nodes=predictor_list_nodes)
        elif issubclass(type(model_param), self.model_class):
            return model_param
        else:
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format(
                    f"{name}_model", f"a list or a '{name}_model'"
                )
            )

    def __init_loss__(self, loss_param, name):
        """
        Get an initialized loss.

        Parameters
        ----------
        loss_param
            given loss. If auto-inferred, apply get_loss. Otherwise, use
            loss_param as loss function directly
        name : str
            name of model, either "predictor" or "adversary"
        """
        if callable(loss_param):
            return loss_param
        elif isinstance(loss_param, str):
            return self.get_loss(loss_param)
        else:
            raise ValueError(_NO_LOSS.format(f"{name}_loss"))

    def __init_optimizers__(self, optim_param, model, name):
        """
        Get an initialized optimizer.

        Parameters
        ----------
        optim_param
            Optimizer parameter. If a subclass instance of self.optim_class
            then we use this directly. If it is a callable then we call
            this (and pass the model through this call) and set the result
            of this call as the optimizer. If it is a string, we apply
            get_optimizer
        model
            initialized model
        name : str
            name of model, either "predictor" or "adversary"
        """
        # Preinitialized optimizer
        if issubclass(type(optim_param), self.optim_class):
            return optim_param
        # Optimizer constructor
        elif callable(optim_param):
            return optim_param(model)
        # Optimizer keyword
        elif isinstance(optim_param, str):
            got_optim = self.get_optimizer(optim_param, model)
            if got_optim is None:
                raise ValueError(
                    _KWARG_ERROR_MESSAGE.format(
                        f"{name}_optimizer",
                        (
                            f"a string that is supported by {self.__name__}, "
                            + "an already initialized optimizer (must "
                            + f"subclass {self.optim_class}), or a constructor "
                            + f"that takes as parameter the {name}_model and "
                            + "returns an initialized optimizer."
                        ),
                    )
                )
            else:
                return got_optim
        # Invalid optim_param
        raise ValueError(
            _KWARG_ERROR_MESSAGE.format(
                f"{name}_optimizer",
                (
                    "a string that indicates a certain optimizer, "
                    + "an already initialized optimizer (must subclass "
                    + f"{self.optim_class}), or a constructor that takes "
                    + f"as parameter the {name}_model and returns an "
                    + "initialized optimizer."
                ),
            )
        )

    def shuffle(self, X, Y, A):
        """
        Shuffle the rows of X, Y, A.

        Needs to be overriden by backends that are non-compatible such as torch.
        """
        X, Y, A = shuffle(X, Y, A, random_state=self.base.random_state_)
        return X, Y, A

    def evaluate(self, X: ndarray) -> ndarray:
        """
        Evaluate the model given input `X`.

        Feed 2d `numpy.ndarray` through model and receive output as
        2d `numpy.ndarray`.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def train_step(self, X: ndarray, Y: ndarray, A: ndarray):
        """
        Perform one training step over data.

        Returns
        -------
        (LP, LA) : tuple of (float, float)
            predictor loss and adversary loss.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def get_optimizer(self, optim_param, model):
        """Create the predictor_optimizer and adversary_optimizer here."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def get_loss(self, dist_type):
        """
        Infer loss from keyword or callable.

        This is the base function which can be used as fall-back for subclass
        methods.
        """
        raise ValueError(
            _DIST_TYPE_NOT_IMPLEMENTED.format(self.__class__.__name__, dist_type)
        )

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
            list of keywords.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED)
