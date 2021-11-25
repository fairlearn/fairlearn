from ._util import interpret_keyword
from ._constants import _KWARG_ERROR_MESSAGE, _NOT_IMPLEMENTED
from sklearn.utils import shuffle
from numpy import ndarray


class BackendEngine:
    """The interface of a mixin class."""

    model_class = None

    def __init__(self, base, X, Y, Z):
        self.base = base

        # Set up predictor_model
        if isinstance(base.predictor_model, list):
            predictor_list_nodes = (
                [X.shape[1]] + base.predictor_model + [Y.shape[1]]
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
            adversarial_in = Y.shape[1] * (2 if base.pass_y_ else 1)
            adversary_list_nodes = (
                [adversarial_in] + base.adversary_model + [Z.shape[1]]
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
        self.predictor_loss = self.get_loss(
            interpret_keyword(Y, base.predictor_loss, "predictor_loss", "y")
        )
        self.adversary_loss = self.get_loss(
            interpret_keyword(
                Z, base.adversary_loss, "adversary_loss", "sensitive_features"
            )
        )

    def shuffle(self, X, Y, Z):
        """Shuffle the rows of X, Y, Z.

        Needs to be overriden by backends that do not work with sklearn."""
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

    def get_loss(self, keyword):
        """Infer loss from keyword or callable."""
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def get_model(self, list_nodes):
        raise NotImplementedError(_NOT_IMPLEMENTED)
