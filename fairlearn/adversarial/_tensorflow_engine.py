from ._backend_engine import BackendEngine
from ._util import Keyword
from ._constants import _KWARG_ERROR_MESSAGE

from numpy import finfo, float32

# dynamic imports.
tensorflow = None


class TensorflowEngine(BackendEngine):
    """Adds TensorFlow specific functions."""

    def __init__(self, base, X, Y, Z):
        global tensorflow
        import tensorflow

        self.model_class = tensorflow.keras.Model
        super(TensorflowEngine, self).__init__(base, X, Y, Z)

        self.setup_optimizer()

    def evaluate(self, X):
        """
        Evaluate the model given input `X`.

        Feed 2d `numpy.ndarray` through model and receive output as
        2d `numpy.ndarray`.
        """
        Y_pred = self.predictor_model(X, training=False)
        Y_pred = Y_pred.numpy()
        return Y_pred

    def train_step(self, X, Y, Z):
        """
        Perform one training step over data in TensorFlow models.

        Returns
        -------
        (LP, LA) : tuple of (float, float)
            predictor loss and adversary loss.
        """
        with tensorflow.GradientTape(persistent=True) as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            Y_hat = self.predictor_model(X, training=True)

            LP = self.predictor_loss(Y, Y_hat)

            # For equalized odds
            if self.base.pass_y_:
                Y_hat = tensorflow.concat((Y_hat, Y), axis=1)

            Z_hat = self.adversary_model(Y_hat)

            LA = self.adversary_loss(Z, Z_hat)

        dW_LP = tape.gradient(LP, self.predictor_model.trainable_variables)
        dU_LA = tape.gradient(LA, self.adversary_model.trainable_variables)
        dW_LA = tape.gradient(LA, self.predictor_model.trainable_variables)

        del tape  # Because persistent=True !

        for i in range(len(dW_LP)):
            # Normalize dW_LA
            unit_dW_LA = dW_LA[i] / (
                tensorflow.norm(dW_LA[i]) + finfo(float32).tiny
            )
            # Project
            proj = tensorflow.reduce_sum(
                tensorflow.multiply(dW_LP[i], unit_dW_LA)
            )
            # Calculate dW
            dW_LP[i] = (
                dW_LP[i] - (proj * unit_dW_LA) - (self.base.alpha * dW_LA[i])
            )

        self.predictor_optimizer.apply_gradients(
            zip(dW_LP, self.predictor_model.trainable_variables)
        )
        self.adversary_optimizer.apply_gradients(
            zip(dU_LA, self.adversary_model.trainable_variables)
        )

        return (LP.numpy().item(), LA.numpy().item())

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
                learning_rate=self.base.learning_rate
            )
        else:
            self.predictor_optimizer = self.base.predictor_optimizer

        if isinstance(self.base.adversary_optimizer, str):
            optim = self.get_optimizer(
                self.base.adversary_optimizer, "adversary_optimizer"
            )
            self.adversary_optimizer = optim(
                learning_rate=self.base.learning_rate
            )
        else:
            self.adversary_optimizer = self.base.adversary_optimizer

    def get_optimizer(self, optimizer, keyword_name):
        """Get the optimizer base class corresponding to the string name.

        The parameter `optimizer` should be a string that tells us which optimizer
        to use."""
        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                return tensorflow.keras.optimizers.Adam
            elif optimizer.lower() == "sgd":
                return tensorflow.keras.optimizers.SGD
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
            # Use sigmoid as final layer
            return tensorflow.keras.losses.BinaryCrossentropy(from_logits=False)
        elif keyword == Keyword.CATEGORY:
            # User softmax as final layer
            return tensorflow.keras.losses.CategoricalCrossentropy(
                from_logits=False
            )
        elif keyword == Keyword.CONTINUOUS:
            return tensorflow.keras.losses.MeanSquaredError()

    def get_model(self, list_nodes):
        from ._models import getTensorflowModel as getModel

        return getModel(list_nodes=list_nodes)
