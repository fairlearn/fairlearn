# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from math import ceil
from time import time

from sklearn.utils import check_scalar

from ._constants import (
    _IMPORT_ERROR_MESSAGE,
    _KWARG_ERROR_MESSAGE,
    _PREDICTION_FUNCTION_AMBIGUOUS,
    _PROGRESS_UPDATE,
    _CALLBACK_RETURNS_ERROR,
)
from ._backend_engine import BackendEngine
from ._pytorch_engine import PytorchEngine
from ._tensorflow_engine import TensorflowEngine
from ._preprocessor import (
    FloatTransformer,
    _get_type,
)
from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
    BaseEstimator,
    TransformerMixin,
)
from sklearn.utils.validation import (
    check_is_fitted,
    check_random_state,
    check_array,
)
from sklearn.exceptions import NotFittedError
from numpy import zeros, argmax, arange
import logging

logger = logging.getLogger(__name__)


class _AdversarialFairness(BaseEstimator):
    r"""Train PyTorch or TensorFlow predictors while mitigating unfairness .

    This model implements the supervised learning method proposed in
    `"Mitigating Unwanted Biases with Adversarial Learning"
    <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_ [1]_.
    This algorithm takes as input two neural network
    models, a predictor model and an adversarial model, defined either as a
    `PyTorch module
    <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ or
    `TensorFlow2 model
    <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_. You train this
    predictor using an API that is similar to estimators in `sklearn`.

    The predictor model takes the features :code:`X` as input and seeks
    to predict :code:`y`.
    For real-valued predictions of :code:`y`, regression takes place.
    For binary classification, the predictor model should return a single
    real-valued score, which is transformed into a probability of the
    positive class via the logistic function (aka sigmoid), similarly to
    logistic regression. For multi-class classification, the predictor
    model should return a vector of real-valued scores, which are
    transformed into class probabilities via the softmax function,
    similarly to multinomial logistic regression. The training loss is
    the negative log likelihood (aka log loss, logistic loss,
    cross-entropy loss). For regression, the squared error is taken, as
    it is proven to be equivalent to a scaled cross-entropy loss (under
    certain distribution assumptions).

    The adversarial model for demographic parity takes scores
    produced by the predictor model as input, and seeks to predict
    :code:`sensitive_features`. Depending on the type of the provided
    sensitive features, the model should produce a scalar
    or vector output. Three types of sensitive features are supported:
    (1) a single binary feature; (2) a single discrete feature; (3) one or
    multiple real-valued features. For a single binary sensitive feature
    and a single discrete feature, the network outputs are transformed
    by the logistic function and the softmax function, respectively, and
    the loss is the negative log likelihood. For one or multiple
    real-valued features, the network output is left as is, and the
    loss is a square loss.

    The adversarial model for equalized odds additionaly takes
    :code:`y` as input. For multi-class classification, :code:`y` is
    transformed using one-hot encoding.

    Parameters
    ----------
    backend : str, BackendEngine, default = 'auto'
        The backend to use. Must be one of :code:`'torch'`, :code:`'tensorflow'`,
        or :code:`'auto'` which indicates PyTorch, TensorFlow, or to
        automatically infer
        the backend from the :code:`predictor_model`.
        You can also pass in a BackendEngine class.

    predictor_model : list, torch.nn.Module, tensorflow.keras.Model
        The predictor model to train.
        Instead of a neural network model, it is possible to pass a list
        :math:`[k_1, k_2, \dots]`, where each :math:`k_i` either indicates
        the number of nodes (if :math:`k_i` is an integer) or
        an activation function (if :math:`k_i` is a string) or
        a layer or activation function instance directly (if :math:`k_i` is
        a callable). The default parameter is :code:`[]`, which indicates
        a neural network without any hidden layers.
        However, the number of nodes in the input
        and output layer are automatically inferred from data, and the final
        activation function (such as softmax for categorical
        predictors) are inferred from data.
        If :code:`backend` is specified, you cannot pass a model
        that uses a different backend.

    adversary_model : list, torch.nn.Module, tensorflow.keras.Model
        The adversary model to train. Defined similarly as :code:`predictor_model`.
        Must be the same type as the
        :code:`predictor_model`.

    predictor_optimizer : str, torch.optim, tensorflow.keras.optimizers, callable, default = 'Adam'
        The optimizer class to use. If a string is passed instead, this must be
        either 'SGD' or 'Adam'. A corresponding SGD or Adam optimizer is
        initialized with the given predictor model and learning rate.
        If an instance of a subclass of torch.optim.Optimizer
        or tensorflow.keras.optimizers.Optimizer is passed, this
        is used directly. If a callable :code:`fn` is passed, we call this
        callable and pass our model, and set the result of this call
        as the optimizer, so: :code:`predictor_optimizer=fn(predictor_model)`.

    adversary_optimizer : str, torch.optim, tensorflow.keras.optimizers, callable, default = 'Adam'
        The optimizer class to use. Defined similarly as
        :code:`predictor_optimizer`.

    constraints : str, default = 'demographic_parity'
        The fairness constraint. Must be either 'demographic_parity'
        or 'equalized_odds'.

    y_transform : str, sklearn.base.TransformerMixin, default = "auto"
        The preprocessor to use on the predictions :code:`y`.
        Either a keyword such as :code:`"auto"` or :code:`"one_hot_encoder",
        None, or a preprocessor. None indicates to pass-through data directly
        without preprocessing.
        The given transformer *must* map data
        to a 2d ndarray containing only floats.

    sf_transform : str, sklearn.base.TransformerMixin, default = "auto"
        The preprocessor to use on the :code:`sensitive_features`.
        Either a keyword such as :code:`"auto"` or :code:`"one_hot_encoder",
        None, or a preprocessor. None indicates to pass-through data directly
        without preprocessing.
        The given transformer *must* map data
        to a 2d ndarray containing only floats.

    learning_rate : float, default = 0.001
        A small number greater than zero to set as a learning rate.

    alpha : float, default = 1.0
        A small number :math:`\alpha` as specified in the paper. It
        is the factor that balances the training towards predicting :code:`y`
        (choose :math:`\alpha` closer to zero) or enforcing fairness constraint
        (choose larger :math:`\alpha`).

    epochs : int, default = 1
        Number of epochs to train for.

    batch_size : int, default = 32
        Batch size. For no batching, set this to -1.

    shuffle : bool, default = False
        When true, shuffle the data before every epoch (including the first).

    progress_updates : number, optional, default = None
        If a number :math:`t` is provided, we print an update
        about the training loop after processing a batch and :math:`t` seconds
        have passed since the previous update.

    skip_validation : bool, default = False
        Skip the validation of the data. Useful because validate_input is
        a costly operation, and we may instead pass all data to validate_input
        at an earlier stage. Note that not only checking :code:`X`
        is skipped, but also no tranform is applied to :code:`y` and
        :code:`sensitive_features`.

    callbacks : callable
        Callback function, called after every batch. For instance useable when
        wanting to validate. A list of callback functions can also be provided.
        Each callback function is passed two arguments :code:`self` (the
        estimator instance) and :code:`step` (the completed iteration), and
        may return a Boolean value. If the returned value is `True`, the
        optimization algorithm terminates. This can be used to implement
        *early stopping*.

    cuda : str, default = None
        A string to indicate which device to use when training. For instance,
        set :code:`cuda='cuda:0'` to train on the first GPU. Only for PyTorch
        backend.

    warm_start : bool, default = False
        Normally, when set to False, a call to :code:`fit()` triggers reinitialization,
        which discards the models and intializes them again. Setting to
        True triggers reuse of these models. Note: if pre-initialized models
        are passed, the models (and their parameters) are never discarded.

    random_state : int, RandomState, default = None
        Controls the randomized aspects of this algorithm, such as shuffling.
        Useful to get reproducible output across multiple function calls.

    References
    ----------
    .. [1] Zhang, Lemoine, Mitchell `"Mitigating Unwanted Biases with
       Adversarial Learning" <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_,
       AIES, 2018.

    """  # noqa : E501

    def __init__(
        self,
        *,
        backend="auto",
        predictor_model=[],  # [] is a NN with no hidden layers.
        adversary_model=[],
        predictor_loss="auto",
        adversary_loss="auto",
        predictor_function="auto",
        threshold_value=0.5,
        predictor_optimizer="Adam",
        adversary_optimizer="Adam",
        constraints="demographic_parity",
        y_transform="auto",
        sf_transform="auto",
        learning_rate=0.001,
        alpha=1.0,
        epochs=1,
        batch_size=32,
        max_iter=-1,
        shuffle=False,
        progress_updates=None,
        skip_validation=False,
        callbacks=None,
        cuda=None,
        warm_start=False,
        random_state=None,
    ):
        """Initialize class by only storing (kw)args, as per sklearn API."""
        self.backend = backend
        self.predictor_model = predictor_model
        self.adversary_model = adversary_model
        self.predictor_loss = predictor_loss
        self.adversary_loss = adversary_loss
        self.predictor_function = predictor_function
        self.threshold_value = threshold_value
        self.predictor_optimizer = predictor_optimizer
        self.adversary_optimizer = adversary_optimizer
        self.constraints = constraints
        self.y_transform = y_transform
        self.sf_transform = sf_transform
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.progress_updates = progress_updates
        self.skip_validation = skip_validation
        self.callbacks = callbacks
        self.cuda = cuda
        self.warm_start = warm_start
        self.random_state = random_state

    def __setup(self, X, Y, A):
        """
        Initialize the entire model from the parameters and the given data.

        Following sklearn API, we do not do intialization in `__init__`, but
        instead in `fit`. Firstly, we validate the backend. Then, we validate
        some key-word arguments. Then, we initialize the BackendEngine, which
        handles the initialization of the losses and optimizers. Among these
        steps, if a loss or function is not explicitely defined, we try to
        infer something appropriate from data (see interpret_keyword).
        """
        self._validate_backend()

        # Verify the constraints and set up the corresponding network structure.
        if self.constraints == "demographic_parity":
            self.pass_y_ = False
        elif self.constraints == "equalized_odds":
            self.pass_y_ = True
        else:
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format(
                    "constraints", "'demographic_parity' or 'equalized_odds'"
                )
            )

        # Numbers
        check_scalar(self.threshold_value, "threshold_value", (int, float))

        # Non-negative numbers
        for kw, kwname in (
            (self.learning_rate, "learning_rate"),
            (self.alpha, "alpha"),
            (self.progress_updates, "progress_updates"),
        ):
            if kw:
                check_scalar(
                    kw,
                    kwname,
                    (int, float),
                    min_val=0.0,
                    include_boundaries="left",
                )

        # Positive or -1 numbers
        for kw, kwname in (
            (self.batch_size, "batch_size"),
            (self.epochs, "epochs"),
            (self.max_iter, "max_iter"),
        ):
            check_scalar(
                kw, kwname, (int, float), min_val=-1, include_boundaries="left"
            )
            if kw <= 0.0 and kw != -1:
                raise ValueError(
                    _KWARG_ERROR_MESSAGE.format(kwname, "a positive number or -1")
                )

        for kw, kwname in (
            (self.shuffle, "shuffle"),
            (self.skip_validation, "skip_validation"),
            (self.warm_start, "warm_start"),
        ):
            if not isinstance(kw, bool):
                raise ValueError(_KWARG_ERROR_MESSAGE.format(kwname, "a boolean"))

        self.callbacks_ = None
        if self.callbacks:
            if not callable(self.callbacks):
                if not isinstance(self.callbacks, list):
                    raise ValueError(
                        _KWARG_ERROR_MESSAGE.format(
                            "callbacks", "a callable or list of callables"
                        )
                    )
                else:
                    for cb in self.callbacks:
                        if not callable(cb):
                            raise ValueError(
                                _KWARG_ERROR_MESSAGE.format(
                                    "callbacks",
                                    "a callable or list of callables",
                                )
                            )
                    self.callbacks_ = self.callbacks
            else:
                self.callbacks_ = [self.callbacks]

        def read_kw(data, kw_or_func, kwname):
            if isinstance(kw_or_func, str) or kw_or_func is None:
                # Possible remove some of these keywords in the future
                if kw_or_func in [
                    "auto",
                    "classification",
                    "binary",
                    "continuous",
                    "category",
                ]:
                    expected_dist = kw_or_func
                elif kw_or_func in [
                    "logistic_loss",
                    "cross_entropy_loss",
                    "log_loss",
                ]:
                    expected_dist = "classification"
                elif kw_or_func in [
                    "multinomial_logistic_loss",
                    "argmax",
                    "categorical_cross_entropy_loss",
                ]:
                    expected_dist = "category"
                elif kw_or_func in ["threshold"]:
                    expected_dist = "binary"
                elif kw_or_func in ["square_loss", None]:
                    expected_dist = "continuous"
                else:
                    raise ValueError(
                        _KWARG_ERROR_MESSAGE.format(
                            kwname,
                            "A valid keyword or a callable",
                        )
                    )
                return _get_type(data, expected_dist)
            else:
                return kw_or_func

        self.predictor_loss_ = read_kw(Y, self.predictor_loss, "predictor_loss")
        self.adversary_loss_ = read_kw(A, self.adversary_loss, "adversary_loss")
        self.predictor_function_ = read_kw(
            Y, self.predictor_function, "predictor_function"
        )

        for kw, kwname in (
            (self.y_transform, "y_transform"),
            (self.sf_transform, "sf_transform"),
        ):
            if not (
                (
                    issubclass(type(kw), TransformerMixin)
                    and hasattr(kw, "fit")
                    and hasattr(kw, "transform")
                )
                or kw is None
                or isinstance(kw, str)
            ):
                raise ValueError(
                    _KWARG_ERROR_MESSAGE.format(
                        kwname,
                        "a keyword or a sklearn Transformer"
                        + "(subclass TransformerMixin)",
                    )
                )

        self._y_transform = FloatTransformer(transformer=self.y_transform)
        self._sf_transform = FloatTransformer(transformer=self.sf_transform)
        self._y_transform.fit(Y)
        self._sf_transform.fit(A)

        if self.cuda and not isinstance(self.cuda, str):
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format(
                    "cuda",
                    'None, or the device to use (such as "cuda:0")',
                )
            )

        self.random_state_ = check_random_state(self.random_state)

        self._set_predictor_function()

        # Initialize backend
        # here, losses and optimizers are also set up.
        self.backendEngine_ = self.backend_(self, X, Y, A)

        # Sklearn-parameters
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = self._y_transform.n_features_in_

        self._is_setup = True

    def fit(self, X, y, *, sensitive_features=None):
        """
        Fit the model based on the given training data and sensitive features.

        Currently, for discrete y and sensitive_features
        *ALL* classes need to be passed in
        the first call to fit!

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional numpy array containing training data

        y : array
            Array-like containing training targets

        sensitive_features : array
            Array-like containing the sensitive features of the
            training data.
        """
        X, Y, A = self._validate_input(X, y, sensitive_features, reinitialize=True)

        # Not checked in __setup, because partial_fit may not require it.
        if self.epochs == -1 and self.max_iter == -1:
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format(
                    "epochs",
                    (
                        "a positive number, or key-word argument "
                        + "max_iter should be a positive number."
                    ),
                )
            )

        if self.batch_size == -1:
            batch_size = X.shape[0]
        else:
            batch_size = self.batch_size
        batches = ceil(X.shape[0] / batch_size)

        if self.epochs == -1:
            epochs = ceil(self.max_iter / batches)
        else:
            epochs = self.epochs

        start_time = time()
        last_update_time = start_time

        predictor_losses = [None]
        adversary_losses = []

        self.step_ = 0
        for epoch in range(epochs):
            if self.shuffle:
                X, Y, A = self.backendEngine_.shuffle(X, Y, A)
            for batch in range(batches):
                if self.progress_updates:
                    if (time() - last_update_time) > self.progress_updates:
                        last_update_time = time()
                        progress = (epoch / epochs) + (batch / (batches * epochs))
                        if (
                            progress > 0
                            and len(predictor_losses) >= 1
                            and len(adversary_losses) >= 1
                        ):
                            ETA = (
                                (last_update_time - start_time + 1e-6)
                                / (progress + 1e-6)
                            ) * (1 - progress)
                            # + 1e-6 for numerical stability
                            logger.info(
                                _PROGRESS_UPDATE.format(  # noqa : G001
                                    "=" * round(20 * progress),
                                    " " * round(20 * (1 - progress)),  # noqa : G003
                                    epoch + 1,  # noqa : G003
                                    epochs,
                                    " "  # noqa : G003
                                    * (
                                        len(str(batch + 1))  # noqa : G003
                                        - len(str(batches))  # noqa : G003
                                    ),  # noqa : G003
                                    batch + 1,  # noqa : G003
                                    batches,
                                    ETA,
                                    predictor_losses[-1],
                                    adversary_losses[-1],
                                )
                            )
                batch_slice = slice(
                    batch * batch_size,
                    min((batch + 1) * batch_size, X.shape[0]),
                )
                (LP, LA) = self.backendEngine_.train_step(
                    X[batch_slice], Y[batch_slice], A[batch_slice]
                )
                predictor_losses.append(LP)
                adversary_losses.append(LA)

                self.step_ += 1

                # Purposefully first stop and then handle callbacks
                if self.max_iter != -1 and self.step_ >= self.max_iter:
                    return self

                if self.callbacks_:
                    stop = False
                    for cb in self.callbacks_:
                        result = cb(self, self.step_)
                        if result and not isinstance(result, bool):
                            raise RuntimeError(_CALLBACK_RETURNS_ERROR)
                        stop = stop or result

                    if stop:
                        return self

        return self

    def partial_fit(self, X, y, *, sensitive_features=None):
        """
        Perform one epoch on given samples and update model.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional numpy array containing training data

        y : array
            Array-like containing training targets

        sensitive_features : array
            Array-like containing the sensitive feature of the
            training data.
        """
        X, Y, A = self._validate_input(X, y, sensitive_features, reinitialize=False)
        self.backendEngine_.train_step(X, Y, A)

        return self

    def decision_function(self, X):
        """
        Compute predictor output for given test data.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional numpy array containing test data

        Returns
        -------
        Y_pred : numpy.ndarray
            Two-dimensional array containing the model's (soft-)predictions
        """
        check_is_fitted(self)
        X = check_X(X)

        Y_pred = self.backendEngine_.evaluate(X)
        return Y_pred

    def predict(self, X):
        """
        Compute predictions for given test data.

        Predictions are discrete for classifiers, making use of the
        predictor_function.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional numpy array containing test data

        Returns
        -------
        Y_pred : array
            array-like containing the model's predictions fed through
            the (discrete) :code:`predictor_function`
        """
        check_is_fitted(self)
        X = check_X(X)

        Y_pred = self.backendEngine_.evaluate(X)
        Y_pred = self.predictor_function_(Y_pred)
        Y_pred = self._y_transform.inverse_transform(Y_pred)
        return Y_pred

    def _validate_input(self, X, Y, A, reinitialize=False):
        """
        Validate the input data and possibly setup this estimator.

        Important note is that we follow call `__setup` from here, because the
        setup procedure requires the validated data. If `reintialize` is True,
        then always call `__setup`.
        """
        if not self.skip_validation:
            X = check_X(X)

        try:  # TODO check this
            check_is_fitted(self)
            is_fitted = True
        except NotFittedError:
            is_fitted = False

        if A is None:
            logger.warning("No sensitive_features provided")
            logger.warning("Setting sensitive_features to zeros")
            A = zeros(len(Y))

        if (not is_fitted) or (reinitialize):
            self.__setup(X, Y, A)

        if not self.skip_validation:
            Y = self._y_transform.transform(Y)
            A = self._sf_transform.transform(A)

        # Check for equal number of samples
        if not (X.shape[0] == Y.shape[0] and X.shape[0] == A.shape[0]):
            raise ValueError(
                "Input data has an ambiguous number of rows: {}, {}, {}.".format(
                    X.shape[0], Y.shape[0], A.shape[0]
                )
            )

        if not self.skip_validation:
            # Some backendEngine may want to do some additional preprocessing,
            # such as moving to GPU.
            attr = getattr(self.backendEngine_, "validate_input", None)
            if attr:
                X, Y, A = attr(X, Y, A)

        return X, Y, A

    def _validate_backend(self):
        """
        Import either PyTorch or TensorFlow, depending on predictor.

        Given a backend and the predictor/adversary models, we do some steps.
        Firstly, if the backend is a BackendEngine subclass, we just use this.
        Else, we check if parameters comply with a torch backend. Do this by
        checking if backend is set to torch or auto, we have torch installed,
        and whether the models are torch models (or lists). If any step fails,
        we try the same for tensorflow, and if that also fails then we give an
        appropriate error message.
        """
        # Discover which backends are available
        torch_installed = False
        tensorflow_installed = False

        if isinstance(self.backend, type) and issubclass(self.backend, BackendEngine):
            self.backend_ = self.backend
            return
        if self.backend == "torch" or self.backend == "auto":
            select = False
            try:
                from torch.nn import Module as model

                torch_installed = True
                if isinstance(self.predictor_model, (list, model)) and isinstance(
                    self.adversary_model, (list, model)
                ):
                    select = True
                elif self.backend == "torch":
                    raise ValueError(
                        _KWARG_ERROR_MESSAGE.format(
                            "predictor_model and adversary_model",
                            "a list or torch.nn.Module",
                        )
                    )
            except ImportError:
                if self.backend == "torch":
                    raise RuntimeError(_IMPORT_ERROR_MESSAGE.format("torch"))
            if select:
                self.backend_ = PytorchEngine
                return
        if self.backend == "tensorflow" or self.backend == "auto":
            select = False
            try:
                from tensorflow.keras import Model as model

                tensorflow_installed = True
                if isinstance(self.predictor_model, (list, model)) and isinstance(
                    self.adversary_model, (list, model)
                ):
                    select = True
                elif self.backend == "tensorflow":
                    raise ValueError(
                        _KWARG_ERROR_MESSAGE.format(
                            "predictor_model and adversary_model",
                            "a list or tensorflow.keras.Model",
                        )
                    )
            except ImportError:
                if self.backend == "tensorflow":
                    raise RuntimeError(_IMPORT_ERROR_MESSAGE.format("tensorflow"))
            if select:
                self.backend_ = TensorflowEngine
                return
        # The keyword self.backend was weird
        if self.backend not in ["torch", "tensorflow", "auto"]:
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format(
                    "backend", "one of ['auto', 'torch','tensorflow']"
                )
            )
        # Or no backend is installed
        if not (torch_installed or tensorflow_installed):
            raise RuntimeError(_IMPORT_ERROR_MESSAGE.format("torch or tensorflow"))
        # Or all other cases, a mismatch between model model (and installation)
        raise ValueError(
            _KWARG_ERROR_MESSAGE.format(
                "predictor_model and adversary_model",
                "a list, torch.nn.Module, or tensorflow.keras.Model. Also, "
                + "make sure to have installed the corresponding backend",
            )
        )

    def _set_predictor_function(self):
        """
        Infer prediction function.

        The :code:`predictor_function` should be either the string
        :code:`'auto'`, :code:`'binary'`, :code:`'category'`,
        :code:`'continuous'`, or a callable. The string
        :code:`'auto'` indicates to infer the predictor function
        from the distribution type of :code:`y`
        A callable should be a
        function that maps the continuous output of the predictor model to
        a discrete prediction.
        """
        if callable(self.predictor_function_):
            pass
        elif isinstance(self.predictor_function_, str):
            kw = self.predictor_function_
            if kw == "binary":
                self.predictor_function_ = lambda pred: (
                    pred >= self.threshold_value
                ).astype(float)
            elif kw == "category":

                def loss(pred):
                    shape = pred.shape
                    c = argmax(pred, axis=1)
                    b = zeros(shape, dtype=float)
                    a = arange(shape[0])
                    b[a, c] = 1
                    return b

                self.predictor_function_ = loss
            elif kw == "continuous":
                self.predictor_function_ = lambda pred: pred
            else:
                raise ValueError(_PREDICTION_FUNCTION_AMBIGUOUS)
        else:
            raise ValueError(_PREDICTION_FUNCTION_AMBIGUOUS)

    def __sklearn_is_fitted__(self):
        """Speed up check_is_fitted."""
        return hasattr(self, "_is_setup")


class AdversarialFairnessClassifier(_AdversarialFairness, ClassifierMixin):
    r"""Train PyTorch or TensorFlow classifiers while mitigating unfairness.

    This estimator implements the supervised learning method proposed in
    `"Mitigating Unwanted Biases with Adversarial Learning"
    <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_ [1]_.
    The training algorithm takes as input two neural network
    models, a predictor model and an adversarial model, defined either as a
    `PyTorch module
    <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ or
    `TensorFlow2 model
    <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_. The API
    follows conventions of `sklearn` estimators.

    The predictor model takes the features :code:`X` as input and seeks
    to predict :code:`y`.
    For binary classification, the predictor model should return a single
    real-valued score, which is transformed into a probability of the
    positive class via the logistic function (aka sigmoid), similarly to
    logistic regression. For multi-class classification, the predictor
    model should return a vector of real-valued scores, which are
    transformed into class probabilities via the softmax function,
    similarly to multinomial logistic regression. The training loss is
    the negative log likelihood (aka log loss, logistic loss,
    cross-entropy loss).

    The adversarial model for demographic parity takes scores
    produced by the predictor model as input, and seeks to predict
    :code:`sensitive_features`. Depending on the type of the provided
    sensitive features, the model should produce a scalar
    or vector output. Three types of sensitive features are supported:
    (1) a single binary feature; (2) a single discrete feature; (3) one or
    multiple real-valued features. For a single binary sensitive feature
    and a single discrete feature, the network outputs are transformed
    by the logistic function and the softmax function, respectively, and
    the loss is the negative log likelihood. For one or multiple
    real-valued features, the network output is left as is, and the
    loss is a square loss.

    The adversarial model for equalized odds additionaly takes
    :code:`y` as input. For multi-class classification, :code:`y` is
    transformed using one-hot encoding.

    Parameters
    ----------
    backend : str, BackendEngine, default = 'auto'
        The backend to use. Must be one of :code:`'torch'`, :code:`'tensorflow'`,
        or :code:`'auto'` which indicates PyTorch, TensorFlow, or to
        automatically infer
        the backend from the :code:`predictor_model`.
        You can also pass in a BackendEngine class.

    predictor_model : list, torch.nn.Module, tensorflow.keras.Model
        The predictor model to train.
        Instead of a neural network model, it is possible to pass a list
        :math:`[k_1, k_2, \dots]`, where each :math:`k_i` either indicates
        the number of nodes (if :math:`k_i` is an integer) or
        an activation function (if :math:`k_i` is a string) or
        a layer or activation function instance directly (if :math:`k_i` is
        a callable). The default parameter is :code:`[]`, which indicates
        a neural network without any hidden layers.
        However, the number of nodes in the input
        and output layer are automatically inferred from data, and the final
        activation function (such as softmax for categorical
        predictors) are inferred from data.
        If :code:`backend` is specified, you cannot pass a model
        that uses a different backend.

    adversary_model : list, torch.nn.Module, tensorflow.keras.Model
        The adversary model to train. Defined similarly as :code:`predictor_model`.
        Must be the same type as the
        :code:`predictor_model`.

    predictor_optimizer : str, torch.optim, tensorflow.keras.optimizers, callable, default = 'Adam'
        The optimizer class to use. If a string is passed instead, this must be
        either 'SGD' or 'Adam'. A corresponding SGD or Adam optimizer is
        initialized with the given predictor model and learning rate.
        If an instance of a subclass of torch.optim.Optimizer
        or tensorflow.keras.optimizers.Optimizer is passed, this
        is used directly. If a callable :code:`fn` is passed, we call this
        callable and pass our model, and set the result of this call
        as the optimizer, so: :code:`predictor_optimizer=fn(predictor_model)`.

    adversary_optimizer : str, torch.optim, tensorflow.keras.optimizers, callable, default = 'Adam'
        The optimizer class to use. Defined similarly as
        :code:`predictor_optimizer`.

    constraints : str, default = 'demographic_parity'
        The fairness constraint. Must be either 'demographic_parity'
        or 'equalized_odds'.

    learning_rate : float, default = 0.001
        A small number greater than zero to set as a learning rate.

    alpha : float, default = 1.0
        A small number :math:`\alpha` as specified in the paper. It
        is the factor that balances the training towards predicting :code:`y`
        (choose :math:`\alpha` closer to zero) or enforcing fairness constraint
        (choose larger :math:`\alpha`).

    epochs : int, default = 1
        Number of epochs to train for.

    batch_size : int, default = 32
        Batch size. For no batching, set this to -1.

    shuffle : bool, default = False
        When true, shuffle the data before every epoch (including the first).

    progress_updates : number, optional, default = None
        If a number :math:`t` is provided, we print an update
        about the training loop after processing a batch and :math:`t` seconds
        have passed since the previous update.

    skip_validation : bool, default = False
        Skip the validation of the data. Useful because validate_input is
        a costly operation, and we may instead pass all data to validate_input
        at an earlier stage. Note that not only checking :code:`X`
        is skipped, but also no tranform is applied to :code:`y` and
        :code:`sensitive_features`.

    callbacks : callable
        Callback function, called after every batch. For instance useable when
        wanting to validate. A list of callback functions can also be provided.
        Each callback function is passed two arguments :code:`self` (the
        estimator instance) and :code:`step` (the completed iteration), and
        may return a Boolean value. If the returned value is `True`, the
        optimization algorithm terminates. This can be used to implement
        *early stopping*.

    cuda : str, default = None
        A string to indicate which device to use when training. For instance,
        set :code:`cuda='cuda:0'` to train on the first GPU. Only for PyTorch
        backend.

    warm_start : bool, default = False
        Normally, when set to False, a call to :code:`fit()` triggers reinitialization,
        which discards the models and intializes them again. Setting to
        True triggers reuse of these models. Note: if pre-initialized models
        are passed, the models (and their parameters) are never discarded.

    random_state : int, RandomState, default = None
        Controls the randomized aspects of this algorithm, such as shuffling.
        Useful to get reproducible output across multiple function calls.

    References
    ----------
    .. [1] Zhang, Lemoine, Mitchell `"Mitigating Unwanted Biases with
       Adversarial Learning" <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_,
       AIES, 2018.

    """  # noqa : E501

    def __init__(
        self,
        *,
        backend="auto",
        predictor_model=[],  # [] is a NN with no hidden layers (linear NN).
        adversary_model=[],
        predictor_optimizer="Adam",
        adversary_optimizer="Adam",
        constraints="demographic_parity",
        learning_rate=0.001,
        alpha=1.0,
        epochs=1,
        batch_size=32,
        shuffle=False,
        progress_updates=None,
        skip_validation=False,
        callbacks=None,
        cuda=None,
        warm_start=False,
        random_state=None,
    ):
        """Initialize model by setting the predictor loss and function."""
        self._estimator_type = "classifier"
        super(AdversarialFairnessClassifier, self).__init__(
            backend=backend,
            predictor_model=predictor_model,
            adversary_model=adversary_model,
            predictor_loss="classification",
            adversary_loss="auto",
            predictor_function="classification",
            threshold_value=0.5,
            predictor_optimizer=predictor_optimizer,
            adversary_optimizer=adversary_optimizer,
            constraints=constraints,
            learning_rate=learning_rate,
            alpha=alpha,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            progress_updates=progress_updates,
            skip_validation=skip_validation,
            callbacks=callbacks,
            cuda=cuda,
            warm_start=warm_start,
            random_state=random_state,
        )


class AdversarialFairnessRegressor(_AdversarialFairness, RegressorMixin):
    r"""Train PyTorch or TensorFlow regressors while mitigating unfairness.

    This estimator implements the supervised learning method proposed in
    `"Mitigating Unwanted Biases with Adversarial Learning"
    <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_ [1]_.
    The training algorithm takes as input two neural network
    models, a predictor model and an adversarial model, defined either as a
    `PyTorch module
    <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ or
    `TensorFlow2 model
    <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_. The API
    follows conventions of `sklearn` estimators.

    The regressor model takes the features :code:`X` as input and seeks
    to predict :code:`y`.
    The training loss is measured using the squared error.

    The adversarial model for demographic parity takes scores
    produced by the predictor model as input, and seeks to predict
    :code:`sensitive_features`. Depending on the type of the provided
    sensitive features, the model should produce a scalar
    or vector output. Three types of sensitive features are supported:
    (1) a single binary feature; (2) a single discrete feature; (3) one or
    multiple real-valued features. For a single binary sensitive feature
    and a single discrete feature, the network outputs are transformed
    by the logistic function and the softmax function, respectively, and
    the loss is the negative log likelihood. For one or multiple
    real-valued features, the network output is left as is, and the
    loss is a square loss.

    The adversarial model for equalized odds additionaly takes
    :code:`y` as input.

    Parameters
    ----------
    backend : str, BackendEngine, default = 'auto'
        The backend to use. Must be one of :code:`'torch'`, :code:`'tensorflow'`,
        or :code:`'auto'` which indicates PyTorch, TensorFlow, or to
        automatically infer
        the backend from the :code:`predictor_model`.
        You can also pass in a BackendEngine class.

    predictor_model : list, torch.nn.Module, tensorflow.keras.Model
        The predictor model to train.
        Instead of a neural network model, it is possible to pass a list
        :math:`[k_1, k_2, \dots]`, where each :math:`k_i` either indicates
        the number of nodes (if :math:`k_i` is an integer) or
        an activation function (if :math:`k_i` is a string) or
        a layer or activation function instance directly (if :math:`k_i` is
        a callable). The default parameter is :code:`[]`, which indicates
        a neural network without any hidden layers.
        However, the number of nodes in the input
        and output layer are automatically inferred from data, and the final
        activation function (such as softmax for categorical
        predictors) are inferred from data.
        If :code:`backend` is specified, you cannot pass a model
        that uses a different backend.

    adversary_model : list, torch.nn.Module, tensorflow.keras.Model
        The adversary model to train. Defined similarly as :code:`predictor_model`.
        Must be the same type as the
        :code:`predictor_model`.

    predictor_optimizer : str, torch.optim, tensorflow.keras.optimizers, callable, default = 'Adam'
        The optimizer class to use. If a string is passed instead, this must be
        either 'SGD' or 'Adam'. A corresponding SGD or Adam optimizer is
        initialized with the given predictor model and learning rate.
        If an instance of a subclass of torch.optim.Optimizer
        or tensorflow.keras.optimizers.Optimizer is passed, this
        is used directly. If a callable :code:`fn` is passed, we call this
        callable and pass our model, and set the result of this call
        as the optimizer, so: :code:`predictor_optimizer=fn(predictor_model)`.

    adversary_optimizer : str, torch.optim, tensorflow.keras.optimizers, callable, default = 'Adam'
        The optimizer class to use. Defined similarly as
        :code:`predictor_optimizer`.

    constraints : str, default = 'demographic_parity'
        The fairness constraint. Must be either 'demographic_parity'
        or 'equalized_odds'.

    learning_rate : float, default = 0.001
        A small number greater than zero to set as a learning rate.

    alpha : float, default = 1.0
        A small number :math:`\alpha` as specified in the paper. It
        is the factor that balances the training towards predicting :code:`y`
        (choose :math:`\alpha` closer to zero) or enforcing fairness constraint
        (choose larger :math:`\alpha`).

    epochs : int, default = 1
        Number of epochs to train for.

    batch_size : int, default = 32
        Batch size. For no batching, set this to -1.

    shuffle : bool, default = False
        When true, shuffle the data before every epoch (including the first).

    progress_updates : number, optional, default = None
        If a number :math:`t` is provided, we print an update
        about the training loop after processing a batch and :math:`t` seconds
        have passed since the previous update.

    skip_validation : bool, default = False
        Skip the validation of the data. Useful because validate_input is
        a costly operation, and we may instead pass all data to validate_input
        at an earlier stage. Note that not only checking :code:`X`
        is skipped, but also no tranform is applied to :code:`y` and
        :code:`sensitive_features`.

    callbacks : callable
        Callback function, called after every batch. For instance useable when
        wanting to validate. A list of callback functions can also be provided.
        Each callback function is passed two arguments :code:`self` (the
        estimator instance) and :code:`step` (the completed iteration), and
        may return a Boolean value. If the returned value is `True`, the
        optimization algorithm terminates. This can be used to implement
        *early stopping*.

    cuda : str, default = None
        A string to indicate which device to use when training. For instance,
        set :code:`cuda='cuda:0'` to train on the first GPU. Only for PyTorch
        backend.

    warm_start : bool, default = False
        Normally, when set to False, a call to :code:`fit()` triggers reinitialization,
        which discards the models and intializes them again. Setting to
        True triggers reuse of these models. Note: if pre-initialized models
        are passed, the models (and their parameters) are never discarded.

    random_state : int, RandomState, default = None
        Controls the randomized aspects of this algorithm, such as shuffling.
        Useful to get reproducible output across multiple function calls.

    References
    ----------
    .. [1] Zhang, Lemoine, Mitchell `"Mitigating Unwanted Biases with
       Adversarial Learning" <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_,
       AIES, 2018.

    """  # noqa : E501

    def __init__(
        self,
        *,
        backend="auto",
        predictor_model=[],  # [] is a NN with no hidden layers (linear NN).
        adversary_model=[],
        predictor_optimizer="Adam",
        adversary_optimizer="Adam",
        constraints="demographic_parity",
        learning_rate=0.001,
        alpha=1.0,
        epochs=1,
        batch_size=32,
        shuffle=False,
        progress_updates=None,
        skip_validation=False,
        callbacks=None,
        cuda=None,
        warm_start=False,
        random_state=None,
    ):
        """Initialize model by setting the predictor loss and function."""
        self._estimator_type = "regressor"
        super(AdversarialFairnessRegressor, self).__init__(
            backend=backend,
            predictor_model=predictor_model,
            adversary_model=adversary_model,
            predictor_loss="continuous",
            adversary_loss="auto",
            predictor_function=None,
            predictor_optimizer=predictor_optimizer,
            adversary_optimizer=adversary_optimizer,
            y_transform=None,
            constraints=constraints,
            learning_rate=learning_rate,
            alpha=alpha,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            progress_updates=progress_updates,
            skip_validation=skip_validation,
            callbacks=callbacks,
            cuda=cuda,
            warm_start=warm_start,
            random_state=random_state,
        )


def check_X(X):
    """
    Validate the input array, and possible coerce to 2D.

    Calls :code:`sklearn.utils.check_array` on parameter X with the
    parameters suited for Adversarial Mitigation.

    Returns
    -------
    X : numpy.ndarray
        Cleaned data.
    """
    return check_array(
        X,
        accept_sparse=False,
        accept_large_sparse=False,
        dtype=float,
        allow_nd=True,
    ).astype(float)
