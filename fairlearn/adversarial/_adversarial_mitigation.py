# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

import logging
import warnings
from math import ceil
from time import time

from numpy import arange, argmax, unique, zeros
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    is_classifier,
)
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.utils import check_scalar
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import (
    check_consistent_length,
    check_is_fitted,
    check_random_state,
)

from fairlearn.utils._fixes import validate_data

from ._backend_engine import BackendEngine
from ._constants import (
    _CALLBACK_RETURNS_ERROR,
    _IMPORT_ERROR_MESSAGE,
    _KWARG_ERROR_MESSAGE,
    _PREDICTION_FUNCTION_AMBIGUOUS,
    _PROGRESS_UPDATE,
)
from ._preprocessor import FloatTransformer
from ._pytorch_engine import PytorchEngine
from ._tensorflow_engine import TensorflowEngine

logger = logging.getLogger(__name__)


class _AdversarialFairness(BaseEstimator):
    r"""Train PyTorch or TensorFlow predictors while mitigating unfairness .

    This model implements the supervised learning method proposed in
    "Mitigating Unwanted Biases with Adversarial Learning". :footcite:`zhang2018mitigating`
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

    predictor_model : list, torch.nn.Module, keras.Model
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

    adversary_model : list, torch.nn.Module, keras.Model
        The adversary model to train. Defined similarly as :code:`predictor_model`.
        Must be the same type as the
        :code:`predictor_model`.

    predictor_optimizer : str, torch.optim, keras.optimizers, callable, default = 'Adam'
        The optimizer class to use. If a string is passed instead, this must be
        either 'SGD' or 'Adam'. A corresponding SGD or Adam optimizer is
        initialized with the given predictor model and learning rate.
        If an instance of a subclass of torch.optim.Optimizer
        or keras.optimizers.Optimizer is passed, this
        is used directly. If a callable :code:`fn` is passed, we call this
        callable and pass our model, and set the result of this call
        as the optimizer, so: :code:`predictor_optimizer=fn(predictor_model)`.

    adversary_optimizer : str, torch.optim, keras.optimizers, callable, default = 'Adam'
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
    .. footbibliography::

    """  # noqa : E501

    def __init__(
        self,
        *,
        backend="auto",
        predictor_model=None,
        adversary_model=None,
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

    def __setup(self, X, y, A):
        """
        Initialize model from parameters and data.

        Validates backend and arguments, then initializes BackendEngine.
        Infers appropriate losses and functions if not explicitly defined.
        Called from `fit` method, not `__init__`, following sklearn API.

        Parameters:
        X : array-like, input features
        y : array-like, target values
        A : array-like, sensitive features
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
            check_scalar(kw, kwname, (int, float), min_val=-1, include_boundaries="left")
            if kw <= 0.0 and kw != -1:
                raise ValueError(_KWARG_ERROR_MESSAGE.format(kwname, "a positive number or -1"))

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
                        _KWARG_ERROR_MESSAGE.format("callbacks", "a callable or list of callables")
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

        self.predictor_loss_ = self.predictor_function_ = type_of_target(y)
        self.adversary_loss_ = type_of_target(A)

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
                        "a keyword or a sklearn Transformer" + "(subclass TransformerMixin)",
                    )
                )

        self._y_transform = FloatTransformer(transformer=self.y_transform)
        self._sf_transform = FloatTransformer(transformer=self.sf_transform)
        self._y_transform.fit(y)
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
        self.backendEngine_ = self.backend_(self, X, y, A)

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
        first_call = not hasattr(self, "classes_")

        X, y, A = self._validate_input(X, y, sensitive_features, first_call)

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

        if self.predictor_model is not None:
            predictor_model = self.predictor_model
        else:
            predictor_model = []  # [] is a NN with no hidden layers # noqa: F841

        if self.adversary_model is not None:
            adversary_model = self.adversary_model
        else:
            adversary_model = []  # noqa: F841

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

        self.n_iter_ = 0
        for epoch in range(epochs):
            if self.shuffle:
                X, y, A = self.backendEngine_.shuffle(X, y, A)
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
                            ETA = ((last_update_time - start_time + 1e-6) / (progress + 1e-6)) * (
                                1 - progress
                            )
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
                    X[batch_slice], y[batch_slice], A[batch_slice]
                )
                predictor_losses.append(LP)
                adversary_losses.append(LA)

                self.n_iter_ += 1

                # Purposefully first stop and then handle callbacks
                if self.max_iter != -1 and self.n_iter_ >= self.max_iter:
                    return self

                if self.callbacks_:
                    stop = False
                    for cb in self.callbacks_:
                        result = cb(
                            self, step=self.n_iter_, X=X, y=y, z=sensitive_features, pos_label=1
                        )
                        if result and not isinstance(result, bool):
                            raise RuntimeError(_CALLBACK_RETURNS_ERROR)
                        stop = stop or result

                    if stop:
                        return self

        return self

    def partial_fit(self, X, y, *, classes=None, sensitive_features=None):
        """
        Perform one training step on given samples and update model.

        This method allows for incremental fitting on batches of data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        classes : array-like of shape (n_classes,), default=None
            List of all the classes that can possibly appear in the y vector.
            Must be provided at the first call to partial_fit, can be omitted
            in subsequent calls.

        sensitive_features : array-like of shape (n_samples,), default=None
            The sensitive features for each sample. If None, a vector of zeros
            will be used.

        Returns
        -------
        self : object
            Returns self.
        """

        first_call = not hasattr(self, "classes_")

        if first_call and classes is not None:
            self.classes_ = classes
        if not first_call:
            if self.n_features_in_ != X.shape[1]:
                raise ValueError(
                    "X has %d features, but %s is expecting %d features as input"
                    % (X.shape[1], self.__class__.__name__, self.n_features_in_)
                )

        X, y, A = self._validate_input(X, y, sensitive_features, first_call)
        self.backendEngine_.train_step(X, y, A)

        return self

    def _raw_predict(self, X):
        """
        Compute predictor output for given test data.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional numpy array containing test data

        Returns
        -------
        y_pred : numpy.ndarray
            Two-dimensional array containing the model's (soft-)predictions
        """
        check_is_fitted(self)
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            accept_large_sparse=False,
            dtype=float,
            allow_nd=True,
            reset=False,
        )
        y_pred = self.backendEngine_.evaluate(X)
        return y_pred

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
        y_pred : array
            array-like containing the model's predictions fed through
            the (discrete) :code:`predictor_function`
        """
        y_pred = self._raw_predict(X)
        y_pred = self.predictor_function_(y_pred)
        y_pred = self._y_transform.inverse_transform(y_pred)
        return y_pred

    def _validate_input(self, X, y, A, reinitialize=False):
        """
        Validate input data and optionally set up the estimator.

        Parameters
        ----------
        X : array-like
            The input features.
        y : array-like
            The target values.
        A : array-like
            The sensitive features.
        reinitialize : bool, default=False
            If True, force reinitialization of the estimator.

        Returns
        -------
        X : array-like
            Validated input features.
        y : array-like
            Validated target values.
        A : array-like
            Validated sensitive features.

        Notes
        -----
        This method calls `__setup` if the estimator is not fitted or if
        `reinitialize` is True. The setup procedure requires validated data.
        """
        if not self.skip_validation:
            X = validate_data(
                self,
                X,
                accept_sparse=False,
                accept_large_sparse=False,
                dtype=float,
                allow_nd=True,
                ensure_2d=True,
            )
            if y is not None:
                y = validate_data(self, y, dtype=None, ensure_2d=False)
                if y.ndim != 1:
                    warnings.warn(
                        (
                            "A column-vector y was passed when a "
                            "1d array was expected. Please change "
                            "the shape of y to (n_samples,), for "
                            "example using ravel()."
                        ),
                        DataConversionWarning,
                        stacklevel=2,
                    )

            check_consistent_length(X, y)

            if is_classifier(self) and type_of_target(y) == "continuous":
                raise ValueError(
                    "Unknown label type: Regression targets have been passed to AdversarialFairnessClassifier."
                )

        try:  # TODO check this
            check_is_fitted(self)
            is_fitted = True
        except NotFittedError:
            is_fitted = False

        if A is None:
            logger.warning("No sensitive_features provided")
            logger.warning("Setting sensitive_features to zeros")
            A = zeros(len(X))

        if not self.skip_validation:
            check_consistent_length(X, A)

        if (not is_fitted) or (reinitialize):
            self.__setup(X, y, A)

        if not hasattr(self, "classes_"):
            self.classes_ = unique(y)

        y = self._y_transform.transform(y)
        A = self._sf_transform.transform(A)

        if not self.skip_validation:
            # Some backendEngine may want to do some additional preprocessing,
            # such as moving to GPU.
            attr = getattr(self.backendEngine_, "validate_input", None)
            if attr:
                X, y, A = attr(X, y, A)

        return X, y, A

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
                from keras import Model as model

                tensorflow_installed = True
                if isinstance(self.predictor_model, (list, model)) and isinstance(
                    self.adversary_model, (list, model)
                ):
                    select = True
                elif self.backend == "tensorflow":
                    raise ValueError(
                        _KWARG_ERROR_MESSAGE.format(
                            "predictor_model and adversary_model",
                            "a list or keras.Model",
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
                _KWARG_ERROR_MESSAGE.format("backend", "one of ['auto', 'torch','tensorflow']")
            )
        # Or no backend is installed
        if not (torch_installed or tensorflow_installed):
            raise RuntimeError(_IMPORT_ERROR_MESSAGE.format("torch or tensorflow"))
        # Or all other cases, a mismatch between model model (and installation)
        raise ValueError(
            _KWARG_ERROR_MESSAGE.format(
                "predictor_model and adversary_model",
                "a list, torch.nn.Module, or keras.Model. Also, "
                + "make sure to have installed the corresponding backend",
            )
        )

    def _binary_predictor_function(self, pred):
        return (pred >= self.threshold_value).astype(float)

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
                self.predictor_function_ = self._binary_predictor_function
            elif kw == "multiclass":

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


class AdversarialFairnessClassifier(ClassifierMixin, _AdversarialFairness):
    r"""Train PyTorch or TensorFlow classifiers while mitigating unfairness.

    This estimator implements the supervised learning method proposed in
    "Mitigating Unwanted Biases with Adversarial Learning". :footcite:`zhang2018mitigating`
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

    predictor_model : list, torch.nn.Module, keras.Model
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

    adversary_model : list, torch.nn.Module, keras.Model
        The adversary model to train. Defined similarly as :code:`predictor_model`.
        Must be the same type as the
        :code:`predictor_model`.

    predictor_optimizer : str, torch.optim, keras.optimizers, callable, default = 'Adam'
        The optimizer class to use. If a string is passed instead, this must be
        either 'SGD' or 'Adam'. A corresponding SGD or Adam optimizer is
        initialized with the given predictor model and learning rate.
        If an instance of a subclass of torch.optim.Optimizer
        or keras.optimizers.Optimizer is passed, this
        is used directly. If a callable :code:`fn` is passed, we call this
        callable and pass our model, and set the result of this call
        as the optimizer, so: :code:`predictor_optimizer=fn(predictor_model)`.

    adversary_optimizer : str, torch.optim, keras.optimizers, callable, default = 'Adam'
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
        Each callback function is called as::

            callback(
                self, step=self.step_, X=X, y=y, z=sensitive_features, pos_label=1
            )

        which is passed the ``self`` object, the step number, the inputs ``X``,
        the targets ``y``, the sensitive features ``z``, and the positive label.
        The callback may return a Boolean value. If the returned value is `True`,
        the optimization algorithm terminates. This can be used to implement
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
    .. footbibliography::

    """  # noqa : E501

    def __init__(
        self,
        *,
        backend="auto",
        predictor_model=None,
        adversary_model=None,
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

    def _more_tags(self):
        return {"poor_score": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if tags.classifier_tags:
            tags.classifier_tags.poor_score = True
        return tags


class AdversarialFairnessRegressor(RegressorMixin, _AdversarialFairness):
    r"""Train PyTorch or TensorFlow regressors while mitigating unfairness.

    This estimator implements the supervised learning method proposed in
    "Mitigating Unwanted Biases with Adversarial Learning". :footcite:`zhang2018mitigating`
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

    predictor_model : list, torch.nn.Module, keras.Model
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

    adversary_model : list, torch.nn.Module, keras.Model
        The adversary model to train. Defined similarly as :code:`predictor_model`.
        Must be the same type as the
        :code:`predictor_model`.

    predictor_optimizer : str, torch.optim, keras.optimizers, callable, default = 'Adam'
        The optimizer class to use. If a string is passed instead, this must be
        either 'SGD' or 'Adam'. A corresponding SGD or Adam optimizer is
        initialized with the given predictor model and learning rate.
        If an instance of a subclass of torch.optim.Optimizer
        or keras.optimizers.Optimizer is passed, this
        is used directly. If a callable :code:`fn` is passed, we call this
        callable and pass our model, and set the result of this call
        as the optimizer, so: :code:`predictor_optimizer=fn(predictor_model)`.

    adversary_optimizer : str, torch.optim, keras.optimizers, callable, default = 'Adam'
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
    .. footbibliography::

    """  # noqa : E501

    def __init__(
        self,
        *,
        backend="auto",
        predictor_model=None,
        adversary_model=None,
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

    def _more_tags(self):
        return {"poor_score": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if tags.regressor_tags:
            tags.regressor_tags.poor_score = True
        return tags
