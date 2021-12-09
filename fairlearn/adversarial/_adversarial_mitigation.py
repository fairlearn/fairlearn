# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from math import ceil
from time import time
from ._constants import (
    _IMPORT_ERROR_MESSAGE,
    _KWARG_ERROR_MESSAGE,
    _PREDICTION_FUNCTION_AMBIGUOUS,
    _PROGRESS_UPDATE,
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


class AdversarialFairness(BaseEstimator):
    r"""Train complex predictors while mitigating biases in PyTorch or Tensorflow.

    This model implements the supervised learning method proposed in
    `"Mitigating Unwanted Biases with Adversarial Learning"
    <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_ [1]_.
    This algorithm takes as input two neural network
    models, a predictor and an adversarial, defined either as a `PyTorch module
    <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ or
    `Tensorflow2 model
    <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_. You train this
    predictor using an API that is similar to estimators in `sklearn`.

    As per [1]_, the neural network models :code:`predictor_model` and
    :code:`adversary_model` may not have a discrete prediction at the end of the
    model. Instead, even if we are dealing with discrete predictions, we may output
    the sigmoidal or soft-max, but we may not output discrete integer predictions.

    The distribution types of :code:`y` and :code:`sensitive_features`
    are set by their preprocessor :code:`y_transform` and :code:`z_transform`
    respectively. The default transformer :code:`FloatTransformer("auto")`
    attempts to automatically infer
    whether to assume binomial, multinomial, or normally distributed data.
    You can force the transformer to assume one of the above by passing
    :code:`FloatTransformer("binary")`, :code:`FloatTransformer("category")`,
    :code:`FloatTransformer("continuous")` instead.
    Loss functions and decision functions
    for the target and sensitive features are also predefined for keywords
    "binary", "category", and "continuous"
    For more information, visit the user guide.

    Parameters
    ----------
    backend : str, BackendEngine, default = 'auto'
        The backend to use. Must be one of :code:`'torch'`, :code:`'tensorflow'`,
        or :code:`'auto'` which indicates PyTorch, TensorFlow, or to
        automatically infer
        the backend from the :code:`predictor_model` and which are installed.
        You can also pass in a BackendEngine class.

    predictor_model : list, torch.nn.Module, tensorflow.keras.Model
        The predictor model to train.
        Instead of a neural network model, we can pass a list of keywords
        :math:`k_1, k_2, \dots` that indicate either
        the number of nodes :math:`k_i` (if :math:`k_i` is integer) or a keyword
        that indicates an activation function (if :math:`k_i` is a string) or
        a layer or activation function instance directly (if :math:`k_i` is
        callable).
        However, the number of nodes in the input
        and output layer are automatically inferred from data, and the final
        activation function (such as softmax for categorical
        predictors) are inferred from data.
        If :code:`backend` is specified, we cannot pass a model
        that uses a different backend.

    adversary_model : list, torch.nn.Module, tensorflow.keras.Model
        The adversary model to train. Must be the same type as the
        :code:`predictor_model`.

    predictor_loss : str, callable, default = 'auto'
        Either the string
        :code:`'auto'`, :code:`'binary'`, :code:`'category'`,
        :code:`'continuous'`, or a callable. The string
        :code:`'auto'` indicates to infer the loss
        from the distribution type of :code:`y`
        The string-keyword indicates a
        distribution type, see user guide for more information.
        A callable should be a
        loss function with an API that follows the chosen backend (torch or
        tensorflow). Note that torch and tensorflow loss functions don't agree
        on parameter order, as in Pytorch it is :math:`l(\hat y, y)` while in
        Tensorflow it is :math:`l(y, \hat y)`.

    adversary_loss : str, callable, default = 'auto'
        Either the string
        :code:`'auto'`, :code:`'binary'`, :code:`'category'`,
        :code:`'continuous'`, or a callable. The string
        :code:`'auto'` indicates to infer the loss
        from the distribution type of :code:`sensitive_features`
        The string-keyword indicates a
        distribution type, see user guide for more information.
        A callable should be a
        loss function with an API that follows the chosen backend (torch or
        tensorflow).

    predictor_function : str, callable, default = 'auto'
        Either the string
        :code:`'auto'`, :code:`'binary'`, :code:`'category'`,
        :code:`'continuous'`, or a callable. The string
        :code:`'auto'` indicates to infer the predictor function
        from the distribution type of :code:`y`
        The string-keyword indicates a
        distribution type, see user guide for more information.
        A callable should be a
        function that maps the continuous output of the predictor model to
        a discrete prediction.

    predictor_optimizer : str, torch.optim, tensorflow.keras.optimizers, callable, default = 'Adam'
        The optimizer class to use. If a string is passed instead, this must be
        either "SGD" or "Adam". A corresponding SGD or Adam optimizer is
        initialized with the model and given learning rate.
        If an instance of a subclass of torch.optim.Optimizer
        or tensorflow.keras.optimizers.Optimizer is passed, this
        is used directly. If a callable :code:`fn` is passed, we call this
        callable and pass our model, and set the result of this call
        as the optimizer, so: :code:`predictor_optimizer=fn(predictor_model)`.


    adversary_optimizer : str, torch.optim, tensorflow.keras.optimizers, callable, default = 'Adam'
        The optimizer class to use. Similarly defined as
        :code:`predictor_optimizer`

    constraints : str, default = 'demographic_parity'
        The fairness measure to optimize for. Must be either 'demographic_parity'
        (Demographic Parity) or 'equalized_odds' (Equalized Odds).

    y_transform : sklearn.base.TransformerMixin, default = fairlearn.adversarial.FloatTransformer("auto")
        The preprocessor to use on the predictions :code:`y`.
        The given transformer *must* map data
        to a 2d ndarray containing only floats. Per default, we use a
        FloatTransformer that maps strictly binary or categorical tables
        to binary or
        one-hot encodings, and it maps strictly continuous-valued (possible 2d)
        to itself.

    z_transform : sklearn.base.TransformerMixin, default = fairlearn.adversarial.FloatTransformer("auto")
        The preprocessor to use on the :code:`sensitive_features`.
        The given transformer *must* map data
        to a 2d ndarray containing only floats. Per default, we use a
        FloatTransformer that maps strictly binary or categorical tables
        to binary or
        one-hot encodings, and it maps strictly continuous-valued (possible 2d)
        to itself.

    learning_rate : float, default = 0.001
        A small number greater than zero to set as initial learning rate

    alpha : float, default = 1.0
        A small number :math:`\alpha` as specified in the paper.

    epochs : int, default = 1
        Number of epochs to train for.

    batch_size : int, default = -1
        Batch size. For no batching, set this to -1.

    shuffle : bool, default = False
        Iff true, shuffle the data after every iteration.

    progress_updates : number, optional, default = None
        If a number :math:`t` is provided, we print an update
        about the training loop after processing a batch and :math:`t` seconds
        have passed since the previous update.

    skip_validation : bool, default = False
        Skip the validation of the data. Useful because validate_input is
        a costly operation, and we may instead pass all data to validate_input
        at an earlier stage. Note that not only checking :math:`X`
        is skipped, but also no tranform is applied to :math:`y` and
        :math:`z` (sensitive features).

    callback_fn : callable
        Callback function, called after every batch. For instance useable when
        wanting to validate. We pass as arguments:
        :code:`(self, epoch, batch)`. If a callable returns something non-None
        (such as True) then the fitting stops, which is useful when
        implementing *early stopping*.

    cuda : str, default = None
        A string to indicate which device to use when training. For instance,
        set :code:`cuda='cuda:0'` to train on the first GPU. Only for PyTorch
        backend.

    warm_start : bool, default = False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.

    random_state : int, RandomState instance, default = None
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
        predictor_model,
        adversary_model,
        predictor_loss="auto",
        adversary_loss="auto",
        predictor_function="auto",
        predictor_optimizer="Adam",
        adversary_optimizer="Adam",
        constraints="demographic_parity",
        y_transform=FloatTransformer("auto"),
        z_transform=FloatTransformer("auto"),
        learning_rate=0.001,
        alpha=1.0,
        epochs=1,
        batch_size=-1,
        shuffle=False,
        progress_updates=None,
        skip_validation=False,
        callback_fn=None,
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
        self.predictor_optimizer = predictor_optimizer
        self.adversary_optimizer = adversary_optimizer
        self.constraints = constraints
        self.y_transform = y_transform
        self.z_transform = z_transform
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.progress_updates = progress_updates
        self.skip_validation = skip_validation
        self.callback_fn = callback_fn
        self.cuda = cuda
        self.warm_start = warm_start
        self.random_state = random_state

    def __setup(self, X, Y, Z):
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

        for kw, kwname in (
            (self.learning_rate, "learning_rate"),
            (self.alpha, "alpha"),
            (self.epochs, "epochs"),
            (self.progress_updates, "progress_updates"),
        ):
            if kw and kw < 0.0:
                raise ValueError(
                    _KWARG_ERROR_MESSAGE.format(kwname, "a non-negative number")
                )
        if self.batch_size <= 0.0 and self.batch_size != -1:
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format(
                    "batch_size", "a positive number or -1"
                )
            )

        for kw, kwname in (
            (self.shuffle, "shuffle"),
            (self.skip_validation, "skip_validation"),
            (self.warm_start, "warm_start"),
        ):
            if not isinstance(kw, bool):
                raise ValueError(
                    _KWARG_ERROR_MESSAGE.format(kwname, "a boolean")
                )

        if self.callback_fn and not callable(self.callback_fn):
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format("callback_fn", "a callable")
            )

        # NOTE: inferring distribution type should happen before transforming
        read_kw = (
            lambda data, kw_or_func: _get_type(data, kw_or_func)
            if isinstance(kw_or_func, str)
            else kw_or_func
        )

        self.predictor_loss_ = read_kw(Y, self.predictor_loss)
        self.adversary_loss_ = read_kw(Z, self.adversary_loss)
        self.predictor_function_ = read_kw(Y, self.predictor_function)

        kws = ["binary", "category", "continuous"]
        for kw, kwname in (
            (self.predictor_loss_, "predictor_loss"),
            (self.adversary_loss_, "adversary_loss"),
            (self.predictor_function_, "predictor_function"),
        ):
            # Some keywords can also be a callable instead of a str.
            if not ((isinstance(kw, str) and kw in kws) or callable(kw)):
                raise ValueError(
                    _KWARG_ERROR_MESSAGE.format(
                        kwname,
                        (
                            "'auto', 'binary', 'category', 'continuous', "
                            + "or a callable",
                        ),
                    )
                )

        for kw, kwname in (
            (self.y_transform, "y_transform"),
            (self.z_transform, "z_transform"),
        ):
            if not (
                issubclass(type(kw), TransformerMixin)
                and hasattr(kw, "fit")
                and hasattr(kw, "transform")
            ):
                raise ValueError(
                    _KWARG_ERROR_MESSAGE.format(
                        kwname,
                        "a sklearn Transformer (subclass TransformerMixin)",
                    )
                )

        self.y_transform_ = self.y_transform.fit(Y)
        self.z_transform_ = self.z_transform.fit(Z)

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
        self.backendEngine_ = self.backend_(self, X, Y, Z)

        # Sklearn-parameters
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = self.y_transform_.n_features_in_

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
        X, Y, Z = self._validate_input(X, y, sensitive_features)

        if self.batch_size == -1:
            batch_size = X.shape[0]
        else:
            batch_size = self.batch_size
        batches = ceil(X.shape[0] / batch_size)

        start_time = time()
        last_update_time = start_time

        # logger = logging.getLogger(__name__) FIXME: use logger

        predictor_losses = [None]
        adversary_losses = []
        for epoch in range(self.epochs):
            for batch in range(batches):
                if self.progress_updates:
                    if (time() - last_update_time) > self.progress_updates:
                        last_update_time = time()
                        progress = (epoch / self.epochs) + (
                            batch / (batches * self.epochs)
                        )
                        print(
                            _PROGRESS_UPDATE.format(
                                "=" * round(20 * progress),
                                " " * round(20 * (1 - progress)),
                                epoch + 1,
                                self.epochs,
                                " " * (len(str(batch + 1)) - len(str(batches))),
                                batch + 1,
                                batches,
                                ((last_update_time - start_time) / progress)
                                * (1 - progress),
                                predictor_losses[-1],
                                adversary_losses[-1],
                            ),
                            end="\n",
                        )
                batch_slice = slice(
                    batch * batch_size,
                    min((batch + 1) * batch_size, X.shape[0]),
                )
                (LP, LA) = self.backendEngine_.train_step(
                    X[batch_slice], Y[batch_slice], Z[batch_slice]
                )
                predictor_losses.append(LP)
                adversary_losses.append(LA)

                if self.callback_fn and callable(self.callback_fn):
                    stop = self.callback_fn(self, epoch, batch)
                    if stop:
                        return
            if self.shuffle and epoch != self.epochs - 1:
                X, Y, Z = self.backendEngine_.shuffle(X, Y, Z)

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
        X, Y, Z = self._validate_input(X, y, sensitive_features)
        self.backendEngine_.train_step(X, Y, Z)

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
        Compute discrete predictions for given test data.

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
        Y_pred = self.y_transform_.inverse_transform(Y_pred)
        return Y_pred

    def _validate_input(self, X, Y, Z):
        """
        Validate the input data and possibly setup this estimator.

        Important note is that we follow call `__setup` from here, because the
        setup procedure requires the validated data.
        """
        if not self.skip_validation:
            X = check_X(X)

        try:  # TODO check this
            check_is_fitted(self)
            is_fitted = True
        except NotFittedError:
            is_fitted = False

        if Z is None:
            print("Warning: no sensitive_features provided")  # FIXME : logger?
            Z = [0] * X.shape[0]

        if (not is_fitted) or (not self.warm_start):
            self.__setup(X, Y, Z)

        if not self.skip_validation:
            Y = self.y_transform_.transform(Y)
            Z = self.z_transform_.transform(Z)

        # Check for equal number of samples
        if not (X.shape[0] == Y.shape[0] and X.shape[0] == Z.shape[0]):
            raise ValueError(
                "Input data has an ambiguous number of rows: {}, {}, {}.".format(
                    X.shape[0], Y.shape[0], Z.shape[0]
                )
            )

        if not self.skip_validation:
            # Some backendEngine may want to do some additional preprocessing,
            # such as moving to GPU.
            attr = getattr(self.backendEngine_, "validate_input", None)
            if attr:
                X, Y, Z = attr(X, Y, Z)

        return X, Y, Z

    def _validate_backend(self):
        """
        Import either PyTorch or Tensorflow, depending on predictor.

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

        if isinstance(self.backend, type) and issubclass(
            self.backend, BackendEngine
        ):
            self.backend_ = self.backend
            return
        if self.backend == "torch" or self.backend == "auto":
            select = False
            try:
                from torch.nn import Module as model

                torch_installed = True
                if isinstance(
                    self.predictor_model, (list, model)
                ) and isinstance(self.adversary_model, (list, model)):
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
                if isinstance(
                    self.predictor_model, (list, model)
                ) and isinstance(self.adversary_model, (list, model)):
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
                    raise RuntimeError(
                        _IMPORT_ERROR_MESSAGE.format("tensorflow")
                    )
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
            raise RuntimeError(
                _IMPORT_ERROR_MESSAGE.format("torch or tensorflow")
            )
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
                self.predictor_function_ = lambda pred: (pred >= 0.5).astype(
                    float
                )
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


class AdversarialFairnessClassifier(AdversarialFairness, ClassifierMixin):
    """Creates an AdversarialFairness with loss and predictions set to classification."""

    def __init__(self, **kwargs):
        """Initialize model by setting the predictor loss and function."""
        kwargs["y_transform"] = kwargs.get(
            "y_transform", FloatTransformer("classification")
        )
        super(AdversarialFairnessClassifier, self).__init__(**kwargs)


class AdversarialFairnessRegressor(AdversarialFairness, RegressorMixin):
    """Create an AdversarialFairness that has predictor loss set to regression."""

    def __init__(self, *args, **kwargs):
        """Initialize model by setting the predictor loss."""
        kwargs["y_transform"] = kwargs.get(
            "y_transform", FloatTransformer("continuous")
        )
        super(AdversarialFairnessRegressor, self).__init__(*args, **kwargs)


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
    ).astype(float)
