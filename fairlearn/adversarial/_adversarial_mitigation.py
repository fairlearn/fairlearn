# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from math import ceil
from time import time
from ._constants import (
    _IMPORT_ERROR_MESSAGE,
    _KWARG_ERROR_MESSAGE,
    _PREDICTION_FUNCTION_AMBIGUOUS,
    _PROGRESS_UPDATE,
    _TRANSFORM_NO_PARAM,
    _TRANSFORMER_WEIRD_DIST_TYPE,
)
from ._backend_engine import BackendEngine
from ._pytorch_engine import PytorchEngine
from ._tensorflow_engine import TensorflowEngine
from ._preprocessor import FloatTransformer
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

    The distribution types of the data are automatically assumed,
    and appropriate loss functions, decision functions and preprocessors
    for the target and sensitive features are chosen accordingly.
    For more information, visit the user guide. #FIXME how to reference?

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
        Either the string :code:`'auto'` or a callable. The string
        :code:`'auto'` indicates to infer the loss
        from the distribution type of :code:`y`
        (computed in :code:`dist_type_` of
        the transformer). A callable should be a
        loss function with an API that follows the chosen backend (torch or
        tensorflow). Note that torch and tensorflow loss functions don't agree
        on parameter order, as in Pytorch it is :math:`l(\hat y, y)` while in
        Tensorflow it is :math:`l(y, \hat y)`.

    adversary_loss : str, callable, default = 'auto'
        Either the string :code:`'auto'` or a callable. The string
        :code:`'auto'` indicates to infer the loss
        from the distribution type of :code:`sensitive_features`
        (computed in :code:`dist_type_` of
        the transformer). A callable should be a
        loss function with an API that follows the chosen backend (torch or
        tensorflow).

    predictor_function : str, callable, default = 'auto'
        Either the string :code:`'auto'` or a callable. The string
        :code:`'auto'` indicates to infer the predictor function
        from the distribution type of :code:`y`
        (computed in :code:`dist_type_` of
        the transformer). A callable should be a
        function that maps the continuous output of the predictor model to
        a discrete prediction. For example, if the transform of :code:`y` has
        has as attribute :code:`dist_type_=="binary", then map the predictors
        sigmoidal output :math:`y` to 1 iff :math:`y \geq \frac12`. Note
        that it is important that the model has sigmoidal outputs in this case.

    predictor_optimizer : str, torch.optim, tensorflow.keras.optimizers, default = 'Adam'
        The optimizer class to use. If a string is passed instead, this must be
        either "SGD" or "Adam". A corresponding SGD or Adam optimizer is
        initialized with the model and given learning rate.
        If not a string but an already initialized optimizer is passed, this
        optimizer is used instead.

    adversary_optimizer : str, torch.optim, tensorflow.keras.optimizers, default = 'Adam'
        The optimizer class to use. Similarly defined as
        :code:`predictor_optimizer`

    constraints : str, default = 'demographic_parity'
        The fairness measure to optimize for. Must be either 'demographic_parity'
        (Demographic Parity) or 'equalized_odds' (Equalized Odds).

    transformer : sklearn.base.TransformerMixin, default = fairlearn.adversarial.FloatTransformer
        The preprocessor to use on the predictions :code:`y`
        and :code:`sensitive_features`. The given transformer *must* map data
        to a 2d ndarray containing only floats. Per default, we use a
        FloatTransformer that maps strictly binary or categorical tables
        to binary or
        one-hot encodings, and it maps strictly continuous-valued (possible 2d)
        to itself.

    y_distribution : str, default = 'auto'
        This is a string that indicates an assumption about the distribution of
        the target values :math:`y`. Possible assumptions are
        "binary", "category", "continuous", "classification", "auto" (default).
        This assumption is fed to the transformer. If the predictions are not
        describable using one of these keywords, you must provide explicitely
        the following keywords: :code:`transformer`, :code:`predictor_loss`,
        :code:`predictor_function`, as the transformer, loss, and discrete
        prediction function cannot be automatically inferred, as the
        assumption about the distribution of :math:`y`
        can not be automatically inferred.

    z_distribution : str, default = 'auto'
        This is a string that indicates an assumption about the distribution of
        the sensitive features :math:`z`. Possible assumptions are
        "binary", "category", "continuous", "classification", "auto" (default).
        This assumption is fed to the transformer. If the predictions are not
        describable using one of these keywords, you must provide explicitely
        the following keywords: :code:`transformer`, :code:`adversary_loss`,
        as the transformer and loss function cannot be
        automatically inferred, as the
        assumption about the distribution of :math:`z` (sensitive features)
        can not be automatically inferred.

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

    """

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
        transformer=FloatTransformer,
        y_distribution="auto",
        z_distribution="auto",
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
        self.transformer = transformer
        self.y_distribution = y_distribution
        self.z_distribution = z_distribution
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

        # FIXME Want to get rid of this variable, but check_if_fitted can be expensive?
        self._setup = False

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

        kws = ["auto", "binary", "category", "continuous", "classification"]
        for kw, kwname in (
            (self.y_distribution, "y_distribution"),
            (self.z_distribution, "z_distribution"),
        ):
            if kw not in kws:
                raise ValueError(
                    _KWARG_ERROR_MESSAGE.format(
                        kwname,
                        "one of {}".format(kws),
                    )
                )

        for kw, kwname in (
            (self.predictor_loss, "predictor_loss"),
            (self.adversary_loss, "adversary_loss"),
            (self.predictor_function, "predictor_function"),
        ):
            # Some keywords can also be a callable instead of a str.
            if isinstance(kw, str) and kw != "auto":
                raise ValueError(
                    _KWARG_ERROR_MESSAGE.format(
                        kwname,
                        "auto or a callable",
                    )
                )

        if self.cuda and not isinstance(self.cuda, str):
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format(
                    "cuda",
                    'None, or the device to use (such as "cuda:0")',
                )
            )

        self.random_state_ = check_random_state(self.random_state)

        if not (
            self.transformer
            and hasattr(self.transformer, "fit")
            and hasattr(self.transformer, "transform")
            and issubclass(self.transformer, TransformerMixin)
        ):
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format(
                    "transformer",
                    "a transformer that implements TransformerMixin.",
                )
            )

        # Create preprocessors
        if self.y_distribution == "auto":
            self.y_transform_ = self.transformer()
        else:
            try:
                self.y_transform_ = self.transformer(
                    dist_assumption=self.y_distribution
                )
            except TypeError as e:
                e.value += _TRANSFORM_NO_PARAM
                raise e
        self.y_transform_.fit(Y)

        if self.z_distribution == "auto":
            self.z_transform_ = self.transformer()
        else:
            try:
                self.z_transform_ = self.transformer(
                    dist_assumption=self.z_distribution
                )
            except TypeError as e:
                e.value += _TRANSFORM_NO_PARAM
                raise e
        self.z_transform_.fit(Z)

        # Check if preprocessors supply correct information (no info is also ok)
        for transform in (self.y_transform_, self.z_transform_):
            if hasattr(
                transform, "dist_type_"
            ) and transform.dist_type_ not in (
                "binary",
                "category",
                "continuous",
            ):
                raise ValueError(
                    _TRANSFORMER_WEIRD_DIST_TYPE.format(
                        transform, transform.dist_type_
                    )
                )

        self._set_predictor_function()

        # Initialize backend
        # here, losses and optimizers are also set up.
        self.backendEngine_ = self.backend_(self, X, Y, Z)

        # Sklearn-parameters
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = self.y_transform_.n_features_in_

    def fit(self, X, y, *, sensitive_features):
        """
        Fit the model based on the given training data and sensitive features.

        Currently, for discrete y and sensitive_features
        *ALL* classes need to be passed in
        the first call to fit!

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional numpy array containing training data

        y : numpy.ndarray
            Two-dimensional numpy array containing training targets

        sensitive_features : numpy.ndarray
            Two-dimensional numpy array containing the sensitive features of the
            training data.
        """
        X, Y, Z = self._validate_input(X, y, sensitive_features)

        if self.batch_size == -1:
            self.batch_size = X.shape[0]
        batches = ceil(X.shape[0] / self.batch_size)

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
                    batch * self.batch_size,
                    min((batch + 1) * self.batch_size, X.shape[0]),
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

    def partial_fit(self, X, y, *, sensitive_features):
        """
        Perform one epoch on given samples and update model.

        Parameters
        ----------
        X : numpy.ndarray
            Two-dimensional numpy array containing training data

        y : numpy.ndarray
            Two-dimensional numpy array containing training targets

        sensitive_features : numpy.ndarray
            Two-dimensional numpy array containing the sensitive feature of the
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
            Two-dimensional array containing the model predictions
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
        Y_pred : numpy.ndarray
            Two-dimensional array containing the model predictions fed through
            the :code:`predictor_function`
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

        Provide a callable predictor function that maps soft-probabilities
        (or more precisely, predictor_model output) to discrete prediction.
        Alternatively, pass a Transformer with the :code:`transformer` key-word
        that has a :code:`dist_type_` attribute that a BackendEngine
        can use to interpret a loss function.
        """
        if callable(self.predictor_function):
            self.predictor_function_ = self.predictor_function
        elif hasattr(self.y_transform_, "dist_type_"):
            if self.y_transform_.dist_type_ == "binary":
                self.predictor_function_ = lambda pred: (pred >= 0.5).astype(
                    float
                )
            elif self.y_transform_.dist_type_ == "category":

                def loss(pred):
                    shape = pred.shape
                    c = argmax(pred, axis=1)
                    b = zeros(shape, dtype=float)
                    a = arange(shape[0])
                    b[a, c] = 1
                    return b

                self.predictor_function_ = loss
            elif self.y_transform_.dist_type_ == "continuous":
                self.predictor_function_ = lambda pred: pred
        if not self.predictor_function_:
            raise ValueError(_PREDICTION_FUNCTION_AMBIGUOUS)


class AdversarialFairnessClassifier(AdversarialFairness, ClassifierMixin):
    """Creates an AdversarialFairness with loss and predictions set to classification."""

    def __init__(self, **kwargs):
        """Initialize model by setting the predictor loss and function."""
        kwargs["y_distribution"] = kwargs.get(
            "y_distribution", "classification"
        )
        super(AdversarialFairnessClassifier, self).__init__(**kwargs)


class AdversarialFairnessRegressor(AdversarialFairness, RegressorMixin):
    """Create an AdversarialFairness that has predictor loss set to regression."""

    def __init__(self, *args, **kwargs):
        """Initialize model by setting the predictor loss."""
        kwargs["y_distribution"] = kwargs.get("y_distribution", "continuous")
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
