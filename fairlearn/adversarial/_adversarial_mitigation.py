# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from torch._C import Value
from ._constants import (
    _IMPORT_ERROR_MESSAGE,
    _KWARG_ERROR_MESSAGE,
    _PROGRESS_UPDATE,
    _NO_DATA,
)
from ._util import _get_function, interpret_keyword, Keyword, _check_array
from ._backend_engine import BackendEngine
from ._pytorch_engine import PytorchEngine
from ._tensorflow_engine import TensorflowEngine
from sklearn.base import ClassifierMixin, RegressorMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_random_state
from math import ceil
from time import time


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
    # TODO say something about easy-to-use List[int] models.

    There are three important key-words to identify the assumption about the
    underlying distribution of the data, which are:
    :code:`"binary"`, :code:`"category"`, or :code:`"continuous"`.
    Additionally, passing
    :code:`"classification"` will infer whether to use :code:`"binary"` or
    :code:`"category"`, and passing :code:`"auto"` will infer whether to use
    :code:`"binary"`, :code:`"category"`, or :code:`"continuous"`.

    We make the important design choice to allow for arbitrary loss functions
    :code:`predictor_loss` and :code:`adversary_loss` while
    also providing a simple interface for simple use cases. So, for simple use
    cases, the user may for example pass :code:`predictor_loss="binary"` to indicate
    that we assume that we are predicting binary data, and that we should use an
    instance of :mod:`torch.nn.BCEWithLogitsLoss` (or the
    tensorflow equivalent of binary cross entropy if the models are tensorflow
    models). Or, the user could pass an instance of
    :mod:`torch.nn.BCEWithLogitsLoss` directly to accomplish the same.
    Besides :code:`"binary"`, we provide loss functions for :code:`"category"`
    (categorical cross entropy)
    and :code:`"continuous"` (mean squared error) as well.
    We handle :code:`adversary_loss` in the same way.

    Additionally, one may pass one of these key-words as a :code:`predictor_function`.
    For instance, setting :code:`predictor_function="category"` maps the model's
    continuous output vector (assumped to be the logits or probabilities per
    category) to a discrete category prediction. However, we could also pass such
    a mapping function as :code:`predictor_function` directly, which may be
    helpful in complex use cases where we are predicting multiple discrete
    values from one model.

    When passing arbitrary functions as :code:`predictor_loss`, :code:`adversary_loss`,
    or :code:`predictor_function`, one has to take care that the underlying backend
    supports these functions. That is to say, one should not use
    TensorFlow loss functions with PyTorch models for instance, foremostly
    because a TensorFlow loss function may not work with PyTorch's automatic
    differentation, and TensorFlow loss functions expect their parameters differently
    than PyTorch loss functions expect their parameters.

    Parameters
    ----------
    backend : str, BackendEngine, default = 'auto'
        The backend to use. Must be one of :code:`'torch'`, :code:`'tensorflow'`,
        or :code:`'auto'` which indicates PyTorch, TensorFlow, or to
        automatically infer
        the backend from the :code:`predictor_model` and which are installed.
        You can also pass in a BackendEngine class.

    predictor_model : list, torch.nn.Module, tensorflow.keras.Model
        The predictor model to train. If a list of integers
        :math:`[n_1, n_2, \dots, n_k]` is passed, a fully
        connected neural network with sigmoidal activation functions is
        constructed with :math:`k` hidden layers that have :math:`n_i` nodes
        respectively. If :code:`backend` is specified, we cannot pass a model
        that uses a different backend.

    adversary_model : list, torch.nn.Module, tensorflow.keras.Model
        The adversary model to train. Must be the same type as the
        :code:`predictor_model`.

    predictor_loss : str, callable, default = 'auto'
        Either a string that indicates the type of :code:`y`,
        or :code:`'auto'` to infer the type of :code:`y`, or a callable
        loss function with an API that follows the chosen backend (torch or
        tensorflow). Note that torch and tensorflow loss functions don't agree
        on parameter order, as in Pytorch it is :math:`l(\hat y, y)` while in
        Tensorflow it is :math:`l(y, \hat y)`.

    adversary_loss : str, callable, default = 'auto'
        Either a string that indicates the type of :code:`sensitive_features`,
        or :code:`'auto'` to infer the type of :code:`sensitive_features`, or a
        callable loss function with an API that follows the chosen backend
        (torch or tensorflow). Note that torch and tensorflow loss functions
        don't agree on parameter order.

    predictor_function : str, callable, default='auto'
        Either a string that indicates the type of :code:`y`,
        or :code:`'auto'` to infer the type of :code:`y`, or a callable
        prediction function maps the continuous output of the predictor model to
        a discrete prediction. For example, passing 'binary' maps the predictor's
        sigmoidal output :math:`y` to 1 iff :math:`y \geq \frac12`.

    constraints : str, default = 'demographic_parity'
        The fairness measure to optimize for. Must be either 'demographic_parity'
        (Demographic Parity) or 'equalized_odds' (Equalized Odds).

    predictor_optimizer : str, torch.optim, tensorflow.keras.optimizers, default = 'Adam'
        The optimizer class to use. If a string is passed instead, this must be
        either "SGD" or "Adam". A corresponding SGD or Adam optimizer is
        initialized with the model and given learning rate.
        If not a string but an already initialized optimizer is passed, this
        optimizer is used instead.
    
    adversary_optimizer : str, torch.optim, tensorflow.keras.optimizers, default = 'Adam'
        The optimizer class to use. Similarly defined as
        :code:`predictor_optimizer`

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

    progress_updates : number, optional
        If a number :math:`t` is provided, we print an update
        about the training loop after processing a batch and :math:`t` seconds
        have passed since the previous update.

    skip_validation : bool
        Skip the validation of the data. Useful because validate_input is
        a costly operation, and we may instead pass all data to validate_input
        at an earlier stage.

    callback_fn : callable
        Callback function, called every epoch. For instance useable when
        wanting to validate. Should take zero arguments.

    cuda : str, default = None
        A string to indicate which device to use when training. For instance,
        set :code:`cuda='cuda:0'` to train on the first GPU. Only for PyTorch
        backend.
    
    warm_start : bool, default=False
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

    Examples
    --------
    
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
        self.backend = backend
        self.predictor_model = predictor_model
        self.adversary_model = adversary_model
        self.predictor_loss = predictor_loss
        self.adversary_loss = adversary_loss
        self.predictor_function = predictor_function
        self.predictor_optimizer = predictor_optimizer
        self.adversary_optimizer = adversary_optimizer
        self.constraints = constraints
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

        # Want to get rid of this variable, but check_if_fitted can be expensive?
        self._setup = False

    def __setup(self, X, Y, Z):
        """
        Setup the entire model from the parameters and the given data.

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
            (self.learning_rate, 'learning_rate'),
            (self.alpha, 'alpha'),
            (self.epochs, 'epochs'),
            (self.progress_updates, 'progress_updates')
        ):
            if kw and kw <= 0.:
                raise ValueError(
                    _KWARG_ERROR_MESSAGE.format(kwname, 'a positive number')
                )

        self.random_state_ = check_random_state(self.random_state)

        self.predictor_function_ = _get_function(
            interpret_keyword(
                Y, self.predictor_function, "predictor_function", "y"
            )
        )

        # Initialize backend
        self.backendEngine_ = self.backend_(self, X, Y, Z)

        self._setup = True

        # Sklearn-parameters
        self.n_features_in_ = Y.shape[1]

    def fit(self, X, y, *, sensitive_features):
        """
        Fit the model based on the given training data and sensitive features.

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

                if callable(self.callback_fn):
                    stop = self.callback_fn(self, epoch, batch)
                    if stop:
                        return
            if self.shuffle and epoch != self.epochs - 1:
                X, Y, Z = self.backendEngine_.shuffle(X, Y, Z)

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
        if not self._setup:
            raise UserWarning(_NO_DATA)

        check_is_fitted(self)
        X = _check_array(X)

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
        if not self._setup:
            raise UserWarning(_NO_DATA)

        check_is_fitted(self)
        X = _check_array(X)

        Y_pred = self.backendEngine_.evaluate(X)
        Y_pred = self.predictor_function_(Y_pred)
        return Y_pred

    def _validate_input(self, X, Y, Z):
        """
        Validate the input data and possibly setup this estimator.
        
        Important note is that we follow call `__setup` from here, because the
        setup procedure requires the validated data.
        """
        if not self.skip_validation:
            X = _check_array(X)
            Y = _check_array(Y)
            Z = _check_array(Z)

            X = X.astype(float)
            Y = Y.astype(float)
            Z = Z.astype(float)

        # Check for equal number of samples
        if not (X.shape[0] == Y.shape[0] and X.shape[0] == Z.shape[0]):
            raise ValueError(
                "Input data has an ambiguous number of rows: {}, {}, {}.".format(
                    X.shape[0], Y.shape[0], Z.shape[0]
                )
            )

        if (not self._setup) or (not self.warm_start):
            self.__setup(X, Y, Z)

        if not self.skip_validation:
            # Some backendEngine may want to do some additional preprocessing,
            # such as moving to GPU.
            # FIXME: Maybe we move this to backendEngine_.prepareData?
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
        # Or all other cases
        raise ValueError(
            _KWARG_ERROR_MESSAGE.format(
                "predictor_model and adversary_model",
                "a list, torch.nn.Module, or tensorflow.keras.Model. Also, "
                + "make sure to have installed the corresponding backend",
            )
        )


class AdversarialFairnessClassifier(AdversarialFairness, ClassifierMixin):
    """Creates an AdversarialFairness with loss and predictions set to classification."""

    def __init__(self, **kwargs):
        """Initialize model by setting the predictor loss and function."""
        kwargs["predictor_loss"] = kwargs.get(
            "predictor_loss", Keyword.CLASSIFICATION.value
        )
        kwargs["predictor_function"] = kwargs.get(
            "predictor_function", Keyword.CLASSIFICATION.value
        )
        super(AdversarialFairnessClassifier, self).__init__(**kwargs)


class AdversarialFairnessRegressor(AdversarialFairness, RegressorMixin):
    """Create an AdversarialFairness that has predictor loss set to regression."""

    def __init__(self, *args, **kwargs):
        """Initialize model by setting the predictor loss."""
        kwargs["predictor_loss"] = kwargs.get(
            "predictor_loss", Keyword.CONTINUOUS.value
        )
        kwargs["predictor_function"] = kwargs.get(
            "predictor_function", Keyword.CONTINUOUS.value
        )
        super(AdversarialFairnessRegressor, self).__init__(*args, **kwargs)
