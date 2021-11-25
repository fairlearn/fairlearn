# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

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
    backend : str, default = 'auto'
        The backend to use. Must be one of :code:`'torch'`, :code:`'tensorflow'`,
        or :code:`'auto'` which indicates PyTorch, TensorFlow, or to automatically infer
        the backend from the :code:`predictor_model` and which are installed.

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
        on parameter order.

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
        output logits :math:`y` to 1 iff :math:`y \geq 0`.

    constraints : str, default = 'demographic_parity'
        The fairness measure to optimize for. Must be either 'demographic_parity'
        (Demographic Parity) or 'equalized_odds' (Equalized Odds).

    optimizer : str, torch.optim, tensorflow.keras.optimizers, default = 'Adam'
        The optimizer class to use. If a string is passed instead, this must be
        either "SGD" or "Adam".

    learning_rate : float, default = 0.001
        A small number greater than zero to set as initial learning rate

    alpha : float, default = 1.0
        A small number $\alpha$ as specified in the paper.

    cuda : str, default = None
        A string to indicate which device to use when training. For instance,
        set :code:`cuda='cuda:0'` to train on the first GPU. Only for PyTorch
        backend.

    epochs : int, default = 1
        Number of epochs to train for.

    batch_size : int, default = -1
        Batch size. For no batching, set this to -1.

    shuffle : bool, default = False
        Iff True, shuffle the data after every iteration. Default is False

    progress_updates : number, optional
        If a number :math:`t` is provided, we regularly print an update
        about the training loop after at least every :math:`t` seconds.

    skip_validation : bool
        Skip the validation of the data. Useful because validate_input is
        a costly operation, and we may instead pass all data to validate_input
        at an earlier stage.

    callback_fn : callable
        Callback function, called every epoch. For instance useable when
        wanting to validate. Should take zero arguments.

    References
    ----------
    .. [1] Zhang, Lemoine, Mitchell `"Mitigating Unwanted Biases with
       Adversarial Learning" <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_,
       AIES, 2018.

    Examples
    --------
    >>> from fairlearn.adversarial import AdversarialFairness
    >>> from fairlearn.metrics import MetricFrame, selection_rate
    >>> from numpy import number
    >>> from sklearn.metrics import accuracy_score
    >>> from sklearn.compose import make_column_transformer, make_column_selector
    >>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
    >>> from sklearn.datasets import fetch_openml
    >>> from pandas import Series
    >>>
    >>> mitigator = AdversarialFairness(
    ...     predictor_model=[50, 20],
    ...     adversary_model=[6, 6],
    ...     learning_rate=0.0001,
    ...     epochs=100,
    ...     batch_size=2**9,
    ...     shuffle=True,
    ...     progress_updates=5
    ... )
    >>>
    >>> def transform(X):
    ...     if isinstance(X, Series):
    ...         X = X.to_frame()
    ...     ct = make_column_transformer(
    ...         (StandardScaler(),
    ...          make_column_selector(dtype_include=number)),
    ...         (OneHotEncoder(drop='if_binary', sparse=False),
    ...          make_column_selector(dtype_include="category")))
    ...     return ct.fit_transform(X)
    ...
    >>> X, y = fetch_openml(data_id=1590, as_frame=True, return_X_y=True)
    >>> non_NaN_rows = ~X.isna().any(axis=1)
    >>> X, y = X[non_NaN_rows], y[non_NaN_rows]
    >>> sensitive_feature = X['sex']
    >>>
    >>> X = transform(X)
    >>> y = transform(y)
    >>> sensitive_feature = transform(sensitive_feature)
    >>>
    >>> mitigator.fit(
    ...     X,
    ...     y,
    ...     sensitive_features=sensitive_feature
    ... )
    |==>                  | Epoch: 12/100, Batch: 50/89, ETA: 38.30 sec. Loss (pred/adv): 0.55/0.63
    |=====>               | Epoch: 24/100, Batch: 35/89, ETA: 32.78 sec. Loss (pred/adv): 0.57/0.62
    |=======>             | Epoch: 36/100, Batch: 54/89, ETA: 27.15 sec. Loss (pred/adv): 0.56/0.61
    |==========>          | Epoch: 48/100, Batch: 68/89, ETA: 21.89 sec. Loss (pred/adv): 0.37/0.63
    |============>        | Epoch: 60/100, Batch: 81/89, ETA: 16.74 sec. Loss (pred/adv): 0.34/0.58
    |==============>      | Epoch: 73/100, Batch: 12/89, ETA: 11.60 sec. Loss (pred/adv): 0.38/0.58
    |=================>   | Epoch: 85/100, Batch: 13/89, ETA: 6.60 sec. Loss (pred/adv): 0.36/0.65
    |===================> | Epoch: 97/100, Batch: 2/89, ETA: 1.66 sec. Loss (pred/adv): 0.35/0.64
    >>>
    >>> y_pred = mitigator.predict(X)
    >>>
    >>> mf = MetricFrame(
    ...     metrics={
    ...         'accuracy': accuracy_score,
    ...         'selection_rate': selection_rate},
    ...     y_true=y,
    ...     y_pred=y_pred,
    ...     sensitive_features=sensitive_feature)
    >>>
    >>> print(mf.by_group)
                        accuracy selection_rate
    sensitive_feature_0
    0.0                  0.906295       0.147397
    1.0                  0.809447       0.228322
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
        constraints="demographic_parity",
        predictor_optimizer="Adam",
        adversary_optimizer="Adam",
        learning_rate=0.001,
        alpha=1.0,
        cuda=None,
        epochs=1,
        batch_size=-1,
        shuffle=False,
        progress_updates=None,
        skip_validation=False,
        callback_fn=None,
        warm_start=False,
        random_state=None,
    ):
        self.backend = backend
        self.predictor_model = predictor_model
        self.adversary_model = adversary_model
        self.predictor_loss = predictor_loss
        self.adversary_loss = adversary_loss
        self.predictor_function = predictor_function
        self.constraints = constraints
        self.predictor_optimizer = predictor_optimizer
        self.adversary_optimizer = adversary_optimizer
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.cuda = cuda
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.progress_updates = progress_updates
        self.skip_validation = skip_validation
        self.callback_fn = callback_fn
        self.warm_start = warm_start
        self.random_state = random_state

        # Want to get rid of this variable, but check_if_fitted can be expensive?
        self._setup = False

    def setup(self, X, Y, Z):
        """Finalize setup that is required as soon as the first data is given."""
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
                    self.callback_fn(self, epoch, batch)
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
        if not self._setup:
            raise UserWarning(_NO_DATA)

        check_is_fitted(self)
        X = _check_array(X)

        Y_pred = self.backendEngine_.evaluate(X)
        return Y_pred

    def predict(self, X):
        """
        Gather predictions for given test data.

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
        """Validate the input data and possibly setup this estimator."""
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
            self.setup(X, Y, Z)

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

        Better description.
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
