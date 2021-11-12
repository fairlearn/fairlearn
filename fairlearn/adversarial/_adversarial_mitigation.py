# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

from ._constants import _IMPORT_ERROR_MESSAGE, _KWARG_ERROR_MESSAGE, \
    _PROGRESS_UPDATE, _TYPE_CHECK_ERROR
from numpy import finfo, float32, ndarray, zeros, argmax, logical_or, arange
from numpy import all as np_all
from numpy import sum as np_sum
from sklearn.utils import check_array, shuffle
from math import ceil
from time import time

torch = None
tf = None

# TODO Is this a sensible way?
class Keyword:
    AUTO = "auto"
    CLASSIFICATION = "classification"
    BINARY = "binary"
    CATEGORY = "category"
    CONTINUOUS = "continuous"

    TYPES = [AUTO, CLASSIFICATION, BINARY, CATEGORY, CONTINUOUS]

class AdversarialFairness():
    r"""Train complex predictors while mitigating biases in PyTorch or Tensorflow.

    This algorithm is our implementation of the supervised learning method proposed in
    `"Mitigating Unwanted Biases with Adversarial Learning"
    <https://dl.acm.org/doi/pdf/10.1145/3278721.3278779>`_ [1]_.
    This algorithm takes as input two neural network
    models, a predictor and an adversarial, defined either as a `PyTorch module
    <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_ or
    `Tensorflow2 model
    <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_. You train this
    predictor using an API that is similar to estimators in `sklearn`.

    To summarize [1]_, assume we have data :math:`X, Y, Z`, where we want to
    predict :math:`Y` from :math:`X` while being fair with respect to :math:`Z`
    w.r.t. some fairness measure. We firstly create predictor and adversary neural
    networks with learnable weights :math:`W` and :math:`U` respectively. Without
    considering fairness yet, this typical supervised-learning problem aims to
    minimize the predictor loss :math:`L_P`. Now, to improve fairness, we not
    only want to minimize the predictor loss, but we want to decrease the
    adversary's ability to predict the sensitive features from the predictor's
    predictions. Suppose the adversary has loss term :math:`L_A`, then the paper
    trains the predictor with gradient:

    .. math::
        \nabla_W L_P - \text{proj}_{\nabla_W L_A} \nabla_W L_P - \alpha \nabla_W L_A

    In this implementation, we accept :math:`X, Y, Z` as 1d or 2d array-like. We
    provide both a full-fledged training loop by using :func:`AdversarialFairness.fit`,
    though one may also implement their own training loop by continuously calling
    :func:`AdversarialFairness.fit` or :func:`AdversarialFairness.partial_fit`.

    As per [1]_, the neural network models :code:`predictor_model` and
    :code:`adversary_model` may not have an activation function at the end of the
    model. Instead, even if we are dealing with discrete predictions, we must output
    logits. # TODO say something about easy-to-use List[int] models.

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
    or :code:`predictor_function`, one has to take care that the underlying library
    supports these functions. That is to say, one should not use
    TensorFlow loss functions with PyTorch models for instance, foremostly
    because a TensorFlow loss function may not work with PyTorch's automatic
    differentation, and TensorFlow loss functions expect their parameters differently
    than PyTorch loss functions expect their parameters.

    Parameters
    ----------
    library : str, default = 'auto'
        The library to use. Must be one of :code:`'torch'`, :code:`'tensorflow'`,
        or :code:`'auto'` which indicates PyTorch, TensorFlow, or to automatically infer
        the library from the :code:`predictor_model` and which are installed.

    predictor_model : list, torch.nn.Module, tensorflow.keras.Model
        The predictor model to train. If a list of integers
        :math:`[n_1, n_2, \dots, n_k]` is passed, a fully
        connected neural network with sigmoidal activation functions is
        constructed with :math:`k` hidden layers that have :math:`n_i` nodes
        respectively. If :code:`library` is specified, we cannot pass a model
        that uses a different library.

    adversary_model : list, torch.nn.Module, tensorflow.keras.Model
        The adversary model to train. Must be the same type as the
        :code:`predictor_model`.

    predictor_loss : str, callable, default = 'auto'
        Either a string that indicates the type of :code:`y`,
        or :code:`'auto'` to infer the type of :code:`y`, or a callable
        loss function with an API that follows the chosen library (torch or
        tensorflow). Note that torch and tensorflow loss functions don't agree
        on parameter order.

    adversary_loss : str, callable, default = 'auto'
        Either a string that indicates the type of :code:`sensitive_features`,
        or :code:`'auto'` to infer the type of :code:`sensitive_features`, or a
        callable loss function with an API that follows the chosen library
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

    cuda : bool, default = False
        A boolean to indicate whether we can use cuda:0 (first GPU) when training
        a PyTorch model.

    Notes
    -----
    Hey
    # TODO

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
    ...     learning_rate=0.0001
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
    ...     sensitive_features=sensitive_feature,
    ...     epochs=100,
    ...     batch_size=2**9,
    ...     shuffle=True,
    ...     progress_updates=5
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

    def __init__(self, *,
                 library='auto',
                 predictor_model,
                 adversary_model,
                 predictor_loss='auto',
                 adversary_loss='auto',
                 predictor_function='auto',
                 constraints='demographic_parity',
                 optimizer='Adam',
                 predictor_optimizer=None,
                 adversary_optimizer=None,
                 learning_rate=0.001,
                 alpha=1.0,
                 cuda=False
                 ):
        """Initialize Adversarial Mitigation."""
        self._init_models(library, predictor_model, adversary_model)

        # Inherit library-specific code
        if self.torch:
            self._extend_instance(AdversarialPytorchMixin)
        elif self.tensorflow:
            self._extend_instance(AdversarialTensorflowMixin)

        self._init_losses(predictor_loss, adversary_loss, predictor_function)
        self._init_constraints(constraints)
        self.optimizer = optimizer
        self.predictor_optimizer = predictor_optimizer
        self.adversary_optimizer = adversary_optimizer
        self.learning_rate = learning_rate
        self.alpha = alpha
        self._init_cuda(cuda)

        # The design is very data-dependent. Instead of having users specify
        # every nitty-gritty detail of the data they will pass, we will try to
        # infer as much detail as possible the first time that data is passed
        self.setup_with_data_ = False

    def fit(self, X, y, *, sensitive_features,
            epochs=1,
            batch_size=-1,
            shuffle=False,
            progress_updates=None,
            skip_validation=False,
            callback_fn=None):
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
        """
        X, Y, Z = self._validate_input(X, y, sensitive_features)
        # TODO decreasing learning rate: not really necessary with adam
        # TODO stopping condition!? If |grad| < eps
        if batch_size == -1:
            batch_size = X.shape[0]
        batches = ceil(X.shape[0] / batch_size)

        start_time = time()
        last_update_time = start_time
        predictor_losses = [None]
        adversary_losses = []
        for epoch in range(epochs):
            for batch in range(batches):
                if progress_updates:
                    if (time() - last_update_time) > progress_updates:
                        last_update_time = time()
                        progress = (epoch / epochs) + (batch / (batches * epochs))
                        print(_PROGRESS_UPDATE.format(
                            "=" * round(20 * progress),
                            " " * round(20 * (1 - progress)),
                            epoch + 1, epochs,
                            " " * (len(str(batch + 1)) - len(str(batches))),
                            batch + 1, batches,
                            ((last_update_time - start_time) / progress) * (1 - progress),
                            predictor_losses[-1],
                            adversary_losses[-1]
                        ), end='\n')
                batch_slice = slice(batch * batch_size, min((batch + 1) * batch_size, X.shape[0]))
                (LP, LA) = self._train_step(X[batch_slice],
                                            Y[batch_slice],
                                            Z[batch_slice])
                predictor_losses.append(LP)
                adversary_losses.append(LA)
            if shuffle and epoch != epochs - 1:
                X, Y, Z = self._shuffle(X, Y, Z)
            if callable(callback_fn):
                callback_fn()

    def _shuffle(self, X, Y, Z):
        """Shuffle the rows of X, Y, Z."""
        X, Y, Z = shuffle(X, Y, Z)
        return X, Y, Z

    def partial_fit(self, X, y, *, sensitive_features, skip_validation=False):
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

        skip_validation : bool
            Skip the validation of the data. Useful because validate_input is
            a costly operation, and we may instead pass all data to validate_input
            at an earlier stage.
        """
        X, Y, Z = X, y, sensitive_features
        if not skip_validation:
            X, Y, Z = self._validate_input(X, Y, Z)
        self._train_step(X, Y, Z)

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
        if not self._initialized_models:
            raise RuntimeError(
                "Models have not been initialized, as we have seen no data yet.")
        if not self._setup_with_data:
            raise UserWarning("Havent seen data yet")

        X = _check_array(X)

        Y_pred = self._evaluate(X)

        assert Y_pred.ndim == 2

        Y_pred = self.predictor_function(Y_pred)

        return Y_pred

    def _check_type(self, Y, choice, data_name):
        """Identify user query :code:`choice`."""
        if choice == Keyword.CLASSIFICATION:
            if np_all(logical_or(Y == 0, Y == 1)):
                if Y.shape[1] == 1:
                    return Keyword.BINARY
                elif Y.shape[1] > 1:
                    if np_all(np_sum(Y, axis=1) == 1):
                        return Keyword.CATEGORY
        elif choice == Keyword.BINARY:
            if np_all(logical_or(Y == 0, Y == 1)):
                if Y.shape[1] == 1:
                    return Keyword.BINARY
        elif choice == Keyword.CATEGORY:
            if np_all(logical_or(Y == 0, Y == 1)):
                if Y.shape[1] > 1:
                    if np_all(np_sum(Y, axis=1) == 1):
                        return Keyword.CATEGORY
        elif choice == Keyword.AUTO:
            if np_all(logical_or(Y == 0, Y == 1)):
                if Y.shape[1] == 1:
                    return Keyword.BINARY
                else:
                    if np_all(np_sum(Y, axis=1) == 1):
                        return Keyword.CATEGORY
            else:
                return Keyword.CONTINUOUS
        elif choice == Keyword.CONTINUOUS:
            return choice
        # This means choice was one of the above but the data disagrees
        raise ValueError(_TYPE_CHECK_ERROR.format(data_name, choice))

    def _interpret_choice(self, Y, choice, data_name):
        """Infer choice from either a function or Keyword.BINARY, Keyword.CATEGORY, or Keyword.CONTINUOUS."""
        if callable(choice):
            return choice
        elif isinstance(choice, str):
            choice = self._check_type(Y, choice, data_name)
            if choice == Keyword.BINARY or choice == Keyword.CATEGORY or choice == Keyword.CONTINUOUS:
                return choice
        raise ValueError(_KWARG_ERROR_MESSAGE.format(
            "choice", "one of {} or a callable".format(Keyword.TYPES)))        

    def _get_function(self, Y, choice, data_name):
        """Infer prediction function."""
        choice = self._interpret_choice(Y, choice, data_name)
        if choice == Keyword.BINARY:
            return lambda pred: (pred >= 0.).astype(float)
        elif choice == Keyword.CATEGORY:
            shape = Y.shape

            def loss(pred):
                c = argmax(pred, axis=1)
                b = zeros(shape, dtype=float)
                a = arange(shape[0])
                b[a, c] = 1
                return b
            return loss
        elif choice == Keyword.CONTINUOUS:
            return lambda pred: pred
        else:
            return choice
    def _setup_with_data(self, X, Y, Z):
        """Finalize setup that is required as soon as the first data is given."""
        self.setup_with_data_ = True

        # Initialize models if not done yet
        if not self._initialized_models:
            predictor_list_nodes = [X.shape[1]] + self.predictor_model + [Y.shape[1]]
            adversarial_in = Y.shape[1] * (2 if self.pass_y else 1)
            adversary_list_nodes = [adversarial_in] + self.adversary_model + [Z.shape[1]]
            if self.torch:
                from ._models import getTorchModel as getModel
            elif self.tensorflow:
                from ._models import getTensorflowModel as getModel

            self.predictor_model = getModel(list_nodes=predictor_list_nodes)
            self.adversary_model = getModel(list_nodes=adversary_list_nodes)
            self._initialized_models = True

        # Setup losses, if they are set to 'auto'
        self.predictor_loss = self._get_loss(Y, self.predictor_loss, "y")
        self.adversary_loss = self._get_loss(Z, self.adversary_loss, "sensitive_features")
        self.predictor_function = self._get_function(Y, self.predictor_function, "y")

        # Setup optimizers, because now we definitely have models set up
        self._setup_optimizer(self.optimizer)

        # Use CUDA
        if self.cuda:
            self.adversary_model = self.adversary_model.to(self.device)
            self.predictor_model = self.predictor_model.to(self.device)

    def _validate_input(self, X, Y, Z):
        """Validate the input data."""
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
                    X.shape[0], Y.shape[0], Z.shape[0]))

        if not self.setup_with_data_:
            self._setup_with_data(X, Y, Z)

        return X, Y, Z

    def _init_models(self, library, predictor_model, adversary_model):
        """
        Import either PyTorch or Tensorflow, depending on predictor.

        if library is 'auto', then infer from predictor_model. If predictor_model
        is a list, then choose torch or tensorflow, depending on which is installed.
        """
        # The library to use
        self.torch = False
        self.tensorflow = False

        # Discover which librarys are available
        torch_installed = False
        tf_installed = False
        global torch
        global tf
        try:
            import torch
            torch_installed = True
        except ImportError:
            pass
        try:
            import tensorflow as tf
            tf_installed = True
        except ImportError:
            pass

        # At this point, either tensorflow or torch is installed
        if library == 'torch':
            if not torch_installed:
                raise RuntimeError(_IMPORT_ERROR_MESSAGE.format("torch"))
            self.torch = True
        elif library == 'tensorflow':
            if not tf_installed:
                raise RuntimeError(_IMPORT_ERROR_MESSAGE.format("tensorflow"))
            self.tensorflow = True
        elif library == 'auto':
            if (not torch_installed) and (not tf_installed):
                raise RuntimeError(_IMPORT_ERROR_MESSAGE.format("torch or tensorflow"))
            if isinstance(predictor_model, list):
                if torch_installed:
                    self.torch = True
                elif tf_installed:
                    self.tensorflow = True
            elif torch_installed and isinstance(predictor_model, torch.nn.Module):
                self.torch = True
            elif tf_installed and isinstance(predictor_model, tf.keras.Model):
                self.tensorflow = True
        else:
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format(
                    'library',
                    "one of [\'auto\', \'torch\',\'tensorflow\']"))

        if not (self.torch or self.tensorflow):
            raise ValueError(
                _KWARG_ERROR_MESSAGE.format(
                    'predictor_model',
                    "one of [\'list\', \'torch\',\'tensorflow\']"))

        # At this point, either self.torch or self.tensorflow is selected
        if isinstance(predictor_model, list):
            if not isinstance(adversary_model, list):
                raise ValueError(_KWARG_ERROR_MESSAGE.format(
                    "adversary_model", "a list"))
            self._initialized_models = False
        else:
            if isinstance(predictor_model, torch.nn.Module):
                if not isinstance(adversary_model, torch.nn.Module):
                    raise ValueError(_KWARG_ERROR_MESSAGE.format(
                        "adversary_model", "a \'torch.nn.Module\'"))

            if isinstance(predictor_model, tf.keras.Model):
                if not isinstance(adversary_model, tf.keras.Model):
                    raise ValueError(_KWARG_ERROR_MESSAGE.format(
                        "adversary_model", "a \'tensorflow.keras.Model\'"))
            self._initialized_models = True
        # Note, if initialized_models == False, the model will still be a list
        self.predictor_model = predictor_model
        self.adversary_model = adversary_model

    def _verify_dist_type(self, input_type, dist_name):
        if callable(input_type):
            return input_type
        elif isinstance(input_type, str):
            if input_type.lower() in Keyword.TYPES:
                return input_type.lower()
        raise ValueError(_KWARG_ERROR_MESSAGE.format(
            dist_name, "one of {} or a callable".format(Keyword.TYPES)))

    def _init_losses(self, predictor_loss, adversary_loss, predictor_function):
        self.predictor_loss = self._verify_dist_type(predictor_loss, "predictor_loss")
        self.adversary_loss = self._verify_dist_type(adversary_loss, "adversary_loss")
        self.predictor_function = self._verify_dist_type(predictor_function, "predictor_function")

    def _init_constraints(self, constraints):
        """Verify the constraints and set up the corresponding network structure."""
        self.constraints = constraints
        if (constraints == "demographic_parity"):
            self.pass_y = False
        elif (constraints == "equalized_odds"):
            self.pass_y = True
        else:
            raise ValueError(_KWARG_ERROR_MESSAGE.format(
                "constraints", "\'demographic_parity\' or \'equalized_odds\'"))

    def _init_cuda(self, cuda):
        """Verify whether we can use the GPU and move pytorch model to it."""
        if (not cuda):
            self.cuda = False
        elif (cuda):
            if (not self.torch):
                raise ValueError("Cuda is PyTorch-only")
            if not torch.cuda.is_available():
                raise ValueError("Cuda is not available")
            self.cuda = True
            self.device = torch.device("cuda:0")

    def _extend_instance(self, cls_):
        """Apply mixins to a class instance after creation."""
        base_cls = self.__class__
        base_cls_name = self.__class__.__name__
        self.__class__ = type(base_cls_name, (cls_, base_cls), {})


def _check_array(X):
    """
    Validate the input array, and possible coerce to 2D.

    Calls :code:`sklearn.utils.check_array` on parameter X with the
    parameters suited for Adversarial Mitigation.
    """
    return check_array(
        X, accept_sparse=False, accept_large_sparse=False,
        dtype=float, order=None, copy=False, force_all_finite=True,
        ensure_2d=True, allow_nd=False, ensure_min_samples=1,
        ensure_min_features=1, estimator=None
    )


class AdversarialMixin():
    """The interface of a mixin class."""

    def _evaluate(self, X: ndarray) -> ndarray:
        """
        Evaluate the model given input `X`.

        Feed 2d `numpy.ndarray` through model and receive output as
        2d `numpy.ndarray`.
        """
        pass

    def _train_step(self, X: ndarray, Y: ndarray, Z: ndarray):
        """
        Perform one training step over data.

        Returns
        -------
        (LP, LA) : tuple of (float, float)
            predictor loss and adversary loss.
        """
        pass

    def _setup_optimizer(self, optimizer):
        """
        Initialize the optimizers.

        Setup self.predictor_optimizer and self.adversary_optimizer using the
        parameter optimizer given by the user.
        """
        if not isinstance(optimizer, str) and \
                (not self.predictor_optimizer or not self.adversary_optimizer):
            raise ValueError(_KWARG_ERROR_MESSAGE.format(
                "optimizer",
                "'Adam' or 'SGD' for standard optimizers."
                + " Or, predictor_optimizer and adversary_optimizer are optimizers."
            ))
        if (self.predictor_optimizer or self.adversary_optimizer) \
                and isinstance(optimizer, str):
            raise ValueError("Either use kwarg optimizer or use kwargs "
                             + "predictor_optimizer and adversary_optimizer")
        if (self.predictor_optimizer and not self.adversary_optimizer) or \
                (self.adversary_optimizer and not self.predictor_optimizer):
            raise ValueError("Either define both predictor_optimizer and" +
                             " adversary_optimizer explicitely, or define" +
                             " none and use kwarg optimizer")

    def _get_loss(self, Y, choice, data_name):
        """Infer loss from ndarray Y under assumption choice."""
        pass

class AdversarialPytorchMixin(AdversarialMixin):
    """Adds PyTorch specific functions."""

    def _shuffle(self, X, Y, Z):
        """Override base's shuffle to work with `torch.FloatTensor`."""
        idx = torch.randperm(X.shape[0])
        X = X[idx].view(X.size())
        Y = Y[idx].view(Y.size())
        Z = Z[idx].view(Z.size())
        return X, Y, Z

    def _evaluate(self, X):
        """
        Evaluate the model given input `X`.

        Feed 2d `numpy.ndarray` through model and receive output as
        2d `numpy.ndarray`.
        """
        self.predictor_model.eval()
        X = torch.from_numpy(X).float()
        if self.cuda:
            X = X.to(self.device)
        with torch.no_grad():
            Y_pred = self.predictor_model(X)
        if self.cuda:
            Y_pred = Y_pred.detach().cpu().numpy()
        else:
            Y_pred = Y_pred.numpy()
        return Y_pred

    def _train_step(self, X, Y, Z):
        """
        Perform one training step over data in PyTorch models.

        Returns
        -------
        (LP, LA) : tuple of (float, float)
            predictor loss and adversary loss.
        """
        self.predictor_model.train()
        self.adversary_model.train()

        # Clear gradient
        self.predictor_optimizer.zero_grad()
        self.adversary_optimizer.zero_grad()

        Y_hat = self.predictor_model(X)
        LP = self.predictor_loss(Y_hat, Y)
        LP.backward(retain_graph=True)  # Check what this does at some point in time

        dW_LP = [torch.clone(p.grad.detach()) for p in self.predictor_model.parameters()]

        self.predictor_optimizer.zero_grad()
        self.adversary_optimizer.zero_grad()

        # For equalized odds
        if self.pass_y:
            Y_hat = torch.cat((Y_hat, Y), dim=1)

        Z_hat = self.adversary_model(Y_hat)
        LA = self.adversary_loss(Z_hat, Z)
        LA.backward()

        dW_LA = [torch.clone(p.grad.detach()) for p in self.predictor_model.parameters()]

        for i, p in enumerate(self.predictor_model.parameters()):
            # Normalize dW_LA
            unit_dW_LA = dW_LA[i] / (torch.norm(dW_LA[i]) + torch.finfo(float).tiny)
            # Project
            proj = torch.sum(torch.inner(unit_dW_LA, dW_LP[i]))
            # Calculate dW
            p.grad = dW_LP[i] - (proj * unit_dW_LA) - (self.alpha * dW_LA[i])

        self.predictor_optimizer.step()
        self.adversary_optimizer.step()

        return (LP.item(), LA.item())

    def _setup_optimizer(self, optimizer):
        """
        Create the optimizers for PyTorch.

        Setup self.predictor_optimizer and self.adversary_optimizer using the
        parameter optimizer given by the user.
        """
        super(AdversarialPytorchMixin, self)._setup_optimizer(optimizer)
        if isinstance(optimizer, str):
            # keyword cases.
            if optimizer.lower() == "adam":
                self.predictor_optimizer = torch.optim.Adam(
                    self.predictor_model.parameters(), lr=self.learning_rate)
                self.adversary_optimizer = torch.optim.Adam(
                    self.adversary_model.parameters(), lr=self.learning_rate)
            elif optimizer.lower() == "sgd":
                self.predictor_optimizer = torch.optim.SGD(
                    self.predictor_model.parameters(), lr=self.learning_rate)
                self.adversary_optimizer = torch.optim.SGD(
                    self.adversary_model.parameters(), lr=self.learning_rate)
            else:
                raise ValueError(_KWARG_ERROR_MESSAGE.format(
                    'optimizer', '"Adam" or "SGD" or an optimizer'))
    
    def _get_loss(self, Y, choice, data_name):
        """Infer loss."""
        choice = self._interpret_choice(Y, choice, data_name)
        if callable(choice):
            return choice
        if choice == Keyword.BINARY:
            return torch.nn.BCEWithLogitsLoss(reduction='mean')
        elif choice == Keyword.CATEGORY:
            return torch.nn.CrossEntropyLoss(reduction='mean')
        elif choice == Keyword.CONTINUOUS:
            return torch.nn.MSELoss(reduction='mean')

    def _validate_input(self, X, Y, Z):
        """Extend the base `_validate_input` to send data to GPU when required."""
        X, Y, Z = super(AdversarialPytorchMixin, self)._validate_input(X, Y, Z)

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        Z = torch.from_numpy(Z).float()

        if self.cuda:
            X = X.to(self.device)
            Y = Y.to(self.device)
            Z = Z.to(self.device)

        return X, Y, Z


class AdversarialTensorflowMixin(AdversarialMixin):
    """Adds TensorFlow specific functions."""

    def _evaluate(self, X):
        """
        Evaluate the model given input `X`.

        Feed 2d `numpy.ndarray` through model and receive output as
        2d `numpy.ndarray`.
        """
        Y_pred = self.predictor_model(X, training=False)
        Y_pred = Y_pred.numpy()
        return Y_pred

    def _train_step(self, X, Y, Z):
        """
        Perform one training step over data in TensorFlow models.

        Returns
        -------
        (LP, LA) : tuple of (float, float)
            predictor loss and adversary loss.
        """
        with tf.GradientTape(persistent=True) as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            Y_hat = self.predictor_model(X, training=True)

            LP = self.predictor_loss(Y, Y_hat)

            # For equalized odds
            if self.pass_y:
                Y_hat = tf.concat((Y_hat, Y), axis=1)

            Z_hat = self.adversary_model(Y_hat)

            LA = self.adversary_loss(Z, Z_hat)

        dW_LP = tape.gradient(LP, self.predictor_model.trainable_variables)
        dU_LA = tape.gradient(LA, self.adversary_model.trainable_variables)
        dW_LA = tape.gradient(LA, self.predictor_model.trainable_variables)

        del tape  # Because persistent=True !

        for i in range(len(dW_LP)):
            # Normalize dW_LA
            unit_dW_LA = dW_LA[i] / (tf.norm(dW_LA[i]) + finfo(float32).tiny)
            # Project
            proj = tf.reduce_sum(tf.multiply(dW_LP[i], unit_dW_LA))
            # Calculate dW
            dW_LP[i] = dW_LP[i] - (proj * unit_dW_LA) - (self.alpha * dW_LA[i])

        self.predictor_optimizer.apply_gradients(
            zip(dW_LP, self.predictor_model.trainable_variables))
        self.adversary_optimizer.apply_gradients(
            zip(dU_LA, self.adversary_model.trainable_variables))

        return (LP.numpy().item(), LA.numpy().item())

    def _setup_optimizer(self, optimizer):
        """
        Create the optimizers for TensorFlow.

        Setup self.predictor_optimizer and self.adversary_optimizer using the
        parameter optimizer given by the user.
        """
        super(AdversarialTensorflowMixin, self)._setup_optimizer(optimizer)
        if isinstance(optimizer, str):
            # keyword cases.
            if optimizer.lower() == "adam":
                self.predictor_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate)
                self.adversary_optimizer = tf.keras.optimizers.Adam(
                    learning_rate=self.learning_rate)
            elif optimizer.lower() == "sgd":
                self.predictor_optimizer = tf.keras.optimizers.SGD(
                    learning_rate=self.learning_rate)
                self.adversary_optimizer = tf.keras.optimizers.SGD(
                    learning_rate=self.learning_rate)
            else:
                raise ValueError(_KWARG_ERROR_MESSAGE.format(
                    'optimizer', '"Adam" or "SGD" or an optimizer'))
    
    
    def _get_loss(self, Y, choice, data_name):
        """Infer loss."""
        choice = self._interpret_choice(Y, choice, data_name)
        if callable(choice):
            return choice
        if choice == Keyword.BINARY:
            return tf.keras.losses.BinaryCrossentropy(from_logits=True)
        elif choice == Keyword.CATEGORY:
            return tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        elif choice == Keyword.CONTINUOUS:
            return tf.keras.losses.MeanSquaredError()


class AdversarialFairnessClassifier(AdversarialFairness):
    """Creates an AdversarialFairness with loss and predictions set to classification."""

    def __init__(self, **kwargs):
        """Initialize model by setting the predictor loss and function."""
        kwargs['predictor_loss'] = Keyword.CLASSIFICATION
        kwargs['predictor_function'] = Keyword.CLASSIFICATION
        super(AdversarialFairnessClassifier, self).__init__(**kwargs)


class AdversarialFairnessRegressor(AdversarialFairness):
    """Create an AdversarialFairness that has predictor loss set to regression."""

    def __init__(self, *args, **kwargs):
        """Initialize model by setting the predictor loss."""
        kwargs['predictor_loss'] = Keyword.CONTINUOUS
        super(AdversarialFairnessRegressor, self).__init__(*args, **kwargs)
