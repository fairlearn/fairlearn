# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

PREDICTOR_OR_ESTIMATOR_REQUIRED_ERROR_MESSAGE = "One of 'unconstrained_predictor' and " \
                                                "'estimator' need to be passed."
EITHER_PREDICTOR_OR_ESTIMATOR_ERROR_MESSAGE = "Only one of 'unconstrained_predictor' and " \
                                              "'estimator' can be passed."
MISSING_FIT_PREDICT_ERROR_MESSAGE = "The model does not have callable 'fit' or 'predict' methods."
MISSING_PREDICT_ERROR_MESSAGE = "The predictor does not have a callable 'predict' method."


class PostProcessing:
    """Abstract base class for postprocessing approaches for disparity mitigation.

    :param unconstrained_predictor: a predictor with a `predict` function that has already been
        trained on the training data; the predictor will subsequently be used in the mitigator
        for unconstrained predictions; can only be specified if `estimator` is None
    :type unconstrainted_predictor: predictor
    :param estimator: an estimator with a `fit` and `predict` function that will be trained on the
        training data and subsequently used in the mitigator for unconstrained predictions; can
        only be specified if `unconstrainted_predictor` is None
    :type estimator: estimator
    :param constraints: the parity constraints to be enforced represented as a string
    :type constraints: str
    """

    def __init__(self, *, unconstrained_predictor=None, estimator=None,
                 constraints=None):
        if unconstrained_predictor and estimator:
            raise ValueError(EITHER_PREDICTOR_OR_ESTIMATOR_ERROR_MESSAGE)
        elif unconstrained_predictor:
            self._unconstrained_predictor = unconstrained_predictor
            self._estimator = None
            self._validate_predictor()
        elif estimator:
            self._unconstrained_predictor = None
            self._estimator = estimator
            self._validate_estimator()
        else:
            raise ValueError(PREDICTOR_OR_ESTIMATOR_REQUIRED_ERROR_MESSAGE)

    def fit(self, X, y, *, sensitive_features, **kwargs):
        """Fits the model.

        The fit is based on training features and labels, sensitive features,
        as well as the fairness-unaware predictor or estimator. If an estimator was passed
        in the constructor this fit method will call :code:`fit(X, y, **kwargs)` on said
        estimator.

        :param X: Feature matrix
        :type X: numpy.ndarray or pandas.DataFrame
        :param y: Label vector
        :type y: numpy.ndarray, pandas.DataFrame, pandas.Series, or list
        :param sensitive_features: Sensitive features to identify groups by, currently allows
            only a single column
        :type sensitive_features: currently 1D array as numpy.ndarray, list, pandas.DataFrame,
            or pandas.Series
        """
        raise NotImplementedError(self.fit.__name__ + " is not implemented")

    def predict(self, X, *, sensitive_features):
        """Predict label for each sample in X while taking into account sensitive features.

        :param X: Feature matrix
        :type X: numpy.ndarray or pandas.DataFrame
        :param sensitive_features: Sensitive features to identify groups by, currently allows
            only a single column
        :type sensitive_features: Currently 1D array as numpy.ndarray, list, pandas.DataFrame,
            or pandas.Series
        :return: predictions in numpy.ndarray
        """
        raise NotImplementedError(self.predict.__name__ + " is not implemented")

    def _pmf_predict(self, X, *, sensitive_features):
        """Probabilistic mass function.

        :param X: Feature matrix
        :type X: numpy.ndarray or pandas.DataFrame
        :param sensitive_features: Sensitive features to identify groups by, currently allows
            only a single column
        :type sensitive_features: Currently 1D array as numpy.ndarray, list, pandas.DataFrame,
            or pandas.Series
        :return: array of tuples with probabilities for predicting 0 or 1, respectively. The sum
            of the two numbers in each tuple needs to add up to 1.
        :rtype: numpy.ndarray
        """
        raise NotImplementedError(self._pmf_predict.__name__ + " is not implemented")

    def _validate_predictor(self):
        """Validate that the _unconstrained_predictor member has a predict function."""
        predict_function = getattr(self._unconstrained_predictor, "predict", None)
        if not predict_function or not callable(predict_function):
            raise ValueError(MISSING_PREDICT_ERROR_MESSAGE)

    def _validate_estimator(self):
        """Validate that the `_estimator` member has both a fit and a predict function."""
        fit_function = getattr(self._estimator, "fit", None)
        predict_function = getattr(self._estimator, "predict", None)
        if not predict_function or not fit_function or not callable(predict_function) or \
                not callable(fit_function):
            raise ValueError(MISSING_FIT_PREDICT_ERROR_MESSAGE)


# Ensure that PostProcessing shows up in correct place in documentation
# when it is used as a base class
PostProcessing.__module__ = "fairlearn.postprocessing"
