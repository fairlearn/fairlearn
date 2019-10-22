# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

MODEL_OR_ESTIMATOR_REQUIRED_ERROR_MESSAGE = "One of 'unconstrained_model' and " \
                                            "'unconstrained_estimator' need to be passed."
EITHER_MODEL_OR_ESTIMATOR_ERROR_MESSAGE = "Only one of 'unconstrained_model' and " \
                                          "'unconstrained_estimator' can be passed."
MISSING_FIT_PREDICT_ERROR_MESSAGE = "The model does not have callable 'fit' or 'predict' methods."
MISSING_PREDICT_ERROR_MESSAGE = "The model does not have a callable 'predict' method."


class PostProcessing:
    def __init__(self, *, unconstrained_model=None, unconstrained_estimator=None,
                 parity_criteria=None):
        if unconstrained_model and unconstrained_estimator:
            raise ValueError(EITHER_MODEL_OR_ESTIMATOR_ERROR_MESSAGE)
        elif unconstrained_model:
            self._unconstrained_model = unconstrained_model
            self._unconstrained_estimator = None
            self._validate_model()
        elif unconstrained_estimator:
            self._unconstrained_model = None
            self._unconstrained_estimator = unconstrained_estimator
            self._validate_estimator()
        else:
            raise ValueError(MODEL_OR_ESTIMATOR_REQUIRED_ERROR_MESSAGE)

    def fit(self, X, y, sensitive_features, **kwargs):
        """ Fit the model based on training features and labels, auxiliary data,
        as well as the fairness-unaware model or estimator. If an estimator was passed
        in the constructor this fit method will call fit(X, y, **kwargs) on said estimator.
        """
        raise NotImplementedError(self.fit.__name__ + " is not implemented")

    def predict(self, X, sensitive_features):
        raise NotImplementedError(self.predict.__name__ + " is not implemented")

    def predict_proba(self, X, sensitive_features):
        raise NotImplementedError(self.predict_proba.__name__ + " is not implemented")

    def _validate_model(self):
        predict_function = getattr(self._unconstrained_model, "predict", None)
        if not predict_function or not callable(predict_function):
            raise ValueError(MISSING_PREDICT_ERROR_MESSAGE)

    def _validate_estimator(self):
        fit_function = getattr(self._unconstrained_estimator, "fit", None)
        predict_function = getattr(self._unconstrained_estimator, "predict", None)
        if not predict_function or not fit_function or not callable(predict_function) or \
                not callable(fit_function):
            raise ValueError(MISSING_FIT_PREDICT_ERROR_MESSAGE)
