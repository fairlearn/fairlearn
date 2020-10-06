# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

SCORE_KEY = "score"
LABEL_KEY = "label"
SENSITIVE_FEATURE_KEY = "sensitive_feature"
P0_KEY = "p0"
P1_KEY = "p1"

OUTPUT_SEPARATOR = "-"*65

_MATPLOTLIB_IMPORT_ERROR_MESSAGE = "Please make sure to install fairlearn[customplots] to use " \
                                   "the postprocessing plots."
BASE_ESTIMATOR_NONE_ERROR_MESSAGE = "The base estimator cannot be `None`."
BASE_ESTIMATOR_NOT_FITTED_WARNING = "The value of `prefit` is `True`, but `check_is_fitted` " \
    "raised `NotFittedError` on the base estimator.\n\n" \
    "If the provided base estimator has been fitted, this could mean that (1) its " \
    "implementation does not conform to the sklearn estimator API, or (2) the enclosing {0} " \
    "has been cloned (for instance by `sklearn.model_selection.cross_validate`).\n\n" \
    "In case (1), please file an issue with the base estimator developers, but continue " \
    "to use the enclosing {0} with `prefit=True`. In case (2), please use `prefit=False`."
