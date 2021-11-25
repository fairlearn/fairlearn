# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

_IMPORT_ERROR_MESSAGE = "Please make sure to install {} in order to use this module."
_KWARG_ERROR_MESSAGE = "Key-word argument \'{}\' should be set to {}."
_TYPE_CHECK_ERROR = "Data of \'{}\' does not follow distribution assumption \'{}\'."
_TYPE_COMPLIANCE_ERROR = _TYPE_CHECK_ERROR + " The data looks \'{}\'."
_INVALID_DATA = \
    "Data of \'{}\' can not be interpreted. Only continuous data or array with only 0s and 1s."
_PREDICTION_FUNCTION_AMBIGUOUS = \
    "prediction_function {} cannot be inferred. Provide function mapping " + \
    "soft-probabilities to discrete prediction, or a keyword {}."
_PROGRESS_UPDATE = \
    "|{}>{}| Epoch: {}/{}, Batch: {}{}/{}, ETA: {:.2f} sec. Loss (pred/adv): {:.2f}/{:.2f}"
_NOT_IMPLEMENTED = "Interface-class methods have not been implemented"
_NO_DATA = "Haven't seen data yet. Call fit or partial_fit first to set up."