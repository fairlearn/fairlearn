# Copyright (c) Fairlearn contributors.
# Licensed under the MIT License.

_IMPORT_ERROR_MESSAGE = (
    "Please make sure to install {} in order to use this module."
)
_ARG_ERROR_MESSAGE = "Argument '{}' should be set to {}."
_KWARG_ERROR_MESSAGE = "Key-word argument '{}' should be set to {}."
_TYPE_CHECK_ERROR = "Data does not follow distribution assumption '{}'."
_TYPE_COMPLIANCE_ERROR = _TYPE_CHECK_ERROR + " The data looks '{}'."
_INVALID_DATA = (
    "Data of '{}' can not be interpreted. Only continuous data or array "
    + "with only 0s and 1s."
)
_TRANSFORMER_WEIRD_DIST_TYPE = (
    "Transformer {} has an unsupported dist_type_ {}. "
    + "Make sure the provided dist_type_ is either "
    + '"binary", "category", or "continuous".'
)
_PREDICTION_FUNCTION_AMBIGUOUS = (
    "Can not interpret prediction_function. Please provide a callable as "
    + "key-word argument that maps soft-probabilities "
    + "(or more precisely, predictor_model output) "
    + "to discrete prediction. Or, pass a Transformer with the `transformer` "
    + "key-word that has a dist_type_ attribute that "
    + "a BackendEngine can use to interpret a loss function."
)
_DIST_TYPE_NOT_IMPLEMENTED = (
    "BackendEngine {} has no loss function defined for the given "
    + "distribution type {}."
)
_PROGRESS_UPDATE = (
    "|{}>{}| Epoch: {}/{}, Batch: {}{}/{}, ETA: {:.2f} sec. Loss "
    + "(pred/adv): {:.2f}/{:.2f}"
)
_NOT_IMPLEMENTED = "Interface-class method has not been implemented"
_NO_DATA = "Haven't seen data yet. Call fit or partial_fit first to set up."
_NO_LOSS = (
    "Can not interpret {}. Please provide a callable as key-word argument. "
    + "Or, pass a Transformer with the `transformer` key-word that "
    + "has a dist_type_ attribute that "
    + "a BackendEngine can use to interpret a loss function."
)
_TRANSFORM_NO_PARAM = (
    ". Make sure your provided transform accepts a distribution "
    + 'keyword (such as "binary" and "category") in '
    + "its __init__"
)
_MODEL_UNRECOGNIZED_STR = (
    "Key-word argument predictor_model or adversary_model received "
    + "an unrecognized string: {}. If you were trying to signal a "
    + "certain activation function, try passing an instance of that "
    + "function directly."
)
_MODEL_UNRECOGNIZED_ITEM = (
    "Key-word argument predictor_model or adversary_model received "
    + "an unrecognized item: {}. Either pass an int to signal a linear layer, "
    + "a callable to add directly to the model (such as an activation "
    + "function) or a supported keyword (string)."
)
_TYPE_UNKNOWN_ERROR = "Unknown label type."