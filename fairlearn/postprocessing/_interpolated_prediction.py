# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
from ._constants import OUTPUT_SEPARATOR

logger = logging.getLogger(__name__)


class InterpolatedPredictor:
    """Predictor for computing predictions between two actual predictions.

    The predictions are represented through the threshold rules operation0 and operation1.

    :param p_ignore: p_ignore changes the interpolated prediction P to the desired
        solution using the transformation p_ignore * prediction_constant + (1 - p_ignore) * P
    :param prediction_constant: 0 if not required, otherwise the x value of the best
        solution should be passed
    :param p0: interpolation multiplier for prediction from the first predictor
    :param operation0: threshold rule for the first predictor
    :param p1: interpolation multiplier for prediction from the second predictor
    :param operation1: threshold rule for the second predictor
    :return: an anonymous function that scales the original prediction to the desired one
    :rtype: lambda
    """

    def __init__(self, p_ignore, prediction_constant, p0, operation0, p1, operation1):
        self._operation0 = operation0
        self._operation1 = operation1
        self._p_ignore = p_ignore
        self._prediction_constant = prediction_constant
        self._p0 = p0
        self._p1 = p1

        logger.debug(OUTPUT_SEPARATOR)
        logger.debug("p_ignore: %s", p_ignore)
        logger.debug("prediction_constant: %s", prediction_constant)
        logger.debug("p0: %s", p0)
        logger.debug("operation0: %s", operation0)
        logger.debug("p1: %s", p1)
        logger.debug("operation1: %s", operation1)
        logger.debug(OUTPUT_SEPARATOR)

    def __repr__(self):  # noqa: D105
        return "[p_ignore: {}, prediction_constant: {}, " \
            "p0: {}, operation0: {}, p1: {}, operation1: {}]" \
            .format(self._p_ignore, self._prediction_constant, self._p0, self._operation0,
                    self._p1, self._operation1)

    def predict(self, scores):
        """Create the interpolated prediction.

        The interpolation is based on two threshold operations and the
        transformation adjustment.

        :param scores: the scores from an unconstrained predictor to which the threshold
            operations are applied
        :type scores: numpy.ndarray
        :return: the interpolated prediction
        :rtype: numpy.ndarray
        """
        transformation_adjustment = self._p_ignore * self._prediction_constant
        weighted_predictions0 = self._p0 * self._operation0.get_predictor_from_operation()(scores)
        weighted_predictions1 = self._p1 * self._operation1.get_predictor_from_operation()(scores)
        interpolated_predictions = (1 - self._p_ignore) * (weighted_predictions0 + weighted_predictions1)  # noqa: E501
        return transformation_adjustment + interpolated_predictions
