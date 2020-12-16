import numpy as np
class SquareLoss:
    """Class to evaluate the square loss."""

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.min = 0
        self.max = (max_val-min_val) ** 2

    def eval(self, y_true, y_pred):  # noqa: A003
        """Evaluate the square loss for the given set of true and predicted values."""
        return (np.clip(y_true, self.min_val, self.max_val)
                - np.clip(y_pred, self.min_val, self.max_val)) ** 2


class AbsoluteLoss:
    """Class to evaluate absolute loss."""

    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        self.min = 0
        self.max = np.abs(max_val-min_val)

    def eval(self, y_true, y_pred):  # noqa: A003
        """Evaluate the absolute loss for the given set of true and predicted values."""
        return np.abs(np.clip(y_true, self.min_val, self.max_val)
                      - np.clip(y_pred, self.min_val, self.max_val))


# Ensure that AbsoluteLoss shows up in correct place in documentation
# when it is used as a base class
AbsoluteLoss.__module__ = "fairlearn.reductions"

class ZeroOneLoss(AbsoluteLoss):
    """Class to evaluate a zero-one loss."""

    def __init__(self):
        super().__init__(0, 1)