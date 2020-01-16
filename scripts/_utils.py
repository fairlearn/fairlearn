import logging
import os

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def _ensure_cwd_is_fairlearn_root_dir():
    # To ensure we're in the right directory check both the directory name
    # and that there's a fairlearn directory inside.
    if not os.path.basename(os.getcwd()) == 'fairlearn' or \
            not os.path.exists(os.path.join(os.getcwd(), "fairlearn")):
        raise Exception("Please run this from the fairlearn root directory. "
                        "Current directory: {}".format(os.getcwd()))


class LogWrapper:
    def __init__(self, description):
        self._description = description

    def __enter__(self):
        _logger.info("Starting {}".format(self._description))

    def __exit__(self, type, value, traceback):  # noqa: A002
        # raise exceptions if any occurred
        if value is not None:
            raise value
        _logger.info("Completed {}".format(self._description))
