# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import ast
import logging
import os
from pathlib import Path

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def _ensure_cwd_is_fairlearn_root_dir():
    # To ensure we're in the right directory that there's a fairlearn directory inside the
    # current working directory as well as the presence of a README.rst file.
    if not os.path.exists(os.path.join(os.getcwd(), "fairlearn")) or not os.path.exists(
        os.path.join(os.getcwd(), "README.rst")
    ):
        raise Exception(
            "Please run this from the fairlearn root directory. Current directory: {}".format(
                os.getcwd()
            )
        )


def _get_fairlearn_version():
    """Read ``__version__`` from ``fairlearn/__init__.py`` without importing the package.

    This avoids requiring fairlearn (and its runtime dependencies) to be installed
    before the version can be read, which is useful in packaging/release scripts
    that run before ``pip install``.
    """
    init_path = Path(os.getcwd()) / "fairlearn" / "__init__.py"
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__version__":
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, str
                    ):
                        return node.value.value
    raise RuntimeError(
        "Could not find a string literal assignment to __version__ in {}".format(init_path)
    )


class _LogWrapper:
    def __init__(self, description):
        self._description = description

    def __enter__(self):
        _logger.info("Starting %s", self._description)

    def __exit__(self, type, value, traceback):  # noqa: A002
        # raise exceptions if any occurred
        if value is not None:
            raise value
        _logger.info("Completed %s", self._description)
