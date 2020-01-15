import logging
import os
import shutil
import subprocess

from _utils import _ensure_cwd_is_fairlearn_root_dir, LogWrapper


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

_doc_build_dir = 'docbuild'
_doc_config_dir = "docs"


if __name__ == "__main__":
    _ensure_cwd_is_fairlearn_root_dir()

    if os.path.exists(_doc_build_dir):
        with LogWrapper("deletion of {}".format(_doc_build_dir)):
            shutil.rmtree(_doc_build_dir)

    with LogWrapper("copying config files to {}".format(_doc_build_dir)):
        shutil.copytree(_doc_config_dir, _doc_build_dir)

    with LogWrapper("creation of expected directories"):
        os.mkdir(os.path.join(_doc_build_dir, "_static"))
        os.mkdir(os.path.join(_doc_build_dir, "_build"))
        os.mkdir(os.path.join(_doc_build_dir, "_templates"))

    with LogWrapper("build of API documentation with sphinx"):
        subprocess.check_call(["sphinx-apidoc", 'fairlearn', '-o', _doc_build_dir])
        subprocess.check_call(["sphinx-build", _doc_build_dir, os.path.join(_doc_build_dir, "_build")])
