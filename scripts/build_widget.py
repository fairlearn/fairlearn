import logging
import os
import shutil
import subprocess

from _utils import _ensure_cwd_is_fairlearn_root_dir, LogWrapper


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

_widget_js_directory = os.path.join("fairlearn", "widget", "js")


if __name__ == "__main__":
    _ensure_cwd_is_fairlearn_root_dir()

    LogWrapper("yarn install of dependencies"):
        subprocess.check_call(["yarn", "install"])

    LogWrapper("yarn build"):
        subprocess.check_call(["yarn", "build:all"])
    
    LogWrapper("removal of extra directories"):
        shutil.rmtree("dist")
        shutil.rmtree("lib")
        shutil.rmtree("node_modules")
