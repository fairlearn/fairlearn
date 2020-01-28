import argparse
import logging
import os
import subprocess
import sys

from _utils import _ensure_cwd_is_fairlearn_root_dir, _LogWrapper
from process_readme import process_readme


_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


_fairlearn_dev_version_env_var_name = 'FAIRLEARN_DEV_VERSION'


def build_argument_parser():
    desc = "Build wheels for fairlearn"

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--target-type",
                        help="Either Prod or Test",
                        choices=['Prod', "Test"],
                        required=True)
    parser.add_argument("--dev-version",
                        help="Optional version suffix to indicate the development state of the "
                             "package.",
                        required=False)
    parser.add_argument("--version-filename",
                        help="The file where the version will be stored.",
                        required=True)

    return parser


def main(argv):
    _ensure_cwd_is_fairlearn_root_dir()
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if _fairlearn_dev_version_env_var_name in os.environ:
        raise Exception("Environment variable {} must not be set"
                        .format(_fairlearn_dev_version_env_var_name))

    if args.target_type == "Test":
        os.environ[_fairlearn_dev_version_env_var_name] = args.dev_version

    with _LogWrapper("installation of fairlearn"):
        subprocess.check_call(["pip", "install", "-e", ".[customplots]"])

    with _LogWrapper("processing README.md"):
        process_readme("README.md", "README.md")

    with _LogWrapper("storing fairlearn version in {}".format(args.version_filename)):
        import fairlearn
        with open(args.version_filename, 'w') as version_file:
            version_file.write(fairlearn.__version__)

    with _LogWrapper("creation of packages"):
        subprocess.check_call(["python", "setup.py", "sdist", "bdist_wheel"])


if __name__ == "__main__":
    main(sys.argv[1:])
