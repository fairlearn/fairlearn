# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Utility methods to print system info for debugging.

Adapted from :py:func:`pandas.show_versions` and :py:func:`sklearn.show_versions`.
"""  # noqa: RST304

import platform
import sys


def _get_sys_info():
    """System information.

    Returns
    -------
    sys_info : dict
        system and Python version information
    """
    python = sys.version.replace("\n", " ")

    blob = [
        ("python", python),
        ("executable", sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info():
    """Overview of the installed version of main dependencies.

    This function does not import the modules to collect the version numbers
    but instead relies on standard Python package metadata.

    Returns
    -------
    deps_info: dict
        version information on relevant Python libraries
    """
    deps = sorted(
        [
            "pip",
            "setuptools",
            "numpy",
            "scipy",
            "Cython",
            "pandas",
            "matplotlib",
            "sklearn",
            "lightgbm",
            "pytorch",
            "tensorflow",
        ]
    )

    from fairlearn import __version__

    deps_info = {"fairlearn": __version__}

    from importlib.metadata import PackageNotFoundError, version

    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None
    return deps_info


def show_versions():
    """Print useful debugging information."""
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print("\nSystem:")
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))

    print("\nPython dependencies:")
    for k, stat in deps_info.items():
        print("{k:>13}: {stat}".format(k=k, stat=stat))
