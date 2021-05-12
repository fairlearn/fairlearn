# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Utility methods to print system info for debugging.

Adapted from :py:func:`pandas.show_versions` and :py:func:`sklearn.show_versions`.
"""  # noqa: RST304

import platform
import sys
import importlib


def _get_sys_info():
    """System information.

    :returns: system and Python version information
    :rtype: dict
    """
    python = sys.version.replace('\n', ' ')

    blob = [
        ("python", python),
        ('executable', sys.executable),
        ("machine", platform.platform()),
    ]

    return dict(blob)


def _get_deps_info():
    """Overview of the installed version of main dependencies.

    :return: version information on relevant Python libraries
    :rtype: dict
    """
    deps = sorted([
        "pip",
        "setuptools",
        "sklearn",
        "numpy",
        "scipy",
        "Cython",
        "pandas",
        "matplotlib",
        "tempeh"
        ])

    def get_version(module):
        return module.__version__

    deps_info = {}

    for modname in deps:
        try:
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = importlib.import_module(modname)
            ver = get_version(mod)
            deps_info[modname] = ver
        except ImportError:
            deps_info[modname] = None

    return deps_info


def show_versions():
    """Print useful debugging information."""
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()

    print('\nSystem:')
    for k, stat in sys_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))

    print('\nPython dependencies:')
    for k, stat in deps_info.items():
        print("{k:>10}: {stat}".format(k=k, stat=stat))
