# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""Combine a complete CI matrix's line coverage without mixing in branch data."""

import argparse
import subprocess
import sys
from pathlib import Path

from coverage import CoverageData


def combine_coverage(reports: Path, expected: set[str]) -> None:
    actual = {path.name for path in reports.iterdir()}
    if actual != expected:
        raise ValueError(
            f"Incomplete coverage matrix: missing {sorted(expected - actual)}, "
            f"unexpected {sorted(actual - expected)}"
        )

    data_paths = []
    for name in sorted(expected):
        path = reports / name / ".coverage"
        if not path.is_file():
            raise ValueError(f"Missing raw coverage data: {path}")
        data = CoverageData(basename=str(path))
        try:
            data.read()
            if not data.measured_files():
                raise ValueError(f"Empty coverage data: {path}")
            if data.has_arcs():
                raise ValueError(f"Branch coverage cannot enter the line aggregate: {path}")
        finally:
            # Older coverage.py versions close connections after each operation.
            if hasattr(data, "close"):
                data.close()
        data_paths.append(str(path))

    for command in [
        ["combine", "--keep", *data_paths],
        ["xml", "-o", "coverage.xml"],
        ["json", "-o", "coverage.json"],
        ["report"],
    ]:
        subprocess.run([sys.executable, "-m", "coverage", *command], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--os", nargs="+", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--min-python", required=True)
    parser.add_argument("--other-ml", nargs="+", required=True)
    args = parser.parse_args()
    expected = {
        f"coverage-line-{suite}-{os}-{python}"
        for suite, python in [
            ("full", args.python),
            ("min", args.min_python),
            ("install", args.python),
        ]
        for os in args.os
    }
    expected.update(
        f"coverage-line-other-{package}-ubuntu-latest-{args.python}" for package in args.other_ml
    )
    try:
        combine_coverage(args.reports, expected)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
