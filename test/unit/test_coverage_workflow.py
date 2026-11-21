# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

import json
import runpy
import shutil
from pathlib import Path

import pytest
from coverage import CoverageData

ROOT = Path(__file__).resolve().parents[2]
combine_coverage = runpy.run_path(str(ROOT / "scripts" / "combine_coverage.py"))[
    "combine_coverage"
]


@pytest.fixture
def reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for variable in (
        "COVERAGE_FILE",
        "COV_CORE_SOURCE",
        "COV_CORE_CONFIG",
        "COV_CORE_DATAFILE",
        "COV_CORE_BRANCH",
    ):
        monkeypatch.delenv(variable, raising=False)
    shutil.copyfile(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    package = tmp_path / "fairlearn"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "example.py").write_text(
        "def choose(flag):\n"
        "    if flag:\n"
        "        return 1\n"
        "    return 2\n"
        "def install_only():\n"
        "    return 3\n"
    )
    (package / "uncovered.py").write_text("value = 1\n")
    reports = tmp_path / "coverage-data"
    reports.mkdir()
    return reports


def write_report(reports, name, filename, lines, untouched=()):
    directory = reports / name
    directory.mkdir()
    data = CoverageData(basename=str(directory / ".coverage"))
    data.add_lines({filename: lines})
    data.touch_files(untouched)
    data.write()
    if hasattr(data, "close"):
        data.close()
    return directory


def test_combines_cross_platform_reports_and_install_only_coverage(reports):
    write_report(
        reports,
        "linux",
        "/home/runner/work/fairlearn/fairlearn/fairlearn/example.py",
        {1, 2, 3, 5},
        untouched=[
            "/home/runner/work/fairlearn/fairlearn/fairlearn/__init__.py",
            "/home/runner/work/fairlearn/fairlearn/fairlearn/uncovered.py",
        ],
    )
    write_report(
        reports, "windows", r"D:\a\fairlearn\fairlearn\fairlearn\example.py", {1, 2, 4, 5}
    )
    write_report(
        reports, "install", "/Users/runner/work/fairlearn/fairlearn/fairlearn/example.py", {5, 6}
    )
    combine_coverage(reports, {"linux", "windows", "install"})
    report = json.loads(Path("coverage.json").read_text())
    files = {name.replace("\\", "/"): data for name, data in report["files"].items()}
    assert set(files) == {
        "fairlearn/__init__.py",
        "fairlearn/example.py",
        "fairlearn/uncovered.py",
    }
    assert report["totals"]["covered_lines"] == 6
    assert report["totals"]["num_statements"] == 7
    assert files["fairlearn/uncovered.py"]["missing_lines"] == [1]
    assert not report["meta"]["branch_coverage"]
    assert Path("coverage.xml").is_file()
    assert all(
        (reports / name / ".coverage").is_file() for name in {"linux", "windows", "install"}
    )


@pytest.mark.parametrize("actual,expected", [("full", {"full", "install"}), ("branch", {"full"})])
def test_rejects_incomplete_or_unexpected_reports(reports, actual, expected):
    write_report(reports, actual, "fairlearn/example.py", {1})
    with pytest.raises(ValueError, match="Incomplete coverage matrix"):
        combine_coverage(reports, expected)
    assert not Path("coverage.xml").exists()


def test_rejects_artifact_without_raw_data(reports):
    (reports / "full").mkdir()
    with pytest.raises(ValueError, match="Missing raw coverage data"):
        combine_coverage(reports, {"full"})


def test_rejects_empty_raw_data(reports):
    directory = reports / "full"
    directory.mkdir()
    data = CoverageData(basename=str(directory / ".coverage"))
    data.add_lines({})
    data.write()
    if hasattr(data, "close"):
        data.close()
    with pytest.raises(ValueError, match="Empty coverage data"):
        combine_coverage(reports, {"full"})


def test_rejects_branch_data_in_line_aggregate(reports):
    directory = reports / "branch"
    directory.mkdir()
    data = CoverageData(basename=str(directory / ".coverage"))
    data.add_arcs({"fairlearn/example.py": {(-1, 1), (1, 2), (2, 3)}})
    data.write()
    if hasattr(data, "close"):
        data.close()
    with pytest.raises(ValueError, match="Branch coverage cannot enter the line aggregate"):
        combine_coverage(reports, {"branch"})
