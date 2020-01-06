# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import nbformat as nbf
import papermill as pm
import pytest
import scrapbook as sb


class ScrapSpec:
    def __init__(self, code, expected):
        self.code = code
        self.expected = expected

    @property
    def code(self):
        """Return the code to be inserted (string)."""
        return self._code

    @code.setter
    def code(self, value):
        self._code = value

    @property
    def expected(self):
        """Return the expected evaluation of the code (Python object)."""
        return self._expected

    @expected.setter
    def expected(self, value):
        self._expected = value


def append_scrapbook_commands(input_nb_path, output_nb_path, scrap_specs):
    notebook = nbf.read(input_nb_path, as_version=nbf.NO_CONVERT)

    scrapbook_cells = []
    # Always need to import nteract-scrapbook
    scrapbook_cells.append(nbf.v4.new_code_cell(source="import scrapbook as sb"))

    # Create a cell to store each key and value in the scrapbook
    for k, v in scrap_specs.items():
        source = "sb.glue(\"{0}\", {1})".format(k, v.code)
        scrapbook_cells.append(nbf.v4.new_code_cell(source=source))

    # Append the cells to the notebook
    [notebook['cells'].append(c) for c in scrapbook_cells]

    # Write out the new notebook
    nbf.write(notebook, output_nb_path)


def assay_one_notebook(notebook_name, test_values):
    input_notebook = "notebooks/" + notebook_name + ".ipynb"
    processed_notebook = "./test/notebooks/" + notebook_name + ".processed.ipynb"
    output_notebook = "./test/notebooks/" + notebook_name + ".output.ipynb"

    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)

    for k, v in test_values.items():
        assert nb.scraps[k].data == v.expected


@pytest.mark.notebooks
def test_group_metrics_notebook():
    overall_recall_key = "overall_recall"
    by_groups_key = "recall_by_groups"

    test_values = {}
    test_values[overall_recall_key] = ScrapSpec("group_metrics.overall", 0.5)
    test_values[by_groups_key] = ScrapSpec(
        "results.by_group", {'a': 0.0, 'b': 0.5, 'c': 0.75, 'd': 0.0})

    assay_one_notebook("Group Metrics", test_values)
