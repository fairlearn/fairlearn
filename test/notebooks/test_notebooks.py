# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import nbformat as nbf
import papermill as pm
import pytest
import scrapbook as sb


class ScrapSpec:
    def __init__(self, key, command):
        self.scrap_key = key
        self.scrap_command = command

    @property
    def scrap_key(self):
        return self._scrap_key

    @scrap_key.setter
    def scrap_key(self, value):
        self._scrap_key = value

    @property
    def scrap_command(self):
        return self._scrap_command

    @scrap_command.setter
    def scrap_command(self, value):
        self._scrap_command = value


def append_scrapbook_commands(input_nb_path, output_nb_path, scrap_specs):
    notebook = nbf.read(input_nb_path, as_version=nbf.NO_CONVERT)

    scrapbook_cells = []
    scrapbook_cells.append(nbf.v4.new_code_cell(source="import scrapbook as sb"))

    for s in scrap_specs:
        source = "sb.glue(\"{0}\", {1})".format(s.scrap_key, s.scrap_command)
        scrapbook_cells.append(nbf.v4.new_code_cell(source=source))

    [notebook['cells'].append(c) for c in scrapbook_cells]

    nbf.write(notebook, output_nb_path)


@pytest.mark.notebooks
def test_group_metrics_notebook():
    notebookname = "Group Metrics"
    input_notebook = "notebooks/" + notebookname + ".ipynb"
    processed_notebook = "./test/notebooks/" + notebookname + ".processed.ipynb"
    output_notebook = "./test/notebooks/" + notebookname + ".output.ipynb"

    cmds = []
    cmds.append(ScrapSpec("overall_recall", "group_metrics.overall"))
    cmds.append(ScrapSpec("recall_by_groups", "results.by_group"))

    append_scrapbook_commands(input_notebook, processed_notebook, cmds)

    pm.execute_notebook(processed_notebook, output_notebook)

    nb = sb.read_notebook(output_notebook)

    expected_overall_recall = 0.5
    assert nb.scraps['overall_recall'].data == expected_overall_recall

    expected_by_groups = {'a': 0.0, 'b': 0.5, 'c': 0.75, 'd': 0.0}
    assert nb.scraps['recall_by_groups'].data == expected_by_groups
