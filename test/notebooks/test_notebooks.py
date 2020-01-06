# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import nbformat as nbf
import papermill as pm
import pytest
import scrapbook as sb


@pytest.mark.notebooks
def test_group_metrics_notebook():
    notebookname = "Group Metrics"
    input_notebook = "notebooks/" + notebookname + ".ipynb"
    processed_notebook = "./test/notebooks/" + notebookname + ".processed.ipynb"
    output_notebook = "./test/notebooks/" + notebookname + ".output.ipynb"

    notebook = nbf.read(input_notebook, as_version=nbf.NO_CONVERT)
    # sb.glue("overall_recall", group_metrics.overall)
    # sb.glue("recall_by_groups", results.by_group)
    scrapbook_cells = []
    scrapbook_cells.append(nbf.v4.new_code_cell(source="import scrapbook as sb"))
    scrapbook_cells.append(nbf.v4.new_code_cell(
        source="sb.glue(\"overall_recall\", group_metrics.overall)"))
    scrapbook_cells.append(nbf.v4.new_code_cell(
        source="sb.glue(\"recall_by_groups\", results.by_group)"))
    [notebook['cells'].append(c) for c in scrapbook_cells]

    nbf.write(notebook, processed_notebook)

    pm.execute_notebook(processed_notebook, output_notebook)

    nb = sb.read_notebook(output_notebook)

    expected_overall_recall = 0.5
    assert nb.scraps['overall_recall'].data == expected_overall_recall

    expected_by_groups = {'a': 0.0, 'b': 0.5, 'c': 0.75, 'd': 0.0}
    assert nb.scraps['recall_by_groups'].data == expected_by_groups
