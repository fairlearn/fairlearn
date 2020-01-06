import papermill as pm
import scrapbook as sb
import pytest


@pytest.mark.notebooks
def test_group_metrics_notebook():
    notebookname = "Group Metrics"
    input_notebook = "notebooks/" + notebookname + ".ipynb"
    output_notebook = "./test/notebooks/" + notebookname + ".output.ipynb"

    pm.execute_notebook(input_notebook, output_notebook)

    nb = sb.read_notebook(output_notebook)

    expected_overall_recall = 0.5
    assert nb.scraps['overall_recall'].data == expected_overall_recall

    expected_by_groups = {'a': 0.0, 'b': 0.5, 'c': 0.75, 'd': 0.0}
    assert nb.scraps['recall_by_groups'].data == expected_by_groups
