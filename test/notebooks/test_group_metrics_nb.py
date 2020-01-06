import papermill as pm
import scrapbook as sb
import pytest


@pytest.mark.notebooks
def test_group_metrics_notebook():
    notebookname = "Group Metrics"
    input_notebook = "notebooks/" + notebookname + ".ipynb"
    output_notebook = "./test/notebooks" + notebookname + ".output.ipynb"

    pm.execute_notebook(input_notebook, output_notebook)

    nb = sb.read_notebook(output_notebook)
    print(nb.scraps)  # print a dict of all scraps by name
