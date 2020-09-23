# Copyright (c) Microsoft Corporation and Fairlearn contributors.
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
        """The code to be inserted (string)."""  # noqa:D401
        return self._code

    @code.setter
    def code(self, value):
        self._code = value

    @property
    def expected(self):
        """The expected evaluation of the code (Python object)."""  # noqa:D401
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
    """Test a single notebook.

    This uses nbformat to append `nteract-scrapbook` commands to the
    specified notebook. The content of the commands and their expected
    values are stored in the `test_values` dictionary. The keys of this
    dictionary are strings to be used as scrapbook keys. They corresponding
    value is a `ScrapSpec` tuple. The `code` member of this tuple is
    the code (as a string) to be run to generate the scrapbook value. The
    `expected` member is a Python object which is checked for equality with
    the scrapbook value

    Makes certain assumptions about directory layout.
    """
    input_notebook = "notebooks/" + notebook_name + ".ipynb"
    processed_notebook = "./test/notebooks/" + notebook_name + ".processed.ipynb"
    output_notebook = "./test/notebooks/" + notebook_name + ".output.ipynb"

    append_scrapbook_commands(input_notebook, processed_notebook, test_values)
    pm.execute_notebook(processed_notebook, output_notebook)
    nb = sb.read_notebook(output_notebook)

    for k, v in test_values.items():
        assert nb.scraps[k].data == v.expected


@pytest.mark.notebooks
def test_grid_search_for_binary_classification():
    nb_name = "Grid Search for Binary Classification"

    test_values = {}
    test_values["best_lambda_second_grid"] = ScrapSpec(
        "lambda_best_second", pytest.approx(0.8333333333))
    test_values["best_coeff_second0"] = ScrapSpec(
        "second_sweep.predictors_[second_sweep.best_idx_].coef_[0][0]",
        pytest.approx(2.53725364))

    assay_one_notebook(nb_name, test_values)


@pytest.mark.notebooks
def test_mitigating_disparities_in_ranking_from_binary_data():
    nb_name = "Mitigating Disparities in Ranking from Binary Data"
    test_values = {}
    # Needs wider bound due to randomness in ExponentiatedGradient
    test_values["sel_eg_X_alt_disparity"] = ScrapSpec(
        "sel_expgrad_X_alt.loc[ 'disparity', :][0]",
        pytest.approx(0.35, abs=0.08))
    assay_one_notebook(nb_name, test_values)


@pytest.mark.notebooks
def test_binary_classification_with_the_uci_credit_card_default_dataset():
    nb_name = "Binary Classification with the UCI Credit-card Default Dataset"
    test_values = {}
    test_values["Contains_Unmitigated"] = ScrapSpec(
        "'Unmitigated' in model_sweep_dict",
        True
    )
    test_values["Contains_ThresholdOptimizer"] = ScrapSpec(
        "'ThresholdOptimizer' in model_sweep_dict",
        True
    )
    assay_one_notebook(nb_name, test_values)
