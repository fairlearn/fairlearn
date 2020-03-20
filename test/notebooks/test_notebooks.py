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
def test_group_metrics_notebook():
    overall_recall_key = "overall_recall"
    by_groups_key = "recall_by_groups"

    test_values = {}
    test_values[overall_recall_key] = ScrapSpec("group_metrics.overall", 0.5)
    test_values[by_groups_key] = ScrapSpec(
        "results.by_group", {'a': 0.0, 'b': 0.5, 'c': 0.75, 'd': 0.0})

    assay_one_notebook("Group Metrics", test_values)


@pytest.mark.notebooks
def test_grid_search_for_binary_classification():
    nb_name = "Grid Search for Binary Classification"

    test_values = {}
    test_values["best_lambda_second_grid"] = ScrapSpec(
        "lambda_best_second", pytest.approx(0.8333333333))
    test_values["best_coeff_second0"] = ScrapSpec(
        "second_sweep.best_result.predictor.coef_[0][0]", pytest.approx(2.53725364))

    assay_one_notebook(nb_name, test_values)


@pytest.mark.notebooks
def test_binary_classification_on_compas_dataset():
    nb_name = "Binary Classification on COMPAS dataset"

    test_values = {}
    test_values["pp_eo_aa_pignore"] = ScrapSpec(
        "postprocessed_predictor_EO._post_processed_predictor_by_sensitive_feature['African-American']._p_ignore",  # noqa: E501
        pytest.approx(0.2320703126)
    )

    assay_one_notebook(nb_name, test_values)


@pytest.mark.notebooks
def test_grid_search_with_census_data():
    nb_name = "Grid Search with Census Data"
    test_values = {}
    test_values["len_nondominated"] = ScrapSpec("len(non_dominated)", 13)
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
def test_binary_classification_with_the_uci_credit_card_payment_default_dataset():
    nb_name = "Binary Classification with the UCI Credit Card Payment Default Dataset"
    test_values = {}
    assay_one_notebook(nb_name, test_values)
