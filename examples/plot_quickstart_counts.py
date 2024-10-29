# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
=============
Value counts
=============

.. note::

    View this example on
    `GitHub <https://github.com/fairlearn/fairlearn/blob/main/examples/plot_quickstart_counts.py>`_,
    or open an `issue
    <https://github.com/fairlearn/fairlearn/issues/new?title=DOC%20Issue:%20Quickstart%20Counts&body=https://github.com/fairlearn/fairlearn/blob/main/examples/plot_quickstart_counts.py>`_
    related to it.
"""
import matplotlib.pyplot as plt

from fairlearn.datasets import fetch_diabetes_hospital

# %%
fig, ax = plt.subplots()

data = fetch_diabetes_hospital(as_frame=True)
X = data.data.copy()
X.drop(columns=["readmitted", "readmit_binary"], inplace=True)
y_true = data.target
race = X["race"]

df = race.value_counts().reset_index()

ax.bar(df["race"], df["count"])
ax.set_title("Counts by race")
ax.tick_params(axis="x", labelrotation=45)

plt.tight_layout()
plt.show()
