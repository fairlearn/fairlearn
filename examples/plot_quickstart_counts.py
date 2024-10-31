# Copyright (c) Microsoft Corporation and Fairlearn contributors.
# Licensed under the MIT License.

"""
=============
Value counts
=============
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
