"""
==============================
Plotting the UCI Adult Dataset
==============================
"""
print(__doc__)

import matplotlib.pyplot as plt
from shap.datasets import adult  # shap is only used its dataset utility
X, y_true = adult()
y_true = y_true * 1
sex = X['Sex'].apply(lambda sex: "female" if sex == 0 else "male")


def percentage_with_label_1(sex_value):
    return y_true[sex == sex_value].sum() / (sex == sex_value).sum()


plt.bar([0, 1], [percentage_with_label_1("female"), percentage_with_label_1("male")], color='g')
plt.xticks([0, 1], ["female", "male"])
plt.ylabel("percentage earning over $50,000")
plt.xlabel("sex")
plt.show()
