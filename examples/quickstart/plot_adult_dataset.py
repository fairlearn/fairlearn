"""
==============================
Plotting the UCI Adult Dataset
==============================
"""
print(__doc__)

from fairlearn.datasets import fetch_adult
import matplotlib.pyplot as plt


data = fetch_adult(as_frame=True)
X = data.data
y_true = (data.target == '>50K') * 1
sex = X['sex'].apply(lambda sex: "female" if sex == 0 else "male")


def percentage_with_label_1(sex_value):
    return y_true[sex == sex_value].sum() / (sex == sex_value).sum()


plt.bar([0, 1], [percentage_with_label_1("female"), percentage_with_label_1("male")], color='g')
plt.xticks([0, 1], ["female", "male"])
plt.ylabel("percentage earning over $50,000")
plt.xlabel("sex")
plt.show()
