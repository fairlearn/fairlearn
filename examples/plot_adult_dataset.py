"""
==============================
Plotting the UCI Adult Dataset
==============================
"""
import matplotlib.pyplot as plt
from fairlearn.metrics import selection_rate_group_summary
from fairlearn.datasets import fetch_adult
print(__doc__)


data = fetch_adult(as_frame=True)
X = data.data
y_true = (data.target == '>50K') * 1
sex = X['sex']

selection_rates = selection_rate_group_summary(y_true, y_true, sensitive_features=sex)

plt.bar([0, 1], [selection_rates.by_group["Female"], selection_rates.by_group["Male"]], color='g')
plt.xticks([0, 1], ["Female", "Male"])
plt.ylabel("percentage earning over $50,000")
plt.xlabel("sex")
plt.show()
