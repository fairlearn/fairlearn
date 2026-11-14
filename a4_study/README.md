# Learning Fair Representations (Zemel et al. 2013) — study for Assignment 4

Paper: *Learning Fair Representations*, Zemel, Wu, Swersky, Pitassi & Dwork, ICML 2013.
Code: `fairlearn.preprocessing.PrototypeRepresentationLearner` (fairlearn 0.15).

LFR is a **pre-processing** method. It learns K prototypes in the input space and maps
every person to a softmax over the negative distances to them. The mapping is trained to
keep reconstruction error low, keep label accuracy high, and make the two protected
groups use the prototypes at the same rate. A classifier then trains on the prototype
coordinates and inherits the fairness of the representation.

This repository contains the implementation I studied, the data, the sweep behind the
chosen parameters, the final comparison, and the two tables in the report.

## How to reproduce

```bash
git clone -b a4-fair-representations-study https://github.com/RendelGadaffi/fairlearn.git
cd fairlearn/a4_study
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py     # prints the summary and writes results/results.json
python src/sweep.py              # the fairness-weight grid, writes results/sweep.csv
```

`german.data` and `compas-scores-two-years.csv` are committed under `data/`. The two Adult
files are not: the upstream fairlearn `.gitignore` excludes `*.data` and `*.test`, and I did
not want to change the library's ignore rules for an assignment. Grab `adult.data` and
`adult.test` from the UCI Census Income (Adult) dataset and drop them in `data/`, or run
`python src/fetch_data.py`, which downloads all three from their original sources and prints
what it saved. From the repository root, clone with `--depth 1` to skip the library's history.

Expected output of `run_experiment.py` (the three lines below are what you should get;
`results/results.json` has the full numbers):

```
[adult]  done in ... (LFR iterations: 22)
[adult]  dp_diff -0.1945 -> +0.0225, accuracy 0.7889
[german] done in ... (LFR iterations: 24)
[german] dp_diff -0.1176 -> +0.0471, accuracy 0.6900
[compas] done in ... (LFR iterations: 94)
[compas] dp_diff -0.1231 -> -0.0602, accuracy 0.5713
```

Runtime on a 4-core laptop, roughly: German 2 minutes, COMPAS 13 minutes, Adult about
1.5 hours per LFR fit. The logistic baselines take seconds. `sweep.py` runs five fairness
weights per dataset, so a full sweep is about five Adult fits; the weights are independent
of each other, so running them as parallel processes cuts it to about one fit.


## 1. Setup

### 1.1 Datasets

15|
16|| dataset | n | label (positive) | protected attribute (unprivileged) |
17||---|---|---|---|
18|| Adult (UCI) | 48,842 | income > 50K | sex = Female |
19|| German Credit (UCI) | 1,000 | credit granted | age < 25 |
20|| COMPAS (ProPublica) | 5,714 | two-year recidivism | sex = Female |
21|
22|`data/README.md` documents the sources. One note on Adult: the LFR paper cites the
23|45,222-row collection; the current UCI release has 48,842 rows and no cleaning of it
24|reproduces that number, so I keep all rows (the rows the old cleaning dropped carry
25|"?" only in native-country, a column standard UCI setups drop anyway).

26|
27|### 1.2 Measures
28|
29|| measure | definition |
30||---|---|
31|| DP_Diff | P(ŷ=1 \| unprivileged) − P(ŷ=1 \| privileged); 0 = parity |
32|| DIR | ratio of the two selection rates; 1 = parity, <0.8 = adverse impact |
33|| consistency | k-NN individual-fairness measure from the paper (Eq. 15): per test point, the distance-weighted disagreement between its label and the labels of its 10 nearest neighbours. Lower is better. |
34|
35|Three settings per dataset, all on the same 80/20 test split (stratified on label and
36|group):
37|
38|1. **labels before** – the measures computed from the original labels, before training anything (bias already present in the data)
39|2. **baseline** – plain logistic regression, no mitigation
40|3. **LFR** – prototypes fitted on the training fold, predictions (and accuracy/F1) on the test fold
41|
42|### 1.3 Parameters
43|
44|| parameter | value | where in code |
45||---|---|---|
46|| random seed | 42 | `src/run_experiment.py`, `src/sweep.py` |
47|| train/test split | 80/20, stratified on label and group | `src/run_experiment.py` |
48|| n_prototypes | 10 | `src/run_experiment.py`, `src/sweep.py` |
49|| max_iter | 100 | `src/run_experiment.py`, `src/sweep.py` |
50|| fairness_weight | Adult **50.0**, German **5.0**, COMPAS **20.0** | `src/run_experiment.py` |
51|
52|`max_iter=100` everywhere: every fit in `results/sweep.csv` converged in 21–97
53|iterations at fairlearn's default tolerance (1e-6), so the cap never binds.
54|
55|## 2. Results
56|
57|All numbers on the same held-out test split. Best fairness value per dataset in **bold**.
58|
59|### Adult (income > 50K, sex)
60|
61|| setting | accuracy | F1 | DP_Diff | DIR | consistency |
62||---|---|---|---|---|---|
63|| labels before | – | – | −0.1945 | 0.360 | 0.0078 |
64|| baseline (logistic regression) | **0.8487** | **0.6566** | −0.1898 | 0.282 | 0.0044 |
65|| LFR (w=50) | 0.7889 | 0.2934 | **+0.0225** | **1.433** | **0.0013** |
66|
67|### German Credit (credit granted, age < 25)
68|
69|| setting | accuracy | F1 | DP_Diff | DIR | consistency |
70||---|---|---|---|---|---|
71|| labels before | – | – | −0.1176 | 0.836 | 0.1727 |
72|| baseline (logistic regression) | **0.7350** | **0.8191** | −0.1157 | 0.852 | 0.1220 |
73|| LFR (w=5) | 0.6900 | 0.8133 | **+0.0471** | **1.049** | **0.0205** |
74|
75|### COMPAS (two-year recidivism, sex)
76|
77|| setting | accuracy | F1 | DP_Diff | DIR | consistency |
78||---|---|---|---|---|---|
79|| labels before | – | – | −0.1231 | 0.757 | 0.0143 |
80|| baseline (logistic regression) | **0.6448** | **0.6111** | −0.2614 | 0.458 | 0.0052 |
81|| LFR (w=20) | 0.5713 | 0.5964 | **−0.0602** | **0.898** | **0.0036** |
82|
83|## 3. Discussion
84|
85|**LFR closes the group gap on all three datasets.** DP_Diff moves from −0.19 to +0.02
86|on Adult, −0.12 to +0.05 on German and −0.12 to −0.06 on COMPAS, and DIR crosses the
87|conventional 0.8 "four-fifths" threshold on German and COMPAS. Consistency improves on
88|every dataset, most dramatically on German (0.173 → 0.021, more than an 8× reduction), which
89|matches the paper's claim that the reconstruction term makes the code treat similar
90|people alike.
91|
92|**The unmitigated baseline is not neutral, on COMPAS it is worse than the data.**
93|Logistic regression pushes DP_Diff from −0.123 in the raw labels to −0.261. It is
94|learning recidivism partly through features that proxy sex, so training on the data as-is
95|amplifies the disparity that is already there. This is the clearest argument in the study
96|for why mitigation is needed, and it is why reporting "labels before" next to "baseline"
97|matters: the baseline is not a fair yardstick.
98|
99|**Fairness is bought with predictive performance, and the price varies a lot.** On
100|German the trade is cheap: 4.5 accuracy points, F1 within 0.006 of baseline, for a DP_Diff
101|that goes from adverse (0.836 DIR) to parity (1.049). On COMPAS it costs 7 accuracy
102|points for DP_Diff −0.06. On Adult the price is steep: F1 falls from 0.657 to 0.293
103|while accuracy only drops 6 points, because the positive class is 11% of the women and
104|45% of the men in the test fold. A model that equalizes the rates must stop predicting
105|the minority class, and that is exactly what F1 exposes.
106|
107|**Why my Adult numbers are weaker than the paper's.** The paper reports strong Adult
108|results with a tuned number of prototypes, its own L-BFGS solver, and learned
109|per-feature distance weights (α, and a separate α per group). Fairlearn's
110|`PrototypeRepresentationLearner` implements the core method with plain Euclidean distance
111|and no α, uses a fixed prototype count (I used 10, the paper does not state one), and its
112|optimization is a scipy loop over the full training matrix. With 48,842 rows and 58
113|features, 10 prototypes is a hard bottleneck: the code cannot carry the income signal
114|across 58 dimensions, so the fairness term buys parity by discarding information. On
115|German, where the signal is 1:1 with fewer features, the same settings lose almost nothing.
116|So the accuracy cost I measure is partly a property of the implementation, not just of the
117|method.
118|
119|**What I would change next.** Tune n_prototypes with the sweep (10 is one point on a
120|grid; the paper's Adult setting is unknown), use the paper's own optimizer with α weights,
121|and sweep the trade-off curve at fixed accuracy rather than picking a single weight per
122|dataset. The `sweep.py` script is set up for exactly that: add weights to the list and the
123|table regenerates.
124|
125|The `fairness_weight` per dataset in `src/run_experiment.py` comes from the sweep
142|(`results/sweep.csv`): the lowest weight that reaches |DP_Diff| ≤ 0.06 while keeping DIR
143|at or above 0.8 and F1 within about 0.02 of the baseline F1.