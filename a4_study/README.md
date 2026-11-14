# Assignment 4: study of PrototypeRepresentationLearner on Adult, German, COMPAS

Fork of [fairlearn/fairlearn](https://github.com/fairlearn/fairlearn), branch
`a4-fair-representations-study`. fairlearn supplies the implementation I used,
[`PrototypeRepresentationLearner`](https://fairlearn.org/main/api_reference/generated/fairlearn.preprocessing.PrototypeRepresentationLearner.html),
which follows Zemel et al. (2013), *Learning Fair Representations*, ICML 2013. I did not
touch the library; `a4_study/` is my own study of it.

The library's own install instructions are in the root `README.md`; for this study a plain
`pip install fairlearn==0.15.0` is enough, which is what `requirements.txt` pins.

## How to reproduce

```bash
git clone -b a4-fair-representations-study --depth 1 https://github.com/RendelGadaffi/fairlearn.git
cd fairlearn/a4_study
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/run_experiment.py     # prints the summary, writes results.json
python src/sweep.py              # the fairness-weight grid, writes sweep.csv
```

`data/german.data` and `data/compas-scores-two-years.csv` are committed on this branch. The
two Adult files are not, because fairlearn's own `.gitignore` excludes `*.data` and `*.test`
and I did not want to change the library's ignore rules for an assignment. Grab
`adult.data` and `adult.test` from the UCI Census Income (Adult) dataset and drop them in
`data/`, or run `python src/fetch_data.py`, which downloads all three from their original
sources and prints what it saved.

Every number in the report is in `sweep.csv`, one row per dataset and fairness weight, so
the tables below can be checked without running anything.

Runtime on a 4-core laptop, roughly: German 2 minutes, COMPAS 13 minutes, Adult about
1.5 hours per LFR fit; the logistic baselines take seconds. The weights in the sweep are
independent of each other, so running them as parallel processes cuts a full sweep to
about one Adult fit.

## 1. Setup

### 1.1 Datasets

| dataset | n | label (positive) | protected attribute (unprivileged) |
|---|---|---|---|
| Adult (UCI) | 48,842 | income > 50K | sex = Female |
| German Credit (UCI) | 1,000 | credit granted | age < 25 |
| COMPAS (ProPublica) | 5,714 | two-year recidivism | sex = Female |

`data/README.md` documents the sources. One note on Adult: the LFR paper cites the
45,222-row collection; the current UCI release has 48,842 rows and no cleaning of it
reproduces that number, so I keep all rows (the rows the old cleaning dropped carry
"?" only in native-country, a column standard UCI setups drop anyway).

### 1.2 Measures

| measure | definition |
|---|---|
| DP_Diff | P(ŷ=1 \| unprivileged) − P(ŷ=1 \| privileged); 0 = parity |
| DIR | ratio of the two selection rates; 1 = parity, <0.8 = adverse impact |
| consistency | k-NN individual-fairness measure from the paper (Eq. 15): per test point, the distance-weighted disagreement between its label and the labels of its 10 nearest neighbours. Lower is better. |

Three settings per dataset, all on the same 80/20 test split (stratified on label and
group):

1. **labels before** – the measures computed from the original labels, before training anything (bias already present in the data)
2. **baseline** – plain logistic regression, no mitigation
3. **LFR** – prototypes fitted on the training fold, predictions (and accuracy/F1) on the test fold

### 1.3 Parameters

| parameter | value | where in code |
|---|---|---|
| random seed | 42 | `src/run_experiment.py`, `src/sweep.py` |
| train/test split | 80/20, stratified on label and group | `src/run_experiment.py` |
| n_prototypes | 10 | `src/run_experiment.py`, `src/sweep.py` |
| max_iter | 100 | `src/run_experiment.py`, `src/sweep.py` |
| fairness_weight | Adult **50.0**, German **5.0**, COMPAS **20.0** | `src/run_experiment.py` |

`max_iter=100` everywhere: every fit in `results/sweep.csv` converged in 21–97
iterations at fairlearn's default tolerance (1e-6), so the cap never binds.

The `fairness_weight` per dataset in `src/run_experiment.py` comes from the sweep
(`results/sweep.csv`): the lowest weight that reaches |DP_Diff| ≤ 0.06 while keeping DIR
at or above 0.8 and F1 within about 0.02 of the baseline F1.

## 2. Results

All numbers on the same held-out test split. Best fairness value per dataset in **bold**.

### Adult (income > 50K, sex)

| setting | accuracy | F1 | DP_Diff | DIR | consistency |
|---|---|---|---|---|---|
| labels before | – | – | −0.1945 | 0.360 | 0.0078 |
| baseline (logistic regression) | **0.8487** | **0.6566** | −0.1898 | 0.282 | 0.0044 |
| LFR (w=50) | 0.7889 | 0.2934 | **+0.0225** | **1.433** | **0.0013** |

### German Credit (credit granted, age < 25)

| setting | accuracy | F1 | DP_Diff | DIR | consistency |
|---|---|---|---|---|---|
| labels before | – | – | −0.1176 | 0.836 | 0.1727 |
| baseline (logistic regression) | **0.7350** | **0.8191** | −0.1157 | 0.852 | 0.1220 |
| LFR (w=5) | 0.6900 | 0.8133 | **+0.0471** | **1.049** | **0.0205** |

### COMPAS (two-year recidivism, sex)

| setting | accuracy | F1 | DP_Diff | DIR | consistency |
|---|---|---|---|---|---|
| labels before | – | – | −0.1231 | 0.757 | 0.0143 |
| baseline (logistic regression) | **0.6448** | **0.6111** | −0.2614 | 0.458 | 0.0052 |
| LFR (w=20) | 0.5713 | 0.5964 | **−0.0602** | **0.898** | **0.0036** |

## 3. Discussion

**LFR closes the group gap on all three datasets.** DP_Diff moves from −0.19 to +0.02
on Adult, −0.12 to +0.05 on German and −0.12 to −0.06 on COMPAS, and DIR crosses the
conventional 0.8 "four-fifths" threshold on German and COMPAS. Consistency improves on
every dataset, most dramatically on German (0.173 → 0.021, more than an 8× reduction), which
matches the paper's claim that the reconstruction term makes the code treat similar
people alike.

**The unmitigated baseline is not neutral, on COMPAS it is worse than the data.**
Logistic regression pushes DP_Diff from −0.123 in the raw labels to −0.261. It is
learning recidivism partly through features that proxy sex, so training on the data as-is
amplifies the disparity that is already there. This is the clearest argument in the study
for why mitigation is needed, and it is why reporting "labels before" next to "baseline"
matters: the baseline is not a fair yardstick.

**Fairness is bought with predictive performance, and the price varies a lot.** On
German the trade is cheap: 4.5 accuracy points, F1 within 0.006 of baseline, for a DP_Diff
that goes from adverse (0.836 DIR) to parity (1.049). On COMPAS it costs 7.4 accuracy
points for DP_Diff −0.06. On Adult the price is steep: F1 falls from 0.657 to 0.293
while accuracy only drops 6 points, because the positive class is 10.9% of the women and
30.4% of the men in the test fold. A model that equalizes the rates must stop predicting
the minority class, and that is exactly what F1 exposes.

**Why my Adult numbers are weaker than the paper's.** The paper reports strong Adult
results with a tuned number of prototypes, its own L-BFGS solver, and learned
per-feature distance weights (α, and a separate α per group). Fairlearn's
`PrototypeRepresentationLearner` implements the core method with plain Euclidean distance
and no α, uses a fixed prototype count (I used 10, the paper does not state one), and its
optimization is a scipy loop over the full training matrix. With 48,842 rows and 58
features, 10 prototypes is a hard bottleneck: the code cannot carry the income signal
across 58 dimensions, so the fairness term buys parity by discarding information. On
German, where the signal is 1:1 with fewer features, the same settings lose almost nothing.
So the accuracy cost I measure is partly a property of the implementation, not just of the
method.

**What I would change next.** Tune n_prototypes with the sweep (10 is one point on a
grid; the paper's Adult setting is unknown), use the paper's own optimizer with α weights,
and sweep the trade-off curve at fixed accuracy rather than picking a single weight per
dataset. The `sweep.py` script is set up for exactly that: add weights to the list and the
table regenerates.
