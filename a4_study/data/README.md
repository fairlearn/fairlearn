# Data sources

- `adult.data`, `adult.test` — UCI Machine Learning Repository, Census Income
  (Adult) dataset, downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/adult/
- `german.data` — UCI Machine Learning Repository, Statlog (German Credit) dataset,
  https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/
- `compas-scores-two-years.csv` — the ProPublica download of the Northpointe
  COMPAS two-year recidivism scores (the same export the fairlearn tutorial
  uses), https://www.propublica.org/datastore/dataset/compas-recidivism

All three are public, require no registration, and are used in the fairness
literature as-is. Nothing is modified in place; the cleaning each loader applies
is documented in `src/data_loading.py`.
