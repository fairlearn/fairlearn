# Assignment 4 — studying LFR with fairlearn

This branch is a fork of [fairlearn/fairlearn](https://github.com/fairlearn/fairlearn).
fairlearn supplies the implementation I used,
[`PrototypeRepresentationLearner`](https://fairlearn.org/main/api_reference/generated/fairlearn.preprocessing.PrototypeRepresentationLearner.html),
which follows Zemel et al. (2013), *Learning Fair Representations*, ICML 2013.

I did not touch the library. The `a4_study/` folder holds my own study of it: the data
loaders, the fairness metrics, the sweep over the `fairness_weight` knob, the final
comparison on Adult / German Credit / COMPAS, and the README that explains the setup, the
results and what I think they mean.

Start at [`a4_study/README.md`](a4_study/README.md).

The upstream repository's own instructions for installing fairlearn from source are in the
root [README.md](../README.md); I used a plain `pip install fairlearn==0.15.0` instead,
which is what `a4_study/requirements.txt` pins.
