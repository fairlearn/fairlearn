# Notebooks and Fairlearn Versioning

We are still exploring possibilities for the APIs in Fairlearn,
meaning that we sometimes have breaking changes.
As a result, notebooks taken from the `master` branch on GitHub may
not always be compatible with a version of Fairlearn installed using
`pip` (notebooks on `master` will always be compatible with `master`).

If you encounter compatibility problems with notebooks, go to your terminal and run:
```
pip show fairlearn
```
to show the version of Fairlean which you currently have
installed.
Then, on the GitHub page, navigate to that version
(e.g. [v0.4.5](https://github.com/fairlearn/fairlearn/tree/v0.4.5) or
[v0.4.6](https://github.com/fairlearn/fairlearn/tree/v0.4.6)),
and download the notebooks from there.

Alternatively, [install Fairlearn from a cloned repository](https://fairlearn.github.io/contributor_guide/development_process.html#advanced-installation-instructions)
in order to use the notebooks from `master`.