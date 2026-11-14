"""Download the three datasets into data/.

The files live in the repository, so you do not need this script to run the
experiments. It exists so the data can be re-fetched from the original sources,
and so the loaders can be checked against a fresh download.
"""

import pathlib
import urllib.request

URLS = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    "german.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
}


def main():
    here = pathlib.Path(__file__).resolve().parents[1]
    out = here / "data"
    out.mkdir(exist_ok=True)
    for name, url in URLS.items():
        path = out / name
        if path.exists():
            print(f"{name}: already there, keeping it")
            continue
        print(f"{name}: downloading")
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        path.write_bytes(data)
        print(f"{name}: {len(data)} bytes")

    compas = out / "compas-scores-two-years.csv"
    if compas.exists():
        print("compas-scores-two-years.csv: already there")
    else:
        print("compas-scores-two-years.csv: not in the repo checkout, see data/README.md")


if __name__ == "__main__":
    main()
