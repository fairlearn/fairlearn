"""Download the three datasets into data/.

The files are committed, so you don't need this to run the experiments. It's
here so the committed copies can be checked against the original sources.

  python src/fetch_data.py
"""

import pathlib
import urllib.request

URLS = {
    "adult.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    "german.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
    "compas-two-years.csv": "https://raw.githubusercontent.com/compas-dev/compas/main/compas-two-years.csv",
}


def main():
    out = pathlib.Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    for name, url in URLS.items():
        path = out / name
        if path.exists():
            print(f"{name}: already there, keeping it")
            continue
        print(f"{name}: downloading")
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
        print(f"{name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
