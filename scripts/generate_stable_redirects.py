"""Generate stable documentation redirects for Fairlearn's GitHub Pages site."""

import argparse
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

STABLE_URL_RE = re.compile(r"https://fairlearn\.org/(v\d+\.\d+)/?")


def stable_directory(versions_file: Path) -> str:
    """Find the stable release in the documentation version switcher."""
    versions = json.loads(versions_file.read_text(encoding="utf-8"))
    stable = [item for item in versions if "(stable)" in item["name"]]

    if len(stable) != 1:
        raise ValueError("Expected exactly one stable version in versions.json")

    match = STABLE_URL_RE.fullmatch(stable[0]["url"])
    if match is None:
        raise ValueError("Stable version must have a fairlearn.org/vX.Y/ URL")

    return match.group(1)


def published_html_pages(site_repo: Path, version_dir: str) -> list[str]:
    """List published HTML pages, including sparse-checkout exclusions."""
    result = subprocess.run(
        [
            "git",
            "-C",
            str(site_repo),
            "ls-tree",
            "-r",
            "--name-only",
            "HEAD",
            "--",
            f"{version_dir}/",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    prefix = f"{version_dir}/"
    pages = [
        path[len(prefix) :]
        for path in result.stdout.splitlines()
        if path.startswith(prefix) and path.endswith(".html")
    ]

    if "index.html" not in pages:
        raise ValueError(f"Published documentation {version_dir}/index.html not found")

    return pages


def redirect_page(url: str) -> str:
    """Generate an HTML redirect preserving URL queries and fragments."""
    escaped = html.escape(url, quote=True)

    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f'<meta http-equiv="refresh" content="0; url={escaped}">\n'
        f'<link rel="canonical" href="{escaped}">\n'
        "<title>Fairlearn documentation redirect</title>\n"
        "<script>\n"
        f"window.location.replace({json.dumps(url)}"
        " + window.location.search + window.location.hash);\n"
        "</script>\n</head>\n<body>\n"
        f'<p>This page has moved to <a href="{escaped}">'
        f"{escaped}</a>.</p>\n"
        "</body>\n</html>\n"
    )


def generate(site_repo: Path, versions_file: Path) -> tuple[str, int]:
    """Generate redirects matching the currently published stable release."""
    version_dir = stable_directory(versions_file)
    pages = published_html_pages(site_repo, version_dir)

    stable_dir = site_repo / "stable"
    if stable_dir.exists():
        shutil.rmtree(stable_dir)

    for relative in pages:
        target = stable_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        url = f"https://fairlearn.org/{version_dir}/{relative}"
        target.write_text(redirect_page(url), encoding="utf-8")

    # Sphinx needs the actual inventory, not an HTML redirect.
    # Read from Git because sparse checkout may exclude this file.
    try:
        inventory = subprocess.run(
            [
                "git",
                "-C",
                str(site_repo),
                "show",
                f"HEAD:{version_dir}/objects.inv",
            ],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"Published documentation {version_dir}/objects.inv not found"
        ) from exc

    (stable_dir / "objects.inv").write_bytes(inventory)

    return version_dir, len(pages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_repo", type=Path)
    parser.add_argument("versions_file", type=Path)
    args = parser.parse_args()

    version, count = generate(args.site_repo, args.versions_file)
    print(f"Generated {count} stable redirects to {version}/")


if __name__ == "__main__":
    main()
