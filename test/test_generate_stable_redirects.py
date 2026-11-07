"""Tests for stable documentation redirects."""

import json
import subprocess
import tempfile
import unittest
import zlib
from pathlib import Path

from scripts.generate_stable_redirects import generate, stable_directory


class TestStableRedirects(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.root = Path(self.temp.name)
        self.site = self.root / "website"
        self.versions = self.root / "versions.json"

        subprocess.run(
            ["git", "init", "-q", str(self.site)],
            check=True,
        )

        for page in (
            "v0.14/index.html",
            "v0.14/legacy.html",
            "v0.15/index.html",
            "v0.15/api_reference/generated/example.html",
        ):
            path = self.site / page
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Documentation", encoding="utf-8")

        for version in ("v0.14", "v0.15"):
            inventory = self.site / version / "objects.inv"
            inventory.write_bytes(
                b"# Sphinx inventory version 2\n"
                b"# Project: Fairlearn\n"
                + f"# Version: {version}\n".encode()
                + b"# The remainder of this file is compressed using zlib.\n"
                + zlib.compress(b"example py:class 1 api_reference/generated/example.html -\n")
            )

        subprocess.run(
            ["git", "-C", str(self.site), "add", "."],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(self.site),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "Add documentation",
            ],
            check=True,
        )

        self.set_versions("v0.15")

    def set_versions(self, version):
        self.versions.write_text(
            json.dumps(
                [
                    {
                        "name": f"{version}.0 (stable)",
                        "url": f"https://fairlearn.org/{version}/",
                    },
                ]
            ),
            encoding="utf-8",
        )

    def test_selects_stable_version(self):
        self.assertEqual(
            stable_directory(self.versions),
            "v0.15",
        )

    def test_generates_homepage_and_deep_links(self):
        version, count = generate(self.site, self.versions)

        self.assertEqual(version, "v0.15")
        self.assertEqual(count, 2)

        homepage = self.site / "stable/index.html"
        api_page = self.site / "stable/api_reference/generated/example.html"

        self.assertTrue(homepage.is_file())
        self.assertTrue(api_page.is_file())

        content = api_page.read_text(encoding="utf-8")
        self.assertIn(
            "https://fairlearn.org/v0.15/api_reference/generated/example.html",
            content,
        )
        self.assertIn(
            "window.location.search + window.location.hash",
            content,
        )

    def test_copies_inventory_from_published_git_tree(self):
        """The inventory is available even with a sparse checkout."""
        source = self.site / "v0.15/objects.inv"
        expected = source.read_bytes()

        # Simulate the inventory being excluded by sparse checkout.
        source.unlink()

        generate(self.site, self.versions)

        stable_inventory = self.site / "stable/objects.inv"
        self.assertTrue(stable_inventory.is_file())
        self.assertEqual(stable_inventory.read_bytes(), expected)

        parts = stable_inventory.read_bytes().split(b"\n", 4)
        self.assertEqual(parts[0], b"# Sphinx inventory version 2")
        self.assertEqual(len(parts), 5)
        self.assertIn(
            b"example py:class",
            zlib.decompress(parts[4]),
        )

    def test_new_release_replaces_old_redirects(self):
        self.set_versions("v0.14")
        generate(self.site, self.versions)

        old_page = self.site / "stable/legacy.html"
        self.assertTrue(old_page.exists())

        self.set_versions("v0.15")
        generate(self.site, self.versions)

        self.assertFalse(old_page.exists())
        self.assertTrue((self.site / "stable/api_reference/generated/example.html").exists())

    def test_rejects_missing_stable_version(self):
        self.versions.write_text("[]", encoding="utf-8")

        with self.assertRaises(ValueError):
            stable_directory(self.versions)

    def test_rejects_invalid_stable_url(self):
        self.versions.write_text(
            json.dumps(
                [
                    {
                        "name": "v0.15.0 (stable)",
                        "url": "https://example.com/v0.15/",
                    },
                ]
            ),
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            stable_directory(self.versions)

    def test_rejects_unpublished_version(self):
        self.set_versions("v0.16")

        with self.assertRaises(ValueError):
            generate(self.site, self.versions)


if __name__ == "__main__":
    unittest.main()
