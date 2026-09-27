"""Cross-platform guarantees: the pipeline must run the same on macOS, Windows and Linux.

Run:  python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline import common  # noqa: E402
from pipeline import assemble  # noqa: E402


class CaptionFontTests(unittest.TestCase):
    def test_default_caption_font_is_an_existing_file(self):
        self.assertTrue(Path(common.caption_font()).is_file())

    def test_falls_back_to_bundled_font_when_no_system_font_exists(self):
        path = common.caption_font(["/definitely/not/here.ttf"])
        self.assertEqual(path, str(common.BUNDLED_FONT))
        self.assertTrue(Path(path).is_file(), "bundled font missing from assets/fonts")

    def test_bundled_font_renders_in_pillow(self):
        from PIL import ImageFont
        font = ImageFont.truetype(str(common.BUNDLED_FONT), 64)
        self.assertGreater(font.getlength("DAY 1 · THE CARRY"), 0)


class FfmpegPathTests(unittest.TestCase):
    def test_drawtext_escapes_windows_drive_colon_in_fontfile(self):
        s = assemble._drawtext("hi", r"C:\Windows\Fonts\arial.ttf", 1920)
        self.assertIn("fontfile='C\\:/Windows/Fonts/arial.ttf'", s)

    def test_drawtext_without_font_has_no_fontfile(self):
        self.assertNotIn("fontfile", assemble._drawtext("hi", None, 1920))

    def test_find_font_never_returns_none(self):
        self.assertEqual(assemble._find_font(["/nope.ttf"]), str(common.BUNDLED_FONT))

    def test_concat_list_uses_forward_slashes_and_quotes_apostrophes(self):
        text = assemble._concat_list_text([PureWindowsPath(r"C:\w\clip_000.mp4"),
                                           PureWindowsPath(r"C:\Ian's Trip\clip_001.mp4")])
        self.assertEqual(text, "file 'C:/w/clip_000.mp4'\nfile 'C:/Ian'\\''s Trip/clip_001.mp4'\n")


class Utf8Tests(unittest.TestCase):
    def test_log_survives_a_cp1252_console(self):
        # Simulates Windows piping stdout through a cp1252 code page.
        code = "from pipeline.common import log; log('reel → lyric ’ · —')"
        env = dict(os.environ, PYTHONIOENCODING="cp1252")
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("reel", r.stdout)

    @unittest.skipIf(sys.platform.startswith("win"), "POSIX locale env only")
    def test_read_json_decodes_utf8_under_a_latin1_locale(self):
        # Simulates a machine whose default text encoding is not UTF-8 (Windows cp1252 and friends).
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "captions.json"
            p.write_bytes('{"sub": "luke knew the way · i just knew luke"}'.encode("utf-8"))
            code = "import json, sys; from pipeline.common import read_json; print(json.dumps(read_json(sys.argv[1])))"
            env = dict(os.environ, LC_ALL="en_US.ISO8859-1", LANG="en_US.ISO8859-1")
            r = subprocess.run([sys.executable, "-c", code, str(p)], cwd=REPO, env=env,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.loads(r.stdout)["sub"], "luke knew the way · i just knew luke")


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        import bootstrap  # repo-root script; imported here so a missing file fails only these tests
        self.b = bootstrap
    def test_venv_python_path_per_platform(self):
        root = Path("/repo")
        self.assertEqual(self.b.venv_python(root, "win32"), root / ".venv" / "Scripts" / "python.exe")
        self.assertEqual(self.b.venv_python(root, "darwin"), root / ".venv" / "bin" / "python")
        self.assertEqual(self.b.venv_python(root, "linux"), root / ".venv" / "bin" / "python")

    def test_python_version_floor(self):
        self.assertFalse(self.b.python_ok((3, 8, 10)))
        self.assertTrue(self.b.python_ok((3, 9, 6)))
        self.assertTrue(self.b.python_ok((3, 13, 1)))

    def test_install_hints_name_the_platform_package_manager(self):
        self.assertIn("brew install ffmpeg", self.b.install_hint("ffmpeg", "darwin"))
        self.assertIn("winget", self.b.install_hint("ffmpeg", "win32"))
        self.assertIn("apt", self.b.install_hint("ffmpeg", "linux"))
        self.assertIn("winget", self.b.install_hint("python", "win32"))


if __name__ == "__main__":
    unittest.main()
