#!/usr/bin/env python3
"""One-command setup for the Facebook Reel Generator on macOS, Windows or Linux.

    python3 bootstrap.py              # create .venv, install packages, check ffmpeg, fetch the lyric model
    python3 bootstrap.py --check      # report only, install nothing
    python3 bootstrap.py --no-whisper # skip the ~1.5 GB Whisper model download (prep fetches it later)

Standard library only, so it runs before anything is installed. Exit code 0 = ready to make reels.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PY = (3, 9)
TESTED_MAX_PY = (3, 13)  # newer interpreters may not have torch / whisper wheels yet
IMPORT_CHECK = "import cv2, numpy, librosa, scenedetect, whisper, PIL, yaml, torch"
MAC_FONT = Path("/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf")
BUNDLED_FONT = ROOT / "assets" / "fonts" / "BarlowCondensed-Bold.ttf"

HINTS = {
    ("ffmpeg", "darwin"): "brew install ffmpeg    (no Homebrew yet? https://brew.sh installs it with one command)",
    ("ffmpeg", "win32"): "winget install --id Gyan.FFmpeg -e    (then close and reopen the app so PATH refreshes)",
    ("ffmpeg", "linux"): "sudo apt install ffmpeg    (or your distro's package manager)",
    ("python", "darwin"): "brew install python@3.12    (or: xcode-select --install)",
    ("python", "win32"): "winget install --id Python.Python.3.12 -e    (tick 'Add python to PATH' if asked)",
    ("python", "linux"): "sudo apt install python3 python3-venv python3-pip",
}

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def say(msg: str) -> None:
    print(msg, flush=True)


def venv_python(root: Path, plat: str = sys.platform) -> Path:
    """The interpreter inside the project's virtualenv, per OS layout."""
    if plat.startswith("win"):
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def python_ok(version=tuple(sys.version_info[:3])) -> bool:
    return tuple(version[:2]) >= MIN_PY


def install_hint(tool: str, plat: str = sys.platform) -> str:
    key = "win32" if plat.startswith("win") else ("darwin" if plat == "darwin" else "linux")
    return HINTS[(tool, key)]


def run(cmd: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    say("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=check, **kw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only; install nothing")
    ap.add_argument("--no-whisper", action="store_true", help="skip pre-downloading the Whisper lyric model")
    args = ap.parse_args(argv)
    problems: list[str] = []

    say(f"Facebook Reel Generator setup  ({platform.system()} {platform.release()}, {platform.machine()})")
    say(f"Python {platform.python_version()}  at {sys.executable}")
    if not python_ok():
        problems.append(f"Python {MIN_PY[0]}.{MIN_PY[1]} or newer is required. Install: {install_hint('python')}")
    elif sys.version_info[:2] > TESTED_MAX_PY:
        say("  note: this Python is newer than the tested range; if the package install fails, "
            "install Python 3.12 and rerun with it")

    vpy = venv_python(ROOT)
    if not vpy.exists() and not problems:
        if args.check:
            problems.append("no .venv yet (run bootstrap.py without --check to create it)")
        else:
            say("Creating .venv (a private Python environment for this project) ...")
            run([sys.executable, "-m", "venv", str(ROOT / ".venv")])
    if vpy.exists() and not args.check:
        say("Installing Python packages (first time: several minutes and about 1 GB) ...")
        run([str(vpy), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=False)
        r = run([str(vpy), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=False)
        if r.returncode != 0:
            problems.append("pip install failed (see the errors above). Known-good versions are in "
                            "requirements.lock.txt if a newer package broke.")

    deps_ok = False
    if vpy.exists():
        r = subprocess.run([str(vpy), "-c", IMPORT_CHECK], capture_output=True, text=True)
        deps_ok = r.returncode == 0
        say(f"Python packages: {'ok' if deps_ok else 'MISSING'}")
        if not deps_ok:
            problems.append("Python packages did not import:\n    " + r.stderr.strip()[-600:].replace("\n", "\n    "))

    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        say(f"{tool}: {path or 'MISSING'}")
        if not path:
            problems.append(f"{tool} is not installed or not on PATH. Install: {install_hint('ffmpeg')}")

    if MAC_FONT.is_file():
        say("caption font: DIN Condensed Bold (macOS)")
    elif BUNDLED_FONT.is_file():
        say("caption font: bundled Barlow Condensed Bold")
    else:
        problems.append("assets/fonts/BarlowCondensed-Bold.ttf is missing - re-download the project")

    if deps_ok and shutil.which("ffmpeg") and not args.check and not args.no_whisper:
        say("Fetching the Whisper lyric model (one time, about 1.5 GB) ...")
        code = ("import yaml, whisper; cfg = yaml.safe_load(open('config.yaml', encoding='utf-8')); "
                "m = (cfg.get('lyrics') or {}).get('model', 'medium'); whisper.load_model(m); "
                "print('whisper model ready:', m)")
        r = run([str(vpy), "-c", code], cwd=str(ROOT), check=False)
        if r.returncode != 0:
            problems.append("Whisper model download failed (internet?). prep will retry it on first use.")

    say("")
    if problems:
        say("SETUP INCOMPLETE - fix these, then run bootstrap.py again:")
        for p in problems:
            say("  - " + p)
        return 1
    say("SETUP COMPLETE. Run pipeline steps with this interpreter (no 'activate' needed), e.g.")
    say(f"  {vpy} -m pipeline.prep projects/<name>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
