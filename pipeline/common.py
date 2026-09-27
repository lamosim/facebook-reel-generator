"""Shared helpers: paths, config loading, ffprobe, subprocess wrapper."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
ASSETS_DIR = REPO_ROOT / "assets"
# Caption / map-card font. DIN Condensed Bold ships with macOS and is the original look; any
# machine without it (Windows, Linux) falls back to the bundled SIL-OFL Barlow Condensed Bold,
# the closest free match (assets/fonts/).
BUNDLED_FONT = ASSETS_DIR / "fonts" / "BarlowCondensed-Bold.ttf"
CAPTION_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
]

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tif", ".tiff", ".bmp"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def caption_font(candidates: list[str] | None = None) -> str:
    """Path of the first caption font present on this machine, else the bundled one."""
    for cand in (CAPTION_FONT_CANDIDATES if candidates is None else candidates):
        if Path(cand).is_file():
            return str(cand)
    return str(BUNDLED_FONT)


# ── config ────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── project paths ───────────────────────────────────────────────────────────
@dataclass
class Project:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def music(self) -> Path:
        return self.root / "music"

    @property
    def work(self) -> Path:
        return self.root / "work"

    @property
    def frames(self) -> Path:
        return self.work / "frames"

    @property
    def clips(self) -> Path:
        return self.work / "clips"

    @property
    def output(self) -> Path:
        return self.root / "output"

    # generated artifacts
    @property
    def manifest_path(self) -> Path:
        return self.work / "manifest.json"

    @property
    def segments_path(self) -> Path:
        return self.work / "segments.json"

    @property
    def beats_path(self) -> Path:
        return self.work / "beats.json"

    @property
    def lyrics_path(self) -> Path:
        return self.work / "lyrics.json"

    @property
    def lyric_sheet_path(self) -> Path:
        return self.work / "lyric_sheet.json"

    @property
    def catalog_raw_path(self) -> Path:
        return self.work / "catalog_raw.json"

    @property
    def catalog_path(self) -> Path:
        return self.work / "catalog.json"

    @property
    def edl_path(self) -> Path:
        return self.work / "edl.json"

    @property
    def timeline_path(self) -> Path:
        return self.work / "timeline.json"

    @property
    def reel_path(self) -> Path:
        return self.output / "reel.mp4"

    def ensure_dirs(self) -> None:
        for d in (self.work, self.frames, self.clips, self.output):
            d.mkdir(parents=True, exist_ok=True)

    def raw_videos(self) -> list[Path]:
        if not self.raw.is_dir():
            return []
        return sorted(
            p for p in self.raw.iterdir()
            if p.is_file()
            and p.suffix.lower() in VIDEO_EXTS
            and p.stat().st_size > 0  # skip empty/untransferred files (e.g. iCloud)
        )

    def raw_images(self) -> list[Path]:
        if not self.raw.is_dir():
            return []
        return sorted(
            p for p in self.raw.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTS
            and p.stat().st_size > 0
        )

    def music_track(self) -> Path | None:
        if not self.music.is_dir():
            return None
        tracks = sorted(
            p for p in self.music.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        )
        return tracks[0] if tracks else None


def open_project(path_str: str) -> Project:
    root = Path(path_str)
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    if not root.is_dir():
        die(f"Project folder not found: {root}")
    return Project(root=root)


# ── subprocess / ffmpeg ──────────────────────────────────────────────────────
def run(cmd: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
    """Run a command, raising on failure with captured stderr."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr or "")
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd[:6])}…")
    return result


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        die(f"Required tool '{name}' not found on PATH.")


def ffprobe_info(path: Path) -> dict:
    """Return duration/width/height/fps/rotation/has_audio for a media file."""
    require_tool("ffprobe")
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(path),
    ]
    data = json.loads(run(cmd).stdout)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = data.get("format", {})

    try:
        duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0  # images report duration "N/A"

    info: dict = {
        "path": str(path),
        "name": path.name,
        "duration": duration,
        "has_audio": a is not None,
        "is_image": is_image(path),
    }
    if v is not None:
        info["width"] = int(v.get("width", 0) or 0)
        info["height"] = int(v.get("height", 0) or 0)
        info["fps"] = _parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1")
        info["rotation"] = _parse_rotation(v)
    return info


def _parse_fps(rate: str) -> float:
    try:
        num, den = rate.split("/")
        den = float(den)
        return float(num) / den if den else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _parse_rotation(stream: dict) -> int:
    # rotation can live in tags or side_data_list depending on ffmpeg version
    tags = stream.get("tags", {})
    if "rotate" in tags:
        try:
            return int(tags["rotate"]) % 360
        except ValueError:
            pass
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                return int(sd["rotation"]) % 360
            except (ValueError, TypeError):
                pass
    return 0


# ── json io ──────────────────────────────────────────────────────────────────
def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── logging ────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    print(f"[reel] {msg}", flush=True)


def die(msg: str) -> None:
    sys.stderr.write(f"[reel] ERROR: {msg}\n")
    sys.exit(1)
