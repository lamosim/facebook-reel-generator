"""Map the song's timestamped lyrics onto the REEL's timeline.

Once the music-start window is chosen (the same captivation picker render uses),
each lyric line gets a reel-relative time range — so Claude can author the EDL to
place each clip where its imagery emotionally matches the lyric playing then.

Usage:  python -m pipeline.lyricmap projects/<trip-name>

Writes work/lyric_sheet.json and prints a readable reel-time → lyric table.
Run this AFTER prep and BEFORE writing work/edl.json. If the EDL already pins
`music_start` (recommended, to lock alignment), that value is used; otherwise the
window is auto-selected with `music_section_strategy`.
"""
from __future__ import annotations

import sys

from . import common
from .build_timeline import _select_music_window
from .common import Project, log


def lyrics_in_window(
    lyrics: dict, music_start: float, target: float,
) -> list[dict]:
    """Lyric lines overlapping [music_start, music_start+target], expressed in
    REEL time (0 = first frame). Clipped to the window."""
    out: list[dict] = []
    for ln in lyrics.get("lines", []):
        rs = ln["start"] - music_start
        re = ln["end"] - music_start
        if re <= 0 or rs >= target:
            continue
        out.append({
            "reel_start": round(max(0.0, rs), 2),
            "reel_end": round(min(target, re), 2),
            "song_start": round(ln["start"], 2),
            "text": ln["text"],
        })
    return out


def lyric_at(sheet: list[dict], t0: float, t1: float) -> str | None:
    """The lyric line that most overlaps the reel-time span [t0, t1] (or None)."""
    best, best_ov = None, 0.0
    for ln in sheet:
        ov = min(t1, ln["reel_end"]) - max(t0, ln["reel_start"])
        if ov > best_ov:
            best, best_ov = ln["text"], ov
    return best


def _resolve_music_start(project: Project, beats: dict, cfg: dict, target: float) -> tuple[float, str]:
    """Match build_timeline's choice: EDL `music_start` wins; else auto-select."""
    edit_cfg = cfg["edit"]
    beat_times = beats.get("beat_times", [])
    beat_energy = beats.get("beat_energy", [])
    edl = common.read_json(project.edl_path) if project.edl_path.exists() else {}

    if "music_start" in edl:
        ms = float(edl["music_start"])
        idx = next((k for k, t in enumerate(beat_times) if t >= ms), 0)
        ms = beat_times[idx] if beat_times else ms
        return ms, f"manual (EDL music_start {ms:.1f}s)"

    strategy = edl.get("music_section_strategy",
                       edit_cfg.get("music_section_strategy", "energy"))
    if edit_cfg.get("auto_music_section", True) and beat_times:
        idx, win = _select_music_window(beat_times, beat_energy, target, strategy)
        return win.get("start_time", beat_times[idx]), f"auto [{strategy}]"
    return 0.0, "start of track"


def build_lyric_sheet(project: Project, cfg: dict) -> dict:
    beats = common.read_json(project.beats_path)
    if not project.lyrics_path.exists():
        common.die("No work/lyrics.json — run prep (lyrics step) first.")
    lyrics = common.read_json(project.lyrics_path)

    edl = common.read_json(project.edl_path) if project.edl_path.exists() else {}
    target = float(edl.get("target_duration", cfg["edit"]["target_duration"]))

    music_start, how = _resolve_music_start(project, beats, cfg, target)
    sheet = lyrics_in_window(lyrics, music_start, target)

    result = {
        "project": project.root.name,
        "track": lyrics.get("track"),
        "music_start": round(music_start, 3),
        "music_start_source": how,
        "target_duration": target,
        "lines": sheet,
    }
    common.write_json(project.lyric_sheet_path, result)

    log(f"Lyric sheet: music starts at {music_start:.1f}s in '{lyrics.get('track')}' "
        f"({how}); {len(sheet)} line(s) land in the {target:.0f}s reel.")
    print(f"\n  reel time   lyric  (track '{lyrics.get('track')}', start {music_start:.1f}s, {how})")
    print("  " + "-" * 64)
    if not sheet:
        print("  (no transcribed vocals fall in this window — likely an instrumental "
              "section; pick a different music_start if you want lyrics)")
    for ln in sheet:
        print(f"  {ln['reel_start']:5.1f}–{ln['reel_end']:<5.1f}  {ln['text']}")
    print()
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.lyricmap projects/<trip-name>")
    project = common.open_project(argv[0])
    cfg = common.load_config()
    build_lyric_sheet(project, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
