"""Verify a render: probe, per-clip anchors (reel time, song time, lyric), and review tiles.

Usage:  python -m pipeline.verify projects/<name>
Writes work/checks/story.jpg (one frame per clip, 8 per row), open.jpg (first 8 s) and end.jpg (last 6 s).
Look at the tiles: captions legible? cards where expected? nothing cut mid-caption?
"""
from __future__ import annotations

import sys

from . import common
from .common import log


def _tile(reel, frames, cols, path, w=180, h=320):
    sel = "+".join(f"eq(n\\,{n})" for n in frames)
    rows = -(-len(frames) // cols)
    common.run(["ffmpeg", "-y", "-i", str(reel), "-vf", f"select='{sel}',scale={w}:{h},tile={cols}x{rows}",
                "-frames:v", "1", str(path)])


def verify(project: common.Project) -> dict:
    reel = project.reel_path
    if not reel.exists() or not project.timeline_path.exists():
        common.die("Render first (output/reel.mp4 + work/timeline.json).")
    info = common.ffprobe_info(reel)
    t = common.read_json(project.timeline_path)
    ms = float(t["music_start"])
    log(f"{reel.name}: {info.get('width')}x{info.get('height')} {info['duration']:.2f}s")
    log(f"{'#':>2} {'clip':16s} {'reel':>6} {'song':>7} {'dur':>5} {'trans':9s} lyric")
    mids = []
    for c in t["clips"]:
        tr = (c.get("transition") or {}).get("type", "-")
        name = c["clip_path"].split("/")[-1][:16]
        log(f"{c['index']:2d} {name:16s} {c['reel_start']:6.2f} {ms + c['reel_start']:7.2f} {c['duration']:5.2f} {tr:9s} {str(c.get('lyric'))[:40]}")
        mids.append(int(round((c["reel_start"] + 0.55 * c["duration"]) * 30)))
    log(f"total {t['total_duration']:.3f}s  music_start {ms}  clips {len(t['clips'])}")
    checks = project.work / "checks"
    checks.mkdir(exist_ok=True)
    _tile(reel, mids, 8, checks / "story.jpg")
    end = int(t["total_duration"] * 30)
    _tile(reel, [end - 190, end - 160, end - 130, end - 100, end - 60, end - 25], 6, checks / "end.jpg", 202, 360)
    _tile(reel, [3, 40, 100, 165, 195, 230], 6, checks / "open.jpg", 202, 360)
    log(f"tiles: {checks / 'story.jpg'}, open.jpg, end.jpg")
    return t


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.verify projects/<name>")
    verify(common.open_project(argv[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
