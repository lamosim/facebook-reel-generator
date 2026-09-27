"""Beat-walk planner: lay an ordered shot list on the ACTUAL beat grid and show where each shot
lands in song time (with the lyric line playing), then emit the matching EDL segments with
whole-beat `fixed_duration`s and a correctly pinned `music_start`.

Usage:
  python -m pipeline.plan projects/<name>                 # print the walk from work/plan.json
  python -m pipeline.plan projects/<name> --write-edl     # also write work/edl.json (keeps other top-level fields)

work/plan.json
{
  "start_beat": 205,                       # index into work/beats.json beat_times (or "start_time": 113.36)
  "target_duration": 82.0,                 # optional (default: the walk's length)
  "defaults": {"hold": 0.7, "motion": "calm", "dir": "../work/prerender"},
  "shots": [
    {"name": "cold_open", "beats": 11, "note": "hook over 'burn, burn, burn'"},
    {"name": "map1", "beats": 9, "clip": "../work/maps/map1.mp4"},
    {"name": "surge", "beats": 8, "motion": "dynamic"}
  ]
}
A shot's `clip` defaults to "<dir>/<name>.mp4". `music_start` is pinned a hair BELOW the beat time
because the pipeline snaps to the first beat >= music_start.
"""
from __future__ import annotations

import sys

import numpy as np

from . import common
from .common import log


def lyric_at(lines: list[dict], t0: float, t1: float) -> str:
    hits = [l["text"] for l in lines if l["start"] < t1 and l["end"] > t0]
    return " / ".join(hits)[:60]


def walk(project: common.Project, write_edl: bool = False) -> dict:
    plan = common.read_json(project.work / "plan.json")
    beats = common.read_json(project.beats_path)
    bt = beats["beat_times"]
    med = float(np.median(np.diff(bt)))
    lyrics = common.read_json(project.lyrics_path)["lines"] if project.lyrics_path.exists() else []
    if "start_beat" in plan:
        start = int(plan["start_beat"])
    else:
        st = float(plan["start_time"])
        start = next((k for k, t in enumerate(bt) if t >= st - 1e-3), 0)
    d = plan.get("defaults", {})
    i, reel, total = start, 0.0, 0
    segs = []
    log(f"music_start = beat {start} @ {bt[start]:.4f}s   (median beat {med:.4f}s)")
    log(f"{'#':>2} {'shot':16s} {'beats':>5} {'song_in':>8} {'song_out':>8} {'dur':>5} {'reel_in':>7}  lyric")
    for k, s in enumerate(plan["shots"]):
        n = int(s["beats"])
        j = min(i + n, len(bt) - 1)
        t0, t1 = bt[i], bt[j]
        log(f"{k:2d} {s['name'][:16]:16s} {n:5d} {t0:8.2f} {t1:8.2f} {t1 - t0:5.2f} {reel:7.2f}  {lyric_at(lyrics, t0, t1)}")
        seg = {"clip": s.get("clip", f"{d.get('dir', '../work/prerender')}/{s['name']}.mp4"),
               "in": 0.0, "out": 60.0, "hold": s.get("hold", d.get("hold", 0.7)),
               "motion": s.get("motion", d.get("motion", "calm")),
               "fixed_duration": round(n * med, 4),
               "_beat": f"{t0:.2f} {s.get('note', '')}".strip()}
        if s.get("transition"):
            seg["transition"] = s["transition"]
        segs.append(seg)
        reel += t1 - t0
        i = j
        total += n
    log(f"total {total} beats, ends at song {bt[i]:.2f}s, reel ≈ {reel:.2f}s")
    result = {"music_start": round(bt[start] - 0.0002, 4), "target_duration": plan.get("target_duration", round(reel + 0.5, 1)),
              "segments": segs}
    if write_edl:
        edl = common.read_json(project.edl_path) if project.edl_path.exists() else {}
        edl.update({"music_start": result["music_start"], "target_duration": result["target_duration"]})
        edl.setdefault("reframe_mode", "fill")
        edl.setdefault("transition", "cut")
        edl.setdefault("pacing", "dynamic")
        edl.setdefault("keep_original_audio", False)
        edl["segments"] = segs
        common.write_json(project.edl_path, edl)
        log(f"wrote {project.edl_path} ({len(segs)} segments, music_start {result['music_start']})")
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.plan projects/<name> [--write-edl]")
    walk(common.open_project(argv[0]), write_edl="--write-edl" in argv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
