"""PHASE 1 orchestrator: ingest → segment → score → frames → beats.

Usage:  python -m pipeline.prep projects/<trip-name>

Produces everything Claude needs to make the creative edit decision:
  work/manifest.json   – probed metadata for every raw clip
  work/segments.json   – candidate shots with quality scores + kept flags
  work/frames/*.jpg     – representative frame per kept segment (+ _contact.jpg)
  work/beats.json      – tempo + beat grid of the music track
"""
from __future__ import annotations

import sys

from . import common
from .beats import build_beats
from .catalog import build_catalog
from .common import Project, log
from .lyrics import build_lyrics
from .segment_score import build_segments


def ingest(project: Project) -> dict:
    videos = project.raw_videos()
    if not videos:
        common.die(f"No video files found in {project.raw}")
    log(f"Probing {len(videos)} clip(s)…")
    clips = []
    for v in videos:
        try:
            info = common.ffprobe_info(v)
        except RuntimeError:
            log(f"  ! skipping unreadable/corrupt file: {v.name}")
            continue
        if info.get("duration", 0.0) <= 0.0 or not info.get("width"):
            log(f"  ! skipping file with no usable video stream: {v.name}")
            continue
        clips.append(info)
    if not clips:
        common.die("No readable video clips found.")
    track = project.music_track()
    manifest = {
        "project": project.root.name,
        "clips": clips,
        "total_raw_duration": round(sum(c["duration"] for c in clips), 2),
        "music_track": track.name if track else None,
    }
    common.write_json(project.manifest_path, manifest)
    log(f"Manifest → {project.manifest_path} ({manifest['total_raw_duration']}s of footage)")
    return manifest


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.prep projects/<trip-name>")
    project = common.open_project(argv[0])
    cfg = common.load_config()
    project.ensure_dirs()

    log(f"=== PREP: {project.root.name} ===")
    ingest(project)
    build_segments(project, cfg)
    build_catalog(project, cfg)
    build_beats(project, cfg)
    build_lyrics(project, cfg, force="--force-lyrics" in argv)

    log("=== PREP complete ===")
    log("Next: review work/catalog_raw.json + the dense work/frames/<id>_dense.jpg sheets,")
    log("      write work/catalog.json (descriptions/tags/people/best-trim-windows),")
    log(f"      map lyrics to reel time:  python -m pipeline.lyricmap {argv[0]}")
    log("      then write work/edl.json (place clips so imagery matches the lyric),")
    log(f"      then run:  python -m pipeline.render {argv[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
