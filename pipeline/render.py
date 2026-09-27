"""PHASE 2 orchestrator: build_timeline → assemble → (post_audio voice mix, if the EDL has one).

Usage:  python -m pipeline.render projects/<trip-name>

Requires work/edl.json (authored by Claude) and the work/*.json artifacts
produced by `pipeline.prep`.
"""
from __future__ import annotations

import sys

from . import common
from .assemble import assemble
from .build_timeline import build_timeline
from .common import log
from . import voicemix


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.render projects/<trip-name>")
    project = common.open_project(argv[0])
    cfg = common.load_config()

    if not project.edl_path.exists():
        common.die(
            f"No EDL at {project.edl_path}. Review work/segments.json + work/frames/, "
            "then write work/edl.json (see README)."
        )
    for required in (project.beats_path, project.segments_path):
        if not required.exists():
            common.die(f"Missing {required}. Run `python -m pipeline.prep {argv[0]}` first.")

    log(f"=== RENDER: {project.root.name} ===")
    build_timeline(project, cfg)
    out = assemble(project, cfg)
    voicemix.apply(project, cfg)      # no-op unless the EDL lists post_audio lines
    log("=== RENDER complete ===")
    log(f"Upload-ready reel: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
