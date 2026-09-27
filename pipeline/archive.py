"""Archive the current render as the next output/reel_revNN.mp4 (never overwrites an existing rev).

Usage:  python -m pipeline.archive projects/<name> ["note"]
"""
from __future__ import annotations

import hashlib
import re
import shutil
import sys

from . import common
from .common import log


def archive(project: common.Project, note: str = "") -> str:
    reel = project.reel_path
    if not reel.exists():
        common.die(f"No {reel}; render first.")
    revs = [int(m.group(1)) for p in project.output.glob("reel_rev*.mp4") if (m := re.search(r"reel_rev(\d+)\.mp4$", p.name))]
    nxt = (max(revs) + 1) if revs else 1
    dest = project.output / f"reel_rev{nxt:02d}.mp4"
    shutil.copy2(reel, dest)
    h = hashlib.md5(dest.read_bytes()).hexdigest()
    dur = common.ffprobe_info(dest)["duration"]
    log(f"archived {reel.name} -> {dest.name}  ({dur:.2f}s, md5 {h[:8]}){'  # ' + note if note else ''}")
    log_path = project.output / "revisions.md"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"- rev{nxt:02d}: {dur:.2f}s, md5 {h[:8]}{' - ' + note if note else ''}\n")
    return dest.name


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.archive projects/<name> [note]")
    archive(common.open_project(argv[0]), " ".join(argv[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
