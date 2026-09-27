"""Post-render audio: drop short candid lines from the raw clips ("well, we made it") under
specific shots, ducking the music around each one. Runs automatically at the end of
`pipeline.render` when the EDL has a `post_audio` list; re-run alone with
`python -m pipeline.voicemix projects/<name>` after tweaking gains (it re-mixes from the
untouched render stashed at work/reel_nomix.mp4).

EDL field:
  "post_audio": [
    {"src": "IMG_4174.mov", "ss": 7.9, "t": 1.1,        # the line, in the raw clip
     "clip": "bridge.mp4", "offset": 0.85,               # which shot (clip_path suffix, or "index") + seconds into it
     "gain_db": 4.5, "duck": 0.42}                       # optional: voice gain, music gain under the line
  ]
Find candidate lines with `python -m pipeline.candid projects/<name>`.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from . import common
from .common import log


def _find_clip(timeline: dict, spec: dict) -> dict:
    if "index" in spec:
        return timeline["clips"][int(spec["index"])]
    key = str(spec.get("clip", ""))
    for c in timeline["clips"]:
        if c["clip_path"].endswith(key):
            return c
    common.die(f"post_audio: no timeline clip matches {spec!r}")


def apply(project: common.Project, cfg: dict) -> Path | None:
    edl = common.read_json(project.edl_path)
    specs = edl.get("post_audio") or []
    if not specs:
        return None
    timeline = common.read_json(project.timeline_path)
    out_cfg = cfg["output"]
    reel = project.reel_path
    stash = project.work / "reel_nomix.mp4"
    # the stash is the untouched render; refresh it whenever the timeline is newer than it
    if not stash.exists() or stash.stat().st_mtime < project.timeline_path.stat().st_mtime:
        shutil.copy2(reel, stash)
    duration = common.ffprobe_info(stash)["duration"]

    inputs = ["-i", str(stash)]
    duck_expr = "1"
    voice_chains = []
    for i, spec in enumerate(specs, start=1):
        clip = _find_clip(timeline, spec)
        at = float(clip["reel_start"]) + float(spec.get("offset", 0.0))
        t = float(spec["t"])
        a, b = at - float(spec.get("duck_pre", 0.10)), at + t + float(spec.get("duck_post", 0.15))
        duck = float(spec.get("duck", 0.42))
        duck_expr += f"*(1-{1 - duck:.3f}*min(1,max(0,(t-{a:.3f})/0.12))*min(1,max(0,({b:.3f}-t)/0.30)))"
        src = project.raw / spec["src"]
        inputs += ["-ss", f"{float(spec['ss']):.3f}", "-t", f"{t:.3f}", "-i", str(src)]
        delay_ms = int(round(at * 1000))
        voice_chains.append(
            f"[{i}:a]afade=t=in:st=0:d={float(spec.get('fade_in', 0.05)):.2f},"
            f"afade=t=out:st={max(0.0, t - float(spec.get('fade_out', 0.15))):.2f}:d={float(spec.get('fade_out', 0.15)):.2f},"
            f"volume={float(spec.get('gain_db', 4.5)):.1f}dB,"
            f"aformat=sample_rates={out_cfg['audio_sample_rate']}:channel_layouts=stereo,"
            f"adelay={delay_ms}|{delay_ms}[v{i}]")
        log(f"  voice {spec['src']} @{spec['ss']}s -> reel {at:.2f}s (under {Path(clip['clip_path']).name}), music x{duck} for {a:.2f}-{b:.2f}s")
    n = len(specs) + 1
    filt = (f"[0:a]volume='{duck_expr}':eval=frame[m];" + ";".join(voice_chains) + ";"
            f"[m]{''.join(f'[v{i}]' for i in range(1, n))}amix=inputs={n}:duration=first:normalize=0,"
            f"alimiter=limit=0.97:attack=3:release=60[a]")
    common.run(["ffmpeg", "-y", *inputs, "-filter_complex", filt, "-map", "0:v:0", "-map", "[a]",
                "-c:v", "copy", "-c:a", "aac", "-b:a", out_cfg["audio_bitrate"],
                "-ar", str(out_cfg["audio_sample_rate"]), "-ac", str(out_cfg["audio_channels"]),
                "-t", f"{duration:.3f}", "-movflags", "+faststart", str(reel)])
    log(f"  post_audio: {len(specs)} line(s) mixed into {reel.name} (untouched render kept at {stash.name})")
    return reel


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.voicemix projects/<name>")
    project = common.open_project(argv[0])
    cfg = common.load_config()
    if apply(project, cfg) is None:
        log("EDL has no post_audio entries; nothing to do.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
