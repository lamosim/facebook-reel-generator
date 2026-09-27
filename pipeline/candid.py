"""Find quotable candid lines in the raw clips ("well, we made it!") by transcribing each clip's
audio with Whisper. Writes work/candid_lines.json and prints every line with its clip + timestamps,
so a good one can be dropped under a shot via the EDL's `post_audio` (see pipeline/voicemix.py).

Usage:  python -m pipeline.candid projects/<name> [--model small] [--lively]
Every clip with an audio track is transcribed (a few minutes on CPU); --lively restricts it to the clips
the catalog flags as `lively` (talking/laughing), which is faster but can miss a quiet line.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from . import common
from .common import log


def _lively_clips(project: common.Project) -> set[str] | None:
    raw = project.catalog_raw_path
    if not raw.exists():
        return None
    doc = common.read_json(raw)
    segs = doc.get("segments") or doc.get("shots") or []
    items = segs.values() if isinstance(segs, dict) else segs
    names = {Path(s.get("clip") or s.get("clip_path") or "").name
             for s in items if isinstance(s, dict) and (s.get("audio") or {}).get("lively")}
    names.discard("")
    return names or None


def transcribe_clips(project: common.Project, model_name: str = "small", lively_only: bool = False) -> list[dict]:
    import whisper  # local dependency, already used for lyrics

    videos = project.raw_videos()
    if lively_only:
        wanted = _lively_clips(project)
        if wanted:
            videos = [v for v in videos if v.name in wanted] or videos
    log(f"=== CANDID: transcribing {len(videos)} clip(s) with whisper-{model_name} ===")
    model = whisper.load_model(model_name)
    lines: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        for v in videos:
            wav = Path(td) / (v.stem + ".wav")
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(v), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                               capture_output=True)
            if r.returncode != 0 or not wav.exists():
                continue  # no audio track
            res = model.transcribe(str(wav), language="en", word_timestamps=True)
            for s in res.get("segments", []):
                text = s["text"].strip()
                if not text or s.get("no_speech_prob", 0) > 0.6:
                    continue
                lines.append({"clip": v.name, "start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2),
                              "text": text, "no_speech_prob": round(float(s.get("no_speech_prob", 0)), 2)})
                log(f"  {v.name}  {s['start']:5.1f}-{s['end']:5.1f}  {text}")
    out = project.work / "candid_lines.json"
    common.write_json(out, {"model": model_name, "lines": lines})
    log(f"=== CANDID complete: {len(lines)} line(s) -> {out} ===")
    return lines


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.candid projects/<name> [--model small] [--lively]")
    project = common.open_project(argv[0])
    model = argv[argv.index("--model") + 1] if "--model" in argv else "small"
    transcribe_clips(project, model, lively_only="--lively" in argv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
