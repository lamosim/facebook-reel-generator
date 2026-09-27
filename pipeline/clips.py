"""Pre-render story clips: graded, 9:16-fitted, real-time footage with BAKED on-screen captions,
plus small composites (a cold-open teaser, a pair of shots sharing one caption, a shot dissolving
into a map card). The EDL then references the files (`"clip": "../work/prerender/<name>.mp4"`)
with `in: 0` and a whole-beat `fixed_duration`, so every anchor stays exact.

Usage:  python -m pipeline.clips projects/<name> [name ...]     # build all, or only the named clips

Spec: work/clips.json
{
  "out_dir": "work/prerender",            # optional
  "beat": 0.5572,                          # optional; default = median beat of work/beats.json
  "tail_pad": 0.3,                         # extra real footage after each clip's beat span (see README)
  "grade": {"sat": 0.92, "gamma": 1.0},    # optional defaults for the filmic grade
  "style": {"big_px": 124, ...},           # optional caption style overrides (see STYLE)
  "clips": [
    {"name": "day1_carry",
     "shots": [{"src": "IMG_4059.mov", "ss": 0.0, "t": "6B"}],           # t: seconds or "nB[+s]"
     "captions": [{"big": "DAY 1  ·  THE CARRY", "sub": "8 miles of trail.", "t0": 0.35}]},
    {"name": "paddle",
     "shots": [{"src": "IMG_4117.mov", "ss": 1.5, "t": "4B"}, {"src": "IMG_4118.mov", "ss": 2.4, "t": "4B"}],
     "captions": [{"sub": "paddle across.  carry over.  repeat.", "t0": 0.4}]},
    {"name": "cold_open",
     "shots": [...three quick cuts...],
     "hook": {"lines": [["WHAT 30 MILES", 0.1], ["LOOKS LIKE", 0.1], ["WHEN YOU HAVE", "4B+0.05"], ["TO CARRY THE BOAT.", "4B+0.05"]]}},
    {"name": "finish",
     "shots": [{"src": "IMG_4175.mov", "ss": 5.45, "t": "3B+0.5"}, {"src": "../work/maps/map4.mp4", "ss": 0, "t": 5.0, "grade": false}],
     "xfade": 0.5}
  ]
}
`src` resolves like an EDL `clip` (relative to raw/, so "../work/maps/map4.mp4" works).
Caption fields: big (white headline), sub (warm lowercase line), t0 (fade-in start), t1 (end; default =
clip end), fade, fade_out, sub_delay. A shot may override sat / gamma / grade(false).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import common
from .common import log

FPS, OW, OH = 30, 1080, 1920
FONT = common.caption_font()  # DIN Condensed Bold on macOS; bundled Barlow Condensed elsewhere
GRADE = ("curves=master='0/0.04 0.25/0.24 0.5/0.5 0.75/0.77 1/0.98',"
         "eq=saturation={sat}:contrast=1.03:gamma={gamma},"
         "colorbalance=rs=-0.04:bs=0.07:rm=0.01:bm=-0.01:rh=0.05:gh=0.01:bh=-0.06,"
         "vignette=a=PI/8")
FIT = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1"
STYLE = {  # caption look: white condensed headline in the top zone, warm lowercase sub line under it
    "font": FONT, "big_px": 124, "big_y": 118, "big_spacing": 8,
    "sub_px": 54, "sub_y": 262, "solo_px": 60, "solo_y": 150, "sub_spacing": 4,
    "white": [255, 255, 255], "warm": [255, 205, 160], "max_w": 980,
    "top_gradient_h": 520, "top_gradient_a": 0.62,
    "hook_px": 132, "hook_y0": 96, "hook_line_h": 136, "hook_spacing": 6, "hook_warm_from": 2,
}

_fonts: dict = {}


def font(path: str, px: int):
    key = (path, px)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, px)
    return _fonts[key]


def ease(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def parse_t(v, beat: float) -> float:
    """seconds as a number, or a beat expression: "6B", "3B+0.5", "12B-0.2"."""
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace(" ", "")
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)B)?([+-]\d+(?:\.\d+)?)?", s)
    if not m or (m.group(1) is None and m.group(2) is None):
        common.die(f"clips.json: bad time expression {v!r} (use seconds or e.g. \"6B+0.3\")")
    return float(m.group(1) or 0) * beat + float(m.group(2) or 0)


# ── frames ─────────────────────────────────────────────────────────────────
def decode(src: Path, ss: float, t: float, sat: float, gamma: float, grade: bool = True) -> list[np.ndarray]:
    """graded, 9:16-fitted frames of src[ss, ss+t) as exactly round(t*FPS) BGR arrays (short reads pad)."""
    n = int(round(t * FPS))
    vf = (GRADE.format(sat=sat, gamma=gamma) + "," if grade else "") + FIT
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-t", f"{t + 0.2:.3f}", "-i", str(src),
           "-vf", vf, "-r", str(FPS), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    fsz = OW * OH * 3
    frames = [np.frombuffer(raw[i * fsz:(i + 1) * fsz], np.uint8).reshape(OH, OW, 3) for i in range(len(raw) // fsz)]
    if not frames:
        common.die(f"no frames decoded from {src} @ {ss}s")
    while len(frames) < n:
        frames.append(frames[-1])
    return frames[:n]


# ── text ───────────────────────────────────────────────────────────────────
def _top_gradient(fr: np.ndarray, a: float, st: dict) -> np.ndarray:
    if a <= 0:
        return fr
    h = int(st["top_gradient_h"])
    g = np.zeros(OH, np.float32)
    ys = np.arange(h, dtype=np.float32)
    g[:h] = (1 - ys / h) ** 1.3
    f = fr.astype(np.float32) * (1 - st["top_gradient_a"] * a * g)[:, None, None]
    return np.clip(f, 0, 255).astype(np.uint8)


def _spaced_line(d, txt, path, px, y, col, alpha, spacing, max_w, cx=None):
    """letter-spaced centered text, auto-shrunk to max_w."""
    while True:
        f = font(path, px)
        widths = [d.textlength(ch, font=f) for ch in txt]
        total = sum(widths) + spacing * (len(txt) - 1)
        if total <= max_w or px <= 30:
            break
        px -= 2
    x = (OW - total) / 2 if cx is None else cx - total / 2
    for ch, w in zip(txt, widths):
        d.text((x + 3, y + 4), ch, font=f, fill=(0, 0, 0, int(170 * alpha)))
        d.text((x, y), ch, font=f, fill=tuple(col) + (int(255 * alpha),))
        x += w + spacing


def draw_lines(fr: np.ndarray, lines: list, st: dict) -> np.ndarray:
    """lines: (text, px, y, color, alpha, spacing). Fully transparent lines are skipped."""
    lines = [l for l in lines if l[4] > 0.002]
    if not lines:
        return fr
    pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for txt, px, y, col, a, sp in lines:
        _spaced_line(d, txt, st["font"], px, y, col, a, sp, st["max_w"])
    pil = Image.alpha_composite(pil, layer).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def caption_alpha(c: dict, t: float, end: float) -> float:
    t1 = end if c.get("t1") is None else c["t1"]
    a_in = ease((t - c.get("t0", 0.3)) / c.get("fade", 0.28))
    fo = c.get("fade_out", 0.25)
    a_out = 1.0 if fo <= 0 else ease((t1 - t) / fo)
    return min(a_in, a_out)


def apply_captions(fr: np.ndarray, t: float, end: float, caps: list, st: dict) -> np.ndarray:
    lines, grad = [], 0.0
    for c in caps:
        a = caption_alpha(c, t, end)
        if a <= 0:
            continue
        grad = max(grad, a)
        if c.get("big"):
            lines.append((c["big"], st["big_px"], st["big_y"], st["white"], a, st["big_spacing"]))
        if c.get("sub"):
            a_sub = min(a, ease((t - c.get("t0", 0.3) - c.get("sub_delay", 0.0)) / c.get("fade", 0.28)))
            px, y = (st["sub_px"], st["sub_y"]) if c.get("big") else (st["solo_px"], st["solo_y"])
            lines.append((c["sub"], px, y, st["warm"], a_sub, st["sub_spacing"]))
    if grad > 0:
        fr = _top_gradient(fr, grad, st)
    return draw_lines(fr, lines, st)


def apply_hook(fr: np.ndarray, t: float, hook: dict, st: dict, beat: float) -> np.ndarray:
    """stacked headline lines that appear on their own cue times (the cold-open teaser)."""
    lines, grad = [], 0.0
    px = hook.get("px", st["hook_px"]); y0 = hook.get("y0", st["hook_y0"]); lh = hook.get("line_h", st["hook_line_h"])
    warm_from = hook.get("warm_from", st["hook_warm_from"])
    for i, (txt, t0) in enumerate(hook["lines"]):
        a = ease((t - parse_t(t0, beat)) / hook.get("fade", 0.22))
        grad = max(grad, a)
        lines.append((txt, px, y0 + i * lh, st["white"] if i < warm_from else st["warm"], a, hook.get("spacing", st["hook_spacing"])))
    return draw_lines(_top_gradient(fr, grad, st), lines, st)


# ── build ──────────────────────────────────────────────────────────────────
def probe_duration(path: Path) -> float:
    return common.ffprobe_info(path)["duration"]


def encode(path: Path, frames) -> int:
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "14",
                          "-pix_fmt", "yuv420p", "-r", str(FPS), str(path)], stdin=subprocess.PIPE)
    n = 0
    for fr in frames:
        p.stdin.write(np.ascontiguousarray(fr).tobytes())
        n += 1
    p.stdin.close()
    p.wait()
    return n


def build_clip(project: common.Project, spec: dict, clip: dict, out_dir: Path, beat: float, st: dict) -> Path:
    g = spec.get("grade", {})
    tail = float(spec.get("tail_pad", 0.3))
    shots = clip["shots"]
    seq: list[np.ndarray] = []
    xfade = float(clip.get("xfade", 0.0))
    for j, sh in enumerate(shots):
        src = project.raw / sh["src"]
        t = parse_t(sh["t"], beat)
        grade = bool(sh.get("grade", True))
        if j == len(shots) - 1 and grade:
            t += tail        # the last shot outlasts its beat span; the pipeline trims at the actual beat
        fr = decode(src, float(sh.get("ss", 0.0)), t, float(sh.get("sat", g.get("sat", 0.92))),
                    float(sh.get("gamma", g.get("gamma", 1.0))), grade)
        if seq and xfade > 0:
            k = min(int(round(xfade * FPS)), len(seq), len(fr))
            for i in range(k):
                w = (i + 1) / (k + 1)
                seq[-k + i] = np.clip(seq[-k + i].astype(np.float32) * (1 - w) + fr[i].astype(np.float32) * w, 0, 255).astype(np.uint8)
            seq.extend(fr[k:])
        else:
            seq.extend(fr)
    end = len(seq) / FPS
    caps = [dict(c, t0=parse_t(c.get("t0", 0.3), beat), t1=(None if c.get("t1") is None else parse_t(c["t1"], beat)))
            for c in clip.get("captions", [])]
    hook = clip.get("hook")

    def frames():
        for i, fr in enumerate(seq):
            t = i / FPS
            if hook:
                fr = apply_hook(fr, t, hook, st, beat)
            yield apply_captions(fr, t, end, caps, st)

    out = out_dir / f"{clip['name']}.mp4"
    n = encode(out, frames())
    log(f"  {clip['name']:14s} {n:4d} frames  {n / FPS:5.2f}s  -> {out.name}")
    return out


def build_clips(project: common.Project, only: set[str] | None = None) -> list[Path]:
    spec_path = project.work / "clips.json"
    if not spec_path.exists():
        common.die(f"No {spec_path}. Write the clip spec first (see pipeline/clips.py docstring).")
    spec = common.read_json(spec_path)
    out_dir = project.root / spec.get("out_dir", "work/prerender")
    out_dir.mkdir(parents=True, exist_ok=True)
    beat = float(spec.get("beat", 0.0)) or median_beat(project)
    st = dict(STYLE, **spec.get("style", {}))
    log(f"=== CLIPS: {project.root.name}  (beat {beat:.4f}s, tail pad {spec.get('tail_pad', 0.3)}s) ===")
    built = []
    for clip in spec["clips"]:
        if only and clip["name"] not in only:
            continue
        built.append(build_clip(project, spec, clip, out_dir, beat, st))
    log(f"=== CLIPS complete: {len(built)} file(s) in {out_dir} ===")
    return built


def median_beat(project: common.Project) -> float:
    beats = common.read_json(project.beats_path)["beat_times"]
    return float(np.median(np.diff(beats)))


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.clips projects/<name> [clip-name ...]")
    project = common.open_project(argv[0])
    build_clips(project, set(argv[1:]) or None)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
