"""PHASE 2 renderer: timeline.json + music → output/reel.mp4 (ffmpeg).

Per clip: trim → reframe to 1080x1920 → optional text → re-encode to uniform
spec params. Then concat (hard cut or xfade), overlay the music bed (with an end
fade), and final-encode to the Facebook Reels spec.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

from . import common
from .common import Project, log

# Candidate caption fonts (first existing wins); the bundled font ships in the repo, so
# there is always one.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/SFNS.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    str(common.BUNDLED_FONT),
]


def _find_font(candidates: list[str] | None = None) -> str:
    for f in (_FONT_CANDIDATES if candidates is None else candidates):
        if Path(f).is_file():
            return f
    return str(common.BUNDLED_FONT)


def _filter_path(path: str | Path) -> str:
    """A file path as an ffmpeg filter option value: forward slashes, drive colon escaped
    (`C\\:/...`), to be wrapped in single quotes by the caller."""
    return str(path).replace("\\", "/").replace(":", "\\:")


def _concat_list_text(parts) -> str:
    """Body of an ffmpeg concat-demuxer list: posix slashes (Windows-safe), quotes escaped."""
    return "".join(f"file '{p.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for p in parts)


@functools.lru_cache(maxsize=1)
def _has_drawtext() -> bool:
    """Some ffmpeg builds ship without libfreetype (no drawtext filter)."""
    try:
        out = common.run(["ffmpeg", "-hide_banner", "-filters"]).stdout
    except RuntimeError:
        return False
    return " drawtext " in out


def _reframe_filter(mode: str, w: int, h: int, sigma: int) -> str:
    if mode == "fill":
        return (f"scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},setsar=1")
    if mode == "pad":
        return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    # blur-pad (default): blurred fill behind aspect-correct foreground
    return (
        f"split=2[bg][fg];"
        f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},gblur=sigma={sigma}[bg];"
        f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def _drawtext(text: str, font: str | None, h: int) -> str:
    safe = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")
    parts = [f"text='{safe}'", "fontcolor=white", f"fontsize={max(40, h // 24)}",
             "borderw=3", "bordercolor=black@0.6",
             "x=(w-text_w)/2", f"y=h-{h // 7}"]
    if font:
        parts.insert(0, f"fontfile='{_filter_path(font)}'")
    return "drawtext=" + ":".join(parts)


def _x264_args(out_cfg: dict) -> list[str]:
    return [
        "-c:v", "libx264", "-profile:v", "high", "-preset", out_cfg["x264_preset"],
        "-pix_fmt", "yuv420p",
        "-r", str(out_cfg["fps"]),
        "-g", str(out_cfg["gop"]), "-keyint_min", str(out_cfg["gop"]),
        "-sc_threshold", "0",
        "-b:v", out_cfg["video_bitrate"], "-maxrate", out_cfg["video_maxrate"],
        "-bufsize", out_cfg["video_bufsize"],
    ]


def _aac_args(out_cfg: dict) -> list[str]:
    return [
        "-c:a", "aac", "-b:a", out_cfg["audio_bitrate"],
        "-ar", str(out_cfg["audio_sample_rate"]), "-ac", str(out_cfg["audio_channels"]),
    ]


def _build_intermediates(project: Project, timeline: dict, cfg: dict) -> list[Path]:
    """Trim + reframe + re-encode each clip to uniform spec params (with a
    guaranteed silent-or-real audio track so concat is clean)."""
    out_cfg = cfg["output"]
    img_cfg = cfg.get("image", {})
    w, h, fps = out_cfg["width"], out_cfg["height"], out_cfg["fps"]
    sigma = cfg["edit"]["blur_sigma"]
    font = _find_font()
    reframe = _reframe_filter(timeline["reframe_mode"], w, h, sigma)
    ar = out_cfg["audio_sample_rate"]

    paths: list[Path] = []
    for c in timeline["clips"]:
        idx = c["index"]
        dest = project.clips / f"clip_{idx:03d}.mp4"
        dur = c["duration"]
        is_img = c.get("is_image")

        vf = reframe
        if is_img and img_cfg.get("ken_burns", True):
            # subtle slow zoom so a still feels alive, not frozen
            total = max(2, round(dur * fps))
            zmax = float(img_cfg.get("zoom_max", 1.10))
            zinc = (zmax - 1.0) / total
            vf = (f"{reframe},zoompan=z='min(zoom+{zinc:.6f},{zmax})':d={total}:"
                  f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}:fps={fps},setsar=1")
        if c.get("text"):
            if _has_drawtext():
                vf = f"{vf},{_drawtext(c['text'], font, h)}"
            else:
                log("  ! ffmpeg has no 'drawtext' filter (build lacks libfreetype) — "
                    "skipping text overlays.")

        if is_img:
            # a photo: hold it for `dur`, Ken Burns it, add a silent audio bed
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-framerate", str(fps), "-t", f"{dur:.3f}", "-i", c["clip_path"],
                "-f", "lavfi",
                "-i", f"anullsrc=channel_layout=stereo:sample_rate={ar}",
                "-t", f"{dur:.3f}",
                "-vf", vf,
                "-map", "0:v:0", "-map", "1:a:0",
                *_x264_args(out_cfg), *_aac_args(out_cfg),
                "-shortest", str(dest),
            ]
        else:
            # Guarantee an audio track (real if the source has one, silent otherwise)
            # so the concat demuxer sees uniform streams across every clip.
            info = common.ffprobe_info(Path(c["clip_path"]))
            cmd = [
                "ffmpeg", "-y",
                "-ss", f"{c['in']:.3f}", "-i", c["clip_path"],
                "-f", "lavfi",
                "-i", f"anullsrc=channel_layout=stereo:sample_rate={ar}",
                "-t", f"{dur:.3f}",
                "-vf", vf,
                "-map", "0:v:0",
                "-map", ("0:a:0" if info["has_audio"] else "1:a:0"),
                *_x264_args(out_cfg), *_aac_args(out_cfg),
                "-shortest", str(dest),
            ]
        kind = "photo" if is_img else "clip"
        log(f"  encoding {kind} {idx} ({dur:.2f}s, {Path(c['clip_path']).name})…")
        common.run(cmd)
        paths.append(dest)
    return paths


def _concat_cut(project: Project, parts: list[Path]) -> Path:
    """Hard-cut concat via the concat demuxer (stream copy — fast, uniform)."""
    list_path = project.work / "concat_list.txt"
    list_path.write_text(_concat_list_text(parts), encoding="utf-8")
    out = project.work / "concat.mp4"
    common.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-c", "copy", str(out),
    ])
    return out


def _concat_transitions(project: Project, parts: list[Path], timeline: dict, cfg: dict) -> Path:
    """Concat with per-boundary transitions decided by build_timeline.

    Hard cuts are real cuts (zero overlap): we group consecutive cut-joined clips
    and `concat` them, then `xfade`/`acrossfade` ONLY at the genuine crossfade
    boundaries. This both honors the edit and avoids ffmpeg's xfade chain breaking
    on sub-frame "cut" overlaps (a tiny xfade silently collapses the whole chain)."""
    out_cfg = cfg["output"]
    clips = timeline["clips"]
    durations = [c["duration"] for c in clips]
    n = len(parts)

    # Group clips: a "cut" stays in the current group (hard concat); any other
    # transition starts a new group joined to the previous one by a crossfade.
    groups: list[list[int]] = [[0]]
    between: list[tuple[str, float]] = []
    for i in range(1, n):
        tr = clips[i].get("transition") or {"type": "cut", "name": "fade", "dur": 0.06}
        if tr.get("type") == "cut":
            groups[-1].append(i)
        else:
            between.append((tr.get("name", "fade"), float(tr.get("dur", 0.5))))
            groups.append([i])

    inputs: list[str] = []
    for p in parts:
        inputs += ["-i", str(p)]

    fps = out_cfg["fps"]
    ar = out_cfg["audio_sample_rate"]
    filt: list[str] = []
    # Normalize every stream to a common timebase/fps/rate first — concat and
    # xfade reject mismatched timebases (concat emits 1/1000000, raw inputs don't).
    for i in range(n):
        filt.append(f"[{i}:v]fps={fps},settb=AVTB[nv{i}]")
        filt.append(f"[{i}:a]aresample={ar},asettb=AVTB[na{i}]")

    g_labels: list[tuple[str, str]] = []
    g_durs: list[float] = []
    for gi, g in enumerate(groups):
        if len(g) == 1:
            g_labels.append((f"[nv{g[0]}]", f"[na{g[0]}]"))
        else:
            cat = "".join(f"[nv{i}][na{i}]" for i in g)
            filt.append(f"{cat}concat=n={len(g)}:v=1:a=1[cv{gi}][ca{gi}]")
            filt.append(f"[cv{gi}]settb=AVTB[gv{gi}]")
            filt.append(f"[ca{gi}]asettb=AVTB[ga{gi}]")
            g_labels.append((f"[gv{gi}]", f"[ga{gi}]"))
        g_durs.append(sum(durations[i] for i in g))

    cur_v, cur_a = g_labels[0]
    acc = g_durs[0]
    for gi in range(1, len(groups)):
        name, x = between[gi - 1]
        x = max(0.04, min(x, g_durs[gi] - 0.05, acc - 0.05))
        offset = acc - x
        nv, na = f"[xv{gi}]", f"[xa{gi}]"
        filt.append(f"{cur_v}{g_labels[gi][0]}xfade=transition={name}:duration={x:.3f}:offset={offset:.3f}{nv}")
        filt.append(f"{cur_a}{g_labels[gi][1]}acrossfade=d={x:.3f}{na}")
        cur_v, cur_a = nv, na
        acc += g_durs[gi] - x

    out = project.work / "concat.mp4"
    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filt),
        "-map", cur_v, "-map", cur_a,
        *_x264_args(out_cfg), *_aac_args(out_cfg),
        str(out),
    ]
    common.run(cmd)
    return out


def _mux_music(project: Project, concat: Path, timeline: dict, cfg: dict) -> Path:
    """Lay the music bed under the cut video and final-encode to spec."""
    out_cfg = cfg["output"]
    edit_cfg = cfg["edit"]
    music = timeline.get("music_track_path")
    music_start = float(timeline.get("music_start", 0.0))
    duration = common.ffprobe_info(concat)["duration"]
    fade = float(edit_cfg["music_fade_out"])
    fade_in = float(edit_cfg.get("music_fade_in", 0.0))
    fade_start = max(0.0, duration - fade)
    out = project.reel_path

    if not music or not Path(music).exists():
        log("  no music track — keeping original audio only.")
        common.run([
            "ffmpeg", "-y", "-i", str(concat),
            "-c:v", "copy", *_aac_args(out_cfg),
            "-movflags", "+faststart", str(out),
        ])
        return out

    music_fx = f"afade=t=out:st={fade_start:.3f}:d={fade:.3f}"
    if fade_in > 0:
        music_fx = f"afade=t=in:st=0:d={fade_in:.3f}," + music_fx
    if timeline.get("keep_original_audio"):
        duck = edit_cfg["duck_volume"]
        filt = (
            f"[0:a]volume={duck}[orig];"
            f"[1:a]{music_fx}[mus];"
            f"[orig][mus]amix=inputs=2:duration=first:dropout_transition=0[a]"
        )
        amap = "[a]"
    else:
        filt = f"[1:a]{music_fx}[a]"
        amap = "[a]"

    common.run([
        "ffmpeg", "-y",
        "-i", str(concat),
        "-ss", f"{music_start:.3f}", "-i", str(music),
        "-filter_complex", filt,
        "-map", "0:v:0", "-map", amap,
        "-c:v", "copy", *_aac_args(out_cfg),
        "-t", f"{duration:.3f}",
        "-movflags", "+faststart",
        str(out),
    ])
    return out


def assemble(project: Project, cfg: dict) -> Path:
    common.require_tool("ffmpeg")
    timeline = common.read_json(project.timeline_path)
    if not timeline.get("clips"):
        common.die("Timeline has no clips. Did build_timeline run?")
    project.ensure_dirs()
    # clear stale intermediates
    for old in project.clips.glob("clip_*.mp4"):
        old.unlink()

    clips = timeline["clips"]
    log(f"Encoding {len(clips)} clips (reframe={timeline['reframe_mode']}, "
        f"transitions={timeline.get('transition_mode', 'auto')})…")
    parts = _build_intermediates(project, timeline, cfg)

    all_cut = all((c.get("transition") or {}).get("type", "cut") == "cut" for c in clips[1:])
    if len(parts) == 1 or all_cut:
        log("Hard-cut concatenating…")
        concat = _concat_cut(project, parts)
    else:
        log("Concatenating with per-cut transitions…")
        concat = _concat_transitions(project, parts, timeline, cfg)

    log("Muxing music + final encode…")
    out = _mux_music(project, concat, timeline, cfg)

    info = common.ffprobe_info(out)

    # Integrity guard: the VIDEO stream must be ~as long as the planned timeline.
    # A short video stream means concat/xfade silently truncated it (e.g. a
    # timebase or bad-overlap failure) — fail loudly instead of shipping it.
    vdur_raw = common.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=duration", "-of", "csv=p=0", str(out),
    ]).stdout.strip()
    try:
        vdur = float(vdur_raw)
    except ValueError:
        vdur = info["duration"]
    # total_duration is pre-overlap; crossfades legitimately shorten the result,
    # so use a relative floor that still catches gross truncation (e.g. 3s vs 60s).
    expected = timeline.get("total_duration", info["duration"])
    if expected > 0 and vdur < 0.7 * expected:
        common.die(f"Rendered video stream is only {vdur:.1f}s but the timeline is "
                   f"{expected:.1f}s — assembly truncated the video. Not shipping a broken reel.")

    log(f"=== DONE → {out}")
    log(f"    {info.get('width')}x{info.get('height')} @ {info.get('fps'):.2f}fps, "
        f"video {vdur:.1f}s / container {info['duration']:.2f}s, audio={info['has_audio']}")
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.assemble projects/<trip-name>")
    project = common.open_project(argv[0])
    cfg = common.load_config()
    assemble(project, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
