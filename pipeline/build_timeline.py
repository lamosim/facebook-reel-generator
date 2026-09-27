"""Turn the Claude-authored EDL + the beat grid into an exact, beat-snapped timeline.

EDL schema (work/edl.json, written by Claude):
{
  "target_duration": 30.0,           # optional, falls back to config
  "reframe_mode": "blur-pad",        # optional override of config
  "transition": "cut",               # optional override: cut | xfade
  "cut_density": 2,                  # optional: make a cut every N beats
  "keep_original_audio": false,      # optional override
  "segments": [
    {"id": "c00_s01", "in": 12.0, "text": "Day 1"},   # `id` refs segments.json
    {"clip": "DSC_0012.mp4", "in": 3.0, "out": 9.0},  # or explicit clip+in/out
    ...
  ]
}

Output (work/timeline.json): per-slot {clip_path, in, out, duration, text, ...}
with cut points snapped to the music's beat grid.
"""
from __future__ import annotations

import sys

from . import common
from .common import Project, log


def _segment_index(segments_doc: dict) -> dict:
    return {s["id"]: s for s in segments_doc.get("segments", [])}


def _resolve_entry(entry: dict, seg_idx: dict, project: Project, cfg: dict) -> dict:
    """Resolve an EDL entry to concrete clip_path / in / out / available footage,
    plus content attributes used to time the clip and pick its transition:
      hold   0..1  how much the content wants to linger (payoff vista=high,
                   quick detail=low). EDL `hold`; default 0.5.
      motion calm|dynamic  visual energy. EDL `motion`; else auto from segment
                   shake (handheld/action → dynamic).
      transition  optional per-shot override of the transition INTO this clip.
    """
    seg_shake = None
    is_img = False
    if "id" in entry:
        seg = seg_idx.get(entry["id"])
        if seg is None:
            common.die(f"EDL references unknown segment id '{entry['id']}'")
        clip_path = seg["clip_path"]
        seg_start, seg_end = seg["start"], seg["end"]
        seg_shake = seg.get("shake")
        is_img = bool(seg.get("is_image"))
    else:
        clip = entry.get("clip")
        if not clip:
            common.die(f"EDL entry needs either 'id' or 'clip': {entry}")
        clip_path = str(project.raw / clip)
        is_img = common.is_image(project.raw / clip)
        if is_img:
            seg_start, seg_end = 0.0, 0.0
        else:
            info = common.ffprobe_info(project.raw / clip)
            seg_start, seg_end = 0.0, info["duration"]

    if is_img:
        # a still can be shown for any length — no seek, unbounded "footage"
        in_pt, out_pt, available = 0.0, 0.0, 1.0e6
    else:
        in_pt = float(entry.get("in", seg_start))
        out_pt = float(entry.get("out", seg_end))
        in_pt = max(seg_start, min(in_pt, seg_end))
        out_pt = max(in_pt, min(out_pt, seg_end))
        available = out_pt - in_pt

    motion = entry.get("motion")
    if motion not in ("calm", "dynamic"):
        thr = float(cfg["edit"].get("motion_shake_threshold", 11.0))
        motion = "dynamic" if (seg_shake is not None and seg_shake >= thr) else "calm"

    return {
        "clip_path": clip_path,
        "is_image": is_img,
        "in": in_pt,
        "max_out": out_pt,
        "available": available,
        "text": entry.get("text"),
        "hold": max(0.0, min(1.0, float(entry.get("hold", 0.5)))),
        "motion": motion,
        "transition": entry.get("transition"),
        "fixed_duration": (float(entry["fixed_duration"])
                           if entry.get("fixed_duration") is not None else None),
    }


def _uniform_slots(beat_times: list[float], cut_density: int, target: float) -> list[float]:
    """Pick every `cut_density`-th beat as a cut boundary; return slot lengths
    that together cover ~`target` seconds (all roughly equal)."""
    if len(beat_times) < 2:
        n = max(1, int(round(target / 2.5)))
        return [target / n] * n

    boundaries = beat_times[::max(1, cut_density)]
    durations: list[float] = []
    acc = 0.0
    for i in range(1, len(boundaries)):
        d = boundaries[i] - boundaries[i - 1]
        if d <= 0.05:
            continue
        durations.append(d)
        acc += d
        if acc >= target:
            break
    return durations or [target]


def _smooth(values: list[float], window: int) -> list[float]:
    """Centered moving average so pacing follows musical sections, not single beats."""
    if window <= 1 or not values:
        return list(values)
    out = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def _dynamic_slots(
    beat_times: list[float], beat_energy: list[float], target: float,
    bmin: int, bmax: int, smooth: int,
) -> list[float]:
    """Walk the beat grid, holding each clip for a beat-count that scales with
    local music energy: loud/energetic passages cut fast (≈bmin beats), quiet
    passages hold long (≈bmax beats). Every cut lands on a beat → varied,
    music-synced clip durations. Falls back to uniform if energy is unavailable."""
    n = len(beat_times)
    if n < bmax + 2 or len(beat_energy) != n:
        return _uniform_slots(beat_times, max(2, (bmin + bmax) // 2), target)

    energy = _smooth(beat_energy, smooth)
    # Rescale energy to the reel's own time window so the full pacing range is
    # used (the calmest moment in-reel → longest hold, loudest → fastest cut),
    # instead of being compressed by louder sections later in the full track.
    win = [energy[k] for k, t in enumerate(beat_times) if t <= target * 1.1]
    if win:
        wlo, whi = min(win), max(win)
        if whi > wlo:
            energy = [(e - wlo) / (whi - wlo) for e in energy]

    slots: list[float] = []
    acc = 0.0
    i = 0
    while acc < target and i < n - 1:
        e = max(0.0, min(1.0, energy[i]))
        # high energy → fewer beats (faster cut); low energy → more beats (longer hold)
        beats = round(bmax - e * (bmax - bmin))
        beats = max(bmin, min(bmax, beats))
        j = min(i + beats, n - 1)
        if j <= i:
            break
        dur = beat_times[j] - beat_times[i]
        if dur < 0.4:  # guard against degenerate spacing
            i = j
            continue
        slots.append(dur)
        acc += dur
        i = j
    # Stop at whichever beat boundary is closest to target instead of always
    # overshooting by a partial slot (keeps final duration predictable).
    if len(slots) > 1 and (acc - target) > (target - (acc - slots[-1])):
        slots.pop()
    return slots or _uniform_slots(beat_times, bmax, target)


def _window_score(e: list[float], strategy: str = "energy") -> dict:
    """Score a candidate music window for captivation.

    strategy="energy" (default): favor a section that is loud from the very
        first beat and stays energetic — leads with the hook/drop for immediate
        attention.
    strategy="build": favor a section that *builds* (ends louder than it starts)
        and peaks late — for a montage that escalates to a synchronized climax.
    """
    m = len(e)
    if m < 4:
        return {"score": 0.0}
    q = max(1, m // 4)
    mean_e = sum(e) / m
    peak_e = max(e)
    start_e = sum(e[:q]) / q                              # energy at the window's start
    rise = (sum(e[-q:]) / q) - start_e                    # -1..1, want positive for "build"
    rise01 = (max(-1.0, min(1.0, rise)) + 1.0) / 2.0
    amax = e.index(peak_e) / (m - 1)                      # 0..1 position of peak
    late_peak = max(0.0, 1.0 - abs(amax - 0.85) / 0.85)   # best when peak ~85% in

    if strategy == "build":
        score = 0.25 * mean_e + 0.20 * peak_e + 0.30 * rise01 + 0.25 * late_peak
    else:  # "energy" — immediate, sustained energy
        score = 0.40 * mean_e + 0.35 * start_e + 0.25 * peak_e
    return {
        "score": round(score, 4), "mean_energy": round(mean_e, 3),
        "peak_energy": round(peak_e, 3), "start_energy": round(start_e, 3),
        "build": round(rise, 3), "peak_at_frac": round(amax, 3),
    }


def _select_music_window(
    beat_times: list[float], beat_energy: list[float], target: float,
    strategy: str = "energy",
) -> tuple[int, dict]:
    """Slide a `target`-second window across the track and return the start beat
    index of the most captivating section (+ its score breakdown)."""
    n = len(beat_times)
    if n < 6 or len(beat_energy) != n:
        return 0, {"score": 0.0, "reason": "no energy data — starting at 0:00"}

    best_i, best = 0, {"score": -1.0}
    for i in range(n):
        t0 = beat_times[i]
        j = i
        while j + 1 < n and (beat_times[j + 1] - t0) < target:
            j += 1
        # not enough song left to fill the window → no later start can either
        if (beat_times[j] - t0) < target * 0.7:
            break
        info = _window_score(beat_energy[i:j + 1], strategy)
        if info["score"] > best["score"]:
            best_i, best = i, info
    best["start_time"] = round(beat_times[best_i], 2)
    best["strategy"] = strategy
    return best_i, best


def _window_energy(
    sub_times: list[float], sub_energy: list[float], target: float, smooth: int
) -> list[float]:
    """Smoothed, window-renormalized per-beat energy (0..1) for the reel's span."""
    if not sub_energy or len(sub_energy) != len(sub_times):
        return []
    e = _smooth(sub_energy, smooth)
    win = [e[k] for k, t in enumerate(sub_times) if t <= target * 1.1]
    if win:
        lo, hi = min(win), max(win)
        if hi > lo:
            e = [(x - lo) / (hi - lo) for x in e]
    return [max(0.0, min(1.0, x)) for x in e]


def _energy_at_time(t: float, sub_times: list[float], energy: list[float]) -> float:
    """Music energy (0..1) at timeline time `t` (nearest beat)."""
    if not energy:
        return 0.5
    idx = min(range(len(sub_times)), key=lambda k: abs(sub_times[k] - t))
    return energy[min(idx, len(energy) - 1)]


def _content_music_slots(
    entries: list, sub_times: list[float], sub_energy: list[float], target: float,
    bmin: int, bmax: int, smooth: int, music_w: float, content_w: float,
) -> list[float]:
    """Give each shot (1:1 with the ordered EDL) a beat-snapped duration set by
    BOTH the music energy where it lands AND its content `hold`. Calm music and a
    high-hold payoff shot stretch the clip; energetic music and a low-hold/quick
    shot shorten it. Durations are allocated proportionally so the total ≈ target."""
    m = len(entries)
    n = len(sub_times)
    if m == 0:
        return []
    if n < 3:
        return [target / m] * m

    energy = _window_energy(sub_times, sub_energy, target, smooth)

    # Relative duration weight per shot (estimate position uniformly to sample energy).
    weights: list[float] = []
    for i, e in enumerate(entries):
        em = _energy_at_time(((i + 0.5) / m) * target, sub_times, energy) if energy else 0.5
        music_factor = 1.0 - music_w * (em - 0.5) * 2.0       # calm → >1, loud → <1
        content_factor = 1.0 + content_w * (e["hold"] - 0.5) * 2.0  # hi-hold → >1
        weights.append(max(0.25, music_factor * content_factor))

    # Median beat interval, to convert any pinned `fixed_duration` (seconds) → beats.
    diffs = sorted(sub_times[k + 1] - sub_times[k] for k in range(n - 1))
    med = diffs[len(diffs) // 2] if diffs else (target / max(1, n - 1))

    total_beats = next((k for k, t in enumerate(sub_times) if t >= target), n - 1)

    # Pinned shots (e.g. a cold-open hook) take a fixed beat-count that bypasses
    # the min/max range; the rest share the remaining beats by content×music weight.
    fixed = {i: max(1, round(e["fixed_duration"] / med))
             for i, e in enumerate(entries) if e.get("fixed_duration")}
    non_fixed = [i for i in range(m) if i not in fixed]
    total_beats = max(total_beats, sum(fixed.values()) + len(non_fixed) * bmin)
    target_nf = max(len(non_fixed) * bmin, total_beats - sum(fixed.values()))
    sw = sum(weights[i] for i in non_fixed) or 1.0

    beats = [0] * m
    for i in fixed:
        beats[i] = fixed[i]
    for i in non_fixed:
        beats[i] = max(bmin, min(bmax, round(target_nf * weights[i] / sw)))

    # Clamping to [bmin, bmax] skews the sum (badly when energy swings hard, as in
    # a "build" section). Balance it back so the non-fixed beats sum to target_nf,
    # adding/removing single beats from shots that still have headroom → the total
    # duration tracks `target` regardless of how lopsided the energy is.
    def _nf_sum() -> int:
        return sum(beats[i] for i in non_fixed)

    guard = 0
    while _nf_sum() > target_nf and guard < 100000:
        cands = [i for i in non_fixed if beats[i] > bmin]
        if not cands:
            break
        beats[max(cands, key=lambda i: beats[i])] -= 1
        guard += 1
    while _nf_sum() < target_nf and guard < 100000:
        cands = [i for i in non_fixed if beats[i] < bmax]
        if not cands:
            break
        beats[min(cands, key=lambda i: beats[i])] += 1
        guard += 1

    # Walk the beat grid, snapping each clip to its beat-count.
    durs: list[float] = []
    ib = 0
    for b in beats:
        j = ib + b
        if j >= n:
            j = n - 1
        if j <= ib:
            j = min(ib + 1, n - 1)
        durs.append(sub_times[j] - sub_times[ib])
        ib = j
        if ib >= n - 1:
            # ran out of song; give any remaining shots a short tail beat
            break
    while len(durs) < m:
        durs.append(durs[-1] if durs else target / m)
    return durs


# Map our transition types → ffmpeg xfade transition names.
# NOTE: the clean, smooth crossfade is ffmpeg's "fade" (linear alpha blend).
# ffmpeg's "dissolve" is a noisy pixel-swap that looks cheap — only used if
# explicitly requested as a per-shot override.
_XFADE_NAME = {
    "cut": "fade",            # ~2-frame linear blend, reads as a hard cut
    "crossfade": "fade",      # smooth linear crossfade (default soft transition)
    "fade": "fade",
    "fadeblack": "fadeblack", "fadewhite": "fadewhite",
    "dissolve": "dissolve",   # noisy pixel dissolve — available but off by default
    "wipeleft": "wipeleft", "wiperight": "wiperight",
    "slideup": "slideup", "slidedown": "slidedown", "smoothleft": "smoothleft",
}


def _decide_transitions(
    entries: list, durs: list[float], sub_times: list[float], sub_energy: list[float],
    target: float, smooth: int, mode: str, cfg: dict,
) -> list:
    """Per-cut transition (the one INTO each clip), synced to music + content:
      • hard CUT on energetic moments or when either shot is dynamic (punchy)
      • soft DISSOLVE between calm/scenic shots
      • per-shot `transition` override always wins (e.g. 'fadeblack' at a break)
    Returns a list aligned to `entries`; index 0 is None (no transition-in)."""
    ec = cfg["edit"]
    cut_e = float(ec.get("transition_cut_energy", 0.55))
    soft = float(ec.get("crossfade_duration", 0.5))
    durmap = {
        "cut": float(ec.get("cut_duration", 0.06)),
        "crossfade": soft, "fade": soft, "dissolve": soft,
        "fadeblack": float(ec.get("fadeblack_duration", 0.6)),
    }
    energy = _window_energy(sub_times, sub_energy, target, smooth)

    out: list = [None]
    t_cursor = durs[0] if durs else 0.0
    for k in range(1, len(entries)):
        override = entries[k].get("transition")
        if override:
            ttype = override
        elif mode == "cut":
            ttype = "cut"
        elif mode in ("xfade", "crossfade", "dissolve"):
            ttype = "crossfade"
        else:  # auto — combine music energy + content into a "cut pressure"
            em = _energy_at_time(t_cursor, sub_times, energy) if energy else 0.5
            dynamic = entries[k]["motion"] == "dynamic" or entries[k - 1]["motion"] == "dynamic"
            avg_hold = (entries[k]["hold"] + entries[k - 1]["hold"]) / 2.0
            # loud/dynamic → harder cut; calm high-hold scenic pair → smooth crossfade
            pressure = em + (0.4 if dynamic else 0.0) - 0.35 * (avg_hold - 0.5) * 2.0
            ttype = "cut" if pressure >= cut_e else "crossfade"

        dur = durmap.get(ttype, durmap["dissolve"])
        # never let a transition eat more than 40% of either adjacent clip
        dur = max(0.04, min(dur, 0.4 * min(durs[k - 1], durs[k])))
        out.append({"type": ttype, "name": _XFADE_NAME.get(ttype, "fade"), "dur": round(dur, 3)})
        t_cursor += durs[k]
    return out


def build_timeline(project: Project, cfg: dict) -> dict:
    edl = common.read_json(project.edl_path)
    beats = common.read_json(project.beats_path)
    segments_doc = common.read_json(project.segments_path)
    seg_idx = _segment_index(segments_doc)

    edit_cfg = cfg["edit"]
    target = float(edl.get("target_duration", edit_cfg["target_duration"]))
    pacing = edl.get("pacing", edit_cfg.get("pacing", "dynamic"))
    cut_density = int(edl.get("cut_density", edit_cfg["cut_density"]))
    reframe_mode = edl.get("reframe_mode", edit_cfg["reframe_mode"])
    transition_mode = edl.get("transition", edit_cfg["transition"])
    keep_original = bool(edl.get("keep_original_audio", edit_cfg["keep_original_audio"]))

    entries = [_resolve_entry(e, seg_idx, project, cfg) for e in edl.get("segments", [])]
    if not entries:
        common.die("EDL has no segments.")

    beat_times = beats.get("beat_times", [])
    beat_energy = beats.get("beat_energy", [])

    # Choose where in the track the music overlay starts. Auto-pick the most
    # captivating window unless disabled or overridden by the EDL.
    music_start = 0.0
    start_idx = 0
    if "music_start" in edl:
        music_start = float(edl["music_start"])
        start_idx = next((k for k, t in enumerate(beat_times) if t >= music_start), 0)
        music_start = beat_times[start_idx] if beat_times else music_start
        log(f"Music section: manual start at {music_start:.1f}s (snapped to beat).")
    elif edit_cfg.get("auto_music_section", True) and beat_times:
        strategy = edl.get("music_section_strategy",
                           edit_cfg.get("music_section_strategy", "energy"))
        start_idx, win = _select_music_window(beat_times, beat_energy, target, strategy)
        music_start = win.get("start_time", beat_times[start_idx])
        log(f"Music section [{strategy}]: auto-selected window → start "
            f"{music_start:.1f}s (mean energy {win.get('mean_energy','?')}, "
            f"start energy {win.get('start_energy','?')}, "
            f"peak {win.get('peak_energy','?')} at {int(100*win.get('peak_at_frac',0))}% in, "
            f"score {win.get('score','?')}).")

    # Re-base the beat grid to the chosen start so cut spacing tracks that
    # section (durations are differences, so absolute offset doesn't matter).
    sub_times = [t - music_start for t in beat_times[start_idx:]]
    sub_energy = beat_energy[start_idx:] if len(beat_energy) == len(beat_times) else []

    smooth = int(edit_cfg["energy_smooth_beats"])
    bmin, bmax = int(edit_cfg["beats_per_cut_min"]), int(edit_cfg["beats_per_cut_max"])

    # One beat-snapped duration per shot (1:1 with the ordered EDL).
    if pacing == "dynamic":
        slots = _content_music_slots(
            entries, sub_times, sub_energy, target, bmin, bmax, smooth,
            float(edit_cfg["duration_music_weight"]), float(edit_cfg["duration_content_weight"]),
        )
        log(f"Pacing=dynamic → {len(slots)} shots timed by content × music "
            f"(target {target}s). Clip lengths vary {min(slots):.1f}–{max(slots):.1f}s.")
    else:
        per = _uniform_slots(sub_times, cut_density, target)
        avg = sum(per) / len(per) if per else target / max(1, len(entries))
        slots = [avg] * len(entries)
        log(f"Pacing=uniform → {len(entries)} shots of ~{avg:.1f}s (every {cut_density} beat(s)).")

    # Clamp each slot to the shot's available footage.
    eff_durs: list[float] = []
    for i, (slot, e) in enumerate(zip(slots, entries)):
        dur = slot
        if e["available"] < slot:
            dur = e["available"]
            log(f"  ! shot {i} ({e['clip_path'].split('/')[-1]}): only {dur:.2f}s available "
                f"(< {slot:.2f}s) — slightly shorter than planned.")
        eff_durs.append(max(0.2, dur))

    transitions = _decide_transitions(
        entries, eff_durs, sub_times, sub_energy, target, smooth, transition_mode, cfg,
    )

    # Lyric overlay map (reel time) so each clip can be annotated with the line
    # playing during it — the emotional-sync contract, verifiable in timeline.json.
    lyric_sheet: list[dict] = []
    if project.lyrics_path.exists():
        from .lyricmap import lyrics_in_window, lyric_at
        lyrics_doc = common.read_json(project.lyrics_path)
        lyric_sheet = lyrics_in_window(lyrics_doc, music_start, target)
    else:
        lyric_at = lambda *a, **k: None  # noqa: E731

    # Walk the ACTUAL playback timeline. Crossfades overlap the previous clip by
    # their duration (the assembler xfades them); hard cuts are concatenated with
    # no overlap. Tracking this is what makes reel_start — and therefore the lyric
    # assigned to each clip — line up with what's really heard/seen, instead of
    # drifting later and later by the accumulated crossfade time.
    timeline: list[dict] = []
    play_end = 0.0
    for i, (e, dur, tr) in enumerate(zip(entries, eff_durs, transitions)):
        if i == 0:
            t0 = 0.0
        elif tr and tr.get("type") != "cut":
            t0 = max(0.0, play_end - float(tr.get("dur", 0.0)))
        else:
            t0 = play_end
        t1 = t0 + dur
        timeline.append({
            "index": i,
            "clip_path": e["clip_path"],
            "is_image": e.get("is_image", False),
            "in": round(e["in"], 3),
            "out": round(e["in"] + dur, 3),
            "duration": round(dur, 3),
            "reel_start": round(t0, 3),
            "hold": e["hold"],
            "motion": e["motion"],
            "transition": tr,        # how to transition INTO this clip (None for clip 0)
            "lyric": lyric_at(lyric_sheet, t0, t1) if lyric_sheet else None,
            "text": e["text"],
        })
        play_end = t1
    total = play_end

    # Report the transition mix so the edit is legible.
    if transition_mode == "auto":
        from collections import Counter
        mix = Counter(c["transition"]["type"] for c in timeline if c["transition"])
        log(f"Transitions [auto]: " + ", ".join(f"{n}×{t}" for t, n in mix.items()))

    if lyric_sheet:
        n_synced = sum(1 for c in timeline if c["lyric"])
        log(f"Lyric sync: {n_synced}/{len(timeline)} clips land on a lyric line "
            f"({len(lyric_sheet)} lines in window). Per-clip lyric in timeline.json.")

    result = {
        "project": project.root.name,
        "reframe_mode": reframe_mode,
        "transition_mode": transition_mode,
        "keep_original_audio": keep_original,
        "music_track_path": beats.get("track_path"),
        "music_start": round(music_start, 3),
        "total_duration": round(total, 3),
        "lyric_sheet": lyric_sheet,
        "clips": timeline,
    }
    common.write_json(project.timeline_path, result)
    log(f"Timeline: {len(timeline)} clips, {result['total_duration']}s → {project.timeline_path}")
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.build_timeline projects/<trip-name>")
    project = common.open_project(argv[0])
    cfg = common.load_config()
    build_timeline(project, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
