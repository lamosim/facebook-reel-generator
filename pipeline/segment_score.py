"""Segment raw clips into shots (PySceneDetect) and score each (OpenCV).

Writes work/segments.json and a representative frame per kept segment to
work/frames/, plus a contact-sheet montage for quick visual review.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from . import common
from .common import Project, log


# ── segmentation ─────────────────────────────────────────────────────────────
def detect_shots(video: Path, cfg: dict) -> list[tuple[float, float]]:
    """Return list of (start_sec, end_sec) candidate shots for one clip."""
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import AdaptiveDetector, ContentDetector

    info = common.ffprobe_info(video)
    duration = info.get("duration", 0.0)
    seg_cfg = cfg["segment"]

    # Short clips: treat the whole thing as a single shot (no detection cost).
    if duration <= seg_cfg["short_clip_threshold"]:
        return [(0.0, duration)]

    sd_video = open_video(str(video))
    fps = sd_video.frame_rate or info.get("fps", 30.0) or 30.0
    min_scene_frames = max(1, int(seg_cfg["min_scene_len"] * fps))

    if seg_cfg["detector"] == "content":
        detector = ContentDetector(min_scene_len=min_scene_frames)
    else:
        detector = AdaptiveDetector(
            adaptive_threshold=seg_cfg["adaptive_threshold"],
            min_scene_len=min_scene_frames,
        )

    sm = SceneManager()
    sm.add_detector(detector)
    sm.detect_scenes(sd_video)
    scenes = sm.get_scene_list()

    if not scenes:
        return [(0.0, duration)]
    return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]


# ── per-frame quality metrics ────────────────────────────────────────────────
def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _luma(frame_bgr: np.ndarray) -> float:
    v = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    return float(v.mean())


def _flow_magnitude(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(mag.mean())


def _frame_interest(gray: np.ndarray, frame_bgr: np.ndarray, cfg: dict) -> float:
    """Per-frame 'how compelling is this moment' in ~0..1: sharp + well-exposed +
    visually detailed (contrast). Used to find where a shot's PAYOFF lands."""
    sc = cfg["score"]
    sharp_n = min(1.0, _sharpness(gray) / (sc["min_sharpness"] * 4.0))
    luma = _luma(frame_bgr)
    target = (sc["min_luma"] + sc["max_luma"]) / 2.0
    span = (sc["max_luma"] - sc["min_luma"]) / 2.0
    expo_n = max(0.0, 1.0 - abs(luma - target) / span) if span else 0.0
    detail_n = min(1.0, float(gray.std()) / 64.0)  # contrast/composition richness
    return 0.5 * sharp_n + 0.3 * expo_n + 0.2 * detail_n


def _payoff_metrics(interest: list[float], flow_vals: list[float]) -> dict:
    """Derive a shot's narrative shape from its per-sample interest + motion curves:
      peak_pos   0..1 where the most compelling frame lands (late = builds to a reveal)
      settle     -1..1 motion change first→last third (+ = camera settles into a vista)
      motion_arc settling | rising | steady
      payoff_score 0..1 — does this shot build to a satisfying, holdable moment?
    These guide trimming so each clip starts/ends on a meaningful beat, not mid-motion.
    """
    if not interest:
        return {"payoff_score": 0.0, "peak_pos": 0.0, "settle": 0.0,
                "motion_arc": "steady", "max_interest": 0.0, "interest_curve": []}

    n = len(interest)
    peak_i = max(range(n), key=lambda k: interest[k])
    peak_pos = peak_i / (n - 1) if n > 1 else 0.0
    max_interest = interest[peak_i]

    settle = 0.0
    if len(flow_vals) >= 2:
        h = max(1, len(flow_vals) // 3)
        first = sum(flow_vals[:h]) / h
        last = sum(flow_vals[-h:]) / h
        denom = first + last + 1e-6
        settle = max(-1.0, min(1.0, (first - last) / denom))
    arc = "settling" if settle > 0.2 else "rising" if settle < -0.2 else "steady"

    # Reward a strong peak that lands with room to breathe (~70% in) and a camera
    # that settles rather than still moving when we'd cut.
    late_peak = max(0.0, 1.0 - abs(peak_pos - 0.7) / 0.7)
    payoff = 0.45 * max_interest + 0.30 * max(0.0, settle) + 0.25 * late_peak

    # coarse curve (≤6 pts) for at-a-glance reference in segments.json
    step = max(1, n // 6)
    curve = [round(interest[k], 2) for k in range(0, n, step)]
    return {
        "payoff_score": round(payoff, 3),
        "peak_pos": round(peak_pos, 3),
        "settle": round(settle, 3),
        "motion_arc": arc,
        "max_interest": round(max_interest, 3),
        "interest_curve": curve,
    }


def score_segment(
    video: Path, start: float, end: float, cfg: dict
) -> dict:
    """Sample frames across [start, end] and compute composite quality metrics
    plus a content-PAYOFF shape (where the shot peaks / whether it settles)."""
    score_cfg = cfg["score"]
    interval = score_cfg["sample_interval"]

    cap = cv2.VideoCapture(str(video))
    duration = max(0.01, end - start)
    n_samples = max(2, int(duration / interval) + 1)
    sample_times = [start + (i / (n_samples - 1)) * duration for i in range(n_samples)]
    # downscale for fast optical flow
    flow_w = 320

    sharp_vals: list[float] = []
    luma_vals: list[float] = []
    flow_vals: list[float] = []
    interest: list[float] = []
    interest_times: list[float] = []
    best_sharp = -1.0
    best_frame: np.ndarray | None = None
    best_time = start
    prev_small: np.ndarray | None = None

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        s = _sharpness(gray)
        sharp_vals.append(s)
        luma_vals.append(_luma(frame))
        interest.append(_frame_interest(gray, frame, cfg))
        interest_times.append(t)
        if s > best_sharp:
            best_sharp, best_frame, best_time = s, frame, t

        h, w = gray.shape
        small = cv2.resize(gray, (flow_w, max(1, int(h * flow_w / w))))
        if prev_small is not None and prev_small.shape == small.shape:
            flow_vals.append(_flow_magnitude(prev_small, small))
        prev_small = small

    cap.release()

    if not sharp_vals:
        return {
            "ok": False, "sharpness": 0.0, "luma": 0.0, "shake": 999.0,
            "best_frame": None, "best_time": start, "payoff": _payoff_metrics([], []),
            "peak_time": start,
        }

    payoff = _payoff_metrics(interest, flow_vals)
    # absolute time of the most compelling moment (where the payoff lands)
    peak_idx = max(range(len(interest)), key=lambda k: interest[k])
    peak_time = interest_times[peak_idx]

    return {
        "ok": True,
        "sharpness": float(np.median(sharp_vals)),
        "luma": float(np.median(luma_vals)),
        "shake": float(np.median(flow_vals)) if flow_vals else 0.0,
        "best_frame": best_frame,
        "best_time": best_time,
        "payoff": payoff,
        "peak_time": peak_time,
    }


def score_image(path: Path, cfg: dict) -> dict:
    """Score a still photo as a single 'frame'. No motion → shake 0, steady arc.
    A strong, sharp, well-exposed photo earns a high payoff (it IS its own peak)."""
    img = cv2.imread(str(path))
    if img is None:
        return {"ok": False, "sharpness": 0.0, "luma": 0.0, "shake": 0.0,
                "best_frame": None, "best_time": 0.0,
                "payoff": _payoff_metrics([], []), "peak_time": 0.0}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    interest = _frame_interest(gray, img, cfg)
    return {
        "ok": True,
        "sharpness": _sharpness(gray),
        "luma": _luma(img),
        "shake": 0.0,
        "best_frame": img,
        "best_time": 0.0,
        "peak_time": 0.0,
        "payoff": {"payoff_score": round(interest, 3), "peak_pos": 0.0, "settle": 0.0,
                   "motion_arc": "still", "max_interest": round(interest, 3),
                   "interest_curve": [round(interest, 2)]},
    }


def _composite_keep_score(m: dict, cfg: dict) -> float:
    """Normalize metrics to ~0..1 and combine by configured weights."""
    w = cfg["score"]["weights"]
    sc = cfg["score"]
    # sharpness: saturating curve relative to floor
    sharp_n = min(1.0, m["sharpness"] / (sc["min_sharpness"] * 4.0))
    # exposure: 1.0 in the comfortable mid-range, falling off at extremes
    target = (sc["min_luma"] + sc["max_luma"]) / 2.0
    span = (sc["max_luma"] - sc["min_luma"]) / 2.0
    expo_n = max(0.0, 1.0 - abs(m["luma"] - target) / span) if span else 0.0
    # stability: less flow = better, normalized against the shake ceiling
    stab_n = max(0.0, 1.0 - m["shake"] / sc["max_shake"]) if sc["max_shake"] else 0.0
    total_w = w["sharpness"] + w["exposure"] + w["stability"]
    return (
        w["sharpness"] * sharp_n
        + w["exposure"] * expo_n
        + w["stability"] * stab_n
    ) / total_w


def _passes_hard_thresholds(m: dict, cfg: dict) -> tuple[bool, str]:
    sc = cfg["score"]
    if m["sharpness"] < sc["min_sharpness"]:
        return False, "blurry"
    if m["luma"] < sc["min_luma"]:
        return False, "too_dark"
    if m["luma"] > sc["max_luma"]:
        return False, "blown_out"
    if m["shake"] > sc["max_shake"]:
        return False, "shaky"
    return True, ""


# ── frame extraction ─────────────────────────────────────────────────────────
def _save_best_frame(frame: np.ndarray, dest: Path, max_w: int = 640) -> None:
    h, w = frame.shape[:2]
    if w > max_w:
        frame = cv2.resize(frame, (max_w, int(h * max_w / w)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _save_filmstrip(
    video: Path, start: float, end: float, dest: Path, peak_time: float,
    n: int = 5, cell_w: int = 300, cell_h: int = 300,
) -> bool:
    """Lay N evenly-spaced frames across [start, end] in one row, each labeled with
    its IN-CLIP offset (seconds). Lets Claude see how a shot evolves and pick clean
    in/out trim points (so clips don't start/end mid-motion). The frame nearest the
    content peak is marked ★."""
    duration = max(0.01, end - start)
    times = [start + (i / (n - 1)) * duration for i in range(n)] if n > 1 else [start]
    cap = cv2.VideoCapture(str(video))
    label_h = 22
    pad = 4
    strip = np.full((cell_h + label_h, n * (cell_w + pad) + pad, 3), 20, dtype=np.uint8)
    got = 0
    peak_cell = min(range(len(times)), key=lambda k: abs(times[k] - peak_time))
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        got += 1
        h, w = frame.shape[:2]
        scale = min(cell_w / w, cell_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame, (nw, nh))
        cell = np.full((cell_h, cell_w, 3), 35, dtype=np.uint8)
        oy, ox = (cell_h - nh) // 2, (cell_w - nw) // 2
        cell[oy:oy + nh, ox:ox + nw] = resized
        x = pad + i * (cell_w + pad)
        strip[label_h:label_h + cell_h, x:x + cell_w] = cell
        tag = f"+{t - start:0.1f}s" + (" ★" if i == peak_cell else "")
        cv2.putText(strip, tag, (x + 4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (80, 220, 255), 1, cv2.LINE_AA)
    cap.release()
    if got == 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True


def _build_contact_sheet(
    project: Project, seg_ids: list[str], cols: int = 5
) -> Path | None:
    """Grid the kept-segment thumbnails into one labeled montage (OpenCV).

    Uses uniform letterboxed cells so landscape and portrait frames coexist
    (ffmpeg's `tile` silently drops mismatched sizes), and labels each cell
    with its segment id for easy EDL reference.
    """
    if not seg_ids:
        return None
    cell_w, cell_h, pad = 360, 360, 6
    label_h = 26
    cols = min(cols, len(seg_ids))
    rows = math.ceil(len(seg_ids) / cols)

    sheet = np.full(
        (rows * (cell_h + pad) + pad, cols * (cell_w + pad) + pad, 3),
        20, dtype=np.uint8,
    )
    for i, sid in enumerate(seg_ids):
        fp = project.frames / f"{sid}.jpg"
        img = cv2.imread(str(fp))
        if img is None:
            continue
        # fit-within the image area (below the label strip), preserving aspect
        area_h = cell_h - label_h
        h, w = img.shape[:2]
        scale = min(cell_w / w, area_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh))

        cell = np.full((cell_h, cell_w, 3), 35, dtype=np.uint8)
        oy = label_h + (area_h - nh) // 2
        ox = (cell_w - nw) // 2
        cell[oy:oy + nh, ox:ox + nw] = resized
        cv2.putText(cell, sid, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (80, 220, 255), 1, cv2.LINE_AA)

        r, c = divmod(i, cols)
        y = pad + r * (cell_h + pad)
        x = pad + c * (cell_w + pad)
        sheet[y:y + cell_h, x:x + cell_w] = cell

    out = project.frames / "_contact.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out


# ── orchestration ────────────────────────────────────────────────────────────
def build_segments(project: Project, cfg: dict) -> dict:
    common.require_tool("ffmpeg")
    project.ensure_dirs()
    videos = project.raw_videos()
    if not videos:
        common.die(f"No video files found in {project.raw}")

    seg_cfg = cfg["segment"]
    segments: list[dict] = []
    kept_ids: list[str] = []

    for vi, video in enumerate(videos):
        log(f"Segmenting {video.name} ({vi + 1}/{len(videos)})…")
        try:
            shots = detect_shots(video, cfg)
        except Exception as e:  # noqa: BLE001 — never let one bad clip abort the batch
            log(f"  ! skipping {video.name}: segmentation failed ({e})")
            continue
        for si, (start, end) in enumerate(shots):
            if (end - start) < seg_cfg["min_segment_duration"]:
                continue
            seg_id = f"c{vi:02d}_s{si:02d}"
            m = score_segment(video, start, end, cfg)
            keep, reason = (False, "unreadable")
            if m["ok"]:
                keep, reason = _passes_hard_thresholds(m, cfg)
            keep_score = _composite_keep_score(m, cfg) if m["ok"] else 0.0

            strip_rel = None
            if keep and m["best_frame"] is not None:
                _save_best_frame(m["best_frame"], project.frames / f"{seg_id}.jpg")
                n_strip = int(cfg["score"].get("filmstrip_frames", 5))
                if _save_filmstrip(
                    video, start, end, project.frames / f"{seg_id}_strip.jpg",
                    m.get("peak_time", m["best_time"]), n=n_strip,
                ):
                    strip_rel = f"frames/{seg_id}_strip.jpg"
                kept_ids.append(seg_id)

            payoff = m.get("payoff", {})
            segments.append({
                "id": seg_id,
                "clip": video.name,
                "clip_path": str(video),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "sharpness": round(m["sharpness"], 1),
                "luma": round(m["luma"], 1),
                "shake": round(m["shake"], 2),
                "keep_score": round(keep_score, 3),
                "best_time": round(m["best_time"], 3),
                # ── content PAYOFF shape (guides selection + clean trimming) ──
                "payoff_score": payoff.get("payoff_score", 0.0),
                "peak_pos": payoff.get("peak_pos", 0.0),
                "peak_time": round(m.get("peak_time", m["best_time"]), 3),
                "settle": payoff.get("settle", 0.0),
                "motion_arc": payoff.get("motion_arc", "steady"),
                "interest_curve": payoff.get("interest_curve", []),
                "kept": keep,
                "drop_reason": "" if keep else reason,
                "frame": f"frames/{seg_id}.jpg" if keep else None,
                "filmstrip": strip_rel,
            })

    # ── still photos: each becomes one still segment (id prefix 'p') ──
    images = project.raw_images()
    still_dur = float(cfg.get("image", {}).get("still_duration", 3.0))
    img_min_sharp = float(cfg.get("image", {}).get("min_sharpness", 40.0))
    for pi, img_path in enumerate(images):
        log(f"Scoring photo {img_path.name} ({pi + 1}/{len(images)})…")
        seg_id = f"p{pi:02d}_s00"
        m = score_image(img_path, cfg)
        keep, reason = (False, "unreadable")
        if m["ok"]:
            if m["sharpness"] < img_min_sharp:
                keep, reason = False, "blurry"
            elif m["luma"] < cfg["score"]["min_luma"]:
                keep, reason = False, "too_dark"
            elif m["luma"] > cfg["score"]["max_luma"]:
                keep, reason = False, "blown_out"
            else:
                keep, reason = True, ""
        keep_score = _composite_keep_score(m, cfg) if m["ok"] else 0.0

        strip_rel = None
        if keep and m["best_frame"] is not None:
            _save_best_frame(m["best_frame"], project.frames / f"{seg_id}.jpg")
            kept_ids.append(seg_id)
        payoff = m.get("payoff", {})
        segments.append({
            "id": seg_id,
            "clip": img_path.name,
            "clip_path": str(img_path),
            "is_image": True,
            "start": 0.0,
            "end": round(still_dur, 3),
            "duration": round(still_dur, 3),
            "sharpness": round(m["sharpness"], 1),
            "luma": round(m["luma"], 1),
            "shake": 0.0,
            "keep_score": round(keep_score, 3),
            "best_time": 0.0,
            "payoff_score": payoff.get("payoff_score", 0.0),
            "peak_pos": 0.0,
            "peak_time": 0.0,
            "settle": 0.0,
            "motion_arc": "still",
            "interest_curve": payoff.get("interest_curve", []),
            "kept": keep,
            "drop_reason": "" if keep else reason,
            "frame": f"frames/{seg_id}.jpg" if keep else None,
            "filmstrip": strip_rel,
        })

    # rank kept segments by quality for convenience
    kept_ids.sort(key=lambda sid: next(s["keep_score"] for s in segments if s["id"] == sid), reverse=True)
    contact = _build_contact_sheet(project, kept_ids)

    result = {
        "project": project.root.name,
        "n_clips": len(videos),
        "n_segments": len(segments),
        "n_kept": sum(1 for s in segments if s["kept"]),
        "contact_sheet": "frames/_contact.jpg" if contact else None,
        "segments": segments,
    }
    common.write_json(project.segments_path, result)
    log(f"Segments: {result['n_segments']} total, {result['n_kept']} kept → {project.segments_path}")
    if contact:
        log(f"Contact sheet → {contact}")
    return result
