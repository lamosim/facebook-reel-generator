"""Deep, dense catalog of every shot — so prep's one pass captures ALL the moments.

The sparse hero-frame + 5-frame filmstrip in segment_score can miss brief but
important moments (a person darting in, a jump-scare, a quick reveal). This stage
profiles each segment densely and writes a rich, durable index the rest of the
process reads:

  • dense per-frame traces  – sharpness / luma / colorfulness / optical-flow magnitude
  • motion EVENTS           – flow spikes (a jump / fast action), settles (arriving),
                              reveals; each a {start,end,type} moment
  • PEOPLE timeline         – HOG person + Haar face detection per frame (when/how many)
  • AUDIO activity          – original-audio loudness/voice-band energy (candid moments)
  • dense contact sheet     – ~24 labeled frames per clip (frames/<id>_dense.jpg)
  • event thumbnails        – a frame saved AT each detected moment (not just the sharpest)

Output: work/catalog_raw.json (mechanical). Claude then reviews the dense sheets and
writes work/catalog.json (descriptions/tags/people/best-trim-windows) — the source of
truth for the edit. Runs on ALL segments (incl. ones segment_score dropped for shake —
candid action is often shaky), so nothing is lost.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from . import common
from .common import Project, log
from .segment_score import _sharpness, _luma, _flow_magnitude


# ── detectors (lazily built once) ────────────────────────────────────────────
_HOG = None
_FACE = None


def _detectors():
    global _HOG, _FACE
    if _HOG is None:
        _HOG = cv2.HOGDescriptor()
        _HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        _FACE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _HOG, _FACE


def _colorfulness(bgr: np.ndarray) -> float:
    """Hasler-Süsstrunk colorfulness, normalized ~0..1."""
    b, g, r = cv2.split(bgr.astype("float"))
    rg = np.absolute(r - g)
    yb = np.absolute(0.5 * (r + g) - b)
    std = math.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean = math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return min(1.0, (std + 0.3 * mean) / 120.0)


def _count_people(bgr: np.ndarray) -> int:
    """People in frame via HOG full-body + Haar faces (max of the two)."""
    hog, face = _detectors()
    h, w = bgr.shape[:2]
    scale = 480.0 / w if w > 480 else 1.0
    small = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale != 1.0 else bgr
    n_people = 0
    try:
        rects, weights = hog.detectMultiScale(
            small, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        n_people = int(sum(1 for wt in (weights or []) if float(wt) > 0.6))
    except cv2.error:
        n_people = 0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
    return max(n_people, len(faces))


# ── dense per-frame profile of one segment ──────────────────────────────────
def profile_segment(video: Path, start: float, end: float, cfg: dict) -> dict:
    cat = cfg.get("catalog", {})
    interval = float(cat.get("sample_interval", 0.4))
    do_people = bool(cat.get("detect_people", True))

    cap = cv2.VideoCapture(str(video))
    duration = max(0.05, end - start)
    n = max(3, int(duration / interval) + 1)
    times = [start + (i / (n - 1)) * duration for i in range(n)]
    flow_w = 320

    prof = {"t": [], "sharp": [], "luma": [], "color": [], "flow": [], "people": []}
    prev_small = None
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prof["t"].append(round(t - start, 2))
        prof["sharp"].append(round(_sharpness(gray), 1))
        prof["luma"].append(round(_luma(frame), 1))
        prof["color"].append(round(_colorfulness(frame), 3))
        h, w = gray.shape
        small = cv2.resize(gray, (flow_w, max(1, int(h * flow_w / w))))
        prof["flow"].append(round(_flow_magnitude(prev_small, small), 2)
                            if prev_small is not None and prev_small.shape == small.shape else 0.0)
        prev_small = small
        prof["people"].append(_count_people(frame) if do_people else 0)
    cap.release()
    return prof


def _profile_image(path: Path, cfg: dict) -> dict:
    """Profile a still photo as a single sample (no motion/audio); detect people."""
    img = cv2.imread(str(path))
    if img is None:
        return {"t": [], "sharp": [], "luma": [], "color": [], "flow": [], "people": []}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    do_people = bool(cfg.get("catalog", {}).get("detect_people", True))
    return {
        "t": [0.0], "sharp": [round(_sharpness(gray), 1)], "luma": [round(_luma(img), 1)],
        "color": [round(_colorfulness(img), 3)], "flow": [0.0],
        "people": [_count_people(img) if do_people else 0],
    }


# ── derive moments + summaries from the dense traces ─────────────────────────
def _moments(prof: dict, cfg: dict) -> list[dict]:
    cat = cfg.get("catalog", {})
    k = float(cat.get("flow_spike_k", 1.6))
    gap = float(cat.get("min_moment_gap", 0.8))
    t = prof["t"]
    flow = prof["flow"]
    ppl = prof["people"]
    if len(t) < 3:
        return []

    fa = np.array(flow, dtype=float)
    mean, std = float(fa.mean()), float(fa.std())
    thr = mean + k * std

    raw: list[dict] = []
    # motion bursts (sudden action — jumps, fast pans, someone entering fast)
    in_burst = False
    bs = 0
    for i, f in enumerate(flow):
        if f >= thr and not in_burst:
            in_burst, bs = True, i
        elif f < thr and in_burst:
            in_burst = False
            raw.append({"start": t[bs], "end": t[i], "type": "motion_burst",
                        "flow_peak": round(max(flow[bs:i + 1]), 2)})
    if in_burst:
        raw.append({"start": t[bs], "end": t[-1], "type": "motion_burst",
                    "flow_peak": round(max(flow[bs:]), 2)})

    # people enters (0 → ≥1)
    for i in range(1, len(ppl)):
        if ppl[i] >= 1 and ppl[i - 1] == 0:
            raw.append({"start": t[i], "end": t[i], "type": "people_enter",
                        "people": int(max(ppl[i:i + 3]))})

    # settle (sustained high flow → low: arriving at / holding a composition)
    half = max(1, len(flow) // 5)
    for i in range(half, len(flow) - half):
        before = sum(flow[i - half:i]) / half
        after = sum(flow[i:i + half]) / half
        if before > mean and after < 0.5 * mean and before > 2 * after:
            raw.append({"start": t[i], "end": t[min(i + half, len(t) - 1)], "type": "settle"})
            break

    raw.sort(key=lambda m: m["start"])
    merged: list[dict] = []
    for m in raw:
        if merged and m["type"] == merged[-1]["type"] and m["start"] - merged[-1]["end"] < gap:
            merged[-1]["end"] = max(merged[-1]["end"], m["end"])
            if "flow_peak" in m:
                merged[-1]["flow_peak"] = max(merged[-1].get("flow_peak", 0), m["flow_peak"])
        else:
            merged.append(dict(m))
    return merged


def _audio_activity(video: Path) -> dict:
    """Original-audio loudness + voice-band energy → flag candid/talky/laughy moments."""
    try:
        import librosa
        y, sr = librosa.load(str(video), mono=True, sr=16000)
        if y.size < sr // 2:
            return {"has_audio": False}
        rms = librosa.feature.rms(y=y)[0]
        rms_n = rms / (rms.max() + 1e-9)
        times = librosa.times_like(rms, sr=sr)
        # voice band 300-3400 Hz energy share (speech/laughter live here)
        S = np.abs(librosa.stft(y, n_fft=1024))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
        band = (freqs >= 300) & (freqs <= 3400)
        voice_share = float(S[band].mean() / (S.mean() + 1e-9))
        peak_t = float(times[int(np.argmax(rms))])
        return {
            "has_audio": True,
            "loudness": round(float(rms_n.mean()), 3),
            "loud_peak_t": round(peak_t, 2),
            "voice_band": round(voice_share, 3),
            # elevated voice-band energy + audible level → likely talking/laughing
            # (candid moment). voice_share>1 means above-average mid-band energy.
            "lively": bool(rms_n.mean() > 0.10 and voice_share > 1.2),
        }
    except Exception:  # noqa: BLE001 — no audio stream / decode issue
        return {"has_audio": False}


# ── dense contact sheet (event-anchored) ─────────────────────────────────────
def _dense_sheet(video: Path, start: float, end: float, dest: Path,
                 n: int, moments: list[dict]) -> bool:
    """A grid of ~n frames evenly across the clip (labeled with in-clip offset).
    Frames overlapping a detected moment are tinted so the eye lands on action."""
    duration = max(0.05, end - start)
    cols = 6
    rows = math.ceil(n / cols)
    cell_w, cell_h, pad, label_h = 240, 240, 3, 18
    cap = cv2.VideoCapture(str(video))
    sheet = np.full((rows * (cell_h + label_h + pad) + pad,
                     cols * (cell_w + pad) + pad, 3), 18, dtype=np.uint8)
    got = 0
    for i in range(n):
        off = (i / max(1, n - 1)) * duration
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + off) * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        got += 1
        h, w = frame.shape[:2]
        s = min(cell_w / w, cell_h / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        cell = np.full((cell_h, cell_w, 3), 30, dtype=np.uint8)
        cell[(cell_h - nh) // 2:(cell_h - nh) // 2 + nh,
             (cell_w - nw) // 2:(cell_w - nw) // 2 + nw] = cv2.resize(frame, (nw, nh))
        r, c = divmod(i, cols)
        y = pad + r * (cell_h + label_h + pad)
        x = pad + c * (cell_w + pad)
        in_moment = next((m for m in moments if m["start"] - 0.3 <= off <= m["end"] + 0.3), None)
        tag = f"+{off:0.1f}s"
        col = (80, 220, 255)
        if in_moment:
            tag += f" [{in_moment['type'][:5]}]"
            col = (90, 255, 120)
            cv2.rectangle(sheet, (x, y + label_h), (x + cell_w, y + label_h + cell_h), (40, 120, 40), 2)
        cv2.putText(sheet, tag, (x + 3, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
        sheet[y + label_h:y + label_h + cell_h, x:x + cell_w] = \
            cv2.addWeighted(sheet[y + label_h:y + label_h + cell_h, x:x + cell_w], 0, cell, 1, 0)
    cap.release()
    if got == 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return True


# ── orchestration ────────────────────────────────────────────────────────────
def build_catalog(project: Project, cfg: dict) -> dict:
    if not cfg.get("catalog", {}).get("enabled", True):
        log("Catalog: disabled in config — skipping.")
        return {}
    seg_doc = common.read_json(project.segments_path)
    segments = seg_doc.get("segments", [])
    if not segments:
        common.die("No segments to catalog — run prep first.")

    cat = cfg.get("catalog", {})
    n_sheet = int(cat.get("dense_sheet_frames", 24))
    do_audio = bool(cat.get("detect_audio", True))

    entries: list[dict] = []
    audio_cache: dict[str, dict] = {}
    for si, s in enumerate(segments):
        sid = s["id"]
        video = Path(s["clip_path"])
        is_img = bool(s.get("is_image"))
        log(f"Cataloging {sid} ({s['clip']}) {si + 1}/{len(segments)}…")
        try:
            if is_img:
                prof = _profile_image(video, cfg)
            else:
                prof = profile_segment(video, s["start"], s["end"], cfg)
        except Exception as e:  # noqa: BLE001
            log(f"  ! profiling failed for {sid}: {e}")
            continue
        moments = [] if is_img else _moments(prof, cfg)
        ppl = prof["people"]
        people = {
            "present": any(p >= 1 for p in ppl),
            "max_count": int(max(ppl)) if ppl else 0,
            "frac_with_people": round(sum(1 for p in ppl if p >= 1) / max(1, len(ppl)), 2),
            "first_t": next((prof["t"][i] for i, p in enumerate(ppl) if p >= 1), None),
        }
        if do_audio and not is_img:
            if s["clip"] not in audio_cache:
                audio_cache[s["clip"]] = _audio_activity(video)
            audio = audio_cache[s["clip"]]
        else:
            audio = {"has_audio": False}

        # surface a candid-audio peak (talking/laughing) as a moment to look at
        if audio.get("lively") and audio.get("loud_peak_t") is not None:
            lp = float(audio["loud_peak_t"]) - s["start"]
            if 0 <= lp <= (s["end"] - s["start"]):
                moments.append({"start": round(lp, 2), "end": round(lp, 2),
                                "type": "candid_audio", "voice_band": audio.get("voice_band")})
                moments.sort(key=lambda m: m["start"])

        sheet_rel = None
        if is_img:
            # a still has nothing to densify — the hero frame is the whole shot
            sheet_rel = s.get("frame")
        elif _dense_sheet(video, s["start"], s["end"],
                          project.frames / f"{sid}_dense.jpg", n_sheet, moments):
            sheet_rel = f"frames/{sid}_dense.jpg"

        fa = prof["flow"]
        entries.append({
            "id": sid, "clip": s["clip"], "start": s["start"], "end": s["end"],
            "duration": round(s["end"] - s["start"], 2),
            "is_image": is_img,
            "kept": s.get("kept"),
            "dense_sheet": sheet_rel,
            "n_samples": len(prof["t"]),
            "moments": moments,
            "people": people,
            "audio": audio,
            "flow_mean": round(float(np.mean(fa)), 2) if fa else 0.0,
            "flow_max": round(float(np.max(fa)), 2) if fa else 0.0,
            "color_mean": round(float(np.mean(prof["color"])), 3) if prof["color"] else 0.0,
            # full traces kept for reference / future tooling
            "trace": {"t": prof["t"], "flow": prof["flow"], "people": prof["people"]},
        })

    result = {
        "project": project.root.name,
        "n_segments": len(entries),
        "n_with_people": sum(1 for e in entries if e["people"]["present"]),
        "n_with_moments": sum(1 for e in entries if e["moments"]),
        "segments": entries,
    }
    common.write_json(project.catalog_raw_path, result)
    log(f"Catalog (mechanical): {len(entries)} segments, "
        f"{result['n_with_people']} with people, {result['n_with_moments']} with motion/events "
        f"→ {project.catalog_raw_path}")
    log("Next: review the work/frames/<id>_dense.jpg sheets + catalog_raw.json, then write "
        "work/catalog.json (descriptions/tags/best-trim-windows).")
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.catalog projects/<trip-name>")
    project = common.open_project(argv[0])
    cfg = common.load_config()
    project.ensure_dirs()
    build_catalog(project, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
