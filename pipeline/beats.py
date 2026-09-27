"""Beat detection on the project's music track (librosa)."""
from __future__ import annotations

from pathlib import Path

from . import common
from .common import Project, log


def detect_beats(track: Path, cfg: dict) -> dict:
    import librosa
    import numpy as np

    log(f"Analyzing beats in {track.name}…")
    y, sr = librosa.load(str(track), mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # trim=False: keep beats through the final chorus/outro (default trimming drops
    # trailing beats, which truncated the grid at ~153s on a 197s track).
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Per-beat energy so the editor can pace cuts to the music's dynamics:
    # RMS loudness (overall energy) + onset strength (transient "busyness").
    def _norm(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

    beat_energy: list[float] = []
    energy_peak_time = 0.0
    if len(beat_frames):
        rms = librosa.feature.rms(y=y)[0]
        oenv = librosa.onset.onset_strength(y=y, sr=sr)
        # beat_frames index the hop grid shared by rms/oenv
        bf = np.clip(beat_frames, 0, min(len(rms), len(oenv)) - 1)
        rms_n = _norm(rms[bf])
        onset_n = _norm(oenv[bf])
        combined = _norm(0.6 * rms_n + 0.4 * onset_n)  # loudness-weighted energy
        beat_energy = [round(float(v), 4) for v in combined]
        # absolute time of the loudest moment (for aligning the visual climax)
        rms_times = librosa.times_like(rms, sr=sr)
        energy_peak_time = float(rms_times[int(np.argmax(rms))])

    result: dict = {
        "track": track.name,
        "track_path": str(track),
        "duration": round(duration, 3),
        "tempo": round(float(tempo), 2) if hasattr(tempo, "__float__") else float(tempo),
        "energy_peak_time": round(energy_peak_time, 3),
        "beat_times": [round(float(t), 4) for t in beat_times],
        "beat_energy": beat_energy,
    }

    if cfg["beats"].get("use_onsets_fallback", True):
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        result["onset_times"] = [round(float(t), 4) for t in onset_times]

    return result


def build_beats(project: Project, cfg: dict) -> dict:
    track = project.music_track()
    if track is None:
        common.die(f"No music track found in {project.music} (.mp3/.wav/.m4a/…)")
    result = detect_beats(track, cfg)
    common.write_json(project.beats_path, result)
    log(
        f"Beats: tempo≈{result['tempo']} BPM, {len(result['beat_times'])} beats "
        f"over {result['duration']}s → {project.beats_path}"
    )
    return result
