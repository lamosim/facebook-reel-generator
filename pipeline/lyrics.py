"""Transcribe the music track's lyrics WITH timestamps, locally (OpenAI Whisper).

Writes work/lyrics.json:
{
  "track": "song.mp3", "model": "small", "language": "en",
  "duration": 212.4,
  "lines": [ {"start": 12.30, "end": 16.80, "text": "..."} , ... ]
}

The timestamps are in absolute song time. build_timeline maps them onto the
reel's timeline once the music-start window is chosen, so Claude can place each
clip where its imagery emotionally matches the lyric playing then.

Transcription is the slow step; the result is cached and only recomputed when the
track changes (or `force`). No API key, no network — Whisper runs on-device.
"""
from __future__ import annotations

import sys

from pathlib import Path

from . import common
from .common import Project, log


def _whisper_transcribe(track: Path, cfg: dict) -> dict:
    """Run local Whisper and return {language, lines:[{start,end,text}]}."""
    try:
        import whisper  # openai-whisper
    except ImportError:
        common.die(
            "openai-whisper is not installed. Install it into the venv:\n"
            "  . .venv/bin/activate && pip install openai-whisper\n"
            "(or set lyrics.enabled: false in config.yaml to skip lyric sync)."
        )

    lcfg = cfg.get("lyrics", {})
    model_name = lcfg.get("model", "small")
    language = lcfg.get("language") or None  # None → auto-detect

    log(f"Transcribing lyrics with Whisper '{model_name}' (this can take a few minutes)…")
    model = whisper.load_model(model_name)
    result = model.transcribe(
        str(track),
        language=language,
        word_timestamps=False,
        # Music confuses Whisper into repetition loops; these tame it.
        condition_on_previous_text=False,
        temperature=0.0,
        no_speech_threshold=float(lcfg.get("no_speech_threshold", 0.6)),
        verbose=False,
    )

    lines: list[dict] = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append({
            "start": round(float(seg["start"]), 2),
            "end": round(float(seg["end"]), 2),
            "text": text,
        })
    return {"language": result.get("language", language or "?"), "lines": lines}


def build_lyrics(project: Project, cfg: dict, force: bool = False) -> dict | None:
    """Transcribe (or load cached) lyrics for the project's music track."""
    if not cfg.get("lyrics", {}).get("enabled", True):
        log("Lyrics: disabled in config — skipping transcription.")
        return None

    track = project.music_track()
    if track is None:
        log("Lyrics: no music track found — skipping.")
        return None

    # Cache: reuse if we already transcribed this exact track.
    if project.lyrics_path.exists() and not force:
        try:
            cached = common.read_json(project.lyrics_path)
            if cached.get("track") == track.name:
                log(f"Lyrics: using cached transcription ({len(cached.get('lines', []))} "
                    f"lines) → {project.lyrics_path}")
                return cached
        except Exception:  # noqa: BLE001 — corrupt cache → just re-transcribe
            pass

    tx = _whisper_transcribe(track, cfg)
    info = common.ffprobe_info(track)
    result = {
        "track": track.name,
        "track_path": str(track),
        "model": cfg.get("lyrics", {}).get("model", "small"),
        "language": tx["language"],
        "duration": round(info.get("duration", 0.0), 2),
        "lines": tx["lines"],
    }
    common.write_json(project.lyrics_path, result)
    log(f"Lyrics: {len(result['lines'])} timed line(s) [{result['language']}] "
        f"→ {project.lyrics_path}")
    return result


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        common.die("Usage: python -m pipeline.lyrics projects/<trip-name> [--force]")
    project = common.open_project(argv[0])
    cfg = common.load_config()
    project.ensure_dirs()
    build_lyrics(project, cfg, force="--force" in argv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
