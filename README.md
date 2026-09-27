# Facebook Reel Generator

Turn a folder of raw trip clips + a music track into a single polished, beat-synced,
vertical **Facebook Reel** (`1080×1920` `.mp4`), ready to upload.

The creative brain is **Claude Code**: Python scripts do all the mechanical work (probe,
segment, score, frame-extract, beat-detect, assemble via `ffmpeg`); Claude looks at the
extracted frames, reads your stated *vibe*, and writes the edit decision. No API key, no
per-run cost.

## One-time setup

Works on macOS, Windows and Linux. You need Python 3.9+ and ffmpeg; everything else goes
into a private `.venv` created by the bootstrap script:

```bash
python3 bootstrap.py          # Windows: py bootstrap.py
```

It creates `.venv`, installs the packages, checks that `ffmpeg`/`ffprobe` are on your PATH (and
prints the install command for your OS if not), and pre-downloads the Whisper lyric model
(~1.5 GB, once). Re-run it any time; `--check` only reports. When it says SETUP COMPLETE, run
every pipeline step with the venv's own interpreter — no activation needed:

```bash
.venv/bin/python -m pipeline.prep projects/iceland        # macOS / Linux
.venv\Scripts\python -m pipeline.prep projects/iceland    # Windows
```

(The commands further down write `python -m pipeline.…` for short: use the interpreter above, or
activate the venv first. Manual setup still works: `python3 -m venv .venv`, activate,
`pip install -r requirements.txt`. `requirements.lock.txt` holds the versions known to work if a
newer package ever breaks.)

**Fonts:** captions and map cards use DIN Condensed Bold on macOS; other systems use the bundled
Barlow Condensed Bold (`assets/fonts/`, SIL Open Font License), the closest free match.

## Making a reel

1. **Create a project folder** (copy the template):
   ```bash
   cp -R projects/_template projects/iceland
   ```
2. **Drop your media in:**
   - video clips → `projects/iceland/raw/`  (`.mp4 .mov .m4v .avi .mkv .webm`)
   - **photos** → also `projects/iceland/raw/`  (`.jpg .jpeg .png .heic .webp .tif`) — each
     still becomes a short shot with a subtle **Ken Burns** zoom, reframed to 9:16
   - one music track → `projects/iceland/music/`  (`.mp3 .wav .m4a …`)
3. **Tell Claude the vibe**, e.g. *"make a chill, nostalgic ~30s reel from the iceland
   folder."* Claude then:
   - runs **prep**: `python -m pipeline.prep projects/iceland` (scores shots incl. a
     content-**payoff** metric, saves **filmstrips**, detects beats, and transcribes the
     song's **lyrics with timestamps** via local Whisper)
   - reviews `work/segments.json` + **looks at** `work/frames/_contact.jpg` and the
     per-shot `work/frames/<id>_strip.jpg` filmstrips
   - maps lyrics onto reel time: `python -m pipeline.lyricmap projects/iceland`
   - writes `work/edl.json` — ordering/trimming each clip so its imagery emotionally
     matches the **lyric + beat + energy** playing then, telling one story
   - runs **render**: `python -m pipeline.render projects/iceland`
4. **Result:** `projects/iceland/output/reel.mp4` — upload it to Facebook as a Reel.

You can also run the steps yourself; the EDL is just a JSON file (schema below).

## The two phases

```
prep   →  [catalog + lyricmap + Claude writes work/edl.json]  →  render
(probe, segment, score+payoff,   (Claude writes catalog.json;   (beat-snap timeline,
 DEEP CATALOG, beats, lyrics)     map lyrics→reel; place clips)   reframe, concat, music, encode)
```

## Deep catalog: capturing every moment

Prep runs once, so it profiles each shot **densely** rather than keeping a single hero frame
(which can hide brief but important moments — a person entering, a jump, a quick reveal). The
catalog stage (`pipeline/catalog.py`, also runnable standalone) writes `work/catalog_raw.json`
with, per shot: dense per-frame traces (sharpness/luma/colour/optical-flow), detected **moments**
(motion bursts, people-enter, settles, candid-audio peaks), a **people** summary (HOG + face
detection), **audio** activity (loudness + voice-band → `lively` flags likely talking/laughing),
and an event-anchored **dense contact sheet** per clip (`frames/<id>_dense.jpg`, ~24 frames,
moments tinted). It profiles *all* segments, including shake-dropped ones (candid action is
often handheld). Claude then reviews the dense sheets and writes `work/catalog.json` — a
vision-grounded index (descriptions, tags, people, notable moments, best trim windows) that the
rest of the edit is built from. Tunables live under `catalog:` in `config.yaml`.

## Telling a story: payoff, lyrics, and emotional sync

Three signals let the edit read as a narrative locked to the song, not just cuts on beats:

- **Content payoff (per shot, from prep).** Each segment in `segments.json` carries
  `payoff_score` (0–1: does it build to a satisfying, holdable moment?), `peak_pos` (0–1
  where that moment lands — late = builds to a reveal), `peak_time` (its absolute time in
  the source), `settle`/`motion_arc` (does the camera settle into a vista or is it still
  moving?), and a coarse `interest_curve`. Use these to pick shots that *resolve*, and to
  set each clip's `in`/`out` so it starts and ends on a meaningful beat — never abruptly cut
  mid-motion or left lingering past its payoff (unless that's the intent).
- **Filmstrips (per shot, from prep).** `work/frames/<id>_strip.jpg` shows ~5 frames across
  the shot, each labeled with its in-clip offset (the ★ marks the peak). Look at it to choose
  clean trim points.
- **Lyric sync.** Prep transcribes the track's lyrics with timestamps (`work/lyrics.json`).
  `python -m pipeline.lyricmap projects/<name>` maps them onto the reel's timeline (reel
  second → lyric) so you can place each clip where its imagery emotionally matches the words.
  `render` annotates every clip in `timeline.json` with the `lyric` playing during it, so the
  emotional match is verifiable. Pin `music_start` in the EDL to lock the lyric alignment
  exactly. Lyric sync is visual-only (no on-screen text); turn it off with
  `lyrics.enabled: false` in `config.yaml`.

## EDL schema (`work/edl.json`)

```json
{
  "target_duration": 30.0,
  "reframe_mode": "blur-pad",
  "transition": "auto",
  "pacing": "dynamic",
  "keep_original_audio": false,
  "segments": [
    { "id": "c00_s01", "in": 4.0, "hold": 0.9, "motion": "calm", "text": "Reykjavík" },
    { "id": "c03_s00", "in": 4.0, "hold": 0.3, "motion": "dynamic" },
    { "clip": "DJI_0042.mp4", "in": 2.0, "out": 7.5, "transition": "crossfade" }
  ]
}
```

- `id` references a segment in `work/segments.json`; or give an explicit `clip` (+`in`/`out`).
- `post_audio` (optional list): candid audio lines mixed under shots after assembly — see *Story tools*.
- All top-level fields are optional and fall back to `config.yaml`.
- `reframe_mode`: `blur-pad` (blurred-fill background, best for horizontal footage) ·
  `fill` (crop to fit) · `pad` (solid letterbox bars).
- Per-shot **content attributes** (this is where your judgment of the footage lives):
  - `hold` (0–1): how much the shot's content deserves to linger — a payoff vista ≈ 0.9, a
    quick detail ≈ 0.3, default 0.5.
  - `motion`: `calm` or `dynamic`. If omitted it's auto-derived from the shot's measured
    camera shake (`motion_shake_threshold` in config); set it explicitly to override (e.g. a
    serene slow pan that happens to read as shaky → `calm`).
- `pacing`: `dynamic` (default) sets each clip's length from **content × music** — the music
  energy at that moment *and* the shot's `hold`. Calm music + a high-`hold` vista → long hold;
  loud music + a low-`hold`/`dynamic` shot → fast cut. Every cut lands on a beat, durations are
  allocated so the total ≈ `target_duration`. `uniform` gives every shot the same length.
- `transition`: `auto` (default) picks each cut from music energy + content — hard **cut** on
  loud/dynamic moments, smooth **crossfade** (clean linear blend) between calm high-`hold`
  scenic shots. Force a mode globally with `cut` or `crossfade`, or override a single boundary
  with a per-shot `transition` (`cut` · `crossfade` · `fadeblack` · `wipeleft` · … and
  `dissolve` for the noisy pixel-dissolve if you ever want it). The transition is the one
  *into* that clip.
- Shots play in the order listed, one each — curate the list to roughly fill `target_duration`,
  and place your climax shot last.
- `fixed_duration` (optional, seconds): pin a shot to a set length (beat-snapped) instead of the
  content×music allocation — e.g. a ~1.5 s **cold-open hook** (lead with your most striking shot,
  hard-cut into the body) or a held ~3 s finale. Tip: open and close on the same hero shot so the
  reel's loop seam lands cleanly.
- `cut_density`: uniform mode only — a cut every N beats. Dynamic mode uses
  `beats_per_cut_min/max` and the duration/transition weights in `config.yaml`.
- `music_start` (optional, seconds): start the overlay at a specific point in the track. If
  omitted and `auto_music_section: true` (config), the system **analyzes the whole song and
  auto-selects its most captivating window** — energetic, building, and peaking late — then
  starts there (skipping dead intros). The strategy is set by `music_section_strategy` (config
  or EDL): `energy` (default) leads with the loudest/hookiest part for immediate attention;
  `build` picks a section that builds to a late peak so the song's climax aligns with the
  reel's. A short `music_fade_in` smooths the mid-song entry.
- `text` draws a caption on that clip. **Requires an ffmpeg built with libfreetype** (the
  `drawtext` filter). If your ffmpeg lacks it, captions are skipped automatically with a warning;
  install one that has it (`brew install ffmpeg` on macOS, `winget install Gyan.FFmpeg` on Windows;
  most builds include freetype).

## Story tools (multi-day trip reels)

For a trip with days, camps and a route, the reel is a story, not a montage. Four small specs drive it
(skeletons to copy: `projects/_template/work/`):

| step | command | spec → output |
|---|---|---|
| quotable lines | `python -m pipeline.candid projects/<name>` | clip audio → `work/candid_lines.json` |
| map chapter cards | `python -m pipeline.mapcards projects/<name>` | `work/maps.json` → `work/maps/mapN.mp4` |
| graded, captioned clips | `python -m pipeline.clips projects/<name> [name…]` | `work/clips.json` → `work/prerender/*.mp4` |
| beat plan → EDL | `python -m pipeline.plan projects/<name> --write-edl` | `work/plan.json` → `work/edl.json` |
| render (+ voice mix) | `python -m pipeline.render projects/<name>` | EDL `post_audio` → `output/reel.mp4` |
| verify | `python -m pipeline.verify projects/<name>` | `work/checks/{story,open,end}.jpg` |
| archive | `python -m pipeline.archive projects/<name> "note"` | `output/reel_revNN.mp4` |

- **mapcards** extracts the drawn route from a map screenshot (onX/Gaia/AllTrails), orders it by
  geodesic distance from `start`, and animates it: an opener that plans the loop, `DAY N` cards that
  light up the segment travelled, a finish card with an arrival flare timed to the song's last hit,
  and labelled checkpoints (`NIGHT 1`). Cover app widgets with `ui_patches`.
- **clips** pre-renders every shot in real time through a filmic grade with captions baked per frame
  (white condensed headline + warm lowercase sub line, top zone), plus composites: a cold-open teaser
  with stacked hook lines, pairs sharing one caption, a shot dissolving into a map card. Times are
  beat expressions (`"6B"`, `"3B+0.5"`); each file gets a 0.3 s tail so the beat-snapped timeline
  never runs out of footage.
- **plan** walks the actual beat grid, prints where every shot lands in song time with the lyric
  playing, and writes whole-beat `fixed_duration` segments with a correctly pinned `music_start`.
- **post_audio** (EDL field) lays candid lines from the raw clips under a shot with the music ducked:
  `{"src": "IMG_4174.mov", "ss": 7.9, "t": 1.1, "clip": "bridge.mp4", "offset": 0.85}`.

The editorial rules that produced the approved cut (structure, caption voice, what to leave out) live
in `CLAUDE.md` under *Story-reel playbook*.

## Tuning

All thresholds, weights, and encode settings live in **`config.yaml`** — segmentation
sensitivity, quality drop thresholds (blur/dark/shake), scoring weights, target duration,
and the full Reels encode profile. No code changes needed to tune.

## Output spec (Facebook Reels)

`.mp4` · H.264/`yuv420p` · `1080×1920` · 30 fps · 2 s closed GOP · ~10 Mbps ·
AAC-LC 48 kHz stereo 192 kbps · `+faststart`.

## Music & licensing

You supply the track. "Royalty-free" still means *licensed* — and trending/commercial songs
can be **auto-muted or blocked by Facebook's audio detection**. For safe automated use, prefer
no-attribution, commercial-OK sources (e.g. Pixabay, Uppbeat) or a paid library that clears
Meta (Epidemic Sound, Artlist). Licensing of whatever you drop in `music/` is your responsibility.
```
