# Working in this repo (instructions for Claude)

This project turns a folder of raw trip clips + a music track into a polished vertical
**Facebook Reel**. **You (Claude) are the creative brain** — Python does the mechanical work,
and you make the edit decision by looking at frames and the user's stated vibe.

## When the user asks for a reel

Example asks: *"make a chill reel from the iceland folder"*, *"build a hype 20s recap of my
ski trip."* Do this:

1. **Identify the project folder** under `projects/<name>/`. If it doesn't exist, tell the user
   to drop clips in `projects/<name>/raw/` and one music track in `projects/<name>/music/`
   (or create it by `cp -R projects/_template projects/<name>`). **Photos** (`.jpg/.png/.heic…`)
   go in `raw/` alongside videos — each becomes a still shot with a subtle Ken Burns zoom (ids
   are prefixed `p`, e.g. `p00_s00`).

2. **Run prep:**
   ```bash
   .venv/bin/python -m pipeline.prep projects/<name>      # Windows: .venv\Scripts\python.exe
   ```
   (If `.venv` is missing or imports fail, run `python3 bootstrap.py` first — on macOS, Windows or
   Linux it creates the venv, installs `requirements.txt`, checks ffmpeg and fetches the Whisper
   model. Fix whatever it reports before continuing; the user may need to install Python or
   ffmpeg — tell them the exact command it printed.)

3. **Catalog the footage, then make the edit decision (this is your job):**
   - Read `projects/<name>/work/segments.json` — quality scores (`keep_score`, `sharpness`,
     `luma`, `shake`) and content-payoff signals (`payoff_score`, `peak_pos`, `peak_time`,
     `settle`, `motion_arc`) per shot, plus a `kept` flag.
   - Read `projects/<name>/work/catalog_raw.json` — the **deep mechanical catalog**: per-shot
     `moments` (motion bursts, people-enter, settles, candid_audio), a `people` summary
     (present / max_count / frac), `audio` (loudness, `voice_band`, `lively` = likely
     talking/laughing), and flow/colour stats. Use this to find candid/action/people moments
     that a single hero frame would hide.
   - **Look with your vision** — for each clip open the dense `frames/<id>_dense.jpg` sheet
     (~24 labeled frames; green-tinted = a detected moment). These capture the WHOLE clip, so
     brief action (a jump, someone entering, a laugh) is visible. (`_contact.jpg` and the
     5-frame `frames/<id>_strip.jpg` still exist for a quick overview.)
   - **Write `projects/<name>/work/catalog.json`** — your durable, vision-grounded index: for
     each shot a `description`, `tags` (road/trail/vista/canyon/salt-flat/forest/candid/people…),
     who/what is in it, notable `moments` with timestamps + what happens, the clean
     `best_windows` (in/out + why), and a payoff rating. This is the source of truth for the
     edit; build the EDL from it.
   - **Trim with intent:** use the catalog's `best_windows` + the dense sheet + `peak_time`/
     `settle` to set each clip's `in`/`out` so it *starts and ends on a meaningful beat* —
     never cut abruptly mid-motion, and don't linger past the payoff unless intended.
   - **Map the lyrics to reel time:** run `python -m pipeline.lyricmap projects/<name>` and read
     the reel-second → lyric table (also `work/lyric_sheet.json`). Order/trim clips so each
     one's imagery is **emotionally tied to the lyric, beat, energy, and instrumentation**
     playing at that moment. The whole reel should tell one story.
   - Use only `kept: true` segments unless the user wants something specific.
   - Decide ordering and pacing to match the **stated vibe** (chill → slower, more
     `crossfade`; hype → faster cuts). Map the vibe to a coherent narrative arc (establishing
     shot → build → highlights → emotional climax → closer), aligned to the song's arc.

4. **Write `projects/<name>/work/edl.json`** (schema in README.md). Reference segments by `id`,
   set `target_duration`, `reframe_mode`, `transition`, `keep_original_audio`, per-shot
   `hold`/`motion`/`fixed_duration`. **Pin `music_start`** to the value `lyricmap` reported so
   the lyric alignment is locked exactly between authoring and render.

5. **Render:**
   ```bash
   python -m pipeline.render projects/<name>
   ```
   Then **verify the emotional sync**: open `work/timeline.json` — each clip carries a `lyric`
   field (the line playing during it) and `reel_start`; confirm the imagery matches the words.

6. **Report** the result: `projects/<name>/output/reel.mp4`, its duration, and a one-line
   summary of the edit you made — **and a proposed post caption** (2–3 options with hashtags,
   written to `output/post.md` and shown in the reply). Offer to re-cut (different vibe / pacing / segment selection /
   lyric pairing) by editing the EDL and re-running render — prep does not need to re-run.

## Notes

- Defaults live in `config.yaml`; the EDL overrides per-project. Prefer editing the EDL over
  editing config for one-off creative choices.
- `prep` is the slow step (scene detection + scoring + the dense catalog pass — dense optical
  flow + people/audio detection on every clip — + Whisper lyric transcription). It runs once;
  thoroughness there beats re-processing. Lyrics are cached per track; `render` is fast-ish and
  cheap to re-run while iterating on the EDL. Re-run `lyricmap` after changing
  `music_start`/`target_duration`. Run the catalog alone with `python -m pipeline.catalog
  projects/<name>` (e.g. after tuning `catalog:` in config).
- The catalog profiles **all** segments, including ones `segment_score` dropped for shake —
  candid/action moments are often handheld and shaky, so don't ignore `kept: false` shots.
- Segment `id`s are positional (sorted filename order), so **adding/removing media reshuffles
  them**. `catalog.json`/EDLs keyed by old ids go stale after a re-prep — re-key by the stable
  `clip` filename (each catalog entry carries it) and only newly review the added files.
- Horizontal source footage looks best with `reframe_mode: blur-pad`. Native-vertical footage
  can use `fill`.
- Do **not** auto-source or swap in copyrighted/trending music — the user supplies the track.

## Inherited house style

The two playbooks below are the house style this repo shipped with, distilled from the original
author's reels (the reference is the CANOE-2026-SEP story reel). **Use them as the defaults on the
first pass.** They are a starting point, not a rulebook: when the user states a different taste
(pacing, caption voice, stills, slow motion, names on cards…), follow the user and update the
relevant line here so the change sticks. The named companion, places and captions quoted below are
that reference reel's worked example — never facts about the current user's trip.

## House style — the reel playbook (distilled from CAPEX-2026-JUL rev01→rev04, 2026-07)

The original author's preferred end-state, reached over four revisions. **Default to producing the rev04-style
cut on the first pass** for any new folder of trip clips + one song:

1. **Music first — cut to the song's climax, not its opening.** After prep, analyze the whole
   track: bucket `beats.json` `beat_energy` per 5s against the timed lyrics, and run
   `_select_music_window` (from `pipeline.build_timeline`) across strategies/targets. Choose a
   **40–70s late-peaking window** — typically the final chorus block + resolving outro
   (energy peak in the last third). Start on a phrase boundary at a section change (e.g. the
   drop into a quiet verse); end where the last sung line **naturally decays** so the 2s music
   fade rides the song's own exhale, never chops a phrase.
2. **One psychological arc, ~10–12 shots, one shot per emotional beat.** The template that
   works: restless motion open (road POV) → one long scenic **exhale** (best vista, longest
   hold ~6s) → **effort/commitment** action (climb) → a **claim** landing on the first chorus
   hit (person plants themselves; funny is good) → short road punch in an **instrumental gap**
   (transitions/jokes live in gaps) → second-location arrival ON a chorus line → wildlife/rest
   beat → **the belonging shot** (family/candid warmth) held across the final chorus refrain —
   this is the emotional center, pair the most tender lyric with the most human image → a
   child-gaze/echo shot as the outro opens → golden-hour going-home → the quietest, most poetic
   image (the moon) fading out with the song.
3. **Cut feeling-duplicates, not just visual ones.** Two contemplative landscapes in a row,
   two climbing beats, three car shots in one outro = one each. Every keep/cut must be
   justifiable in psychological terms. Fewer, longer shots beat coverage: avg hold 4–5s,
   nothing under ~1.8s except a deliberate road punch.
4. **Standing preferences:** distinctive wildlife (sea lions!) gets a real dedicated beat, not
   background. A driving clip must bridge location chapters; let the road recur
   open/pivot/close as the story's spine. People: minimal and purposeful — candid family
   moments are the core; buddies only when funny or thesis-carrying; no generic walking shots.
   Reels run longer/smoother rather than choppy; never auto-shorten below ~45s.
5. **Beat-true trims, never overtrimmed.** Every `fixed_duration` = a whole beat multiple
   (60/BPM × n, n≈4–14); pick `in` so the clip's key internal action (vista opening, step-up,
   wave crash) lands on a beat; enter on action starts, exit after the action completes.

## Story-reel playbook — multi-day trips (distilled from CANOE-2026-SEP rev01→rev17, 2026-09)

The original author's preferred end-state for a **multi-day adventure** (hike / canoe / road trip) with one song,
reached over 17 revisions. **Default to producing this cut on the first pass** for any folder of
clips from a trip with days, camps and a route. It is spec-driven: four JSON files + the EDL.

```bash
python -m pipeline.prep projects/<name>            # once (segments, catalog, beats, lyrics)
python -m pipeline.candid projects/<name>          # quotable lines in the clips' audio -> work/candid_lines.json
python -m pipeline.mapcards projects/<name>        # work/maps.json  -> work/maps/mapN.mp4 (route chapter cards)
python -m pipeline.clips projects/<name>           # work/clips.json -> work/prerender/*.mp4 (graded, captioned, real time)
python -m pipeline.plan projects/<name> --write-edl   # work/plan.json -> beat-walk print + EDL segments + pinned music_start
python -m pipeline.render projects/<name>          # timeline + assemble (+ EDL post_audio voice mix)
python -m pipeline.verify projects/<name>          # anchors table + work/checks/{story,open,end}.jpg — LOOK at them
python -m pipeline.archive projects/<name> "note"  # output/reel_revNN.mp4 (never overwrite; reel.mp4 = latest)
```
Start from the skeletons in `projects/_template/work/{clips,maps,plan}.json` + `edl.example.json`
(cut down from the reference CANOE-2026-SEP specs; on the original author's machine the full worked
example is still in `projects/CANOE-2026-SEP/work/`); copy and adapt them rather than starting blank.

1. **Story spine = the complete adventure, in order.** Chronological chapters (one per day), NIGHT
   markers, the route map as narrator. Real time only — **no slow motion** (rev06 tried it; the author
   dropped it). Video only, no stills. Filmic grade on everything (`pipeline.clips` defaults).
2. **Music map.** Cold open on the *tail of the previous chorus* (~11 beats: three fast cuts of the
   hardest effort + a stacked hook) → the quiet section for the title map card and Day 1 → the
   build for Day 2 (ends with the last-day map card under the vocal pickup) → the final chorus
   drop = the biggest effort of the last day → every chorus hit on a cut → the "burn"/turn hit =
   the sun/grace beat → the last hit = arrival on the finish card → fade to black with the song.
   Reels run ~80 s (147 beats); pin `music_start` on the cold-open beat.
3. **Map chapter cards** (`pipeline.mapcards`): opener that "plans" the whole loop with the title
   (`30 MILES / 3 DAYS · 2 CANOES · ADIRONDACKS`), `DAY N` cards with a chapter subtitle
   (`PONDS AND PORTAGES`, `THE RIVER`) whose glowing segment ends where they slept, `NIGHT 1/2`
   labels beside the camp rings, a finish card whose arrival flare lands on the song's last hit
   and carries the closing line (`CARRIED IN. PADDLED OUT.`). **No names/date line and no
   sign-off question on the end card** — the author removed both.
4. **Hook first.** Facebook viewers decide in 2 s and watch muted: the cold open stacks a
   premise + twist in big type (`WHAT 30 MILES LOOKS LIKE / WHEN YOU HAVE TO CARRY THE BOAT.`).
5. **Captions carry the story, sparingly** (`pipeline.clips` bakes them; white condensed headline
   + warm lowercase sub line, top zone). Final count that worked: chapter markers
   (`DAY 1 · THE CARRY`, `NIGHT 1`, `NIGHT 2`, `BEAVER DAM.`, `SO WE HAULED.`, `THE OSWEGATCHIE`,
   `AND THEN, THE SUN.`, `THE FINISH LINE`) plus **five** sub-lines in 80 s; the paddle, the fire
   embers, the gray river and the whole home stretch run wordless. "Less says more."
6. **Caption voice (author-approved):** humble and first-person ("we" = the two of them; never a
   crowd), no exclamation marks, specific nouns over adjectives, the joke in the last three words,
   ≤ 50 characters. Give the credit away (the friend, the river, luck: `luke knew the way. i just
   knew luke.`, `we did not plan this part.`) but **never reference the friend's pace/speed,
   weight or eating** — no line anyone could misread as a dig. One honest stumble max (`8.2 miles
   back to the road. we thought about it.`), one grace note, an optional callback gag. Only facts
   the user stated or the footage shows (the sign's `8.2`). Never instructional "you", never a lecture.
7. **Candid audio.** `pipeline.candid` finds lines like "well, we made it"; put the best one under
   the finish shot via EDL `post_audio` (music ducked, +4.5 dB, ~1.1 s). **Audio only — no quote
   caption.** Mention profane lines, leave them out.
8. **Deliver the post caption with the reel.** Write `output/post.md`: 2–3 options in the same
   humble first-person voice (facts only, ≤ 2 lines before the fold, 3–4 hashtags, tag the
   companion) and put them in the reply. **Each option carries "receipts":** which words do
   the work and what they signal (warmth before competence — Fiske; praise of others and a
   small self-directed joke — Cialdini's liking principle, the pratfall effect; no humblebrag —
   Sezer/Gino/Norton 2018), plus what was deliberately left out. The reel is not done until
   the caption and its justification are.
9. **Iterate by offering numbered options.** When a line needs work, list 8–14 alternatives with
   the mechanism varied (understatement, reversal, credit-giving, callback), mark 2–3 picks, and
   build the rev the user names. Each caption change is a two-minute rebuild of one clip + render.

### Settled defaults (every decision from the reference CANOE-2026-SEP session, 2026-09-14/15)

Treat these as settled defaults; don't re-ask, and don't re-propose the rejected items.

**Story & structure**
- Mostly chronological, "so that it shows the complete adventure"; chapters per day, NIGHT markers.
- Rugged / "manly" tone: effort shots (canoe hoist, portage, beaver-dam haul, paddle strokes, fire,
  the rock-on at the takeout). No gear-fiddling, no snack shots.
- Show where they are on the route "in an engaging and profound way" → animated map chapter cards.
- Cold-open hook first (premise + twist in big type), then the title map on the quiet section.
- ~80 s is fine ("reels run longer/smoother"); every chorus hit on a cut; finish on the last hit.

**Visual**
- Video only — "I don't like the still shots." No Ken Burns photos.
- Real time only — no slow motion (rev06's slow motion was dropped for good).
- Filmic grade (lifted blacks, cool shadows / warm highlights, soft vignette) on every clip.
- Map cards: route glows orange as it's travelled, NIGHT 1 / NIGHT 2 labels on the camp rings,
  chapter subtitle under DAY N, the finish card's arrival flare on the song's last hit.

**Captions — voice**
- "Less says more, let the videos do the talking": chapter markers + ~5 sub-lines per reel; long
  stretches wordless (the paddle, the embers, the gray river, the whole home stretch).
- Humble but engaging "such that people like you": first-person "we" (it was two of them — never
  crowd phrasing like "we all"), lowercase warm sub-lines, no exclamation marks, the joke in the
  last three words, ≤ 50 characters.
- Give the credit away (companion, river, luck) — but **never** a line about a companion's pace or
  speed, weight, food or eating, and nothing anyone could misread as a dig.
- Facts only from the user or visible in the footage (reference reel: the sign's "8.2"; Luke did the navigating and
  carried all the food and beer). Never invent mileage or place claims.
- No instructional "you" (except the hook), no lectures, no names/dates, no sign-off question.

**Captions — the approved set (rev17)**
- Cold open: `WHAT 30 MILES / LOOKS LIKE / WHEN YOU HAVE / TO CARRY THE BOAT.`
- Title card: `30 MILES / 3 DAYS · 2 CANOES · ADIRONDACKS`
- `DAY 1 · THE CARRY / 8 miles of trail. the canoes did not help.`
- `NIGHT 1 / a lean-to and a lake.`
- `DAY 2 / PONDS AND PORTAGES` (card) · sign: `8.2 miles back to the road. we thought about it.`
- paddle: `paddle across. carry over. repeat.` · `NIGHT 2` · `DAY 3 / THE RIVER` (card)
- the drop: `luke knew the way. i just knew luke.`
- `BEAVER DAM. / the beaver did not leave a gate.` → `SO WE HAULED.` → `THE OSWEGATCHIE`
- `AND THEN, THE SUN. / we did not plan this part.` → `THE FINISH LINE` (audio: "Well, we made it")
- End card: `30 MILES / CARRIED IN. PADDLED OUT.` then fade to black.

**Rejected for tone (don't bring back):** "the shoulders say more", "YOU HAUL." (cocky /
instructional); "luke set the pace. he always does." (pace); every pack / heaviest-pack / food /
beer line (weight-adjacent); the quote caption “well, we made it.” (audio only); "same time next
year?" and "IAN & LUKE · SEPTEMBER 2026" on the end card. **Cut for quantity, not tone** (fine to
offer again if a spot feels empty): "the best sleep all year", "this part we'd do all day", "still
not helping." (callback), "nothing to do, and all night to do it.", "THE HOME STRETCH / one bend at
a time. then one more.", "downstream from here. the river helped."

**Audio**
- Music supplied by the user only; never auto-source or swap tracks.
- The candid line at the finish stays in (rev15 without it was reversed in rev16); mixed under the
  bridge shot with the music ducked; profane candid lines are mentioned, not used.

**Process**
- Archive every render as `output/reel_revNN.mp4` (never overwrite); `reel.mp4` = latest.
- Offer numbered caption options (8–14, mechanisms varied, 2–3 marked as picks); the user answers
  with a number. When they supply a fact or his own wording, use it verbatim or optimise it lightly.
- Look at the verify tiles before archiving; report what changed in plain words, not settings.
- Every reel comes with a proposed Facebook post caption (options in `output/post.md`); the author asked
  "where is the proposed caption for this post?" when one was missing.
- "I like it when each proposed caption has a justification with receipts on why it makes the
  poster look likeable, down to earth, and humble" — per option: the phrases that carry it,
  the effect each one has on the reader, the principle behind it, and what was avoided.

## Pipeline gotchas (learned the hard way)

- **Lyric-locked cuts need `fixed_duration` on every segment** — dynamic pacing undershoots
  and drifts off the lyric anchors. Beat-snapping + transition overlaps shrink the result:
  expect actual ≈ requested − (0.5s per crossfade + 0.06s per cut + ~0.5–1s snap-down), i.e.
  **request ~7% over** the music-window length. After every render, read `work/timeline.json`
  (`reel_start` + `lyric` per clip) and nudge ±1 beat until anchors hit; extend the final two
  holds if the last line would fade early.
- Workflow order: EDL (with intended `music_start`) → `lyricmap` → **pin the snapped
  `music_start` it reports** back into the EDL → render → verify timeline → iterate.
- Whisper mishears repeated refrains ("Home at last" → "Homeless"/"Holy land"/"Hope at
  last") — trust the timestamps, correct the words from knowledge of the song.
- **Archive every render**: `cp output/reel.mp4 output/reel_revNN.mp4` (next NN; never
  overwrite an existing rev). `reel.mp4` stays the latest.
- Env: background/`run_in_background` shells don't inherit venv activation and the working dir
  persists between calls — invoke `./.venv/bin/python -m pipeline.<step>` (Windows:
  `.venv\Scripts\python.exe`) from the repo root with absolute-ish paths.
- Cross-platform rules: fonts resolve through `common.caption_font()` (DIN Condensed Bold on macOS,
  the bundled Barlow Condensed elsewhere — never hard-code a font path); every text file is opened
  with `encoding="utf-8"`; paths inside ffmpeg filter strings go through `assemble._filter_path`
  and concat lists through `assemble._concat_list_text`. After touching `pipeline/`, run
  `.venv/bin/python -m unittest discover -s tests -t .`.
- `pipeline/catalog.py` can fail per-clip with a numpy "truth value of an array" error; the
  clip is still usable — review its `frames/<id>_strip.jpg` instead of the missing dense sheet.
- Verify output visually with ffmpeg frame tiles, e.g.
  `ffmpeg -i output/reel.mp4 -vf "select='eq(n\,N)+…',scale=270:480,tile=4x1" -frames:v 1 check.jpg`.
- **Pre-rendered clips must outlast their beat span.** Beats drift (up to 0.57 s vs the 0.5572
  median); `build_timeline` clamps `out` to the file, so a file cut to `n × median` runs short and
  every later anchor drifts. `pipeline.clips` adds a 0.3 s tail automatically — keep it.
- **`music_start` snaps to the first beat ≥ the value.** Pin it a hair *below* the beat time
  (`pipeline.plan` does this); `113.36` for beat `113.3598` lands you one beat late.
- Never write pre-rendered files into `work/clips/` — `assemble` deletes `clip_*.mp4` there.
  Use `work/prerender/` (the EDL references `../work/prerender/<name>.mp4`).
- The Write tool turns `\u201c` escapes into literal curly quotes; when patching scripts, match the
  literal characters, and chain *patch → build → render* with `&&` so a failed patch never renders
  stale text (rev08 shipped old captions once because the patch step failed silently).
- Whisper on clip audio: the `small` model is enough; drop segments with `no_speech_prob > 0.6`. It
  mishears words ("Oh, we made it" was really "Well, we made it") — trust the timestamps, confirm
  the words with the user before captioning them, and prefer playing the line as audio only.
- Verify every render with `pipeline.verify` and *look* at the story tile before archiving; an
  end-card line once landed on top of the finish marker and only the tile showed it.
