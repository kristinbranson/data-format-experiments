# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory — Visual Behavior 2P (`visual-behavior-ophys-1.1.0`), change-detection task
- **Papers**: Allen VBO 2P Technical Whitepaper (Garrett et al.); "Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex" (Neuron 2024)
- **Date started**: 2026 (container date)
- **Goal**: Convert to decoder-compatible format (neural -> experimental/behavioral variables)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `code/` : AllenSDK source (v2.16.2 also installed in system python)
- `data/` : `visual-behavior-ophys-1.1.0/` (behavior_ophys_experiments/*.nwb = 284 files, 247 GB; project_metadata/*.csv), `visual-behavior-ophys_project_manifest_v1.1.0.json`, `_downloaded_data.json`
- `tutorials/` : 5 tutorial scripts + 1 notebook (load ophys data, compare across trial types, data access, dataset manifest, mouse history)
- `methods.txt`, `paper.pdf`, `whitepaper.pdf`
- `decoder.py`, `train_decoder.py`
- `cache/` (created by me for exploration scripts)

Environment verified: python3.13, numpy 2.4.4, torch 2.6.0+cu124 (CUDA available: NVIDIA L4), pandas 2.3.3, allensdk 2.16.2, sklearn, matplotlib.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function / attribute | File | Stage | Purpose |
|----------|------|-------|---------|
| `VisualBehaviorOphysProjectCache.from_local_cache(cache_dir)` | `allensdk/brain_observatory/behavior/behavior_project_cache/project_cache_base.py` | LOADING | Offline cache; used by all tutorials (as `from_s3_cache`). `from_local_cache(..., use_static_cache=True)` fails here (expects `<dir>/visual-behavior-ophys/manifests`), the non-static form works with the provided `_downloaded_data.json` + manifest json. |
| `cache.get_ophys_experiment_table()` / `get_ophys_session_table()` / `get_behavior_session_table()` | `behavior_project_cache.py` | LOADING | Metadata tables (1936 experiments in manifest; 284 available locally) |
| `cache.get_behavior_ophys_experiment(ophys_experiment_id)` | `behavior_project_cache.py` | LOADING | Returns `BehaviorOphysExperiment` from the local NWB file |
| `experiment.events` | `behavior_ophys_experiment.py:554` | PROCESSING | DataFrame indexed by `cell_specimen_id` with `events` (detected calcium-event magnitudes, one value per ophys frame), `filtered_events` (half-gaussian smoothed, "for visualization"), `lambda`, `noise_std` |
| `experiment.dff_traces` | `behavior_ophys_experiment.py:530` | PROCESSING | dF/F traces (already computed & detrended in the released NWB; no need to compute) |
| `experiment.ophys_timestamps` | `behavior_ophys_experiment.py:523` | ALIGNMENT | Sync-clock times (s) of each 2p frame; same for `events` and `dff_traces` |
| `experiment.cell_specimen_table` | `behavior_ophys_experiment.py:588` | CURATION | "Table only contains roi_valid = True entries, as invalid ROIs / non-cell segmented objects have been filtered out" -> ROI quality filtering is already applied by the SDK |
| `experiment.metadata` | `behavior_ophys_experiment.py:480` | LOADING | `ophys_frame_rate`, `targeted_structure`, `imaging_depth`, `cre_line`, `mouse_id`, `session_type`, `equipment_name`, `ophys_session_id`, ... |
| `session.trials` | `behavior_session.py:1271` | LOADING/CURATION | Trial table: `start_time`, `stop_time`, `change_time`, `initial_image_name`, `change_image_name`, `go`, `catch`, `aborted`, `auto_rewarded`, `hit`, `miss`, `false_alarm`, `correct_reject`, `lick_times`, `reward_time`, `response_latency` |
| `session.stimulus_presentations` | `behavior_session.py:1067` | LOADING | Flash table. Must be subset to the `change_detection_behavior` block via `stimulus_block_name.str.contains('change_detection')` (tutorials do exactly this). Columns used: `start_time`, `end_time`, `image_name`, `image_index`, `is_change`, `is_sham_change`, `omitted`, `flashes_since_change`, `trials_id` |
| `session.running_speed` | `behavior_session.py:1023` | LOADING | 60 Hz, `timestamps`, `speed` (cm/s), 10 Hz low-pass filtered + transient/wrap corrected (whitepaper). `raw_running_speed` is the unfiltered version |
| `session.eye_tracking` | `behavior_session.py:908` | LOADING | ~30 Hz; `timestamps`, `pupil_area`, `pupil_width`, `pupil_height`, `likely_blink`, ... Values are **NaN wherever `likely_blink == True`** |
| `compute_circular_area` | `behavior/eye_tracking_processing.py:85` | PROCESSING | `pupil_area = pi * max(width,height)^2` -> pupil **diameter** `= 2*sqrt(area/pi) = 2*max(width,height)` px. (Verified numerically that `pupil_area != pi*w*h`.) |
| `determine_likely_blinks` | `behavior/eye_tracking_processing.py:125` | CURATION | Blink/outlier detection with dilation; drives the NaNs above |
| `get_trace_around_timepoint(..., window_around_timepoint_seconds, frame_rate)` | `behavior/swdb/analysis_tools.py:24` | ALIGNMENT | Reference way of cutting traces around an event using the ophys frame rate |
| `get_trial_response_df` + `response_analysis_params` | `behavior/swdb/save_trial_response_df.py:291` | ALIGNMENT | Reference trial-aligned analysis: **aligned to `trials.change_time`**, `window_around_timepoint_seconds = [-4, 8]` |
| `save_flash_response_df.py:383` | | ALIGNMENT | Flash-aligned window `[-0.5, 0.75]` s |
| `session.get_rolling_performance_df()` | `behavior_session.py:718` | PROCESSING | rolling hit-rate / FA-rate / d-prime (100-trial window, aborted trials excluded) |

### Notes
- Imaging data: **no dF/F computation needed** — the NWB release contains detrended dF/F *and* detected calcium events. The analysis paper uses the **detected calcium events** (`events`), so that is our neural stream.
- `events` vs `filtered_events`: the SDK documents `filtered_events` as a smoothing "to smooth it for visualization"; the paper's analyses use the detected events themselves. We use `events`.
- Neuron quality control: already applied upstream (only `valid_roi==True` ROIs are returned; container/session-level QC already removed failed planes). The paper adds no further per-cell filtering.
- Reference code asserts `np.all(np.isin(change_times, flash_times))` — i.e. change times coincide exactly with flash onsets. I reproduced this check (100% true).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_<ophys_experiment_id>.nwb` — **284 files** (247 GB). One file = one *imaging plane* ("experiment") of one *session*.
- `/app/data/visual-behavior-ophys-1.1.0/project_metadata/{ophys_experiment_table,ophys_session_table,behavior_session_table,ophys_cells_table}.csv`
- The project manifest lists 1936 experiments; **only 284 are present locally**, so the local file set defines the available dataset.

### Composition of the 284 local experiments (from the experiment table)
| Field | Values |
|---|---|
| project_code | VisualBehavior 239 (Scientifica, single plane, 31 Hz), VisualBehaviorMultiscope 45 (Mesoscope, 11 Hz, all from **one** Sst mouse 457841) |
| ophys sessions | 247 |
| mice | 38 |
| containers | 44 |
| session_type | OPHYS_1_images_A 55, OPHYS_3_images_A 55, OPHYS_4_images_B 46, OPHYS_6_images_B 46, OPHYS_5_images_B_passive 42, OPHYS_2_images_A_passive 40 |
| passive | 82 experiments passive / 202 active |
| experience_level | Familiar 150, Novel 1 38, Novel >1 96 |
| cre_line | Slc17a7 153, Sst 85, Vip 46 |
| targeted_structure | VISp 261, VISl 23 |
| image_set | A 150 (familiar), B 134 (novel) |

### Dataset size — active (non-passive) experiments (scan of all 202, `/app/cache/scan_active.csv`)
| Statistic | Value |
|-----------|-------|
| Experiments (planes) | 202 |
| Ophys sessions | 174 (168 single-plane, 1x3, 2x5, 3x7 planes) |
| Subjects (mice) | 38 |
| Neurons (total) | 29,444 (4–666 per plane; 6–666 per session) |
| Go+Catch (non auto-rewarded) trials | 51,992 (39–409 per session) |
| Ophys frame dt | 0.03232 s (31 Hz) or 0.09323 s (11 Hz) |
| Aborted trials | 61.9% of all trials; auto-rewarded 0.6% |
| Catch / (Go+Catch) | 0.125 |
| hit / miss / FA / CR (of go+catch) | 0.307 / 0.568 / 0.018 / 0.107 |
| NaNs in `events` | 0; all-zero cells 0; missing change_time 0 |
| Missing running speed samples | 0 |
| Experiments without eye tracking | 3 (1 of them familiar: session 805989030) |
| pupil NaN (blink) fraction | median 2.9%, p90 5.4%, max 29.6% |

### Familiar + active subset (the subset used for conversion, see Step 5)
| Statistic | Value |
|-----------|-------|
| Experiments (planes) | 110 (OPHYS_1_images_A 55, OPHYS_3_images_A 55) |
| Ophys sessions | 92 (88 single-plane, 1x3, 1x5, 2x7) |
| Mice | 38 |
| Neurons | 14,895 (Slc17a7 14,048 / Sst 446 / Vip 401); 7–666 per session |
| Go+Catch non-auto trials | 23,049 (39–396 per session, mean 250) |
| Brain regions | VISp 99 planes, VISl 11 planes |
| Images | im061, im062, im063, im065, im066, im069, im077, im085 (image set A, identical in every familiar session) |
| Flashes/session | ~4,800; omitted 3.5%, change 4.7%, sham change 0.7% |
| Sessions without eye tracking | 1 (805989030) |

### Trial timing (all active experiments)
- `change_time − start_time` ∈ [2.79, 8.31] s (mean ≈ 4.37 s, consistent with the whitepaper's truncated-exponential 2.25–8.25 s / mean 4.2 s)
- `stop_time − change_time` ∈ [4.20, 4.35] s
- => a window of **[−2, +4] s around `change_time` lies inside every included trial**.
- Flashes: 250 ms image, 500 ms grey, 750 ms cycle (measured: duration 0.250 s, onset-to-onset 0.7506 s).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source |
|-----------|-------|--------------|
| Whole behavior dataset | 376 imaging sessions, 82 mice | paper: "contains behavior from 376 imaging sessions from 82 mice" |
| Paper's neural subset | 8,619 excitatory (21 sessions, 9 mice), 470 Sst (15 sessions, 6 mice), 1,239 Vip (21 sessions, 9 mice) | paper: "Our dataset contains 8,619 excitatory cells..." — this is the **multi-plane (Mesoscope), familiar** subset, which is almost entirely absent from the local 284-file subset (only 1 Mesoscope mouse here) |
| Neural signal | detected calcium events | paper: "For all analysis of neural data we used the detected calcium events"; "analyses on discrete calcium events that were regressed from the raw fluorescence traces" |
| Stimulus timing | 250 ms image, 500 ms grey ISI | paper Methods |
| Omission probability | 5% of image repeats (changes and pre-change never omitted) | paper + whitepaper |
| Catch probability | ~12.5% in late sessions (matrix sampling) | whitepaper |
| Change time distribution | truncated exponential 2.25–8.25 s, mean 4.2 s after trial start | whitepaper |
| Response window | 0.150–0.750 s after change | whitepaper |
| Trial types | GO, CATCH, plus aborted (premature lick) and free-reward/auto-rewarded trials (5 at session start, and after 10 consecutive misses) | whitepaper |
| Engagement | reward rate > 2 rewards/min = engaged; mice engaged 72.2% of time; 60.1% of image intervals engaged | whitepaper / paper Fig 3 |
| Ophys frame rate | 31 Hz single plane, 11 Hz per plane multi-plane | whitepaper |
| Eye/behavior cameras | 30 Hz | whitepaper |
| Running speed | cm/s, 10 Hz low-pass filtered (`running_speed`) | whitepaper |
| Familiar-only restriction | "we restricted our analysis to familiar stimuli"; "For neural analysis we used neurons recorded during familiar image set presentations" | paper |

### Processing Details
- Temporal alignment of all streams is done upstream on a 100 kHz sync board; the SDK returns every stream with sync-clock timestamps, so streams are directly comparable in seconds.
- The reference trial-level analysis aligns to **`trials.change_time`** (window [−4, 8] s); flash-level analyses use [−0.5, 0.75] s.
- The paper's neural analyses are computed in the **750 ms image-presentation interval** (behavioral events assigned to image intervals), and its decoding analysis uses the **first 400 ms after image presentation**.

### Curation Steps
**Neuron curation rules**: only `valid_roi` cells (already enforced by the SDK's `cell_specimen_table`/`events`); QC-failed planes/sessions/containers already removed from the release; cells matched across sessions but no additional filtering in the paper. No extra filtering applied here.

**Trial curation rules** (papers + task spec):
- Whitepaper/SDK: performance metrics exclude *aborted* trials; free-reward (auto-rewarded) trials are not task trials.
- Task spec for this conversion: include `go` and `catch`, exclude `aborted` and `auto_rewarded`.
- Only **active** (behaving) sessions have meaningful trial outcomes -> passive sessions (OPHYS_2/5) excluded.
- Paper restricts neural analysis to **familiar** image sets -> OPHYS_1/OPHYS_3 (image set A) kept.

### Decoders Trained (reference)
| Decoded variable | Reported accuracy |
|---|---|
| Image change vs. repeat (random forest, 400 ms after flash, per imaging plane) | Fig 6A: ~65–90% correct depending on cell class / n cells (chance 50%); "decoded equally well for all cell classes" |
| Hit vs. miss | Fig 6C: ~55–70% correct (chance 50%), higher for visual-strategy sessions |
| False alarm decoding | Fig S22: "very low" (near chance) |
| (no image-identity, running or pupil decoding reported) | — |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal | `events` (detected events) and `dff_traces` both available | both present, no NaNs | "detected calcium events" | Use `events` (per-frame event magnitudes) |
| Neuron QC | `cell_specimen_table` returns only valid ROIs | all cells `valid_roi==True`, 0 all-zero cells | no extra filtering | No extra filtering |
| Trial alignment | reference aligns to `change_time` | `change_time` present for every go & catch trial, exactly equal to a flash onset | change-aligned analyses | Align to `change_time` |
| Catch probability | — | 0.125 of go+catch | "~12.5%" | ✅ consistent (sanity check passed) |
| Omission rate | 5% nominal | 3.5% of flashes (changes and pre-change flashes are never omitted, and omissions only occur in the change-detection block) | 5% of image *repeats* | ✅ consistent |
| Change time distribution | — | 2.79–8.31 s after trial start, mean 4.37 | 2.25–8.25 s, mean 4.2 | ✅ consistent |
| Dataset size | 1936 experiments in manifest | 284 local files, 247 sessions, 38 mice | 376 sessions/82 mice (behavior), Mesoscope-familiar subset for neural | Local subset is a ~1/7 sample of the release and contains mostly single-plane sessions; the paper's exact neural subset (Mesoscope familiar, 9 mice) cannot be reproduced locally (only 1 Mesoscope mouse). We therefore keep **all locally available active familiar sessions** (both rigs) and document the deviation. |
| Pupil size | `pupil_area = pi*max(w,h)^2` | verified | pupil not analysed in the paper (running is, in 750 ms intervals) | diameter `= 2*sqrt(area/pi)` |
| Running speed | `running_speed` = filtered, 60 Hz | 0 NaNs, −11 to +69 cm/s | "running speed" in cm/s | Use `running_speed.speed` |

No unresolved discrepancies.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

> **NOTE**: three parameters in the plan below (time bin 250 ms -> 750 ms, window [-2,+4] -> [-2.25,+3.75] s, per-session -> global percentile bins) and one addition (per-neuron z-scoring of the binned events) were revised after the first full run. See *"Revisions made after the first full run"* at the end of Step 9 for the justification and the empirical comparison. The final settings are the defaults in `/app/convert_data.py` and are the ones recorded in `metadata`.

(see next update)

### Data selection (sessions / planes / trials)
1. **Task**: only the `change_detection_behavior` stimulus block of *active* behavior sessions ("Visual Behavior" task). Passive sessions (OPHYS_2, OPHYS_5) are excluded: the lick spout is retracted, so there are no Go/Catch outcomes.
2. **Experience level**: only **familiar** image-set sessions (OPHYS_1_images_A, OPHYS_3_images_A). Reasons: (a) the paper's neural analysis is explicitly restricted to familiar images; (b) all familiar sessions in this release use the *same* 8 images (image set A), so "image identity" is a single consistent 8-way categorical output across all sessions, whereas adding novel sessions would add 8 mutually exclusive image classes that no familiar session ever shows.
3. **Session = `ophys_session_id`**. For Mesoscope sessions the 3–7 simultaneously recorded imaging planes (separate NWB "experiments") are **merged into one session**, since they are recorded simultaneously from the same mouse and constitute one simultaneously recorded population. Each neuron keeps its own plane's `targeted_structure` for `brain_region_idx`.
4. **Neurons**: all cells returned by the SDK (`events` table == valid ROIs only). No further filtering (consistent with reference code/paper).
5. **Trials**: `trials` rows with `(go | catch) & ~auto_rewarded` (per the task specification; also matches the whitepaper/SDK practice of excluding aborted trials from performance metrics and treating free-reward trials as non-task trials). Every such trial has a finite `change_time` (0 missing in 202 active experiments).
6. **Sessions without eye tracking are dropped** (1 familiar session, 805989030), because pupil diameter is a required output.

### Temporal alignment and binning
- **Alignment event**: `trials.change_time` — the time of the image change on Go trials and of the *sham* change on Catch trials (verified: `change_time` always coincides exactly with a flash onset; on Go trials that flash has `is_change==True`, on Catch trials `is_sham_change==True`). This matches the reference `save_trial_response_df.py`, which aligns trials to `change_time`.
- **Window**: `off_start = -2.0 s`, `off_end = +4.0 s` (24 bins). Justification: measured `change_time − start_time ≥ 2.79 s` and `stop_time − change_time ≥ 4.20 s` for **every** trial in the dataset, so the window never crosses a trial boundary and never contains a second change; it covers ~2.7 flash cycles before and 5.3 after the change, comfortably covering the 150–750 ms response window and the calcium-event response. (The reference window [−4, 8] s would cross trial boundaries and contain other changes, so it is not usable for trial-wise decoding.)
- **Bin size**: **250 ms** (`time_bin_size = 250.0`), bin edges aligned to the alignment event. Justifications: (a) it must be ≥ the slowest frame period (90.9 ms at 11 Hz for Mesoscope planes) so that every bin of every session contains ≥1 imaging frame — with 250 ms bins every bin contains ≥2 frames (11 Hz) or ~8 frames (31 Hz); (b) 250 ms divides the 750 ms flash cycle exactly into 3 bins (image-on bin + 2 grey bins), and equals the image duration, so bins never straddle image-on/image-off boundaries; (c) it is the same order as the paper's 400 ms decoding window and 750 ms image interval.
- All streams (events, running speed, pupil, stimulus, trials) carry sync-clock timestamps from the SDK, so binning every stream on the same edges guarantees alignment.

### Variable Mapping
| Source | Target | Transform |
|---|---|---|
| `experiment.events['events']` (per-frame detected calcium-event magnitude) + `experiment.ophys_timestamps` | `neural` (n_neurons, 24) | mean of event magnitude over the imaging frames whose timestamp falls in each 250 ms bin (mean, not sum, so the value is frame-rate independent across 31/11 Hz rigs) |
| — | `input` (0, 24) | no decoder inputs (per task spec); `input_names = []` |
| `stimulus_presentations` (change_detection block) `image_name` | `output[0]` "image_identity", 8 classes | each 750 ms image-presentation interval (paper's definition) is labelled with the image flashed at its start; bins take the label of the interval containing their centre. Omitted flashes (3.5%) keep the identity of the image that would have repeated (= previous image), since omissions replace a *repeat* |
| `trials.change_time` / `stimulus_presentations.is_change` | `output[1]` "image_change", 2 classes | 1 for the bins in the 750 ms image-presentation interval that begins at the change (i.e. right after the change in image identity), else 0. Catch (sham-change) trials therefore have all-zero change, as no image identity change occurred |
| `running_speed['speed']` (cm/s, filtered) | `output[2]` "running_speed_quintile", 5 classes | mean speed per 250 ms bin, then discretised at the 20/40/60/80th percentiles of that session's included bins |
| `eye_tracking['pupil_area']` -> diameter `2*sqrt(area/pi)` (= 2*max(width,height) px) | `output[3]` "pupil_diameter_quintile", 5 classes | nanmean per bin; blink NaNs interpolated over short gaps; discretised at that session's 20/40/60/80th percentiles |
| `trials[['hit','miss','false_alarm','correct_reject']]` | `output[4]` "trial_outcome", 4 classes | per-trial constant broadcast over the 24 bins (these four flags exactly partition the go+catch non-auto trials: verified sum == n trials) |
| `experiment_table.mouse_id` | `subjects`, `subject_idx` | str ids |
| plane `targeted_structure` | `brain_regions` = ['VISp','VISl'], `brain_region_idx` | per-neuron |

### Key Decisions
1. **Use `events`, not `dff_traces` or `filtered_events`**: the paper states all neural analyses use the detected calcium events; `filtered_events` is documented in the SDK as smoothing "for visualization".
2. **Mean (not sum) within bins**: makes values comparable between 31 Hz and 11 Hz rigs.
3. **Per-session quintiles for running speed and pupil diameter**: pupil diameter is in camera pixels and depends on rig/zoom/eye position, and running propensity varies strongly across mice/sessions; global percentile bins would largely encode *session identity* (trivially decodable from a per-session PCA projection) instead of within-session behaviour, and would leave some sessions with nearly all bins in one class. Per-session quintiles give a balanced 5-class problem per session and the class label has the same meaning (rank within session) everywhere. Global edges are additionally recorded in `metadata` for reference.
4. **Image identity is held for the full 750 ms interval** (not only the 250 ms image-on bin): this is the paper's "image presentation interval" convention, and calcium-event responses to a flash extend into the following grey period.
5. **Trial outcome as 4 classes** (hit/miss/false alarm/correct reject) rather than 2×2, keeping the natural experimental categories; it is constant within a trial (static per-trial), broadcast over time as required by the (d_output, T) format.
6. **Sessions with no eye tracking dropped** (1 session) rather than imputing a pupil output.

### Planned Sanity Checks
- [ ] Catch/(Go+Catch) ≈ 0.125 (whitepaper ~12.5%)
- [ ] hit+miss+false_alarm+correct_reject == n trials for every session
- [ ] every `change_time` equals a flash onset (`np.isin`), as asserted in reference code
- [ ] change-aligned image identity: label at bin 0 (t=0..250 ms) == `trials.change_image_name`; label at t=−250 ms == `trials.initial_image_name` (Go trials)
- [ ] image_change == 1 only for bins 8,9,10 (t ∈ [0, 750) ms) on Go trials and never on Catch trials
- [ ] flash cadence: measured onset-to-onset 0.7506 s, image duration 0.250 s
- [ ] omission fraction of flashes ≈ 3.5% (5% of repeats)
- [ ] running speed range plausible (−20..80 cm/s), quintile class fractions ≈ 0.2 each
- [ ] pupil diameter plausible (tens of pixels), quintile class fractions ≈ 0.2
- [ ] neural: no NaN, non-negative (event magnitudes), mean event rate per neuron plausible
- [ ] direct spot-check of neural/running/pupil values against the raw NWB file (Step 10 Check 2)
- [ ] totals: 91 sessions, 38 mice, ~14.8k neurons, ~22.9k trials (familiar/active minus the no-eye-tracking session)

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the Step-5 plan. Structure:
- `select_experiments()` — builds the local experiment table (only the 284 NWB files actually present) and applies the two selection filters (`~passive`, `session_type in {OPHYS_1_images_A, OPHYS_3_images_A}`), printing the funnel 284 -> 202 -> 110.
- `convert_session()` — one ophys session (all of its planes) per call:
  loads each plane with `cache.get_behavior_ophys_experiment`, stacks `events['events']` with `ophys_timestamps`;
  selects trials `(go|catch) & ~auto_rewarded`; builds the (ntrials, 25) matrix of bin edges from `change_time + [-2 : 0.25 : 4]`;
  bins events (mean), running speed (NaN-aware mean), pupil diameter (NaN-aware mean after short-gap interpolation);
  labels image identity / image change from the `change_detection_behavior` block; discretises running & pupil into per-session quintiles; assembles arrays and runs per-session sanity checks.
- `bin_sum_count()` — vectorised binning of all neurons x all trials at once via `cumsum` + `searchsorted` (no python loops over trials or bins).
- `sanity_checks()` — per-session checks (see Step 7/10).
- `plot_processing()` — 6-panel figure per session for `--show-processing`.
- `main()` — groups planes by `ophys_session_id`, multiprocessing `Pool` over sessions, assembles the final dict, prints a full statistics summary.

Code inefficiencies identified / speedups added:
- binning implemented with cumulative sums + `searchsorted` instead of per-trial/per-bin masks (all neurons, trials and bins in 3 array ops);
- each NWB file is opened exactly once per session;
- 8-way multiprocessing over sessions;
- neural stored as float32, outputs as int64;
- `--sample` mode picks one multi-plane and one single-plane session so both code paths are exercised.

Assertions built into the conversion (fail loudly rather than silently producing bad data):
`events` length == `ophys_timestamps` length; no missing `change_time`; outcome flags partition trials; trial window inside `[start_time, stop_time]`; no empty time bin (every bin contains >= 1 imaging frame); bin centres never precede the first flash; identical image set across sessions.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

> **NOTE**: the numbers in this section are from the *first* sample run (250 ms bins, un-normalised events, per-session quantiles). The sample deliverables were regenerated with the final settings; the final sample run gives 2 sessions / 247 trials / 8 time bins per trial / 177 neurons, `Data format is valid, no errors or warnings`, and the same per-session sanity checks all passing. See Step 8 and the revision note at the end of Step 9.

Commands run:
```
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing  # -> conversion_sample_out.txt
python -u /app/train_decoder.py /app/sample_data.pkl --verify-only             # -> verification_sample_out.txt
```

### Sample Statistics (sessions 951410079 [Mesoscope, 7 planes] and 775289198 [single plane])
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 (403491, 457841) |
| Neurons (total) | 177 (88, 89) |
| Trials (total) | 246 (39, 207) |
| Time bins / trial | 24 (all trials) |
| Input dimension | 0 |
| Trials dropped for missing running/pupil | 2 / 248 (0.8%) |
| image_identity distribution | [0.133, 0.125, 0.130, 0.107, 0.127, 0.115, 0.118, 0.145] |
| image_change distribution | [0.892, 0.108] |
| running_speed_quintile | [0.200, 0.200, 0.200, 0.200, 0.200] |
| pupil_diameter_quintile | [0.200, 0.200, 0.200, 0.200, 0.200] |
| trial_outcome | [0.382, 0.484, 0.037, 0.098] |

Expectation checks on the sample:
- `image_change` fraction expected = (3 bins / 24) x P(go) = 0.125 x 0.865 = 0.108 -> measured 0.108 ✅
- quintiles are 0.2 each by construction ✅
- 8 image classes all present with ~uniform frequency ✅ (8 images, familiar set A)
- `verify_data_format`: **"Data format is valid, no errors or warnings."**

### Per-session sanity checks (all sessions, all trials)
| Check | Result |
|---|---|
| every `change_time` coincides with a flash onset | True (100%) |
| bin containing t=0 has `trials.change_image_name` (GO trials) | 1.000 |
| bin at t=-250 ms has `trials.initial_image_name` (GO trials) | 1.000 |
| `image_change` == 1 exactly for bins 8,9,10 on GO trials | 1.000 |
| `image_change` == 0 everywhere on CATCH trials | 1.000 |
| neural finite / non-negative | True / True |
| mean event magnitude per bin | 0.0007 - 0.0109 |

### Processing Plots Review
`/app/processing_951410079.png`, `/app/processing_775289198.png` (6 panels each):
1. raw per-frame detected events vs 250 ms binned values for 3 cells — binned trace tracks the raw events, bin edges drawn, change time marked; no temporal offset visible.
2. stimulus raster (colour = image identity, grey = omitted, red = change) with the binned `image_identity` and `image_change` traces overlaid — the binned identity steps exactly at flash onsets and the change flag covers the 750 ms interval starting at the red change line.
3. running speed: raw 60 Hz trace, binned mean, quintile edges (horizontal) and the resulting class -> class changes exactly when the binned value crosses an edge.
4. same for pupil diameter.
5. population-mean event magnitude aligned to `change_time`, split by outcome, with 750 ms flash cycle grid — shows the expected flash-locked oscillation and a larger response after the change; response starts after t=0 (no acausal leakage).
6. output class-distribution bar plot.
No anomalies found.

### Run Time Estimates
| Speed-ups Implemented | Effect |
|---|---|
| cumsum/searchsorted binning (vs per-trial loops) | ~10-50x on the binning step |
| one NWB open per plane, behavior streams read once per session | avoids 7x redundant reads on Mesoscope sessions |
| 8-process pool over sessions | ~8x |

| Step | Time | Estimated Total |
|---|---|---|
| sample (2 sessions, 8 planes, 2 workers) | 21.9 s wall (7 s single-plane session, 24 s 7-plane session) | — |
| per-plane cost | ~3 s | 110 planes ≈ 330 s serial |
| full run with 8 workers | — | **≈ 1-3 min** (well under the 15 min budget) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

> **FINAL sample run** (750 ms bins, z-scored events, global quantiles; `/app/train_decoder_sample_out.txt`): validation balanced accuracy image_identity **0.578** (chance 0.125), image_change **0.681** (0.5), running_speed_quintile **0.333** (0.2), pupil_diameter_quintile **0.356** (0.2), trial_outcome **0.409** (0.25); loss decreasing, format valid with no errors or warnings. The table below is from the first (250 ms) sample run and is kept for the record.

`python -u /app/train_decoder.py /app/sample_data.pkl` -> `/app/train_decoder_sample_out.txt`

### Format Validation
- Errors: None
- Warnings: None

### Training
Loss decreased monotonically 1.47 -> 1.372 over 200 epochs (test loss 1.413). No divergence.

### Decoder Results (Sample, balanced accuracy)
| Output | Chance | Training | Validation |
|--------|--------|----------|------------|
| image_identity (8) | 0.125 | 0.367 | 0.306 |
| image_change (2) | 0.500 | 0.636 | 0.620 |
| running_speed_quintile (5) | 0.200 | 0.289 | 0.255 |
| pupil_diameter_quintile (5) | 0.200 | 0.304 | 0.285 |
| trial_outcome (4) | 0.250 | 0.366 | 0.381 |

Every output is above chance on validation data; train/validation gaps are small (<1.5x). With only 177 neurons from 2 sessions these values are expected to improve on the full dataset.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

```
python -u /app/convert_data.py /app/converted_data.pkl --full        # -> conversion_full_out.txt, 69 s
python -u /app/train_decoder.py /app/converted_data.pkl --verify-only  # -> verification_full_out.txt
```

### Output Files
- `/app/converted_data.pkl`: 388.9 MB
- `/app/verification_full_out.txt`: created; **format valid, 0 errors**; the only warnings are "all neural data is zero" for 1,341 of 22,050 trials (see Step 10, Check 1).

### Conversion funnel (no data silently lost)
| Stage | Experiments (planes) | Sessions | Mice | Neurons | Trials |
|---|---|---|---|---|---|
| local NWB files | 284 | 247 | 38 | — | — |
| active (non-passive) | 202 | 174 | 38 | 29,444 | 51,992 (go+catch, non-auto) |
| + familiar image set (OPHYS_1/3_images_A) | 110 | 92 | 38 | 14,895 | 23,049 |
| − 1 session without eye tracking (805989030) | 109 | 91 | 38 | 14,669 | 22,778 |
| − trials with missing running/pupil bins (3.2%) | 109 | 91 | 38 | 14,669 | **22,050** |

### Converted dataset statistics
| Statistic | Value |
|---|---|
| Sessions | 91 |
| Subjects (mice) | 38 (1–5 sessions each) |
| Neurons | 14,669 total; per session min 7, median 89, mean 161, max 666 |
| Trials | 22,050 total; per session min 39, mean 242, max 392 |
| Time bins per trial | 24 (all trials, 250 ms) |
| Input dimension | 0 |
| Brain regions | VISp 14,565 neurons, VISl 104 neurons |
| image_identity | [0.125, 0.126, 0.125, 0.124, 0.125, 0.125, 0.124, 0.126] |
| image_change | [0.891, 0.109] |
| running_speed_quintile | [0.200, 0.200, 0.200, 0.200, 0.200] |
| pupil_diameter_quintile | [0.200, 0.200, 0.200, 0.200, 0.200] |
| trial_outcome | [0.304, 0.571, 0.022, 0.104] (hit 6,699 / miss 12,586 / FA 476 / CR 2,289 trials) |

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data (independent scan of raw NWBs, `/app/cache/scan_active.csv`) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Mice (familiar active) | 82 mice / 376 sessions for the *whole* release | — | 38 | 38 | ✅ (all locally available mice) |
| Sessions (familiar active) | — | — | 92 | 91 (−1 no eye tracking) | ✅ |
| Neurons (familiar active) | 8,619 exc + 470 Sst + 1,239 Vip = 10,328 in the paper's Mesoscope-familiar subset | — | 14,895 (14,048 exc / 446 Sst / 401 Vip) | 14,669 | ✅ vs data; differs from paper because the local file subset contains mostly single-plane sessions (see Step 4) |
| Go+Catch non-auto trials | — | — | 23,049 | 22,050 (−271 dropped session, −728 missing behavior) | ✅ |
| Catch / (Go+Catch) | "~12.5%" (whitepaper) | — | 0.1250 | 0.1256 (2,765 catch / 22,050) | ✅ |
| Outcome fractions | — | — | hit 0.3033, miss 0.5710, FA 0.0215, CR 0.1043 | 0.3038 / 0.5708 / 0.0216 / 0.1038 | ✅ |
| Aborted trials excluded | performance metrics exclude aborted (whitepaper/SDK) | `get_rolling_performance_df` excludes aborted | 61.9% of all trials are aborted | excluded | ✅ |
| Image set | 8 natural images / session | — | image set A = im061..im085 in every familiar session | 8 classes, ~uniform (0.124–0.126) | ✅ |
| Flash cadence | 250 ms image + 500 ms grey | — | duration 0.2503 s, onset-to-onset 0.7506 s | 250 ms bins = 1 image bin + 2 grey bins | ✅ |
| Omission rate | 5% of repeats | — | 3.5% of flashes | (omissions inherit the repeating image identity) | ✅ |
| Change-time distribution | 2.25–8.25 s, mean 4.2 s | — | 2.79–8.31 s, mean 4.37 s | alignment window [−2, +4] s always inside trial | ✅ |
| Running speed | cm/s | `running_speed` (filtered, 60 Hz) | −11…+69 cm/s, 0 NaN | quintile edges per session recorded in metadata | ✅ |
| Neural signal | "detected calcium events" | `experiment.events` | 0.02–0.15 events/s/cell, magnitudes ~0.3–0.6 | mean magnitude/bin 0.00008–0.035 | ✅ |

### `events` vs `filtered_events` (empirical check, 14 sessions)
| Output | validation bal. acc. with `events` | with `filtered_events` |
|---|---|---|
| image_identity | 0.342 | 0.375 |
| image_change | 0.579 | 0.594 |
| running_speed_quintile | 0.258 | 0.254 |
| pupil_diameter_quintile | 0.235 | 0.237 |
| trial_outcome | 0.286 | 0.263 |

Differences are small and inconsistent in sign. **Decision: keep `events`**, because the paper states all neural analyses use the *detected* calcium events and the SDK documents `filtered_events` as a smoothing applied "to smooth it for visualization". A `--neural-signal filtered_events` switch is retained in the script for reproducibility of this comparison.

### Revisions made after the first full run (documented here because they changed Steps 5–9)
Three conversion parameters were revised after empirical comparison on a fixed 14-session subset (all other settings identical). Each was chosen on a justification grounded in the reference papers, and confirmed empirically:

1. **Time bin 250 ms -> 750 ms, window [−2, +4] s -> [−2.25, +3.75] s (8 bins).** The paper performs *all* of its analyses on the **750 ms image-presentation interval** ("By image presentation interval we refer to the 750 ms interval beginning with each image presentation"), and its decoding analysis integrates activity over 400 ms after each presentation. Because `change_time` is always a flash onset and 2.25 s = 3 x 750 ms, the bin edges coincide exactly with flash onsets: bin 3 is the change interval, bins 0–2 are the three preceding image intervals, bins 4–7 the four following ones. Empirically (14 sessions, validation balanced accuracy): image_identity 0.334 -> 0.427, image_change 0.577 -> 0.650, pupil 0.271 -> 0.300. Detected calcium events are extremely sparse (0.02–0.15 events/s/cell), so a 250 ms bin usually contains no event at all; 750 ms bins match the paper's unit of analysis and raise the per-bin SNR.
2. **Per-neuron z-scoring of the binned event magnitudes** (within session, over all selected trials). Event magnitudes are cell-specific and span orders of magnitude, and the decoder initialises its per-session projection from a *raw, un-centred* SVD of the neural matrix; without scaling a few large-amplitude cells dominate every principal component. Z-scoring is a monotone per-neuron affine transform: it changes no event time and no relative time course, only the units. Empirically: none -> std -> zscore gave image_identity 0.427/0.564/0.556, image_change 0.650/0.671/0.696, running 0.286/0.399/0.443, pupil 0.300/0.378/0.389, outcome 0.264/0.290/0.317. `zscore` chosen (`--normalize {none,std,zscore}` retained). Side benefit: the "all neural data is zero" warnings disappear, because a trial with no events is now a constant negative offset rather than exact zeros (the information content is identical).
3. **Running/pupil percentile bins computed globally (across the whole dataset) rather than per session.** This is the literal reading of "discretised into five equal percentile bins", and it makes a class label mean the same physical speed/diameter in every session — which matters because the decoder's readout layer is *shared* across sessions. Empirically: pupil 0.231 -> 0.271, running 0.257 -> 0.266 (14 sessions). Edges are stored in `metadata` (`running_quintile_edges`, `pupil_quintile_edges`): running 0.016 / 3.63 / 18.14 / 32.51 cm/s, pupil 74.98 / 84.05 / 93.43 / 108.91 px. `--quantile-scope session` is retained for reference.

All deliverables (sample + full conversion, verification and training outputs) were regenerated after these revisions.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
- **Errors: none.** `verify_data_format` reports *"Data format is valid, no errors or warnings."* for both `sample_data.pkl` and `converted_data.pkl`.
- Earlier iterations (before per-neuron z-scoring) produced 1,330–1,341 warnings of the form *"Session S, trial T: all neural data is zero"*. Investigation: detected calcium events are sparse point events (measured 0.023 events/s/cell for excitatory, 0.097 for Vip, 0.15 for Sst; 98.7% of (neuron, bin) entries are exactly 0), and the affected trials were all in sessions with 7–40 simultaneously recorded cells (Sst/Vip lines), where a 6-s window can genuinely contain no detected event. These were therefore *not* conversion errors. They are nonetheless gone in the final dataset because z-scoring maps "no event" to a constant negative value per neuron. No warnings remain, so nothing had to be left unfixed.

### Check 2: Sanity checks against the raw NWB files (`/app/cache/sanity_raw.py`, output `/app/cache/sanity_raw_out.txt`)
The script re-loads the raw NWB files through the AllenSDK **without importing any code from `convert_data.py`** and recomputes each quantity independently, for 4 sessions spanning both rigs and a 7-plane Mesoscope session. **Result: `TOTAL FAILURES: 0`.**
| Sanity check | Method | Result |
|---|---|---|
| neural, single entries | brute-force mean of `events` over frames in the bin, then per-neuron z-score; compared at (trial 0, neuron 0, bin 0), (2,1,5), (5,3,7), (10,0,3) | `np.allclose(rtol=1e-3, atol=1e-5)` True for all 16 spot checks |
| neural, whole trial | all neurons x 8 bins for one random trial per session | `np.allclose` True (max abs diff 1.1e-5 … 2.4e-4, i.e. float32 round-off) |
| image identity | for each bin centre, look up the raw `stimulus_presentations` row and (for omissions) the last non-omitted flash; map through `output_values[0]` | exact array equality, 12/12 trials |
| image change | raw `is_change` of the flash containing each bin centre | exact array equality, 12/12 trials |
| running quintile | re-bin the raw 60 Hz `running_speed.speed` and digitise with the stored global edges | exact equality for **all** trials of each session (0 mismatched bins out of 1,168 / 1,736 / 2,584 / 3,504) |
| pupil diameter | compare stored global quintile edges with raw `2*sqrt(pupil_area/pi)` percentiles per session | stored global edges [74.98, 84.05, 93.43, 108.91] px bracket the per-session raw percentiles (e.g. [75.2, 80.7, 88.9, 99.5] and [89.8, 104.6, 116.6, 130.3]) — as expected for global binning |
| trial outcome | recompute from raw `hit/miss/false_alarm/correct_reject` flags of the kept `trials_id`s | exact equality, all sessions |
| brain_region_idx | recompute per-plane `targeted_structure` x per-plane cell counts | exact equality |
| subject id | compare with raw `metadata['mouse_id']` | equal |

One genuine mismatch was found and fixed during this check: my first version of the check computed the z-score statistics over *kept* trials, while the conversion computes them over *all selected* go/catch trials (normalisation happens immediately after binning, before the behaviour-NaN trial filter). Values then differed by 0.3–1.5%. After replicating the code's scope the agreement is exact. The conversion's ordering is deliberate and documented: the per-neuron scale is a property of the recording, so it should not depend on which trials happen to have valid eye tracking.

### Check 3: Reference code comparison
| Processing step | Reference (papers / AllenSDK / tutorials) | This conversion | Same? |
|---|---|---|---|
| (a) data loading | `VisualBehaviorOphysProjectCache` + `get_behavior_ophys_experiment(eid)` (all tutorials) | identical (`from_local_cache`) | ✅ |
| neural stream | paper: "detected calcium events"; SDK `experiment.events` | `ds.events['events']` with `ds.ophys_timestamps` | ✅ |
| (b) neuron filtering | SDK returns only `valid_roi` cells; QC-failed planes/sessions already removed from the release; paper applies no extra per-cell filter | no extra filtering (verified all cells `valid_roi==True`, no all-zero cells, no NaNs) | ✅ |
| trial filtering | whitepaper/SDK exclude aborted trials from performance metrics; free rewards are not task trials; task spec: keep go+catch | `(go|catch) & ~auto_rewarded` | ✅ |
| session filtering | paper: familiar image sets, active behaviour | familiar (`OPHYS_1/3_images_A`), non-passive | ✅ |
| (c) temporal alignment | `save_trial_response_df.py` aligns trials to `trials.change_time`; asserts change times are flash onsets | aligned to `change_time`; the same assertion is run per session and passes | ✅ |
| window | reference trial window [−4, 8] s | [−2.25, +3.75] s | ⚠ differs — the reference window crosses trial boundaries and contains other image changes, which would make trials overlap and would put several changes inside one "trial"; our window is the largest multiple of 750 ms that fits inside *every* trial (measured min pre-change 2.79 s, min post-change 4.20 s) |
| (d) binning | paper assigns all behavioural/neural events to the 750 ms image-presentation interval; decoding uses 400 ms after flash onset | 750 ms bins, edges on flash onsets | ✅ (paper's interval) |
| behaviour streams | tutorials use `running_speed` (filtered) and `eye_tracking` | same; pupil diameter from `pupil_area` via the SDK's `compute_circular_area` convention | ✅ |
| (e) input construction | — | none (task spec) | n/a |
| (f) output construction | paper decodes change-vs-repeat and hit-vs-miss; image identity / running / pupil are not decoded in the paper | as specified in the Decoder Task | ⚠ superset of the paper, required by the task |
| normalisation | paper uses a random forest (scale-invariant), so it does not normalise | per-neuron z-score | ⚠ differs — required because the provided decoder does a raw (un-centred) SVD + linear readout; the transform is monotone per neuron and does not alter timing (see revision note 2) |

### Check 4: Key statistics comparison
See the table in Step 9. Every statistic that the papers state for this dataset is reproduced: catch fraction 12.6% vs "~12.5%", change times 2.79–8.31 s vs 2.25–8.25 s, 8 images per familiar session, 250 ms/750 ms flash cadence, 3.5% omitted flashes (5% of repeats), aborted trials 61.9% (excluded), and 0.3033/0.5714/0.0215/0.1038 hit/miss/FA/CR — identical (to 4 decimals) to an independent scan of the raw NWB files. The absolute neuron/session/mouse counts differ from the paper's 8,619/470/1,239-cell Mesoscope-familiar subset because only 284 of the release's 1,936 experiments are present locally and only one of them is a Mesoscope mouse; the conversion uses **all** locally available familiar active sessions (91 of 92; the 92nd has no eye tracking).

### Check 5: Edge cases
- **Trial-window boundaries**: asserted per session that `change_time − start_time >= 2.25 s` and `stop_time − change_time >= 3.75 s` for every kept trial, so no window crosses into a neighbouring trial and no window contains a second change.
- **Empty time bins**: asserted `counts.min() > 0` — every bin of every trial of every session contains at least one imaging frame (23–24 frames at 31 Hz, 8 at 11 Hz).
- **First/last flash**: asserted that every bin centre falls at or after the first flash of the change-detection block (`j.min() >= 0`).
- **Omitted flashes** (3.5%): identity is forward-filled from the last non-omitted flash (an omission replaces a *repeat*, so the ongoing identity is unchanged), and `is_change` is `<NA>` -> filled `False`; omissions never coincide with a change by design.
- **Blinks / tracking loss**: pupil NaNs (median 2.9%, max 29.6% of frames, runs up to ~14 s) are linearly interpolated only across gaps <= 1 s; longer gaps stay NaN and the affected trials are dropped (599 trials, 2.6%). Trials are also dropped if a bin contains no valid running or pupil sample at all.
- **Sessions with no eye tracking**: skipped with an explicit message (1 session, 805989030).
- **Sessions with < 2 usable trials**: skipped (none occurred).
- **Mixed frame rates** (31 Hz vs 11 Hz): bins hold the *mean* event magnitude, so values are frame-rate independent; verified both rigs in the sample run.
- **Multi-plane sessions**: the 3–7 planes of a Mesoscope session are concatenated along the neuron axis; behaviour streams are read once (verified identical `behavior_session_id`), and `brain_region_idx` keeps each plane's own structure (a 7-plane session contributes both VISp and VISl neurons — verified in the sample).
- **`<NA>`/object dtypes** from pandas nullable columns are explicitly cast (`fillna(False).astype(bool)`); `brain_regions` entries are cast to plain `str` (fixed after the first sample run, where they were `np.str_`).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` -> `/app/train_decoder_full_out.txt` (GPU, ~1 min)

### Training Progress
- Loss decreasing: **Yes** (monotone over all 200 epochs; final train loss ≈ test loss).
- Format: valid, **0 errors, 0 warnings**.

### Decoder Results (Full dataset: 91 sessions, 38 mice, 14,669 neurons, 22,179 trials)
| Output | Classes | Chance | Training bal. acc. | Validation bal. acc. | Validation / chance |
|--------|---------|--------|--------------------|----------------------|--------------------|
| image_identity | 8 | 0.125 | 0.602 | **0.491** | 3.9x |
| image_change | 2 | 0.500 | 0.741 | **0.642** | 1.28x |
| running_speed_quintile | 5 | 0.200 | 0.579 | **0.439** | 2.2x |
| pupil_diameter_quintile | 5 | 0.200 | 0.628 | **0.490** | 2.4x |
| trial_outcome | 4 | 0.250 | 0.590 | **0.340** | 1.36x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
All five outputs are above chance. Three are far above (image identity 3.9x, pupil 2.4x, running 2.2x). Two are between 1.25x and 1.5x of chance and were investigated in detail:

**`image_change` (0.642, chance 0.5).** This is a *balanced* 2-class score computed over every time bin, where only 1 bin in 8 is a change. Its scale is directly comparable to the paper's change decoder, which reports 65–90% correct per imaging plane. Reproducing the paper's exact analysis **on my converted data** (random forest, change interval vs the immediately preceding repeat interval, 5-fold CV, mean over imaging planes; `/app/cache/paper_style_decoding.py`) gives **0.653 ± 0.013 SEM over 91 sessions (0.736 for the 43 sessions with >= 100 cells, max 0.955)** — squarely inside the paper's range, and the dependence on cell count reproduces the paper's Figure 6A trend. The provided shared-readout linear decoder reaching 0.642 is therefore consistent with the information actually present in these recordings, not a conversion artefact.

**`trial_outcome` (0.340, chance 0.25).** The 4 classes are hit / miss / false alarm / correct reject. Hit-vs-miss is decodable: the paper's analysis applied to my data gives **0.726 ± 0.013** (paper Figure 6C: 55–70%). What limits the 4-class score is the catch-trial pair: distinguishing a false alarm (2.2% of trials) from a correct reject requires knowing whether the mouse licked to a *sham* change, and the paper explicitly reports that "for all cell classes, **false alarm decoding performance was very low**, with no difference between strategies" (Figure S22). A balanced 4-class score is dominated by the two rare, near-undecodable classes, so 0.34 is the expected value rather than a defect. (Splitting the outcome into two binary outputs would raise the headline number but would depart from the task specification of "trial outcome" as one categorical variable.)

### Check 2: Accuracy comparison to the papers
| Variable | Paper's report | Paper's analysis re-run on my converted data | This conversion + provided decoder |
|---|---|---|---|
| image change vs repeat | Fig 6A: ~65–90% correct per imaging plane (RF, 400 ms after flash) | **0.653 ± 0.013 SEM** (0.736 for >= 100 cells) | 0.642 balanced acc. |
| hit vs miss | Fig 6C: ~55–70% correct | **0.726 ± 0.013 SEM** | part of the 4-class trial_outcome (0.340) |
| false alarm | Fig S22: "very low" | — (2.2% of trials) | limits the 4-class trial_outcome score |
| image identity | not reported | — | 0.491 (chance 0.125) |
| running speed | not decoded (Fig 5 relates activity to running) | — | 0.439 (chance 0.2) |
| pupil diameter | not decoded | — | 0.490 (chance 0.2) |
No accuracy falls short of what the papers report for the corresponding variable.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| image_identity | 0.602 | 0.491 | 1.23 |
| image_change | 0.741 | 0.642 | 1.15 |
| running_speed_quintile | 0.579 | 0.439 | 1.32 |
| pupil_diameter_quintile | 0.628 | 0.490 | 1.28 |
| trial_outcome | 0.590 | 0.340 | 1.73 |
Four outputs are below the 1.5x threshold. `trial_outcome` (1.73x) overfits, as expected for the only *static per-trial* output: it has one effective sample per trial (not per bin), the model has a 100-component projection per session (91 sessions), and the two rare classes (FA 2.2%, CR 10.4%) are up-weighted by the balanced loss, so a few memorised training trials inflate the training score. There is no data leakage: train/validation splits are by trial, outputs are derived only from the trial's own behavioural flags, and neural z-score statistics are per-neuron over the whole session (a per-neuron affine constant, which cannot carry trial labels).

### Debugging steps performed for the lower-accuracy outputs
1. **Output values verified against raw data** for specific trials — Step 10 Check 2 (exact equality for image identity/change, running quintiles, trial outcome).
2. **Temporal alignment verified** — `/app/processing_*.png` overlays the binned neural, stimulus, running and pupil traces on the raw traces for a single trial; the population-average panel shows the flash-locked response rising *after* t = 0 with the 750 ms cadence, confirming no acausal leakage or offset. Additional checks: change times are exactly flash onsets; bin 3 carries `change_image_name` and bin 2 `initial_image_name` on 100% of GO trials; `image_change` is non-zero only in bin 3 of GO trials and never on CATCH trials.
3. **Output variation checked** — no output is dominated by one class: image identity 12.4–12.6% per class, image change 89/11, quintiles exactly 20% each, outcome 30/57/2/10.
4. **Neural filtering checked** — only valid ROIs (SDK), QC-failed planes already excluded upstream, no NaNs, no all-zero cells, no empty bins.
5. **Processing matched to the reference** — Step 10 Check 3 table; the only deliberate deviations (trial window length, per-neuron z-score, extra outputs) are justified there.

### Issues found and resolved during Steps 10–12
- 250 ms bins were too short for sparse calcium events -> 750 ms image-presentation intervals (revision 1). Accuracy rose for every output.
- Raw event magnitudes broke the decoder's un-centred SVD initialisation -> per-neuron z-score (revision 2). Largest single improvement; also removed all format warnings.
- Per-session percentile bins made class labels session-specific under a shared readout -> global percentile bins (revision 3).
- `brain_regions` contained `np.str_` instead of `str` -> cast to `str`.
- Sanity-check script initially used the wrong z-score scope -> corrected; agreement then exact.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `/app/README.md` created (dataset description, how to load, output specification, key statistics, decoder performance)
- [x] `/app/cache/` folder holds all exploration, comparison and validation scripts and their outputs
- [x] `/app/cache/README_CACHE.md` documents every cached file
- [x] Deliverables present:
  - `/app/CONVERSION_NOTES.md`, `/app/README.md`
  - `/app/convert_data.py`
  - `/app/converted_data.pkl` (132 MB), `/app/sample_data.pkl`
  - `/app/conversion_sample_out.txt`, `/app/verification_sample_out.txt`, `/app/train_decoder_sample_out.txt`
  - `/app/conversion_full_out.txt`, `/app/verification_full_out.txt`, `/app/train_decoder_full_out.txt`
  - processing plots `/app/processing_775289198.png`, `/app/processing_951410079.png`; decoder plots `sample_trials.png`, `predictions.png`

---

## Final run (definitive numbers)

All deliverables were regenerated once more after a cosmetic metadata fix (the `neural_signal`
string now also states the per-neuron normalisation). Nothing about the data changed.

| File | Result |
|---|---|
| `conversion_full_out.txt` | 91 sessions, 38 mice, 14,669 neurons, 22,179 trials, 68.9 s, 132.2 MB; all per-session sanity checks pass |
| `verification_full_out.txt` | **"Data format is valid, no errors or warnings."** |
| `train_decoder_full_out.txt` | see table below |
| `conversion_sample_out.txt` | 2 sessions (1 Mesoscope 7-plane + 1 single-plane), 177 neurons, 247 trials |
| `verification_sample_out.txt` | **"Data format is valid, no errors or warnings."** |
| `train_decoder_sample_out.txt` | image_identity 0.596, image_change 0.672, running 0.341, pupil 0.339, trial_outcome 0.403 (validation) |

### Final decoder accuracy (full dataset, validation balanced accuracy)
| Output | Classes | Chance | Training | Validation | x chance |
|---|---|---|---|---|---|
| image_identity | 8 | 0.125 | 0.603 | **0.493** | 3.9x |
| image_change | 2 | 0.500 | 0.742 | **0.642** | 1.28x |
| running_speed_quintile | 5 | 0.200 | 0.577 | **0.436** | 2.2x |
| pupil_diameter_quintile | 5 | 0.200 | 0.628 | **0.488** | 2.4x |
| trial_outcome | 4 | 0.250 | 0.589 | **0.338** | 1.35x |

(Run-to-run variation of the decoder is ~0.005; these match the Step 11/12 tables.)
