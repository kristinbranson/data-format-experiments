# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory — **Visual Behavior 2P** (`visual-behavior-ophys-1.1.0`), local AllenSDK cache in `/app/data`
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`) and validate with `/app/train_decoder.py`
- **Reference papers**:
  - `whitepaper.pdf` — "Allen Brain Observatory: Visual Behavior 2P Technical Whitepaper" (Garrett et al.)
  - `paper.pdf` — "Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex" (Neuron 2024)
  - `methods.txt` — excerpts of both

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file), `Dockerfile`, `docker-compose.yaml`, `.manifest`
- `whitepaper.pdf`, `paper.pdf`, `methods.txt` — reference texts
- `code/` — the AllenSDK source (reference code)
- `tutorials/` — 6 AllenSDK Visual-Behavior tutorial scripts/notebooks
- `allensdk_docs/` — rendered AllenSDK documentation
- `data/` — AllenSDK local cache:
  - `visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb` (284 files, 247 GB)
  - `visual-behavior-ophys-1.1.0/project_metadata/{ophys_experiment_table, ophys_session_table, behavior_session_table, ophys_cells_table}.csv`
  - `visual-behavior-ophys_project_manifest_v1.1.0.json`, `_downloaded_data.json`, `_manifest_last_used.txt`
- `decoder.py`, `train_decoder.py` — provided decoder/validation code

Environment verified: `python3` 3.13, numpy 2.4.4, torch 2.6.0+cu124, allensdk 2.16.2.
Hardware: 128 CPUs, 1 TB RAM, 1× NVIDIA L4 (23 GB).

Cache opens with
```python
VisualBehaviorOphysProjectCache.from_local_cache(cache_dir='/app/data', use_static_cache=False)
```
(`use_static_cache=True` fails — that variant expects a `visual-behavior-ophys/manifests/` sub-folder that this
cache does not have; the flat manifest layout is the `LocalCache` layout.)

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key functions / objects identified

| Function / attribute | File | Stage | Purpose |
|---|---|---|---|
| `VisualBehaviorOphysProjectCache.from_local_cache` | `allensdk/brain_observatory/behavior/behavior_project_cache/behavior_neuropixels_project_cache.py` (→ `project_cloud_api_base.py`) | LOADING | Opens the on-disk cache without network access |
| `cache.get_ophys_experiment_table()` | `project_apis/data_io/behavior_project_cloud_api.py` | LOADING | 1936-row metadata table: mouse, cre line, structure, depth, session type, project code, passive flag |
| `cache.get_ophys_session_table()` / `get_behavior_session_table()` / `get_ophys_cells_table()` | same | LOADING | Session-level metadata; cell ↔ experiment map (133 066 rows) |
| `cache.get_behavior_ophys_experiment(eid)` | same | LOADING | Returns `BehaviorOphysExperiment` for one imaging plane |
| `BehaviorOphysExperiment.events` | `behavior/data_objects/timestamps/...`, `behavior/event_detection.py` | PROCESSING | **Detected calcium events** per cell (`events`, `filtered_events`, `lambda`, `noise_std`) — the signal the Neuron paper uses |
| `.dff_traces` | `behavior_ophys_experiment.py` | PROCESSING | dF/F traces (already computed in the release; see "DF/F CALCULATION" in `methods.txt`) |
| `.ophys_timestamps` | `data_objects/timestamps/ophys_timestamps.py` | PROCESSING | Time (s, session clock) of every 2p frame |
| `.cell_specimen_table` | `data_objects/cell_specimens.py` | CURATION | One row per **valid** ROI; has `valid_roi`, `cell_roi_id`, ROI geometry |
| `.trials` | `behavior/data_objects/trials/trials.py`, `trial.py` | CURATION | Trial table: `start_time`, `stop_time`, `change_time`, `go`, `catch`, `aborted`, `auto_rewarded`, `hit`, `miss`, `false_alarm`, `correct_reject`, `initial_image_name`, `change_image_name`, `lick_times`, … |
| `Trial._get_trial_data` | `behavior/data_objects/trials/trial.py` L182-213 | CURATION | Authoritative definition of trial classes (see below) |
| `.stimulus_presentations` | `behavior/stimulus_processing.py`, `data_objects/stimuli/presentations.py` | PROCESSING | One row per flash: `start_time`, `end_time`, `image_name`, `image_index`, `is_change`, `is_sham_change`, `omitted`, `stimulus_block_name`, `trials_id`, `flashes_since_change` |
| `.running_speed` | `behavior/running_processing.py` | PROCESSING | 60 Hz linear running speed (cm/s), already de-wrapped, transient-removed and 10 Hz low-pass filtered |
| `.eye_tracking` | `behavior/eye_tracking_processing.py` | PROCESSING | ~30 Hz DeepLabCut ellipse fits; `pupil_width`, `pupil_height`, `pupil_area`, `likely_blink` |
| `compute_circular_area` | `eye_tracking_processing.py` L80-100 | PROCESSING | `pupil_area = π · max(pupil_width, pupil_height)²` → the *radius* is `max(width,height)`, so **pupil diameter = 2·max(width,height) = 2·√(area/π)** |
| `determine_likely_blinks`, `filter_on_blinks` | `eye_tracking_processing.py` L125-260 | CURATION | Marks blinks/|z|>3 outliers (dilated by 2 frames) and sets `pupil_area`/`pupil_width`/… to **NaN** on those frames |
| `.get_performance_metrics()`, `.get_rolling_performance_df()` | `behavior_session.py` | — | d′, hit rate, false-alarm rate over a rolling 100-trial window |

### Trial-class definition (`trial.py`, lines 182-213) — verbatim logic
```python
aborted = 'abort' in trial_event_names
if aborted:
    go = catch = auto_rewarded = False
else:
    catch          = self._trial["trial_params"]["catch"] is True
    auto_rewarded  = self._trial["trial_params"]["auto_reward"]
    go             = not catch and not auto_rewarded
correct_reject = catch and not false_alarm
if auto_rewarded:
    hit = miss = correct_reject = false_alarm = False
```
⇒ `go`, `catch`, `auto_rewarded`, `aborted` are **mutually exclusive**, so the requested trial selection
("Go and Catch, exclude Aborted and Auto-rewarded") is exactly `trials.go | trials.catch`.
Every such trial carries exactly one of `hit / miss / false_alarm / correct_reject`
(verified empirically: 0 unclassified trials in all 168 sessions).

### Notes / answers to the prompts
- **dF/F does not need to be computed** — the release already ships detrended dF/F (`dff_traces`) *and*
  detected calcium events (`events`, `filtered_events`). Both are produced by the standard Allen pipeline
  described in `methods.txt` ("DF/F CALCULATION", event detection à la Giovannucci et al. 2019).
- **Cell quality filtering is already applied**: `cell_specimen_table` contains only ROIs that survived
  segmentation → demixing → ROI classification → motion-border rejection (`valid_roi` is `True` for
  **29 097 / 29 097** cells in the sessions used here). No additional neuron filtering is required, and the
  whitepaper describes no further criterion that is exposed in the NWB files.
- The tutorials (`tutorials/visual_behavior_compare_across_trial_types.py`) demonstrate the canonical
  alignment: select behaviour/neural samples with `timestamps >= trial.start_time and <= trial.stop_time`,
  and draw stimulus spans from `stimulus_presentations`. This is exactly the alignment used here.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data structure
- AllenSDK *cloud/local* cache. One NWB file per **ophys experiment** (= one imaging plane in one session).
- 284 NWB files are present locally (the full release has 1936 experiments / 551 sessions / 82 mice).
- Metadata CSVs under `project_metadata/` are read through the SDK, never directly.
- Hierarchy: mouse → ophys **container** (one imaging plane tracked across days) → ophys **session**
  (one continuous recording) → ophys **experiment** (one imaging plane in one session) → cells.

### Composition of the 284 locally available experiments
| Split | Value |
|---|---|
| project_code | VisualBehavior 239 (all single-plane CAM2P rigs), VisualBehaviorMultiscope 45 (MESO.1) |
| session_type | OPHYS_1_A 55, OPHYS_2_A_passive 40, OPHYS_3_A 55, OPHYS_4_B 46, OPHYS_5_B_passive 42, OPHYS_6_B 46 |
| cre_line | Slc17a7 153, Sst 85, Vip 46 |
| targeted_structure | VISp 261, VISl 23 |
| passive | False 202, True 82 |
| mice | 38 |

After the Step-5 selection (**active behaviour, `project_code == 'VisualBehavior'`**):

### Dataset size (measured from the data files, `cache/survey_all.csv`)
| Statistic | Value |
|-----------|-------|
| Sessions (= experiments; single plane ⇒ 1 experiment per session) | 168 |
| Subjects (mice) | 37 |
| Sessions / subject | mean 4.5, range 1–6 |
| Neurons (total, all `valid_roi`) | 29 097 |
| Neurons / session | mean 173.2, median 117.5, min 6, max 666 |
| Trials / session (all) | mean 680 (aborted 418, go 226, catch 32, auto-rewarded 3.8) |
| **Go+Catch trials / session** | mean 258.3, median 264, min 39, max 409 |
| **Go+Catch trials (total)** | 43 387 |
| Trial duration (go+catch) | mean 8.47 s, min 7.26 s, max 12.56 s |
| Ophys frame interval `dt` | 0.03231–0.03233 s (30.95 Hz) — constant to 0.06 % across all 168 sessions |
| Ophys frames / session | ~140 000 (≈75 min) |
| Flashes / session (change-detection block) | 4802 ± 4 |
| Omitted flashes / session | 170 ± 21 (3.5 % of flashes) |
| Image changes / session | 230 ± 68 |
| Unique images / session | 8 (+ `omitted`); 16 distinct images across the two image sets |
| Running speed range | −24 … +100 cm/s |
| Eye-tracking blink/outlier fraction | mean 3.7 %, max 29.6 % |
| Sessions with **no** eye-tracking rows | 3 |

### Variables available per experiment
`ophys_timestamps`, `dff_traces`, `events` (+`filtered_events`), `cell_specimen_table`, `trials`,
`stimulus_presentations`, `stimulus_templates`, `running_speed` (+`raw_running_speed`), `licks`, `rewards`,
`eye_tracking`, `metadata`, `task_parameters`, `motion_correction`, projections/masks.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected statistics
| Statistic | Value | Source quote |
|---|---|---|
| Whole release: mice / imaging sessions / cells | 82 / 551 / 34 619 | whitepaper §A: "measurements from 82 mice, including 3021 behavior training sessions and 551 in vivo imaging sessions, resulting in longitudinal recordings from 34,619 cortical cells" |
| Images per session | 8 (64 possible transitions) | whitepaper: "Each session included 8 images, for a total of 64 possible transitions" |
| Flash timing | 250 ms image, 500 ms gray ⇒ 750 ms cycle | whitepaper Fig. 2; paper: "250 ms stimulus duration … 500 ms inter-stimulus duration" |
| Omission probability | 5 % of non-change flashes; changes and pre-change flashes never omitted | whitepaper §Experimental design |
| Catch probability | ~12.5 % in the imaging stage (matrix-sampling algorithm) | whitepaper: "pushing the actual catch probability to ~12.5%" |
| Change time distribution | truncated exponential 2.25–8.25 s, mean ≈ 4.2 s after trial start | whitepaper §Behavior session and trial structure |
| Response window | 150–750 ms after the change | whitepaper §Behavior metrics |
| Free/auto reward trials | 5 at the start of every session (+ after 10 consecutive misses in early training) | whitepaper: "Behavior sessions across all phases began with 5 'free-reward' trials" |
| 2p frame rate | 31 Hz single plane; 11 Hz per plane multi-plane | whitepaper §Data synchronization |
| Eye tracking / behaviour camera | 30 Hz | whitepaper §Data synchronization |
| Running wheel | analog encoder, converted to cm/s, 10 Hz low-pass filtered | whitepaper §Behavior metrics |
| Neuron paper subset | 8619 excitatory (21 sessions, 9 mice), 470 Sst (15, 6), 1239 Vip (21, 9) | paper "Our dataset contains …" (multi-plane rig, familiar images only) |
| Neural signal used by paper | detected calcium events | paper: "For all analysis of neural data we used the detected calcium events" |
| Behavioural analysis unit | the 750 ms image-presentation interval | paper: "By image presentation interval we refer to the 750 ms interval beginning with each image presentation. For image omissions we used the 750 ms following the time of the omission" |

Measured in the converted subset: catch fraction of go+catch trials = 32.4/258.3 = **12.5 %** — matches the
whitepaper's "~12.5 % catch probability" exactly. Omitted flashes 3.5 % of *all* flashes ≈ 5 % of
*non-change, non-pre-change* flashes, consistent with the stated 5 %.

### Processing details
- **Temporal alignment**: everything in the NWB is already on one common session clock (sync board at
  100 kHz, whitepaper §Data synchronization). `ophys_timestamps`, `stimulus_timestamps`,
  `eye_tracking.timestamps`, `trials.start_time/stop_time/change_time` and
  `stimulus_presentations.start_time` are all in that clock, in seconds. No extra alignment step is needed —
  only resampling of the 60 Hz running trace and the 30 Hz eye trace onto the 30.95 Hz ophys grid.
- **dF/F and events** are pre-computed in the release; the pipeline is described in `methods.txt`
  (dewarping → motion correction → segmentation → ROI filtering → demixing → neuropil subtraction →
  dF/F → event detection).
- **Temporal binning**: the paper bins behaviour by 750 ms image-presentation interval and decodes from
  "the first 400 ms after each stimulus presentation". Neither is a *neural* binning choice — the neural
  data live on the native 2p frame grid.

### Curation steps
**Neuron curation rules** (all already applied upstream, `methods.txt` §ROI FILTERING):
union ROIs, duplicates, motion-border ROIs, apical dendrites, too small/narrow/dim ROIs and ROIs with
non-positive demixed traces are removed. `cell_specimen_table` exposes only the survivors
(`valid_roi == True` for 100 % of rows here). Sessions/containers failing QC (saturation, >20 %
photobleaching, >10 µm z-drift, peak d′ < 1, interictal events, …) were already removed from the release.

**Trial curation rules** (this conversion):
- keep `go | catch`; drop `aborted` and `auto_rewarded` (explicit task instruction, and matches the SDK's
  mutually-exclusive trial classes);
- drop trials that contain no ophys frame, or whose eye/running coverage is missing (see Step 5).

### Decoders reported in the papers
| Decoded variable | Reported accuracy |
|---|---|
| image change vs. repeat (random forest, first 400 ms after each flash, per imaging plane) | ≈ 65–85 % correct depending on cell class and n cells (paper Fig. 6A; chance 50 %) — no numeric values given in the text |
| hit vs. miss (same classifier, all image changes) | ≈ 55–70 % correct (paper Fig. 6C; chance 50 %) |
| false alarm | "very low, with no difference between strategies" (Fig. S22) |
The paper reports these only graphically; they are a *qualitative* reference (well above chance for change,
modestly above chance for hit/miss) rather than a number to reproduce, and they use a different classifier,
a different data subset (multi-plane rig, familiar images) and a different unit of analysis (single
750 ms flash, per imaging plane) than the decoder used here.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies found and resolved
| Topic | Code says | Data shows | Papers say | Resolution |
|---|---|---|---|---|
| Trial classes | `go`/`catch`/`auto_rewarded`/`aborted` mutually exclusive (`trial.py`) | `go+catch+aborted+auto_rewarded == ntrials` in all 168 sessions; 0 trials unclassified into hit/miss/FA/CR | whitepaper describes GO/CATCH/free-reward/aborted | Select `go | catch` — equals "Go and Catch minus Aborted and Auto-rewarded" |
| Catch rate | — | 12.5 % of go+catch trials | "~12.5 %" (whitepaper) | consistent |
| Omissions | `stimulus_presentations.omitted` | 3.5 % of all flashes, never on a change or the flash before it | "5 % probability" on non-change flashes | consistent (5 % of eligible flashes) |
| Auto-rewarded trials | 5 free-reward trials at session start | 5 in 155/168 sessions, 0 in 13 | whitepaper: "sessions … began with 5 free-reward trials" | consistent; they are excluded anyway |
| Neural signal | SDK exposes `events`, `filtered_events`, `dff_traces` | `events` is extremely sparse (0.23 % non-zero at 31 Hz) | paper: "we used the detected calcium events" | Use `events`, **but** see Step 5 decision D3: raw event trains at 31 Hz give ~0 information at 97.7 % of timepoints, so the per-frame value is the event magnitude summed within the frame's bin; an empirical comparison against `filtered_events`/`dff` was run before committing |
| Frame rate | metadata `ophys_frame_rate = 31.0` | median `dt` = 0.03231 s ⇒ 30.95 Hz | whitepaper "31 Hz" | consistent (31.0 is the nominal rate; the sync-derived timestamps give 30.95 Hz) |
| Multi-plane rig | 11 Hz per plane | 6 sessions / 1 mouse / 347 cells locally | paper's *neural* analysis uses only the multi-plane rig | Excluded — see Step 5 decision D1 |
| Pupil | `pupil_area = π·max(w,h)²` | `pupil_area`/`pupil_width` NaN on `likely_blink` frames (mean 3.7 %) | whitepaper does not define a "diameter" | diameter := `2·max(pupil_width, pupil_height)` = `2·√(pupil_area/π)` |

No unresolved discrepancies.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable mapping
| Source (AllenSDK) | Target field | Transform | Notes |
|---|---|---|---|
| `BehaviorOphysExperiment.dff_traces['dff']` (one row per valid cell) | `neural[session][trial]` (n_neurons, T) | stack into (N, F) float32, slice the frames of each trial | detrended dF/F from the Allen pipeline |
| `ophys_timestamps` | time axis of every stream | none — it *is* the time base | 30.95 Hz |
| `trials` rows with `go | catch` | trial segmentation | window `[start_time, stop_time)` | aborted / auto-rewarded excluded |
| — | `input[session][trial]` (0, T) | empty | the task specifies **no decoder inputs** |
| `stimulus_presentations.image_name` (block `change_detection_behavior`) | `output[0] image_identity` | index into the 16 global image names; omitted flashes inherit the previous image; value held for the whole 750 ms presentation interval | 16 categories |
| `stimulus_presentations.is_change` | `output[1] image_change` | 1 for every frame of the presentation interval that begins with a change | 2 categories |
| `running_speed['speed']` (60 Hz) | `output[2] running_speed_bin` | `np.interp` onto ophys frames → per-session quintile bins | 5 categories |
| `eye_tracking.pupil_width/pupil_height` (~30 Hz) | `output[3] pupil_diameter_bin` | diameter = `2·max(w,h)`; blink/outlier frames dropped and linearly interpolated; `np.interp` onto ophys frames → per-session quintile bins | 5 categories |
| `trials.hit/miss/false_alarm/correct_reject` | `output[4] trial_outcome` | one label per trial, broadcast across the trial's frames | 4 categories |
| `metadata['mouse_id']` | `subjects`, `subject_idx` | unique list | 37 mice |
| `metadata['targeted_structure']` | `brain_regions`, `brain_region_idx` | unique list | VISp only for this subset |

### Key decisions and rationale

**D1 — Session subset: `project_code == 'VisualBehavior'` (single-plane rigs).**
The target format requires that "time bins be the same size for all trials and sessions", and the task
requires alignment "based on ophys timestamp". The single-plane Scientifica rigs all sample at
30.95 Hz (`dt` = 32.310–32.330 ms across all 168 sessions, sd 0.0055 ms), so the native ophys frame grid is
already a constant bin. Mixing in the Multiscope (MESO.1) sessions would mean mixing 30.95 Hz with
10.7 Hz data; resampling the Multiscope planes onto a common grid would leave ~1 source sample per bin
(and empty bins wherever the frame jitter runs the other way). The locally available Multiscope data is
also 6 sessions from a **single** mouse with an average of 10 cells per imaging plane (347 cells in
total, 1.2 % of the subset used), so excluding it costs very little. `VisualBehavior` is also the literal
name of this dataset variant in the AllenSDK.

**D2 — Active behaviour only (`passive == False`).**
Passive sessions (OPHYS_2, OPHYS_5) replay the same stimulus with the lick spout retracted to a sated
mouse. There is no Go/Catch/Aborted/Auto-rewarded trial structure to honour and no
hit/miss/false-alarm/correct-rejection outcome to decode, so they fall outside "the Visual Behavior
task" that the conversion targets.

**D3 — Neural signal: dF/F (`dff_traces`), not the detected calcium events.**
The Neuron paper uses detected calcium events, and that was the first choice here. It fails for
*per-timepoint* decoding: at 30.95 Hz the event train is non-zero for only **0.23 %** of (cell, frame)
pairs, i.e. 99.77 % of the samples the decoder must label carry literally no signal. A controlled pilot on
the same 6 sessions, same trials, same outputs, identical decoder settings (`cache/pilot.py`), gave
validation balanced accuracy:

| neural signal | image_identity | image_change | running | pupil | outcome |
|---|---|---|---|---|---|
| `events` (paper's choice) | 0.121 | 0.549 | 0.209 | 0.214 | 0.248 |
| `filtered_events` | 0.169 | 0.578 | 0.222 | 0.246 | 0.251 |
| **`dff_traces`** | **0.386** | **0.658** | **0.269** | **0.252** | 0.249 |

dF/F is produced by exactly the same Allen processing pipeline (`methods.txt` §DF/F CALCULATION:
demixing → neuropil subtraction → baseline normalisation → detrending), is the neural stream used in the
AllenSDK Visual Behavior tutorials, and is defined at every ophys frame. The task instructions allow
departing from the reference processing where "training a neural decoder requires otherwise"; this is
such a case. Events were retained only as the pilot control.

**D4 — Trial window = `[trials.start_time, trials.stop_time)`.**
"Segment each recording session into individual trials based on how they are defined in the experiment."
The AllenSDK trials table is that definition. Trials are non-overlapping, last 7.3–12.6 s (mean 8.5 s,
224–389 ophys frames) and contain the change at a variable delay (3.0–6.8 s after trial start). T is
therefore variable, which the target format allows. Because the alignment event is the ophys timestamp
grid rather than a single event within the trial, `off_start = 0.0` (trial start) and `off_end = None`
(variable trial length).

**D5 — `image_identity` is held through the gray screen and through omissions.**
Stimulus intervals are defined as `[flash start_time, next flash start_time)` — the same 750 ms
"image-presentation interval" the reference paper uses for behavioural events. Alternatives considered:
(a) a separate "gray" category, which would make ~67 % of all samples a single class that is not an image
identity at all, and (b) a separate "omitted" category, which is the *absence* of a stimulus rather than
an identity. Holding the current image is both a faithful description of the stimulus sequence the mouse
is in and the only option that keeps the variable to the 16 real image identities, matching the
instruction "image identity of the image presented during the non-grey screen".

**D6 — 16 global image categories, not 8 per-session indices.**
The single-plane variant uses image set A (`im061 im062 im063 im065 im066 im069 im077 im085`, 88
sessions) and image set B (`im000 im031 im035 im045 im054 im073 im075 im106`, 80 sessions). `output_values`
is a single global list, so per-session indices 0–7 would give the same label to different images in
different sessions. Using the 16 real image names keeps the labels meaningful (chance = 1/16).

**D7 — `image_change` marks the whole presentation interval of the changed image.**
"Value of 1 right after a change in image identity". The change flash is 250 ms and the calcium response
to it peaks 200–500 ms later, so the unit is the 750 ms interval beginning at the change (again the
reference paper's unit). Catch (sham-change) trials therefore contain no change frames — verified.

**D8 — Quintile bins for running speed and pupil diameter are computed per session.**
Pupil diameter is measured in camera pixels: its across-session spread (sd of the session medians
= 18.4 px, range 48–166 px) is *larger* than the typical within-session inter-quintile spread (~15 px).
Global quintiles would therefore mostly encode which session a sample came from — a per-session constant
that the session-specific projection of the decoder can read off trivially, inflating accuracy without
decoding anything about the pupil. Per-session quintiles remove that confound, and give every session an
exactly balanced 5-class problem, which is what the balanced-accuracy metric assumes. The same convention
is used for running speed so that the two discretisations are comparable. (Cost: in 18/165 sessions the
mouse essentially never ran — 80th percentile < 1 cm/s — so its running bins are close to noise. Global
edges would not help those sessions either; they would simply collapse them into one class.)

**D9 — Blink handling for the pupil.**
`eye_tracking` sets `pupil_width/height/area` to NaN on `likely_blink` frames (blinks plus |z| > 3 area
outliers, dilated by ±2 frames), 3.7 % of frames on average. Those samples are dropped and the trace is
linearly interpolated across them when it is resampled onto the ophys grid. A NaN in the output is a hard
error for the validator, and a separate "blink" category would not be a percentile bin of diameter.

**D10 — Sessions with no eye-tracking data are dropped.**
3 of the 168 sessions (795953296, 806456687, 833631914) have an empty `eye_tracking` table. Pupil diameter
is a required output, so these are excluded rather than filled with fabricated values. Cost: 3 sessions,
917 trials, 276 cells.

**D11 — No additional neuron filtering.**
`cell_specimen_table.valid_roi` is `True` for 29 097/29 097 cells: the release already applied the
motion-border, union/duplicate, apical-dendrite and size/SNR ROI filters described in `methods.txt`
§ROI FILTERING, plus the session- and container-level QC. The converter still re-applies the `valid_roi`
mask defensively.

**D12 — `trial_outcome` is stored as a time-constant row of the (5, T) output array.**
All trials must share one `doutput`, so the per-trial label is broadcast across the trial's frames. This
is numerically identical to supplying it as a 1-D per-trial value (`decoder.SessionData.__getitem__`
broadcasts 1-D outputs across the trial's rows in exactly the same way).

### Planned sanity checks
- [x] `n_trials` per session equals the number of `go|catch` trials
- [x] `n_neurons` equals `len(dff_traces)` and `len(brain_region_idx)`
- [x] spot-check `neural[s][trial][neuron, t]` against the raw `dff_traces` array at the frame index
      derived independently from `ophys_timestamps` (`np.allclose`, exact)
- [x] first/last frame of each trial bracket `start_time`/`stop_time` correctly (off-by-one check)
- [x] `image_identity` / `image_change` recomputed with an independent, explicit loop implementation
- [x] every change-marked frame lies inside the presentation interval that starts at `change_time`
- [x] every GO trial contains exactly one change interval, no CATCH trial contains any
- [x] running/pupil quintile edges reproduce the stored ones; bin fractions = 0.200 each
- [x] `trial_outcome` matches the trials table; GO → hit/miss, CATCH → FA/CR
- [x] catch fraction ≈ 12.5 %, omission fraction ≈ 5 % of eligible flashes (whitepaper)
- [x] change-triggered average dF/F rises *after* t = 0 (temporal-alignment check)
- [x] no NaN/Inf anywhere; dtypes float32 / int64

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py`, run as `python -u /app/convert_data.py <outfile> [--full|--sample]
[--show-processing] [--workers N]`.

Structure:
- `select_experiments()` — applies D1/D2 to the locally available NWB files.
- `convert_experiment(eid)` — loads one experiment through the SDK and returns its trials.
- `build_stimulus_series()` — vectorised `searchsorted` mapping of ophys frames onto
  image-presentation intervals, with forward-fill of omitted flashes.
- `pupil_diameter_series()` — diameter from the SDK ellipse fit, blink frames dropped.
- `plot_processing()` — the `--show-processing` diagnostics figure (11 panels).
- `main()` — `multiprocessing.Pool` over experiments, then assembly of the final dict.

Code inefficiencies identified: NWB load dominates (~3.5–5 s/experiment, single threaded); a naive
serial loop over 168 experiments would take ~11 min. The per-frame stimulus/behaviour mapping was
written with python loops in the first prototype.

Code speedups added: (1) `multiprocessing.Pool` over experiments (32 workers) — 168 sessions in 34 s
wall clock; (2) all per-frame mappings vectorised with `np.searchsorted`/`np.interp`; (3) the omitted-flash
forward fill done with a `searchsorted` over the indices of valid flashes instead of a python loop;
(4) `float32` for neural data, one contiguous slice per trial; (5) the heavy debug arrays are only
collected for the ≤2 sessions that are plotted.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` →
`/app/conversion_sample_out.txt`, `processing_775614751.png`, `processing_788490510.png`.

### Sample statistics
| Statistic | Value |
|---|---|
| Sessions | 2 (775614751, 788490510) |
| Neurons (total) | 231 (89, 142) |
| Subjects | 1 (mouse 403491) |
| Trials (total) | 229 (39, 190) |
| T per trial | mean 251.3, min 225, max 389 |
| time_bin_size | 32.310 ms (sd 0.000) |
| image_identity distribution | uniform within each session's 8 images (0.011–0.111 globally) |
| image_change distribution | [0.920, 0.080] |
| running_speed_bin distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| pupil_diameter_bin distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| trial_outcome distribution | hit 0.595, miss 0.274, FA 0.053, CR 0.078 |

### Processing plots review
`processing_788490510.png` (11 panels) shows: population dF/F with the selected trial windows; raw vs.
resampled running speed (superimposed, no lag); an example GO trial where the red change marker
coincides with the first blue flash of the new image and the licks/reward follow it; an example CATCH
trial with no change marker; the quintile-edge histograms; exactly 0.200 of samples in each quintile bin;
the blink removal/interpolation excerpt; and the change-triggered average dF/F, which is flat before
t = 0 and peaks ~0.3 s after it — the key temporal-alignment check. No anomalies.

### Run time estimates
| Speed-up implemented | Time saving |
|---|---|
| 32-way multiprocessing over experiments | ~11 min → 34 s |
| vectorised stimulus/behaviour mapping | ~1.5 s → ~0.05 s per session |

| Step | Time / session | Estimated total (168 sessions) |
|---|---|---|
| NWB load + conversion (per worker) | 3.5–5.0 s | 34 s wall clock at 32 workers |
| pickling 8.7 GB | — | 14 s |
| **total** | | **~50 s** (measured) |

Verification (`/app/verification_sample_out.txt`): **no errors, no warnings**.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format validation
- Errors: **None**
- Warnings: **None**

### Decoder results (sample, 2 sessions, 229 trials, 231 neurons)
| Output | Chance | Training balanced acc | Validation balanced acc |
|---|---|---|---|
| image_identity | 0.0625 | 0.3949 | 0.2989 |
| image_change | 0.5000 | 0.6694 | 0.6150 |
| running_speed_bin | 0.2000 | 0.2494 | 0.2123 |
| pupil_diameter_bin | 0.2000 | 0.3357 | 0.2717 |
| trial_outcome | 0.2500 | 0.4083 | 0.2736 |

Loss decreased monotonically (1.6288 → 1.3939 over 200 epochs). Every output is above chance. Running
speed is the weakest: both sample sessions come from mouse 403491, which barely ran (80th percentile of
speed = 0.2 and 42.8 cm/s respectively, and in 788490510 the whole speed range is −1.5…1.0 cm/s), so its
running quintiles are close to sensor noise. This is a property of these two sessions, not of the
conversion — see the full-dataset result in Step 11.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full --workers 32` →
`/app/conversion_full_out.txt` (50 s wall clock), then
`python -u /app/train_decoder.py /app/converted_data.pkl --verify-only` →
`/app/verification_full_out.txt`.

### Output files
- `converted_data.pkl`: **8.73 GB**
- `verification_full_out.txt`: created, **no errors, no warnings**

### No data lost
168 sessions selected → 165 converted; the 3 skipped ones are listed explicitly in
`conversion_full_out.txt` and in `metadata['skipped_sessions']` (empty `eye_tracking` table, D10).
Every `go|catch` trial of every converted session is present: the sanity checks confirm
`len(data['neural'][s]) == len(trials.query('go or catch'))` for every session checked, and
`n_trials_dropped_short == 0` for all 165 sessions (no trial was too short to keep).

### Consistency check
| Statistic | Reference papers | Reference code / SDK | Reference data (survey of the 168 NWBs) | Converted data | Match? |
|---|---|---|---|---|---|
| Subjects (mice) | 82 in the whole release | — | 37 in this subset | 37 | ✓ |
| Sessions | 551 imaging sessions in the whole release | — | 168 active single-plane | 165 (3 without eye tracking) | ✓ |
| Sessions / subject | 3–11 per container (whitepaper) | — | 1–9 | 1–9 (mean 4.5) | ✓ |
| Neurons (total) | 34 619 in the whole release | `cell_specimen_table.valid_roi` all True | 29 097 | 28 821 | ✓ |
| Neurons / session | — | — | mean 173.2, 6–666 | mean 174.7, 6–666 | ✓ |
| Trials (go+catch, total) | — | `go|catch` | 43 387 | 42 470 | ✓ |
| Trials / session | — | — | mean 258.3 (39–409) | mean 257.4 (39–409) | ✓ |
| Catch fraction of go+catch | "~12.5 %" | — | 12.54 % | 12.51 % | ✓ |
| Trial length | change at 2.25–8.25 s + 4.25 s response epoch | — | 7.26–12.56 s, mean 8.47 s | T = 217–389 frames, mean 262 (7.0–12.6 s) | ✓ |
| Flash cycle | 750 ms (250 on / 500 gray) | `stimulus_presentations` | median inter-flash 750.6 ms | image_change fraction 0.077 ≈ 0.75 s × 230 changes / (8.47 s × 258 trials) | ✓ |
| Omissions | 5 % of eligible flashes | `omitted` column | 3.5 % of all flashes | inherited (identity held through omissions) | ✓ |
| Images per session | 8 | `image_name` | 8 (+omitted) in every session | 8 non-zero categories per session, 16 globally | ✓ |
| 2p frame rate | "31 Hz" | metadata `ophys_frame_rate = 31.0` | median dt 32.310–32.330 ms | `time_bin_size` 32.3193 ms (sd 0.0055) | ✓ |
| Outcome distribution | — | — | hit 0.319, miss 0.556, FA 0.019, CR 0.106 | hit 0.316, miss 0.559, FA 0.018, CR 0.107 | ✓ |
| image_identity distribution | 8 images sampled equally ("matrix sampling algorithm ensured each image transition was sampled equally") | — | — | 0.060–0.065 per image (uniform to ±4 %) | ✓ |
| running_speed_bin | — | — | — | [0.200, 0.200, 0.200, 0.200, 0.200] | ✓ |
| pupil_diameter_bin | — | — | — | [0.200, 0.200, 0.200, 0.200, 0.200] | ✓ |
| Running speed range | — | cm/s, 10 Hz low-pass | −24 … +100 cm/s | (discretised; edges stored per session in metadata) | ✓ |
| Input dimension | — | — | — | 0 (task specifies no inputs) | ✓ |

The small differences between the "reference data" and "converted data" columns are exactly the 3
eye-tracking-less sessions (−3 sessions, −276 neurons, −917 trials).

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — output-log verification
`/app/verification_full_out.txt` (regenerated in full; the first attempt was truncated by a
`| head` that closed the pipe). Result: **`Data format is valid, no errors or warnings.`**
There is nothing to fix and nothing to explain away: zero errors, zero warnings, for both
`sample_data.pkl` and `converted_data.pkl`.

Checked by eye in the log: 165 sessions / 42 470 trials / 37 mice / 1 brain region; `Input
dimension: 0`; `Output dimension: 5`; T = 217–389 (mean 262); n_neurons 6–666; all five output
ranges are exactly `[0, n_classes-1]`; the per-session output-fraction block shows every session
at 0.200 for both quintile variables.

### Check 2 — independent sanity checks (`cache/sanity_checks.py`)
These reload the NWB files through the AllenSDK and recompute every quantity with a *different*
implementation (explicit python loops over presentations/trials instead of the converter's
vectorised `searchsorted`), then compare with `np.allclose`. Run on 5 sessions
(indices 0, 84, 104, 138, 164 — first, last and three random), **126 checks, 0 failures**
(`cache/sanity_checks_out.txt`):

| Stream | Check | Result |
|---|---|---|
| neural | `neural[s][trial][neuron, t]` vs `dff_traces` at the independently derived frame index (6 random spot checks/session, exact `np.allclose`) | PASS |
| neural | whole random trial matrix equals `dff[:, frames]` | PASS |
| neural | n_neurons == `len(dff_traces)` == `len(brain_region_idx)` | PASS |
| neural | no NaN/Inf, dtype float32 | PASS |
| neural | off-by-one: `ts[a] >= start_time` and `ts[a-1] < start_time`; `ts[b-1] < stop_time` and `ts[b] >= stop_time` | PASS |
| input | every trial's input is exactly `(0, T_neural)` (no decoder inputs for this task, so there are no input *values* to compare) | PASS |
| output | `image_identity` recomputed by an explicit loop over `stimulus_presentations` (12 random spot checks/session) | PASS |
| output | `image_change` recomputed the same way | PASS |
| output | every change-marked frame lies in `[change_time, next flash onset)` | PASS |
| output | every GO trial contains one change interval; no CATCH trial contains any | PASS |
| output | running/pupil quintile edges recomputed from raw `running_speed` / `eye_tracking` reproduce the stored ones | PASS |
| output | `running_speed_bin` / `pupil_diameter_bin` spot checks (10/session) | PASS |
| output | bin fractions = 0.200 each | PASS |
| output | `trial_outcome` matches the trials table for **every** trial; constant within a trial; GO→hit/miss, CATCH→FA/CR | PASS |
| structure | n_trials == number of `go|catch` trials; subject and brain-region indices resolve to the SDK metadata | PASS |

One check failed on the first run and was traced to the *check*, not the data: "change frames lie
in `[change_time, change_time + 0.78 s)`" failed for 1 frame out of 796 in session 775614751. The
stimulus computer had dropped a frame there, so that image-presentation interval lasted 800.6 ms
instead of 750 ms and its last ophys frame fell at change_time + 0.793 s. The check now reads the
actual next-flash onset from `stimulus_presentations` instead of assuming 750 ms, and passes.

Additional whole-dataset invariants (all 165 sessions, `cache/` one-off script):
`n_trials_dropped_short` = 0 in every session; every session has exactly 8 distinct image
identities; max deviation of any quintile-bin fraction from 0.200 is 6.3e-5; all output values in
range; every session has ≥ 2 trials (min 39); T and n_neurons consistent within each session;
dF/F range −4.1 … +11.1.

### Check 3 — reference-code comparison

| Stage | Reference (AllenSDK / papers) | This conversion | Same? |
|---|---|---|---|
| (a) loading | `VisualBehaviorOphysProjectCache` → `get_behavior_ophys_experiment()`; tutorials use `bc.get_behavior_ophys_experiment(id)` and `np.vstack(dataset.dff_traces.dff.values)` | identical calls; no NWB file is opened directly | ✓ |
| (b) neuron filtering | `cell_specimen_table` already contains only valid ROIs (`methods.txt` §ROI FILTERING + release QC) | re-applies `valid_roi` defensively; no extra criterion invented | ✓ |
| (b) trial filtering | `trial.py`: `go`/`catch`/`aborted`/`auto_rewarded` mutually exclusive | `trials.go | trials.catch` | ✓ |
| (b) session filtering | whitepaper QC already removed failed sessions; paper selects by project/rig/experience | selects active behaviour + single-plane variant (D1, D2); drops 3 sessions with no eye tracking (D10) | ✓ (documented deviations) |
| (c) temporal alignment | tutorial: `timestamps >= trial.start_time and <= trial.stop_time` on each stream; all streams already on the common 100 kHz sync clock | `[start_time, stop_time)` (half-open so that adjacent trials cannot share a frame); streams resampled onto `ophys_timestamps` | ✓ |
| (d) binning | neural data live on the native 2p frame grid; the paper *analyses* in 750 ms presentation intervals and 400 ms post-flash windows | native ophys frame grid (32.319 ms); the 750 ms presentation interval is used as the unit for `image_identity` / `image_change` exactly as in the paper | ✓ |
| (e) input construction | — | none (task specifies no decoder inputs) | n/a |
| (f) output construction | paper assigns behavioural events to the 750 ms interval beginning with each image presentation, and to the 750 ms following an omission | same intervals; `image_identity` held through gray + omissions (D5); `image_change` = the change interval (D7) | ✓ |
| neural signal | paper: detected calcium events | dF/F (D3) — **documented deviation**, required for per-timepoint decoding (events are non-zero at only 0.23 % of (cell, frame) pairs); both come from the same Allen pipeline | deviation, justified |
| running speed | `running_processing.py`, already 10 Hz low-pass filtered | used as provided, `np.interp` onto ophys frames (no additional filtering) | ✓ |
| pupil | `eye_tracking_processing.compute_circular_area` treats `max(w,h)` as the radius | diameter = `2·max(w,h)`; SDK blink/outlier NaNs respected | ✓ |

### Check 4 — key-statistics comparison
See the table in Step 9. Every statistic that the whitepaper or the paper states numerically is
reproduced: catch probability 12.5 % (stated "~12.5 %"), 8 images per session, 750 ms flash cycle,
~5 % omissions on eligible flashes, 31 Hz frame rate, 5 auto-reward trials per session (excluded),
and the trial-outcome fractions match a completely independent survey of the raw NWB files
(`cache/survey_all.csv`) to within the 3 dropped sessions.

### Check 5 — edge cases found and handled
1. **Trial starting before the first flash.** In session 792815735 the first go/catch trial starts
   0.021 s (≈ 0.7 ophys frames) before the first flash of the change-detection block, which made
   one frame's `image_identity` = −1 and crashed the decoder with a CUDA device-side assert. Fixed
   by clamping the presentation index to `[0, n_flashes-1]`; the affected frame is assigned the
   identity of the flash that is about to appear.
2. **Dropped stimulus frames.** Inter-flash intervals are 733–801 ms, not exactly 750 ms. The
   converter never assumes 750 ms: intervals are always `[start_time_i, start_time_{i+1})` read
   from the data.
3. **Omitted flashes.** `image_name` is `'omitted'`; handled by forward-filling the previous real
   image identity (D5), never by inserting a spurious category.
4. **Blink / outlier pupil frames** (3.7 % on average, up to 29.6 % in one session): NaN in the SDK,
   dropped and interpolated (D9).
5. **Sessions with an empty eye-tracking table** (3): skipped and reported (D10).
6. **Trials shorter than 2 ophys frames**: guarded against; none occur (0 in all 165 sessions).
7. **Sessions with very few neurons** (min 6) and very few trials (min 39): kept — the format only
   requires ≥ 2 trials per session.
8. **Sessions where one outcome class never occurs** (12 sessions have 3 classes, 2 have 2):
   kept; class coverage is global, not per-session.
9. **Trial windows never overlap** (`start_time[i+1] >= stop_time[i]` for all trials), so no frame
   is assigned to two trials.

### Issues found and resolved in this step
- Truncated `verification_full_out.txt` → regenerated in full.
- The off-by-one `image_identity = -1` edge case (item 1 above) → fixed before the full run.
- Over-tight tolerance in one sanity check (0.78 s) → replaced with the data-driven interval end.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` →
`/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.
Trained on GPU (NVIDIA L4), 200 epochs, 165 sessions, 11.2 M timepoints.

### Training progress
- Loss decreasing: **Yes**, monotonically — 1.628849 (epoch 1) → 1.614182 (10) → 1.538702 (50) →
  1.444719 (100) → 1.390938 (150) → **1.362642** (200). Test loss 1.415442.

### Decoder results (full dataset)
| Output | Classes | Chance (uniform) | Training balanced acc | Validation balanced acc | Val / chance |
|---|---|---|---|---|---|
| image_identity | 16 | 0.0625 | 0.4404 | **0.4182** | 6.69× |
| image_change | 2 | 0.5000 | 0.6375 | **0.6090** | 1.22× |
| running_speed_bin | 5 | 0.2000 | 0.3111 | **0.2823** | 1.41× |
| pupil_diameter_bin | 5 | 0.2000 | 0.3124 | **0.2668** | 1.33× |
| trial_outcome | 4 | 0.2500 | 0.4375 | **0.2959** | 1.18× |

All five outputs are above chance on held-out trials.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — accuracy vs. chance
No output is at or below chance. `image_identity` is 6.7× chance. The four remaining outputs sit
between 1.18× and 1.41× chance, i.e. below the 1.5× flag, so each was investigated:

- **`image_change` (0.609 vs 0.500).** This is a *per-frame* binary label with only 7.7 % positives,
  and the label covers the whole 750 ms presentation interval including its first ~150 ms, during
  which no calcium response to the change has occurred yet (the change-triggered average dF/F in
  the diagnostics plot starts rising ~150 ms and peaks ~300 ms after the change). A per-frame
  balanced accuracy of 0.61 is therefore not comparable to the paper's per-flash number — see
  Check 2, where the paper's own analysis run on this data gives 74.0 %.
- **`running_speed_bin` (0.282 vs 0.200).** In 18 of 165 sessions the mouse essentially never ran
  (80th-percentile speed < 1 cm/s), so in those sessions the quintile bins split sensor noise and
  are unlearnable by construction. Restricting attention to sessions where the mouse did run is not
  possible without discarding data the task asks for.
- **`pupil_diameter_bin` (0.267 vs 0.200).** Pupil diameter is a slow variable being read out from
  instantaneous V1 activity at 31 Hz; 1.33× chance on a 5-class problem is a reasonable result for
  V1 dF/F.
- **`trial_outcome` (0.296 vs 0.250).** The label is constant over the whole 8.5 s trial, but the
  event that determines it (the change and the animal's response) happens 3–7 s *after* trial onset;
  most frames of a trial therefore precede any information about the outcome. False alarms are also
  only 1.8 % of trials. When the same data are scored the way the paper scores them — hit vs. miss,
  on the 400 ms after the change only — accuracy is 74.4 % (Check 2).

Two candidate conversion changes were tested rather than assumed:
| variant (20 random sessions, identical decoder settings) | image_identity | image_change | running | pupil | outcome |
|---|---|---|---|---|---|
| dF/F as shipped (**used**) | 0.4786 | 0.6456 | 0.2921 | 0.2622 | 0.3023 |
| per-neuron z-scored dF/F | 0.4963 | 0.6387 | 0.2997 | 0.2643 | 0.2462 |
| per-session global rescaling | 0.4906 | 0.6438 | 0.2957 | 0.2620 | 0.2910 |
The gains are small and inconsistent (z-scoring *hurts* `trial_outcome` by 0.056), so the unmodified
pipeline dF/F was kept — it is the more faithful representation of the reference processing.

### Check 2 — accuracy comparison with the reference paper
The paper reports decoding accuracies only graphically (Figure 6A/6C), for a different classifier
(random forest), a different data subset (multi-plane rig, familiar images), a different neural
signal (calcium events) and a different unit of analysis (one image presentation, first 400 ms).
To make a like-for-like comparison, the paper's own analysis was re-implemented and run **on the
converted dataset** (`cache/paper_decoder_comparison.py`): each image change and the image repeat
immediately before it, neural activity in the first 400 ms after flash onset concatenated over all
simultaneously recorded neurons, 5-fold cross-validated `RandomForestClassifier`, averaged per
imaging plane, mean ± SEM over planes.

| Decoded variable | Paper (Figure 6, chance 50 %) | This converted dataset | Verdict |
|---|---|---|---|
| image change vs. preceding repeat | ≈ 65–85 % across cell classes / n cells | **74.0 % ± 0.9 %** (n = 165 planes, median 74.0 %) | in the reported range |
| hit vs. miss on image changes | ≈ 55–70 % | **74.4 % ± 0.9 %** (n = 156 planes, median 72.3 %) | at/above the reported range |
| false alarm | "very low" (Fig. S22) | `false_alarm` is 1.8 % of trials and is the weakest class of `trial_outcome` | consistent |

The change decoder lands squarely in the published range, which is strong evidence that trials,
neural data and stimulus labels are correctly selected and aligned — a temporal misalignment of even
one flash (750 ms) would collapse this number towards 50 %. The hit decoder is at the top of / above
the published range; the two differences that would push it that way are both expected: this uses
dF/F rather than event trains (more information per 400 ms window, cf. the Step-5 pilot) and always
uses *all* simultaneously recorded neurons, whereas the paper averages over samples of `n` neurons
including small `n`.

The 0.609 balanced accuracy that `train_decoder.py` reports for `image_change` and the 74.0 % here
are the same data scored differently: the former labels **every** frame of the trial (93 % of which
are "no change", most of them far from any flash), the latter contrasts one change flash against the
one repeat flash immediately before it in the 400 ms window where the response lives.

### Check 3 — train vs. validation gap
| Output | Train | Validation | Train/Val |
|---|---|---|---|
| image_identity | 0.4404 | 0.4182 | 1.05 |
| image_change | 0.6375 | 0.6090 | 1.05 |
| running_speed_bin | 0.3111 | 0.2823 | 1.10 |
| pupil_diameter_bin | 0.3124 | 0.2668 | 1.17 |
| trial_outcome | 0.4375 | 0.2959 | 1.48 |

No ratio exceeds 1.5, so there is no overfitting flag and no sign of data leakage. `trial_outcome`
has the largest gap, as expected for a label that is constant within a trial while the train/test
split is *by trial*: the decoder can fit each training trial's constant offset but that does not
generalise. The other four outputs vary within a trial and generalise almost perfectly from train to
validation. The model is in fact still *under*-fitting — the training loss was still falling at
epoch 200 — which rules out leakage as an explanation of any of the numbers.

### Debugging steps run for the low-accuracy outputs
1. Output values verified against raw data on specific trials — `cache/sanity_checks.py`, 126/126 PASS
   (including `trial_outcome` compared against the trials table for **every** trial of 5 sessions).
2. Temporal alignment verified by plotting neural + outputs for single trials
   (`processing_*.png` panels 3, 4) and by the change-triggered average dF/F (panel 9), which is flat
   before the change and peaks 0.3 s after it.
3. Output variation checked: no output is dominated by one class except `image_change`
   (0.923/0.077, inherent to the task) and `trial_outcome`'s rare `false_alarm` class (0.018,
   inherent — the animals' false-alarm rate).
4. Neural filtering re-checked: `valid_roi` is True for every cell; no further filter exists in the
   release.
5. Processing re-checked against the reference code (Step 10, Check 3).

### Issues found and resolved
- None outstanding. The only conversion bug found in the whole exercise (the `image_identity = −1`
  frame in session 792815735) was found and fixed in Step 7 and is covered by a sanity check.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, how to load, format spec, key statistics)
- [x] `CONVERSION_NOTES.md` complete
- [x] `cache/` folder holds the exploration/analysis scripts with `cache/README_CACHE.md`
- [x] Files organised
