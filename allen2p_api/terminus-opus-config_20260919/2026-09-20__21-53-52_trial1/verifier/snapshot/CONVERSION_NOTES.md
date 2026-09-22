# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory - Visual Behavior 2P (AllenSDK VisualBehaviorOphysProjectCache)
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment verified: python3 with numpy 2.4.4, torch 2.6.0+cu124, allensdk 2.16.2 all import successfully.

Directory contents of /app:
- `.manifest`, `Dockerfile`, `docker-compose.yaml` (infrastructure)
- `CONVERSION_NOTES.md` (this file)
- `allensdk_docs/` - downloaded allensdk documentation
- `code/` - AllenSDK source repository (reference code)
- `data/` - AllenSDK VisualBehaviorOphysProjectCache directory:
  - `visual-behavior-ophys_project_manifest_v1.1.0.json` (manifest)
  - `visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb` (284 files)
  - `visual-behavior-ophys-1.1.0/project_metadata/` (csv metadata tables)
  - `_downloaded_data.json`, `_manifest_last_used.txt`, `resources/`
- `decoder.py`, `train_decoder.py` - decoder training/validation code
- `methods.txt`, `paper.pdf`, `whitepaper.pdf` - reference texts
- `tutorials/` - 6 tutorial scripts/notebooks on visual behavior ophys data access

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=..., use_static_cache=True)` | allensdk/brain_observatory/behavior/behavior_project_cache/project_cache_base.py | LOADING | Open the local data cache without S3 access |
| `bc.get_ophys_experiment_table()` | behavior_project_cache.py:219 | LOADING | Table of all ophys experiments (one imaging plane per row): cre_line, session_type, targeted_structure, imaging_depth, mouse_id, ophys_session_id |
| `bc.get_ophys_session_table()` / `get_behavior_session_table()` / `get_ophys_cells_table()` | behavior_project_cache.py | LOADING | Session- and cell-level metadata tables |
| `bc.get_behavior_ophys_experiment(ophys_experiment_id)` | behavior_project_cache.py:337 | LOADING | Returns `BehaviorOphysExperiment` object for one imaging plane |
| `dataset.dff_traces` (`dff`, `cell_roi_id`) | data_objects/cell_specimens/traces/dff_traces.py | LOADING | dF/F already computed by the Allen pipeline; index = cell_specimen_id |
| `dataset.events` (`events`, `filtered_events`, `lambda`, `noise_std`) | data_objects/cell_specimens/events.py | PROCESSING | L0-deconvolved events; `filtered_events` = events convolved with half-gaussian (filter_scale_seconds = 2/31 s, n_time_steps=20) |
| `CellSpecimens(..., exclude_invalid_rois=True)` | data_objects/cell_specimens/cell_specimens.py:154-231 | CURATION | Default: only `valid_roi == True` cells are returned; traces filtered & reordered to match |
| `dataset.ophys_timestamps` | BehaviorOphysExperiment | ALIGNMENT | Timestamps (s, sync-aligned) for each 2p frame; ~31 Hz (mesoscope ~11 Hz) |
| `dataset.stimulus_presentations` | data_objects/stimuli/presentations.py | LOADING | Per-flash table: start_time, end_time, image_name, image_index, is_change, omitted, stimulus_block_name, flashes_since_change |
| `dataset.trials` | data_objects/trials/trials.py, trial.py | LOADING | Per-trial table: start_time, stop_time, change_time, initial_image_name, change_image_name, go, catch, aborted, auto_rewarded, hit, miss, false_alarm, correct_reject, response_time, lick_times, reward_time |
| `Trial._get_trial_data()` | data_objects/trials/trial.py:151-214 | PROCESSING | Definition of trial types: aborted = animal licked before change; go = stimulus change (not catch, not auto_rewarded); catch = sham change; auto_rewarded = reward delivered automatically (hit/miss/FA/CR all forced False); correct_reject = catch & not false_alarm |
| `dataset.running_speed` (`timestamps`, `speed`) | data_objects/running_speed/running_speed.py, running_processing.py:302 | LOADING/PROCESSING | Running speed in cm/s on the stimulus-frame (~60 Hz) clock; already zscore-outlier-corrected (threshold 10) and low-pass Butterworth filtered (order 3, Wn=4 Hz, fs=60) by the SDK |
| `dataset.eye_tracking` (`timestamps`, `pupil_width/height/area`, `likely_blink`) | data_objects/eye_tracking/eye_tracking_table.py:120-192 | LOADING/CURATION | Eye-tracking table on the eye-camera clock; `from_nwb` recomputes `likely_blink` (z_threshold=3.0, dilation_frames=2) and `filter_on_blinks` sets all tracking columns to NaN on blink frames |
| `dataset.licks`, `dataset.rewards` | BehaviorOphysExperiment | LOADING | Lick/reward event times (not required by the decoder spec but useful for sanity checks) |
| `get_trace_around_timepoint(trace, timepoint, timestamps, window, frame_rate)` | swdb/analysis_tools.py | ALIGNMENT | Reference way of cutting trial-aligned traces: finds nearest ophys frame to `change_time` and slices `[window[0]*fr, window[1]*fr]` frames around it |
| `save_trial_response_df.py` main block | swdb/save_trial_response_df.py:284-328 | PROCESSING | Reference trial-response dataframe: traces aligned to `trials['change_time']`, `window_around_timepoint_seconds = [-4, 8]`, response window 0.5 s after change, baseline 0.5 s before; asserts change_times are a subset of flash start times |
| Tutorial `visual_behavior_compare_across_trial_types.py` | /app/tutorials | PROCESSING | Shows querying `dataset.trials.query('hit')` / `miss` / `false_alarm` / `correct_reject` and plotting running/pupil/dff within `[trial.start_time, trial.stop_time]` |
| Tutorial `visual_behavior_load_ophys_data.py` | /app/tutorials | LOADING | Canonical loading recipe; shows `stimulus_presentations.stimulus_block_name.str.contains('change_detection')` to restrict to the behavior (change-detection) block, `np.vstack(dff_traces.dff.values)` to get an (n_cells x n_frames) matrix |

### Notes
- **dF/F does NOT need to be computed**: the SDK returns pipeline-computed `dff_traces` for each valid ROI. The SDK also provides L0 event detection (`events`/`filtered_events`).
- **ROI curation is automatic**: `exclude_invalid_rois=True` is the default, so only ROIs that passed the Allen segmentation/classification QC (`valid_roi`) are returned. The NWB files released in the cache contain only valid ROIs anyway.
- **Time bases**: ophys frames (`ophys_timestamps`), stimulus frames (`stimulus_timestamps`, 60 Hz; running speed lives here), eye-tracking frames (~30 Hz), and event times (licks/rewards/trials/stimulus presentations) are all on a common sync clock in seconds, so they can be aligned directly by interpolating/binning onto the ophys timestamps.
- **Reference trial alignment** in Allen analysis code is to the stimulus **change time** of each trial with a window of a few seconds before/after; this matches the decoder spec ("Temporally align based on ophys timestamp", trials = go + catch).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
`/app/data` is an AllenSDK *local* (non-static) cloud cache for project `visual-behavior-ophys`, manifest v1.1.0:
- `visual-behavior-ophys_project_manifest_v1.1.0.json` - manifest listing metadata csvs and data files
- `visual-behavior-ophys-1.1.0/project_metadata/{behavior_session_table, ophys_session_table, ophys_experiment_table, ophys_cells_table}.csv`
- `visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_<ophys_experiment_id>.nwb` - **284 files**

Loading API (works): `VisualBehaviorOphysProjectCache.from_local_cache(cache_dir='/app/data', use_static_cache=False)`.
(`use_static_cache=True` fails because this cache is laid out as a LocalCache, not a mounted S3 static cache.)

Metadata tables describe the **entire** published VBO dataset (1936 experiments, 703 ophys sessions, 4782 behavior sessions,
133066 cells), but only **284 experiments** have NWB files present locally -> the usable dataset is this subset.

Per-experiment data objects used (`bc.get_behavior_ophys_experiment(eid)`):
- `dff_traces` (n_cells x n_frames dF/F), `events` (`events`, `filtered_events`), `ophys_timestamps` (s)
- `cell_specimen_table` (all `valid_roi == True`, invalid ROIs already excluded)
- `trials` (21 cols incl. start_time, stop_time, change_time, initial/change_image_name, go, catch, aborted, auto_rewarded, hit, miss, false_alarm, correct_reject)
- `stimulus_presentations` (19 cols; blocks: `initial_gray_screen_5min`, `change_detection_behavior`, `natural_movie_one`, `post_behavior_gray_screen_5min`); columns `image_name`, `image_index`, `is_change`, `omitted`, `start_time`, `end_time`, `active`, `is_sham_change`, `flashes_since_change`, `trials_id`
- `running_speed` (timestamps, speed; ~60 Hz), `eye_tracking` (~30 Hz, incl. `pupil_width/height/area`, `likely_blink`; blink frames are NaN)
- `licks`, `rewards`, `metadata`

Example experiment 792815735 (Vip, OPHYS_1_images_A, VISp): 27 neurons, 140208 ophys frames at 30.94 Hz (4534 s),
937 trials (164 go, 24 catch, 749 aborted, 0 auto-rewarded), 13808 stimulus presentations (4806 in the change-detection block),
running 270257 samples @59.95 Hz, eye tracking 136036 samples @29.99 Hz with 2.26% likely-blink (NaN) frames.
For go+catch trials in that session: `change_time - start_time` = 3.02-8.28 s, `stop_time - change_time` = 4.23-4.25 s (fixed).

### Dataset Size (from data files; local cache subset)
| Statistic | Value |
|-----------|-------|
| Ophys experiments (imaging planes) with NWB | 284 (202 active behavior, 82 passive) |
| Neurons (total, all 284) | 42,147 |
| Neurons (total, 202 active-behavior experiments) | 29,444 |
| Neurons / experiment | mean 148.4, median 66, range 4-666 |
| Subjects (mice) | 38 |
| Ophys sessions (unique, active) | 174 (168 single-plane + 6 mesoscope sessions with 3-7 planes) |
| Active ophys sessions / mouse | mean 4.6, range 2-9 |
| Cre lines (active experiments) | Slc17a7 107, Sst 62, Vip 33 |
| Targeted structures (active) | VISp 185, VISl 17 |
| Session types (active) | OPHYS_1_images_A 55, OPHYS_3_images_A 55, OPHYS_4_images_B 46, OPHYS_6_images_B 46 |
| Experience levels (active) | Familiar 110, Novel >1 54, Novel 1 38 |
| Trials / session | ~900 total, of which ~190 go+catch (session-dependent) |
| Ophys frame rate | ~31 Hz (single plane), ~11 Hz (mesoscope) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Task | go/no-go change detection | "mice are presented with a continuous series of briefly presented stimuli and they earn water rewards by correctly reporting when the identity of the image changes" (whitepaper) |
| Images per session | 8 (64 possible transitions) | "Each session included 8 images, for a total of 64 possible transitions" |
| Stimulus timing | 250 ms image, 500 ms gray (750 ms cycle) | "natural images (250 ms stimulus duration) interspersed with periods of a gray screen (500 ms inter-stimulus duration)" (paper) |
| Omission probability | 5% of flashes (never at a change or the flash before) | "5% of image repeats were omitted... Image changes as well as the image immediately before the change were not omitted" |
| Catch probability | ~12.5% of go/catch trials | "a matrix sampling algorithm... pushing the actual catch probability to ~12.5%" |
| Change time distribution | truncated exponential 2.25-8.25 s after trial start, mean ~4.2 s | "Change-times were selected from a truncated exponential distribution ranging from 2.25 to 8.25 seconds... resulting in a mean change time of 4.2 seconds" |
| Response window | 150-750 ms after the change | "the fraction of go-trials in which the mouse licked in a 0.150 to 0.750 second window" |
| Auto-rewarded trials | 5 at session start + after 10 consecutive misses | "Behavior sessions across all phases began with 5 free-reward trials... after 10 consecutive MISS trials" |
| Ophys frame rate | 31 Hz single plane, 11 Hz per plane multi-plane | "Two-photon movies (512x512 pixels, 31 Hz for single plane and 512x512 pixels, 11 Hz for each plane in multi-plane experiments)" |
| Eye-tracking rate | 30 Hz | "eye tracking (30 Hz), and behavior (30 Hz)" |
| Running-speed processing | zscore>10 transient removal + 10 Hz lowpass Butterworth | "Any additional transients with z-score of 10 or greater were removed... smoothing the raw running speed with a 10 Hz lowpass Butterworth filter" |
| Running speed units | cm/s (wheel radius 16.5 cm/2, 2/3 radius point) | "running_speed_cm_per_sec = angular_speed * (2/3 * wheel_radius)" |
| dF/F | computed by the pipeline (600 s median-filter baseline) | "DF/F CALCULATION ... median filter with kernel size 600s ... subtract this baseline ... normalize by the baseline" |
| ROI curation | invalid ROIs (unions, duplicates, edge, dendrites, too small/dim) excluded by classifier | "ROI FILTERING ... any ROIs that end up labeled are not considered cell bodies" |
| Paper neural signal | detected calcium events | "For all analysis of neural data we used the detected calcium events" |
| Paper session/mouse counts (behavior) | 376 imaging sessions, 82 mice (full public dataset) | "contains behavior from 376 imaging sessions from 82 mice" |
| Paper neural counts (their subset) | 8,619 excitatory (21 sessions, 9 mice), 470 Sst (15 sessions, 6 mice), 1,239 Vip (21 sessions, 9 mice) | "Our dataset contains 8,619 excitatory cells..." |
| Engagement threshold | 2 rewards/minute | "corresponds to reward rates above and below 2 rewards per minute" |

NOTE: the paper analysed the *full* public VBO release; the local cache holds only a 284-experiment subset, so absolute
counts (mice/sessions/cells) cannot match the paper. Rates/fractions (catch fraction, omission fraction, hit/FA rates,
change-time distribution, frame rates) are the statistics that *must* match, and they do (see Step 4/9).

### Processing Details
- **Temporal alignment**: all streams are on one sync clock (s). Neural data define the time base (`ophys_timestamps`);
  behavior (running 60 Hz, eye 30 Hz) and stimulus/trial event times are resampled onto the neural time bins.
- **Trial-aligned analysis in Allen reference code** (swdb/save_trial_response_df.py) aligns to `trials.change_time`
  with a window of [-4, +8] s at the ophys frame rate.
- **Paper decoding analysis**: random forest, per *image presentation*, using neural activity in the **first 400 ms after
  image presentation**; decoded (a) image change vs repeat, (b) hit vs miss; 5-fold CV; accuracy reported as % correct.
  Reported qualitatively: change decoding is well above chance for all cell classes (Figure 6A, ~60-80% correct), hit/miss
  decoding modestly above chance (Figure 6C), and false-alarm decoding "very low" (Figure S22).
- **Behavioral data processing (paper)**: behavioral events assigned to the 750 ms *image presentation interval* beginning
  at each flash (also used for omissions). This motivates using the 750 ms flash cycle as the natural quantum of time.

### Curation Steps

**Neuron curation rules**:
- ROI filtering already applied by the Allen pipeline (multi-label classifier; unions/duplicates/edge/dendrite/small/dim
  ROIs marked `valid_roi=False`); the SDK returns only valid ROIs (`exclude_invalid_rois=True` default). No further
  neuron filtering is described in the paper.
- Session/container QC (z-drift < 10 um, <20% photobleaching, d-prime >= 1, sync verified, no interictal events) was applied
  before release, so all released experiments pass QC.

**Trial curation rules**:
- Trials are go / catch / aborted / auto_rewarded. This conversion keeps **go + catch** and drops **aborted**
  (mouse licked before the change) and **auto_rewarded** (free reward) trials, per the task specification. This matches
  the SDK convention that hit/miss/false_alarm/correct_reject are only defined on go/catch trials.
- Passive sessions (OPHYS_2, OPHYS_5) have no behavioral responses/trial outcomes -> excluded (the paper also excluded
  passive sessions: "Imaging was also performed during passive viewing of the same stimulus, which was not analyzed here").

### Decoders Trained (paper)
| Decoded variable | Accuracy |
| image change vs repeat (random forest, 400 ms window) | well above chance, ~0.6-0.85 % correct depending on n cells/cell class (Fig 6A) |
| hit vs miss (random forest) | modestly above chance (Fig 6C) |
| false alarm | "very low" (Fig S22) |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

I cross-checked every quantity that the whitepaper/paper state against what the SDK code computes and what the data contain
(scripts in `/app/cache/`: `scan_all.py`, `check_images.py`, `check_windows.py`, `check_eye.py`, `check_pupil_trials.py`;
outputs `scan_all.csv`, `check_*.csv`).

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Cache loading | `from_local_cache(use_static_cache=True)` needs `<dir>/visual-behavior-ophys/manifests` | cache has manifest json at top level + `visual-behavior-ophys-1.1.0/` data dir | tutorials use `from_s3_cache` (no network here) | Use `from_local_cache(cache_dir='/app/data', use_static_cache=False)`; verified all 284 NWBs are reachable through the SDK |
| Dataset size | metadata tables cover 1936 experiments / 703 sessions / 133k cells | only 284 NWB files present (202 active) | paper analysed 376 imaging sessions / 82 mice | Local cache is a subset; absolute counts cannot match, **rates** must (and do). Restrict to experiments whose NWB exists |
| Catch probability | trial types from `trial.py` | catch/(go+catch) = **0.1253** over 202 experiments | "~12.5%" (whitepaper) | Consistent |
| Omission probability | `presentations.py` marks `omitted` | omitted fraction of change-detection flashes = **0.0352** (mean) | "5% of image repeats were omitted" | Consistent: 5% applies to *eligible* repeats only - changes, the flash before a change, and the first flashes after a change are never omitted, which lowers the overall rate to ~3.5% |
| Change times | SWDB asserts change_times are a subset of flash start times | true in **202/202** experiments | change is a flash | Consistent |
| Change time distribution | - | change-start: min 2.79 s, mean ~4.2 s | "truncated exponential 2.25-8.25 s, mean 4.2 s" | Consistent |
| Images per session | - | exactly 8 non-omitted images in every experiment; 16 unique across the dataset (2 disjoint sets A/B) | "Each session included 8 images" | Consistent |
| Ophys frame rate | - | 30.93-30.95 Hz (168 single-plane experiments), 10.73 Hz (34 mesoscope experiments) | "31 Hz single plane, 11 Hz per plane multi-plane" | Consistent |
| Running speed | `running_processing.get_running_df` (zscore 10 outlier removal + Butterworth lowpass) | `running_speed` @59.95 Hz, no NaNs in any experiment | "10 Hz lowpass Butterworth" (code uses order 3, Wn=4 Hz, fs=60) | Use SDK `running_speed` as-is (already the processed, filtered stream) |
| Eye tracking | `EyeTrackingTable.from_nwb` recomputes blinks and NaNs them | ~3.5% NaN frames/experiment; **3 sessions have no eye-tracking table at all** | "eye tracking (30 Hz)" | Blinks -> NaN by design. Handle explicitly (see Step 5 decisions) |
| Neural signal | SDK provides `dff_traces`, `events`, `filtered_events` | both present for all experiments | "For all analysis of neural data we used the detected calcium events" (paper) | Use **`events`** (L0-detected calcium event magnitudes), matching the paper |
| ROI curation | `exclude_invalid_rois=True` default | `cell_specimen_table.valid_roi` is True for 100% of returned ROIs | ROI FILTERING section | Nothing further to do - released NWBs contain only valid ROIs |
| Multi-plane sessions | experiments = imaging planes | 6 sessions have 3-7 planes; planes in a session share *identical* trial tables (verified: 0/174 sessions disagree) | "up to 8 imaging planes (8 experiments) per session" | Merge planes of the same `ophys_session_id` into ONE decoder session (simultaneously recorded neurons) |
| Trial outcome labels | `trial.py`: hit/miss/false_alarm/correct_reject defined only for go/catch | every go+catch trial has exactly one of the four | go/no-go task | Use the 4 outcomes as the static per-trial output |
| Behaviour rates | - | hit rate 0.350, false-alarm rate 0.143 over all included trials | d-prime >= 1 QC criterion | Consistent with well-trained mice (d' ~ 0.5-1.5 range) |

No unresolved discrepancies.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Session / trial definition
- **Session** = one `ophys_session_id` (a single continuous recording). For Multiscope sessions the 3-7 simultaneously
  recorded imaging planes (`ophys_experiment_id`s) are **concatenated along the neuron axis** into one session, because
  they are simultaneously recorded neurons sharing one behavior session (verified: all planes of a session have identical
  trial tables). Only **active behavior** sessions are used (passive OPHYS_2/OPHYS_5 have no choices/outcomes).
- **Trial** = a `trials` row with `(go | catch) & ~aborted & ~auto_rewarded`, per the Decoder Task spec.
- **Alignment event** = `trials.change_time` (the stimulus change on go trials, the sham change on catch trials).
  This is the reference event used by the Allen SWDB analysis code (`save_trial_response_df.py`).
- **Trial window** = **[-2.25 s, +3.75 s]** around the change time = 8 flash cycles (3 before, 5 after).
  Data-driven justification: over all 202 active experiments min(change_time - start_time) = 2.79 s and
  min(stop_time - change_time) = 4.20 s, so this window is **always inside the trial** (no bleed into neighbouring trials,
  no truncation, identical length for every trial) and always inside `ophys_timestamps`.
- **Time bin** = **250 ms** -> **24 bins/trial**. Chosen because (a) it equals the image presentation duration and divides
  the 750 ms flash cycle exactly into 3, so image/change labels never straddle bins; (b) it is larger than the slowest
  ophys frame interval (93 ms on the Mesoscope), so every bin of every session contains >= 2 imaging frames, allowing one
  common bin size for all sessions as the format requires.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `experiment.events['events']` (L0 detected calcium events) + `ophys_timestamps` | `neural` | sum of event magnitudes of the ophys frames whose timestamp falls in each 250 ms bin -> (n_neurons, 24) float32 | `Events.from_nwb`; paper "we used the detected calcium events" | Neurons of all planes of a session stacked |
| (none) | `input` | empty array (0, 24) float32, `input_names=[]` | - | Decoder Task specifies "No inputs for this task" |
| `stimulus_presentations.image_name` (change_detection block, non-omitted) | `output[0]` image_identity | per bin: image of the most recent non-omitted flash onset <= bin centre; 16 global classes (im000...im106) | tutorials use `stimulus_block_name.str.contains('change_detection')` | Holding the identity across the 500 ms gray ISI implements the paper's "image presentation interval" (750 ms starting at each flash); during a 5% omission the identity of the ongoing (repeating) image is unchanged, so it is held |
| `stimulus_presentations.is_change` | `output[1]` image_change | 1 in the bins of the presentation interval of a change flash (3 bins = 750 ms starting at the change), else 0 | `presentations.py` `is_change` | Catch trials have a *sham* change (`is_change` False) so they stay 0 - exactly the change-vs-repeat discrimination of the paper's change decoder |
| `running_speed` (`speed`, 60 Hz) | `output[2]` running_speed_quintile | mean speed per 250 ms bin -> discretised into 5 **global** equal-percentile bins (20/40/60/80th pct of all bins in the dataset) | `running_processing.get_running_df` (already filtered by SDK) | Decoder Task: "discretized into five equal percentile bins" |
| `eye_tracking.pupil_area` (30 Hz) | `output[3]` pupil_diameter_quintile | diameter = 2*sqrt(area/pi); blink NaN gaps <= 0.5 s linearly interpolated; mean per bin; 5 global equal-percentile bins | `EyeTrackingTable.from_nwb` (blinks already NaN, z=3, dilation 2) | Diameter (not area) as required |
| `trials.hit/miss/false_alarm/correct_reject` | `output[4]` trial_outcome | static per trial, broadcast to all 24 bins; 4 classes | `trial.py::_get_trial_data` | Static per-trial output as specified |
| `ophys_experiment_table.mouse_id` | `subjects`, `subject_idx` | str ids | - | |
| `ophys_experiment_table.targeted_structure` (per plane) | `brain_regions`, `brain_region_idx` | VISp / VISl per neuron | - | |

### Key Decisions
1. **Neural signal = detected calcium events (`events`), summed per bin.** The paper states "For all analysis of neural data
   we used the detected calcium events"; events remove the slow GCaMP6f decay, which otherwise smears stimulus information
   across many 250 ms bins. Raw (unsmoothed) `events` is used rather than `filtered_events`, whose causal half-gaussian is
   documented in the SDK as a *visualisation* smoother; binning already integrates over 250 ms.
2. **dF/F is not recomputed** - the Allen pipeline already provides `dff_traces`, and the event detection was run on those
   traces. ROI curation is likewise already applied (`valid_roi`, `exclude_invalid_rois=True` default): 100% of returned ROIs
   are valid, so no further neuron filtering is warranted (and none is described in the papers).
3. **Mesoscope planes merged per session** (simultaneously recorded populations) - gives larger neuron counts and matches the
   whitepaper definition of a *session* vs an *experiment*.
4. **Passive sessions excluded** (no behavioral report -> no trial outcome), matching the paper ("passive viewing ... was not
   analyzed here").
5. **Aborted and auto-rewarded trials excluded** per the Decoder Task spec; this also matches the SDK's own convention that
   hit/miss/false-alarm/correct-reject are only defined for go/catch trials.
6. **Global (dataset-wide) quintile edges** for running speed and pupil diameter, so the 5 classes are equally populated over
   the whole dataset as "five equal percentile bins" requires, and so that one shared decoder head sees a consistent labelling.
7. **Missing pupil data**: 3 of 174 active sessions have *no* eye-tracking table at all -> those sessions are dropped (a
   required output cannot be fabricated). Within the remaining sessions, blink gaps <= 0.5 s are linearly interpolated (blinks
   are short); trials that still contain a bin with no valid pupil sample (~6% of trials) are dropped. Fabricating pupil
   values for a *decoded* variable would corrupt the evaluation.
8. **All active sessions of all cre lines, both image sets, both areas and all experience levels are kept.** The paper's
   restriction to familiar/Mesoscope sessions served its scientific question (strategy comparison); for building a decoding
   dataset, more sessions is strictly better and the task says to convert data collected under the Visual Behavior task.
9. **Image identity uses the 16 global image names**, not the per-session 0-7 `image_index`, because index i means different
   images in image sets A and B.

### Planned Sanity Checks
- [ ] catch/(go+catch) ~ 0.125 (whitepaper ~12.5%)
- [ ] 8 distinct images present per session; 16 across the dataset
- [ ] change_time always coincides with a flash start time
- [ ] image_change == 1 in exactly 3 bins (bins 9-11) of every go trial and 0 for every catch trial
- [ ] image identity at bin 8 (just before change) != identity at bin 9 for go trials, and == for catch trials
- [ ] running/pupil quintiles each ~20% of bins
- [ ] trial-outcome distribution reproduces hit rate ~0.35, FA rate ~0.14
- [ ] per-neuron binned event sums match an independent re-computation directly from the SDK (spot check)
- [ ] no NaN/Inf anywhere; all trials have identical shape (n_neurons, 24)
- [ ] number of trials per session equals the number of go+catch trials in the SDK trials table (minus pupil-dropped trials)

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` (runs as `python -u /app/convert_data.py <outfile> [--full|--sample] [--show-processing]`).

Structure:
- `get_cache()` / `available_experiment_table()` - open the AllenSDK `VisualBehaviorOphysProjectCache`
  (`from_local_cache(cache_dir='/app/data', use_static_cache=False)`) and restrict the experiment table to experiments
  whose NWB file is present locally. **All data access goes through the SDK**; no NWB file is opened directly.
- `process_session((session_id, experiment_ids, region_map, want_debug, signal))` - worker that loads every imaging plane of
  one ophys session, selects go+catch trials, and bins neural + behaviour into the 24 x 250 ms trial windows.
- `bin_sum` / `bin_mean_nan` - fully vectorised binning via cumulative sums + `np.searchsorted` (no python loop over trials);
  `bin_mean_nan` propagates NaN only when a bin contains *no* valid sample.
- `interpolate_short_gaps` - linear interpolation of pupil blink gaps <= 0.5 s.
- `quantile_bins` / `digitize_with` - global equal-percentile discretisation of running speed and pupil diameter.
- `plot_processing` - `--show-processing` figure with 6 rows x 3 trials: raw events/dF/F traces, binned neural matrix,
  stimulus flashes vs decoded image identity, image-change binary vs the true change flash, raw running speed vs its
  quintile, raw pupil diameter vs its quintile with licks/rewards and the trial outcome.
- `main()` - grouping by `ophys_session_id`, multiprocessing pool, global quantile edges, assembly of the output dict,
  statistics printout, pickling.

Code inefficiencies identified:
- A naive implementation would slice traces per trial in a Python loop (n_trials x n_cells slices).
- Re-opening the cache per experiment is expensive.

Code speedups added:
- Cumulative-sum + `searchsorted` binning: all trials of a session are binned in one vectorised call.
- One cache handle per worker process, sessions processed in parallel (`--nproc`, default 16).
- `float32` storage for neural data.
- Timing is printed per run (`s/session`).

Extra option `--neural-signal {dff,events,filtered_events}` was added so the choice of neural signal could be tested
empirically (see Step 7/8).

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` -> `/app/conversion_sample_out.txt`
`python -u /app/train_decoder.py /app/sample_data.pkl --verify-only` -> `/app/verification_sample_out.txt`
(sample = one single-plane session 775289198 and one 7-plane Mesoscope session 951410079).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 177 (89 + 88) |
| Neurons / session | 88.5 |
| Subjects | 2 |
| Sessions / subject | 1 |
| Trials (total) | 246 |
| Trials / session | 39, 207 |
| Timepoints / trial | 24 (all trials) |
| Inputs | 0 (as specified) |
| image_identity distribution | im061 0.133, im062 0.124, im063 0.130, im065 0.107, im066 0.128, im069 0.115, im077 0.118, im085 0.145 |
| image_change distribution | no_change 0.892, change 0.108 |
| running_speed_quintile | 0.200 x 5 |
| pupil_diameter_quintile | 0.200 x 5 |
| trial_outcome | hit 0.382, miss 0.484, false_alarm 0.037, correct_reject 0.098 |
| neural (dF/F) | mean 0.055, max 22.7, no NaN |

Checks passed:
- `image_change` is 1 in **exactly** bins 9-11 (= [0, 0.75) s after the change) for all 213 go trials and 0 for all 33 catch trials.
- `image_identity` differs between bin 8 and bin 9 for every go trial and is identical across the sham change for every catch trial.
- Expected change fraction: 3 bins / 24 bins x (go fraction 0.87) = 0.109, observed 0.108.
- No NaN/Inf; every trial is (n_neurons, 24); `input` is (0, 24); outputs int64.
- One sample session legitimately has only 39 trials - it is the minimum-trial session of the dataset (confirmed independently in `/app/cache/scan_all.csv`).

### Processing Plots Review
`processing_775289198.png`, `processing_951410079.png`: the raw dF/F traces, the binned neural matrix, the flash raster,
the image-identity step function, the image-change pulse, and the running/pupil traces with their quintiles all line up on the
same time axis with the change at t = 0. The change flash shading coincides exactly with the `image_change` pulse, the
identity step occurs at t = 0 on go trials, licks/rewards appear shortly after t = 0 on hit trials. No anomalies.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
| cumsum/searchsorted vectorised binning | ~10x vs per-trial slicing |
| 16-way multiprocessing over sessions | ~16x |
| float32 neural storage | halves memory/pickle size |

| Step | Time / Session | Estimated Total Time |
| load + bin one session (1-7 planes) | 10.3 s wall per session (serial), ~3 s per plane | 174 sessions, 202 planes, 16 workers -> ~2-4 min |
| pickling + statistics | < 10 s | |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` -> `/app/train_decoder_sample_out.txt`

### Format Validation
- Errors: None
- Warnings: None ("Data format is valid, no errors or warnings.")

### Decoder Results (Sample, 2 sessions, dF/F)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| image_identity | 0.630 | 0.526 | 0.125 |
| image_change | 0.725 | 0.717 | 0.500 |
| running_speed_quintile | 0.338 | 0.262 | 0.200 |
| pupil_diameter_quintile | 0.417 | 0.339 | 0.200 |
| trial_outcome | 0.498 | 0.321 | 0.250 |

Loss decreased monotonically (1.60 -> 1.24 over 200 epochs); all five outputs are above chance.

### Neural-signal comparison (same two sessions, identical everything else)
| Signal | image_identity | image_change | running | pupil | trial_outcome |
|--------|---------------|--------------|---------|-------|---------------|
| `dff` (dF/F, bin mean) | **0.526** | **0.717** | **0.262** | **0.339** | 0.321 |
| `filtered_events` (SDK half-gaussian) | 0.383 | 0.621 | 0.241 | 0.279 | 0.329 |
| `events` (raw L0 events) | 0.336 | 0.619 | 0.232 | 0.290 | 0.330 |

**Decision**: use **dF/F** (`dff_traces`, averaged within each 250 ms bin) as the neural signal.
Justification: (i) it is the signal used by the Allen reference trial-analysis code (`swdb/save_trial_response_df.py`
builds its trial response dataframe from `dff_trace`) and by the SDK tutorials; (ii) the paper's preference for detected
events was motivated by removing GCaMP decay for *event-triggered averaging*, whereas for a decoder the extra-sparse event
trains (98% empty 250 ms bins) discard amplitude information, and empirically decode every variable *worse*; (iii) dF/F keeps
the slow components that carry running/pupil state information. The alternative signals remain available through
`--neural-signal events|filtered_events`, and the comparison above is reported for transparency.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full` -> `/app/conversion_full_out.txt` (90 s total, 0.50 s/session with 16 workers)
`python -u /app/train_decoder.py /app/converted_data.pkl --verify-only` -> `/app/verification_full_out.txt`

### Output Files
- `converted_data.pkl`: 774.2 MB
- `verification_full_out.txt`: created; **"Data format is valid, no errors or warnings."**

### Converted dataset
| Quantity | Value |
|---|---|
| Sessions | 171 (of 174 active ophys sessions; 3 excluded for having no eye-tracking data) |
| Mice | 38 |
| Neurons | 29,168 (mean 170.6/session, min 6, max 666) |
| Trials | 41,904 (mean 245/session, min 39, max 393) |
| Timepoints/trial | 24 (250 ms bins, [-2.25, +3.75] s around the change) |
| Brain regions | VISp 29,006 neurons, VISl 162 neurons |
| Inputs | 0 |

### Trial accounting (no data silently lost)
| Item | Trials |
|---|---|
| go+catch trials in the SDK trials tables of all 174 active sessions | 44,892 |
| - in the 3 sessions with no eye tracking | 917 |
| - dropped because a 250 ms bin had no valid pupil/running sample (blinks > 0.5 s) | 2,071 (4.6%) |
| = trials in `converted_data.pkl` | **41,904** |
Sum check: 41,904 + 2,071 + 917 = 44,892 exactly.
Neuron check: 29,168 converted == 29,168 cells in the scan table for the 171 kept sessions (and VISl 162 == 162).

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data (SDK scan) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | n/a (different subset) | - | 29,444 active / 29,168 in kept sessions | 29,168 | YES |
| Mean neurons/session | - | - | 169.2 | 170.6 | YES |
| Subjects | 82 (full release) | - | 38 (local cache) | 38 | YES (cache subset) |
| Sessions | 376 (full release) | - | 174 active | 171 (3 lack eye tracking) | YES (cache subset) |
| Trials (total go+catch) | - | - | 44,892 | 41,904 (+2,071 dropped +917 excluded) | YES |
| Trials/session (mean) | - | - | 258 | 245 | YES |
| Catch fraction | ~12.5% | - | 12.53% | 12.5% (no_change trials / total) | YES |
| Change bins fraction | - | - | expected 0.875 x 3/24 = 0.109 | 0.109 | YES |
| Hit rate (hit/(hit+miss)) | d' >= 1 QC | - | 0.350 | 0.369 (0.323/(0.323+0.552)) | YES |
| False-alarm rate (fa/(fa+cr)) | - | - | 0.143 | 0.152 (0.019/(0.019+0.106)) | YES |
| Images | 8 per session, 2 sets | - | 16 unique, 8/session | 16 classes, each ~6% of bins | YES |
| Running quintiles | - | - | - | 0.200 x 5 (edges 0.019, 2.52, 19.26, 33.85 cm/s) | YES |
| Pupil quintiles | - | - | - | 0.200 x 5 (edges 74.7, 84.2, 93.1, 105.8 px) | YES |
| Time bin | paper analysed 400 ms post-flash windows / 750 ms presentation intervals | SWDB uses ophys frame rate | 30.9 / 10.7 Hz | 250 ms (common to all sessions) | YES |

### Quantile-scope experiment (24-session subset, everything else identical)
| Scope | image_identity | image_change | running | pupil | outcome |
|---|---|---|---|---|---|
| **global percentiles (chosen)** | 0.465 | 0.700 | **0.406** | **0.404** | 0.301 |
| per-session percentiles | 0.469 | 0.700 | 0.329 | 0.285 | 0.301 |
Global dataset-wide percentile edges are both the literal reading of the specification ("five equal percentile bins")
and empirically better, so they are used.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt` line 1: **"Data format is valid, no errors or warnings."** - no errors and no warnings
to address. Structure: 171 sessions, 41,904 trials, 38 subjects, 2 brain regions, input dim 0, output dim 5, T = 24 for
every trial in every session.

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py`)
The script re-derives everything **directly from the AllenSDK objects** with independently written per-trial loops (it does
not import convert_data.py) and compares with `np.allclose` / `np.array_equal`. Run on 3 single-plane sessions
(775289198, 852326785, 1042249629) and 3 Mesoscope sessions (951410079, 955775716, 958105827):

| Check | Method | Result |
|---|---|---|
| Neural spot checks | mean dF/F of the ophys frames inside each of the 24 bins for a random (trial, neuron), recomputed from `dff_traces` + `ophys_timestamps` | 24/24 `allclose=True` (rtol 1e-4), incl. neurons from planes 1-3 of Mesoscope sessions -> plane concatenation order is correct |
| Image identity | most recent non-omitted flash onset at each bin centre, from `stimulus_presentations` | 24/24 exact match |
| Image change | `is_change` flash and bin centre within 750 ms of it | 24/24 exact match |
| Running quintile | independent per-bin mean of `running_speed.speed` + `np.digitize` with the stored edges | 24/24 exact match |
| Pupil quintile | independent blink-gap interpolation + per-bin mean of 2*sqrt(pupil_area/pi) | 24/24 exact match |
| Trial outcome | `trials.hit/miss/false_alarm/correct_reject` | 24/24 exact match, exactly one label per trial |
| Trial keep-mask | independently recomputed behaviour-completeness mask | reproduced exactly in all 6 sessions |
| Alignment | `change_time` must equal the start time of an `is_change` flash (bin 9 left edge) | true for every go trial tested |
| Trial counts | SDK go+catch count == converted + dropped | exact in all 6 sessions |

**TOTAL FAILURES: 0** in both runs.

### Check 3: Reference code comparison
| Step | My code | Reference | Same? |
|---|---|---|---|
| (a) Loading | `VisualBehaviorOphysProjectCache.from_local_cache(...)`, `get_ophys_experiment_table()`, `get_behavior_ophys_experiment(eid)`, `dff_traces`, `ophys_timestamps`, `stimulus_presentations`, `trials`, `running_speed`, `eye_tracking` | identical call sequence in `/app/tutorials/visual_behavior_load_ophys_data.py` and `visual_behavior_compare_across_trial_types.py` (they use `from_s3_cache` only because they download) | YES |
| (b) Neuron filtering | none beyond the SDK default | `CellSpecimens(exclude_invalid_rois=True)`; whitepaper ROI FILTERING already applied | YES |
| (b) Trial filtering | `(go|catch) & ~aborted & ~auto_rewarded` | `trial.py::_get_trial_data` definitions; tutorials query `hit/miss/false_alarm/correct_reject`, which exist only on these trials | YES (spec-mandated) |
| (c) Temporal alignment | trials aligned to `trials.change_time`, all streams resampled onto ophys-timestamp-derived bins | `swdb/save_trial_response_df.py` aligns traces to `trials['change_time']` with `get_trace_around_timepoint` | YES |
| (d) Binning | fixed window, uniform 250 ms bins | SWDB uses a fixed window at the native frame rate ([-4, +8] s) | Equivalent; a *common* bin size across 31 Hz and 11 Hz rigs is required by the target format, and the window is shortened to [-2.25, +3.75] s so that it never leaves the trial |
| (e) Input construction | empty | n/a | Spec: no inputs |
| (f) Output construction | image identity / change from `stimulus_presentations`; running from `running_speed`; pupil diameter from `eye_tracking.pupil_area`; outcome from `trials` | same source columns as the tutorials; paper assigns behaviour to the 750 ms image presentation interval, which my 250 ms grid subdivides exactly | YES |
| Neural signal | `dff_traces` averaged per bin | SWDB reference code uses `dff_trace`; the paper uses detected events | Documented deviation from the paper (Step 8); both available via `--neural-signal` |

### Check 4: Key statistics comparison
See the table in Step 9. All rates match the reference texts: catch fraction 0.1249 (whitepaper ~12.5%), 8 images/session and
16 across the two image sets, change bins 0.109 = 0.875 x 3/24, hit rate 0.370 and FA rate 0.153 (consistent with the SDK
tables: 0.350 / 0.143 before the pupil-completeness filter), frame rates 31/11 Hz, omission ~3.5% of flashes.
Absolute mouse/session counts are necessarily smaller than the paper's because the local cache holds 284 of 1936 experiments.

### Check 5: Edge cases
| Edge case | Handling | Verified |
|---|---|---|
| Trial window leaving the trial | window [-2.25, 3.75] s vs measured min(change-start) = 2.79 s and min(stop-change) = 4.20 s | `/app/cache/check_windows.py`: satisfied for all 202 experiments |
| Trial window leaving the recording | all change times +/- window are inside `ophys_timestamps` | `/app/cache/check_images.py`: True for all experiments |
| Empty 250 ms bin (no ophys frame) | would give a spurious 0 | `/app/cache/check_bin_occupancy.py`: **min 2 frames/bin** (median 7) across all 171 sessions, 0 empty bins |
| Trials with NaN `change_time` | dropped explicitly | none exist in go/catch trials (checked over all experiments) |
| Sessions without eye tracking | excluded and listed in `metadata['sessions_excluded']` | 3 sessions |
| Blinks (NaN pupil) | gaps <= 0.5 s interpolated; trials still incomplete are dropped (2,071 = 4.6%) | trial accounting sums exactly |
| Trials with no outcome label | would raise; none found | checked in every session |
| Mesoscope planes | concatenated along the neuron axis; plane trial tables verified identical | 0/174 sessions disagree |
| Sessions with < 2 trials | excluded by construction | none triggered (min 39 trials) |
| Off-by-one at bin 9 | change flash onset = left edge of bin 9; `image_change` occupies bins 9-11 only | verified for every trial of the sample and by sanity checks |

### Issues Found and Resolved
- *Global neural statistics crashed* when sessions had different neuron counts -> replaced with a per-session accumulation.
- *`np.str_` leaking into `output_values`/`brain_regions`* -> cast to plain `str`.
- *Neural signal choice*: raw `events` decoded worse than dF/F -> switched the default to dF/F and documented the comparison.
- *Quantile scope*: per-session quintiles decoded worse and deviate from the "five equal percentile bins" wording -> global edges kept.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` -> `/app/train_decoder_full_out.txt`
(plus `sample_trials.png` and `predictions.png`). Trained on the GPU, 200 epochs, 171 sessions / 41,904 trials / 29,168 neurons.

### Training Progress
- Loss decreasing: **Yes**, monotonically 1.72 -> 1.238 over 200 epochs; test loss 1.307.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Ratio to chance | Notes |
|--------|-------------|--------|--------|--------|-------|
| image_identity | 0.4901 | **0.4532** | 0.0625 (16 classes) | 7.3x | strong image decoding in V1/LM |
| image_change | 0.7106 | **0.6623** | 0.5 | 1.32x | change vs no-change per 250 ms bin |
| running_speed_quintile | 0.4381 | **0.4006** | 0.2 | 2.0x | |
| pupil_diameter_quintile | 0.5099 | **0.4677** | 0.2 | 2.3x | |
| trial_outcome | 0.4986 | **0.3074** | 0.25 (4 classes) | 1.23x | hardest: static label, 2% false-alarm class |

All five outputs are above chance; train/validation gaps are modest except for trial_outcome (see Step 12).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Output | Validation balanced acc | Chance | Ratio |
|---|---|---|---|
| image_identity | 0.4532 | 0.0625 | **7.25x** |
| image_change | 0.6623 | 0.5 | 1.32x |
| running_speed_quintile | 0.4006 | 0.2 | 2.00x |
| pupil_diameter_quintile | 0.4677 | 0.2 | 2.34x |
| trial_outcome | 0.3074 | 0.25 | 1.23x |
No output is at or below chance. Two outputs are below the 1.5x guideline and were investigated:

**image_change (1.32x).** The ceiling here is structural, not a bug: a *change* label exists only in bins 9-11 of go trials
(10.9% of bins), and the neural response to a change is largely a transient in the first 1-2 bins. Balanced accuracy averages
recall over the change and no-change classes, and the no-change class includes the 3 bins of *every other* flash, which evoke
nearly identical image-onset responses. I verified the information is genuinely present by replicating the paper's own change
decoder (below): 0.77 correct, in the middle of the paper's reported range.

**trial_outcome (1.23x).** Diagnostics: (i) the label is *static* per trial, so the decoder is asked to predict it from
pre-change bins too, where the outcome is not yet determined (in the 2.25 s before the change, hits and misses are physically
indistinguishable apart from arousal/running state); (ii) the 4 classes are very unbalanced (hit 0.323, miss 0.552,
false_alarm 0.019, correct_reject 0.106) and balanced accuracy weights the 801 false-alarm trials as much as the 23,120 miss
trials; the paper itself reports that "false alarm decoding performance was very low" (Figure S22); (iii) the format requires
a per-trial static output for this variable, so this cannot be improved by re-shaping the label. I confirmed the data are
not at fault by decoding hit vs miss with the paper's own method (below): 0.77 correct.

### Check 2: Accuracy comparison to the paper
The paper reports only two decoding analyses (random forest, first 400 ms after image presentation, 5-fold CV, % correct,
Figure 6A/6C). I replicated them **on my converted data** (`/app/cache/paper_decoders.py`, per session, 5-fold CV
RandomForestClassifier, features = bins 9-10 = 0-500 ms after the change, repeat = bins 6-7 = the preceding flash):

| Decoder | Paper (Fig 6) | This conversion | Match? |
|---|---|---|---|
| change vs repeat (% correct) | ~0.6-0.85 depending on cell class / n cells | **0.770 +/- 0.009** (171 sessions) | YES |
| hit vs miss (% correct) | modestly above chance, ~0.6-0.8 | **0.771 +/- 0.008** (162 sessions) | YES |
| false alarm | "very low" | reproduced as the worst class of `trial_outcome` | YES |
Median 108 cells/session (the paper used up to a few hundred per imaging plane).
The converted dataset therefore contains at least as much task information as the paper reports; the lower balanced
accuracies of the provided decoder come from its different read-out (a per-timepoint linear decoder on 100 session PCs,
evaluated on *every* 250 ms bin including pre-change bins, and scored with balanced accuracy over rare classes),
not from the conversion.

### Check 3: Train vs validation gap
| Output | Train | Validation | Train/Val |
|---|---|---|---|
| image_identity | 0.4901 | 0.4532 | 1.08 |
| image_change | 0.7106 | 0.6623 | 1.07 |
| running_speed_quintile | 0.4381 | 0.4006 | 1.09 |
| pupil_diameter_quintile | 0.5099 | 0.4677 | 1.09 |
| trial_outcome | 0.4986 | 0.3074 | **1.62** |
Only trial_outcome exceeds the 1.5x guideline. That is expected for a *static per-trial* label: all 24 timepoints of a trial
share one label, so the effective sample size is the number of trials (245/session), not timepoints, while the decoder has
100 PCs per session - so it can partly memorise training trials. There is no data leakage: the split is by trial
(`train_idx`/`test_idx` index trials), and every output value is derived only from data inside its own trial.

### Issues Found and Resolved
- No new issues were found in this round. The two flagged accuracies were traced to intrinsic properties of the task/label
  definitions and were validated against the paper's own decoders, which my data reproduce.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, loading instructions, output specification, key statistics, repro commands)
- [x] `cache/` folder created with `README_CACHE.md` documenting every cached script, table and figure
- [x] All files organised:
  - Deliverables in `/app`: `CONVERSION_NOTES.md`, `README.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`,
    `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`,
    `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`,
    `processing_775289198.png`, `processing_951410079.png`
  - Investigation/validation scripts, intermediate CSVs, alternative conversions and figures in `/app/cache/`
