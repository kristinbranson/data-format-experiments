# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory — **Visual Behavior 2P** (AllenSDK cache `visual-behavior-ophys-1.1.0`)
- **Reference papers**: Allen Institute *Visual Behavior 2P Technical Whitepaper* (`whitepaper.pdf`); Piet et al., *"Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex"* (`paper.pdf`, `methods.txt`)
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`) and train the reference decoder.

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `.manifest`, `Dockerfile`, `docker-compose.yaml` — environment bookkeeping
- `allensdk_docs/` — downloaded allensdk documentation
- `code/` — AllenSDK source tree (the reference code base)
- `data/` — AllenSDK S3-style cache (`visual-behavior-ophys-1.1.0/`, 247 GB)
  - `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb` — 284 experiment NWB files
  - `data/visual-behavior-ophys-1.1.0/project_metadata/*.csv` — 4 project tables
- `decoder.py`, `train_decoder.py` — reference decoder code (read-only reference)
- `methods.txt`, `paper.pdf`, `whitepaper.pdf` — reference texts
- `tutorials/` — 5 AllenSDK Visual-Behavior tutorial scripts/notebook

Environment check: `python3` works; `numpy 2.4.4`, `torch 2.6.0+cu124`, `allensdk 2.16.2`.
Hardware: 128 CPU cores, 1006 GB RAM, 1× NVIDIA L4 (23 GB), 3.4 TB free disk.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

The reference code base in `/app/code` is the **AllenSDK** itself (the tool used by both reference
papers to load this dataset). The tutorials in `/app/tutorials` show the canonical access pattern.

### Key Functions Identified
| Function / attribute | File | Stage | Purpose |
|----------|------|-------|---------|
| `bpc.VisualBehaviorOphysProjectCache.from_local_cache(cache_dir)` | `allensdk/brain_observatory/behavior/behavior_project_cache/behavior_project_cache.py` | LOADING | Open the on-disk cache without network access. (`from_s3_cache` needs S3; `use_static_cache=True` needs a `manifests/` subfolder that this cache does not have.) |
| `cache.get_ophys_experiment_table()` | same | LOADING/CURATION | 1936-row table of all imaging planes; columns used for curation: `project_code`, `passive`, `targeted_structure`, `cre_line`, `experience_level`, `mouse_id`, `ophys_session_id` |
| `cache.get_behavior_session_table()` | same | CURATION | per-session behavior summary: `go_trial_count`, `catch_trial_count`, `hit/miss/false_alarm/correct_reject_trial_count` — used here as ground truth for sanity checks |
| `cache.get_behavior_ophys_experiment(oeid)` | same | LOADING | returns a `BehaviorOphysExperiment` object for one imaging plane |
| `ds.events` (DataFrame: `events`, `filtered_events`, `lambda`, `noise_std`) | `allensdk/brain_observatory/behavior/data_objects/timestamps/...`, NWB API | PROCESSING | **detected calcium events** per cell — the neural signal used by the Piet et al. paper (evaluated, not shipped: see Step 6) |
| `ds.dff_traces` (`dff`) | " | PROCESSING | ΔF/F traces. **dF/F is already computed by the Allen pipeline** (see methods.txt "DF/F CALCULATION") — nothing to compute ourselves. **This is the neural signal that is shipped.** |
| `ds.ophys_timestamps` | " | ALIGNMENT | 2-photon frame times (seconds, same clock as everything else). Used as the master time base. |
| `ds.cell_specimen_table` (`valid_roi`, `cell_roi_id`) | `allensdk/.../data_objects/cell_specimens.py` | CURATION | ROI table; released data already contains only `valid_roi == True` ROIs (verified) |
| `ds.trials` | `allensdk/brain_observatory/behavior/data_objects/trials/trials.py` + `trial.py` | PROCESSING | trial table with `start_time`, `stop_time`, `change_time`, `go`, `catch`, `aborted`, `auto_rewarded`, `hit`, `miss`, `false_alarm`, `correct_reject` |
| `Trial._get_trial_data()` | `allensdk/brain_observatory/behavior/data_objects/trials/trial.py:175-215` | PROCESSING | **definitive trial-type logic**: if `aborted` → `go = catch = auto_rewarded = False`; else `catch = params['catch']`, `auto_rewarded = params['auto_reward']`, `go = not catch and not auto_rewarded`. `correct_reject = catch and not false_alarm`. Hence `go` and `catch` are mutually exclusive and **already exclude aborted and auto-rewarded trials**. |
| `Trial.calculate_change_frame()` | same, l.377 | PROCESSING | `change_time` = real image change for `go`, **sham change** for `catch` |
| `ds.stimulus_presentations` | `allensdk/.../data_objects/stimuli/presentations.py` | PROCESSING | flash table: `start_time`, `end_time`, `image_name`, `image_index`, `omitted`, `is_change`, `is_sham_change`, `stimulus_block_name`, `trials_id` |
| `ds.running_speed` (`timestamps`, `speed`) | `allensdk/brain_observatory/behavior/running_processing.py` | PROCESSING | 60 Hz filtered running speed in cm/s (10 Hz low-pass Butterworth, per whitepaper) |
| `ds.eye_tracking` (`timestamps`, `pupil_area`, `likely_blink`, …) | `allensdk/.../data_objects/eye_tracking/` | PROCESSING | ~30 Hz pupil tracking. Non-`_raw` columns are set to **NaN where `likely_blink`** |
| `ds.licks`, `ds.rewards` | " | (unused) | not needed for the requested outputs |

### Notes
- **Do ΔF/F need computing?** No. The Allen pipeline already performs motion correction, segmentation,
  ROI filtering, demixing, neuropil subtraction, ΔF/F and event detection (methods.txt §DATA PROCESSING).
  The SDK returns the finished `dff_traces` / `events`.
- **Cell quality filtering?** Already applied upstream (ROI FILTERING section of methods.txt); released
  `cell_specimen_table.valid_roi` is all `True`. No further neuron filtering is described in either paper.
- The tutorial `visual_behavior_load_ophys_data.py` shows `np.vstack(dataset.dff_traces.dff.values)` to
  get a (cell × time) array — this is the exact idiom used by the conversion.
- The tutorial also shows that the change-detection part of a session must be selected with
  `stimulus_presentations.stimulus_block_name.str.contains('change_detection')` (SDK ≥ 2.16 puts the
  5-min grey screens and the 5-min natural movie in the same table).

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
/app/data/
  visual-behavior-ophys_project_manifest_v1.1.0.json   # manifest used by from_local_cache
  _downloaded_data.json, _manifest_last_used.txt
  visual-behavior-ophys-1.1.0/
      project_metadata/ behavior_session_table.csv (4782×35)
                        ophys_session_table.csv    (703×26)
                        ophys_experiment_table.csv (1936×31)
                        ophys_cells_table.csv      (133066×3)
      behavior_ophys_experiments/ behavior_ophys_experiment_<oeid>.nwb   # 284 files, ~870 MB each
```
One NWB = one **ophys experiment** = one imaging plane in one session.

Of the 1936 experiments in the manifest, **284 NWB files are present locally**:

| project_code | n experiments present | n in full manifest |
|---|---|---|
| **VisualBehavior** | **239 (all of them)** | 239 |
| VisualBehaviorMultiscope | 45 (1 mouse) | 862 |
| VisualBehaviorTask1B | 0 | 199 |
| VisualBehaviorMultiscope4areasx2d | 0 | 636 |

i.e. the local cache contains the **complete `VisualBehavior` dataset variant** plus a single
Multiscope mouse. Of the 239 `VisualBehavior` experiments, 168 are active-behavior and 71 are passive.

### Dataset Size (from data files, after the curation of Step 5: `project_code == 'VisualBehavior'`, active behaviour)
| Statistic | Value |
|-----------|-------|
| Sessions (= imaging planes = NWB files) | 168 |
| Neurons (total) | 29,097 |
| Neurons / session | mean 173.2, median 117.5, min 6, max 666 |
| Subjects (mice) | 37 |
| Sessions / subject | mean 4.5 (168/37) |
| Trials (total, go+catch) | 43,387 (37,948 go + 5,439 catch) |
| Trials / session | mean 258.3, min 39, max 409 |
| Brain region | VISp only (all 168) |
| Cre lines | Slc17a7 107, Vip 33, Sst 28 sessions |
| Experience level | Familiar 88, Novel 1 31, Novel >1 49 |
| Image sets | A = im061,062,063,065,066,069,077,085 (88 sessions); B = im000,031,035,045,054,073,075,106 (80 sessions) |

### Available variables (per experiment) and dtypes
- `ophys_timestamps`: float64 (n_frames,), ~140,000 frames, **31.0 Hz** (`ophys_frame_rate`), median Δt = 32.32 ms (range 32.31–32.33 ms across sessions).
- `events` / `filtered_events` / `dff`: object columns of float64 arrays of length n_frames, one row per cell. **No NaNs anywhere** (checked all 168 sessions).
- `trials`: 680 rows/session on average — 226 go, 32 catch, 418 aborted, 3.8 auto-rewarded.
- `stimulus_presentations`: 4802 change-detection flashes/session, 170 omitted, 230 changes, 32 sham changes, plus 9000 natural-movie frames and 2 grey-screen blocks.
- `running_speed`: 270,257 rows @60 Hz, no NaNs, range checked per session.
- `eye_tracking`: ~136,000 rows @30 Hz; `pupil_area` NaN exactly where `likely_blink` (mean 5.5 % of frames, median 3.0 %, max 29.6 %). **3 of 168 sessions have an empty eye-tracking table.**
- All four streams (ophys, running, eye, stimulus) fully cover the change-detection block in every session (verified: no session has `stream_t0 > block_t0` or `stream_t1 < block_t1`).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Task | go/no-go change detection, 8 natural images | "mice are presented with a continuous series of briefly presented stimuli and they earn water rewards by correctly reporting when the identity of the image changes. Each session included 8 images" |
| Flash timing | 250 ms image, 500 ms grey ⇒ 750 ms cycle | "a series of natural images (250 ms stimulus duration) interspersed with periods of a gray screen (500 ms inter-stimulus duration)" |
| Omissions | 5 % of flashes | "5% of image repeats were omitted…Image changes as well as the image immediately before the change were not omitted" |
| Change time distribution | truncated exponential 2.25–8.25 s, mean 4.2 s | "Change-times were selected from a truncated exponential distribution ranging from 2.25 to 8.25 seconds…resulting in a mean change time of 4.2 seconds" |
| Catch probability | ~12.5 % in later sessions (was ~30 %) | "a matrix sampling algorithm that ensured that each image transition was sampled equally, pushing the actual catch probability to ~12.5%" |
| Free/auto rewards | 5 at session start (+ after 10 consecutive misses) | "Behavior sessions across all phases began with 5 'free-reward' trials" |
| Response window | 150–750 ms after change | "if mice responded correctly within a short, post-change response window (150-750ms…)" |
| Ophys frame rate | 31 Hz single-plane, 11 Hz multi-plane | "512x512 pixels, 31 Hz for single plane and 512x512 pixels, 11 Hz for each plane in multi-plane experiments" |
| Eye tracking / behavior rate | 30 Hz | "eye tracking (30 Hz), and behavior (30 Hz)" |
| Dataset variants | `VisualBehavior` vs `VisualBehaviorTask1B` are the two **single-plane** variants | "compare dataset variants VisualBehavior and VisualBehaviorTask1B in Figure 5. This control was performed using the single-plane 2-photon imaging system." |
| Whole released dataset | 376 imaging sessions, 82 mice | "This dataset contains behavior from 376 imaging sessions from 82 mice" |
| Piet et al. neural subset | 8,619 exc (21 sessions, 9 mice), 470 Sst (15, 6), 1,239 Vip (21, 9) | "Our dataset contains 8,619 excitatory cells (21 imaging sessions, 9 mice), 470 Sst cells…" — **this is the multi-plane, familiar-images subset, which is NOT what is present in this local cache** |
| Neural signal | detected calcium events | "For all analysis of neural data we used the detected calcium events…thus removing the slow decay dynamics of the calcium indicator GCaMP6f" |
| Engagement threshold | reward rate 2/min | "epochs labeled 'engaged' or 'disengaged' corresponds to reward rates above and below 2 rewards per minute" |
| Behavioral epoch | 750 ms image presentation interval | "By image presentation interval we refer to the 750 ms interval beginning with each image presentation" |

### Processing Details
- All data streams are already on a **common clock** (the sync board, whitepaper §DATA SYNCHRONIZATION);
  AllenSDK returns every stream with timestamps in that clock, so alignment = resampling onto
  `ophys_timestamps`.
- Neural pipeline (motion correction → segmentation → ROI filtering → demixing → neuropil subtraction →
  ΔF/F → event detection) is already applied in the released data.
- Piet et al. restricted to familiar images on the multi-plane rig — a restriction motivated by their
  specific novelty control, and **not applicable here** (the local cache contains the single-plane
  `VisualBehavior` variant; applying their restriction would leave 22 planes from **one** mouse).

### Curation Steps
**Neuron curation rules** (from the references): ROI filtering (unions, duplicates, motion-border,
dendrites, too small/narrow/dim) and removal of ROIs with non-positive demixed traces are performed
*upstream*; the released `events`/`dff_traces` contain only valid cells. → **no additional neuron
filtering** (verified `valid_roi` is all True).

**Trial curation rules**: per the Decoder Task, keep `go | catch` and drop `aborted` and
`auto_rewarded`. The SDK's own `Trial` logic makes `go`/`catch` already mutually exclusive with
aborted/auto-rewarded, so `trials[trials.go | trials.catch]` is exactly the requested set
(verified: 0 of 43,387 selected trials are aborted or auto-rewarded).

**Session curation rules**: active behaviour only (passive sessions have a retracted lick spout, hence
no hit/miss/FA/CR outcomes); `project_code == 'VisualBehavior'`; sessions with an empty eye-tracking
table dropped (3 sessions) because pupil diameter is a required decoder output.

### Decoders Trained (in the reference paper)
| Decoded variable | Method | Accuracy reported |
|---|---|---|
| Image change vs. repeat ("change decoder") | random-forest on per-image-presentation population activity, 5-fold CV | Fig. 5/S17: rises with # neurons; ≈ 0.6–0.85 depending on cell class & n (Vip/Sst lower, exc higher) |
| Hit vs. miss ("hit decoder") | random forest on image changes, 5-fold CV | ≈ 0.55–0.70 |
Both are *per-image-presentation* decoders with a different architecture from `decoder.py`, so they
give only an order-of-magnitude expectation, not a directly comparable number.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Which sessions are "Visual Behavior" | `project_code` ∈ {VisualBehavior, VisualBehaviorTask1B, VisualBehaviorMultiscope, …} | only `VisualBehavior` (239, complete) + 45 Multiscope planes of 1 mouse are downloaded | whitepaper calls `VisualBehavior` and `VisualBehaviorTask1B` *dataset variants* | Use `project_code == 'VisualBehavior'` — the named variant, and the one that is completely present locally. The 45 Multiscope planes come from a single mouse and would additionally require merging 3–7 planes with different frame clocks into one "session"; excluded (documented in Step 5). |
| Piet et al. cell counts (8,619/470/1,239) | – | our subset has 27,895 exc / 396 Sst / 806 Vip | their numbers are for multi-plane + familiar only | Not reproducible from this cache; not a valid target. Verified our counts against `ophys_cells_table.csv` instead. |
| Neural signal: `dff` vs `events` vs `filtered_events` | SDK exposes all three; `filtered_events` = `events` convolved with a `stats.halfnorm` kernel (identical mean, 0.0010719 both) | `events` is **99.77 % zeros** at 32 ms resolution, `filtered_events` 96.3 %, `dff` 0 % | whitepaper's processing chain ends at **detrended dF/F**; Piet et al. used "detected calcium events" for their per-flash analyses | Use `ds.dff_traces.dff`. It is the Allen pipeline's neural output (already computed — nothing for us to recompute), and at the 32 ms bin the event traces are almost entirely zeros. Measured head-to-head in Step 6: dF/F is better on every output. Both event variants remain available via `--neural-signal`. |
| auto-rewarded trials | `trial.py`: `go = not catch and not auto_rewarded` | 645 auto-rewarded trials exist, 0 of them are go or catch | "sessions began with 5 free-reward trials" | Consistent. `go|catch` already excludes them. |
| Catch fraction | – | 5,439/43,387 = 12.5 % | "pushing the actual catch probability to ~12.5%" | **Match** ✓ |
| Omission fraction | – | 169.9/4801.8 = 3.54 % of all flashes (= 3.7 % of non-change flashes) | "5% of image repeats were omitted" | Slightly below 5 % because changes and pre-change flashes are never omitted and the 5 % is applied to eligible repeats only; consistent. |
| Change count | – | `is_change` count (229.7/session) vs `go` count (225.9/session) | – | `is_change` also flags the change on **auto-rewarded** trials (3.8/session). 229.7 − 225.9 = 3.8 ✓ exactly. Resolved. |
| Sham change count | – | `is_sham_change` = 32.375 = `catch_trial_count` exactly | – | **Match** ✓ |
| Trial length | – | go/catch trials stop 4.241 ± 0.009 s after `change_time`; length 7.0–12.6 s | change time 2.25–8.25 s after trial start | Consistent (2.78–8.32 s pre-change + 4.24 s post-change). |
| Frame rate | metadata `ophys_frame_rate` = 31.0 | median Δt = 32.319 ms (=30.94 Hz) | "31 Hz for single plane" | Consistent; use the measured mean Δt for `time_bin_size`. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Session / trial definition
- **session** = one `BehaviorOphysExperiment` (one imaging plane; for `VisualBehavior` there is exactly
  one plane per ophys session, so session ≡ experiment ≡ NWB file).
- **trial** = one row of `ds.trials` with `go == True or catch == True`, spanning
  `[start_time, stop_time)` — i.e. the trial exactly as the experiment defines it.
- **time base** = `ds.ophys_timestamps`. A trial's timepoints are the ophys frames `t` with
  `start_time <= t < stop_time` (found with `np.searchsorted`). T therefore varies (≈218–390 frames,
  mean ≈ 262) but the **bin size is constant** at the 2-photon frame interval (32.32 ms).

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `ds.dff_traces.dff` (cell × frame) | `neural[s][k]` (n_neurons, T) | `np.vstack(...)[:, i0:i1]`, cast float32 | `get_behavior_ophys_experiment`, tutorial `np.vstack(dff.values)` idiom | detrended dF/F, already computed by the Allen pipeline |
| — | `input[s][k]` (0, T) | empty; task specifies no decoder inputs | – | `input_names = []` |
| `ds.stimulus_presentations` (change-detection block, `omitted == False`) | `output[s][k][0]` image identity | for each flash, ophys frames in `[start_time, end_time)` ← `image_name`; all other frames ← `grey` | tutorial `stimulus_block_name.str.contains('change_detection')` | 17 categories: `grey` + 16 images (8 in set A, 8 in set B) |
| same table, `is_change == True` | `output[s][k][1]` image change | 1 on the ophys frames of the changed flash's 250 ms presentation, else 0 | `Trial.calculate_change_frame` | catch trials are all-zero (sham change = no image change) |
| `ds.running_speed.speed` (60 Hz) | `output[s][k][2]` running-speed bin | `np.interp` onto ophys timestamps, then **global** quintile bins | `running_processing.py` (already 10 Hz low-pass filtered) | 5 equal-percentile bins over all trial timepoints of the whole dataset |
| `ds.eye_tracking.pupil_area` (30 Hz, NaN at blinks) | `output[s][k][3]` pupil-diameter bin | diameter `= 2·sqrt(area/π)`; linear interpolation across blink NaNs; `np.interp` onto ophys timestamps; **global** quintile bins | `eye_tracking` data object; `likely_blink` masking | monotone transform ⇒ bins identical whether computed on area or diameter |
| `ds.trials.hit/miss/false_alarm/correct_reject` | `output[s][k][4]` trial outcome | one of 4 classes, constant over the trial's T frames | `Trial._get_trial_data` | exactly one is True for every go/catch trial (verified) |
| `experiment_table.mouse_id` | `subjects`, `subject_idx` | unique sorted list | – | 37 mice |
| `experiment_table.targeted_structure` | `brain_regions`, `brain_region_idx` | one entry per neuron | – | all VISp in this subset |

### Key Decisions
1. **Which experiments (`project_code == 'VisualBehavior'`, `passive == False`)**: the task says to
   convert data "under the *Visual Behavior* task". `VisualBehavior` is a literal `project_code` and a
   named *dataset variant* in the whitepaper, and it is the only project whose experiments are
   **completely** present in the local cache (239/239). Passive-viewing sessions are excluded because
   the lick spout is retracted, so no go/catch trial outcomes exist. → 168 sessions, 37 mice.
   *(The 45 locally-present Multiscope planes belong to a single mouse and would require merging
   3–7 planes acquired on different frame clocks into one "session"; excluded.)*
2. **Neural signal = `dff` (detrended dF/F)**: the endpoint of the whitepaper's processing chain and
   already computed in the released data. Compared head-to-head against `events` and
   `filtered_events` in Step 6; dF/F wins on every output, and the event traces are ≥96 % zeros at
   the required 32 ms bin.
3. **No extra neuron filtering**: the Allen pipeline's ROI filtering is already applied upstream and
   `valid_roi` is all-True in the released data; neither reference describes further cell curation.
4. **Trial window = the experiment's own trial** (`start_time`→`stop_time`) rather than a hand-chosen
   window around the change, because the task says to segment "based on how they are defined in the
   experiment". Consequence: `off_start = 0.0` (relative to trial start) and `off_end = None`
   (variable-length trials).
5. **Time base = ophys frames, no re-binning**: the task says to align on ophys timestamps. Bin size is
   the 2-photon frame interval, constant to ±0.01 ms across the 168 sessions.
6. **Image identity during grey screen = a separate `grey` class**, and **omitted flashes count as
   grey** (the screen is literally grey during an omission). The task says "image identity *of the
   image presented during the non-grey screen*".
7. **Image change = 1 for the 250 ms presentation of the changed image** — the window "right after a
   change in image identity", consistent with how image identity is coded.
8. **Quintile bins for running/pupil are computed globally** over all trial timepoints of all included
   sessions, so that the class labels mean the same thing in every session and the dataset-level
   distribution is exactly 20 % per bin.
9. **Blink handling**: `pupil_area` is NaN exactly where `likely_blink`; linearly interpolate across
   those gaps (edges held constant) rather than dropping timepoints, so trials keep a common T with
   the neural data.
10. **3 sessions with an empty eye-tracking table are dropped** (oeid 795953296, 833631914, 806456687;
    276 cells, 917 trials) — pupil diameter is a required output and cannot be fabricated.
11. **Outputs are packed as a single `(5, T)` integer array** per trial (the static trial outcome is
    broadcast across the trial) because the format requires one array per trial.

### Planned Sanity Checks (all executed in Step 10; results there)
- [x] #trials per session == `behavior_session_table.go_trial_count + catch_trial_count`
- [x] #trials with each outcome == `hit/miss/false_alarm/correct_reject_trial_count`
- [x] #neurons per session == rows of `ophys_cells_table.csv` for that experiment
- [x] every go trial contains exactly one `image_change == 1` run; every catch trial contains none
- [x] number of change frames per session ≈ 0.25 s / 0.03232 s ≈ 7–8 frames per change (measured 7.74)
- [x] catch fraction ≈ 12.5 % (whitepaper) — measured 12.54 %
- [x] overall running/pupil bin distribution = [0.2,0.2,0.2,0.2,0.2]
- [x] image identity: grey fraction ≈ 2/3 (measured 0.669), each image ≈ 0.020–0.021
- [x] neural spot-check against a freshly loaded NWB (`np.allclose`)
- [x] running/pupil spot-check against a freshly loaded NWB (`np.allclose`)
- [x] no NaN/Inf anywhere; all outputs integer and within their declared ranges

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` — `python -u /app/convert_data.py <out.pkl> [--full|--sample] [--show-processing]`
(extra debug flags: `--workers N`, `--nsessions N`, `--neural-signal {dff,filtered_events,events}`).

Structure:
1. `select_experiments()` — AllenSDK `get_ophys_experiment_table()` ∩ locally present NWBs,
   `project_code == 'VisualBehavior'`, `passive == False`, sorted by experiment id.
2. `extract_session(oeid)` — one process per session:
   `get_behavior_ophys_experiment` → `dff_traces` / `ophys_timestamps` / `stimulus_presentations`
   / `running_speed` / `eye_tracking` / `trials`; builds the whole-session per-frame timeseries
   once (vectorised `np.searchsorted` for flash boundaries) and then slices out each go/catch trial.
3. Global quintile edges for running speed and pupil diameter are computed *after* all sessions are
   read (they must be global to make the class labels comparable across sessions).
4. Assembly into the target dict + `run_summary()` (prints statistics, asserts the format invariants).
5. `show_processing()` re-opens the session through the SDK and overlays the converted trial data on
   the raw SDK tables, so the plots are an independent check rather than a replay of the conversion.

Code inefficiencies identified:
- Per-trial `np.searchsorted` over the flash table would be O(n_trials · n_flashes); instead the
  per-frame image-identity / change vectors are built **once per session** with a single vectorised
  `searchsorted` over all 4,800 flashes, then sliced per trial.
- Re-reading the NWB for each stream — avoided; the `BehaviorOphysExperiment` object is opened once.

Code speedups added:
- `multiprocessing.Pool(16)` over sessions (NWB decompression is the bottleneck): 2.2 s/session
  serial → 0.55 s/session wall.
- float32 throughout, `np.interp` instead of pandas resampling.
- `pickle.HIGHEST_PROTOCOL` for the (multi-GB) output.

### Neural-signal decision (recorded here because it was made empirically)
Converted the *same* 11 sessions three times and ran the reference decoder on each:

| Validation balanced accuracy | `events` | `filtered_events` | **`dff`** | chance |
|---|---|---|---|---|
| image_identity | 0.178 | 0.248 | **0.347** | 0.059 |
| image_change | 0.620 | 0.640 | **0.680** | 0.500 |
| running_speed_bin | 0.228 | 0.255 | **0.365** | 0.200 |
| pupil_diameter_bin | 0.252 | 0.300 | **0.381** | 0.200 |
| trial_outcome | 0.249 | 0.260 | **0.260** | 0.250 |

`dff` wins on every output, so the converted dataset uses `ds.dff_traces.dff`. Justification:
* The whitepaper's data-processing chain (methods.txt §DATA PROCESSING) **ends at the detrended
  dF/F trace** — that is the pipeline's neural output, and it is already computed in the released
  data (nothing for us to recompute).
* Piet et al. used *detected calcium events* for their event-based, per-image-presentation analyses.
  At the 32 ms ophys bin required here, `events` is 99.77 % zeros (≈0.6 non-zero samples per neuron
  per trial), which carries almost no per-timepoint information; this is exactly the case the task
  description covers with "except where … training a neural decoder require otherwise".
* Both `events` variants remain selectable with `--neural-signal` and the choice is recorded in
  `metadata['neural_signal']`.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing` → `/app/conversion_sample_out.txt`
(2 sessions: oeid 775614751 and 788490510, both mouse 403491).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 231 |
| Neurons / session | 89, 142 |
| Subjects | 1 (mouse 403491) |
| Trials (total) | 229 (200 go + 29 catch) |
| Trials / session | 39, 190 |
| T per trial | mean 254.9, min 225, max 389 |
| Bin size | 32.310 ms |
| Input dimension | 0 (no decoder inputs, as specified) |
| image_identity distribution | grey 0.665, 16 images 0.004–0.038 |
| image_change distribution | [0.973, 0.027] |
| running_speed_bin distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| pupil_diameter_bin distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| trial_outcome (timepoints) | hit 0.595, miss 0.274, FA 0.053, CR 0.078 |
| trial_outcome (trials) | hit 138, miss 62, FA 13, CR 16 |
| catch fraction | 0.1266 (whitepaper: ~12.5 %) ✓ |

### Processing Plots Review (`processing_775614751.png`, `processing_788490510.png`)
- **Image identity** steps up exactly on the cyan flash spans drawn straight from
  `ds.stimulus_presentations` and returns to 0 (grey) in the inter-stimulus intervals and on
  omitted flashes — no lag, no off-by-one frame.
- **Image change** is 1 exactly on the flash that contains `trials.change_time` (green line) and 0
  everywhere else; on the catch trial it is 0 for the whole trial while the green sham-change line
  still falls on a flash of the *unchanged* image. This is the behaviour the task asks for.
- **Running speed / pupil diameter**: the resampled ophys-rate trace lies on top of the raw 60 Hz /
  30 Hz trace with no visible shift; blink gaps (grey, NaN) are bridged smoothly.
- **Session panel**: the population mean dF/F is clearly flash-locked, and the go/catch trial
  windows tile the session with the change time inside each window — confirming the trials were cut
  from the right part of the session.
- **Discretisation panels**: the value→bin map is a monotone step function whose steps sit exactly
  on the printed quintile edges.
- Anomaly noted (not a bug): session 788490510 is a **non-runner** (99.5 % of samples |v| < 0.5 cm/s),
  so in this 2-session sample the global quintile edges collapse to ±0.2 cm/s. With the full dataset
  the edges are set by the many running sessions.

### Verification (`/app/verification_sample_out.txt`)
`Data format is valid, no errors or warnings.`

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| 16-process pool over sessions | 2.2 s → 0.55 s per session (4×) |
| single vectorised flash→frame mapping per session | avoids O(trials × flashes) work |
| float32 + `np.interp` | ~2× less memory traffic than float64/pandas |

| Step | Time / Session | Estimated Total Time (168 sessions) |
|---|---|---|
| Phase 1 read + extract (16 workers) | 0.55 s wall | ≈ 95 s |
| Phase 2 global quantiles | – | ≈ 5 s |
| Phase 3 assemble + self-checks | ≈ 0.1 s | ≈ 20 s |
| Pickle write (~8.3 GB) | – | ≈ 60 s |
| **Total** | | **≈ 3–5 min** (well under the 15 min budget) |

The estimate accounts for trial-count differences: the sample sessions hold 229 of the dataset's
43,387 trials, so the per-session cost was re-measured on a 12-session subset (0.54 s/session) that
spans the typical 39–409 trial and 6–666 neuron range.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`

### Format Validation
- Errors: **None**
- Warnings: **None**

### Training Progress
Loss decreases monotonically: 1.6378 (epoch 1) → 1.5192 (50) → 1.4119 (100) → 1.3488 (150) → 1.3124 (200).

### Decoder Results (Sample, 2 sessions / 231 neurons / 1 mouse)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-------------|--------|
| image_identity | 0.0588 | 0.4195 | **0.3077** |
| image_change | 0.5000 | 0.6996 | **0.6384** |
| running_speed_bin | 0.2000 | 0.2666 | **0.2457** |
| pupil_diameter_bin | 0.2000 | 0.3428 | **0.2481** |
| trial_outcome | 0.2500 | 0.4092 | **0.2766** |

Every output is above chance. The weak running/pupil numbers are a property of this particular
2-session sample (one of the two sessions never runs, and both come from the same mouse, so the
global quintile edges are degenerate here); they are re-examined on the full dataset in Step 11.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/converted_data.pkl --full` → `/app/conversion_full_out.txt`
Runtime **71.5 s** total (Phase 1 read+extract 58.7 s with 16 workers = 0.35 s/session; assemble 1.2 s;
pickle write 7.9 s). This is well inside the Step-7 estimate of 3–5 min, so no re-optimisation was needed.

### Output Files
- `converted_data.pkl`: **8.73 GB**
- `verification_full_out.txt`: created — `Data format is valid, no errors or warnings.`

### Sessions dropped during conversion
`795953296`, `806456687`, `833631914` — empty eye-tracking table (pupil diameter is a required
output). 168 → **165 sessions**, i.e. 917 trials and 276 cells not converted; every one of these
three drops was predicted in advance from the Step-2 scan, so nothing was lost silently.

### Consistency Check
"Reference Data" = recomputed directly from `project_metadata/*.csv` for the same 165 experiments.

| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | n/a (paper used a different subset) | `ophys_cells_table` | 28,821 | 28,821 | ✓ |
| Mean neurons/session | – | – | 174.7 (6–666) | 174.7 (6–666) | ✓ |
| Subjects | 82 mice in the whole release | – | 37 (this variant) | 37 | ✓ |
| Sessions | 376 in the whole release | – | 165 (this variant, active, eye-tracked) | 165 | ✓ |
| Trials (total) | – | `trials[go\|catch]` | 42,470 (37,143 go + 5,327 catch) | 42,470 | ✓ |
| Trials/session (mean) | – | – | 257.4 (39–409) | 257.4 (39–409) | ✓ |
| Catch fraction | **~12.5 %** (whitepaper) | – | 12.54 % | **12.54 %** | ✓ |
| hit / miss / FA / CR (trials) | – | `behavior_session_table` | 13,569 / 23,574 / 814 / 4,513 | identical | ✓ |
| Flash cycle | 250 ms image / 750 ms cycle | – | 7.74 frames mean per change | image_change runs 7–9 frames (mean 7.74 = 250 ms) | ✓ |
| Grey-screen fraction | 500/750 ms = 0.667 | – | – | 0.669 | ✓ |
| Ophys frame rate | 31 Hz single-plane | `metadata['ophys_frame_rate']` = 31.0 | median Δt 32.319 ms | bin size **32.319 ms** (range 32.310–32.331) | ✓ |
| image_change fraction | – | – | ≈0.87 go × 7.74/263 = 0.0256 | 0.0257 | ✓ |
| running_speed_bin | "five equal percentile bins" | – | – | [0.200 ×5] | ✓ |
| pupil_diameter_bin | "five equal percentile bins" | – | – | [0.200 ×5] | ✓ |
| image_identity | 8 images per session, 2 image sets | – | 16 distinct images | grey 0.669 + 16 images at 0.020–0.021 each | ✓ |

Running-speed quintile edges: **[0.043, 4.403, 22.636, 36.295] cm/s** (range −23.98 … 99.92).
Pupil-diameter quintile edges: **[75.42, 84.94, 94.18, 107.17] px** (range 16.93 … 480.86).

No data was lost beyond the three documented eye-tracking drops: every selected go/catch trial of
every kept session produced a trial (`n_trials_kept == n_trials_selected` for all 165 sessions),
and every cell in `dff_traces` became a row of the neural matrix.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1 — Output log verification (`/app/verification_full_out.txt`)
`Data format is valid, no errors or warnings.` — **zero errors, zero warnings**, so there is nothing
to leave unfixed. (The earlier `sklearn` "y_pred contains classes not in y_true" notice comes from
`train_decoder.py`'s scoring on the 2-session sample, not from the format check, and disappears on
the full dataset where every class is present in the test split.)

### Check 2 — Independent sanity checks (`/app/cache/sanity_checks.py`, output `/app/cache/sanity_checks_out.txt`)
The script re-opens each spot-checked session through the AllenSDK from scratch, recomputes the
expected values **without importing `convert_data.py`**, and compares with `np.allclose` /
`np.array_equal`. 5 randomly chosen sessions × 3 random trials per check.

| Check | Stream | What is compared | Result |
|---|---|---|---|
| A0 | – | `len(neural[s])` vs `len(trials[go\|catch])`; `n_neurons` vs `len(dff_traces)` | **PASS** (all 5) |
| A1 | **neural** | `neural[s][k][n,t]` and the whole `(n_neurons, T)` block vs `np.vstack(ds.dff_traces.dff)[:, i0:i1]`, `np.allclose(rtol=1e-6)` | **PASS** (15/15 trials, whole-trial allclose) |
| A2 | **input** | every trial's input is exactly `(0, T)` | **PASS** |
| A3a | **output** | image identity rebuilt frame-by-frame from `ds.stimulus_presentations` | **PASS** (15/15, 100 % of frames agree) |
| A3b | **output** | image change rebuilt from `is_change` flashes | **PASS** (15/15) |
| A3c | **output** | running-speed bin = `digitize(np.interp(t, raw_t, raw_v), edges)` | **PASS** (15/15) |
| A3d | **output** | pupil bin recomputed from `pupil_area` + blink mask | **PASS** (15/15) |
| A3e | **output** | trial outcome vs the `hit/miss/false_alarm/correct_reject` columns | **PASS** (15/15) |
| B1 | dataset | trials/session vs `behavior_session_table.go_trial_count + catch_trial_count` | **PASS** (165/165) |
| B2 | dataset | neurons/session vs `ophys_cells_table.csv` | **PASS** (165/165) |
| B3 | dataset | per-session hit/miss/FA/CR counts vs `behavior_session_table` | **PASS** (165/165) |
| C1 | dataset | every go trial has exactly one `image_change` episode | **PASS** (37,143 go trials, 0 anomalies) |
| C2 | dataset | no catch trial contains an image change | **PASS** (0 offenders) |
| C3 | dataset | the change episode lasts 7–9 frames (250 ms at 30.94 Hz); mean 7.74 | **PASS** |
| D | dataset | all neural finite; every output inside its declared range | **PASS** (42,470 trials) |
| E | dataset | `subject_idx`/`brain_region_idx` resolve to the right mouse/structure | **PASS** |

**0 FAILURES.**

### Check 3 — Reference code comparison
| Stage | Reference (AllenSDK / papers) | `convert_data.py` | Same? |
|---|---|---|---|
| (a) **Loading** | tutorials: `bpc.VisualBehaviorOphysProjectCache` → `get_ophys_experiment_table()` → `get_behavior_ophys_experiment(oeid)`; `np.vstack(dataset.dff_traces.dff.values)` | identical calls (`from_local_cache` instead of `from_s3_cache` because there is no network) | ✓ |
| (b) **Neuron filtering** | whitepaper §ROI FILTERING + demixing: performed upstream; SDK exposes only valid ROIs (`valid_roi` all True) | none added | ✓ |
| (b) **Trial filtering** | `trial.py:_get_trial_data` defines `go`/`catch` to exclude aborted and auto-rewarded | `trials[trials.go | trials.catch]`, with assertions that no selected trial is aborted or auto-rewarded | ✓ |
| (b) **Session filtering** | papers use only active-behaviour sessions for trial-based analysis; Piet et al. additionally restrict to multi-plane + familiar | active-behaviour + `project_code == 'VisualBehavior'`; **not** restricted to familiar/multi-plane | **deliberate difference**, see below |
| (c) **Temporal alignment** | whitepaper §DATA SYNCHRONIZATION: all streams recorded on one 100 kHz sync board, SDK returns every stream on that clock; tutorial plots `ds.ophys_timestamps` against `ds.running_speed.timestamps` and `ds.eye_tracking.timestamps` directly | resample running/pupil onto `ds.ophys_timestamps` with `np.interp`; map flash and trial boundaries with `np.searchsorted` on the same clock | ✓ |
| (d) **Binning** | ophys frame = the native bin (31 Hz); Piet et al. additionally aggregate into 750 ms image-presentation intervals for their per-flash analyses | native ophys frame, no re-binning (the task requires alignment on ophys timestamps) | ✓ / intentional |
| (e) **Input construction** | – | none (task specifies no decoder inputs) | ✓ |
| (f) **Output construction** | `image_name` / `is_change` / `is_sham_change` from `stimulus_presentations`; `speed` from `running_processing`; `pupil_area` + `likely_blink` from `eye_tracking`; outcome flags from `trials` | same fields, nothing recomputed from raw signals | ✓ |

**Deliberate differences and why**
1. *Neural signal `dff` rather than `events`* — justified in Step 6 (the whitepaper's processing chain
   ends at detrended dF/F; `events` is 99.77 % zeros at a 32 ms bin, which the decoder cannot use;
   measured accuracy is higher on every output).
2. *No familiar-only / multi-plane-only restriction* — Piet et al.'s restriction serves their novelty
   control. Applying it to this cache would leave 22 imaging planes from **one** mouse, which cannot
   support a cross-session decoder. All experience levels and both image sets are kept and recorded
   per session in `metadata['session_info']`.
3. *Sessions without eye tracking dropped* — the task requires pupil diameter as an output.

### Check 4 — Key statistics comparison
See the Step-9 table: every statistic available in the reference texts, code and metadata tables
(catch fraction 12.5 %, flash cadence 250/750 ms, 31 Hz frame rate, 8 images per session, 2 image
sets, trial-length distribution, per-session hit/miss/FA/CR counts, per-session cell counts) matches
the converted data. The one paper statistic that does **not** apply is Piet et al.'s cell/session/mouse
counts (8,619 exc / 470 Sst / 1,239 Vip), which describe the multi-plane familiar-only subset that is
not present in this cache; the corresponding local counts (27,895 / 396 / 806 across 168 planes) were
verified against `ophys_cells_table.csv` instead.

### Check 5 — Edge cases examined
| Edge case | Finding | Action |
|---|---|---|
| Trials with no ophys coverage | none (`n_trials_kept == n_trials_selected` everywhere); guard kept in the code | none needed |
| Off-by-one at trial boundaries | trial frames are `start_time <= t < stop_time` (half-open, `searchsorted(..., 'left')`); consecutive trials never share a frame, verified by the whole-trial `allclose` spot checks | none needed |
| Off-by-one at flash boundaries | flash frames are `start_time <= t < end_time`; the independent per-frame rebuild in Check 2 agrees on 100 % of frames | none needed |
| `is_change` on auto-rewarded trials | `is_change` is True for 229.7 flashes/session but only 225.9 go trials — the difference is exactly the 3.8 auto-rewarded trials/session. Those changes never fall inside a go/catch trial window, and Check C1/C2 confirm one change per go trial and none per catch trial | none needed |
| Omitted flashes | `image_name == 'omitted'`, duration 0.25 s; treated as grey screen (the screen really is grey) | by design |
| Blinks (`pupil_area` NaN) | 5.5 % of eye frames on average, max 29.6 % in one session | linearly interpolated |
| Sessions with no eye tracking | 3 | dropped, documented |
| Extreme pupil values | dataset max 480 px vs typical session max ~120–160 px; spot-checked 10 sessions: `>2×` the session median occurs in ≤0.03 % of frames | left in — percentile binning is insensitive to a handful of outliers, and the SDK provides no additional validity flag beyond `likely_blink` |
| Extreme running speeds | up to ~100 cm/s in some sessions; the SDK already removes z>10 transients and 10 Hz low-passes (whitepaper §BEHAVIOR) | left in |
| Sessions with very few cells | min 6 neurons; the decoder's projection layer handles `nneurons < npcs` | kept |
| Sessions with <2 trials | none (min 39) | guard kept in the code |
| Sessions missing an outcome class | 2 sessions have only 2 of the 4 outcomes, 12 have 3; all 165 have ≥2 trials and ≥2 classes overall | kept — the decoder is shared across sessions |

### Issues Found and Resolved
- *(Step 6/7, before the full run)* `show_processing` read `ds.events[NEURAL_SIGNAL]` after the
  neural signal was switched to `dff`, which lives in `ds.dff_traces` — fixed, plots regenerated.
- *(Step 6)* The `--change-window presentation_interval` variant initially measured the next flash
  onset within the non-omitted flash table, which stretched the window to 1.5 s when the flash after
  a change was omitted — fixed to use the full flash block. (This option is only used for the
  experiment in Step 12; the shipped dataset uses the 250 ms flash window.)
- No issue was found that required re-running the full conversion after it completed; all Step-10
  checks passed on the first full artefact.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` → `/app/train_decoder_full_out.txt`
(165 sessions, 42,470 trials, 11,192,974 timepoints; trained on 33,908 trials, tested on 8,562;
NVIDIA L4, 200 epochs, ~9 min).

### Training Progress
- Loss decreasing: **Yes**, monotonically — 1.6407 (ep 1) → 1.6229 (10) → 1.6032 (20) → 1.5298 (50) →
  1.4159 (100) → 1.3488 (150) → 1.3125 (200). Test loss 1.4670.
- Format check before training: `Data format is valid, no errors or warnings.`

### Decoder Results (Full)
| Output | #classes | Chance (1/K) | Training Balanced Acc | Validation Balanced Acc | Val / chance |
|--------|---|---|-------------|--------|-------|
| image_identity | 17 | 0.0588 | 0.4284 | **0.3996** | **6.80×** |
| image_change | 2 | 0.5000 | 0.6685 | **0.6233** | 1.25× |
| running_speed_bin | 5 | 0.2000 | 0.3817 | **0.3628** | 1.81× |
| pupil_diameter_bin | 5 | 0.2000 | 0.4656 | **0.4437** | 2.22× |
| trial_outcome | 4 | 0.2500 | 0.4372 | **0.2952** | 1.18× |

`sample_trials.png` and `predictions.png` were written. In `predictions.png` the middle (input) row
is empty, as it must be for a 0-dimensional input, and the ground-truth (solid) / prediction (dashed)
traces for outputs 0–4 are plotted against each other.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1 — Accuracy vs chance
Every output is **above** chance; none is below. Two are below the 1.5× guideline and were
investigated in depth.

| Output | Val acc | Chance | Ratio | Verdict |
|---|---|---|---|---|
| image_identity | 0.3996 | 0.0588 | 6.80× | fine |
| running_speed_bin | 0.3628 | 0.2000 | 1.81× | fine |
| pupil_diameter_bin | 0.4437 | 0.2000 | 2.22× | fine |
| image_change | 0.6233 | 0.5000 | 1.25× | **investigated — see below** |
| trial_outcome | 0.2952 | 0.2500 | 1.18× | **investigated — see below** |

**image_change.** The 1.5× rule cannot apply to a binary variable (it would demand 0.75 balanced
accuracy). Three alternative explanations were tested and rejected:
1. *Wrong label window?* Re-converted 40 sessions with the change labelled over the full 750 ms
   image-presentation interval instead of the 250 ms flash (Piet et al.'s "image presentation
   interval"). Result: image_change **0.6174** vs **0.6462** for the 250 ms flash window — the
   shipped definition is the better one, and it is also the one consistent with how image identity
   is coded. Kept the 250 ms window.
2. *Misalignment?* Ruled out by the Step-10 frame-by-frame rebuild (100 % agreement) and by the
   `--show-processing` panels, where the change label sits exactly on the flash containing
   `trials.change_time`.
3. *Too little signal in the converted activity?* Ruled out by the paper replication in Check 2 —
   the same converted trials support a 0.78–0.80 change-vs-repeat decoder under the paper's own
   protocol. The 0.62 figure is simply the harder per-32 ms-frame version of the problem, where the
   negative class also contains every grey-screen frame and every image repeat.

**trial_outcome.** `/app/cache/diagnose_outcome.py` re-trains the reference decoder on a 37-session
subset and re-scores output 4 as a function of time relative to the (sham) change:

| window rel. change (s) | balanced acc |
|---|---|
| −4.0 … −3.5 | 0.310 |
| −2.0 … −1.5 | 0.307 |
| −0.5 … 0.0 | 0.284 |
| **0.0 … 0.5** | **0.342** |
| 0.5 … 1.0 | 0.331 |
| 2.5 … 3.0 | 0.314 |
| pre-change overall (238,424 pts) | 0.298 |
| post-change overall (240,123 pts) | 0.306 |

So the outcome is weakly but consistently decodable throughout the trial, with a clear peak in the
0–1 s after the change — exactly the time when the animal's lick/reward (or its absence) occurs.
This is the expected shape, not a bug: (i) the trial window is the experiment's own trial, more than
half of which precedes the change, when the outcome has not yet happened; and (ii) the residual
pre-change signal is the engagement/arousal state that predicts hit-vs-miss, which is the central
claim of the reference paper. The value could be raised by cutting trials to a short post-change
window, but that would contradict the instruction to segment trials "based on how they are defined
in the experiment".

### Check 2 — Accuracy comparison to the papers
The reference paper reports two decoders (methods.txt §"Decoding analysis"): a **change decoder**
(change vs. the immediately preceding repeat) and a **hit decoder** (hit vs. miss on image changes),
both random forests with 5-fold CV over per-image-presentation population activity, summarised per
imaging plane. These are a different problem from `decoder.py` (per-flash vs. per-32 ms-frame;
per-plane random forest vs. a shared linear read-out on 100 session PCs), so rather than dismiss the
difference, the paper's protocol was **re-implemented on the converted data**
(`/app/cache/replicate_paper_decoders.py`, 40 imaging planes ≥20 neurons):

| Decoder | Paper (Piet et al.) | This conversion (mean ± SEM over planes) |
|---|---|---|
| change vs. repeat, excitatory | ≈0.75–0.85 (highest of the three classes) | **0.797 ± 0.017** (37 planes) |
| change vs. repeat, Vip | ≈0.55–0.65 (lowest) | **0.589 ± 0.034** (3 planes) |
| change vs. repeat, all | ≈0.6–0.85 | **0.782 ± 0.018** |
| hit vs. miss | ≈0.55–0.70 | **0.771 ± 0.022** (38 planes) |

The converted data reproduces the paper's numbers, including the cell-class ordering
(excitatory ≫ Vip for change decoding). The hit decoder comes out at the top of / slightly above the
paper's range; the two expected reasons are that this conversion uses dF/F rather than events (see
Step 6) and that these are single-plane VISp recordings with up to 666 simultaneously recorded cells,
whereas the paper's per-plane multi-plane populations are smaller. Both push accuracy up, and neither
indicates a conversion error.

| `decoder.py` output | Achieved (val.) | Expectation from the papers |
|---|---|---|
| image_identity | 0.3996 (17 classes, chance 0.059) | not reported; V1 image identity is strongly decodable — 6.8× chance is consistent |
| image_change | 0.6233 per 32 ms frame | 0.78 for the paper's easier per-flash balanced version, reproduced above |
| running_speed_bin | 0.3628 | not reported; the paper documents strong running modulation of V1 (Fig. S16) |
| pupil_diameter_bin | 0.4437 | not reported |
| trial_outcome | 0.2952 per 32 ms frame (4 classes) | 0.77 for the paper's per-change hit-vs-miss version, reproduced above |

### Check 3 — Train vs validation gap
| Output | Train | Val | Train/Val |
|---|---|---|---|
| image_identity | 0.4284 | 0.3996 | 1.07 |
| image_change | 0.6685 | 0.6233 | 1.07 |
| running_speed_bin | 0.3817 | 0.3628 | 1.05 |
| pupil_diameter_bin | 0.4656 | 0.4437 | 1.05 |
| trial_outcome | 0.4372 | 0.2952 | **1.48** |

No output exceeds the 1.5× threshold. The largest gap is `trial_outcome`, which is the only *static*
output: every one of a trial's ~263 timepoints carries the same label, so the model can memorise a
training trial's whole time course from a single constant. That is ordinary overfitting on ~34k
labelled trials, not leakage — the train/validation split is by **trial**, no trial appears on both
sides, and the quantile edges (the only globally fitted quantity) are computed from the data
distribution, not from labels, and are identical for train and test trials.

### Additional experiment: per-neuron z-scoring (rejected)
Z-scoring each neuron's dF/F within a session before PCA was tested on the same 37-session subset:
image_identity 0.4367 (vs 0.4247), image_change 0.6428 (0.6462), running 0.3451 (0.3561),
pupil 0.3353 (0.3605), outcome 0.2790 (0.3031) — worse on 4 of 5 outputs, and it is a transform
neither reference describes. **Not applied**; the shipped data is the Allen pipeline's dF/F as-is
(itself already normalised by each cell's baseline fluorescence).

### Issues Found and Resolved
- *image_change window*: tested 250 ms vs 750 ms, kept 250 ms (better and more consistent). Resolved.
- *trial_outcome low accuracy*: shown by the time-resolved diagnostic to be an expected property of
  the full-trial window, not a bug. No change.
- No re-conversion was required; all Step-10 checks still pass on the shipped
  `/app/converted_data.pkl` (re-verified after the Step-12 experiments, which wrote only to
  `/app/cache/`).

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created — dataset description, how to load, full output-format specification,
      key statistics, decoder results, file inventory.
- [x] `cache/` folder created with `cache/README_CACHE.md` documenting the five investigation
      scripts (`scan.py`, `sanity_checks.py`, `diagnose_outcome.py`, `replicate_paper_decoders.py`,
      `make_zscore.py`) and their outputs. The 5.6 GB of intermediate comparison pickles used for
      the Step-6/Step-12 experiments were deleted (regenerable from `convert_data.py` flags).
- [x] All required artefacts present in `/app`:
      `CONVERSION_NOTES.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`,
      `README.md`, `conversion_sample_out.txt`, `verification_sample_out.txt`,
      `train_decoder_sample_out.txt`, `conversion_full_out.txt`, `verification_full_out.txt`,
      `train_decoder_full_out.txt`, plus `processing_775614751.png`, `processing_788490510.png`,
      `sample_trials.png`, `predictions.png`.

### Summary of every decision made
| Decision | Choice | Why |
|---|---|---|
| Which experiments | `project_code == 'VisualBehavior'`, active behaviour | the named dataset variant in the whitepaper and the only project fully present locally; passive sessions have no trial outcomes |
| Multiscope planes in the cache | excluded | one mouse only, and merging 3–7 planes with different frame clocks into one "session" is not supported by the target format |
| Session = | one imaging plane / NWB file | the VisualBehavior project is single-plane, so session ≡ experiment |
| Neural signal | `dff_traces.dff` | whitepaper's pipeline output, already computed; event traces are ≥96 % zeros at 32 ms; measured best on every output |
| Neuron filtering | none added | Allen ROI filtering already applied; `valid_roi` all True |
| Trials | `trials[go | catch]`, `[start_time, stop_time)` | the experiment's own trial definition; excludes aborted/auto-rewarded by construction |
| Time base | `ophys_timestamps`, no re-binning | task requires alignment on ophys timestamps; bin constant at 32.319 ms |
| Inputs | none, `(0, T)` arrays | task specifies no decoder inputs |
| Image identity | 250 ms flash → image; ISI and omissions → `grey` | "the image presented during the non-grey screen" |
| Image change | 1 for the 250 ms changed-image presentation | "right after a change"; tested against a 750 ms window, which decoded worse |
| Running / pupil bins | **global** quintiles over all converted timepoints | makes a bin mean the same thing in every session and yields exactly 20 % per bin dataset-wide |
| Pupil diameter | `2·√(pupil_area/π)`, blinks linearly interpolated | blink frames are NaN in the SDK; interpolation keeps T aligned with the neural data (percentile bins are invariant to the monotone area→diameter transform) |
| Sessions without eye tracking | dropped (3) | pupil diameter is a required output |
| Trial outcome | 4 classes, broadcast over T | the format allows only one array per trial, so a static label must be broadcast |
| Per-neuron z-scoring | not applied | not described by either reference and measured worse on 4 of 5 outputs |
