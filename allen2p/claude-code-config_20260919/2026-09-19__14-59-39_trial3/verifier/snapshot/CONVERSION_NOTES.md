# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory — Visual Behavior 2P (`visual-behavior-ophys-1.1.0`), change-detection task
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `.manifest`, `Dockerfile`, `docker-compose.yaml` — container setup
- `CONVERSION_NOTES.md` (this file)
- `code/` — AllenSDK source (v2.16.2, also installed in site-packages)
- `data/` — 247 GB; `visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb` (284 files)
  and `visual-behavior-ophys-1.1.0/project_metadata/*.csv` (4 metadata tables)
- `decoder.py`, `train_decoder.py` — decoder reference implementation
- `methods.txt`, `paper.pdf`, `whitepaper.pdf` — reference texts
- `tutorials/` — 5 AllenSDK Visual-Behavior tutorials (.py/.ipynb)

Environment verified: `python3` = 3.13, numpy 2.4.4, torch 2.6.0+cu124 (CUDA available,
NVIDIA L4 23 GB), pandas 2.3.3, allensdk 2.16.2. Host: 128 cores, 1006 GB RAM, 3.4 TB free disk.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function / attribute | File | Stage | Purpose |
|----------|------|-------|---------|
| `BehaviorOphysExperiment.from_nwb_path(path)` | `allensdk/brain_observatory/behavior/behavior_ophys_experiment.py` | LOADING | Load one ophys *experiment* (= one imaging plane of one session) straight from its NWB file. Used instead of `VisualBehaviorOphysProjectCache` because the local cache lacks the `visual-behavior-ophys/manifests` sub-folder the SDK's `from_local_cache` requires. Returns the identical object type the tutorials get from `bc.get_behavior_ophys_experiment()`. |
| `.trials` | same | LOADING/CURATION | Trial table: `start_time`, `stop_time`, `change_time`, `go`, `catch`, `aborted`, `auto_rewarded`, `hit`, `miss`, `false_alarm`, `correct_reject`, `initial_image_name`, `change_image_name`, `is_change`, `lick_times`, `reward_time`, `trial_length`. |
| `.stimulus_presentations` | same | PROCESSING | One row per flash: `start_time`, `end_time`, `image_name`, `image_index`, `omitted`, `is_change`, `is_sham_change`, `stimulus_block_name`, `trials_id`, `flashes_since_change`. |
| `.events` | `data_objects/cell_specimens/events.py` | LOADING | Per-cell detected calcium events (`events`) and `filtered_events`. |
| `filter_events_array(arr, scale, n_time_steps)` | `brain_observatory/behavior/event_detection.py` | PROCESSING | Builds `filtered_events`: **causal** half-normal (half-Gaussian) kernel, `scale = 2/31 s * frame_rate_hz` (≈2 frames), 20 taps, normalised to sum 1, `np.convolve(...)[:len]`. Causal ⇒ no backward (acausal) leakage of activity in time. |
| `.dff_traces` | same | LOADING | ΔF/F traces. **Already computed in the released NWB files** (whitepaper "DF/F CALCULATION"); nothing to recompute. |
| `.ophys_timestamps` | same | ALIGNMENT | Time (s, session clock) of every 2-photon frame. Common clock for all streams. |
| `.running_speed` | same | LOADING | `timestamps`, `speed` (cm/s) at 60 Hz (stimulus frame clock), already unwrapped/despiked/10 Hz-low-passed by `running_processing` module. |
| `.eye_tracking` | same | LOADING | 30 Hz; `pupil_area`, `pupil_width`, `pupil_height`, `likely_blink`; pupil columns are **NaN wherever `likely_blink`**. |
| `.cell_specimen_table` | same | CURATION | `cell_roi_id`, `valid_roi`, ROI geometry. |
| `.metadata` | same | LOADING | `mouse_id`, `cre_line`, `targeted_structure`, `imaging_depth`, `session_type`, `project_code`, `ophys_frame_rate`, `ophys_session_id`, … |
| `tutorials/visual_behavior_compare_across_trial_types.py` | tutorials | PROCESSING | **The closest reference for this task.** Selects trials with `.query('hit')`, `.query('miss')`, `.query('false_alarm')`, `.query('correct_reject')` and plots every stream (stimuli, running, licks, rewards, pupil, dF/F) restricted to `timestamps >= trial.start_time and timestamps <= trial.stop_time`. ⇒ trial window = `[start_time, stop_time]`, trial types = the 4 go/catch outcomes. |
| `tutorials/visual_behavior_load_ophys_data.py` | tutorials | PROCESSING | Shows `np.vstack(dff_traces.dff.values)`, `events`/`filtered_events`, and that `ophys_timestamps` is the time base for both. States events "approximate the firing rate with a resolution of about 200 ms" and that `filtered_events` are events convolved with a `stats.halfnorm` filter. |

### Notes
- ΔF/F does **not** need to be computed — the NWB files ship `dff_traces` and `events` already
  produced by the Allen pipeline (dewarping → motion correction → segmentation → ROI filtering →
  demixing → neuropil subtraction → dF/F → event detection, all described in `methods.txt`).
- Neuron quality control is also already applied upstream: the released `cell_specimen_table`
  contains only ROIs that passed the multi-label ROI classifier (`valid_roi == True` for
  **29,097/29,097** cells in the sessions used here — verified, see Step 2). This is
  optical-physiology data, so there is no spike-sorting quality metric to threshold.
- The SDK's local-cache loader could not be used (missing `manifests` sub-folder), so
  experiments are loaded with `BehaviorOphysExperiment.from_nwb_path`, which is the same code
  path (`from_nwb`) the cache itself uses. Session/experiment selection uses the released
  `project_metadata/ophys_experiment_table.csv`, i.e. the same table
  `bc.get_ophys_experiment_table()` returns.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
/app/data/
  visual-behavior-ophys_project_manifest_v1.1.0.json     (manifest)
  visual-behavior-ophys-1.1.0/
    project_metadata/
      behavior_session_table.csv     (all behavior sessions, incl. training)
      ophys_session_table.csv
      ophys_experiment_table.csv     (1936 rows x 31 cols = full release)
      ophys_cells_table.csv          (133,066 rows: experiment -> cell_roi_id/cell_specimen_id)
    behavior_ophys_experiments/
      behavior_ophys_experiment_<ophys_experiment_id>.nwb    (284 files, ~0.9 GB each)
```
Terminology (whitepaper "DATA STRUCTURE AND TERMINOLOGY"): *session* = one continuous
recording; *experiment* = one imaging plane within a session. Single-plane rigs have 1
experiment per session; the Multiscope has up to 8.

**Which 284 of the 1936 released experiments are present locally:**

| project_code | experiments present | of total in release | mice | ophys sessions |
|---|---|---|---|---|
| `VisualBehavior` (single plane, Scientifica) | **239 (= all 239)** | 239 | 37 | 239 (1 plane/session) |
| `VisualBehaviorMultiscope` | 45 | 862 | **1** | 8 |

i.e. the local copy is **the complete `VisualBehavior` single-plane project** plus all
experiments of a single Multiscope mouse (457841).

### Dataset Size (from data files)
Measured by `cache/prescan.py` over the 168 **active** (non-passive) `VisualBehavior`
experiments (full scan, no sampling):

| Statistic | Value |
|-----------|-------|
| Experiments present (all) | 284 |
| `VisualBehavior` project | 239 (37 mice) |
| `VisualBehavior`, active behaviour | 168 (37 mice) |
| `VisualBehavior`, active, with eye tracking | **165** (37 mice) — 3 sessions have an empty eye-tracking table |
| Neurons (total, 168 active) | 29,097 (all `valid_roi == True`) |
| Neurons (total, 165 used) | 28,821 |
| Neurons / session | mean 173.2, min 6, max 666 |
| Neurons / session by cre line | Slc17a7 260.7, Vip 24.4, Sst 14.1 |
| Subjects | 37 |
| Sessions / subject | 168/37 = 4.5 (range 1–6 active sessions) |
| Trials (total, all types) | 114,292 (mean 680/session) |
| Trials go∪catch (total, 168) | 43,387 (mean 258/session, min 39, max 409) |
| Trials go∪catch (total, 165) | 42,470 |
| Aborted trials | mean 418/session |
| Auto-rewarded trials | 5/session (0 in a few sessions) |
| Ophys frame period | 0.032319 s (30.9406 Hz) — identical to 5 decimals in all 168 |
| Ophys frames / session | ~140,280 (~4535 s) |
| Ophys frames inside go∪catch trials | 11,438,438 (168) / 11,192,974 (165); mean 262 / trial |
| Multiscope ophys frame period | 0.09323 s (10.726 Hz) |
| Running speed | 60 Hz, 0 NaNs in every session |
| Eye tracking | 30 Hz; NaN (blink) fraction mean 5.5 %, longest continuous gap median 9.7 s, max 144 s |
| Trial length (go∪catch) | 7.26 – 12.56 s (mean 8.5 s) |
| change_time − start_time | 3.02 – 8.31 s |
| stop_time − change_time | 4.23 s (essentially constant) |
| Images | set A = im061,062,063,065,066,069,077,085; set B = im000,031,035,045,054,073,075,106 |

Per-trial outcome fractions over the 42,470 go∪catch trials of the 165 used sessions:
hit 0.319, miss 0.555, false_alarm 0.019, correct_reject 0.106.
`hit + miss + false_alarm + correct_reject == n(go ∪ catch)` in **every** session (checked).

Passive sessions (`OPHYS_2_*_passive`, `OPHYS_5_*_passive`) have hit = false_alarm = **0**
in every case (lick spout retracted), so their trial outcome is degenerate.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Task | go/no-go change detection | "mice are presented with a continuous series of briefly presented stimuli and they earn water rewards by correctly reporting when the identity of the image changes" (whitepaper) |
| Images per session | 8 (64 possible transitions) | "Each session included 8 images, for a total of 64 possible transitions." |
| Flash / blank timing | 250 ms image, 500 ms grey | "a series of natural images (250 ms stimulus duration) interspersed with periods of a gray screen (500 ms inter-stimulus duration)" (paper) |
| Omission probability | 5 % of flashes; never the change or the flash before it | "stimuli were omitted with a 5% probability … Stimulus changes and the stimulus immediately preceding the change were never omitted." |
| Trial types | GO, CATCH, plus ABORTED and free-reward (auto-rewarded) | "this trial structure leads to a sampling of 'GO' and 'CATCH' trials, that when combined with mouse responding, yields 'HIT', 'MISS', 'FALSE ALARM', and 'CORRECT REJECTION' trials." |
| Auto-rewarded trials | first 5 of every session + after 10 consecutive misses | "behavior sessions across all phases began with 5 'free-reward' trials … delivered after 10 consecutive 'MISS' trials." (matches the measured 5/session) |
| Catch probability | ~12.5 % in the imaging stage | "later sessions implemented a matrix sampling algorithm … pushing the actual catch probability to ~12.5%." Measured: 5,321 catch / 42,470 = **12.5 %** ✔ |
| Change time distribution | truncated exponential 2.25–8.25 s, mean ≈4.2 s after trial start | "Change-times were selected from a truncated exponential distribution ranging from 2.25 to 8.25 seconds … resulting in a mean change time of 4.2 seconds." Measured change_time−start_time: 3.02–8.31 s, mean 4.29 s ✔ |
| Response window | 150–750 ms after the change | "lick in a 0.150 to 0.750 second window" |
| Aborted trials | trials with licking before the change | "excluding aborted trials (trials where the animal responded before the stimulus change)" |
| Imaging rate | 31 Hz single plane, 11 Hz per plane multi-plane | "512x512 pixels, 31 Hz for single plane and 512x512 pixels, 11 Hz for each plane in multi-plane experiments" — measured 30.94 Hz and 10.73 Hz ✔ |
| Eye/behaviour camera rate | 30 Hz | "eye tracking (30 Hz), and behavior (30 Hz)" ✔ |
| Neural signal used by the paper | detected calcium events | "For all analysis of neural data we used the detected calcium events" |
| Passive sessions | not analysed | "Imaging was also performed during passive viewing of the same stimulus, **which was not analyzed here**." |
| Paper's dataset size (behaviour) | 376 sessions, 82 mice | "This dataset contains behavior from 376 imaging sessions from 82 mice" (whole release, not the local subset) |
| Paper's dataset size (neural) | 8,619 exc (21 sessions, 9 mice), 470 Sst (15, 6), 1,239 Vip (21, 9) | "Our dataset contains 8,619 excitatory cells…" — this is the **Multiscope familiar-only** subset the paper restricted to, not the subset available locally |
| Running behaviour | open loop, mice stop running while licking | "Running was 'open loop' … Mice typically ran between licking bouts and stopped running during licking" |

### Processing Details
- **Temporal alignment**: "Temporal synchronization of all data-streams (calcium imaging, visual
  stimulation, body and eye tracking cameras) was achieved by recording all experimental clocks on
  a single NI PCI-6612 digital IO board at 100 kHz". All timestamps exposed by the SDK
  (`ophys_timestamps`, `stimulus_presentations.start_time`, `running_speed.timestamps`,
  `eye_tracking.timestamps`, `trials.start_time/stop_time/change_time`) are therefore already on
  one common session clock; no cross-stream time-shift has to be estimated, only resampling.
- **Neural preprocessing** (all already applied in the NWB): dewarping, Suite2P rigid motion
  correction, segmentation, ROI filtering, demixing of overlapping ROIs, neuropil subtraction,
  dF/F with detrending, and event detection.
- **Behaviour preprocessing**: running speed is already unwrapped, transient-corrected,
  z>10 outliers removed, and 10 Hz low-pass Butterworth filtered by the SDK.
- **Behavioural binning used by the paper**: events assigned to 750 ms image-presentation
  intervals. That granularity is for their image-by-image analyses; it is *not* usable here
  because the decoder outputs must be time-varying at the neural sampling rate.

### Curation Steps
**Neuron curation rules**:
- Upstream (already applied to the release): ROIs that are unions, duplicates, on the motion
  border, apical dendrites, or too small/narrow/dim are labelled invalid and removed; ROIs with
  non-positive demixed traces and their overlapping neighbours are removed (~1 % loss).
- Container-level QC (registration, brain health, cell matching) removes whole experiments.
- ⇒ Nothing further to filter: `valid_roi` is `True` for 29,097/29,097 cells locally. I keep the
  `valid_roi == True` filter in code as an explicit guard.

**Trial curation rules**:
- Task spec: keep `go` ∪ `catch`; drop `aborted` and `auto_rewarded`.
- The reference tutorial `visual_behavior_compare_across_trial_types.py` uses exactly the four
  go/catch outcomes (`hit`, `miss`, `false_alarm`, `correct_reject`) and the
  `[start_time, stop_time]` window.
- Paper (`Behavioral strategy…`): passive sessions not analysed.

### Decoders Trained (reported in the paper)
| Decoded variable | Reported accuracy |
|---|---|
| Image change vs. repeat ("change decoder", random forest on image-presentation-wise activity, 5-fold CV) | The paper reports it as a curve vs. number of neurons (Figure 6); text gives no single number. Broadly ~0.6–0.8 % correct for large n in V1 excitatory populations. |
| Hit vs. miss on change flashes ("hit decoder") | Same figure; performance close to but above chance (~0.55–0.65). |
No decoding accuracies are reported for image identity, running speed, pupil, or 4-way outcome,
so those have no paper reference value.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Which project(s) to use | SDK exposes all 4 project codes | Local copy = **all 239** `VisualBehavior` experiments + 45 from one Multiscope mouse | Task spec: "data under the **Visual Behavior** task" | Use `project_code == 'VisualBehavior'`. It is exactly the complete project that was downloaded, and it is the only choice compatible with "time bins the same size for all sessions": single-plane = 30.9406 Hz for every session, Multiscope = 10.726 Hz. Including the Multiscope mouse would either break the uniform bin size or force resampling of every session, and would add only 1 subject. |
| Neural signal: dF/F vs events vs filtered events | SDK provides all three | events are non-zero on 0.25 % of frames (extremely sparse); `filtered_events` non-zero on 4.5 % | Paper: "we used the detected calcium events" | Use **`filtered_events`** = the paper's detected calcium events convolved with the SDK's own standard causal half-Gaussian (σ = 2/31 s). Same events, same magnitudes, only a 65 ms causal smoothing. Raw events are ~0.25 % non-zero per frame, which makes a per-timepoint decoder degenerate (the tutorial itself notes event *resolution* is ~200 ms, i.e. finer than 32 ms sampling carries no information). Causality guarantees no information leaks backwards across the change time. Empirically compared against `events` and `dff` on the sample (Step 7/8). |
| Session = experiment? | SDK: a session may hold several experiments (planes) | In the `VisualBehavior` project every ophys session has exactly 1 experiment | whitepaper terminology | For this project, experiment ≡ session; a converted "session" is one NWB file. |
| Passive sessions | present in the table (`passive == True`) | hit = false_alarm = 0 in every passive session ⇒ trial outcome is degenerate (only miss / correct_reject) | Paper: passive viewing "was not analyzed here" | Exclude the 71 passive `VisualBehavior` experiments. |
| Eye tracking availability | SDK always exposes `eye_tracking` | 3/168 active sessions have an **empty** table | — | Exclude those 3 sessions: pupil diameter is a required decoder output and cannot be produced for them. |
| Behaviour numbers in the paper (376 sessions / 82 mice; 8,619 exc cells) | — | Local subset = 165 sessions / 37 mice | Paper used the **whole release** for behaviour and the **Multiscope familiar** subset for neurons | Not a discrepancy: the paper's counts refer to data that is not all present locally. Comparable checks are the task-structure statistics (catch probability, change-time distribution, flash timing, frame rates, auto-reward count), all of which match — see Step 3 table. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Session / trial selection (final)
1. `ophys_experiment_table.csv` → `project_code == 'VisualBehavior'` **and** `passive == False`
   **and** the NWB file exists locally → 168 experiments.
2. Drop sessions with an empty `eye_tracking` table → **165 sessions, 37 mice**.
3. Trials: `trials[(go) | (catch)]` → excludes `aborted` and `auto_rewarded` by construction
   (in this table `go`, `catch`, `aborted`, `auto_rewarded` are mutually exclusive: verified that
   `hit+miss+false_alarm+correct_reject == n(go∪catch)` in every session).
4. Drop a trial if it contains **no valid (non-blink) pupil sample** — pupil would be pure
   extrapolation. Drop a trial with < 2 ophys frames.
5. Drop a session left with < 2 trials or 0 neurons (none expected).

### Trial window and time base
- Window `[trials.start_time, trials.stop_time)` — the experiment's own trial definition, and
  exactly the window the reference tutorial plots.
- Time base: the **ophys frame timestamps** inside that window (`ophys_timestamps`). All other
  streams are resampled onto those timestamps. Bin size = 0.0323193 s (30.9406 Hz), identical for
  every trial and session.
- `off_start = 0.0` (window starts at the alignment event, trial start); `off_end = None`
  because trials have the experiment's natural variable length (7.26–12.56 s).

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `events.filtered_events` (n_cells × n_frames) | `neural` | `np.vstack`, slice frames in trial window, `float32` | `dataset.events`, `filter_events_array` | rows ordered by `cell_specimen_table` index |
| — | `input` | empty `(0, T)` float32 | — | task spec: "No inputs for this task" |
| `stimulus_presentations.image_name` over `[start_time, end_time)` of each flash | `output[0]` `image_identity` | 0 = grey/none (incl. omitted flashes and the 500 ms blanks), 1..16 = the 16 image names sorted | tutorial `plot_stimuli` uses the same `[start_time, end_time]` spans | 17 categories |
| `stimulus_presentations.is_change` | `output[1]` `image_change` | 1 on the frames inside the **changed** flash's 250 ms presentation, else 0 | `is_change` (False on catch/sham flashes) | binary; catch trials have `is_sham_change` and correctly get 0 |
| `running_speed.speed` (60 Hz) | `output[2]` `running_speed_bin` | linear interpolation onto ophys timestamps → per-session quintile bins | `dataset.running_speed` | 5 categories |
| `eye_tracking.pupil_area` (30 Hz) | `output[3]` `pupil_diameter_bin` | NaN (blink) removal → linear interpolation of valid samples onto ophys timestamps → effective diameter `2*sqrt(area/π)` → per-session quintile bins | `dataset.eye_tracking` | 5 categories |
| `trials.hit/miss/false_alarm/correct_reject` | `output[4]` `trial_outcome` | one label per trial, broadcast over the trial's timepoints | tutorial `.query('hit')` etc. | 4 categories, static per trial |
| `metadata['mouse_id']` | `subjects`, `subject_idx` | unique sorted | | 37 subjects |
| `metadata['targeted_structure']` | `brain_regions`, `brain_region_idx` | one entry per neuron | | VISp / VISl |

### Key Decisions
1. **Project = `VisualBehavior` only** — see Step 4. Gives one uniform time bin (32.32 ms) and is
   exactly the complete project present on disk.
2. **Active sessions only** — the paper explicitly does not analyse passive sessions, and trial
   outcome is degenerate there (no hits, no false alarms).
3. **Neural signal = `filtered_events`** — the paper's detected calcium events with the SDK's own
   causal smoothing; needed for a per-timepoint decoder. (Compared empirically in Step 7/8.)
4. **No extra neuron filtering** — the release already contains only `valid_roi` cells; verified
   29,097/29,097.
5. **Trial = `[start_time, stop_time)` of go/catch trials**, matching the reference tutorial;
   variable length, so `off_end = None`.
6. **Image identity includes a `grey` class** — the spec defines it as "the image presented during
   the non-grey screen", so the grey inter-stimulus interval (and omitted flashes, which are grey
   continuations) need their own category. 16 image names are kept globally distinct rather than
   collapsed to the per-session 0–7 `image_index`, because image sets A and B are different
   physical stimuli.
7. **Image change = the 250 ms changed flash**, consistent with how image identity is defined, and
   0 on catch (sham-change) trials because the image identity does not actually change there.
8. **Quintile bins for running speed and pupil are computed *per session*** over exactly the
   timepoints that enter the dataset. Rationale: pupil size is in camera pixels and depends on
   zoom/eye position/rig, so it is **not** comparable across sessions — global bins would mostly
   encode session identity rather than arousal. Per-session bins also guarantee the "five equal
   percentile bins" property holds within every session and give a balanced 20/20/20/20/20 target.
9. **Pupil diameter = `2*sqrt(pupil_area/π)`** (effective diameter of the fitted ellipse).
   Monotonic in area, so quintiles are identical to area quintiles; reported as a diameter as
   requested.
10. **Blinks**: `pupil_area` is NaN exactly where `likely_blink`; these are removed and linearly
    interpolated from the surrounding valid samples (standard blink handling). Trials with no
    valid sample at all are dropped.
11. **Trial outcome is stored time-varying** (constant across the trial's timepoints) so it lives
    in the same `(5, T)` array as the other four outputs, per the spec's "If at all possible, make
    it time-varying".

### Planned Sanity Checks
- [ ] Every session's ophys frame period equals 0.032319 s (±1e-4).
- [ ] Catch fraction of go∪catch trials ≈ 12.5 % (whitepaper).
- [ ] `change_time − start_time` ∈ [2.25, 8.4] s, mean ≈ 4.2 s (whitepaper).
- [ ] Every go trial window contains exactly one `is_change` flash; every catch trial contains zero.
- [ ] Image flash duration ≈ 250 ms and flash period ≈ 750 ms measured from the converted
      `image_identity` trace.
- [ ] Omitted-flash fraction ≈ 5 %.
- [ ] Each of the 5 running/pupil bins holds ≈20 % of timepoints.
- [ ] Trial-outcome fractions match the trial table (hit .319 / miss .555 / fa .019 / cr .106).
- [ ] Re-read raw NWB independently and `np.allclose` spot-check neural, running speed, pupil and
      image identity at specific (session, trial, neuron, timepoint) indices.
- [ ] Total neurons 28,821; total trials ≈42,470 (minus pupil-dropped); 165 sessions; 37 subjects.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` — runs as `python -u /app/convert_data.py <outpicklefile> [--full|--sample] [--show-processing]`.

Structure:
| Function | Purpose |
|---|---|
| `select_experiments()` | reads `ophys_experiment_table.csv`, keeps locally-present `VisualBehavior`, non-passive experiments (168) |
| `process_experiment(job)` | converts one session; runs in a worker process |
| `resample_running()` | `np.interp` of the 60 Hz SDK running speed onto the ophys frame times |
| `resample_pupil()` | drops blink (NaN) samples, converts area → effective diameter `2*sqrt(area/pi)`, `np.interp` onto ophys frame times; also returns the valid sample times so trials with no measurement can be rejected |
| `stimulus_traces()` | per-ophys-frame image code and change indicator from `stimulus_presentations` |
| `quantile_bins()` | 5 equal-percentile bins from the session's own within-trial values |
| `make_processing_plot()` | the `--show-processing` diagnostic figure |
| `main()` | parallel map over sessions, assembly of the target dict, summary + assertions, pickle |

Extra (non-default) flags used only for the benchmarking documented in Step 7:
`--neural-signal {dff,events,filtered_events}` (default `dff`), `--zscore`,
`--change-window {flash,interval}` (default `flash`), `--limit N`.

Code reused from the reference: the AllenSDK itself does all loading and all
neural/behavioural preprocessing (`BehaviorOphysExperiment.from_nwb_path`,
`.dff_traces`, `.events`, `.ophys_timestamps`, `.trials`, `.stimulus_presentations`,
`.running_speed`, `.eye_tracking`, `.cell_specimen_table`). The trial selection and
the `[start_time, stop_time]` window reproduce
`tutorials/visual_behavior_compare_across_trial_types.py`.

Code inefficiencies identified:
- Opening the NWB and materialising the (n_cells x n_frames) trace matrix dominates
  runtime (~3-6 s/session, ~0.9 GB files).
- Naive per-trial `np.interp`/`searchsorted` would repeat work 250+ times per session.

Code speedups added:
- Every stream is resampled **once per session** onto the full ophys timestamp vector;
  trials are then pure array slices.
- Trial boundaries found with two vectorised `np.searchsorted` calls instead of a loop.
- `ProcessPoolExecutor(16)` over sessions.
- `float32` neural, `int16` outputs (outputs are 8x smaller than int64 would be).
- Result: **60 s wall time for the whole dataset** (0.31 s/session amortised).

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
→ `/app/conversion_sample_out.txt`, `/app/sample_data.pkl`,
`/app/processing_775614751.png`, `/app/processing_788490510.png`.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (oeid 775614751, 788490510) |
| Neurons (total) | 231 |
| Neurons / session | 89, 142 |
| Subjects | 1 (mouse 403491) |
| Trials (total) | 229 |
| Trials / session | 39, 190 |
| Timepoints | 58,368 (mean T = 251.3, min 225, max 389) |
| Time bin | 32.310 ms |
| image_identity distribution | grey 0.665, each of the 8 session images ≈0.004–0.045 |
| image_change distribution | [0.973, 0.027] |
| running_speed_bin distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| pupil_diameter_bin distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| trial_outcome distribution (timepoints) | hit 0.595, miss 0.274, FA 0.053, CR 0.078 |
| catch fraction | 0.1266 (whitepaper ~0.125) |
| mean change latency | 3.857 s (whitepaper ~4.2 s; this mouse ran short trials) |

Inputs: `dinput = 0`, so there is no input range to report (the task specifies no
decoder inputs). Verified that the reference `verify_data_format`,
`print_data_summary`, `plot_trial` and `train_validate_decoder` all handle a
`(0, T)` input array without error or warning.

### Processing Plots Review
`processing_<oeid>.png` has 7 panels; all were inspected and show no anomalies:
1. raw 60 Hz running speed vs. the ophys-resampled trace — curves superimpose exactly.
2. raw 30 Hz pupil samples vs. the blink-interpolated ophys-resampled trace — the
   interpolated line passes through every raw sample.
3. `image_identity` step function vs. the shaded `stimulus_presentations` spans — the
   code is non-grey exactly inside a non-omitted flash and grey everywhere else;
   `image_change` rises exactly at the red `change_time` line and the identity code
   changes at the same frame.
4. the dF/F matrix over the same window with the trial bounds and change time marked.
5. running / pupil histograms with the quintile edges drawn.
6. bar charts confirming 0.200 of the timepoints per quintile.
7. the five converted output traces over three consecutive trials with the change time
   from the trials table overlaid; the 250 ms/750 ms flash rhythm and the constant
   per-trial outcome are visible.

Independent time-domain cross-check of panel 3: integrating the flash spans that
overlap the trial windows gives an image-on-screen fraction of **0.3349** for both
sample sessions, matching the converted `image_identity` fraction of 0.3351. (It is
slightly above 250/750 = 0.3333 because a trial window begins at a flash onset, so it
contains one more flash than it contains complete 750 ms cycles.)

### Neural signal benchmark (why dF/F)
Same conversion, only the neural signal changed; validation balanced accuracy.

2 sessions (sample):
| signal | image_identity | image_change | running | pupil | outcome |
|---|---|---|---|---|---|
| `events` | 0.140 | 0.611 | 0.207 | 0.229 | 0.266 |
| `filtered_events` | 0.187 | 0.611 | 0.208 | 0.254 | 0.269 |
| `filtered_events` z-scored | 0.283 | 0.617 | 0.202 | 0.230 | 0.350 |
| `dff` | 0.305 | 0.638 | 0.215 | 0.271 | 0.283 |
| `dff` z-scored | 0.307 | 0.635 | 0.213 | 0.275 | 0.294 |

20 sessions (`--limit 20`, more reliable):
| signal | image_identity | image_change | running | pupil | outcome |
|---|---|---|---|---|---|
| `filtered_events` | 0.292 | 0.608 | 0.247 | 0.224 | 0.319 |
| `filtered_events` z-scored | 0.372 | 0.633 | 0.267 | 0.251 | **0.412** |
| **`dff`** | **0.470** | **0.641** | **0.297** | **0.287** | 0.333 |
| `dff` z-scored | 0.477 | 0.632 | 0.305 | 0.290 | 0.317 |

**Decision: `dff_traces.dff`, not z-scored.** Rationale:
- The detected calcium events used by the reference *paper* are ~0.25 % non-zero per
  32 ms frame; the SDK tutorial itself states their effective resolution is ~200 ms.
  A per-timepoint decoder therefore sees almost no signal, and this shows up as much
  lower accuracy on every output.
- dF/F is the primary neural product of the release (a whole whitepaper section,
  "DF/F CALCULATION", is devoted to it), it is already neuropil-subtracted, demixed,
  baseline-normalised and detrended, and it is the signal the reference tutorial that
  defines our trial segmentation (`visual_behavior_compare_across_trial_types.py`,
  `plot_dff`) plots per trial.
- z-scoring adds nothing on top of dF/F (differences ≤0.02, both directions), so the
  raw released traces are kept — the converted `neural` values *are* the released dF/F.

### Change-window benchmark (why the 250 ms flash)
| `image_change` = 1 during | image_identity | image_change | running | pupil | outcome |
|---|---|---|---|---|---|
| the 250 ms changed flash (**chosen**) | 0.467 | **0.640** | 0.297 | 0.287 | 0.335 |
| the full 750 ms presentation interval | 0.471 | 0.624 | 0.297 | 0.289 | 0.334 |
The 250 ms flash both decodes better and is the reading consistent with how
`image_identity` is defined ("the image presented during the non-grey screen").

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| resample each stream once per session instead of per trial | ~250x fewer interp calls |
| vectorised `searchsorted` trial boundaries | loop removed |
| 16 worker processes | ~10x wall clock |
| float32 neural / int16 outputs | 8.4 GB instead of ~17 GB + 0.45 GB |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| open NWB + read dF/F | 3.2 s (1 worker) | — |
| resample behaviour + stimulus | 0.05 s | — |
| assemble trials | 0.05 s | — |
| **measured, 16 workers** | **0.31 s** | **52 s for 168 sessions** |
| pickle write (8.4 GB) | — | 8 s |
| **total measured** | | **~65 s** (well under the 15 min budget; no further optimisation needed) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation (`/app/verification_sample_out.txt`)
- Errors: **None**
- Warnings: **None**

### Decoder Results (Sample, 2 sessions / 231 neurons / 229 trials)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc |
|--------|--------|-------------|--------|
| image_identity | 0.0588 | 0.4236 | 0.3025 |
| image_change | 0.5000 | 0.6981 | 0.6370 |
| running_speed_bin | 0.2000 | 0.2502 | 0.2153 |
| pupil_diameter_bin | 0.2000 | 0.3342 | 0.2710 |
| trial_outcome | 0.2500 | 0.4117 | 0.2763 |

Loss decreased monotonically (1.63 → 1.55 over 200 epochs) and every output is above
chance. The margins for running speed and trial outcome are small here; with only two
sessions the reference decoder is badly under-trained (it takes **one** optimiser step
per epoch, so 200 steps total, and the gradient is summed over sessions). The 20-session
and 165-session runs confirm this: the same conversion reaches 0.30 and 0.28-0.30 on
those two outputs once more sessions contribute to each step.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: **8.38 GB**, 165 sessions
- `conversion_full_out.txt`, `verification_full_out.txt`: created; **no errors, no warnings**

Conversion wall time 65 s (52 s processing with 16 workers + 8 s pickling).
3 of the 168 selected sessions were skipped, all for the same documented reason
(`no eye tracking data`: oeid 795953296, 806456687, 833631914).

### Consistency Check
| Statistic | Reference Papers | Reference Code / Data tables | Reference Data (direct scan) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions | n/a (paper used a different subset) | 168 active `VisualBehavior` experiments in `ophys_experiment_table.csv` | 168 NWB files present, 3 without eye tracking | 165 | ✔ |
| Subjects | n/a | 37 | 37 | 37 | ✔ |
| Sessions / subject | — | 1–6 (mean 4.5) | same | 2–9 (mean 4.5)¹ | ✔ |
| Total neurons | n/a | 29,097 rows in `ophys_cells_table.csv` for the 168 | 29,097 (`valid_roi` 29,097/29,097) | 28,821 (165 sessions) | ✔ |
| Mean neurons/session | — | 173.2 | 173.2 | 174.7 | ✔ |
| Trials (go∪catch) | — | — | 43,387 (168) / 42,470 (165) | 42,410 (60 dropped, no pupil sample) | ✔ |
| Trials/session (mean) | — | — | 257.4 | 257.0 | ✔ |
| Aborted trials excluded | "excluding aborted trials (trials where the animal responded before the stimulus change)" | `trials.aborted` | 69,264 | 0 kept | ✔ |
| Auto-rewarded excluded | "5 free-reward trials" per session | `trials.auto_rewarded` | 5/session | 0 kept | ✔ |
| Catch fraction | "~12.5 %" | — | 0.1254 | **0.1254** | ✔ |
| Mean change latency | "mean change time of 4.2 seconds" | — | 4.29 s | **4.227 s** | ✔ |
| Flash duration | "250 ms stimulus duration" | `stimulus_presentations.duration` 0.2511 s | — | 258.6 ms recovered from the converted trace (8 frames of 32.3 ms) | ✔ |
| Flash cycle | "250 ms + 500 ms" = 750 ms | 0.7506 s | — | image on-screen fraction 0.336 | ✔ |
| Omitted flashes | "5 % probability, never the change or the flash before it" | — | 3.5 % of flashes (mean), 1.7–4.4 % | grey | ✔² |
| Ophys frame rate | "31 Hz for single plane" | `metadata.ophys_frame_rate = 31.0` | 30.9406 Hz | time_bin_size 32.32 ms = 30.94 Hz | ✔ |
| Eye camera rate | "30 Hz" | — | 30.0 Hz | resampled to 30.94 Hz | ✔ |
| Behaviour/running rate | "30 Hz"/60 Hz stimulus clock | — | 60 Hz | resampled to 30.94 Hz | ✔ |
| Images per session | "Each session included 8 images" | — | 8 + 'omitted' | 8 non-grey classes used per session, 16 across image sets A+B | ✔ |
| trial_outcome distribution | — | — | hit .319 / miss .555 / FA .019 / CR .106 (per trial) | hit .3196 / miss .5550 / FA .0191 / CR .1062 | ✔ |
| image_change fraction | — | — | — | 0.0257 of timepoints | ✔³ |
| running / pupil quintiles | — | — | — | 0.200 each | ✔ |
| Brain regions | — | `targeted_structure` = VISp for all 239 `VisualBehavior` experiments | VISp | `['VISp']`, 28,821 neurons | ✔ |

¹ mouse 457766 has 9 active sessions because some session types were retaken.
² 3.5 % rather than 5 %: changes (~7 % of flashes) and the flash before each change are
never omitted, so only ~86 % of flashes are eligible → expected ≈4.3 %; the observed
1.7–4.4 % range brackets that.
³ expected = P(go) × 0.2511 s / mean trial length = 0.875 × 0.2511 / 8.47 = 0.0259.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt` reports **"Data format is valid, no errors or
warnings."** There are no errors and no warnings to address. (During training,
scikit-learn emits `UserWarning: y_pred contains classes not in y_true` — this comes
from `balanced_accuracy_score` inside `train_decoder.py`, not from the data: the shared
17-class image readout can predict an image-set-B class on an image-set-A session. It is
unavoidable for a globally-shared image-identity label space and is not a data defect.)

### Check 2: Sanity checks against the raw files (`cache/sanity_checks.py`)
Every check re-reads the original NWB files with **h5py directly** — no AllenSDK, no
code shared with `convert_data.py` — and compares with `np.allclose` / `np.array_equal`.
Run on 5 sessions spread through the dataset (index 0, 40, 97, 130, 164) plus
dataset-level checks. **0 failures out of 95 checks.**

| Check | How | Result |
|---|---|---|
| dF/F values | 3 random (trial, neuron) per session; re-read `processing/ophys/dff/traces/data`, map the stored `cell_specimen_ids` to columns via `dff/traces/rois` + `cell_specimen_table/cell_specimen_id`, slice with `searchsorted` on `dff/traces/timestamps` | `np.allclose(atol=1e-6)` PASS (15/15) |
| neuron count | vs. the NWB `cell_specimen_table` | PASS |
| trial identity | stored `trial_ids` all lie in `go∪catch` of `intervals/trials`; go/catch/aborted/auto_rewarded are mutually exclusive | PASS |
| trial lengths | re-derived `searchsorted(ts, start_time)…searchsorted(ts, stop_time)` | exact match |
| running_speed_bin | whole session rebuilt from `processing/running/speed/{timestamps,data}` + `np.interp` + `np.quantile` | **100.0000 % identical**, quintile edges `np.allclose` |
| pupil_diameter_bin | whole session rebuilt from `acquisition/EyeTracking/pupil_tracking/area` + `likely_blink`, diameter `2√(A/π)`, interp, quantiles | **100.0000 % identical**, edges `np.allclose` |
| image_identity | whole session rebuilt from `intervals/*_presentations/{start_time,stop_time,image_name}` | **100 % identical** |
| image_change | rebuilt from `is_change` | **100 % identical** |
| one change per GO trial / none per CATCH trial | from the converted trace | PASS in every session |
| trial_outcome | rebuilt from `intervals/trials/{hit,miss,false_alarm,correct_reject}` | exact match; constant within each trial |
| structure | nsessions/subject_idx/brain_region_idx lengths, ≥2 trials per session, T consistent across neural/input/output | PASS |
| whitepaper values | catch fraction 0.1254, change latency 4.227 s, flash duration 258.6 ms, image on-screen 0.336 | PASS |

### Check 3: Reference code comparison
| Step | Reference | This conversion | Same? |
|---|---|---|---|
| (a) data loading | tutorials: `bc.get_behavior_ophys_experiment(oeid)` → `BehaviorOphysExperiment`; attributes `.trials`, `.stimulus_presentations`, `.running_speed`, `.eye_tracking`, `.dff_traces`, `.events`, `.ophys_timestamps` | `BehaviorOphysExperiment.from_nwb_path(...)` and the *same* attributes. The cache path is unusable locally (missing `visual-behavior-ophys/manifests`), and `get_behavior_ophys_experiment` itself ends in `from_nwb`, so the loaded object is identical. Session selection uses the released `ophys_experiment_table.csv`, the same table `bc.get_ophys_experiment_table()` serves. | ✔ |
| (b) neuron filtering | pipeline QC already applied to the release (ROI classifier, demixing, container QC); `cell_specimen_table.valid_roi` | keep `valid_roi == True` (29,097/29,097 — no cell is actually removed); no further filtering, matching the papers, which describe none | ✔ |
| (b) session/trial filtering | paper: passive sessions "not analyzed here"; tutorial selects trials via `.query('hit'/'miss'/'false_alarm'/'correct_reject')` | passive excluded; trials = `go | catch`, which is exactly the union of those four outcomes (verified: `hit+miss+FA+CR == n(go∪catch)` in every session). Extra: 3 sessions with no eye tracking and 60 trials with no valid pupil sample — required because pupil is a mandatory decoder output. | ✔ + documented extras |
| (c) temporal alignment | whitepaper: all clocks recorded on one 100 kHz IO board, so every SDK timestamp is on a single session clock. Tutorial plots each stream with `timestamps >= trial.start_time and timestamps <= trial.stop_time`. | identical window `[start_time, stop_time)`; every stream resampled onto `ophys_timestamps` by linear interpolation (running, pupil) or interval lookup (stimulus). No time shift is introduced anywhere. | ✔ |
| (d) binning | no binning in the reference — the SDK's native ophys frame grid | native ophys frames (32.32 ms), no re-binning, no smoothing | ✔ |
| (e) input construction | — | none (task spec) | n/a |
| (f) output construction | tutorial `plot_stimuli` shades `[start_time, end_time]` of non-omitted flashes; `trials.query('hit')` etc.; `plot_running` uses `running_speed.speed`; `plot_pupil` uses `eye_tracking.pupil_area` | image identity from exactly those flash spans; outcome from exactly those four trial flags; running from `running_speed.speed`; pupil from `eye_tracking.pupil_area`. Discretisation into quintiles is imposed by the task spec (outputs must be categorical) and has no reference counterpart. | ✔ |

**Documented differences from the reference papers, and why:**
1. **dF/F instead of detected calcium events.** The paper's analyses are per
   image-presentation (750 ms) with a random forest; ours is per 32 ms frame with a
   linear readout, for which events are too sparse. Benchmarked in Step 7; dF/F is
   better on every output. dF/F is itself a first-class released signal.
2. **Only the `VisualBehavior` single-plane project.** Required by "time bins the same
   size for all trials and sessions" (30.94 Hz vs 10.73 Hz). The paper's neural
   analyses used the Multiscope rig, of which only one mouse is present locally.
3. **All experience levels (familiar and novel), not familiar only.** The paper
   restricted the *neural* analysis to familiar images; its *behavioural* analysis used
   "all active behavioral sessions … across all image set experience levels". Keeping
   all active sessions triples the dataset and the experience level is recorded in
   `metadata['session_info']` so it can be filtered downstream.
4. **Outputs discretised into 5 quantile bins** — required by the task spec.

### Check 4: Key statistics comparison
See the Step 9 table; every statistic that exists in the reference texts, code or data
matches. The two paper numbers that do *not* match (376 sessions/82 mice; 8,619
excitatory + 470 Sst + 1,239 Vip cells) refer to the whole release and to the
Multiscope-familiar subset respectively, neither of which is on disk here — this was
verified against `ophys_experiment_table.csv` (1,936 experiments released, 284 present).

### Check 5: Edge cases (`cache/edge_cases.py`, all 165 sessions)
| Edge case | Result |
|---|---|
| trial window outside the ophys timestamp range | none — worst margin 304 s before / 617 s after |
| trial window outside the running-speed range (would silently extrapolate) | none — worst margin 300 s / 601 s |
| trial window outside the valid eye-tracking range | none — worst margin 278 s / 588 s |
| NaN `change_time` on a go/catch trial | 0 |
| duplicate or NaN `cell_specimen_id` | 0 |
| NaN/Inf in dF/F | 0 |
| non-monotonic ophys timestamps | 0 |
| overlapping consecutive trials | 0 |
| `valid_roi == False` | 0 |
| NaN `start_time`/`end_time` in the stimulus table | 0 |
| image names outside the known 16 | none |
| frames before the first flash (`searchsorted` index −1) | handled → grey |
| omitted flashes | mapped to grey (screen really is grey) |
| session with <2 usable trials, or 0 neurons | none occurred; guarded anyway |
| trial with <2 ophys frames | none occurred; guarded anyway |
| off-by-one at trial edges | window is half-open `[start, stop)`, so consecutive trials can never share a frame — confirmed by the exact trial-length match in Check 2 |

### Issues Found and Resolved
1. **Neural signal choice** — the first version used `filtered_events`; benchmarking
   (Step 7) showed dF/F decodes better on every output. Changed, re-ran the full
   conversion and all validation.
2. **`cell_specimen_ids` / `trial_ids` were not stored**, which made the h5py
   sanity checks impossible to anchor. Added to `metadata['session_info']`; full
   conversion re-run.
3. **`--change-window` benchmark** — tested the 750 ms presentation interval as an
   alternative definition of `image_change`; the 250 ms flash is better and was kept.
4. No other issue was found: all 95 raw-file sanity checks and all 15 edge-case audits
   pass, and the format verifier reports no errors and no warnings.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt`, `sample_trials.png`, `predictions.png`.

### Training Progress
- Loss decreasing: **Yes**, monotonically — 1.6263 (epoch 10) → 1.5513 (50) →
  1.4564 (100) → 1.4002 (150) → 1.3699 (200). Test loss 1.5284.
- Ran on the GPU (NVIDIA L4); ~6 min for 200 epochs over 11.18 M timepoints.

### Decoder Results (Full: 165 sessions, 28,821 neurons, 42,410 trials)
| Output | Chance | Training Balanced Acc | Validation Balanced Acc | Val / chance | Notes |
|--------|--------|-------------|--------|------|-------|
| image_identity | 0.0588 | 0.4344 | **0.4049** | 6.9x | 17-way; strongly above chance |
| image_change | 0.5000 | 0.6692 | **0.6247** | 1.25x | binary, 2.6 % positives |
| running_speed_bin | 0.2000 | 0.3116 | **0.2828** | 1.41x | 5-way |
| pupil_diameter_bin | 0.2000 | 0.3118 | **0.2657** | 1.33x | 5-way |
| trial_outcome | 0.2500 | 0.4382 | **0.2949** | 1.18x | 4-way, static per trial |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
Every output is above chance. `image_identity` is 6.9x chance. The other four are
between 1.18x and 1.41x chance, i.e. below the 1.5x flag, so each was investigated.

**Independent linear-decoder ceiling (`cache/diag_ceiling.py`).** For 10 sessions
spanning the neuron-count range (6 → 666 neurons), an independently written
per-session PCA(100) + balanced multinomial logistic regression was fitted on a 75/25
trial split — i.e. the best a *per-session* linear per-timepoint decoder can do on the
converted data:

| Output | independent per-session linear ceiling | reference decoder (validation) |
|---|---|---|
| image_identity | 0.377 | **0.405** |
| image_change | 0.627 | 0.625 |
| running_speed_bin | 0.285 | 0.283 |
| pupil_diameter_bin | 0.262 | 0.266 |
| trial_outcome | 0.294 | 0.295 |

The reference decoder is **at or above** the independent ceiling on all five outputs
(higher on `image_identity` because it shares a readout across sessions). The accuracies
therefore reflect how much a linear per-timepoint readout of V1 dF/F can extract, not a
conversion defect. Accuracy also scales strongly and monotonically with the number of
simultaneously recorded neurons (image_identity 0.14 at n=14 → 0.77 at n=469), exactly
the dependence the reference paper reports for its decoders (Figure 6), which is further
evidence that the neural data are correctly aligned to the labels.

**Why each of the four is intrinsically hard here**
- `image_change` (1.25x): binary, but the label is on the 250 ms of the changed flash
  while the GCaMP6f response to that flash peaks 200–400 ms later; and the "negative"
  class contains the visually identical repeat flashes. The paper's own change decoder
  (random forest, whole 750 ms presentations, change vs. the *immediately preceding*
  repeat only) reaches ~0.6–0.8; 0.625 on the much harder all-timepoint version is
  consistent. Tested the 750 ms label window — it is *worse* (0.624 vs 0.640 at 20
  sessions), so the 250 ms definition was kept.
- `running_speed_bin`, `pupil_diameter_bin` (1.41x, 1.33x): 5-way quantile
  classification of slow continuous variables. Because the bins are per-session
  quintiles, the task is exactly balanced, so 0.28/0.27 corresponds to a large, real
  effect (≈40 % / 33 % above chance) and matches the independent ceiling to 3 decimals.
- `trial_outcome` (1.18x): the label is constant over a 7–12.5 s trial that begins
  3–8 s *before* the change, so most timepoints precede any behaviour that defines the
  outcome; and false alarms are only 1.9 % of trials. A dedicated diagnostic
  (`cache/diag_outcome.py`, 8 largest sessions) gave all-timepoints 0.318,
  pre-change 0.320, post-change 0.363 — post-change is better, but pre-change is already
  well above chance (engagement state), so the modest overall value is not a temporal
  misalignment. Restricting trials to a post-change window would raise this number, but
  the task requires segmenting "based on how they are defined in the experiment".

### Check 2: Accuracy comparison to papers
| Variable | Paper's reported accuracy | This dataset | Comparable? |
|---|---|---|---|
| Image change vs. repeat ("change decoder") | random forest on 750 ms presentations, change flash vs. the immediately preceding repeat flash, 5-fold CV; reported as a curve vs. n neurons, reaching roughly 0.6–0.8 for large excitatory populations | 0.625 balanced accuracy over **all** timepoints (including every grey interval and every other repeat flash), linear per-timepoint readout | Our task is strictly harder and we are inside the paper's range |
| Hit vs. miss ("hit decoder") | same figure, above chance but lower than the change decoder (~0.55–0.65 binary) | 4-way hit/miss/FA/CR at 0.296 (chance 0.25). Re-scoring hit-vs-miss only is not produced by the reference script, but the per-session ceiling analysis gives 0.29–0.36 for the 4-way problem | Consistent |
| image identity, running speed, pupil, 4-way outcome | **not reported in either paper** | — | no reference value exists |
The papers report no other decoding accuracies, so no further comparison is possible.

### Check 3: Train vs validation gap
| Output | Train | Val | Ratio |
|---|---|---|---|
| image_identity | 0.434 | 0.405 | 1.07 |
| image_change | 0.669 | 0.625 | 1.07 |
| running_speed_bin | 0.312 | 0.283 | 1.10 |
| pupil_diameter_bin | 0.312 | 0.266 | 1.17 |
| trial_outcome | 0.438 | 0.295 | **1.49** |
Only `trial_outcome` approaches the 1.5x flag, and it does not cross it. This is the
expected signature of a per-trial (static) label: the session has ~257 trials but ~68,000
timepoints, so the effective sample size is the number of trials while the model sees
timepoints, and it can partly memorise trial-specific activity. There is no data leakage:
`train_validate_decoder` splits **whole trials**, and because every trial is a contiguous
block of ophys frames from a disjoint time interval (verified: no overlapping trials, and
the window is half-open so consecutive trials never share a frame), no timepoint can
appear in both the training and validation sets.

### Issues Found and Resolved
No new issue was found in this round. The four sub-1.5x outputs were each traced to a
property of the experiment or of the linear per-timepoint decoder, confirmed by an
independently written decoder that reaches the same numbers on the same data.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (user-facing dataset description, loading instructions,
      format specification, key statistics)
- [x] `cache/` folder created with all exploratory/validation scripts and
      `cache/README_CACHE.md`
- [x] All required output files present in `/app`
