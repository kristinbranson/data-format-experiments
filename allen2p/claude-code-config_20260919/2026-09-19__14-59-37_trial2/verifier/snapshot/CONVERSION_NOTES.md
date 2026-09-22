# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory — Visual Behavior 2P (`visual-behavior-ophys-1.1.0`), in `/app/data`
- **Date started**: 2026-09-19
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents (`/app`):
- `CONVERSION_NOTES.md` (this file), `Dockerfile`, `docker-compose.yaml`, `.manifest`
- `code/` — AllenSDK v2.16.2 source (reference code)
- `data/` — `visual-behavior-ophys-1.1.0/` (NWB files + project metadata csvs), manifest json
- `tutorials/` — 5 tutorial scripts + 1 notebook on loading VBO data
- `methods.txt`, `paper.pdf` (Piet et al. 2024, Neuron), `whitepaper.pdf` (VBO 2P technical whitepaper)
- `decoder.py`, `train_decoder.py` — decoder reference implementation

Environment verified: `python3` with numpy 2.4.4, torch 2.6.0+cu124 (CUDA L4 GPU, 23 GB),
pandas 2.3.3, allensdk 2.16.2 (installed *and* available as source in `/app/code`).
Machine: 128 CPUs, 1 TB RAM.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

The reference code base is the AllenSDK (`/app/code/allensdk`). The data are distributed as
per-experiment NWB files; the SDK's `BehaviorOphysExperiment` object is the canonical loader and
is what both tutorials use. All processing (motion correction, segmentation, ROI filtering,
demixing, neuropil subtraction, dF/F, event detection, running-speed processing, eye tracking) is
already applied upstream and materialized in the NWB; the SDK simply exposes it.

### Key Functions Identified
| Function / attribute | File | Stage | Purpose |
|----------------------|------|-------|---------|
| `BehaviorOphysExperiment.from_nwb_path(path)` | `allensdk/brain_observatory/behavior/behavior_ophys_experiment.py` | LOADING | Canonical loader of one imaging plane (= one "experiment") from an NWB file |
| `VisualBehaviorOphysProjectCache.from_s3_cache` + `get_ophys_experiment_table()` | `behavior_project_cache.py` | LOADING | Table of all experiments/metadata. Offline equivalent: `data/.../project_metadata/ophys_experiment_table.csv` |
| `.dff_traces` | `data_objects/cell_specimens/dff_traces.py` | PROCESSING | Per-cell dF/F (already computed by the pipeline: median-filter baseline, detrended — methods.txt "DF/F CALCULATION") |
| `.events` | `data_objects/cell_specimens/events.py` | PROCESSING | Per-cell detected calcium events (`events`) and `filtered_events` = events convolved with a **causal** half-normal kernel (`event_detection.filter_events_array`, `scale = 2/31 s`, 20 time steps). Also `lambda`, `noise_std` |
| `.ophys_timestamps` | `data_objects/timestamps/ophys_timestamps.py` | LOADING | Time (s, sync clock) of every 2P frame; identical for `dff_traces` and `events` |
| `.cell_specimen_table` | `data_objects/cell_specimens/cell_specimens.py` | CURATION | One row per **valid** ROI (`valid_roi`), indexed by `cell_specimen_id`; ROI filtering/QC already applied by the pipeline |
| `.trials` | `data_objects/trials/trials.py` | LOADING | Trial table: `start_time`, `stop_time`, `change_time`, `initial_image_name`, `change_image_name`, `go`, `catch`, `aborted`, `auto_rewarded`, `hit`, `miss`, `false_alarm`, `correct_reject`, `lick_times`, `reward_time`, `response_latency` |
| `.stimulus_presentations` | `data_objects/stimuli/presentations.py` | LOADING | One row per flash: `start_time`, `end_time`, `image_name`, `image_index`, `is_change`, `is_sham_change`, `omitted`, `trials_id`, `flashes_since_change`, `stimulus_block_name` |
| `.running_speed` | `data_objects/running_speed/running_speed.py`, `running_processing.py` | PROCESSING | Linear running speed (cm/s) at ~60 Hz on `stimulus_timestamps`; already unwrapped, transient-corrected and 10 Hz low-pass filtered (`raw_running_speed` = unfiltered) |
| `.eye_tracking` | `data_objects/eye_tracking/eye_tracking_table.py`, `eye_tracking_processing.py` | PROCESSING | Ellipse fits at ~30 Hz. `pupil_width`/`pupil_height`/`pupil_area` are set to **NaN** on `likely_blink` frames (missing fit or \|z\|>3 outlier, dilated by 2 frames) |
| `filter_events_array` | `behavior/event_detection.py` | PROCESSING | Causal half-gaussian smoothing used to build `filtered_events` |
| `.metadata` | — | LOADING | `ophys_frame_rate`, `targeted_structure`, `cre_line`, `mouse_id`, `ophys_session_id`, `session_type`, `project_code`, `imaging_depth`, ... |

### Notes
- dF/F does **not** need to be computed by us — it is precomputed in the NWB (`dff_traces`), as are
  the detected calcium events. Neuropil subtraction, demixing and detrending are already applied.
- Neuron ("cell") quality filtering is also already applied upstream: `cell_specimen_table` contains
  only ROIs that passed the multi-label ROI classifier (`valid_roi == True` for every row in all 199
  surveyed experiments; `len(cell_specimen_table) == len(events) == len(dff_traces)` everywhere), and
  every `cell_specimen_id` is a valid positive id (no -1/NaN).
- Tutorials (`/app/tutorials/visual_behavior_load_ophys_data.py`,
  `visual_behavior_compare_across_trial_types.py`) show the canonical alignment recipe: select a time
  window from the `trials` table (`start_time` … `stop_time`) and slice every stream by its own
  timestamps (`ophys_timestamps` for dff/events, `running_speed.timestamps`,
  `eye_tracking.timestamps`). They label `pupil_width` as "pupil diameter".
- Tutorials restrict the stimulus table to `stimulus_block_name.str.contains('change_detection')`,
  i.e. the behavior block, excluding the 5-min gray screens and the 5-min natural movie.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
data/visual-behavior-ophys-1.1.0/
  behavior_ophys_experiments/behavior_ophys_experiment_<ophys_experiment_id>.nwb   (284 files, 247 GB)
  project_metadata/ophys_experiment_table.csv   (1936 rows = full release; 31 columns)
                   ophys_session_table.csv      (703 rows)
                   behavior_session_table.csv   (4782 rows; includes per-session trial counts)
                   ophys_cells_table.csv        (133066 rows: experiment_id, cell_roi_id, cell_specimen_id)
```
Terminology (whitepaper): **experiment** = one imaging plane in one session; **session** = one
continuous recording (1 plane on Scientifica single-plane rigs, up to 8 planes on the Multiscope);
**container** = same plane tracked across days.

Only **284 of the 1936** experiments in the release are downloaded here. They comprise
*all* 239 experiments of `project_code == 'VisualBehavior'` (single plane, 31 Hz, VISp) plus 45
Multiscope experiments, which all come from a **single mouse** (457841, Sst), 8 sessions, and are the
only source of VISl (LM) neurons.

### Dataset Size (from data files; downloaded subset)
| Statistic | All downloaded | Active behavior only (passive excluded) | After curation (used) |
|-----------|----------------|------------------------------|------------------------|
| Experiments (planes) | 284 | 202 | 199 |
| Ophys sessions | 247 | 174 | 171 |
| Subjects (mice) | 38 | 38 | 38 |
| Neurons (cells, sum over experiments) | 43 927 (cells table) | 29 444 | 29 168 |
| Neurons / session | — | — | mean 170.6, median 108, min 6, max 666 |
| Sessions / subject | — | — | mean 4.5, min 2, max 9 |
| Go+Catch trials (total) | — | 44 892 | 43 975 |
| Go+Catch trials / session | — | — | mean 257, median 263, min 39, max 409 |
| Trial duration (go/catch) | — | — | mean 8.47 s, min 7.02 s, max 12.56 s |
| Ophys frame rate | 31 Hz (single plane) / 11 Hz (Multiscope) | same | dt = 32.31–32.33 ms (31 Hz) / 93.23 ms (11 Hz) |

Session types present (active only): OPHYS_1_images_A (55), OPHYS_3_images_A (55),
OPHYS_4_images_B (46), OPHYS_6_images_B (46). Passive: OPHYS_2/OPHYS_5 (82 experiments) — excluded.
Cre lines (cells, active): Slc17a7 27 669, Sst 743, Vip 756. Structures: VISp 29 006, VISl 162.
Experience levels: Familiar 110, Novel 1 38, Novel >1 54 (experiments).

### Available variables (per experiment)
- Neural: `dff_traces.dff`, `events.events`, `events.filtered_events` (all n_cells × n_frames),
  `ophys_timestamps`, `corrected_fluorescence_traces`, `demixed_traces`, `neuropil_traces`.
- Behavior: `trials`, `licks`, `rewards`, `running_speed` (60 Hz), `eye_tracking` (~30 Hz, 23 cols),
  `stimulus_presentations`, `stimulus_templates`, `task_parameters`, `get_performance_metrics()`.
- Task parameters (constant across sessions): 250 ms flashes, 500 ms blanks, response window
  [0.15, 0.75] s, 5 % omission probability, reward 7 µL / auto-reward 5 µL, 8 images per set.

### Exploration findings (all 202 active experiments surveyed, `/app/cache/survey.csv`)
1. Frame rate is 31 Hz for 165 experiments (median dt 32.31–32.33 ms) and 11 Hz for 34 (93.23 ms).
2. **3 experiments have no eye-tracking data at all** (0 rows): 795953296, 806456687, 833631914.
3. Eye-tracking NaN (blink/outlier) fraction: median 2.9 %, max 29.6 % of frames.
4. Running speed has no NaNs; range over the dataset −24.1 … 99.9 cm/s.
5. `events` are extremely sparse: on average only **0.41 %** of frames are non-zero
   (≈0.13 events/s/cell). `filtered_events` (causal half-gaussian) spreads each event over ~0.6 s.
6. Every go/catch trial has a non-NaN `change_time`, and every go/catch trial lies entirely inside
   the ophys timestamp range (0 exceptions in 202 experiments).
7. Image sets A and B are disjoint: A = im061, im062, im063, im065, im066, im069, im077, im085;
   B = im000, im031, im035, im045, im054, im073, im075, im106 → 16 unique images + "omitted".
8. `trials.change_time` is **exactly** equal (diff = 0.0 s) to the `start_time` of the
   `stimulus_presentations` row with `is_change` (go) or `is_sham_change` (catch) for that trial.
9. n(`is_change` flashes) == n(go)+n(auto_rewarded) and n(`is_sham_change`) == n(catch), per experiment.
10. On catch trials `initial_image_name == change_image_name` (sham change: image does NOT change);
    on go trials they always differ.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Full public dataset (active) | "376 imaging sessions from 82 mice" (Piet et al.); 382 sessions in figures | paper.pdf: "This dataset contains behavior from 376 imaging sessions from 82 mice" |
| Piet et al. neural subset | 8 619 exc (21 sessions, 9 mice), 470 Sst (15, 6), 1 239 Vip (21, 9) | "Our dataset contains 8,619 excitatory cells …" |
| Neural signal used by paper | detected calcium **events** | "For all analysis of neural data we used the detected calcium events" |
| Ophys frame rate | 31 Hz single plane, 11 Hz per plane multi-plane | whitepaper "512x512 pixels, 31 Hz for single plane and … 11 Hz for each plane" |
| Eye tracking / behavior rate | 30 Hz each | whitepaper "eye tracking (30 Hz), and behavior (30 Hz)" (running is polled ~60 Hz) |
| Stimulus timing | 250 ms image, 500 ms gray → 750 ms cycle | paper: "250 ms stimulus duration … 500 ms inter-stimulus duration" |
| Omission probability | 5 %; changes and pre-change flashes never omitted | whitepaper + paper |
| Images per session | 8, 64 possible transitions | whitepaper "Each session included 8 images" |
| Change time distribution | truncated exponential 2.25–8.25 s after trial start, mean ≈4.2 s | whitepaper "Change-times were selected from a truncated exponential distribution ranging from 2.25 to 8.25 seconds" |
| Response window | 0.15–0.75 s after change | whitepaper "150-750ms" + `task_parameters` |
| Catch probability | ~12.5 % in late-stage sessions (matrix sampling) | whitepaper "pushing the actual catch probability to ~12.5%" |
| Free-reward (auto-rewarded) trials | first 5 trials of a session + after 10 consecutive misses | whitepaper "Behavior sessions across all phases began with 5 free-reward trials" |
| Engagement | "mice are engaged 72.2 % of the time"; 60.1 % of image intervals engaged | paper.pdf |
| dF/F | precomputed (median-filter baseline, detrended) | methods.txt "DF/F CALCULATION" |
| Events | "magnitude of events approximates the firing rate … resolution of about 200 ms" | tutorial |

Measured in this downloaded subset (active, curated): catch fraction = 5627/44892 ≈ 12.5 % of
go+catch trials ✔ consistent with the whitepaper's ~12.5 %; hit rate 36.1 %, false-alarm rate 15.0 %.

### Processing Details
- Temporal synchronization of all streams is already done upstream on a single 100 kHz sync board;
  every stream in the NWB carries timestamps on that common clock. Aligning streams therefore only
  requires sampling them at a common set of times — here the **ophys frame times** (per the task's
  "Temporally align based on ophys timestamp").
- Trials are defined by the behavior program (`trials` table); trial types: GO, CATCH, ABORTED
  (lick before change), AUTO-REWARDED (free reward).
- Behavioral analysis in Piet et al. is done per 750 ms **image presentation interval**; decoding in
  that paper uses the first 400 ms after each image presentation. Our decoder instead needs
  continuous time-varying labels, so we keep the full trial at ophys resolution.

### Curation Steps
**Neuron curation rules** (already applied upstream by the Allen pipeline; nothing further needed):
ROIs touching the motion border, unions, duplicates, apical dendrites, too small/narrow/dim ROIs are
excluded; ROIs with non-positive demixed traces removed (~1 % loss). The SDK exposes only these
valid ROIs. We therefore keep **all** cells returned by `cell_specimen_table` / `events`.

**Session curation rules**:
- Active behavior only (`passive == False`, i.e. OPHYS_1/3/4/6) — passive sessions have the lick
  spout retracted, no rewards, so "trial outcome" is undefined (and Piet et al. also analyze only
  active sessions).
- Sessions without eye-tracking data are dropped (pupil diameter is a required decoder output).

**Trial curation rules**: keep `go | catch` (this exactly excludes `aborted` and `auto_rewarded`,
as required by the task); require the trial window to be covered by the ophys timestamps.

### Decoders Trained (in the reference paper)
| Decoded variable | Method | Accuracy |
|---|---|---|
| Image change vs. repeat | random forest, 400 ms after image onset, per imaging plane | Fig. 6A, ≈65–90 % correct depending on n cells/cell class (no numbers in text) |
| Hit vs. miss | random forest, same window | Fig. 6C, ≈55–70 % correct |
| False alarm | same | "very low" (near chance) |
No numerical decoding accuracies are given in the text of either reference document; only figures.
Our decoder is a different architecture (per-timepoint PCA + shared linear readout), so these are
qualitative reference points: image change should be decodable well above chance, and trial outcome
(which is close to hit/miss) decodable but much less well.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal | SDK offers `dff`, `events`, `filtered_events` | `events` non-zero on only 0.41 % of frames | Piet et al.: "detected calcium events" | Use the detected events. Because our decoder classifies **every timepoint** (32 ms), raw `events` are ~99.6 % zeros; the SDK's own `filtered_events` (same events, causal half-gaussian, no backward leakage) is the same signal made usable at frame resolution. Decided empirically in Step 7 (dff vs events vs filtered_events compared on identical sample sessions). |
| Session definition | SDK: one NWB = one *experiment* (plane) | 6 active sessions have 3–7 planes | whitepaper: "data collected in a single continuous recording is defined as a session" | Group experiments by `ophys_session_id` → 171 sessions; concatenate neurons across simultaneously-recorded planes |
| Frame rate | metadata `ophys_frame_rate` = 31 or 11 | dt = 32.32 ms or 93.23 ms | whitepaper: 31 Hz / 11 Hz | Target format requires one common bin size for all sessions → resample onto a common 1/31 s grid (identity for 97 % of sessions, zero-order hold for the 6 Multiscope sessions) |
| Which mice/areas | — | Multiscope = 1 mouse, both VISp+VISl | Piet et al. use Multiscope for neural analysis, VISp+VISl | Keep the Multiscope mouse (it is the only source of VISl and of a 38th mouse) |
| Pupil "diameter" | `eye_tracking` has `pupil_width`, `pupil_height`, `pupil_area` | `pupil_width` is NaN on blinks | tutorial comment "function to plot pupil diameter" uses `pupil_width` | Use `pupil_width` as pupil diameter; linearly interpolate across blink/outlier NaNs |
| Running speed | `running_speed` (10 Hz low-pass) vs `raw_running_speed` | both present | whitepaper describes both; filtered is the default | Use `running_speed` (filtered), which is what the tutorials plot |
| # sessions/mice | — | 171 sessions / 38 mice | 376 sessions / 82 mice (full release) | Consistent: only 284/1936 experiments were downloaded (all of project_code `VisualBehavior` + one Multiscope mouse); we use every active one |

Everything else was consistent across code, data and text (task timing, trial-type definitions,
omission rate, catch rate, response window, ROI curation).

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `events.filtered_events` (n_cells × n_frames) of every plane of the session | `neural[session][trial]` (n_neurons, T) | sample at the trial's bin times using the nearest ophys frame; concatenate planes | `BehaviorOphysExperiment.events`, `event_detection.filter_events_array` | dtype float32. Alternatives (`dff`, raw `events`) benchmarked in Step 7 |
| — | `input[session][trial]` (0, T) | none — task specifies no decoder inputs | — | `input_names = []` |
| `stimulus_presentations` (`start_time`, `end_time`, `image_name`, `omitted`) of the `change_detection_behavior` block | `output[0]` image identity | 0 = gray/no image (incl. omitted flashes and the 500 ms blanks), 1..16 = the 16 unique image names, sorted | tutorial `plot_stimuli` | 17 categories |
| `stimulus_presentations.is_change` | `output[1]` image change | 1 for bins inside the 250 ms flash whose `is_change == True` (go trials only), else 0 | tutorial | 2 categories. Catch trials have `is_sham_change` (image does not change) → all 0 |
| `running_speed` (`timestamps`, `speed`) | `output[2]` running speed quintile | linear interpolation onto bin times, then per-session quintile bins (0–4) | `running_processing.py` | 5 categories |
| `eye_tracking.pupil_width` | `output[3]` pupil diameter quintile | interpolate over blink NaNs, linear interpolation onto bin times, then per-session quintile bins | `eye_tracking_processing.py` | 5 categories |
| `trials.hit/miss/false_alarm/correct_reject` | `output[4]` trial outcome | 0 hit, 1 miss, 2 false alarm, 3 correct reject; constant within the trial | `trials.py` | 4 categories, static per trial (broadcast over T) |
| `trials.start_time`, `trials.stop_time`, `trials.go`, `trials.catch` | trial segmentation | keep `go | catch`; window = [start_time, stop_time) | tutorial `make_trial_plot` | excludes aborted + auto-rewarded |
| `ophys_experiment_table.mouse_id` | `subjects`, `subject_idx` | unique sorted list of mouse ids | — | 38 mice |
| `metadata['targeted_structure']` per experiment | `brain_regions`, `brain_region_idx` | one entry per neuron | — | VISp, VISl |

### Key Decisions
1. **Session = `ophys_session_id`** (not NWB file). The whitepaper defines a session as one
   continuous recording; the 6 active Multiscope sessions contain 3–7 simultaneously recorded planes
   which share one behavior/trial structure. Neurons of all planes of a session are concatenated
   (`brain_region_idx` keeps track of VISp vs VISl). → 171 sessions.
2. **Only active behavior sessions** (OPHYS_1/3/4/6). Passive sessions have no licking/reward and
   hence no Go/Catch trial outcomes; Piet et al. also analyze only active sessions.
3. **Trials = `go | catch`** from the SDK trials table, window `[start_time, stop_time)`. This is
   exactly the task's requirement (Go and Catch, no Aborted, no Auto-rewarded); `go`/`catch` are
   False on aborted and auto-rewarded trials, so the boolean mask is sufficient. Trials are variable
   length (7.0–12.6 s) because that is how the experiment defines them; the format allows variable T.
4. **Time bins**: uniform 1/31 s = 32.258 ms for every trial and session, anchored at each trial's
   `start_time` ("Temporally align based on ophys timestamp": bin k of a trial is the ophys frame
   nearest to `start_time + k·dt`). For the 165 single-plane experiments the frame period is
   32.31 ms, so this is a 1:1 relabelling of the native ophys frames with no interpolation of neural
   data; for the 34 Multiscope planes (93.23 ms) it is a zero-order hold, which also provides the
   common time base needed to merge the planes of a Multiscope session.
5. **Neural signal = detected calcium events, `filtered_events`.** Piet et al. use detected calcium
   events for all neural analyses. Raw `events` are non-zero on only 0.41 % of frames, which makes a
   per-timepoint decoder nearly blind; `filtered_events` is the *same* event train convolved with the
   SDK's own causal half-normal kernel (2/31 s scale) — causal, so it cannot move activity backwards
   in time and cannot create temporal misalignment. This is the "required by the decoding task"
   deviation. Verified empirically against `dff` and raw `events` in Step 7.
6. **No additional neuron filtering**: ROI QC is already applied upstream (see Step 1/3).
7. **Sessions without eye tracking are dropped** (3 experiments = 3 sessions) because pupil diameter
   is a required output and cannot be fabricated.
8. **Running speed / pupil quintiles are computed per session**, over all timepoints of that
   session's included trials. Pupil width is measured in camera pixels and is not comparable across
   sessions/rigs (different zoom and eye position), and running distributions differ strongly across
   mice; per-session quantization yields the "five equal percentile bins" the task asks for in every
   session and a well-posed balanced 5-class problem. (Pooled-dataset quintiles would give the same
   ≈20 % marginal but degenerate per-session distributions.)
9. **Image identity** uses the union of the 16 image names across image sets A and B plus a
   "gray" class for the inter-flash blanks and omitted flashes (17 categories). The image sets are
   disjoint, so a session only ever expresses 8 of the 16 image classes — that is a property of the
   experiment, not of the conversion.
10. **Image change** is marked over the 250 ms of the changed-image flash (≈8 bins) rather than a
    single bin, so that the label covers the stimulus event itself; a 1-bin delta would be 0.4 % of
    the data and almost unlearnable. Catch (sham change) trials contain no image change → all zeros.
11. **Outputs are all time-varying except trial outcome**, which is static per trial and broadcast
    over the trial's bins (the target format requires one array per trial, so all five outputs share
    the (5, T) array).

### Planned Sanity Checks (all executed in Step 10; results there)
- [x] Number of sessions/mice/neurons/trials matches the project metadata tables (171 / 38 / 29 168 / 43 975).
- [x] Per-session go+catch trial counts equal `behavior_session_table.go_trial_count + catch_trial_count`.
- [x] Trial outcome fractions match `behavior_session_table` counts (hit .317, miss .558, FA .019, CR .106).
- [x] Catch fraction of go+catch ≈ 12.5 % (whitepaper).
- [x] Image-change bins per trial ≈ 250 ms/32.26 ms ≈ 7.75; exactly one change epoch per go trial, none on catch trials.
- [x] Image identity at the bin before `change_time` == `initial_image_name`; at `change_time` == `change_image_name`.
- [x] Gray fraction ≈ (750−250)/750 = 66.7 % of bins (slightly more with omissions).
- [x] Each quintile output has ≈20 % of bins in every session.
- [x] Spot check (loading the NWB independently of the conversion code) that `neural[s][t][n, k]`
      equals `filtered_events` of the corresponding cell at the ophys frame nearest
      `trial.start_time + k·dt`.
- [x] Trial durations: T·dt ≈ `stop_time − start_time` for every trial.
- [x] No NaN/Inf anywhere; all outputs integer-valued and within range.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` — `python -u /app/convert_data.py <outfile> [--full|--sample]
[--show-processing] [--trace {dff,filtered_events,events}] [--workers N] [--sessions ids]`.

Structure:
- `select_sessions()` — reads `ophys_experiment_table.csv`, keeps experiments whose NWB file is
  present locally and `passive == False`, groups by `ophys_session_id`.
- `convert_session(job)` — runs in a worker process, one job per ophys session:
  loads every plane's NWB with `BehaviorOphysExperiment.from_nwb_path`, checks that the behavior
  tables agree across planes, selects `go|catch` trials, builds the per-trial bin grid, samples the
  neural traces at the nearest ophys frame, builds the five outputs, computes the session's
  running/pupil quintile edges, returns per-trial arrays.
- `main()` — fans the sessions out over a process pool, assembles the final dict (remapping each
  session's local image labels to the global 16-image list), prints summary statistics, pickles.
- `_plot_processing()` — `--show-processing` diagnostic figure (14 panels) per session.

Robustness/edge cases handled: sessions/planes with missing eye tracking or running data (session
dropped with a logged reason), trials shorter than 2 bins, trials not fully covered by the ophys
timestamps, `end_time` NaN on omitted flashes, go/catch trials without an outcome flag, plotting
failures (never lose data), non-finite interpolated behaviour.

Code efficiency:
- Bottleneck is NWB reading (~250 MB/experiment). `ProcessPoolExecutor` with 16 workers over
  sessions; each NWB is opened exactly once and all five streams are taken from that one object.
- Nearest-frame lookup is a vectorised `np.searchsorted` per trial, not a Python loop over bins.
- Traces are cast to float32 once per plane; trial slices are views→copies of that matrix.
- Behaviour streams are interpolated with `np.interp` (vectorised) on the trial's bin centres.
- Per-session quantile edges are computed once on the concatenated session values, then applied to
  all trials with one `np.searchsorted`.

Code inefficiencies identified: loading the planes of a Multiscope session is serial inside a
worker (7 planes ≈ 17 s); this affects 6 of 171 sessions and is not worth extra process nesting.

### Neural signal choice (decided empirically, see Step 7 table)
Identical conversions of the same 6 sessions with `--trace dff`, `filtered_events`, `events`, then
the reference decoder: dF/F wins on every output, by a wide margin on image identity. The Allen
whitepaper (methods.txt) describes dF/F as *the* processed neural signal of this dataset (there is
no event-detection section in it), so using it is faithful to the dataset reference; the Piet et al.
paper's use of detected events is a choice made for their event-triggered analyses and is not
required here — and raw `events` are non-zero on only 0.4 % of frames, which is unusable for a
per-time-bin decoder.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`
(`--sample` deliberately takes one single-plane session and one 7-plane Multiscope session so both
code paths are exercised.) → `/app/conversion_sample_out.txt`, `/app/sample_data.pkl`,
`processing_775289198.png`, `processing_951410079.png`.

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (775289198 single-plane VISp; 951410079 Multiscope, 7 planes, VISp+VISl) |
| Neurons (total) | 177 (89 + 88) |
| Neurons / session | 88.5 |
| Subjects | 2 |
| Trials (total) | 248 (39 + 209) |
| Trials / session | 124 |
| T per trial | mean 255.8, min 224, max 388 bins (32.26 ms bins) |
| Inputs | none (dinput = 0) |
| image_identity distribution | gray 0.670, 8 images 0.034–0.048 each |
| image_change distribution | 0.975 / 0.025 |
| running quintiles | 0.200 × 5 |
| pupil quintiles | 0.200 × 5 |
| trial_outcome | hit 0.372, miss 0.495, FA 0.031, CR 0.102 |

Consistency of these numbers: gray fraction 0.670 ≈ (750−250)/750 = 0.667 expected from the 250 ms
flash / 750 ms cycle (slightly higher because 5 % of flashes are omitted → gray); change fraction
0.025 ≈ 0.25 s/8.5 s × (fraction of go trials ≈ 0.87) = 0.026.

### Neural signal comparison (6 sessions, 1989 neurons, 1628 trials, identical everything else)
| Output (validation balanced accuracy) | chance | `dff` | `filtered_events` | `events` |
|---|---|---|---|---|
| image_identity (17 classes) | 0.059 | **0.575** | 0.326 | 0.210 |
| image_change | 0.500 | **0.651** | 0.600 | 0.582 |
| running_speed_quintile | 0.200 | **0.307** | 0.249 | 0.229 |
| pupil_diameter_quintile | 0.200 | **0.305** | 0.242 | 0.226 |
| trial_outcome | 0.250 | **0.345** | 0.257 | 0.260 |
→ `dff` selected as the default neural signal.

### Processing Plots Review
`processing_775289198.png` / `processing_951410079.png` (14 panels each) show, from raw to
converted: raw dF/F on native ophys timestamps; raw stimulus flashes (colour = image, black =
change, red dotted = omitted); raw 60 Hz running speed; raw 30 Hz pupil width; the converted neural
matrix of one trial; the converted outputs 0–1 and 2–4 of that trial; plus alignment checks
overlaying the raw stream and the converted samples for neural/running/pupil, the session
distributions with the quintile edges drawn on them, the per-value output fractions, and the
image identity immediately before vs at each change.

No anomalies: converted bins lie exactly on the raw traces (neural overlay is indistinguishable for
the 31 Hz session and a correct zero-order hold for the 11 Hz Multiscope session); the image
identity steps exactly at `change_time` and `image_change` is 1 only over the 250 ms changed flash;
the plotted Multiscope trial is a Catch trial and correctly has a constant image and no change;
quintile edges cut the session distributions into five equal-count parts.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| Process pool over sessions (16 workers) | ~16× (I/O bound on 250 MB NWB files) |
| Single NWB open per experiment, all streams from one object | ~2× vs re-opening per stream |
| Vectorised nearest-frame `searchsorted` + `np.interp` instead of per-bin loops | ~50× on the per-trial step |
| float32 traces | halves RAM/IPC |

| Step | Time / session | Estimated total (171 sessions, 16 workers) |
|---|---|---|
| NWB load (1 plane) | 4–5 s | — |
| NWB load (7 planes, Multiscope) | 17 s | — |
| trace extraction + per-trial assembly | 1–3 s | — |
| **wall-clock per session (1 worker)** | **≈6 s (single-plane), ≈19 s (Multiscope)** | ≈1100 s / 16 ≈ **2 min** + pickling ~8 GB ≈ 2 min → **< 5 min** |

### Format Validation (`/app/verification_sample_out.txt`)
"Data format is valid, no errors or warnings." Output ranges and distributions as tabulated above.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/sample_data.pkl` → `/app/train_decoder_sample_out.txt`.
Loss decreased monotonically from 2.10 (epoch 1) to 1.261 (epoch 200); test loss 1.433.

### Decoder Results (Sample, 2 sessions / 177 neurons only)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| image_identity (9 classes in the sample: 8 set-A images + gray) | 0.5387 | 0.4724 | 0.1111 |
| image_change | 0.7210 | 0.6623 | 0.5000 |
| running_speed_quintile | 0.2737 | 0.2459 | 0.2000 |
| pupil_diameter_quintile | 0.3638 | 0.3164 | 0.2000 |
| trial_outcome | 0.4194 | 0.3248 | 0.2500 |

Every output is above chance, and train/validation gaps are small (≤1.3×). Running speed is the
weakest (1.23× chance) on this 2-session sample; it improves with the full dataset (Step 11).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full --workers 16`
→ `/app/conversion_full_out.txt`. Runtime **79 s** (well under the 15-min budget and under the
< 5 min estimate from Step 7), of which ~25 s is pickling.

### Output Files
- `converted_data.pkl`: 8.83 GB (171 sessions, 43 975 trials, 11 583 870 time bins × 29 168 neurons worth of float32)
- `verification_full_out.txt`: created, **"Data format is valid, no errors or warnings."**

Excluded by the script (logged in `metadata['excluded_sessions']`): ophys sessions 795625712,
805989030, 832881662 — "no eye tracking data".

### Consistency Check
| Statistic | Reference papers | Reference code / metadata tables | Reference data (NWB) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Active ophys sessions (whole release) | 376–382 (Piet et al.) | 495 in `ophys_session_table` (incl. variants Piet excluded) | — | n/a (only 284/1936 experiments are downloaded) | ✔ consistent |
| Mice (whole release) | 82 | 107 | — | n/a | ✔ consistent |
| Sessions used | — | 174 active downloaded | 174 | **171** (3 have no eye tracking) | ✔ |
| Subjects used | — | 38 | 38 | **38** | ✔ |
| Experiments (planes) used | — | 202 active | 199 loadable w/ eye tracking | **199** | ✔ |
| Total neurons | — | 29 444 (`ophys_cells_table`, 202 exps) | 29 168 (199 exps) | **29 168** | ✔ |
| Neurons/session | — | — | mean 170.6 (6–666) | **mean 170.6, min 6, max 666** | ✔ |
| Go+Catch trials | — | 44 892 (174 sessions, `behavior_session_table`) | 43 975 (171 sessions) | **43 975** | ✔ |
| Trials/session | — | — | mean 257.2 (39–409) | **mean 257.2, min 39, max 409** | ✔ |
| Trial outcome counts | — | hit 13 940 / miss 24 520 / FA 834 / CR 4 681 (`behavior_session_table`, 171 sessions) | same | **identical** (checked trial-by-trial, Step 10) | ✔ |
| Hit rate | — | 36.2 % | 36.2 % | **36.2 %** (13 940/38 460 go trials) | ✔ |
| False-alarm rate | — | 15.1 % | 15.1 % | **15.1 %** (834/5 515 catch trials) | ✔ |
| Catch fraction of go+catch | ~12.5 % (whitepaper) | 12.5 % | 12.5 % | **12.5 %** (5 515/43 975) | ✔ |
| Time bin | 31 Hz imaging (whitepaper) | `ophys_frame_rate` 31 Hz; dt = 32.32 ms | 32.31–32.33 ms | **32.258 ms (1/31 s)** | ✔ |
| Trial duration | change 2.25–8.25 s after start, +~4.2 s | — | mean 8.47 s | **mean T 261.9 bins = 8.45 s** | ✔ |
| image_identity: gray | (250 ms of every 750 ms cycle, minus ~4 % omissions) ⇒ 0.667–0.68 | — | 11.7 flashes/trial, 4 % omitted ⇒ predicted image-on 0.332 | **gray 0.670 / image 0.330** | ✔ |
| image_identity: per image | 8 images/session, 16 total, A and B disjoint | — | — | **0.0197–0.0215 each (16 images)** | ✔ |
| image_change | 250 ms per go trial ⇒ 0.026 | — | — | **0.0252** | ✔ |
| running/pupil quintiles | — | — | — | **0.2000 each** | ✔ |
| trial_outcome (per time bin) | — | per-trial hit .317/miss .558/FA .019/CR .106 | same | **hit .313 / miss .562 / FA .018 / CR .107** (per-bin, weighted by trial length) | ✔ |
| Brain regions | V1 (VISp) and LM (VISl) | VISp 29 006, VISl 162 | same | **VISp 29 006, VISl 162** | ✔ |

No data were lost: every active session with eye tracking, every valid ROI of every plane, and
every Go/Catch trial of every session is present (trial counts verified against
`behavior_session_table` session by session in Step 10).

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
First line: **"Data format is valid, no errors or warnings."** There are **no errors and no
warnings** to address (no NaN/Inf, no dtype warnings, no empty sessions, no dimension mismatches,
neural data has variability, `input_names`/`output_names`/`output_values` lengths all match).
Ranges: image_identity 0–16, image_change 0–1, quintiles 0–4, outcome 0–3, T 217–389 bins,
n_neurons 6–666. All as designed.

### Check 2: Independent sanity checks (`/app/cache/sanity_checks.py`, log `/app/cache/sanity_checks_out.txt`)
The script re-derives every quantity **from the NWB files and the project-metadata csvs with the
AllenSDK, without importing the conversion code**, and compares with `np.allclose` /
`np.array_equal`. 6 random sessions + the 7-plane Multiscope session were spot-checked in full,
plus dataset-level checks over all 171 sessions. **All 100 checks pass (0 failures).**

Dataset-level (all sessions):
| Check | Result |
|---|---|
| total neurons == rows of `ophys_cells_table` for the used experiments (29 168) | PASS |
| each session maps to one mouse; `subject_idx` correct | PASS |
| `brain_region_idx` == `targeted_structure` per neuron, in plane order | PASS |
| trials/session == `go_trial_count + catch_trial_count` of `behavior_session_table`, every session | PASS |
| outcome counts == [hit 13940, miss 24520, FA 834, CR 4681] from `behavior_session_table` | PASS (exact) |

Per-session (raw NWB):
| Check | Result |
|---|---|
| **neural**: `neural[t][n,k]` == `dff_traces` of the right cell at the ophys frame nearest `start_time+(k+.5)·dt` (3 random (trial,neuron,bin) per session) | PASS |
| **neural**: whole (n_neurons × T) matrix of a random trial matches, including multi-plane concatenation order | PASS |
| `T == floor((stop_time-start_time)/dt)` for **every** trial | PASS |
| **output 0**: image identity re-derived with a slow, independent "which flash contains this bin centre" implementation, 5 random trials/session | PASS |
| **output 0**: last image before `change_time` == `initial_image_name`; first image at/after == `change_image_name`, every trial | PASS |
| **output 1**: change == bins inside the changed flash on every go trial; all-zero on every catch trial; exactly one contiguous change epoch per go trial | PASS |
| **output 2/3**: quintile labels == independent recomputation from raw `running_speed`/`eye_tracking.pupil_width`; fractions 0.2±0.01 | PASS |
| **output 4**: outcome constant within trial and == trials-table flags | PASS |
| **input**: shape (0, T) for every trial | PASS |

Additional alignment check (`/app/cache/alignment_check.py`, figure `/app/cache/alignment_check.png`),
computed from the converted data over 20 random sessions: the change-triggered population average of
z-scored dF/F is at its **minimum exactly at the change bin**, rises immediately after it and peaks
at **+0.29 s** (GCaMP6f kinetics), then shows the 750 ms-periodic flash responses; non-change flash
onsets give the same periodic response with ~1/4 the amplitude (the classic change/repeat adaptation
effect). Activity never precedes the stimulus → no temporal misalignment and no sign/offset error.

### Check 3: Reference code comparison
| Step | Reference (AllenSDK / tutorials / papers) | This conversion | Same? |
|---|---|---|---|
| (a) Data loading | `bc.get_behavior_ophys_experiment(eid)`, which is literally `BehaviorOphysExperiment.from_nwb_path(nwb)` (`behavior_project_cloud_api.py:181`) | `BehaviorOphysExperiment.from_nwb_path(nwb)` on the local NWB | identical |
| Experiment selection | `bc.get_ophys_experiment_table()` filtered by columns | same table read from `project_metadata/ophys_experiment_table.csv`, filtered on `passive` + file present | identical |
| (b) Neuron filtering | pipeline ROI QC; SDK exposes only valid ROIs (`cell_specimen_table.valid_roi` all True) | no further filtering; all cells of `dff_traces` kept | identical |
| (b) Trial filtering | tutorials query `trials.query('hit')`, `'miss'`, …; whitepaper defines GO/CATCH/ABORTED/AUTO-REWARDED | `trials[trials.go | trials.catch]` (task requirement) | identical definitions |
| (b) Session filtering | Piet et al.: active behavior sessions only | passive excluded; + 3 sessions without eye tracking (output requirement) | same + documented extra |
| (c) Temporal alignment | tutorials slice each stream by its own timestamps over `[trial.start_time, trial.stop_time]`; all streams share the 100 kHz sync clock | identical windows, all streams sampled at the ophys-frame-anchored bin centres of that window | same convention |
| (d) Binning | SDK does not bin; analyses use native ophys frames (Piet et al. bin behaviour into 750 ms image intervals) | uniform 1/31 s bins = the native single-plane frame period (needed for one common bin size across rigs) | consistent |
| (e) Input construction | n/a | none (task says no inputs) | n/a |
| (f) Output construction | `stimulus_presentations.image_name/is_change/omitted` (tutorial `plot_stimuli`), `running_speed.speed` (tutorial `plot_running`), `eye_tracking.pupil_width` (tutorial "plot pupil diameter"), `trials.hit/miss/false_alarm/correct_reject` (tutorial trial-type queries) | exactly these variables | identical |
| Neural signal | whitepaper: dF/F is the pipeline's neural output; Piet et al. use detected events | `dff` (default), `events`/`filtered_events` available via `--trace` | deviation, justified in Step 6/7 (decoding requirement; dF/F is the whitepaper's canonical signal) |
| Running speed variant | whitepaper: `running_speed` filtered (10 Hz low-pass) vs `raw_running_speed` | `running_speed` | identical to tutorials |

Differences and why: (1) neural signal dF/F instead of Piet et al.'s events — required by the
per-time-bin decoding task (events are non-zero in 0.4 % of bins), and dF/F is the signal the
dataset whitepaper documents; (2) sessions grouped by `ophys_session_id` instead of one NWB per
"session" — the whitepaper's own definition of a session, and it gives the Multiscope planes a
common trial structure; (3) a uniform 1/31 s bin instead of each rig's native period — the target
format demands one bin size for all sessions.

### Check 4: Key statistics comparison
See the Step 9 table: sessions, mice, experiments, neurons, neurons/session, trials,
trials/session, per-outcome trial counts, hit rate (36.2 %), false-alarm rate (15.1 %), catch
fraction (12.5 % — whitepaper says "~12.5 %"), image-on fraction (0.330 vs 0.332 predicted from
11.7 flashes/trial × 96 % non-omitted × 250 ms), change fraction (0.0252 vs 0.026 predicted),
bin size (32.26 ms vs 32.32 ms native), trial duration (8.45 s vs 8.47 s), brain-region counts.
Every statistic matches its reference value. The only "mismatch" is with the *whole-release*
numbers quoted in Piet et al. (376 sessions / 82 mice), which is expected because only 284 of the
1936 experiments in the release are present in `/app/data`.

### Check 5: Edge cases
- **No trial is silently dropped**: trials/session equals `go_trial_count+catch_trial_count` for all
  171 sessions, so none of the defensive filters (T < 2 bins, ophys coverage, missing outcome flag)
  ever triggered. The guards remain for robustness.
- **Trial boundaries**: `T = floor(dur/dt)` verified for all trials; the last (partial) bin is
  dropped so a bin never extends past `stop_time`. Bin centres are at `+0.5·dt`, so no bin centre
  ever coincides exactly with a flash boundary (750 ms and 32.258 ms are incommensurate).
- **First/last trial of a session**: every go/catch trial lies inside the ophys timestamp range in
  all 202 active experiments (checked in the Step 2 survey and re-checked per trial in the script).
- **Omitted flashes**: their `end_time` is forced to −inf so they always fall through to "gray";
  verified against the independent implementation.
- **Multi-plane sessions**: behavior tables are asserted identical across planes (trial count and
  all start times) before merging; neuron order is plane order and matches
  `brain_region_idx` (verified in Check 2).
- **Sessions with tiny populations** (6–20 neurons, Sst/Vip) are kept: the reference paper analyses
  exactly these small inhibitory populations; the decoder handles `n_neurons < npcs` by padding the
  projection with orthogonal rows.
- **Eye tracking**: 3 sessions have no eye data at all (dropped, logged); in the rest, blink/outlier
  NaNs (median 2.9 %, max 29.6 % of frames) are linearly interpolated over before sampling.
- **Empty-plane fallbacks**: if plane 0 of a Multiscope session lacked eye/running data the code
  falls back to the other planes before giving up.

### Issues Found and Resolved
1. `--show-processing` crashed (1-D indexing of the trace matrix, then a cell index that could
   exceed plane 0's neuron count in multi-plane sessions) → fixed, and plotting is now wrapped in
   try/except so a plotting bug can never drop a session's data.
2. Initial default neural signal (`filtered_events`, following Piet et al.) decoded much worse than
   dF/F → changed the default to `dff` after the controlled comparison in Step 7.
3. No other issues: the first full conversion already matched every reference statistic.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`
→ `/app/train_decoder_full_out.txt` (ran on the L4 GPU; ~11 min).

### Training Progress
- Loss decreasing: **Yes**, monotonically, 1.6443 (epoch 1) → 1.3712 (epoch 200); test loss 1.5348.
  (The loss is still falling at epoch 200 — with 171 session-specific projection matrices the fixed
  200-epoch budget under-fits slightly; this is a property of the decoder, not of the data.)

### Decoder Results (Full: 171 sessions, 29 168 neurons, 43 975 trials, 11.58 M time bins)
| Output | #classes | Chance | Training Balanced Acc | Validation Balanced Acc | × chance (val) |
|--------|---------|--------|-------------|--------|-------|
| image_identity | 17 | 0.0588 | 0.4362 | **0.4065** | 6.9× |
| image_change | 2 | 0.5000 | 0.6678 | **0.6260** | 1.25× |
| running_speed_quintile | 5 | 0.2000 | 0.3079 | **0.2822** | 1.41× |
| pupil_diameter_quintile | 5 | 0.2000 | 0.3075 | **0.2673** | 1.34× |
| trial_outcome | 4 | 0.2500 | 0.4384 | **0.2969** | 1.19× |

`sample_trials.png` and `predictions.png` were produced; the ground-truth traces show the expected
750 ms flash structure, and the predicted traces track it (noisily, as expected from a
per-time-bin linear read-out of 100 PCs).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Output | Validation balanced acc | Chance | Ratio | Verdict |
|---|---|---|---|---|
| image_identity | 0.4065 | 0.0588 | **6.9×** | strong |
| image_change | 0.6260 | 0.5000 | 1.25× | investigated below |
| running_speed_quintile | 0.2822 | 0.2000 | 1.41× | investigated below |
| pupil_diameter_quintile | 0.2673 | 0.2000 | 1.34× | investigated below |
| trial_outcome | 0.2969 | 0.2500 | 1.19× | investigated below |

Nothing is below chance. Four outputs are below 1.5× chance, so each was investigated:

**(a) Is it a conversion bug, or the decoder/task?** Re-running the identical pipeline on an
*unbiased* random subset of 20 sessions (`/app/cache/rand20.pkl`) reproduces the full-dataset
numbers almost exactly (image identity 0.429, change 0.636, running 0.287, pupil 0.276, outcome
0.307), while the 6 sessions with the **largest populations** (22–666 cells, Step 7) give much
higher numbers (0.575 / 0.651 / 0.307 / 0.305 / 0.345). So performance is limited by how many
neurons a session has, not by dataset size or a scale-related bug. The dataset is dominated by
small inhibitory populations (median 108 cells, 40 sessions with <25 cells — the Sst/Vip mice that
the reference paper studies).

**(b) Is the information actually in the converted data?** Yes — reproducing the reference paper's
own decoders on the converted data (`/app/cache/paper_decoders.py`: random forest, activity in the
**first 400 ms after image onset**, 5-fold CV, per session, exactly Piet et al.'s Fig. 6 protocol)
gives, on the same 20 random sessions:
| Paper decoder | This converted data | Piet et al. Fig. 6 |
|---|---|---|
| change vs. repeat (% correct) | **76.8 % ± 3.1 SEM** (range 56–97 %, r = 0.52 with #cells) | ≈65–90 %, increasing with #cells |
| hit vs. miss (% correct) | **78.7 % ± 2.2 SEM** | ≈55–70 % |
So the change and choice signals are present at the level the paper reports; the per-time-bin
linear read-out in `train_decoder.py` simply cannot exploit all of it, because it must label each
32 ms bin independently while the dF/F response to a change peaks 290 ms after the 250 ms label
window starts (see the change-triggered average in Step 10).

**(c) Would a different output construction help?** Controlled experiments on identical data:
| Variant | image_identity | image_change | running | pupil | outcome |
|---|---|---|---|---|---|
| chosen: change = 250 ms changed flash (6-session set) | 0.5754 | **0.6512** | 0.3073 | 0.3052 | 0.3452 |
| change = full 750 ms image-presentation interval | 0.5796 | 0.6442 | 0.3080 | 0.3071 | 0.3453 |
| per-neuron z-scored dF/F | 0.5726 | 0.6492 | 0.3097 | 0.3040 | 0.2855 |
| image identity persisting through the gray period (20-session set) | 0.4477 (vs 0.4289) | 0.6346 | 0.2861 | 0.2774 | 0.3081 |
None of these is a material improvement, so the literal/most faithful construction was kept in
every case (see "Interpretation notes" below).

**(d) Per-class recall** (20-session subset, `/app/cache/confusion.py`) shows every class is
learned, i.e. no degenerate output:
- image_identity: gray 0.11, individual images 0.29–0.59 (the balanced loss trades the 67 %
  "gray" class away; during the 500 ms gray period the dF/F response to the preceding flash is
  still decaying, so those bins genuinely look like the image).
- image_change: no_change 0.75, change 0.52.
- running quintiles: 0.49 / 0.24 / 0.18 / 0.19 / 0.33 — the extremes (stationary, fast running)
  are decodable, the middle quintiles are not, which is the expected shape for locomotion signals.
- pupil quintiles: 0.54 / 0.18 / 0.14 / 0.18 / 0.34 — same pattern.
- trial_outcome: hit 0.48, miss 0.37, false_alarm 0.22, correct_reject 0.16. Most of a trial
  (the 3–8 s before the change) contains no information about the outcome at all, and false alarms
  are only 1.9 % of trials, so a per-bin 4-way balanced accuracy of 0.30 is what this label can
  support.

### Check 2: Accuracy comparison to papers
Neither reference document reports numerical decoding accuracies in text (only Fig. 6A–C and
Fig. S22 of Piet et al.). The like-for-like comparison is therefore the reproduction of their
decoders on our converted data, tabulated in Check 1(b): change vs repeat 76.8 % (paper ≈65–90 %),
hit vs miss 78.7 % (paper ≈55–70 %, and they subsample neurons whereas we use all cells of a
session, which is expected to help us). False-alarm decoding is "very low" in the paper, matching
the low false-alarm recall (0.22) here. Our data therefore meets or exceeds the paper's decodable
signal, so no conversion bug is indicated.

### Check 3: Train vs validation gap
| Output | Train | Validation | Ratio |
|---|---|---|---|
| image_identity | 0.4362 | 0.4065 | 1.07× |
| image_change | 0.6678 | 0.6260 | 1.07× |
| running_speed_quintile | 0.3079 | 0.2822 | 1.09× |
| pupil_diameter_quintile | 0.3075 | 0.2673 | 1.15× |
| trial_outcome | 0.4384 | 0.2969 | 1.48× |
All below the 1.5× flag. The largest gap is the only **static per-trial** label: within a training
trial the target is constant for ~260 bins, so the model can partly memorise the trial's overall
neural state; held-out trials do not transfer as well. There is no data leakage across the split —
`train_validate_decoder` splits by trial, trials are disjoint time windows, and every quantity
(including the running/pupil quantile edges) is computed per session from all of that session's
trials, i.e. identically for train and test trials (the edges are unsupervised statistics of the
behavioural variable, not of the labels being predicted).

### Interpretation notes (documented alternatives, deliberately not adopted)
1. **Image identity during the gray screen.** The task says "Image identity (of the image presented
   during the non-grey screen)". We label each bin with the image physically on the monitor and use
   a separate `gray` class for the 500 ms inter-flash blanks and for omitted flashes (67 % of bins).
   The alternative reading — carry the last presented image through the blank — was implemented and
   measured (+0.019 balanced accuracy, Check 1c) and rejected as less literal: with it, the screen
   content and the label disagree for two thirds of the time.
2. **Image change window.** 1 during the 250 ms presentation of the changed image ("right after a
   change in image identity"). The 750 ms image-presentation-interval convention of Piet et al. was
   measured and is no better.
3. **Trial window.** The full `[start_time, stop_time)` of the trials table, as the task requires
   ("segment ... based on how they are defined in the experiment"). A short window around the
   change would raise the change/outcome accuracies simply by deleting the uninformative
   pre-change period, but it would throw away ~60 % of the recorded task time.

### Issues Found and Resolved
- None in this round: no check produced a failure, so no re-conversion was required. The two issues
  found in Step 10 (plot crash; neural-signal default) were already fixed before the full run.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, how to load, output-format specification, key
      statistics, decoder accuracy, curation summary).
- [x] `cache/` folder created with all investigation/validation scripts and their logs, documented
      in `cache/README_CACHE.md`. Large intermediate pickles were deleted (regeneration commands
      are given there).
- [x] Files in `/app`:
  - deliverables: `CONVERSION_NOTES.md`, `README.md`, `convert_data.py`, `converted_data.pkl`
    (8.83 GB), `sample_data.pkl` (26 MB)
  - logs: `conversion_sample_out.txt`, `verification_sample_out.txt`,
    `train_decoder_sample_out.txt`, `conversion_full_out.txt`, `verification_full_out.txt`,
    `train_decoder_full_out.txt`
  - figures: `processing_775289198.png`, `processing_951410079.png` (conversion diagnostics),
    `sample_trials.png`, `predictions.png` (from `train_decoder.py --plot-samples`)
  - provided inputs, untouched: `data/`, `code/`, `tutorials/`, `decoder.py`, `train_decoder.py`,
    `methods.txt`, `paper.pdf`, `whitepaper.pdf`, `Dockerfile`, `docker-compose.yaml`, `.manifest`

### Summary of every decision made (quick index)
| Decision | Choice | Where justified |
|---|---|---|
| Loader | `BehaviorOphysExperiment.from_nwb_path` (== the SDK cache loader) | Step 1, Step 10 Check 3 |
| Sessions | active behavior only, grouped by `ophys_session_id`, must have eye tracking → 171 | Step 5 #1,2,7 |
| Neurons | all SDK-returned ROIs (pipeline QC already applied) → 29 168 | Step 3, Step 5 #6 |
| Trials | `go | catch`, window `[start_time, stop_time)` → 43 975 | Step 5 #3 |
| Alignment | ophys frame times; bin k = frame nearest `start_time+(k+0.5)·dt` | Step 5 #4 |
| Bin size | 32.258 ms (1/31 s) for all sessions | Step 4, Step 5 #4 |
| Neural signal | dF/F (`dff_traces.dff`) | Step 6/7 (measured against events) |
| Inputs | none | task specification |
| Output 0 | image on screen, 17 classes incl. `gray` | Step 5 #9, Step 12 note 1 |
| Output 1 | change = 250 ms changed flash, Go trials only | Step 5 #10, Step 12 note 2 |
| Outputs 2,3 | per-session quintiles of running speed / pupil width | Step 5 #8 |
| Output 4 | 4-way trial outcome, static per trial | Step 5 #11 |
