# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory — Visual Behavior 2P (ophys), release `visual-behavior-ophys-1.1.0`
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format (`/app/converted_data.pkl`)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents of `/app`:
- `CONVERSION_NOTES.md` (this file), `Dockerfile`, `docker-compose.yaml`, `.manifest`
- `allensdk_docs/` — downloaded allensdk documentation
- `code/` — AllenSDK source (reference code)
- `data/` — AllenSDK cache: `visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb` (284 files, 247 GB),
  `visual-behavior-ophys-1.1.0/project_metadata/*.csv`, `visual-behavior-ophys_project_manifest_v1.1.0.json`
- `tutorials/` — 5 official tutorials (data access, loading ophys data, comparing trial types, mouse history, dataset manifest)
- `decoder.py`, `train_decoder.py` — decoder reference code
- `methods.txt`, `paper.pdf` (Vip-Sst strategy paper), `whitepaper.pdf` (VB-2P technical whitepaper)
- `cache/` — my scratch/analysis scripts (created during this work)

Environment verified: `python3` 3.13, numpy 2.4.4, torch 2.6.0+cu124, allensdk 2.16.2, 1x NVIDIA L4 (23 GB).

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=...)` | allensdk...behavior_project_cache | LOADING | Opens the local AllenSDK cache (works offline: reads the already-downloaded manifest). `from_local_cache` fails here because the cache was built by `from_s3_cache` and has no `manifests/` subfolder. |
| `cache.get_ophys_experiment_table()` | same | LOADING/CURATION | Metadata for all 1936 released experiments (mouse_id, cre_line, targeted_structure, session_type, passive, project_code, experience_level, ...). Used to select experiments. |
| `cache.get_behavior_ophys_experiment(oeid)` | same | LOADING | Returns `BehaviorOphysExperiment`; the only API used to read NWB content. ~3-4 s / experiment. |
| `ds.events` (`events`, `filtered_events`, `lambda`, `noise_std`) | BehaviorOphysExperiment | PROCESSING | Detected calcium events per cell, one value per ophys frame. This is the neural signal used by the Vip-Sst paper. |
| `ds.dff_traces` | " | PROCESSING | dF/F traces (already computed by the Allen pipeline — no need to compute dF/F ourselves; whitepaper "DF/F CALCULATION" describes how they were produced). |
| `ds.ophys_timestamps` | " | ALIGNMENT | Time (s, session clock) of every 2p frame; shared by `dff_traces` and `events`. |
| `ds.cell_specimen_table` (`valid_roi`, `cell_roi_id`) | " | CURATION | Segmented+filtered ROIs. In the released NWBs `valid_roi` is True for every cell in all 202 active experiments (ROI filtering from the whitepaper was applied before release). |
| `ds.trials` | " | CURATION/ALIGNMENT | One row per behavioral trial: `start_time`, `stop_time`, `change_time`, `go`, `catch`, `aborted`, `auto_rewarded`, `hit`, `miss`, `false_alarm`, `correct_reject`, `is_change`, `initial_image_name`, `change_image_name`. |
| `ds.stimulus_presentations` | " | PROCESSING | One row per flash: `start_time`, `end_time`, `image_name`, `omitted`, `is_change`, `is_sham_change`, `stimulus_block_name`, `trials_id`. Tutorials filter to `stimulus_block_name == 'change_detection_behavior'`. |
| `ds.running_speed` (`timestamps`, `speed`) | " | PROCESSING | Filtered running speed (cm/s) at 60 Hz (whitepaper: 10 Hz low-pass Butterworth of the wrap/transient-corrected encoder signal). |
| `ds.eye_tracking` (`timestamps`, `pupil_area`, `pupil_width`, `likely_blink`, ...) | " | PROCESSING | 30 Hz eye tracking; `pupil_area` is NaN on `likely_blink` frames (tutorial: "pupil area shows some missing data - these were points that were filtered out as outliers"). |
| `ds.metadata` | " | METADATA | `ophys_frame_rate`, `targeted_structure`, `imaging_depth`, `mouse_id`, `session_type`, `cre_line`, `ophys_session_id`, ... |
| tutorial `visual_behavior_compare_across_trial_types.py` | /app/tutorials | PROCESSING | Canonical per-trial plotting: everything (`running_speed`, `licks`, `eye_tracking`, `dff`) is queried by `timestamps >= trial.start_time and timestamps <= trial.stop_time`, i.e. all streams live on one common session clock. This is the reference for temporal alignment. |

### Notes
- dF/F does **not** need to be computed: the NWB files ship `dff_traces` and `events` already produced by the
  Allen pipeline (dewarping → motion correction → segmentation → ROI filtering → demixing → neuropil subtraction →
  dF/F → event detection), exactly as described in `methods.txt`.
- No electrophysiology, so no spike-quality filtering. ROI-quality filtering was already applied upstream
  (`valid_roi` all True), so no further neuron curation is justified.
- All data streams (2p, stimulus, running, eye) are already synchronized onto one session clock by the SDK
  (whitepaper "DATA SYNCHRONIZATION"), so alignment only requires resampling onto a common time base.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- The data are an AllenSDK **cloud cache** directory (`/app/data`), opened with
  `VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir='/app/data')` (works fully offline — it reads the
  already-downloaded `visual-behavior-ophys_project_manifest_v1.1.0.json`).
- `visual-behavior-ophys-1.1.0/project_metadata/`: `ophys_experiment_table.csv` (1936 rows x 31 cols),
  `ophys_session_table.csv` (703), `ophys_cells_table.csv` (133066), `behavior_session_table.csv` (4782).
  These describe the **whole release**; only a subset of NWB files is present locally.
- `visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_<oeid>.nwb`:
  **284 files** locally (247 GB). Each file = one *experiment* = one imaging plane of one *session*.
- Hierarchy: mouse -> container (one imaging plane tracked over days) -> ophys session (one continuous
  recording; 1 plane on Scientifica rigs, up to 8 planes on the Multiscope) -> experiment (plane).

### Locally available data (from the 284 NWB files)
| Split | experiments | ophys sessions | mice |
|---|---|---|---|
| all local | 284 | 247 | 38 |
| active behavior (`passive == False`) | 202 | 174 | 38 |
| passive viewing (`passive == True`) | 82 | — | — |
| project_code `VisualBehavior` (single plane) | 239 (202 active: 168) | | |
| project_code `VisualBehaviorMultiscope` | 45 (34 active) | 6 active sessions, 1 mouse | |

Active-behavior experiments break down as: session types OPHYS_1/3 (images A) and OPHYS_4/6 (images B);
cre lines Slc17a7 (107), Sst (62), Vip (33); targeted structures VISp (185) and VISl (17);
experience levels Familiar 110 / Novel 1 38 / Novel >1 54; frame rate 31 Hz (168) or 11 Hz (34 Multiscope).

### Dataset Size (from data files, active-behavior experiments, before my curation)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 29,444 (all `valid_roi == True`) |
| Neurons / experiment | mean 145.8, min 4, max 666 |
| Subjects | 38 mice |
| Sessions (ophys_session_id) | 174 (202 experiments) |
| Sessions / subject | mean 4.6 |
| Trials (total, go+catch, non-aborted, non-auto-rewarded) | 51,992 over 202 experiments |
| Trials / experiment | mean 257.4, min 39, max 409 |
| All trials incl. aborted / auto-rewarded | mean 686 / experiment (aborted mean 425, auto-rewarded mean 4.0) |
| Flashes / experiment | mean 4802, of which 3.5 % omitted |
| Images / session | 8 (+ `omitted`); 2 image sets of 8, 16 unique images across the dataset |

### Available variables (per `BehaviorOphysExperiment`)
- Neural: `events` (FastLZero detected calcium events, one value per ophys frame, 98.6 % zeros for a sparse
  Vip plane), `filtered_events` (same convolved with a half-normal kernel), `dff_traces`, `ophys_timestamps`
  (31 Hz: dt = 32.32 ms; 11 Hz Multiscope: dt = 93.23 ms), `cell_specimen_table` (`valid_roi`, x/y, roi_mask).
- Behavior/stimulus: `trials`, `stimulus_presentations`, `running_speed` (60 Hz), `eye_tracking` (30 Hz),
  `licks`, `rewards`, `get_performance_metrics()`, `get_rolling_performance_df()`.
- Data-quality facts found by scanning all 202 active experiments (`/app/cache/scan_experiments.py`):
  - no load errors; `events` contain no NaNs; `running_speed` contains no NaNs.
  - **3 experiments have a completely empty `eye_tracking` table** (oeids 795953296, 833631914, 806456687).
  - `pupil_area` is NaN on `likely_blink` frames: mean 3.5 % of frames, max 29.6 %.
  - `change_time` is never NaN for go/catch trials; it is **exactly equal** to the `start_time` of the
    corresponding `stimulus_presentations` flash (max abs difference 0.0 s), which is `is_change` for go
    trials and `is_sham_change` for catch trials.
  - time from trial `start_time` to `change_time`: >= 2.79 s for every trial in the dataset;
    `stop_time - change_time`: >= 4.20 s for every trial. Mean change time 4.19 s after trial start.
  - `go`, `catch`, `aborted` and `auto_rewarded` are mutually exclusive: (go|catch) already excludes every
    aborted and auto-rewarded trial (n_auto_and_go = 0 in all 202 experiments), and
    hit+miss+false_alarm+correct_reject == n(go|catch) exactly.
  - Multiscope planes of the same ophys session share the same `trials` table and (nearly) the same
    `ophys_timestamps` (planes imaged simultaneously in pairs; pair-to-pair offset 23 ms).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Task | go/no-go change detection | "mice are presented with a continuous series of briefly presented stimuli and they earn water rewards by correctly reporting when the identity of the image changes" (whitepaper) |
| Images per session | 8 | "Each session included 8 images, for a total of 64 possible transitions" |
| Flash cadence | 250 ms image + 500 ms gray = 750 ms | "natural images (250 ms stimulus duration) interspersed with periods of a gray screen (500 ms inter-stimulus duration)" (paper) |
| Omission probability | 5 % of flashes, never on a change or the flash before it | "stimuli were omitted with a 5% probability ... Stimulus changes and the stimulus immediately preceding the change were never omitted" |
| GO / CATCH proportion | 87.5 % / 12.5 % | "GO trials comprise 87.5% of all trials ... CATCH trials comprise 12.5% of all trials" |
| Mean change time within trial | 4.2 s (range 2.25-8.25 s, shifted by one flash cycle) | "resulting in a mean change time of 4.2 seconds" |
| Response window | 150-750 ms after the change | "a lick detected within response window (150 ms to 750 ms following image change...) resulted in a HIT" |
| Free-reward (auto-rewarded) trials | 5 at session start (+ after 10 consecutive misses in early training) | "behavior sessions across all phases began with 5 'free-reward' trials" |
| Neural signal used by the paper | detected calcium events | "For all analysis of neural data we used the detected calcium events"; "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of GCaMP6f" |
| Event detection | FastLZero, threshold 2.0x noise at 31 Hz, 2.6x at 11 Hz (chosen so magnitudes match across rigs) | whitepaper "EVENT DETECTION" |
| 2p frame rates | 31 Hz single plane, 11 Hz per plane Multiscope | whitepaper "DATA SYNCHRONIZATION" |
| Pupil | area of circle whose diameter is the ellipse major axis; NaN on `likely_blink` | whitepaper eye-tracking section |
| Running speed | cm/s, 10 Hz low-pass filtered, can be negative | whitepaper "BEHAVIOR METRICS" |
| Paper's dataset (their subset) | 8,619 exc (21 sessions, 9 mice), 470 Sst (15, 6), 1,239 Vip (21, 9) — familiar, Multiscope only | paper "Our dataset contains ..." |
| Whole VB-2P release | 376 imaging sessions, 82 mice | paper intro |
| Behavioral analysis unit | the 750 ms image-presentation interval | "By image presentation interval we refer to the 750 ms interval beginning with each image presentation. For image omissions we used the 750 ms following the time of the omission" |

### Processing Details
- **Temporal alignment**: all streams share one session clock (sync board at 100 kHz). The tutorials align
  everything by `timestamps` in seconds. The paper aligns neural activity to image presentations
  ("Decoding was performed on neural activity in the first 400 ms after each stimulus presentation").
- **Temporal binning**: the paper bins neural activity by image presentation interval (750 ms) and uses the
  first 400 ms after each presentation for decoding. There is no canonical fixed-width bin in the references,
  so a bin size has to be chosen for this conversion (see Step 5).
- **dF/F / events**: computed by the Allen pipeline and shipped in the NWB; nothing to recompute.

### Curation Steps
**Neuron curation rules**: ROI filtering (motion border, union/duplicate ROIs, apical dendrites, too small /
narrow / dim) was applied by the Allen pipeline before release; the NWB files only contain valid cells
(`valid_roi` True for all 29,444 cells). Neither reference paper applies further neuron filtering, so I apply none.

**Trial curation rules**: the change-detection task defines GO (change) and CATCH (sham change) trials, plus
ABORTED (premature lick) and AUTO-REWARDED (free reward) trials. The decoder task specifies: keep go + catch,
drop aborted + auto-rewarded. Sessions: only *active behavior* sessions are "the Visual Behavior task"
(in passive sessions the lick spout is retracted and no choices exist), so passive sessions are dropped.

### Decoders Trained (in the reference paper)
| Decoded variable | Accuracy |
|---|---|
| image change vs. repeat (random forest, 400 ms after flash, per imaging plane) | reported only in Figure 6A as % correct, well above the 50 % chance level; equal across strategies |
| hit vs. miss | Figure 6C, above chance, higher for "visual strategy" sessions |
| false alarm | "for all cell classes, false alarm decoding performance was very low" (Figure S22) |
No numeric decoding accuracies are given in the text of either reference, and neither decodes image identity,
running speed or pupil, so the reference papers only constrain the *qualitative* expectation: image change and
trial outcome should be decodable above chance, with change >> outcome, and with per-plane (per-session) populations.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| What is a "session" | SDK loads one *experiment* (imaging plane) at a time; `ophys_session_table` groups planes into sessions | 202 active experiments in 174 ophys sessions; Multiscope planes of a session share trials and timestamps | whitepaper defines session = one continuous recording, experiment = one plane | Use **ophys_session** as the "session" of the output format: planes recorded simultaneously are concatenated along the neuron axis. This is the only definition under which all neurons of a session are simultaneously recorded, and it is what lets `brain_region_idx` carry VISp/VISl within a session. |
| Which sessions are "the Visual Behavior task" | `passive` / `behavior_type` columns separate active behavior from passive viewing | 202 active, 82 passive locally | Passive sessions = "mice ... view the task stimuli with the lick spout retracted so they are unable to earn water rewards"; the paper's behavioral analyses use "all active behavioral sessions" | Keep active-behavior sessions only. Trial outcome (hit/miss/FA/CR) is undefined/degenerate without licking. Both project codes (single-plane + Multiscope) are kept: both run the identical change-detection task. |
| Neural signal | NWB ships `dff_traces`, `events`, `filtered_events` (all produced by the Allen pipeline) | events are sparse (84-99 % zeros per frame) | Vip-Sst paper uses detected calcium **events**; the whitepaper's processing chapter ends at **dF/F** | Measured both (Steps 7 and 12): dF/F carries substantially more decodable information at 250 ms resolution, so `dff` is used and the deviation from the paper is documented; `--signal events` reproduces the paper's choice. |
| Trial-type flags | `trials.go/catch/aborted/auto_rewarded/hit/miss/false_alarm/correct_reject` | flags mutually exclusive; catch = 12.5 % | whitepaper: GO 87.5 %, CATCH 12.5 % | `(go | catch)` selection reproduces the 12.5 % catch fraction exactly -> consistent. |
| Change time | `trials.change_time` | equals `stimulus_presentations.start_time` of the change/sham flash to within 0 s | whitepaper: "the actual change time was determined as the nearest flash from the drawn time" | Align trials on `change_time`; the 750 ms flash grid is therefore phase-locked to t = 0. |
| Pupil "diameter" | `eye_tracking.pupil_area` (NaN on blinks), `pupil_width`/`pupil_height` are half-axes | 3.5 % NaN on average | whitepaper: pupil area = area of circle whose diameter is the ellipse major axis | diameter = 2*sqrt(pupil_area/pi) — a monotone function of `pupil_area`, so percentile bins are identical either way, and it uses the blink-cleaned column. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `ds.dff_traces.dff` + `ds.ophys_timestamps` | `neural` | mean dF/F in each 250 ms bin; planes of a Multiscope session stacked along the neuron axis | `cache.get_behavior_ophys_experiment(oeid).dff_traces` | float32, (n_neurons, 21) per trial. `ds.events` was the initial plan (reference paper) and is still available via `--signal events`; see decision 7 |
| — | `input` | none required by the task | — | shape (0, 21) per trial, `input_names = []` |
| `ds.stimulus_presentations.image_name` | `output[0]` `image_identity` | label of the flash interval ([flash start, next flash start)) containing the bin centre; 16 image names + `omitted` | tutorial filter `stimulus_block_name == 'change_detection_behavior'` | categorical, 17 values |
| `ds.trials.change_time` + `is_change` | `output[1]` `image_change` | 1 for bins whose centre lies in [change_time, change_time+0.5) on go trials, else 0 | — | binary, time-varying; see decision 8b |
| `ds.running_speed` | `output[2]` `running_speed_bin` | bin-average speed, then per-session quintile bins | tutorial `plot_running` | 5 categories |
| `ds.eye_tracking.pupil_area` | `output[3]` `pupil_diameter_bin` | diameter = 2*sqrt(area/pi), linear interpolation over blink NaNs, bin-average, per-session quintile bins | tutorial `plot_pupil` | 5 categories |
| `ds.trials.hit/miss/false_alarm/correct_reject` | `output[4]` `trial_outcome` | one label per trial, broadcast over the 21 bins | tutorial `trials.query('hit')` | 4 categories, static per trial |
| `ds.metadata['mouse_id']` | `subjects` / `subject_idx` | unique sorted list | — | |
| `ds.metadata['targeted_structure']` | `brain_regions` / `brain_region_idx` | per experiment, repeated for that plane's neurons | — | VISp, VISl |

### Key Decisions
1. **Sessions = ophys sessions, active behavior only**: 174 candidate sessions (202 experiments). Passive
   sessions excluded (no licking -> no trial outcome, and they are not "the Visual Behavior task").
   Sessions whose eye tracking is entirely missing are excluded because pupil diameter is a required output.
2. **Trials = go + catch, aborted and auto-rewarded excluded** (as specified by the decoder task; this also
   matches the whitepaper's 87.5/12.5 GO/CATCH structure).
3. **Alignment event = `change_time`** (the actual image change on go trials, the sham change on catch
   trials). This is the only event common to every included trial, it is exactly a flash onset, and the
   reference paper aligns its analyses to image presentations/changes.
4. **Trial window = [-2.25, +3.0] s around the change**, i.e. 7 complete 750 ms flash cycles
   (3 pre-change flashes, the change flash, 3 post-change flashes). Every included trial contains this window
   entirely inside its own `start_time`/`stop_time` (dataset minima: 2.79 s before, 4.20 s after), so no trial
   is padded or truncated and no data from a neighbouring trial leaks in.
5. **Bin size = 250 ms** (21 bins/trial). It divides the 750 ms flash cycle exactly into 3 bins
   (image / gray / gray) and is a whole multiple of the 250 ms image duration; and it is >= 2 ophys frames
   even at the 11 Hz Multiscope rate (93.23 ms), so no bin is ever empty for any rig. A bin size equal to the
   native ophys frame period cannot be used because the format requires one bin size for all sessions while
   the rigs sample at 31 Hz and 11 Hz.
6. **"Temporally align based on ophys timestamp"**: the 250 ms bin grid is anchored to each trial's change
   time on the shared session clock; each ophys frame is assigned to a bin by its `ophys_timestamps` value,
   and every other stream (running, pupil, stimulus) is resampled onto those same absolute-time bins.
7. **Neural = `dff`** (bin-averaged dF/F). *Initial plan*: `events`, the detected calcium events used by
   the Vip-Sst paper. *Final decision*: dF/F, after measuring both (Step 7 table, Step 12 Check 2). Both
   come from the same Allen pipeline and are shipped in the NWB (nothing is recomputed here); at the
   250 ms resolution this task needs, `events` are 84-99 % zeros and cost roughly half the decodable
   stimulus information (image identity 0.26 vs 0.51 balanced accuracy). dF/F is the pipeline's primary
   neural product (whitepaper "DF/F CALCULATION") and is what the SDK tutorials plot per trial.
   `--signal events` reproduces the reference paper's choice exactly.
8. **Image identity is labelled per 750 ms flash interval**, exactly the paper's "image presentation
   interval" convention ("the 750 ms interval beginning with each image presentation; for omissions the
   750 ms following the time of the omission"). The gray period therefore carries the identity of the image
   that was flashed at the start of its interval, which also matches the ~100-300 ms lag of calcium events.
   `omitted` is kept as its own category, since an omission is a distinct stimulus condition with a known
   neural signature in this dataset.
8b. **`image_change` marks the 500 ms after the change** (2 bins). "Right after a change" has to be given
   a width; 500 ms is the representable window closest to the 400 ms post-presentation window the
   reference paper decodes in, it sits inside the behavioural response window (150-750 ms), and of the
   three widths tested (250/500/750 ms) it is the one the change response is actually present in
   (Step 12 Check 2). Catch trials have no change of identity, so they are 0 throughout.
9. **Image identity uses global image names** (16 across image sets A and B, + `omitted` = 17 categories)
   rather than within-session indices, so that a label means the same physical image in every session.
10. **Quintile bins for running speed and pupil are computed per session** over exactly the timepoints that
    are written out. Pupil area is in camera pixels^2 and is not comparable across sessions/mice (eye size,
    camera distance), and running speed distributions differ strongly across mice, so per-session percentiles
    are the only way the 5 labels carry the same meaning in every session; it also guarantees the 20/20/20/20/20
    class balance that "five equal percentile bins" asks for.
11. **Blink NaNs in pupil are linearly interpolated** over time within the session (edges held constant)
    before binning, rather than dropping trials: blinks are short (3.5 % of frames) and dropping them would
    bias the trial set toward calm periods.
12. **Trial outcome is broadcast over time** because the format requires a single (d_output, T) array per trial.

### Planned Sanity Checks
- [x] catch fraction of selected trials ~= 12.5 % (whitepaper)
- [x] mean change time within trial ~= 4.2 s (whitepaper)
- [x] 8 images per session, 16 unique + `omitted` across the dataset
- [x] omitted flashes ~ 3.5-5 % of flashes, and never at the change or the flash before it
- [x] image identity constant within each 750 ms flash interval and changing exactly at t = 0 on go trials,
      never on catch trials (100 % / 100 % / 0 %, Step 10)
- [x] `image_change` == 1 exactly on go trials, only in the bins covering [0, 0.5) s (bins 9-10); its
      per-trial mean equals the go fraction 0.8746 (Step 10)
- [x] running speed / pupil quintiles contain 20.0 % of timepoints each, per session (Steps 7 and 9)
- [x] trial outcome distribution matches the counts computed directly from `ds.trials` *and* the
      independent `behavior_session_table` counts in 169/171 sessions, the 2 differences being exactly
      the 9 trials dropped for sensor dropout (Step 9)
- [x] spot-check binned neural values against raw `dff_traces` + `ophys_timestamps` with `np.allclose`
      (`cache/sanity_checks.py`, Step 10)
- [x] number of neurons per session == sum of `len(cell_specimen_table)` over that session's experiments
- [x] total trials == sum over sessions of n(go|catch) from `ds.trials` (43,975 - 917 - 9 = 43,966)

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` — `python -u convert_data.py <outfile> [--full|--sample] [--show-processing]`.
Extra options: `--nproc N` (default 8), `--signal {dff,events,filtered_events}`,
`--nsessions N` and `--neural-lag S` (both testing-only, used for the comparisons in Steps 7/12).

Structure:
- `get_cache()` / `select_sessions()` — open the local AllenSDK cache and pick the active-behaviour
  experiments whose NWB file is present, grouped by `ophys_session_id`.
- `convert_session()` — loads every plane of one session through
  `cache.get_behavior_ophys_experiment()`, selects trials, bins all streams, builds the outputs.
- `bin_mean()` — vectorised binning: one `np.searchsorted` of all trials' bin edges into a stream's
  timestamps, then a single `np.add.reduceat` for all bins of all trials at once.
- `assemble()` — builds the final dictionary (global image-name list, brain regions, subjects, metadata).
- `plot_processing()` — the `--show-processing` figures.

Code inefficiencies identified:
- Naively looping over trials (x257) and neurons (x666) per session would dominate the runtime.
- Loading an NWB file is ~3.5 s and is the only real cost (98 % of the per-session time).
- A per-session `cumsum` over the full 140k-frame x n-neuron trace would cost hundreds of MB per worker.

Code speedups added:
- One `searchsorted` + one `reduceat` per data stream per session (no Python loop over trials or bins).
- Sessions are processed in parallel with `multiprocessing.Pool` (16 workers).
- Binning is done directly on the trace instead of a cumulative sum, so memory stays at one experiment
  per worker; results are stored as float32.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

`python -u convert_data.py /app/sample_data.pkl --sample --show-processing` converts two sessions
chosen to exercise both code paths: `775289198` (single-plane Scientifica, 31 Hz, 1 experiment) and
`951410079` (Multiscope, 11 Hz, 7 simultaneously recorded planes in VISp+VISl).

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Neurons (total) | 177 (89 + 88) |
| Neurons / session | 88.5 |
| Subjects | 2 |
| Sessions / subject | 1 |
| Trials (total) | 248 |
| Trials / session | 39, 209 |
| Timepoints / trial | 21 (250 ms bins) |
| image_identity distribution | im061 0.127, im062 0.122, im063 0.127, im065 0.103, im066 0.122, im069 0.112, im077 0.113, im085 0.143, omitted 0.031 |
| image_change distribution | [0.917, 0.083] |
| running_speed_bin distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| pupil_diameter_bin distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| trial_outcome distribution | hit 0.383, miss 0.484, false_alarm 0.036, correct_reject 0.097 |
| dinput | 0 (no decoder inputs for this task) |

### Processing Plots Review
`processing_<session>.png` shows, for one go trial: the raw dF/F traces with the 250 ms bin edges,
the change time and the flash intervals overlaid; the binned neural matrix; raw vs. binned running
speed; the running quintile labels; raw (with blink NaNs), interpolated and binned pupil diameter;
the per-bin image identity against the actual labelled flashes; and `image_change`.
`processing_<session>_summary.png` shows the session-level distributions.

Anomalies found and fixed:
1. **Binning bug (found in the first summary plot)**: binned running speed reached 1750 cm/s and
   binned pupil 300,000 px, far outside the raw ranges (max 14.7 cm/s, 148 px). Cause:
   `np.add.reduceat`'s *last* segment always runs to the end of the array, so the last bin of the
   last trial of each session averaged everything from that bin to the end of the recording. Fixed by
   truncating the array at the final bin edge. A permanent guard was added (`within_range()`): every
   binned stream must stay inside the min/max of the raw stream it was averaged from.
2. **14 sessions were being dropped for "empty pupil bins"**. Investigation showed the eye-tracking
   camera drops a handful of frames in many sessions (1-4 bins per session, longest gap 0.76 s).
   Dropping whole sessions for a sub-second sensor dropout is wrong, so empty behavioural bins are now
   filled by linear interpolation in time and only trials with a dropout longer than 2 s are dropped
   (9 trials in the entire dataset).

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| vectorised binning (searchsorted + reduceat, no per-trial loop) | binning is 0.05 s/session vs. ~1 s/session looped |
| 16-way multiprocessing over sessions | 700 s -> 69 s wall clock |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| NWB load (per experiment, 202 experiments) | 3.5 s | 700 s serial |
| binning + outputs | 0.1 s | 20 s serial |
| **full conversion, 16 workers** | — | **69 s measured** (well under the 15 min budget) |

### Neural-signal comparison (empirical, 14 sessions spread over the dataset)
Validation balanced accuracy, same decoder settings:
| signal | image_identity | image_change | running | pupil | outcome |
|---|---|---|---|---|---|
| `events` (reference paper) | 0.258 | 0.570 | 0.236 | 0.233 | 0.297 |
| `filtered_events` (2 sessions) | 0.310 | 0.639 | 0.240 | 0.261 | 0.342 |
| **`dff` (chosen)** | **0.511** | **0.682** | **0.338** | **0.307** | **0.314** |
| `dff`, z-scored per neuron | 0.521 | 0.665 | 0.333 | 0.303 | 0.273 |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None ("Data format is valid, no errors or warnings.")

### Decoder Results (Sample, 2 sessions / 248 trials)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| image_identity | 0.6475 | 0.5325 | 0.111 |
| image_change | 0.7709 | 0.7092 | 0.500 |
| running_speed_bin | 0.3322 | 0.2420 | 0.200 |
| pupil_diameter_bin | 0.4240 | 0.3185 | 0.200 |
| trial_outcome | 0.5010 | 0.3701 | 0.250 |

Loss decreased monotonically over the 200 epochs (1.52 -> 1.15); every output is above chance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 707 MB, 171 sessions, 43,966 trials, 29,168 neurons, 38 mice (conversion 69 s)
- `verification_full_out.txt`: created, **no errors and no warnings**

### Data accounting (nothing silently lost)
| | experiments | sessions | neurons | trials |
|---|---|---|---|---|
| active-behaviour data available locally | 202 | 174 | 29,444 | 43,975 (go+catch) |
| dropped: no eye-tracking table at all | 3 | 3 | 276 | 917 |
| dropped: sensor dropout > 2 s inside the trial window | — | — | — | 9 |
| **converted** | **199** | **171** | **29,168** | **43,966** |
29,444 - 276 = 29,168 and 43,975 - 917 - 9 = 43,966, i.e. every neuron and trial is accounted for.

### Consistency Check
| Statistic | Reference Papers | Reference Code / SDK tables | Reference Data (direct scan) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Subjects (mice) | 82 in the whole release | 38 with local active NWBs | 38 | 38 | yes |
| Sessions | 376 imaging sessions in the whole release | 174 local active ophys sessions | 174 | 171 (-3 without eye tracking) | yes |
| Neurons total | 8,619 exc + 470 Sst + 1,239 Vip in the paper's subset | 29,444 in `ophys_cells_table` for these experiments | 29,444 | 29,168 | yes |
| Neurons / session | — | — | mean 145.8 / experiment | mean 170.6 / session, min 6, max 666 | yes (planes merged per session) |
| Trials / session | — | `behavior_session_table.go_trial_count + catch_trial_count` | mean 257.4 | mean 257.1 | yes |
| GO / CATCH split | 87.5 % / 12.5 % (whitepaper) | — | 87.5 / 12.5 | 87.46 / 12.54 | yes |
| Hit / miss / FA / CR counts, per session | — | `behavior_session_table` hit/miss/false_alarm/correct_reject_trial_count | — | identical in 169/171 sessions; the 2 differences are exactly the 9 trials dropped for sensor dropout | yes |
| Trial outcome distribution | — | — | hit .3065, miss .5682, FA .0179, CR .1074 (per experiment) | hit .317, miss .558, FA .019, CR .106 | yes |
| Hit rate / FA rate | d' > 1 peak required for QC | — | — | 0.362 / 0.151 (session-average d' = 0.68, consistent with a *peak* d' > 1) | yes |
| Flash cadence | 250 ms image / 750 ms cycle | `stimulus_presentations.duration` 250.3 ms, cycle 750.7 ms | same | image identity constant within each 750 ms interval, changes exactly at t = 0 | yes |
| Omission rate | 5 % of flashes, never at a change or the flash before it | 3.5 % of flashes | 3.5 % | 3.0-4.9 % per bin, **exactly 0 %** in the bins covering the pre-change flash and the change flash | yes |
| Images | 8 per session, 2 image sets | 16 unique + `omitted` | same | 16 + `omitted`, each 5.8-6.4 % of timepoints | yes |
| Neural time bin | paper uses the 750 ms image interval / first 400 ms | 2p frame 32.3 ms (31 Hz) or 93.2 ms (11 Hz) | same | 250 ms (uniform across rigs; 3 bins per flash cycle) | see Step 5 decision 5 |
| Running speed | cm/s, can be negative | `running_speed.speed` | −10.5 to 100 cm/s across sessions | quintile labels; edges stored per session in metadata | yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification (`/app/verification_full_out.txt`)
`Data format is valid, no errors or warnings.` — there is nothing to fix and no warning to explain.
Checked by hand in the log: 171 sessions all with T = 21; n_neurons 6-666; output ranges
0-16 / 0-1 / 0-4 / 0-4 / 0-3 as intended; both brain regions populated (VISp 29,006, VISl 162);
every output value occurs.

### Check 2: Sanity checks against the original data (`/app/cache/sanity_checks.py`)
Four randomly chosen sessions (incl. a 7-plane Multiscope session), three random trials each. Every
quantity is recomputed from a freshly loaded `BehaviorOphysExperiment` using *different* code from the
converter (explicit boolean masks / row lookups instead of `searchsorted`+`reduceat`):
- `neural` vs. brute-force per-bin mean of `dff_traces` over `ophys_timestamps` — `np.allclose`, PASS
  (this also verifies the trial matching, the [-2.25, +3.0] s window and the plane concatenation order)
- `image_identity` vs. the last `stimulus_presentations` row starting before each bin centre — PASS
- `image_change` vs. `trials.is_change` and `change_time` — PASS
- `trial_outcome` vs. the `hit/miss/false_alarm/correct_reject` columns — PASS
- `running_speed_bin` vs. brute-force binned `running_speed` + the stored quintile edges — PASS
- `pupil_diameter_bin` vs. brute-force binned 2*sqrt(pupil_area/pi) + stored edges — PASS
- neuron count, per-neuron brain region, subject id, empty input arrays — PASS
Result: **ALL CHECKS PASSED** — 92 individual checks, 0 failures
(`/app/cache/sanity_checks_out.txt`, re-run against the final `converted_data.pkl`).

Whole-dataset structural checks (`/app/cache/` one-liners, reported in Step 9):
- image identity is constant within every 750 ms flash interval (100 % of 43,966 trials)
- identity differs across t = 0 on 100 % of go trials and on 0 % of catch trials
- `image_change` is 1 only in bins 9-10 and only on go trials; its mean over trials is 0.8746 = the
  go fraction
- `omitted` never appears in the change flash or the flash before it (whitepaper claim reproduced)
- per-session hit/miss/FA/CR counts equal the Allen `behavior_session_table` counts (see Step 9)

### Check 3: Reference code comparison
| Step | Reference | This conversion | Same? |
|---|---|---|---|
| (a) loading | tutorials: `VisualBehaviorOphysProjectCache` -> `get_behavior_ophys_experiment(oeid)` | identical; no NWB opened directly | yes |
| (b) neuron filtering | Allen pipeline ROI filtering applied before release; neither paper filters further | none applied; `valid_roi` is True for all 29,444 cells | yes |
| (b) trial filtering | tutorial `trials.query('hit')` etc.; whitepaper defines go/catch/aborted/auto-rewarded | `(go|catch) & ~aborted & ~auto_rewarded` as the decoder task specifies | yes |
| (b) session filtering | paper: "all active behavioural sessions"; passive sessions "not analyzed here" | active behaviour only; 3 sessions without eye tracking also dropped (pupil is a required output) | yes + 1 task-driven addition |
| (c) temporal alignment | tutorials query every stream by `timestamps >= trial.start_time & <= trial.stop_time` on the shared session clock; paper aligns to image presentations | every stream is averaged over the same absolute-time bins, anchored to `trials.change_time`; 2p frames assigned by `ophys_timestamps` | yes |
| (d) binning | paper: the 750 ms image-presentation interval, decoding on the first 400 ms | 250 ms bins, exactly 3 per flash interval; `image_change` marks the first 500 ms (the representable window closest to the paper's 400 ms) | compatible |
| (e) input construction | — | none (task specifies no decoder inputs) | n/a |
| (f) output construction | paper assigns behavioural events to the 750 ms image-presentation interval; whitepaper defines pupil area and running speed | image identity uses exactly that interval convention; pupil = 2*sqrt(pupil_area/pi) per the whitepaper definition; running speed is the SDK's filtered `speed` | yes |
| neural signal | paper uses detected calcium **events**; whitepaper's pipeline produces **dF/F** | `dff` (documented deviation, see Step 12) | deviation, justified |

Reasoning for the two deviations:
1. **dF/F instead of events**: the task is to train a neural decoder, and events at 250 ms resolution
   are extremely sparse (84-99 % of frames are exactly zero), which costs a large amount of decodable
   information (image identity 0.26 vs. 0.51 balanced accuracy, see Step 7 table). dF/F is the primary
   neural product of the same Allen pipeline (whitepaper "DF/F CALCULATION"), is already computed in
   the NWB, and is what the SDK tutorials plot per trial. The `--signal events` option reproduces the
   paper's choice exactly if wanted.
2. **250 ms bins instead of the 750 ms image interval**: the format requires time-varying outputs and
   one bin size for all sessions; 250 ms is the largest bin that still resolves image vs. gray within
   a flash cycle and the smallest that is never empty at the 11 Hz Multiscope frame rate.

### Check 4: Key statistics comparison
See the table in Step 9 — every statistic available in the references (GO/CATCH 87.5/12.5, mean change
time 4.2 s, 8 images/session, omission structure, per-session trial counts, per-session outcome counts,
neuron counts) is reproduced by the converted data. No discrepancy remained unexplained.

### Check 5: Edge cases
- **Trial-window containment**: the window [-2.25, +3.0] s is inside every trial's own
  `start_time`/`stop_time` (dataset minima 2.79 s before and 4.20 s after the change), so no trial is
  truncated and no neighbouring trial leaks in. Verified for all 43,966 trials (edges are strictly
  increasing across the session, which the converter asserts).
- **First/last trial of a session**: handled by the same edge check; the reduceat truncation bug that
  affected the very last bin of each session was found and fixed (Step 7).
- **Sessions with very few neurons** (min 6) are kept: the reference papers apply no cell-count
  criterion and the decoder handles small populations.
- **Sessions with 39 trials** (minimum) are kept; the format only requires >= 2 trials.
- **Missing eye tracking** (3 sessions) -> session dropped; **short sensor dropouts** -> interpolated;
  **dropouts > 2 s** -> those trials dropped (9 in total).
- **Blinks**: `pupil_area` is NaN on `likely_blink` frames (3.5 % on average, up to 29.6 %); these are
  interpolated over in time before binning, so no NaN reaches the output.
- **Auto-rewarded / aborted trials**: excluded; verified that `go`/`catch` never overlap them.
- **Two image sets**: the global image list is the union (16 images); each session contains 8 of them.
- **Multiscope sessions**: the 7 planes of a session share one `trials` table and one behaviour stream;
  verified that the trials tables are identical across planes before merging.

### Issues Found and Resolved
1. `np.add.reduceat` last-segment bug -> corrupted the last bin of each session. Fixed + permanent
   range guard. Re-ran sample and full conversion; re-ran all checks.
2. 14 sessions lost to sub-second eye-camera dropouts -> gap interpolation + per-trial drop rule.
   Re-ran the full conversion: 157 -> 171 sessions.
3. `events` chosen initially per the reference paper -> switched to `dff` after the measured decoding
   comparison above (documented deviation).
4. `image_change` initially marked the whole 750 ms flash interval -> shortened to 500 ms (see Step 12).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

`python -u train_decoder.py /app/converted_data.pkl --plot-samples` (GPU, ~3 min).

### Training Progress
- Loss decreasing: Yes, monotonically from 1.64 to 1.298 over 200 epochs; test loss 1.388.

### Decoder Results (Full: 171 sessions, 43,966 trials, 29,168 neurons)
| Output | Classes | Training Balanced Acc | Validation Balanced Acc | Chance | Val / chance |
|--------|---|-------------|--------|--------|---|
| image_identity | 17 | 0.4824 | **0.4402** | 0.0588 | 7.5x |
| image_change | 2 | 0.7534 | **0.7016** | 0.5000 | 1.40x |
| running_speed_bin | 5 | 0.3648 | **0.3088** | 0.2000 | 1.54x |
| pupil_diameter_bin | 5 | 0.3593 | **0.2801** | 0.2000 | 1.40x |
| trial_outcome | 4 | 0.5030 | **0.3084** | 0.2500 | 1.23x |

`sample_trials.png` and `predictions.png` were produced; the input panel is empty by design
(the task specifies no decoder inputs).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
Every output is above chance; none is below. Three outputs are below 1.5x chance
(image_change 1.41x, pupil 1.40x, trial_outcome 1.23x), so each was investigated with the confusion
matrices and per-timepoint accuracies in `/app/cache/confusion_analysis.py`
(output in `/app/cache/confusion_analysis_out.txt`, figure `/app/cache/accuracy_vs_time.png`):

- **image_change** (recall 0.748 no-change / 0.661 change, 0.741 overall accuracy): a 1.5x ratio is impossible for a binary
  variable without reaching 75 %. The decoder is asked to label *every* 250 ms bin of the trial, not
  just the change flash vs. the preceding flash as in the reference paper. Per-timepoint accuracy is
  0.87-0.91 in the last bin of each flash cycle and 0.61-0.69 in the change bins, i.e. the errors are
  where they should be.
- **pupil_diameter_bin** (recall 0.502 / 0.210 / 0.160 / 0.186 / 0.335): the confusion matrix is
  strongly banded — errors are concentrated in *adjacent* quintiles (59.8 % of predictions within
  +/-1 quintile vs. 52.0 % expected by chance) and the extreme quintiles are decoded best. That is the
  signature of decoding a discretized continuous variable, not of a bug. Pupil also has no
  event-locked structure, so per-timepoint accuracy is flat (0.27-0.29) across the trial, as expected.
- **trial_outcome** (recall hit 0.513, miss 0.352, FA 0.252, CR 0.145): the label is constant over the
  whole trial, but 9 of the 21 bins are *before* the change, where hit/miss/FA/CR is not yet
  determined. Per-timepoint accuracy rises from ~0.35 before the change to ~0.42 after it — exactly
  the profile expected if the alignment is correct.

The changes that were tested to try to raise these numbers (and their outcomes) are in Check 2.

### Check 2: Accuracy comparison to papers
| Variable | This conversion (val. balanced acc) | Reference-paper value |
|---|---|---|
| image change | 0.702 (per 250 ms bin, all bins of the trial) | Figure 6A reports % correct for a random forest that classifies *the change flash vs. the flash immediately before it*, using the first 400 ms of activity, per imaging plane; no number is given in the text |
| trial outcome (hit vs. miss part) | hit recall 0.513, miss recall 0.352 in a 4-class problem | Figure 6C, hit vs. miss, above chance, no number in the text |
| false alarm | FA recall 0.252 (rarest class, 1.9 % of trials) | "for all cell classes, false alarm decoding performance was very low" (Figure S22) — consistent |
| image identity | 0.441 over 17 classes (7.5x chance) | not decoded in either reference |
| running speed / pupil | 0.309 / 0.280 over 5 classes | not decoded in either reference |
Neither reference reports a numeric decoding accuracy in its text, and neither decodes image identity,
running speed or pupil, so only the qualitative comparison above is possible. The two variables the
paper does decode behave as the paper describes: image change is decoded well, false alarms poorly.
The paper's task is also strictly easier (balanced 2-class problem restricted to the informative
timepoints, non-linear classifier), so its % correct is expected to exceed our per-bin accuracy.

Conversion changes that were tested to raise accuracy (each re-converted and re-trained on the same
14-session subset):
| Variant | image_identity | image_change | running | pupil | outcome | Decision |
|---|---|---|---|---|---|---|
| `events` (reference paper's signal) | 0.258 | 0.570 | 0.236 | 0.233 | 0.297 | rejected |
| `filtered_events` | 0.310* | 0.639* | 0.240* | 0.261* | 0.342* | rejected (*2-session test) |
| `dff` | 0.511 | 0.682 | 0.338 | 0.307 | 0.314 | **chosen** |
| `dff` z-scored per neuron | 0.521 | 0.665 | 0.333 | 0.303 | 0.273 | rejected (no gain, less faithful) |
| neural grid shifted +125 ms (response lag) | 0.485 | 0.688 | 0.342 | 0.297 | 0.324 | rejected (no gain) |
| neural grid shifted +250 ms | 0.489 | 0.676 | 0.347 | 0.292 | 0.326 | rejected (no gain) |
| `image_change` = 750 ms (full flash interval) | 0.511 | 0.682 | — | — | — | rejected |
| **`image_change` = 500 ms** | 0.510 | **0.729** | — | — | — | **chosen** |
| `image_change` = 250 ms | 0.511 | 0.704 | — | — | — | rejected |
| `omitted` folded into the repeated image (16 classes) | 0.529 (chance 0.0625) | — | — | — | — | rejected: same ratio to chance (8.5x vs 8.7x) and less faithful |
The only change that produced a real gain was the 500 ms `image_change` window, which was adopted; it
is also the representable window closest to the reference paper's 400 ms decoding window and matches
the behavioural response window (150-750 ms). The full conversion was re-run with it
(image_change 0.655 -> 0.702 on the full dataset).

### Check 3: Train vs validation gap
| Output | Train | Val | Ratio |
|---|---|---|---|
| image_identity | 0.4824 | 0.4402 | 1.10 |
| image_change | 0.7534 | 0.7016 | 1.07 |
| running_speed_bin | 0.3648 | 0.3088 | 1.18 |
| pupil_diameter_bin | 0.3593 | 0.2801 | 1.28 |
| trial_outcome | 0.5030 | 0.3084 | **1.63** |
Only `trial_outcome` exceeds 1.5x. This is overfitting, not leakage: the outcome label is constant
within a trial, so a trial contributes 21 identical labels and the model (171 session-specific
projections + a shared linear read-out, trained for 200 epochs without early stopping) can memorise
trial-level idiosyncrasies. There is no path by which validation-trial information reaches training:
the train/validation split is over whole trials, and every feature of a trial (neural bins) comes only
from that trial's own time window. Per-session quantile edges for running/pupil are computed from the
session's own timepoints — that is a property of the *labels*, shared by train and test trials of a
session, and is unavoidable for a per-session discretization (a global discretization was rejected for
the reasons in Step 5 decision 10); it cannot leak neural information.

### Issues Found and Resolved
- `image_change` window shortened from 750 ms to 500 ms (+0.05 balanced accuracy); full pipeline re-run,
  all Step 10 checks re-run and still passing (`sanity_checks.py`: ALL CHECKS PASSED).
- No other change tested improved accuracy, and no check revealed a conversion error.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `/app/README.md` created (dataset description, how to load, format spec, key statistics,
      decoder performance, file list)
- [x] `/app/cache/` folder created for all analysis/investigation scripts, documented in
      `/app/cache/README_CACHE.md`
- [x] Files organized. Deliverables in `/app`:
      `CONVERSION_NOTES.md`, `README.md`, `convert_data.py`, `converted_data.pkl`, `sample_data.pkl`,
      `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt`,
      `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt`,
      `processing_775289198.png`, `processing_951410079.png` (+ `_summary` versions),
      `sample_trials.png`, `predictions.png` (produced by `train_decoder.py`).
