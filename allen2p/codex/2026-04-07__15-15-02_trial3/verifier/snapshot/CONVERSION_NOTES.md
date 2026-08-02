# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory Visual Behavior 2P / local paper dataset
- **Date started**: 2026-04-07
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`
- `tutorials/`
- `whitepaper.pdf`

Python environment check:
- `python3`: 3.13.12
- `numpy`: 2.3.5
- `torch`: 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `VisualBehaviorOphysProjectCache.get_behavior_ophys_experiment` | `code/allensdk/brain_observatory/behavior/behavior_project_cache/behavior_project_cache.py` | LOADING | Project-cache entry point returning a `BehaviorOphysExperiment` for one `ophys_experiment_id`. |
| `BehaviorOphysExperiment.from_lims` / `from_nwb` | `code/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py` | LOADING | Assembles experiment object from behavior session, sync-derived ophys timestamps, cell specimens, projections, motion correction, metadata. |
| `BehaviorSession.from_lims` / `from_nwb` | `code/allensdk/brain_observatory/behavior/behavior_session.py` | LOADING | Loads behavior-side streams: stimulus timestamps, running, licks, rewards, trials, eye tracking, metadata. |
| `StimulusTimestamps.from_sync_file` | `code/allensdk/brain_observatory/behavior/data_objects/timestamps/stimulus_timestamps/stimulus_timestamps.py` | PROCESSING | Builds stimulus/behavior timestamps from sync file, then adds monitor delay; supports multiple stimulus blocks when needed. |
| `OphysTimestamps.from_sync_file` / `OphysTimestampsMultiplane.from_sync_file` | `code/allensdk/brain_observatory/behavior/data_objects/timestamps/ophys_timestamps.py` | PROCESSING | Loads microscope frame timestamps; for multiplane sessions subsamples interleaved frames by imaging plane group. |
| `Trials.from_stimulus_file` | `code/allensdk/brain_observatory/behavior/data_objects/trials/trials.py` | PROCESSING | Parses `trial_log`, constructs per-trial dataframe with trial bounds, outcomes, change timing, image names, lick/reward timing. |
| `Trial._get_trial_data` / `_get_trial_timing` / `add_change_time` | `code/allensdk/brain_observatory/behavior/data_objects/trials/trial.py` | PROCESSING | Defines `go`, `catch`, `aborted`, `auto_rewarded`, hit/miss/FA/CR logic and computes `start_time`, `stop_time`, `change_time`, `response_latency`. |
| `trial_masks.contingent_trials` | `code/allensdk/brain_observatory/behavior/trial_masks.py` | CURATION | Explicitly defines contingent trials as GO and CATCH only. |
| `CellSpecimens.from_lims` | `code/allensdk/brain_observatory/behavior/data_objects/cell_specimens/cell_specimens.py` | LOADING | Loads ROI table plus dF/F, corrected fluorescence, demixed/neuropil traces, and event detections for one experiment. |
| `CellSpecimens.__init__` | `code/allensdk/brain_observatory/behavior/data_objects/cell_specimens/cell_specimens.py` | CURATION | Filters to `valid_roi` when `exclude_invalid_rois=True` and reorders traces/events to the filtered ROI list. |
| `DFFTraces.from_data_file` / `DFFFile.load_data` | `code/allensdk/brain_observatory/behavior/data_objects/cell_specimens/traces/dff_traces.py`; `code/allensdk/brain_observatory/behavior/data_files/dff_file.py` | LOADING | dF/F is loaded directly from the stored HDF5 trace file; SDK does not recompute it here. |
| `Events.from_data_file` | `code/allensdk/brain_observatory/behavior/data_objects/cell_specimens/events.py` | PROCESSING | Loads event detections and creates `filtered_events` by causal half-Gaussian smoothing for visualization. |
| `RunningSpeed.from_stimulus_file` / `_get_running_speed_df` | `code/allensdk/brain_observatory/behavior/data_objects/running_speed/running_speed.py` | PROCESSING | Computes running speed from wheel signals on stimulus timestamps with no monitor delay and optional low-pass filtering. |
| `EyeTrackingTable.from_data_file` | `code/allensdk/brain_observatory/behavior/data_objects/eye_tracking/eye_tracking_table.py` | PROCESSING | Aligns eye-tracking frames to timestamps, computes blink flags/outlier filtering, and returns pupil/eye/corneal measurements. |
| `get_stimulus_presentations` / `is_change_event` | `code/allensdk/brain_observatory/behavior/stimulus_processing.py` | PROCESSING | Builds per-flash stimulus table, includes omitted flashes, and marks image identity changes from successive non-omitted images. |

### Notes
- AllenSDK documentation in `code/doc_template/visual_behavior_optical_physiology.rst` states Visual Behavior Ophys data are delivered in NWB and should be accessed through AllenSDK helpers.
- Relevant experiment container for this task is `BehaviorOphysExperiment`, which bundles neural traces/events, behavior, stimuli, running, and eye tracking for a single imaging plane in one session.
- Reference loading path uses sync-derived timestamps for both stimulus/behavior streams and ophys frames. Stimulus timestamps carry monitor-delay compensation; running speed timestamps explicitly require zero monitor delay.
- Trial definitions come from the behavior stimulus file `trial_log`, not from ad hoc segmentation. Trial boundaries are contiguous via `Trials._get_trial_bounds`.
- Within `Trial._get_trial_data`, aborted trials force `go = catch = auto_rewarded = False`; auto-rewarded trials clear hit/miss/false-alarm/correct-reject labels.
- `trial_masks.contingent_trials` confirms the reference definition for kept trial types is GO plus CATCH only, matching the user requirement to exclude aborted and auto-rewarded trials.
- `CellSpecimens` applies ROI curation by dropping rows with `valid_roi == False` when `exclude_invalid_rois=True` (the default in `BehaviorOphysExperiment.from_lims/from_nwb`).
- dF/F is not recomputed in the SDK path explored here. `DFFFile.load_data` reads stored traces from an HDF5 dataset named `data`, and `DFFTraces.from_data_file` wraps them unchanged.
- Event traces are also already produced upstream and loaded from event-detection files; the SDK additionally derives `filtered_events` for visualization by causal smoothing.
- Stimulus presentation tables include omitted flashes and change annotations. For this decoder task, trial-level segmentation should still be anchored to `Trials`, while image identity and image-change outputs can be derived on the ophys timeline from stimulus presentations/trial image fields.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`
  contains per-experiment NWB files named `behavior_ophys_experiment_<ophys_experiment_id>.nwb`.
- `data/visual-behavior-ophys-1.1.0/project_metadata/`
  contains CSV manifests:
  `ophys_experiment_table.csv`, `ophys_session_table.csv`,
  `behavior_session_table.csv`, and `ophys_cells_table.csv`.
- Top-level metadata files:
  `visual-behavior-ophys_project_manifest_v1.1.0.json`,
  `_downloaded_data.json`, `_manifest_last_used.txt`.
- Observed on-disk organization matches AllenSDK Visual Behavior Ophys CloudCache layout.
- Sample NWB file structure (via `h5py`) contains:
  - top-level groups: `acquisition`, `analysis`, `general`, `intervals`, `processing`, `stimulus`, etc.
  - `processing/ophys`: `dff`, `event_detection`, `image_segmentation`, `corrected_fluorescence`, `demixed_trace`, `neuropil_trace`, motion correction.
  - `processing/running`: `speed`, `speed_unfiltered`, `dx`.
  - `acquisition/EyeTracking`: pupil, eye, corneal-reflection ellipse fits plus blink flags.
  - `intervals/trials`: trial table with `aborted`, `auto_rewarded`, `catch`, `go`, `hit`, `miss`, `false_alarm`, `correct_reject`, `initial_image_name`, `change_image_name`, `change_time`, etc.
  - stimulus interval tables such as `*_presentations` for image flashes / natural movie / spontaneous blocks.
- Sample file `behavior_ophys_experiment_1007107386.nwb`:
  - `dff` data shape `(140204, 13)` stored as time x ROI
  - `event_detection` shape `(140204, 13)`
  - running speed length `270240`
  - pupil time series length `135981`
  - trials in NWB table: `503`
  - stimulus presentation rows in active image block: `4806`

Available variables identified from raw files:
- Neural: dF/F, event detections, corrected fluorescence, demixed traces, neuropil traces, ROI metadata including `valid_roi`.
- Behavior: running speed, licks, rewards, trial table/outcomes.
- Eye tracking: pupil area/geometry, eye area/geometry, likely blink mask.
- Stimulus: image identities, stimulus presentation timing, omissions, stimulus block labels.
- Metadata: mouse ID, sex, genotype/Cre line, area/depth, session type, experience level, project code, passive flag.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 133,066 ROI rows in `ophys_cells_table.csv` |
| Neurons / session | 68.73 mean ROI rows per experiment file (median 22; min 1; max 666) |
| Subjects | 107 unique `mouse_id` values in `ophys_experiment_table.csv` |
| Sessions / subject | 18.09 experiment files per mouse on average; 6.57 unique ophys sessions per mouse on average |
| Trials (total) | 412,112 total trials across 703 unique ophys-linked behavior sessions (`behavior_session_table.csv`, deduplicated by `behavior_session_id`) |
| Trials / session | 586.22 mean trials per unique ophys behavior session (min 397; max 1403) |

Additional raw dataset counts:
- Experiment NWB files present locally: 284
- Rows in `ophys_experiment_table.csv`: 1,936 experiments
- Unique ophys sessions in metadata: 703
- Unique behavior sessions with ophys: 703
- Rows in `behavior_session_table.csv`: 4,782 total behavior sessions
- Unique targeted structures: `VISp`, `VISl`, `VISal`, `VISam`
- Project codes present: `VisualBehavior`, `VisualBehaviorTask1B`, `VisualBehaviorMultiscope`, `VisualBehaviorMultiscope4areasx2d`
- Session types observed include active and passive variants (`OPHYS_1`, `OPHYS_3`, `OPHYS_4`, `OPHYS_6` active; `OPHYS_2`, `OPHYS_5` passive)

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | Paper subset: 8,619 excitatory + 470 Sst + 1,239 Vip = 10,328 cells | `methods.txt:179` “Our dataset contains 8,619 excitatory cells … 470 Sst cells … and 1,239 Vip cells …” |
| Neurons / session | Paper subset is tens to hundreds of cells per imaging plane; raw release contains 68.7 mean ROI rows per experiment | No single text value found; paper gives cell totals over 15–21 imaging sessions depending on class (`methods.txt:179`) |
| Subjects | Paper subset: 82 mice for behavior analyses; cell-class neural subsets use 6–9 mice | `methods.txt:174` “376 imaging sessions from 82 mice”; `methods.txt:179` gives 6–9 mice for cell-class subsets |
| Sessions / subject | Paper behavior subset: 376 / 82 = 4.59 imaging sessions per mouse on average | Derived from `methods.txt:174` subset counts |
| Trials (total) | Not explicitly tabulated in paper/whitepaper text | Task structure and session duration described, but no total trial count stated in text |
| Trials / session | Not explicitly tabulated; sessions are ~1 hour and sample file has ~500 trials | `methods.txt:58` “Recording sessions were ~1 hour long” |
| Neural data time bin | Native acquisition: 31 Hz single-plane; 11 Hz per plane multi-plane | `methods.txt:58` “31 Hz for single plane and … 11 Hz for each plane in multi-plane experiments” |
| Behavior data time bin | Native behavior and eye tracking: 30 Hz | `methods.txt:58` “eye tracking (30 Hz), and behavior (30 Hz)” |
| Reward rate | Engagement threshold is 2 rewards/min | `methods.txt:44` “reward rates above and below 2 rewards per minute” |
| Stimulus cadence | 250 ms image + 500 ms gray screen | `methods.txt:176` “250 ms stimulus duration” and “500 ms inter-stimulus duration” |
| Omission rate | 5% omitted image repeats during imaging | `methods.txt:10` / `methods.txt:176` “stimuli were omitted with a 5% probability” |
| Response window | 150–750 ms after change/sham change before lag compensation | `methods.txt:24` / `methods.txt:42` |
| Catch probability | ~36% in stages 1–2; ~30% then effectively ~12.5% in later stage 3+ sessions | `methods.txt:32` |
| Change-time distribution | Truncated exponential 2.25–8.25 s, mean 4.25 s; actual mean ~4.2 s after flash alignment shift | `methods.txt:32` |


### Processing Details
- Task is flashed-image go/no-go change detection. Each session uses 8 images, giving 64 possible image transitions (`methods.txt:4`).
- Trial structure:
  - GO and CATCH trial types are pre-selected before each trial (`methods.txt:32`).
  - Premature licks before the change reset the trial and generate aborted trials/timeouts (`methods.txt:32`).
  - Sessions also contain free-reward / auto-reward-like trials at the start and after long miss streaks (`methods.txt:34`).
- Temporal details:
  - Stimuli are presented for 250 ms with a 500 ms gray inter-stimulus interval (`methods.txt:176`).
  - Image changes and the immediately preceding image are never omitted (`methods.txt:10`, `methods.txt:176`).
  - Response window is 150–750 ms relative to non-display-lag-compensated change time; reaction times are recalculated after display-lag correction of roughly 20–35 ms (`methods.txt:42`, `methods.txt:18`).
  - Neural data are recorded at 31 Hz (single plane) or 11 Hz/plane (multiplane); eye tracking and behavior at 30 Hz (`methods.txt:58`).
- Neural processing:
  - Paper analyses frequently use discrete calcium events rather than raw dF/F (`methods.txt:179`, `methods.txt:208`).
  - Whitepaper event detection uses FastLZero on fluorescence-derived traces; factor 2.0 at 31 Hz and 2.6 at 11 Hz (`whitepaper.pdf` p.42 extracted text).
- ROI/trace processing:
  - Motion border invalidates ROIs touching the border (`methods.txt:97`).
  - ROIs are filtered if duplicate, union, edge/motion-corrupted, apical dendrite, too small/narrow/dim (`methods.txt:109`-`111`; `whitepaper.pdf` p.34).
  - Duplicate/union ROIs are removed before demixing; missed problematic ROIs causing non-positive demixed traces are removed afterward, costing ~1% of ROIs (`methods.txt:118`-`124`; `whitepaper.pdf` p.35-36).
  - Neuropil subtraction is performed before downstream analyses; dF/F and event traces are derived after standardized processing (`methods.txt:127`-`131`).
- Behavioral engagement:
  - AllenSDK rolling metrics exclude aborted trials and define engagement by reward rate > 2 rewards/min (`methods.txt:44`).

### Curation Steps

**Neuron curation rules**:
- Exclude invalid ROIs using the whitepaper/SDK ROI filtering logic: motion-border ROIs, duplicate ROIs, union ROIs, likely dendrites, and ROIs too small/narrow/dim to be confident cell bodies (`methods.txt:109`-`111`).
- Additional post-demixing removal when negative/zero demixed traces reveal missed duplicate/union ROIs (`methods.txt:124`).
- Paper neural analyses use extracted calcium events, not raw fluorescence (`methods.txt:179`, `methods.txt:208`).

**Trial curation rules**:
- Aborted trials are excluded from rolling performance metrics and should be excluded for this decoder task (`methods.txt:44`).
- Auto/free-reward trials are behaviorally present (`methods.txt:34`) and should be excluded for this decoder task.
- GO and CATCH trials are the contingent trial types of interest (`methods.txt:32`).
- Image omissions occur only during imaging, but changes and the immediately preceding image are never omitted (`methods.txt:10`, `methods.txt:176`).

### Decoders Trained
| Decoded variable | Accuracy |
| Image change vs repeat | Not numerically stated in paper text; described as decodable “equally well” across cell classes/strategies (`paper.pdf` p.12 extraction) |
| Hit vs miss | Not numerically stated in paper text; decoder performance reported higher for excitatory and Vip cells in visual vs timing strategy sessions (`paper.pdf` p.12 extraction) |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Dataset scope | SDK/code supports full Visual Behavior Ophys release, including active/passive, familiar/novel, single- and multi-plane | Local metadata contains 1,936 experiments, 703 ophys sessions, 107 mice, of which 1,363 experiments / 495 sessions are active | Neuron paper analyzes a narrower subset for strategy analyses (376 imaging sessions from 82 mice; familiar multiplane subset for neural analyses) | Use the broader active-task release, not the paper’s strategy-specific subset. Rationale: user asked for data “under the Visual Behavior task” with trial outcomes, go/catch trials, running, pupil, and image variables. Passive sessions are not task performance and make outcome labels degenerate. |
| Passive sessions | Trial table exists in NWB, but still shares trial schema | Passive example shows `go`/`catch` counts but all outcomes collapse to `miss`/`correct_reject` and no aborted/auto-rewarded trials | Paper distinguishes passive viewing from active behavior; behavior analyses use active sessions | Exclude passive sessions (`passive == True` / `behavior_type == passive_viewing`) from conversion. |
| Neural signal choice | SDK exposes both `dff_traces` and `events`; paper subset analyses use calcium events | NWB files contain both dF/F and event detection | Paper methods explicitly use detected calcium events for neural analyses | Use raw event magnitude traces from `processing/ophys/event_detection/data` as `neural`. Do not use `filtered_events` (visualization-only) and do not recompute dF/F. |
| Native sampling rates | SDK preserves native per-session sampling (31 Hz single-plane, 11 Hz multiplane) | Raw files reflect multiple rig types | Papers/whitepaper describe mixed acquisition rates and also describe 30 Hz interpolation for event-triggered analyses | Resample all streams to a common 30 Hz grid (`33.333... ms` bins). This preserves ophys-based timing while satisfying the decoder requirement that all trials/sessions share a common bin size. |
| Trial segmentation source | SDK builds trials from behavior stimulus file via `Trials` / `Trial` | NWB already contains processed `intervals/trials` table with AllenSDK-style columns | Methods define GO/CATCH/aborted/auto-rewarded logic and change timing | Use the NWB `trials` table directly as the authoritative processed trial definition, then filter to GO/CATCH and exclude aborted/auto-rewarded. This matches Allen processing while avoiding fragile reimplementation and bypasses local SDK/NWB version issues. |
| Stimulus identity/change source | SDK can derive from `stimulus_presentations` and trial image fields | NWB contains `*_presentations` intervals and trial-level `initial_image_name` / `change_image_name` | Methods specify 8 flashed natural images, omissions, and non-omitted pre-change/change flashes | Use stimulus presentation interval tables to build time-varying image identity and image-change signals on the resampled trial grid; use trial table to sanity-check change times and image names. |
| Environment mismatch | Reference code expects AllenSDK/NWB stack to load NWB directly | Local environment cannot instantiate `NWBFile` for these NWBs due `external_resources`/version mismatch | Not a scientific discrepancy | Read NWB files directly with `h5py` and mirror the AllenSDK/whitepaper semantics from the processed NWB contents rather than relying on broken high-level loading in this environment. |

Final understanding:
- Sessions for conversion will be active `behavior_ophys_experiment` NWB files only.
- Trials will be defined by the processed NWB `trials` table, keeping GO and CATCH only, excluding `aborted` and `auto_rewarded`.
- Neural activity will be event traces, aligned on absolute ophys time and resampled to a common 30 Hz grid within each trial.
- Running speed, pupil diameter, and stimulus variables will be interpolated or sampled onto the same 30 Hz trial grid.
- Brain region will come from experiment metadata `targeted_structure`; each experiment is one imaging plane/region.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/event_detection/data` (NWB, time x ROI) | `neural` | Transpose to ROI x time, then linearly interpolate event magnitudes from native ophys timestamps onto a common 30 Hz trial grid | `BehaviorOphysExperiment.events`; whitepaper event detection; paper uses extracted calcium events | Use raw event magnitudes, not `filtered_events` |
| `processing/ophys/dff/traces/data` | not used for `neural` | None | `BehaviorOphysExperiment.dff_traces` | Retained only for sanity/debug plots if needed; paper analyses favor events |
| `intervals/trials` table (`start_time`, `stop_time`, `go`, `catch`, `aborted`, `auto_rewarded`, `hit`, `miss`, `false_alarm`, `correct_reject`, `change_time`, `initial_image_name`, `change_image_name`) | Trial segmentation and `output[trial_outcome]` | Filter to GO/CATCH with `aborted==False` and `auto_rewarded==False`; outcome encoded as constant categorical trace across each trial | `Trials.from_stimulus_file`, `Trial._get_trial_data`, `trial_masks.contingent_trials` | NWB already stores Allen-processed trial table |
| `intervals/*_presentations` task image block (`image_name`, `omitted`, `is_change`, `start_time`, `stop_time`, `trials_id`, `stimulus_block_name`) | `output[image_identity]` | Build per-bin categorical state on 30 Hz grid: actual `image_name` during image flashes; `gray` during gray/omitted periods | `get_stimulus_presentations`, `is_change_event` | Restrict to `stimulus_block_name == change_detection_behavior` when present |
| Same stimulus-presentation rows | `output[image_change]` | Binary per-bin trace: 1 during change-image flash interval, else 0 | `is_change_event`; trial `change_time` for cross-check | Change and pre-change flashes are never omitted |
| `processing/running/speed` (`data`, `timestamps`) | `output[running_speed_bin]` | Interpolate filtered running speed onto 30 Hz trial grid; discretize globally across included data into 5 equal-frequency bins | `RunningSpeed.from_stimulus_file`; running-processing module | Use filtered running speed in cm/s |
| `acquisition/EyeTracking/pupil_tracking/{width,height,timestamps}` plus blink-filtered fields | `output[pupil_diameter_bin]` | Compute pupil diameter as `2 * max(width, height)` after blink filtering; interpolate onto 30 Hz grid; discretize globally into 5 equal-frequency bins | `process_eye_tracking_data`, `filter_on_blinks`, whitepaper eye-tracking description | Exclude sessions with missing eye-tracking acquisition entirely; interpolate within-session for blink-related NaNs |
| None required by task | `input` | Use empty arrays of shape `(0, n_timepoints)` for every trial | N/A | Decoder task specifies no decoder inputs |
| `mouse_id` from `ophys_experiment_table.csv` | `subjects`, `subject_idx` | Unique string list + per-session index | metadata tables | Sessions correspond to experiment files after filtering |
| `targeted_structure` from `ophys_experiment_table.csv` | `brain_regions`, `brain_region_idx` | Unique region list + constant region index repeated for all neurons in each session | metadata tables | Each experiment file is one imaging plane / one targeted structure |
| Local NWB filename stem `behavior_ophys_experiment_<id>.nwb` | Session order | Sort by `ophys_experiment_id` and keep only locally present active files with required data | metadata + filesystem | Local subset is 284 files total, 202 active; sessions with no eye tracking will be excluded |

### Key Decisions
1. **Use only locally available active experiment NWBs**: The workspace contains 284 experiment files, of which 202 are active and 82 passive. Passive sessions collapse trial-outcome variability and are outside the active Visual Behavior task.
2. **Treat each `ophys_experiment_id` file as one converted session**: This matches the AllenSDK object granularity (`BehaviorOphysExperiment`) and yields a single imaging plane / neuron set / brain region per session.
3. **Segment trials using the processed NWB `trials` table**: This mirrors the Allen reference processing without re-deriving trial logic from lower-level files and avoids the broken local NWB/AllenSDK high-level loader.
4. **Keep only contingent trials**: Include GO and CATCH trials; exclude `aborted` and `auto_rewarded` exactly as required and consistent with reference definitions.
5. **Use event traces as neural activity**: This best matches the paper’s neural analyses and uses the whitepaper-defined event detection pipeline already embedded in the NWB files.
6. **Use a common 30 Hz trial grid**: Native acquisition rates vary across rigs (31 Hz single-plane, 11 Hz multiplane). Resampling all streams to 30 Hz gives one shared bin size while remaining close to behavior/eye-tracking rate and consistent with paper event-triggered interpolation onto 30 Hz timestamps.
7. **Align by absolute ophys time, then cut into trials**: For each trial, create bin centers from trial `start_time` to `stop_time` at 30 Hz and sample/interpolate all streams onto that grid.
8. **Represent all outputs as time-varying traces**: `image_identity`, `image_change`, `running_speed_bin`, and `pupil_diameter_bin` are naturally time-varying; `trial_outcome` will be repeated across bins within a trial as a constant categorical trace to keep one consistent `(n_output, T)` format.
9. **Encode gray/omission periods explicitly**: `image_identity` will include a `gray` category for ISI and omitted-image periods, because the user specifically asks for the image identity during non-gray screen and those periods still occupy trial time.
10. **Discretize running and pupil globally across the included dataset**: Five equal-percentile bins will be computed from all finite samples across all included sessions/trials, not per session, so class semantics are consistent dataset-wide.
11. **Require pupil availability at session level**: Three active local files lack eye-tracking acquisition entirely; these sessions will be excluded. Remaining sessions have modest blink-related missingness and can be filled by time interpolation before discretization.
12. **Restrict stimulus rows to the task block**: Use interval groups / block labels corresponding to `change_detection_behavior` and trial overlap, ignoring natural-movie or spontaneous blocks stored elsewhere in NWB.

### Planned Sanity Checks
- [ ] Session/trial count check: for each converted session, converted trial count equals raw NWB count of `go|catch` trials after excluding `aborted` and `auto_rewarded`
- [ ] Neural spot-check: independently reconstruct one trial from raw `event_detection` + ophys timestamps and verify `np.allclose()` to converted neural matrix for selected neurons/bins
- [ ] Stimulus spot-check: independently reconstruct image identity / image-change traces from raw stimulus-presentation rows for 3 trials and verify `np.allclose()` against converted categorical codes
- [ ] Running spot-check: independently interpolate raw running speed timestamps for a selected trial and verify `np.allclose()` to pre-discretized converted running trace
- [ ] Pupil spot-check: independently compute `2*max(width,height)` from raw eye-tracking arrays, interpolate for a selected trial, and verify `np.allclose()` to pre-discretized converted pupil trace
- [ ] Outcome consistency: converted `trial_outcome` labels match mutually exclusive raw trial columns (`hit`, `miss`, `false_alarm`, `correct_reject`)
- [ ] Region/subject consistency: session metadata in converted file matches `ophys_experiment_table.csv` for `mouse_id` and `targeted_structure`
- [ ] Distribution checks: global percentile bins for running/pupil produce approximately balanced class counts across included finite samples

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `convert_data.py` with the required CLI:
- `python -u convert_data.py <outpicklefile>`
- `--full`
- `--sample`
- `--show-processing`

Implementation notes:
- Directly reads local NWB files with `h5py` because the local AllenSDK/NWB stack cannot instantiate these NWB files in this environment.
- Uses a two-pass conversion:
  1. pass 1 computes global running-speed and pupil-diameter percentile edges and filters out unusable sessions
  2. pass 2 converts trials into session/trial matrices
- Uses local active experiment files only.
- Aligns all streams on a common 30 Hz grid within each trial using absolute time and ophys timestamps as the neural reference stream.
- Saves empty decoder inputs as `(0, T)` arrays.
- Builds optional processing plots for up to two sessions.

Code inefficiencies identified:
- Full conversion may still be I/O-heavy because each NWB event matrix must be read from disk.
- Global binning requires a first pass over sessions, so conversion reads each file twice.

Code speedups added:
- Vectorized linear interpolation for neural event matrices using `searchsorted` + broadcasting rather than per-neuron `np.interp`.
- Session-level streaming design avoids storing continuous raw traces for the whole dataset in memory.
- Uses processed NWB tables directly rather than re-deriving trials/stimuli from lower-level files.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 231 |
| Neurons / session | [89, 142] (mean 115.5) |
| Subjects | 1 (`403491`) |
| Sessions / subject | 2 |
| Trials (total) | 229 |
| Trials / session | [39, 190] |
| Running bin range | [0, 4] |
| Pupil bin range | [0, 4] |
| Image identity values present | 17 categories total (`gray` + 16 images) |
| Image change distribution | [0.974, 0.026] |
| Running bin distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| Pupil bin distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| Trial outcome distribution | [0.595, 0.274, 0.053, 0.078] for `[hit, miss, false_alarm, correct_reject]` |

### Processing Plots Review
- Initial diagnostic plot version was too compressed because it showed full-session running/pupil traces against a single-trial context. This was fixed by plotting relative-to-trial-start windows only.
- Final diagnostic plots (`processing_775614751.png`, `processing_788490510.png`) show:
  - event traces and 30 Hz resampled traces aligned on the same trial window
  - running and pupil streams smoothly aligned to the trial grid
  - image flashes alternating with gray periods at the expected 250 ms / 500 ms cadence
  - change indicator aligned to the change flash
  - percentile bin edges overlaid on running/pupil histograms
- No obvious temporal misalignment was observed in the corrected plots.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Vectorized event interpolation | Avoided per-neuron interpolation loops; sample conversion finished in ~4.5 s for 2 sessions |
| Direct NWB processed-table reads | Avoided expensive/fragile AllenSDK high-level loading in this environment |

| Step | Time / Session | Estimated Total Time |
| Pass 1 bin-stat collection | ~0.36 s / session | ~1.2 min for 202 active local sessions |
| Pass 2 conversion | ~1.86 s / session | ~6.3 min for 202 active local sessions |
| Total conversion | ~2.22 s / session | ~7.5 min for 202 active local sessions |

Validation notes:
- `sample_data.pkl` created successfully (`31M`).
- `verification_sample_out.txt` created successfully.
- `train_decoder.py --verify-only` reported: “Data format is valid, no errors or warnings.”
- Environment note: the requested `python -u ...` command could not be used because only `python3` is installed here; sample conversion was run with `python3 -u ...` instead.
- Per-session running and pupil bin distributions are skewed in the 2-session sample, which is expected because bin edges are computed globally across the sample and not per session.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings:
  - `sklearn` emitted `y_pred contains classes not in y_true` during one balanced-accuracy computation on the small held-out split. This reflects missing classes in the small validation subset rather than a format failure.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| image_identity | 0.2062 | 0.1651 |
| image_change | 0.6531 | 0.6163 |
| running_speed_bin | 0.2370 | 0.2292 |
| pupil_diameter_bin | 0.2564 | 0.2144 |
| trial_outcome | 0.2951 | 0.2633 |

Sample training notes:
- Loss decreased monotonically from `1.638766` at epoch 1 to `1.510723` at epoch 200.
- Test loss was `1.554734`.
- All outputs were above uniform-chance on validation:
  - `image_identity`: chance `1/17 = 0.0588`
  - `image_change`: chance `0.5000`
  - `running_speed_bin`: chance `0.2000`
  - `pupil_diameter_bin`: chance `0.2000`
  - `trial_outcome`: chance `0.2500`
- The two-session sample is small and class coverage is uneven, so the modest margins above chance for running, pupil, and trial outcome are not by themselves conclusive.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: `8.1G`
- `verification_full_out.txt`: created
- `conversion_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | 10,328 in paper neural subset | `valid_roi` cells from included local active + eye-tracking sessions | 29,168 | 29,168 | Yes for local subset; paper differs by subset |
| Mean neurons/session | Not explicitly reported | valid cells / included experiment | 146.57 | 146.57 | Yes |
| Subjects | 82 mice in paper imaging subset | included local active + eye-tracking experiment files | 38 | 38 | Yes for local subset; paper differs by subset |
| Sessions | 376 imaging sessions in paper subset | included local active + eye-tracking experiment files | 199 | 199 | Yes for local subset; paper differs by subset |
| Trials (total) | Not explicitly reported | GO/CATCH and not aborted/auto-rewarded | 51,075 | 51,075 | Yes |
| Trials/session (mean) | Not explicitly reported | same contingent-trial filter as reference logic | 256.66 | 256.66 | Yes |
| Neural time bin | 31 Hz single-plane; 11 Hz multi-plane native acquisition | common resampled grid chosen after reference review | 30 Hz target grid | 30 Hz target grid | Yes |
| Behavior / eye time bin | 30 Hz native | same | 30 Hz native, resampled on 30 Hz trial grid | 30 Hz trial grid | Yes |
| Running bin range | N/A | 5 quantile bins | [0, 4] | [0, 4] | Yes |
| Pupil bin range | N/A | 5 quantile bins | [0, 4] | [0, 4] | Yes |
| Image change distribution | Flash-change task implies sparse positives | derived from stimulus presentations | [0.975, 0.025] | [0.975, 0.025] | Yes |
| Trial outcome distribution | Strategy paper emphasizes hit/miss imbalance; exact global fraction not tabulated | derived from raw trial table outcome flags | [0.307, 0.568, 0.018, 0.107] | [0.307, 0.568, 0.018, 0.107] | Yes |

Full conversion notes:
- `python3 -u convert_data.py converted_data.pkl --full` completed in `395.58 s` and saved `199` sessions, `51,075` trials, and `29,168` neurons.
- Raw-data cross-checks on the local available subset matched the converted file exactly:
  - included sessions: `199`
  - included subjects: `38`
  - kept contingent trials: `51,075`
  - valid ROI neurons: `29,168`
  - trial-outcome distribution: `[0.30703867, 0.56759667, 0.01801273, 0.10735193]`
- `train_decoder.py converted_data.pkl --verify-only` completed successfully and reported valid structure, dimensions, and categorical ranges.
- The verifier did report many warnings of the form `all neural data is zero` for some trials. Quantification on the converted dataset showed `2,474 / 51,075 = 4.84%` of kept trials had all-zero event traces.
- Raw-NWB spot check on warned session index `3` (experiment `792815735`), warned kept trial index `2`, showed the underlying `event_detection` segment itself was exactly zero across all `27` neurons and `388` native frames. This indicates the warnings are due to sparse event detections in the source data rather than a conversion bug.
- Spot checks from `verification_full_out.txt`:
  - output ranges were correct: `image_identity [0,16]`, `image_change [0,1]`, `running_speed_bin [0,4]`, `pupil_diameter_bin [0,4]`, `trial_outcome [0,3]`
  - running and pupil bins were globally balanced at `~0.2` per class
  - brain regions in the converted local subset were `VISp` and `VISl`, matching metadata for the available files

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` contained one warning class only: `all neural data is zero`. There were `2,474` such trials (`4.84%` of kept trials). No structural or dimensional errors were reported.
2. **Neural/input/output raw-data sanity checks**: Independently reconstructed three converted trials directly from raw NWB files and compared with `np.allclose()`.
3. **Reference code comparison**: Matched each major processing stage in `convert_data.py` to the corresponding AllenSDK/data-object logic explored in Step 1.
4. **Key statistics comparison**: Recomputed sessions, subjects, contingent trials, valid neurons, and trial-outcome fractions directly from raw local NWB files and compared to the converted pickle.
5. **Edge-case review**: Checked trial-boundary binning, missing eye tracking, sparse-event trials, and sessions with repeated `behavior_session_id` / `ophys_session_id` across multiple experiment planes.

### Issues Found and Resolved
- **Verifier warnings for zero neural trials**: Not fixed by filtering. Direct raw-NWB inspection showed that warned trials can be exactly zero in the source `event_detection` matrix itself. Example: session index `3`, experiment `792815735`, kept trial index `2`, `27` neurons, `388` native frames, raw event sum `0.0`. Resolution: retain these trials because they are valid source-data trials rather than a conversion artifact; document warning cause.
- **Potential trial-count mismatch risk**: Recomputed kept trials directly from raw tables using `(go | catch) & ~aborted & ~auto_rewarded` and obtained `51,075`, exactly matching the converted dataset. Resolution: no change required.
- **Potential neuron-count mismatch risk**: Recomputed valid cell counts directly from raw NWB `cell_specimen_table` and obtained `29,168`, exactly matching the converted dataset. Resolution: no change required.
- **Potential temporal-alignment bug risk**: Independent trial reconstructions matched converted arrays exactly for neural, image, change, running, pupil, and outcome traces. Resolution: no change required.

Detailed review notes:
- **Check 1: Output log verification**
  - `train_decoder.py --verify-only` reported valid structure and categorical ranges.
  - Warning class present: all-zero neural trials only.
  - No negative values, shape inconsistencies, or output-range errors were reported.
- **Check 2: Constructed sanity checks from original raw files**
  - Trial `(session 0, trial 0, experiment 775614751)`:
    - `neural_allclose == True`, max abs diff `2.98e-08`
    - `image_identity`, `image_change`, `running_speed_bin`, `pupil_diameter_bin`, `trial_outcome` all matched exactly
  - Trial `(session 3, trial 2, experiment 792815735)`:
    - `neural_allclose == True`, max abs diff `0.0`
    - all output traces matched exactly
    - this was a warned all-zero neural trial, confirming the warning arises from source data
  - Trial `(session 150, trial 0, experiment 960410028)`:
    - `neural_allclose == True`, max abs diff `1.19e-07`
    - all output traces matched exactly
  - These checks were performed by loading raw NWB arrays directly with `h5py`, reconstructing the 30 Hz trial bins, interpolating neural/running/pupil traces independently, rebuilding stimulus traces from raw interval tables, and comparing to the converted arrays with `np.allclose()`.
- **Check 3: Reference code comparison**
  - Data loading:
    - reference: `BehaviorOphysExperiment` / `BehaviorSession` load processed NWB contents
    - converter: direct `h5py` reads from processed NWB groups because the local AllenSDK/NWB stack cannot instantiate these files
    - comparison result: same processed sources are used; only the loader mechanism differs due environment incompatibility
  - Neuron filtering:
    - reference: `CellSpecimens.__init__` keeps `valid_roi == True`
    - converter: included-session raw NWB files already had all listed cells valid (`29,168` total valid of `29,168` total listed), so event matrices matched converted neuron counts exactly
  - Trial filtering:
    - reference: contingent trials via `trial_masks.contingent_trials`, excluding aborted and auto-rewarded logic from `Trial`
    - converter: `(go | catch) & ~aborted & ~auto_rewarded`
    - comparison result: exact match
  - Temporal alignment and binning:
    - reference texts describe mixed native ophys rates and 30 Hz behavior/eye streams, with event-triggered analyses interpolated onto a common time base
    - converter: absolute-time alignment on a common 30 Hz trial grid using ophys timestamps as the reference time axis
    - comparison result: consistent with the paper/whitepaper constraints and decoder requirement for a shared bin size
  - Input construction:
    - reference does not define decoder inputs for this task
    - converter uses empty `(0, T)` arrays
  - Output construction:
    - stimulus traces derived from processed stimulus-presentation tables
    - running/pupil discretized globally into 5 quantile bins
    - trial outcomes derived from mutually exclusive trial outcome flags
    - comparison result: consistent with task specification and processed raw variables
- **Check 4: Key statistics comparison**
  - Included sessions: raw `199`, converted `199`
  - Included subjects: raw `38`, converted `38`
  - Included contingent trials: raw `51,075`, converted `51,075`
  - Included neurons: raw `29,168`, converted `29,168`
  - Trial-outcome fractions: raw `[0.30703867, 0.56759667, 0.01801273, 0.10735193]`, converted identical
  - Paper subset counts (`376` sessions, `82` mice, `10,328` neural-subset cells) remain different because the workspace contains only a smaller local subset of active NWB files. This discrepancy was already resolved in Step 4 as a scope difference, not a processing mismatch.
- **Check 5: Edge cases**
  - Trial binning uses centers strictly within `[start_time, stop_time)`, avoiding off-by-one inclusion past trial end.
  - Sessions missing `EyeTracking` are excluded up front because pupil output is required.
  - All-zero event trials are retained because they are present in the source data and still have valid behavioral/stimulus labels.
  - Multiple experiment files can share the same `behavior_session_id` or `ophys_session_id`; the conversion intentionally treats each `ophys_experiment_id` plane as a separate session because that is the AllenSDK experiment granularity and each file has its own neuron set.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| image_identity | 0.1934 | 0.1903 | 17-way output; well above chance `0.0588` |
| image_change | 0.5955 | 0.5890 | Binary output; above chance `0.5000` |
| running_speed_bin | 0.2457 | 0.2440 | 5-way output; above chance `0.2000` |
| pupil_diameter_bin | 0.2748 | 0.2714 | 5-way output; above chance `0.2000` |
| trial_outcome | 0.2940 | 0.2684 | 4-way output; above chance `0.2500`; false alarms are rare |

Full-training notes:
- Command run: `python3 -u train_decoder.py converted_data.pkl --plot-samples 2>&1 | tee train_decoder_full_out.txt`
- Training completed on `cuda` without needing the `--cpu` fallback.
- Optimization progressed smoothly from loss `1.632803` at epoch `1` to `1.565087` at epoch `200`.
- Final test loss was `1.579317`.
- Plot outputs created by the trainer: `predictions.png` and `sample_trials.png`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
|----------|-------------------|-------------------------|
| image_identity | 0.1903 validation balanced accuracy | No directly comparable identity-decoding number reported in text; expected to be above chance if stimulus alignment is correct |
| image_change | 0.5890 validation balanced accuracy | Paper reports above-chance change-vs-repeat decoding from neural activity in the first 400 ms after image presentation, but no exact text-extracted percentage value was available |
| running_speed_bin | 0.2440 validation balanced accuracy | No directly comparable paper decoder reported |
| pupil_diameter_bin | 0.2714 validation balanced accuracy | No directly comparable paper decoder reported |
| trial_outcome | 0.2684 validation balanced accuracy | Paper reports hit-vs-miss decoding qualitatively, but not a text-extracted number comparable to this 4-class full-trial outcome decoder |

Analysis:
- **Chance comparison**
  - `image_identity`: `0.1903 / 0.0588 = 3.24x` chance
  - `image_change`: `0.5890 / 0.5000 = 1.18x` chance
  - `running_speed_bin`: `0.2440 / 0.2000 = 1.22x` chance
  - `pupil_diameter_bin`: `0.2714 / 0.2000 = 1.36x` chance
  - `trial_outcome`: `0.2684 / 0.2500 = 1.07x` chance
- Outputs below `1.5x` chance were investigated rather than accepted at face value:
  - raw-trial spot checks already verified output correctness for image, change, running, pupil, and outcome traces on three specific trials
  - temporal alignment was checked both numerically with `np.allclose()` and visually in `processing_775614751.png` / `processing_788490510.png`
  - output variation was adequate:
    - running and pupil bins are essentially perfectly balanced by construction
    - image change has sparse positives (`2.5%` of bins) but balanced accuracy accounts for class imbalance
    - trial outcome is imbalanced, especially false alarms (`1.8%` of trials), which likely limits achievable balanced accuracy
  - no conversion bug was found in these investigations
- **Accuracy comparison to papers**
  - The paper contains two relevant neural decoders:
    - change vs. repeat
    - hit vs. miss
  - Extracted text:
    - “Decoding was performed on neural activity in the first 400 ms after each stimulus presentation.”
    - “Cross-validated random forest classifier performance at decoding image changes and repeats (% correct).”
    - “Cross-validated random forest classifier performance at decoding hits and misses (% correct).”
  - However, the PDF text available in this environment did not expose the figure values numerically, so an exact percentage table could not be reproduced from text alone.
  - These paper decoders are not directly matched to this benchmark anyway:
    - they operate on selected cell classes and strategy-defined subsets
    - they decode only the first `400 ms` after each image
    - they use `% correct`, not balanced accuracy
    - they decode binary `hit vs miss`, not the 4-class `trial_outcome` used here
  - Qualitatively, our `image_change` and `trial_outcome` results are consistent with the paper’s claim that these variables are decodable from visual-cortex activity.
- **Train vs validation gap**
  - `image_identity`: `1.02x`
  - `image_change`: `1.01x`
  - `running_speed_bin`: `1.01x`
  - `pupil_diameter_bin`: `1.01x`
  - `trial_outcome`: `1.10x`
  - No output showed a train/validation ratio above `1.5x`, so there was no evidence of severe overfitting or leakage.

### Issues Found and Resolved
- **Potential low-accuracy concern for running/pupil/trial outcome**: Investigated via raw-data spot checks, temporal-alignment review, and class-distribution review; no conversion issue identified.
- **Paper comparison limitation**: The paper text exposes the presence and setup of neural decoders but not the exact figure values in extractable text, so only qualitative comparison was possible from local materials.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Cleanup notes:
- Added `README.md` with dataset summary, file format, loading instructions, and validation snapshot.
- Added `cache/README_CACHE.md` documenting retained investigation artifacts.
- Added `cache/raw_sanity_checks.py` so the raw-vs-converted `np.allclose()` spot checks used in Step 10 can be rerun.
