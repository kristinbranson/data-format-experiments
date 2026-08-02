# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory Visual Behavior 2P dataset, local project files in `/app` with AllenSDK reference code in `code/`
- **Date started**: 2026-04-08
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

Environment verification:
- `python3`: `3.13.12`
- `numpy`: `2.3.5`
- `torch`: `2.6.0+cu124`
- `CONVERSION_NOTES.md` existence check passed via `ls -la CONVERSION_NOTES.md`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `BehaviorProjectCache.get_behavior_ophys_experiment` | `code/allensdk/brain_observatory/behavior/behavior_project_cache/behavior_project_cache.py` | LOADING | Main cache entry point that returns a `BehaviorOphysExperiment` for one `ophys_experiment_id`. |
| `BehaviorOphysExperiment.from_lims` / `from_nwb` | `code/allensdk/brain_observatory/behavior/behavior_ophys_experiment.py` | LOADING | Assembles one experiment from behavior session data, ophys timestamps, cell specimens, metadata, projections, and motion correction. |
| `BehaviorSession._read_data_from_stimulus_file` | `code/allensdk/brain_observatory/behavior/behavior_session.py` | PROCESSING | Builds session stimulus timestamps, licks, rewards, trials, stimuli, and task parameters from the stimulus/sync files. |
| `StimulusTimestamps.from_sync_file` / `subtract_monitor_delay` | `code/allensdk/brain_observatory/behavior/data_objects/timestamps/stimulus_timestamps/stimulus_timestamps.py` | PROCESSING | Reads stimulus-frame timestamps from sync and applies/removes monitor delay. |
| `OphysTimestamps.from_sync_file` / `validate` | `code/allensdk/brain_observatory/behavior/data_objects/timestamps/ophys_timestamps.py` | PROCESSING | Loads ophys frame timestamps and truncates extra sync frames when needed to match trace length. |
| `Trials.from_stimulus_file` / `_get_trial_bounds` | `code/allensdk/brain_observatory/behavior/data_objects/trials/trials.py` | PROCESSING | Defines trials from the behavior stimulus `trial_log`, using consecutive `trial_start` frames to create contiguous trial intervals. |
| `Trial._get_trial_data` / `_get_trial_timing` / `add_change_time` | `code/allensdk/brain_observatory/behavior/data_objects/trials/trial.py` | PROCESSING | Determines `go`, `catch`, `aborted`, `auto_rewarded`, hit/miss/false-alarm/correct-reject, plus `start_time`, `stop_time`, `change_frame`, `change_time`, and image names. |
| `Presentations.from_stimulus_file` | `code/allensdk/brain_observatory/behavior/data_objects/stimuli/presentations.py` | PROCESSING | Creates the per-stimulus presentation table with `image_name`, `start_time`, `is_change`, `omitted`, `flashes_since_change`, and `trials_id`. |
| `CellSpecimens.from_nwb` / `from_lims` | `code/allensdk/brain_observatory/behavior/data_objects/cell_specimens/cell_specimens.py` | LOADING | Loads cell ROI table and associated traces/events, validates trace lengths against ophys timestamps, and joins data by ROI. |
| `CellSpecimens.__init__` with `exclude_invalid_rois` | `code/allensdk/brain_observatory/behavior/data_objects/cell_specimens/cell_specimens.py` | CURATION | Filters the cell table to `valid_roi == True`, then filters/reorders all traces and events to the remaining ROIs. |
| `DFFTraces.from_nwb` / `from_data_file` | `code/allensdk/brain_observatory/behavior/data_objects/cell_specimens/traces/dff_traces.py` | LOADING | Loads already-computed dF/F traces; no dF/F recomputation is required by the SDK. |
| `Events.from_nwb` / `from_data_file` | `code/allensdk/brain_observatory/behavior/data_objects/cell_specimens/events.py` | PROCESSING | Loads event-detection outputs and computes `filtered_events` with a causal half-Gaussian filter. |
| `RunningSpeed.from_stimulus_file` | `code/allensdk/brain_observatory/behavior/data_objects/running_speed/running_speed.py` | PROCESSING | Computes running speed aligned to stimulus timestamps with zero monitor delay; includes polarity correction if mean speed is implausibly negative. |
| `EyeTrackingTable.from_nwb` / `from_data_file` | `code/allensdk/brain_observatory/behavior/data_objects/eye_tracking/eye_tracking_table.py` | PROCESSING | Loads pupil/eye ellipse data, recomputes likely blinks, and filters blink frames. |

### Notes
- The AllenSDK reference object for this dataset is `BehaviorOphysExperiment` (the older `BehaviorOphysSession` is only a deprecated alias).
- One experiment corresponds to one imaging plane in one session. For multiplane sessions, `OphysTimestampsMultiplane.from_sync_file()` subsamples interleaved frame times by plane group before validation.
- Trial segmentation is not inferred from image changes directly. It is defined from the behavior `trial_log` using each trial’s `trial_start` frame and the next trial’s start frame as the current trial end boundary.
- Trial labels come from SDK logic in `Trial._get_trial_data()`:
  - `aborted` if an `abort` event occurred.
  - Non-aborted trials are `catch` if `trial_params["catch"]` is true.
  - `auto_rewarded` comes from `trial_params["auto_reward"]`.
  - `go` is `not catch and not auto_rewarded`.
  - `correct_reject = catch and not false_alarm`.
  - Auto-rewarded trials are explicitly prevented from being counted as hit/miss/false_alarm/correct_reject.
- `change_time` is generated from `change_frame` using stimulus timestamps that include monitor delay; licks and rewards are derived from timestamps with monitor delay removed.
- Session-level stimulus timestamps returned by `BehaviorSession._read_data_from_stimulus_file()` are stored with monitor delay removed.
- Running speed is built on stimulus-timebase timestamps with `monitor_delay=0.0`, not on ophys timestamps.
- Eye tracking is aligned to synchronized eye-camera frame times; likely blinks are recomputed and blink frames are filtered.
- Neural data in the SDK are already processed:
  - dF/F traces are loaded directly from `DFFTraces`; no extra dF/F computation is needed.
  - Events are loaded from event-detection files and optionally smoothed for visualization as `filtered_events`.
- Cell curation in the SDK is ROI-quality based:
  - invalid ROIs are removed when `exclude_invalid_rois=True` (default in `BehaviorOphysExperiment.from_nwb/from_lims`);
  - all trace tables and events are then filtered/reordered to match the surviving `cell_roi_id`s.
- I did not find electrophysiology unit-quality filtering in the relevant Visual Behavior ophys path because this task is 2-photon calcium imaging, not ephys.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Root layout inside `data/`:
  - `visual-behavior-ophys_project_manifest_v1.1.0.json`: Allen project manifest
  - `visual-behavior-ophys-1.1.0/project_metadata/*.csv`: project-level metadata tables
  - `visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb`: per-experiment NWB files
  - `_downloaded_data.json`, `_manifest_last_used.txt`: cache bookkeeping
- File formats:
  - NWB/HDF5 for raw experiment payloads
  - CSV for experiment/session/cell metadata tables
  - JSON for manifest metadata
- Important distinction from the local data:
  - metadata CSVs describe the full Allen release (`703` ophys sessions, `1936` ophys experiments, `133066` cell rows, `4782` behavior sessions)
  - the locally downloaded NWB payload available for conversion is a subset (`284` NWB experiment files)
- Representative NWB hierarchy for one experiment:
  - `/general/metadata`: session and experiment metadata (`ophys_session_id`, `ophys_experiment_id`, `project_code`, `session_type`, etc.)
  - `/general/subject`: subject ID and genotype/sex metadata
  - `/processing/ophys`: `dff`, `corrected_fluorescence`, `demixed_trace`, `neuropil_trace`, `event_detection`, `image_segmentation`, motion correction, images
  - `/processing/running`: `speed`, `speed_unfiltered`, `dx`
  - `/processing/stimulus/timestamps`: stimulus timestamps
  - `/acquisition/EyeTracking/*`: pupil/eye/corneal reflection time series when present
  - `/intervals/trials`: trial table
  - `/intervals/*_presentations`: stimulus presentation tables
- Variables confirmed present in raw NWB files:
  - Trial table columns: `aborted`, `auto_rewarded`, `catch`, `change_frame`, `change_image_name`, `change_time`, `correct_reject`, `false_alarm`, `go`, `hit`, `initial_image_name`, `is_change`, `lick_times`, `miss`, `response_latency`, `response_time`, `reward_time`, `reward_volume`, `start_time`, `stop_time`, `trial_length`
  - Stimulus presentation columns: `image_name`, `is_change`, `omitted`, `start_time`, `stop_time`, `flashes_since_change`, `trials_id`, `active`, plus image and block metadata
  - Neural arrays: dF/F traces in `/processing/ophys/dff/traces/data` with shape `(n_timepoints, n_rois)` and ROI metadata under `/processing/ophys/image_segmentation/cell_specimen_table`
  - Running speed and eye tracking are on their own timestamp bases and will need interpolation/reindexing to ophys timestamps during conversion
  - Brain areas present in the local subset: `VISp`, `VISl`
  - Session types present in the local subset: `OPHYS_1_images_A`, `OPHYS_2_images_A_passive`, `OPHYS_3_images_A`, `OPHYS_4_images_B`, `OPHYS_5_images_B_passive`, `OPHYS_6_images_B`

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 42,147 local ROI rows across available NWB experiment files; 133,066 rows in full metadata table |
| Neurons / session | 148.405 per experiment locally (mean); experiment files are the unit of raw neural recording payload |
| Subjects | 38 locally available in NWB files; 107 in full release metadata |
| Sessions / subject | 6.5 mean locally (range 4-11) |
| Trials (total) | 171,887 across locally available NWB experiment files |
| Trials / session | 600.126 mean per unique local ophys session; 605.236 mean per experiment file |

Additional local subset observations:
- Local NWB experiment files: `284`
- Unique local ophys sessions: `247`
- Unique local containers: `44`
- Experiments per local ophys session: range `1-7`, mean `1.15`
- Total size of `data/`: `247G`
- Ophys frames per experiment: range `48,284-149,508`
- Running-speed samples per experiment: range `269,540-287,886`
- Pupil samples per experiment with eye tracking present: range `104,039-272,837`
- Eye tracking missing in `3` local NWB files

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 34,619 cortical cells in whitepaper v1.0 dataset scope | Whitepaper p.3 paraphrase: dataset includes 82 mice, 3021 behavior sessions, 551 imaging sessions, and 34,619 cortical cells. | 
| Neurons / session | Not stated explicitly in text; implied by 34,619 / 551 ≈ 62.8 in whitepaper scope | Derived from whitepaper p.3 counts; not explicitly reported as a mean. |
| Subjects | 82 mice in whitepaper and paper analysis scope | Whitepaper p.3 paraphrase: dataset includes neural and behavioral measurements from 82 mice. Paper p.4 paraphrase: behavior from 376 imaging sessions from 82 mice. |
| Sessions / subject | Not stated explicitly | Paper and whitepaper report total mice and session counts, not a mean per mouse. |
| Trials (total) | Not explicitly stated | No direct total-trial count found in whitepaper, paper, or methods excerpt. |
| Trials / session | Not explicitly stated | Session duration and trial timing are given, but mean trials/session is not directly reported. |
| Neural data time bin | Native acquisition rate: 31 Hz single-plane; 11 Hz per plane multi-plane | Whitepaper/methods p.25: two-photon movies are 31 Hz single-plane and 11 Hz per plane multi-plane. |
| Behavior data time bin | Native acquisition rate: behavior 30 Hz; eye tracking 30 Hz | Whitepaper/methods p.25: eye tracking and behavior are recorded at 30 Hz. |
| Reward rate | Engagement threshold 2 rewards/minute | Whitepaper/methods p.19 paraphrase: SDK engaged vs disengaged corresponds to reward rates above vs below 2 rewards/minute. | 
| Omission fraction | 5% of non-change image repeats during imaging | Whitepaper p.5 / methods paraphrase: omissions occur with 5% probability; change and pre-change stimuli are never omitted. | 
| Response window | 150-750 ms after non-display-lag-compensated change/sham-change | Methods excerpt: response window is 0.150 to 0.750 s for hit/false-alarm classification. |
| Change-time distribution | Truncated exponential 2.25-8.25 s; mean actual change time about 4.2 s | Methods excerpt p.18 paraphrase: scheduled change times 2.25-8.25 s, shifted by one 750 ms cycle to a mean around 4.2 s. |
| Catch probability | ~30% initially in stage 3+, later ~12.5% with equal transition sampling | Methods excerpt p.18 paraphrase: stage 3+ catch probability started near 30% and later effective catch probability was ~12.5%. |


### Processing Details
- Task structure from whitepaper/methods:
  - go/no-go visual change detection with 8 images and 64 possible transitions
  - stimuli shown for 250 ms with 500 ms gray inter-stimulus interval
  - imaging sessions include 5 min gray before task, 5 min gray after task, then repeated natural movie presentation
- Temporal alignment:
  - all clocks were synchronized on a single NI PCI-6612 board sampled at 100 kHz
  - imaging, visual stimulation, behavior, and eye-tracking streams are therefore aligned by shared synchronization signals
  - hit/false-alarm classification uses the non-display-lag-compensated image display time, while reaction times are recalculated after display-lag correction
- Imaging/native sampling:
  - single-plane 2P: 31 Hz
  - multi-plane 2P: 11 Hz per plane
  - eye tracking: 30 Hz
  - behavior/running: 30 Hz
- Neural signal processing described in whitepaper:
  - raw fluorescence undergoes neuropil correction
  - dF/F is computed by subtracting a 600 s median-filter baseline and normalizing by baseline or estimated noise when baseline is very small
  - dF/F is then detrended with a constrained 3.33 s median-filter trend estimate
- Paper-specific analysis choice:
  - the paper’s main figures operate on discrete calcium events regressed from raw fluorescence traces, not directly on dF/F
  - the paper further restricts analyses to familiar-stimulus sessions for its strategy study
- Running speed:
  - computed from wheel encoder voltage, with unwrapping, wrap detection, and derivative-based conversion to linear speed
  - whitepaper explicitly points to AllenSDK running-processing code for the implementation
- Pupil:
  - whitepaper states pupil size is derived from ellipse fits; major axis is treated as diameter and area is computed from that diameter

### Curation Steps

**Neuron curation rules**:
- Whitepaper describes experiment/session QC including temporal sync, residual motion, z-drift, task performance, and interictal-event checks before data release.
- A specific session-level QC threshold is stated: experiments with z-drift above 10 μm were excluded.
- The paper uses extracted discrete calcium events for analysis, implying an additional event-extraction stage beyond fluorescence traces.
- The SDK code path identified in Step 1 also filters invalid ROIs (`valid_roi == True`) at load time; this is consistent with the whitepaper’s QC emphasis, though the exact `valid_roi` classifier rule is described in code/data rather than the methods excerpt.

**Trial curation rules**:
- Trial structure consists of GO and CATCH trial types, which combine with behavior to yield HIT, MISS, FALSE ALARM, and CORRECT REJECTION outcomes.
- Trials with premature licking before the scheduled change are reset and treated as aborted.
- Aborted trials are excluded from rolling hit/false-alarm/d-prime metrics in the SDK documentation and methods text.
- Imaging-session omissions occur only on non-change image repeats; change and immediately pre-change stimuli are never omitted.
- The paper’s analysis subset uses familiar-stimulus sessions only; this is a paper-specific restriction, not a dataset-wide rule for the conversion task.

### Decoders Trained
| Decoded variable | Accuracy |
| Behavioral strategy model (paper logistic model on licking bouts) | Average AUC 0.83 |
| Image changes vs repeats from neural activity (paper random forest) | No exact numeric accuracy reported in main text; described qualitatively as decoded equally well across strategy groups |
| Hits vs misses from neural activity (paper random forest) | No exact numeric accuracy reported in main text; described qualitatively as higher in visual-strategy sessions for excitatory and Vip cells |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Dataset version / size | AllenSDK code path is generic and supports released Visual Behavior ophys NWB files; doc text in repo references larger release counts in later docs | Local manifest is `visual-behavior-ophys` v`1.1.0`; metadata CSVs list 703 ophys sessions / 1936 experiments / 133066 cells, while local downloaded NWBs are only 284 experiments | Whitepaper PDF is V1.0-era and reports 82 mice / 551 imaging sessions / 34619 cortical cells | Treat size/count discrepancies as release-version and local-download-scope differences, not processing mismatches. Use local files as authoritative for actual conversion scope. |
| Analysis subset vs full conversion scope | SDK exposes all experiments and all trial/session metadata | Local data include both active and passive session types, familiar (`images_A`) and novel (`images_B`) sessions | Paper strategy study restricts to familiar stimuli and uses 376 imaging sessions from 82 mice | Do not inherit the paper’s familiar-only restriction for the dataset conversion unless required by the decoder task. The paper subset is informative, not the conversion scope. |
| Neural representation | SDK loads both dF/F traces and event-detection outputs | NWB files contain both `/processing/ophys/dff` and `/processing/ophys/event_detection` | Whitepaper describes dF/F processing; paper states analyses were performed on discrete calcium events | Use the available processed neural signal in a way that matches the paper/code path. Tentative resolution for later mapping: prefer event-detection outputs for neural activity because the analysis paper explicitly uses them, while preserving SDK ROI filtering and timestamps. |
| Trial definitions | SDK trial table defines `go`, `catch`, `aborted`, `auto_rewarded`, outcomes, `change_time`, and image names from the stimulus log | Raw NWBs contain the same trial columns under `/intervals/trials` | Methods text describes GO/CATCH trial types, aborted trials from premature licking, and response-window logic | Trial logic is consistent across code, data, and text. Use SDK-style trial definitions directly. |
| Temporal alignment | SDK keeps separate stimulus, running, eye, and ophys timestamp streams; all are sync-derived | Local NWBs store separate time bases for dF/F, running, pupil, and stimulus presentations | Whitepaper states all data streams were synchronized on one board at 100 kHz | Align converted outputs to ophys timestamps by resampling/interpolating from their native synchronized time bases. |
| Neuron/ROI curation | SDK filters to `valid_roi` by default | In local NWBs all ROI rows inspected so far are marked valid (`valid_roi` sum equals ROI count), but the field exists explicitly | Whitepaper emphasizes QC, residual motion, temporal sync, z-drift, task performance; paper uses processed events | Keep SDK `valid_roi` filtering logic even if many local files already appear fully valid. |
| Omission logic | SDK presentations table has `omitted` and change flags | Local presentations tables include `omitted`, `is_change`, and `flashes_since_change` | Whitepaper/methods state 5% omission on non-change repeats only; change and pre-change stimuli are never omitted | Omission handling is consistent. Include omission-aware image/change outputs, but trial inclusion still follows go/catch trial table rules. |
| Aborted-trial handling | SDK rolling performance excludes aborted trials; trial table labels them explicitly | Local trials table contains `aborted` and `auto_rewarded` flags | Methods text explicitly says aborted trials are those with premature licking before change | Exclude `aborted` and `auto_rewarded` trials from converted trial set, as required by the user task. |

### Final Understanding
- Authoritative loading/processing logic should come from the AllenSDK object model and the actual NWB fields present in `data/`.
- The whitepaper and paper remain important for:
  - experimental timing and task definitions,
  - omission and response-window rules,
  - QC/curation principles,
  - deciding between dF/F and event-based neural representations.
- Dataset-size mismatches across sources are explained by different scopes:
  - whitepaper PDF: earlier V1.0 dataset scope,
  - paper: a further analysis subset,
  - local data: v1.1.0 metadata tables plus a downloaded NWB subset.
- No discrepancy was found in the core trial logic or in the existence of the required raw variables (neural traces/events, stimulus presentations, running speed, pupil, outcomes).

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `/processing/ophys/event_detection` events aligned to ophys frames | `neural` | Use valid-ROI-filtered event traces; rebin from native ophys timestamps into one common bin size across all sessions | `BehaviorOphysExperiment.from_nwb`, `CellSpecimens.from_nwb`, `Events.from_nwb` | Prefer events over dF/F because the analysis paper explicitly uses discrete calcium events. Use raw events, not visualization-only `filtered_events`. |
| No decoder inputs requested | `input` | Store empty arrays of shape `(0, n_timepoints)` for every trial | N/A | User explicitly specified no decoder inputs. |
| Stimulus presentation `image_name` + presentation timing + omission state | `output[0]` (`image_identity`) | Project stimulus presentations onto trial bins; assign image category during image flashes and `gray` during ISI/omissions/no-image periods | `Presentations.from_nwb`, stimulus presentations table | Time-varying categorical output. Global category set will include all local image names plus `gray`. |
| Stimulus presentation `is_change` + presentation timing | `output[1]` (`image_change`) | Binary series that is 1 during the changed-image presentation immediately after a true image-identity change, else 0 | `Presentations.from_nwb`, trial `change_time` for sanity checks | Time-varying binary output. |
| Running speed timeseries | `output[2]` (`running_speed_bin`) | Interpolate running speed to rebinned trial time axis, then discretize into 5 global percentile bins | `RunningSpeed.from_nwb` | Use global bin edges computed over all included samples in included sessions/trials. |
| Eye-tracking pupil signal | `output[3]` (`pupil_diameter_bin`) | Use processed pupil area after blink filtering, convert to equivalent diameter `2*sqrt(area/pi)`, interpolate to rebinned trial axis, discretize into 5 global percentile bins | `EyeTrackingTable.from_nwb` | Requires eye-tracking availability. Sessions with missing eye tracking will be excluded. |
| Trial outcome flags `hit`, `miss`, `false_alarm`, `correct_reject` | `output[4]` (`trial_outcome`) | Single categorical value per trial, repeated across all time bins in that trial to keep output arrays uniformly time-varying | `Trials.from_nwb`, `Trial._get_trial_data` | Aborted and auto-rewarded trials excluded before conversion. |
| `/general/subject/subject_id` or metadata `mouse_id` | `subjects`, `subject_idx` | String subject identifiers with session-level index mapping | NWB subject metadata | Session order follows converted session order. |
| Imaging plane location / targeted structure | `brain_regions`, `brain_region_idx` | One region index repeated for every neuron in a session | Imaging plane metadata / `targeted_structure` | In local subset, brain regions are `VISp` and `VISl`. |

### Key Decisions
1. **Neural signal = event-detection output, not dF/F**: The AllenSDK and NWBs provide both; the paper explicitly states that analyses were performed on discrete calcium events. This is the closest match to the paper while still using the SDK-defined loading and ROI filtering.
2. **Use SDK-valid trials only**: Keep GO and CATCH trials, exclude `aborted` and `auto_rewarded` exactly as instructed and consistent with the trial table semantics.
3. **Keep passive sessions if they have valid GO/CATCH trials**: Passive sessions still contain well-defined trial tables and outcomes (`miss`/`correct_reject` dominant), so they remain valid for the requested decoder outputs.
4. **Exclude sessions with missing eye tracking**: Pupil diameter is a required output; local scan found exactly 3 NWB files without eye-tracking data, so those sessions will be dropped rather than fabricating pupil values.
5. **Common time base via uniform rebinned trial bins**: Native ophys sampling differs between single-plane (31 Hz) and multi-plane (11 Hz), but the target format requires one bin size across all sessions. I will rebin all trials to a common bin width based on ophys-time alignment.
6. **Tentative common bin size = 100 ms**: This is coarse enough to avoid pathological upsampling of 11 Hz recordings, still resolves 250 ms stimulus flashes, and remains compatible with 30 Hz behavior/eye streams. This choice will be validated during sample conversion.
7. **Alignment event = trial start**: The trial itself is the natural unit requested by the user. Metadata will therefore use `trial start` as the alignment event with `off_start = 0.0` and variable trial lengths (`off_end = None`).
8. **Image identity will include a `gray` class**: Trials contain gray ISI and omission periods; using an explicit `gray`/blank class keeps the categorical time series defined at every bin.
9. **Image-change target will mark the changed-image presentation, not only a single instant**: Marking the post-change image flash interval is more robust after binning and is consistent with the task structure.
10. **Running and pupil bin edges will be global, not per-session**: One consistent categorical definition across all sessions is required for decoder outputs and `output_values`.
11. **Trial outcome will be repeated across time bins**: Although static per-trial, repeating it across the trial keeps every `output` array in `(n_output, n_timepoints)` form.
12. **No decoder inputs**: `input_names` will be empty and every `input` trial entry will be an empty 2D array with the same time dimension as the corresponding trial.

### Planned Sanity Checks
- [ ] Raw-vs-converted neural spot check: for a chosen experiment/trial/bin, verify converted neural values equal the sum of raw event amplitudes from ophys timestamps falling into that bin.
- [ ] Raw-vs-converted image identity spot check: for chosen trials, verify bins overlapping raw stimulus-presentation intervals carry the correct image label and gray periods remain gray.
- [ ] Raw-vs-converted image change spot check: verify the converted binary change series turns on only for the changed-image flash and matches trial `change_time` / presentation `is_change`.
- [ ] Raw-vs-converted running spot check: compare rebinned running-speed values against raw `running/speed` sampled at the same absolute times.
- [ ] Raw-vs-converted pupil spot check: compare converted pupil-diameter bins to raw eye-tracking-derived diameter after interpolation at selected timestamps.
- [ ] Trial-count sanity check: per session, valid converted trial count must equal raw trial count minus aborted minus auto_rewarded.
- [ ] Outcome sanity check: per session, converted trial-outcome counts must match raw `hit`/`miss`/`false_alarm`/`correct_reject` counts after filtering.
- [ ] Session-inclusion sanity check: exactly the 3 sessions with missing eye tracking should be excluded for pupil-output completeness, unless another issue is discovered.
- [ ] Output-balance sanity check: global running and pupil bins should each contain roughly 20% of included samples by construction.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `convert_data.py` with the required CLI:
- `python -u convert_data.py <outpicklefile>`
- `--full`
- `--sample`
- `--show-processing`

Current implementation notes:
- Uses direct HDF5 reads from local NWB files rather than `pynwb` because the installed NWB stack is incompatible with these files in this environment.
- Mirrors SDK semantics already serialized into NWB:
  - trial inclusion/exclusion from `/intervals/trials`
  - stimulus timing from `/intervals/*_presentations`
  - running from `/processing/running/speed`
  - pupil from `/acquisition/EyeTracking/*`
  - neural activity from `/processing/ophys/event_detection`
- Uses a two-pass workflow:
  1. lightweight preview pass to identify eligible sessions, collect global image categories, and compute global running/pupil percentile edges
  2. conversion pass to build neural/input/output trial arrays and optional processing plots
- Uses a common 100 ms trial-relative time base across all sessions to satisfy the shared-bin-size requirement while keeping alignment anchored to ophys time.
- Excludes sessions with missing eye tracking because pupil diameter is a required decoder target.
- Produces empty input arrays with zero rows because the decoder task specifies no inputs.

Code inefficiencies identified:
- Full preview scans all NWB files once and the conversion pass opens them again; this is intentional to avoid storing large neural matrices before global percentile/bin definitions are known.
- Session conversion currently rebins event data with per-bin slice sums instead of using cumulative sums; this reduces peak memory at the cost of some extra CPU.

Code speedups added:
- Preview pass avoids loading neural event matrices.
- Conversion uses direct dataset reads and processes one session at a time to cap peak memory.
- Processing plots are limited to at most 2 sessions.
- Sample-mode preview now short-circuits once 2 eligible sessions are found, which keeps Step 7 runtime proportional to the sample size.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 1,270 |
| Neurons / session | [666, 604] |
| Subjects | 1 |
| Sessions / subject | 2 for subject `448900` |
| Trials (total) | 603 |
| Trials / session | [324, 279] |
| Decoder inputs | none (`d_input = 0`) |
| Image identity distribution | [gray 0.670, im061 0.039, im062 0.041, im063 0.041, im065 0.042, im066 0.042, im069 0.043, im077 0.041, im085 0.041] |
| Image change distribution | [no_change 0.974, change 0.026] |
| Running-speed bin distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| Pupil-diameter bin distribution | [0.204, 0.202, 0.198, 0.199, 0.198] |
| Trial-outcome distribution | [hit 0.114, miss 0.766, false_alarm 0.003, correct_reject 0.117] |

### Processing Plots Review
- Generated `processing_877018118.png` and `processing_873968820.png`.
- Plots were produced without runtime errors and are sized plausibly for manual inspection.
- No obvious conversion failure was exposed by the sample run.
- The higher-cell-count sample removed the earlier all-zero-trial warnings; `verification_sample_out.txt` now reports no errors and no warnings.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Sample-mode preview short-circuit after 2 eligible sessions | Reduced sample preview from scanning 284 files to scanning 2 files |
| Preview avoids neural loading | Keeps first pass lightweight |
| Preview now pools raw running/pupil samples within valid trials instead of interpolating every trial during preview | Reduced preview CPU cost substantially |
| Session-wise conversion | Caps peak memory and keeps per-session timing visible |

| Step | Time / Session | Estimated Total Time |
| | | |
| Preview pass (optimized sample-mode observed) | ~0.25 s / file for first 2 files | Naive full preview estimate well under 2 min |
| Conversion pass (sample observed on high-neuron sessions) | ~5.95 s / session | Conservative full-conversion upper bound ~28.2 min for 284 similarly large sessions |
| Total practical full estimate | N/A | Likely far below the conservative upper bound because the sample sessions are much larger than the dataset average (635 neurons/sample session vs ~148 neurons mean locally) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None from `verification_sample_out.txt`; `train_decoder_sample_out.txt` also completed successfully. A downstream sklearn warning noted `y_pred contains classes not in y_true` during validation, which appears to be due to missing classes in the held-out sample split rather than a data-format issue.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| image_identity | 0.5189 | 0.4921 |
| image_change | 0.7002 | 0.6555 |
| running_speed_bin | 0.3276 | 0.2857 |
| pupil_diameter_bin | 0.3718 | 0.3509 |
| trial_outcome | 0.4792 | 0.3201 |

Additional sample-training notes:
- Loss decreased monotonically from `1.511646` at epoch 1 to `1.234237` at epoch 200.
- All validation balanced accuracies were above uniform-chance levels:
  - image identity: chance `0.1111`
  - image change: chance `0.5000`
  - running speed bin: chance `0.2000`
  - pupil diameter bin: chance `0.2000`
  - trial outcome: chance `0.2500`

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: `4.3G`
- `verification_full_out.txt`: created
- `conversion_full_out.txt`: created

Full conversion run summary:
- Preview pass: `65.0 s`
- Conversion pass: `686.0 s`
- Total elapsed: `757.7 s`
- Eligible local experiment files: `281 / 284`
- Exclusions: `3` experiments with missing eye tracking, `0` experiments with fewer than two valid go/catch trials

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | `34,619` cortical cells in whitepaper V1.0 scope | `41,871` valid ROIs after SDK-style `valid_roi` filtering on eligible local experiments | `41,871` counted directly from local NWBs after eye-tracking + trial eligibility filter | `41,871` | Matches local code/data; paper differs because of release/scope |
| Mean neurons/session | `62.8` implied by `34,619 / 551` whitepaper imaging sessions | `149.0` | `149.0` | `149.0` | Matches local code/data; paper differs because of release/scope |
| Subjects | `82` mice in whitepaper/paper scope | `38` local mice represented in eligible experiments | `38` | `38` | Matches local code/data; paper differs because of scope |
| Sessions | `551` imaging sessions in whitepaper scope; paper subset `376` familiar-stimulus imaging sessions | `281` eligible local experiment files | `281` eligible local experiment files | `281` | Matches local code/data; paper differs because of scope |
| Trials (total) | Not explicitly reported | `84,313` go/catch, non-aborted, non-auto-rewarded trials in eligible local experiments | `84,313` counted directly from `/intervals/trials` in local NWBs | `84,313` | Yes |
| Trials/session (mean) | Not explicitly reported | `300.05` | `300.05` | `300.05` | Yes |
| Input dimension | Not applicable | `0` | `0` by design for this decoder task | `0` | Yes |
| image_change distribution | Papers give task timing but not exact binwise fraction | `[0.974, 0.026]` from local eligible trials after ophys-bin construction | `[0.974, 0.026]` | `[0.9737, 0.0263]` | Yes |
| running_speed_bin distribution | Not reported in papers | Approximately quintiles by construction | Derived from global local running-speed percentiles | `[0.1994, 0.2013, 0.1992, 0.1999, 0.2002]` | Yes |
| pupil_diameter_bin distribution | Not reported in papers | Approximately quintiles by construction | Derived from global local pupil-diameter percentiles after blink masking | `[0.2029, 0.2058, 0.1929, 0.1866, 0.2119]` | Yes |
| trial_outcome distribution | Outcome classes defined in methods; exact pooled fractions not reported | `[0.182, 0.692, 0.010, 0.115]` | `[0.182, 0.692, 0.010, 0.115]` | `[0.1823, 0.6920, 0.0104, 0.1153]` | Yes |

Verification summary:
- `train_decoder.py --verify-only` reported no errors.
- It reported `3947` warnings of the form `all neural data is zero` (`4.68%` of trials).
- These warnings are consistent with sparse event-detection matrices and are concentrated in low-neuron and/or low-event trials rather than indicating a schema mismatch.
- This warning class is carried into Step 10 for explicit review against raw NWB event data.
- After the Step 10 fix below, `image_identity` verification range is now `[0.0, 16.0]`, matching `gray + 16` task images with no stray omitted-stimulus class.

Spot checks completed:
- Direct raw-NWB recount confirmed `281` eligible experiments, `84,313` eligible trials, and `41,871` valid ROIs before any converted-pickle inspection.
- Converted pickle dimensions matched the direct raw-NWB recount exactly.
- Brain-region totals matched expectations from the local subset: `VISp 41,633` neurons, `VISl 238` neurons.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Re-read `verification_full_out.txt` after the full rerun. Result: no errors, `3947` warnings for all-zero neural trials. Root-cause review found one genuine issue before the rerun: `output_values[0]` incorrectly included an unused `omitted` image label even though omitted flashes were encoded as `gray` in the time series. Fixed by excluding `omitted` from `unique_nonempty_images()` and re-running `sample_data.pkl`, `converted_data.pkl`, `verification_sample_out.txt`, `verification_full_out.txt`, and `train_decoder_sample_out.txt`.
2. **Constructed raw-data sanity checks with `np.allclose()`**: Added `cache/step10_raw_checks.py`, which reconstructs trial bins directly from the source NWB files without importing `convert_data.py`. Result: all checks passed.
   - `representative_nonzero`: experiment `877018118`, session `153`, converted trial `0`, raw trial `19`; neural/input/output all matched exactly and the raw neural matrix was not all zero.
   - `warned_zero`: experiment `1007107386`, session `0`, converted trial `108`, raw trial `145`; neural/input/output all matched exactly and the raw neural matrix was all zero.
   - `warned_zero`: experiment `836258957`, session `103`, converted trial `0`, raw trial `1`; neural/input/output all matched exactly and the raw neural matrix was all zero.
   - `warned_zero`: experiment `960410042`, session `268`, converted trial `0`, raw trial `70`; neural/input/output all matched exactly and the raw neural matrix was all zero.
   - `min_length_edge`: experiment `856938751`, session `138`, converted trial `42`, raw trial `147`; neural/input/output all matched exactly on a minimum-length trial.
3. **Reference code comparison**:

| Processing Step | `convert_data.py` | Reference code / methods | Comparison |
|-----------------|-------------------|---------------------------|------------|
| Data loading | `read_trials()`, `read_task_presentations()`, `load_neural_events()`, `convert_session()` | AllenSDK `BehaviorOphysExperiment`, `BehaviorSession`, `Presentations`, `Trials` object model identified in Step 1 | Same underlying NWB tables/fields are used; I read them directly with `h5py` because `pynwb` failed in this environment. |
| Neuron / trial filtering | `load_neural_events()` keeps `valid_roi`; `build_trial_specs()` excludes `aborted` and `auto_rewarded` trials and uses go/catch outcome fields | AllenSDK default ROI loading uses `valid_roi`; trial semantics are defined by the SDK `Trials` table and methods text | Logic matches the reference semantics serialized in the NWB files. |
| Temporal alignment | `build_bin_centers()`, event sums on ophys timestamps, running/pupil interpolation to ophys-aligned bin centers | Whitepaper says streams are sync-aligned on a shared board; SDK exposes synchronized ophys, stimulus, running, and eye tables | Alignment matches the reference time bases; the only added step is the decoder-required common 100 ms rebinning. |
| Binning | Per-bin event sums and percentile discretization for running/pupil | Native SDK data remain at native sample rates; the papers do not prescribe a common decoder bin size | Difference is intentional and required by the target format, not a mismatch in source processing. |
| Input construction | Empty `(0, T)` arrays | Not specified by AllenSDK; raw variables exist but the user-specified decoder task requests no decoder inputs | Intentional task-driven difference. |
| Output construction | `make_image_series()`, `discretize_with_edges()`, per-trial outcome series | SDK tables provide `image_name`, `omitted`, `is_change`, running speed, eye tracking, and trial outcomes | Same source variables and semantics; Step 10 fix ensured omitted flashes map only to `gray`, not to a separate class label. |

4. **Key statistics comparison**: Recounted eligible raw data directly from NWBs after the fix and compared with the converted pickle. Result: `281` eligible experiments, `84,313` eligible trials, `41,871` valid ROIs, `38` subjects, brain regions `VISp` and `VISl`. Converted data matched raw-NWB recount exactly. The corrected `image_identity` vocabulary is now `17` classes total (`gray + 16` task images), consistent with the paper/whitepaper task description of two eight-image sets.
5. **Edge-case review**: Checked missing-eye-tracking handling (`3` excluded sessions), verified no experiments had fewer than two valid go/catch trials after filtering, validated a minimum-length trial with raw reconstruction, and confirmed that all-zero neural warnings arise from genuine zero-event trials in the raw event-detection matrices rather than off-by-one errors at trial boundaries.

### Issues Found and Resolved
- `output_values[0]` mistakenly contained `omitted` as an unused image class: fixed by excluding `omitted` from image vocabulary construction and re-running affected sample/full artifacts.
- `all neural data is zero` warnings remained after the rerun: investigated and not fixed because direct raw-NWB reconstruction showed that these trials are truly all-zero under the paper-consistent event-detection representation. Removing them or replacing events with dF/F would diverge from the reference processing and task definition.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| image_identity | 0.2580 | 0.2504 | Chance `0.0588` for `17` classes |
| image_change | 0.6188 | 0.6040 | Chance `0.5000`; class is sparse (`2.63%` positive bins) |
| running_speed_bin | 0.2734 | 0.2704 | Chance `0.2000`; bins are near-uniform by construction |
| pupil_diameter_bin | 0.3203 | 0.3172 | Chance `0.2000`; strongest continuous-behavior decoder |
| trial_outcome | 0.3307 | 0.2890 | Chance `0.2500`; four static trial-outcome classes |

Additional full-training notes:
- Training ran successfully on `cuda`; CPU fallback was not needed.
- Loss decreased from `1.635368` at epoch 1 to `1.495033` at epoch 200.
- Final test loss was `1.523579`.
- Full split sizes: `67,343` training trials and `16,970` testing trials.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| image_identity | Validation balanced accuracy `0.2504` vs chance `0.0588` (`4.26x` chance) | No direct paper accuracy reported for this exact task; above-chance decoding is expected for visual identity. |
| image_change | Validation balanced accuracy `0.6040` vs chance `0.5000` (`1.21x` chance) | Paper reports image changes vs repeats are decodable from neural activity but does not give a numeric accuracy in the main text. |
| running_speed_bin | Validation balanced accuracy `0.2704` vs chance `0.2000` (`1.35x` chance) | No direct paper accuracy reported for this discretized behavioral output. |
| pupil_diameter_bin | Validation balanced accuracy `0.3172` vs chance `0.2000` (`1.59x` chance) | No direct paper accuracy reported for this discretized behavioral output. |
| trial_outcome | Validation balanced accuracy `0.2890` vs chance `0.2500` (`1.16x` chance) | Paper reports hit/miss decoding qualitatively but not as a four-class balanced-accuracy number. |

Analysis of lower-ratio outputs:
- `image_change`, `running_speed_bin`, and `trial_outcome` are below the heuristic `1.5x chance` threshold, so I explicitly investigated them rather than dismissing them.
- Raw-value verification had already passed in Step 10 on one representative nonzero trial, three warned all-zero trials, and one minimum-length trial; output matrices reconstructed directly from NWB matched the converted pickle exactly with `np.allclose()`.
- Temporal alignment was rechecked using the processing plots generated during sample conversion; stimulus identity/change, running interpolation, pupil interpolation, and neural event traces are synchronized on the same ophys-aligned trial window with no visible lag mismatch.
- Output variation is adequate:
  - `image_change` positive bins are sparse (`2.63%`), so balanced accuracy modestly above `0.5` is still meaningful.
  - `running_speed_bin` is close to uniform by design, which makes it a relatively hard five-way decoder from visual cortical activity alone.
  - `trial_outcome` is imbalanced (`hit 18.2%`, `miss 69.2%`, `false_alarm 1.0%`, `correct_reject 11.5%`) and is a four-class static label, making it harder than the paper’s qualitative two-class hit/miss analyses.
- Train/validation gaps do not indicate major overfitting or leakage:
  - `image_identity`: `1.03x`
  - `image_change`: `1.03x`
  - `running_speed_bin`: `1.01x`
  - `pupil_diameter_bin`: `1.01x`
  - `trial_outcome`: `1.14x`
- I searched the paper/methods for directly comparable decoder metrics. The only explicit numeric decoding-style metric I found was the paper’s behavioral-strategy logistic model (`AUC 0.83`), which is not the same prediction target as any output in this conversion. The paper’s neural decoders for image change and hit/miss are described qualitatively without full-dataset balanced-accuracy numbers, so a strict numeric comparison is not possible.
- Given the raw-NWB sanity checks, the lack of train/validation pathology, and the corrected class vocabulary, I did not identify another conversion bug to fix at this stage.

### Issues Found and Resolved
- Potential concern: some outputs were less than `1.5x` chance. Resolution: investigated raw labels, temporal alignment, class balance, and train/validation gaps; no conversion error was found, and the observed accuracies are plausible for these targets under the chosen full-trial event-based representation.
- Missing direct paper baselines for most decoder heads. Resolution: documented the absence of directly comparable numeric results instead of inventing a false comparison.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Cleanup notes:
- Created `README.md` summarizing the dataset, conversion choices, file layout, and reproduction commands.
- Created `cache/README_CACHE.md` documenting cached helper artifacts.
- Moved the Step 10 raw-check helper script and its output log into `cache/`.
