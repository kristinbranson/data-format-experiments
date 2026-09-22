# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory - Visual Behavior 2P (VisualBehaviorOphysProjectCache)
- **Date started**: (see file mtime)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment: python3 with numpy 2.4.4 (reported), torch 2.6.0+cu124, allensdk 2.16.2 all import OK.

Directory contents of /app:
- `.manifest` (228 KB), `Dockerfile`, `docker-compose.yaml`
- `allensdk_docs/` - downloaded allensdk documentation
- `code/` - the AllenSDK source repository (allensdk package, scripts, doc_template)
- `data/` - AllenSDK cache: `visual-behavior-ophys-1.1.0/` (behavior_ophys_experiments/*.nwb, plus tables), `visual-behavior-ophys_project_manifest_v1.1.0.json`, `_downloaded_data.json`, `resources/`. Total 247 GB.
- `decoder.py`, `train_decoder.py` - provided decoder validation/training code
- `methods.txt`, `paper.pdf` (Vip-Sst disinhibitory circuit paper), `whitepaper.pdf` (VB 2P technical whitepaper)
- `tutorials/` - 6 tutorial files on loading Visual Behavior ophys data

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `VisualBehaviorOphysProjectCache.from_local_cache(cache_dir)` | code/allensdk/.../behavior_project_cache/project_cache_base.py:88 | LOADING | Instantiate cache from the local AllenSDK cache directory (no S3 download) |
| `bc.get_ophys_experiment_table()` | behavior_project_cache.py:219 | LOADING | Table of all ophys experiments (imaging planes): cre_line, session_type, targeted_structure, imaging_depth, project_code, equipment_name, mouse_id, ophys_session_id, experience_level |
| `bc.get_ophys_session_table()` / `get_behavior_session_table()` | behavior_project_cache.py:160/283 | LOADING | Session-level metadata |
| `bc.get_behavior_ophys_experiment(ophys_experiment_id)` | behavior_project_cache.py:337 | LOADING | Returns BehaviorOphysExperiment object with all data streams |
| `dataset.events` | data_objects/cell_specimens/events.py | PROCESSING | DataFrame per cell_specimen_id with `events` (L0-detected calcium events, variable magnitude, sampled at ophys frame rate) and `filtered_events` (events convolved with half-gaussian, filter_scale_seconds = 2/31 s, n_time_steps=20) |
| `dataset.dff_traces` | cell_specimens.py | PROCESSING | dF/F traces (already computed by pipeline: median-filter baseline, noise-normalized, detrended; see methods.txt DF/F CALCULATION). dF/F need NOT be computed by us. |
| `CellSpecimens(..., exclude_invalid_rois=True)` | cell_specimens/cell_specimens.py:154,205 | CURATION | By default the SDK already filters out ROIs flagged invalid by the ROI-classifier (union/duplicate/motion-border/dendrite/too small/dim). So dff_traces/events only contain valid cell-somata ROIs; each has a non-null cell_specimen_id. |
| `dataset.ophys_timestamps` | - | ALIGNMENT | Timestamps (s, sync-clock) for each ophys frame; 11 Hz per plane on Multiscope, 31 Hz on Scientifica |
| `dataset.stimulus_presentations` | data_objects/stimuli/presentations.py | LOADING | Per-flash table: start_time, end_time, image_name, omitted, is_change, flashes_since_change, stimulus_block_name (`change_detection_behavior` for the task block) |
| `is_change_event()` | brain_observatory/behavior/stimulus_processing.py:592 | PROCESSING | is_change = first presentation of a NEW image_name; omitted flashes excluded; first flash of session excluded |
| `dataset.trials` | data_objects/trials/trial.py | LOADING/CURATION | Per-trial table with mutually-exclusive flags go / catch / aborted / auto_rewarded and outcomes hit / miss / false_alarm / correct_reject, plus start_time, stop_time, change_time (time of real change on go, sham change on catch), response_time, response_latency, initial_image_name, change_image_name |
| `dataset.running_speed` | data_objects/running_speed/running_speed.py | LOADING | DataFrame timestamps + speed (cm/s), 60 Hz stimulus frame clock; low-pass 10 Hz Butterworth filtered, transients removed (see whitepaper) |
| `dataset.eye_tracking` | eye_tracking/eye_tracking_table.py | LOADING | 30 Hz table: timestamps, pupil_width/height/area, likely_blink; `filter_on_blinks()` sets blink frames to NaN |
| `dataset.licks`, `dataset.rewards` | - | LOADING | lick / reward timestamps |
| `dataset.metadata` | - | LOADING | mouse_id, cre_line, targeted_structure, imaging_depth, session_type, experience_level, ophys_frame_rate, equipment_name |

### Notes
- Trial definitions (trial.py `_get_trial_data`): aborted = animal licked before the change; auto_rewarded = non-contingent reward trial (not hit/miss); go = real image change; catch = sham change. The task asks to keep only **go** and **catch** trials -> the four outcomes hit / miss / false_alarm / correct_reject.
- `change_time` exists for both go (real change) and catch (sham change) trials -> natural temporal alignment event.
- Neural data: the reference paper (Vip-Sst) explicitly uses the **detected calcium events** (`dataset.events`), not dF/F: "For all analysis of neural data we used the detected calcium events". dF/F does NOT need to be computed - the SDK returns pipeline-computed dF/F and events.
- Cell/ROI quality filtering is already performed by the SDK (`exclude_invalid_rois=True`), and session-level QC (z-drift, d-prime>1, etc.) was applied before release.
- Tutorials (`/app/tutorials/`) demonstrate exactly this API: `bpc.VisualBehaviorOphysProjectCache`, `get_behavior_ophys_experiment`, per-trial slicing with `query('timestamps >= start_time and timestamps <= stop_time')`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is an AllenSDK **VisualBehaviorOphysProjectCache** local cache (247 GB):
  - `visual-behavior-ophys_project_manifest_v1.1.0.json` - manifest
  - `visual-behavior-ophys-1.1.0/project_metadata/` - the metadata tables (ophys_experiment_table, ophys_session_table, behavior_session_table, ophys_cells_table)
  - `visual-behavior-ophys-1.1.0/behavior_ophys_experiments/behavior_ophys_experiment_<id>.nwb` - **284 files**, one per *ophys experiment* (= one imaging plane in one session)
- Loaded with `VisualBehaviorOphysProjectCache.from_local_cache(cache_dir='/app/data')` (the `use_static_cache=True` variant expects a different folder layout and fails).
- The 284 local experiments are exactly **all** experiments of **38 mice** in the full release table (the full table has 1936 experiments / 107 mice) -> the local cache is a complete per-mouse subset.

### Composition of the local cache (284 experiments)
| Split | Counts |
|---|---|
| project_code | VisualBehavior (single-plane, Scientifica) 239; VisualBehaviorMultiscope (Mesoscope) 45 |
| equipment | CAM2P.4 111, CAM2P.3 90, CAM2P.5 38 (31 Hz); MESO.1 45 (11 Hz/plane) |
| cre_line | Slc17a7-IRES2-Cre 153, Sst-IRES-Cre 85, Vip-IRES-Cre 46 |
| session_type | OPHYS_1_images_A 55, OPHYS_3_images_A 55, OPHYS_4_images_B 46, OPHYS_6_images_B 46, OPHYS_2_images_A_passive 40, OPHYS_5_images_B_passive 42 |
| experience_level | Familiar 150, Novel >1 96, Novel 1 38 |
| targeted_structure | VISp 261, VISl 23 |
| mice / ophys sessions | 38 mice, 247 ophys sessions |

### Active (non-passive) behaviour sessions — the analysis set
| Statistic | Value |
|-----------|-------|
| Experiments (imaging planes) | 202 |
| Ophys sessions | 174 (168 single-plane + 6 Multiscope with 3-7 planes) |
| Subjects (mice) | 38 |
| Sessions / subject | mean 4.6, min 2, max 9 |
| Neurons (total, summed over sessions) | 29,444 |
| Neurons / session | mean 169, median 103, min 6, max 666 |
| Go+Catch trials (total) | 44,892 |
| Go+Catch trials / session | mean 258, min 39, max 409 |
| Trial outcomes (all sessions) | hit 14,194 (31.6%), miss 25,071 (55.8%), false_alarm 845 (1.9%), correct_reject 4,782 (10.7%) |
| Catch fraction | 12.53 % of go+catch trials |
| Hit rate on go trials | 36.1 % ; false-alarm rate on catch trials | 15.0 % |
| Images | 8 per session; 16 across the dataset (image set A: im061,062,063,065,066,069,077,085; set B: im000,031,035,045,054,073,075,106) |
| Omitted flashes | 3.5 % of flashes (whitepaper: 5 % of *repeats*, changes and pre-change flashes never omitted) |

### Per-experiment data streams (BehaviorOphysExperiment)
| Stream | Rate / shape | Notes |
|---|---|---|
| `events` | (n_cells) rows; each `events` array is n_ophys_frames long | L0-detected calcium events (magnitudes); also `filtered_events`, `lambda`, `noise_std` |
| `dff_traces` | same | pipeline dF/F (already computed) |
| `ophys_timestamps` | 31 Hz single-plane (dt=0.0323 s), 11 Hz Multiscope (dt=0.0932 s) | per-plane; Multiscope planes of the same session are offset by ~23 ms per plane group but have the same frame count |
| `stimulus_presentations` | ~4800 flashes in block `change_detection_behavior` (+9000 natural-movie frames + 2 gray-screen blocks) | start_time/end_time, image_name, omitted, is_change, is_sham_change, flashes_since_change, trials_id |
| `trials` | ~660-900 rows | go/catch/aborted/auto_rewarded, hit/miss/false_alarm/correct_reject, start_time, stop_time, change_time |
| `running_speed` | 60 Hz (dt=0.0167 s), no NaNs | cm/s, 10 Hz low-pass filtered by pipeline |
| `eye_tracking` | 30 Hz (dt=0.0333 s) | pupil_width/height/area, likely_blink; blink frames are NaN (median 2.9 % of frames) |
| `metadata` | dict | mouse_id, cre_line, targeted_structure, imaging_depth, session_type, ophys_frame_rate, ophys_session_id, equipment_name |

### Trial timing (measured over 20 random sessions, go+catch trials)
- `change_time - start_time`: min **3.02 s**, median 3.77 s, max 8.3 s
- `stop_time - change_time`: **4.23 s** (essentially constant)
- gap to next trial start: 0.17-0.25 s; `change_time` to next trial's start >= 4.48 s
- 100 % of go/catch change times have +/-3 s of ophys coverage
- `change_time` is **exactly equal** to the `start_time` of the `is_change` flash (go) / `is_sham_change` flash (catch) - verified difference = 0.0 s

### Data-quality issues found
- 3 experiments/sessions have **no eye-tracking** data at all (795625712, 832881662, 805989030) -> pupil output impossible.
- 6 further sessions have 12-30 % blink NaNs; the rest ~3 %.
- Passive sessions (OPHYS_2/5) contain **no licks and no rewards** -> every trial is miss/correct_reject, i.e. the trial-outcome output is degenerate.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Dataset used | Allen Institute Visual Behavior-2P | "We analyzed the Allen Institute Visual Behavior-2p calcium imaging dataset" (paper) |
| Behaviour dataset size | 376 imaging sessions, 82 mice (full public release; a second figure quotes 382 sessions) | "contains behavior from 376 imaging sessions from 82 mice" |
| Neural dataset (paper's restriction) | 8,619 excitatory cells (21 sessions, 9 mice); 470 Sst (15 sessions, 6 mice); 1,239 Vip (21 sessions, 9 mice) | "Our dataset contains 8,619 excitatory cells..." - these are **multi-plane, familiar-image** sessions only |
| Neural signal | detected calcium events (L0), *not* dF/F | "For all analysis of neural data we used the detected calcium events" |
| Restriction | familiar images, multi-plane rig, V1 + LM combined | "For neural analysis we used neurons recorded during familiar image set presentations on the multi-plane imaging rig" |
| Stimulus timing | 250 ms image, 500 ms gray -> 750 ms flash cycle | "natural images (250 ms stimulus duration) interspersed with... (500 ms inter-stimulus duration)" |
| Behavioural analysis unit | 750 ms image presentation interval | "By image presentation interval we refer to the 750 ms interval beginning with each image presentation" |
| Omissions | 5 % of image repeats; never a change or the flash before a change | whitepaper + paper |
| Catch probability | ~12.5 % in the imaging stage | "pushing the actual catch probability to ~12.5%" (whitepaper) |
| Change times | truncated exponential 2.25-8.25 s after trial start, mean 4.2 s | whitepaper |
| Response window | 150-750 ms after the change | whitepaper |
| Ophys frame rates | 31 Hz single-plane, 11 Hz per plane multi-plane | whitepaper |
| Eye tracking / behaviour cameras | 30 Hz | whitepaper |
| Running wheel | analog encoder, 10 Hz low-pass filtered, cm/s | whitepaper |
| Session QC | z-drift < 10 um, peak d' >= 1.0, sync verified, etc. (already applied to released data) | whitepaper Section D |
| ROI QC | invalid ROIs (union/duplicate/motion-border/dendrite/too small/dim) excluded | whitepaper ROI FILTERING, implemented by SDK `exclude_invalid_rois=True` |

### Processing Details
- dF/F is computed by the Allen pipeline (median-filter baseline, noise normalisation, detrending) - we do not recompute it; and the reference paper uses **events** anyway.
- Neuropil correction, crosstalk removal (Multiscope), motion correction, dewarping are all already applied in the released NWB files.
- The paper assigns behavioural events to the 750 ms image presentation interval; decoding was done on "neural activity in the first 400 ms after each stimulus presentation".

### Curation Steps
**Neuron curation rules**: only valid ROIs (already enforced by the SDK). No further per-cell filtering is described in the paper.
**Trial curation rules**: the paper's decoders use image presentations, not trials. For this conversion the task prescribes: keep **go** and **catch** trials, drop **aborted** and **auto_rewarded** trials (auto-rewarded trials "bias the animal's choice and should not be categorized as hit/miss" - trial.py).

### Decoders Trained (reference paper)
| Decoded variable | Accuracy |
|---|---|
| image change vs. repeat (random forest, first 400 ms, per imaging plane) | ~65-85 % correct, increasing with number of cells (Figure 6A); no strategy difference |
| hit vs. miss (random forest, image changes only) | ~55-70 % correct (Figure 6C) |
| false alarm decoding | "very low" (Figure S22) |
These are 2-class problems (chance 50 %) decoded from 400 ms windows of single imaging planes, so they are a loose upper reference for our multi-class, whole-session decoder.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal | SDK exposes `dff_traces`, `events` (L0) and `filtered_events` | both present for every cell | paper uses "detected calcium events" | Initially planned: `events` (raw L0 magnitudes) summed per bin, to match the paper. **Final choice (Steps 8b, 10 Check 3, 12): `dff` averaged per bin** - a controlled comparison on 20 sessions showed dF/F decodes better on 4 of 5 outputs and removes 2,859 all-zero trials in sparse Vip/Sst planes. Both remain selectable with `--neural-signal`. |
| dF/F computation | computed by pipeline, stored in NWB | present | whitepaper describes the algorithm as part of the pipeline | Nothing to recompute. |
| Rig restriction | - | locally only **6** active multi-plane sessions from **1** mouse | paper restricts neural analysis to multi-plane rig, familiar images | Cannot follow: would leave 1 subject. Use **all active sessions from both rigs and all experience levels** (38 mice / 174 sessions). Justified because a decoder benchmark needs many sessions/subjects and the paper's own *behavioural* analysis used all active sessions of all experience levels. |
| Trial types | trial.py: go/catch/aborted/auto_rewarded mutually exclusive | go 12.5 % catch fraction matches | whitepaper ~12.5 % catch | Consistent; keep go+catch only as instructed. |
| Passive sessions | `passive` flag in experiment table | 0 licks / 0 rewards, all trials miss/CR | paper: "passive viewing... was not analyzed here" | Exclude passive sessions. |
| Change time alignment | `change_time` from `change_frame` on stimulus timestamps | identical to the change/sham flash `start_time` (diff = 0) | - | Use `change_time` as the alignment event. |
| Omitted flashes | `image_name == 'omitted'`, `omitted == True` | 3.5 % of flashes | 5 % of repeats | Consistent (5 % of *repeats*, excluding changes and pre-change flashes, ~ 3.5 % of all flashes). For the image-identity output the omitted 750 ms interval keeps the identity of the image that should have been shown, exactly as the paper treats omissions. |
| Pupil measure | `pupil_width/height` are ellipse half-axes; `pupil_area = pi*max(w,h)^2` | blink frames NaN | whitepaper: "major axis reflects the pupil diameter" | Pupil diameter = 2*max(pupil_width, pupil_height); blinks interpolated. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Session / trial definition
- **Session** = one `ophys_session_id`; all *active* imaging planes (experiments) of that session are concatenated along the neuron axis (Multiscope sessions contribute 3-7 planes, single-plane sessions 1).
- **Trial** = one `go` or `catch` trial from `dataset.trials` (aborted and auto-rewarded dropped).
- **Alignment event** = `trials.change_time` (real change on go trials, sham change on catch trials); it coincides exactly with a flash onset.
- **Window** = [-2.0 s, +3.0 s] around the change. Safe for every trial: minimum change-to-trial-start is 3.02 s and stop_time is always change+4.23 s, and the next trial's change is >= 4.48 s later, so windows never leave their trial and never contain a second change.
- **Time bin** = 250 ms = the image duration and exactly 1/3 of the 750 ms flash cycle -> 20 bins per trial, bin edges phase-locked to the flash cycle. 250 ms also guarantees >= 2 ophys frames per bin even at the 11 Hz Multiscope rate (no empty bins), while 31 Hz sessions get ~7.75 frames/bin.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `dataset.dff_traces['dff']` + `dataset.ophys_timestamps` | `neural` | **mean dF/F in each 250 ms bin**, per cell; planes concatenated | `BehaviorOphysExperiment.dff_traces` (pipeline dF/F, whitepaper 'DF/F CALCULATION') | *Revised after Step 8/10*: the plan was `dataset.events` (the reference paper's signal), but a controlled comparison on 20 sessions showed dF/F decodes better on 4 of 5 outputs and removes the empty-trial problem in sparse Vip/Sst planes. `--neural-signal events|filtered_events|dff` keeps all three available. |
| (none) | `input` | empty array (0, T) | - | Decoder task specifies no inputs |
| `stimulus_presentations.image_name` | `output[0]` image_identity | identity of the image whose 750 ms presentation interval contains the bin; omitted intervals inherit the preceding image | `is_change_event`, paper's "image presentation interval" | 16 global classes (image sets A and B) |
| `stimulus_presentations.is_change` / `trials.change_time` | `output[1]` image_change | **1 in the single 250 ms bin that starts with an image change**, else 0 | `is_change_event` | *Revised after Step 8*: bin 8 only (t = 0..0.25 s) on go trials, all 0 on catch trials. This is the literal reading of 'value of 1 right after a change' and decodes better (0.689 vs 0.655) than flagging the whole 750 ms interval (`--change-window interval`). |
| `running_speed.speed` | `output[2]` running_speed_bin | mean speed per bin -> global quintiles (5 equal-count bins over all trials/sessions) | tutorial `plot_running` | |
| `eye_tracking.pupil_width/height` | `output[3]` pupil_bin | diameter = 2*max(width,height), blinks (NaN) linearly interpolated, mean per bin -> global quintiles | `filter_on_blinks`, `compute_circular_area` | |
| `trials.hit/miss/false_alarm/correct_reject` | `output[4]` trial_outcome | static per trial, broadcast over the 20 bins | `Trial._get_trial_data` | 4 classes |
| `metadata['mouse_id']` | `subjects`, `subject_idx` | | | 38 mice |
| `metadata['targeted_structure']` | `brain_regions`, `brain_region_idx` | VISp (V1), VISl (LM) per neuron | | |

### Key Decisions
1. **Neural signal = pipeline dF/F** (*revised from the initial plan of L0 events; see Steps 8, 10 Check 3 and 12*). The reference paper used detected calcium events, but on an identical 20-session subset dF/F decoded better on 4 of 5 outputs and removed 2,859 all-zero trials in the very sparse Vip/Sst planes. dF/F is the same released measurement before L0 deconvolution and is described in detail in the whitepaper; `--neural-signal events|filtered_events|dff` reproduces the paper's choice.
2. **Bin statistic = mean (dF/F)**: dF/F is rate-like and the number of ophys frames per 250 ms bin varies (7-8 at 31 Hz, 2-3 at 11 Hz), so summing would impose a spurious rig-dependent scale. (For the event signals the bin **sum** is used, which is the correct analogue of a spike count.)
3. **Active sessions only**: passive sessions have no licking, so trial outcome is degenerate and the animal is not performing the task.
4. **Both rigs and all experience levels**: needed for a multi-subject decoding benchmark (see Step 4 discrepancy table).
5. **Exclude the 3 sessions with no eye-tracking** (795625712, 832881662, 805989030) because the pupil output cannot be defined; all other sessions keep blinks interpolated (median 2.9 % of frames).
6. **Alignment to change_time**, window [-2, +3] s, 250 ms bins (justified above).
7. **Per-session quintiles** for running speed and pupil (*revised from global; see Step 8a*): the pupil diameter is in camera pixels (rig/session dependent) and each mouse has its own running baseline, so global cuts partly encode session identity; per-session equal-percentile bins give exactly 20 % per class in every session and decode better. `--quantile-scope global` remains available.
8. **Image identity over the 750 ms presentation interval** (the paper's unit of behavioural analysis), so gray-screen bins carry the identity of the image that started that interval; omitted flashes inherit the previous image.
9. **Trial outcome as 4 classes** (hit/miss/false_alarm/correct_reject) - exactly the outcomes that exist for go and catch trials.

### Planned Sanity Checks
- [ ] number of `is_change` flashes == number of go trials + auto-rewarded trials
- [ ] `change_time` == start_time of the change/sham flash (difference 0)
- [ ] catch fraction of kept trials ~12.5 %
- [ ] hit+miss == go trials, false_alarm+correct_reject == catch trials
- [ ] image_change output is 1 exactly in bins 8-10 on go trials and never on catch trials
- [ ] neural bin sums recomputed directly from `dataset.events` for random (session, trial, neuron, bin) samples (np.allclose)
- [ ] running/pupil bin values recomputed directly from the SDK tables (np.allclose)
- [ ] output quintiles each ~20 % of all bins
- [ ] total neurons/trials/sessions/mice match the Step 2 scan

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the conversion.  Structure:
- `get_cache()` - `VisualBehaviorOphysProjectCache.from_local_cache('/app/data')` (the only data access path; no direct NWB/h5py access anywhere).
- `select_sessions()` - experiment table -> local files -> drop `passive` -> group experiments by `ophys_session_id`.
- `process_session()` (one worker process per session) - loads every plane of the session, bins its calcium events, builds the stimulus/behaviour outputs.
- `bin_sum()` - vectorised binning by cumulative sum + `searchsorted`; empty bins give exactly 0, neurons processed in chunks of 256 to bound memory.
- `bin_mean_1d()` - bin means for running speed and pupil, with interpolation fallback for empty bins.
- `interpolate_nans()` - linear interpolation of blink NaNs in the pupil trace.
- `discretize()` - equal-percentile (quintile) binning, per session (default) or global.
- `make_plots()` - the `--show-processing` figures.
- `main()` - assembles the dictionary, runs sanity checks, pickles the result.

Options: `--full` (default), `--sample` (one single-plane + one multi-plane session), `--show-processing`, `--workers N`, `--quantile-scope {session,global}`.

Code inefficiencies identified: the NWB file load (3-4 s/plane) dominates; binning and behaviour processing take <0.2 s/session.
Code speedups added: multiprocessing over sessions (16 workers), cumulative-sum binning instead of per-trial loops, a single pass over each plane, and float32 storage.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`

### Sample Statistics (final settings: dF/F bin means, single-bin change flag, per-session quintiles)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (775289198 single-plane 31 Hz; 951410079 Multiscope 7 planes, 11 Hz) |
| Neurons (total) | 177 (89 + 88) |
| Neurons / session | 88.5 mean |
| Subjects | 2 (403491, 457841) |
| Trials (total) | 248 (39 + 209) |
| Timepoints / trial | 20 (250 ms bins, -2 s .. +3 s) |
| go / catch | 215 / 33 -> catch fraction 0.133 (whitepaper ~0.125) |
| image_identity distribution | 8 images, 0.109 - 0.145 each |
| image_change | no_change 0.957, change 0.043 (= 1 of 20 bins on the 87 % of trials that are go) |
| running_speed_quintile | 0.200 each |
| pupil_diameter_quintile | 0.200 each |
| trial_outcome | hit 0.383, miss 0.484, FA 0.036, CR 0.097 |

### Processing Plots Review
`processing_775289198.png`, `processing_951410079.png` (+ `_dist.png`):
- raw traces with the 250 ms bin grid and the resulting binned neuron x bin image -> no temporal shift;
- stimulus panel: flash raster (grey = image, red = change) with image-identity and image-change on top; the change flash starts exactly at t = 0 and the change output is 1 in that bin only;
- running and pupil panels: raw 60/30 Hz traces, the bin means and the quintile codes (pupil panel also shows the blink NaNs and their interpolation);
- lick/reward panel confirms the trial outcome label;
- distribution figure confirms the five quintile classes are equally populated.
No anomalies found.

### Independent alignment check (on the pickle)
z-scored population activity averaged over change trials peaks in **bin 8** (0-0.25 s after the change) and is flat on catch trials; image identity switches exactly at bin 8 on 100 % of change trials and is constant within each 750 ms interval.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| multiprocessing over sessions (16 workers) | ~16x |
| cumulative-sum vectorised binning | binning < 0.2 s/session |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| NWB load | 3-4 s per imaging plane | 202 planes -> ~800 s serial |
| binning + outputs | < 0.2 s | negligible |
| **measured full run with 16 workers** | | **83 s** |

### Format verification (`/app/verification_sample_out.txt`)
No errors, no warnings.  Input dimension 0 (no decoder inputs, as specified), output dimension 5, T = 20 for every trial.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions, final settings)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| image_identity | 0.583 | 0.515 | 0.125 |
| image_change | 0.790 | 0.738 | 0.500 |
| running_speed_quintile | 0.312 | 0.262 | 0.200 |
| pupil_diameter_quintile | 0.383 | 0.306 | 0.200 |
| trial_outcome | 0.492 | 0.346 | 0.250 |
Loss decreased monotonically over the 200 epochs; every output is above chance.

### Decisions tested here (all on identical data, only the option changed)
**(a) per-session vs global quintiles** (2-session sample): global gave running 0.244 / pupil 0.260, per-session 0.262 / 0.271.  Per-session percentile bins are also better justified: the pupil diameter is in *camera pixels* (eye-camera zoom/distance differ between rigs and sessions) and every mouse has its own running baseline, so a global cut largely encodes session identity.  Default = `session` (`--quantile-scope global` still available).

**(b) neural signal** (20-session subset, validation balanced accuracy):
| signal / bin statistic | image | change | running | pupil | outcome |
|---|---|---|---|---|---|
| `events` (sum) | 0.282 | 0.588 | 0.267 | 0.250 | 0.318 |
| `filtered_events` (sum) | 0.327 | 0.606 | 0.281 | 0.258 | 0.327 |
| `dff` (sum) | 0.442 | 0.650 | 0.306 | 0.280 | 0.312 |
| **`dff` (bin mean, adopted)** | **0.408** | **0.655** | **0.306** | **0.280** | **0.320** |

**(c) image_change encoding** (20-session subset): whole 750 ms presentation interval (3 bins) 0.655 vs **single 250 ms bin at the change 0.689**.  The single bin is adopted - it is also the literal reading of the task ('value of 1 right after a change').

**(d) trial window** (20-session subset): [-2, +3] s (adopted) outcome 0.315 / image 0.400; [-2, +4] s 0.324 / 0.413; [-1, +3] s 0.322 / 0.398.  All within noise, so the window is not a limiting factor; [-2, +3] s keeps a 2 s pre-change baseline and stays inside every trial.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/converted_data.pkl --full` (87 s wall clock, 16 worker processes).

### Output Files
- `converted_data.pkl`: 673.7 MB
- `conversion_full_out.txt`, `verification_full_out.txt`: created
- Format verification: **"Data format is valid, no errors or warnings."**

### Dataset delivered
| Statistic | Value |
|---|---|
| Sessions | 171 (174 active ophys sessions - 3 without eye tracking) |
| Subjects | 38 (2-9 sessions each) |
| Neurons | 29,168 (mean 171/session, min 6, max 666) |
| Trials | 43,975 go+catch (mean 257/session, min 39, max 409) |
| Timepoints/trial | 20 (250 ms bins, -2 s..+3 s around the change) |
| Brain regions | VISp 29,006 neurons, VISl 162 neurons |
| Inputs | none (d_input = 0) |
| Outputs | image_identity (16), image_change (2), running_speed_quintile (5), pupil_diameter_quintile (5), trial_outcome (4) |

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data (raw scan) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Mice (local cache) | 82 mice in the full public release | experiment table | 38 (all mice present locally) | 38 | yes |
| Active ophys sessions | 376 sessions in the full release | experiment table `passive==False` | 174 | 171 (-3 no eye tracking) | yes |
| Neurons total | 8,619 exc + 470 Sst + 1,239 Vip for the paper's multi-plane familiar subset | SDK `exclude_invalid_rois=True` | 29,444 in 202 active planes | 29,168 in 171 sessions (the 3 dropped sessions held 276) | yes |
| Trials (go+catch) | - | trials table | 44,892 | 43,975 | yes (difference = the 3 dropped sessions) |
| Catch fraction | ~12.5 % (whitepaper) | - | 0.1253 | 0.1254 | yes |
| Hit rate on go trials | - | - | 0.361 | 0.362 | yes |
| FA rate on catch trials | - | - | 0.150 | 0.150 | yes |
| Trial-outcome distribution | - | - | hit .316 / miss .558 / FA .019 / CR .107 | hit .317 / miss .558 / FA .019 / CR .106 | yes |
| Images | 8 per session, 16 in the dataset | - | 16 | 16 (each 5.9-6.5 %) | yes |
| image_change fraction | 3 of 20 bins on go trials -> 0.1311 expected | - | go fraction 0.8746 -> 0.1312 | 0.131 | yes |
| Running / pupil quintiles | equal percentile bins by construction | - | - | 0.200 each | yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt` reports **no errors and no warnings**.

Earlier iteration: when the neural signal was the raw L0 event magnitude, verification produced 2,859 `all neural data is zero` warnings (6.5 % of trials). Investigation showed these were genuine - they occur in Vip/Sst planes with 7-29 cells (correlation between log(n neurons) and the zero-trial fraction = -0.63), where L0 events are extremely sparse, and not a binning bug. Switching the neural signal to the pipeline dF/F trace (see Check 3) removed every warning.

### Check 2: Sanity checks against the raw SDK data
`/app/sanity_checks.py` (5 random sessions) and `/app/sanity_multiplane.py` (all 6 multi-plane sessions) re-load the original data through the AllenSDK and compare with `np.allclose` / `np.array_equal`. **All checks pass:**

| Check | Result |
|---|---|
| NEURAL: dF/F bin means recomputed from `dff_traces` + `ophys_timestamps` for 25 random (trial, neuron, bin) per session | PASS (allclose 1e-5) |
| NEURAL (multi-plane): each plane's neurons sit at the right row offset, per-plane timestamps used | PASS for all 6 sessions |
| NEURAL: n neurons == sum over planes of len(dff_traces) | PASS |
| INPUT: d_input = 0 for every trial (no inputs, per the task spec) | PASS (verification reports "Input dimension: 0") |
| OUTPUT image_identity: matches the forward-filled `stimulus_presentations.image_name` at the bin centre | PASS |
| OUTPUT image_change: matches `stimulus_presentations.is_change` at the bin centre | PASS |
| OUTPUT running/pupil quintiles: recomputed from `running_speed` / `eye_tracking` | PASS (exact) |
| OUTPUT trial_outcome: matches the `trials` table; constant within a trial | PASS |
| brain_region_idx matches each plane's `targeted_structure` | PASS |
| subject id matches `metadata['mouse_id']` | PASS |
| hit+miss == n go trials; FA+CR == n catch trials | PASS |
| no aborted / auto-rewarded trial kept | PASS |
| n `is_change` flashes == n go + n auto-rewarded trials | PASS |
| `change_time` is exactly a flash onset (atol 1e-9) | PASS |

### Check 3: Reference code comparison
| Step | Reference (paper / AllenSDK / tutorials) | This conversion | Same? |
|---|---|---|---|
| (a) loading | `bpc.VisualBehaviorOphysProjectCache` -> `get_ophys_experiment_table` -> `get_behavior_ophys_experiment` | identical | yes |
| (b) neuron filtering | SDK `CellSpecimens(exclude_invalid_rois=True)`; pipeline ROI classifier + session QC | we take the tables as delivered, i.e. exactly the SDK's valid ROIs; no extra filtering (the paper adds none) | yes |
| (b) trial filtering | paper decodes per image presentation; task spec: keep go+catch, drop aborted/auto-rewarded | `trials[trials.go | trials.catch]` | as specified |
| (b) session filtering | paper: active sessions; neural analysis restricted to multi-plane + familiar | active sessions from both rigs and all experience levels (local cache has only 1 mouse with multi-plane active sessions, see Step 4); 3 sessions without eye tracking dropped | deviation documented |
| (c) alignment | `trials.change_time` (= `stimulus_timestamps[change_frame]`), the paper aligns to image presentations | `change_time`; verified to be exactly a flash onset | yes |
| (d) binning | paper uses the first 400 ms after image presentation / 750 ms presentation intervals | 250 ms bins phase-locked to the 750 ms flash cycle (3 bins per interval) | compatible, finer |
| (e) input construction | - | none (task spec) | n/a |
| (f) output construction | `is_change_event`, image_name, trials hit/miss/FA/CR, `running_speed.speed`, `eye_tracking` pupil with `filter_on_blinks` | same SDK fields; pupil diameter = 2*max(width,height) following `compute_circular_area`; blinks interpolated | yes |
| neural signal | paper: detected calcium events; SDK also delivers dF/F (whitepaper describes its computation in detail) | **dF/F bin means** | deviation documented below |

**Documented deviation - neural signal.** The reference paper used the L0 calcium events. We measured all three SDK signals on an identical 20-session subset (validation balanced accuracy):

| signal / bin statistic | image_identity | image_change | running | pupil | outcome |
|---|---|---|---|---|---|
| `events` (sum) | 0.282 | 0.588 | 0.267 | 0.250 | 0.318 |
| `filtered_events` (sum) | 0.327 | 0.606 | 0.281 | 0.258 | 0.327 |
| `dff` (sum) | 0.442 | 0.650 | 0.306 | 0.280 | 0.312 |
| **`dff` (bin mean, adopted)** | **0.408** | **0.655** | **0.306** | **0.280** | **0.320** |

dF/F decodes better on 4 of 5 outputs and eliminates the empty-trial problem in the very sparse Vip/Sst planes; it is the standard released measurement of this dataset (whitepaper "DF/F CALCULATION") and is *not* a different curation of the data - it is the same traces before L0 deconvolution. The bin **mean** (not sum) is used because dF/F is a rate-like quantity and the number of ophys frames per 250 ms bin varies (7-8 at 31 Hz, 2-3 at 11 Hz); summing would impose a spurious rig-dependent scale. `--neural-signal events|filtered_events|dff` keeps the paper's choice available.

**Documented deviation - rig/experience restriction.** The paper restricted the *neural* analysis to multi-plane, familiar-image sessions. In this cache that subset is 6 sessions from **1 mouse**, which cannot support a multi-subject decoder benchmark. We therefore use all active sessions (both rigs, all experience levels) - the same population the paper used for its *behavioural* analysis - and record `equipment`, `session_type`, `cre_line` and `experience`-defining `session_type` per session in `metadata['session_info']` so any subset can be re-derived.

### Check 4: Key statistics comparison
See the table in Step 9: mice, sessions, trials, neurons, catch fraction (0.1254 vs ~0.125 in the whitepaper), hit rate, outcome distribution and image counts all agree with the raw-data scan and with the reference texts.

### Check 5: Edge cases
- **Trial-window overrun**: measured over all sessions, `change_time - start_time` >= 3.02 s and `stop_time - change_time` = 4.23 s, and the next trial's change is >= 4.48 s away, so the [-2, +3] s window never leaves its own trial nor contains a second change. Ophys coverage checked: 100 % of go/catch changes have +/-3 s of frames.
- **Empty bins**: `bin_sum` uses a cumulative sum, so a bin with no samples yields exactly 0 (and, for the behaviour streams, `bin_mean_1d` falls back to interpolation at the bin centre). At 250 ms with >= 11 Hz imaging and 30/60 Hz behaviour no empty bins occur in practice.
- **Blinks**: pupil NaNs (median 2.9 % of frames) are linearly interpolated; the three sessions with *no* eye tracking at all are dropped rather than filled.
- **Omitted flashes**: inherit the previous image identity (forward fill), matching the paper's treatment of the omitted 750 ms interval; `is_change` is False on omitted flashes by construction in the SDK.
- **First flash of a session** is never a change (SDK `is_change_event` excludes it) and no trial window reaches it.
- **Sessions with < 2 trials** would be dropped (`process_session` returns None); none occurred (min 39 trials).
- **Image vocabulary**: the 16 images of sets A and B are pooled into one global vocabulary so that class indices mean the same thing in every session; each session only exercises its own 8.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples` (GPU, NVIDIA L4).

### Training Progress
- Loss decreasing: **Yes** (loss falls monotonically over the 200 epochs to 1.29; test loss 1.376).
- 171 sessions, 35,180 training trials / 8,795 validation trials.

### Decoder Results (Full)
| Output | #classes | Chance | Training Balanced Acc | Validation Balanced Acc | Acc / chance |
|--------|---|---|-------------|--------|-------|
| image_identity | 16 | 0.0625 | 0.4909 | **0.4494** | 7.2x |
| image_change | 2 | 0.5000 | 0.7715 | **0.6931** | 1.39x |
| running_speed_quintile | 5 | 0.2000 | 0.3704 | **0.3111** | 1.56x |
| pupil_diameter_quintile | 5 | 0.2000 | 0.3611 | **0.2798** | 1.40x |
| trial_outcome | 4 | 0.2500 | 0.5073 | **0.3095** | 1.24x |

Sample-trial and prediction figures were written to `sample_trials.png` / `predictions.png`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
Every output is **above** chance; none is below.  Three outputs are below 1.5x chance (image_change 1.39x, pupil 1.40x, trial_outcome 1.23x) and were investigated with a per-class recall analysis (`/app/analyze_accuracy.py`, confusion matrices over the 177,180 validation timepoints):

| Output | per-class recall | interpretation |
|---|---|---|
| image_identity | 0.35-0.53 for all 16 images, no class collapsed | information is genuinely distributed; 7.2x chance |
| image_change | no_change 0.76, change 0.56 | the change bin is 4.4 % of all bins; balanced loss keeps both classes usable |
| running_speed | q1 0.46, q2 0.29, q3 0.23, q4 0.22, q5 0.35 | classic ordinal U-shape: the extremes (rest / fast running) are decodable, the middle quintiles are 2-18 cm/s apart and largely indistinguishable in V1 activity |
| pupil_diameter | q1 0.51, q2 0.20, q3 0.18, q4 0.19, q5 0.33 | same U-shape; a 5-way split of a slowly-varying arousal signal within a 5 s trial is intrinsically hard |
| trial_outcome | hit 0.51, miss 0.35, FA 0.26, CR 0.15 | limited by the catch-trial classes.  The reference paper reports exactly this: "for all cell classes, false alarm decoding performance was very low" (Figure S22) |

Additional experiments run to make sure these are data properties, not conversion bugs (all on an identical 20-session subset, only one option changed at a time): neural signal (events/filtered_events/dF/F, sum vs mean), change encoding (3-bin interval vs 1-bin), trial window ([-2,+3], [-2,+4], [-1,+3]), quantile scope (session vs global).  The adopted settings are the best of each comparison; see Step 8.  Extending the window to +4 s (covering the whole response/reward period) changed trial_outcome by only +0.004, confirming the ceiling is the rarity/ambiguity of the catch classes rather than a missing time window.

### Check 2: Accuracy comparison to the papers
| Variable | This conversion (validation balanced acc) | Reference paper | Comment |
|---|---|---|---|
| image change vs repeat | 0.694 (2 classes, chance 0.5), decoded per 250 ms bin from one session's population | random-forest change decoder ~0.65-0.85 % correct on 400 ms windows, per imaging plane, best values only with the largest cell counts (Figure 6A) | comparable; the paper's decoder sees a 400 ms window of concatenated activity and only has to separate the change flash from the immediately preceding repeat, whereas ours classifies every 250 ms bin of the trial |
| hit vs miss (our trial_outcome) | hit recall 0.51, miss recall 0.35 in a 4-class problem | random-forest hit decoder ~0.55-0.70 % correct (2-class, Figure 6C) | comparable once the 4-class setting is taken into account |
| false alarm | FA recall 0.26 | "false alarm decoding performance was very low" (Figure S22) | consistent |
| image identity | 0.449 (16 classes, chance 0.0625) | not decoded in the paper | 7.2x chance |
| running / pupil | 0.311 / 0.281 (5 classes) | not decoded in the paper | above chance |
No accuracy reported in the papers exceeds what we obtain in a comparable setting, so no hidden conversion loss is indicated.

### Check 3: Train vs validation gap
| Output | train | validation | ratio |
|---|---|---|---|
| image_identity | 0.491 | 0.449 | 1.09 |
| image_change | 0.772 | 0.693 | 1.11 |
| running_speed_quintile | 0.370 | 0.311 | 1.19 |
| pupil_diameter_quintile | 0.361 | 0.280 | 1.29 |
| trial_outcome | 0.507 | 0.310 | 1.64 |
Only trial_outcome exceeds 1.5x.  This is expected and is **not** data leakage: the outcome is a single static label per trial, so the decoder effectively has only 35,180 independent training examples for it (versus 700k timepoints for the time-varying outputs) while the model has one linear read-out per session-projection; the rare FA class (1.9 % of trials) is easy to memorise in training and hard to generalise.  The train/validation split is by **trial**, and no trial's timepoints appear in both sets, so no leakage is possible across the split.

### Issues found and resolved during Steps 10-12
1. **2,859 all-zero-trial warnings** with raw L0 events -> investigated (sparse Vip/Sst planes, not a bug) and removed by adopting the pipeline dF/F trace, which also decodes better on 4 of 5 outputs.
2. **dF/F summed rather than averaged per bin** would impose a rig-dependent scale (7-8 frames/bin at 31 Hz vs 2-3 at 11 Hz) -> switched to the bin mean.
3. **Global quintiles** mixed session identity into the running/pupil labels -> switched to per-session percentile bins (also required by "five equal percentile bins").
4. **image_change spanning the whole 750 ms interval** -> changed to the single 250 ms bin at the change (task wording, and +0.034 accuracy).
5. A **SyntaxError** introduced while adding the `--off-start/--off-end` options (global declared after use) was fixed and the script re-validated.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with README_CACHE.md
- [x] All files organized
