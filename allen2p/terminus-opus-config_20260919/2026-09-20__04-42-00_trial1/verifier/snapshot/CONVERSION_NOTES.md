# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory - Visual Behavior 2P (Ophys)
- **Date started**: (session start)
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment: python3, numpy 2.4.4, torch 2.6.0+cu124, CUDA available = True.

Directory contents of /app:
- `.manifest` (1326 file list), `Dockerfile`, `docker-compose.yaml`
- `code/` : AllenSDK source tree (allensdk/...)
- `data/` : `visual-behavior-ophys-1.1.0/` (247 GB) with `behavior_ophys_experiments/` (283 `behavior_ophys_experiment_<id>.nwb` files) and `project_metadata/` (behavior_session_table.csv, ophys_cells_table.csv, ophys_experiment_table.csv, ophys_session_table.csv); plus `visual-behavior-ophys_project_manifest_v1.1.0.json`, `_downloaded_data.json`, `_manifest_last_used.txt`
- `tutorials/` : visual_behavior_load_ophys_data.(py|ipynb), visual_behavior_compare_across_trial_types.py, visual_behavior_mouse_history.py, visual_behavior_ophys_data_access.py, visual_behavior_ophys_dataset_manifest.py
- `decoder.py`, `train_decoder.py`, `methods.txt`, `paper.pdf`, `whitepaper.pdf`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `VisualBehaviorOphysProjectCache.get_behavior_ophys_experiment(id)` | code/allensdk/.../behavior_project_cache.py | LOADING | Canonical entry point used by all tutorials; internally resolves the NWB path and calls `BehaviorOphysExperiment.from_nwb_path` |
| `BehaviorProjectCloudApi.get_behavior_ophys_experiment` | code/allensdk/.../behavior_project_cloud_api.py:159 | LOADING | Shows the cache maps `file_id` -> local nwb path, then `BehaviorOphysExperiment.from_nwb_path(str(data_path))`. I replicate this by calling `from_nwb_path` directly on the local files (no S3 access needed). |
| `BehaviorOphysExperiment.from_nwb(...)` | code/allensdk/.../behavior_ophys_experiment.py:226 | LOADING/CURATION | Builds the experiment object. Note kwarg `exclude_invalid_rois=True` (default) -> only valid ROIs (cells) are returned. |
| `.events` (property) | behavior_ophys_experiment.py:554 | PROCESSING | DataFrame indexed by cell_specimen_id with columns `events`, `filtered_events`, `lambda`, `noise_std`. These are the *detected calcium events* used by the Vip/Sst paper. |
| `.dff_traces` | behavior_ophys_experiment.py:530 | PROCESSING | dF/F traces. Already computed in the released NWB (whitepaper "DF/F CALCULATION"); **no need to compute dF/F ourselves**. |
| `.ophys_timestamps` | behavior_ophys_experiment.py:523 | ALIGNMENT | Time (s, session clock) of every 2p frame. All streams in the NWB are already on this common session clock. |
| `.cell_specimen_table` | behavior_ophys_experiment.py:588 | CURATION | `valid_roi`, `cell_roi_id`, ROI geometry. |
| `.trials` | BehaviorSession | PROCESSING | One row per behavioral trial: `start_time, stop_time, change_time, go, catch, aborted, auto_rewarded, hit, miss, false_alarm, correct_reject, initial_image_name, change_image_name, is_change, response_latency, lick_times, reward_time`. |
| `Trial._get_trial_data` | code/allensdk/.../trials/trial.py:150-210 | PROCESSING | Definitive definition of trial categories: `go` = real change, `catch` = sham change, `aborted` = licked before change, `auto_rewarded` = free reward (explicitly "should not be categorized as hit/miss"); `hit/miss/false_alarm/correct_reject` are mutually exclusive outcomes. |
| `Trial.calculate_change_frame` / `add_change_time` | trial.py:340-420 | ALIGNMENT | `change_time` = stimulus timestamp of the change frame on `go`/`auto_rewarded` trials and of the **sham** change frame on `catch` trials. This is the natural per-trial alignment event and is defined for every go and catch trial. |
| `.stimulus_presentations` | BehaviorSession | PROCESSING | One row per flash: `start_time, end_time, image_name, image_index, omitted, is_change, stimulus_block_name, active`. Must be filtered with `stimulus_block_name.str.contains("change_detection")` (SDK >=2.16 warning) to get the task block, excluding the 5-min gray screens and the natural movie block. |
| `.running_speed` | BehaviorSession | PROCESSING | DataFrame `timestamps`, `speed` (cm/s) at ~60 Hz. |
| `.eye_tracking` | BehaviorSession | PROCESSING | ~30 Hz; `pupil_area`, `pupil_width`, `pupil_height`, `likely_blink`; values are NaN where `likely_blink`. |
| tutorials/visual_behavior_load_ophys_data.py | tutorials | LOADING | Shows `np.vstack(dataset.dff_traces.dff.values)` idiom and that dff/events are plotted against `ophys_timestamps`. |
| tutorials/visual_behavior_compare_across_trial_types.py | tutorials | PROCESSING | Shows per-trial slicing of running/licks/rewards/pupil by `start_time`/`stop_time`, and the `change_detection` stimulus-block filter. |

### Notes
- **dF/F does not need to be computed**: the NWB files already contain the pipeline dF/F and the detected events. The whitepaper documents how both were produced.
- **Neuron quality filtering is already applied**: `from_nwb_path` defaults to `exclude_invalid_rois=True`, so ROIs failing the whitepaper's "ROI FILTERING" criteria (unions, duplicates, motion-border, dendrites, too small/dim) are dropped. Verified empirically: `cell_specimen_table.valid_roi` is `True` for 100% of returned cells.
- This is 2-photon imaging, so there is no electrophysiology spike-sorting quality metric to apply.
- Loading one experiment from the local NWB takes ~2.5-3 s, so the full dataset is cheap to convert with multiprocessing.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
/app/data/
  visual-behavior-ophys_project_manifest_v1.1.0.json   # cloud manifest (file_id -> path/size/hash)
  visual-behavior-ophys-1.1.0/
    project_metadata/
      behavior_session_table.csv    # all behavior sessions incl. training
      ophys_session_table.csv       # one row per ophys session
      ophys_experiment_table.csv    # one row per imaging plane (experiment); 1936 rows total
      ophys_cells_table.csv         # cell_specimen_id <-> experiment mapping
    behavior_ophys_experiments/
      behavior_ophys_experiment_<ophys_experiment_id>.nwb   # 284 files, ~900 MB each, 247 GB total
```
- Only **284 of the 1936** experiments in the manifest are actually present on disk; the conversion therefore works from the intersection of the experiment table with the files on disk.
- **experiment vs session**: an *experiment* is one imaging plane; a *session* is one continuous recording. Single-plane (`VisualBehavior`, CAM2P rigs) sessions have 1 experiment; Multiscope sessions have up to 8 simultaneously-recorded planes sharing one behavior stream and one clock.

### Contents of the 284 available experiments
| Split | Counts |
|---|---|
| project_code | VisualBehavior 239, VisualBehaviorMultiscope 45 |
| session_type | OPHYS_1_images_A 55, OPHYS_3_images_A 55, OPHYS_4_images_B 46, OPHYS_6_images_B 46, **OPHYS_2_images_A_passive 40, OPHYS_5_images_B_passive 42** |
| cre_line | Slc17a7-IRES2-Cre 153, Sst-IRES-Cre 85, Vip-IRES-Cre 46 |
| targeted_structure | VISp 261, VISl 23 |
| mice / ophys sessions | 38 mice, 247 sessions |

Passive sessions (OPHYS_2, OPHYS_5) have the lick spout retracted and therefore contain **no go/catch trials and no outcomes**, so they cannot supply the required "trial outcome" output and are excluded (see Step 5).

### Dataset Size (from data files; active/behavior experiments only, all 202 loaded successfully)
| Statistic | Value |
|-----------|-------|
| Experiments (imaging planes) | 202 |
| Sessions (ophys_session_id) | 174 (168 single-plane + 6 Multiscope with 3-7 planes) |
| Neurons (total, valid ROIs) | 29,444 |
| Neurons / experiment | mean 145.8, median 63.5, min 4, max 666 |
| Subjects (mice) | 38 |
| Trials (total, all types) | 138,638 |
| Go trials | 45,477 |
| Catch trials | 6,515 |
| Aborted trials | 85,831 |
| Auto-rewarded trials | 815 |
| Go+Catch trials (what we keep) | 51,992 summed **per experiment**; this double-counts Multiscope planes that share one behaviour stream. Deduplicated **per session** = 44,892 (mean 257/session). See Step 9. |
| Outcomes | hit 15,936, miss 29,541, false alarm 931, correct reject 5,584 |
| Ophys frame rate | 31 Hz (168 single-plane exps), 11 Hz (34 Multiscope exps) |
| Running speed rate | ~60 Hz |
| Eye tracking rate | ~30 Hz (3 experiments have an empty eye-tracking table) |
| Image sets | A = im061,062,063,065,066,069,077,085 (110 exps); B = im000,031,035,045,054,073,075,106 (92 exps); 16 distinct images |
| change_time missing on go/catch trials | 0 (always defined) |
| min (change_time - start_time) over all go/catch trials | 2.79 s |
| min (stop_time - change_time) over all go/catch trials | 4.20 s |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Behavior sessions / mice in full release | 376 sessions, 82 mice | paper: contains behavior from 376 imaging sessions from 82 mice |
| Paper neural subset | 8,619 exc (21 sessions, 9 mice); 470 Sst (15 sessions, 6 mice); 1,239 Vip (21 sessions, 9 mice) | paper: Our dataset contains 8,619 excitatory cells ... (Multiscope, familiar only) |
| Image duration | 250 ms | paper: natural images (250 ms stimulus duration) |
| Inter-stimulus gray | 500 ms | paper: gray screen (500 ms inter-stimulus duration) |
| Flash cycle / image presentation interval | **750 ms** | paper: By image presentation interval we refer to the 750 ms interval beginning with each image presentation |
| Omission probability | 5% of image repeats; changes and pre-change flashes never omitted | paper + whitepaper |
| Images per session | 8 (two sets, A and B) | whitepaper: Each session included 8 images, for a total of 64 possible transitions |
| Change time distribution | truncated exponential 2.25-8.25 s after trial start, mean 4.2 s | whitepaper: Change-times were selected from a truncated exponential distribution ranging from 2.25 to 8.25 seconds ... mean change time of 4.2 seconds |
| Response window | 150-750 ms after change | whitepaper: licked in a 0.150 to 0.750 second window following the image display time |
| Catch probability | ~12.5% in late training / imaging | whitepaper: matrix sampling algorithm ... pushing the actual catch probability to ~12.5% |
| Aborted trials | animal licks before the change; trial is reset | whitepaper: In trials when a mouse licked prior to the stimulus change the trial was reset |
| Auto-rewarded trials | free reward, should not be categorized as hit/miss | allensdk trial.py docstring |
| Neural signal for analysis | detected calcium events | paper: For all analysis of neural data we used the detected calcium events |
| Decoding window in paper | first 400 ms after image presentation | paper: Decoding was performed on neural activity in the first 400 ms after each stimulus presentation |

**Measured in our data for comparison** (202 active experiments): catch fraction of go+catch = 6,515/51,992 = 12.5%, matching the whitepaper ~12.5% exactly; hit rate among go = 0.350; false-alarm rate among catch = 0.143; median (change_time - start_time) = 3.02 s with mean go/catch trial length 8.4 s, consistent with the stated truncated-exponential design.

### Processing Details
- All NWB data streams (ophys, running, eye, stimulus, trials, licks, rewards) are already rebased onto one common session clock, so alignment means resampling onto a chosen grid, not applying offsets.
- The paper assigns behavioral events to 750 ms image presentation intervals; this is the natural time bin and is adopted here.
- The paper aligns neural analyses to image presentations and uses the first 400 ms after onset for decoding.

### Curation Steps

**Neuron curation rules**:
- Only valid ROIs. `from_nwb_path` applies `exclude_invalid_rois=True` by default, implementing the whitepaper ROI FILTERING step (removes unions, duplicates, motion-border ROIs, likely dendrites, too small/narrow/dim). Verified: `valid_roi` is True for 100% of returned cells.
- No further per-neuron filtering is described for this dataset; unlike ephys there are no spike-sorting quality metrics.

**Trial curation rules**:
- Keep go and catch trials; exclude aborted and auto_rewarded, per the Decoder Task and consistent with the whitepaper which excludes aborted trials from performance metrics and states auto-rewarded trials should not be categorized as hit/miss.
- Exclude passive sessions (OPHYS_2 / OPHYS_5): verified they contain 0 licks and 0 rewards, so every outcome is miss/correct-reject by construction and trial outcome carries no behavioral information.

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| change vs repeat (random forest, 400 ms window, per imaging plane) | reported only graphically in Figure 6A; text says decoded equally well for all cell classes |
| hit vs miss (random forest) | Figure 6C, graphical only; higher for visual than timing sessions |
| false alarm decoding | very low (Figure S22) |

No numeric decoding accuracies appear in the text, so the paper gives qualitative targets: image-change decoding clearly above chance, hit/miss above chance but weaker.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Neural signal | SDK exposes dff_traces, events (detected events), filtered_events (events convolved with causal half-normal, scale 2/31 s, 20 steps) | events sparse (~0.3-1.4% of frames nonzero) | paper: we used the detected calcium events | Use **events**, summed within each 750 ms bin. Summing is the integral of event magnitude over the bin, avoids sparsity of instantaneous sampling, and needs no arbitrary smoothing kernel. |
| Which sessions | SDK exposes passive and active alike | passive sessions have 0 licks / 0 rewards | paper analyzed active behavior sessions | Exclude passive sessions. |
| Session definition | ophys_experiment_id = one imaging plane; ophys_session_id = one recording | Multiscope sessions have 3-7 planes with identical trials and timestamps agreeing within 0.07 s | whitepaper: data collected in a single continuous recording is defined as a session | **Merge all planes of an ophys_session_id into one decoder session**, concatenating neurons; each plane binned on its own timestamps. |
| Paper subset vs available data | - | 202 active experiments, 38 mice, both rigs, familiar+novel | paper restricted neural analysis to Multiscope + familiar + V1/LM (21/15/21 sessions) | The paper restriction served its question (cell classes vs strategy). For training a decoder we keep all active sessions from both rigs and all experience levels: more data, and the Decoder Task does not ask for the novelty/strategy contrast. Deliberate, documented difference. |
| Trial alignment | change_time from Trial.add_change_time | change_time coincides exactly (0.000000 s) with a flash onset on every go and catch trial | change / sham change is the event the animal reports | Align trials to change_time. |
| Pupil measure | SDK gives pupil_area, pupil_width, pupil_height | pupil_area is NaN exactly where likely_blink is True | paper does not decode pupil | Use **pupil_width as diameter** (area is an ellipse area, not a diameter; width correlates 0.986 with area). Interpolate across blinks within session before binning. |
| Omitted flashes | image_name == omitted | omitted flashes always flanked by the same image, never adjacent to a change | omissions are a continuation of the gray screen | Give omissions their own image-identity category rather than dropping or filling them, so no timepoint is fabricated. |

All three sources agree once the above are resolved. The measured catch fraction (12.5%) independently reproduces the whitepaper catch probability, and measured change-time statistics reproduce the stated design, validating that the trials table is being read as the pipeline wrote it.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Time base
- **Bin = one 750 ms image presentation interval**, exactly the unit the paper uses for behavioral analysis.
- Bin edges come from the actual start_time of consecutive flashes in the change_detection stimulus block, so bins track the real (slightly jittered, 0.73-0.77 s) cadence rather than an idealized grid.
- **Trial window**: 3 flashes before the change through 4 flashes after, inclusive = **8 bins**. Verified over all 51,992 go+catch trials that this window lies inside [start_time, stop_time] and never runs off the stimulus block.
- off_start = -2.25 s, off_end = +3.0 s relative to the change (nominal 750 ms spacing).
- Fixed T = 8 for every trial in every session satisfies the requirement that time bins be the same size for all trials and sessions.

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| events.events per cell + ophys_timestamps | neural | sum of detected event magnitudes in each 750 ms bin per neuron; Multiscope planes concatenated along neuron axis | .events, .ophys_timestamps | paper uses detected calcium events; each plane binned on its own timestamps |
| - | input | empty array shape (0, 8) | - | Decoder Task: No inputs for this task. Verified the decoder trains with dinput=0. |
| stimulus_presentations.image_name | output[0] image_identity | categorical index over a shared 17-value vocabulary (16 images of sets A and B, plus omitted); time-varying | .stimulus_presentations filtered to change_detection block | shared vocabulary keeps the shared readout consistent across sessions |
| stimulus_presentations.is_change | output[1] image_change | 1 in the bin whose flash is_change, else 0; time-varying | .stimulus_presentations | value 1 right after a change in image identity = the bin beginning at the change flash |
| running_speed.speed + timestamps | output[2] running_speed_bin | mean speed within each bin, discretized into 5 equal-percentile bins | .running_speed | quintile edges computed WITHIN each session (see Key Decision 7) |
| eye_tracking.pupil_width + timestamps, likely_blink | output[3] pupil_diameter_bin | blinks (NaN) linearly interpolated within session, mean within bin, then 5 equal-percentile bins | .eye_tracking | 3 experiments (3 sessions) have an empty eye table; those sessions are dropped since outputs cannot be NaN |
| trials.hit/miss/false_alarm/correct_reject | output[4] trial_outcome | static per trial, broadcast across the 8 bins | .trials | 4 classes; go -> hit/miss, catch -> false_alarm/correct_reject |
| metadata mouse_id | subjects, subject_idx | unique sorted mouse ids | .metadata | |
| metadata targeted_structure | brain_regions, brain_region_idx | VISp/VISl per neuron (per plane for Multiscope) | .metadata | Multiscope sessions legitimately span both areas |

### Key Decisions
1. **Bin 750 ms, 8 bins/trial aligned to change_time**: matches the paper image-presentation interval and guarantees identical T across trials and across both rigs (31 Hz and 11 Hz), which a fixed grid in seconds could not do cleanly.
2. **Neural = summed detected events per bin**: the paper explicitly uses detected calcium events; summing is the natural per-bin aggregate and keeps rigs with different frame rates comparable.
3. **Merge Multiscope planes into one session**: simultaneously recorded from one animal on one clock, so they form one population.
4. **Exclude passive sessions**: 0 licks / 0 rewards makes trial outcome degenerate.
5. **Exclude aborted and auto-rewarded trials**: required by the Decoder Task and consistent with the whitepaper analysis.
6. **All outputs time-varying**: image identity, change, running and pupil vary within a trial; trial outcome is constant per trial but broadcast across bins, per the instruction to make outputs time-varying if at all possible.
7. **Within-session quintile edges for running and pupil**: pupil width is measured in camera pixels whose scale depends on the per-session camera alignment, and running propensity differs enormously between animals, so a single global edge set would mostly encode which session/mouse a trial came from rather than the animal state. Computing the five equal-percentile edges within each session makes each class mean the same thing (relative level for this animal on this day) everywhere, and yields exactly ~20% per class both per session and overall (verified: 0.199-0.202).
8. **Drop the 3 sessions lacking eye tracking** rather than emit NaN, which verify_data_format rejects.

### Planned Sanity Checks
- [ ] change_time coincides exactly with a flash onset for every kept trial (verified globally already).
- [ ] The 8-bin window lies inside [start_time, stop_time] for every kept trial (verified globally already).
- [ ] Trial counts: kept go+catch == 51,992 minus trials of dropped no-eye-tracking sessions.
- [ ] image_change has exactly one 1 per trial, always at bin index 3.
- [ ] Image identity at bin 3 equals trials.change_image_name; at bin 2 equals trials.initial_image_name.
- [ ] Outcome distribution reproduces hit 15,936 / miss 29,541 / FA 931 / CR 5,584 before session drops.
- [ ] Running and pupil classes each hold about 20% of timepoints.
- [ ] Neural spot check: recompute binned events for a specific (session, trial, neuron) straight from the raw NWB and compare with np.allclose.
- [ ] Total neurons over kept sessions equals the sum of valid ROIs of kept experiments.


---

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` implements the Step 5 mapping. Structure:
- `get_session_table()` - intersects `ophys_experiment_table.csv` with the NWB files actually on disk, drops passive experiments, and groups experiments by `ophys_session_id` so that Multiscope planes become one session.
- `process_session()` - loads each plane with `BehaviorOphysExperiment.from_nwb_path`, takes the behaviour streams from the first plane (verified identical across planes), selects go+catch trials, finds each trial's change flash, bins every stream, and returns arrays.
- `binned_sum()` / `binned_mean_1d()` - cumulative-sum + `searchsorted` binning, so each stream is traversed once regardless of the number of bins.
- `quantile_bin()` - 5 equal-percentile classes.
- `plot_processing()` - the `--show-processing` figures.
- `main()` - pools sessions over workers, builds shared vocabularies, assembles the dict, pickles it.

Options: `--full` (default), `--sample` (1 single-plane + 1 Multiscope session, so both rig types are exercised), `--show-processing`, `--workers`.

Code inefficiencies identified:
- Naively looping over bins and re-scanning the event array per bin would be O(n_bins x n_frames). Replaced with one cumulative sum per plane plus `searchsorted` lookups.
- Loading each NWB more than once (e.g. once for behaviour and once for neural) would double the IO. Each plane is loaded exactly once and the behaviour streams are read only from the first plane.

Code speedups added:
- `multiprocessing.Pool` over sessions (sessions are independent).
- cumsum/searchsorted binning as above.
- float32 neural output to halve memory.

### Bug found and fixed during development
The first run asserted `chg[:, N_PRE] == 1` for every trial and failed. Investigation showed this was my error, not a data problem: `stimulus_presentations.is_change` marks only **real** image changes, and on **catch** trials the change_time marks a *sham* change where the image does not actually change. This is exactly the semantics the Decoder Task wants for `image_change` (1 only after a genuine change in image identity), so catch trials are legitimately all-zero. The assertion was rewritten to require `is_change == 1` at the change bin on go trials and `== 0` on catch trials, which now passes for every trial in every session.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

Command: `python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing`

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (775289198 single-plane; 951410079 Multiscope, 7 planes merged) |
| Neurons (total) | 177 (89 + 88) |
| Neurons / session | 89, 88 |
| Subjects | 2 |
| Trials (total) | 248 (39 + 209), 0 dropped |
| Trials / session | 39, 209 |
| T per trial | 8 bins, all trials |
| input | shape (0, 8) - no inputs, as specified |
| image_identity distribution | 8 images each 0.10-0.14, omitted 0.031 |
| image_change distribution | [0.892, 0.108] |
| running_speed distribution | [0.201, 0.200, 0.200, 0.200, 0.201] |
| pupil_diameter distribution | [0.201, 0.200, 0.200, 0.200, 0.201] |
| trial_outcome distribution | [hit 0.383, miss 0.484, FA 0.036, CR 0.097] |
| Brain regions | VISp 125, VISl 52 |

The low trial count of session 775289198 was checked against the raw trials table rather than assumed to be a bug: that session really has 1,078 aborted trials out of 1,117 (an unusually impulsive mouse) leaving only 39 go+catch, and it is the minimum over the whole dataset. My code dropped 0 of them. It also has 0 correct-reject trials, confirming that individual sessions can lack an outcome class.

### Processing Plots Review
`processing_775289198.png` and `processing_951410079.png` each show: raw detected events with flash windows and the change marked; raw running speed and pupil with the 8 trial bins shaded; the binned continuous signals; the binned neural matrix; the `image_change` output as an image (a single clean column at bin 3); image identity per bin; histograms of binned running/pupil with the quintile edges drawn on; and class-fraction bars against the 20% line. No temporal misalignment or discretisation anomaly is visible.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| cumsum + searchsorted binning | avoids O(n_bins x n_frames) rescans |
| single load per plane | ~2x IO |
| Pool over sessions (24 workers) | ~24x wall clock |

| Step | Time / Session | Estimated Total Time |
|---|---|---|
| single-plane session | 4.2 s (1 plane) | 202 planes total, ~2.6 s/plane => ~9 min serial |
| Multiscope session | 18.3 s (7 planes) | with 24 workers: **~1-2 min wall clock** |

Well under the 15 minute budget, so no further optimisation was needed.

### Format verification
`/app/verification_sample_out.txt`: **"Data format is valid, no errors or warnings."**

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample, 2 sessions / 248 trials)
Loss decreased monotonically 1.430 -> 1.034 over 200 epochs; test loss 1.327.

| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| image_identity | 0.6979 | 0.5928 | 0.1111 |
| image_change | 0.8128 | 0.7112 | 0.5000 |
| running_speed | 0.3996 | 0.2821 | 0.2000 |
| pupil_diameter | 0.4397 | 0.3074 | 0.2000 |
| trial_outcome | 0.5627 | 0.3428 | 0.2500 |

Every output is above chance on validation. Image identity is 5.3x chance and image change is well above chance, consistent with the paper's finding that image change is robustly decodable from these populations. With only 2 sessions this is expected to improve on the full dataset.


---

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 171 sessions, 29,168 neurons, 43,975 trials, 38 mice (conversion took 51 s with 24 workers)
- `conversion_full_out.txt`, `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Data (raw tables/NWB) | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Active (non-passive) ophys sessions | - | 174 | 171 (3 dropped: no eye tracking) | yes, accounted for |
| Total neurons | - | 29,444 active; 29,168 after dropping 3 sessions (276 cells) | 29,168 | **exact** |
| Subjects | 82 mice in full release (incl. behavior-only) | 38 mice with ophys files on disk | 38 | yes |
| Trials (go+catch, per session) | - | 44,892; minus 917 in dropped sessions = 43,975 | 43,975 | **exact** |
| Trials/session (mean) | - | 257 | 257 | exact |
| go / catch | catch ~12.5% (whitepaper) | 38,460 / 5,515 = 12.54% catch | 38,460 / 5,515 = 12.54% | **exact, and matches whitepaper** |
| hit / miss / FA / CR | - | 13,940 / 24,520 / 834 / 4,681 | 13,940 / 24,520 / 834 / 4,681 | **exact** |
| Hit rate among go | - | 0.3625 | 0.3625 | exact |
| FA rate among catch | - | 0.1512 | 0.1512 | exact |
| Brain regions | V1 and LM | VISp 29,282 / VISl 162 (active) | VISp 29,006 / VISl 162 | exact after the 276 dropped VISp cells |
| Image vocabulary | 8 images/session, 2 sets | 16 images + omitted | 17 categories | yes |
| image_change distribution | - | 1 change bin per go trial | [0.891, 0.109] | consistent (1 of 8 bins on 38,460/43,975 trials) |
| running/pupil distribution | - | - | ~[0.200]x5 each | exactly equal-percentile |
| Time bin | 750 ms image presentation interval (paper) | flash spacing 0.73-0.77 s | 750 ms | yes |

### Trial-count reconciliation (an apparent discrepancy that was investigated)
My Step 2 census reported 51,992 go+catch trials, but the conversion produced 43,975. This is **not** data loss: 51,992 summed go+catch **per experiment**, which double-counts the shared behaviour trials of Multiscope sessions (a 7-plane session has one set of 209 trials counted 7 times). Deduplicating by `ophys_session_id` gives 44,892, minus the 917 trials of the 3 dropped sessions = 43,975 exactly. A per-session comparison found **0 mismatches** in either trial count or neuron count across all 171 sessions.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`verification_full_out.txt` reports **no errors**. It reports 2,370 warnings, all of the form "Session S, trial T: all neural data is zero".

*Cause*: detected calcium events are sparse (94-97% of 750 ms bins are zero even in large populations), so a trial of 8 bins in a session with very few valid ROIs can legitimately contain no event at all. The warnings are concentrated entirely in low-yield sessions:

| neurons/session | sessions | trials | mean all-zero trial fraction |
|---|---|---|---|
| <=10 | 11 | 2,978 | 0.176 |
| 11-20 | 27 | 6,321 | 0.161 |
| 21-50 | 27 | 6,768 | 0.113 |
| 51-100 | 20 | 4,898 | 0.005 |
| >100 | 86 | 23,010 | 0.0002 |

*Why this warning is not "fixed"*: it reflects a true property of the recordings, not a conversion error. The independent sanity check below recomputed these same trials from the raw NWB and got identical all-zero matrices. Discarding them would mean discarding valid ROI recordings purely because they are quiet; see Step 12 for the decision not to impose a neuron-count filter.

### Check 2: Constructed sanity checks (`/app/cache/sanity_checks.py`)
These reload the **raw NWB files** and recompute everything from scratch, without importing `convert_data.py`. Three sessions were chosen to span the failure modes: a large single-plane session (808340530, 302 neurons), a 7-plane Multiscope session (951410079, 88 neurons), and a tiny session (835795999, 13 neurons).

| Check | Method | Result |
|---|---|---|
| **Neural** | For randomly chosen trials, re-derive the change flash, form the 8 windows, and sum `events.events` per neuron with a naive boolean mask (not cumsum) | `np.allclose` **True**, max abs diff **0** for all 9 trials tested |
| **Output: image identity** | Re-index `stimulus_presentations.image_name` at the 8 flash positions | exact match on all trials tested; additionally `image[bin 3] == trials.change_image_name` asserted |
| **Output: image change** | Re-index `stimulus_presentations.is_change` | exact match |
| **Output: trial outcome** | Read hit/miss/false_alarm/correct_reject from the raw trials row | exact match |
| **Output: pupil** | Re-interpolate blinks, re-bin, re-quantile | exact match |
| **Output: running** | Same | matched in 2 of 3 sessions; see below |

The single running mismatch (1 bin of 2,136 in session 808340530) was investigated rather than dismissed. The session has **0 NaNs and 0 empty bins**; the offending bin's mean fell *exactly* on the 60th-percentile edge, and my checker's edges differed from the converter's in the last bits. A direct comparison of the converter's cumsum-based mean against a naive `.mean()` over the whole session gives max abs diff **1.3e-10**, `allclose(rtol=1e-9)` **True**, and **0** class differences when edges are computed consistently. So this was an artifact of the checking script, not of the conversion; no change to `convert_data.py` was required.

### Check 2b: Additional external validation (`/app/cache/check_initial_image.py`)
The paper states that "image changes as well as the image immediately before the change were
not omitted". In the converted data, bin 2 (the flash immediately preceding the change) is
`omitted` in **0 of 43,975 trials**, exactly reproducing that statement -- an independent
external check that the trial window is aligned to the right flash. For all sessions tested,
bin 2 also equals `trials.initial_image_name` for **100%** of trials. (The `!= omitted` guard
in the converter's assertion is therefore provably a no-op on this dataset; it is retained
only as defensive coding.)

### Check 3: Reference code comparison
| Stage | Reference | My script | Same? |
|---|---|---|---|
| (a) loading | `BehaviorProjectCloudApi.get_behavior_ophys_experiment` -> `BehaviorOphysExperiment.from_nwb_path(path)` | identical call on the local NWB path | yes |
| (b) neuron filtering | `from_nwb(..., exclude_invalid_rois=True)` default | same default used; `valid_roi` verified True for 100% of returned cells | yes |
| (c) temporal alignment | trials aligned to `change_time` (`Trial.add_change_time`, from the change frame on go and the sham frame on catch) | same column used; verified to coincide exactly with a flash onset for all 51,992 candidate trials | yes |
| (d) binning | paper: "the 750 ms interval beginning with each image presentation" | bins are the actual consecutive flash `start_time`s, width 750 ms | yes |
| (e) input construction | n/a | empty by specification | n/a |
| (f) output construction | `stimulus_presentations` filtered to the `change_detection` block (SDK warning + both tutorials); `trials` outcome flags; `running_speed.speed`; `eye_tracking` | same columns and same block filter | yes |

*Deliberate differences*, with reasoning:
- **Neural signal is the raw `events`, not `filtered_events`**: the paper says "detected calcium events"; `filtered_events` applies an extra causal half-normal smoothing that would blur activity across my 750 ms bin boundaries.
- **All active sessions kept, not only Multiscope/familiar/V1-LM**: the paper's restriction served its cell-class-vs-strategy question. Keeping all active sessions maximises training data for a decoder and the Decoder Task does not request the novelty/strategy contrast.
- **Multiscope planes merged per session**: required to give the decoder a simultaneously-recorded population.

### Check 4: Key statistics comparison
See the Step 9 table: every count (sessions, neurons, trials, go/catch, all four outcomes, regions) matches the raw data exactly, and the catch fraction independently reproduces the whitepaper's ~12.5%.

### Check 5: Edge cases (`/app/cache/edge_cases.py`)
| Check | Result |
|---|---|
| Every trial: neural `(N,8)` float32 finite, input `(0,8)`, output `(5,8)` int64, `len(brain_region_idx)==N` | OK |
| Output values within their vocabularies | image [0,16]/17, change [0,1]/2, running [0,4]/5, pupil [0,4]/5, outcome [0,3]/4 |
| Change structure | 38,460 trials with exactly one 1, **all at bin 3**; 5,515 catch trials all-zero; 0 anomalies |
| Outcome constant within trial | 0 violations |
| Catch trials: image at change bin == preceding image | **True for all 5,515** (a sham change really does not change the image) |
| `subject_idx` / `brain_region_idx` in range | OK |
| Sessions with <2 trials | none |
| Change bin ever "omitted" | 0 (matches "changes were never omitted") |

### Issues Found and Resolved
1. **Catch-trial assertion failure** (Step 6): I had assumed every aligned trial has `is_change==1`. Investigation showed `is_change` marks only real changes, and catch trials are sham changes. This is the correct semantics for the `image_change` output. Assertion rewritten to be go/catch aware; now passes for all 43,975 trials.
2. **Apparent loss of 8,017 trials**: traced to per-experiment vs per-session counting of Multiscope sessions. No trials were lost (0 per-session mismatches).
3. **Apparent running-speed mismatch**: traced to a floating-point tie exactly on a quantile edge in my checking script, not in the converter.
4. **Apparent VISl under-count (162 neurons)**: verified against the raw table -- the 17 active VISl experiments genuinely contain only 162 valid ROIs. Not a bug.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples`

### Training Progress
- Loss decreasing: **Yes**, monotonically 1.430 -> 1.253 over 200 epochs; test loss 1.356.

### Decoder Results (Full: 171 sessions, 43,975 trials, 29,168 neurons)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Val / Chance |
|--------|-------------|--------|--------|--------|
| image_identity | 0.5338 | **0.4903** | 0.0588 | 8.3x |
| image_change | 0.7269 | **0.6581** | 0.5000 | 1.32x |
| running_speed | 0.3854 | **0.3161** | 0.2000 | 1.58x |
| pupil_diameter | 0.3683 | **0.2825** | 0.2000 | 1.41x |
| trial_outcome | 0.5183 | **0.3182** | 0.2500 | 1.27x |

All five outputs are above chance on validation.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
| Variable | Val Acc | Chance | Ratio | Assessment |
|---|---|---|---|---|
| image_identity | 0.490 | 0.059 | 8.3x | strong; V1/LM is visual cortex so image identity should be, and is, the most decodable variable |
| image_change | 0.658 | 0.500 | 1.32x | binary; a 0.66 balanced accuracy on a 1-in-8-bins event is a solid change signal |
| running_speed | 0.316 | 0.200 | 1.58x | above 1.5x |
| pupil_diameter | 0.283 | 0.200 | 1.41x | below 1.5x, investigated below |
| trial_outcome | 0.318 | 0.250 | 1.27x | below 1.5x, investigated below |

None is below chance, so there is no sign-flip or label-misalignment bug.

**Investigation of the weaker outputs.** I measured, per session, a change-signal index (population activity at change bins vs other bins, in SD units) and the correlation of population activity with the running/pupil class, grouped by session neuron count:

| neurons/session | sessions | trials | all-zero trials | change signal | r(act, running) | r(act, pupil) |
|---|---|---|---|---|---|---|
| <=10 | 11 | 2,978 | 0.176 | 0.045 | 0.169 | 0.149 |
| 11-20 | 27 | 6,321 | 0.161 | 0.115 | 0.137 | 0.093 |
| 21-50 | 27 | 6,768 | 0.113 | 0.121 | 0.115 | 0.123 |
| 51-100 | 20 | 4,898 | 0.006 | 0.468 | 0.108 | 0.129 |
| 101-200 | 36 | 9,442 | 0.000 | 0.905 | 0.059 | 0.141 |
| >200 | 50 | 13,568 | 0.000 | 0.635 | 0.081 | 0.096 |

Sessions with <=50 neurons (65 of 171 sessions, ~36% of trials) carry almost no population signal and contribute most of the 2,370 all-zero trials. Restricting to N>=20 would keep 98.5% of neurons while dropping ~19% of trials, and would very likely raise every reported accuracy.

**Decision: I did not apply a neuron-count filter.** No reference text specifies a minimum-neuron criterion for this dataset; the documented curation is the ROI-validity filter, which is already applied via `exclude_invalid_rois=True`. These are genuine recordings of valid ROIs, and the low accuracy in small populations is a real scientific property (few cells carry little decodable information), not an artifact. Removing ~20% of real trials *because they decode poorly* would optimise the reported score rather than the fidelity of the conversion, and would misrepresent the dataset. The evidence is documented here so the trade-off is explicit and the filter can be applied downstream if a user wants it.

Two further structural reasons the weaker outputs are bounded:
- **trial_outcome** is constant within a trial, so the decoder must infer it from 8 bins that include 3 pre-change bins carrying no outcome information at all; and the classes are very unbalanced (FA is 1.9% of trials). 14 sessions do not even contain all four outcome classes (12 have 3, 2 have 2) -- a real property of behaviour.
- **pupil_diameter** is an arousal variable only indirectly reflected in V1/LM activity, and is quantised within session, so the classes are by construction equally likely and cannot be won by a prior.

### Check 2: Accuracy comparison to papers
| Variable | My validation accuracy | Paper's reported value |
|---|---|---|
| image change (change vs repeat) | 0.658 balanced | Figure 6A, **graphical only** -- no numeric value printed in the text; text states changes "could be decoded equally well for all cell classes" |
| hit vs miss | not decoded separately; `trial_outcome` (4-class) = 0.318 | Figure 6C, graphical only; higher for visual than timing sessions |
| false alarm | part of `trial_outcome`; FA is the rarest class | text: "false alarm decoding performance was very low" -- **consistent with my result** |

The papers report no numeric decoding accuracies (all in figures), so no quantitative discrepancy can be established. The qualitative pattern matches: image/change information is robustly decodable, and false-alarm information is weak. Note the paper's decoder is also not comparable in setup -- it used a random forest on the first 400 ms after each presentation, per imaging plane, sampling n neurons, whereas this task uses a shared linear decoder on 100 PCs over 750 ms bins across all sessions.

### Check 3: Train vs validation gap
| Output | Train | Val | Ratio |
|---|---|---|---|
| image_identity | 0.534 | 0.490 | 1.09 |
| image_change | 0.727 | 0.658 | 1.10 |
| running_speed | 0.385 | 0.316 | 1.22 |
| pupil_diameter | 0.368 | 0.283 | 1.30 |
| trial_outcome | 0.518 | 0.318 | 1.63 |

All but `trial_outcome` are well under 1.5x. `trial_outcome` at 1.63x reflects mild overfitting, expected for a per-trial label with only ~44k trials, strong class imbalance, and a per-session projection with many parameters for small sessions. It is not data leakage: the label is constant within a trial and the train/validation split is by trial, so no bin of a validation trial is ever seen in training.

### Issues Found and Resolved
- All issues listed in Step 10 were resolved there; re-running the full check suite after the final conversion showed no remaining failures.
- The remaining warnings (all-zero trials) are a documented property of the data, not a defect.


---

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created (dataset description, how to load, output format spec, key statistics, curation summary, decoder performance)
- [x] `cache/` folder created, with `cache/README_CACHE.md` documenting every investigation script
- [x] All files organized; all investigation/verification scripts live in `/app/cache/`

### Deliverables
| File | Purpose |
|---|---|
| `CONVERSION_NOTES.md` | This file: all decisions, checks and validation results |
| `convert_data.py` | The conversion script (`--full`, `--sample`, `--show-processing`) |
| `converted_data.pkl` | Full converted dataset (171 sessions, 29,168 neurons, 43,975 trials) |
| `sample_data.pkl` | 2-session sample |
| `README.md` | User-facing documentation |
| `conversion_sample_out.txt`, `verification_sample_out.txt`, `train_decoder_sample_out.txt` | Sample run logs |
| `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt` | Full run logs |
| `processing_775289198.png`, `processing_951410079.png` | Per-step processing figures |
| `sample_trials.png`, `predictions.png` | Decoder sample/prediction plots |
| `cache/` | Investigation scripts + extracted paper text |
