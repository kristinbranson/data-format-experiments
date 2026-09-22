# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory Visual Behavior 2P (provided local export)
- **Date started**: 2026-09-19
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

Environment check: Python 3.13.15, NumPy 2.4.4, PyTorch 2.6.0+cu124; imports succeeded and CUDA is available. The required notes-file checkpoint (`ls -la /app/CONVERSION_NOTES.md`) passed.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `BehaviorOphysExperiment.from_nwb` | `allensdk/brain_observatory/behavior/behavior_ophys_experiment.py` | LOADING | Builds a synchronized experiment object from NWB, including behavior, ophys timestamps, cells, metadata, motion, and eye tracking. |
| `BehaviorOphysExperiment.ophys_timestamps` | same | LOADING | Returns microscope frame timestamps; these are the requested master temporal grid. |
| `BehaviorOphysExperiment.dff_traces` | same | LOADING | Returns precomputed baseline-normalized dF/F traces, indexed by cell. No new dF/F computation is needed. |
| `BehaviorOphysExperiment.events` | same | PROCESSING | Returns L0-detected event traces plus causally filtered events; available as an alternative neural representation. |
| `CellSpecimens.__init__` | `.../cell_specimens/cell_specimens.py` | CURATION | With default `exclude_invalid_rois=True`, keeps only `valid_roi`, then filters/reorders all traces consistently. |
| `BehaviorSession.trials` / `Trials` / `Trial` | `behavior_session.py`; `.../trials/{trials,trial}.py` | LOADING | Provides trial bounds and mutually exclusive go/catch/aborted/auto-rewarded flags plus hit/miss/false-alarm/correct-reject outcomes. |
| `BehaviorSession.stimulus_presentations` | `behavior_session.py` | LOADING | Provides image identity, exact start/end times, omitted and change flags, activity block, and trial mapping. |
| `is_change_event` | `stimulus_processing.py` | PROCESSING | Defines change as first non-omitted presentation of a new image (ignores omitted flashes and the session's first stimulus). |
| `compute_trials_id_for_stimulus` | `stimulus_processing.py` | PROCESSING | Assigns stimulus flashes to trials based on stimulus start and trial bounds. |
| `get_running_df` | `.../running_speed/running_processing.py` | PROCESSING | Corrects encoder wraps/outliers and applies the released low-pass running-speed filter (implementation is 3rd-order Butterworth, 4 Hz cutoff at 60 Hz despite legacy 10 Hz wording). |
| `EyeTrackingTable.from_nwb` | `.../eye_tracking/eye_tracking_table.py` | PROCESSING | Loads pupil ellipse measures, recomputes blink mask with z=3 and 2-frame dilation, and masks blink-contaminated samples. |
| `VisualBehaviorOphysProjectCache.get_ophys_experiment_table` | `.../behavior_project_cache/behavior_project_cache.py` | LOADING | Provides released experiment metadata used for project/session selection. |

### Notes
- The SDK documentation recommends NWB access through `BehaviorOphysExperiment`; provided data may already be a derived local export, which will be checked in Step 2.
- dF/F is already calculated, neuropil-subtracted/demixed upstream, baseline normalized, and stored in NWB. Recomputing it would diverge from the release processing.
- The default SDK cell path excludes invalid/non-cell ROIs. This is the applicable neuron-quality rule for optical physiology; electrophysiology unit filtering is not relevant.
- Running speed is the released filtered `speed` stream in cm/s, sampled behaviorally at about 60 Hz, and should be interpolated/aligned to ophys frames rather than rederived.
- Pupil output should use a diameter measure from the pupil ellipse (the precise available source field is deferred to Step 2); blink-masked missing samples require explicit handling.
- Stimulus output should be formed from the active change-detection stimulus presentations. Omitted flashes are gray periods and should not be mislabeled as image presentations.
- Trial outcomes are `hit`, `miss`, `false_alarm`, and `correct_reject` after explicitly excluding `aborted` and `auto_rewarded`, exactly as requested.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`: 284 NWB 2.0 experiment files (246.79 GiB). Each file is one imaging plane/experiment.
- `/app/data/visual-behavior-ophys-1.1.0/project_metadata/`: CSV tables for 4,782 behavior sessions, 1,936 released experiments, 703 ophys sessions, and 133,066 experiment-cell rows in the complete v1.1.0 release metadata.
- Local NWBs comprise all 239 experiments with exact `project_code == "VisualBehavior"` plus 45 experiments from one `VisualBehaviorMultiscope` mouse. The latter project is outside the explicitly named task and will not be converted.
- Of 239 local `VisualBehavior` experiments, 168 are active behavior (`OPHYS_1`, `OPHYS_3`, `OPHYS_4`, or `OPHYS_6`) and 71 are passive replay (`OPHYS_2` or `OPHYS_5`). Passive replay trials carry labels copied from an earlier active stimulus sequence rather than contemporaneous behavioral outcomes, so the active 168-session cohort is the native-data cohort relevant to all requested outputs.
- NWB native streams: precomputed dF/F and L0 events `(ophys_frames, valid_cells)` with common ophys timestamps; filtered/unfiltered running speed with timestamps; pupil ellipse `width`, `height`, `area`, center, angle, blink-masked values, and timestamps; stimulus-presentation dynamic tables with image/change/omission/timing/trial IDs; and trial dynamic tables with bounds, type, outcome, response, reward, and image fields.
- Sample native dtypes are float64 for traces/timestamps/continuous behavior, bool for trial flags, int64 for IDs/frames, and variable strings for image names. The conversion will cast dense arrays to compact dtypes after exact alignment.
- Active `VisualBehavior` sessions contain 16 global natural-image identities (`im000`, `im031`, `im035`, `im045`, `im054`, `im061`, `im062`, `im063`, `im065`, `im066`, `im069`, `im073`, `im075`, `im077`, `im085`, `im106`) split into 8-image A/B session sets; `omitted` represents gray, not an image identity.
- Ophys frame interval is extremely stable: median 0.032310 s overall (session medians 0.032300–0.032320 s; about 30.95 Hz). Session traces contain 139,884–149,472 frames.
- Three active sessions have no eye-tracking samples; all 165 sessions with eye tracking contain blink-masked NaNs (median missing fraction 2.94%, maximum 29.64%). This must be handled explicitly.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 29,097 valid session-neuron recordings; 13,676 unique longitudinal cell specimen IDs |
| Neurons / session | mean 173.20; median 117.5; range 6–666 |
| Subjects | 37 in active `VisualBehavior` |
| Sessions / subject | mean 4.54; median 4; range 2–9 |
| Sessions | 168 active experiments/sessions (single-plane project); 239 including 71 passive replays |
| Trials (total) | 114,292 native trial rows; 43,387 eligible go/catch rows after the specified aborted/auto-rewarded exclusion |
| Trials / session | eligible mean 258.26; median 264; range 39–409 |
| Eligible go / catch | 37,948 / 5,439 |
| Eligible outcomes | hit 13,823; miss 24,125; false alarm 825; correct reject 4,614; every eligible trial has exactly one outcome |
| Brain regions | VISp for all 168 sessions |
| Session types | OPHYS_1 A: 48; OPHYS_3 A: 40; OPHYS_4 B: 39; OPHYS_6 B: 41 |
| Genetic classes | Slc17a7: 107 sessions; Vip: 33; Sst: 28 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (release total) | 34,619 unique cells in whitepaper v1.0; later final release documentation says 50,476 | Whitepaper Table 1 totals 34,619 (v1.0); the SDK v1.1 documentation describes the later expanded release. | 
| Neurons (`VisualBehavior`, v1.0) | 16,122 unique cells across transgenic rows (5,915 + 9,743 + 84 + 380) | Whitepaper Table 1. This is a longitudinal unique-cell statistic, not a sum of session-cell recordings. |
| Neurons / session | Not reported for the requested exact cohort | Whitepaper reports aggregate unique cells and sessions, which cannot be divided to obtain session ROI count because cells repeat longitudinally. |
| Subjects | 82 release-wide in whitepaper v1.0; exact `VisualBehavior` table rows sum to 35; final v1.1 docs: 107 release-wide | Whitepaper Table 1; SDK release documentation. |
| Sessions / subject | Containers contain 3–11 sessions | Whitepaper data structure/terminology. |
| Trials (total) | Not reported | Native NWB must be authoritative. |
| Trials / session | Task runs 60 min; counts not reported | Whitepaper trial structure. |
| Neural data time bin | Single plane 31 Hz (~32.26 ms); multiplane 11 Hz | “Two-photon movies … 31 Hz for single plane … 11 Hz for each plane in multi-plane experiments.” |
| Behavior data time bin | Behavior 30 Hz; eye tracking 30 Hz | Whitepaper data synchronization section. |
| Reward rate | No cohort-wide fraction reported; SDK engagement threshold is 2 rewards/min | Whitepaper behavior metrics. | 
| Trial type prior | Go 87.5%, catch 12.5% after matrix sampling | Whitepaper Figure 4 / trial structure. | 
| Stimulus timing | 250 ms image + 500 ms gray; 5% non-change omissions | Whitepaper/paper task descriptions. |
| Change timing | Truncated exponential 2.25–8.25 s; mean about 4.2 s after flash alignment | Whitepaper trial structure. |
| Response window | 150–750 ms after (sham) change before display-lag correction | Whitepaper behavior metrics. |
| Paper behavior cohort | 376 imaging sessions, 82 mice | Paper abstract/results. |
| Paper neural cohort | 8,619 excitatory (21 sessions/9 mice), 470 Sst (15/6), 1,239 Vip (21/9) | Paper results; this is a familiar, multiplane analysis subset and not the requested project cohort. |


### Processing Details
- All clocks were sampled by one synchronization board; released timestamps are already brought into a common clock. The task explicitly requires ophys timestamps as the target grid.
- Whitepaper single-plane imaging is nominally 31 Hz; the measured NWB interval of 32.31 ms is consistent. Behavior/eye are nominally 30 Hz and therefore require timestamp-based interpolation rather than sample-index matching.
- The paper used **unfiltered L0-detected calcium event magnitude traces** for all neural analysis and decoding, linearly interpolated event-triggered responses to a consistent 30 Hz grid. Since this task explicitly says align to ophys timestamps, retain the event samples on their native ophys grid instead of re-interpolating neural data to 30 Hz; interpolate behavior to those ophys timestamps.
- Paper decoders used the first 400 ms after image presentation and random forests. Figure 6 reports approximate change-vs-repeat accuracy of 51–65% and hit-vs-miss accuracy of 52–74%, depending on cell class, strategy, and neuron count (50% chance). These are related benchmarks, not direct expectations for the five requested time-varying/per-trial labels.
- Pupil processing fits ellipses with DeepLabCut. The whitepaper defines pupil diameter as the ellipse major axis and the released blink-masked pupil area as the area of a circle whose diameter is that major axis. Thus diameter can be recovered exactly as `2*sqrt(pupil_area/pi)`; likely blinks/outlier fits are NaN.
- Running speed follows encoder unwrapping, wrap clipping in +/-0.25 s, z-score >=10 rejection, conversion to cm/s, and low-pass filtering. Use the released processed stream.
- The paper assigns events to 750-ms image-presentation intervals and uses detected events. For this task, exact continuous trial segmentation and ophys-frame output labels supersede the paper’s presentation-level aggregation.

### Curation Steps

**Neuron curation rules**:
Release QC excludes bad sessions/planes (sync failures, >10 µm z drift, task performance failure, hardware/software issues, stress, excessive motion, saturation/bleaching, interictal activity). ROI classification excludes unions, duplicates, edge/motion-border objects, dendrites, and ROIs too small/narrow/dim; demixing failures remove about 1% more ROIs. NWB dF/F/event matrices already contain the valid released ROIs, so no ad hoc signal-based cell filter should be added. The reference paper uses detected FastLZero calcium events (31-Hz event threshold factor 2.0).

**Trial curation rules**:
The task yields go/catch trials and hit/miss/false-alarm/correct-reject outcomes. Premature licks abort/restart trials; free/auto-reward trials bias choice. Follow the explicit decoder specification by keeping only rows with `go OR catch`, excluding `aborted` and `auto_rewarded`, and requiring exactly one of the four valid outcomes. Exclude passive replay sessions: the paper explicitly states passive viewing “was not analyzed,” and these sessions cannot provide contemporaneous trial outcomes.

### Decoders Trained
| Decoded variable | Accuracy |
| Image change vs repeat (paper RF, first 400 ms) | approximately 51–65% across plotted conditions; chance 50% |
| Hit vs miss (paper RF, first 400 ms) | approximately 52–74% across plotted conditions; chance 50% |
| False alarm | described as “very low”; no numeric value in main paper |
| Image identity, running quintile, pupil quintile | not reported in supplied references |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Release size | Local SDK docs describe final 107-mouse/703-session/50,476-cell release | Metadata are v1.1.0; exact `VisualBehavior` has 239 sessions/37 mice | v1.0 whitepaper table has 82/551/34,619 release-wide and 225 sessions/35 mice for `VisualBehavior` | Version expansion, not data loss. Use exact local v1.1.0 metadata and report both versions explicitly. |
| Session cohort | SDK exposes active and passive and copies trial mapping into replay stimulus tables | 168 active + 71 passive exact-project sessions; passive files still contain replay-associated trial rows | Paper says passive viewing “was not analyzed”; behavioral outcomes require active task performance | Keep 168 active sessions initially; exclude three more only because pupil output is entirely absent, leaving 165. |
| Neural representation | SDK exposes both dF/F and L0 `events`, on identical ophys timestamps | Representative files: shapes and timestamps match exactly | Paper neural analyses and RF decoding use detected event magnitudes | Use raw/unfiltered `event_detection/data`, not `filtered_events` visualization and not recomputed dF/F. |
| ROI filtering | Default `exclude_invalid_rois=True`; trace tables are reordered to valid ROI table | NWB cell table length equals event columns and every stored `valid_roi=True` | Whitepaper describes classifier/motion/duplicate/union filtering | Accept NWB-valid cells; add no second filter. |
| Time grid | SDK keeps per-stream timestamps | Ophys median dt 32.31 ms; behavior and eye have distinct grids | Whitepaper: single-plane 31 Hz, behavior/eye 30 Hz; paper often interpolates to 30 Hz | Task specifically requests ophys timestamps: keep neural native; interpolate behavior to each selected ophys frame. |
| Running filter | SDK prose/whitepaper call it 10-Hz low-pass; current code constructs 3rd-order Butterworth with `Wn=4, fs=60` | NWB contains released processed `speed` and `speed_unfiltered` | Whitepaper says use processed low-pass stream | Do not attempt to resolve/reapply legacy cutoff wording; load released `processing/running/speed`. |
| Pupil diameter | SDK computes circular area as `pi*max(width,height)^2`; stored ellipse width/height are radii | Direct checks match that formula; blink samples are NaN | Whitepaper says ellipse major axis reflects diameter and area is circularized from it | Diameter is `2*sqrt(pupil_area/pi)` (equivalently twice major radius), retaining blink masking before interpolation. |
| Trial proportions/timing | Trial flags define go/catch and outcomes | Eligible cohort: go 87.464%, catch 12.536%, mean change-start 4.230 s | Expected 87.5%/12.5%, mean about 4.2 s | Excellent agreement; filtering logic is correct. |
| Trial outcomes | SDK makes hit/miss/FA/CR mutually exclusive after abort/auto handling | All 43,387 eligible rows have exactly one outcome; hit rate 36.43%, FA rate 15.17% | Whitepaper definitions match | Map the four flags directly to outcome classes. |
| Paper subset vs task | Project metadata distinguish exact project codes and session types | Requested project includes GCaMP6f/6s, familiar/novel, single-plane VISp | Paper neural figures use familiar GCaMP6f multiplane VISp/LM subset | Decoder task’s named project and required outputs take priority; use all eligible exact-project active sessions rather than reproduce the paper’s narrower scientific subset. |

Final consistent understanding: use exact-project (`VisualBehavior`) active-behavior NWBs; read already-QC'd event traces and cells; select requested trial rows; slice by native trial bounds on ophys timestamps; construct stimulus labels from active natural-image presentation intervals; timestamp-interpolate processed running and blink-masked pupil diameter; discretize behavior into within-session quintiles. Three sessions (`795953296`, `806456687`, `833631914`) have no eye data at all and cannot supply a required output without fabrication, so they will be excluded. All 37 mice remain represented.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/event_detection/data` | `neural[session][trial]` | Slice rows whose event timestamps satisfy `trial_start <= t < trial_stop`, transpose time-by-cell to cell-by-time, cast float32 | `BehaviorOphysExperiment.events`; `Events.from_nwb` | Unfiltered L0 event magnitude, as used by paper decoding. |
| none | `input[session][trial]` | Empty float32 array `(0, n_time)` | N/A | Decoder task explicitly specifies no inputs; `input_names=[]`. |
| natural-image presentation `image_name`, `start_time`, `stop_time`, `omitted`, `active` | output row 0: `image_identity` | Initialize gray (0); during each active, non-omitted `[start, stop)` interval assign one of 16 globally sorted image codes 1–16 | `BehaviorSession.stimulus_presentations` | Gray includes normal inter-stimulus gray and omitted flashes. |
| presentation `is_change` | output row 1: `image_change` | Value 1 for ophys frames during the changed-image presentation interval, else 0 | `is_change_event` | This represents the immediate post-identity-change image flash (~250 ms); catch sham changes remain 0 because identity did not change. |
| processed running `speed`, `timestamps` | output row 2: `running_speed_quintile` | Linear timestamp interpolation to selected ophys frames; session-specific 20/40/60/80th-percentile edges across all eligible trial frames; codes 0–4 | `BehaviorSession.running_speed`; `get_running_df` upstream | Use released filtered cm/s stream; negative filter excursions are retained before rank binning. |
| blink-masked pupil `area`, `timestamps` | output row 3: `pupil_diameter_quintile` | `diameter=2*sqrt(area/pi)`; discard nonfinite source points, linear timestamp interpolation to ophys frames, then session-specific quintiles across eligible frames | `compute_circular_area`; `filter_on_blinks`; `EyeTrackingTable.from_nwb` | Three sessions with no source eye samples are excluded. Interpolation is necessary because every remaining session has short masked blink/outlier gaps. |
| trial flags `hit`, `miss`, `false_alarm`, `correct_reject` | output row 4: `trial_outcome` | Map mutually exclusive flag to 0/1/2/3; repeat the static code over the trial’s time axis | `Trial._get_trial_data`; `BehaviorSession.trials` | Repetition permits static and time-varying outputs in one rectangular `(5,time)` array. |
| `/general/subject/subject_id` / metadata `mouse_id` | `subjects`, `subject_idx` | String mouse IDs, sorted globally; session indices point into list | `BehaviorOphysExperiment.metadata` | Direct NWB and CSV IDs must agree. |
| metadata `targeted_structure` | `brain_regions`, `brain_region_idx` | Global `['VISp']`; zero int array per session of length n_cells | `BehaviorOphysExperiment.metadata` | All exact-project active sessions are VISp. |
| experiment/trial metadata | `metadata.session_info` | Store IDs, mouse, session type, experience, image set, cell/trial counts, raw trial IDs, ophys median dt, and behavior quantile edges | metadata tables / NWB trials | Supports reproducibility and raw spot checks. |

Other available variables not mapped because they are outside the requested decoder outputs: dF/F, corrected/demixed/neuropil fluorescence, filtered visualization events, lick/reward times and volumes, response latency, eye position/angle, motion correction, projections/ROI masks, image novelty, omission identity, and natural movie/spontaneous blocks. They remain in source NWBs.

### Key Decisions
1. **Cohort**: Select metadata rows with exact `project_code == 'VisualBehavior'`, `passive == False`, an existing local NWB, and usable pupil data. This is 165 sessions, 37 subjects, 28,821 valid session-neurons (13,627 unique longitudinal cells), and 42,470 eligible trials. Excluding the three eye-absent experiments removes 276 session-neurons and 917 trials but avoids fabricated required labels.
2. **Trials**: Keep `(go OR catch) AND NOT aborted AND NOT auto_rewarded`; verify exactly one valid outcome. Preserve native SDK trial boundaries and use half-open `[start_time, stop_time)` slices, preventing shared boundary frames.
3. **Master clock**: The event trace’s ophys timestamps are authoritative. No index-based alignment across streams is permitted. Native single-plane dt is effectively common (median 32.31 ms, max session-median deviation 0.01 ms), satisfying common time-bin size while preserving the explicitly requested ophys alignment.
4. **Neural activity**: Use raw L0 event magnitudes because this is the paper’s neural/decoder representation and already incorporates the whitepaper’s movie correction, segmentation, demixing, neuropil subtraction, dF/F, and event-detection pipeline.
5. **Image classes**: Define global codes `['gray', 'im000', 'im031', 'im035', 'im045', 'im054', 'im061', 'im062', 'im063', 'im065', 'im066', 'im069', 'im073', 'im075', 'im077', 'im085', 'im106']`. A session uses gray plus its 8-image set; global coding keeps meanings constant across sessions.
6. **Change pulse**: Label every ophys sample falling in the changed image’s on-screen interval. A one-frame impulse would underrepresent the defined changed presentation and be less consistent with the presentation table/paper’s post-presentation decoder window.
7. **Behavior quintiles**: Compute within session and only over frames actually retained in eligible trials. This yields five percentile-defined categories in the converted analysis population and avoids treating session-specific pupil camera scale as biologically meaningful. Edges are stored in metadata.
8. **Missing pupil**: Use the released blink mask; interpolate only from finite blink-filtered diameter values, including endpoint hold through `np.interp`. Do not use `area_raw`, which would reintroduce invalid blink fits. Entirely missing eye sessions are excluded.
9. **Data types**: Neural/input float32; categorical output int16; indices integer. Expected neural payload is ~7.71 GiB before pickle overhead, so process one session at a time and avoid retaining full raw session matrices after slicing.
10. **Output categories**: `image_change=['no_change','change']`; running/pupil=`['Q1 (lowest)','Q2','Q3','Q4','Q5 (highest)']`; outcomes=`['hit','miss','false_alarm','correct_reject']`.
11. **Session independence**: Each experiment is a target-format session even when longitudinal cell IDs recur across days. This matches the single-plane project’s one experiment per recording session and avoids inventing cross-day time concatenation.

Pre-implementation mapping checks on experiments 775614751 and 1007107386 found finite aligned running/pupil values, exactly 9 output image categories per session (gray + 8 images), about 33% image-on frames as expected from 250 ms on/500 ms off, and about 2.6–2.8% change-positive frames with the changed-flash definition.

### Planned Sanity Checks
- [ ] Cohort IDs exactly equal eligible metadata IDs minus the three documented eye-absent sessions; expect 165 sessions, 37 mice, 28,821 session-neurons, 42,470 trials, and 11,192,974 aggregate trial frames.
- [ ] Trial counts/outcomes from converted arrays equal direct raw flags: go 37,143; catch 5,327; hit 13,569; miss 23,574; false alarm 814; correct reject 4,513.
- [ ] For raw trial 5 (or first eligible fallback) in selected sessions, `np.allclose(converted_neural, raw_event_slice.T.astype(float32))`.
- [ ] For the same raw trial, construct expected empty input from raw frame count and verify `np.allclose` plus `(0,time)` shape.
- [ ] Reconstruct image/change codes directly from raw presentation intervals and verify `np.allclose` against converted output rows.
- [ ] Reconstruct interpolated running/pupil and quintile codes using stored edges and verify `np.allclose` against output rows.
- [ ] Verify trial outcome code directly from raw mutually exclusive flags and that it is constant over each trial.
- [ ] Validate no NaN/Inf in retained neural/output arrays; neuron/time dimensions agree; every session has >=2 trials.
- [ ] Confirm image-on fraction near 1/3, go/catch near 87.5/12.5%, ophys dt near 32.31 ms, quintile distributions near 20% per class, and all image/change intervals remain within their trials without off-by-one overlap.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with `--full` (default), `--sample`, and `--show-processing`. It reads NWB/HDF5 tables directly but implements the same released SDK semantics documented in Steps 1–5. It validates source IDs, trial exclusivity, timestamps, ROI/event correspondence, finite neural values, nonoverlapping half-open trial slices, categorical ranges, and cross-stream dimensions before serialization. Metadata records session identities, trial IDs, cell IDs, quantile edges, timing, source release, and all transformation definitions.

Development smoke test: `python -u /app/convert_data.py /app/dev_sample.pkl --sample` completed without errors in 0.57 s. It produced 2 sessions, 229 trials, 231 session-neurons, 58,368 frames, and a 0.030-GiB pickle. The temporary pickle was moved to `/app/cache/dev_sample.pkl` during cleanup. `python3 -m py_compile` also passed.

Code inefficiencies identified:
- Loading complete 140k-frame float64 event matrices would transiently double memory and process spontaneous/movie periods that are never retained.
- Reading each trial independently would cause tens of thousands of small HDF5 reads.
- Keeping transposed views would pin bounding source blocks containing excluded trial intervals.
- Parallel full-file reads would compete for storage bandwidth and multiply peak memory.

Code speedups added:
- One bounding HDF5 read per session via `read_direct` converts event data straight to float32.
- Vectorized timestamp alignment, presentation lookup, interpolation, percentile coding, and outcome construction operate once per session.
- Trial matrices are copied from the in-memory task block, after which excluded raw frames can be released.
- Sequential session processing bounds raw-read memory while accumulating only required float32 output.
- Per-session timing and rolling ETA are printed. Smoke-test conversion averaged about 0.27 s/session plus serialization; even allowing much larger cell populations, the full conversion is expected comfortably below 15 minutes.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 231 session-neurons |
| Neurons / session | [89, 142]; mean 115.5 |
| Subjects | 1 (`403491`) |
| Sessions / subject | 2 |
| Trials (total) | 229 |
| Trials / session | [39, 190] |
| Trial timepoints | mean 251.29; min 225; max 389 |
| Input dimensions | 0 (range is empty as specified) |
| Image identity range/distribution | 0–16; gray 0.665; each of 16 images 0.004–0.038 overall |
| Image change distribution | no-change 0.973; change 0.027 |
| Running quintile distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| Pupil quintile distribution | [0.200, 0.200, 0.200, 0.200, 0.200] |
| Trial outcome trial distribution | hit 0.603, miss 0.271, false alarm 0.057, correct reject 0.070 |
| Trial outcome counts | [138, 62, 13, 16] |
| Brain region | VISp: 231 neurons |

### Processing Plots Review
Reviewed both `processing_775614751.png` and `processing_788490510.png`. Event rasters are sparse nonnegative L0 magnitudes; trial windows start/end cleanly; images occupy 250-ms flashes separated by 500-ms gray; the change label coincides exactly with the changed image flash; running and pupil traces are smooth after timestamp interpolation; quintile transitions occur at the plotted thresholds; and outcome is constant. Negative near-zero running values in the second plot are expected low-pass-filter excursions retained from the released stream, not an alignment artifact. No anomalies or temporal offsets were seen.

Manual raw checks loaded the NWBs independently for the first eligible trial of both sessions. `np.allclose` passed for raw event slice vs converted neural, the expected `(0,T)` input, directly reconstructed image/change outputs, and direct trial outcome. All neural and output values are finite. Metadata totals and array counts agree.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Direct float32 bounding read plus vectorized alignment | Sample conversion without plotting: 0.57 s total; avoids thousands of per-trial HDF5 reads |
| Session-at-a-time processing | Bounded peak raw memory; no parallel-I/O contention |

| Step | Time / Session | Estimated Total Time |
| Source read/alignment/slicing (sample) | 0.21–0.32 s | 35–55 s by session count; allow 2–3 min for larger cell matrices |
| Diagnostic plot (first two only) | about 0.45 s | under 1 s full mode (plots are not requested for full run) |
| Serialization | 0.04 s for 0.030 GiB | roughly 10–30 s for expected ~7.8 GiB |
| Total | 1.45 s with two plots | conservatively 2–4 min, far below 15 min |

The required verifier reports: **valid, no errors, no warnings**. Output ranges cover every declared class across the two-session sample. The sample pickle is 31 MiB; conversion and verification logs exist at the required paths.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: No format warnings. Scikit-learn emitted one evaluation warning because a predicted class was absent from the small held-out label subset; this does not indicate malformed data and is expected with only 39 trials in one sample session.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| image_identity | 0.1620 | 0.1436 |
| image_change | 0.6286 | 0.6056 |
| running_speed_quintile | 0.2085 | 0.2042 |
| pupil_diameter_quintile | 0.2352 | 0.2261 |
| trial_outcome | 0.2885 | 0.2680 |

Training completed successfully on CUDA. Mean loss decreased monotonically from 1.633954 (epoch 1) to 1.572538 (epoch 200); test loss was 1.606636. Every validation balanced accuracy exceeds uniform chance: image identity 2.44x chance, image change 1.21x, running 1.02x, pupil 1.13x, and outcome 1.07x. The weaker behavior/outcome margins are plausible for a two-session sample and will be reassessed on the complete cohort.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 7.815 GiB (filesystem display 7.9G)
- `verification_full_out.txt`: created (135 KiB)
- `conversion_full_out.txt`: created (17 KiB)

Full conversion took 66.69 s: 58.6 s to load/align/slice all 165 sessions and 8.09 s to serialize. This is substantially faster than the conservative 2–4 minute estimate. The verifier completed successfully with no errors. It issued 1,729 “all neural data is zero” trial warnings (4.07% of retained trials across 62 sessions), which are expected for sparse L0 events—especially sessions with as few as 6 valid cells—and are reviewed formally in Step 10.

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | v1.0 exact project: 16,122 unique longitudinal cells across active/passive; not directly comparable | Valid ROI/event columns | 28,821 active usable session-neurons; 13,627 unique IDs | 28,821 session-neurons | Yes (exact source selection) |
| Mean neurons/session | not reported | valid ROI table length | 174.67 (range 6–666) | 174.67 (range 6–666) | Yes |
| Subjects | exact project v1.0 table: 35; local v1.1 expanded | metadata `mouse_id` | 37 after task/output filters | 37 | Yes (v1.1) |
| Sessions | exact project v1.0: 225 active+passive | experiment table/session flags | 239 v1.1 total; 168 active; 165 active with pupil | 165 | Yes |
| Trials (total) | not reported | requested SDK flags | 42,470 eligible | 42,470 | Yes |
| Trials/session (mean) | not reported | requested SDK flags | 257.39 (range 39–409) | 257.39 | Yes |
| Ophys bin size | nominal 31 Hz (~32.26 ms) | event timestamps | median 32.31 ms | 32.31 ms | Yes |
| Brain region | single-plane project is VISp | targeted structure | VISp, 28,821 rows | VISp, 28,821 | Yes |
| Input dimensions | task says none | N/A | 0 | 0 | Yes |
| Image identity distribution | 250-ms on / 500-ms gray predicts ~1/3 image-on | presentation bounds | gray 0.669; each global image 0.020–0.021 | identical | Yes |
| Image change distribution | go 87.5%; one changed 250-ms flash/trial | `is_change` presentation | no-change 0.974, change 0.026 | identical | Yes |
| Running quintiles | five equal percentile bins | released filtered speed | [0.200 x5] | [0.200 x5] | Yes |
| Pupil quintiles | five equal percentile bins | blink-masked circular area | [0.200 x5] | [0.200 x5] | Yes |
| Trial outcomes (trial fraction) | definitions only, no cohort rate | mutually exclusive flags | hit 0.3195, miss 0.5551, FA 0.0192, CR 0.1063 | same; counts 13,569/23,574/814/4,513 | Yes |

Independent spot checks loaded raw NWBs for the first, middle, and final converted sessions (`775614751`, `889771676`, `1086048031`) and compared the first/middle/final retained trial in each. All nine neural event slices and reconstructed stimulus/change rows matched with `np.allclose`; neuron/trial counts and metadata totals also matched.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `/app/verification_full_out.txt` contains no errors and finishes with “Data verification complete.” Its only warnings are 1,729 “all neural data is zero” messages. The independent audit confirmed every referenced converted array is exactly zero and direct raw-NWB reads confirmed representative first/middle/last warnings are also exactly zero. This cannot be “fixed” without either (a) deleting scientifically valid quiet trials, violating the requested go/catch inclusion and biasing the dataset toward active neurons, (b) fabricating events, or (c) switching away from the paper’s L0 event representation. The warnings are therefore legitimate, irreducible diagnostics for sparse detected events, especially in low-cell-count sessions.
2. **Independent original-data sanity checks**: Created and ran `/app/cache/review_conversion.py` without importing conversion code. For sessions 0/82/164 and trial indices 0/5/final, it independently loaded raw event slices and reconstructed the empty input, image identity, image change, running interpolation/quintiles, pupil diameter interpolation/quintiles, and outcome. Every check passed `np.allclose`. Full trial arrays—not only scalars—were compared. As an explicit scalar spot check, trial index 5/timepoint 10/neuron 3 matched raw data in all three sessions.
3. **Reference code comparison**: Compared all six major processing stages line by line (table below). No unintended mismatch was found.
4. **Key-statistics comparison**: Rebuilt session IDs and totals from the v1.1 metadata and every source NWB. Exact matches: 165 sessions, 37 mice, 28,821 session-neurons, 42,470 trials, 11,192,974 trial frames, 37,143 go, 5,327 catch, and outcomes 13,569/23,574/814/4,513. Output frame fractions are image gray 0.668979; change 0.025687; exact 0.2 quintiles; outcomes 0.315722/0.559001/0.018435/0.106842. Median source presentation count is about 4,802/session; median image duration is 0.2502 s; overall omission fraction is 3.54% (correctly below the configured 5% of eligible non-change repeats because changes/pre-changes cannot be omitted); mean selected-trial change time is 4.277 s, consistent with the paper’s about 4.2 s.
5. **Edge cases/off-by-one audit**: For all trials, confirmed first frame `>= start`, preceding frame `< start`, last included frame `< stop`, next frame `>= stop`, positive length, no overlap, and matching neural/input/output time axes. Confirmed all static outcomes are constant, change implies a non-gray image, all category bounds are valid, all cells map to VISp, all stored ROIs are valid, first/final sessions and subjects are present, and every session has at least two trials. All checks passed in 11.55 s.

### Major Processing-Step Comparison

| Stage | Converter | Reference | Comparison / rationale |
|---|---|---|---|
| (a) Loading | `convert_data.py:252–320` direct NWB groups | `BehaviorOphysExperiment.from_nwb` at `behavior_ophys_experiment.py:226`; exposed properties at 523/554 | Same released event, timestamp, running, eye, trial, stimulus, and valid-cell datasets. Direct HDF access avoids loading projections/templates and is value-identical in raw checks. |
| (b) Filtering | exact project/active/pupil at 82–111; trials at 267; ROI assertions at 370 | SDK default `exclude_invalid_rois=True`; `CellSpecimens` valid filter at 205–208; `Trial` mutually exclusive flags | Same released ROI curation and requested trial flags. Passive/no-eye session exclusion is required for contemporaneous requested outputs and explicitly documented. |
| (c) Alignment | timestamp `searchsorted` half-open slicing at 287–295; interpolation at 298–311 | SDK synchronized `ophys_timestamps`; common sync process in whitepaper | Same synchronized clock. Using ophys rather than 30-Hz paper grid is explicitly required by this task. Every boundary independently checked. |
| (d) Binning | native ~32.31-ms ophys frames retained; quintiles at 124–131 | Paper interpolates event-triggered analyses to 30 Hz; SDK keeps native ophys | No neural rebinning because task says align based on ophys timestamp. Behavior quintiles are task-required, within-session percentile thresholds stored in metadata. |
| (e) Input | empty `(0,T)` at 339 | none in Decoder Inputs | Exact match to task; verifier accepts dinput=0. |
| (f) Output | stimulus at 149–189/333–337; processed behavior and outcomes at 298–337 | SDK `stimulus_presentations`, `is_change_event` at `stimulus_processing.py:592`, processed running, blink filtering and circular pupil area | Same source definitions; conversion to gray/image codes, change flash, quintiles, and repeated static outcome is the minimum transformation required by decoder specifications. |

### Issues Found and Resolved
- **No conversion mismatch was found in iteration 1**, so no code or data regeneration was necessary. The only flagged condition was source-native all-zero L0 event trials; direct raw verification established that removing or altering them would be less faithful than retaining them.
- **Reference-version count discrepancy** was already resolved in Step 4: whitepaper Table 1 is v1.0, whereas local data/metadata are expanded v1.1.0. Exact local cohort comparisons are internally consistent.
- **Reference-paper cohort discrepancy** was already resolved in Step 4: its 376-session behavior and familiar-multiplane neural subsets are not the exact single-plane `VisualBehavior` project requested here. They are documented benchmarks, not expected converted totals.

Audit output: `/app/cache/review_conversion_out.txt` ends with `ALL CHECKS PASS`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples 2>&1 | tee /app/train_decoder_full_out.txt`
- Device: NVIDIA L4 CUDA; the run completed normally without an out-of-memory fallback.
- Loss decreasing: **Yes**. Weighted cross-entropy decreased monotonically from 1.633122 (epoch 1) to 1.581270 (epoch 200); held-out test loss was 1.597059.
- Split: 33,914 training trials and 8,556 held-out trials (session-stratified split performed by the supplied decoder).
- The supplied script wrote `sample_trials.png` and `predictions.png` and ended with `train_decoder.py finished successfully.`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| Image identity | 0.1637 | 0.1605 | 2.73x chance (1/17 = 0.0588) |
| Image change | 0.5888 | 0.5833 | 1.17x chance (0.5) |
| Running-speed quintile | 0.2210 | 0.2182 | 1.09x chance (0.2) |
| Pupil-diameter quintile | 0.2182 | 0.2142 | 1.07x chance (0.2) |
| Trial outcome | 0.2823 | 0.2612 | 1.04x chance (0.25) |

All five held-out balanced accuracies are above chance. The small train/validation differences (0.0032, 0.0055, 0.0028, 0.0040, and 0.0211 absolute) give no evidence of a >1.5x overfitting gap. Lower-than-1.5x-chance variables are investigated explicitly in Step 12.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation balanced accuracy | Chance | Multiple of chance | Finding |
|----------|------------------------------|--------|--------------------|---------|
| Image identity | 0.1605 | 0.0588 | 2.73x | Clear signal; passes the 1.5x screen. |
| Image change | 0.5833 | 0.5000 | 1.17x | Above chance; within the paper's approximate 0.51–0.65 range despite a stricter all-frame task. |
| Running-speed quintile | 0.2182 | 0.2000 | 1.09x | Above chance; low instantaneous linear predictability is plausible and raw labels/alignment pass. |
| Pupil-diameter quintile | 0.2142 | 0.2000 | 1.07x | Above chance; low instantaneous linear predictability is plausible and raw labels/alignment pass. |
| Trial outcome | 0.2612 | 0.2500 | 1.04x | Above chance; outcome is intentionally static across long trials, while outcome-related neural information is localized near change/response. |

The 1.5x screen triggered investigation for four outputs; it is a diagnostic threshold, not an instruction to alter verified source labels. None is below chance. The exact train/validation accuracy ratios are 1.020, 1.009, 1.013, 1.019, and 1.081, respectively, all far below the 1.5x overfitting threshold.

### Accuracy Comparison to Supplied Papers

| Paper decoding result | Paper accuracy | Closest requested decoder output | This conversion | Interpretation |
|-----------------------|----------------|----------------------------------|-----------------|----------------|
| Image change vs immediately preceding repeat, random forest over first 400 ms (Figure 6A) | approximately 0.51–0.65 across plotted cell classes, strategies, and neuron counts | Image change | 0.5833 | Inside the plotted paper range. Our score is balanced timepoint accuracy across every trial frame, so it is not an identical estimator. |
| Hit vs miss, random forest over first 400 ms (Figure 6C) | approximately 0.52–0.74 across plotted conditions | Four-class trial outcome | 0.2612 (four-class chance 0.25) | Not numerically like-for-like: the requested label includes hit, miss, false alarm, and correct reject and is scored at every frame of the full trial. Raw outcome flags were nevertheless rechecked exactly. |
| False-alarm decoding (text referring to Figure S22) | Described only as “very low”; no numeric value in the supplied paper PDF | Four-class trial outcome | 0.2612 | Qualitatively compatible; no scalar paper value exists for comparison. |

The whitepaper reports processing/QC and cohort statistics but no neural decoding accuracy. A full-text search of the supplied paper found no other numeric decoder performance report. Figure values above are approximate readings because the paper reports curves/points rather than a numeric table.

### Targeted Low-Accuracy Audit

1. **Three-plus raw trials**: the independent raw-data audit reconstructed neural, image, change, running, pupil, and outcome arrays for the first/middle/final retained trials in each of experiments 775614751, 889771676, and 1086048031 (nine trials total). Every array passed `np.allclose`; the exact outcome flags were independently read from each NWB.
2. **Temporal alignment plots**: `processing_775614751.png` and `processing_788490510.png` show neural events and all labels on the native ophys clock. Changed-image pulses coincide with changed flashes, behavior is smoothly interpolated, and trial outcome remains constant. Visual review of full-training `sample_trials.png` and `predictions.png` found no offsets or category-range anomalies.
3. **Variation**: frame distributions are change 0.974313/0.025687, exactly balanced quintiles (approximately 0.2 each), and outcomes 0.315722/0.559001/0.018435/0.106842. No output is a 99% single class; balanced loss and balanced accuracy address the remaining imbalance.
4. **Neural filtering**: the converter uses released valid ROIs and unfiltered L0 event magnitudes, exactly the event representation specified by the paper. The 1,729 all-zero trials were verified to be genuinely zero in source arrays rather than alignment failures.
5. **Processing match**: independent all-session checks confirmed half-open ophys slices, every retained trial ID, all neuron counts, quintile construction, categorical bounds, and static outcomes. The supplied decoder is an instantaneous shared linear classifier after session-specific PCA-like projections; the paper classifier instead concatenates a 400-ms neural window and restricts examples to change/pre-change or hit/miss. This architectural/task distinction explains why weak continuous-behavior and long-trial outcome scores can be real even when labels are correct; it was considered only after raw and temporal checks passed.

No conversion change was justified. Relabeling frames around behavior/outcome events, shortening trials, or adding task variables as decoder inputs could raise accuracy but would violate the requested full-trial outputs or leak target information. All requested outputs remain faithful to the independently verified source streams.

### Issues Found and Resolved
- **Potential concern—four outputs below 1.5x chance**: resolved as a model/task sensitivity rather than a conversion error after the five audits above; all remain above chance, with small generalization gaps.
- **Potential concern—paper hit/miss score appears higher**: resolved by confirming that Figure 6C is a binary, 400-ms, go-trial-only random-forest analysis, whereas the required output is four-class and static over full variable-duration go/catch trials. The conversion cannot adopt the paper's restricted label without violating the target specification.
- **No new mismatch found**: no regeneration or retraining iteration was required.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with loading example, format, categories, reproducibility commands, cohort statistics, and decoder results.
- [x] cache/ folder created with `README_CACHE.md`; independent audit code/output, temporary development data, reference-page renders, and bytecode were moved there.
- [x] All files organized. Required conversion, validation, training, documentation, and plot artifacts remain in `/app`; every required-file existence check passed.

Final checks: all 14 workflow step statuses are `COMPLETE`; `train_decoder_full_out.txt` records epoch 200, validation metrics, and successful termination; all eleven explicitly required output/log/document files exist. The final converted pickle is 8,391,542,706 bytes (7.81 GiB).
