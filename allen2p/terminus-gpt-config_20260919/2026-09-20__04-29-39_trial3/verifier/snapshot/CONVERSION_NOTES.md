# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory Visual Behavior 2P
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `app/`
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
- Python 3.13.15
- NumPy 2.4.4
- PyTorch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `BehaviorOphysExperiment.from_nwb` / `from_lims` | `behavior_ophys_experiment.py` | LOADING | Construct one ophys experiment and expose behavior plus imaging streams. |
| `BehaviorOphysExperiment.dff_traces` | `behavior_ophys_experiment.py` | LOADING | Return per-cell precomputed delta-F/F traces; dF/F does not need to be recomputed. |
| `BehaviorOphysExperiment.ophys_timestamps` | `behavior_ophys_experiment.py` | LOADING | Return imaging-frame timestamps, the required master temporal grid. |
| `CellSpecimens.from_nwb` | `data_objects/cell_specimens/cell_specimens.py` | CURATION | Load segmented cell ROIs and traces; SDK cell table/trace indexing keeps cell identity aligned and uses valid ROI records. |
| `DFFTraces.from_nwb` | `data_objects/cell_specimens/traces/dff_traces.py` | LOADING | Read NWB `dff` RoiResponseSeries and associate each row with cell ROI IDs. |
| `Trials.from_nwb` and trial metric properties | `data_objects/trials/trials.py` | LOADING | Read trial intervals and expose go/catch/aborted/auto-rewarded and outcome columns. |
| `Trial._match_to_sync_timestamps` / trial event parsing | `data_objects/trials/trial.py` | PROCESSING | Build start/stop/change/response/reward times and hit, miss, false-alarm, correct-reject labels from task events. |
| `StimulusPresentations.from_nwb` | `data_objects/stimuli/presentations.py` | LOADING | Read presentation intervals including image identity, omission, and change information. |
| `RunningSpeed.from_nwb` | `data_objects/running_speed/running_speed.py` | LOADING | Read timestamped running speed in cm/s. |
| `EyeTrackingTable.from_nwb` / `from_data_file` | `data_objects/eye_tracking/eye_tracking_table.py` | PROCESSING | Load timestamped eye/pupil ellipse data; raw processing aligns video frames to stimulus timestamps, removes metadata frame, and marks/dilates blink/outlier frames. |
| `OphysTimestamps.from_nwb` | `data_objects/timestamps/ophys_timestamps.py` | LOADING | Read the timestamps attached to ophys acquisition. |

### Notes
- AllenSDK documentation recommends loading Visual Behavior 2P NWB through SDK objects rather than manually interpreting NWB.
- Neural activity is calcium imaging, and NWB already contains processed delta-F/F traces. Recomputing dF/F from fluorescence would diverge from the released/reference processing.
- The cell specimen table and dF/F rows are keyed by ROI/cell IDs; preserve this order and include released valid segmented ROIs rather than applying electrophysiology quality criteria.
- Trial semantics are explicit: go and catch are valid task trials; aborted and auto-rewarded are separately flagged. Outcomes map to hit/miss for go and false-alarm/correct-reject for catch.
- Stimulus presentations provide `image_name`/identity, presentation start/stop times, omissions, and change flags. These should be sampled onto ophys timestamps rather than assumed to share frame indices.
- Running and eye streams have independent timestamps. SDK eye processing performs video/stimulus frame alignment and blink/outlier handling; released NWB tables should be used directly. Pupil is represented by ellipse measurements including pupil area; requested diameter will require a documented equivalent-circle transform if no direct diameter field exists.
- No new filtering or conversion decision is finalized here; those decisions will be reconciled with actual files and reference texts in Steps 2-5.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Root: `/app/data/visual-behavior-ophys-1.1.0/` plus project manifest `visual-behavior-ophys_project_manifest_v1.1.0.json`.
- `behavior_ophys_experiments/`: 284 local NWB/HDF5 files, one file per ophys experiment (imaging plane), totaling 264,986,580,075 bytes.
- `project_metadata/`: `behavior_session_table.csv` (4,782 rows), `ophys_session_table.csv` (703), `ophys_experiment_table.csv` (1,936), and `ophys_cells_table.csv` (133,066). These tables describe the full release; local NWBs are a 284-experiment subset.
- Local files map to 247 ophys acquisition sessions and 38 mice. Most local sessions contain one supplied plane; eight have multiple supplied planes (3-7). Each NWB/plane is treated natively as an ophys experiment with its own neural population.
- Neural: `processing/ophys/dff/traces/data`, shape time x neuron, float64; corresponding timestamps and ROI references. Target format will require transpose.
- Trials: `intervals/trials`, with start/stop/change/response/reward times, initial/change image names, go/catch/aborted/auto-rewarded flags, and hit/miss/false-alarm/correct-reject outcomes.
- Stimulus: image-presentation indices and timestamps in `stimulus/presentation`; eight image templates per image set. Trial image fields are explicit strings.
- Running: filtered and unfiltered speed with independent timestamps under `processing/running`.
- Eye: `acquisition/EyeTracking` contains pupil/eye/corneal-reflection ellipse center, width, height, angle, processed/raw area, timestamps, and likely-blink flags. Some experiments lack eye data.
- Metadata provides mouse, session/experiment IDs, Cre line, session type/experience, targeted structure, imaging depth, equipment, and cell specimen/ROI IDs.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 42,147 across local experiments |
| Neurons / session | 4-666; mean 148.405 per NWB experiment |
| Subjects | 38 mice |
| Sessions / subject | 284 ophys experiments across 247 acquisition sessions; mean 7.47 experiments/mouse |
| Trials (total) | 171,887 native rows across experiments |
| Trials / session | 399-1,241; mean 605.236 native rows |
| Eligible requested trials | 85,230 go/catch, excluding aborted and auto-rewarded; 39-412 per experiment (mean 300.106) |
| Brain regions | VISp: 261 experiments; VISl: 23 |
| Session types | Ophys 1/3 images A, 2 A passive, 4/6 images B, 5 B passive |
| Genotypes | Slc17a7: 153 experiments; Sst: 85; Vip: 46 |

### Native Dimensions and Quality Observations
- Ophys traces contain 48,284-149,508 frames. Two timestamp intervals occur: about 32.31 ms and 93.23 ms, reflecting single-plane versus multiplane acquisition; target bins must be made uniform later.
- Running streams have roughly 269,540-287,886 samples/file and independent timestamps.
- Eye streams are variable and may be absent; conversion must not silently fabricate measurements.
- Exact metadata cell count (42,147) equals summed NWB dF/F columns, validating neuron identity/count alignment.
- Every experiment has at least 39 eligible trials, satisfying the minimum two-trial requirement.
- Native trial totals include 85,842 aborted and 815 auto-rewarded flags; requested filtering is therefore material.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (paper familiar-session subset) | 10,328 total: 8,619 excitatory, 470 Sst, 1,239 Vip | Methods: “8,619 excitatory cells ... 470 Sst cells ... 1,239 Vip cells.” |
| Subjects (paper neural subset) | 9 excitatory, 6 Sst, 9 Vip mice (overlapping classes not implied) | Methods |
| Imaging sessions (paper neural subset) | 21 excitatory, 15 Sst, 21 Vip | Methods |
| Supplied local subset | 42,147 cells, 38 mice, 284 experiments / 247 acquisition sessions | Direct data scan; broader than paper neural subset |
| Stimulus cadence | 250 ms image + 500 ms gray = 750 ms interval | Paper Fig. 1 caption and whitepaper task description |
| Omission rate | 5% of image repeats | Paper task description |
| Neural analysis time grid (paper) | Resampled to common 30 Hz before averaging windows | Paper Methods |
| Native ophys sampling | ~31 Hz single-plane or ~10.7 Hz multiplane | Whitepaper acquisition description and direct timestamps |
| Behavior data time bin | Native asynchronous timestamps; paper assigns events to 750 ms image intervals | Methods |
| Reward rate / engagement context | Paper defines engaged behavior as 1 reward/120 s and 1 lick bout/10 s; 60.1% image intervals engaged | Paper results; not a trial reward fraction |
| Behavioral strategy model | mean AUC 0.83 over 382 sessions | Paper Fig. 2; not neural decoder accuracy |

### Processing Details
- Visual change-detection task: eight natural images, each displayed for 250 ms and followed by 500 ms gray. Changes occur between image identities; catch trials have no change. Licks in the post-change response window determine outcomes.
- The paper assigns behavior to each 750 ms image-presentation interval and segments licking bouts with a 700 ms inter-lick threshold.
- Neural traces are aligned by ophys timestamps. The paper used released detected calcium events, resampled traces to a common 30 Hz grid, and often averaged specified windows. Its decoding used the first 400 ms after image presentation.
- Whitepaper/released pipeline includes motion correction, ROI segmentation/classification, neuropil correction and dF/F, plus event detection. The released dF/F and events are processed products; neither should be recomputed from raw fluorescence.
- Running speed is supplied in cm/s with timestamps. Eye tracking fits pupil/eye ellipses; processed pupil area is NaN during `likely_blink` frames. Width/height are also available.

### Curation Steps

**Neuron curation rules**:
Use released valid cell ROIs/cell specimen rows. The SDK aligns cell/ROI IDs across tables and processed traces. Do not impose electrophysiology criteria. The cited paper further restricted neural analyses to familiar-image multiplane sessions, but that is a paper-specific scientific subset rather than the supplied Decoder Task, which requests collection/conversion under Visual Behavior.

**Trial curation rules**:
The requested conversion explicitly includes go and catch and excludes aborted and auto-rewarded trials. This is consistent with native independent flags. Passive sessions may have many rows marked aborted and relatively few valid go/catch rows, but every supplied experiment retains at least 39 eligible rows.

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Change vs repeat (random forest, first 400 ms) | Reported graphically as % correct; no numeric table/text recoverable |
| Hit vs miss (random forest, first 400 ms) | Reported graphically as % correct; no numeric table/text recoverable |
| Behavioral strategy dynamic model | AUC 0.83 (not a neural decoder and not directly comparable) |

### Applicability to this conversion
The reference paper decoder differs from the required task: it uses event traces and image-level binary labels, whereas this conversion must predict time-varying image identity/change/running/pupil plus static trial outcome. Native dF/F is selected as the continuous neural signal exposed by SDK and suitable for framewise decoding; this difference will be explicitly checked against native events in Step 4/5.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Neural representation | SDK exposes released dF/F and detected events | Both arrays share ophys timing/cell axes | Paper analyses detected events | Use dF/F as the standard SDK continuous calcium-activity property requested by target `neural`; do not recompute. Record the divergence from the paper's specialized event analysis. |
| Session subset | SDK can load every experiment | 284 supplied experiments span active/passive, familiar/novel, single/multiplane | Paper neural analysis restricts familiar multiplane | Convert all supplied experiments because Decoder Task says collect Visual Behavior and explicitly defines trial filtering; paper subset restriction was analysis-specific. |
| Session unit | `BehaviorOphysExperiment` is one imaging plane | A few ophys sessions have 3-7 files with identical behavior but distinct cells/rates | Paper decodes each imaging plane | Treat each NWB experiment/plane as one target session, matching SDK and paper decoding unit. |
| Sampling rate | Ophys timestamps are authoritative | 32.31 ms and 93.23 ms native intervals | Paper resampled to common 30 Hz | Construct one common grid from absolute ophys time. A coarser uniform bin must avoid inventing temporal precision for 10.7 Hz data; final choice in Step 5. |
| Stimulus representation | SDK reconstructs presentation table and image names | NWB presentation indices map to eight template descriptions; trials contain initial/change names | 250 ms image + 500 ms gray | Build image identity from actual presentation timestamps/data, with a dedicated gray class between image presentations. |
| Image change | Trial and stimulus tables expose change flags/times | Eligible trial `change_time` is present for go/catch | Paper change decoder compares changes to repeats | Emit a one-bin pulse immediately after each identity change; catch trials have no change pulse. |
| Pupil measure | Eye table supplies area, width, height, likely-blink | Processed area is NaN during likely blinks; some files lack eye | Task requests diameter; paper uses pupil area | Convert processed pupil area to equivalent-circle diameter `2*sqrt(area/pi)`, preserving NaN. This is orientation-invariant and uses reference processing. |
| Trial labels | SDK exposes mutually exclusive outcomes | Every eligible row has exactly one outcome; totals 15,936 hit, 58,602 miss, 932 false alarm, 9,760 correct reject | Go/catch outcome semantics agree | Encode four outcome classes in this order and retain only non-aborted, non-auto-rewarded go/catch rows. |
| Passive-session eligible rows | Flags are authoritative | Passive files contain many aborted rows but 39+ eligible rows each, often misses | Paper excludes passive neural sessions | Retain explicit eligible rows per task; do not infer trial validity from session name. Document class imbalance. |
| Simultaneous planes | Each NWB repeats behavior streams | Trial/running values match across planes | Paper analyzes imaging planes separately | Keep each plane as a session; duplicated behavior is expected, not accidental duplication of neurons. |

### Final Reconciled Understanding
- Load experiment-level NWB products and metadata; preserve released cell ordering and IDs.
- Use absolute ophys timestamps as the temporal reference and interpolate/sample asynchronous stimulus, running, and pupil streams onto a uniform grid.
- Segment with native trial start/stop boundaries and apply the explicit requested trial flags.
- Native outcome checks passed in every file: one and only one of hit/miss/false-alarm/correct-reject per eligible trial, with correct go/catch correspondence.
- Metadata cell total (42,147) exactly matches summed dF/F columns.
- Missing eye data and blink NaNs are genuine missingness; percentile discretization must be fit only on finite training/session observations and missingness represented explicitly or sessions without usable pupil excluded. The final policy is specified in Step 5.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `processing/ophys/dff/traces/data` + timestamps | `neural[session][trial]` | Linear interpolation at 100 ms grid centers; transpose to neuron x time; float32 | `BehaviorOphysExperiment.dff_traces`, `OphysTimestamps.from_nwb` | Released dF/F, no recomputation; cell order follows ROI response series. |
| none | `input[session][trial]` | Empty float32 array with shape (0, time) | N/A | Decoder Task explicitly specifies no inputs. |
| stimulus presentation index/timestamps + template descriptions | `output[...,0,:]` | Categorical image identity while image is visibly presented; gray during 500 ms inter-image periods | `StimulusPresentations.from_nwb` | Classes: gray plus 16 named images. |
| trial `change_time` / presentation identity transition | `output[...,1,:]` | Binary pulse in first 100 ms bin at/after change | `Trial` parsing / stimulus presentations | 1 immediately after identity change; catch trials remain zero. |
| `processing/running/speed` + timestamps | `output[...,2,:]` | Linear interpolation, then session-wise equal-frequency quintiles | `RunningSpeed.from_nwb` | Five classes `Q1`…`Q5`; fit edges from finite eligible-trial samples. |
| processed pupil `area` + eye timestamps | `output[...,3,:]` | Equivalent-circle diameter `2*sqrt(area/pi)`, interpolate finite points, then session-wise equal-frequency quintiles | `EyeTrackingTable.from_nwb` | Blink samples are NaN in released area and omitted from interpolation anchors. |
| trial hit/miss/false_alarm/correct_reject flags | `output[...,4]` conceptually static | Repeat static class across time only if validator requires homogeneous output matrix; otherwise static scalar channel | `Trials.from_nwb` | Values: hit=0, miss=1, false_alarm=2, correct_reject=3. |
| metadata `mouse_id` | `subjects`, `subject_idx` | Unique sorted string IDs and per-experiment index | project metadata | 38 source mice; final may omit mice only if all their files lack required pupil. |
| metadata `targeted_structure` | `brain_regions`, `brain_region_idx` | Regions `VISp`, `VISl`; repeat experiment region for all cells | project metadata | Each imaging plane has one targeted structure. |

### Output Classes
- `image_identity`: `gray`, then lexicographically sorted `im000, im031, im035, im045, im054, im061, im062, im063, im065, im066, im069, im073, im075, im077, im085, im106`.
- `image_change`: `no_change`, `change`.
- `running_speed_quintile`: `Q1_lowest` through `Q5_highest`.
- `pupil_diameter_quintile`: `Q1_smallest` through `Q5_largest`.
- `trial_outcome`: `hit`, `miss`, `false_alarm`, `correct_reject`.

### Key Decisions
1. **Session unit**: One NWB ophys experiment/imaging plane is one target session, matching SDK and paper imaging-plane decoding. Simultaneous planes legitimately repeat behavior but contain distinct cells.
2. **Trial curation**: Keep exactly `(go OR catch) AND NOT aborted AND NOT auto_rewarded`. Require all trial bounds within ophys support (verified for all source eligible rows).
3. **Uniform temporal bin**: Use 100 ms (`metadata.time_bin_size=100.0`). This is close to but not finer than the slowest 93.23 ms native interval, avoids pretending multiplane data have 30 Hz resolution, resolves the target's cross-session uniformity requirement, and still provides 2-3 samples during each 250 ms image flash.
4. **Grid definition**: Grid centers are `start_time + 0.05 + 0.1*k`, with only centers strictly before native trial stop. Trials retain their native variable duration (71-126 typical points); `off_start=0`, `off_end=None` because alignment is to each trial's native start and end varies.
5. **Neural interpolation**: Linear interpolation of released dF/F on absolute ophys timestamps. This is conservative at 100 ms and uses no extrapolation.
6. **Stimulus state**: Decode NWB presentation control indices using template `control_description`. Mark the actual 250 ms image display after each presentation timestamp, gray otherwise. This honors the experiment rather than labeling the entire 750 ms interval as an image.
7. **Change pulse**: Set the first grid center at or after native `change_time` to 1. This implements “right after” and avoids anticipatory labeling.
8. **Percentile discretization**: Compute quintile edges separately per experiment from finite values sampled over retained trials. Use rank/quantile edges with duplicate-edge handling; this minimizes apparatus/session scaling effects and targets equal occupancy. Apply one set of edges to every trial in that session to avoid trial leakage.
9. **Pupil diameter**: Processed area is preferred because reference blink handling has already set bad frames to NaN. Equivalent-circle diameter is a standard scalar diameter preserving ellipse area. Interpolate only between finite anchors; do not extrapolate outside eye support.
10. **Missing pupil policy**: Exclude the three experiments with no pupil series (`795953296`, `806456687`, `833631914`) because pupil is a required output and no defensible value can be fabricated. Within remaining experiments, linearly bridge reference-marked blink gaps using surrounding finite samples; drop an individual trial only if any grid point lies outside finite eye support or interpolation cannot produce a finite value. Report all losses.
11. **Running support**: Require finite interpolated running at every retained grid point; drop only unsupported trials and report.
12. **Outcome representation**: Outcome is static by specification. If the supplied validator requires one rectangular output matrix, repeat the static class over time while documenting `output_static=[False,False,False,False,True]` in metadata; semantically it remains per-trial.
13. **Types/memory**: Neural float32; outputs compact integer arrays; empty inputs shape `(0,T)`. Estimated 7.35M timepoints makes full conversion practical but memory-intensive, so process/write session-by-session in one pass and avoid loading raw image templates.
14. **All experimental variables considered**: Available trials also contain lick, response, reward, omission and timing fields; metadata contain genotype/depth/session experience. They are not decoder inputs because the task says no inputs, nor outputs because the required output list is exhaustive. They remain described in metadata/session info.

### Planned Sanity Checks
- [ ] For three experiments/rate regimes, compare raw dF/F values at exact/near ophys timestamps to converted interpolation with `np.allclose`.
- [ ] Reconstruct image labels directly from raw NWB presentation timestamps and compare selected converted trial bins with `np.allclose`.
- [ ] Compare raw running interpolation and pupil area-to-diameter interpolation at selected bins with `np.allclose`.
- [ ] Verify each change pulse is in the first bin at/after raw `change_time`, never on catch trials.
- [ ] Verify all eligible outcomes are mutually exclusive and converted static labels match raw flags with `np.allclose`.
- [ ] Verify quintile occupancies are approximately 20% per retained experiment and all values lie in 0-4.
- [ ] Verify every neural/input/time-varying output has identical T within each trial and all sessions retain at least two trials.
- [ ] Compare source vs converted mice, sessions, neurons, trials, regions, outcome distribution, and frame-rate groups; explain only explicit pupil/support exclusions.
- [ ] Plot raw timestamps/continuous streams and converted categorical labels for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with:
- Required `python -u convert_data.py OUTFILE`, `--full` (default/all), `--sample`, and `--show-processing` interfaces.
- Direct HDF5 reads of released NWB arrays and metadata CSV joins, avoiding network/cache dependencies.
- Explicit trial filtering, absolute-time interpolation, stimulus reconstruction, quintile discretization, static outcome construction, subject/region indexing, metadata, assertions, and per-session timing.
- Two-rate sample selection and six-panel processing plots for up to two sessions.
- Missing-pupil experiment exclusion and trial-level finite/support checks with reported loss counts.

Code inefficiencies identified:
- Initial draft converted the entire HDF5 dF/F dataset to NumPy once per trial, which would cause severe redundant I/O.

Code speedups added:
- Load each experiment's dF/F matrix once as float32, then vectorized interpolation by index/weight arrays for each trial.
- Read only metadata/behavior vectors needed for processing; never load large stimulus image templates.
- Fit quantile edges once per experiment from concatenated retained samples.
- Script passes `py_compile` and CLI help invocation.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 689 |
| Neurons / session | 666, 23 |
| Subjects | 2 |
| Sessions / subject | 1 each |
| Trials (total) | 730 retained / 730 source eligible |
| Trials / session | 324, 406 |
| Trial points at 100 ms | 72-125; mean 86.00 |
| Image identity distribution | gray 0.667; each of 16 images ~0.014-0.028 pooled |
| Image change distribution | no-change 0.990; change 0.010 after correction |
| Running quintile distribution | each 0.200 |
| Pupil quintile distribution | each 0.200 |
| Outcome distribution | hit 0.210, miss 0.665, false alarm 0.009, correct reject 0.116 |

### Format Validation
- `/app/verification_sample_out.txt`: “Data format is valid, no errors or warnings.”
- Every trial has aligned neural/input/output T, finite float32 neural values, input shape `(0,T)`, output shape `(5,T)`, and static outcome constant through time.
- Output ranges exactly match declared categorical values.

### Processing Plots Review
- Final sample created `processing_877018118.png` (native ~31 Hz, 666 cells) and `processing_953659749.png` (~10.7 Hz, 23 cells). Earlier low-neuron diagnostic plots are retained as audit artifacts.
- Plots contain neural heatmap, visible-image states, change pulse, continuous and binned running, continuous and binned pupil, and uniform grid intervals. No shape/time-grid anomalies were observed.

### Issue Found and Fixed
- Initial sample had one `image_change` pulse on every go and catch trial because native catch `change_time` marks the scheduled sham-change time. This contradicted identity-change semantics.
- Fixed construction to gate pulses by native `is_change`; added a catch assertion and reconverted/revalidated.
- Recheck: converted pulse counts 306 and 181 exactly equal raw eligible changes; catch counts 43 and 28 contribute zero pulses.

### Run Time Estimates
| Speed-ups Implemented | Time Savings |
|---|---|
| One dF/F read/session; vectorized interpolation | Avoids hundreds of redundant HDF5 reads/session |
| No image-template pixel loading | Avoids very large irrelevant arrays |

| Step | Time / Session | Estimated Total Time |
|---|---:|---:|
| Sample conversion compute | 0.7-1.3 s | ~3-6 min for 281 retained sessions plus I/O |
| Full pickle serialization | not dominant in sample | Expected under 15 min total |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None.
- Validator warnings: None during verify-only.
- Training emitted sklearn warnings that predictions included classes absent from sample validation truth. The two selected sessions use one image set and contain no false alarms, while `output_values` correctly declares full-dataset classes. This cannot be fixed without abandoning the required two-session sample or selecting sessions solely for labels; full data contain every class.

### Decoder Iterations
1. Initial rate-representative sample had only 10 and 12 neurons. Loss decreased, and 4/5 validation outputs exceeded chance, but trial outcome was 0.2101 vs 0.25 chance. This was investigated rather than accepted.
2. Sample selection was revised to choose the neuron-richest experiment in each native rate group (666 fast-rate cells and 23 slow-rate cells). Conversion logic was unchanged. Format revalidation passed.

### Decoder Results (Sample, final)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------:|------------------------:|-------:|
| image_identity | 0.5054 | 0.4581 | 0.0588 |
| image_change | 0.7629 | 0.6326 | 0.5000 |
| running_speed_quintile | 0.3664 | 0.3248 | 0.2000 |
| pupil_diameter_quintile | 0.3712 | 0.3016 | 0.2000 |
| trial_outcome | 0.6751 | 0.4457 | 0.2500 |

- Loss decreased monotonically from 1.643833 at epoch 1 to 1.153258 at epoch 200; test loss 1.322341.
- All final validation balanced accuracies are above chance. Image identity is especially strong, supporting stimulus/neural temporal alignment.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 4,543,137,779 bytes (4.3 GiB)
- `conversion_full_out.txt`: created; conversion completed in 251.83 s
- `verification_full_out.txt`: created; “Data format is valid, no errors or warnings.”

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Paper subset 10,328 | Released valid cells | 42,147 | 41,871 | Yes after 276 cells in 3 no-pupil experiments excluded |
| Mean neurons/session | N/A | experiment/plane unit | 148.405 | 149.01 | Yes after explicit exclusion |
| Subjects | paper subsets 6-9/class | mouse_id | 38 | 38 | Yes |
| Sessions | paper familiar subset 57 class-sessions | experiment/plane unit | 284 | 281 | Yes after 3 no-pupil experiments excluded |
| Trials (eligible) | task-defined | native flags | 85,230 | 84,313 | Yes after 917 eligible trials in excluded experiments |
| Trials/session | N/A | native trial table | 39-412 eligible | 39-412 retained | Yes, zero trial drops in retained sessions |
| Brain regions | V1/LM | targeted_structure | VISp/VISl | VISp/VISl | Yes |
| Image identity | 8 images/set, 250 ms on/500 ms gray | presentation controls | 16 union images | gray 0.6659; each image 0.0200-0.0218 | Yes |
| Image change | go/change versus catch | `is_change` | 73,733 retained go changes | 73,733 pulses (0.010264 of bins) | Yes |
| Running quintiles | continuous cm/s | timestamped filtered speed | continuous | each 0.200 ± 0.00002 | Yes |
| Pupil quintiles | processed ellipse area/NaN blink | eye timestamps/processed area | 281 usable files | each 0.200 ± 0.00002 | Yes |
| Outcome distribution | no direct release-wide value | exclusive flags | retained trial labels | time-weighted: hit .1823, miss .6920, FA .0104, CR .1153 | Consistent |

### Spot Checks
- First, middle, and last retained experiments passed direct raw-vs-converted np.allclose checks for neural, image, change, running, pupil, and outcome.
- All 281 retained experiments preserved every eligible trial; no support or nonfinite trial drops.
- Full verifier reports T 70-126 (mean 84.77) and neurons 4-666/session.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Searched all of `verification_full_out.txt`; validator reports no errors or warnings, and a token scan found no invalid/NaN/Inf diagnostics.
2. **Raw-data np.allclose sanity checks** (`cache_work/audit_full.py`):
   - Neural: independently loaded raw dF/F/timestamps and reconstructed linear interpolation for trial 5 of experiments 775614751, 936494970, and 1086048031; all matrices matched with rtol/atol 2e-5.
   - Input: verified every input has exact shape `(0,T)`, as required by “No inputs,” and aligned T.
   - Outputs: independently reconstructed image visibility, change pulse, running interpolation/quintile code, pupil area-to-diameter interpolation/quintile code, and static outcome from raw NWB; every tested array matched with np.allclose.
3. **Reference code comparison**:
   - (a) Loading: script reads the same NWB products exposed by `BehaviorOphysExperiment.from_nwb`; direct HDF5 is used only for speed and avoids semantic alteration.
   - (b) Curation: cell columns are the released dF/F ROI response rows/cell table, matching `CellSpecimens`; trials use native SDK flags exactly. Only files unable to provide mandatory pupil output are excluded.
   - (c) Alignment: all streams use their explicit timestamps, matching `OphysTimestamps`, `RunningSpeed`, and `EyeTrackingTable`; no frame-index alignment across clocks.
   - (d) Binning: reference paper resampled to 30 Hz for its event analysis; this task requires one cross-session bin and source includes 10.7 Hz data, so 100 ms is used to avoid invented precision. This is the justified task-specific difference.
   - (e) Input construction: empty `(0,T)` arrays exactly implement no decoder inputs.
   - (f) Output construction: stimulus controls and 250 ms visibility follow `StimulusPresentations`; trial flags follow `Trials`; processed running and blink-cleaned pupil area follow SDK products. Diameter/quintiles are required downstream transforms.
4. **Key-statistics comparison**: Source 284/42,147/85,230 sessions-neurons-eligible trials minus explicit no-pupil totals 3/276/917 equals converted 281/41,871/84,313 exactly. All 38 mice remain. Regions and image sets match metadata/templates.
5. **Edge cases/off-by-one review**:
   - Half-open grid uses centers `start+0.05+0.1*k < stop`; observed 70-126 points agrees with 7.02-12.61 s native durations.
   - All eligible bounds were verified within ophys support.
   - Initial catch-pulse bug was found in Step 7 and fixed by gating native scheduled `change_time` with `is_change`; full pulse total now equals raw retained changes.
   - Simultaneous planes intentionally duplicate behavior while retaining distinct neural populations.
   - Blink NaNs are omitted as interpolation anchors; no extrapolation and no retained-trial nonfinite values.

### Issues Found and Resolved
- **Catch sham-change incorrectly pulsed in initial sample**: gated by `is_change`, added assertion, reconverted, and reran all validation/audit checks successfully.
- **Initial dF/F redundant I/O**: moved full trace read outside trial loop before sample/full processing.
- **Three files lack pupil tracking**: excluded and fully accounted (276 neurons, 917 eligible trials); fabrication would violate reference handling and required pupil output.
- **Warnings**: None in full verification, so no unfixable validator warnings remain.

### Review Conclusion
Every available source statistic and every major processing stage has been reconciled. Independent raw-file comparisons pass, losses are entirely explicit, and no unresolved conversion issue remains.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Full command completed successfully on CUDA using 67,343 training and 16,970 validation trials.
- Loss decreasing: Yes, smoothly from 1.641700 (epoch 1) to 1.371061 (epoch 200).
- Test loss: 1.514340.
- `--plot-samples` completed as part of the required command.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|----------------------:|------------------------:|-------:|-------|
| image_identity | 0.3965 | 0.3634 | 0.0588 | 6.18× chance validation |
| image_change | 0.7283 | 0.6281 | 0.5000 | 1.26× chance; rare one-bin pulse |
| running_speed_quintile | 0.2974 | 0.2698 | 0.2000 | 1.35× chance |
| pupil_diameter_quintile | 0.3116 | 0.2702 | 0.2000 | 1.35× chance |
| trial_outcome | 0.4575 | 0.3219 | 0.2500 | 1.29× chance; static trial label |

All five validation balanced accuracies exceed chance. Strong image decoding supports correct stimulus/neural alignment. The complete output is in `train_decoder_full_out.txt`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Achieved Validation Accuracy | Chance | Ratio to Chance | Expectation from Papers |
|----------|-----------------------------:|-------:|----------------:|-------------------------|
| image_identity | 0.3634 | 0.0588 | 6.18 | Not decoded in cited paper |
| image_change | 0.6281 | 0.5000 | 1.26 | Paper reports change-vs-repeat % correct graphically, first 400 ms; no numeric text/table |
| running quintile | 0.2698 | 0.2000 | 1.35 | Not decoded in cited paper |
| pupil quintile | 0.2702 | 0.2000 | 1.35 | Not decoded in cited paper |
| trial outcome | 0.3219 | 0.2500 | 1.29 | Paper reports binary hit-vs-miss graphically; current four-class task is not directly comparable |

### Check 1: Accuracy vs Chance
- Every output exceeds chance. Image identity is far above chance.
- Image change, running, pupil, and outcome are below the cautionary 1.5×-chance threshold and were investigated rather than dismissed.
- **Three raw trials / output correctness**: Step 10 independently reconstructed trial 5 in first, middle, and last retained sessions from original NWB. Image, change, running, pupil, and outcome all passed `np.allclose`.
- **Temporal alignment**: raw timestamps and converted labels were plotted for fast and slow ophys regimes. Stimulus identity uses actual 250 ms visibility; change occurs in the first grid bin at/after raw change time; asynchronous behavior uses its own timestamps.
- **Variation**: image-change prevalence is 1.026% of time bins by design because it is a one-bin event. Running and pupil classes are exactly balanced. Outcome classes are imbalanced but balanced loss/accuracy are used. Thus low ratios are not caused by accidental constant outputs.
- **Filtering/processing**: released valid dF/F traces and native trial flags are used. All retained trials are within ophys support and all outputs are finite.
- Conclusion: no conversion change is justified. Weak-to-moderate behavior/outcome decoding is plausible across heterogeneous mice, cell classes, regions, passive/active experience types, and sessions with as few as four neurons.

### Check 2: Accuracy Comparison to Papers
- The cited paper's neural decoders differ materially: detected calcium events, familiar multiplane subset, binary change-vs-repeat or hit-vs-miss, first 400 ms, random forest, and per-plane neuron sampling.
- Extracted paper text and methods provide no numeric neural-decoder accuracy; values are only graphical. Therefore no exact numeric claim can be responsibly invented.
- The paper's stated AUC 0.83 is for a behavioral strategy model, not neural decoding, and is not used as a benchmark.
- Current image-change and four-class outcome results are directionally consistent with decodable task information, while image identity strongly confirms alignment.

### Check 3: Train vs Validation Gap
| Output | Train / Validation Ratio | Above 1.5? |
|--------|-------------------------:|:----------:|
| image_identity | 1.09 | No |
| image_change | 1.16 | No |
| running quintile | 1.10 | No |
| pupil quintile | 1.15 | No |
| trial outcome | 1.42 | No |

No output has the specified concerning >1.5 train/validation gap. There is no evidence of severe overfitting or leakage.

### Issues Found and Resolved
- No new conversion issue was found in this review.
- Lower-than-1.5×-chance outputs were fully checked against raw values, timing, variation, filtering, and reference processing. All checks passed, so changing labels or leaking behavioral information into inputs would be unjustified.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created with loading, structure, statistics, and reproduction instructions
- [x] cache/ folder created
- [x] `cache/README_CACHE.md` documents investigation artifacts
- [x] Analysis/audit scripts and extracted PDF text organized under cache
- [x] All required conversion, validation, training, and documentation files retained

<!-- Step 8 iteration note: Initial two-rate sample used only 10 and 12 neurons. Loss decreased and 4/5 outputs exceeded chance, but trial outcome validation balanced accuracy was 0.2101 vs 0.25 chance. This was treated as an inadequate low-neuron sample test, not dismissed. Sample selection was revised to retain fast/slow rate coverage while choosing the largest available neural population in each group; conversion logic was unchanged. -->
