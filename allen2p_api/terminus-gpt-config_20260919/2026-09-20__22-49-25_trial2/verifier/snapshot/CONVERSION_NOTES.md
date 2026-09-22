# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory Visual Behavior 2P
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment checks:
- Python 3.13.15
- NumPy 2.4.4
- PyTorch 2.6.0+cu124

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `Dockerfile`
- `allensdk_docs/`
- `code/`
- `data/`
- `decoder.py`
- `docker-compose.yaml`
- `methods.txt`
- `paper.pdf`
- `train_decoder.py`
- `tutorials/`
- `whitepaper.pdf`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `VisualBehaviorOphysProjectCache.from_local_cache` | `behavior_project_cache/project_cache_base.py` | LOADING | Instantiate the required AllenSDK cache against an existing local cache directory. |
| `get_ophys_experiment_table` | `behavior_project_cache/behavior_project_cache.py` | LOADING | Return experiment-level metadata used to enumerate recordings and mouse/area/session attributes. |
| `get_ophys_cells_table` | `behavior_project_cache/behavior_project_cache.py` | LOADING | Return project cell metadata. |
| `get_behavior_ophys_experiment` | `behavior_project_cache/behavior_project_cache.py` | LOADING | Load one `BehaviorOphysExperiment` through AllenSDK. |
| `BehaviorOphysExperiment.ophys_timestamps` | `behavior_ophys_experiment.py` | PROCESSING | Return microscope frame timestamps; these are the required common temporal grid. |
| `BehaviorOphysExperiment.dff_traces` | `behavior_ophys_experiment.py` | PROCESSING | Return precomputed change-in-fluorescence / fluorescence traces by cell. |
| `BehaviorOphysExperiment.events` | `behavior_ophys_experiment.py` | PROCESSING | Return inferred calcium events and SDK-filtered events by cell. |
| `BehaviorOphysExperiment.cell_specimen_table` | `behavior_ophys_experiment.py` | CURATION | Return cell metadata with invalid ROIs/non-cell objects already removed (`roi_valid=True`). |
| `BehaviorSession.trials` | `behavior_session.py` | CURATION | Return trial timing and mutually exclusive trial/outcome flags. |
| `BehaviorSession.stimulus_presentations` | `behavior_session.py` | PROCESSING | Return image presentation intervals and identities. |
| `BehaviorSession.running_speed` | `behavior_session.py` | PROCESSING | Return running speed with timestamps. |
| `BehaviorSession.eye_tracking` | `behavior_session.py` | PROCESSING | Return eye/pupil measurements with timestamps. |
| `Trial._get_trial_data` | `data_objects/trials/trial.py` | CURATION | Define go, catch, aborted, auto-rewarded, hit, miss, false alarm, and correct reject flags. |
| `filter_events_array` via `Events` | `data_objects/cell_specimens/events.py` | PROCESSING | Produce SDK-filtered inferred event traces using event-detection parameters. |

### Notes
- The supplied reference code is AllenSDK. Its visual-behavior documentation explicitly recommends loading and interacting with NWB-backed data through AllenSDK; conversion will never open NWB files directly.
- Cache construction should use `VisualBehaviorOphysProjectCache.from_local_cache(cache_dir=...)`, followed by table methods and `get_behavior_ophys_experiment(id)`.
- `dff_traces` are already computed and exposed by the SDK; delta F/F must not be recomputed from raw fluorescence. The SDK also exposes inferred `events` and `filtered_events`; the final neural representation will be resolved against paper methods in Steps 3-5.
- `cell_specimen_table` is already curated to valid ROIs (`roi_valid=True`), so this SDK-level curation must be retained.
- SDK trial logic makes go/catch mutually exclusive. Aborted trials have neither flag; auto-rewarded trials are excluded from go and are not assigned hit/miss. Valid requested outcomes are hit, miss, false alarm, and correct reject.
- Newer releases contain multiple stimulus blocks. The cache warns that legacy task stimuli are the rows whose `stimulus_block_name` contains `change_detection`; non-task blocks must not be used to construct image outputs.
- Neural and behavioral streams expose their own timestamps. All streams will be sampled/aligned explicitly to microscope (`ophys_timestamps`) without assuming equal sampling rates.
- No electrophysiology quality filtering applies because this is two-photon calcium imaging.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is a 247 GB AllenSDK cloud-cache for `visual-behavior-ophys` release 1.1.0.
- It contains a project manifest, four project metadata CSVs, and 284 locally available behavior-ophys experiment NWB files. NWB files were **not** opened directly; all metadata and experiment contents were read via `VisualBehaviorOphysProjectCache`.
- Full-release project tables exposed by the cache contain 1,936 experiments, 703 ophys/behavior sessions, 107 mice, and 133,066 valid cell records. Only the locally supplied 284 experiments are convertible without downloading data, so all conversion statistics below refer to that provided subset.
- The local subset has 284 experiments from 247 ophys sessions (239 single-plane sessions; multiscope sessions contribute multiple simultaneous planes), 38 mice, and 42,147 experiment-cell records.
- Areas are VISp (261 experiments) and VISl (23). Cre lines are Slc17a7-IRES2-Cre (153), Sst-IRES-Cre (85), and Vip-IRES-Cre (46). Experience levels are Familiar (150), Novel 1 (38), and Novel >1 (96).
- Native experiment objects expose: `ophys_timestamps` (float seconds); `dff_traces` and inferred `events`/`filtered_events` arrays by valid cell; trial table; stimulus-presentation intervals; running speed (`timestamps`, `speed`); eye tracking (`timestamps`, processed/raw areas, ellipse width/height, blink flag); and metadata.
- Representative experiment 775614751: 89 valid cells, 149,472 neural frames, median frame interval 0.03231 s (~31 Hz), 1,117 trial attempts, 13,793 stimulus rows, 287,868 running samples, and 144,962 eye samples. Neural trace/event lengths exactly equal ophys timestamp length.
- Stimulus tables contain multiple blocks (task, gray screen, natural movie). Task rows are identified by `stimulus_block_name` containing `change_detection`; representative task images included eight image identities plus omitted flashes.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (experiment-cell records, local) | 42,147 |
| Neurons / experiment | min 4; median 66; mean 148.40; max 666 |
| Subjects | 38 |
| Sessions / subject | 247 sessions across 38 mice (mean 6.50) |
| Ophys sessions | 247 |
| Ophys experiments / planes | 284 |
| Trials (all attempts) | 148,231 |
| Requested valid trials (go + catch) | 74,476 |
| Valid trials / session | mean 301.52 (go mean 263.68; catch mean 37.85) |
| Go outcomes | 14,194 hit + 50,934 miss = 65,128 |
| Catch outcomes | 846 false alarm + 8,502 correct reject = 9,348 |

### Available Variables and Types
- Trial-static: initial/change image, go/catch, hit/miss/false-alarm/correct-reject, aborted, auto-rewarded, trial start/stop/change times, response/reward times, lick arrays.
- Stimulus time-varying: image identity, onset/end, `is_change`, omitted, novelty, active/task block.
- Behavior time-varying: running speed (float), processed pupil area and ellipse dimensions (float), and blink validity (bool).
- Neural: precomputed dF/F and inferred/filtered event arrays (float, one value per ophys frame).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Behavioral cohort | 376 imaging sessions, 82 mice | Paper/methods: dataset "contains behavior from 376 imaging sessions from 82 mice." |
| Focused excitatory cohort | 8,619 cells; 21 imaging sessions; 9 mice | Paper/methods neural results cohort. |
| Focused Sst cohort | 470 cells; 15 imaging sessions; 6 mice | Paper/methods neural results cohort. |
| Focused Vip cohort | 1,239 cells; 21 imaging sessions; 9 mice | Paper/methods neural results cohort. |
| Single-plane acquisition | 31 Hz | Whitepaper: 512x512 movies at 31 Hz for single plane. |
| Multiplane acquisition | 11 Hz per plane | Whitepaper: 512x512 movies at 11 Hz for each multiplane plane. |
| Stimulus cycle | 250 ms image + 500 ms gray = 750 ms | Paper task description. |
| Behavioral response window | 150-750 ms after change/sham change (before display-lag compensation) | Whitepaper/methods. |
| Change-time distribution | truncated exponential 2.25-8.25 s; mean about 4.2 s after flash alignment | Methods task description. |
| Catch fraction | about 12.5% in later matrix-sampling sessions | Methods task description. |
| Omission fraction | 5% of repeats | Paper task description. |
| Neural analysis representation | detected calcium events with time and magnitude | Paper/methods: "For all analysis of neural data we used the detected calcium events." |
| Image-response interval | 50-800 ms after presentation | Paper methods uses 50 ms delay to account for signal propagation. |
| Published decoding interval | first 400 ms after image presentation | Paper Figure 6 caption. |
| Engagement threshold | reward rate >2 rewards/min | Methods behavioral processing. |

### Processing Details
- The experiment is go/no-go visual change detection. A natural image appears for 250 ms followed by 500 ms gray; one image repeats until an identity change. Premature licking resets/delays the change and creates an aborted attempt.
- Go trials contain a real identity change; catch trials contain a sham change. Combined with behavior, requested outcomes are hit, miss, false alarm, and correct reject. Auto-rewarded and aborted trials are not valid choice trials.
- Behavioral analysis assigns events to each 750 ms image-presentation interval. Omitted flashes use the corresponding 750 ms interval.
- Reference neural analyses use inferred discrete calcium event magnitudes, not newly computed dF/F. Event extraction removes slow GCaMP decay dynamics. The whitepaper notes sampling-rate-aware noise estimation and compares 31 Hz to downsampled traces.
- For image response analyses, the paper uses 50-800 ms to account for neural latency. Its decoder uses the first 400 ms, concatenating each neuron's event samples per image.
- Running is open-loop and does not control task progression.
- Eye tracking fits pupil/eye/corneal-reflection ellipses. Processed `pupil_area` treats the observed ellipse major axis as the diameter of an underlying circular pupil. Frames marked likely blink (fit missing or area z-score >3, expanded by two adjacent frames) are represented as NaN.
- Single-plane and multiplane data have different native frame rates, so no fixed sample count can represent the same physical interval without temporal resampling/binning.

### Curation Steps

**Neuron curation rules**:
- Experiments undergo Allen Institute session QC. ROIs that are unions, duplicates/ghosts, non-cell objects, or otherwise invalid are excluded. AllenSDK `cell_specimen_table` already exposes only valid ROIs.
- The paper's focused biological analyses use selected familiar-image, multiplane sessions, but the decoder task requests the supplied Visual Behavior recordings generally; the implications are resolved in Steps 4-5.

**Trial curation rules**:
- Include go and catch only; exclude aborted trials (lick before change) and auto-rewarded trials.
- Valid outcome classes are hit, miss, false alarm, and correct reject.
- Reference SDK performance metrics also exclude aborted trials.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Image change vs immediately preceding repeat | Cross-validated % correct shown graphically in Figure 6A; no exact numeric table in text. |
| Hit vs miss on image changes | Cross-validated % correct shown graphically in Figure 6C; no exact numeric table in text. |

The paper used 5-fold cross-validated random forests and reports means/SEM over imaging planes. Its decoder tasks/windows differ from the required multi-output neural decoder, so these graphical results are qualitative alignment checks rather than exact expected accuracies.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Stimulus blocks | SDK warns to select block names containing `change_detection` | Each file also has gray screens and natural movie; active block is `change_detection_behavior`, passive is `change_detection_passive` | Paper analyzes task image presentations and states passive viewing was not analyzed | Keep active task sessions only and use `change_detection_behavior` rows. |
| Passive trial labels | SDK creates go/catch-style rows from replayed changes even without animal behavior | Passive sessions have all rows valid-looking and zero aborted trials; outcomes are dominated by no-response labels | Paper explicitly excludes passive viewing from its analyses; required trial outcome must reflect behavior | Exclude session types containing `passive`. |
| Session versus imaging plane | Cache table has one experiment per plane; multiscope sessions have 3-7 locally passing planes | Planes in a multiscope session share identical trial times and ophys timestamps but contain different cells/areas | Paper samples and reports decoding over imaging planes | Treat each `ophys_experiment_id` as one decoder session; do not merge planes or duplicate cells across areas. |
| Neural representation | SDK offers dF/F, raw events, and filtered events | All have one value per ophys frame | Paper says all neural analyses use detected calcium events | Use SDK `filtered_events`, preserving event magnitude and avoiding slow fluorescence decay. |
| Frame rate | SDK timestamps vary by rig | Single-plane approximately 31 Hz; multiscope approximately 11 Hz | Whitepaper confirms 31 Hz versus 11 Hz | Resample/bin by physical time onto one common bin width; never equate frame indices across rigs. |
| Pupil variable | SDK exposes processed area and ellipse dimensions, not a column named diameter | `pupil_area` is NaN around likely blinks | Whitepaper defines processed area from a circle whose diameter is ellipse major axis | Derive diameter as `2*sqrt(pupil_area/pi)` and preserve NaN until interpolation/validity handling. |
| Cohort sizes | Full project metadata has 1,936 experiments/107 mice | Supplied cache contains 284 experiment files; active subset has 202 experiments/174 ophys sessions/38 mice | Paper reports other analysis-specific cohorts (376 sessions/82 mice; smaller neural subsets) | Convert all locally supplied active task experiments; explicitly distinguish release, provided, and paper-specific cohorts. |

### Final Consistent Understanding
- Input data are a curated local subset of the public release, not the exact focused cohort used for every paper figure.
- Active subset: 202 imaging-plane experiments, 174 unique behavioral/ophys sessions, 38 mice, and 29,444 valid experiment-cell records. Unique behavioral sessions contain 44,892 valid go/catch trials (39,265 go and 5,627 catch), with 14,194 hit, 25,071 miss, 845 false alarm, and 4,782 correct reject.
- Each experiment is independently decodable and has at least one curated imaging plane. Shared behavioral streams in multiscope experiments are expected, while neural populations differ.
- AllenSDK handles release loading, timestamp corrections, trial classification, ROI validity, event filtering, and blink-cleaned pupil area. Conversion should not duplicate these native processing steps.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `events.filtered_events` + `ophys_timestamps` | `neural` | Sum detected event magnitudes in consecutive 100 ms physical-time bins within each trial | `BehaviorOphysExperiment.events`, `.ophys_timestamps` | Preserves event magnitude; common physical bin across 10.7/30.9 Hz rigs. |
| none | `input` | Empty float32 array of shape `(0, n_timepoints)` | N/A | Decoder task explicitly specifies no inputs. |
| task `stimulus_presentations.image_name/start_time/end_time` | `output[0]` image identity | 0=gray/no image; 1-16=the global sorted real image names; assign only bins whose centers fall within non-omitted image intervals | `BehaviorSession.stimulus_presentations` | `omitted` is gray/no image, not an image identity. |
| task `stimulus_presentations.is_change/start_time` | `output[1]` image change | 1 in the first 100 ms bin containing each real change onset, otherwise 0 | stimulus presentation table | A one-bin event pulse at identity change onset. |
| `running_speed.timestamps/speed` | `output[2]` running speed | Linear interpolation to bin centers, then discretize using experiment-wide quintile edges | `BehaviorSession.running_speed` | Five equal-percentile bins, labels 0-4. |
| processed `eye_tracking.pupil_area/timestamps` | `output[3]` pupil diameter | `2*sqrt(pupil_area/pi)`; interpolate over valid nonblink samples to bin centers; experiment-wide quintile edges | `BehaviorSession.eye_tracking` | Three sessions with no eye data are excluded; labels 0-4. |
| trial flags `hit/miss/false_alarm/correct_reject` | `output[4]` trial outcome | Static integer vector shape `(1,)`: hit=0, miss=1, false alarm=2, correct reject=3 | `Trials` / `Trial._get_trial_data` | Exactly one class per retained go/catch trial. |
| `mouse_id` | `subjects`, `subject_idx` | Sorted unique strings and per-experiment lookup | experiment metadata | Experiment/plane is target session unit. |
| `targeted_structure` | `brain_regions`, `brain_region_idx` | Global sorted regions; repeat experiment's area index for every cell | experiment metadata | Each supplied experiment is one imaging plane in one area. |

### Key Decisions
1. **Use active sessions only**: Exclude session types containing `passive`, because replayed passive changes receive artificial no-response outcomes and the paper excludes passive data.
2. **Use one imaging plane per decoder session**: Matches paper decoding and preserves homogeneous neuron populations; multiscope planes legitimately share behavioral labels.
3. **Use SDK filtered events**: Matches the paper's detected-event neural representation and uses SDK sampling-rate-aware filtering rather than recomputing dF/F.
4. **Retain SDK go/catch trial boundaries**: Keep only `go | catch`, which inherently excludes aborted and auto-rewarded rows. Trial matrices may have variable timepoint counts, while all use the same 100 ms bins.
5. **100 ms bins**: This common physical bin has roughly 1 sample for multiscope and 3 for single-plane data, provides enough temporal resolution for 250 ms images and the one-bin change pulse, and greatly limits memory. Event values are summed (not averaged) so each bin represents detected event magnitude in that duration.
6. **No alignment-event cropping**: Trials are segmented at native `start_time` and `stop_time`; the required segmentation is trial-based, not a fixed event-centered analysis. Metadata alignment is trial start (`off_start=0`, variable `off_end=None`).
7. **Image gray class**: The required identity is the image presented during non-gray screen. Gray periods and omitted flashes are class 0. Sixteen image names across sets A/B are globally stable classes.
8. **Behavior quintiles are fitted per experiment**: This yields equal percentile bins within each recording and avoids cross-rig calibration/units differences. Repeated quantile edges are handled monotonically; digitized labels are clipped to 0-4.
9. **Pupil diameter and missing values**: Use whitepaper-defined processed pupil area. Interpolate only over SDK-valid points; do not use raw blink-contaminated values. Exclude the three active experiments with no eye samples (795953296, 833631914, 806456687), because an entire required output cannot be imputed responsibly.
10. **Trace edges**: Use bins fully anchored to trial start, with `ceil((stop-start)/0.1)` centers and clip final edge to stop. Nearest stream interpolation is not used for categorical stimuli; interval membership prevents onset smearing.
11. **Storage types**: Neural float32, empty input float32, categorical outputs int64, indices int64. This is validator-compatible and memory efficient.

### Planned Sanity Checks
- [ ] Assert all traces per trial share exactly the same `n_timepoints` and all values are finite.
- [ ] Assert event aggregation against an independent direct timestamp mask using `np.allclose()` for selected neuron/trial/bin.
- [ ] Assert interpolated running/pupil values and resulting bins against independently loaded SDK streams using `np.allclose()`.
- [ ] Assert image interval labels/change pulses against raw SDK stimulus rows using `np.allclose()`.
- [ ] Assert retained trial count equals SDK `(go | catch).sum()` for every included experiment and outcomes sum to trial count.
- [ ] Assert each included session has at least two trials and one cell; excluded sessions are only passive or missing the mandatory pupil stream.
- [ ] Check global image classes, output ranges, quintile fractions, frame-rate groups, and subject/region mappings.
- [ ] Compare active/source and converted counts, accounting explicitly for multiplane duplication and three eye-data exclusions.

### Efficiency Plan
- Load each experiment once through AllenSDK and construct all arrays in that pass.
- Precompute experiment-wide interpolated behavior and percentile edges once, not per trial.
- Use `searchsorted`, `reduceat`/bin indices, and interval slicing rather than per-timepoint Python loops.
- Initial exhaustive SDK scan took ~15 minutes for 202 active files; conversion should avoid its redundant table accesses and is expected to be similar or faster.

---

## Step 6: Script Development
**Status**: COMPLETE

`/app/convert_data.py` was created and passes `python3 -m py_compile`. It supports `--full` (default behavior), `--sample`, and `--show-processing`, and writes the requested pickle structure. All experiment data are loaded through `VisualBehaviorOphysProjectCache.get_behavior_ophys_experiment`; no direct NWB reader is used.

Implementation includes explicit assertions for event/timestamp lengths, finite values, exclusive outcomes, trial counts, and matching per-trial dimensions. Processing plots show event activity, image/change alignment, continuous and discretized running, continuous and discretized pupil, and event distributions.

Code inefficiencies identified:
- AllenSDK object creation dominates runtime (~15 minutes for a prior exhaustive 202-file scan).
- Repeated per-bin/per-cell masks would be expensive, and retaining SDK objects would inflate memory.

Code speedups added:
- Each experiment is loaded exactly once and all source tables/streams are reused.
- Event aggregation uses timestamp `searchsorted` plus cumulative sums across all cells.
- Continuous streams and percentile edges are prepared once per experiment.
- Only overlapping stimulus rows are iterated per trial; categorical labels use interval membership.
- Arrays are stored as float32/int64 and SDK experiment references are released after each session.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (experiment-cell records) | 101 |
| Neurons / session | 12, 89 |
| Subjects | 2 |
| Sessions / subject | 1 each |
| Trials (total) | 248 |
| Trials / session | 209, 39 |
| Timepoints / trial | 73-126 (100 ms bins) |
| Image identity distribution | gray 0.672; eight set-A images each 0.034-0.048 |
| Image change distribution | 0: 0.990; 1: 0.010 |
| Running quintiles | [0.210, 0.213, 0.224, 0.202, 0.152] |
| Pupil quintiles | [0.169, 0.258, 0.231, 0.170, 0.171] |
| Trial outcomes | hit 0.372, miss 0.494, false alarm 0.031, correct reject 0.102 (time-weighted validator fractions) |
| Neural finite values | 100% |

### Processing Plots Review
- `processing_951980471.png` (multiscope, 10.73 Hz) and `processing_775614751.png` (single-plane, 30.95 Hz) were created.
- Plots include event matrices, interval-derived image labels, one-bin change pulses, continuous behavior, quintile labels, and event histograms on the same 100 ms trial grid.
- Manual array inspection confirmed every trial has matching `(neural T, input T, output T)`, image classes occur only during image intervals, and outcome is constant within trial.
- No nonfinite converted values were found.

### Format Validation
- `/app/verification_sample_out.txt`: "Data format is valid, no errors or warnings."
- Input dimension 0 and output dimension 5 match the task.
- Output ranges are valid: image 0-15 in this image-set-A sample, change 0-1, behavior bins 0-4, outcome 0-3.
- Not all global image/outcome classes need occur in two sampled sessions; full `output_values` remains the stable global mapping.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| One SDK load per experiment | Avoids repeated ~seconds-long file reads |
| Cumulative-sum event binning | Avoids neuron x bin masking loops |
| Precomputed behavior edges | Avoids per-trial quantile calculations |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Sample conversion | 6.6 s mean including startup/plots | naive ~22 min for 199 sessions |
| Prior metadata/stream scan | ~4.5 s/session | ~15 min |

The full run will be monitored; plotting is disabled and conversion reuses loaded streams, so expected runtime is near 15-22 minutes. If runtime exceeds 1.5x this estimate, it will be stopped and optimized.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None
- Loss decreased monotonically from approximately 1.62 to 1.49 over 200 epochs; test loss was 1.5615.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Uniform Chance |
|--------|-----------------------|-------------------------|----------------|
| Image identity | 0.1982 | 0.1804 | 0.0588 |
| Image change | 0.5429 | 0.5183 | 0.5000 |
| Running speed quintile | 0.2308 | 0.2210 | 0.2000 |
| Pupil diameter quintile | 0.2318 | 0.2032 | 0.2000 |
| Trial outcome | 0.2874 | 0.2578 | 0.2500 |

All outputs exceeded nominal chance. Image identity is about 3.1x chance. The small margins for change, pupil, and outcome are not treated as final performance estimates because this sample has only two sessions, one with 12 neurons and one with only 39 valid trials.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 2.902 GB (2.8 GiB)
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; verification completed without errors

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Subjects | paper cohort 82 | cache metadata | provided active subset 38 | 38 | Yes for supplied subset |
| Decoder sessions | paper reports selected imaging planes | experiment = imaging plane | 202 active experiments | 199 after 3 missing-eye exclusions | Yes, documented exclusions |
| Unique ophys sessions | paper cohorts differ | ophys session metadata | 174 active | 171 after eye exclusions | Yes |
| Experiment-cell records | focused paper cohorts differ | valid ROIs only | 29,444 active | 29,168 after eye exclusions | Yes |
| Neurons/session | N/A | valid cell table | active min 4, median 63.5, max 666 | min 4, median 64, mean 146.57, max 666 | Yes |
| Trials | task requires go+catch | SDK `go|catch` | per retained experiments | 51,075 plane-trials | Yes |
| Trials/session | N/A | SDK trial table | min at least 39 | min 39, median 263, mean 256.66, max 409 | Yes |
| Trial outcomes | hit/miss/FA/CR | mutually exclusive SDK flags | retained source rows | [15,682, 28,990, 920, 5,483] | Yes |
| Change pulses | one per real go change | `is_change` task rows | go outcomes = 44,672 | 44,672 | Exact |
| Frame rates | 31 Hz single; 11 Hz/plane multiscope | ophys timestamps | two rate groups | ~30.94 and ~10.73 Hz | Yes |
| Regions | V1 and LM in paper | targeted structure | VISp/VISl | VISp 29,006; VISl 162 cells | Yes |
| Image range | two sets of eight | task stimulus table | 16 identities + gray | global 0-16 | Yes |
| Behavior output range | five percentile bins required | continuous SDK streams | finite streams | both 0-4 | Yes |

### Validation Findings
- All converted arrays are finite and every session has at least 39 trials.
- The validator reports warnings for some all-zero neural trial matrices. These are legitimate no-detected-event intervals from sparse SDK `filtered_events`, especially in low-cell planes. They cannot be "fixed" without fabricating activity or applying unsupported activity-dependent trial filtering; they are retained and quantified in Step 10.
- Full conversion initially projected too slowly because a session-wide event cumulative sum was recomputed per trial. The run was stopped, this was moved outside the trial loop, syntax was rechecked, and conversion was restarted. Runtime improved from 20-33 s to typically 4-10 s/session; final runtime was 1,109.2 s (18.5 min).
- Multiscope behavioral trials appear once per independently decoded plane, matching the paper's imaging-plane decoder unit. Therefore converted plane-trial totals intentionally exceed unique behavioral-session totals.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: Full format verification completed with no errors. The only warnings were 2421 all-zero neural trial matrices (4.74% of 51,075 trials). These are source-faithful intervals with no detected events across sparse populations, not missing data. Removing them would apply unsupported activity-dependent selection; adding values would fabricate neural activity. They are retained.
2. **Independent neural sanity check**: Loaded original experiments 775614751 (~31 Hz) and 951980471 (~10.7 Hz) directly through `VisualBehaviorOphysProjectCache`, independently summed source `filtered_events` selected by timestamp masks in first, interior, and final bins, and compared with converted values using `np.allclose(rtol=1e-5, atol=1e-6)`. Both passed.
3. **Independent input sanity check**: Confirmed converted input shape equals `(0,T)` for raw trial length in both source experiments. This exactly implements the explicit no-input task specification.
4. **Independent output sanity checks**: Reconstructed image interval labels, change-onset pulses, interpolated running/pupil quintiles, and trial outcomes directly from freshly loaded SDK tables without calling conversion functions. Every comparison passed `np.allclose()` for both rig types.
5. **Reference code comparison**:

| Processing stage | Conversion logic | Reference/SDK logic | Comparison |
|------------------|------------------|---------------------|------------|
| Data loading | `VisualBehaviorOphysProjectCache.from_local_cache` and `get_behavior_ophys_experiment` | Required SDK cache API | Exact |
| Neuron filtering | SDK `cell_specimen_table`/events rows | SDK exposes only valid ROIs; whitepaper excludes invalid/non-cell ROIs | Exact |
| Trial filtering | `trials[go | catch]`, active sessions only | SDK flags make aborted/auto-rewarded non-go/catch; paper excludes passive and aborted | Exact |
| Temporal alignment | Source stream timestamps mapped to absolute 100 ms trial-start grid | SDK timestamps are corrected stream times; task requires ophys temporal basis | Consistent; physical resampling required by mixed rig rates |
| Neural processing | Sum SDK `filtered_events` magnitudes in bins | Paper uses detected calcium events; SDK supplies filtered event magnitudes | Consistent |
| Input construction | Empty `(0,T)` | Decoder task says no inputs | Exact |
| Image output | Task-block real-image intervals; gray/omitted class 0 | SDK release warning requires change-detection block; paper defines 250 ms image/500 ms gray and omissions | Exact |
| Change output | One-bin pulse at source `is_change` onset | SDK presentation change flag | Exact |
| Running/pupil | SDK processed streams, physical-time interpolation, within-experiment quintiles | SDK running timestamps and blink-cleaned pupil area; task requires five percentile bins | Consistent; discretization is decoder-specific |
| Outcome | Repeat one SDK static class across T | SDK mutually exclusive trial outcome flags; task says static per trial | Semantically exact and compatible with joint time-varying output matrix |

6. **Key statistics comparison**: 38 supplied active-subset mice; 199 retained imaging planes; 171 unique ophys sessions; 29,168 valid cells; 51,075 plane-trials. Regions, frame-rate groups, cell ranges, and session types match the source tables after exactly the documented passive/missing-eye exclusions. Go outcomes total 44,672 and exactly equal converted change pulses.
7. **Edge-case review**: Tested clipped final bin directly; `searchsorted(..., side='left')` assigns frames to half-open bins without double counting. Stimulus intervals are half-open `[start,end)`. Change at an exact boundary maps by floor to the bin immediately after onset. Trials use `ceil(duration/bin)` so no tail is dropped. Static outcome is constant over all T. Every retained session has >=39 trials and >=4 cells.
8. **Data integrity**: All converted neural/output values are finite; categorical ranges match `output_values`; subject and region indices are in bounds; session lists have equal lengths.

### Issues Found and Resolved
- **Severe initial runtime bottleneck**: A session-wide event cumulative sum was mistakenly recomputed once per trial, projecting >60 minutes. It was moved outside the trial loop, all affected conversion/validation checks were rerun, and full runtime fell to 18.5 minutes.
- **Passive sessions carried misleading outcomes**: Resolved before conversion by excluding passive session types in agreement with the paper.
- **Three active files had no eye data**: Excluded because pupil diameter is mandatory and whole-session imputation is unjustified.
- **Sparse all-zero event trials**: Verified as valid source behavior and retained, with rationale above.

`/app/sanity_step10_out.txt` records the independent source comparisons and ends with `ALL SANITY CHECKS PASSED`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes, monotonically from 1.6341 at epoch 1 to 1.5159 at epoch 200.
- Test loss: 1.5676.
- Device: CUDA.
- Split: 40,786 training trials and 10,289 validation trials.
- `/app/train_decoder_full_out.txt` ends with `train_decoder.py finished successfully.`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-----------------------|-------------------------|--------|-------|
| Image identity | 0.3000 | 0.2909 | 0.0588 | 4.95x chance validation |
| Image change | 0.5961 | 0.5630 | 0.5000 | Above chance; sparse one-bin event |
| Running speed quintile | 0.2603 | 0.2531 | 0.2000 | 1.27x chance |
| Pupil diameter quintile | 0.2702 | 0.2573 | 0.2000 | 1.29x chance |
| Trial outcome | 0.3545 | 0.2907 | 0.2500 | 1.16x chance |

All outputs are above chance. Training/validation ratios are 1.03, 1.06, 1.03, 1.05, and 1.22 respectively, so none exceed the 1.5 overfitting threshold.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Validation Accuracy | Chance | Accuracy / Chance | Expectation from Papers |
|----------|---------------------|--------|-------------------|-------------------------|
| Image identity | 0.2909 | 0.0588 | 4.95x | Not decoded in the reference paper; strongly above chance. |
| Image change | 0.5630 | 0.5000 | 1.13x | Paper decodes balanced change vs immediately preceding repeat over the first 400 ms; exact values are only graphical. Required output is a much sparser one-bin event over full trials. |
| Running speed quintile | 0.2531 | 0.2000 | 1.27x | No running decoder reported in the paper. |
| Pupil diameter quintile | 0.2573 | 0.2000 | 1.29x | No pupil decoder reported in the paper. |
| Trial outcome | 0.2907 | 0.2500 | 1.16x | Paper decodes balanced hit vs miss only at image changes; required output has four classes including rare catch outcomes and is static per trial. |

### Checks Performed
1. **Accuracy versus chance**: Every output exceeds chance. Image identity is nearly 5x chance. Change, running, pupil, and outcome are below 1.5x chance and were therefore investigated rather than accepted without review.
2. **Three raw trial checks**: Fresh AllenSDK loading of experiment 775614751 verified trials 0, 19, and 38. The trials included go/miss, catch/false-alarm, and go/hit examples. Converted outcome classes and change-pulse counts exactly matched source flags/presentations.
3. **Temporal alignment plot**: `/app/cache/accuracy_debug_alignment.png` overlays mean neural event activity, image identity, image-change pulse, running bin, and pupil bin on the common trial-start time axis. It shows no temporal shift or edge artifact.
4. **Output variation**: Full timepoint class maxima were image identity 0.669 (gray), change 0.990 (expected sparse event), running 0.207, pupil 0.223, and outcome 0.571. Running and pupil bins are well populated; trial-level outcomes are [15,682, 28,990, 920, 5,483].
5. **Neural processing**: Reconfirmed use of SDK `filtered_events`, matching the paper's detected calcium-event representation and avoiding slow dF/F decay.
6. **Train/validation gap**: Ratios are image 1.03, change 1.06, running 1.03, pupil 1.05, and outcome 1.22. None approaches the 1.5 threshold, so there is no evidence of severe overfitting or leakage.
7. **Paper comparison**: The reference paper reports only graphical random-forest results for balanced image-wise change/repeat and hit/miss tasks. It does not report image-identity, running, pupil, or four-way outcome decoding. Exact numerical comparison is therefore unavailable. The required decoder uses full variable-length trials and one-bin change events, making direct equality inappropriate.

### Interpretation of Lower-Margin Outputs
- **Image change** is positive at only 44,672 of 4,374,171 timepoints (1.02%). Balanced accuracy and balanced loss prevent majority-class inflation, but a single 100 ms pulse is intrinsically harder than the paper's balanced 400 ms image snippets. Expanding the pulse would violate the instruction to use value 1 "right after" a change and would distort timing.
- **Running and pupil** are five-way within-session percentile labels. Their near-uniform class occupancy rules out class-collapse bugs. Above-chance decoding is plausible but modest because behavior is not deterministically encoded by every small imaging plane.
- **Trial outcome** includes rare false alarms and correct rejects, unlike the paper's hit/miss-only decoder. It is constant across each trial and source checks passed, so low margin is not due to output jitter or misalignment.

### Issues Found and Resolved
- No conversion bug was found in the low-accuracy investigation. All prescribed raw checks, timing checks, variation checks, and processing comparisons passed.
- The investigation script and output are retained as `/app/cache/accuracy_debug.py` and `/app/cache/accuracy_debug_out.txt`.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

### Finalization
- `README.md` documents loading, structure, mappings, key statistics, reproduction commands, warnings, and decoder results.
- Investigation scripts, outputs, and the alignment plot are organized under `/app/cache/` and described by `README_CACHE.md`.
- All required conversion, sample/full validation, and sample/full training artifacts exist.
- `CONVERSION_NOTES.md` contains decisions, source comparisons, consistency statistics, warning rationale, independent raw-data checks, runtime optimization history, and decoder accuracy reviews for all workflow steps.
