# Dataset Conversion Notes

## Overview
- **Dataset**: Allen Brain Observatory Visual Behavior 2P, local AllenSDK cache
- **Date started**: 2026-09-20
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`, `Dockerfile`, `docker-compose.yaml`
- `whitepaper.pdf`, `paper.pdf`, `methods.txt`
- `code/`, `tutorials/`, `allensdk_docs/`
- `data/`
- `decoder.py`, `train_decoder.py`
- `CONVERSION_NOTES.md`

Environment checks: Python 3 runs successfully; NumPy 2.4.4 and PyTorch 2.6.0+cu124 import successfully. Checkpoint verified with `ls -la /app/CONVERSION_NOTES.md`.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `VisualBehaviorOphysProjectCache.from_s3_cache` | `behavior_project_cache/project_cache_base.py` / cache class | LOADING | Instantiate the official cloud/local cache interface. |
| `get_ophys_experiment_table` | `behavior_project_cache/behavior_project_cache.py` | LOADING | Return experiment metadata indexed by ophys experiment ID. |
| `get_behavior_ophys_experiment` | same | LOADING | Return a `BehaviorOphysExperiment` through the SDK API. |
| `BehaviorOphysExperiment.ophys_timestamps` | `behavior_ophys_experiment.py` | LOADING | Microscope frame timestamps; master temporal grid required by task. |
| `BehaviorOphysExperiment.events` | same | PROCESSING | SDK-provided inferred and filtered calcium events. |
| `BehaviorOphysExperiment.dff_traces` | same | PROCESSING | SDK-provided ΔF/F; no recomputation is needed. |
| `BehaviorOphysExperiment.cell_specimen_table` | same | CURATION | Contains only `roi_valid=True` ROIs, so invalid/non-cell segmentations are already excluded. |
| `BehaviorSession.trials` | `behavior_session.py` | CURATION | Trial bounds and go/catch/aborted/auto-rewarded/outcome flags. |
| `BehaviorSession.stimulus_presentations` | same | PROCESSING | Image presentation intervals, image identity, change/omission flags, and trial IDs. |
| `BehaviorSession.running_speed` | same | PROCESSING | SDK-filtered speed (cm/s) with synchronized timestamps. |
| `BehaviorSession.eye_tracking` | same | PROCESSING | Filtered eye/pupil ellipse metrics; blink/outlier-derived values are NaN. |
| `trial_masks.contingent_trials` | `trial_masks.py` | CURATION | Selects go and catch trials. |

### Notes
- The cache is the required entrypoint; detailed session objects must be obtained using `get_behavior_ophys_experiment`, never by opening NWB files directly.
- Neural modality is two-photon calcium imaging, not electrophysiology. SDK files already contain computed ΔF/F and inferred events. The paper analysis is expected to determine which representation to use; ΔF/F must not be recomputed.
- The SDK explicitly guarantees `cell_specimen_table` has invalid ROIs removed (`roi_valid=True`).
- `ophys_timestamps` index both ΔF/F and event arrays and therefore provide exact neural-frame alignment.
- The modern stimulus table includes several blocks; the SDK warning directs users to the active `change_detection` block (or equivalently active task rows), rather than unrelated passive/movie blocks.
- The trial schema distinguishes go, catch, aborted, and auto-rewarded trials and supplies mutually interpretable outcome flags: hit, miss, false alarm, correct reject.
- Filtered running speed is preferred to `raw_running_speed`; eye tracking already masks likely blinks and ellipse-fit outliers in processed columns.
- No additional electrophysiology quality filter applies.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Allen cache root: `/app/data`; release directory `visual-behavior-ophys-1.1.0` plus `visual-behavior-ophys_project_manifest_v1.1.0.json`.
- `project_metadata/` contains SDK-managed experiment, session, behavior-session, and cell tables. `behavior_ophys_experiments/` contains 284 cached NWB assets. These were enumerated by filename only; their contents were accessed exclusively through `VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir='/app/data')` and `get_behavior_ophys_experiment`.
- Full release metadata exposed by the SDK: 1,936 experiments, 703 ophys sessions, and 133,066 cell-table rows. Locally cached subset: 284 experiments representing 247 ophys/behavior sessions and 38 mice. Some multiscope sessions have several simultaneous plane experiments.
- Cached subset composition: 239 `VisualBehavior`, 45 `VisualBehaviorMultiscope`; 202 active-behavior experiments and 82 passive-viewing experiments; experience levels 150 Familiar, 38 Novel 1, 96 Novel >1. Regions: 261 VISp and 23 VISl experiments. Cre lines: 153 Slc17a7, 85 Sst, 46 Vip.
- One SDK-loaded example had 149,472 ophys frames, 89 valid cells, 1,117 trial rows, 13,793 stimulus-presentation rows, 287,868 running samples, and 144,962 eye samples.
- Native SDK variables and types: `ophys_timestamps` float64 vector; `dff_traces.dff`, `events.events`, and `events.filtered_events` per-cell arrays; trial DataFrame with time bounds, image names, change time, go/catch, four outcome flags, aborted and auto-rewarded; stimulus table with interval bounds, image name/index, change/omission/active flags and trial ID; running table (`timestamps`, `speed`); eye table including timestamp, processed/raw pupil area, pupil ellipse width/height, and `likely_blink`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 42,147 valid ROI entries across 284 cached experiments |
| Neurons / session | min 4, mean 148.40, median 66, max 666 |
| Subjects | 38 |
| Sessions / subject | experiments: min 4, mean 7.47, median 6, max 45 |
| Trials (total) | 148,231 raw trial rows over 247 unique source behavior sessions; 74,476 go+catch. Active subset: 118,636 raw / 44,892 go+catch over 174 unique behavior sessions. |
| Trials / session | all unique source behavior sessions: raw mean 600.13 (399–1,241); active: raw mean 681.82 (400–1,241), contingent mean 257.89 |

Outcome counts over all unique cached behavior sessions: 65,128 go, 9,348 catch, 14,194 hit, 50,934 miss, 846 false alarm, 8,502 correct reject. Counts are native metadata statistics; final counts will differ because passive experiments and auto-rewarded trials require task-specific exclusion, and simultaneous plane experiments are separate neural recording entries.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 34,619 cortical cells in complete release; paper analysis subset 8,619 excitatory + 470 Sst + 1,239 Vip | Whitepaper p.3: release has “34,619 cortical cells”; paper Methods reports subset by class. | 
| Neurons / session | Paper subset: 10,328 cells / 57 imaging sessions (cell entries, not necessarily unique longitudinal IDs) | Paper Methods: 21 excitatory, 15 Sst, 21 Vip sessions. |
| Subjects | 82 complete release; paper neural subset 24 class-count mouse entries (9 excitatory, 6 Sst, 9 Vip) | Whitepaper p.3 and paper Methods. |
| Sessions / subject | Release: 551 imaging sessions / 82 mice; a container has 3–11 sessions | Whitepaper pp.3,7. |
| Trials (total) | Not reported for the paper neural subset | Reference texts define trials but do not tabulate total. |
| Trials / session | ≥100 hit+miss trials in each of 3 consecutive sessions required for transition readiness; task duration ~1 h | Whitepaper task/training and acquisition sections. |
| Neural data time bin | Native microscope frames: ~31 Hz single-plane; ~11 Hz per plane multiscope | Whitepaper synchronization/acquisition section. |
| Behavior data time bin | Eye and behavior ~30 Hz; stimulus flash cycle 750 ms | Whitepaper: 30 Hz streams; paper: 250 ms image + 500 ms gray. |
| Reward rate | Engagement threshold >2 rewards/min (25-trial rolling measure) | Methods/whitepaper task metrics. | 
| Catch probability | ~12.5% after matrix sampling in stage 3+ | Whitepaper trial structure. | 
| Omission probability | 5%; changes and preceding flashes never omitted | Whitepaper and paper. |
| Response window | 150–750 ms after non-display-lag-compensated change/sham-change | Whitepaper behavioral metrics. |


### Processing Details
- Task: go/no-go visual change detection with eight natural images. Each image appears for 250 ms, followed by 500 ms gray; image identity repeats until a go-trial change. Catch trials contain a sham change but no identity change. Premature licks abort/reset trials. Four contingent outcomes are hit, miss, false alarm, and correct rejection.
- All experimental clocks were recorded on one 100 kHz synchronization board. Use SDK-synchronized timestamps and ophys microscope frames as the requested master timebase.
- The reference neural analysis used detected calcium events rather than ΔF/F, explicitly to remove slow GCaMP decay. It used activity in the first 400 ms after image presentations for random-forest change and hit/miss decoding.
- FastLZero event inference produces time-and-magnitude events. The unfiltered `events` stream best matches the paper’s “discrete calcium events”; `filtered_events` is documented by SDK as visualization smoothing only.
- Running speed: unwrap encoder, remove >5.1 V artifacts, correct wraps, convert angular to linear speed assuming radius position 2/3 from center, remove wrap/outlier transients, then 10 Hz low-pass Butterworth. The SDK `running_speed` property supplies this processed signal.
- Eye processing fits ellipses using DeepLabCut landmarks. Processed pupil metrics are NaN during blink/outlier frames; raw variants retain those frames. The task asks for pupil diameter, so ellipse-derived diameter should be used rather than area.
- The supplied paper reports behavior-model AUC 0.83, but that model predicts lick-bout initiation from behavioral covariates and is not comparable to the provided neural decoder. Neural change and hit/miss accuracies appear graphically in Figure 6 but are not stated numerically in extracted text; they used a different random-forest/image-window task.

### Curation Steps

**Neuron curation rules**:
Use only SDK-returned valid ROIs. Upstream QC excludes non-cells, unions, duplicates, motion-edge ROIs, dendrites, small/narrow/dim ROIs, negative/zero demixing failures (~1%), failed experiments, >10 µm z-drift, possible epileptiform recordings, and failed longitudinal registration. Do not impose a new downstream cell filter absent from the reference analysis.

**Trial curation rules**:
Decoder requirement overrides paper analysis scope: retain go and catch only, explicitly remove aborted and auto-rewarded trials. Restrict to active task experiments because passive viewing has no behavioral trial outcomes and the paper explicitly did not analyze passive sessions. Do not apply an engagement threshold unless demanded by a specific analysis; the paper states strategy effects were independent of engagement and used all active behavioral sessions for behavioral analysis.

### Decoders Trained
| Decoded variable | Accuracy |
| Lick-bout initiation (behavioral dynamic model, not neural decoder) | mean ROC AUC 0.83 |
| Image change vs preceding repeat from first 400 ms calcium events | Figure 6 graphical only; no exact text value |
| Hit vs miss from first 400 ms calcium events | Figure 6 graphical only; no exact text value |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| “Session” vs “experiment” | SDK defines a neural object per ophys experiment/imaging plane | 284 cached experiments map to 247 continuous recordings because multiscope has multiple planes | Whitepaper defines one continuous recording as a session and each imaging plane as an experiment; paper decoding is per imaging plane | Each target session will be one ophys experiment. A neural matrix cannot combine different frame clocks/regions/planes, and the paper’s unit of decoding is an imaging plane. Shared behavioral trials are therefore intentionally represented once per plane. |
| Neural representation | SDK exposes ΔF/F, inferred `events`, and visualization-only `filtered_events` | Arrays share ophys-frame length | Paper says all neural analyses used detected calcium events to remove slow GCaMP decay | Use unfiltered inferred event magnitudes, not recomputed ΔF/F or smoothed visualization events. |
| Passive sessions | SDK contains trials copied/associated with passive stimulus blocks and labels `behavior_type` | 82/284 cached experiments are passive | Paper states passive viewing was not analyzed; requested trial outcome is undefined behaviorally during passive viewing | Retain active-behavior experiments only. |
| Trial selection | SDK table provides independent go, catch, aborted, auto-rewarded flags | Raw tables include aborted trials and free-reward trials; go+catch summary counts are contingent trials | Whitepaper describes aborted reset trials and non-contingent free rewards separately from four outcomes | Select `(go OR catch) AND NOT aborted AND NOT auto_rewarded`; require exactly one of hit/miss/false_alarm/correct_reject. |
| Cell counts | SDK cells metadata includes valid ROI entries for cached experiments | 42,147 entries, which exceeds 34,619 release unique cortical cells | Whitepaper count is longitudinal unique cells, while experiment entries repeat matched cells across sessions | Preserve every valid recorded cell per experiment, as needed for within-session decoding; do not deduplicate longitudinal IDs across sessions. |
| Regions | Experiment metadata provides VISp/VISl | Cached subset is 261 VISp, 23 VISl | Paper calls these V1 and LM | Save canonical SDK names `VISp`, `VISl`; explain aliases in metadata. |
| Timing rates | Native 31 Hz single-plane / 11 Hz multiplane | Ophys timestamp spacing varies by acquisition type; behavior/eye use separate rates | References require clock synchronization, while target requires one bin size across sessions | Align all modalities by timestamps, then aggregate/interpolate to a fixed bin width (chosen in Step 5); never align by raw sample index. |
| Pupil measure | SDK offers processed ellipse area, width, height and raw counterparts | Processed values are NaN on blink/outlier frames | Whitepaper documents ellipse half-axes/area and masking; requested output says diameter | Derive equivalent circular diameter `2*sqrt(pupil_area/pi)` from processed area, preserving ellipse size while respecting SDK masking; interpolate only short/internal gaps during temporal resampling and exclude trials with irrecoverable pupil coverage. |

Final consistent understanding: the provided cache is a task-focused subset of release 1.1.0, but still includes passive sessions and both microscope types. Conversion will use active ophys experiments as independent target sessions, SDK-curated cells and unfiltered inferred events, SDK timestamps, active stimulus intervals, processed running and pupil signals, and only valid contingent trials. Dataset-wide paper totals are comparison context rather than expected equality because the local subset and decoder selection differ.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `experiment.events['events']` + `ophys_timestamps` | neural | Sum inferred event magnitudes into non-overlapping 100 ms bins bounded by each retained trial; float32 `(n_cells,T)` | `BehaviorOphysExperiment.events`, FastLZero pipeline | Unfiltered discrete events match paper; bin sum preserves event magnitude/count-like activity. |
| No variables | input | Empty float32 array `(0,T)` | N/A | Decoder Task explicitly says no inputs; `input_names=[]`. |
| Active `stimulus_presentations.image_name`, interval times | output[0] image identity | At each 100 ms bin center, categorical image name during a non-omitted 250 ms image interval; otherwise `gray` | `BehaviorSession.stimulus_presentations` | Active change-detection block only; global vocabulary is gray + all cached active image names. |
| Active `stimulus_presentations.is_change`, interval times | output[1] image change | Binary 1 at bin centers during the newly changed image's on-screen interval (~250 ms immediately after identity changes); 0 otherwise | same | Catch sham changes are not image-identity changes and remain 0. |
| `running_speed.timestamps`, `speed` | output[2] running speed quintile | Linearly interpolate SDK-filtered cm/s to bin centers, then discretize using pooled 20/40/60/80 percentiles over retained data | `BehaviorSession.running_speed`; SDK running pipeline | Five categorical bins, codes 0–4. |
| `eye_tracking.timestamps`, processed `pupil_area` | output[3] pupil diameter quintile | Equivalent diameter `2*sqrt(area/pi)`, interpolate valid processed samples to bin centers, pooled quintile thresholds | SDK eye pipeline | Blink/outlier values are already NaN. Trials outside pupil coverage are excluded; internal gaps are interpolated and tracked in metadata. |
| trial flags `hit`, `miss`, `false_alarm`, `correct_reject` | output[4] trial outcome | Map to 0–3 and repeat constant across all T bins | SDK trial schema | Static per trial represented as a constant time series so all requested outputs coexist in one `(5,T)` array. |
| `mouse_id` | `subjects`, `subject_idx` | Sorted string IDs and per-experiment index | experiment metadata | One target session per experiment/plane. |
| `targeted_structure` | `brain_regions`, `brain_region_idx` | Global sorted SDK region names; fill all cells in experiment with its region index | experiment metadata | VISp=V1, VISl=LM. |

### Key Decisions
1. **Experiment selection**: Locally cached experiments with `behavior_type == 'active_behavior'`; passive experiments cannot supply meaningful trial outcome. Require at least two successfully converted trials and at least one valid cell.
2. **Trial selection**: `(go | catch) & ~aborted & ~auto_rewarded`, finite ordered time bounds, exactly one of the four contingent outcomes, and sufficient synchronized ophys/running/pupil coverage. This directly implements the task and avoids invented outcomes.
3. **Temporal grid**: Fixed 100 ms bins (`time_bin_size=100.0` ms) for every experiment. This is close to the slower multiscope frame period (~91 ms), avoids pretending it has 31 Hz resolution, and retains multiple bins per 250 ms image and first 400 ms paper decoding window. Bin centers define categorical/continuous labels; event magnitudes are summed between bin edges using timestamps.
4. **Variable trial duration**: Use native trial `start_time` to `stop_time`; choose complete 100 ms bins whose centers lie in the interval. Trial arrays may have different T, which validator/model explicitly supports. `off_start` and `off_end` are `None` because alignment is to absolute ophys timestamps/trial boundaries rather than a fixed event window.
5. **Image labels**: Build a single deterministic global class list: `gray` followed by sorted image names across retained active stimulus rows. Omitted images and 500 ms inter-stimulus periods are genuinely gray and receive `gray`, not the preceding identity.
6. **Change interval**: Use the full stimulus presentation carrying `is_change`, not trial `change_time` independently, because this display-lag-corrected row is the newly changed image shown immediately after identity changes. At 100 ms resolution this yields 2–3 positive bins rather than an arbitrarily undersampled mathematical impulse. Assert each retained go trial has one change presentation and catch has none.
7. **Quintiles**: Calculate thresholds from all finite, temporally aligned samples in the final retained sessions/trials, so each requested continuous output has dataset-wide equal-percentile definitions. Use `np.searchsorted(thresholds, value, side='right')`; record thresholds. Repeated quantile edges, if any, are reported (not expected).
8. **Pupil missingness**: Use processed pupil area, never raw blink-corrupted values. Convert to equivalent diameter before interpolation. Do not extrapolate outside valid eye coverage; drop affected trials. Linear interpolation across internal missing frames avoids losing whole trials for short blinks; report missing fraction and sensitivity in checks.
9. **Neural curation**: Trust SDK `cell_specimen_table`/events rows, which already contain valid ROIs. Preserve SDK row order and assert every trace length equals timestamps. Do not filter on activity/SNR because neither paper nor decoder specification calls for it.
10. **Storage**: neural/input float32, outputs integer (int16), indices integer. Include experiment IDs, cell specimen IDs, class mappings, quintile thresholds, microscope rate, region, trial IDs, and exclusions in metadata/session_info.

### Planned Sanity Checks
- [ ] Independently reload representative raw experiment through the SDK and `np.allclose` converted neural bin sums to direct timestamp-selected event sums.
- [ ] `np.allclose` converted running values before discretization to direct interpolation from SDK running samples; verify quintile counts.
- [ ] `np.allclose` pupil diameters before discretization to equivalent-diameter interpolation from SDK eye samples; confirm no raw likely-blink value was used.
- [ ] Match converted image labels at selected bins to direct active stimulus interval lookup and change impulses to `is_change` onset.
- [ ] Verify every trial is contingent, non-aborted, non-auto-rewarded, has one valid outcome; go/catch and outcome totals agree with direct SDK tables after documented coverage exclusions.
- [ ] Confirm fixed 100 ms binning, monotonic edges, no off-by-one overlap, all modality lengths equal T, finite neural values, label ranges, >=2 trials/session, and constant outcome within trial.
- [ ] Compare subject/session/cell/trial counts and region/cell-class/session-type distributions against cache metadata and paper subsets; explain selection differences.
- [ ] Plot native events, binned neural activity, stimulus/image/change, continuous and discretized running/pupil for up to two sessions.

---

## Step 6: Script Development
**Status**: COMPLETE

Created `/app/convert_data.py`. It supports the required positional output, mutually exclusive `--full`/`--sample` (full is default), and `--show-processing`. Syntax compilation and CLI help completed successfully. The implementation imports the local AllenSDK and uses `VisualBehaviorOphysProjectCache.from_s3_cache` plus public session properties exclusively. It validates trace/timestamp shapes, finite events, trial flags, modality lengths, and go/catch stimulus-change consistency; errors are logged per experiment rather than silently accepted.

Code inefficiencies identified:
Potential bottlenecks are SDK/NWB object construction, stacking full-session event arrays, and repeated trial-level aggregation. Full trace loading is unavoidable through the SDK, but copying full arrays more than once would inflate memory.

Code speedups added:
Metadata selection avoids loading passive assets. Event binning uses vectorized `searchsorted` and trial-local cumulative sums rather than neuron/bin Python loops or a second full-session cumulative array. Continuous streams use vectorized interpolation. Float32 neural arrays and int16 categorical outputs reduce pickle size. Objects are released and garbage-collected after each experiment. Timing is printed per experiment and overall.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 231 experiment-cell entries |
| Neurons / session | 89, 142 (mean 115.5) |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 229 |
| Trials / session | 39, 190 |
| Timepoints / trial | min 72, mean 80.62, max 125 |
| Input dimensions | 0 (required) |
| Image identity distribution | gray .661; 16 image classes collectively .339 |
| Image change distribution | [no change .972, changed-image interval .028] |
| Running quintile distribution | [.200,.200,.200,.200,.200] |
| Pupil quintile distribution | [.200,.200,.200,.200,.200] |
| Trial outcome distribution | [hit .595, miss .274, false alarm .053, correct reject .078] |

`sample_data.pkl` is 10.2 MB. Validator reported valid format with no errors or warnings. Both sessions have >=2 trials, all requested output classes/ranges are legal, all continuous-derived outputs achieve exact pooled quintiles, and metadata records trial/cell IDs and exclusion counts. The 39-trial session is not data loss: direct trial flags show 1,078/1,117 rows are non-contingent or aborted and exactly 39 satisfy the requested selection; no coverage/stimulus consistency exclusions occurred.

### Processing Plots Review
Reviewed `processing_775614751.png` and `processing_788490510.png`. Native event peaks fall inside their corresponding summed 100 ms bins; binned activity is the expected sum rather than an interpolation. Image flashes occupy ~250 ms and are separated by ~500 ms gray. Each go change interval exactly overlays the newly changed image presentation. SDK running samples and aligned traces overlap. Processed pupil samples and aligned equivalent diameter overlap, including smooth interpolation across masked frames. Discrete quintiles track their continuous traces. No temporal shift or edge anomaly was observed. Session-specific quintile skew is expected from pooled dataset-wide thresholds; pooled fractions are exactly equal.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Four independent experiment workers | Estimated ~4x reduction of SDK loading/processing wall time; unused diagnostics removed outside plot mode. |

| Step | Time / Session | Estimated Total Time |
| SDK load + conversion | 5.32 s/experiment sequential sample | ~4.5 min for 202 selected experiments with 4 workers; allow ~6–8 min for larger files/IPC/assembly |
| Assembly + save | ~1.7 s sample including plots | Expected <2 min; total remains below 15 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None from format validation. During scoring, sklearn notes that a predicted class was absent from the small validation truth split; this is a sample-size/class-rarity metric warning, not a data-format warning.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| image_identity | 0.3114 | 0.2497 |
| image_change | 0.7076 | 0.6496 |
| running_speed_quintile | 0.2665 | 0.2529 |
| pupil_diameter_quintile | 0.2938 | 0.2356 |
| trial_outcome | 0.3402 | 0.2804 |

Loss decreased monotonically from 1.6374 to 1.4177 over 200 epochs; test loss was 1.5233. Every validation balanced accuracy exceeds uniform chance (image identity 1/17=.0588; change .5; running .2; pupil .2; outcome .25).

Iteration: the initial one-bin change-onset representation produced validation balanced accuracy .4916, slightly below chance. Review against the SDK and paper showed `is_change` labels the newly presented image interval and paper decoding uses post-presentation neural activity. The output was corrected to 1 across bin centers in that ~250 ms on-screen interval (“right after” identity changes), not just a single undersampled onset bin. Steps 7–8 were rerun fully; format remained warning-free and change accuracy rose to .6496. No other mapping changed.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 2.740 GB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Total neurons | Release 34,619 unique; paper subset 10,328 entries | Valid ROI/event rows | Active cache 29,302 entries before pupil selection | 29,168 experiment-cell entries | Yes for subset |
| Mean neurons/session | Paper subset ~181 | No extra filter | Active cache 145.1 | 146.57 (median 64; 4–666) | Yes |
| Subjects | Release 82 | `mouse_id` | Active cache 38 | 38 | Yes |
| Sessions | Release 551; paper subset 57 planes | One target session/plane | 202 active experiments | 199; 3 lacked pupil | Yes after coverage selection |
| Trials | Not tabulated | Contingent, non-aborted/free reward | Direct eligible rows | 51,075 | Yes; ledgers reconcile |
| Trials/session | Criterion ≥100 hit/miss, not analysis count | Variable bounds | Performance dependent | mean 256.66 (39–409) | Plausible/direct |
| Image identity | 8/session, two sets | Active shown intervals | 16 names + gray | gray .665; images .019–.023 each | Yes |
| Image change | Changed presentation | active `is_change` | One interval/go | [.973,.027] | Yes |
| Running quintiles | SDK filtered cm/s | timestamp interpolation | pooled | [.2,.2,.2,.2,.2] | Yes |
| Pupil quintiles | SDK masked ellipse | equivalent diameter | pooled | [.2,.2,.2,.2,.2] | Yes |
| Trial outcome | Four outcomes | direct flags | eligible trials | [.303,.571,.017,.108], time-weighted | Plausible/direct |

Full conversion completed in 348.33 s (5.81 min), below estimate/cutoff. Verification had no structural errors. It warns for 2,502/51,075 trials (4.90%) having all-zero inferred events; these occur in 88 sessions and concentrate in low-cell experiments (cell-count/zero-fraction correlation -0.216; worst sessions have 4–18 cells). This is expected for paper-matched sparse FastLZero events. Removing them would select on decoder input and bias behavior; replacing events with ΔF/F or smoothed visualization events would violate the reference representation. The warnings are explicitly accepted.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log**: `verification_full_out.txt` has no errors. Only warning class is all-zero sparse-event trials (2,502, 4.90%); quantified and justified in Step 9. No legal reference-consistent fix exists without neural-activity selection bias or changing the paper-matched signal.
2. **Independent original-data sanity checks**: `/app/cache/sanity_checks.py` reloads experiments 775614751, 940352367, and 1086048031 through `VisualBehaviorOphysProjectCache`, without importing conversion code. For first/middle/last converted trial in each, it independently reconstructed and passed `np.allclose` for (a) every neuron/bin event sum, (b) running quintile after direct interpolation, (c) pupil equivalent-diameter quintile after direct interpolation, (d) image identity, (e) changed-image state and constant outcome. Result: 45/45 modality comparisons passed. It additionally audited every trial shape, finite neural values, empty inputs, output ranges, constant outcome, region-index length, and >=2 trials/session; all passed.
3. **Reference code comparison—loading**: converter uses cache `from_s3_cache`, experiment table, and `get_behavior_ophys_experiment`, matching SDK docs. No direct NWB library is imported or called.
4. **Reference comparison—filtering**: cells are exactly SDK event rows/valid ROI table (no invented SNR filter). Trials are active go/catch, non-aborted, non-auto-rewarded with exactly one outcome. This matches SDK contingent definition plus explicit task exclusions. Three experiments lacking the required processed pupil stream were excluded and logged; no fatal load errors occurred.
5. **Reference comparison—alignment**: all sources are queried by synchronized timestamps. Ophys timestamps define half-open 100 ms neural bins; behavior is interpolated to centers; displayed stimulus intervals are queried at the same centers. This follows the whitepaper synchronization model and satisfies the common-bin requirement.
6. **Reference comparison—binning**: the paper used native event samples in 400 ms image windows but target needs a uniform bin across 31/11 Hz experiments. Summed 100 ms unfiltered FastLZero magnitudes preserve discrete event signal and approximate the slowest frame period. Independent sums passed exactly.
7. **Reference comparison—inputs/outputs**: no decoder inputs as specified. Image and true change come from active stimulus rows; running uses SDK filtered speed; pupil uses processed/masked ellipse area converted to equivalent diameter; outcome uses direct trial flags. Differences from paper (time-varying image/running/pupil and quintiles) are mandated by this decoder task.
8. **Key statistics**: 38 subjects match active cache; 199/202 active experiments retained after 3 no-pupil exclusions; 29,168 experiment-cell entries; 51,075 trials; two regions; 16 image identities plus gray; exact pooled quintiles; outcome/image/change distributions listed in Step 9. Release/paper totals differ only due explicitly different local/analysis subsets and longitudinal unique-cell counting.
9. **Edge cases/off-by-one audit**: all bins are half-open `[edge_i,edge_{i+1})`; only complete bins fit inside trial bounds; output uses centers. First/middle/last trials from beginning/middle/end sessions passed. Empty inputs have exact `(0,T)` shape. All session/trial/neuron list lengths agree. Go/catch change-row consistency is asserted. Trial lengths 70–125 bins are plausible for 7.0–12.5 s contingent trials.

### Issues Found and Resolved
- Initial one-bin change impulse underrepresented the SDK `is_change` interval and decoded below chance; changed to full newly shown image interval and reran sample conversion/validation/training.
- Three active experiments (795953296, 806456687, 833631914) have no processed pupil data through the SDK; excluded entire sessions because categorical pupil output cannot be honestly imputed from absent data.
- All-zero event warnings were investigated, not suppressed: sparse inferred events and low neuron counts explain them; retained to avoid conditioning trial inclusion on neural activity.

No check found a remaining conversion mismatch; no further iteration was required after the change-interval correction already rerun in Steps 7–9.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes, monotonically from 1.634203 (epoch 1) to 1.494686 (epoch 200); test loss 1.532081.

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| image_identity | 0.2620 | 0.2557 | chance .0588 |
| image_change | 0.6278 | 0.6112 | chance .5000 |
| running_speed_quintile | 0.2930 | 0.2883 | chance .2000 |
| pupil_diameter_quintile | 0.3093 | 0.3025 | chance .2000 |
| trial_outcome | 0.3216 | 0.2760 | chance .2500 |

Required command completed successfully on GPU for 200 epochs (40,786 training and 10,289 validation trials). All validation outputs exceed uniform balanced chance. `sample_trials.png` and `predictions.png` were generated by `--plot-samples`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
|----------|-------------------|-------------------------|
| image_identity | .2557 (4.35× chance) | Not decoded in supplied paper; strong above-chance sensory coding expected in visual cortex. |
| image_change | .6112 (1.22× chance) | Paper Figure 6 reports above-chance random-forest change-vs-repeat decoding from first 400 ms, but no exact numeric accuracy is stated in text. Our task predicts every 100 ms bin and includes both rigs/experience levels, so it is not numerically identical. |
| running_speed_quintile | .2883 (1.44× chance) | No neural decoding accuracy reported. |
| pupil_diameter_quintile | .3025 (1.51× chance) | No neural decoding accuracy reported. |
| trial_outcome | .2760 (1.10× chance) | Paper Figure 6 reports binary hit-vs-miss decoding graphically, no exact text value. Our four-class outcome includes catch outcomes and is repeated over all trial bins, so chance/task differ. |
| lick-bout initiation (paper behavioral model only) | N/A | Paper reports mean cross-validated ROC AUC .83. This uses behavioral covariates, predicts licking, and is not a neural decoding accuracy. |

All outputs exceed uniform balanced chance. Image change, running, and outcome are below 1.5× chance and were investigated completely:

- Output correctness was independently checked on 9 raw SDK trials spanning 3 sessions; stimulus/change and outcome matched with `np.allclose` in all cases.
- Neural temporal alignment and each output were independently reconstructed at all bins of those trials; 45/45 checks passed. Processing and final sample plots show synchronized changes, flashes, continuous behavior, and event bins.
- Variation is adequate: image-change positives are 2.7% of time bins; running and pupil have exactly 20% in each class; outcomes are [.303,.571,.017,.108] time-weighted. No output is 99% one class except neither (change has strong imbalance but balanced loss/accuracy handle it).
- Neural filtering uses precisely SDK-valid ROIs and paper-matched unfiltered discrete events. The 4.9% zero-event trials are genuine sparsity, not missing alignment; direct SDK sums match exactly.
- The provided decoder is instantaneous per 100 ms bin. Trial outcome is a static four-class variable repeated across many pre/post-change bins, so much neural data at each bin has weak outcome information; .276 remains meaningfully above .25. Running is also only indirectly encoded. Change .611 is consistent with an above-chance signal and the paper's qualitatively reported result under a different 400 ms/random-forest design.
- Train/validation ratios are image 1.025, change 1.027, running 1.016, pupil 1.022, outcome 1.165—none approaches the >1.5 overfitting threshold. Validation generalization is therefore stable with no leakage signature.
- Visual review of `sample_trials.png` and `predictions.png` found expected sparse events, blank input panel (zero inputs required), valid time-varying labels, and predictions tracking multiple labels without an alignment shift.

### Issues Found and Resolved
- No new conversion issue was found. The only earlier accuracy issue (one-bin change impulse) was fixed and all affected conversions/validations/trainings were rerun before this review.
- No paper supplies directly comparable exact accuracies for the five required outputs; differences in task definitions are documented rather than used to dismiss a numeric mismatch.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with `README_CACHE.md`
- [x] All investigation code/logs organized under cache; required conversion, validation, training, and plot artifacts retained at project root

Final audit: all 11 required deliverable/log files exist and are non-empty. Every workflow step is COMPLETE. The full training log ends with successful completion, and all five validation balanced accuracies are above chance.
