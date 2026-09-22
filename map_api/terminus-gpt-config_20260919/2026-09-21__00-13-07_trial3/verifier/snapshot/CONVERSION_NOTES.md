# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement (NWB dataset)
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Environment: Python 3.13.15; NumPy 2.4.4; PyTorch 2.6.0+cu124. Imports succeeded.

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `ChenLiuEtAl2023_SpikeSortingQC.pdf`
- `Dockerfile`
- `code`
- `data`
- `datapaper.pdf`
- `decoder.py`
- `docker-compose.yaml`
- `methodpaper.pdf`
- `methods.txt`
- `pynwb_docs`
- `train_decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadmat`, `loadh5mat` | `VideoAnalysisUtils/preprocessing_utils.py` | LOADING | Reference loaders for legacy MATLAB/HDF5 data and cluster-note strings. These are informative only; conversion will use `pynwb` as required. |
| `sliding_histogram` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Histograms spike times over a specified interval using bin width/stride and optionally divides counts by bin width to produce Hz. |
| `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Loads trial metadata and per-probe spikes, gathers trial event/behavior fields, and orchestrates per-area firing-rate output. |
| `process_one_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Selects units for an anatomical area/QC mode, bins trial-relative spikes, and saves trial × time × neuron firing-rate arrays plus behavioral metadata. |
| `helper_get_neuron_id_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Selects units by annotation/QC, hemisphere, and anatomical region. |
| `helper_filter_by_neuron_id` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Applies selected unit IDs consistently to spikes, unit metadata, CCF coordinates, and labels. |
| `get_regular_trial_mask` | `VideoAnalysisUtils/functions_for_r2.py`; `population_decoding_utils.py` | CURATION | Defines regular trials using no early lick, no stimulation, and non-ignore/non-miss trial conditions in the processed representation. |
| `temporal_alignment_embed_and_ephys` | `VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Matches video and ephys trials by trial number and intersects their available time indices. |
| `get_frames_between_limits` | `VideoAnalysisUtils/end_to_end.py` | PROCESSING | Converts video frame indices to times relative to go cue and selects a requested go-aligned interval. |
| `create_4fold_trial_type_mask` | `VideoAnalysisUtils/functions_for_r2.py` | CURATION | Builds trial-type strata from stimulus/choice and correctness for balanced folds. |
| `load_session` | `VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Concatenates area-specific processed firing rates and CCF metadata into a session-level representation. |

### Notes
- Repository: MapVideoAnalysis. The README says production HPC analyses are in `Sherlock`, the reusable module is `VideoAnalysisUtils`, and `Archive` is old/unused/potentially broken. Archive code was therefore not treated as authoritative.
- Modality is electrophysiology, not calcium imaging; delta-F/F is not applicable. Neural data are spike times converted to firing rates.
- Reference firing-rate processing uses a sliding histogram: spike counts in bins over trial-relative time, divided by bin width when `rate=True`. The paper code commonly references processed folders such as `stride17_bw40`; the task-specific conversion instead requires non-overlapping 50-ms bins, which will be documented as an explicit required difference.
- Reference processed neural arrays use trial × time × neuron orientation (`fr.shape[2]` is repeatedly used as neuron count). Target conversion will transpose each trial to neuron × time.
- Reference curation supports manual cluster annotations and classifier QC, then anatomical region/hemisphere selection. Exact applicable NWB unit quality columns must be established from the supplied data and papers in later ordered steps.
- The reusable `get_regular_trial_mask` excludes early-lick and photostimulation trials and restricts to ordinary completed trials for the method-paper analyses. This decoder explicitly requires early lick and photostimulation labels, so those trials cannot automatically be removed; the discrepancy will be resolved after examining NWB trial semantics and reference methods.
- Video/ephys alignment in the method code is by trial identity followed by overlap of relative time indices. Go-relative frame times are computed as `frame_inds * dt - go_time`.
- Trial splits are stratified/balanced by trial type in Sherlock scripts. Trial IDs are parsed carefully because source trial numbering is one-based while NumPy indexing is zero-based.
- Region metadata are CCF labels/coordinates and area-specific files; session loading concatenates neurons across recorded areas while preserving area identity.
- No NWB loader exists in this legacy reference repository. Per the task constraint, all supplied NWB data inspection and conversion will use `pynwb`, never the legacy `h5py` loader.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains 174 NWB files (about 50 GB), organized as one directory per subject and one `behavior+ecephys+ogen.nwb` file per session. All inspection used `pynwb.NWBHDF5IO`; `h5py` was not used.
- All sessions share one trial-table schema and one unit-table schema.
- Trial table (one row/trial): `start_time`, `stop_time`, one-based `trial`, `photostim_onset`, `photostim_power`, `photostim_duration`, `trial_uid`, `task`, `task_protocol`, `trial_instruction`, `early_lick`, `outcome`, `auto_water`, and `free_water`.
- Unit table (one row/raw sorted unit): `spike_times`; manual `unit_quality`; classifier `classification`; anatomical `anno_name`; per-unit Boolean `is_good_trials`; electrode links; waveform summaries; and QC metrics including firing rate, ISI violation, presence ratio, amplitude cutoff, isolation distance, L-ratio, d-prime, nearest-neighbor rates, silhouette, and drift metrics.
- `acquisition/BehavioralEvents` contains timestamps for delay, go, photostimulation, presample, sample, trial end, and left/right lick events. Go-start event count equals trial count in every session.
- `acquisition/BehavioralTimeSeries` contains Camera0 jaw, nose, and tongue tracking in every session. Each is `(samples, 3)` with x, y, likelihood and timestamps normally spaced about 0.0034 s. Optional whisker (20 sessions), Camera3 tracking (3), or lick-port tracking (4) creates four acquisition variants.
- Camera0 tongue tracking is the uniform source for tongue y and visibility. One session (`SC066_20210413_112028_s6`) has only 13,995 samples despite 550 trials; this truncated video stream is an explicit edge case for later mapping.
- Electrodes store `location` as JSON containing a lateralized coarse `brain_regions` label and geometry. `anno_name` provides finer Allen CCF labels for classifier-labeled units.
- Unit `is_good_trials` arrays have exactly one Boolean per session trial. Some sessions/units contain false values and these periods must be considered during trial/unit curation.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272,227 raw sorted units; 69,453 classifier-good units |
| Neurons / session | 1,564.5 raw mean (range 480–2,675); 399.2 classifier-good mean (range 39–863) |
| Subjects | 28 |
| Sessions / subject | 3–10; counts: {'440956': 4, '440957': 4, '440958': 5, '440959': 8, '441666': 5, '442571': 5, '449141': 6, '455219': 4, '455220': 6, '456772': 10, '456773': 4, '456774': 3, '460432': 4, '460434': 4, '460436': 4, '479121': 6, '479149': 7, '480133': 9, '480134': 7, '480135': 7, '480927': 10, '480928': 9, '484672': 6, '484673': 9, '484674': 9, '484675': 6, '484676': 7, '484677': 6} |
| Trials (total) | 94,990 |
| Trials / session | 545.9 mean (range 264–800) |

### Dataset-wide Trial Distributions
| Variable | Counts |
|----------|--------|
| Task / protocol | audio delay: 94,990; protocol 1: 94,990 |
| Trial instruction | left: 46,077; right: 48,913 |
| Early lick | no early: 84,185; early: 10,805 |
| Outcome | hit: 65,254; miss: 15,641; ignore: 14,095 |
| Photostimulation | off (`N/A` power/duration): 76,402; on (5.5 power, 0.5 s): 18,588 |
| Auto water | no: 93,651; yes: 1,339 |
| Free water | no: 92,540; yes: 2,450 |

Event completeness was checked for every stream; every session has one go-start event per trial (94,990 each). Six sessions have zero photostimulation events, which is valid and agrees with the corresponding trial fields.

### Exploration Correction
- An initial statistic obtained by parsing the verbose per-file scan before checking its completeness incorrectly reported 24 subjects, 100,750 trials, and 259,171 units. The independent completed aggregate pynwb scan and its 174 explicit per-session rows showed the correct values: 28 subjects, 94,990 trials, and 272,227 raw units. All table entries above were corrected; per-session totals sum exactly to the corrected dataset totals.

### Available Values and Quality Notes
- `trial_instruction` supplies instructed left/right trials; `outcome` values are hit/miss/ignore; `early_lick` supplies early/no-early; photostimulation is represented both in the trial table and timestamped event streams.
- Manual spike-sort labels include `good` and `multi`; classifier labels include `good` and `unlabelled`. The correct paper-matched filter will be resolved from the methods/QC paper in Step 3 rather than guessed here.
- Representative numeric unit fields are finite and broad (e.g. firing rates, ISI violation, presence ratio, amplitude cutoff), enabling reference QC rules.
- Native spike times and behavioral/event timestamps are on the same session time base. Trial rows and go events provide the mapping for go-aligned extraction.
- Trial schema descriptions explicitly state that `trial` is one-based, confirming the reference code's subtraction of one when converting to NumPy indices.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote / Source |
|-----------|-------|-----------------------|
| Neurons (total) | 69,943 classifier-good units | Data paper Methods: “Overall, the dataset consisted of 69,943 good units recorded across 173 behavioral sessions.” |
| Neurons / session | about 404 good units/session | Derived from 69,943 / 173; paper reports regional totals rather than a per-session range. |
| Subjects | 28 mice | Data paper Fig. 1J caption: aggregated over 173 behavioral sessions and 28 mice. |
| Sessions / subject | 173 total behavioral sessions | Data paper Fig. 1J and Methods. |
| Trials (total) | Not explicitly tabulated | Paper gives session-level summary after analysis selection. |
| Trials / session | mean 476, range 130–785 | Data paper Methods. |
| Neural data time bin | 40-ms bin width, 17-ms stride in method-paper video/neural analyses | Method paper Methods, firing-rate calculation. Task conversion mandates 50-ms non-overlapping bins instead. |
| Behavior data time bin | Video acquired at 300 Hz; NWB timestamps are approximately 3.4 ms apart | Data paper Methods and supplied data. |
| Reward/correct rate | mean 84%, range 65–99% in selected sessions | Data paper Methods; performance is fraction correct control trials excluding early lick. |
| Probe insertions | 655 in Methods; Fig. 1J says 660 penetrations | Data paper; likely insertion/penetration terminology or revision difference. |
| Classifier yield | 25.9% of Kilosort2 clusters | Data paper Methods. |
| Regional good units | ALM 8,717; striatum 7,664; thalamus 12,808; midbrain 7,495; medulla 2,928; plus other regions | Data paper Methods. |

### Processing Details
- Task: auditory delayed response. Instruction tones are 3 or 12 kHz, presented three times for 150 ms with 100-ms inter-tone intervals. A 1.2-s delay follows. The 100-ms auditory go cue (6-kHz carrier, 360-Hz modulation) ends the delay. The answer period is 1.5 s.
- Early licking during sample/delay triggers replay. Correct response gives water; incorrect response gives a 1–3-s timeout. No response is represented as ignore in the supplied trial table.
- Two camera views were acquired near 300 Hz; DeepLabCut tracked tongue, jaw, and nose. The supplied NWB Camera0 tongue series has y position and model likelihood.
- Ephys was acquired at 30 kHz and sorted with Kilosort2 for published analyses.
- Method-paper video/neural prediction computed firing rate in 40-ms windows advanced by 17 ms. It excluded units with mean firing rate below 2 Hz for those prediction analyses and aligned video/neural samples over common trial-relative time.
- Method-paper video analyses used correct, non-photostimulation trials and stratified train/test folds by lick direction; other analyses used 20-fold stratified cross-validation. Those are analysis-specific choices, not universally applicable to this decoder task.
- The current decoder task overrides the reference bin/window where stated: align to go onset, extract −2.5 to +1.5 s, and use 50-ms-width bins.

### Curation Steps

**Neuron curation rules**:
- Published dataset uses region-specific logistic-regression QC classifiers trained from blinded manual Phy labels (`good` versus `unlabeled`) and 15 Allen quality metrics.
- Separate classifiers were trained for cortex, striatum, thalamus, midbrain, and medulla and then applied to related structures. Published false-alarm rates were cortex 7.8%, striatum 6.4%, thalamus 7.3%, midbrain 5.5%, medulla 4.3% using held-out/10-fold validation.
- Therefore the NWB `classification == "good"` field is the direct paper-matched unit inclusion flag. Manual `unit_quality` (`good`/`multi`) is not a substitute for this classifier output.
- The method paper's additional >2-Hz filter was specific to video-to-neural prediction and should not automatically be imposed on a general neural decoder unless consistency analysis indicates it is required.

**Trial/session curation rules**:
- Published primary analyses excluded early-lick and no-response trials.
- Published sessions were selected for control-trial performance >65% and at least 50 correct left plus 50 correct right trials.
- Performance used no-photostimulation trials and excluded early-lick trials.
- This task explicitly requires early lick, outcome (including ignore/miss), and photostimulation as decoder variables. Consequently those trial classes must be retained unless data validity fails; this is a justified task-required departure from paper-specific trial exclusions.
- Per-unit `is_good_trials` marks invalid unit/trial periods and must be respected when deciding session/trial/neuron inclusion.

### Decoders Trained
| Decoded variable | Accuracy / metric |
|------------------|-------------------|
| Choice from behavioral video/markers | ROC-AUC, stratified cross-validation; numeric values mainly presented graphically, not as one tabled accuracy. |
| Uninstructed movement grouping and choice modulation of single neurons | ROC-AUC; chance 0.5, with nested cross-validation for regularization. |
| Neural firing rate from video | Explained variance, not categorical accuracy. |

### Decoder Accuracy Expectations
- The reference papers do not report accuracies for the exact supplied `train_decoder.py` architecture or the exact four requested outputs. Direct numerical equality is therefore not expected.
- For categorical outputs, balanced validation accuracy must exceed chance (1/3 for choice and outcome, 1/2 for early lick, 1/4 for tongue category). Strong neural choice selectivity and movement encoding suggest above-chance performance, while tongue visibility/category will depend on correct temporal alignment and likelihood handling.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session count | Process each available session/area file | 174 NWB files; one session has zero `classification == good` units | 173 behavioral sessions | Exclude the zero-good-unit session because it cannot form a neural decoder session. This yields 173 usable sessions, matching the paper. Do not reapply prose behavioral thresholds naively to the already curated release. |
| Good-unit count | Reference preprocessing uses classifier QC annotations | 69,453 classifier-good units across all 174 files | 69,943 good units | Use the direct NWB classifier flag, the closest source-of-truth for this release. The 490-unit difference is attributed to release/version/export differences and cannot be repaired by substituting manual labels. Document converted count exactly. |
| Session performance thresholds | Reference scripts mostly consume preprocessed selected sessions | Recalculation yields 106 or 152 passing sessions depending on whether ignores enter the denominator | >65% control performance and ≥50 correct trials each direction; 173 sessions published | The NWB files are an already selected release and event/outcome semantics do not reproduce the prose selection verbatim. Applying either reconstruction would wrongly remove 21–68 sessions. Keep usable released sessions; exclude only objective neural/data validity failures. |
| Trial curation | `get_regular_trial_mask`: no early lick, auto/free water, ignore, or stimulation | All these labels are available and common | Main paper analyses excluded early/no-response and often stimulation trials | Decoder explicitly requires predicting early lick/outcome and receiving photostimulation, so retain these trial types. Auto/free-water flags remain metadata and will be assessed as potential atypical trials in mapping. |
| Neural binning | 40-ms sliding window, 17-ms stride for method-paper video prediction | Raw spike times permit arbitrary bins | Method paper uses 40/17 ms for that analysis | Use mandated non-overlapping 50-ms bins over −2.5 to +1.5 s. This is an explicit task override. |
| Neural orientation | Reference processed `fr`: trial × time × neuron | Raw spikes per unit | Target requires neuron × time per trial | Bin each unit and transpose to neuron × time. |
| Video alignment | Match trial IDs and overlapping relative-time indices | Shared session timestamps; one go event/trial; sample/delay events can repeat after early licks | Go cue defines trial timing; video at 300 Hz | Use NWB timestamps and one-to-one go events for alignment. Tone onset mapping must account for replay events and is finalized in Step 5. |
| Tongue stream | Reference uses high-rate marker positions | Camera0 tongue x/y/likelihood in all sessions; one stream is truncated | DeepLabCut tongue tracking at ~300 Hz | Use Camera0 tongue y and likelihood. Mark bins without valid/visible samples as class 3; do not fabricate values or discard otherwise valid neural trials. |
| Unit-specific bad trials | Reference source contains validity/annotation mechanisms | 1,826 raw units have at least one false `is_good_trials`; 99.86% of all unit-trial flags are true | Papers emphasize QC but do not specify fixed-tensor handling | Respect these flags. Exact fixed-session tensor policy (unit exclusion versus trial exclusion) will be chosen quantitatively in Step 5 to avoid silent contamination or variable neuron dimensions. |

### Final Consistent Understanding
- Use `pynwb` to load the supplied release and `classification == "good"` as the paper-matched neural QC flag.
- Treat the release as already session-curated; a session with no good neural units is objectively unusable and its removal reconciles the 174 files with 173 published neural sessions.
- Preserve early-lick, hit/miss/ignore, and stimulated trials because they are required decoder variables, despite narrower paper-analysis masks.
- Use go-start timestamps as the unambiguous alignment event and raw spike times for mandated 50-ms firing-rate bins.
- Use timestamped Camera0 tongue tracking, with model likelihood and temporal coverage determining visibility.
- Brain regions should come from unit-linked electrodes/coarse lateralized region JSON or fine `anno_name`; mapping granularity will be fixed in Step 5 after checking completeness for selected units.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units.spike_times` for `classification == "good"` and all-true `is_good_trials` | `neural` | Histogram absolute spikes in 80 non-overlapping 50-ms bins spanning go−2.5 s to go+1.5 s; divide counts by 0.05 s; float32 neuron × time | `sliding_histogram`, `process_one_area` | Task-mandated window/bin replaces reference 40-ms/17-ms sliding analysis bins. |
| Latest `sample_start_times` before each trial's go cue | `input[0]` | At every bin center, absolute bin-center time minus tone/sample onset; continuous seconds | reference event alignment in `process_one_sess` | Most recent onset handles replay after early lick. All selected onsets are inside their trial and before go. |
| Paired `photostim_start_times` / `photostim_stop_times` | `input[1]` | Binary 1 where a 50-ms bin center lies in the trial's photostimulation interval, else 0 | reference `stimulation` and regular-trial mask | Event timestamps are preferred to string-valued trial columns after equivalence checks. |
| `trial_instruction` + `outcome` | `output[0]` choice | hit → instructed side; miss → opposite side; ignore → no lick; encode left=0, right=1, no lick=2 | `trial_type`, `correctness`, lick fields | Agrees with first response-period lick in 99.65% of raw trials; outcome-derived choice avoids incidental lick-event ambiguity. Per-trial scalar. |
| `outcome` | `output[1]` outcome | ignore=0, miss=1, hit=2 | `correctness` | Per-trial scalar; all required classes retained. |
| `early_lick` | `output[2]` early lick | no early=0, early=1 | `early_lick_trials` | Per-trial scalar; retained despite paper-analysis exclusion because it is required output. |
| Camera0 `TongueTracking` y and likelihood | `output[3]` tongue y-position | Nearest camera sample to each bin center if within 10 ms and likelihood ≥0.9. Using percentiles of all visible y samples in that session: y < p40 → 0; p40 ≤ y ≤ p60 → 1; y > p60 → 2; unavailable/low likelihood → 3 | paper DeepLabCut tracking; temporal overlap logic in `temporal_alignment_embed_and_ephys` | Time-varying 80-vector. A conservative 0.9 confidence threshold defines visible because no paper/code threshold is specified. |
| `subject.subject_id` | `subjects`, `subject_idx` | Unique sorted subject IDs and per-session index | — | 28 subjects. |
| Unit-linked electrode `location` JSON `brain_regions` | `brain_regions`, `brain_region_idx` | Preserve 14 lateralized coarse recording-target names and map every selected unit | `load_session` area concatenation | Complete and stable; fine `anno_name` has 293 Allen labels and is retained in session metadata summaries but is too granular for primary region categories. |

### Key Decisions
1. **Sessions**: Include the 173 files containing at least one classifier-good unit; exclude `SC017_20190216_162508_s4`, whose 1,852 units all have missing classifier/anatomy labels. This exactly reconciles the 174 files with the paper's 173 sessions.
2. **Neuron QC and fixed dimensions**: Start with `classification == "good"`, then exclude the 565 good units whose `is_good_trials` is false on any trial. This leaves 68,888 units and preserves all 94,370 trials in usable sessions. The alternative of removing every trial invalid for any selected unit would remove 1,569 trials and unnecessarily alter output distributions. Session neuron identity/dimension remains fixed.
3. **Trials**: Retain all required behavioral classes, but require every selected unit’s `obs_intervals` to contain the complete go−2.5 to go+1.5-s window. Critical review found 18,337 released trial rows outside simultaneous neural observation periods; excluding them leaves 76,033 valid trials. Every session retains at least two trials.
4. **Time grid**: Edges are `np.arange(-2.5, 1.5 + 0.05, 0.05)` (81 edges); centers are −2.475 through +1.475 s (80 points). Left-closed/right-open histogram semantics avoid off-by-one overlap; the last edge is included only by NumPy's final-bin convention.
5. **Tone input semantics**: The user explicitly requests “time from tone onset in seconds,” so this is a continuous time-varying ramp, not a binary onset pulse. Early-lick replay creates 10,921 extra sample events; selecting the latest sample onset before go maps the final epoch correctly. All mapped onsets fall within trial boundaries.
6. **Photostimulation**: Use paired timestamp intervals and evaluate state at bin centers. Nonstimulated trials are all-zero. This captures the late-delay 0.5-s intervention described in the paper without relying on string `N/A` parsing.
7. **Actual choice**: Derive actual lick choice from instruction and outcome. A hit means instructed side, a miss means the opposite side, and ignore means no lick. This matches task semantics more robustly than choosing the first among many raw lick events.
8. **Tongue visibility**: Camera0 exists uniformly and is the designated side view. A sample must be within 10 ms of the bin center (greater than the normal ~1.7-ms nearest-frame distance but detects gaps) and likelihood ≥0.9. Across usable sessions, 98.0% of centers have temporal coverage and 15.8% are visible at this confidence, consistent with intermittent tongue protrusion. Missing coverage and low confidence both map to class 3.
9. **Tongue percentiles**: Compute p40/p60 separately for each session from all finite Camera0 y samples with likelihood ≥0.9, as explicitly requested “over the session.” Do not compute thresholds per trial or only within the go window. If no visible samples exist, all tongue outputs are class 3.
10. **Truncated video**: `SC066_20210413_112028_s6` has only 13,995 tongue samples; neural/task trials remain valid, and out-of-coverage bins become class 3 rather than dropping the session.
11. **Region granularity**: Use complete 14-way lateralized coarse recording targets (`left/right` ALM, BLA, ECT, Medulla, Midbrain, Striatum, Thalamus). This matches experimental targeting and avoids hundreds of sparse fine subdivisions while preserving hemisphere.
12. **Storage and efficiency**: Expected raw float32 neural payload is about 12.0 GB. Process and close one NWB session at a time, vectorize spike histogramming per unit/trial where practical, avoid global dense intermediates, and pickle once at the end. Inputs/outputs use compact float32/int64 arrays.

### Expected Converted Counts
| Statistic | Planned value |
|-----------|---------------|
| Sessions | 173 |
| Subjects | 28 |
| Trials | 92,247 after excluding only zero-total-spike acquisition gaps |
| Neurons | 68,888 after classifier and all-trial-valid filtering |
| Time points/trial | 80 |
| Inputs | 2: time from tone onset, photostimulation on |
| Outputs | 4: choice, outcome, early lick, tongue y category |
| Brain regions | 14 lateralized coarse targets |

### Planned Sanity Checks
- [ ] For three sessions/trials, independently histogram raw `spike_times` with pynwb and verify converted neural matrices with `np.allclose()`.
- [ ] Verify firing rates are nonnegative multiples of 20 Hz and matrix shape is selected neurons × 80.
- [ ] Independently reconstruct tone-relative center times from sample/go timestamps and compare converted input[0] using `np.allclose()`.
- [ ] Independently reconstruct photostimulation center-state vectors from paired event intervals and compare input[1] using `np.allclose()`.
- [ ] Verify output choice mapping on hit, miss, and ignore examples directly from raw trial rows and raw lick events.
- [ ] Recompute session p40/p60 from visible raw tongue samples; compare three time bins per selected session and verify classes including not-visible with `np.allclose()`.
- [ ] Confirm all output values are in declared ranges and class distributions agree with the raw 173-session subset.
- [ ] Confirm selected go events map through stored original trial indices, every final tone precedes go, every requested window is inside all selected-unit observation intervals, and all retained trials have exactly 80 bins.
- [ ] Confirm subject/session/neuron/trial/region totals and selected-unit IDs match the independently computed planning statistics.
- [ ] Plot spike rates, time-from-tone, stimulation state, tongue y/likelihood/categories, and event markers for up to two sessions in `--show-processing` mode.

---

## Step 6: Script Development
**Status**: COMPLETE

- Implemented `/app/convert_data.py` with required invocation and `--full` (default), `--sample`, and `--show-processing` modes.
- The script imports and uses `pynwb.NWBHDF5IO` for every NWB read and never imports `h5py`.
- Each session is opened, processed, and closed sequentially. Selected spike times are binned directly from the NWB unit table; trial/event/behavior streams remain on their common absolute timestamp base.
- Outputs use 80-bin uniform matrices: the three per-trial outputs are repeated across time and tongue category remains time-varying, producing a decoder-compatible 4×80 output matrix.
- Assertions check event/trial counts, non-overlapping requested windows, tone placement, fixed tensor dimensions, finite/nonnegative rates, and all input/output dimensions.
- `--show-processing` creates one five-panel figure per requested session showing mean firing rate, time from tone, photostimulation state, tongue y with percentile thresholds, likelihood, and category.
- A direct first-session smoke test succeeded in 1.20 s: 368 trials, 459 selected neurons, neural 459×80 float32, input 2×80 float32, output 4×80 int8; rates ranged 0–360 Hz and were exact multiples of 20 Hz.
- Available resources are sufficient: approximately 979 GiB RAM and 3.4 TiB disk available versus an estimated 12-GB float32 neural payload.

Code inefficiencies identified:
- Naive nested trial × unit calls to `np.histogram` would perform tens of millions of Python operations.
- Reopening NWB files for each stream or retaining full NWB objects would add substantial I/O/memory overhead.

Code speedups added:
- Verified the minimum go-cue spacing is 4.5827 s, greater than the 4.0-s requested window, so windows never overlap.
- For each unit, spikes are assigned vectorially to the latest trial-window start and accumulated with one `np.bincount`, exactly matching independent non-overlapping histograms.
- Session-level timestamp grids, nearest video indices, and event vectors are computed with NumPy search/sort operations.
- Files are read once per conversion session; only compact converted arrays and metadata remain in memory.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 across sessions (459, 375) |
| Neurons / session | mean 417; range 375–459 |
| Subjects | 1 |
| Sessions / subject | 2 sessions for subject 440956 |
| Trials (total) | 848 |
| Trials / session | 368, 480 |
| Time points | exactly 80 for every trial |
| Time from tone range | [−0.625, 5.722] s |
| Photostimulation range | [0, 1] |
| Choice distribution | left 0.469, right 0.441, no lick 0.090 |
| Outcome distribution | ignore 0.090, miss 0.200, hit 0.710 |
| Early lick distribution | no 0.954, yes 0.046 |
| Tongue distribution | low 0.056, middle 0.030, high 0.069, not visible 0.845 |

### Format Validation
- `/app/sample_data.pkl` created (112,585,409 bytes) with all required top-level keys.
- Neural/input/output shapes are respectively neuron×80, 2×80, and 4×80 for every inspected trial; dtypes are float32, float32, and int8.
- All neural values are finite and nonnegative; rate values are multiples of 20 Hz as expected for 50-ms count bins.
- `/app/verification_sample_out.txt` was created by the required verifier. It reports no errors and ends with “Data verification complete.”
- Both inputs and every declared output class are present in the sample.

### Processing Plots Review
- `processing_SC015_20190207_120657_s1.png` and `processing_SC015_20190208_133600_s2.png` are valid, nonempty PNGs (five aligned panels each).
- Mean firing rates use the same −2.5 to +1.5-s go-relative axis as task inputs and tongue output.
- Time-from-tone traces increase linearly at slope one; stimulation is binary and confined to the expected late-delay interval; tongue category changes only at percentile/visibility boundaries.
- No temporal discontinuity or array-length mismatch was observed.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
|-----------------------|--------------|
| Vectorized per-unit trial/bin assignment with `searchsorted` + `bincount` | Avoids nested trial×unit histogram calls |
| One NWB open/read per session | Avoids redundant 50-GB source scans |
| Vectorized video nearest-neighbor lookup | Avoids trial×time Python loops |

| Step | Time / Session | Estimated Total Time |
|------|----------------|----------------------|
| Session conversion | 0.94–1.34 s in sample | about 3–5 min for 173 sessions, allowing larger sessions |
| Sample pickle | 0.11 s for 0.113 GB | full 12-GB pickle expected to add roughly 1–3 min |
| Total full conversion | — | conservatively <10 min, below the 15-min optimization threshold |

### Issue Found and Fixed
- First sample attempt processed both sessions but failed in the final summary because neuron totals were computed from session lists as if they were arrays. The summary expression was corrected to read each session's first trial shape; the sample conversion was then rerun cleanly from scratch and passed all checks.


### Final Sample Rerun After Critical Review
- Final sample policy retains behavioral classes and excludes only simultaneous zero-total-spike acquisition gaps.
- Sample conversion, verification, plots, and training were regenerated successfully.
- Final validation balanced accuracies: choice 0.6235, outcome 0.6453, early lick 0.7294, tongue y-position 0.5216; all exceed chance.

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None.
- Warnings: None.
- The verifier reported expected dimensions/ranges/distributions and completed successfully.

### Training Progress
- Loss decreased from 5.21 at epoch 5 to 0.706 at epoch 200; test loss was 0.820.
- Training completed normally and `/app/train_decoder_sample_out.txt` was created.

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-----------------------|-------------------------|--------|
| Lick direction choice | 0.6794 | 0.6018 | 0.3333 |
| Outcome | 0.7112 | 0.6120 | 0.3333 |
| Early lick | 0.8136 | 0.6411 | 0.5000 |
| Tongue y-position | 0.5286 | 0.4307 | 0.2500 |

All four validation accuracies are above chance. Choice and outcome are about 1.8× chance, tongue is about 1.7× chance, and early lick is 1.28× chance. Train/validation gaps are modest and do not exceed the later 1.5× review threshold.


### Final Sample Decoder Results (supersedes initial table)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-----------------------|-------------------------|--------|
| Lick direction choice | 0.7326 | 0.6235 | 0.3333 |
| Outcome | 0.7524 | 0.6453 | 0.3333 |
| Early lick | 0.8359 | 0.7294 | 0.5000 |
| Tongue y-position | 0.6777 | 0.5216 | 0.2500 |

Loss decreased from 7.74 (epoch 5) to 0.571 (epoch 200); test loss 0.730.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 12.212 GB (12,211,xxx,xxx bytes; filesystem display 12 GB)
- `conversion_full_out.txt`: created
- `verification_full_out.txt`: created; no errors/warnings and ends with “Data verification complete.”

### Full Conversion Performance
- Completed in 246.29 s (4.10 min), including a 12.23-s pickle write; this was below the conservative <10-min estimate and far below the 15-min optimization threshold.
- Exactly one file was skipped: `SC017_20190216_162508_s4`, whose units all lack classifier labels.

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 classifier-good (release/version) | classifier QC | 69,453 classifier-good; 68,888 valid on every trial | 68,888 | Yes to documented selected-unit rule; paper differs by 490 before validity filter |
| Mean neurons/session | ~404 paper good units | area files concatenated | 398.2 selected | 398.2 (range 90–923) | Yes |
| Subjects | 28 | session metadata | 28 | 28 | Yes |
| Sessions | 173 | selected processed sessions | 174 files, one without classifier labels | 173 | Yes |
| Trials (total) | not tabulated; mean 476 for paper analysis subset | analysis-specific regular masks | 94,370 in 173 usable sessions | 94,370 | Yes |
| Trials/session | paper mean 476, range 130–785 after analysis selection | varies by mask | mean 545.5, range 264–800 | mean 545.5, range 264–800 | Yes to released/task-retained set |
| Time points | task mandates 80 | reference differs (40-ms/17-ms) | raw timestamps | 80 uniformly | Yes |
| Time from tone range | no target range specified | final sample epoch before go | raw mapped events | [−1.525, 11.8943] s | Plausible; includes pre-tone bins and replay-delayed go trials |
| Photostimulation range | binary state | stimulation event intervals | 0/1 | 0/1 | Yes |
| Choice distribution | not directly tabulated | trial type + correctness | expected from 173-session rows | left 0.4286, right 0.4220, no lick 0.1494 | Yes |
| Outcome distribution | not directly tabulated | hit/miss/no response | ignore 14.94%, miss 16.46%, hit 68.60% | same | Yes |
| Early lick distribution | excluded in paper analyses | early-lick field | no 88.62%, yes 11.38% | same | Yes |
| Tongue distribution | intermittent tracked movement | video marker positions | visible 15.80% at threshold | low 6.18%, middle 3.16%, high 6.46%, not visible 84.20% | Yes |
| Brain regions | broad brain-wide targets | area-specific files | 14 complete lateralized targets | 14 | Yes |

### Spot Checks
- Loaded the 12.2-GB pickle in 5.54 s and checked first/middle/last trials of first/middle/last sessions.
- Every checked neural/input/output shape was neuron×80, 2×80, and 4×80 with float32/float32/int8 dtypes.
- `subject_idx` has shape (173,) and spans 0–27; every region index count matches verifier output.
- All required output classes and both stimulation states occur globally.
- One randomly selected spot (`SC067_20210420_170516_s28`, trial index 325) had an all-zero neural matrix. This is flagged for mandatory raw-data investigation in Step 10 rather than assumed valid.


### Final Full Rerun (supersedes earlier intermediate counts)
- Final conversion: 173 sessions, 28 subjects, 90,859 trials, 68,888 neurons, 14 regions.
- Final pickle size: 11.760 GB; conversion completed in 264.80 s.
- Trial range 159–800, mean 525.20/session; neuron range 90–923.
- Final verifier reports no errors/warnings and ends with “Data verification complete.”
- Final output fractions: choice left/right/no lick = 0.4286/0.4223/0.1491; outcome ignore/miss/hit = 0.1491/0.1664/0.6845; early no/yes = 0.8846/0.1154; tongue low/middle/high/not-visible = 0.0621/0.0317/0.0651/0.8410.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output Log Verification
- Read the complete final `/app/verification_full_out.txt`.
- Errors: none. Warnings: none. The final line is “Data verification complete.”
- All 173 sessions have consistent 80-point neural/input/output time dimensions and fixed neuron counts within session.

### Check 2: Independent Raw-Data Sanity Checks
All original-data checks loaded NWB files directly with `pynwb.NWBHDF5IO` and did not call conversion functions.
- Neural: independently counted raw spikes for first/middle/last sessions, selected units, trials, and bins; converted Hz values matched with `np.allclose()`.
- Input 0: independently selected the latest sample/tone event before go and calculated bin-center time minus tone onset; matched with `np.allclose()`.
- Input 1: independently tested bin centers against paired photostimulation event intervals, including an actually stimulated bin; matched with `np.allclose()`.
- Trial outputs: independently mapped raw instruction/outcome/early-lick fields for hit, miss, ignore, and early trials; matched with `np.allclose()`.
- Tongue: independently recomputed session p40/p60 from likelihood≥0.9 samples and nearest-frame visible/not-visible classes; all tested bins matched with `np.allclose()`.
- Original trial mapping: stored `original_trial_indices` correctly maps every converted trial to its raw NWB row after gap exclusion.
- Final script `/tmp/step10_final_sanity.py` reports “ALL FINAL SANITY CHECKS PASSED.”

### Check 3: Reference Code Comparison
| Stage | Reference implementation | Conversion implementation | Comparison / reason for difference |
|-------|--------------------------|---------------------------|------------------------------------|
| Data loading | Legacy MATLAB/HDF5 loaders in `preprocessing_utils.py` | `pynwb.NWBHDF5IO` only | Mandatory NWB/pynwb difference; source fields are semantically equivalent. |
| Neuron filtering | Region-specific classifier annotations; analysis-specific optional >2-Hz threshold | `classification == good`; remove 565 units with any false `is_good_trials` | Matches paper QC. The >2-Hz rule was specific to video→neural prediction and is not imposed. |
| Trial filtering | `get_regular_trial_mask` excludes early, ignore, water, and stimulation trials | Retain required behavioral/stimulation classes; exclude only zero-total-spike acquisition gaps | Required by decoder outputs/inputs. Gap exclusion removes no-information periods without class-specific paper masks. |
| Temporal alignment | Trial identity plus overlapping relative time; frame time relative to go | Exact NWB go-start timestamps and stored raw trial indices | Same conceptual alignment, with NWB common clock. |
| Binning | Sliding histogram, 40-ms width/17-ms stride in method-paper video analysis | Non-overlapping 50-ms bins, −2.5 to +1.5 s | Explicit decoder-task override. Counts divided by 0.05 to Hz. |
| Input construction | Relative event times and stimulation metadata | Continuous time from latest tone/sample onset; binary stimulation state from event intervals | Matches requested decoder inputs and replay semantics. |
| Output construction | Trial type/correctness/early fields; video marker coordinates | Actual choice from instruction+outcome, categorical outcome/early, session-percentile tongue y with visibility | Matches task definitions; per-trial values repeated over 80 points for uniform decoder tensors. |

### Check 4: Key Statistics Comparison
| Statistic | Paper | Raw supplied release | Final converted |
|-----------|-------|----------------------|-----------------|
| Subjects | 28 | 28 | 28 |
| Sessions | 173 | 174 files; one lacks classifier labels | 173 |
| Good units | 69,943 (paper release) | 69,453 classifier-good | 68,888 after all-trial-valid unit filtering |
| Trials/session | mean 476, range 130–785 for paper analysis subset | mean 545.9, range 264–800 across all classes | mean 525.2, range 159–800 after acquisition-gap exclusion |
| Outcome fractions | correct mean 84% in selected control analyses | ignore/miss/hit 14.84/16.47/68.69% in 173 usable files | 14.91/16.64/68.45% |
| Time bins | reference analysis 40-ms/17-ms stride | raw spikes | mandated 50 ms, 80 points |
- The 490-unit paper/release discrepancy is not repairable by substituting manual labels and is documented as a release/export difference.
- Final class fractions remain close to the released data, demonstrating that gap exclusion is not outcome-selective.

### Check 5: Edge Cases and Off-by-One Review
- Verified 81 edges produce exactly 80 left-closed 50-ms bins from −2.5 to +1.5 s; centers are −2.475 to +1.475 s.
- Minimum go spacing is 4.5827 s, so requested windows do not overlap.
- Trial numbering is one-based in NWB metadata, but conversion uses row indices explicitly and stores original row indices; no one-based subtraction ambiguity remains.
- Latest sample onset before go handles 10,921 replay events; every mapped tone lies inside its trial and before go.
- All 18,441 photostimulation events in usable sessions pair correctly and last 0.5 s.
- Truncated tongue session is retained; uncovered bins are class 3, never fabricated.
- Fine anatomy is complete but sparse (293 labels); stable complete 14-way lateralized recording targets are used.

### Issues Found and Resolved
1. **Summary-only sample bug**: neuron total treated a session list as an array. Fixed and reran sample conversion.
2. **Unrecorded all-zero periods**: initial conversion retained source periods with zero spikes across every selected unit. Independent raw checks confirmed acquisition gaps. Final conversion excludes 3,511 such trials and stores original indices.
3. **Incorrect intermediate `obs_intervals` interpretation**: filtering by these intervals reduced miss outcomes from 16.5% to 0.93%, revealing they encode an analysis subset rather than general recording validity. Rejected that policy, restored all behavioral classes, and reran sample/full conversion, validation, training, and every critical check.
4. **Final result**: 90,859 trials, no all-zero neural matrices, raw class proportions preserved, all independent comparisons pass.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Command: `python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples 2>&1 | tee /app/train_decoder_full_out.txt`.
- Device: CUDA GPU; no memory fallback or runtime error.
- Training used 72,633 trials and testing used 18,226 trials.
- Loss decreasing: Yes. Loss fell from 18.8457 at epoch 1 to 0.6699 at epoch 200; test loss was 0.6697.
- Full script finished normally with `train_decoder.py finished successfully.`

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-----------------------|-------------------------|--------|-------|
| Lick direction choice | 0.7244 | 0.6880 | 0.3333 | 2.06× chance validation |
| Outcome | 0.6953 | 0.6583 | 0.3333 | 1.97× chance validation |
| Early lick | 0.7838 | 0.7483 | 0.5000 | 1.497× chance validation; reviewed in Step 12 |
| Tongue y-position | 0.6759 | 0.6161 | 0.2500 | 2.46× chance validation |

Validation accuracy is above chance for every output. Training and validation values are close, indicating good generalization rather than leakage or severe overfitting.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Validation Accuracy | Chance | Accuracy / Chance | Expectation from Papers |
|----------|---------------------|--------|-------------------|-------------------------|
| Lick direction choice | 0.6880 | 0.3333 | 2.064 | Papers show robust choice selectivity and report video-based choice ROC-AUC graphically; exact architecture/metric not reported. |
| Outcome | 0.6583 | 0.3333 | 1.975 | No paper accuracy for three-way hit/miss/ignore decoding; strong task modulation supports above chance. |
| Early lick | 0.7483 | 0.5000 | 1.497 | No paper decoder; early trials were excluded in paper analyses. Marginally below the suggested 1.5× review threshold and investigated below. |
| Tongue y-position | 0.6161 | 0.2500 | 2.464 | Paper reports widespread movement encoding and video/neural predictability, but not this four-class tongue metric. |

### Check 1: Accuracy vs Chance
- Every output is above chance; choice, outcome, and tongue substantially exceed 1.5× chance.
- Early lick is 1.497× chance, only 0.0017 absolute accuracy below the heuristic 1.5× threshold. Investigation found no conversion defect:
  1. Raw `early_lick` values were independently checked on first/middle/last NWB sessions and exactly matched converted labels.
  2. Both classes have ample support: final fractions are no=0.8846, yes=0.1154.
  3. Alignment uses one go event per source trial with stored raw row indices; independent time/input checks pass.
  4. Sample validation early-lick accuracy was 0.7294 and full accuracy improved to 0.7483, consistent with a genuine but difficult pre-go neural signal.
  5. No filtering change can improve this without outcome leakage or removing the required early-lick class.
- Conclusion: the 0.0034 shortfall from exactly 1.5× is statistical/architectural variation, not evidence of a conversion bug.

### Check 2: Accuracy Comparison to Papers
- Exhaustive text search found paper decoders based primarily on ROC-AUC for choice/uninstructed movement and explained variance for video→neural prediction.
- The papers do not report balanced accuracies for the exact four outputs, 50-ms bins, time window, or supplied neural-network decoder.
- Direct numeric comparison would mix metrics and tasks. Qualitatively, the achieved high choice and tongue accuracies agree with reported brain-wide choice/movement encoding; no achieved score is in conflict with a reported value.

### Check 3: Train vs Validation Gap
| Variable | Train / Validation | Absolute Gap | Review |
|----------|--------------------|--------------|--------|
| Choice | 1.053 | 0.0364 | Small |
| Outcome | 1.056 | 0.0370 | Small |
| Early lick | 1.047 | 0.0355 | Small |
| Tongue | 1.097 | 0.0598 | Small |
- No ratio approaches the 1.5× overfitting threshold. Similar train/test loss (0.6699/0.6697) also argues against overfitting or leakage.

### Low-Accuracy Debugging Checks
- Checked three specific raw trials for each trial-level mapping during Step 10; all matched with `np.allclose()`.
- Plotted aligned neural, task input, stimulation, tongue y/likelihood, and tongue categories; no temporal offset was observed.
- Every output has adequate variation and all declared classes occur.
- Classifier QC and zero-total-spike acquisition-gap filtering are applied; raw outcome proportions are preserved.
- Processing was compared stage-by-stage to reference code/papers in Step 10.

### Issues Found and Resolved
- No new conversion issue was found in the accuracy review.
- Earlier invalid-period and outcome-biased-filter issues were already fixed, fully rerun, and materially improved sample/full reliability.
- Final conclusion: decoder performance is high, above chance for all outputs, generalizes well, and is consistent with the neuroscience reported in the papers.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] `README.md` created with dataset description, loading example, schema, statistics, reproduction commands, and decoder results.
- [x] `cache/` folder created; investigation/sanity scripts copied there.
- [x] `cache/README_CACHE.md` documents cached files.
- [x] All required conversion, sample/full validation, and sample/full training logs exist.
- [x] Final `converted_data.pkl` and `sample_data.pkl` exist and pass the reference verifier.
- [x] `CONVERSION_NOTES.md` includes all decisions, discrepancies, iterations, raw-data checks, and final accuracy tables.
- [x] Final top-level directory retains user-facing outputs and processing plots; investigation scripts are organized under `cache/`.

