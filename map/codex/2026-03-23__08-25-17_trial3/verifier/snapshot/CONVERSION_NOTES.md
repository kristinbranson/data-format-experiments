# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement dataset from Steinmetz et al. reference materials provided in this directory
- **Date started**: 2026-03-23
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `.manifest`
- `CONVERSION_NOTES.md`
- `ChenLiuEtAl2023_SpikeSortingQC.pdf`
- `Dockerfile`
- `code/`
- `data/`
- `datapaper.pdf`
- `decoder.py`
- `docker-compose.yaml`
- `methodpaper.pdf`
- `methods.txt`
- `train_decoder.py`

Environment checks:
- `python3` available: `3.13.12`
- `numpy` import works: `2.3.5`
- `torch` import works: `2.6.0+cu124`
- Step 0 checkpoint passed: `ls -la CONVERSION_NOTES.md` confirmed file exists

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadmat` | `code/VideoAnalysisUtils/preprocessing_utils.py` | LOADING | Reads MATLAB exports and recursively converts nested MATLAB structs/cell arrays into Python dictionaries/lists. |
| `process_one_sess` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Loads all probe files for a session, extracts task/behavior variables, intersects ephys units with histology, and aggregates per-session data across probes. |
| `sliding_histogram` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Converts per-trial spike times into binned spike counts or firing rates; output firing rates have shape `(n_bins, n_trials, n_neurons)`. |
| `helper_get_neuron_id_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Uses QC-selected unit indices plus side assignment from CCF ML coordinate (`5700` midline) to select neurons for each side/region. |
| `process_one_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates spike times to analysis window, bins them into firing rates, and saves per-area session pickle files with behavior/task metadata. |
| `process_all_sess_parallel` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Batch driver for preprocessing all sessions in parallel. The Sherlock script sets `bw=0.04`, `stride=0.0034`, `begin_time=-3.0`, `end_time=3.0`, `qc_mode='classifier'`. |
| `load_session` | `code/VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Re-loads all per-area pickle files from one session and concatenates neurons across files into a single session dictionary. |
| `get_regular_trial_mask` | `code/VideoAnalysisUtils/population_decoding_utils.py` | CURATION | Defines “regular trials” as no early lick, no auto water, no free water, correctness not equal to `-1`, and no photostimulation (`stimulation[:,0] == 0`). |
| `create_4fold_trial_type_mask` | `code/VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Builds stratification labels from `trial_type` and `correctness`: hit right, miss right, hit left, miss left; pools misses if some classes are too small. |
| `temporal_alignment_embed_and_ephys` | `code/VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Aligns video-derived embeddings and ephys firing rates onto overlapping time points assuming common timestep `dt=0.0034 s`. |

### Notes
- `code/README.md` confirms this repository is the analysis code for the 2025 movement-encoding paper and points to processed MAP dataset usage.
- The primary reference preprocessing path is `code/Sherlock/preprocess_all_ephys.py`, which calls `preprocessing_DJ_2022Aug.process_all_sess_parallel(...)`.
- In `process_one_sess`, behavioral/task variables are read once from the first probe file of a session and then copied to all saved area files. Key variables include `early_lick_trials`, `auto_water_trials`, `free_water_trials`, `lick_directions`, `lick_times`, `correctness`, `gocue_time`, `delay_period`, `sample_period`, `stimulation`, and `trial_type`.
- Raw lick and stimulation times are explicitly shifted into go-cue coordinates by subtracting per-trial go cue time. Spike times are treated as already go-cue aligned in the exported files.
- Neuron curation in the reference code is not based on ad hoc thresholds inside the preprocessing script. Instead it relies on externally generated QC files (`goodunits`) and then further requires intersection with histology/unit IDs so only neurons with both ephys and anatomical assignment are retained.
- The preprocessed session format used throughout the paper stores data per area but downstream analysis concatenates areas back into one session matrix across neurons.
- Trial filtering for most downstream decoders excludes early-lick, auto-water, free-water, no-response, and stimulation trials. This is specific to the paper’s analyses and may differ from the decoder task here; I need to decide later whether to preserve all trials or reproduce this curation exactly where compatible with the requested outputs.
- `trial_type` is encoded in the reference preprocessing as `1` for left instruction (`'l'`) and `0` for right instruction (`'r'`).
- There is an internal comment inconsistency around `lick_directions`: one comment says `0=left, 1=right`, another says `0=right, 1=left`. This must be resolved from raw data and reference texts before final mapping.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` is a DANDI-style NWB dataset (`dandiset.yaml` + one folder per subject).
- There are `28` subject folders named `sub-<subject_id>`.
- Each session is a single NWB file inside its subject folder, for a total of `174` NWB files.
- File naming pattern: `data/sub-<subject>/sub-<subject>_ses-<datetime>_behavior+ecephys[+ogen].nwb`.
- `dandiset.yaml` identifies the dataset as **Mesoscale Activity Map Dataset** and reports `174` files and `28` subjects, matching the local copy.
- Representative NWB organization:
  - `intervals/trials`: trial table with columns `trial`, `photostim_onset`, `photostim_power`, `photostim_duration`, `trial_uid`, `task`, `task_protocol`, `trial_instruction`, `early_lick`, `outcome`, `auto_water`, `free_water`, plus `start_time`/`stop_time`.
  - `units`: spike-sorted units with metadata columns including `unit_quality`, `classification`, `anno_name`, QC metrics, `is_good_trials`, and ragged `spike_times`.
  - `acquisition/BehavioralEvents`: event time series for `presample`, `sample`, `delay`, `go`, `trialend`, `left_lick`, `right_lick`, and photostimulation start/stop times.
  - `acquisition/BehavioralTimeSeries`: continuous tracked positions. All sessions include `Camera0_side_JawTracking`, `Camera0_side_NoseTracking`, and `Camera0_side_TongueTracking`; some sessions additionally include whisker or lickport tracking.
- Tongue/jaw/nose tracking time series are shaped `(n_frames, 3)` with columns `(x, y, likelihood)`. Example `Camera0_side_TongueTracking` description: `('tongue_x', 'tongue_y', 'tongue_likelihood')`.
- Trial value encodings observed directly from NWB tables:
  - `trial_instruction`: `left` / `right`
  - `outcome`: `hit` / `miss` / `ignore`
  - `early_lick`: `early` / `no early`
  - `task`: `audio delay`
- Unit annotations:
  - `classification` contains `good` / `unlabelled` plus occasional `nan`.
  - `unit_quality` contains `good` / `multi`.
  - `anno_name` stores detailed Allen-style anatomical labels per unit; examples include `Caudoputamen`, `Secondary motor area, layer 5`, `Mediodorsal nucleus of thalamus`, `Midbrain reticular nucleus`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272227 raw units across NWB files |
| Neurons / session | min 493, mean 1564.52, max 3191 raw units |
| Subjects | 28 |
| Sessions / subject | min 3, mean 6.21, max 10 |
| Trials (total) | 94990 |
| Trials / session | min 264, mean 545.92, max 800 |

Additional raw-data statistics:
- Units with `classification == "good"`: `69453` total, min `0`, mean `399.16`, max `923` per session.
- Trial instruction counts across all raw trials: `right=48913`, `left=46077`.
- Outcome counts across all raw trials: `hit=65254`, `miss=15641`, `ignore=14095`.
- Early lick counts across all raw trials: `no early=84185`, `early=10805`.
- Sessions with tongue tracking: `174/174`.
- Sessions with at least one non-`N/A` photostimulation trial: `168/174`.
- Core trial-table schema and core behavioral-event schema are identical in all `174/174` sessions.

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units | “69,943 good units recorded across 173 behavioral sessions” |
| Neurons / session | ~404 good units/session mean (69943/173) | Derived from paper totals |
| Subjects | 28 mice | “based on data from 28 mice” |
| Sessions / subject | 173 behavioral sessions total | “173 behavioral sessions” |
| Trials (total) | not stated globally in text | Session average reported instead |
| Trials / session | mean 476, range 130-785 | “average 476 (Mean; range, 130-785) trials per session” |
| Neural data time bin | 40 ms bin width; 3.4 ms stride in method paper preprocessing | “binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms” |
| Behavior data time bin | 300 Hz video tracking | “High-speed videos … were acquired at 300 Hz” |
| Reward rate | 84% correct control-trial rate, range 65-99% | “84% correct rate (range, 65-99%)” |
| Photostim trial fraction | ~25% of trials in subset of mice/sessions | “deployed on a subset of ~25% randomly interleaved trials” |
| Probe insertions | 655 in STAR Methods; figure summary says 660 penetrations | “655 probe insertions” / “660 penetrations” |
| Good-unit fraction | 25.9% of Kilosort2 clusters | “25.9 % of clusters reported by Kilosort2” |
| Video->choice AUC before sample | 0.51 ± 0.06 s.d., n=106 sessions | “before the sample epoch … AUC … 0.51 ± 0.06” |
| Video->choice AUC sample+delay | 0.66 ± 0.12 s.d., n=106 sessions | “In the sample and delay epochs, the mean AUC increased significantly (0.66 ± 0.12)” |
| Video->choice AUC response | 0.99 ± 0.01 s.d., n=106 sessions | “Soon after the go cue, prediction saturated … 0.99 ± 0.01” |


### Processing Details
- Behavioral task structure from `methods.txt` / datapaper:
  - Sample epoch: three 150 ms tones with 100 ms inter-tone intervals.
  - Delay epoch: `1.2 s`.
  - Go cue: `0.1 s` auditory cue marking end of delay.
  - Response epoch / answer period: `1.5 s`.
  - Consumption period: `1.5 s`.
  - Trial ends after `1.5 s` without licking, followed by `250 ms` ITI.
- Temporal alignment convention in the reference preprocessing is go-cue-centered. The code subtracts go cue time from lick times and stimulation times, and the papers discuss peri-go-cue timing throughout (`Time to go (s)`).
- Photoinhibition timing: late-delay ALM photoinhibition lasts the final `0.5 s` of delay including `100 ms` ramp-down and ends before the go cue.
- Method paper preprocessing for the video/neural analysis:
  - Spikes converted to firing rates using `40 ms` bins and `3.4 ms` stride.
  - Only side-view video frames were used.
  - DeepLabCut markers included jaw, paws, and tongue.
  - Marker outliers were identified using a five-sigma frame-to-frame velocity rule and imputed from nearby frames.
  - Tongue position was imputed to its mean when occluded before the response epoch.
- For PSTHs in the datapaper, spike counts were binned at `1 ms` then boxcar-averaged over `200 ms`; that is for figure generation, not the main preprocessed dataset used by the method paper.

### Curation Steps

**Neuron curation rules**:
- Spike sorting performed with Kilosort2; each cluster assigned 15 quality metrics.
- Region-specific logistic-regression classifiers were trained from manually curated labels in five major area groups (cortex, striatum, thalamus, midbrain, medulla).
- The classifier outputs defined the “good” units used in the papers.
- Region mapping from `methods.txt`:
  - Cortex classifier: ALM, other cortex, hippocampus, olfactory, cortical subplate.
  - Striatum classifier: striatum, pallidum.
  - Thalamus classifier: thalamus, hypothalamus.
  - Midbrain classifier: midbrain, pons.
  - Medulla classifier: medulla, cerebellum.
- False alarm rates on held-out manually labeled units: cortex `7.8%`, striatum `6.4%`, thalamus `7.3%`, midbrain `5.5%`, medulla `4.3%`.

**Trial curation rules**:
- Early lick trials and no-response trials were excluded for analysis in the datapaper.
- Overall behavioral performance for session inclusion was computed on control trials only, excluding early licks.
- Sessions were selected for analysis with:
  - performance `> 65%`
  - at least `50` correct lick-left and `50` correct lick-right trials.
- Many downstream analyses in the method paper further use only correct trials, or only non-stimulation “regular” trials, depending on the analysis.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice from behavioral videos, pre-sample | ROC AUC `0.51 ± 0.06` (n=106 sessions) |
| Choice from behavioral videos, sample+delay | ROC AUC `0.66 ± 0.12` (n=106 sessions) |
| Choice from behavioral videos, response | ROC AUC `0.99 ± 0.01` (n=106 sessions) |
| Single-neuron choice / movement prediction during delay | ROC AUC used as metric in method paper; neurons considered modulated at AUC `> 0.6` in some analyses |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session count | Reference preprocessing / papers operate on analyzed sessions; code paths and text consistently refer to `173` behavioral sessions | Local NWB release contains `174` raw sessions | Datapaper / methods: `173 behavioral sessions` | Exactly one raw session (`sub-440958_ses-20190216T162508_behavior+ecephys+ogen.nwb`) has zero units with `classification == good`. Excluding that session yields `173` sessions, matching the papers. |
| Probe insertion count | Reference text methods: `655 probe insertions`; datapaper figure caption: `660 penetrations` | Raw NWB HDF5 electrode-group count is `659` insertions across 174 sessions; excluding the zero-good-unit session yields `655` | Figure caption says `660 penetrations`, STAR Methods says `655 probe insertions` | Use `655` as the analyzed-set count because it matches the code/paper analyzed session set after excluding the zero-good-unit session. Treat `660` as a figure-caption inconsistency or broader raw-count convention. |
| Good-unit total | Reference code uses external classifier-produced `goodunits` files; whitepaper/papers report `69,943` good units | NWB `classification == good` gives `69,453` units | Papers: `69,943 good units` | Small residual mismatch of `490` units remains. Most likely causes are release/version differences or a difference between the external `goodunits` files used by the reference code and the NWB `classification` field. For NWB-native conversion, `classification == good` is the closest directly available analogue and matches the session / insertion counts after excluding the zero-good session. I will keep checking this later against decoder behavior and dataset statistics. |
| Trial filtering | `get_regular_trial_mask` excludes early lick, auto water, free water, no-response, and stimulation trials | Raw trials include all of these categories | Papers describe exclusion of early lick and no-response trials; some downstream analyses also restrict to correct or control-only trials | Distinguish three levels: raw trial table, paper session-selection criteria, and stricter downstream “regular trial” filters. The conversion should preserve requested outputs and only apply exclusions required by the decoder task or by clear reference consistency. |
| Trial/event alignment | Reference code assumes go-cue-aligned exported spike/behavior variables and stores one trial row per trial | In NWB, `go_start_times` matches trial count exactly, but `sample_start_times` differs in `173/174` sessions and `delay_start_times` differs in `174/174` sessions because replayed epochs are stored explicitly | Papers state that early licks triggered replay of sample/delay epochs | Use `go_start_times` as the unique per-trial alignment anchor. When sample/delay timing is needed, use the final sample/delay event within each trial window rather than assuming one event per trial. |
| Photostim timing representation | Reference code stores stimulation times relative to go cue after subtracting go cue time | NWB trial table stores `photostim_onset` relative to trial start; in example trials, subtracting `(go_time - trial_start)` gives about `-1.2 s`, matching late-delay stimulation | Papers: photoinhibition in last `0.5 s` of delay, ending before go cue | Convert photostimulation onset/offset from trial-start coordinates into go-aligned coordinates during conversion. |
| Trial instruction coding | Reference preprocessing stores `trial_type = 1*(trial_type == 'l')` | NWB trial table stores strings `left` / `right` | Papers discuss lick-left / lick-right trial types | Map NWB `trial_instruction` to left/right categorical outputs directly; this is consistent with the reference code once recoded. |

Final consistency understanding:
- The NWB files are the raw session-level source.
- The paper/code analyzed dataset is best approximated in NWB as: sessions with at least one good classified unit, using go-cue-centered alignment, and QC-filtered units corresponding to `classification == good`.
- `go_start_times` is the reliable one-per-trial event; sample/delay event streams need within-trial disambiguation because of replayed epochs.
- The method paper’s `40 ms` / `3.4 ms` neural preprocessing describes the reference analysis pipeline, but the requested decoder conversion will intentionally deviate to `50 ms` bins while preserving the same alignment and curation principles where possible.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times`, `units/classification`, `acquisition/BehavioralEvents/go_start_times` | `neural` | Keep only units with `classification == "good"`; exclude the single zero-good-unit session; bin absolute spike times into go-aligned `[-2.5, 1.5)` windows using `50 ms` bins; convert counts to firing rates in Hz | `process_one_sess`, `process_one_area`, `sliding_histogram` | One trial array per trial, shape `(n_good_units, 80)` |
| Task structure from papers (`sample = 0.65 s`, `delay = 1.2 s`) + go cue | `input[0]` | Compute `time_from_tone_onset_s = bin_center - (-1.85 s)` for every bin | `process_one_sess` stores sample/delay periods; go-centered alignment throughout reference code | Raw NWB sample-event streams contain replay-related extra events and are not reliable one-to-one trial markers; using the canonical `-1.85 s` tone onset is more consistent with the task definition |
| `intervals/trials/photostim_onset`, `photostim_duration`, `start_time`, `go_start_times` | `input[1]` | Convert photostim onset/duration from trial-start coordinates to go-centered coordinates; emit binary `0/1` per bin center for photostim on/off | `process_one_sess` aligns stimulation by subtracting go cue | Keep stimulation trials because photostim is an explicit decoder input |
| `intervals/trials/trial_instruction`, `intervals/trials/outcome`, lick events | `output[0]` (`choice`) | Encode left=`0`, right=`1`; for `hit`, use instructed side; for `miss`, use opposite side; for `ignore`, use first post-go lick side if present, else first lick side anywhere in trial if present, else fall back to instructed side | `process_one_sess` stores `trial_type`; lick-direction handling is implicit in exported behavior variables | Output is constant across time bins for a trial |
| `intervals/trials/outcome` | `output[1]` (`outcome`) | Map `ignore -> 0`, `miss -> 1`, `hit -> 2` | Paper/task variable; used in trial filtering logic throughout code | Constant across time bins |
| `intervals/trials/early_lick` | `output[2]` (`early_lick`) | Map `no early -> 0`, `early -> 1` | `get_regular_trial_mask` and datapaper methods | Constant across time bins |
| `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` `(x, y, likelihood)` + timestamps | `output[3]` (`tongue_y_position`) | Clean tracking using method-paper-inspired velocity-outlier interpolation; align to go-centered bin centers using last-frame-carried-forward; compute per-session 40th/60th percentiles over all included aligned bins; discretize y into `0/1/2` | Method paper preprocessing + `align_markers_between_lims` | Time-varying output over 80 bins |
| `subject` folder / NWB subject metadata | `subjects`, `subject_idx` | Use exact subject IDs (`sub-xxxxx`) and map each session to its subject index | N/A | Session order will follow sorted NWB file paths |
| `units/anno_name` for good units | `brain_regions`, `brain_region_idx` | Use unique detailed anatomical annotations as region labels and map each good unit to its annotation index | Reference code keeps detailed `ccf_label` per neuron | More faithful at unit level than probe insertion target |
| Fixed task specification | `input_names`, `output_names`, `output_values`, `metadata` | Populate names, class labels, and alignment/binning metadata exactly for the decoder task | N/A | Metadata will explicitly document the requested 50 ms deviation from reference 40 ms / 3.4 ms preprocessing |

### Key Decisions
1. **Session inclusion**: Use all sessions with at least one good classified unit (`173` sessions). This matches the paper session count and avoids the single raw session with zero analyzable units.
2. **Trial inclusion**: Keep all trials from included sessions that have full neural coverage for the requested `[-2.5, 1.5)` go-aligned window according to `units/obs_intervals`; within that valid-coverage set, retain stimulation, early-lick, ignore, and miss trials because they are required by the decoder inputs/outputs. This intentionally differs from the paper’s “regular trial” mask, and the difference is required by the user task.
3. **Neural unit filter**: Use NWB `classification == "good"` as the closest NWB-native equivalent of the reference code’s external `goodunits` files.
4. **Temporal anchor**: Use `go_start_times` as the unique per-trial alignment event. Do not assume one sample/delay event per trial because replayed epochs are explicitly stored in NWB.
5. **Sample/tone onset definition**: Use the canonical tone onset at `-1.85 s` relative to go cue from the task structure (`0.65 s` sample epoch + `1.2 s` delay), because raw NWB sample-event streams contain replay-related extra events.
6. **Input representation**: Both decoder inputs will be fully time-varying arrays of shape `(2, 80)` per trial.
7. **Output representation**: All outputs will be stored as time-varying arrays of shape `(4, 80)` per trial; choice, outcome, and early-lick labels will be repeated across time, while tongue position varies across bins.
8. **Ignore-trial choice edge case**: Because most ignore trials have no post-go lick (`13770 / 14095`), choice must use a documented fallback. I will use lick-derived side when available, otherwise instructed side.
9. **Tongue alignment**: Use last-frame-carried-forward at each bin center to mirror the reference alignment scripts rather than linear interpolation.
10. **Tongue discretization support set**: Compute per-session percentiles from the aligned tongue-y samples that actually enter the converted dataset, not from unrelated off-trial video periods.
11. **Brain-region labels**: Use detailed `anno_name` labels per good unit rather than coarse probe targets, because they are unit-resolved and directly stored in NWB.
12. **Required deviation from reference bins**: Use `50 ms` bins because the decoder task explicitly requires this, while preserving the reference session/unit curation and go-cue alignment principles.

### Planned Sanity Checks
- [ ] Neural sanity check: for selected session/trial/unit/bin, recompute spike counts directly from raw spike times and verify converted firing rate equals `count / 0.05` with `np.allclose()`.
- [ ] Input sanity check: for selected stimulated and non-stimulated trials, verify `time_from_tone_onset_s` and `photostim_on` against raw sample/go/stimulation event times with `np.allclose()`.
- [ ] Output sanity check (trial-level): verify outcome and early-lick labels exactly match raw trial table values for selected trials.
- [ ] Output sanity check (choice): verify hit/miss trials match lick- or instruction-derived choice, and explicitly inspect ignore-trial fallback behavior on selected trials.
- [ ] Output sanity check (tongue): for selected session/trial/bin, verify aligned tongue y and discretized class match direct recomputation from raw tracking timestamps/data.
- [ ] Coverage check: confirm all included sessions have at least 2 trials and all included trials have 80 aligned bins with no NaNs.

---

## Step 6: Script Development
**Status**: COMPLETE

- Implemented `convert_data.py` with CLI:
  - `python -u convert_data.py <outpicklefile>`
  - `--full`
  - `--sample`
  - `--show-processing`
- Main implementation pieces:
  - NWB loading via `h5py` rather than higher-level wrappers for lower overhead and explicit access to ragged arrays.
  - Session discovery and exclusion of the single zero-good-unit session.
  - Good-unit filtering from `units/classification == "good"`.
  - Trial-validity filtering from `units/obs_intervals` so only trials with complete neural coverage for the requested go-aligned window are retained.
  - Spike binning into `50 ms` go-aligned bins and conversion to firing rate in Hz.
  - Time-varying decoder inputs for canonical time-from-tone-onset and photostimulation on/off.
  - Time-varying decoder outputs for choice, outcome, early lick, and discretized tongue-y position.
  - Per-session processing visualization saved to `processing_<session_id>.png` in `--show-processing` mode.
  - Shape/type validation and per-session timing prints.

Code inefficiencies identified:
- Initial implementation trusted the behavioral trial table directly and therefore processed many trials that were outside the ephys recording intervals, producing all-zero neural matrices and downstream verification warnings.
- Initial tongue discretization used percentile edges with `np.digitize`, which can collapse the middle class when `q40 == q60` or when many values land exactly on the percentile boundary.

Code speedups added:
- Added an `obs_intervals`-based valid-trial mask before spike binning, reducing wasted binning work on out-of-recording trials.
- Kept all heavy operations vectorized in NumPy and used `float16` for stored neural/input arrays and `int16` for outputs to reduce pickle size.
- Used search-sorted spike binning on per-unit spike vectors rather than per-bin Python loops.
- Reused aligned bin-center arrays and session-level tongue thresholds instead of recomputing them per trial.
- Added a residual all-zero-trial check after binning as a defensive guard against edge-case recording coverage mismatches.
- Replaced percentile-edge discretization with explicit `< q40`, `q40..q60`, `> q60` logic so the middle tongue class remains populated.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 |
| Neurons / session | mean 417, min 375, max 459 |
| Subjects | 1 (`sub-440956`) |
| Sessions / subject | 2 |
| Trials (total) | 370 |
| Trials / session | 233, 137 |
| `time_from_tone_onset_s` range | [-0.6, 3.3] |
| `photostim_on` range | [0.0, 1.0] |
| `choice` distribution | [0.446, 0.554] |
| `outcome` distribution | [0.200, 0.000, 0.800] |
| `early_lick` distribution | [0.941, 0.059] |
| `tongue_y_position` distribution | [0.067, 0.848, 0.086] |

### Processing Plots Review
- Generated plots for both sample sessions:
  - `processing_sub-440956_ses-20190207T120657_behavior+ecephys+ogen.png`
  - `processing_sub-440956_ses-20190208T133600_behavior+ecephys+ogen.png`
- The underlying converted sample passed `train_decoder.py --verify-only` with no errors or warnings after the `obs_intervals` trial-coverage fix and the tongue discretization fix.
- No further anomalies were flagged by the conversion script summaries: both sessions had nonzero neural activity, expected input ranges, and populated tongue classes.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| `obs_intervals` pre-filtering of invalid trials | Removes out-of-recording trials before spike binning; eliminates wasted work and zero-trial warnings |
| Vectorized spike binning / compact dtypes | Keeps sample conversion to `0.6-0.8 s` per session in the tested sessions |
| Session-level tongue thresholding reuse | Avoids repeated percentile recomputation |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion observed | `0.6-0.8 s / session` | `1.4 s` for 2 sessions |
| Full conversion estimate from per-session scaling | ~`0.7 s / analyzed session` | ~`2.0 min` for 173 sessions |
| Full conversion estimate from unit-trial-product scaling | N/A | ~`4.5 min` for 173 sessions / 76,033 valid trials / 69,453 good units |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| choice | 0.7963 | 0.7397 |
| outcome | 0.9047 | 0.8695 |
| early_lick | 0.8848 | 0.8086 |
| tongue_y_position | 0.8123 | 0.6805 |

Notes:
- Training ran on `295` trials and validated on `75` trials.
- Loss decreased from `13.0885` at epoch 1 to `0.3618` at epoch 200.
- Every output exceeded chance on validation:
  - choice: `0.7397` vs chance `0.5000`
  - outcome: `0.8695` vs chance `0.3333`
  - early_lick: `0.8086` vs chance `0.5000`
  - tongue_y_position: `0.6805` vs chance `0.3333`
- The sample subset contains no miss trials after neural-coverage filtering, so the outcome result is encouraging but not yet sufficient for full-dataset validation.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: `4.5G`
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | `69,943` good units | external `goodunits` files used by authors | `69,453` NWB `classification == good` | `69,453` | Matches NWB release; paper/code vs NWB release mismatch remains documented |
| Mean neurons/session | `~404.3` | `~404.3` implied by paper analyzed set | `401.46` | `401.46` | Close; matches NWB release |
| Subjects | `28` | `28` | `28` | `28` | Yes |
| Sessions | `173` | `173` | `173` sessions with at least one good unit | `173` | Yes |
| Trials (total) | not stated globally; `476` mean/session reported | analysis-specific trial masks | `76,033` windows passing first-unit `obs_intervals`; `73,910` with nonzero all-good-unit spikes | `73,910` | Yes for converted set; lower than paper raw-session mean because only full neural windows were retained |
| Trials/session (mean) | `476` | analysis-specific | `439.5` first-unit `obs_intervals`; `427.2` with nonzero all-good-unit spikes | `427.2` | Explained by full-window neural coverage requirement and removal of zero-spike windows |
| `time_from_tone_onset_s` range | task-defined from requested window | go-aligned time variables in reference code | `[-0.6, 3.3]` | `[-0.6, 3.3]` | Yes |
| `photostim_on` range | binary on/off | binary aligned stim epochs | `[0.0, 1.0]` | `[0.0, 1.0]` | Yes |
| `choice` distribution | roughly balanced left/right task | left/right trial coding in reference code | `[0.491, 0.509]` | `[0.491, 0.509]` | Yes |
| `outcome` distribution | `84%` correct control-trial rate reported | analysis-specific filters often exclude misses/ignores | `[0.173, 0.009, 0.818]` | `[0.173, 0.009, 0.818]` | Converted hit fraction is close to paper reward rate given coverage filtering and inclusion of stimulation trials |
| `early_lick` distribution | early licks excluded in many paper analyses | regular-trial mask excludes early licks | `[0.885, 0.115]` | `[0.885, 0.115]` | Yes for converted set |
| `tongue_y_position` distribution | not reported in papers | not reported | `[0.082, 0.834, 0.085]` | `[0.082, 0.834, 0.085]` | Yes |

Additional Step 9 notes:
- Full conversion runtime was `7.24 min`, below the `15 min` optimization threshold.
- `train_decoder.py converted_data.pkl --verify-only` reported: `Data format is valid, no errors or warnings.`
- Spot-check of raw data showed that the `2,123` trials removed after the initial `obs_intervals` coverage filter were genuinely all-zero across **all** good units in the requested `[-2.5, 1.5)` window, even when neighboring trials contained tens of thousands of spikes. These appear to be raw session-wide silent gaps rather than a conversion bug.
- Example spot-check:
  - Session `sub-484676_ses-20210419T130102_behavior+ecephys+ogen`: first-unit `obs_intervals` yielded `507` candidate trials; converted data retained `469`.
  - Direct raw recomputation showed dropped trials such as go times `308.8702 s`, `922.9372 s`, and `4167.0008 s` had exactly `0` spikes across all `530` good units, while adjacent retained trials had `24,783-32,862` spikes in the same `4 s` window.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` reported `Data format is valid, no errors or warnings.` No fixes were required from the verifier.
2. **Raw-vs-converted sanity checks**: Wrote and ran `cache/step10_sanity_checks.py` against the raw NWB files and `converted_data.pkl`. All selected checks passed with `np.allclose()`:
   - Neural firing-rate check:
     - Session `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`
     - Converted trial `0`, unit `0`, bin `4`
     - Direct raw spike count / `0.05 s` matched the stored firing rate exactly.
   - Input checks:
     - Full `time_from_tone_onset_s` vector matched the independent formula from fixed bin centers and canonical tone onset.
     - Photostimulation vector for session `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`, trial `44` matched direct recomputation from raw `photostim_onset`, `photostim_duration`, `start_time`, and `go_time`.
   - Output checks:
     - Hit-trial check: session `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`, trial `0`
     - Ignore-trial check: session `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`, trial `2`
     - Miss-trial check: session `sub-440958_ses-20190215T141028_behavior+ecephys+ogen`, trial `371`
     - In each case, independently recomputed outcome, early-lick, and choice vectors matched the converted outputs exactly.
   - Tongue check:
     - Independently recomputed cleaning, alignment, percentile thresholds, and discretization matched the converted tongue classes for session `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`, trial `0`.
3. **Reference code comparison**:
   - Data loading:
     - Reference code loads author-exported `.mat` session structures (`loadmat`, `load_session`) that are already go-aligned.
     - Conversion code loads raw NWB files directly with `h5py` and reconstructs the same session/trial organization from source event and unit tables.
   - Neuron filtering:
     - Reference code uses external `goodunits` classifier outputs.
     - Conversion uses NWB `classification == "good"` as the closest NWB-native analogue.
     - This is the main remaining release-level discrepancy versus the paper total (`69,943` vs `69,453` in NWB).
   - Trial filtering:
     - Reference code often applies `get_regular_trial_mask` to exclude early-lick, no-response, auto-water, free-water, and stimulation trials for specific analyses.
     - Conversion intentionally retains stimulation, ignore, miss, and early-lick trials when they have valid neural coverage, because these variables are required decoder inputs/outputs.
   - Temporal alignment:
     - Reference code and papers are consistently go-cue-centered.
     - Conversion also centers everything on go cue and converts photostimulation timing into go-centered coordinates.
   - Binning:
     - Reference method-paper preprocessing uses `40 ms` bins with `3.4 ms` stride.
     - Conversion uses `50 ms` non-overlapping bins because that is an explicit task requirement.
   - Video / tongue processing:
     - Reference code aligns behavioral markers by last-frame-carried-forward and cleans outliers with a velocity rule.
     - Conversion mirrors these principles for side-view tongue `y`.
4. **Key statistics comparison**:
   - Subjects: `28` in papers, raw NWB, and converted data.
   - Sessions: `173` analyzed sessions after excluding the single zero-good-unit NWB session.
   - Probe insertions: raw NWB count matches the paper analyzed-set value (`655`) once the zero-good-unit session is excluded.
   - Good units: converted data matches NWB release count (`69,453`); paper/code report `69,943`, likely reflecting a different release or external `goodunits` snapshot.
   - Trials: converted data retains `73,910` full neural windows after removing `2,123` raw windows with zero spikes across all good units.
5. **Edge-case checks**:
   - Confirmed all converted trials have `80` time bins and there are no all-zero neural trials.
   - Confirmed `photostim_on` handles `N/A` values as all-zero vectors.
   - Confirmed the tongue middle class remains populated after changing discretization from percentile-edge binning to explicit `< q40`, `q40..q60`, `> q60`.
   - Confirmed ignore-trial choice fallback behaves as documented when there is no post-go lick.

### Issues Found and Resolved
- **Issue**: Full-format verification initially showed many all-zero neural trials in the sample conversion.
  **Resolution**: Added the `obs_intervals` trial-coverage filter and retained a defensive post-binning all-zero-trial removal step.
- **Issue**: Tongue discretization initially collapsed the middle class in tied-percentile cases.
  **Resolution**: Replaced `np.digitize` with explicit threshold logic.
- **Issue**: A residual paper-vs-NWB good-unit mismatch (`69,943` vs `69,453`) remains after all checks.
  **Resolution**: No code fix is possible from the local NWB release; documented as a source-data / release discrepancy rather than a conversion bug.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| choice | 0.7727 | 0.7420 | Above chance; slightly below `1.5 x` chance (`0.75`) |
| outcome | 0.8619 | 0.7386 | Strong despite severe class imbalance |
| early_lick | 0.8099 | 0.7574 | Above `1.5 x` chance |
| tongue_y_position | 0.7929 | 0.7537 | Strong time-varying behavioral decoding |

Additional Step 11 notes:
- Training split: `59,063` train trials / `14,847` validation trials.
- Device: `cuda`
- Final test loss: `0.5359`
- Loss milestones:
  - epoch 1: `21.1037`
  - epoch 10: `6.8316`
  - epoch 50: `1.6247`
  - epoch 100: `0.7537`
  - epoch 150: `0.5350`
  - epoch 200: `0.4605`

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| choice | `0.7420` balanced accuracy | No directly matching neural-decoder number reported; strong post-go choice/movement signal expected from both papers |
| outcome | `0.7386` balanced accuracy | No directly matching paper number found |
| early_lick | `0.7574` balanced accuracy | No directly matching paper number found |
| tongue_y_position | `0.7537` balanced accuracy | No directly matching paper number found; strong movement encoding expected |

[Analysis of any low accuracies]
- No output was below chance.
- No output had validation accuracy below `1.5 x` chance except `choice`, which was `0.7420` versus a heuristic threshold of `0.7500`. This is a very small miss (`0.008`) and does not by itself indicate a bug.
- Additional evidence against a conversion bug for `choice`:
  - Raw-vs-converted hit / miss / ignore choice checks passed in Step 10.
  - Temporal alignment and photostim timing checks passed.
  - Train/validation gap for choice is small (`0.7727` vs `0.7420`; ratio `1.04`), so there is no sign of leakage or severe overfitting.
  - The papers report very strong movement-related signals near and after go cue, and the converted decoder window spans both weak pre-go and strong post-go periods; a single whole-window balanced-accuracy summary is therefore not directly comparable to the paper’s epoch-specific AUC results.

### Issues Found and Resolved
- **Issue**: Need to rule out accuracy failures below chance or major train/validation gaps.
  **Resolution**: No such failures were found. All outputs were comfortably above chance, and all train/validation ratios were well below the `1.5 x` overfitting concern threshold.
- **Issue**: `choice` accuracy was slightly below the heuristic `1.5 x chance` line.
  **Resolution**: Investigated against Step 10 sanity checks and training behavior; found no evidence of a conversion error. Retained current conversion logic.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Created files:
- `README.md`
- `cache/README_CACHE.md`
- `cache/step10_sanity_checks.py`
