# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement dataset
- **Date started**: 2026-09-21
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

Environment verification:
- `python3` runs successfully
- `numpy` import OK, version `2.4.4`
- `torch` import OK, version `2.6.0+cu124`
- Checkpoint verified with `ls -la /app/CONVERSION_NOTES.md`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `sliding_histogram` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Bin go-cue-aligned spike times into spike counts or firing rates using specified bin width and stride. Returns `(bin_centers, fr)` with `fr` shaped `(n_bins, n_trials, n_neurons)`. |
| `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Load all probe `.mat` files for one session, extract behavior/task variables, align lick/stimulation times to go cue, concatenate probes, and prepare session-level structures. |
| `process_one_sess` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Load classifier QC files from `goodunits`, intersect ephys units with histology, and restrict to QC-approved neurons. |
| `process_one_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Subset neurons for one side-region, truncate spike times to analysis window, bin to firing rates, and save region-level pickles. |
| `helper_get_neuron_id_area` | `VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Build side-specific region neuron indices by intersecting QC neuron IDs with hemisphere split at ML midpoint `5700` and CCF annotations. |
| `load_session` | `VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Load all per-area pickles for a session, concatenate neural data across areas, and expose behavior/task metadata plus CCF coordinates/labels. |
| `get_regular_trial_mask` | `VideoAnalysisUtils/population_decoding_utils.py` | CURATION | Define “regular” trials as no early lick, no auto water, no free water, no no-response, and no photostimulation. |
| `temporal_alignment_embed_and_ephys` | `VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Align time bases of video embeddings and electrophysiology by overlapping valid time support with `dt=0.0034 s`. |
| `align_markers_between_lims` | `Sherlock/align_markers.py` | PROCESSING | Align tracked marker trajectories to go cue with frame step `0.0034 s`; per bin uses the last frame in `[t-dt, t)`. |
| `get_bad_trial_inds` | `Sherlock/align_markers.py` / `align_embed_vecs.py` | CURATION | Remove video trials whose frame count is inconsistent with trial end time. |

### Notes
- Repository README identifies this as MAP dataset analysis code for the movement-encoding paper.
- The core reference preprocessing is for electrophysiology, not imaging. No delta-F/F computation is relevant.
- Raw session files are MATLAB exports from DataJoint. Spike times are already go-cue aligned in the raw data according to `process_one_area`.
- Behavior/task fields used by the reference code include `behavior_early_report`, `behavior_is_auto_water`, `behavior_is_free_water`, `behavior_lick_directions`, `behavior_lick_times`, `behavior_report`, `task_cue_time`, `task_delay_time`, `task_sample_time`, `task_stimulation`, and `task_trial_type`.
- Lick times and stimulation on/off times are explicitly shifted by subtracting the go cue time for each trial.
- Neural curation in the reference code is driven by per-session classifier QC files in `goodunits`, not by ad hoc thresholds in the script itself.
- Only neurons with both electrophysiology metadata and histology entries are retained.
- Reference code organizes neurons by bilateral high-level regions: `ALM`, `Medulla`, `Midbrain`, `Striatum`, `Thalamus`, `Pons`, `Cerebellum`, `Hypothalamus`, `Hippocampus`, `Orbital`, `OtherCortex`, `Olfactory`, `CorticalSubplate`, `Pallidum`.
- Session preprocessing in `Sherlock/preprocess_all_ephys.py` uses `bw=0.04 s`, `stride=0.0034 s`, `begin_time=-3 s`, `end_time=3 s`, and `qc_mode='classifier'`.
- For the current decoder task I will preserve the same source loading, QC logic, and go-cue alignment where applicable, but adapt the neural binning/window to the required `50 ms` bins and `[-2.5, +1.5] s` interval.
- The reference trial-quality mask excludes early lick, auto water, free water, no-response trials, and stimulated trials for “regular trial” analyses. I will decide later whether full-task decoder targets should include all trials or a curated subset, based on consistency across code, data, and papers.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` is a DANDI/NWB dataset (`dandiset.yaml` + one folder per subject).
- There are `28` subject folders named `sub-<mouse_id>`.
- Each session is stored as one NWB file inside its subject folder, for a total of `174` NWB files.
- File naming pattern: `sub-<subject>_ses-<timestamp>_behavior+ecephys[+ogen].nwb`.
- NWB top-level groups in a sample file: `acquisition`, `analysis`, `general`, `intervals`, `processing`, `stimulus`, `units`, plus identifiers/timestamps metadata.
- Trial data live in `intervals/trials`. Trial columns present in sample files:
  - `start_time`, `stop_time`, `trial`, `photostim_onset`, `photostim_power`, `photostim_duration`, `trial_uid`, `task`, `task_protocol`, `trial_instruction`, `early_lick`, `outcome`, `auto_water`, `free_water`
- Neural data live in `units`:
  - Unit QC/metadata columns include `classification`, `unit_quality`, `presence_ratio`, `isi_violation`, `amplitude_cutoff`, `avg_firing_rate`, `drift_metric`, `is_good_trials`, `anno_name`, `spike_times`, `electrodes`, `electrode_group`, waveform metrics, etc.
- Behavioral event streams live in `acquisition/BehavioralEvents`:
  - `presample_*`, `sample_*`, `delay_*`, `go_*`, `trialend_*`, `left_lick_times`, `right_lick_times`, `photostim_start_times`, `photostim_stop_times`
- Continuous behavior/tracking streams live in `acquisition/BehavioralTimeSeries`:
  - `Camera0_side_TongueTracking`, `Camera0_side_JawTracking`, `Camera0_side_NoseTracking`
  - In a sample file, each tracking sample has shape `(n_frames, 3)`, consistent with `x`, `y`, and a confidence/likelihood-like channel.
- All `174` sessions contain the three side-camera tracking streams.
- `168` sessions are `+ogen` sessions and show at least some non-`N/A` photostimulation trial entries; `6` sessions are ephys-only and have `N/A` photostimulation columns throughout.
- Trial categorical values observed across the full dataset:
  - `task`: only `audio delay`
  - `task_protocol`: only `1`
  - `trial_instruction`: `left`, `right`
  - `early_lick`: `early`, `no early`
  - `outcome`: `hit`, `miss`, `ignore`

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272227 raw units across all sessions |
| Neurons / session | mean 1564.52, min 493, max 3191 |
| Subjects | 28 |
| Sessions / subject | min 3, max 10 |
| Trials (total) | 94990 |
| Trials / session | mean 545.92, min 264, max 800 |

Additional dataset counts:
- Units with non-empty `anno_name`: `71305` total, mean `409.80` per session
- Trial instruction counts: `48913` right, `46077` left
- Early lick counts: `10805` early, `84185` no early
- Outcome counts: `65254` hit, `15641` miss, `14095` ignore

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69943 good units | `"69,943 good units recorded across 173 behavioral sessions"` |
| Neurons / session | 404.29 good units/session (derived) | `"69,943 good units"` and `"173 behavioral sessions"` |
| Subjects | 28 mice | `"Mice (n = 28, Table S1)"` |
| Sessions / subject | 6.18 mean (derived from 173/28) | `"28 mice"` and `"173 behavioral sessions"` |
| Trials (total) | ~82348 analyzed trials (derived from 173 x 476 mean) | `"476 ... trials per session"` |
| Trials / session | mean 476, range 130-785 | `"476 ... trials per session with 84% correct rate"` |
| Neural data time bin | 40 ms for video-to-neural prediction; 200 ms causal windows for population choice decoder | `"40 ms bin width"` and `"200 ms ... step size of 10 ms"` |
| Behavior data time bin | 300 Hz video (~3.33 ms/frame; reference code uses 3.4 ms) | `"acquired at 300 Hz"` |
| Reward rate | 84% correct control-trial performance, range 65-99% | `"84% correct rate (range, 65-99%)"` |
| Photostim trial fraction | ~25% of trials in photoinhibition subset | `"typically 25%"` / `"subset of ~25% randomly interleaved trials"` |
| Photostim sessions | 93 sessions in 17 VGAT-ChR2-EYFP mice | `"N = 17 ... n = 93 sessions"` |
| QC classifier quality | AUC > 0.9 average in 10-fold cross-validation | `"AUC was on average > 0.9"` |
| False alarm rates | cortex 7.8%, striatum 6.4%, thalamus 7.3%, midbrain 5.5%, medulla 4.3% | `"false alarm rate: 7.8% ... 4.3%"` |


### Processing Details
- Behavioral task:
  - Sample epoch: three 150 ms tones with 100 ms inter-tone intervals
  - Delay epoch: `1.2 s`
  - Go cue: `0.1 s`
  - Response/answer period: `1.5 s`
  - Consumption period: `1.5 s`
- Temporal reference in the papers/code is the go cue for neural and movement analyses.
- Early lick trials trigger replay behavior online, and early lick plus no-response trials were excluded from many paper analyses.
- Control-trial performance in the papers excludes photostim trials and early licks.
- Experimental sessions used in the data paper analysis were selected with:
  - behavioral performance `> 65%`
  - at least `50` correct lick-left and `50` correct lick-right trials
- Multi-view video was acquired at `300 Hz`; reference code aligns frame-based streams to go cue with `dt=0.0034 s`.
- For predicting neural activity from behavioral video, the papers used:
  - marker tracking and/or 16-D autoencoder embeddings
  - linear regression
  - `40 ms` neural bin width
- For population choice decoding in the data paper:
  - causal `200 ms` sliding windows
  - `10 ms` step
  - logistic regression
  - nested `5-fold` CV, preserving `<LL>, <RR>, <LR>, <RL>` ratios
- Method paper shows movement-related neural predictability strongest in medulla, then midbrain, then forebrain regions.
- Method paper also reports direct choice prediction from video:
  - pre-sample AUC `0.51 ± 0.06`
  - sample/delay AUC `0.66 ± 0.12`
  - response AUC `0.99 ± 0.01`

### Curation Steps

**Neuron curation rules**:
- Raw traces were band-pass filtered `250 Hz - 9 kHz`.
- Kilosort2 output clusters were scored with `15` QC metrics.
- Recordings with substantial drift were rejected before downstream analysis.
- Five region-specific logistic-regression classifiers were trained from manually curated labels:
  - cortex classifier for ALM/other cortex/hippocampus/olfactory/cortical subplate
  - striatum classifier for striatum/pallidum
  - thalamus classifier for thalamus/hypothalamus
  - midbrain classifier for midbrain/pons
  - medulla classifier for medulla/cerebellum
- Only units classified as `good` were used in the paper analyses.

**Trial curation rules**:
- Early lick and no-response trials excluded for core task analyses in the data paper.
- Control-trial performance excludes photostimulation trials.
- Photoinhibition analysis additionally required at least `10` photostimulation trials per unit.
- Method paper screened sessions for video artifacts and excluded affected sessions.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice from behavioral video (method paper) | AUC `0.51 ± 0.06` pre-sample, `0.66 ± 0.12` sample/delay, `0.99 ± 0.01` response |
| Choice from neural populations (data paper) | Exact values shown graphically in Fig. 6 / Fig. S6; clearly above chance before go cue, highest/earliest in ALM |

Additional paper statistics to use later:
- Good-unit counts by major brain compartment (data paper Figure 1 / Figure 2): `ALM 8717`, `orbital 10223`, `striatum 7664`, `pallidum 1092`, `hippocampus 1944`, `thalamus 12808`, `hypothalamus 815`, `midbrain 7495`, `pons 347`, `medulla 2928`, `cerebellum 1820`, `other areas 14090` (sum = `69943`).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Raw format | Reference code loads DataJoint-exported `.mat` files with behavior/task/spike arrays and separate QC `.mat` files | Local archive is NWB (`DANDI:000363/0.230822.0128`) with trials/events/time-series and unit tables | Papers describe the same experiment and NWB data release in DANDI | Treat NWB as the authoritative raw format available here; map NWB trial/event/unit fields to the variables used by the reference code. |
| Session count | Data paper/white paper report `173 behavioral sessions`; method-paper code/README references a later DANDI release | Local data contain `174` NWB files | Data paper reports `173`; method paper used subsets such as `105` sessions for video analyses | One NWB session (`sub-440958_ses-20190216T162508...`) has `classification == "nan"` for all `1852` units and no usable good-unit labels. Excluding that session yields `173` analyzable sessions, consistent with the papers. |
| Unit QC source | Reference code uses external per-session `goodunits` classifier outputs and intersects with histology | NWB stores unit-level `classification` (`good`/`unlabelled`) directly, plus `unit_quality` and `anno_name` | White paper says analysis used units labeled `good` by region-specific classifiers | Use `units.classification == "good"` as the NWB equivalent of the paper’s good-unit lists. |
| Good-unit total | Papers report `69943` good units | NWB `classification == "good"` totals `69453` units | Data paper + white paper say `69943` | Small residual gap (`490` units, ~0.7%) remains. Most likely explanation is archive/version mismatch: code README cites DANDI `0.231012.2129`, local data are `0.230822.0128`. Because NWB column description explicitly defines `classification` as the single-unit QC label used for the paper, I will use `classification == "good"` despite the small count mismatch. |
| Histology / region labels | Reference code combines QC lists with histology labels and hemisphere split to create broad paper regions | NWB provides fine-grained `anno_name` per unit and broad probe insertion region in electrode `location["brain_regions"]` | Papers analyze both broad compartments and finer subregions | Use unit-level annotations where possible rather than insertion-site labels; later map `anno_name` into decoder brain-region labels while preserving unit specificity. |
| Trial/session filtering | Papers describe excluding early lick and no-response trials; session inclusion based on performance and correct-left/right counts | Naively applying NWB trial-table fields yields only `152` passing sessions, not `173` | Papers say `173` analyzed sessions after those criteria | Interpret the performance-based session filter as analysis-specific and not directly reproducible from archived NWB trial columns alone. Reference preprocessing code also loads sessions first and applies trial masks later. For conversion, keep analyzable sessions and represent trial outcomes/early lick explicitly rather than hard-dropping them up front. |
| Photostim fields | Reference code stores stimulation with on/off times relative to go cue | NWB stores trial-level photostim columns plus absolute event timestamps for photostim start/stop | Papers state late-delay photostim ended before the go cue | Reconstruct time-varying photostim from NWB event timestamps relative to trial go cue, which is equivalent in content to reference code after alignment. |
| Lick/tone/go alignment | Reference code subtracts go cue from lick and stimulation times and bins spikes around go cue | NWB stores absolute timestamps for go/sample/delay/lick/photostim events and absolute trial start/stop times | Papers define sample/delay/go epochs explicitly | Align all streams by subtracting per-trial go-cue onset, matching the reference code and decoder task. |
| Choice window semantics | Reference papers define a `1.5 s` answer period after the go cue | NWB `go_stop_times` are not consistent across sessions: early sessions use `go_stop - go_start = 1.5 s`, but later sessions often use about `0.05 s` | Papers define the answer period explicitly as `1.5 s` | Derive choice from the first lick in `[go_start, min(go_start + 1.5 s, trial_stop)]`, not from the raw `go_stop_times` field. |
| `is_good_trials` field | Reference code does not rely on an analogous field during preprocessing | NWB `units/is_good_trials` is inconsistent: `9` sessions have widths not equal to the trial table, and semantics appear insertion-specific | Papers do not describe using this field for the published preprocessing | Do not use `is_good_trials` for session-wide trial filtering in the conversion. Restrict filtering to explicit event-validity checks that are well defined in NWB. |
| Neural coverage vs trial table | Reference MATLAB preprocessing starts from already aligned/usable spike data per session | In NWB, some sessions have behavioral trial tables that extend beyond the actual usable spike-time support. Example: `sub-440956_ses-20190208T133600...` has `480` trial-table rows, but only `160` trials whose decoder window fits inside the shared `units/obs_intervals` support; one additional late trial still produced an all-zero neural tensor. | Papers do not mention this archive edge case explicitly. | Require the full decoder window to lie inside the shared good-unit `obs_intervals` support, and conservatively drop any remaining trial whose binned neural activity is all zero. |
| Video subset used in method paper | Method paper reports `n = 105 sessions` for video-prediction comparisons | All `174` local sessions contain side-camera tracking time series | Method paper also notes sessions with video artifacts were excluded | Decoder task only needs tongue y from side-camera tracking, not the exact video-modeling subset. I will use sessions with valid neural data and required tracking, while handling missing/low-confidence tongue samples explicitly. |

Final understanding after consistency check:
- The datapaper/white-paper analysis target is best approximated from the local NWB archive by:
  - excluding the one all-`nan` classification session
  - keeping units with `classification == "good"`
  - aligning all streams to per-trial go cue
  - preserving trial types instead of pre-applying paper-specific analysis masks, because the decoder task explicitly needs outcome and early-lick labels
- The later method-paper code is informative about alignment and movement analyses but uses a later release/subset, so exact unit/session counts need not match that paper.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times` + per-trial go cue timestamps + `units/classification` | `neural` | Keep units with `classification == "good"`. For each trial, subtract go cue time and histogram spikes into non-overlapping `50 ms` bins on `[-2.5, 1.5)` s, then divide by `0.05` to get firing rate (Hz). | `sliding_histogram`, `process_one_area` | This preserves the reference go-cue alignment and spike-time processing but changes bin width/window to match the decoder task. |
| `acquisition/BehavioralEvents/sample_start_times` + trial intervals + go cue | `input[0]` | For each trial, find the last `sample_start_time` between trial start and go cue. Compute `time_from_tone_onset = bin_center - (sample_start - go_time)` for each neural bin. | behavior alignment logic in `process_one_sess`; event alignment patterns in `align_markers.py` | Must be trial-specific because replays can make tone onset earlier than `-1.85 s`. |
| `acquisition/BehavioralEvents/photostim_start_times` / `photostim_stop_times` + go cue | `input[1]` | Binary vector per trial/bin: `1` if bin center is within the photostim interval relative to go cue, else `0`. | stimulation alignment in `process_one_sess` | Event counts are 0 or 1 start/stop pair per trial in sampled sessions. |
| `left_lick_times`, `right_lick_times`, `go_start_times`, `trial stop time` | `output[0]` (`choice`) | Derive actual choice from the first lick in the answer window `[go_start, min(go_start + 1.5 s, trial_stop)]`: left/right; if no lick in the answer window, label `no lick`. Broadcast across all bins in the trial. | `lick_times`, `lick_directions` usage in `process_one_sess` | This follows the paper-defined answer period and avoids inconsistent NWB `go_stop_times` semantics across sessions. |
| `intervals/trials/outcome` | `output[1]` (`outcome`) | Map string labels to categorical ids and broadcast across all bins in the trial. | `correctness` logic in `process_one_sess`; trial masks in `get_regular_trial_mask` | Use explicit three-way task labels required by decoder: `ignore`, `miss`, `hit`. |
| `intervals/trials/early_lick` | `output[2]` (`early_lick`) | Map `no early` / `early` to `0/1` and broadcast across all bins. | `early_lick_trials` in `process_one_sess` / `get_regular_trial_mask` | Keep these trials instead of excluding them, because early lick is a decoder target. |
| `BehavioralTimeSeries/Camera0_side_TongueTracking` (`y`, likelihood) | `output[3]` (`tongue_y_bin`) | Align side-camera tongue tracking to neural bins by taking the last visible frame within each `50 ms` bin. If no visible frame, class `3` (`not visible`). For visible bins, discretize y-position using session-wide 40th and 60th percentiles from all visible tongue frames in that session. | alignment idea from `align_markers.py` | Likelihood is effectively bimodal (near-zero or `1.0`), so a high threshold is reasonable. |
| `general/subject/subject_id` / subject folder name | `subjects`, `subject_idx` | Collect unique subject ids, index sessions accordingly. | N/A | Use strings like `sub-440956` for stability. |
| `units/anno_name` with fallback to probe insertion `location["brain_regions"]` | `brain_regions`, `brain_region_idx` | Use cleaned unit-level histology annotation strings directly; if annotation is missing/`nan`, fall back to insertion target. | `helper_get_neuron_id_area` | This preserves the most specific available unit location in the NWB release. |
| Decoder/task metadata | `metadata` | Record task description, bin size, alignment event, offsets, session ids, conversion rules, and QC/filtering summary. | N/A | Include exact offsets `off_start=-2.5`, `off_end=1.5`, `time_bin_size=50.0`. |

### Key Decisions
1. **Session inclusion**: Keep sessions with at least one `classification == "good"` unit.
Rationale: this reproduces the paper-reported `173` analyzable sessions from `174` raw NWB files by excluding the one all-`nan` QC session.

2. **Neuron QC rule**: Use NWB `units.classification == "good"` as the primary good-unit filter.
Rationale: the NWB field description explicitly says this is the paper’s single-unit classification label; it is the closest equivalent to the reference `goodunits` lists.

3. **Trial curation rule**: Do not exclude early-lick, miss, ignore, or photostim trials globally.
Rationale: these states are required as decoder outputs/inputs. The papers excluded some of them for specific analyses, but the conversion task explicitly needs them represented.

4. **Bad-trial mask usage**: Do not use `is_good_trials` for conversion-wide trial filtering.
Rationale: in this NWB release the field is not consistently aligned to the trial table across sessions, so using it would create arbitrary shape-dependent behavior.

5. **Neural representation**: Store firing rates, not raw counts.
Rationale: the task explicitly asks for `50-ms-width bins for computing firing rates`, and the reference preprocessing also converts spike times to firing rates.

6. **Output dimensionality**: Store all outputs as time-varying `(n_output, T)` arrays, with per-trial labels broadcast across time and tongue y as genuinely time-varying.
Rationale: each trial must have a single output array; mixing 1D and 2D outputs inside one trial is not possible.

7. **Tone-onset input source**: Derive tone onset from event timestamps, not a fixed `-1.85 s`.
Rationale: raw NWB trials can contain replay structure; some trials have sample onset substantially earlier than `-1.85 s`.

8. **Choice source**: Derive choice from the first post-go lick in the paper-defined `1.5 s` answer window rather than only from `outcome + instruction` or the raw NWB `go_stop_times`.
Rationale: it uses the most direct behavioral measurement and avoids a real NWB inconsistency where `go_stop_times` can mark only the short go-cue duration in later sessions.

9. **Tongue visibility threshold**: Use a high DeepLabCut-likelihood threshold (planned: `>= 0.9`) to define visibility.
Rationale: tongue likelihood is strongly bimodal (near zero vs `1.0`), so the exact high threshold has little effect and a strict threshold avoids false visibility.

10. **Tongue percentile thresholds**: Compute 40th/60th percentiles from all visible tongue-y frames in the full session before trial binning.
Rationale: the task says “over the session”; using all visible session frames is the most literal interpretation.

11. **Brain-region labels**: Preserve cleaned unit-level `anno_name` labels, with insertion-target fallback for missing annotations.
Rationale: the NWB release does not provide an authoritative coarse-group mapping equivalent to the paper’s MATLAB QC files, and preserving the actual unit annotation is more defensible than hand-crafted coarse remapping.

12. **Neural-support trial filter**: Keep only trials whose full decoder window lies inside the shared `units/obs_intervals` support of retained good units, and drop any residual trial whose binned neural tensor is entirely zero.
Rationale: some NWB sessions contain more behavioral trials than the stored spike data actually support; without this guard the sample verifier reports unsupported all-zero late trials.

### Planned Sanity Checks
- [ ] Neural binning check: for at least 3 random `(session, trial, neuron)` combinations, compare converted firing-rate bins against direct raw-spike histograms with `np.allclose`.
- [ ] Tone-onset alignment check: for at least 3 trials, verify that the converted `time_from_tone_onset` equals `bin_center - (sample_start_last_before_go - go_time)` from raw events.
- [ ] Photostim alignment check: for at least 3 stim and 3 non-stim trials, verify the binary input vector against raw photostim start/stop timestamps.
- [ ] Choice-label check: for at least 3 hit, 3 miss, and 3 ignore trials, verify the converted choice label against raw lick timestamps in the response window.
- [ ] Tongue discretization check: for at least 3 trials and multiple bins, verify visibility class and percentile bin from raw tongue `y` and likelihood values.
- [ ] Session-count check: confirm converted sessions equal `173` after excluding the all-`nan` QC session.
- [ ] Good-unit count check: confirm converted neuron count is close to `69453` for this NWB release, and document the remaining gap versus the paper’s `69943`.
- [ ] Trial-validity check: verify that every kept trial has a valid go cue and at least one sample-start event before go within the trial interval.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with:
- CLI: `python -u /app/convert_data.py <outpicklefile> [--full|--sample] [--show-processing]`
- NWB session loader using `h5py`
- Good-unit filtering from `units.classification`
- Session exclusion for the one all-`nan` QC session
- Trial filtering from shared good-unit `obs_intervals`, required event checks, and all-zero-neural fallback removal (not `is_good_trials`, which is inconsistent in this NWB release)
- Go-cue-aligned neural firing rates in `50 ms` bins over `[-2.5, 1.5)` s
- Trial-specific tone-onset timing from raw event timestamps
- Time-varying photostim input from raw event timestamps
- Choice/outcome/early-lick/tongue-y outputs, with choice extracted from the paper-defined `1.5 s` answer window
- Session-specific processing plots
- Timing printouts per major processing block

Code inefficiencies identified:
- Full dataset will be large because neural activity is stored trial-by-trial for all sessions.
- Spike binning and tongue alignment are the main expected bottlenecks.

Code speedups added:
- Vectorized spike binning per unit using `np.searchsorted` over all trial bin edges at once.
- Vectorized tongue-frame alignment using `np.searchsorted`.
- Reduced output file size by storing neural data as `float16` and categorical outputs as small integers.
- Reused precomputed bin edges/centers and session-wide event arrays.
- Filtered out observation-unsupported trials before most downstream processing, which avoids wasted work on unusable late-session trials.

Sanity of script implementation:
- `python3 -m py_compile /app/convert_data.py` passed.
- `python3 /app/convert_data.py --help` shows the required interface.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 |
| Neurons / session | 459, 375 (mean `417.0`) |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 527 |
| Trials / session | 368, 159 |
| `time_from_tone_onset_s` range | `[-0.625, 5.722]` |
| `photostim_on` range | `[0.0, 1.0]` |
| `choice` distribution | `[0.452, 0.408, 0.140]` for `[left, right, no lick]` |
| `outcome` distribution | `[0.140, 0.298, 0.562]` for `[ignore, miss, hit]` |
| `early_lick` distribution | `[0.943, 0.057]` for `[no, yes]` |
| `tongue_y_bin` distribution | `[0.053, 0.031, 0.060, 0.856]` for `[lt40, 40to60, gt60, not visible]` |

### Processing Plots Review
- The example rasters and input traces are aligned to the go cue as expected; photostim occupies late-delay bins and never extends past the go cue in the sample sessions.
- Tone-onset timing is centered at the expected `-1.85 s` but includes an earlier replay tail down to about `-4.25 s`, confirming that trial-specific event extraction is necessary.
- Tongue output is dominated by `not visible`, which matches the raw side-camera coverage; visible samples spread cleanly around the session-specific 40th/60th percentile thresholds.
- Initial sample verification exposed one late-session all-zero neural trial. Investigation of raw NWB data showed that this session’s behavioral table outlasts the stored spike support. I fixed this by filtering on shared `obs_intervals` support and dropping the remaining all-zero neural trial. Re-running verification then produced no warnings.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| Shared `obs_intervals` trial prefilter | Reduced one sample session from `480` raw trials to `160` neural-supported trials before expensive downstream processing |
| All-zero neural fallback drop | Removed one unsupported edge trial and eliminated the sample validation warning |
| Vectorized spike and tongue alignment | Kept per-session processing below `0.5 s` in the sample runs |

| Step | Time / Session | Estimated Total Time |
| Sample conversion with plots | `0.30-0.49 s` for the two sample sessions (`1.85 s` wall time total including startup/plot overhead) | N/A |
| Full conversion without plots | estimated from all `173` analyzable sessions using sample-derived seconds per `(good units × kept trials)` and archive-wide session metadata | `~99.5 s` total (`~1.66 min`), comfortably below the `15 min` threshold |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `choice` | `0.7300` | `0.6239` |
| `outcome` | `0.7590` | `0.6549` |
| `early_lick` | `0.8411` | `0.7222` |
| `tongue_y_bin` | `0.6648` | `0.5059` |

Additional notes:
- Training used `421` trials and validation used `106` trials.
- Loss decreased from `14.71` at epoch 1 to `0.59` at epoch 200, with test loss `0.75`.
- Every output exceeded chance on validation:
  - `choice`: `0.6252` vs chance `0.3333`
  - `outcome`: `0.6549` vs chance `0.3333`
  - `early_lick`: `0.7222` vs chance `0.5000`
  - `tongue_y_bin`: `0.5059` vs chance `0.2500`

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: `5,977,410,148` bytes (`~5.98 GB`)
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | `69943` good units | QC-approved units only | `69453` units with `classification == good` in local NWB release | `69453` | Close; archive-version difference already documented |
| Mean neurons/session | `404.29` (derived) | QC-approved units per session | `401.46` | `401.46` | Close |
| Subjects | `28` | `28` mice in session lists | `28` | `28` | Yes |
| Sessions | `173` behavioral sessions | Reference analyses use `173` analyzable sessions | `173` analyzable (`174` raw files minus one all-`nan` QC session) | `173` | Yes |
| Trials (total) | `~82348` analyzed-paper estimate (`173 x 476`) | Code loads full sessions, then applies analysis masks later | `94990` raw trial-table rows; `90844` decoder-supported trials after observation/event filtering | `90844` | Intentional difference: conversion keeps all decoder-relevant supported trials |
| Trials/session (mean) | `476` analyzed-paper mean | Trial masks applied downstream per analysis | `545.92` raw, `525.11` after conversion filters | `525.11` | Intentional difference |
| `time_from_tone_onset_s` range | sample/delay timing implies trial-specific replay can extend earlier than `-1.85 s` | Reference code uses trial-specific task timing | `[-1.5, 11.9]` over kept trials | `[-1.5, 11.9]` | Yes |
| `photostim_on` range | binary stimulation state | Binary aligned stimulation intervals | `[0, 1]` | `[0, 1]` | Yes |
| `choice` distribution | Balanced left/right among responded trials; no-lick should track ignore trials | Direct lick-based choice in answer period | Raw-derived `[0.430, 0.421, 0.149]` for `[left, right, no lick]` | `[0.430, 0.421, 0.149]` | Yes |
| `outcome` distribution | Hit-dominant, paper reports `84%` correct on stricter control-trial subset | Outcome labels used for trial masks | Raw-derived `[0.149, 0.166, 0.685]` for `[ignore, miss, hit]` | `[0.149, 0.166, 0.685]` | Yes |
| `early_lick` distribution | Early licks present but minority | Explicit early-lick mask in code | Raw-derived `[0.885, 0.115]` for `[no, yes]` | `[0.885, 0.115]` | Yes |
| Control-trial correct rate (`no early`, `no photostim`, responded) | `0.84` correct | Reference “regular trial” logic excludes early/no-response/photostim | `0.816` from converted subset using the same exclusions | `0.816` | Close |
| Photostim trial fraction | `~0.25` in photoinhibition subset | Photostim tracked explicitly per trial | `0.200` across all retained trials; `0.207` restricting to sessions with any stim trials | same | Reasonable; aggregate includes non-ogen/ephys-only sessions |

Additional full-validation notes:
- Full verification reported `Data format is valid, no errors or warnings.`
- Spot-checking the problematic later session `sub-480928_ses-20210128T133949...` confirmed that the corrected choice window changed a pathological all-`no lick` distribution to a sensible `[left, right, no lick] = [0.289, 0.252, 0.460]` at the raw-session level.
- The original sample-based runtime estimate (`~99.5 s`) under-predicted the full run because later sessions have much larger neuron-by-trial products than the first two sample sessions. Actual full conversion time was `184.30 s`, still comfortably below the `15 min` threshold.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `/app/verification_full_out.txt` reported `Data format is valid, no errors or warnings.` No additional verifier-side fixes were required after correcting the choice window.
2. **Direct raw-vs-converted sanity checks**: I loaded raw NWB files directly and reconstructed one representative converted trial from each of three sessions:
   - `sub-440956_ses-20190207T120657...`, converted trial `67`, raw trial `67`
   - `sub-440956_ses-20190208T133600...`, converted trial `19`, raw trial `19`
   - `sub-480928_ses-20210128T133949...`, converted trial `82`, raw trial `82`
   For all three cases:
   - `np.allclose(raw_neural, converted_neural) == True`
   - `np.allclose(raw_input, converted_input) == True`
   - `np.allclose(raw_output, converted_output) == True`
3. **Reference code comparison**:
   - **Data loading**: reference `process_one_sess` loads session-level behavior/task/spike arrays; my script loads the NWB equivalents from `intervals/trials`, `acquisition/BehavioralEvents`, `BehavioralTimeSeries`, and `units`.
   - **Neuron filtering**: reference code uses external classifier QC lists; my script uses the NWB-carried equivalent `units.classification == "good"`.
   - **Trial filtering**: reference code applies downstream analysis masks (for example “regular trials”); my script intentionally keeps all decoder-relevant supported trials, only excluding trials that lack valid alignment events or usable neural support.
   - **Temporal alignment**: both pipelines align to go cue.
   - **Binning**: reference code bins spikes after go alignment; my script changes the bin width/window to the decoder specification (`50 ms`, `[-2.5, 1.5)`), which is an intentional task-driven difference.
   - **Inputs/outputs**: my script constructs decoder-specific inputs/outputs from the same underlying task variables, with the main task-driven additions being continuous time-from-tone input and discretized tongue-y output.
4. **Key statistics comparison**:
   - Session count, subject count, and good-unit count matched the local NWB release expectations (`173` sessions, `28` subjects, `69453` good units).
   - Control-trial correct rate computed on the converted data with paper-style exclusions (`no early`, `no photostim`, responded only) was `0.816`, close to the paper’s `0.84`.
   - Full converted choice distribution became self-consistent with outcome labels after the answer-window fix: `[left, right, no lick] = [0.430, 0.421, 0.149]`, with `no lick` closely matching `ignore = 0.149`.
5. **Edge-case review**:
   - Verified and fixed sessions whose behavioral trial table extends beyond neural support using `units/obs_intervals` plus the all-zero-neural fallback drop.
   - Verified and fixed inconsistent `go_stop_times` semantics across sessions by switching choice extraction to the paper-defined `1.5 s` answer window.
   - Confirmed replayed sample/tone timing exists in raw data, so trial-specific sample-start extraction is necessary.
   - Quantified residual label consistency after the choice-window fix:
     - fraction of hit trials labeled `no lick`: `0.00079`
     - fraction of ignore trials labeled as a lick choice: `0.00229`
     These are very small and consistent with rare edge-case behavioral/event mismatches rather than a systematic bug.

### Issues Found and Resolved
- **Behavior trials extending beyond usable spike support**: Fixed by requiring the full decoder window to lie inside the shared `units/obs_intervals` support and dropping any residual all-zero-neural trial.
- **Inconsistent `go_stop_times` semantics**: Fixed by deriving choice from the first lick in `[go_start, min(go_start + 1.5 s, trial_stop)]` instead of trusting the raw `go_stop_times` field.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `choice` | `0.7073` | `0.6795` | `2.04x` chance |
| `outcome` | `0.6924` | `0.6610` | `1.98x` chance |
| `early_lick` | `0.7998` | `0.7562` | `1.51x` chance |
| `tongue_y_bin` | `0.6728` | `0.6130` | `2.45x` chance |

Additional notes:
- Training used `72,621` trials and validation used `18,223` trials.
- Loss decreased from `14.79` at epoch 1 to `0.67` at epoch 200, with test loss `0.66`.
- Training completed successfully on `cuda` without requiring the `--cpu` fallback.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| `choice` | Validation balanced accuracy `0.6795` | Papers show neural choice decoding clearly above chance before go cue and strongest near/after go; no directly comparable scalar balanced-accuracy value was tabulated for this exact decoding setup, but the achieved value is consistent with that expectation. |
| `outcome` | Validation balanced accuracy `0.6610` | No directly comparable decoder accuracy reported in the papers. Outcome should be decodable from the same task/movement signals that dominate the dataset; achieved accuracy is well above chance. |
| `early_lick` | Validation balanced accuracy `0.7562` | No directly comparable decoder accuracy reported in the papers. Early-lick status is behaviorally salient and should be decodable; achieved accuracy is above the `1.5x chance` threshold. |
| `tongue_y_bin` | Validation balanced accuracy `0.6130` | Method paper reports very strong movement-related information in neural activity across the dataset; no exact scalar for this discretized tongue-y target is reported, but the achieved value is strongly above chance and directionally consistent. |

Checks performed:
- **Accuracy vs chance**:
  - `choice`: `0.6795 / 0.3333 = 2.04x chance`
  - `outcome`: `0.6610 / 0.3333 = 1.98x chance`
  - `early_lick`: `0.7562 / 0.5000 = 1.51x chance`
  - `tongue_y_bin`: `0.6130 / 0.2500 = 2.45x chance`
  No output was below chance or below the `1.5x chance` investigation threshold.
- **Accuracy comparison to papers**:
  - The reference papers do not provide directly comparable scalar accuracies for `outcome`, `early_lick`, or this discretized `tongue_y_bin` target.
  - For `choice`, the papers provide only graphical neural-decoding summaries (data paper) and behavioral-video AUCs (method paper), neither of which is directly comparable to this balanced-accuracy neural decoder. The achieved value is nevertheless consistent with the papers’ qualitative conclusion that choice is strongly represented.
- **Train vs validation gap**:
  - `choice`: `0.7073 / 0.6795 = 1.04`
  - `outcome`: `0.6924 / 0.6610 = 1.05`
  - `early_lick`: `0.7998 / 0.7562 = 1.06`
  - `tongue_y_bin`: `0.6728 / 0.6130 = 1.10`
  None approaches the `>1.5x` overfitting concern threshold.

### Issues Found and Resolved
- No new conversion issues were identified in this review step.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Cleanup notes:
- Created `/app/README.md` with dataset summary, conversion rules, format description, and loading example.
- Created `/app/cache/README_CACHE.md` documenting nonessential cached artifacts.
- Moved diagnostic plots and the raw-vs-converted sanity-check artifact into `/app/cache/`.
