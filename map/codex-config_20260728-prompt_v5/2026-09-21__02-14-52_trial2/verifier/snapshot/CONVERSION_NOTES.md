# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement dataset, from the provided paper/code/data bundle in `/app`
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

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadmat` | `code/VideoAnalysisUtils/preprocessing_utils.py` | LOADING | Custom MATLAB loader that recursively converts MATLAB structs/cells into Python dict/list structures; reference loader for raw `.mat` exports. |
| `process_one_sess` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Loads one session across probe files, reads behavior/task fields, reads spike times / unit QC / histology, and concatenates probes. |
| `helper_get_neuron_id_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Selects neurons by hemisphere and QC-approved region membership using session-level good-unit indices and CCF labels. |
| `process_one_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates per-trial spike times to an analysis window, bins spikes into firing rates, and saves per-area session pickle files. |
| `sliding_histogram` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Converts per-trial spike times into binned spike counts or firing rates for evenly spaced bin centers. |
| `load_session` | `code/VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Re-loads preprocessed per-area pickles and concatenates areas within a session into one session-level firing-rate array. |
| `get_regular_trial_mask` | `code/VideoAnalysisUtils/population_decoding_utils.py` | CURATION | Defines “regular trials” as no early lick, no auto water, no free water, no no-response, and no stimulation. |
| `align_markers_between_lims` | `code/Sherlock/align_markers.py` | PROCESSING | Aligns raw tracking marker trajectories to go cue on a 3.4 ms grid over `[-3, 1.5)` s by carrying forward the last frame in each bin. |
| `temporal_alignment_embed_and_ephys` | `code/VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Crops/alignment helper used when bringing video-derived signals onto the ephys time base. |

### Notes
- Repository README identifies this as analysis code for the MAP dataset and points to processed-data / DANDI releases; lightweight methods are in `VideoAnalysisUtils`, cluster scripts in `Sherlock`.
- Neural modality is electrophysiology, not imaging. No `dF/F` computation is involved anywhere in the reference loading path.
- The raw session loader (`process_one_sess`) reads behavior/task fields directly from the `.mat` export:
  - `behavior_auto_learn`
  - `behavior_early_report`
  - `behavior_is_auto_water`
  - `behavior_is_free_water`
  - `behavior_lick_directions`
  - `behavior_lick_times`
  - `behavior_report`
  - `task_cue_time`
  - `task_delay_time`
  - `task_sample_time`
  - `task_stimulation`
  - `task_trial_type`
- Reference code explicitly subtracts per-trial go-cue time from lick times and stimulation on/off times, then stores `gocue_time` separately. This confirms trialwise temporal alignment is centered on go cue in the preprocessed representation.
- Spike times in the raw export are treated as already aligned to go cue: comment in `process_one_area` says “in Susu's data, spike_times are relative to go cue time”.
- Neuron curation in the preprocessing path is not based on recomputing QC metrics from scratch. Instead, the code loads a session-specific `goodunits` file (`qc_mode='classifier'`) and uses those approved neuron indices; it also requires neurons to have both ephys and histology entries.
- Histology / region handling:
  - probes are concatenated within session
  - units are assigned CCF coordinates and labels
  - units are split by hemisphere using ML midpoint `5700`
  - units are grouped into 14 high-level regions (`ALM`, `Medulla`, `Midbrain`, `Striatum`, `Thalamus`, `Pons`, `Cerebellum`, `Hypothalamus`, `Hippocampus`, `Orbital`, `OtherCortex`, `Olfactory`, `CorticalSubplate`, `Pallidum`)
- The main Sherlock preprocessing script uses `bw=0.04` and `stride=0.0034` over `[-3, 3]` s. The standalone `__main__` block in the same file shows an older example with `bw=0.1`, `stride=0.05`, `[-3, 3.5]`; the Sherlock script is more likely the paper path.
- `align_markers.py` builds marker arrays from raw tracking under `tracking['camera_0_side']`, using marker keys:
  - `nose_x`, `nose_y`
  - `tongue_x`, `tongue_y`
  - `jaw_x`, `jaw_y`
  - `whisker_x`, `whisker_y`
- Marker alignment uses `dt=0.0034`, `tt=np.arange(-3, 1.5, dt)`, and per bin copies the last frame with timestamp in `[t-dt, t)`. This is the relevant reference processing for tongue trajectories.
- `get_regular_trial_mask` is important reference curation for decoding analyses, but it is a downstream analysis mask rather than part of raw preprocessing. I will decide in later steps whether the decoder task here should include all valid trials or only these “regular” trials, based on consistency with data and papers.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `/app/data` contains:
  - `dandiset.yaml`
  - 28 subject folders named `sub-<id>/`
  - 174 NWB files total, one NWB file per session
- File naming convention:
  - `sub-<subject>_ses-<YYYYMMDDTHHMMSS>_behavior+ecephys(+ogen).nwb`
  - 168 filenames include `+ogen`; 6 are `behavior+ecephys.nwb` without `+ogen` in the filename
- The NWB organization in a representative file is:
  - `intervals/trials`: per-trial metadata table
  - `units`: per-unit metadata + spike times + QC fields
  - `acquisition/BehavioralEvents`: event timestamps for trial epochs, go cue, licks, and photostim
  - `acquisition/BehavioralTimeSeries`: dense camera tracking streams
  - `general/extracellular_ephys/electrodes`: electrode metadata including JSON-encoded high-level brain region labels
- Trial table columns are consistent across all 174 files:
  - `start_time`, `stop_time`, `trial`, `trial_uid`
  - `task`, `task_protocol`, `trial_instruction`
  - `outcome`, `early_lick`, `auto_water`, `free_water`
  - `photostim_onset`, `photostim_power`, `photostim_duration`
- Behavioral event streams are consistent across all 174 files:
  - `presample_start_times`, `presample_stop_times`
  - `sample_start_times`, `sample_stop_times`
  - `delay_start_times`, `delay_stop_times`
  - `go_start_times`, `go_stop_times`
  - `left_lick_times`, `right_lick_times`
  - `photostim_start_times`, `photostim_stop_times`
  - `trialend_start_times`, `trialend_stop_times`
- Behavioral time series:
  - all 174 files have `Camera0_side_TongueTracking`, `Camera0_side_JawTracking`, and `Camera0_side_NoseTracking`
  - `Camera0_side_TongueTracking` description says it stores `('tongue_x', 'tongue_y', 'tongue_likelihood')`
  - timestamps are explicit and sampled every 3.4 ms in the inspected session
  - some sessions additionally have `Camera0_side_LickPortTracking`, `Camera0_side_WhiskerTracking_whisker`, or `Camera3_side_*`
- Units table:
  - contains spike times as one concatenated per-unit vector over the session
  - contains `obs_intervals`, which in the inspected file were one interval per trial
  - contains QC / metadata fields analogous to the `.mat` reference code: `unit_quality`, `unit_amp`, `unit_snr`, `isi_violation`, `avg_firing_rate`, `drift_metric`, `presence_ratio`, `amplitude_cutoff`, `classification`, `anno_name`, `is_good_trials`, waveform metrics, and electrode references
- High-level electrode region labels present in the NWB files are:
  - `left/right ALM`
  - `left/right BLA`
  - `left/right ECT`
  - `left/right Medulla`
  - `left/right Midbrain`
  - `left/right Striatum`
  - `left/right Thalamus`
- Important mismatch vs reference code:
  - reference code was written for DataJoint-exported `.mat` files with session-level `goodunits` QC files and 14 broader regions
  - provided source data here are already-converted NWB session files, so later steps must map NWB fields back to the reference concepts carefully

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272,227 raw units across all NWB `units` tables |
| Neurons / session | mean 1,564.523 raw units/session; range 493-3,191 |
| Subjects | 28 |
| Sessions / subject | mean 6.214; range 3-10 |
| Trials (total) | 94,990 |
| Trials / session | mean 545.920; range 264-800 |

Additional raw-data observations:
- Trial variables observed over the full dataset:
  - `trial_instruction`: `left`, `right`
  - `outcome`: `hit`, `miss`, `ignore`
  - `early_lick`: `early`, `no early`
  - `auto_water`: mostly `0`, with some `1`
  - `free_water`: mostly `0`, with some `1`
  - photostim trials are present in 18,588 / 94,990 trials (`19.57%`)
- Unit labels observed over the full dataset:
  - `unit_quality`: `good`, `multi`
  - `classification`: mostly `unlabelled`, some `good`, a few `nan`
  - non-empty `anno_name` appears for 71,305 / 272,227 raw units

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units in paper analyses | “Overall, the dataset consisted of 69,943 good units recorded across 173 behavioral sessions” |
| Neurons / session | ~404.3 good units/session on average (inferred from 69,943 / 173) | “69,943 good units recorded across 173 behavioral sessions” |
| Subjects | 28 mice | “This study is based on data from 28 mice” |
| Sessions / subject | 173 total analysis sessions; ~6.18 per mouse on average (inferred) | “69,943 good units recorded across 173 behavioral sessions” |
| Trials (total) | Not stated directly for the curated paper subset | No direct total found in text |
| Trials / session | 476 mean; range 130-785 | “Mice performed on average 476 (Mean; range, 130-785) trials per session” |
| Neural data time bin | 40 ms bin width | “we took … 40 ms bin width for estimating spike rates” |
| Behavior data time bin | 3.4 ms video frames / stride | “High-speed videos … were acquired at 300 Hz”; method paper code/text uses 3.4 ms frame spacing and 3.4 ms neural stride |
| Reward rate | 84% correct control-trial rate; range 65-99% | “476 … trials per session with 84% correct rate (range, 65-99%)” |
| Photostimulation trial fraction | Typically ~25% of trials | “In a subset of randomly selected trials (typically 25%), one of the ALM hemispheres was transiently inactivated” |
| Photoinhibition duration/timing | Late delay, final 0.5 s, ending before Go cue | “We silenced ALM activity during the late delay epoch (last 0.5 s) … Thus, photoinhibition always ended before the ‘Go’ cue.” |
| Good-unit yield fraction | 25.9% of Kilosort2 clusters | “This corresponds to 25.9 % of clusters reported by Kilosort2.” |
| Video-analysis session count | 105-106 sessions depending analysis | method paper text reports `n = 106 sessions` for choice-from-video AUC and `n = 105 sessions` / `n = 106 sessions` in explained-variance comparisons |


### Processing Details
- Task timing from datapaper / methods:
  - sample epoch: three 150 ms tones with 100 ms inter-tone intervals
  - delay epoch: 1.2 s
  - go cue: 0.1 s auditory cue
  - response epoch: 1.5 s
  - consumption period: 1.5 s
- Paper analyses are consistently go-cue referenced for neural selectivity and decoding summaries; example windows in datapaper:
  - stimulus selectivity: `[-1.85, -1.2]` s
  - choice selectivity: late delay, last `0.6 s` before go cue
  - action selectivity: first `1.5 s` after go cue
  - outcome selectivity: `2-3 s` after go cue, with lingering-lick trials removed
- The recent method paper states explicitly:
  - spikes were binned into firing rates with `40 ms` bin width and `3.4 ms` stride
  - side-view video only was used
  - behavioral markers were tracked from video and cleaned by 5-sigma velocity-based outlier rejection
  - when the tongue was occluded in the mouth, tongue position was imputed to its mean value
- The datapaper STAR Methods state high-speed video was acquired at `300 Hz` and DeepLabCut tracked tongue, jaw, and nose. The methodpaper text says jaw, paws, and tongue. The reference code uses jaw, nose, tongue, and whisker. This discrepancy will need resolution in Step 4 against the actual data.
- Population choice decoder in datapaper:
  - logistic regression
  - pseudo-populations
  - causal sliding windows of `200 ms`
  - `10 ms` step size
  - nested 5-fold CV for evaluation and inner 4-fold CV for regularization
- Video-based choice decoder in methodpaper:
  - pre-sample: AUC `0.51 ± 0.06` s.d. across 106 sessions
  - sample/delay mean: AUC `0.66 ± 0.12` s.d. across 106 sessions
  - second half of response epoch: AUC `0.99 ± 0.01` s.d. across 106 sessions
- Movement-to-neural prediction in methodpaper:
  - medulla had strongest explained variance
  - embedding-based prediction outperformed marker-based prediction
  - example insertion-averaged embedding `R2` values (Extended Data Fig. 3):
    - sample: medulla `0.0833 ± 0.0066`, ALM `0.0629 ± 0.0047`, thalamus `0.0364 ± 0.0027`
    - delay: medulla `0.0927 ± 0.0065`, ALM `0.0654 ± 0.0069`, thalamus `0.0347 ± 0.0028`
    - response: medulla `0.1764 ± 0.0061`, ALM `0.0923 ± 0.0037`, thalamus `0.0611 ± 0.0026`

### Curation Steps

**Neuron curation rules**:
- Datapaper / methods:
  - Kilosort2 output used for analyses
  - 15 QC metrics computed per cluster
  - five region-specific logistic-regression classifiers trained from manually curated labels
  - cortex classifier applied to ALM / other cortex / hippocampus / olfactory / cortical subplate
  - striatum classifier applied to striatum / pallidum
  - thalamus classifier applied to thalamus / hypothalamus
  - midbrain classifier applied to midbrain / pons
  - medulla classifier applied to medulla / cerebellum
- QC white paper adds:
  - recordings with substantial drift were rejected before classifier application
  - classifiers were trained on 28 manually curated penetrations
  - 10-fold CV AUCs were > 0.9
- Reported false alarm rates on held-out manually labeled units:
  - cortex 7.8%
  - striatum 6.4%
  - thalamus 7.3%
  - midbrain 5.5%
  - medulla 4.3%

**Trial curation rules**:
- Datapaper / methods:
  - early lick trials excluded for analysis
  - no-response (`ignore`) trials excluded for many paper analyses
  - overall behavioral performance computed on control (no-photostim) trials excluding early licks
  - sessions included for analysis required `>65%` performance and at least `50` correct lick-left and `50` correct lick-right trials
  - for photoinhibition effect analyses, units needed at least `10` photostimulation trials
- Methodpaper:
  - video-based analyses excluded free-water, early-lick, and ignore trials
  - sessions with video artifacts were screened out
  - sessions with fewer than 20 trials in any of the 4 decoder-defined groups were excluded in some choice-vs-movement analyses

### Decoders Trained
| Decoded variable | Accuracy |
| Choice from video before sample epoch | AUC `0.51 ± 0.06` s.d., `n = 106` sessions |
| Choice from video during sample/delay | AUC `0.66 ± 0.12` s.d., `n = 106` sessions |
| Choice from video during response epoch | AUC `0.99 ± 0.01` s.d., `n = 106` sessions |
| Choice from neural population | Logistic-regression decoder clearly above chance in figures; exact numerical values mostly figure-based rather than stated directly in text |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Source file format | Reference code loads DataJoint-exported `.mat` files via `loadmat` | Provided raw source is 174 NWB files | Papers discuss the same dataset, but not the exact file format used here | Use NWB as the raw source, but reproduce the logical processing from the `.mat` pipeline: go-cue-centered alignment, QC-based unit selection, behavioral/tracking reconstruction from raw events and time series. |
| Session count | Preprocessing / analyses refer to a curated session set; datapaper says 173 behavioral sessions | Raw NWB bundle has 174 sessions | Datapaper / QC paper: “69,943 good units recorded across 173 behavioral sessions” | One NWB session, `sub-440958_ses-20190216T162508_behavior+ecephys+ogen.nwb`, has 1,852 units but zero `classification == "good"` units and all classifier labels are `nan`. Excluding this session recovers the paper’s 173-session count. |
| Unit count / QC field | Reference preprocessing uses session-level `goodunits` classifier outputs | NWB `units` table contains all clusters, with fields `unit_quality` and `classification` | Papers describe logistic-regression classifiers used to define “good” units | `unit_quality == "good"` is too permissive (154,948 units). `classification == "good"` yields 69,453 units, close to the paper’s 69,943. Therefore `classification` is the correct NWB analog of the paper’s good-unit list. |
| Remaining good-unit count gap | Paper total is 69,943 | NWB `classification == "good"` total is 69,453 | QC white paper / datapaper quote 69,943 | Small residual gap of 490 units (~0.7%) remains. Most likely explanation is a release/version difference or small conversion discrepancy between the DANDI NWB export and the exact paper-analysis snapshot. Use `classification == "good"` as the closest defensible match and document the small residual mismatch. |
| Trial-count statistics | Datapaper reports 476 trials/session mean, 84% correct | Raw NWB mean is 545.9 trials/session including early / ignore / all sessions; mean non-early is 483.4 over 173 kept sessions | Datapaper stats are for analysis sessions / trials | The raw NWB contains all recorded trials, whereas paper statistics are for curated analyses. The no-early mean (483.4) is close to the reported 476; the difference is consistent with additional analysis exclusions. |
| Behavioral session selection | Datapaper says sessions required >65% performance and at least 50 correct left and 50 correct right trials | Applying a strict control-trial criterion to NWB does not recover 173 sessions; depending on denominator it yields 106 or 152 sessions | Datapaper says 173 sessions overall; methodpaper often uses 105-106 sessions for video analyses | Interpret 173 as the paper-wide curated ephys/QC session set (sessions with classifier-approved good units), while 105-106 reflects a narrower video-analysis subset after additional screening (for example, video artifacts / analysis-specific trial requirements). |
| Choice variable availability | Reference `.mat` data expose `lick_directions` per trial | NWB trials table has `trial_instruction` and session-level left/right lick events, but no explicit per-trial choice column | Papers define choice as lick direction | Reconstruct per-trial choice from the first left/right lick after `go_start` within the 1.5 s response window. In checks across sample sessions this matched hits perfectly, matched misses as opposite-to-instruction essentially perfectly, and mapped ignore trials to no-lick. |
| Tone/sample timing availability | Reference `.mat` stores `task_sample_time`; code uses task-aligned fields | NWB has `sample_start_times`, `sample_stop_times`, `delay_start_times`, `delay_stop_times`, `go_start_times`; sample/delay event counts can exceed trial count | Papers state early licks trigger replay of sample/delay epochs | The extra sample/delay events are consistent with replayed epochs after early licks. For each trial, the definitive tone/sample onset should be the **last** `sample_start` before that trial’s go cue, not simply the first event in the trial. |
| Marker variables | Datapaper text says tongue/jaw/nose; methodpaper says jaw/paws/tongue; code aligns nose/tongue/jaw/whisker markers | NWB consistently contains `Camera0_side_TongueTracking`, `JawTracking`, `NoseTracking`; some sessions also have whisker / lickport / Camera3 streams | Text and code are not fully identical | For the required output, only tongue y-position matters. Use the NWB tongue tracking stream directly and align it in the same go-cue-centered manner as the reference marker code. |
| Video frame timing | Methodpaper says 300 Hz; code uses `dt=0.0034` s | NWB tracking timestamps increment by 0.0034 s in inspected files | Same dataset and codebase | Treat 3.4 ms as the operative discrete time step used by the reference analyses. It is consistent with the code and the stored NWB timestamps. |
| Neural binning | Sherlock preprocessing uses 40 ms bin width, 3.4 ms stride | Raw NWB stores spike times, not firing rates | Methodpaper explicitly states 40 ms bins with 3.4 ms stride | For consistency checks, match reference-style 40 ms / 3.4 ms processing when comparing to paper/code. For the final converted decoder dataset, adapt to the user-required 50 ms bins while preserving the same underlying alignment and QC choices where possible. |

Final consistency understanding:
- Paper-wide ephys dataset:
  - 173 sessions = NWB sessions with at least one `classification == "good"` unit
  - “good units” in NWB are best represented by `classification == "good"`
- Methodpaper video/movement analyses:
  - use a narrower session subset (~105-106 sessions)
  - use side-view video and dense 3.4 ms alignment
  - screen sessions for video artifacts and apply additional trial filters depending on the analysis
- Behavioral variable reconstruction from NWB is feasible and consistent with the papers:
  - `choice` from first post-go lick
  - `outcome`, `early_lick`, `auto_water`, `free_water`, `photostim_*` from trial table
  - final pre-go tone onset from the last `sample_start` before each trial’s go cue

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| NWB `units` rows with `classification == "good"` and session-level `spike_times` | `neural` | Keep only classifier-approved good units; bin absolute spike times into go-cue-centered firing rates over `[go-2.5, go+1.5)` using non-overlapping 50 ms bins; divide counts by 0.05 s to get firing rates | `preprocessing_DJ_2022Aug.process_one_area`, `sliding_histogram` | Final converted trial arrays will have shape `(n_good_units_session, 80)` |
| Trial `go_start_times` | alignment / metadata | Define trial-relative time zero | `process_one_sess` (go-cue-centered logic), marker alignment code | Alignment event is go cue onset |
| Last `sample_start_times` before each trial’s go cue | `input[0]` | Compute signed time-from-tone-onset at each 50 ms bin center: `t_rel - sample_onset_rel` | `align_markers.py` style event alignment; datapaper timing text | Use the **last** sample start before go to handle replayed epochs after early licks |
| `photostim_start_times` / `photostim_stop_times` (fallback: trial table `photostim_onset` + `photostim_duration`) | `input[1]` | Binary 0/1 at each bin center: 1 when photostimulation is active | `process_one_sess` subtracts go cue from stimulation times | Kept as time-varying because decoder input explicitly requires it |
| Trial instruction + post-go lick events (`left_lick_times`, `right_lick_times`) | `output[0]` choice | Derive first lick after go cue in the 1.5 s response window: left / right / no lick; broadcast across all 80 bins | Reference `.mat` used `lick_directions`; NWB reconstruction validated in Step 4 | Matches hit/miss/ignore labels nearly perfectly in checks |
| Trial table `outcome` | `output[1]` outcome | Map `ignore->0`, `miss->1`, `hit->2`; broadcast across all 80 bins | Behavioral fields in `.mat` / papers | Preserve ignore trials because decoder output explicitly includes them |
| Trial table `early_lick` | `output[2]` early lick | Map `no early->0`, `early->1`; broadcast across all 80 bins | Behavioral fields in `.mat` / papers | Keep early-lick trials because decoder output explicitly includes them |
| `Camera0_side_TongueTracking` y coordinate and likelihood | `output[3]` tongue_y | For each 50 ms bin, take the last frame in the bin (reference-style sample-and-hold). If likelihood < threshold or frame flagged as tracking outlier, assign class 3 (not visible). Otherwise discretize visible y by session-specific 40th/60th percentiles into 0/1/2 | `align_markers.py` (last frame per bin), methodpaper marker-cleaning / tongue occlusion notes | Percentiles computed over visible, cleaned tongue-y samples within session |
| NWB subject folder name `sub-xxxxx` | `subjects`, `subject_idx` | Preserve as subject IDs | N/A | One subject index per kept session |
| Unit anatomical information (`anno_name` with fallback to electrode `brain_regions`) | `brain_regions`, `brain_region_idx` | Use a per-unit region label; prefer detailed histology label when present, else fallback to insertion target region label | Reference code uses histology-derived region grouping | Final implementation will use a stable, session-independent region vocabulary |

### Key Decisions
1. **Use 173 sessions, not all 174 raw NWBs**: Exclude the single session with zero classifier-approved good units (`classification` all `nan`) to match the paper-wide curated ephys dataset.
2. **Use `classification == "good"` as the unit filter**: This is the closest NWB equivalent to the paper’s logistic-regression good-unit list and reproduces the paper’s order of magnitude (69,453 versus reported 69,943).
3. **Ignore `unit_quality` for primary curation**: `unit_quality == "good"` retains far too many units (154,948) and does not match the paper’s QC definition.
4. **Do not use `is_good_trials` as a primary filter**: It is almost always true for classifier-good units in sampled sessions, and the reference code did not use a per-unit per-trial mask in preprocessing. I may still use it for diagnostics.
5. **Keep photostimulation trials**: Decoder inputs explicitly require photostimulation state, so these trials must remain.
6. **Keep early-lick trials and ignore trials**: Decoder outputs explicitly require `early lick` and `outcome` including `ignore`, so excluding them would remove requested labels.
7. **Exclude auto-water and free-water trials**: These are atypical task contingencies that the reference analyses usually exclude, and they are not represented in the requested decoder inputs/outputs. Keeping them would add label noise the decoder cannot account for.
8. **Use all outputs as time-varying `(doutput, T)` arrays**: Choice, outcome, and early-lick labels will be broadcast across the 80 bins so they can coexist with time-varying tongue-y output in a single consistent tensor shape.
9. **Use visible-frame percentiles for tongue discretization**: The task definition includes a dedicated “not visible” class, so percentiles should be computed from visible tongue samples only, not from imputed/occluded values.
10. **Use a conservative tongue visibility rule**: Preliminary likelihood distributions are strongly bimodal (mostly near 0 or 1), so a high threshold such as 0.9 is appropriate and robust.
11. **Reproduce marker alignment in spirit, then downsample**: Reference marker code aligns with sample-and-hold from the last frame in each interval. I will keep that rule when collapsing 3.4 ms tracking to 50 ms bins.
12. **Respect reference timing, then adapt binning/window only where required**: The user-requested output needs 50 ms bins and `[-2.5, 1.5)` s, but the underlying event alignment, replay handling, and QC should otherwise mirror the reference processing.

Planned session / trial / unit inclusion:
- Sessions:
  - include sessions with at least one `classification == "good"` unit
- Units:
  - include units with `classification == "good"`
- Trials:
  - require membership in the session’s `units.obs_intervals` coverage subset
  - require valid `go_start`
  - require at least one `sample_start` before `go_start`
  - require tongue tracking coverage over the decoder window
  - exclude `auto_water == 1`
  - exclude `free_water == 1`
  - retain photostim / early / ignore trials
  - drop any residual trial whose binned neural matrix is all zero, because this indicates missing ephys rather than a valid decoder input

Planned tensor shapes:
- `neural[session][trial]`: `(n_neurons_session, 80)` float32
- `input[session][trial]`: `(2, 80)` float32
- `output[session][trial]`: `(4, 80)` int64

Planned categorical mappings:
- choice:
  - `0 = left`
  - `1 = right`
  - `2 = no lick`
- outcome:
  - `0 = ignore`
  - `1 = miss`
  - `2 = hit`
- early lick:
  - `0 = no`
  - `1 = yes`
- tongue y:
  - `0 = <40th percentile`
  - `1 = 40th–60th percentile`
  - `2 = >60th percentile`
  - `3 = not visible`

### Planned Sanity Checks
- [ ] Confirm that session filtering yields 173 sessions and one excluded no-good-unit session.
- [ ] Confirm that unit filtering by `classification == "good"` yields ~69.5k units, close to the paper’s 69,943.
- [ ] Manually verify on sampled trials that first post-go lick reconstruction matches hit/miss/ignore labels.
- [ ] Manually verify on sampled trials that the final `sample_start` before go cue is at `-1.85 s` in ordinary trials and shifts earlier only in replayed early-lick trials.
- [ ] Spot-check raw spike-time histograms against converted 50 ms firing-rate bins with `np.allclose()`.
- [ ] Spot-check photostim binary input against raw event times, including that stimulation ends before go cue.
- [ ] Spot-check tongue-y class labels against raw tracking traces and likelihood, including pre-go invisibility and post-go visibility.
- [ ] Verify every kept session has at least 2 trials after filtering and non-zero kept neurons.

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- Created `/app/convert_data.py` with CLI:
  - `python -u /app/convert_data.py <outpicklefile>`
  - `--full` to process all curated sessions
  - `--sample` to process the first 2 curated sessions
  - `--show-processing` to save `processing_<session_id>.png` for up to 2 sessions
- Implemented session discovery directly from `/app/data` NWB files and curated-session selection as “has at least one `units.classification == good` unit”.
- Implemented raw NWB loading with `h5py` only, so the conversion stays close to the provided source data and avoids extra dependency / serialization layers.
- Implemented per-trial go-cue-centered extraction over `[-2.5, 1.5)` s with exactly 80 non-overlapping 50 ms bins.
- Implemented spike binning as firing rates (`spike count / 0.05 s`) from absolute session spike times using `np.searchsorted` against absolute trial bin edges.
- Implemented choice reconstruction from the first post-go left/right lick within the 1.5 s response window.
- Implemented tone onset reconstruction as the last `sample_start` before each go cue.
- Implemented photostimulation input from trial-table onset/duration fields as a time-varying binary input at 50 ms bin centers.
- Implemented tongue output from `Camera0_side_TongueTracking` using:
  - likelihood thresholding
  - 5-sigma velocity outlier rejection
  - session-specific 40th / 60th percentile thresholds computed on visible cleaned samples only
  - last-frame-in-bin sample-and-hold to match the reference marker-alignment logic
- Implemented dataset assembly into the required dictionary fields plus metadata documenting alignment, bin size, session/trial filters, and output semantics.
- Confirmed the script compiles with `python3 -m py_compile /app/convert_data.py`.
- Confirmed the script runs successfully in sample mode and writes `/app/sample_data.pkl` plus processing plots.
- After sample verification exposed all-zero neural trials, updated the script to:
  - keep only trials present in the raw NWB `units.obs_intervals` coverage subset
  - drop any remaining all-zero-neural trials after binning

Code inefficiencies identified:
- The main cost is repeated per-trial spike binning for every good unit; this scales with `n_trials * n_units`.
- Tongue discretization currently loops over bins per trial; cost is modest relative to spike binning.
- Session discovery opens every NWB file once to inspect `classification`, which is acceptable but still non-zero startup overhead.

Code speedups added:
- Pre-sliced each good unit’s spike train once per session before looping over kept trials.
- Used `np.searchsorted` on sorted spike times / frame timestamps rather than scanning samples in Python loops.
- Kept all trial tensors in `float32` / `int64` arrays and avoided redundant file I/O during processing.
- Limited processing plots to at most 2 sessions even when `--show-processing` is enabled.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 |
| Neurons / session | 459, 375 |
| Subjects | 1 |
| Sessions / subject | `sub-440956`: 2 |
| Trials (total) | 513 |
| Trials / session | 354, 159 |
| `time_from_tone_onset_s` range | `[-0.625, 5.722]` |
| `photostimulation_on` range | `[0.0, 1.0]` |
| `choice` distribution | `[0.456, 0.407, 0.136]` for `[left, right, no lick]` |
| `outcome` distribution | `[0.136, 0.298, 0.565]` for `[ignore, miss, hit]` |
| `early_lick` distribution | `[0.942, 0.058]` for `[no, yes]` |
| `tongue_y` distribution | `[0.054, 0.031, 0.061, 0.855]` for `[<40th, 40-60th, >60th, not visible]` |

### Processing Plots Review
- Reviewed `/app/processing_sub-440956_ses-20190207T120657.png`:
  - final sample onset histogram is sharply concentrated at `-1.85 s`, with a very small tail of earlier replayed sample epochs
  - example tongue trace is mostly not visible, and the binned tongue output is correspondingly class `3` throughout the example trial
  - neural heatmap shows non-zero firing throughout the decoder window with no obvious alignment discontinuity
- Reviewed `/app/processing_sub-440956_ses-20190208T133600.png`:
  - final sample onset histogram is again concentrated at `-1.85 s`
  - tongue becomes visible primarily after go cue in the example trial, and the binned tongue classes switch between visible bins (`0/1/2`) and not-visible bins (`3`) in a way consistent with the raw tracking points
  - no obvious temporal offset between go-cue alignment and the onset of response-period tongue movement
- Initial sample verification exposed a block of all-zero neural trials in session 2. Investigation showed these were trials outside the session’s usable ephys coverage despite being present in the behavioral table. I fixed this by filtering to `units.obs_intervals` and dropping residual all-zero-neural trials, after which verification passed cleanly.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Pre-sliced spike trains once per unit per session | Avoided repeated ragged indexing inside the trial loop |
| Trial prefiltering via `obs_intervals` coverage | Prevented binning known-invalid trials |
| `np.searchsorted` spike binning | Kept per-trial spike counting vectorized |
| Full conversion will run without `--show-processing` | Avoids per-session plotting overhead |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion with plots | `2.31 s/session` from `4.62 s / 2 sessions` | Upper bound `~12.7 min` by scaling with total `good_units * recorded_trials` workload |
| Expected full conversion without plots | slightly lower than sample timing | Expected below `~12 min`, therefore no additional optimization required before Step 9 |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| choice | 0.7407 | 0.6192 |
| outcome | 0.7619 | 0.6395 |
| early_lick | 0.8336 | 0.7191 |
| tongue_y | 0.6705 | 0.5105 |

Additional notes:
- Train/test split used by `train_decoder.py`: 410 training trials, 103 validation trials.
- Loss decreased monotonically in broad trend from `13.8635` at epoch 1 to `0.5769` at epoch 200.
- All validation accuracies were above chance:
  - choice: `0.6192` vs chance `0.3333`
  - outcome: `0.6395` vs chance `0.3333`
  - early_lick: `0.7191` vs chance `0.5000`
  - tongue_y: `0.5105` vs chance `0.2500`

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: `11G`
- `verification_full_out.txt`: created
- `conversion_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | `69,943` good units reported | Good-unit classifier output used throughout preprocessing | `69,453` NWB `classification == good` units | `69,453` | Close (`-490`, about `-0.7%`) |
| Mean neurons/session | `~404.3` inferred from paper totals | Same curated-session concept | `401.46` | `401.46` | Close |
| Subjects | `28` | Same dataset | `28` | `28` | Yes |
| Sessions | `173` behavioral sessions | Same curated-session concept | `173` curated NWB sessions | `173` | Yes |
| Trials (total) | Not stated directly | Analysis-dependent masks downstream | `88,779` after explicit conversion filters | `88,779` | Yes |
| Trials/session (mean) | `476` average behavioral trials/session | Depends on downstream trial mask | `513.17` kept trials/session | `513.17` | Different by design; converted data retains early / ignore trials because decoder outputs require them |
| Control no-early hit rate | `0.84` correct rate | Same control-trial criterion used in papers | `0.8147` on converted trial labels with `early=0`, `photostim=0`, response trials only | `0.8147` | Close |
| Photostim trial fraction | “typically 25%” | Stimulation encoded trialwise | `0.2001` | `0.2001` | Same order; slightly below typical text description |
| `time_from_tone_onset_s` range | Final sample onset expected near `-1.85 s`, with replay tail on early-lick trials | Code aligns sample/tone events relative to go cue | `[-1.5, 11.9]` after conversion construction | `[-1.5, 11.9]` | Yes |
| `photostimulation_on` range | Binary on/off | Binary on/off after event alignment | `[0.0, 1.0]` | `[0.0, 1.0]` | Yes |
| `choice` distribution | Not given numerically | Left / right / no-response is the paper behavior structure | `[0.430, 0.421, 0.149]` for `[left, right, no lick]` | `[0.430, 0.421, 0.149]` | Yes |
| `outcome` distribution | Not given numerically; control performance summarized instead | `ignore / miss / hit` structure used throughout | `[0.149, 0.167, 0.683]` for `[ignore, miss, hit]` | `[0.149, 0.167, 0.683]` | Yes |
| `early_lick` distribution | Early licks are present and excluded in many paper analyses | Same field exists in reference behavior exports | `[0.884, 0.116]` for `[no, yes]` | `[0.884, 0.116]` | Yes |
| `tongue_y` distribution | Not stated numerically; tongue mostly occluded outside movement epochs | Methodpaper expects frequent tongue occlusion / invisibility | `[0.063, 0.032, 0.066, 0.839]` for `[<40th, 40-60th, >60th, not visible]` | `[0.063, 0.032, 0.066, 0.839]` | Yes |

Additional full-dataset checks:
- Full conversion runtime was `298.40 s` for all 173 sessions, well below the conservative `~12 min` upper-bound estimate from Step 7.
- No format errors or warnings were reported by `/app/verification_full_out.txt`.
- Good-unit preservation:
  - converted dataset contains exactly the `69,453` classifier-good units from the curated NWB sessions; no extra unit loss occurred after curation
- Trial-loss breakdown from raw NWB to converted dataset:
  - raw trials across all 174 NWB files: `94,990`
  - trials in the excluded no-good-unit session: `620`
  - raw trials in curated sessions: `94,370`
  - removed because trial not covered by `units.obs_intervals`: `1,060`
  - removed because `auto_water == 1` or `free_water == 1`: `3,764`
  - removed because the full `[-2.5, 1.5)` decoder window was not covered by tongue timestamps: `765`
  - removed because the binned neural matrix was all zero after all other filters: `2`
  - final converted trials: `88,779`
- Spot checks of session integrity:
  - `sub-440956_ses-20190208T133600`: raw `obs_intervals` matched the first `160` behavioral trials exactly; converted session kept `159` after dropping one residual all-zero-neural trial
  - `sub-455219_ses-20190807T134913`: raw `obs_intervals` matched a contiguous behavioral-trial block beginning at trial index `125`; converted session kept `505` trials from that covered block after explicit trial filters
  - minimum converted trial count per session is `7`, so every kept session still exceeds the “at least 2 trials” decoder requirement

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**:
   - Re-read `/app/verification_full_out.txt`.
   - Result: `Data format is valid, no errors or warnings.`
   - Interpretation: after the earlier `obs_intervals` / all-zero-neural fix, no remaining verifier issues were present.
2. **Raw-vs-converted sanity checks using `np.allclose()`**:
   - Neural check:
     - session `sub-440956_ses-20190207T120657`
     - converted trial `10`
     - corresponding raw behavioral trial `10`
     - compared full neural matrix `(459, 80)` recomputed directly from NWB spike times against converted data
     - result: `np.allclose == True`, max absolute difference `0.0`
   - Input check:
     - session `sub-440956_ses-20190207T120657`
     - converted trial `64`
     - corresponding raw behavioral trial `67` after trial filtering
     - compared full input matrix `(2, 80)` recomputed directly from NWB events against converted data
     - result: `np.allclose == True`, max absolute difference `0.0`
     - importance: confirms the converted trial index is correctly offset relative to raw trial index after filtering, including photostimulation timing
   - Output check:
     - session `sub-440956_ses-20190207T120657`
     - converted trial `1`
     - corresponding raw behavioral trial `1`
     - compared full output matrix `(4, 80)` recomputed directly from NWB outcome / early-lick / lick events / tongue tracking against converted data
     - result: `np.allclose == True`, max absolute difference `0.0`
   - Conclusion: neural, input, and output tensors all reproduce exactly from the original NWB source under independent reimplementation.
3. **Reference code comparison by processing stage**:
   - `(a) data loading`
     - reference: `process_one_sess` reads behavior, task, stimulation, spike, QC, and histology from `.mat` exports
     - conversion: loads the NWB analogs of the same data streams (`intervals/trials`, `BehavioralEvents`, `BehavioralTimeSeries`, `units`, `electrodes`)
     - assessment: logically matched, with file-format adaptation only
   - `(b) neuron / trial filtering`
     - reference: classifier-approved good units only; downstream “regular trial” masks often exclude early / ignore / stimulation / auto-water / free-water
     - conversion: keeps only `classification == good` units; excludes `auto_water` and `free_water`; keeps early / ignore / photostim because decoder outputs / inputs explicitly require them; additionally restricts to `obs_intervals`-covered trials and valid tongue-tracking windows because the decoder uses both neural and tongue streams
     - assessment: matched where applicable; divergences are required by task specification or by raw NWB multimodal coverage
   - `(c) temporal alignment`
     - reference: go-cue-centered alignment for licks, stimulation, neural data, and markers
     - conversion: all trial tensors are aligned to `go_start_times`
     - assessment: matched
   - `(d) binning`
     - reference: 40 ms width with 3.4 ms stride for paper analyses
     - conversion: 50 ms non-overlapping bins over `[-2.5, 1.5)` s
     - assessment: intentional task-required deviation; underlying spike counting / event alignment logic preserved
   - `(e) input construction`
     - reference: sample/tone and stimulation timing represented relative to go cue
     - conversion: `time_from_tone_onset_s` comes from the last `sample_start` before go; `photostimulation_on` is reconstructed as a binary per-bin input from trial-table stimulation timing
     - assessment: matched in source variables and alignment, adapted to required target format
   - `(f) output construction`
     - reference: choice from licks, outcome / correctness / early behavior fields from behavior tables, tongue markers from side-view tracking with 5-sigma outlier cleaning and aligned markers
     - conversion: choice reconstructed from first post-go lick, outcome / early from trial table, tongue from side-view tongue tracking with the same style of visibility cleaning and last-frame-per-bin alignment, then discretized into the required 4-class output
     - assessment: matched in raw source streams and timing; discretization is a task-required adaptation
   - Additional metadata difference:
     - conversion keeps fine-grained `anno_name` histology labels as `brain_regions` rather than collapsing immediately to the paper’s coarser analysis families
     - rationale: target format only requires stable per-neuron region labels, and the decoder does not consume this field; preserving finer labels retains information rather than discarding it
4. **Key statistics comparison**:
   - Exact matches:
     - subjects: `28`
     - curated sessions: `173`
     - converted units equal raw `classification == good` units: `69,453`
   - Close matches:
     - paper good-unit total `69,943` versus converted `69,453` (`-490`, about `-0.7%`)
     - paper mean units/session `~404.3` versus converted `401.46`
     - paper control performance `84%` versus converted control/no-early/no-stim response-trial hit rate `81.47%`
   - Trial accounting:
     - raw curated-session trials `94,370`
     - minus `1,060` not covered by ephys `obs_intervals`
     - minus `3,764` auto/free-water trials
     - minus `765` lacking full tongue-window coverage
     - minus `2` residual all-zero-neural trials
     - equals `88,779` converted trials
   - Interpretation: no unexplained trial or unit loss remains.
5. **Edge-case checks**:
   - Partial-coverage session `sub-455219_ses-20190807T134913`:
     - `obs_intervals` map exactly to raw behavioral trials `125` through `629`
     - kept-trial count is `505`
     - confirms no off-by-one error when the recording block begins mid-session
   - Low-trial session `sub-484676_ses-20210413T112028`:
     - all `550` behavioral trials are covered by ephys
     - only `7` survive because `515` fail the tongue-window coverage requirement and `28` are auto/free-water
     - confirms the low converted trial count is due to missing video coverage, not a conversion bug
   - Early-replay handling:
     - sample-onset histograms remain concentrated at `-1.85 s` with only a sparse earlier tail, consistent with selecting the last `sample_start` before go
   - Session exclusion:
     - the single raw session with zero classifier-good units remains excluded, which is required to match the paper’s 173-session curated ephys dataset

### Issues Found and Resolved
- All-zero neural trials in partially covered sessions:
  - Cause: some behavioral trials were present in the NWB trial table even though the ephys recording block did not cover them, and one additional trial survived initial filters despite having no spikes anywhere in the decoder window
  - Resolution: restricted trials to the raw `units.obs_intervals` coverage subset and dropped any residual all-zero-neural trial after binning
  - Re-check result: both sample and full verification logs are clean
- No additional unresolved issues were found in this review pass.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| choice | 0.7206 | 0.6856 | Well above chance `0.3333` |
| outcome | 0.6956 | 0.6582 | Well above chance `0.3333` |
| early_lick | 0.7983 | 0.7506 | Above chance `0.5000`; just over the `1.5x chance` threshold |
| tongue_y | 0.6651 | 0.6133 | Strongly above chance `0.2500` |

Additional notes:
- Train/test split used by `train_decoder.py`: `70,958` training trials, `17,821` validation trials.
- Device: `cuda`.
- Final loss trajectory:
  - epoch 1: `18.0851`
  - epoch 50: `1.6727`
  - epoch 100: `0.8216`
  - epoch 150: `0.7003`
  - epoch 200: `0.6607`
- Final test loss: `0.6619`

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| choice | Validation balanced accuracy `0.6856` | Reference papers show robust above-chance neural choice decoding qualitatively, and methodpaper reports side-video choice AUC `0.66 ± 0.12` during sample/delay and `0.99 ± 0.01` during response; not directly the same modality / metric but our result is in the expected above-chance range |
| outcome | Validation balanced accuracy `0.6582` | No exact paper decoder metric found for outcome |
| early_lick | Validation balanced accuracy `0.7506` | No exact paper decoder metric found for early lick |
| tongue_y | Validation balanced accuracy `0.6133` | No exact paper decoder metric found for discretized tongue y; methodpaper shows strong movement information in this dataset, so above-chance decoding is expected |

Accuracy-vs-chance check:
- choice: `0.6856 / 0.3333 = 2.06x` chance
- outcome: `0.6582 / 0.3333 = 1.97x` chance
- early_lick: `0.7506 / 0.5000 = 1.50x` chance
- tongue_y: `0.6133 / 0.2500 = 2.45x` chance
- Result: all outputs are above chance; none are below chance, and none fall meaningfully below the `1.5x chance` audit threshold

Train-vs-validation gap check:
- choice: `0.7206 / 0.6856 = 1.05x`
- outcome: `0.6956 / 0.6582 = 1.06x`
- early_lick: `0.7983 / 0.7506 = 1.06x`
- tongue_y: `0.6651 / 0.6133 = 1.08x`
- Result: no output shows the `>1.5x` train/validation gap that would suggest severe overfitting or leakage

Comparison-to-paper notes:
- I found explicit decoder-performance numbers in the methodpaper for **choice decoded from video**, not from neural activity for the exact outputs used here:
  - pre-sample choice AUC `0.51 ± 0.06`
  - sample/delay choice AUC `0.66 ± 0.12`
  - response-epoch choice AUC `0.99 ± 0.01`
- These are not directly comparable to the present neural balanced-accuracy decoder because:
  - modality differs (`video` versus `neural`)
  - metric differs (`AUC` versus `balanced accuracy`)
  - our decoder predicts whole-trial categorical outputs plus time-varying tongue class, not the paper’s exact sliding-epoch setup
- I did not find exact numerical paper benchmarks for outcome, early lick, or discretized tongue-y decoding from neural activity.
- The datapaper reports neural choice decoding qualitatively as clearly above chance, which is consistent with the present result.

### Issues Found and Resolved
- No new conversion issues were revealed by the full decoder results.
- Because all outputs are above chance, train/validation gaps are small, and the choice-decoding result is qualitatively aligned with the reference papers, no further conversion changes were required after the full training run.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Files added during cleanup:
- `/app/README.md`
- `/app/cache/README_CACHE.md`
- `/app/cache/raw_sanity_checks.py`

Cleanup notes:
- Required deliverable logs were kept at their workflow-specified `/app/*.txt` paths.
- Cached audit logic was moved into `/app/cache/` as a reusable standalone script rather than leaving investigation snippets only in shell history.
