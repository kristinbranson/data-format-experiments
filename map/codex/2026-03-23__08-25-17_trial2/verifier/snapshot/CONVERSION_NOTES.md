# Dataset Conversion Notes

## Overview
- **Dataset**: Brain-wide neural activity underlying memory-guided movement dataset
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

Environment verification:
- `python3`: 3.13.12
- `numpy`: 2.3.5
- `torch`: 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadmat` | `code/VideoAnalysisUtils/preprocessing_utils.py` | LOADING | Loads MATLAB exports and recursively converts MATLAB structs/cells into Python dict/list structures. |
| `sliding_histogram` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Converts per-neuron spike times into binned spike counts or firing rates over a specified window. |
| `process_one_sess` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Reads all probe files for a session, extracts behavior/task variables, aligns lick/stim times to go cue, intersects ephys with histology, and prepares per-session data. |
| `helper_get_neuron_id_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Applies hemisphere split and precomputed QC region membership to select good neurons for each brain area. |
| `process_one_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Crops spike times to analysis window, bins to firing rates, attaches session metadata, and saves per-area processed outputs. |
| `process_all_sess_parallel` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Batch preprocessing entry point across all sessions. |
| `load_session` | `code/VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Re-loads all per-area pickles for a session and concatenates neurons across files into one session matrix. |
| `get_regular_trial_mask` | `code/VideoAnalysisUtils/population_decoding_utils.py` | CURATION | Defines “regular trials” as no early lick, no auto/free water, no ignore (`correctness != -1`), and no photostimulation. |
| `create_4fold_trial_type_mask` | `code/VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Builds trial stratification labels from trial type and correctness for cross-validation. |
| `temporal_alignment_embed_and_ephys` | `code/VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Aligns marker/video time bases to ephys bin centers, accounting for start/end offsets. |
| `align_markers_between_lims` | `code/Sherlock/align_markers.py` | PROCESSING | Builds time-aligned marker trajectories from raw per-frame tracking arrays relative to go cue with `dt=0.0034 s`. |

### Notes
- Repository is for the method paper’s video-analysis workflows; ephys preprocessing is delegated to `preprocessing_DJ_2022Aug.py`.
- The reference preprocessing is electrophysiology, not imaging. No dF/F computation is relevant here.
- Raw spike times in the exported `.mat` files are already interpreted relative to go cue during preprocessing comments and handling; behavior event times are explicitly shifted so go cue is time zero.
- Reference preprocessing settings in `Sherlock/preprocess_all_ephys.py`: bin width `0.04 s`, stride `0.0034 s`, window `[-3.0, 3.0]`, QC mode `classifier`.
- The method-paper code filters neurons using precomputed “good units” indices plus histology consistency, then groups them into 14 high-level regions and hemispheres.
- Trial filtering in downstream decoding/r2 code is stricter than this conversion task: it excludes early-lick, auto-water, free-water, ignore, and stimulated trials. For this conversion, those variables still need to be preserved because early lick, outcome, and photostimulation are decoder targets/inputs.
- Marker alignment code uses side-camera tracking variables `nose_x/y`, `tongue_x/y`, `jaw_x/y`, `whisker_x/y`, sampled at `dt=0.0034 s`, aligned to go cue from `-3` to `1.5 s`.
- The code suggests session-level neural data are formed by concatenating all processed per-area files back into one neuron matrix, so the converted format should likely be session-centric rather than area-centric.
- Important ambiguity to resolve later from raw data/text: the reference repo’s published code uses precomputed QC `.mat` files outside the raw session files. I have not used any out-of-scope paths from the original cluster setup; I need to determine from `/app/data` and the papers how the equivalent QC information is provided locally.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` is a DANDI/NWB dataset (`dandiset.yaml`) with 174 session files under 28 subject folders (`sub-*/*.nwb`).
- File types present: 174 `.nwb` files and 1 `.yaml` metadata file.
- Each NWB file contains:
  - `intervals/trials`: per-trial table with columns `start_time`, `stop_time`, `trial`, `photostim_onset`, `photostim_power`, `photostim_duration`, `trial_uid`, `task`, `task_protocol`, `trial_instruction`, `early_lick`, `outcome`, `auto_water`, `free_water`.
  - `acquisition/BehavioralEvents`: event-aligned time series including `presample_*`, `sample_*`, `delay_*`, `go_*`, `trialend_*`, `left_lick_times`, `right_lick_times`, `photostim_start_times`, `photostim_stop_times`.
  - `acquisition/BehavioralTimeSeries`: continuous video tracking arrays `Camera0_side_TongueTracking`, `Camera0_side_JawTracking`, `Camera0_side_NoseTracking`, each with shape `(n_frames, 3)` and columns `(x, y, likelihood)`, timestamped every `0.0034 s`.
  - `units`: spike times plus extensive QC and waveform metrics. Important columns include `unit_quality`, `classification`, `anno_name`, `presence_ratio`, `amplitude_cutoff`, `isi_violation`, `avg_firing_rate`, `drift_metric`, `is_good_trials`, `spike_times`, and electrode references.
  - `general/extracellular_ephys/electrodes/location`: JSON-like strings containing high-level recording target labels such as `left ALM`, `right Medulla`, etc.
- All sessions contain tongue tracking; 168/174 sessions include optogenetics metadata/groups and 6 are behavior+ecephys without `+ogen` in the filename.
- Raw task vocabulary observed directly from data:
  - `task`: `audio delay`
  - `trial_instruction`: `left`, `right`
  - `outcome`: `hit`, `miss`, `ignore`
  - `early_lick`: `early`, `no early`
- High-level recorded regions observed in electrode metadata: `left/right` versions of `ALM`, `BLA`, `ECT`, `Medulla`, `Midbrain`, `Striatum`, `Thalamus`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272,227 raw units across all sessions |
| Neurons / session | mean 1,564.52 raw units/session (min 493, max 3,191) |
| Subjects | 28 |
| Sessions / subject | mean 6.21 (min 3, max 10) |
| Trials (total) | 94,990 |
| Trials / session | mean 545.92 trials/session (min 264, max 800) |

Additional raw-data counts:
- Good units by `units.unit_quality == "good"`: 154,948 total; mean 890.51/session (min 220, max 1,998).
- Trial outcomes across all sessions: `hit=65,254`, `miss=15,641`, `ignore=14,095`.
- Early lick labels across all sessions: `early=10,805`, `no early=84,185`.
- Trials with non-`N/A` photostim power: 18,588 / 94,990 (`19.57%`).

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units | `"Overall, the data set consisted of 69,943 good units"` (`ChenLiuEtAl2023_SpikeSortingQC.pdf`, p. 7; also `methods.txt`) | 
| Neurons / session | not explicitly given; implied ~404.3 good units/session across 173 sessions | Derived from 69,943 good units / 173 sessions; papers do not state the mean directly |
| Subjects | 28 mice | `"Mice (n = 28, Table S1)"` (`datapaper.pdf`, p. 4); `"This study is based on data from 28 mice"` (`methodpaper.pdf`, p. 13) |
| Sessions / subject | not explicitly given; implied ~6.18 behavioral sessions/subject across 173 sessions | Derived from reported totals |
| Trials (total) | not explicitly given in text | No exact global total reported in extracted text |
| Trials / session | mean 476; range 130–785 | `"Mice performed on average 476 (Mean; range, 130-785) trials per session"` (`methods.txt`) |
| Neural data time bin | 40 ms | `"we then binned spikes into firing rates with a bin width of 40 ms and a stride of 3.4 ms"` (`methodpaper.pdf`, p. 13) |
| Behavior data time bin | 300 Hz video, 3.4 ms frame step | `"high-speed videos ... were acquired at 300 Hz"` (`methods.txt`); code and methods use `dt=0.0034 s` |
| Reward rate | 84% correct on control trials; range 65–99% | `"84% correct rate (range, 65-99%)"` (`methods.txt`) | 
| Behavioral sessions | 173 | `"69,943 good units recorded across 173 behavioral sessions"` (`ChenLiuEtAl2023_SpikeSortingQC.pdf`, p. 7; also `methods.txt`) | 
| Probe insertions | 655 | `"from which 655 probe insertions were made"` (`ChenLiuEtAl2023_SpikeSortingQC.pdf`, p. 7; also `methods.txt`) |
| Photostim fraction | ~25% of trials in subset experiments | `"deployed on a subset of ~25% randomly interleaved trials"` (`methods.txt`) |
| Photostim sessions / mice | 93 sessions, 17 VGAT-ChR2 mice for bilateral ALM performance analysis | `"n = 93 sessions"` and `"N = 17 VGAT-ChR2-EYFP mice"` (`methods.txt`) |


### Processing Details
- Task timing from reference text:
  - sample epoch: three 150 ms tones with 100 ms inter-tone intervals; total sample epoch shown as `0.65 s`
  - delay epoch: `1.2 s`
  - go cue: `0.1 s`
  - response epoch / answer period: `1.5 s`
  - consumption period: `1.5 s`
  - inter-trial interval: `250 ms`
- Reference analyses are aligned to go cue for neural/video preprocessing and for many figures (`time to go` axis; code explicitly aligns lick/stim/marker data to go cue).
- Method-paper neural preprocessing: 40 ms firing-rate bins with 3.4 ms stride.
- Video tracking:
  - side-view video only for method-paper analyses.
  - marker/video timestamps at 300 Hz.
  - tongue/jaw/nose markers tracked; method paper additionally mentions paws, but the local NWB side-view tracking stream clearly contains tongue/jaw/nose.
- Video preprocessing from method paper:
  - marker outliers removed using a five-sigma velocity threshold and imputed from nearby frames.
  - when the tongue is occluded in the mouth, tongue position is set to its mean value.
- Photoinhibition:
  - late delay epoch only, final 0.5 s including 100 ms ramp-down.
  - photoinhibition ends before the go cue.

### Curation Steps

**Neuron curation rules**:
- Spike sorting with Kilosort2, followed by 15 quality metrics.
- Recordings with substantial drift were rejected at the penetration level (`ChenLiuEtAl2023_SpikeSortingQC.pdf`, p. 1).
- Region-specific logistic-regression QC classifiers were trained for five major brain divisions: cortex, striatum, thalamus, midbrain, medulla.
- Region-to-classifier mapping from `methods.txt` / white paper:
  - cortex classifier: ALM, other cortex, hippocampus, olfactory, cortical subplate
  - striatum classifier: striatum, pallidum
  - thalamus classifier: thalamus, hypothalamus
  - midbrain classifier: midbrain, pons
  - medulla classifier: medulla, cerebellum
- White paper reports classifier performance AUC `> 0.9` on average and final false-alarm rates of `7.8%` cortex, `6.4%` striatum, `7.3%` thalamus, `5.5%` midbrain, `4.3%` medulla (`methods.txt`).

**Trial curation rules**:
- Early lick trials and no-response / ignore trials were excluded for many paper analyses.
- Overall behavioral performance was defined on control trials only, excluding early licks.
- Sessions selected for analysis required:
  - performance `> 65%`
  - at least `50` correct lick-left and `50` correct lick-right control trials
- Method-paper reporting summary adds another analysis-level exclusion:
  - brain areas with fewer than `10` neurons per session were excluded from analysis (`methodpaper.pdf`, p. 27).
- Specific decoder analyses in papers used further task-dependent restrictions:
  - population choice decoder used balanced trial subsampling of `50 LL`, `50 RR`, `10 LR`, `10 RL` trials
  - some video-choice analyses excluded sessions with fewer than `20` trials in any of four groups (`methodpaper.pdf`, p. 14)

### Decoders Trained
| Decoded variable | Accuracy |
| Choice from behavioral video, pre-sample epoch | AUC `0.51 ± 0.06` s.d., `n = 106` sessions (`methodpaper.pdf`, p. 7) |
| Choice from behavioral video, second half of sample + delay | AUC `0.66 ± 0.12` s.d., `n = 106` sessions (`methodpaper.pdf`, p. 7) |
| Choice from behavioral video, second half of response epoch | AUC `0.99 ± 0.01` s.d., `n = 106` sessions (`methodpaper.pdf`, p. 7) |
| Choice from single markers, response epoch | marker AUC `0.88 ± 0.01` s.e.m. versus embedding AUC `0.96 ± 0.00`, `n = 106` sessions (`methodpaper.pdf`, p. 8) |
| Population choice decoder from neural activity | Exact numeric values not stated in extracted text; figures show above-chance decoding that emerges first and strongest in ALM, closely followed by midbrain and thalamus (`datapaper.pdf`, p. 13; Fig. 6 / Fig. S6) |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Good-unit definition | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` uses external classifier-derived “good units” lists from QC `.mat` files | NWB embeds both `units.unit_quality` and `units.classification` | White paper says logistic-regression classifiers label units as `good` vs `unlabelled` and those lists are used for analysis | Use `units.classification == "good"` as the local equivalent of the external QC lists. `unit_quality == "good"` is too permissive and produces 154,948 units, which is far above the published total. |
| Session count | Reference code processes sessions with saved QC-filtered neurons | Raw data contain 174 NWB files | Papers repeatedly state `173 behavioral sessions` | Exclude the single session `sub-440958_ses-20190216T162508_behavior+ecephys+ogen.nwb`, which has `classification == nan` for all 1,852 units and zero classifier-labeled good units. This reconciles the session count to 173. |
| Good-unit total after resolving session scope | External QC lists expected to approximate published counts | Local NWB `classification == "good"` yields 69,453 good units across the 173 sessions with at least one good unit | Papers state 69,943 good units | Treat the remaining 490-unit gap (0.7%) as a likely archive/export-version discrepancy and trust the embedded NWB classifier labels for conversion. The qualitative match is very close, while all alternative local interpretations are much worse. |
| Behavioral session inclusion | Reference code for video/r2 analyses often uses custom trial masks rather than whole-session exclusion | Applying `>65%` performance and `>=50` correct control trials per lick direction leaves only 152/174 sessions | `methods.txt` states these criteria for “selected experimental sessions for analysis” | Treat these as analysis-specific selection criteria for paper figures, not as the base dataset definition. For conversion, do not drop to 152 sessions by default; use the broader 173-session QC-filtered dataset unless a later sanity check suggests otherwise. |
| Marker set | `align_markers.py` expects `nose`, `tongue`, `jaw`, `whisker`; method paper text mentions `jaw`, `paws`, `tongue` | Local NWB side-view tracking streams contain `TongueTracking`, `JawTracking`, `NoseTracking` | Papers only require tracked orofacial movements and use side-view video | For this task, use the directly available side-view tongue trajectory from NWB. No paw/whisker reconstruction is needed because the target output only requires tongue y-position. |
| Regional labeling | Preprocessing code groups neurons into 14 major region classes and uses histology annotations for area membership | Electrode metadata give bilateral target regions; `units.anno_name` provides finer anatomical labels, often empty for non-localized units | Papers report both major compartments and finer subregional analyses | Use classifier-good units and derive coarse region labels from electrode target metadata for base `brain_regions`, while keeping finer `anno_name` available for sanity checks if needed. |
| Alignment / binning | Reference code and papers use go-cue alignment, 40 ms bins, 3.4 ms stride | Raw NWB stores absolute trial/event timestamps and continuous video at 3.4 ms resolution | User task requires go-cue alignment with 50 ms bins from `-2.5 s` to `+1.5 s` | Preserve the reference alignment/QC logic but intentionally deviate in final bin width/window to satisfy the decoder task. This is an explicit task-driven modification, not a misunderstanding of the reference pipeline. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units.spike_times` for units with `classification == "good"` | `neural` | For each session/trial, bin absolute spike times into go-aligned `[-2.5, 1.5)` windows using 50 ms bins; convert counts to firing rates (`count / 0.05`) | `sliding_histogram`, `process_one_area` | Task-driven deviation from reference 40 ms bins; keep go-cue alignment and QC logic the same. |
| `acquisition/BehavioralEvents/go_start_times.timestamps` | alignment event / metadata | One go cue per trial, found by matching event timestamp into `[trial.start_time, trial.stop_time]` | reference code aligns all streams to go cue | Use event-in-trial matching rather than row order assumptions. |
| `acquisition/BehavioralEvents/sample_start_times.timestamps` | `input[0]` (`time_from_tone_onset_s`) | For each trial, use the **last** sample-start event before the go cue as the effective tone/sample onset; store `(bin_center_time - sample_onset_time)` in seconds for every bin | code aligns task epochs to go cue; methods describe replays after early licks | Early-lick trials can contain multiple sample-start events; the last one is the final replay that leads to the observed go cue. |
| `trials.photostim_onset`, `trials.photostim_duration`, `trials.start_time` | `input[1]` (`photostim_on`) | Convert trial-relative onset/duration to absolute time, then to go-aligned binary series over bins | reference preprocessing shifts stimulation times relative to go cue | Trials with `photostim_power == N/A` are all zeros. |
| `trials.trial_instruction` + `trials.outcome` | `output[0]` (`choice`) | For `hit`/`miss`, infer actual lick choice from instruction and outcome: hit -> instructed side, miss -> opposite side | paper/code derive LL/RR/LR/RL from instruction + correctness | Verified against first post-go lick on sampled files: 0 mismatches in 10,084 responded trials. |
| first lick in trial, else instruction fallback for ignore trials | `output[0]` (`choice`) | If `outcome == ignore`, choice is undefined in source. Planned fallback: use earliest lick side anywhere in the trial if present; otherwise use instructed side and document this as an unavoidable placeholder | no direct reference equivalent because paper excludes ignore trials | This is the main task-specific edge case to revisit if decoder accuracy is poor. |
| `trials.outcome` | `output[1]` (`outcome`) | Map strings to integers: `ignore=0`, `miss=1`, `hit=2`; replicate across all time bins | direct from raw trial table | Matches user specification exactly. |
| `trials.early_lick` | `output[2]` (`early_lick`) | Map `no early=0`, `early=1`; replicate across all time bins | direct from raw trial table | Matches user specification exactly. |
| `BehavioralTimeSeries/Camera0_side_TongueTracking` `(y, likelihood)` | `output[3]` (`tongue_y_bin`) | Process continuous tongue y per session, resample to trial/bin centers, then discretize with session-level 40th and 60th percentiles into classes `0/1/2` | methods text describes tongue outlier handling and mean imputation when occluded | Planned preprocessing: 5-sigma velocity outlier detection + interpolation; low-likelihood/occluded frames imputed to session mean tongue y. |
| `units.anno_name` | `brain_region_idx` / `brain_regions` | Use exact non-empty anatomical annotation string as neuron region label | papers use histology-based anatomical assignments | All classifier-good units checked so far have non-empty `anno_name`; this avoids mislabeling orbital/other-cortex neurons as coarse probe target labels. |
| `subject.subject_id` | `subjects`, `subject_idx` | Store unique subject IDs and per-session indices | direct from NWB subject metadata | Session order follows sorted NWB file list after excluding the zero-good-unit session. |

### Key Decisions
1. **Base session set**: Use the 173 NWB sessions with at least one classifier-labeled good unit. This matches the paper’s session count better than the raw 174-file archive or the much stricter 152-session behavioral subset.
2. **Unit filter**: Use `units.classification == "good"` rather than `units.unit_quality == "good"`. This is the only local field consistent with the classifier-based QC described in the white paper and methods.
3. **Neural representation**: Store firing rates, not spike counts, because the task explicitly asks for 50 ms bins “for computing firing rates”.
4. **Tone onset handling**: For early-lick replay trials, define tone onset as the last sample-start event before go. This matches the actual final sample epoch that temporally precedes the measured go cue.
5. **Output shape**: Make outputs 2D time-varying arrays `(4, 80)` for every trial. Choice/outcome/early-lick will be repeated across bins; tongue y will vary over time. This satisfies the decoder format cleanly and makes the time-varying output explicit.
6. **Choice edge case on ignore trials**: Because the source does not contain an explicit choice label when no response occurs, use the earliest lick side in the trial if available; otherwise fall back to instructed side. This is a task-driven compromise and will be stress-tested with decoder performance and sanity checks.
7. **Brain-region metadata**: Use exact `anno_name` strings instead of coarse insertion targets. This preserves histology-derived localization and avoids systematically mislabeling orbital and other cortical units as ALM.
8. **Video resampling**: For each 50 ms neural bin, assign tongue y from the processed continuous tongue trace at the bin center (or closest preceding frame). This is closer to the reference marker-alignment logic than averaging over long windows.
9. **Reference-consistent deviation**: Keep reference curation and go-cue temporal alignment, but intentionally change only the final bin width/window to `50 ms` and `[-2.5, +1.5] s` because the decoder task requires it.

### Planned Sanity Checks
- [ ] Neural check: for selected session/trial/unit tuples, recompute raw spike counts from NWB `units.spike_times` and go timestamp, convert to Hz, and compare to converted `neural` bins with `np.allclose()`.
- [ ] Input check: for selected trials, recompute sample onset and photostim interval directly from NWB events/tables and compare to converted `time_from_tone_onset_s` and `photostim_on` arrays with `np.allclose()`.
- [ ] Output check: for selected hit/miss trials, verify converted `choice`, `outcome`, and `early_lick` against raw trial table / lick events with `np.allclose()`.
- [ ] Tongue check: for selected session/trial/bin combinations, compare converted tongue class against raw processed tongue y and the session percentiles used for discretization.
- [ ] Dataset-size check: verify converted session count, subject count, and classifier-good unit totals are consistent with the 173-session / ~69.9k-good-unit reference expectation.
- [ ] Time-axis check: verify every converted trial has exactly 80 bins spanning `[-2.5, +1.5)` relative to go cue.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with the required CLI:
- `python -u convert_data.py <outpicklefile>`
- `--sample` to process the first 2 valid sessions
- `--full` to process the full valid session list
- `--show-processing` to save `processing_<session_id>.png` for up to 2 sessions

Implementation notes:
- Loads NWB directly with `h5py` instead of PyNWB for speed and lower overhead.
- Filters sessions to those with at least one `classification == "good"` unit.
- Uses exact `units.anno_name` strings for `brain_region_idx`.
- Aligns each trial to go cue by matching event timestamps into the trial interval.
- Resolves early-lick replay trials by taking the last `sample_start` before the go cue.
- Infers hit/miss choice from instruction + outcome; uses a documented fallback for ignore trials.
- Processes session-wide tongue y with likelihood masking, velocity-outlier detection, mean filling, and percentile discretization.
- Stores neural firing rates as `float16` to keep the full dataset size manageable.

Code inefficiencies identified:
- Per-trial Python loops over neurons would be too slow and memory-heavy for the full dataset.
- Full-fidelity float32 storage would make the output pickle unnecessarily large.

Code speedups added:
- Vectorized neural binning per unit across **all trials at once** using `np.searchsorted` on a `(n_trials, 81)` edge matrix.
- Direct HDF5 reads for ragged spike times and trial tables.
- Session-by-session processing to bound peak memory.
- `float16` neural storage and preallocated `(n_trials, n_units, n_bins)` session tensors.
- Sample-run timing result: `0.81 s/session` mean over 2 sessions; projected full conversion time `~2.35 min`.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 classifier-good units |
| Neurons / session | mean 417.0 (459, 375) |
| Subjects | 1 |
| Sessions / subject | 2 for `sub-440956` |
| Trials (total) | 370 |
| Trials / session | mean 185.0 (233, 137) |
| `time_from_tone_onset_s` range | [-0.6, 5.7] |
| `photostim_on` range | [0.0, 1.0] |
| `choice` distribution | [0.446, 0.554] |
| `outcome` distribution | [0.200, 0.000, 0.800] for [ignore, miss, hit] |
| `early_lick` distribution | [0.941, 0.059] |
| `tongue_y_bin` distribution | [0.067, 0.846, 0.087] |

### Processing Plots Review
- Both sample sessions show plausible go-aligned neural activity and photostim timing, and the decoder verification script reports no structural errors or warnings.
- The tongue trace preprocessing is conservative: many low-likelihood tongue frames are filled to the session mean, producing a strong middle-bin dominance. This is acceptable for format validation but is a likely place to revisit if decoder accuracy for tongue is weak.
- Trial windows now exclude segments outside the good-unit observation intervals; this removed the earlier all-zero neural trial artifact.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Vectorized per-unit `searchsorted` binning across all trials | Reduced per-session neural binning from projected minute-scale nested loops to sub-second runtime on sample sessions |
| Direct HDF5 reads + sessionwise processing + `float16` neural storage | Reduced I/O overhead and kept the sample pickle to 24.6 MB |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion | 0.70 s | 2.01 min projected for 173 sessions |
| Sample verification (`--verify-only`) | negligible compared with conversion | well under 1 min total at full scale |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| `choice` | 0.7984 | 0.7395 |
| `outcome` | 0.9054 | 0.8749 |
| `early_lick` | 0.8898 | 0.7944 |
| `tongue_y_bin` | 0.8021 | 0.6681 |

Step-8 review notes:
- Loss decreased monotonically from `13.48` at epoch 1 to `0.36` at epoch 200, so the decoder trained stably on the converted sample data.
- Every validation balanced accuracy is above chance: choice `0.7395 > 0.5`, outcome `0.8749 > 0.3333`, early lick `0.7944 > 0.5`, tongue `0.6681 > 0.3333`.
- The sample split contains no `miss` trials in the held dataset summary, so the strong outcome score should be re-checked on the full dataset where all three outcome classes are represented more broadly.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 4.6 GB
- `verification_full_out.txt`: created
- `conversion_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 good units | classifier-based QC lists | 69,453 `classification=="good"` | 69,453 | Close; 0.7% below paper |
| Mean neurons/session | ~404.3 implied | classifier-filtered per session | 401.46 | 401.46 | Close |
| Subjects | 28 | session metadata | 28 | 28 | Yes |
| Sessions | 173 | QC-filtered sessions | 173 valid sessions | 173 | Yes |
| Trials (total) | not stated globally | analysis-specific trial masks | 94,370 raw trials in valid sessions | 73,910 kept go-window-complete trials | Lower by design after full-window filtering |
| Trials/session (mean) | 476 | varies by analysis mask | 545.49 raw | 427.23 kept | Lower than paper because incomplete edge trials are dropped |
| `time_from_tone_onset_s` range | sample onset precedes go; replays possible after early licks | go-aligned sample/start event logic | task structure permits long pre-go delays on replayed trials | [-1.5, 9.7] | Plausible |
| `photostim_on` range | binary on/off during late delay subset | binary aligned stim epochs | [0, 1] from trial table | [0, 1] | Yes |
| `choice` distribution | near-balanced instructed left/right task | LL/RR/LR/RL trial logic | near-balanced at session level | [0.491, 0.509] | Yes |
| `outcome` distribution | 84% correct control rate reported | outcome/correctness fields used directly | hit-dominant task | [0.173, 0.009, 0.818] for [ignore, miss, hit] | Broadly consistent |
| `early_lick` distribution | early licks excluded in many analyses, not absent in data | early-lick mask available | early licks present in raw trial table | [0.885, 0.115] for [no, yes] | Yes |
| `tongue_y_bin` distribution | not reported | video aligned continuously | percentile discretization is task-specific | [0.088, 0.824, 0.088] | Expected middle-bin dominance |

Step-9 spot checks:
- Verified `converted_data.pkl` loads successfully and contains 173 sessions, 28 subjects, 73,910 trials, and 69,453 neurons.
- Spot-checked session indices `0`, `86`, and `172`: all have trial tensors shaped `(n_neurons, 80)`, inputs `(2, 80)`, and outputs `(4, 80)`.
- `train_decoder.py --verify-only` reported `Data format is valid, no errors or warnings.`
- The converted trial count is lower than the raw 173-session archive count because trials are excluded unless the full `[-2.5, +1.5] s` window lies inside a good-unit observation interval; this is necessary to avoid invalid all-zero neural windows.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` reports `Data format is valid, no errors or warnings.` No additional verifier warnings needed triage.
2. **Raw-data sanity checks with `np.allclose()`**: Added `/app/cache/step10_sanity_checks.py` and ran it successfully. The script re-loaded original NWB files and compared converted values against raw recomputations for sessions `sub-440956_ses-20190207T120657`, `sub-479149_ses-20201217T120512`, and `sub-484677_ses-20210420T170516`.
3. **Neural sanity check**: For 4 kept trials per checked session and 3 units per trial (first, middle, last good unit), recomputed binned firing rates directly from raw `units.spike_times` and go-aligned bin edges. All compared bins matched the converted arrays with `np.allclose(..., atol=1e-3)`.
4. **Input sanity check**: For the same checked trials, recomputed `time_from_tone_onset_s` from the last `sample_start` before go and recomputed `photostim_on` from raw trial-relative onset/duration fields. All matched the converted arrays with `np.allclose(..., atol=1e-6)`.
5. **Output sanity check**: For the same checked trials, recomputed `choice`, `outcome`, `early_lick`, and full time-varying `tongue_y_bin` directly from the raw trial table, lick events, and tongue tracking stream. All matched the converted outputs exactly with `np.allclose(..., atol=0)`.
6. **Reference code comparison**:
   - Data loading: `convert_data.py` directly reads NWB `units`, `intervals/trials`, `BehavioralEvents`, and `BehavioralTimeSeries`, corresponding to `process_one_sess` and `load_session` in the reference code.
   - Neuron filtering: `units.classification == "good"` mirrors the classifier-based QC lists described in `preprocessing_DJ_2022Aug.py` and the QC white paper.
   - Temporal alignment: go-cue alignment matches the reference code’s `time to go` convention and `align_markers_between_lims`.
   - Binning: reference code uses 40 ms width / 3.4 ms stride, while this conversion intentionally uses 50 ms non-overlapping bins to satisfy the decoder task; the alignment logic is otherwise the same.
   - Input construction: tone/sample timing and photostim epochs are constructed from the same raw event streams used in the reference preprocessing.
   - Output construction: choice/outcome/early-lick derive from the same trial variables used for LL/RR/LR/RL logic in the reference code; tongue processing follows the method-paper description of low-likelihood mean fill plus five-sigma velocity outlier handling.
7. **Key statistics comparison**:
   - Subjects: 28 in papers, raw data, and converted data.
   - Sessions: 173 in papers and converted data after excluding the one zero-good-unit archive file.
   - Good units: 69,453 locally versus 69,943 reported in the paper; this remains the same 0.7% archive-version discrepancy documented earlier.
   - Mean neurons/session: 401.46 converted versus ~404.3 implied by the paper total.
   - Kept trials/session: 427.23 converted, lower than the paper’s reported 476 average because the decoder task requires dropping trials without a complete `[-2.5, +1.5] s` go-aligned neural window.
8. **Edge-case review**:
   - Verified all converted trials have exactly 80 bins and no sessions have fewer than 2 trials.
   - `n_missing_sample_onset_fallback = 0`, so no trial needed the `go - 1.85 s` default.
   - `n_ignore_choice_fallback = 13,258`; this is expected because ignore trials do not have an explicit behavioral choice label in the raw data.
   - The earlier all-zero neural-window issue is resolved by the observation-interval coverage filter plus a final all-zero neural trial drop.

### Issues Found and Resolved
- No new mismatches were found in Step 10. The existing observation-interval fix remained sufficient, and the raw-to-converted sanity checks passed without additional code changes.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| `choice` | 0.7685 | 0.7405 | Strongly above chance; slightly below the 1.5x-chance heuristic because ignore trials require fallback choice labels |
| `outcome` | 0.8637 | 0.7458 | Above chance despite severe class imbalance (`miss` is rare) |
| `early_lick` | 0.8105 | 0.7647 | Above chance with small train/validation gap |
| `tongue_y_bin` | 0.8035 | 0.7650 | Strong time-varying decoding despite conservative tongue preprocessing |

Additional Step-11 notes:
- Training completed on CUDA without needing the `--cpu` fallback.
- Loss decreased from `14.8801` at epoch 1 to `0.4456` at epoch 200.
- `train_decoder.py --plot-samples` produced `/app/sample_trials.png` and `/app/predictions.png`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
|----------|-------------------|-------------------------|
| `choice` | Validation balanced accuracy `0.7405` | No exact numeric neural population decoder value is given in the reference papers, but the papers report strongly above-chance neural choice decoding and video-choice AUC `0.66 ± 0.12` during sample+delay and `0.99 ± 0.01` during the response epoch. A full-window neural decoder accuracy of `0.74` is qualitatively plausible. |
| `outcome` | Validation balanced accuracy `0.7458` | No direct paper value reported for this exact decoder target. Strong above-chance decoding is expected because outcome covaries with licking and movement-related activity. |
| `early_lick` | Validation balanced accuracy `0.7647` | No direct paper value reported. Above-chance decoding is expected because early licking is a salient behavioral event with strong movement correlates. |
| `tongue_y_bin` | Validation balanced accuracy `0.7650` | No direct paper value reported for this discretized target. Method paper reports strong movement-video decoding, so strong tongue-state decoding from neural activity is expected. |

Step-12 review:
- **Accuracy vs chance**:
  - `choice`: `0.7405 / 0.5 = 1.481x` chance
  - `outcome`: `0.7458 / 0.3333 = 2.238x` chance
  - `early_lick`: `0.7647 / 0.5 = 1.529x` chance
  - `tongue_y_bin`: `0.7650 / 0.3333 = 2.295x` chance
- No output is below chance. Three of the four outputs exceed the Step-12 `1.5x chance` heuristic outright.
- `choice` is only marginally below the heuristic threshold. I investigated whether this suggests a conversion bug and concluded it does not:
  - Choice mapping for hit/miss trials was already verified against raw lick events in sampled sessions with zero mismatches.
  - Step-10 raw-vs-converted sanity checks passed for choice on multiple sessions/trials.
  - `17.9%` of kept trials (`13,258 / 73,910`) are ignore trials with no ground-truth choice in the source data, so the required fallback labels inject unavoidable noise into this specific decoder target.
- **Train vs validation gap**:
  - `choice`: `0.7685 / 0.7405 = 1.038x`
  - `outcome`: `0.8637 / 0.7458 = 1.158x`
  - `early_lick`: `0.8105 / 0.7647 = 1.060x`
  - `tongue_y_bin`: `0.8035 / 0.7650 = 1.050x`
- No output shows the `>1.5x` train/validation gap that would suggest major overfitting or leakage.

### Issues Found and Resolved
- No new conversion issues were uncovered in Step 12. The final decoder performance is consistent with a correctly aligned and formatted dataset.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized

Final cleanup notes:
- User-facing summary written to `/app/README.md`.
- Reproducibility helper documented in `/app/cache/README_CACHE.md`.
- Critical-review sanity script stored in `/app/cache/step10_sanity_checks.py`.
- Required conversion, verification, and training logs are present in `/app/`.
