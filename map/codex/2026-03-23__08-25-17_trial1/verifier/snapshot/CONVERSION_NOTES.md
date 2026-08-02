# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
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
- `python3` available
- `numpy==2.3.5`
- `torch==2.6.0+cu124`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `loadmat` | `code/VideoAnalysisUtils/preprocessing_utils.py` | LOADING | Loads MATLAB/DataJoint-exported `.mat` files while converting nested MATLAB structs/cells into Python dict/list structures. |
| `process_one_sess` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | LOADING | Reads all probe files for one session; loads behavior/task fields, spike times, unit QC fields, and histology annotations; concatenates probes into a session-level representation. |
| `helper_get_neuron_id_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | CURATION | Applies hemisphere split using CCF ML coordinate (`5700`) and intersects session-level QC-selected neuron ids with histology labels for each major brain area. |
| `process_one_area` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Truncates spike times to a go-cue-aligned window, bins spikes into firing rates with `sliding_histogram`, and saves per-area pickle files containing `fr`, `bin_centers`, task variables, and QC metadata. |
| `sliding_histogram` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Converts per-neuron, per-trial spike times into binned spike counts or firing rates. Output `fr` is shaped `(n_bins, n_trials, n_neurons)`. |
| `process_all_sess_parallel` | `code/VideoAnalysisUtils/preprocessing_DJ_2022Aug.py` | PROCESSING | Batch-preprocesses all sessions in parallel after grouping probe files by session. |
| `load_session` | `code/VideoAnalysisUtils/population_decoding_utils.py` | LOADING | Loads all per-area pickle files for one session and concatenates them into a single session-level firing-rate tensor and metadata dict. |
| `get_regular_trial_mask` | `code/VideoAnalysisUtils/population_decoding_utils.py` and `code/VideoAnalysisUtils/functions_for_r2.py` | CURATION | Defines “regular trials” by excluding early-lick, auto-water, free-water, no-response, and photostimulation trials. |
| `create_4fold_trial_type_mask` | `code/VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Creates stratification labels combining trial type and correctness for decoder/train-test splitting. |
| `temporal_alignment_embed_and_ephys` | `code/VideoAnalysisUtils/functions_for_r2.py` | PROCESSING | Crops embedding/video and ephys streams onto a shared time axis using the common frame/bin step (`dt=0.0034`). |
| `align_markers_between_lims` | `code/Sherlock/align_markers.py` | PROCESSING | Aligns marker trajectories (including `tongue_y`) to go cue across `t_min=-3` to `t_max=1.5` with `dt=0.0034`, taking the last frame in each time bin. |

### Notes
- `code/README.md` identifies this repository as the analysis code for the 2025 movement-encoding paper and points to processed MAP data from DANDI/Zenodo.
- The reference code is for electrophysiology, not calcium imaging. There is no dF/F step in the explored reference path.
- Raw spike times are treated as already aligned to go cue in `process_one_area` (“in Susu's data, spike_times are relative to go cue time”).
- The preprocessing script used in the paper repository (`code/Sherlock/preprocess_all_ephys.py`) sets `bw=0.04`, `stride=0.0034`, `begin_time=-3.`, `end_time=3.`, and `qc_mode='classifier'`.
- Neuron curation is required. It is done by:
- intersecting units that have both electrophysiology and histology annotations,
- loading session QC files from `goodunits/` when `qc_mode='classifier'`,
- selecting area-specific good neuron ids from the QC file,
- splitting left/right hemisphere by CCF ML coordinate,
- then carrying forward unit QC metrics such as `presence_ratio`, `amplitude_cutoff`, `isi_violation`, `avg_firing_rate`, and `drift_metric`.
- Trial curation used for downstream decoding excludes early-lick, auto-water, free-water, no-response (`correctness == -1`), and photostimulation trials.
- Behavioral/task fields retained in the preprocessed ephys pickles include `lick_directions`, `lick_times`, `correctness`, `stimulation`, `trial_type`, `delay_period`, and `sample_period`.
- Marker alignment code confirms the raw files contain side-camera tracked markers including `tongue_y`, aligned to go cue with `dt=0.0034`.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- `data/` is a DANDI-style NWB dataset for the Mesoscale Activity Map Dataset (`dandiset.yaml` + per-subject folders).
- Subject folders are named `sub-<subject_id>/`.
- Each subject folder contains one NWB file per session, named like `sub-440956_ses-20190207T120657_behavior+ecephys+ogen.nwb`.
- File format is NWB/HDF5.
- Across all 174 NWB files, the trial schema and unit schema are consistent (`n_unique_trial_schema = 1`, `n_unique_unit_schema = 1`).
- Each sampled file inspected contains:
- `intervals/trials`: trial table with columns `trial`, `photostim_onset`, `photostim_power`, `photostim_duration`, `trial_uid`, `task`, `task_protocol`, `trial_instruction`, `early_lick`, `outcome`, `auto_water`, `free_water`, plus `start_time` and `stop_time`.
- `units`: unit table with spike times and quality metrics including `unit_quality`, `avg_firing_rate`, `presence_ratio`, `amplitude_cutoff`, `isi_violation`, `classification`, and `anno_name`.
- `acquisition/BehavioralEvents`: event series including `sample_start_times`, `delay_start_times`, `go_start_times`, `left_lick_times`, `right_lick_times`, `photostim_start_times`, `photostim_stop_times`, and trial-end events.
- `acquisition/BehavioralTimeSeries`: continuous tracked markers including `Camera0_side_TongueTracking`, `Camera0_side_JawTracking`, and `Camera0_side_NoseTracking`.
- Sample file inspection (`sub-440956_ses-20190207T120657_behavior+ecephys+ogen.nwb`):
- 368 trials, 1952 units.
- `trial_instruction` values are strings like `right`.
- `early_lick` values are strings like `early` / `no early`.
- `outcome` values are strings like `hit` / `ignore`.
- `photostim_*` trial-table fields are strings such as `N/A` on no-stim trials.
- Tongue tracking is a continuous `(n_frames, 3)` time series with columns `(tongue_x, tongue_y, tongue_likelihood)` and explicit timestamps.
- Units include ragged `spike_times` plus per-unit CCF annotation in `anno_name`.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272,227 raw units across NWB files |
| Neurons / session | mean 1,564.52 (min 493, max 3,191) |
| Subjects | 28 |
| Sessions / subject | mean 6.21 (min 3, max 10) |
| Trials (total) | 94,990 |
| Trials / session | mean 545.92 (min 264, max 800) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units | “Overall, the dataset consisted of 69,943 good units recorded across 173 behavioral sessions” |
| Neurons / session | ~404 good units/session mean (69,943 / 173) | Derived from the paper’s total good units and sessions |
| Subjects | 28 mice | “This study is based on data from 28 mice” |
| Sessions / subject | 173 / 28 = ~6.18 sessions/mouse mean | Derived from paper totals |
| Trials (total) | Not directly tabulated in text | No paper text found with a dataset-wide total trial count |
| Trials / session | Mean 476; range 130-785 | “Mice performed on average 476 (Mean; range, 130-785) trials per session” |
| Neural data time bin | 200 ms for published choice decoder; 10 ms step | “Population choice decoders were calculated using spike rates with causal sliding time windows of 200 ms and a step size of 10 ms” |
| Behavior data time bin | 300 Hz video sampling | “High-speed videos from a side view and a bottom view … were acquired at 300 Hz” |
| Reward rate | 84% correct rate on selected sessions; range 65-99% | “84% correct rate (range, 65-99%)” |
| Session inclusion | performance > 65%; at least 50 correct left and 50 correct right trials | “overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each” |
| Photostim fraction | ~25% randomly interleaved trials | “Photoinhibition … was deployed on a subset of ~25% randomly interleaved trials” |
| Photostim timing | late delay, last 0.5 s, ending before go cue | “We silenced ALM activity during the late delay epoch (last 0.5 s)… photoinhibition always ended before the ‘Go’ cue.” |
| QC yield | 25.9% of Kilosort2 clusters retained as good units | “This corresponds to 25.9 % of clusters reported by Kilosort2.” |
| QC classifier performance | ROC AUC > 0.9; false alarm rates 7.8/6.4/7.3/5.5/4.3% by major area | White paper: “Ten-fold crossvalidation demonstrated that the AUC was on average > 0.9” and datapaper false alarm-rate text |


### Processing Details
- Task structure from `methods.txt` / datapaper:
- sample epoch: three pure tones (150 ms each) with 100 ms inter-tone intervals,
- delay epoch: 1.2 s,
- go cue: 0.1 s auditory cue marking delay end,
- response epoch: 1.5 s answer period,
- consumption period: 1.5 s after correct lick.
- Early licks during sample/delay trigger replay of the epoch in the task, and early-lick plus no-response trials are excluded for analysis.
- Published analyses align task epochs to go cue (`Time to go (s)` in figures), and reference code treats spike times as already go-cue referenced.
- Video tracking uses DeepLabCut markers for tongue, jaw, and nose at 300 Hz.
- Published population choice decoding uses causal 200 ms windows stepped every 10 ms; our conversion must deviate here only because the task specification requires 50 ms bins over [-2.5, 1.5] s.

### Curation Steps

**Neuron curation rules**:
- Kilosort2 output was quality-controlled with 15 metrics.
- Five region-specific logistic-regression classifiers were trained from manually curated `good` versus `unlabeled` units.
- Area mapping for classifiers:
- cortex classifier: ALM, other cortex, hippocampus, olfactory, cortical subplate,
- striatum classifier: striatum and pallidum,
- thalamus classifier: thalamus and hypothalamus,
- midbrain classifier: midbrain and pons,
- medulla classifier: medulla and cerebellum.
- False alarm rates on held-out data: cortex 7.8%, striatum 6.4%, thalamus 7.3%, midbrain 5.5%, medulla 4.3%.

**Trial curation rules**:
- Exclude early-lick trials.
- Exclude no-response trials.
- For overall performance in the paper, use control trials only (no photostimulation) and exclude early licks.
- For photostimulation-effect analyses, require at least 10 photostimulation trials per unit.

### Decoders Trained
| Decoded variable | Accuracy |
| Choice (published population decoder, Figure 6D) | Above chance (> 0.5) with 200-neuron pseudo-populations; exact numeric trace shown in figure, not tabulated in text |
| QC classifier (white paper, not behavior decoder) | ROC AUC > 0.9 on average across major-area classifiers |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session count | Reference preprocessing code groups and processes all sessions it finds; downstream analyses operate on QC-filtered/preprocessed sessions | 174 raw NWB session files | 173 behavioral sessions after stringent QC | One raw NWB session (`sub-440958_ses-20190216T162508_behavior+ecephys+ogen.nwb`) has zero units with `classification == good`; after excluding sessions with zero good units, raw data and papers agree at 173 analyzable sessions. |
| Neuron count | Code relies on classifier-selected good units from QC files | 272,227 raw units across NWB files; 69,453 units with `classification == good` in NWB | 69,943 good units | Use classifier-approved units as the intended population. The NWB `classification == good` total is close but 490 below the paper total; likely due to release/version differences or minor pipeline differences between paper and NWB export. Track this in later sanity checks. |
| Trials/session | Raw NWB retains all task trials | 545.9 trials/session mean raw | 476 trials/session mean for selected experimental sessions | The papers’ trial statistic reflects analyzed sessions/trials after exclusions; raw NWB stores all trials. Conversion should keep raw trial availability first, then apply explicit trial filters required by the decoder/reference logic. |
| Session behavioral filter | Reference text states session inclusion by performance and >=50 correct left/right trials | Applying a simple raw-NWB interpretation of that rule yields 152 sessions, not 173 | 173 sessions reported overall | The published 173-session count is therefore not explained solely by a naive behavioral filter on NWB trial-table labels. For conversion, do not use this raw behavioral filter as the primary session-selection rule. Prefer the QC-consistent rule of excluding sessions with zero good units, then retain trial-level metadata for later filtering. |
| Binning/alignment | Reference preprocessing code bins spikes with `bw=0.04 s`, `stride=0.0034 s`, aligned to go cue | Raw NWB contains spike times plus go-cue and behavioral timestamps | Published choice decoder uses causal 200 ms windows, 10 ms step, aligned to go cue | Preserve the core reference principle of go-cue alignment and classifier-based unit curation, but deliberately re-bin to 50 ms non-overlapping bins over [-2.5, 1.5] s because that is the explicit decoder-task requirement. |
| Trial filtering | Reference code’s `get_regular_trial_mask` excludes early-lick, auto-water, free-water, no-response, and stimulation trials for specific downstream analyses | Raw NWB exposes these fields directly in the trial table | Papers exclude early-lick and no-response trials for analysis; control-trial metrics exclude photostim | Distinguish between raw trial retention and analysis-specific masks. Build the converted dataset from raw trials with required metadata, but document and implement any exclusions that are necessary for alignment validity, QC, or decoder robustness. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times` for units with `classification == good` | `neural` | Bin spikes into firing rates (Hz) in non-overlapping 50 ms bins over [-2.5, 1.5) s relative to go cue | `sliding_histogram`, `process_one_area` | Keep sessions with at least 1 good unit; final session inclusion target is 173 sessions after excluding the zero-good-unit session. |
| `units/is_good_trials` plus `units/obs_intervals` | `neural` validity mask during construction | Keep only trials whose full `[-2.5, +1.5] s` go-aligned window lies inside the union of good-unit observation intervals. When the selected trial count matches the NWB `is_good_trials` columns, use `is_good_trials` directly; otherwise fall back to raw `obs_intervals` support. | No direct code equivalent in explored reference code; NWB-specific validity fields | This resolved a discovered bug where ephys-backed trials were not always a simple prefix of the behavioral trial table. |
| `acquisition/BehavioralEvents/go_start_times.timestamps` | temporal alignment anchor | Per-trial go cue absolute timestamp; subtract from neural/behavioral timestamps | Reference code treats spike times as go-cue aligned | One go cue per trial. |
| Earliest `sample_start_times` event within trial window | `input[0]` (`time_from_tone_onset_s`) | For each bin, set value to `(bin_center_rel_go - sample_onset_rel_go)` in seconds | Raw-task timing consistent with `sample_time` use in preprocessing code | Normal trials give ~-1.85 s tone onset; early-lick trials can have earlier tone onset because replay extends trial structure. |
| `photostim_onset`, `photostim_duration`, `photostim_power` trial-table columns | `input[1]` (`photostim_on`) | Binary time series: 1 during `[onset, onset+duration)` relative to go cue, else 0 | Reference code keeps `stimulation` per trial; papers say photostim ends before go cue | Trial-table onset/duration matched the event-series timestamps in spot checks. |
| First post-go lick direction inferred from left/right lick events | `output[0]` (`choice`) | Encode left=0, right=1 | Paper’s choice variable is actual lick direction on responded trials | Primary rule: first post-go lick. Fallback 1: first lick anywhere in trial. Fallback 2: trial instruction for no-lick ignore trials. This fallback is required because ignore trials have no post-go lick but the decoder spec still requires a binary choice output. |
| `trials/outcome` | `output[1]` (`outcome`) | Map `ignore -> 0`, `miss -> 1`, `hit -> 2` | Consistent with paper’s hit/error/no-response categories | Use exact trial-table labels. |
| `trials/early_lick` | `output[2]` (`early_lick`) | Map `no early -> 0`, `early -> 1` | Consistent with reference code’s `early_lick_trials` mask | Keep all trials; do not exclude early trials because early lick itself is a decoder target. |
| `BehavioralTimeSeries/Camera0_side_TongueTracking[:,1]` with timestamps | `output[3]` (`tongue_y_bin`) | Align absolute timestamps to go cue; within each 50 ms bin take the last available `tongue_y` sample; discretize within session using 40th and 60th percentiles over all aligned binned values | `align_markers_between_lims` | Reference marker alignment uses the last frame within each time step rather than averaging. Use `tongue_likelihood` only for QC diagnostics, not thresholding, to stay close to reference code. |
| `subject.subject_id` or folder name `sub-<id>` | `subjects`, `subject_idx` | Deduplicate subjects; map each session to subject index | N/A | Subject order follows converted session order. |
| `units/anno_name` | `brain_regions`, `brain_region_idx` | Use exact non-empty CCF annotation strings for each kept unit | Reference code carries exact `ccf_label` strings | Exact annotations are closer to the raw NWB + reference code than imposing a new coarse atlas grouping. |

### Key Decisions
1. **Session inclusion**: Keep sessions with at least one `classification == good` unit and at least two trials. This matches the raw-to-paper session discrepancy best: 174 raw NWB files become 173 analyzable sessions once the one zero-good-unit session is removed.
2. **Neuron inclusion**: Use units with `classification == good` as the primary QC filter, because the papers and code describe classifier-selected good units as the analysis set.
3. **Brain-region labels**: Use exact `anno_name` strings for `brain_regions` rather than collapsing to 14 coarse groups. This is the least lossy mapping and matches the reference code’s use of exact `ccf_label` values.
4. **Neural binning**: Use 50 ms non-overlapping bins and convert to firing rates in Hz. This intentionally departs from the paper’s 40 ms / 3.4 ms preprocessing and 200 ms / 10 ms choice-decoder windows only because the user’s decoder task explicitly requires 50 ms bins.
5. **Tone-onset input**: Use the earliest sample-start event in each trial as tone onset. This preserves replay-induced timing shifts visible in early-lick trials and is the most faithful raw-data representation of “time from tone onset”.
6. **Photostim input**: Represent photostimulation as a time-varying binary series, not a static trial label, because the decoder task requests the on/off state at every time point.
7. **Choice output edge case**: For hit/miss trials, decode actual lick direction from lick-event timing. For ignore trials with no lick, use the fallback hierarchy documented above. This is the only unavoidable ambiguity created by the target spec relative to the raw data.
8. **Trial retention and alignment validity**: Retain early-lick, miss, hit, ignore, and photostimulation trials because these are either decoder outputs or decoder inputs. Do not apply the reference code’s `regular_trial_mask` wholesale. However, do require that the full neural decoding window is actually supported by the raw session observation interval.
9. **NWB trial mapping rule**: Do not assume `units/is_good_trials.shape[1]` means “use the first N behavioral trials”. In several sessions the ephys-backed trials form an offset block relative to the full behavioral trial table; use raw go times plus `obs_intervals` to select valid trials.
10. **Tongue discretization**: Discretize after session-level alignment and binning, using percentiles over the kept binned tongue-y values from that session, exactly as requested by the decoder task.

### Planned Sanity Checks
- [ ] For random good units/trials, compare converted spike counts in selected bins against direct counts from raw `spike_times` using bin-edge search.
- [ ] For random trials, compare converted photostim binary vectors against raw trial-table onset/duration and against behavioral-event photostim timestamps.
- [ ] For random trials, compare converted tone-onset input against raw sample-start/go timestamps from the NWB file.
- [ ] For random responded trials, verify the converted `choice` agrees with the first post-go lick direction inferred from raw lick timestamps.
- [ ] For random ignore trials, verify that there is no post-go lick and that the documented fallback path was used.
- [ ] For random session/trial/bin points, compare aligned tongue-y values against the raw last frame in that 50 ms bin.
- [ ] Verify that converted session count after unit QC is 173 and that subject count remains 28.

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `convert_data.py` with:
- CLI: `python -u convert_data.py <outpicklefile> [--full|--sample] [--show-processing]`
- Session discovery from `data/sub-*/*.nwb`
- Good-unit filtering using `units/classification == good`
- Automatic exclusion of the single zero-good-unit session
- Go-cue alignment from `BehavioralEvents/go_start_times`
- Neural binning from raw ragged `units/spike_times` into 50 ms firing-rate bins over [-2.5, 1.5) s
- Time-varying decoder inputs:
- `time_from_tone_onset_s` from per-trial earliest sample onset,
- `photostim_on` from trial-table photostim onset/duration
- Time-varying decoder outputs:
- repeated per-bin `choice`, `outcome`, `early_lick`,
- per-bin discretized `tongue_y_bin`
- Session metadata aggregation for subjects, brain regions, and conversion rules
- Optional processing plots saved as `processing_<session_id>.png`

Smoke-run result:
- `python3 -u /app/convert_data.py /app/_step6_smoke.pkl --sample`
- Completed successfully in 1.44 s for 2 sessions

Code inefficiencies identified:
- Full PyNWB object loading produces avoidable overhead and warning spam.
- Per-spike/per-trial Python loops would be too slow for the full dataset.

Code speedups added:
- Used `h5py` directly instead of PyNWB for the conversion path.
- Used ragged-array indexing (`spike_times_index`) to slice unit spike trains efficiently.
- Used vectorized `np.searchsorted` across all trial/bin edges per unit for spike binning.
- Used vectorized timestamp-to-bin alignment for tongue tracking.
- Stored neural firing rates as `float16` to reduce pickle size while remaining valid floating-point input for the decoder (training code converts to `float32`).

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 good units |
| Neurons / session | [459, 375] |
| Subjects | 1 |
| Sessions / subject | [2] |
| Trials (total) | 495 |
| Trials / session | [368, 127] |
| `time_from_tone_onset_s` range | [-0.625, 5.723] |
| `photostim_on` range | [0, 1] |
| `choice` distribution | [0.483 left, 0.517 right] |
| `outcome` distribution | [0.147 ignore, 0.273 miss, 0.580 hit] |
| `early_lick` distribution | [0.949 no, 0.051 yes] |
| `tongue_y_bin` distribution | [0.399 low, 0.199 mid, 0.402 high] |

### Processing Plots Review
- Saved:
- `processing_sub-440956_ses-20190207T120657_behavior+ecephys+ogen.png`
- `processing_sub-440956_ses-20190208T133600_behavior+ecephys+ogen.png`
- Numeric spot checks backing the plots:
- trial-table photostim onset/duration matched behavioral-event photostim timestamps,
- sample onset / delay onset / go cue alignment matched the raw event times in inspected trials,
- early-lick trials showed extended tone-to-go intervals as expected from replayed epochs,
- one session initially revealed that ephys-backed trials were not a simple behavioral-trial prefix; switching to raw go-time plus `obs_intervals` selection fixed that alignment bug,
- after the fix, sample session 2 kept 127 trials with full supported neural windows and no format warnings remained.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Direct `h5py` reads instead of PyNWB objects | Removed heavy object construction and warning spam; sample conversion stayed under 4 s |
| Vectorized `searchsorted` spike binning per unit | Kept 2-session sample conversion to ~3.2 s |
| Float16 neural storage | Reduced memory/pickle size while preserving verifier compatibility |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion observed | ~1.60 s / kept session | ~4.6 minutes for 173 kept sessions |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| choice | 0.6925 | 0.6451 |
| outcome | 0.6783 | 0.6074 |
| early_lick | 0.8379 | 0.7506 |
| tongue_y_bin | 0.4380 | 0.3802 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 3,706,245,117 bytes
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 good units | classifier-selected good units | 69,453 `classification == good` | 69,453 | Close; matches NWB release, 490 below paper |
| Mean neurons/session | ~404 | QC-selected population per session | 401.46 | 401.46 | Yes |
| Subjects | 28 | 28 subjects expected | 28 | 28 | Yes |
| Sessions | 173 | analyses operate on QC-filtered sessions | 173 after excluding zero-good-unit session | 173 | Yes |
| Trials (total) | not directly tabulated | analysis-specific trial subsets | 93,310 recorded ephys-trial columns; 82,602 no-early trials | 51,346 | Different by task-specific valid-window selection |
| Trials/session (mean) | 476 mean in paper text | analysis-specific | 539.36 recorded; 477.47 no-early | 296.80 | Different by task-specific valid-window selection |
| `time_from_tone_onset_s` range | sample onset near -1.85 s; replay can extend timing | sample/go timing used per trial | raw events support extended replay timing | [-1.525, 11.894] | Yes |
| `photostim_on` range | binary perturbation state | stimulation stored per trial | [0, 1] | [0, 1] | Yes |
| `choice` distribution | not tabulated | depends on selected trials | roughly balanced left/right in task design | [0.495 left, 0.505 right] | Yes |
| `outcome` distribution | paper reports 84% correct on selected control/no-early trials | correctness-based masks used in analyses | no-early/control performance is a separate statistic | [0.155 ignore, 0.093 miss, 0.752 hit] | Different because decoder keeps early/stim/ignore structure |
| `early_lick` distribution | early licks excluded in most paper analyses | regular-trial mask excludes them | raw NWB contains them explicitly | [0.888 no, 0.112 yes] | Expected difference; decoder target requires retention |

Notes:
- Full verification reported `Data format is valid, no errors or warnings.`
- The corrected converter now selects trials using the raw neural support window instead of assuming the first `N` behavioral trials line up with `units/is_good_trials`.
- The neuron/session totals match the QC-filtered NWB release exactly. The trial totals are intentionally lower because the converted dataset keeps only trials whose full `[-2.5, +1.5] s` go-aligned neural window is actually supported by the raw observation interval, which is stricter than the paper’s behavior-table summary statistics.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: `verification_full_out.txt` reported `Data format is valid, no errors or warnings.` No verifier issues remained after the trial-selection fix.
2. **Raw-data sanity checks with `np.allclose()`**:
   - Neural check: directly recounted spikes from raw `units/spike_times` for session `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`, converted trial 6, neuron 4, bin 11; converted value `20.0 Hz` matched raw recomputation exactly (`np.allclose == True`).
   - Input check: for the same session/trial, the full `time_from_tone_onset_s` vector and full `photostim_on` vector matched direct raw recomputation exactly (`np.allclose == True`, max abs diff `0.0`).
   - Output check: for the same session/trial, the full output matrix (`choice`, `outcome`, `early_lick`, `tongue_y_bin`) matched direct raw recomputation exactly (`np.allclose == True`, max abs diff `0.0`).
   - Offset-session check: repeated the same neural/input/output comparisons for `sub-440956_ses-20190208T133600_behavior+ecephys+ogen`; all checks passed exactly, and the raw recomputation confirmed the selected trial block starts at behavioral trial index 1 (0-based), not at the beginning of the full trial table.
   - Large-offset edge-case check: repeated the same neural/input/output comparisons for `sub-455219_ses-20190807T134913_behavior+ecephys+ogen`; all checks passed exactly, and the raw recomputation confirmed the selected trial block starts at behavioral trial index 126 (0-based), demonstrating that the previous “first N trials” assumption was incorrect.
3. **Reference code comparison**:
   - Data loading: still matches the reference intent of using QC-approved good units (`classification == good`) and exact CCF labels.
   - Neuron/trial filtering: we intentionally do **not** apply `get_regular_trial_mask` wholesale because early lick and outcome are decoder targets, but we now enforce a stricter raw-data validity rule that the full go-aligned neural window must exist in the raw observation interval.
   - Temporal alignment: still go-cue aligned, matching the papers and preprocessing code.
   - Binning: still 50 ms non-overlapping bins by explicit task requirement, replacing the reference 40 ms / 3.4 ms preprocessing and 200 ms / 10 ms published decoder windows.
   - Input construction: tone/sample timing and photostim timing are still taken from the raw behavioral timing streams and trial table, consistent with the reference variables.
   - Output construction: choice is inferred from lick timing; outcome and early lick come directly from raw trial labels; tongue uses last-frame-within-bin alignment, which matches the marker-alignment convention in the reference code.
4. **Key statistics comparison**:
   - Sessions and subjects match exactly after excluding the one zero-good-unit session: 173 sessions, 28 subjects.
   - Good-unit count matches the QC-filtered NWB release exactly: 69,453.
   - The paper’s 476 trials/session statistic is not directly comparable to the converted dataset because the paper reports behavior-level session statistics, whereas the converter now retains only trials with a fully supported raw neural decoding window. For reference, raw NWB no-early trials average 477.47/session, which reproduces the paper’s behavior-level statistic closely.
5. **Edge-case / off-by-one review**:
   - Surveying raw NWB files showed 86 kept sessions with a nonzero behavioral-trial start offset for the valid neural window and 120 kept sessions where the number of trials with full neural-window support did not equal `units/is_good_trials.shape[1]`.
   - This demonstrated that the earlier “behavioral trial prefix” interpretation was wrong and justified the obs-interval-based trial selection fix.

### Issues Found and Resolved
- **Issue**: The original converter treated the first `units/is_good_trials.shape[1]` behavioral trials as the ephys-backed trial set.
  **Resolution**: Replaced that assumption with raw go-time plus `obs_intervals` selection of trials whose full `[-2.5, +1.5] s` window is supported by the session recording interval; use `is_good_trials` directly only when its column count matches the selected trials.
- **Issue**: Raw-data review showed that this bug was not rare; many sessions had offset or shortened valid-trial blocks.
  **Resolution**: Added explicit raw-data sanity checks on a normal session, a short offset session, and a large-offset session. All converted neural/input/output spot checks matched raw recomputation exactly after the fix.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| choice | 0.7053 | 0.6037 | Above chance; qualitatively consistent with published choice decoding being above chance |
| outcome | 0.6369 | 0.5310 | Strongest categorical task-state signal after hit/miss/ignore discretization |
| early_lick | 0.7797 | 0.6848 | Above chance with moderate class imbalance |
| tongue_y_bin | 0.4922 | 0.4470 | Above 3-class chance; time-varying motor output is decodable |

Additional outputs:
- `sample_trials.png` and `predictions.png` were generated by `train_decoder.py --plot-samples`.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| choice | 0.6037 validation balanced accuracy | Papers report above-chance choice decoding, but do not tabulate an exact directly comparable value for this 50 ms-bin / all-session / all-region decoder |
| outcome | 0.5310 validation balanced accuracy | No directly comparable paper decoder found for this exact categorical outcome target |
| early_lick | 0.6848 validation balanced accuracy | No directly comparable paper decoder found for this exact binary early-lick target |
| tongue_y_bin | 0.4470 validation balanced accuracy | Method paper shows strong movement-related predictability from video/markers, but not this exact discretized tongue-y decoder |

Accuracy vs chance review:
- Choice: chance `0.5000`, achieved `0.6037` (`1.21x` chance).
- Outcome: chance `0.3333`, achieved `0.5310` (`1.59x` chance).
- Early lick: chance `0.5000`, achieved `0.6848` (`1.37x` chance).
- Tongue y bin: chance `0.3333`, achieved `0.4470` (`1.34x` chance).

Train vs validation gap review:
- Choice train/val ratio: `1.17`
- Outcome train/val ratio: `1.20`
- Early lick train/val ratio: `1.14`
- Tongue y bin train/val ratio: `1.10`
- No output exceeded the `>1.5x` train-vs-validation overfitting threshold.

Additional investigation for the lower-margin outputs (`choice`, `early_lick`, `tongue_y_bin`):
- Raw-label spot checks already passed exactly for three sessions, including an offset-trial session and a large-offset session.
- Class balance is acceptable for all outputs; none are near the “99% one class” failure mode.
- The saved processing plots and the successful raw-data sanity checks support correct temporal alignment of neural/activity/task streams after the Step 10 fix.
- The papers do not provide exact decoder accuracies for these four targets under the same task definition, trial-retention rules, and binning. The only directly comparable qualitative benchmark is choice decoding above chance, which this dataset satisfies.

### Issues Found and Resolved
- **Issue**: Choice / early-lick / tongue-y validation accuracy is below `1.5x` chance, which required investigation under the workflow.
  **Resolution**: Re-checked raw labels, alignment, class balance, and train-vs-validation gap. No conversion bug was found after the Step 10 fix; retained the current conversion because it is raw-data-consistent and all outputs remain above chance with modest train/val gaps.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
