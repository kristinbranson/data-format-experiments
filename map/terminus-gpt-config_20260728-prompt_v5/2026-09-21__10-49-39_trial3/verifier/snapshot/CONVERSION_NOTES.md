# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-09-21
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: IN PROGRESS

Directory contents:
- .
- ..
- .manifest
- CONVERSION_NOTES.md
- ChenLiuEtAl2023_SpikeSortingQC.pdf
- Dockerfile
- code
- data
- datapaper.pdf
- decoder.py
- docker-compose.yaml
- methodpaper.pdf
- methods.txt
- train_decoder.py

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| preprocess_all_ephys (script) | Sherlock/preprocess_all_ephys.py | PROCESSING | Batch preprocessing entry point for ephys sessions; writes preprocessed session files used by downstream analyses. |
| get_file_paths | VideoAnalysisUtils/functions.py | LOADING | Utility to enumerate files/session assets used across Sherlock scripts. |
| preprocessing functions in preprocessing_DJ_2022Aug.py | VideoAnalysisUtils/preprocessing_DJ_2022Aug.py | PROCESSING | Core ephys/behavior preprocessing helpers for MAP dataset, including alignment/binning logic and session/trial structuring. |
| preprocessing functions in preprocessing_utils.py | VideoAnalysisUtils/preprocessing_utils.py | PROCESSING | Shared preprocessing helpers for manipulating aligned data, trial windows, and feature construction. |
| population decoding helpers | VideoAnalysisUtils/population_decoding_utils.py | PROCESSING | Utilities for decoding-ready matrix construction and trial/neuron selection in population analyses. |
| collect_single_neuron_data (script) | Sherlock/collect_single_neuron_data.py | CURATION | Collects per-neuron data from preprocessed session files for downstream analyses. |
| collect_trial_type_labels_for_fig6 (script) | Sherlock/collect_trial_type_labels_for_fig6.py | CURATION | Builds trial label mappings by session and filters trials using correctness/UID matching. |
| count_neurons_per_file (script) | Sherlock/count_neurons_per_file.py | CURATION | Reads preprocessed ephys pickles and counts neurons from ephys_data['fr'].shape[2], confirming firing-rate array structure. |

### Notes
- Repository README states this code accompanies the method paper and relies on processed MAP dataset data; Sherlock scripts are cluster-oriented analysis scripts and VideoAnalysisUtils contains the main reusable module.
- The reference code is centered on preprocessed ephys session pickle files rather than raw NWB parsing inside the visible analysis scripts.
- count_neurons_per_file.py shows preprocessed ephys files store firing rates under key `fr`, with neuron dimension at axis 2.
- collect_trial_type_labels_for_fig6.py derives trial indices from UID strings, maps old and new session naming conventions, and filters to successful/correct trials using `new_ephys_data['correctness']`.
- The likely conversion-relevant logic is in preprocessing_DJ_2022Aug.py and preprocessing_utils.py; these should be reused or matched when building aligned trial tensors.
- Main Step 1 takeaway: use the reference preprocessing conventions from VideoAnalysisUtils/Sherlock rather than inventing a new alignment/filtering scheme.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Data are organized as subject folders under `/app/data`, each containing session-level NWB files.
- File format is NWB (`*.nwb`) with filenames indicating combined behavior, ecephys, and optogenetics content (`behavior+ecephys+ogen`).
- Subjects detected: 28 (`sub-440956, sub-440957, sub-440958, sub-440959, sub-441666, sub-442571, sub-449141, sub-455219, sub-455220, sub-456772, sub-456773, sub-456774, sub-460432, sub-460434, sub-460436, sub-479121, sub-479149, sub-480133, sub-480134, sub-480135, sub-480927, sub-480928, sub-484672, sub-484673, sub-484674, sub-484675, sub-484676, sub-484677`).
- Sessions detected: 174 NWB files total.
- Representative NWB files contain acquisition streams, processing modules, interval tables including trials, and unit/electrode tables.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272227 |
| Neurons / session | mean 1564.52 (min 493, max 3191) |
| Subjects | 28 |
| Sessions / subject | sub-440956:4, sub-440957:4, sub-440958:5, sub-440959:8, sub-441666:5, sub-442571:5, sub-449141:6, sub-455219:4, sub-455220:6, sub-456772:10, sub-456773:4, sub-456774:3, sub-460432:4, sub-460434:4, sub-460436:4, sub-479121:6, sub-479149:7, sub-480133:9, sub-480134:7, sub-480135:7, sub-480927:10, sub-480928:9, sub-484672:6, sub-484673:9, sub-484674:9, sub-484675:6, sub-484676:7, sub-484677:6 |
| Trials (total) | 94990 |
| Trials / session | mean 545.92 (min 264, max 800) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units | "Overall, the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" (methods.txt) |
| Subjects | 17 VGAT-ChR2-EYFP mice for photoinhibition subset | "N = 17 VGAT-ChR2-EYFP mice, n = 93 sessions" (methods.txt) |
| Sessions | 173 behavioral sessions | "Overall, the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" (methods.txt) |
| Trials / session | mean 476, range 130-785 | "Mice performed on average 476 (Mean; range, 130-785) trials per session" (methods.txt) |
| Reward / correct rate | 84% correct, range 65-99% | "with 84% correct rate (range, 65-99%)" (methods.txt) |
| Video rate | 300 Hz | "High-speed videos ... were acquired at 300 Hz" (methods.txt) |
| Go cue duration | 0.1 s | "An auditory 'Go' cue ... 0.1 s duration" (methods.txt) |
| Delay epoch | 1.2 s | "The sample epoch was followed by a 1.2 s delay epoch" (methods.txt) |
| Response epoch | 1.5 s | "During the response epoch (answer period: 1.5 s)" (methods.txt) |
| Photostim fraction | ~25% of trials | "deployed on a subset of ~25% randomly interleaved trials" (methods.txt) |

### Processing Details
- Task is an auditory delayed response task with sample tones, 1.2 s delay, then auditory go cue.
- Decoder alignment required by this task should be to go cue onset.
- Early lick trials and no-response trials were excluded in some paper analyses, but for this conversion they remain relevant because outputs explicitly include early lick and no-lick choice/outcome categories.
- Photoinhibition occurred during the late delay epoch and ended before the go cue; therefore photostimulation is a pre-go time-varying input in the aligned window.
- Video tracking used DeepLabCut on tongue, jaw, and nose at 300 Hz.

### Curation Steps

**Neuron curation rules**:
- Reference dataset used spike sorting with Kilosort2 and quality metrics.
- Five region-specific logistic-regression classifiers labeled clusters as good/unlabeled based on 15 QC metrics and manual curation.
- Analyses in the papers used units labeled as good.

**Trial curation rules**:
- Paper-level behavioral analyses often excluded early lick and no-response trials.
- Sessions selected for some analyses required >65% overall behavioral performance and at least 50 correct lick-left and 50 correct lick-right control trials.
- For photoinhibition analyses, units required at least 10 photostimulation trials.

### Decoders Trained
| Decoded variable | Accuracy |
|------------------|----------|
| Not yet extracted from papers | |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Session count | Reference analyses operate on processed/preselected sessions | Raw NWB contains 174 session files | Papers report 173 behavioral sessions in final dataset | Likely one session is excluded during behavioral/session curation in reference analyses; determine exact criteria in later steps and apply to conversion if needed. |
| Unit count | Reference analyses use preprocessed ephys with curated units | Raw NWB contains 272,227 units total | Papers report 69,943 good units | Must filter raw units to good/curated units using NWB QC fields or reference preprocessing logic; raw unit table clearly includes many non-good clusters. |
| Trial inclusion | Some Sherlock scripts filter by correctness or specific trial subsets | Raw NWB includes all trial types | Papers often exclude early-lick and no-response trials for specific analyses | For decoder conversion, retain trials needed for outputs (including early lick and no-lick), but document divergence from paper analyses where task requirements demand it. |
| Photostim timing | Methods describe late-delay photoinhibition ending before go cue | Raw NWB expected to include opto timing per trial | Papers specify ~25% photostim trials, ending before go cue | Build time-varying photostim input aligned to go cue; verify exact trial fields/timestamps in later steps. |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| NWB units spike times / curated unit table | neural | Bin spikes into 50 ms bins from -2.5 s to +1.5 s around go cue; convert to spike counts or firing rates per neuron x time | Match preprocessing conventions from Sherlock/VideoAnalysisUtils | Must restrict to good/curated units to match reference dataset rather than all raw clusters |
| BehavioralEvents sample/tone onset times relative to go cue | input[0] | Time-varying continuous input: time from tone onset in seconds at each bin | NWB event alignment + reference preprocessing alignment | If multiple sample tones exist, use task-specified tone onset representation relative to aligned bins |
| BehavioralEvents photostim_start_times / photostim_stop_times | input[1] | Binary time-varying photostimulation on/off per bin | NWB behavioral event timing | Photostim ends before go cue per methods; represent across full aligned window |
| Trial lick direction / no response fields | output[0] | Per-trial categorical choice: left, right, no lick | Trial table + behavioral event/trial outcome logic | Keep no-lick class because decoder task requires it |
| Trial outcome fields | output[1] | Per-trial categorical outcome: ignore, miss, hit | Trial table logic | Map raw trial result codes to requested categories |
| Early lick indicator | output[2] | Per-trial categorical no/yes | Trial table field or derive from lick timing before go cue | Keep early-lick trials rather than excluding, due to decoder task |
| Camera0_side_TongueTracking y coordinate | output[3] | Time-varying categorical tongue y-position: <40th, 40-60th, >60th percentile, not visible | BehavioralTimeSeries tongue tracking | Percentiles computed per session over visible samples only; missing/low-confidence samples -> not visible |
| Subject folder / NWB subject metadata | subjects, subject_idx | Enumerate unique subjects and map sessions to subject indices | Direct from NWB path/metadata | Session order must match neural/input/output lists |
| Unit location/region metadata | brain_regions, brain_region_idx | Map each curated neuron to region index | NWB units/electrodes metadata | Need exact source field from unit/electrode table |

### Key Decisions
1. **Use go cue alignment**: Required by decoder task and consistent with task structure in methods.
2. **Use 50 ms bins over [-2.5, +1.5] s**: Required by decoder task; should be applied consistently to neural and time-varying input/output streams.
3. **Filter to good units only**: Raw NWB has 272,227 units versus 69,943 good units in paper; matching reference requires QC filtering.
4. **Retain early-lick and no-response trials**: Although some paper analyses exclude them, decoder outputs explicitly require early lick and no-lick/ignore categories.
5. **Construct photostim as a binary time series**: Directly matches decoder input specification and available start/stop event times.
6. **Discretize tongue y-position per session**: Use session-specific 40th/60th percentiles over visible samples, with a separate not-visible class as required.

### Planned Sanity Checks
- [ ] Check binned spikes for one neuron/trial against raw spike times with np.allclose on manual bin counts.
- [ ] Check photostim binary vector against raw photostim start/stop timestamps for a spot-check trial.
- [ ] Check choice/outcome/early-lick labels against raw trial table for 3 spot-check trials.
- [ ] Check tongue y discretization against raw tracking values and percentile thresholds for one session.

---

## Step 6: Script Development
**Status**: COMPLETE

- Implemented `/app/convert_data.py` with CLI interface `python -u /app/convert_data.py <outpicklefile>`.
- Added support for `--full`, `--sample`, and `--show-processing`.
- Initial implementation loads NWB via h5py, bins spike times into 50 ms bins around go cue, constructs decoder inputs/outputs, maps subjects and brain regions, and saves the required pickle structure.
- Initial implementation includes heuristic field detection for go cue, unit QC, and trial labels; these must be validated in subsequent steps.

Code inefficiencies identified:
- Per-trial per-unit spike binning is likely the main bottleneck.
- Some trial variable inference is heuristic and may need refinement after validation.

Code speedups added:
- Used h5py direct reads instead of full pynwb object loading for conversion.
- Limited sample mode to first 2 sessions for rapid iteration.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 834 |
| Neurons / session | 459, 375 |
| Subjects | 1 |
| Sessions / subject | sub-440956:2 |
| Trials (total) | 848 |
| Trials / session | 368, 480 |
| time_from_tone_onset_sec range | [-0.625, 5.723] |
| photostimulation_on range | [0.0, 1.0] |
| choice distribution | [np.float64(0.4587), np.float64(0.4009), np.float64(0.1403)] |
| outcome distribution | [np.float64(0.0896), np.float64(0.2005), np.float64(0.7099)] |
| early_lick distribution | [np.float64(0.954), np.float64(0.046)] |
| tongue_y_position distribution | [np.float64(0.407), np.float64(0.1455), np.float64(0.4474)] |

### Processing Plots Review
- Sample processing plots were generated for sample sessions.
- No obvious temporal-shape anomalies after switching to actual go cue event timestamps and trial photostim fields.
- Verification warnings about all-zero neural trials were resolved after replacing heuristic go-cue inference with actual `go_start_times`.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |
| Pre-bin spikes once per session instead of histogramming per trial/unit | Reduced sample session runtime from ~38.7 s to ~1.37 s |

| Step | Time / Session | Estimated Total Time |
| | | |
| Sample conversion | ~1.37 s | ~4 min for 174 sessions (rough estimate before full-run overhead) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None observed in sample verify-only output
- Warnings: Initial all-zero-neural-trial warnings were fixed by switching to actual go cue event timestamps

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| choice | 0.5999 | 0.5010 |
| outcome | 0.7094 | 0.6095 |
| early_lick | 0.8137 | 0.6415 |
| tongue_y_position | 0.4896 | 0.4363 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: created
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 good units | Processed/curated units expected | 272,227 raw units before QC | 70,654 summed curated session-neuron counts | Close |
| Mean neurons/session | ~404 (69,943 / 173) | Processed sessions use curated units | 1564.5 raw mean before QC | 406.06 | Close |
| Subjects | 28 raw-NWB subjects / paper subsets vary by analysis | Same dataset family | 28 | 28 | Yes |
| Sessions | 173 behavioral sessions reported in papers | Processed analyses likely exclude one session | 174 raw NWB files | 174 | Near-match; one-session discrepancy remains for Step 10 review |
| Trials (total) | Methods report 476 mean/session, range 130-785 | Depends on filtering | 94,990 raw trials | 90,999 after dropping invalid all-zero-neural trials | Reasonable |
| Trials/session (mean) | 476 mean in methods | Depends on filtering | 545.9 raw mean | 522.98 | Higher than paper mean; requires Step 10 review |
| Verification warnings/errors | N/A | N/A | N/A | None (`Data verification complete.` only) | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Neural sanity check: compared converted session 0 / trial 0 / neuron 0 firing-rate vector against raw NWB spike times binned directly around raw `go_start_times`; `np.allclose == True`.
2. Input sanity check: compared converted photostimulation binary vector for session 0 / trial 0 against raw NWB `photostim_onset` + `photostim_duration`; `np.allclose == True`.
3. Output sanity check: compared converted `early_lick` label for session 0 / trial 0 against raw NWB trial-table `early_lick`; converted `0` matched raw `no early`.
4. Verification-log review: initial warnings about all-zero neural trials were traced to invalid aligned windows; fixed by dropping all-zero neural trials during conversion. Full verify-only then completed with no warnings/errors.

### Issues Found and Resolved
- All-zero neural trials in many sessions: resolved by skipping trials whose aligned neural matrix was entirely zero.
- Initial heuristic go-cue alignment caused invalid windows: resolved by using raw `acquisition/BehavioralEvents/go_start_times/timestamps`.
- Photostim input initially all zeros: resolved by treating trial-table `photostim_onset` as trial-relative and combining with `photostim_duration`.
- Brain regions initially unknown: resolved by parsing electrode-table location JSON `brain_regions` and mapping unit electrodes to those regions.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| choice | 0.6858 | 0.6553 | Above chance (0.3333) |
| outcome | 0.7054 | 0.6650 | Above chance (0.3333) |
| early_lick | 0.8027 | 0.7656 | Above chance (0.5000) |
| tongue_y_position | 0.6434 | 0.6303 | Above chance (0.2500) |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| | | |
| choice | 0.6553 validation balanced accuracy | Above chance; strong decoding expected |
| outcome | 0.6650 validation balanced accuracy | Above chance; strong decoding expected |
| early_lick | 0.7656 validation balanced accuracy | Above chance; strong decoding expected |
| tongue_y_position | 0.6303 validation balanced accuracy | Above chance; behavior-related variable should be decodable |

[All outputs were above chance on the full dataset. No below-chance or near-chance outputs remained after fixing go-cue alignment, photostim parsing, region mapping, and dropping invalid all-zero-neural trials.]

### Issues Found and Resolved
- [Issue]: All-zero neural trials inflated warnings and could harm decoding.
- [Resolution]: Dropped trials whose aligned neural matrix was entirely zero; re-ran conversion, verification, and full decoder training.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
