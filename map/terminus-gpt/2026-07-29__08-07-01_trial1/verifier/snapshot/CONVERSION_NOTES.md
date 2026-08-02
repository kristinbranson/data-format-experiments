# Dataset Conversion Notes

## Overview
- **Dataset**: [Name and source]
- **Date started**: 2026-07-29
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
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
| get_regular_trial_mask | code/Archive/neural_and_embedding_data_functions.py | CURATION | Selects regular trials by excluding early lick, auto water, free water, and no-response trials |
| align_markers_between_lims | code/Archive/getting_started.ipynb | PROCESSING | Aligns marker/embedding data to go cue within specified time limits |
| [alignment helper for ephys + embedding] | code/Archive/neural_and_embedding_data_functions.py | PROCESSING | Crops/aligned ephys firing rates and embedding vectors onto a shared time base |

### Notes
- Reference code contains explicit go-cue-centered alignment for marker/embedding data.
- Trial curation in reference code defines “regular trials” as excluding early lick, auto water, free water, and no-response trials.
- Tracking variables visible in code include tongue_x/tongue_y, jaw, nose, and whisker markers.
- Need to inspect data files next to determine how these variables are stored natively and which sessions/files are available.


---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- Data are organized as per-subject directories under `data/`, containing NWB files named like `sub-<mouse>_ses-<timestamp>_behavior+ecephys+ogen.nwb`.
- Each NWB session contains a `units` table with spike times and quality metrics, an `intervals/trials` table with trial-wise metadata, and acquisition groups including `BehavioralEvents` and `BehavioralTimeSeries`.
- Behavioral event streams observed in NWB include go cue, photostim, left/right lick, presample/sample/delay, and trial end timing.
- Tracking/behavioral time series are present in acquisition/processing (to be mapped later), and code references tongue/jaw/nose/whisker markers.
- Example trial table keys include: auto_water, early_lick, free_water, id, outcome, photostim_duration, photostim_onset, photostim_power, start_time, stop_time, task, task_protocol, trial, trial_instruction, trial_uid

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 272227 |
| Neurons / session | 1564.52 |
| Subjects | 28 |
| Sessions / subject | 6.21 |
| Trials (total) | 94990 |
| Trials / session | 545.92 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 69,943 good units | "Overall, the dataset consisted of 69,943 good units recorded across 173 behavioral sessions" |
| Sessions | 173 behavioral sessions | same as above |
| Subjects | 28 mice in provided data | From data directory count; methods excerpt here did not state overall mice count in visible lines |
| Photostim trial fraction | ~25% | "Photoinhibition ... was deployed on a subset of ~25% randomly interleaved trials" |
| Photostim end relative to go cue | Ends before Go cue | "photoinhibition always ended before the 'Go' cue" |
| QC method | region-specific logistic classifiers, good units only | "Applying the trained classifiers ... provided lists of units that were labeled as 'good' and were used for the analysis" |

### Processing Details
- Behavioral sessions are memory-guided movement trials with presample, sample, delay, go, and trial-end epochs.
- Photoinhibition was delivered in a subset of randomly interleaved trials and ended before the go cue.
- Reference analyses use go-cue-centered alignment in code for behavioral/tracking streams.
- Need to verify exact firing-rate binning used in reference code/paper for decoding-style analyses.

### Curation Steps

**Neuron curation rules**:
- Use only units labeled as "good" by the region-specific QC classifier trained on spike sorting quality metrics.
- For specific photoinhibition effect analyses, units with at least 10 photostimulation trials were selected.

**Trial curation rules**:
- Reference code defines regular trials as excluding early lick, auto water, free water, and no-response trials.

### Decoders Trained
| Decoded variable | Accuracy |
| [to extract from methodpaper.pdf] | |


---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Unit counts | Reference analyses use curated units | Raw NWB files contain 272,227 total units across 174 NWB files; `unit_quality == "good"` yields 154,948 units | Methods report 69,943 good units across 173 behavioral sessions | Raw `unit_quality == "good"` yields 154,948 units, but restricting to the major broad regions emphasized in the paper (ALM, Striatum, Thalamus, Midbrain, Medulla) gives a total close to the reported 69,943; likely the paper total excludes additional regions such as ECT/BLA and possibly one extra NWB session. |
- Electrode `location` fields encode JSON with broad `brain_regions` labels (e.g. `left ALM`, `left Striatum`), which can be mapped to units via `units/electrodes` for neuron region labels.

| Trial selection | Reference code defines regular trials excluding early lick, auto water, free water, and no-response trials | Trial table contains `early_lick`, `auto_water`, `free_water`, and `outcome` columns | Methods/code are consistent on excluding these for regular-trial analyses | Implement trial masks consistent with reference analysis where needed |
| Photostim timing | Code aligns to go cue and methods state photostim ends before go cue | NWB contains `photostim_onset` and `photostim_duration` trial fields plus event times | Methods state photoinhibition ended before the Go cue | Build photostim input relative to go cue and verify all stimulation bins are pre-go |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `units/spike_times` for units passing curation; grouped by NWB session | neural | Bin spikes in 50 ms bins from -2.5 s to +1.5 s around each trial's go cue; convert to firing rate or spike count per bin consistently across sessions | go-cue alignment logic in `getting_started.ipynb`; trial curation in `neural_and_embedding_data_functions.py` | Use neuron x time arrays per trial; likely restrict to curated `good` units in broad task regions |
| `sample_start_times` or tone-on event relative to go cue | input[0] | Time-varying representation of time from tone onset within aligned trial window | behavioral event alignment in reference code | Need exact event choice: sample/tone onset is the task tone-related input; represent consistently for all bins |
| `photostim_start_times`/`photostim_stop_times` and/or trial `photostim_onset`/`photostim_duration` | input[1] | Binary time series indicating photostimulation on/off in each 50 ms bin | methods + NWB event streams | Methods state photostim ends before go cue; verify alignment |
| `trial_instruction` | output[0] | Map left=0, right=1 | NWB trial table | Use per-trial categorical output named choice |
| `outcome` | output[1] | Map ignore=0, miss=1, hit=2 | NWB trial table | Per-trial categorical output |
| `early_lick` | output[2] | Map no early=0, early=1 | NWB trial table | Per-trial categorical output |
| `Camera0_side_TongueTracking/data[:,1]` | output[3] | Interpolate/aligned tongue y to trial bins; discretize per session by 40th/60th percentiles into 3 classes | marker alignment logic in `getting_started.ipynb` | Need to confirm column order and handle low-likelihood / missing tracking |

### Key Decisions
1. **Neural curation**: Start from `unit_quality == "good"`; refine by excluding non-paper regions if needed to match reference statistics.
2. **Session unit regions**: Map neuron brain regions from electrode `location` JSON `brain_regions` via `units/electrodes`.
3. **Temporal alignment**: Align every modality to `go_start_times` and extract [-2.5 s, +1.5 s] windows with 50 ms bins.
4. **Trial inclusion**: Keep trials with valid go cue and sufficient aligned data; use trial fields to exclude invalid cases if needed.
5. **Tongue output**: Build a time-varying categorical output from tongue y after session-wise percentile discretization.

### Planned Sanity Checks
- [ ] Check that binned neural activity for a spot-checked neuron/trial matches raw spike times around go cue.
- [ ] Check that photostim binary input matches raw photostim start/stop events for selected trials.
- [ ] Check that trial-level outputs (`trial_instruction`, `outcome`, `early_lick`) exactly match raw NWB trial table values for selected trials.
- [ ] Check that aligned tongue y values at selected bins match interpolated raw tracking timestamps before discretization.


---

## Step 6: Script Development
**Status**: COMPLETE

- Implemented initial `convert_data.py` with NWB loading, unit filtering, go-cue alignment, 50 ms spike binning, input/output construction, and pickle export.
- CLI supports positional output path plus `--full`, `--sample`, and `--show-processing` flags.
- Plot generation for `--show-processing` is not yet implemented and may need follow-up if validation requires it.

Code inefficiencies identified:
[Note]

Code speedups added:
[Note]

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 2236 |
| Neurons / session | 1118.00 |
| Subjects | 1 |
| Sessions / subject | 2 |
| Trials (total) | 679 |
| Trials / session | 339.5 |
| time_from_tone_onset range | [0.0, 5.7] |
| photostim_on range | [0.0, 1.0] |
| choice distribution | [0.473,0.527] |
| outcome distribution | [0.112,0.001,0.887] |

### Processing Plots Review
- No format errors in verification.
- `--show-processing` flag currently does not emit plots yet; this should be addressed later if required.
- Time-from-tone input range corrected after fixing per-trial sample-event matching.

### Run Time Estimates

| Speed-ups Implemented | Time Savings |
| | |

| Step | Time / Session | Estimated Total Time |
| | | |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: sklearn warning on small validation split: `y_pred contains classes not in y_true`

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| choice | 0.8027 | 0.7578 | Above chance, strong decoding |
| outcome | 0.8023 | 0.6093 |
| early_lick | 0.8502 | 0.7539 | Above chance, strong decoding |
| tongue_y | 0.5844 | 0.5456 | Above chance, moderate decoding |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 20G
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Code | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|----------------|--------|
| Total neurons | 69,943 good units in major reported regions | major-region subset implied by code/data | 154,948 `good` units before region restriction; 142,233 in retained major regions | 142,233 | Partial match; broader than paper total |
| Mean neurons/session | ~404 good units/session if using 69,943/173 | N/A | retained sessions vary widely | 817.43 | No; conversion retains more good units than paper total |
| Subjects | 1 | | | | |
| Sessions | 173 behavioral sessions | N/A | 174 NWB files | 174 | Near-match; one extra session/file included |
| Trials (total) | 679 | | | | |
| Trials/session (mean) | N/A | N/A | varies by session | See verification log | |
| <Input 1 range> | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| ... | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | [MIN, MAX] | |
| <Output 1 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |
| <Output 2 distribution> | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | [FRAC0,FRAC1,...] | |

---

## Step 10: Critical Review 1
**Status**: IN PROGRESS

### Checks Performed
1. Verification log check: `verification_full_out.txt` completed with no explicit ERROR/WARNING lines.
2. Raw-to-converted output sanity check: first valid sample session/trial trial-level outputs matched raw NWB `trial_instruction`, `outcome`, and `early_lick`.
3. Raw-to-converted input sanity check: first valid sample session/trial photostim binary series and time-from-tone series matched direct reconstruction from raw NWB event timestamps using `np.allclose()`.

### Issues Found and Resolved
- Time-from-tone input was initially incorrect because `sample_start_times` is a global event stream, not trial-indexed; fixed by selecting the last sample event within each trial before go cue.
- Full conversion runtime was initially too slow; optimized spike binning using `searchsorted` + `bincount`, reducing sample conversion time from ~30.6 s to ~11.0 s for 2 sessions.
- Remaining discrepancy: converted dataset includes 174 sessions and more good units than the paper's 173-session / 69,943-unit summary, likely because the NWB release includes extra non-ogen/extra-region sessions beyond the exact analysis subset in the paper.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| choice | 0.8027 | 0.7578 | Above chance, strong decoding |
| outcome | 0.8698 | 0.7366 | Above chance, strong decoding |
| early_lick | 0.8502 | 0.7539 | Above chance, strong decoding |
| tongue_y | 0.5844 | 0.5456 | Above chance, moderate decoding |

---

## Step 12: Critical Review 2
**Status**: IN PROGRESS

### Accuracy Analysis

| Variable | Achieved Accuracy | Expectation from Papers |
| | |

[Analysis of any low accuracies]

### Issues Found and Resolved
- [Issue]: [Resolution]

---

## Step 13: Documentation and Cleanup
**Status**: NOT STARTED

- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
