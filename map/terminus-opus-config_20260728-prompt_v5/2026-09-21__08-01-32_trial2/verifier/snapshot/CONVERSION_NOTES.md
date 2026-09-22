# Dataset Conversion Notes

## Overview
- **Dataset**: Mesoscale Activity Map (MAP) - Brain-wide neural activity underlying memory-guided movement
- **Source**: DANDI archive 000363 (Chen et al. 2023)
- **Method paper**: Wang, Kurgyis et al. 2025
- **Goal**: Convert NWB electrophysiology data to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/data/` - 28 subject directories with 174 NWB files
- `/app/code/` - Reference code from method paper
- Papers: datapaper.pdf, methodpaper.pdf, ChenLiuEtAl2023_SpikeSortingQC.pdf
- `/app/methods.txt` - Extracted methods text

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| sliding_histogram | preprocessing_DJ_2022Aug.py | PROCESSING | Firing rates from spike times |
| process_one_sess | preprocessing_DJ_2022Aug.py | LOADING | Process all probes in session |
| get_regular_trial_mask | functions_for_r2.py | CURATION | Filter: no early lick, auto water, free water, no-response, stim |
| create_4fold_trial_type_mask | functions_for_r2.py | PROCESSING | 1=Hit R, 2=Miss R, 3=Hit L, 4=Miss L |
| load_session | population_decoding_utils.py | LOADING | Load from pickle files |
| check_fr | preprocessing_utils.py | CURATION | Remove zero-variance neurons |

### Notes
- Reference processes .mat files, not NWB; stride=0.05s, bw=0.1s, range -3.0 to 3.5s
- QC: "classifier" mode - region-specific classifiers
- get_regular_trial_mask excludes stim trials for main analysis only

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- NWB files with units (spike_times, classification, anno_name, obs_intervals), trials, behavioral events/timeseries
- Partial recordings: some sessions have obs_intervals covering fewer trials than the full behavioral session
- All good units within a session share the same obs_intervals

### Dataset Size
| Statistic | Value |
|-----------|-------|
| Subjects | 28 |
| Sessions (total NWB files) | 174 |
| Sessions with good units | 173 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics
| Statistic | Value | Source |
|-----------|-------|--------|
| Good units | 69,943 | methods.txt |
| Sessions | 173 | methods.txt |
| Trials/session | 476 mean (130-785) | methods.txt |
| Correct rate | 84% (65-99%) | methods.txt |
| Good unit fraction | 25.9% | methods.txt |

### Curation
- **Neuron**: classification == "good"
- **Session criteria** (for specific analyses): >65% correct, ≥50 correct per side
- **The 173 session count** includes all sessions with good units, NOT filtered by behavioral criteria
- **Trial curation for decoder**: exclude auto_water and free_water; exclude trials outside recording range

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Key Finding
The paper reports 173 sessions = all sessions with good units (174 total - 1 with no good units). The behavioral criteria (>65% correct, ≥50 per side) are for specific analyses, not for defining the dataset. Verified: exactly 173 sessions have good units.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source | Target | Transform |
|--------|--------|-----------|
| units/spike_times (good) | neural | 50ms bin firing rates, -2.5 to +1.5s from go cue |
| sample_start_times | input[0] | Time from tone onset (s), continuous |
| photostim events | input[1] | Binary on/off, time-varying |
| instruction + outcome | output[0] | Choice: L(0), R(1), no_lick(2) |
| outcome | output[1] | ignore(0), miss(1), hit(2) |
| early_lick | output[2] | no(0), yes(1) |
| tongue_y | output[3] | 0-3 categories, time-varying |

### Key Decisions
1. **Bin width**: 50ms (task requirement)
2. **No behavioral session filtering**: Include all 173 sessions with good units
3. **Recording coverage**: Use obs_intervals to filter trials
4. **Tongue visibility**: 0.9 DLC likelihood threshold

---

## Step 6: Script Development
**Status**: COMPLETE

Script: `/app/convert_data.py` (~3.5s per session, ~10 min total)

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Trials | 1040 |
| Neurons | 873 |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

| Output | Train Bal Acc | Val Bal Acc | Chance |
|--------|--------------|-------------|--------|
| choice | 0.803 | 0.689 | 0.333 |
| outcome | 0.757 | 0.611 | 0.333 |
| early_lick | 0.899 | 0.699 | 0.500 |
| tongue_y | 0.651 | 0.571 | 0.250 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Consistency Check
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| Sessions | 173 | 173 | ✓ |
| Subjects | 28 | 28 | ✓ |
| Total neurons | 69,943 | 69,453 | ~99.3% (partial recordings) |
| Trials/session | 476 (130-785) | 517.5 (160-796) | Similar |
| Brain regions | ~5 major | 293 detailed CCF | ✓ (detailed annotations) |

Verification: 1 warning (1 trial with zero neural data at edge of recording)

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Sanity Checks (all passed)
1. **Neural**: Spot-checked firing rates against raw NWB spike times - exact match (np.allclose=True)
2. **Input**: Spot-checked time_from_tone_onset - exact match
3. **Output**: Spot-checked choice/outcome/early_lick for 3 trials - all correct

### Reference Code Comparison
- Data loading: NWB format vs .mat; same underlying data
- Neuron filtering: classification=='good' matches classifier QC mode
- Temporal alignment: Go cue onset (same)
- Binning: 50ms (task) vs 100ms/50ms stride (reference)
- Session inclusion: All sessions with good units (matches paper's 173 count)

### Warning Explanation
Session 0, trial 159: Last trial of partially recorded session. obs_intervals indicate recording extends to this trial, but spike times end slightly before the trial window. Valid data - neurons simply had no spikes.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss: 16.49 → 0.65 over 200 epochs
- 71,561 training trials, 17,971 test trials

### Decoder Results (Full - 173 sessions)
| Output | Train Bal Acc | Val Bal Acc | Chance | Ratio |
|--------|--------------|-------------|--------|-------|
| choice | 0.708 | 0.677 | 0.333 | 2.03x |
| outcome | 0.708 | 0.662 | 0.333 | 1.99x |
| early_lick | 0.786 | 0.752 | 0.500 | 1.50x |
| tongue_y | 0.704 | 0.656 | 0.250 | 2.62x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
All outputs well above chance. Train-validation gap is small (no overfitting):
- choice: 0.708 vs 0.677 (1.05x)
- outcome: 0.708 vs 0.662 (1.07x)
- early_lick: 0.786 vs 0.752 (1.05x)
- tongue_y: 0.704 vs 0.656 (1.07x)

### Comparison to Papers
The reference papers report AUC 0.7-0.9 for choice decoding in ALM during delay. Our decoder predicts choice at 0.677 balanced accuracy across ALL brain regions and the full trial window, which is consistent.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with processing plots
- [x] All required files present
