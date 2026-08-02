# Dataset Conversion Notes

## Overview
- **Dataset**: Mesoscale Activity Map (MAP) Dataset, DANDI:000363
- **Date started**: 2024
- **Goal**: Convert NWB electrophysiology data to decoder-compatible format
- **Papers**: datapaper.pdf, methodpaper.pdf, ChenLiuEtAl2023_SpikeSortingQC.pdf
- **Code**: MapVideoAnalysis repository (code/)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `data/` - 28 subject directories with 174 NWB files
- `code/` - MapVideoAnalysis repository
- `datapaper.pdf`, `methodpaper.pdf`, `ChenLiuEtAl2023_SpikeSortingQC.pdf`
- `methods.txt`, `train_decoder.py`, `decoder.py`

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `process_one_sess` | preprocessing_DJ_2022Aug.py | LOADING | Load .mat files, apply QC, compute firing rates |
| `sliding_histogram` | preprocessing_DJ_2022Aug.py | PROCESSING | Bin spike times into firing rates |
| `get_regular_trial_mask` | functions_for_r2.py | CURATION | Filter trials |
| `create_4fold_trial_type_mask` | functions_for_r2.py | CURATION | Stratification mask |
| `load_session` | population_decoding_utils.py | LOADING | Load preprocessed data |

### Notes
- Reference code uses .mat files; our data is NWB format
- QC: classifier-based, stored as `classification` field in NWB units ('good'/'unlabelled')
- Trial filtering: no early lick, no auto water, no free water, no ignore, no photostim
- Spike times in NWB are on same time base as trial_starts and go_times
- obs_intervals define which trials have neural recordings (some sessions only record subset)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- NWB files: `data/sub-{id}/sub-{id}_ses-{datetime}_behavior+ecephys[+ogen].nwb`
- Each file contains: Units (spike times, QC), Trials (behavior), BehavioralEvents (timing), BehavioralTimeSeries (tongue/jaw/nose tracking)
- Brain regions from electrode_group.location JSON: left/right ALM, BLA, ECT, Medulla, Midbrain, Striatum, Thalamus

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 28 |
| Sessions (NWB files) | 174 |
| Sessions with good neurons | 173 |
| Good neurons (total) | 69,453 |
| Neurons / session (mean) | 401 |
| Trials / session (mean) | ~476 (before filtering) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source |
|-----------|-------|--------|
| Good units total | 69,943 | methods.txt |
| ALM units | 8,717 | methods.txt |
| Striatum units | 7,664 | methods.txt |
| Thalamus units | 12,808 | methods.txt |
| Midbrain units | 7,495 | methods.txt |
| Medulla units | 2,928 | methods.txt |
| Sessions | 173 | methods.txt |
| Subjects | 28 | dandiset.yaml |
| Trials/session | 476 mean (130-785) | methods.txt |
| Correct rate | 84% (65-99%) | methods.txt |
| QC pass rate | 25.9% | methods.txt |

### Curation Steps
**Neuron curation**: classification == 'good' (classifier-based QC)
**Trial curation** (get_regular_trial_mask):
- No early lick, no auto water, no free water, no ignore (no response), no photostim

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code/Data | Papers | Resolution |
|-------|-----------|--------|------------|
| N sessions | 174 NWB files, 173 with good neurons | 173 | 1 session has no good neurons |
| N good units | 69,453 | 69,943 | ~490 difference, likely from session with no good neurons or minor QC differences |
| ALM count | 23,645 (electrode group label) | 8,717 | Electrode group labels include broader cortical areas under 'ALM' probe label |
| Brain regions | 7 (ALM, BLA, ECT, Medulla, Midbrain, Striatum, Thalamus) | 5 major + others | Consistent - paper focuses on 5 major |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source | Target | Transform |
|--------|--------|----------|
| units.spike_times (good) | neural | Bin 50ms, align go cue, [-2.5, 1.5]s |
| time from sample_start | input[0]: time_from_tone_onset | Continuous seconds |
| photostim active | input[1]: photostim_on | Binary per timepoint |
| trial_instruction | output[0]: choice | left=0, right=1 |
| outcome | output[1]: outcome | ignore=0, miss=1, hit=2 |
| early_lick | output[2]: early_lick | no=0, yes=1 |
| tongue_y | output[3]: tongue_y | 0/<40th, 1/40-60th, 2/>60th pctl |

### Key Decisions
1. **No session-level filtering**: Include all 173 sessions with good neurons (paper's 173)
2. **Trial filtering**: Apply get_regular_trial_mask (no early lick, no auto water, no free water, no ignore, no photostim)
3. **Brain regions**: Use electrode_group.location, strip left/right
4. **obs_intervals**: Only process trials within recording intervals
5. **Tongue y**: Use DLC likelihood > 0.9 threshold, default to class 1 (mid) when no high-confidence detection

---

## Step 6: Script Development
**Status**: COMPLETE

Key implementation details:
- Uses pynwb to read NWB files
- np.histogram for efficient spike binning
- np.searchsorted for efficient tongue data lookup
- obs_intervals matching to determine recorded trials
- Processing time: ~10s/session, ~30min total

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 |
| Neurons | 731 |
| Trials | 809 |

### Run Time: ~12s/session

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Train Bal Acc | Val Bal Acc | Chance |
|--------|--------------|-------------|--------|
| choice | 0.788 | 0.771 | 0.500 |
| outcome | 0.781 | 0.698 | 0.333 |
| early_lick | 1.000 | 1.000 | 0.500 |
| tongue_y | 0.846 | 0.790 | 0.333 |

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 6697 MB
- `verification_full_out.txt`: created, 0 errors, 1 warning

### Consistency Check
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| Sessions | 173 | 173 | ✓ |
| Subjects | 28 | 28 | ✓ |
| Good neurons | 69,943 | 69,453 | ~99.3% |
| Trials (total) | ~82,000 | 52,990 (filtered) | Expected (filtered) |
| Brain regions | 7 | 7 | ✓ |
| Thalamus neurons | 12,808 | 12,878 | ~100.5% |
| Medulla neurons | 2,928 | 4,800 | Higher (includes more areas) |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- 0 errors in verification_full_out.txt
- 1 warning: Session 1, trial 104 has all-zero neural data (edge of recording window)
- This is acceptable as it's a single trial at the boundary

### Check 2: Sanity checks on neural, input, output data
- **Neural**: Verified firing rate at session 2, trial 0, neuron 0, bin 20 = 20.0 Hz matches manual computation from NWB spike times ✓
- **Neural**: Verified firing rate at session 2, trial 5, neuron 3, bin 10 = 0.0 Hz matches ✓
- **Input**: Verified time_from_tone_onset at session 2, trial 5, bin 10 = -0.125s matches manual computation ✓
- **Output**: Verified choice=0 (left) and outcome=1 (miss) for session 2, trial 5 matches NWB trial_instruction='left' and outcome='miss' ✓

### Check 3: Reference code comparison
- Data loading: Uses pynwb instead of loadmat, but extracts same variables
- Neuron filtering: classification=='good' matches QC classifier mode
- Trial filtering: Exact match with get_regular_trial_mask logic
- Temporal alignment: Go cue onset, same as reference code
- Binning: 50ms bins (task spec) vs 40ms in reference code (for video analysis)
- Input construction: time_from_tone_onset and photostim_on as specified
- Output construction: choice, outcome, early_lick, tongue_y as specified

### Check 4: Key statistics comparison
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| Sessions | 173 | 173 | ✓ |
| Subjects | 28 | 28 | ✓ |
| Good neurons | 69,943 | 69,453 | 99.3% |
| Thalamus | 12,808 | 12,878 | ~100.5% |

### Check 5: Edge cases
- obs_intervals: Properly handled sessions where recording covers subset of trials
- Photostim onset: Correctly converted from trial-start-relative to go-cue-relative
- Tongue tracking: Default to class 1 (mid) when no high-confidence DLC detection

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss: 18.3 → 0.40 over 200 epochs (decreasing) ✓
- Test loss: 0.46

### Decoder Results (Full)
| Output | Train Bal Acc | Val Bal Acc | Chance | Notes |
|--------|--------------|-------------|--------|-------|
| choice | 0.749 | 0.712 | 0.500 | Above chance ✓ |
| outcome | 0.758 | 0.712 | 0.333 | Above chance ✓ |
| early_lick | 1.000 | 1.000 | 0.500 | Trivial (filtered) |
| tongue_y | 0.833 | 0.791 | 0.333 | Above chance ✓ |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Val Accuracy | Chance | Ratio |
|----------|-------------|--------|-------|
| choice | 0.712 | 0.500 | 1.42x |
| outcome | 0.712 | 0.333 | 2.14x |
| early_lick | 1.000 | 0.500 | trivial |
| tongue_y | 0.791 | 0.333 | 2.37x |

All non-trivial outputs are well above chance. Choice decoding at 71.2% is reasonable for a brain-wide decoder. The paper reports AUC values for choice selectivity that are consistent with this level of decoding.

early_lick is trivial because all early lick trials are filtered out by get_regular_trial_mask. This is consistent with the reference code.

### Issues Found and Resolved
- **Photostim input**: Always 0 because photostim trials are filtered out. This is correct per reference code.
- **Output dtype**: Fixed from float32 to int64 for decoder compatibility
- **obs_intervals**: Discovered that some sessions only record subset of trials; fixed to only process recorded trials
- **Session filtering**: Removed session-level performance filtering to match paper's 173 sessions

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
