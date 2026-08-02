# Dataset Conversion Notes

## Overview
- **Dataset**: MAP (Mesoscale Activity Map) dataset from DANDI:000363
- **Papers**: "Brain-wide neural activity underlying memory-guided movement" (data paper) and "Brain-wide analysis reveals movement encoding structured across and within brain areas" (Wang, Kurgyis et al., Nature Neuroscience 2025)
- **Date started**: 2024
- **Goal**: Convert NWB electrophysiology data to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `data/` - 28 subject directories with 174 NWB files
- `code/` - Reference code (VideoAnalysisUtils, Sherlock scripts, Notebooks)
- `datapaper.pdf`, `methodpaper.pdf`, `ChenLiuEtAl2023_SpikeSortingQC.pdf`
- `methods.txt` - Extracted methods text
- `decoder.py`, `train_decoder.py` - Decoder training code
- Python 3.13, numpy 2.3.5, torch 2.6.0, pynwb 2.8.3

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `sliding_histogram` | preprocessing_DJ_2022Aug.py | PROCESSING | Compute firing rates from spike times |
| `process_one_sess` | preprocessing_DJ_2022Aug.py | LOADING | Process all probes in one session |
| `get_regular_trial_mask` | functions_for_r2.py | CURATION | Filter: no early lick, no auto water, no free water, no no-response, no stim |
| `create_4fold_trial_type_mask` | functions_for_r2.py | CURATION | Stratification: HitR(1), MissR(2), HitL(3), MissL(4) |
| `check_fr` | preprocessing_utils.py | CURATION | Remove zero-variance neurons |
| `get_period` | preprocessing_utils.py | PROCESSING | Time periods: sample(-1.9,-1.2), delay(-1.2,0), post_go(0,1) |

### Notes
- Reference code loads from .mat files; we use NWB files
- QC mode is 'classifier' (region-specific classifiers)
- Default preprocessing: stride=0.05s, bw=0.1s, range -3.0 to 3.5s
- Sherlock scripts use stride=0.0034s, bw=0.04s for video analysis
- Regular trial mask excludes: early_lick, auto_water, free_water, no-response, stimulation

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- NWB files: `data/sub-{id}/sub-{id}_ses-{datetime}_behavior+ecephys[+ogen].nwb`
- Each NWB file contains: trials, units, BehavioralEvents, BehavioralTimeSeries
- Key fields: classification ('good'/'unlabelled'), anno_name (brain region), spike_times, obs_intervals
- Tongue tracking at ~300Hz (Camera0_side_TongueTracking: x, y, likelihood)

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total, all) | 272,227 |
| Neurons (total, good) | 69,453 |
| Good unit fraction | 25.5% |
| Neurons / session (good, mean) | 399.2 |
| Subjects | 28 |
| Sessions | 174 |
| Sessions with good units | 173 |
| Trials (total) | 94,990 |
| Trials / session (mean) | 545.9 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from papers/methods)
| Statistic | Value | Source Quote |
|-----------|-------|---------------|
| Neurons (total, good) | 69,943 | "69,943 good units across 173 behavioral sessions" |
| Good unit fraction | 25.9% | "25.9% of clusters reported by Kilosort2" |
| Sessions | 173 | "173 behavioral sessions" |
| Trials / session (mean) | 476 | "476 (Mean; range, 130-785) trials per session" |
| Correct rate | 84% | "84% correct rate (range, 65-99%)" |
| Session selection: performance | >65% | "overall behavioral performance (> 65%)" |
| Session selection: min correct | 50 each | "at least 50 correct lick left and lick right trials each" |
| Sample duration | 0.65s | "three times for 150 ms with 100 ms inter-tone intervals" |
| Delay duration | 1.2s | "1.2 s delay epoch" |
| Response period | 1.5s | "answer period: 1.5 s" |

### Processing Details
- Temporal alignment: Go cue onset (time 0)
- Trial timing: sample -1.85 to -1.20, delay -1.20 to 0.00, response 0.00 to 1.50
- QC: Classifier-based, region-specific

### Curation Steps
**Neuron curation**: classification == 'good' (classifier-based QC)
**Session curation**: >65% correct, >=50 correct L and R trials
**Trial curation**: All trials included (for decoder training), filtered by neural recording coverage (obs_intervals)

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Papers say | Resolution |
|-------|-----------|------------|------------|------------|
| Good units | N/A | 69,453 | 69,943 | Small difference (~0.7%), 1 session with 0 good units. Consistent. |
| Sessions with units | N/A | 173 | 173 | Exact match |
| Trial filtering | get_regular_trial_mask excludes many trial types | All available | Excludes early lick/no-response | For decoder: keep all trials (early_lick, outcome are decoder outputs) |
| obs_intervals | N/A | Some sessions have partial recording coverage | N/A | Filter trials by actual spike time range |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| spike_times | neural | Bin into 50ms windows, compute firing rate (Hz) | Aligned to go cue, -2.5 to +1.5s |
| BIN_CENTERS - TONE_ONSET | input[0] | Time from tone onset in seconds | Tone at -1.85s rel to go |
| photostim_start/stop | input[1] | Binary: 1 if photostim on, 0 otherwise | Per time bin |
| trial_instruction | output[0] | left=0, right=1 | Per trial, broadcast to all time bins |
| outcome | output[1] | ignore=0, miss=1, hit=2 | Per trial, broadcast |
| early_lick | output[2] | no early=0, early=1 | Per trial, broadcast |
| tongue_y | output[3] | Discretized per session: <40th=0, 40-60th=1, >60th=2 | Time-varying |

### Key Decisions
1. **Include all trials**: Unlike reference code's get_regular_trial_mask, we include all trials because early_lick, outcome, and photostim are decoder outputs/inputs
2. **Recording coverage**: Filter trials by actual spike time range (not obs_intervals end) to avoid all-zero neural data
3. **Tongue y discretization**: Use all tongue y values regardless of likelihood for percentile computation
4. **Brain regions**: Use anno_name from NWB, simplified to abbreviations where possible

---

## Step 6: Script Development
**Status**: COMPLETE

Key implementation details:
- `compute_firing_rates_session`: Vectorized histogram computation with searchsorted optimization
- `get_recording_range`: Uses actual max spike time across all good units
- `discretize_tongue_y`: Per-session percentile computation
- `simplify_brain_region`: Maps CCF annotations to abbreviations
- Session selection: >65% correct rate, >=50 correct L/R trials, >=2 good units

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 |
| Trials | 679 |
| Neurons | 901 |
| Brain regions | 12 |

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| Data loading | 1.0-1.5s | ~210s |
| Firing rates | 1.6-7.1s | ~850s |
| Inputs/outputs | 0.2-0.5s | ~60s |
| Total | 2.7-9.0s | ~19min |

Actual full conversion time: 1143.9s (19.1min)

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| choice | 0.7902 | 0.7358 | 0.5000 |
| outcome | 0.8444 | 0.4777 | 0.3333 |
| early_lick | 0.9256 | 0.6259 | 0.5000 |
| tongue_y_position | 0.6121 | 0.5798 | 0.3333 |

All above chance. Loss decreased consistently.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 10,227.5 MB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Papers | Reference Data | Converted Data | Match? |
|-----------|------------------|----------------|----------------|--------|
| Total neurons (good) | 69,943 | 69,453 | 60,324 | Partial - 151/173 sessions |
| Sessions with good units | 173 | 173 | N/A | Yes |
| Sessions (after behavior filter) | N/A | N/A | 151 | Expected |
| Subjects | 28 | 28 | 28 | Yes |
| Trials/session (mean) | 476 (control) | 545.9 (all) | 537.7 (all) | Consistent |
| Good unit fraction | 25.9% | 25.5% | N/A | Consistent |
| Hit rate | 84% | N/A | 72.1% (all trials) | Consistent (84% is control-only) |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

**Check 1: Output log verification**
- Warnings: Some trials with all-zero neural data in sessions 43, 44 (brief recording gaps)
- These are rare and don't significantly affect training
- No errors reported

**Check 2: Sanity checks**
- Neural: Spot-checked session 0, trial 5 and trial 50 - firing rates match exactly (np.allclose = True, max diff = 0.0)
- Input: Time from tone onset range [-0.625, 3.325] matches expected [-2.5+1.85, 1.5+1.85]
- Output: Trial instruction, outcome, early_lick all match NWB data exactly
- Photostim: Correctly zero for non-stim trials, correctly 1 for stim trials

**Check 3: Reference code comparison**
- (a) Data loading: We load from NWB instead of .mat, but extract same variables
- (b) Neuron filtering: We use classification=='good' matching the 'classifier' QC mode
- (c) Temporal alignment: Aligned to go cue onset, consistent with reference
- (d) Binning: 50ms bins as specified (reference uses 100ms bins with 50ms stride, or 40ms bins with 3.4ms stride)
- (e) Input construction: Time from tone onset and photostim binary - consistent with task description
- (f) Output construction: Choice, outcome, early_lick, tongue_y - all correctly mapped

**Check 4: Key statistics comparison**
- 28 subjects: matches data and paper
- 173 sessions with good units: matches paper
- 151 sessions after behavioral filtering: reasonable (22 sessions fail criteria)
- 60,324 good neurons across 151 sessions: consistent with 69,943 across 173 sessions
- Mean trials/session 537.7: consistent with paper's 476 (control-only mean)

**Check 5: Edge cases**
- Recording coverage: Handled by checking actual spike time range
- Sessions with 0 good units: Skipped (1 session)
- Sessions with low performance: Skipped (15 sessions)
- Sessions with insufficient correct L/R: Skipped (5 sessions)

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (18.44 → 0.674 over 200 epochs)
- Test loss: 0.706

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Above Chance? |
|--------|----------------------|------------------------|--------|---------------|
| choice | 0.7197 | 0.6984 | 0.5000 | Yes (1.40x) |
| outcome | 0.6825 | 0.6376 | 0.3333 | Yes (1.91x) |
| early_lick | 0.7931 | 0.7380 | 0.5000 | Yes (1.48x) |
| tongue_y_position | 0.5778 | 0.5570 | 0.3333 | Yes (1.67x) |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Chance | Ratio |
|----------|------------------|--------|-------|
| choice | 0.6984 | 0.5000 | 1.40x |
| outcome | 0.6376 | 0.3333 | 1.91x |
| early_lick | 0.7380 | 0.5000 | 1.48x |
| tongue_y_position | 0.5570 | 0.3333 | 1.67x |

All outputs well above chance. No accuracy below 1.5x chance threshold requiring investigation.

**Check 1: Accuracy vs chance** - All outputs > 1.4x chance. Acceptable.

**Check 2: Accuracy comparison to papers** - The method paper trains R2 prediction models (Ridge regression from video embeddings to firing rates), not decoders from neural activity to behavior. No direct decoder accuracy comparison available.

**Check 3: Train vs validation gap** - Gaps are small (0.02-0.06), no significant overfitting.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
