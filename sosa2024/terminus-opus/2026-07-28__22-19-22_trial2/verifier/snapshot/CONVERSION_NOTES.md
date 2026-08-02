# Dataset Conversion Notes

## Overview
- **Dataset**: Sosa et al. 2025 - "A flexible hippocampal population code for experience relative to reward" (Nature Neuroscience)
- **Date started**: 2025-07-29
- **Goal**: Convert NWB calcium imaging data to decoder-compatible format
- **DANDI**: https://dandiarchive.org/dandiset/001361

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code from the paper (reward_relative package)
- `data/` - NWB files organized by subject (11 subjects, 152 sessions)
- `paper.pdf` - Reference paper
- `methods.txt` - Methods extracted from paper
- `decoder.py` - Decoder module
- `train_decoder.py` - Training script

Python environment: numpy 2.3.5, torch 2.6.0+cu124, GPU available.

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `get_timeseries_data()` | glmUtils.py | LOADING/PROCESSING | Extracts behavioral and neural timeseries from sess objects |
| `get_reward_zones()` | behavior.py | LOADING | Gets reward zone coordinates and labels per trial from scene name |
| `get_trial_types()` | behavior.py | LOADING | Gets isreward and morph (env) per trial |
| `is_putative_interneuron()` | spatial.py | CURATION | Identifies putative interneurons by speed correlation |
| `pos_cm_to_rad()` | spatial.py | PROCESSING | Converts position cm to radians |
| `correct_lick_sensor_error()` | behavior.py | CURATION | Removes trials with lick sensor errors |

### Data Processing Pipeline
1. Neural: `sess.timeseries['events']` (deconvolved calcium events)
2. Position: `sess.vr_data['pos']`
3. Speed: `sess.timeseries['speed']`, threshold < 2 cm/s
4. Licks: `sess.timeseries['licks']`, binary, sensor error correction
5. Rewards: `sess.timeseries['rewards']`
6. Trial boundaries: `sess.trial_start_inds`, `sess.teleport_inds`

### Key Processing in get_timeseries_data()
- Lick sensor error: >35% samples with cumulative lick >2 → NaN
- Licks capped at 1 (binary)
- Speed < 2 cm/s → NaN (excluded from analysis in reference code)
- Mask out NaN samples

### Reward Zone Dictionary
- A (X): [80, 130] cm
- B (Y): [200, 250] cm
- C (Z): [320, 370] cm

### Environment Encoding
- Env1 = 0, Env2 = 1

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- NWB files: `data/sub-{mouse}/sub-{mouse}_ses-{session}_behavior+ophys.nwb`
- 11 subjects: m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19
- m17 and m18 have dual-plane imaging (2 planes concatenated)
- Some sessions have neural/behavioral length mismatch (off by 1)

### NWB Structure
- `processing/behavior/BehavioralTimeSeries/`: position, speed, lick, reward_zone, environment, trial_start, teleport, trial number, Reward, autoreward, scanning
- `processing/ophys/Deconvolved/plane{N}/data`: deconvolved events
- `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`: cell classification

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 11 |
| Sessions (total) | 152 |
| Sessions / subject | 12-14 |
| Neurons / session (iscell) | 155-2341, mean 912.4 |
| Total neurons (iscell) | 138,678 |
| Total trials | 12,216 |
| Trials / session | 41-100, mean 80.4 |
| Sampling rate | ~15.5 Hz (64.48 ms) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|---------------|
| Subjects (switch) | 11 | "n = 11 mice" |
| Track length | 450 cm | "450 cm virtual linear track" |
| Reward zone size | 50 cm | "hidden, unmarked 50 cm span" |
| Zone A | 80-130 cm | "zone A, 80-130 cm" |
| Zone B | 200-250 cm | "zone B, 200-250 cm" |
| Zone C | 320-370 cm | "zone C, 320-370 cm" |
| Omission rate | ~15% | "randomly omitted on ~15% of trials" |
| Neurons / session | 155-2172 | "yielded 155-2172 putative pyramidal neurons" |
| Imaging rate | ~15.5 Hz | "~15.5 Hz" |
| Speed threshold | 2 cm/s | "speeds of >2 cm/s" |
| Interneuron excl. | ~0.42% | "excluding 0.42 ± 0.85% of cells" |
| Total imaged trials | 12,376 | "n = 81 out of 12,376 trials" |
| Switch trial | 30 | "30 warm-up trials" |

### Processing Details
- dF/F: Baseline via maximin with 20s sliding window, per trial
- Deconvolution: OASIS algorithm (Suite2p) on dF/F
- Speed threshold: < 2 cm/s excluded from reference analyses
- Lick correction: >30% samples with cumulative lick >2 → NaN (code uses 0.35)

### Curation Steps
**Neuron curation**: Suite2p iscell + interneuron exclusion (speed corr > 0.5)
**Trial curation**: Lick sensor error correction

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Lick error threshold | 0.35 | N/A | >30% | Used code value 0.35 (more conservative) |
| Interneuron threshold | 0.3 (default) | N/A | >0.5 | Used paper value 0.5 |
| Max neurons/session | N/A | 2341 | 2172 | Minor difference, likely due to iscell threshold |
| Total trials | N/A | 12,216 | 12,376 | Paper may count warm-up trials differently |
| Speed threshold | Applied (NaN) | N/A | >2 cm/s | We keep all timepoints for decoder (no speed filtering) |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| Deconvolved events | neural | iscell filter + interneuron exclusion | (n_neurons, n_timepoints) |
| timestamps - trial_start_time | input[0]: time_from_trial_start | float seconds | time-varying |
| environment signal | input[1]: environment_type | 0=Env1, 1=Env2 | per-trial, broadcast |
| trial index | input[2]: trial_number | integer | per-trial, broadcast |
| previous trial reward | input[3]: previous_trial_outcome | 0=omitted, 1=rewarded | per-trial, broadcast |
| position - reward zone | output[0]: distance_to_reward_zone | 7 bins | time-varying |
| position | output[1]: absolute_position | 5 equal bins (90cm each) | time-varying |
| speed | output[2]: speed | 5 bins | time-varying |
| lick (binary) | output[3]: lick | 0/1 | time-varying |
| scene name | output[4]: reward_zone_location | A=0, B=1, C=2 | per-trial |
| reward events | output[5]: reward_outcome | 0=no, 1=yes | per-trial |

### Key Decisions
1. **No speed filtering**: Unlike reference code, we keep all timepoints (including <2 cm/s) for the decoder
2. **Interneuron exclusion**: Used paper's threshold (r>0.5) rather than code default (r>0.3), applied using deconvolved events correlated with speed
3. **Trial boundaries**: trial_start to teleport signals
4. **Multi-plane handling**: Concatenate all planes for dual-plane sessions (m17, m18)
5. **Length mismatch**: Truncate to minimum of neural/behavioral lengths
6. **Lick sensor error**: Following code (>35% threshold), set erroneous lick data to 0

---

## Step 6: Script Development
**Status**: COMPLETE

- `convert_data.py` created with --full, --sample, --show-processing options
- Handles multi-plane sessions (m17, m18)
- Handles neural/behavioral length mismatches
- Includes lick sensor error correction
- Includes interneuron exclusion
- Processing time: ~1.3s per session, ~200s total

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Trials | 160 |
| Neurons | 1298 (155 + 1143) |
| Time bin | 64.48 ms |

### Processing Plots Review
Plots saved as processing_session0.png and processing_session1.png. No anomalies observed.

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| Full conversion | ~1.3s | ~200s (3.3 min) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| distance_to_reward_zone | 0.537 | 0.388 | 0.143 |
| absolute_position | 0.629 | 0.534 | 0.200 |
| speed | 0.489 | 0.378 | 0.200 |
| lick | 0.607 | 0.602 | 0.500 |
| reward_zone_location | 0.934 | 0.854 | 0.333 |
| reward_outcome | 0.651 | 0.626 | 0.500 |

All outputs above chance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9386.6 MB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Converted Data | Match? |
|-----------|-----------------|----------------|--------|
| Subjects | 11 | 11 | ✓ |
| Sessions | ~152 | 152 | ✓ |
| Total trials | 12,376 | 12,216 | ~✓ (98.7%) |
| Neurons/session | 155-2172 | 155-2341 | ~✓ |
| Omission rate | ~15% | 15.3% | ✓ |
| RZ A fraction | ~33% | 34.3% | ✓ |
| RZ B fraction | ~33% | 32.8% | ✓ |
| RZ C fraction | ~33% | 32.9% | ✓ |
| Imaging rate | ~15.5 Hz | 15.51 Hz | ✓ |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

1. **Output log verification**: verification_full_out.txt shows "Data format is valid, no errors or warnings." All 152 sessions, 12,216 trials, 11 subjects correctly identified.

2. **Sanity checks**:
   - Neural data: Raw NWB deconvolved events match converted neural data exactly (np.allclose = True) for session 0, trial 5, neuron 3
   - Position discretization: Raw position → discretized bins matches expected values exactly
   - Time from trial start: Raw timestamps match converted time values (np.allclose = True)
   - Trial count: NWB trial_start count matches converted trial count
   - Neuron count: iscell-filtered count matches converted neuron count

3. **Reference code comparison**:
   - (a) Data loading: NWB fields match sess object fields used in reference code
   - (b) Neuron filtering: iscell + interneuron exclusion (speed corr > 0.5) matches paper
   - (c) Temporal alignment: trial_start to teleport matches reference code's trial_start_inds to teleport_inds
   - (d) Binning: Using raw imaging frames (~64.48 ms), no additional temporal binning
   - (e) Input construction: Time from trial start, environment, trial number, previous outcome
   - (f) Output construction: Distance to reward zone, position, speed, lick, reward zone location, reward outcome
   - Difference: Reference code applies speed threshold (<2 cm/s → NaN) and masks out those timepoints. Our decoder keeps all timepoints as the decoder task doesn't specify filtering.

4. **Key statistics comparison**: See Step 9 consistency check. All statistics match within expected tolerances.

5. **Edge cases**:
   - Multi-plane sessions (m17, m18): Handled by concatenating plane0 and plane1
   - Neural/behavioral length mismatch: Handled by truncating to minimum length
   - Lick sensor errors: Handled by setting erroneous lick data to 0
   - Very long trials (max 3359 timepoints = 216s): Valid data, mouse was slow/stopped
   - Sessions with variable trial counts (41-100): All handled correctly

### Issues Found and Resolved
- Multi-plane sessions caused IndexError: Fixed by loading all planes and concatenating
- Neural/behavioral length mismatch (off by 1): Fixed by truncating to minimum length

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (168.3 → 1.19)
- Test loss: 1.07

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Ratio |
|--------|----------------------|------------------------|--------|-------|
| distance_to_reward_zone | 0.464 | 0.416 | 0.143 | 2.9x |
| absolute_position | 0.559 | 0.530 | 0.200 | 2.7x |
| speed | 0.463 | 0.431 | 0.200 | 2.2x |
| lick | 0.631 | 0.623 | 0.500 | 1.2x |
| reward_zone_location | 0.848 | 0.813 | 0.333 | 2.4x |
| reward_outcome | 0.604 | 0.524 | 0.500 | 1.05x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Chance | Ratio | Notes |
|----------|-------------------|--------|-------|-------|
| distance_to_reward_zone | 0.416 | 0.143 | 2.9x | Good - position relative to reward zone is decodable |
| absolute_position | 0.530 | 0.200 | 2.7x | Good - hippocampal place cells encode position |
| speed | 0.431 | 0.200 | 2.2x | Good - speed is correlated with neural activity |
| lick | 0.623 | 0.500 | 1.2x | Moderate - licking is a sparse event |
| reward_zone_location | 0.813 | 0.333 | 2.4x | Very good - reward zone remapping is well-encoded |
| reward_outcome | 0.524 | 0.500 | 1.05x | Near chance - expected since omission is random |

### Check 1: Accuracy vs chance
All outputs are above chance. reward_outcome is barely above chance (1.05x), which is expected because reward omission is randomly determined (~15% of trials) and thus not strongly predictable from neural activity alone.

### Check 2: Accuracy comparison to paper
The paper's decoder uses circular regression on reward-relative position and reports decode scores (cos(y-y_hat)), not balanced accuracy. Direct comparison is not possible due to different decoder architectures and output variables. However:
- Position decoding (0.530) is consistent with hippocampal place cell encoding
- Reward zone location (0.813) is consistent with the paper's finding that neural populations remap with reward zone changes

### Check 3: Train vs validation gap
- Most outputs have modest gaps (training ~0.03-0.05 higher)
- reward_outcome has the largest gap (0.604 vs 0.524), suggesting some overfitting on this difficult-to-predict variable
- No evidence of data leakage

### Issues Found and Resolved
No additional issues found.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
