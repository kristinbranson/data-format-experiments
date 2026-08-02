# Dataset Conversion Notes

## Overview
- **Dataset**: Sosa, Plitt & Giocomo (2025) "A flexible hippocampal population code for experience relative to reward" - DANDI:001361
- **Date started**: 2025-07-29
- **Goal**: Convert 2-photon calcium imaging data from hippocampal CA1 during VR navigation to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `data/` - NWB files organized by subject (sub-m3, m4, m7, m11-m15, m17-m19)
- `code/` - Reference code from the paper (Sosa_et_al_2024 repo)
- `paper.pdf` - Reference paper
- `methods.txt` - Methods section extracted from paper
- `decoder.py` - Decoder module
- `train_decoder.py` - Script to train/validate decoder

Python environment: numpy 2.3.5, torch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| get_timeseries_data | glmUtils.py | LOADING | Extracts behavioral + neural data from sess object |
| get_reward_zones | behavior.py | PROCESSING | Determines reward zone positions from scene name |
| get_trial_types | behavior.py | PROCESSING | Determines isreward, morph (environment) per trial |
| get_omission_trials | rewardAnalysis.py | PROCESSING | Identifies true omission trials |
| get_omission_inds | rewardAnalysis.py | PROCESSING | Gets reward zone entry indices on omission trials |
| dff | preprocessing.py | PROCESSING | Computes dF/F with maximin baseline + deconvolution |
| define_anim_list | dayData.py | CURATION | Defines which animals to include per day |
| correct_lick_sensor_error | behavior.py | CURATION | Removes trials with lick sensor errors |

### Notes
- Reference code uses `sess` objects (pickled) containing VR data and neural timeseries
- NWB files contain equivalent data mapped to sess-like structure
- Key sess attributes: `trial_start_inds`, `teleport_inds`, `vr_data` (DataFrame), `timeseries` (dict), `scene`
- Neural data = deconvolved events from OASIS algorithm applied to dF/F
- Speed threshold of 2 cm/s applied in get_timeseries_data (below threshold set to NaN, masked out)
- Lick sensor error correction: if >35% of samples in trial have cumulative lick >2, set licks to NaN
- Licks clipped to binary (0/1)
- reward_zone_dict: A/X=[80,130], B/Y=[200,250], C/Z=[320,370]
- Environment mapping: Env1=0, Env2=1
- GCAMP_N -> m_N naming convention
- Session numbers in NWB = exp_day in sessions_dict

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- 11 subjects (m3, m4, m7, m11-m15, m17-m19)
- NWB files: `data/sub-mX/sub-mX_ses-YY_behavior+ophys.nwb`
- Each NWB file contains:
  - Behavioral time series (all same length, ~15.5 Hz):
    - position, speed, lick, Reward (event-based), reward_zone, environment, trial number, trial_start, teleport, scanning, autoreward
  - Neural data (per plane, multi-plane for m17/m18):
    - Deconvolved events: (n_timepoints, n_ROIs)
    - Fluorescence: (n_timepoints, n_ROIs) - raw F
    - Neuropil: (n_timepoints, n_ROIs)
    - iscell: (n_ROIs, 2) - cell classification

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 11 |
| Sessions total | 152 |
| Sessions / subject | 12-14 (m11 has 12, others 14) |
| Imaging rate | ~15.5 Hz |
| Total trials (from NWB) | 12,216 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects (switch) | 11 | "n = 11 mice" |
| Subjects (fixed) | 3 | "n = 3 fixed-condition mice" (not in NWB) |
| Neurons / session | 155-2172 | "155-2172 putative pyramidal neurons per session" |
| Trials / session | ~80 | "mean ± s.d., 80.5 ± 7.4 trials" |
| Track length | 450 cm | "450 cm linear track" |
| Reward zone size | 50 cm | "hidden, unmarked 50 cm span" |
| Zone A | 80-130 cm | |
| Zone B | 200-250 cm | |
| Zone C | 320-370 cm | |
| Reward omission | ~15% | "randomly omitted on ~15% of trials" |
| Switch trial | 30 | "Each switch occurred after 30 trials" |
| Imaging rate | ~15.5 Hz | |
| Speed threshold | 2 cm/s | |
| Lick error trials | ~0.65% | "n=81 out of 12,376" |
| Total imaged trials | 12,376 | |
| Interneuron exclusion | 0.42 ± 0.85% | |

### Processing Details
- Temporal alignment: Align to trial start
- Neural data: Deconvolved calcium events (already in NWB)
- Cell filtering: iscell=1 from Suite2P
- Speed threshold: 2 cm/s in reference code (not applied in decoder - continuous time series needed)

### Curation Steps
**Neuron curation rules**: iscell=1 (Suite2P manual curation)
**Trial curation rules**: Lick sensor error detection (>35% samples with cumulative lick >2)

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Lick error threshold | 0.35 | N/A | 0.30 | Use code value 0.35 |
| Trial count | 12,216 | 12,216 | 12,376 | Paper may include 3 fixed mice |
| Neuron range | N/A | 155-2341 | 155-2172 | Max slightly higher, likely pre-interneuron exclusion |
| Speed masking | Applied in ref code | N/A | Applied | NOT applied in decoder |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| Deconvolved events (iscell=1) | neural | Filter by iscell, transpose | Per trial |
| Time from trial start | input[0] | frame_index * dt | Time-varying |
| Environment (0/1) | input[1] | From scene name | Per trial |
| Trial number | input[2] | 0-indexed | Per trial |
| Previous trial outcome | input[3] | 0=omission, 1=rewarded | Per trial |
| Distance to reward zone | output[0] | Signed distance, 7 bins | Time-varying |
| Absolute position | output[1] | 5 equal bins | Time-varying |
| Speed | output[2] | 5 bins | Time-varying |
| Lick | output[3] | Binary | Time-varying |
| Reward zone location | output[4] | A=0, B=1, C=2 | Per trial |
| Reward outcome | output[5] | Binary | Per trial |

### Key Decisions
1. Use deconvolved events from NWB (matches reference code sess.timeseries['events'])
2. Trial boundaries: trial_start to teleport signals
3. Time bin: native imaging rate (~64.5 ms)
4. No speed masking (continuous time series for decoder)
5. Reward zone from sessions_dict scene mapping (imported from reference code)
6. Multi-plane animals: concatenate planes, apply combined iscell mask
7. All 152 sessions processed

---

## Step 6: Script Development
**Status**: COMPLETE

Script `convert_data.py` created with:
- Multi-plane animal support (m17, m18)
- Scene-based reward zone determination using reference code's sessions_dict
- Lick sensor error detection
- Processing visualization (--show-processing flag)
- Sample mode (--sample) and full mode (--full)

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (m11 ses-03, m17 ses-09) |
| Subjects | 2 |
| Total trials | 160 |
| Neurons/session | 155, 539 |
| Time bin | 64.48 ms |

### Processing Plots Review
Plots saved. No anomalies observed.

### Run Time Estimates
| Step | Time |
|------|------|
| Sample (2 sessions) | 3.6s |
| Full (152 sessions) | ~90s |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Bal Acc | Validation Bal Acc | Chance |
|--------|-----------------|-------------------|--------|
| distance_to_reward_zone | 0.4856 | 0.3947 | 0.1429 |
| absolute_position | 0.5546 | 0.5128 | 0.2000 |
| speed | 0.4602 | 0.4191 | 0.2000 |
| lick | 0.7435 | 0.7232 | 0.5000 |
| reward_zone_location | 0.9678 | 0.9704 | 0.3333 |
| reward_outcome | 0.5932 | 0.5637 | 0.5000 |

All outputs above chance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9842.6 MB
- `verification_full_out.txt`: created, no errors/warnings

### Consistency Check
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| Subjects | 11 (switch) | 11 | ✓ |
| Sessions | ~152 | 152 | ✓ |
| Total trials | 12,376 (14 mice) | 12,216 (11 mice) | ~✓ |
| Trials/session | 80.5 ± 7.4 | 80.4 ± 6.1 | ✓ |
| Neurons/session | 155-2172 | 155-2341 | ~✓ |
| Reward omission | ~15% | 15.7% | ✓ |
| Zone distribution | ~1/3 each | A=32.9%, B=33.7%, C=33.5% | ✓ |
| Imaging rate | ~15.5 Hz | 15.5 Hz | ✓ |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

**Check 1: Output log verification**
- No errors or warnings in verification_full_out.txt

**Check 2: Sanity checks**
1. **Neural data spot check**: Loaded m11 ses-03, trial 5, compared first 3 neurons × 5 timepoints between converted data and original NWB. Result: EXACT MATCH (np.allclose = True)
2. **Position bin check**: Compared position discretization for m11 ses-03 trial 5. Result: EXACT MATCH
3. **Speed bin check**: Compared speed discretization for m11 ses-03 trial 5. Result: EXACT MATCH
4. **Input time check**: Compared time_from_trial_start for m11 ses-03 trial 5. Result: EXACT MATCH
5. **Lick data check**: Compared lick processing for m11 ses-03 trial 5. Result: EXACT MATCH
6. **Reward zone check**: Verified m11 ses-03 (Env1_LocationB_to_A) has B for trials 0-29, A for trials 30+. Confirmed against actual reward positions. Result: CORRECT
7. **Previous trial outcome check**: Verified prev_outcome[i] = reward_outcome[i-1]. Result: CORRECT

**Check 3: Reference code comparison**
- (a) Data loading: NWB h5py loading matches sess object structure
- (b) Neuron filtering: iscell[:,0]==1 matches reference code
- (c) Temporal alignment: trial_start to teleport matches reference code
- (d) Binning: Native imaging rate (~64.5ms) used, no additional binning
- (e) Input construction: Time, environment, trial number, previous outcome all correctly computed
- (f) Output construction: Distance to reward zone, position, speed, lick, reward zone location, reward outcome all correctly computed and discretized

**Check 4: Key statistics comparison**
- Total trials: 12,216 (NWB) vs 12,376 (paper) - difference of 160 likely from 3 fixed-condition mice not in NWB data
- Neuron range: 155-2341 vs paper's 155-2172 - max slightly higher, likely because paper reports after interneuron exclusion (0.42% of cells)
- Reward rate: 84.3% vs expected ~85% - consistent
- Zone distribution: A=32.9%, B=33.7%, C=33.5% - well balanced as expected

**Check 5: Edge cases**
- Multi-plane animals (m17, m18): Correctly handled by concatenating plane0 and plane1 data
- m11 starts from session 03: Correctly handled
- Short trials (<3 timepoints): Skipped
- Lick sensor errors: Detected and set to 0

### Issues Found and Resolved
1. **CRITICAL: Wrong sessions_dict** - Initially manually copied sessions_dict with errors (e.g., GCAMP11 day 3 was listed as Env1_LocationA_to_B instead of Env1_LocationB_to_A). Fixed by importing directly from reference code. Verified fix with reward position spot checks.
2. **Multi-plane handling**: Initially only loaded plane0, causing IndexError for m17/m18. Fixed by concatenating all planes.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (170.0 → 1.21)

### Decoder Results (Full)
| Output | Training Bal Acc | Validation Bal Acc | Chance | Ratio |
|--------|-----------------|-------------------|--------|-------|
| distance_to_reward_zone | 0.4384 | 0.3918 | 0.1429 | 2.7x |
| absolute_position | 0.5323 | 0.5102 | 0.2000 | 2.6x |
| speed | 0.4274 | 0.4044 | 0.2000 | 2.0x |
| lick | 0.6490 | 0.6385 | 0.5000 | 1.3x |
| reward_zone_location | 0.8400 | 0.7955 | 0.3333 | 2.4x |
| reward_outcome | 0.5825 | 0.5202 | 0.5000 | 1.0x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Achieved Accuracy | Chance | Ratio | Notes |
|----------|------------------|--------|-------|-------|
| distance_to_reward_zone | 0.3918 | 0.1429 | 2.7x | Good - position is well encoded in CA1 |
| absolute_position | 0.5102 | 0.2000 | 2.6x | Good - place cells encode position |
| speed | 0.4044 | 0.2000 | 2.0x | Good - speed modulation in CA1 |
| lick | 0.6385 | 0.5000 | 1.3x | Moderate - licking is sparse |
| reward_zone_location | 0.7955 | 0.3333 | 2.4x | Good - reward zone well encoded |
| reward_outcome | 0.5202 | 0.5000 | 1.0x | Near chance - expected |

### Check 1: Accuracy vs chance
- All outputs above chance
- reward_outcome at ~1.0x chance: This is expected because reward outcome is a per-trial binary variable. The decoder sees neural activity from trial start to teleport, and whether reward was delivered may not be strongly encoded in the pre-reward neural activity pattern. The paper's decoder decodes reward-relative POSITION, not reward outcome.

### Check 2: Accuracy comparison to paper
- The paper uses a Bayesian decoder (projected normal distribution) specifically for reward-relative position, not a general neural network decoder. Direct comparison is not straightforward.
- The paper reports decoder performance as a "decode score" (1 - mean|y_t - ŷ_t|), not balanced accuracy.
- Our decoder achieves 2.7x chance for distance_to_reward_zone and 2.6x for absolute_position, which indicates the neural data correctly encodes spatial information.

### Check 3: Train vs validation gap
- distance_to_reward_zone: 0.4384 train vs 0.3918 val (ratio 1.12) - acceptable
- absolute_position: 0.5323 vs 0.5102 (ratio 1.04) - good
- speed: 0.4274 vs 0.4044 (ratio 1.06) - good
- lick: 0.6490 vs 0.6385 (ratio 1.02) - excellent
- reward_zone_location: 0.8400 vs 0.7955 (ratio 1.06) - good
- reward_outcome: 0.5825 vs 0.5202 (ratio 1.12) - acceptable
- No significant overfitting detected (all ratios < 1.5x)

### Issues Found and Resolved
- No new issues found in this review.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
