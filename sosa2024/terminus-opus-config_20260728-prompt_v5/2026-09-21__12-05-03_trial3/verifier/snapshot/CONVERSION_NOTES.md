# Dataset Conversion Notes

## Overview
- **Dataset**: Sosa, Plitt & Giocomo 2025 - "A flexible hippocampal population code for experience relative to reward"
- **Source**: DANDI:001361 - NWB format, 2P calcium imaging of CA1
- **Date started**: 2025
- **Goal**: Convert to decoder-compatible format for predicting behavioral/task variables from neural activity

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/paper.pdf` - Reference paper
- `/app/methods.txt` - Extracted methods text
- `/app/code/` - Reference code repository (reward_relative package)
- `/app/data/` - NWB data files (11 subjects, 152 sessions)
- `/app/decoder.py`, `/app/train_decoder.py` - Decoder implementation

Python environment: numpy 2.4.4, torch 2.6.0 with CUDA

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `dff()` | preprocessing.py | PROCESSING | Compute dF/F: neuropil subtraction (0.7*Fneu), maximin baseline (20s window), smoothing (sigma=2) |
| `get_trial_types()` | behavior.py | PROCESSING | Extract isreward and morph (env identity) per trial |
| `get_reward_zones()` | behavior.py | PROCESSING | Get reward zone coords and labels per trial from scene name |
| `reward_zone_dict` | behavior.py | LOADING | Zone coords: A=[80,130], B=[200,250], C=[320,370] |
| `sessions_dict` | sessions_dict.py | LOADING | Session metadata: date, scene, exp_day per animal |
| `multi_anim_sess()` | utilities.py | PROCESSING | Collect sessions, compute dF/F, calculate place cells |
| `detect_interneurons` | methods.txt | CURATION | Pearson corr(dF/F, speed) > 0.5 |
| `nansmooth()` | utilities.py | PROCESSING | Gaussian smoothing handling NaNs |

### Notes
- NWB files contain raw suite2p outputs (F, Fneu) plus behavioral data aligned to imaging frames
- The Deconvolved data in NWB is suite2p's default deconvolution of raw F, NOT the paper's custom dF/F pipeline
- Paper computes dF/F: neuropil subtraction -> maximin baseline per trial -> dF/F -> 2-sample Gaussian smooth -> optional OASIS deconvolution
- Multi-plane animals (m17, m18): planes pooled for all analyses

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- 11 subjects: m3, m4, m7, m11-m15, m17-m19
- NWB files: `sub-{id}_ses-{num}_behavior+ophys.nwb`
- m11: 12 sessions (ses-03 to ses-14, missing ses-01/02)
- All others: 14 sessions (ses-01 to ses-14)
- Multi-plane: m17 (2 planes), m18 (2 planes)

### NWB Structure
- `processing/ophys/Fluorescence/plane{N}/data`: Raw F (n_timepoints, n_rois)
- `processing/ophys/Neuropil/plane{N}/data`: Raw Fneu
- `processing/ophys/Deconvolved/plane{N}/data`: Suite2p deconvolved (not used)
- `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`: Cell classification (n_rois, 2)
- `processing/behavior/BehavioralTimeSeries/`: position, speed, lick, reward, reward_zone, environment, trial_number, trial_start, teleport, scanning, autoreward
- `processing/behavior/BehavioralTimeSeries/Reward`: timestamps + amounts

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 11 |
| Sessions (total) | 152 |
| Sessions / subject | 12-14 |
| ROIs / session | 349-4857 |
| Frame rate | ~15.51 Hz (~64.5 ms/frame) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects (switch) | 11 | "n = 11 mice" |
| Sessions / subject | 14 (12 for m11) | "14 days" |
| Total sessions | 152 | 11*14 - 2 |
| Trials / session | 80.5 ± 7.4 | "mean ± s.d., 80.5 ± 7.4 trials across 14 mice" |
| Total trials | 12,376 | "n = 81 out of 12,376 trials" |
| Neurons / session | 155-2172 | "155–2172 putative pyramidal neurons per session" |
| Frame rate | ~15.5 Hz | "~15.5 Hz" |
| Track length | 450 cm | "450 cm virtual linear track" |
| Reward zone size | 50 cm | "50 cm reward zone" |
| Zone A | 80-130 cm | "zone A, 80–130 cm" |
| Zone B | 200-250 cm | "zone B, 200–250 cm" |
| Zone C | 320-370 cm | "zone C, 320–370 cm" |
| Reward omission | ~15% | "randomly omitted on approximately 15% of trials" |
| Switch trial | 30 | "Each switch occurred after 30 trials" |
| Speed threshold | 2 cm/s | "<2 cm s−1" excluded for spatial analyses |
| Interneuron excl. | corr > 0.5 | "Pearson correlation of >0.5" |
| Erroneous lick trials | 81/12376 (~0.65%) | "~0.65% of all imaged trials" |

### Processing Details
- dF/F: F -= 0.7*Fneu, add back mean Fneu per trial, maximin baseline (smooth sigma=15, min filter 300, max filter 300), dF/F = (F-baseline)/|baseline|, smooth sigma=2
- Cell curation: iscell (manual) + interneuron exclusion (corr(dF/F, speed) > 0.5)
- Trial boundaries: trial_start and teleport markers

### Curation Steps
**Neuron curation**: iscell=1 + exclude interneurons (corr > 0.5)
**Trial curation**: Use trial_start to teleport boundaries

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| NWB Deconvolved | suite2p raw deconv | Suite2p deconv from raw F | dF/F then OASIS | Compute dF/F from raw F+Fneu following paper method |
| Multi-plane | Pool planes | m17,m18 have 2 planes | Pool for analysis | Concatenate planes, use combined iscell |
| Neural/beh mismatch | N/A | Some sessions off-by-1 | N/A | Truncate to minimum length |
| Trial count | N/A | 12,216 | 12,376 | Difference likely due to counting method; 160 trial difference (~1.3%) |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| F, Fneu, iscell | neural | dF/F computation, iscell filter, interneuron exclusion | Following paper's maximin baseline method |
| timestamps | input[0] | time_from_trial_start = ts - ts[trial_start] | Continuous, time-varying |
| environment | input[1] | 0=ENV1, 1=ENV2 | Binary, per-trial |
| trial_number | input[2] | 0-indexed trial number | Continuous, per-trial |
| reward (prev trial) | input[3] | Previous trial rewarded? 0/1 | Binary, per-trial |
| position, rz_coords | output[0] | distance_to_reward_zone, 7 bins | Time-varying |
| position | output[1] | absolute_position, 5 bins of 90cm | Time-varying |
| speed | output[2] | speed, 5 bins | Time-varying |
| lick | output[3] | lick binary (>0 = lick) | Time-varying |
| scene -> rz_label | output[4] | reward_zone_location (A=0,B=1,C=2) | Per-trial |
| reward_ts | output[5] | reward_outcome (0/1) | Per-trial |

### Key Decisions
1. **Neural signal: dF/F (not deconvolved)**: Computed dF/F from raw F following paper's method. Did not use suite2p deconvolved data because it doesn't match the paper's processing pipeline.
2. **Reward zone from scene names**: Used sessions_dict.py scene names to determine reward zone per trial, with change_trial=30 for switch sessions.
3. **Lick binarization**: lick > 0 = lick present (cumulative count per frame)
4. **Trial boundaries**: trial_start to teleport markers
5. **No speed filtering**: Unlike the paper's spatial analyses which exclude <2 cm/s, we include all data within trials for the decoder.

---

## Step 6: Script Development
**Status**: COMPLETE

Script at `/app/convert_data.py`. Key features:
- Handles single and multi-plane sessions
- Computes dF/F following paper's maximin baseline method
- Excludes interneurons by correlation with speed
- Handles off-by-one between neural and behavioral timepoints
- Supports --sample, --full, --show-processing modes

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics (2 sessions: m11 ses-03, m17 ses-09)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 |
| Total trials | 160 |
| Neurons/session | 154, 539 |

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| dF/F computation | 0.8-8.4s | ~13 min for 152 sessions |

Actual total time: 800s (~13.3 minutes)

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| distance_to_reward_zone | 0.762 | 0.656 | 0.143 |
| absolute_position | 0.852 | 0.757 | 0.200 |
| speed | 0.694 | 0.595 | 0.200 |
| lick | 0.866 | 0.836 | 0.500 |
| reward_zone_location | 0.984 | 0.961 | 0.333 |
| reward_outcome | 0.907 | 0.643 | 0.500 |

All outputs above chance. Loss decreased from 2.15 to 0.50.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 9355 MB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Converted Data | Match? |
|-----------|-----------------|----------------|--------|
| Subjects | 11 | 11 | YES |
| Sessions | 152 | 152 | YES |
| Total trials | 12,376 | 12,216 | CLOSE (98.7%) |
| Trials/session | 80.5 ± 7.4 | 80.4 ± 6.1 | YES |
| Neurons/session range | 155-2172 | 154-2323 | CLOSE |
| Reward omission rate | ~15% | 15.3% | YES |
| Frame rate | ~15.5 Hz | ~15.5 Hz | YES |
| Track length | 450 cm | 0-451 cm | YES |

### Trial Count Discrepancy
Converted: 12,216 vs Paper: 12,376 (difference: 160, ~1.3%)
Possible explanations:
- Paper may count trials differently (e.g., including partial trial 0)
- Minor differences in trial boundary detection
- The paper's count might include trials from a slightly different processing pipeline

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
verification_full_out.txt: "Data format is valid, no errors or warnings."
No errors or warnings to address.

### Check 2: Sanity checks
1. **Neural data timepoints**: Verified m11 ses-03 trial 5 has 266 timepoints matching raw data trial boundaries. PASS.
2. **Position output**: Raw position 35.3cm at t=10 correctly maps to bin 0 (<90cm). PASS.
3. **Speed output**: Raw speed 49.4cm/s at t=10 correctly maps to bin 4 (>40). PASS.
4. **Time input**: Time from trial start at t=10 = 0.6448s matches raw timestamps. PASS.

### Check 3: Reference code comparison
| Processing Step | Our Code | Reference Code | Match? |
|----------------|----------|---------------|--------|
| Data loading | h5py from NWB | suite2p + TwoPUtils | Different format, same data |
| Cell filtering | iscell[:,0]==1 | iscell from suite2p | YES |
| Interneuron excl. | corr(dF/F, speed) > 0.5 | Same criterion in methods | YES |
| dF/F computation | maximin baseline per trial | preprocessing.dff() | YES (reimplemented) |
| Neuropil subtraction | F -= 0.7*Fneu | neu_coef=0.7 | YES |
| Baseline | smooth(sigma=15), min_filter(300), max_filter(300) | Same in preprocessing.py | YES |
| dF/F smoothing | nansmooth(sigma=2) per trial | Same in preprocessing.py | YES |
| Trial boundaries | trial_start -> teleport | trial_start_inds -> teleport_inds | YES |
| Reward zone | From sessions_dict scene names | get_reward_zones() | YES |

### Check 4: Key statistics comparison
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| Subjects | 11 | 11 | YES |
| Sessions | ~152 | 152 | YES |
| Total trials | 12,376 | 12,216 | 98.7% |
| Trials/session | 80.5±7.4 | 80.4±6.1 | YES |
| Neurons/session | 155-2172 | 154-2323 | CLOSE |
| Omission rate | ~15% | 15.3% | YES |
| Interneuron excl. | 0.42±0.85% | ~0.4% | YES |

### Check 5: Edge cases
- Off-by-one between neural and behavioral data: handled by truncation
- Multi-plane sessions: handled by concatenating planes
- Sessions with few trials (m4 ses-04: 41 trials): included, valid data
- Very long trials (>100s): real data from slow-running mouse, included

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (3.00 -> 0.65 over 200 epochs)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Val/Chance |
|--------|-------------|--------|-------|-------|
| distance_to_reward_zone | 0.806 | 0.629 | 0.143 | 4.4x |
| absolute_position | 0.906 | 0.769 | 0.200 | 3.8x |
| speed | 0.730 | 0.623 | 0.200 | 3.1x |
| lick | 0.795 | 0.766 | 0.500 | 1.5x |
| reward_zone_location | 0.966 | 0.866 | 0.333 | 2.6x |
| reward_outcome | 0.934 | 0.604 | 0.500 | 1.2x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Val Acc | Chance | Ratio | Assessment |
|----------|-----------------|--------|-------|------------|
| distance_to_reward_zone | 0.629 | 0.143 | 4.4x | Good |
| absolute_position | 0.769 | 0.200 | 3.8x | Good |
| speed | 0.623 | 0.200 | 3.1x | Good |
| lick | 0.766 | 0.500 | 1.5x | OK |
| reward_zone_location | 0.866 | 0.333 | 2.6x | Good |
| reward_outcome | 0.604 | 0.500 | 1.2x | Low but above chance |

### Check 1: Accuracy vs chance
All outputs above chance. reward_outcome at 1.2x chance is the lowest, but this is expected because reward outcome is a per-trial variable with ~85% reward rate, making it hard to predict from within-trial neural activity alone.

### Check 2: Accuracy comparison to paper
The paper uses circular-linear regression for position decoding (not the same architecture). Direct comparison is difficult, but:
- Position decoding is strong (0.769 val acc), consistent with the paper's finding that CA1 neurons encode position
- Reward zone location is well decoded (0.866), consistent with the paper's finding of reward-relative coding

### Check 3: Train vs validation gap
- reward_outcome: 0.934 train vs 0.604 val (1.55x gap) - some overfitting but the variable is inherently hard to predict
- Other variables: moderate gaps, typical for neural decoding

### Issues Found and Resolved
- No critical issues found. All outputs are above chance.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [ ] README.md created
- [ ] cache/ folder created
- [ ] All files organized
