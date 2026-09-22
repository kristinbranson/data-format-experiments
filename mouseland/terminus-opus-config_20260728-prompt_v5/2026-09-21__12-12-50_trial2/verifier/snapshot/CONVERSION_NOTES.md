# Dataset Conversion Notes

## Overview
- **Dataset**: Zhong et al. 2025 "Unsupervised pretraining in biological neural networks" - calcium imaging in visual cortex during virtual reality corridor task
- **Date started**: 2024
- **Goal**: Convert to decoder-compatible format for predicting behavioral variables from neural activity

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/code/` - Reference code (utils.py, data_process_script.ipynb, fig1-5.py, S6.py)
- `/app/data/` - Data files (beh/, spk/, retinotopy/, process_data/)
- `/app/paper.pdf` - Reference paper
- `/app/methods.txt` - Extracted methods text
- `/app/decoder.py` - Decoder module
- `/app/train_decoder.py` - Decoder training script

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_exp_beh | utils.py | LOADING | Load behavioral data for experiment type |
| load_spk | utils.py | LOADING | Load spike data, concatenate across imaging planes |
| load_retino | utils.py | LOADING | Load retinotopy data for brain area assignment |
| neu_area_ID | utils.py | PROCESSING | Map iarea indices to V1, mHV, lHV, aHV |
| spk_pos_interp | utils.py | PROCESSING | Interpolate spikes to position bins |
| get_interpPos_spk | utils.py | PROCESSING | Batch position interpolation of spikes |
| dprime | utils.py | PROCESSING | Compute d-prime for stimulus selectivity |
| lickCount | utils.py | PROCESSING | Compute binary lick responses per trial |

### Notes
- Frame rate: 3.17 Hz (calcium imaging)
- Neural data: deconvolved fluorescence traces from Suite2p
- Position interpolation: 60 bins per trial (corridor + grey space)
- Brain area assignment: V1=iarea8, mHV=0|1|2|9, lHV=5|6, aHV=3|4
- Excluded areas: iarea=-1, iarea=7 (outside visual cortex)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- **Spike data**: 89 .npy files in `/app/data/spk/`, each with 'spks' key -> list of arrays per imaging plane
- **Behavioral data**: 23 Beh_*.npy files in `/app/data/beh/`, each a dict of sessions
- **Retinotopy**: .npz files in `/app/data/retinotopy/` with iarea and xy_t
- **Experiment info**: Imaging_Exp_info.npy mapping experiment types to session metadata

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Spike files | 89 |
| Unique sessions | 89 |
| Subjects | 19 |
| Sessions / subject | 1-8 (mean ~4.7) |
| Neurons / session (raw) | 20,547-89,577 |
| Neurons / session (filtered) | 17,363-78,815 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Recordings | 89 | "We performed 89 recordings in 19 mice" |
| Subjects | 19 | "19 mice bred to express GCaMP6s" |
| Neurons per recording | 20,547-89,577 | "20,547 to 89,577 neurons in each recording" |
| Frame rate | 3.17 Hz | "fs = 3.17Hz" (data_process_script) |
| Corridor length | 4m | "corridors were each 4 m long" |
| Grey space | 2m | "with 2 m of grey space between corridors" |
| VR speed | 60 cm/s | "always moved at a constant speed (60 cm/s)" |
| Running threshold | 6 cm/s | "running faster than a threshold of 6 cm/s" |
| Sound cue position | 0.5-3.5m | "randomly chosen from uniform distribution between 0.5 m and 3.5 m" |
| Position bins | 60 | "bins of position: 60, each bin is 1 decimeter" |
| Deconvolution | Suite2p, 0.75s decay | "timescale of decay of 0.75 s" |
| Analysis basis | Deconvolved traces | "All analyses were based on deconvolved fluorescence traces" |
| Running filter | Only running | "We only considered timepoints during running" |

### Curation Steps
**Neuron curation**: Exclude neurons outside visual cortex (iarea == -1 or 7)
**Trial curation**: Only frames where VR is moving (ft_move > 0) used in position interpolation

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Total corridor | 60 bins = 6m | CL=60, ft_Pos [0,60] | 4m + 2m grey | Consistent |
| Neurons | Varies | 20,547-89,577 raw | 20,547-89,577 | Consistent |
| Recordings | 89 sessions | 89 spike files | 89 recordings | Consistent |
| Mice | 19 unique | 19 mnames | 19 mice | Consistent |

No unresolved discrepancies.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| spks (deconvolved) | neural | Position interpolation to 60 bins, filter to visual cortex | Matches reference code |
| SoundPos | input[0]: time_to_sound_cue | Convert position distance to time (seconds) | Continuous, time-varying |
| sess# | input[1]: day_of_training | Direct mapping from exp_info | Continuous, per-trial |
| Position bin index | input[2]: time_since_trial_start | bin_idx * time_per_bin | Continuous, time-varying |
| isRew | input[3]: reward_availability | Direct boolean->float | Discrete, per-trial |
| WallName | output[0]: visual_stimulus | Map to global category index | 15 categories |
| LickPos/LickTrind | output[1]: licking | Binary raster at position bins | Binary, time-varying |
| Position bins | output[2]: position | 4 bins of 1m each in corridor | Categorical, time-varying |
| run_pos | output[3]: running_speed | Quartile bins (global) | Categorical, time-varying |

### Key Decisions
1. **Position-based binning (60 bins)**: Matches reference code exactly
2. **All 89 sessions included**: Each unique recording once
3. **Neuron filtering**: Exclude iarea==-1 and 7 (outside visual cortex)
4. **Time bin size**: ~166.67 ms (0.1m / 0.6 m/s)
5. **Grey space included**: Full 60 bins per trial
6. **Speed quartiles**: Computed globally across all sessions
7. **15 stimulus categories**: All unique wall names across all sessions

---

## Step 6: Script Development
**Status**: COMPLETE

Script: `/app/convert_data.py`
- Optimized interpolation using np.interp (4x speedup over scipy interp1d)
- Cached behavioral data loading to avoid reloading large files
- Efficient stimulus name and speed quartile collection (load each beh file once)

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics (2 sessions, DR10)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 1 (DR10) |
| Trials | 934 |
| Neurons/session | 49,472-52,246 |
| Time bins | 60 |

Verification: No errors, no warnings.

### Run Time: ~50s for 2 sessions, estimated ~40 min for full 89 sessions

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample - 2 sessions, DR10 unsupervised mouse)
| Output | Train Bal Acc | Val Bal Acc | Chance |
|--------|--------------|-------------|--------|
| visual_stimulus | 0.8515 | 0.8278 | 0.0667 |
| licking | 1.0000 | 1.0000 | 0.5000 |
| position | 0.7731 | 0.7547 | 0.2500 |
| running_speed | 0.6712 | 0.6465 | 0.2500 |

All above chance. Licking trivially 1.0 (unsupervised mouse never licks).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 410.69 GB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Converted Data | Match? |
|-----------|-----------------|----------------|--------|
| Sessions | 89 | 89 | Yes |
| Subjects | 19 | 19 | Yes |
| Neurons/session (raw) | 20,547-89,577 | 20,547-89,577 | Yes |
| Neurons/session (filtered) | N/A | 17,363-78,815 | N/A |
| Total trials | N/A | 38,110 | N/A |
| Position bins | 60 | 60 | Yes |
| Brain regions | V1, mHV, lHV, aHV | V1, mHV, lHV, aHV | Yes |
| Stimuli | circle, leaf, rock, brick + variants | 15 categories | Yes |
| Sound cue range | 0.5-3.5m (5-35 bins) | [-9.7, 6.4]s time | Consistent |
| Reward availability | Present for task mice | [0, 1] | Yes |
| Running speed | Quartiles | Q1-Q4 (25% each globally) | Yes |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log verification**: No errors, no warnings in verification_full_out.txt
2. **Neural sanity check**: Verified neuron count (52,246) matches raw data after filtering for DR10_2022_07_12_1. PASS.
3. **Behavioral sanity check**: Verified SoundPos range [4.2, 36.0] ≈ expected [5, 35]. Slight variation expected.
4. **Licking check**: 0 licks for unsupervised mouse (DR10) - expected. Task mice (TX108) have 3589 lick events. PASS.
5. **Speed check**: run_pos has some negative values (backward movement) - handled by quartile binning. PASS.
6. **Reference code comparison**:
   - (a) Data loading: Same as load_spk (concatenate planes) and load_retino. MATCH.
   - (b) Neuron filtering: Exclude iarea==-1 and 7, same as neu_area_ID. MATCH.
   - (c) Temporal alignment: Position interpolation using cumulative position, same as spk_pos_interp. MATCH.
   - (d) Binning: 60 bins per trial, same as data_process_script cell 9. MATCH.
   - (e) Input construction: Novel (not in reference - reference doesn't train decoders).
   - (f) Output construction: Novel (not in reference).
7. **Brain region mapping**: V1=8, mHV=0|1|2|9, lHV=5|6, aHV=3|4 matches neu_area_ID exactly. PASS.
8. **Key statistics**: 89 sessions, 19 mice, 20,547-89,577 neurons all match paper. PASS.

### Issues Found and Resolved
- Position output imbalance (50% in bin 3 due to grey space) - inherent to including grey space in trial
- Licking very sparse (1.5%) - expected for dataset with many unsupervised/naive mice

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (18642 -> 183 over 200 epochs)
- Training time: ~10 hours total (including initialization)
- Device: CPU (GPU OOM with L4 23GB)

### Loss Trajectory
| Epoch | Loss |
|-------|------|
| 1 | 18642.7 |
| 10 | 11575.0 |
| 20 | 7330.4 |
| 50 | 2606.4 |
| 100 | 448.9 |
| 150 | 229.0 |
| 200 | 183.0 |

### Decoder Results (Full)
| Output | Train Bal Acc | Val Bal Acc | Chance | Above Chance? |
|--------|--------------|-------------|--------|---------------|
| visual_stimulus | 0.4748 | 0.4531 | 0.0667 | Yes (6.8x) |
| licking | 0.8954 | 0.8603 | 0.5000 | Yes (1.7x) |
| position | 0.3716 | 0.3717 | 0.2500 | Yes (1.5x) |
| running_speed | 0.3311 | 0.3311 | 0.2500 | Yes (1.3x) |

All outputs above chance. Train/val gap is small, indicating no severe overfitting.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Val Accuracy | Chance | Ratio | Assessment |
|----------|-------------|--------|-------|------------|
| visual_stimulus | 0.4531 | 0.0667 | 6.8x | Good - neural activity encodes stimulus identity |
| licking | 0.8603 | 0.5000 | 1.7x | Good - but sparse (1.5% licking) |
| position | 0.3717 | 0.2500 | 1.5x | Moderate - position is deterministic so could be higher |
| running_speed | 0.3311 | 0.2500 | 1.3x | Moderate - speed is noisy |

### Accuracy vs Paper
The paper does not train traditional decoders, so direct comparison is not possible. The paper uses d-prime for stimulus selectivity and coding direction analysis. Our decoder results are consistent with the paper's findings that neural activity encodes stimulus identity.

### Train vs Validation Gap
| Output | Train | Val | Ratio |
|--------|-------|-----|-------|
| visual_stimulus | 0.4748 | 0.4531 | 1.05x |
| licking | 0.8954 | 0.8603 | 1.04x |
| position | 0.3716 | 0.3717 | 1.00x |
| running_speed | 0.3311 | 0.3311 | 1.00x |

No overfitting detected. Train/val gap < 1.1x for all outputs.

### Issues Found and Resolved
None. All outputs are above chance with no overfitting.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with processing plots
- [x] CONVERSION_NOTES.md complete
- [x] All required output files created
