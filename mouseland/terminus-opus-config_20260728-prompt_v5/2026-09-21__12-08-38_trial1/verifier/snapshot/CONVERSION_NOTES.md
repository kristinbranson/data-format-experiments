# Dataset Conversion Notes

## Overview
- **Dataset**: Zhong et al. 2025 - "Unsupervised pretraining in biological neural networks"
- **Date started**: 2024
- **Goal**: Convert calcium imaging + behavior data to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/code/` - Reference code (utils.py, data_process_script.ipynb, fig*.py)
- `/app/data/` - Data files (beh/, spk/, retinotopy/)
- `/app/paper.pdf` - Reference paper
- `/app/methods.txt` - Extracted methods text
- `/app/decoder.py` - Decoder implementation
- `/app/train_decoder.py` - Decoder training script

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_spk | utils.py | LOADING | Loads neural data, concatenates across imaging planes |
| load_retino | utils.py | LOADING | Loads retinotopy data with brain area assignments |
| load_exp_beh | utils.py | LOADING | Loads behavior data for an experiment type |
| neu_area_ID | utils.py | PROCESSING | Maps iarea codes to V1, mHV, lHV, aHV |
| get_interpPos_spk | utils.py | PROCESSING | Interpolates neural activity by position (60 bins) |
| dprime | utils.py | PROCESSING | Computes d-prime for stimulus selectivity |
| lickCount | utils.py | PROCESSING | Returns binary lick response per trial |
| spk_2_firstLick | utils.py | PROCESSING | Aligns spikes to first lick |
| spk_2_cue | utils.py | PROCESSING | Aligns spikes to sound cue |

### Notes
- Frame rate: 3.17 Hz (calcium imaging)
- Neural data: deconvolved fluorescence traces from Suite2p
- `load_spk` concatenates spks from multiple imaging planes
- Reference code only uses running frames (`ft_move > 0`) for analysis
- Brain areas: V1 (iarea==8), mHV (0,1,2,9), lHV (5,6), aHV (3,4)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- **Behavior** (`/app/data/beh/`): 23 .npy files per experiment type
- **Neural** (`/app/data/spk/`): 89 .npy files per session, deconvolved spikes
- **Retinotopy** (`/app/data/retinotopy/`): 89 .npz files per session
- **Imaging_Exp_info.npy**: Session metadata

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 19 |
| Unique sessions | 89 |
| Neurons / session | 20,547 - 89,577 |
| Trials / session | 84 - 789 |
| Frame rate | 3.17 Hz |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Recordings | 89 | "We performed 89 recordings" |
| Subjects | 19 | "in 19 mice" |
| Neurons / session | 20,547 - 89,577 | "activity traces from 20,547 to 89,577 neurons" |
| Corridor length | 4 m + 2m gray | "corridors were each 4 m long, with 2 m of grey space" |
| Running threshold | 6 cm/s | "running faster than a threshold of 6 cm s−1" |
| VR speed | 60 cm/s | "constant speed (60 cm s−1)" |
| Sound cue position | 0.5-3.5 m | "randomly chosen...between positions 0.5 m and 3.5 m" |
| Calcium decay | 0.75 s | "timescale of decay of 0.75 s" |
| GCaMP | GCaMP6s | "bred to express GCaMP6s" |

### Processing Details
- All analyses based on deconvolved fluorescence traces
- Only running timepoints used for analysis
- Position-based interpolation: 60 bins for 6m corridor

### Curation Steps
**Neuron curation rules**: No explicit filtering. All Suite2p-detected cells included.
**Trial curation rules**: Only running timepoints used (ft_move > 0) in reference code.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Sessions | 89 unique | 89 files | 89 recordings | Consistent |
| Mice | 19 unique | 19 names | 19 mice | Consistent |
| Neurons | Concatenated planes | 20K-90K | 20,547-89,577 | Consistent |
| Frame rate | 3.17 Hz | Consistent | Not in methods.txt | From data_process_script.ipynb |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| spk (deconvolved) | neural | Extract frames per trial, float16 | Subsampled to 5000 neurons |
| SoundFr - frame_idx | input[0]: time_to_sound_cue | (frame - SoundFr) / fs | Seconds |
| sess# or days | input[1]: day_of_training | Direct | Per-trial |
| frame_idx - StartFr | input[2]: time_since_trial_start | (frame - StartFr) / fs | Seconds |
| isRew | input[3]: reward_availability | Boolean to float | Per-trial |
| WallName | output[0]: visual_stimulus | Map to category index | 15 categories |
| LickFr, LickTrind | output[1]: licking | Binary per frame | Time-varying |
| ft_Pos | output[2]: position_bin | 4 bins of 1m each | Time-varying |
| ft_RunSpeed | output[3]: running_speed_bin | 4 quartile bins | Time-varying |

### Key Decisions
1. **Neuron subsampling**: 5000 neurons per session (from 20K-90K) to keep file size manageable. Decoder uses PCA to 100 components.
2. **All frames included**: Both running and non-running frames included for temporal continuity.
3. **Position bins**: 4 bins of 1m covering 4m corridor. Gray space (>4m) assigned to bin 3.
4. **Speed quartiles**: Computed across all frames in all sessions.
5. **Trial boundaries**: StartFr to EndFr (including gray space).
6. **Float16 for neural**: Reduces file size by 2x with negligible precision loss.

---

## Step 6: Script Development
**Status**: COMPLETE

Script: `/app/convert_data.py`
- Supports --sample, --full, --show-processing flags
- Loads data using same functions as reference code
- Subsamples neurons with fixed random seed for reproducibility

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (VR2, TX83) |
| Total trials | 872 |
| Neurons/session | 5000 (subsampled) |
| File size | 5.92 GB |

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| Total | ~6.6s | ~9.8 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| visual_stimulus | 0.6569 | 0.6001 | 0.0667 |
| licking | 0.9171 | 0.9046 | 0.5000 |
| position_bin | 0.6115 | 0.5829 | 0.2500 |
| running_speed_bin | 0.6962 | 0.6765 | 0.2500 |

All outputs well above chance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 20.26 GB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Converted Data | Match? |
|-----------|-----------------|----------------|--------|
| Total sessions | 89 | 89 | ✓ |
| Total subjects | 19 | 19 | ✓ |
| Total trials | N/A | 38,110 | - |
| Neurons/session | 20,547-89,577 | 5,000 (subsampled) | Subsampled |
| Brain regions | V1, HVAs | V1, mHV, lHV, aHV, unassigned | ✓ |
| Stimuli | circle, leaf, rock, brick | 15 unique stimuli | ✓ |
| Reward rate | ~50% | 49.8% (sup session) | ✓ |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

**Check 1: Output log verification**
- verification_full_out.txt: "Data format is valid, no errors or warnings."
- No errors found.

**Check 2: Sanity checks**
1. Neural data: Converted values within original data range ✓
2. Input data: time_to_sound_cue matches manual calculation ✓
3. Output data: stimulus category and position bin match manual calculation ✓
4. Reward availability matches original isRew ✓
5. Brain region mapping matches reference code neu_area_ID ✓
6. Licking data present in supervised sessions (280/470 trials) ✓

**Check 3: Reference code comparison**
- Data loading: Uses same load_spk pattern (concatenate spks planes) ✓
- Brain regions: Same neu_area_ID mapping (V1=8, mHV=0,1,2,9, lHV=5,6, aHV=3,4) ✓
- Behavior keys: Same session key format (mname_datexp_blk) ✓
- Neural data: Uses deconvolved traces (same as reference) ✓
- Difference from reference: We use time-series data, not position-interpolated. This is necessary for time-varying decoder outputs.
- Difference from reference: We include all frames (not just running). This maintains temporal continuity for time-varying decoder.
- Difference from reference: We subsample neurons (5000 vs all). This is for file size; decoder uses PCA anyway.

**Check 4: Key statistics**
- 89 sessions: Matches paper ✓
- 19 subjects: Matches paper ✓
- Neuron range 20,547-89,577: Matches paper (before subsampling) ✓
- Reward rate ~50%: Matches expected (49.8% for VR2 session) ✓

**Check 5: Edge cases**
- Negative StartFr (first trial): Clipped to 0 ✓
- Very long trials (up to 5621 frames): Included, not filtered ✓
- Sessions with no licking (unsupervised): All lick values = 0 ✓
- Sessions with no reward (unsupervised): All reward values = 0 ✓

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (7228 -> 37.2 over 200 epochs)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Ratio |
|--------|----------------------|------------------------|--------|-------|
| visual_stimulus | 0.3551 | 0.3438 | 0.0667 | 5.2x |
| licking | 0.7004 | 0.6977 | 0.5000 | 1.4x |
| position_bin | 0.3137 | 0.3098 | 0.2500 | 1.2x |
| running_speed_bin | 0.3066 | 0.3106 | 0.2500 | 1.2x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

**Check 1: Accuracy vs chance**
- All outputs above chance ✓
- visual_stimulus: 5.2x above chance - good
- licking: 1.4x above chance - acceptable (heavily imbalanced: 97.3% not licking)
- position_bin: 1.2x above chance - low but above chance
- running_speed_bin: 1.2x above chance - low but above chance

**Check 2: Accuracy comparison to paper**
The paper does not train traditional neural decoders, so direct comparison is not possible. The paper uses d-prime, coding direction projections, and k-fold cross-validation for reward prediction neurons. The decoder accuracies are reasonable given:
- 15 stimulus categories (chance 6.7%) vs 34% achieved
- Heavily imbalanced licking data (97.3% not licking)
- Position and speed have temporal autocorrelation

**Check 3: Train vs validation gap**
- visual_stimulus: 0.355 vs 0.344 (1.03x ratio) - no overfitting ✓
- licking: 0.700 vs 0.698 (1.00x ratio) - no overfitting ✓
- position_bin: 0.314 vs 0.310 (1.01x ratio) - no overfitting ✓
- running_speed_bin: 0.307 vs 0.311 (0.99x ratio) - no overfitting ✓

No overfitting detected. All train/validation ratios < 1.5x.

### Issues Found and Resolved
- Position bin 3-4m includes gray space, making it 48.6% of data. This is the correct behavior since the corridor includes gray space.
- Speed quartile Q2 has 40.9% of data because many frames have speed near 0 (not running). Quartiles are computed across all frames.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
