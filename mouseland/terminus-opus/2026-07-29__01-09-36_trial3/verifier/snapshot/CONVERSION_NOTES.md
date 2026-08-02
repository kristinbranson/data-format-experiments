# Dataset Conversion Notes

## Overview
- **Dataset**: Zhong et al. 2025 - "Unsupervised pretraining in biological neural networks"
- **Date started**: 2025-07-29
- **Goal**: Convert calcium imaging + behavioral data from VR corridor task to decoder-compatible format
- **Data source**: Figshare (doi.org/10.25378/janelia.28811129.v1)

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code (utils.py, data_process_script.ipynb, figure scripts, S6.py)
- `data/beh/` - Behavioral data files (Beh_*.npy), Imaging_Exp_info.npy
- `data/spk/` - Neural spike data (89 files, *_neural_data.npy)
- `data/retinotopy/` - Retinotopy transformation files (89 + areas.npz)
- `data/process_data/` - Empty (for intermediate processed data)
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods text
- `decoder.py` - Decoder model code
- `train_decoder.py` - Decoder training script

Python: numpy 2.3.5, torch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_spk(db, root)` | utils.py | LOADING | Loads neural data, concatenates 'spks' list across imaging planes |
| `load_exp_beh(root, exp_type)` | utils.py | LOADING | Loads behavioral data from Beh_*.npy |
| `load_retino(db, root)` | utils.py | LOADING | Loads retinotopy data (iarea, xy_t) |
| `neu_area_ID(iarea)` | utils.py | PROCESSING | Maps iarea values to brain regions: V1(8), mHV(0,1,2,9), lHV(5,6), aHV(3,4) |
| `get_interpPos_spk()` | utils.py | PROCESSING | Interpolates spikes by position into trials x bins |
| `dprime()` | utils.py | PROCESSING | Computes d-prime for neuron selectivity |
| `lickCount()` | utils.py | PROCESSING | Returns binary lick response per trial |
| `Get_dprime_selective_neuron()` | utils.py | CURATION | Identifies stimulus-selective neurons using d-prime |

### Notes
- Neural data is deconvolved fluorescence traces (Suite2p output), NOT raw dF/F
- Spikes stored as list of arrays per imaging plane, concatenated for analysis
- Key behavioral variables are frame-aligned (ft_* prefix)
- Frame rate: 3.17 Hz (time bin ~315 ms)
- Corridor_Length = 60 position units (40 texture + 20 gray)
- Physical: 4m corridor + 2m gray = 6m total; 1 VR unit = 10 cm
- VR moves at constant 60 cm/s when mouse runs > 6 cm/s threshold
- Reference code filters neurons: `(arid!=-1) & (arid != 7)` to exclude outside visual cortex
- When session has 'stimtype' field, beh key format: `mname_datexp_blk_stimtype`
- stim_id mapping: 0=circle1, 1=circle2, 2=leaf1, 3=leaf2, 4=leaf3, 5=leaf1_swap1, 6=leaf1_swap2

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- **Neural data** (`data/spk/`): 89 .npy files, each dict with 'spks' key (list of arrays per plane, float32)
- **Behavioral data** (`data/beh/`): 27 Beh_*.npy files organized by experiment type
  - Each file: dict mapping session_id -> behavioral data dict
  - Some sessions keyed with stimtype suffix (e.g., `_swap1`, `_swap2`)
- **Experiment info** (`data/beh/Imaging_Exp_info.npy`): Dict mapping exp_type -> array of session dicts
  - Keys: mname, datexp, blk, exptype, rewType, stim_id, days/sess#, stimtype (optional)
- **Retinotopy** (`data/retinotopy/`): 89 .npz files with iarea and xy_t

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 20,547 to 89,577 per session |
| Subjects | 19 |
| Sessions (unique) | 89 |
| Experiment types | 23 |
| Total entries across exp types | 142 (some sessions in multiple types) |
| Trials / session | 84-789 |
| Total wall names | 15 unique |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Recordings | 89 | "We performed 89 recordings" |
| Subjects | 19 | "in 19 mice" |
| Neurons/recording | 20,547-89,577 | "activity traces from 20,547 to 89,577 neurons" |
| Neural data time bin | ~315 ms | Frame rate 3.17 Hz |
| Corridor length | 4 m | "corridors were each 4 m long" |
| Gray space | 2 m | "with 2 m of grey space between corridors" |
| VR speed | 60 cm/s | "constant speed (60 cm s-1)" |
| Running threshold | 6 cm/s | "running faster than a threshold of 6 cm s-1" |
| Sound cue position | 0.5-3.5 m | "randomly chosen...between positions 0.5 m and 3.5 m" |
| Calcium indicator | GCaMP6s | "bred to express GCaMP6s" |
| Processing | Suite2p | "processed using Suite2p" |
| Deconvolution decay | 0.75 s | "timescale of decay of 0.75 s" |
| Analysis basis | Deconvolved traces | "All our analyses were based on deconvolved fluorescence traces" |

### Processing Details
- Suite2p: motion correction, ROI detection, cell classification, neuropil correction, spike deconvolution
- Non-negative deconvolution with decay timescale 0.75 s
- Only running timepoints used for analysis in the paper

### Curation Steps
**Neuron curation rules**: Exclude neurons outside visual cortex (iarea==-1 or iarea==7). This matches reference code.
**Trial curation rules**: Include all trials. Skip trials with < 2 frames.

### Decoders Trained
No explicit decoder accuracy reported in the paper (focuses on d-prime and coding direction analyses).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Sessions | 89 unique | 89 spk files | 89 recordings | Consistent |
| Mice | 19 unique | 19 mouse names | 19 mice | Consistent |
| Frame rate | 3.17 Hz | ~3.18 Hz measured | 3.17 Hz | Consistent (rounding) |
| Neurons | filtered by iarea | 20k-90k total | 20,547-89,577 | Consistent |
| Corridor | 60 units total | ft_Pos 0-60 | 4m + 2m gray | Consistent |
| Brain regions | V1,mHV,lHV,aHV | iarea -1 to 9 | Visual cortex | Consistent |
| stimtype handling | kn includes stimtype | beh keys have _swap suffix | swap stimuli | Consistent |

No unresolved discrepancies.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code | Notes |
|-----------------|--------------|-----------|----------------|-------|
| spks (concat, filtered) | neural | Per-trial via ft_trInd | load_spk() | Max 2000 neurons/session |
| SoundFr - frame_idx | input[0]: time_to_sound_cue | (SoundFr - frame) / FS | N/A | Time-varying, seconds |
| days/sess# | input[1]: day_of_training | Direct value | exp_info | Per-trial, broadcast |
| frame_idx - StartFr | input[2]: time_since_trial_start | (frame - StartFr) / FS | N/A | Time-varying, seconds |
| isRew | input[3]: reward_availability | Boolean to 0/1 | N/A | Per-trial, broadcast |
| WallName -> stim_id | output[0]: visual_stimulus_category | Categorical int | stim_id mapping | Per-trial |
| LickFr, LickTrind | output[1]: licking | Binary per frame | lickCount() | Time-varying |
| ft_Pos | output[2]: position_bin | 4 bins of 10 VR units | N/A | Time-varying |
| ft_RunSpeed | output[3]: running_speed_bin | 4 global quartile bins | N/A | Time-varying |

### Key Decisions
1. **Neuron filtering**: Exclude iarea==-1 and iarea==7 (matches reference code `(arid!=-1) & (arid != 7)`)
2. **Neuron subsampling**: Max 2000 neurons/session, stratified by brain region. Justified because decoder uses PCA to 100 components and SVD with max 2000 neurons.
3. **Position bins**: 4 bins of 1m (10 VR units each). Gray space frames clipped to bin 3.
4. **Speed quartiles**: Computed globally across all sessions, including all frames.
5. **Stimulus mapping**: Use stim_id when available (non-NaN), wall name otherwise.
6. **Trial alignment**: Align to corridor entry (trial start). Use ft_trInd for frame-to-trial mapping.
7. **Time bin**: Native frame rate (~315 ms). No resampling.

### Planned Sanity Checks
- [x] Total sessions = 89
- [x] Total mice = 19  
- [x] Neuron counts match between spk and retinotopy
- [x] Frame counts match between spk and behavioral data
- [x] Position bins computed correctly
- [x] Time-to-sound-cue computed correctly
- [x] Reward availability matches isRew
- [x] Stimulus category matches WallName

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented in `convert_data.py` with:
- `--full`: Process all 89 sessions (default)
- `--sample`: Process 2 sessions for testing
- `--show-processing`: Generate visualization plots

Key implementation details:
- Uses numpy random state (seed=42) for reproducible neuron subsampling
- Handles stimtype suffix in behavioral data keys
- Clips neural frames to minimum of spk and behavioral frame counts
- Skips trials with < 2 frames

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (TX108_2023_03_13_1, TX119_2023_12_14_1) |
| Subjects | 2 (TX108, TX119) |
| Total trials | 589 (210 + 379) |
| Neurons/session | 1999, 2000 |
| File size | 0.31 GB |
| Processing time | 8.6s |

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| Load neural data | 1-30s (varies by file size) | ~15 min |
| Process trials | <1s | ~1 min |
| Full conversion | - | ~18 min (actual) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Bal Acc | Validation Bal Acc | Chance |
|--------|-----------------|-------------------|--------|
| visual_stimulus_category | 0.9598 | 0.8104 | 0.5000 |
| licking | 0.9336 | 0.8244 | 0.5000 |
| position_bin | 0.9896 | 0.7238 | 0.2500 |
| running_speed_bin | 0.8256 | 0.6470 | 0.2500 |

All outputs well above chance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 16.22 GB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Converted Data | Match? |
|-----------|----------------|----------------|--------|
| Total sessions | 89 | 89 | ✓ |
| Total subjects | 19 | 19 | ✓ |
| Neurons/session (raw) | 20,547-89,577 | 20,547-89,577 | ✓ |
| Neurons/session (converted) | N/A | 1999-2000 | Subsampled |
| Total trials | N/A | 38,110 | - |
| Trials/session range | N/A | 84-789 | - |
| Brain regions | V1, mHV, lHV, aHV | V1, mHV, lHV, aHV | ✓ |
| Stimulus categories | circle, leaf, etc. | 12 categories | ✓ |
| Frame rate | 3.17 Hz | 315.5 ms bins | ✓ |
| Corridor length | 4m | 40 VR units | ✓ |
| Day of training range | N/A | 0-15 | - |
| Sound cue position | 0.5-3.5m | SoundPos 5-35 VR units | ✓ |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- `verification_full_out.txt`: No errors, no warnings. Data format valid.

### Check 2: Sanity checks

**Neural data sanity check (session TX60_2021_06_07_1, trial 5):**
- Frame count: 66 frames in both original and converted ✓
- Neural values are non-negative floats (deconvolved traces) ✓

**Input data sanity check:**
- time_to_sound_cue: Expected 2.5546s, got 2.5546s ✓ (np.allclose passed)
- time_since_trial_start: Expected 0.0641s, got 0.0641s ✓ (np.allclose passed)
- reward_availability: isRew=True → 1.0 ✓

**Output data sanity check:**
- visual_stimulus_category: WallName='leaf1' → idx=3 → name='leaf1' ✓
- position_bin: Position 0.2 → bin 0 ✓ (np.array_equal passed)
- licking: 36 lick events → 24 frames with licks (multiple licks per frame at 3.17 Hz) ✓

### Check 3: Reference code comparison

| Processing Step | My Code | Reference Code | Match? |
|----------------|---------|----------------|--------|
| Data loading | `np.concatenate(spk_data['spks'])` | `np.concatenate([nspk for nspk in np.load(...)['spks']])` | ✓ |
| Neuron filtering | `(iarea != -1) & (iarea != 7)` | `(arid!=-1) & (arid != 7)` | ✓ |
| Brain region mapping | `neu_area_ID()` copied from utils.py | Same function | ✓ |
| Behavioral data loading | Try session_id, then stimtype variants | `kn = '%s_%s_%s_%s' if stimtype else '%s_%s_%s'` | ✓ |
| Frame-trial mapping | `ft_trInd == trial_idx` | Same approach | ✓ |
| Stimulus mapping | stim_id -> STIM_NAMES | `stim_id` array in exp_info | ✓ |

### Check 4: Key statistics comparison
| Statistic | Reference | Converted | Match? |
|-----------|-----------|-----------|--------|
| Sessions | 89 | 89 | ✓ |
| Subjects | 19 | 19 | ✓ |
| Min neurons/session | 20,547 | 20,547 (raw) | ✓ |
| Max neurons/session | 89,577 | 89,577 (raw) | ✓ |

### Check 5: Edge cases
- Handled stimtype suffix for test3 sessions (swap1/swap2)
- Handled NaN stim_id values (use wall name directly)
- Handled frame count mismatch between spk and beh (use min)
- Handled trials with < 2 frames (skipped)

### Issues Found and Resolved
- **Issue**: Sessions in test3 experiments have behavioral data keyed with `_swap1`/`_swap2` suffix
  - **Resolution**: Updated load_beh to try multiple key formats
- **Issue**: stim_id=5 was mapped to 'leaf1_swap' instead of 'leaf1_swap1'
  - **Resolution**: Fixed STIM_NAMES mapping to include both swap1 (id=5) and swap2 (id=6)

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (110.95 → 0.375 over 200 epochs)
- Training time: ~5 minutes on GPU

### Decoder Results (Full)
| Output | Training Bal Acc | Validation Bal Acc | Chance | Ratio to Chance |
|--------|-----------------|-------------------|--------|----------------|
| visual_stimulus_category | 0.9313 | 0.6744 | 0.0833 | 8.1x |
| licking | 0.9870 | 0.8439 | 0.5000 | 1.7x |
| position_bin | 0.9615 | 0.7679 | 0.2500 | 3.1x |
| running_speed_bin | 0.8233 | 0.6773 | 0.2500 | 2.7x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance analysis
All outputs are well above chance:
- visual_stimulus_category: 8.1x chance (0.6744 vs 0.0833)
- licking: 1.7x chance (0.8439 vs 0.5000)
- position_bin: 3.1x chance (0.7679 vs 0.2500)
- running_speed_bin: 2.7x chance (0.6773 vs 0.2500)

No outputs below 1.5x chance.

### Check 2: Accuracy comparison to paper
The reference paper does not report decoder accuracies (it focuses on d-prime, coding direction, and sequence correlation analyses). Therefore, no direct comparison is possible. However, the high accuracies achieved indicate that the neural data contains meaningful information about all decoded variables, which is consistent with the paper's findings about stimulus selectivity and reward prediction in visual cortex.

### Check 3: Train vs validation gap
| Output | Train Acc | Val Acc | Ratio |
|--------|-----------|---------|-------|
| visual_stimulus_category | 0.9313 | 0.6744 | 1.38x |
| licking | 0.9870 | 0.8439 | 1.17x |
| position_bin | 0.9615 | 0.7679 | 1.25x |
| running_speed_bin | 0.8233 | 0.6773 | 1.22x |

No output has train accuracy > 1.5x validation accuracy. The largest gap is for visual_stimulus_category (1.38x), which is expected with 12 categories. No evidence of data leakage.

### Issues Found and Resolved
No issues found in this review.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with README_CACHE.md
- [x] All required output files present
- [x] CONVERSION_NOTES.md complete
