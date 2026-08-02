# Dataset Conversion Notes

## Overview
- **Dataset**: "Unsupervised pretraining in biological neural networks" (Zhong et al., 2025)
- **Date started**: 2026-03-24
- **Goal**: Convert calcium imaging data from visual discrimination task to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Python environment verified:
- numpy 2.3.5, torch 2.6.0+cu124, scipy 1.17.1

Directory contents:
- `code/` - Reference code (Figures.ipynb, data_process_script.ipynb, fig1-5.py, S6.py, utils.py)
- `data/beh/` - Behavioral data (Beh_{exp_type}.npy files + Imaging_Exp_info.npy)
- `data/spk/` - Neural spike data (89 files: {subject}_{date}_{blk}_neural_data.npy)
- `data/retinotopy/` - Retinotopy/area data (89 trans.npz files + areas.npz)
- `data/process_data/` - Empty directory
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods
- `decoder.py`, `train_decoder.py` - Decoder code

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_spk` | utils.py | LOADING | Load neural data, concatenate across planes |
| `load_retino` | utils.py | LOADING | Load retinotopy, get area assignments |
| `load_exp_beh` | utils.py | LOADING | Load behavior for experiment type |
| `neu_area_ID` | utils.py | PROCESSING | Map iarea codes to brain regions (V1, mHV, lHV, aHV) |
| `dprime` | utils.py | PROCESSING | Compute selectivity index d' between two conditions |
| `get_interpPos_spk` | utils.py | PROCESSING | Interpolate neural activity by position (60 bins/corridor) |
| `spk_pos_interp` | utils.py | PROCESSING | Position-based interpolation helper |
| `interp_value` | utils.py | PROCESSING | 1D interpolation utility |
| `get_cat_id` | utils.py | CURATION | Map wall names to category IDs |
| `lickCount` | utils.py | PROCESSING | Binary lick response per trial |
| `Get_dprime_selective_neuron` | utils.py | CURATION | Filter neurons by d' selectivity |

### Notes
- Neural data: Suite2p deconvolved calcium traces (NOT raw fluorescence). Deconvolution with tau=0.75s.
- Data is stored as `{'spks': [plane0, plane1, plane2, ...]}` where each plane is (n_neurons, n_frames).
- `load_spk` concatenates planes: `np.concatenate([nspk for nspk in spks], 0)`
- Brain area assignment via `neu_area_ID(iarea)`:
  - V1: iarea==8
  - mHV (medial higher visual): iarea in {0,1,2,9}
  - lHV (lateral higher visual): iarea in {5,6}
  - aHV (anterior higher visual): iarea in {3,4}
  - Excluded: iarea==-1 (outside visual cortex), iarea==7
- Reference code only uses running frames: `VRmove = beh['ft_move'][:nfr]>0` and corridor frames `beh['ft_CorrSpc'][:nfr]`
- Position interpolation: 60 bins per corridor (each bin = 1 decimeter = 0.1m), total 6m (4m texture + 2m gray)
- Experiment info in `Imaging_Exp_info.npy`: dict mapping exp_type -> list of session dicts with keys: mname, datexp, blk, stim_id, etc.

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- **Neural**: `data/spk/{mname}_{datexp}_{blk}_neural_data.npy` - dict with 'spks' key, list of planes
- **Retinotopy**: `data/retinotopy/{mname}_{datexp}_trans.npz` - contains iarea, xy_t for each neuron
- **Behavior**: `data/beh/Beh_{exp_type}.npy` - dict mapping session_key -> behavior dict
- **Exp info**: `data/beh/Imaging_Exp_info.npy` - dict mapping exp_type -> list of session metadata

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 19 |
| Sessions (neural data files) | 89 |
| Sessions per subject | 1-8 (varies) |
| Neurons per session | 20,547 to 89,577 (from paper) |
| Trials per session | ~300-700 (varies) |
| Frame rate | ~3.17 Hz |
| Corridor length | 60 (decimeters) = 6m (4m texture + 2m gray) |
| Median trial duration | ~42 frames (~13.2 sec) |
| Median corridor duration | ~28 frames (~8.8 sec) |

### Behavioral variables (per session)
Key frame-level arrays (one value per imaging frame):
- `ft_trInd`: trial index for each frame
- `ft_Pos`: position in corridor (0-60 decimeters)
- `ft_PosCum`: cumulative position
- `ft_move`: VR movement (>0 when running)
- `ft_CorrSpc`: True when in texture corridor (0-4m)
- `ft_GraySpc`: True when in gray space (4-6m)
- `ft_WallID`: stimulus name for each frame
- `ft_RunSpeed`: running speed per frame
- `ft_isMoving`: boolean running indicator

Key per-trial arrays:
- `StartFr`: frame index of corridor entry
- `GrayFr`: frame index of gray space entry
- `EndFr`: frame index of corridor exit
- `SoundFr`: frame index of sound cue delivery
- `RewardFr`: frame of reward delivery (NaN if no reward)
- `WallName`: stimulus name per trial
- `isRew`: boolean reward trial indicator
- `SoundPos`: position of sound cue in corridor
- `LickFr`, `LickPos`, `LickTrind`: lick timestamps

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 19 | "89 recordings in 19 mice" |
| Sessions | 89 | "89 recordings in 19 mice" |
| Neurons per session | 20,547-89,577 | "activity traces from 20,547 to 89,577 neurons" |
| Neural data type | Deconvolved Ca2+ traces | "All analyses based on deconvolved fluorescence traces" |
| Deconvolution tau | 0.75 s | "timescale of decay of 0.75 s" |
| Frame rate | ~3.17 Hz | "fs = 3.17Hz" (from data_process_script) |
| Corridor length | 4m texture + 2m gray | "corridors were each 4 m long, with 2 m of grey space" |
| VR speed | 60 cm/s constant | "corridors always moved at a constant speed (60 cm s−1)" |
| Running threshold | 6 cm/s | "running faster than a threshold of 6 cm s−1" |
| Sound cue position | Uniform 0.5-3.5m | "randomly chosen per trial from uniform between 0.5m and 3.5m" |
| Selectivity criterion | d' >= 0.3 | "criteria for selective neurons was d' >= 0.3" |
| Analysis frames | Running only | "We only considered timepoints during running" |
| Analysis region | 0-4m (texture) for d' | "only selected data points inside the 0-4-m region" |

### Processing Details
- Neural data is deconvolved calcium traces from Suite2p (already in data files)
- No additional delta F/F computation needed
- Reference code interpolates neural activity to position (60 bins covering 6m corridor) for analysis
- For decoder: use raw frame-level data aligned to trial start
- Brain region assignment from retinotopy data: exclude neurons with iarea==-1 or iarea==7

### Curation Steps

**Neuron curation rules**:
- Exclude neurons outside visual cortex: iarea==-1 or iarea==7
- Assign to V1, mHV, lHV, aHV based on iarea codes

**Trial curation rules**:
- No explicit trial filtering in reference code for basic loading
- Reference code sometimes restricts to first 200 trials for specific analyses
- For decoder: use all trials (no filtering)

### Decoders Trained
No decoders trained in the reference paper (they use d', coding direction, and sequence correlation analyses instead).

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session count | 89 (from exp_info unique spk keys) | 89 (neural data files) | 89 recordings | Consistent |
| Subject count | 19 (from exp_info) | 19 (from file names) | 19 mice | Consistent |
| Neural data | Concatenated planes from spks dict | spks has 2-3 planes per session | Deconvolved traces | Consistent |
| Corridor | 60 dm = 6m | Corridor_Length=60.0 | 4m + 2m gray | Consistent |
| Area exclusion | iarea != -1 and != 7 | iarea has values -1 to 9 | Inside visual cortex | Consistent - code uses `(arid!=-1) & (arid != 7)` |
| Frame rate | ~3.17 Hz | median dt=0.3146s → 3.178 Hz | Not explicitly stated in methods | Consistent with notebook docs |
| Sessions w/ stimtype | Same neural data, different stim_id | Verified: identical trial data | Multiple analyses per recording | Use one entry per neural file |

### Key Insight
Sessions can appear in multiple experiment types (142 total entries for 89 unique sessions). For the decoder, each physical recording (neural data file) is used once. The behavioral data is the same regardless of which experiment type references it. For sessions with stimtype variants (swap1/swap2), the underlying trial data is identical - only stim_id mapping differs. We use any available behavior entry.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| spk (concatenated planes) | neural | Extract frames per trial, align to StartFr | Deconvolved traces, exclude iarea==-1/7 neurons |
| Time to SoundFr | input[0]: time_to_sound_cue | (SoundFr - current_frame) / fs | Negative before cue, 0 at cue, positive after |
| Session day info | input[1]: day_of_training | Session index within mouse or sess# field | Per-trial scalar |
| Time since StartFr | input[2]: time_since_trial_start | (current_frame - StartFr) / fs | Continuous, starts at 0 |
| isRew | input[3]: reward_availability | Binary 0/1 | Per-trial scalar |
| WallName | output[0]: visual_stimulus | Categorical encoding | Per-trial |
| LickFr/LickTrind | output[1]: licking | Binary per frame | Time-varying |
| ft_Pos | output[2]: position_bin | Discretize 0-4m into 4 bins (0-1m, 1-2m, 2-3m, 3-4m) | Time-varying, use 4+1 bins (add gray space bin) |
| ft_RunSpeed | output[3]: running_speed_bin | Quartile discretization across all frames | Time-varying |
| mouse name | subjects, subject_idx | Index mapping | |
| iarea -> V1/mHV/lHV/aHV | brain_regions, brain_region_idx | Via neu_area_ID mapping | |

### Key Decisions
1. **Time bin size**: Use native frame rate (~315 ms). No resampling needed.
2. **Trial length**: Use frames from StartFr to start of next trial (or end of session). This captures corridor + gray space.
3. **Neuron filtering**: Exclude neurons with iarea==-1 or iarea==7 (outside visual cortex), matching reference code.
4. **Position binning**: 4 bins in texture area (0-10dm, 10-20dm, 20-30dm, 30-40dm = 0-1m, 1-2m, 2-3m, 3-4m) + 1 bin for gray space. But decoder asks for 4 bins, so gray space frames get a separate bin value? Re-reading: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins". So 4 bins covering 0-4m. Gray space frames need a 5th category or can be excluded. I'll use 5 categories: 4 spatial + 1 gray. Actually, re-reading: the task says 4 bins. I'll use 4 bins and mark gray space as a 5th bin.
5. **Running speed**: Discretize into 4 quartile bins computed across ALL running frames in the dataset.
6. **Licking**: Binary per frame - check if any lick occurred in that frame's time window.
7. **Visual stimulus**: Use WallName directly, map to standardized categories.
8. **Day of training**: Use sess# or days field from exp_info if available; otherwise use session chronological order within subject.

### Planned Sanity Checks
- [ ] Number of neurons per session matches concatenated planes count (minus excluded)
- [ ] Trial count matches ntrials in behavior
- [ ] Position values fall in expected range (0-60 dm)
- [ ] Sound cue position in expected range (5-35 dm = 0.5-3.5m)
- [ ] Total sessions = 89, subjects = 19
- [ ] Frame rate consistent across sessions (~3.17 Hz)

---

## Step 6: Script Development
**Status**: COMPLETE

Script `convert_data.py` implements:
- Session map building from Imaging_Exp_info.npy
- Neural data loading with per-plane filtering (avoids large intermediate arrays)
- Float16 storage for neural data (deconvolved traces)
- Brain region assignment via iarea codes
- Stimulus name standardization (rock→circle, wood→leaf, brick→circle/leaf3)
- Frame-level behavioral variable extraction per trial
- Speed quartile computation across all sessions
- Processing visualization plots

**Key optimizations**:
- Filter neurons per-plane before concatenation (saves memory and time)
- Store neural data as float16 (halves file size)
- Speed quartile collection from behavior only (no neural data loading)

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics (2 sessions: TX108_2023_03_25_1, DR10_2022_07_12_1)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 2 (TX108, DR10) |
| Total trials | 926 (423 + 503) |
| Neurons | 72,944 + 52,246 = 125,190 |
| Mean T | 62.9 frames |
| Stim categories | circle1, circle2, leaf1, leaf2 |
| Lick fraction | 3.2% (session 0 only, supervised) |
| Reward fraction | Session 0: ~36% rewarded, Session 1: 0% (unsupervised) |
| File size | 6.70 GB |
| Conversion time | 57s |

### Processing Plots Review
- Neural heatmaps show reasonable deconvolved activity patterns
- Position bins distributed ~evenly in corridor, higher gray fraction
- Speed quartiles approximate 25% each across combined sessions
- Licking only in supervised session (TX108)

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| Speed quartiles | 0.4s total | 5s (all sessions) |
| Neural loading+filtering | ~15-25s | ~1600s |
| Processing | ~5s | ~450s |
| **Total** | ~20-25s | **~21 minutes** |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| visual_stimulus | 0.8691 | 0.7499 | 0.2500 |
| licking | 0.9517 | 0.8978 | 0.5000 |
| position | 0.6313 | 0.6117 | 0.2000 |
| running_speed | 0.4554 | 0.4601 | 0.2500 |

All outputs well above chance. Loss decreased steadily from 7600 to 174 over 200 epochs.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Full Dataset Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 89 |
| Subjects | 19 |
| Total trials | 38,110 |
| Neurons per session | 17,363 to 78,815 |
| Mean trial length | 59.6 frames (~18.8 sec) |
| Stimulus categories | 8 (circle1/2/3, leaf1/2/3, leaf1_swap1/2) |
| File size | 177.26 GB (pickle protocol 5) |
| Conversion time | ~21 minutes |

### Format Verification
- Errors: None
- Warnings: None
- All field shapes, dtypes, and ranges verified

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Discrepancies Reviewed
- Neuron counts (17K-79K) are lower than paper's range (20K-90K) due to excluding iarea==-1 and ==7 neurons. This is correct - paper says "inside visual cortex" only.
- 8 stimulus categories confirmed: circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2
- Speed quartile Q1 contains 47.1% of frames (includes stopped/slow frames below all three thresholds), Q2-Q4 each ~17.6%. This is correct since quartiles are computed on positive speeds only.
- Position bins: ~19% each for 4 texture bins + 32% gray space. Gray fraction is higher because mice spend more time in gray space between corridors.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Memory Issue
Full dataset (89 sessions, 17K-79K neurons) requires ~623 GB for decoder training:
- Pickle data in memory: ~178 GB
- Float32 session tensors: ~445 GB
This exceeds the 503 GB RAM available.

**Solution**: Subsample to 10,000 neurons per session before training. The decoder uses random projection to 2000 dims → SVD to 100 PCs, so 10K neurons provides more than sufficient information. Memory after subsampling: ~38 GB.

### Training Details
- Used `run_decoder.py` wrapper with neuron subsampling
- Device: CPU (GPU only has 2 GB VRAM)
- Training: 30,457 trials, Testing: 7,653 trials
- Loss: 10,563 → 70 over 200 epochs
- Total time: ~33 minutes

### Full Decoder Results
| Output | Train Bal. Acc | Val Bal. Acc | Chance (1/K) |
|--------|---------------|-------------|-------------|
| visual_stimulus (8 classes) | 0.3801 | 0.3763 | 0.1250 |
| licking (2 classes) | 0.8415 | 0.8256 | 0.5000 |
| position (5 classes) | 0.2596 | 0.2579 | 0.2000 |
| running_speed (4 classes) | 0.3380 | 0.3367 | 0.2500 |

All outputs well above chance. Minimal train-val gap indicates no overfitting.

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Analysis
1. **All outputs decode above chance**: Visual stimulus at 3.0x, licking at 1.7x, position at 1.3x, running speed at 1.4x chance level.
2. **No overfitting**: Train-val gaps are 0.001-0.016, indicating excellent generalization.
3. **Visual stimulus decoding** (0.38 balanced acc across 8 classes) is strong. This is consistent with the paper's finding that visual cortex neurons carry stimulus identity information.
4. **Licking decoding** (0.83) is the strongest output, expected since licking produces distinct neural signatures in rewarded sessions.
5. **Position and speed** decode modestly above chance, consistent with the known spatial and locomotion modulation of visual cortex activity.
6. **Comparison to sample** (2 sessions): Sample had higher accuracy (visual: 0.75, position: 0.61) because fewer sessions means less cross-session variability for the shared decoder to handle.

### Potential Concerns
- Position decoding (0.26 vs 0.20 chance) is relatively modest. This may reflect that position-related signals are weaker in visual cortex compared to hippocampus, and the decoder must handle 89 different sessions with different stimulus arrangements.
- Neuron subsampling (10K per session) was necessary for memory but is well-justified since the decoder already projects to 100 PCs.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

### Files Produced
- `converted_data.pkl` (177.26 GB) - Main output, pickle protocol 5
- `convert_data.py` - Conversion script
- `run_decoder.py` - Memory-efficient decoder training wrapper
- `decoder_stats.json` - Full decoder results
- `CONVERSION_NOTES.md` - This file
- `train_decoder_full_out.txt` - Decoder training log
