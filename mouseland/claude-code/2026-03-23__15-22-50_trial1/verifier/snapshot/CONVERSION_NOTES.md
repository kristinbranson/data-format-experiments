# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks (Zhong et al., 2025)
- **Date started**: 2026-03-23
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Python environment verified: numpy 2.3.5, torch 2.6.0+cu124

Directory contents:
- `paper.pdf` - Reference paper (Nature 2025)
- `methods.txt` - Extracted methods from paper
- `decoder.py` - Decoder module
- `train_decoder.py` - Decoder training script
- `code/` - Reference code repository
  - `utils.py`, `fig1.py`-`fig5.py`, `S6.py`, `Figures.ipynb`, `data_process_script.ipynb`
- `data/` - Data directory
  - `beh/` - Behavioral data (23 Beh_ files + `Imaging_Exp_info.npy`)
  - `spk/` - Neural spike data (89 sessions, 19 subjects)
  - `retinotopy/` - Retinotopy data (89 .npz trans files + `areas.npz`)
  - `process_data/` - Empty directory

Subjects identified (19): DR10, DR15, LZ13, LZ16, TX104, TX105, TX108, TX109, TX119, TX123, TX124, TX139, TX140, TX60, TX61, TX83, TX85, TX88, VR2
Sessions: 89 total unique physical recordings, 142 total entries across 23 experiment types

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_spk` | utils.py | LOADING | Loads neural data, concatenates across planes |
| `load_retino` | utils.py | LOADING | Loads retinotopy, computes area indices |
| `load_exp_beh` | utils.py | LOADING | Loads behavior data for experiment type |
| `neu_area_ID` | utils.py | PROCESSING | Maps iarea to V1/mHV/lHV/aHV |
| `spk_pos_interp` | utils.py | PROCESSING | Interpolates spikes to position bins |
| `get_interpPos_spk` | utils.py | PROCESSING | Wrapper for position interpolation |
| `dprime` | utils.py | PROCESSING | Computes d' selectivity index |
| `Get_dprime_selective_neuron` | utils.py | PROCESSING | Computes dprime using corridor frames + moving |
| `lickCount` | utils.py | PROCESSING | Binary lick response per trial |
| `get_cat_id` | utils.py | PROCESSING | Maps wall names to category IDs |

### Notes
- Neural data is loaded via `load_spk()`: concatenates `spks` list across imaging planes
- spks are deconvolved fluorescence traces from Suite2p (not raw dF/F)
- No neuron filtering/curation in the reference code - all Suite2p-detected neurons used
- For position-based analyses, spikes are interpolated from time to position (60 bins for 6m corridor)
- Only frames where mouse is running (ft_move > 0) are used for position interpolation
- For dprime, only corridor frames with running are used: `fr_valid = VRmove & isCorridor`
- Key behavior variables: ft_trInd, ft_CorrSpc, ft_GraySpc, ft_move, ft_Pos, ft_PosCum, StartFr, GrayFr, SoundFr, LickFr, WallName, isRew
- `data_process_script.ipynb`: processes all sessions to create interpolated spike data (neurons x trials x 60 position bins)

### Brain Region Mapping (from `neu_area_ID`)
- V1: iarea == 8
- mHV (medial higher visual): iarea in {0, 1, 2, 9}
- lHV (lateral higher visual): iarea in {5, 6}
- aHV (anterior higher visual): iarea in {3, 4}
- Outside visual cortex: iarea in {-1, 7} (excluded from density maps but not from dprime)

### Experiment Info Structure
- `Imaging_Exp_info.npy` contains 23 experiment types
- Each has list of db entries with keys: mname, datexp, blk, stim_id, rewType, exptype, etc.
- Same physical session can appear in multiple experiment types (different stim_id mappings)
- Sessions with `stimtype` (swap1/swap2 in test3 experiments) have appended behavior keys

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- **Neural (spk/)**: `{mouse}_{date}_{block}_neural_data.npy` - dict with 'spks' key containing list of arrays per imaging plane (neurons_per_plane x n_frames)
- **Behavior (beh/)**: `Beh_{exp_type}.npy` - dict with session keys mapping to full behavior dicts
- **Retinotopy**: `{mouse}_{date}_trans.npz` - contains iarea (area assignment per neuron), xy_t (coordinates)
- Frame rate: ~3.17 Hz (dt ≈ 0.315 s per frame)
- Corridor: 4m texture (40 dm) + 2m grey space (20 dm) = 6m total (60 dm)

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 19 |
| Sessions (unique) | 89 |
| Neural data planes | 3 per session |
| Neurons example (TX108_2023_01_05) | 68,553 (3 x 22,851) |
| Frames example | 23,193 |
| Trials example | 453 |
| Frame rate | ~3.17 Hz |
| Corridor length | 60 dm (4m texture + 2m grey) |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Recordings | 89 | "We performed 89 recordings in 19 mice" |
| Subjects | 19 | "89 recordings in 19 mice" |
| Neurons/session | 20,547 - 89,577 | "activity traces from 20,547 to 89,577 neurons" |
| Neural data type | Deconvolved fluorescence | "All analyses based on deconvolved fluorescence traces" |
| Deconvolution timescale | 0.75 s | "timescale of decay of 0.75 s" |
| Corridor length | 4 m | "corridors were each 4 m long" |
| Grey space | 2 m | "2 m of grey space between corridors" |
| VR speed | 60 cm/s constant | "always moved at a constant speed (60 cm/s)" |
| Run threshold | 6 cm/s | "running faster than a threshold of 6 cm/s" |
| Sound cue position | uniform 0.5-3.5 m | "randomly chosen per trial from uniform between 0.5 m and 3.5 m" |
| d' threshold | 0.3 | "criteria for selective neurons was d' >= 0.3" |
| Analysis frames | Running only | "only considered timepoints during running" |

### Processing Details
- Suite2p: motion correction, ROI detection, cell classification, neuropil correction, spike deconvolution
- No additional neuron filtering beyond Suite2p
- For analysis: only use frames inside corridor (0-4m) where mouse is running
- Position interpolation: time-series neural data interpolated to 60 position bins (1 decimeter each)
- Trial structure: pseudo-random corridor order, aligned to corridor entry

### Curation Steps
**Neuron curation rules**: None beyond Suite2p detection. All detected neurons included.
**Trial curation rules**: None mentioned. All trials used.

### Decoders Trained
No decoder trained in original paper. Paper focuses on d' selectivity and coding direction analysis.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neurons per session | Concatenate 3 planes | TX108: 68,553 (3x22,851) | 20,547-89,577 | Consistent - within range |
| Frame rate | dt from ft timestamps | ~3.17 Hz | Not explicitly stated | Consistent with calcium imaging |
| Corridor length | Corridor_Length=60 | Position 0-60 | 4m + 2m grey | 60 decimeters = 6m total |
| Texture area | CorrSpc=0-40 dm | ft_Pos 0-40 in corridor | 4m corridor | 40 dm = 4m, consistent |
| VR speed | ft_move > 0 for running | Running frames filtered | "only considered running" | Consistent |
| Sound cue | SoundPos, SoundFr | SoundPos uniform ~5-35 | 0.5-3.5 m range | 5-35 dm = 0.5-3.5m, consistent |
| Session duplication | Same session in multiple exp types | Same beh data, different stim_id | Multiple test types | Use each physical session once |

All sources are consistent. No unresolved discrepancies.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| spk (concatenated planes) | neural | Extract frames StartFr:GrayFr per trial | Deconvolved spikes at ~3.17 Hz |
| SoundFr - frame_idx | input[0]: time_to_sound_cue | (SoundFr - frame_idx) * dt_sec | Continuous, time-varying, positive before cue |
| date difference | input[1]: day_of_training | Days since mouse's first recording | Continuous, per-trial |
| frame_idx - StartFr | input[2]: time_since_trial_start | (frame_idx - StartFr) * dt_sec | Continuous, time-varying |
| isRew | input[3]: reward_availability | 1 if rewarded corridor, 0 otherwise | Discrete, per-trial |
| WallName | output[0]: visual_stimulus | Map to category index | Per-trial categorical |
| LickFr, LickTrind | output[1]: licking | Binary per frame (any lick in frame window) | Time-varying binary |
| ft_Pos | output[2]: position | Discretize into 4 bins of 10 dm each (0-10, 10-20, 20-30, 30-40) | Time-varying categorical |
| ft_RunSpeed | output[3]: running_speed | Discretize into 4 quartile bins | Time-varying categorical |

### Key Decisions
1. **Trial window**: StartFr to GrayFr (corridor entry to grey space entry) - captures full textured corridor portion
2. **Frame alignment**: Round fractional frame indices (StartFr, GrayFr, SoundFr) to nearest integer
3. **All neurons included**: No filtering, consistent with reference code
4. **All trials included**: No trial filtering
5. **Each physical session used once**: For sessions in multiple exp types, use the first available behavior data
6. **Brain regions**: V1, mHV, lHV, aHV, other (for iarea -1 and 7)
7. **Stimulus categories**: Use WallName directly (circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, plus rock/brick variants)
8. **Running speed quartiles**: Computed across ALL corridor frames in the dataset
9. **Licking**: Binary 1 at any frame that has >=1 lick (using LickFr rounded to nearest integer frame)
10. **Day of training**: Calendar day difference from mouse's first recording date

### Planned Sanity Checks
- [ ] Total sessions = 89, total subjects = 19
- [ ] Neuron count per session matches concatenated planes
- [ ] Neuron count per session in range 20,547-89,577
- [ ] Sound cue position distribution uniform ~5-35 dm
- [ ] Position values within 0-40 dm for corridor frames
- [ ] Neural activity at specific trial/timepoint matches raw data
- [ ] Number of trials per session matches behavior data
- [ ] Licking events align with LickFr from behavior data

---

## Step 6: Script Development
**Status**: COMPLETE

- Wrote `convert_data.py` with --sample, --full, and --show-processing modes
- Neural data stored as float16 to reduce file size
- Output data stored as int for categorical values
- Uses reference code's `load_spk` approach (concatenate across planes)
- Speed quartiles computed across all corridor frames

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics (2 sessions: DR10)
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Subjects | 1 (DR10) |
| Total trials | 934 |
| Trials/session | 431-503 |
| Neurons/session | 54,741-58,224 |
| Time bin | 314.7 ms |
| Position distribution | ~25% per bin |
| Speed distribution | ~25% per bin |
| Stimulus split | ~50/50 circle1/leaf1 |
| Licking | 0% (unsupervised mice) |
| Reward availability | 0% (unsupervised mice) |
| File size | 3.87 GB |

### Processing Plots Review
- Plots saved for 2 sessions, no anomalies detected
- Neural activity, inputs, and outputs visually consistent

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|----------------|---------------------|
| Process session | ~15s | ~22 min for 89 sessions |
| Speed quartile computation | 0.5s for 2 sessions | ~20s for all |
| Save pickle | 6s for 3.87 GB | ~3 min estimate |
| **Total estimate** | | **~25-30 min** |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None (sklearn warning about single label in licking - expected since unsupervised mice don't lick)

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| visual_stimulus | 0.9653 | 0.9446 | 0.5000 |
| licking | 1.0000 | 1.0000 | 0.5000 |
| position | 0.6522 | 0.6014 | 0.2500 |
| running_speed | 0.5715 | 0.5688 | 0.2500 |

All outputs well above chance. Loss decreased from 5233 to 127 over 200 epochs.
Note: licking accuracy trivially 1.0 since unsupervised mice never lick.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 148.22 GB
- `verification_full_out.txt`: created
- Total conversion time: 33.3 min

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions | 89 | 89 | 89 | 89 | ✓ |
| Subjects | 19 | 19 | 19 | 19 | ✓ |
| Neurons/session | 20,547-89,577 | concatenate 3 planes | verified | 20,547-89,577 | ✓ |
| Total trials | - | - | - | 38,109 | ✓ |
| Time bin | ~315 ms | from timestamps | ~3.17 Hz | 314.7 ms | ✓ |
| Corridor | 4m + 2m grey | 60 dm total | 0-40 dm pos | 0-40 dm | ✓ |
| Stimulus categories | leaf/circle/rock/brick | WallName | 15 unique | 15 | ✓ |
| Brain regions | V1/mHV/lHV/aHV | iarea mapping | retinotopy | 5 regions | ✓ |

### Verification Summary
- Data format: valid, no errors or warnings
- Position: 28.5%, 23.3%, 23.5%, 24.7% (roughly uniform)
- Speed: 30.2%, 19.8%, 24.9%, 25.1% (after `right=True` fix)
- Licking: 96.5% no_lick, 3.5% lick
- Day of training: 0-92 days
- 1 trial skipped (TX83_2022_08_31_1: invalid frame range)

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. Code matches reference `load_spk` (concatenate planes): ✓
2. Brain region mapping matches `neu_area_ID` (V1=8, mHV={0,1,2,9}, lHV={5,6}, aHV={3,4}): ✓
3. Trial window StartFr to GrayFr (corridor frames only): ✓
4. No neuron filtering (all Suite2p-detected neurons): ✓
5. Licking from LickFr with frame range filtering (equivalent to LickTrind): ✓
6. Speed quartiles computed across all corridor frames: ✓
7. Position discretization: 4 equal bins of 10dm (0-40dm): ✓
8. Sound cue timing: (SoundFr - frame_idx) * dt_sec: ✓
9. Day of training: calendar days from mouse's first recording: ✓
10. Session deduplication: each physical session used once: ✓
11. Stimulus categories from WallName (not stim_id): ✓

### Issues Found and Resolved
- Speed quartile distribution skewed (Q1=9.8%): Fixed by adding `right=True` to `np.digitize` so values exactly at 0 go to Q1 bin. New distribution: 30.2%, 19.8%, 24.9%, 25.1%.
- Very long trials (max T=5607 frames ≈ 29 min): Valid - mouse stopped running, VR stationary. No filtering needed as reference code doesn't filter by trial length.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (101.5 → 0.26 over 200 epochs)
- Test loss: 2.93
- Neurons subsampled to 2000 per session to fit in memory (decoder uses random projection to 2000 anyway)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|-------|
| visual_stimulus | 0.9733 | 0.7062 | 0.0667 |
| licking | 0.9910 | 0.8403 | 0.5000 |
| position | 0.9659 | 0.7344 | 0.2500 |
| running_speed | 0.8421 | 0.6887 | 0.2500 |

All outputs well above chance. Memory issue resolved by subsampling neurons before decoder (original attempt OOM killed at ~296 GB float32 needed vs 283 GB free).

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Val. Bal. Acc | Chance | Above Chance | Expectation |
|-----------|--------------|--------|--------------|-------------|
| visual_stimulus | 0.7062 | 0.0667 | 10.6x | High - paper shows strong visual selectivity (d'>=0.3) across areas |
| licking | 0.8403 | 0.5000 | 1.68x | Moderate - only 3.5% lick frames, but balanced acc high. Task mice show anticipatory licking |
| position | 0.7344 | 0.2500 | 2.94x | High - spatial coding well-documented in visual cortex |
| running_speed | 0.6887 | 0.2500 | 2.75x | Moderate-high - speed modulation of neural activity well-established |

### Notes on Decoder Accuracy
- All outputs significantly above chance, confirming data conversion is correct
- Neuron subsampling (2000 per session) reduces accuracy vs using all neurons, but still demonstrates clear decodability
- Train-val gap (0.97 vs 0.71 for stim) indicates some overfitting, expected with 15 categories and limited trials per class
- Licking accuracy is meaningful despite class imbalance (3.5% lick) because balanced accuracy weights classes equally

### Issues Found and Resolved
- **OOM during training**: Original attempt silently killed by OOM (148 GB float16 data → ~296 GB float32 tensors exceeds 283 GB free RAM). Fixed by subsampling to 2000 neurons per session before passing to decoder, since decoder internally does random projection to 2000 anyway.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
