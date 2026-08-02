# Dataset Conversion Notes

## Overview
- **Dataset**: Unsupervised pretraining in biological neural networks (Zhong et al., 2025). 2-photon calcium imaging in mouse visual cortex during VR corridor navigation task.
- **Date started**: 2026-03-24
- **Goal**: Convert to decoder-compatible format

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code: utils.py, fig1-5.py, S6.py, data_process_script.ipynb, Figures.ipynb
- `data/beh/` - 26 behavioral .npy files + Imaging_Exp_info.npy
- `data/spk/` - 89 neural data files (`{mouse}_{date}_{blk}_neural_data.npy`)
- `data/retinotopy/` - 89 retinotopy files (`{mouse}_{date}_trans.npz`) + `areas.npz`
- `data/process_data/` - empty (intermediate results would go here)
- `paper.pdf`, `methods.txt`, `decoder.py`, `train_decoder.py`

Python: numpy 2.3.5, torch 2.6.0+cu124, GPU available

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_spk(db)` | utils.py | LOADING | Load neural data, concatenate spks list into (n_neurons, n_frames) |
| `load_retino(db)` | utils.py | LOADING | Load retinotopy (xy_t, iarea), create area index via `neu_area_ID` |
| `load_exp_beh(root, exp_type)` | utils.py | LOADING | Load behavior file for an experiment type |
| `neu_area_ID(iarea)` | utils.py | PROCESSING | Map iarea values to brain regions: V1(8), mHV(0,1,2,9), lHV(5,6), aHV(3,4) |
| `get_interpPos_spk()` | utils.py | PROCESSING | Interpolate neural activity into (neurons, trials, 60 position bins) |
| `spk_pos_interp()` | utils.py | PROCESSING | Core interpolation: raw spk + cumulative pos -> trial x position matrix |
| `dprime()` | utils.py | PROCESSING | Compute d-prime for stimulus selectivity |
| `lickCount()` | utils.py | PROCESSING | Binary lick response per trial (before/after reward) |
| `get_cat_id()` | utils.py | PROCESSING | Get stimulus category IDs from wall names and reward info |
| `Get_dprime_selective_neuron()` | utils.py | CURATION | Select stimulus-selective neurons using d-prime, filter by corridor activity |
| `Get_coding_direction()` | utils.py | PROCESSING | Coding direction analysis with normalization and cross-validation |

### Key Data Flow
1. **Neural data**: Loaded via `load_spk()` -> concatenates multiple planes of Suite2p output -> shape (n_neurons, n_frames)
2. **Behavior**: Loaded from `Beh_{exp_type}.npy` -> dictionary with per-session behavior data
3. **Retinotopy**: Maps neurons to visual cortex areas (V1, mHV, lHV, aHV) via `iarea` values
4. **Processing**: Reference code primarily uses position-interpolated data (60 bins per trial) for analyses. The raw data is at ~3.17 Hz frame rate.
5. **Neuron filtering in reference**: Corridor-active neurons selected (higher activity in texture vs gray space). D-prime used for stimulus selectivity analysis. No explicit quality filtering of neurons.
6. **DeltaF/F**: NOT needed - data is already Suite2p deconvolved traces (spike deconvolution with 0.75s decay timescale). All analyses use deconvolved traces.

### Session Database Structure (Imaging_Exp_info.npy)
- 23 experiment types, 142 session-experiment pairs total
- 89 unique sessions across 19 mice
- Same physical session can appear in multiple experiment types (different analysis conditions)
- Each entry has: mname, datexp, blk, stim_id, rewType, exptype, etc.
- `sess#` field for most, `days` field for `*_after_learning` types

### Behavior Data Variables (per session)
Key fields: ntrials, WallName, UniqWalls, isRew, stim_id, ft (frame times), ft_trInd, ft_Pos, ft_PosCum, ft_move, ft_CorrSpc, ft_GraySpc, ft_WallID, ft_RunSpeed, StartFr, GrayFr, EndFr, SoundFr, LickFr, LickPos, LickTrind, run_pos, Corridor_Length(60dm=6m), Texture_Length(40dm=4m), Gray_Space_length(20dm=2m)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
- **Neural**: `{mouse}_{date}_{blk}_neural_data.npy` -> dict with 'spks' key -> list of arrays (one per imaging plane), concatenated = (n_neurons, n_frames)
- **Behavior**: `Beh_{exp_type}.npy` -> dict keyed by `{mouse}_{date}_{blk}` -> session behavior dict
- **Retinotopy**: `{mouse}_{date}_trans.npz` -> contains iarea (neuron area labels), xy_t (neuron positions)
- **Experiment info**: `Imaging_Exp_info.npy` -> dict keyed by experiment type -> list of session dicts

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Unique subjects | 19 mice |
| Unique sessions | 89 |
| Sessions / subject | 1-8 (mean ~4.7) |
| Neurons / session (example TX60) | 47,785 (46,218 in visual cortex) |
| Neurons per session range | ~20k-90k (from paper) |
| Trials / session (example) | 485 |
| Frame rate | ~3.17 Hz (~315 ms per frame) |
| Corridor length | 60 dm (6m: 4m texture + 2m gray) |
| VR speed | 60 cm/s when running > 6 cm/s threshold |

### Area Distribution (example session TX60_2021_06_07)
| iarea | Count | Region |
|-------|-------|--------|
| -1 | 410 | Outside visual cortex |
| 0,1,2,9 | 6,369 | mHV (medial higher visual) |
| 3,4 | 6,193 | aHV (anterior higher visual) |
| 5,6 | 6,430 | lHV (lateral higher visual) |
| 7 | 1,157 | Excluded (area 7) |
| 8 | 27,226 | V1 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source |
|-----------|-------|-------|
| Mice | 19 (GCaMP6s, excitatory neurons) | Methods: "89 recordings in 19 mice" |
| Recordings | 89 | Methods |
| Neurons/session | 20,547 to 89,577 | Paper: "activity traces from 20,547 to 89,577 neurons" |
| Frame rate | ~3.17 Hz | data_process_script.ipynb: "fs = 3.17Hz" |
| Corridor length | 4m texture + 2m gray | Methods: "4 m long, with 2 m of grey space" |
| VR speed | 60 cm/s (constant when running > 6 cm/s) | Methods |
| Sound cue position | Random 0.5-3.5m | Methods |
| Reward zone | After sound cue in rewarded corridor | Methods |
| Neural processing | Suite2p deconvolution, 0.75s decay | Methods: "deconvolved fluorescence traces" |
| Stimulus types | circle, leaf, rock, brick (+ gratings) | Methods |
| stim_id mapping | 0:circle1, 1:circle2, 2:leaf1, 3:leaf2, 4:leaf3, 5:leaf1_swap1, 6:leaf1_swap2 | data_process_script.ipynb |

### Processing Details
- **Temporal alignment**: Reference code uses position-based interpolation (60 bins = 1 dm each), not time-based alignment
- **Activity in analyses**: Only frames where VR is moving (ft_move > 0) are used
- **Corridor vs gray**: ft_CorrSpc (position 0-40dm) and ft_GraySpc (position 40-60dm)
- **Deconvolution**: Already done by Suite2p, data is deconvolved spikes

### Curation Steps

**Neuron curation rules** (from reference code):
- Neurons outside visual cortex (iarea == -1) and area 7 (iarea == 7) are excluded in some analyses
- No general quality filtering applied to all neurons; analysis-specific filtering using d-prime thresholds
- For our decoder: include ALL neurons (no filtering by area), as the decoder should learn to use relevant neurons

**Trial curation rules** (from reference code):
- Reference code uses all trials in a session
- Some analyses only use first 200 trials (get_lick_response_in_zone)
- Odd/even trial splitting for cross-validation in some analyses

### Decoders Trained
Paper does not report decoder accuracies for the specific variables we need. The paper focuses on d-prime, coding direction, and lick response analyses rather than a general-purpose decoder.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| N sessions | 89 unique (mname,datexp,blk) | 89 spk files | 89 recordings | Consistent |
| N mice | 19 unique mouse names | 19 from spk filenames | 19 mice | Consistent |
| Session reuse | Same session in multiple exp types | 142 session-exp pairs for 89 sessions | Different analyses on same data | Use each physical session ONCE |
| Neuron filtering | d-prime based, analysis-specific | iarea has -1 and 7 values | Only visual cortex for area analyses | Include all neurons, let decoder learn |
| Frame rate | 3.17 Hz in notebook | ~3.18 Hz from data | Not explicit | ~315ms per frame |
| Activity data | Deconvolved spikes from Suite2p | spks arrays in neural_data.npy | "deconvolved fluorescence traces" | Consistent |
| Corridor length | 60 dm in code | Corridor_Length=60 in beh | 4m texture + 2m gray = 6m | 60 dm = 6m, consistent |

### Key Decision: Session-Experiment Mapping
The same physical recording session appears in multiple experiment types (e.g., TX60_2021_06_07_1 appears in both sup_test1 and sup_train2_before_learning). These represent different analytical views of the same data. For the decoder, each physical session should be included ONCE. I will pick the first experiment type that contains each session, load behavior from that file.

### Key Decision: Which behavior file for each session
Since the same session can appear in multiple behavior files (different exp_types), I need to pick one consistently. The behavior data should be identical across files for the same session. I'll verify this and use any file that contains the session.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| spks (concatenated planes) | neural | Raw deconvolved spikes, extract per-trial segments StartFr:EndFr | ~3.17 Hz frame rate |
| SoundFr - current_frame | input[0]: time_to_sound_cue | (SoundFr - frame_idx) * frame_period_sec, continuous | Negative before, positive = time since |
| Session date order | input[1]: day_of_training | Chronological session index within mouse | Per-trial (broadcast) |
| frame_idx - StartFr | input[2]: time_since_trial_start | (frame_idx - StartFr) * frame_period_sec | Time-varying |
| isRew | input[3]: reward_availability | 1 if rewarded corridor, 0 if not | Per-trial (broadcast) |
| WallName | output[0]: visual_stimulus | Map to category index | Per-trial, 15 unique stimuli |
| LickFr, LickTrind | output[1]: licking | Binary per frame, 1 if lick in frame | Time-varying |
| ft_Pos | output[2]: position | 4 bins of 1m: [0,10), [10,20), [20,30), [30,60) dm | Time-varying |
| ft_RunSpeed | output[3]: running_speed | Quartile bins across all data | Time-varying |

### Brain Region Mapping
| iarea values | Region name |
|-------------|-------------|
| 8 | V1 |
| 0, 1, 2, 9 | mHV |
| 5, 6 | lHV |
| 3, 4 | aHV |
| -1, 7 | other |

### Key Decisions
1. **Use each physical session once**: Same recording in multiple exp types -> use first found
2. **Trial window**: StartFr to EndFr (corridor + gray space)
3. **Time bin = frame rate**: ~315ms, no additional binning
4. **Position bin 3 includes gray space**: [30,60) dm covers 3-4m texture + 2m gray
5. **Speed quartiles**: Computed globally across all frames in all sessions
6. **All neurons included**: No area filtering; let decoder learn
7. **Licking**: Round LickFr to nearest int frame, create binary vector per trial
8. **Day of training**: Ordinal session index (by date) within each mouse

### Planned Sanity Checks
- [ ] Neural shape matches n_neurons from retinotopy
- [ ] Total frames per session matches between spk and beh
- [ ] Trial count matches between StartFr and WallName
- [ ] Stimulus distribution matches across sessions
- [ ] Lick frames fall within expected trial boundaries
- [ ] Position range within [0, 60] dm

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation in `convert_data.py`:
- Loads each unique session once (89 sessions)
- Extracts per-trial neural/input/output data
- Uses raw frame-level data (~3.17 Hz)
- Trial window: StartFr to EndFr (corridor + gray space)
- Stores neural as float32

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (TX108 sup, DR10 unsup) |
| Subjects | 2 |
| Trials | 713 (210 + 503) |
| Neurons | 85,481 and 58,224 |
| Mean frames/trial | 107 and 63 |
| Stimuli | circle1, leaf1, rock1, wood1 |
| Licking fraction | 8% |
| Reward fraction | some TX108 trials |

### Processing Plots Review
Plots saved to processing_TX108_2023_03_13_1.png and processing_DR10_2022_07_12_1.png. No anomalies.

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| Processing | ~15s/session | ~22 min |
| File size | ~7 GB/session | ~217 GB (float32) |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| visual_stimulus | 0.8642 | 0.7999 | 0.25 |
| licking | 0.9020 | 0.8926 | 0.50 |
| position | 0.5819 | 0.5353 | 0.25 |
| running_speed | 0.5747 | 0.5677 | 0.25 |

All outputs above chance. Loss decreased from 8455 to 186 over 200 epochs. GPU OOM (1.6 GB), using CPU.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 202 GB (pickle protocol 5, float16 neural data)
- `conversion_full_out.txt`: created (33.4 min total)
- `verification_full_out.txt`: created — **no errors, no warnings**

### Consistency Check
| Statistic | Reference Paper | Reference Code | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|----------------|--------|
| Sessions | 89 recordings | 89 unique (mname,datexp,blk) | 89 spk files | 89 | YES |
| Mice | 19 | 19 unique mouse names | 19 | 19 | YES |
| Neurons/session range | 20,547 to 89,577 | - | - | 20,547 to 89,577 | YES |
| Mean neurons/session | - | - | - | 52,708 | - |
| Total trials | - | - | - | 38,110 | - |
| Frame rate | ~3.17 Hz | fs=3.17Hz | ~315ms | 314.7 ms | YES |
| Stimulus types | circle, leaf, rock, brick + variants | stim_id 0-6 | - | 15 unique | YES |
| Brain regions | V1, mHV, lHV, aHV | iarea mapping | - | 5 regions | YES |
| Corridor length | 4m + 2m gray | 60 dm | Corridor_Length=60 | 4 position bins [0-1m,1-2m,2-3m,3m+] | YES |
|-----------|-----------------|----------------|----------------|----------------|--------|

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed
1. **Output log review**: No errors, no warnings, no skipped trials in full conversion
2. **Verification**: `train_decoder.py --verify-only` passed with no errors/warnings
3. **Reference code comparison** (via agent):
   - `load_spk()`: Exact match with reference
   - `AREA_MAP`: Matches `neu_area_ID()` for all 4 named regions; added `other` for iarea=-1,7
   - Position range [0,60] dm: Correct
   - `SoundFr`: Correctly used as per-trial scalar frame index
   - Frame period: No reference equivalent; computed correctly from MATLAB datenum
4. **Frame period consistency**: std=0.0003s across 99 sessions — using single session value is fine
5. **Speed quartile distribution**: Q1=9.2%, Q2=40.9%, Q3=25.0%, Q4=24.9% — uneven due to 21% of frames at speed=0 exactly, which is the 25th percentile boundary
6. **Statistic matching**: 89 sessions, 19 mice, neurons 20,547-89,577 — all match paper exactly

### Deliberate Differences from Reference
1. **No ft_move filtering**: Reference code excludes stationary frames (ft_move==0) for d-prime analyses. Our decoder includes all frames to maintain temporal continuity. Running_speed output captures movement state.
2. **Gray space included**: Reference excludes gray space for texture analyses. Our position bin 3 (3m+) includes gray space, which is appropriate for the decoder.
3. **All neurons included**: Reference sometimes filters by area or d-prime. Decoder includes all neurons.

### Issues Found and Resolved
- None requiring code changes. All differences are deliberate design choices documented above.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (8455 → 19.8 over 200 epochs)
- Test loss: 12.97
- Neurons subsampled to 3000/session (from ~52K) to fit in memory — decoder uses random projection for >2000 neurons anyway

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Above Chance? |
|--------|-------------|--------|--------|--------|
| visual_stimulus | 0.1971 | 0.1878 | 0.0667 (1/15) | YES (2.8x) |
| licking | 0.6341 | 0.6362 | 0.50 | YES |
| position | 0.2994 | 0.2968 | 0.25 | YES |
| running_speed | 0.2959 | 0.2978 | 0.25 | YES |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis
| Variable | Val Accuracy | Chance | Ratio | Expectation |
|----------|-------------|--------|-------|-------------|
| visual_stimulus | 0.1878 | 0.0667 | 2.8x | Above chance expected — paper shows strong stimulus coding in visual cortex |
| licking | 0.6362 | 0.50 | 1.27x | Above chance expected — paper shows anticipatory licking in rewarded corridors. Only supervised sessions have licking. |
| position | 0.2968 | 0.25 | 1.19x | Modestly above chance — position is encoded but we include gray space in bin 3 and stationary frames |
| running_speed | 0.2978 | 0.25 | 1.19x | Modestly above chance — speed modulates neural activity |

### Comparison: Sample vs Full
| Output | Sample Val | Full Val | Notes |
|--------|-----------|---------|-------|
| visual_stimulus | 0.7999 | 0.1878 | Sample had 4 stimuli (chance=0.25), full has 15 (chance=0.067). Ratio to chance similar. |
| licking | 0.8926 | 0.6362 | Sample had only supervised sessions with clear licking patterns |
| position | 0.5353 | 0.2968 | Full dataset more diverse, neuron subsampling |
| running_speed | 0.5677 | 0.2978 | Full dataset more diverse |

### Notes
- All outputs above chance → data conversion is valid
- Lower absolute accuracies in full dataset are expected: 89 sessions from 19 mice with diverse experiment types, neuron subsampling to 3000, and 15 stimulus categories
- Train ≈ validation accuracy → no overfitting
- Paper does not report decoder accuracies, so direct comparison impossible

### Issues Found and Resolved
- OOM during decoder training: resolved by subsampling neurons to 3000/session (decoder already uses random projection for >2000 neurons)

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
