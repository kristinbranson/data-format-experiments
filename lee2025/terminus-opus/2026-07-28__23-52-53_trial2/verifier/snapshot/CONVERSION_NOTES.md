# Dataset Conversion Notes

## Overview
- **Dataset**: Lee et al. (2025) "Identifying representational structure in CA1" - Calcium imaging in hippocampal CA1 during geometric environment exploration
- **Date started**: 2025-07-29
- **Goal**: Convert to decoder-compatible format. Decode mouse position (3x3 spatial bins) from CA1 neural activity, with environment geometry as decoder input.

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code from the paper (georepca1 repository)
- `data/` - Data files (7 animals as joblib + .mat, behav_dict, precomputed_results)
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods from paper
- `decoder.py` - Decoder library
- `train_decoder.py` - Decoder training script

Python environment: numpy 2.3.5, torch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_dat | utils.py:61 | LOADING | Load animal data from joblib/MATLAB files |
| generate_behav_dict | utils.py:130 | LOADING | Create lightweight behavior dictionary |
| get_env_mat | utils.py:215 | PROCESSING | Convert environment name to 3x3 binary matrix |
| get_rate_maps | utils.py:313 | PROCESSING | Create rate maps from position and trace (15x15 bins, Gaussian smoothing sigma=1.5) |
| get_split_half | utils.py:356 | CURATION | Calculate split-half reliability of rate maps |
| get_shuffle_split_half | utils.py:393 | CURATION | Shuffled split-half reliability (1000 circular shuffles) |
| get_place_cells | utils.py:415 | CURATION | Identify place cells (p<0.01 from shuffle test) |
| get_shr_within | utils.py:442 | CURATION | Iterate SHR across days for an animal |
| clean_rate_maps | utils.py:463 | PROCESSING | NaN out extraneous pixels for each environment |
| fit_decoder | utils.py:1776 | PROCESSING | Fit GaussianNB decoder: temporal bin 3 frames, Gaussian smooth, one-hot position |
| test_decoder | utils.py:1806 | PROCESSING | Test decoder predictions |
| decode_position_within | utils.py:1845 | PROCESSING | Full within-session decoding: velocity filter (>5cm/s), cell activity threshold (>5 events), 5-fold CV |

### Notes
- Data is pre-processed: trace is already binarized rising-phase calcium events (0/1)
- Position is raw x,y in cm (0-75 range)
- The reference decode_position_within uses:
  - Velocity filtering: smoothed velocity > 5 cm/s
  - Cell activity threshold: >5 events during movement periods
  - Temporal binning: AvgPool1d with kernel_size=3 (100ms bins)
  - Gaussian smoothing of traces: sigma=3 frames before temporal binning
  - Position binned into 15x15 spatial bins
  - GaussianNB with flat priors
- For our decoder task, we need 3x3 spatial bins (not 15x15)
- The fit_decoder function applies: gaussian_filter1d(traces, sigma=temporal_bin_size=3) then AvgPool1d(kernel_size=3)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
Each animal file (joblib) contains a dict with animal ID as key:
- `trace`: (n_days, n_cells, n_frames) - binary calcium events (0/1), NaN for unregistered cells
- `position`: (n_days, 2, n_frames) - x,y position in cm (0-75)
- `envs`: (n_days, 1) - environment name strings
- `blocked`: list of n_days, each containing array of blocked partition indices (0-8, or -1 for none)
- `maps`: dict with smoothed/unsmoothed/sampling rate maps (15,15,n_cells,n_days)
- `SFPs`: (35,35,n_cells,n_days) - spatial footprints
- `centroids`: (n_cells,2,n_days) - cell centroids

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total unique) | 5,413 |
| Neurons / session | 515, 875, 942, 554, 862, 713, 952 (per animal) |
| Subjects | 7 |
| Sessions / subject | 31 (6 animals), 21 (QLAK-CA1-51) |
| Sessions (total) | 207 |
| Registered cell-day maps | 69,744 |
| Place cell maps (p<0.01) | 44,293 |
| Frames / session | ~72,000-72,219 (40 min at 30 Hz) |
| Frame rate | 30 Hz |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 | "5,413 unique neurons" |
| Sessions | 207 | "across 207 sessions" |
| Geometries | 10 | "10 geometries" |
| Rate maps | 69,744 | "forming 69,744 rate maps" |
| Subjects | 7 | From data (7 animal files) |
| Session duration | 40 min | "All sessions were 40 min" |
| Frame rate | 30 Hz | "acquired... at 30 Hz" |
| Environment size | 75x75 cm | "75 cm x 75 cm" |
| Place cell threshold | p<0.01 | "exceeded the 99th percentile" |
| SHR shuffles | 1000 | "1000 circular shuffles" |
| Decoding method | 5-fold GaussianNB | "5-fold split... naive Bayesian" |
| Mean decoding error | ~10-18 cm | From precomputed within_decoding |

### Processing Details
- Calcium trace: binarized rising-phase extraction (z-score > 2.5)
- Position: DeepLabCut tracking
- Cells tracked across sessions via spatial footprints and centroids
- Binary vector treated as firing rate in all analyses
- Rate maps: 15x15 spatial bins, Gaussian smoothing (sigma=1.5 bins ≈ 2.5 cm)

### Curation Steps

**Neuron curation rules**:
- Place cells identified by split-half reliability (SHR) with p<0.01 (99th percentile of 1000 shuffles)
- NaN trace = cell not registered on that day (excluded from that session)
- In decode_position_within: cells need >5 events during movement periods

**Trial curation rules**:
- Velocity filtering in decode_position_within: smoothed velocity > 5 cm/s
- No explicit trial-level curation in the paper (sessions are continuous recordings)

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Position (15x15 bins) | Mean error ~10-18 cm across animals |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Rate maps count | n_cells * n_days | 69,744 registered | "69,744 rate maps" | 69,744 = registered cell-days (non-NaN), not total |
| Place cells | p<0.01 threshold | 44,293 place cell maps | "99th percentile" | Consistent: 99th percentile = p<0.01 |
| Frame count | - | 72,000-72,219 | 40 min at 30 Hz = 72,000 | Minor variation in actual recording length |
| Velocity filter | >5 cm/s in code | - | Not mentioned in methods | Used in decode_position_within only |
| Cell threshold | >5 events in code | - | Not mentioned in methods | Used in decode_position_within only |

All discrepancies resolved. The data, code, and paper are consistent.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| trace (binary events) | neural | Gaussian smooth (sigma=3) + temporal bin (AvgPool1d k=3) | fit_decoder | Produces continuous firing rates at ~10 Hz |
| envs → get_env_mat() | input[0:9] | Convert env name to 3x3 binary matrix, flatten to 9 values | get_env_mat | Static per trial (1D array of 9 values) |
| position (x,y) | output[0] | Discretize into 3x3 bins (0-8), temporally bin same as neural | decode_position_within | Time-varying, 9 categories |

### Key Decisions
1. **Temporal binning**: Use reference code approach: Gaussian smooth trace with sigma=3, then AvgPool1d with kernel_size=3. This gives 100ms time bins at ~10 Hz. Applied to both neural and position data.
2. **Trial splitting**: Split each 40-min session into 1-minute trials = 40 trials per session (discard remainder frames). Each trial = 1800 frames = 600 time bins after temporal binning.
3. **Neural data**: Use ALL registered cells per session (not just place cells). Cells with NaN trace on a given day are excluded from that session. No velocity filtering (that was specific to the reference Bayesian decoder).
4. **Position discretization**: Bin x,y position into 3x3 grid (each bin = 25x25 cm). Convert to single category 0-8. The 3x3 grid matches the environment partition structure.
5. **Environment input**: Use get_env_mat() to convert environment name to 3x3 binary matrix. Flatten to 9 values. This is static per trial.
6. **Sessions**: Each day for each animal = 1 session. Total 207 sessions.
7. **No velocity filtering**: The task asks to decode position at all times, not just during movement.
8. **No place cell filtering**: Use all registered cells to give the decoder maximum information.

### Position binning scheme
- Position range: 0-75 cm in both x and y
- 3 bins: [0-25), [25-50), [50-75] cm
- Combined into single category: bin_x * 3 + bin_y = 0-8
- This matches the 3x3 partition grid of the environment

### Planned Sanity Checks
- [ ] Verify total sessions = 207
- [ ] Verify total unique neurons = 5,413
- [ ] Verify registered cell-days = 69,744
- [ ] Verify position range 0-75 cm
- [ ] Verify trace values are 0/1 (binary)
- [ ] Verify neural data after processing has continuous values (not just 0/1)
- [ ] Verify environment geometry matches blocked field
- [ ] Verify output position bins cover all 9 categories
- [ ] Spot-check: compare neural activity at specific trial/timepoint/neuron with raw data

---

## Step 6: Script Development
**Status**: COMPLETE

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 |
| Trials | 78 (39 per session) |
| Neurons / session | 185, 153 |
| Time bins / trial | 600 |
| Input range | [0, 1] |
| Output range | [0, 8] |

### Processing Plots Review
- processing_QLAK-CA1-08_day0.png and day1.png saved
- Neural data shows continuous values after smoothing (not binary)
- Position bins correctly reflect environment geometry (bin 4 = 0% for 'o' env)
- No anomalies detected

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|----------------|---------------------|
| Loading | ~20-80s per animal | ~210s total |
| Processing | ~0.7-1.0s per session | ~170s total |
| Total | | ~380s (~6.3 min) |

Actual full conversion time: 427s (~7.1 min) - within estimate

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| position | 0.4406 | 0.3799 | 0.1111 |

Accuracy is 3.4x chance level on just 2 sessions - good sign.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 6353.8 MB
- `verification_full_out.txt`: created

### Consistency Check
| Statistic | Reference Paper | Converted Data | Match? |
|-----------|-----------------|----------------|--------|
| Total sessions | 207 | 207 | ✓ |
| Total unique neurons | 5,413 | 5,413 (7 animals) | ✓ |
| Total registered cell-days | 69,744 | 69,744 | ✓ |
| Sessions per subject | 31/31/31/21/31/31/31 | 31/31/31/21/31/31/31 | ✓ |
| Frame rate | 30 Hz | 30 Hz | ✓ |
| Session duration | 40 min | ~40 min (71866-72219 frames) | ✓ |
| Geometries | 10 | 10 unique env names | ✓ |
| Position range | 75x75 cm | [0, 75] cm | ✓ |
| Time bin size | 100 ms (3 frames) | 100 ms | ✓ |

All statistics match the reference paper.

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

**Check 1: Output log verification**
- verification_full_out.txt: "Data format is valid, no errors or warnings."
- No errors or warnings to address.

**Check 2: Sanity checks**
1. Neural data: Compared processed neural data for session 0, trial 5, neuron 3 against independently processed raw data. Result: EXACT MATCH (max diff = 0.0)
2. Input data: Verified environment geometry for 'square' (all 1s) and 'o' (center=0). Result: MATCH
3. Output data: Verified position bin discretization for trial 5, timepoint 10. Result: MATCH
4. Registered neuron count: Day 0 = 185 neurons. Result: MATCH
5. Subject/session mapping: Subject idx correct for sessions 0-4 (all 0) and session 31 (= 1). Result: MATCH
6. Total registered neurons: 69,744. Result: MATCH (paper says 69,744 rate maps)
7. Neural variability: 230 unique values in sample trial (not binary). Result: PASS

**Check 3: Reference code comparison**
- (a) Data loading: Using joblib.load() same as reference load_dat(). ✓
- (b) Neuron filtering: Including all registered cells (non-NaN trace). Reference decode_position_within uses cell_threshold>5 events, but that's specific to Bayesian decoder. We include all cells for neural network decoder. Justified difference.
- (c) Temporal alignment: No velocity filtering applied. Reference uses velocity>5cm/s for Bayesian decoder. We decode at all timepoints. Justified difference.
- (d) Binning: Gaussian smooth sigma=3, AvgPool1d kernel=3. Matches reference fit_decoder. ✓
- (e) Input construction: get_env_mat() copied from reference. ✓
- (f) Output construction: Position discretized into 3x3 bins (task requirement, reference uses 15x15). Justified difference per task spec.

**Check 4: Key statistics comparison**
- Animals: 7 ✓
- Sessions: 207 ✓
- Registered cell-days: 69,744 ✓
- Session duration: ~40 min ✓
- Frame rate: 30 Hz ✓

**Check 5: Edge cases**
- Frames per session varies (71866-72219). Handled by discarding remainder after last complete trial.
- All cells registered on at least one day (no empty sessions).
- Position range [0, 75] handled with buffer for binning.

### Issues Found and Resolved
- Output dtype was float32, causing index error in decoder.py. Fixed to int64.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (2.33 → 1.16 over 200 epochs)
- Test loss: 1.136

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|----------------------|------------------------|--------|-------|
| position | 0.6946 | 0.6071 | 0.1111 | 5.5x chance, train/val gap 1.14x |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Chance | Ratio | Expectation from Paper |
|----------|------------------|--------|-------|------------------------|
| position | 0.6071 | 0.1111 | 5.46x | Consistent with paper's ~10-18 cm decoding error on finer grid |

**Check 1: Accuracy vs chance**
- Position accuracy 0.607 is 5.46x chance (0.111). Well above 1.5x threshold. PASS.

**Check 2: Accuracy comparison to paper**
- Paper uses Bayesian decoder on 15x15 bins (5 cm resolution) with mean error ~10-18 cm.
- Our decoder uses neural network on 3x3 bins (25 cm resolution).
- 61% balanced accuracy is consistent with good spatial decoding at this resolution.

**Check 3: Train vs validation gap**
- Train: 0.695, Val: 0.607. Ratio: 1.14x. Below 1.5x threshold. PASS.

### Issues Found and Resolved
- No issues found in this review.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with processing plots
- [x] All files organized
