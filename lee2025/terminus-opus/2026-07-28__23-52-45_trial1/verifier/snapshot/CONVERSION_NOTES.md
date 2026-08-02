# Dataset Conversion Notes

## Overview
- **Dataset**: CA1 neural recordings from Lee, Keinath, Cianfarano & Brandon (2025) - "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping" (Neuron)
- **Date started**: 2025
- **Goal**: Convert to decoder-compatible format for position decoding from CA1 neural activity

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code from the paper (georepca1 repository)
- `data/` - Data files: 7 animal joblib files, 7 .mat files, behav_dict, precomputed_results/
- `paper.pdf` - The reference paper
- `methods.txt` - Extracted methods from the paper
- `decoder.py` - Decoder library
- `train_decoder.py` - Decoder training script

Python environment verified: numpy 2.3.5, torch 2.6.0+cu124, scipy 1.18.0

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat` | utils.py:61 | LOADING | Load animal data from joblib or MATLAB format |
| `generate_behav_dict` | utils.py:130 | LOADING | Generate lightweight behavioral dict for all animals |
| `get_env_mat` | utils.py:215 | PROCESSING | Get binary 3x3 matrix for environment geometry |
| `get_rate_maps` | utils.py:313 | PROCESSING | Compute rate maps from position and trace (15x15 bins) |
| `get_split_half` | utils.py:356 | CURATION | Calculate split-half reliability correlation |
| `get_shuffle_split_half` | utils.py:393 | CURATION | Shuffle-based split-half reliability |
| `get_place_cells` | utils.py:415 | CURATION | Identify place cells via split-half reliability |
| `get_shr_within` | utils.py:442 | CURATION | Within-session split-half reliability for all days |
| `fit_decoder` | utils.py:1776 | PROCESSING | Fit Naive Bayes decoder with temporal binning |
| `test_decoder` | utils.py:1806 | PROCESSING | Test decoder predictions |
| `decode_position_within` | utils.py:1845 | PROCESSING | Full within-session position decoding pipeline |
| `get_all_decoding_within` | utils.py:1969 | PROCESSING | Aggregate decoding results across animals |

### Notes
- Data is loaded via `load_dat(animal, p, format="joblib")` which returns `{animal: dataset_dict}`
- The decode pipeline transposes position and trace before passing to decode_position_within:
  `decode_position_within(dat[animal]['position'].T, dat[animal]['trace'].T, dat[animal]['maps']['smoothed'])`
- Decoding uses 15x15 spatial bins, velocity filtering (>5 cm/s), cell activity threshold (>5 events), 5-fold CV
- Temporal binning of 3 frames in fit_decoder (AvgPool1d with kernel_size=3)
- Gaussian smoothing of traces with sigma=3 before temporal binning
- The trace data is BINARY (0/1) - rising phase of calcium transients
- No additional dF/F computation needed - data is already preprocessed

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
Each animal file (joblib) contains a dict with animal ID as key, containing:
- `SFPs`: (35, 35, n_cells, n_days) - spatial footprints
- `blocked`: list of arrays - blocked partition indices per day
- `centroids`: (n_cells, 2, n_days) - cell centroids
- `envs`: (n_days, 1) - environment name strings
- `maps`: dict with 'sampling' (15,15,n_days), 'smoothed' (15,15,n_cells,n_days), 'unsmoothed'
- `position`: (n_days, 2, max_timepoints) - x,y position at 30 Hz
- `trace`: (n_days, n_cells, max_timepoints) - binary calcium events (0/1)

Cells not registered on a given day have NaN traces.

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 7 |
| Total unique neurons | 5,413 |
| Total sessions | 207 |
| Total rate maps (valid cell-days) | 69,744 |
| Recording duration | ~40 min/session (71866-72219 frames at 30 Hz) |
| Arena size | 75 x 75 cm |
| Environments | 10 unique geometries |
| Spatial bins (original) | 15 x 15 |

Per-animal breakdown:
| Animal | Days | Total Cells | Timepoints | Valid cells/day (min/max/mean) |
|--------|------|-------------|------------|-------------------------------|
| QLAK-CA1-08 | 31 | 515 | 71866 | 153/254/214 |
| QLAK-CA1-30 | 31 | 875 | 71866 | 336/422/381 |
| QLAK-CA1-50 | 31 | 942 | 71866 | 214/564/401 |
| QLAK-CA1-51 | 21 | 554 | 72219 | 113/323/230 |
| QLAK-CA1-56 | 31 | 862 | 72091 | 251/529/381 |
| QLAK-CA1-74 | 31 | 713 | 72060 | 258/405/312 |
| QLAK-CA1-75 | 31 | 952 | 72071 | 263/535/405 |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total) | 5,413 | "5,413 unique neurons across 207 sessions" |
| Sessions | 207 | "across 207 sessions" |
| Subjects | 7 | 7 animal IDs in code |
| Geometries | 10 | "10 geometrically distinct environments" |
| Rate maps | 69,744 | "forming 69,744 rate maps" |
| Arena size | 75x75 cm | "square open field (75x75 cm2)" |
| Partition size | 25 cm | "blocking select partitions with 25 cm walls" |
| Recording rate | 30 Hz | "recorded at 30 Hz" |
| Session duration | ~40 min | ~71866-72219 frames / 30 fps |
| Sequences | up to 3 | "up to three total repetitions (31 days)" |
| Place cell threshold | p < 0.01 | "split-half reliable (p < 0.01)" |
| Shuffles for place cells | 1000 | "1000 circular shuffles" |
| Decoding method | Naive Bayes | "naive Bayesian method" |
| Decoding CV | 5-fold | "5-fold split" |
| Spatial bins for decoding | one-hot position | "transformed binned positions to one-hot vector" |

### Processing Details
- Calcium traces preprocessed: motion correction, cell segmentation, transient extraction
- Rising phase extraction: derivative smoothed with Gaussian (sigma=5 frames), z-scored, binarized at z=2.5
- Binary vector treated as firing rate
- Position tracked with DeepLabCut
- Cells tracked across sessions via spatial footprints and centroids

### Curation Steps

**Neuron curation rules**:
- Cells not registered on a given day have NaN traces (already handled in data)
- Place cells identified via split-half reliability (p < 0.01, 1000 shuffles)
- In decode_position_within: cells filtered by activity threshold (>5 events when animal moving)
- In decode_position_within: velocity filter applied (>5 cm/s after Gaussian smoothing)
- NOTE: The reference code does NOT filter to only place cells for decoding - it uses ALL registered cells
  (with only the activity threshold filter in decode_position_within)

**Trial curation rules**:
- No explicit trial curation in reference code - all days/sessions are used
- The task description says to split sessions into 1-minute trials

### Decoders Trained
| Decoded variable | Metric | Values |
|-----------------|--------|--------|
| Position (x,y) | Euclidean error (cm) | Mean ~10-17 cm across animals |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Place cell alpha | alpha=0.05 in get_place_cells | N/A | p < 0.01 | Paper says 99th percentile. Code default is 0.05 but main.py uses nsims=1000 which with alpha=0.05 would be 95th. The paper figure caption says p<0.01. This discrepancy is in the reference code, not relevant for our task since we don't filter by place cell status for decoding. |
| Decoding spatial bins | 15x15 in decode_position_within | Maps are 15x15 | One-hot vector from binned positions | Task requires 3x3=9 bins for output |
| Blocked vs env_mat | blocked field uses indices | env_mat matches geometry | 3x3 grid structure | Use get_env_mat(env_name) for input, not blocked field directly |
| Session = Day | Each day is a session | 207 days total | 207 sessions | Confirmed: 1 session per day |
| Neuron filtering for decoding | cell_threshold=5, velocity filter | N/A | Not specified in detail | Reference code filters by activity and velocity within decode_position_within |

### Key Understanding
- Each "session" = one day of recording (~40 min) in one environment
- The reference code uses ALL registered cells for decoding (not just place cells)
- Velocity and activity thresholds are applied within the decoding function
- For our task: split each ~40 min session into 1-min trials
- Output: position in 3x3 grid (9 classes), time-varying
- Input: environment geometry (3x3 binary matrix = 9 values), static per trial

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| trace (binary events) | neural | Use all registered (non-NaN) cells per session, no temporal binning beyond what the time bin size provides | Binary 0/1 values |
| env_mat from envs | input[0:9] | get_env_mat(env_name).flatten() → 9 values | Static per trial (same for all timepoints in trial) |
| position (x,y) | output[0] | Bin into 3x3 grid → 9 classes (0-8) | Time-varying, bin_idx = x_bin * 3 + y_bin |

### Key Decisions

1. **Time bin size**: Use 1 second (30 frames) bins. This provides reasonable temporal resolution while reducing data size. The reference code uses temporal_bin_size=3 (100ms) for decoding, but for our decoder format, 1-second bins are more practical and still capture spatial behavior well.

2. **Trial definition**: Split each ~40 min session into 1-minute trials (60 seconds = 60 time bins at 1s resolution). This gives ~39-40 trials per session.

3. **Neural data**: Use the binary trace data directly. Average within each 1-second time bin to get firing rates. Only include cells that are registered (non-NaN) on that day.

4. **Position binning**: Bin position into 3x3 grid (9 bins). Use the same binning approach as the reference code: pos_binned = floor(pos / bin_size) where bin_size = (max_pos + buffer) / 3. Output = x_bin * 3 + y_bin.

5. **Position output per time bin**: Use the mode (most frequent) position bin within each 1-second window.

6. **Environment input**: Use get_env_mat(env_name).flatten() → 9 binary values. Static per trial.

7. **No velocity filtering**: The reference code applies velocity filtering within the decoding function, but for our format we include all timepoints. The decoder can learn to handle stationary periods.

8. **No cell activity filtering**: Include all registered cells. The reference code filters by activity threshold within the decoding function, but we include all cells and let the decoder handle it.

9. **Brain region**: CA1 for all neurons.

10. **Subject identification**: Use animal ID strings.

### Planned Sanity Checks
- [ ] Total neurons = 5,413
- [ ] Total sessions = 207
- [ ] Total rate maps (valid cell-days) = 69,744
- [ ] Position bin 4 (center) should have ~0 occupancy for "o" environment
- [ ] Trace values should be binary (0 or 1)
- [ ] Each session should produce ~39-40 trials (1 min each from ~40 min)
- [ ] Neural data dimensions: (n_valid_cells, 60) per trial
- [ ] Input dimensions: (9,) per trial (static)
- [ ] Output dimensions: (1, 60) per trial (time-varying)

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- convert_data.py written with 422 lines
- Loads joblib data files for each animal
- For each day/session: extracts valid (non-NaN) cells, bins neural data into 1s time bins, bins position into 3x3 grid
- Splits each ~40 min session into 39 1-minute trials (60 time bins each)
- Environment geometry (3x3 binary matrix) used as static input per trial
- Position bin index (0-8) used as time-varying output

Code inefficiencies identified:
- Mode computation in bin_position_temporal uses a loop (could vectorize with scipy.stats.mode)
- Data loading is sequential (could parallelize but not needed given short runtime)

Code speedups added:
- Vectorized temporal binning using reshape+mean
- Efficient position binning using floor division

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Subjects | 2 (QLAK-CA1-08, QLAK-CA1-30) |
| Sessions | 62 |
| Trials (total) | 2418 |
| Trials / session | 39 |
| Neurons / session | 153-422 (mean 297) |
| Time bins / trial | 60 |
| Input dimension | 9 |
| Output classes | 9 |
| File size | 174.3 MB |

### Processing Plots Review
- Processing plots saved for both animals
- Position trajectories look correct
- Environment geometries match expected patterns
- Neural binning shows sparse activity (binary events averaged over 1s)
- Position bins show expected distributions (blocked areas have 0 occupancy)

### Run Time Estimates
| Step | Time / Animal | Estimated Total Time |
|------|--------------|---------------------|
| Load + process | ~23s/animal | ~160s for 7 animals |
| Save | 0.5s | ~2s |
| Total | ~23s/animal | ~3 min |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None
- "Data format is valid, no errors or warnings."

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| position | 0.7541 | 0.6308 | 0.1111 |

Loss decreased consistently from 2.33 to 1.08 over 200 epochs.
Validation accuracy is 5.7x above chance - strong performance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 667.8 MB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Converted Data | Match? |
|-----------|-----------------|----------------|--------|
| Total unique neurons | 5,413 | 5,413 | ✓ |
| Total sessions | 207 | 207 | ✓ |
| Total rate maps (valid cell-days) | 69,744 | 69,744 | ✓ |
| Subjects | 7 | 7 | ✓ |
| Sessions per subject | 31,31,31,21,31,31,31 | 31,31,31,21,31,31,31 | ✓ |
| Mean neurons/session | ~337 | 336.93 | ✓ |
| Time bins/trial | 60 (1 min) | 60 | ✓ |
| Trials/session | ~39-40 | 39-40 | ✓ |
| Total trials | - | 8,187 | - |
| Output classes | 9 (3x3) | 9 | ✓ |
| Input dimension | 9 (env geometry) | 9 | ✓ |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

**Check 1: Output log verification**
- verification_full_out.txt: "Data format is valid, no errors or warnings."
- No errors or warnings to address.

**Check 2: Sanity checks (loading original data independently)**
1. Neural data spot-check (session 0, trial 0, bin 0): PASS - exact match with original trace data
2. Neural data spot-check (session 0, trial 5, bin 10): PASS - exact match
3. Input data check (square env = all 1s): PASS
4. Input data check (o env = center blocked): PASS
5. Output data check (position bin mode, 2 spot checks): PASS
6. Neuron counts per session (all 207 sessions): PASS - all match original data
7. Blocked environment occupancy (o env, bin 4 = 0.000): PASS
8. Last trial boundary check: PASS - all values finite, correct shape
9. Brain region idx consistency: PASS
10. Total neuron entries = 69,744: PASS - matches paper exactly

**Check 3: Reference code comparison**
| Processing Step | Reference Code | Our Code | Match? |
|----------------|---------------|----------|--------|
| Data loading | `load_dat(animal, p, format="joblib")` → joblib.load | `joblib.load(f'data/{animal}')` | ✓ |
| Cell filtering | Non-NaN cells used (NaN = not registered) | `~np.isnan(trace[:, 0])` | ✓ |
| Temporal alignment | Start of recording | Start of recording | ✓ |
| Neural binning | 3-frame bins in decode (AvgPool1d) | 30-frame (1s) bins via reshape+mean | Different bin size (task requirement) |
| Spatial binning | 15x15 bins in decode | 3x3 bins (task requirement) | Different (task requirement) |
| Input construction | `get_env_mat(env)` for geometry | Same function logic | ✓ |
| Output construction | One-hot position from binned coords | bin_idx = x_bin * 3 + y_bin | ✓ (same logic, different resolution) |

Differences are intentional and required by the task specification (3x3 output bins, 1-min trials).

**Check 4: Key statistics comparison**
| Statistic | Paper | Our Data | Match? |
|-----------|-------|----------|--------|
| Total unique neurons | 5,413 | 5,413 | ✓ |
| Total sessions | 207 | 207 | ✓ |
| Total rate maps | 69,744 | 69,744 | ✓ |
| Subjects | 7 | 7 | ✓ |
| Arena size | 75x75 cm | Position range [0,75] | ✓ |
| Recording rate | 30 Hz | 30 Hz | ✓ |
| Environments | 10 | 10 | ✓ |

**Check 5: Edge cases**
- End of session: ~55s remainder discarded (39-40 full 1-min trials from ~40 min)
- NaN cells: Properly excluded per session
- All 10 environments correctly mapped to 3x3 geometry
- No off-by-one errors in trial splitting or temporal binning

### Issues Found and Resolved
- No issues found. All checks pass.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (2.356 → 0.986 over 200 epochs)
- Training on 6531 trials, testing on 1656 trials
- Using GPU (cuda)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|----------------------|------------------------|--------|-------|
| position | 0.7937 | 0.6697 | 0.1111 | 6.0x above chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

**Check 1: Accuracy vs chance**
| Variable | Validation Accuracy | Chance | Ratio |
|----------|-------------------|--------|-------|
| position | 0.6697 | 0.1111 | 6.0x |

Accuracy is well above chance (6x), indicating correct data conversion.

**Check 2: Accuracy comparison to paper**
The paper reports decoding error in Euclidean distance (cm) using 15x15 spatial bins with Naive Bayes:
- Mean error: ~10-17 cm across animals (from precomputed results)
- Decoding error decreases over sessions

Our decoder uses 3x3 bins (9 classes) with a neural network, achieving 0.67 balanced accuracy.
Direct comparison is not possible due to different spatial resolution (3x3 vs 15x15) and different
decoder architecture (neural network vs Naive Bayes). However, the strong above-chance performance
confirms the neural data contains position information as expected.

**Check 3: Train vs validation gap**
- Training accuracy: 0.7937
- Validation accuracy: 0.6697
- Ratio: 1.19x (training/validation)
- This is below the 1.5x threshold, indicating no significant overfitting or data leakage.

### Issues Found and Resolved
- No issues found. All checks pass.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with processing plots and intermediate files
- [x] All files organized
- [x] CONVERSION_NOTES.md complete with all steps documented
