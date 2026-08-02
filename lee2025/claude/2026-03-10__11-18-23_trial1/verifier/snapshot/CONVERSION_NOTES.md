# Dataset Conversion Notes

## Overview
- **Dataset**: CA1 neural recordings from "Identifying representational structure in CA1 to benchmark theoretical models of cognitive mapping" (georepca1), Lee et al. 2025, Neuron.
- **Date started**: 2026-03-10
- **Goal**: Convert to decoder-compatible format. Decode mouse location (3x3=9 spatial bins) from CA1 neural activity, with environment geometry as decoder input.

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `code/` - Reference code (georepca1 package: main.py, src/utils.py, src/plots.py)
- `data/` - Neural data files (7 mice: QLAK-CA1-08, 30, 50, 51, 56, 74, 75)
  - Joblib files (no extension) - preprocessed Python data
  - `.mat` files - original MATLAB data
  - `behav_dict` - behavioral data dictionary
  - `precomputed_results/` - precomputed analysis results (RSMs, SHR, decoding, etc.)
- `paper.pdf` - Reference paper (20 pages)
- `methods.txt` - Extracted methods text
- `train_decoder.py` - Decoder training script
- `decoder.py` - Decoder model code

Python environment verified: numpy 2.3.5, torch 2.6.0+cu124, scipy 1.17.1

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_dat` | utils.py:61 | LOADING | Load animal data from joblib or MATLAB format |
| `get_rate_maps` | utils.py:313 | PROCESSING | Create rate maps from position + trace (15x15 bins, 30fps, gaussian smoothing sigma=1.5) |
| `get_split_half` | utils.py:356 | PROCESSING | Split-half reliability via Pearson correlation of rate maps |
| `get_place_cells` | utils.py:415 | CURATION | Identify place cells via SHR > 99th percentile of 1000 shuffles |
| `get_shr_within` | utils.py:442 | CURATION | Iterate through days to get SHR p-values for all cells |
| `get_env_mat` | utils.py:215 | PROCESSING | Get binary 3x3 matrix for environment geometry |
| `decode_position_within` | utils.py:1845 | PROCESSING | Bayesian position decoding (5-fold CV, velocity filter >5cm/s, cell_threshold>5) |
| `get_transition_matrix` | utils.py:271 | PROCESSING | Transition matrix between spatial bins (step_size=15 frames = 2Hz) |
| `clean_rate_maps` | utils.py:463 | PROCESSING | NaN out pixels outside environment geometry |

### Notes
- Data is already preprocessed: binary calcium trace (0/1 for significant events from rising-phase extraction)
- No delta F/F computation needed - trace is already binarized
- Cells tracked across sessions via CellReg (spatial footprints)
- Rate maps pre-computed at 15x15 spatial bins with gaussian smoothing
- The paper uses ALL cells for RSM analyses ("motivated the inclusion of all cells in subsequent analyses")
- Place cell identification uses alpha=0.05 in code default, but paper says "99th percentile" (p<0.01)
- For decoding within session: velocity filter (>5 cm/s after gaussian smoothing), cell activity threshold (>5 events when moving)
- Recording at 30 Hz, sessions are ~40 minutes (~72000 frames)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
Each animal stored as joblib file: `data/{animal_id}` -> dict with animal_id key containing:
- `trace`: shape (n_days, n_cells, n_frames) - binary calcium events (0/1), NaN for unregistered cells
- `position`: shape (n_days, 2, n_frames) - x,y position in cm (0-75 range)
- `envs`: shape (n_days, 1) - environment name strings
- `blocked`: list of n_days arrays - blocked partition indices (-1 for square)
- `maps`: dict with 'smoothed', 'unsmoothed' (15,15,n_cells,n_days), 'sampling' (15,15,n_days)
- `SFPs`: (35,35,n_cells,n_days) - spatial footprints
- `centroids`: (n_cells, 2, n_days) - cell centroid locations

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 7 |
| Neurons per animal | 515, 875, 942, 554, 862, 713, 952 |
| Neurons (total unique) | 5413 |
| Sessions / subject | 31, 31, 31, 21, 31, 31, 31 |
| Sessions total | 207 |
| Frames / session | ~71866-72219 (~40 min @ 30Hz) |
| Environments | 10 unique: square, +, bit donut, o, u, t, rectangle, glenn, i, l |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Neurons (total unique) | 5,413 | "5,413 unique neurons across 207 sessions" |
| Rate maps total | 69,744 | "forming 69,744 rate maps" |
| Sessions total | 207 | "207 sessions" |
| Geometries | 10 | "10 geometrically distinct environments" |
| Subjects | 7 (4M, 3F) | "Naive male (4) and female (3) mice" |
| Mean cells/animal | 773 +/- 68 SE | "mean number of cells per animal = 773 +/- 68 SE" |
| Min cells/animal | 515 | "minimum cells per animal = 515" |
| Recording fps | 30 Hz | "acquired at 30 Hz" |
| Session duration | 40 min | "All sessions were 40 min" |
| Environment size | 75 x 75 cm | "75 x 75 cm" |
| Spatial bins (rate maps) | 15 x 15 (5cm bins) | "5cm x 5cm grid" |
| Sequences per animal | Up to 3 | "up to three total repetitions (31 days)" |

### Processing Details
- Neural data: already binarized rising-phase calcium events (z-score > 2.5 on derivative)
- Rate maps: 15x15 bins (5cm each), smoothed with gaussian kernel
- Position tracked with DeepLabCut
- Paper uses ALL cells for analyses

### Curation Steps

**Neuron curation rules**: Use ALL registered cells on each day (non-NaN trace). No place-cell filtering.

**Trial curation rules**: No trial filtering. Each session (day) is one continuous 40-min recording, split into 1-minute segments.

### Decoders Trained
| Decoded variable | Method | Metric |
|---|---|---|
| Position (15x15 bins) | Gaussian Naive Bayes, 5-fold CV | Euclidean distance error (cm) |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Total unique neurons | Sum: 5413 | 5413 | 5413 | Consistent |
| Sessions | Sum: 207 | 207 sessions | 207 | Consistent |
| Rate maps (neuron-sessions) | - | 69,744 | 69,744 | Consistent: sum of registered cells per day |
| Session duration | - | ~71866-72219 frames | 40 min @30Hz | ~39.9 min. Consistent. |
| Spatial bins | n_bins=15 | maps 15x15 | "5cm x 5cm grid" (75/5=15) | Consistent |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| `trace[day]` (binary events) | neural | Registered cells only, (n_registered, n_frames_per_trial) | `load_dat` | Binary 0/1, NaN cells excluded |
| Environment geometry 3x3 | input[0-8] | `get_env_mat(env)` -> flatten to 9 values, static per trial | `get_env_mat` | 1=accessible, 0=blocked |
| Position binned to 3x3 | output[0] | Bin position into 3x3 grid (25cm/bin), integer 0-8 | Custom | Time-varying |

### Key Decisions
1. **Trial definition**: 1-minute segments (1800 frames at 30Hz), ~40 trials per session
2. **Neural data**: Raw binary trace (0/1), only registered cells per session
3. **Time bin**: 30Hz native sampling (33.33ms)
4. **Position discretization**: 3x3 grid, row-major ordering (bin_idx = x_bin*3 + y_bin)
5. **Input**: Environment geometry, static per trial
6. **No velocity filtering**: Not specified in decoder task
7. **Brain region**: CA1 (single)

---

## Step 6: Script Development
**Status**: COMPLETE

Script `convert_data.py` written with:
- `--full` (default) and `--sample` modes
- `--show-processing` for visualization
- Efficient vectorized operations (no inner loops for trace/position)
- Timing output for each session
- Uses `get_env_mat` from reference code

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Animals | 2 (QLAK-CA1-08, QLAK-CA1-30) |
| Sessions | 62 |
| Trials | 2480 |
| Neurons/session | min=153, max=422, mean=297 |
| Time | 42.5s |

### Processing Plots Review
- Processing plots saved for both animals
- Neural activity shows sparse binary events (expected for calcium trace)
- Position traces cover full arena
- 3x3 binning correctly maps position
- Environment geometry correctly encoded

### Run Time Estimates
- ~21s per animal average, 42.5s for 2 animals
- Full run estimated: ~160s (actual: 159.8s)

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| position_bin | 0.5649 | 0.4966 | 0.1111 |

Loss decreased from 2.31 to 1.38. Accuracy ~4.5x chance.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 19,255 MB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Subjects | 7 | 7 | 7 | YES |
| Sessions | 207 | 207 | 207 | YES |
| Neuron-sessions | 69,744 | 69,744 | 69,744 | YES |
| Mean neurons/session | ~337 | ~337 | 337 | YES |
| Sessions/subject | 31,31,31,21,31,31,31 | same | same | YES |
| Min neurons/session | - | 113 | 113 | YES |
| Max neurons/session | - | 564 | 564 | YES |
| Trials/session | N/A | N/A | 40 | N/A (new) |
| Total trials | N/A | N/A | 8280 | N/A (new) |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

**Check 1: Output log verification**
- `verification_full_out.txt`: "Data format is valid, no errors or warnings."
- All 207 sessions have 40 trials each
- Input ranges correct: [0,1] for all 9 env partitions
- Output range: [0, 8] as expected

**Check 2: Sanity checks (spot-checking original data vs converted)**
- Neural data: QLAK-CA1-50, day 0, trial 5 -> `np.allclose` returns True (exact match)
- Position bins: Same session/trial -> `np.allclose` returns True (exact match)
- Environment input: Square environment correctly encoded as all 1s
- Subject index: Session 62 correctly maps to QLAK-CA1-50

**Check 3: Reference code comparison**
| Processing Step | My Code | Reference Code | Match? |
|---|---|---|---|
| Data loading | `joblib.load(f'/app/data/{animal}')` | `load_dat` via `joblib.load` | YES |
| Cell registration | Filter by `~np.isnan(trace[:, 0])` | Same approach in reference | YES |
| Env geometry | `get_env_mat(env_name).flatten()` | `get_env_mat` identical | YES |
| Position binning | `np.floor(pos / bin_size)`, 3 bins | Similar to `get_rate_maps` approach | YES |
| No cell filtering | Use ALL registered cells | Paper: "motivated inclusion of all cells" | YES |
| Binary trace | Used as-is (0/1) | "treated as firing rate" | YES |

**Check 4: Key statistics comparison**
| Statistic | Paper | Converted | Match? |
|---|---|---|---|
| Unique neurons | 5,413 | 5,413 (sum of per-animal) | YES |
| Sessions | 207 | 207 | YES |
| Neuron-sessions (rate maps) | 69,744 | 69,744 | YES |
| Mean cells/animal | 773 +/- 68 | 773.3 (5413/7) | YES |
| Min cells/animal | 515 | 515 (CA1-08) | YES |

**Check 5: Edge cases**
- Last trial has fewer frames when session length isn't divisible by 1800 (min T=1666)
- Partial trials <900 frames (30s) are dropped (correct behavior)
- QLAK-CA1-51 has only 21 sessions (2 sequences) - correctly handled
- All sessions produce >=2 trials (required for decoder evaluation)

### Issues Found and Resolved
No issues found. All checks pass.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (2.307 -> 1.259 over 200 epochs)
- Test loss: 1.232

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-------------|--------|-------|-------|
| position_bin | 0.6197 | 0.5516 | 0.1111 | 5.0x chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Val Acc | Chance | Ratio | Above 1.5x? |
|---|---|---|---|---|
| position_bin | 0.5516 | 0.1111 | 4.96x | YES |

**Check 1: Accuracy vs chance** - 55.16% vs 11.11% chance = 4.96x. Well above threshold.

**Check 2: Accuracy comparison to paper** - Paper reports Euclidean decoding error (cm) using Gaussian Naive Bayes on 15x15 bins, not balanced accuracy on 3x3 bins. Direct comparison not possible since different metrics and bin sizes. However, our strong 5x-chance accuracy is consistent with the paper's finding that CA1 populations encode position well ("achieving maximum decoding accuracy reported in recent work").

**Check 3: Train vs validation gap** - Train 0.6197 / Val 0.5516 = 1.12x ratio. Well below 1.5x threshold. No overfitting concern.

### Issues Found and Resolved
No issues found.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
