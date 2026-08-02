# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p - longitudinal tracking of neuronal activity from barrel cortex (Majnik et al. 2025)
- **Date started**: 2026-03-10
- **Goal**: Convert to decoder-compatible format. Decode motion energy (discretized, 5 bins) from neural activity (dF/F) using time as input.

## Step 0: Setup and Initialization
**Status**: COMPLETE

Python environment verified:
- numpy: 2.3.5, torch: 2.6.0+cu124, scipy: 1.17.1

Directory contents:
- `code/` - Reference code (track2p repository)
- `data/` - Data directory with 6 subjects + load_data.ipynb + README.md
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods from paper
- `decoder.py` - Decoder module; `train_decoder.py` - Decoder training script
- Each subject has session directories containing suite2p output and movement data

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_traces() | data/load_data.ipynb | LOADING | Loads F.npy from suite2p |
| load_fov() | data/load_data.ipynb | LOADING | Loads mean image from ops.npy |
| zscore_rows() | data/load_data.ipynb | PROCESSING | Z-scores fluorescence for visualization |
| iscell filtering | code/notebooks/demo_t2p_ouputs.ipynb | CURATION | Filter neurons by iscell threshold |
| match matrix indexing | code/notebooks/demo_t2p_ouputs.ipynb | CURATION | Select neurons matched across all days |

### Notes
- The provided data is ALREADY track2p output in suite2p format — neurons are pre-filtered
- All iscell values are 1.0 in the provided data (pre-filtered by track2p)
- The eval notebooks evaluate tracking quality (F1 scores), not downstream analysis
- The reference code doesn't contain downstream decoding analysis code

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 6 (jm031, jm032, jm038, jm039, jm040, jm046) |
| Sessions / subject | 7, 7, 7, 7, 6, 7 = 41 total |
| Neurons / subject | 221, 370, 685, 746, 541, 435 |
| Mean neurons / subject | 500 |
| Frames / session | 36000 (jm031, jm032) or 54000 (others) |
| Duration | 20 min or 30 min |
| Missing video frames | Up to 148 frames missing in some sessions |

### Suite2p Parameters (from ops.npy)
- fs: 30 Hz, neucoeff: 0.7, baseline: 'maximin'
- win_baseline: 60.0s, sig_baseline: 10.0 (frames), prctile_baseline: 8.0, tau: 0.3

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 6 | "full dataset of 6 mice" |
| Sessions / subject | ≥6 consecutive days | "within the second postnatal week (P7 to P14)" |
| Mean neurons / mouse | 526 ± 190 std | "On average 526 (± 190 std) neurons per mouse" |
| Imaging rate | 30 Hz | "Imaging rate was 30 Hz (resonant scanner)" |
| Session duration | 20 minutes | "each session lasted 20 minutes" |
| Neural data time bin | 10 frames (333 ms) | "averaging in bins of 10 consecutive timestamps" |
| Iscell threshold | 0.5 (default) | "all ROIs above the default threshold of 0.5" |

### Processing Details

**Neural data (dF/F)**:
- Paper: "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)"
- Suite2p processing: Fc = F - 0.7*Fneu → maximin baseline → dF/F = Fc - F0
- sig_baseline is used directly as Gaussian sigma in frames (Suite2p convention)

**Behavioral data (motion energy)**:
- Already computed in motion_energy_glob.npy

**Binning for decoding**:
- "averaging in bins of 10 consecutive timestamps" (effective rate: 3 Hz)
- "splits were done on consecutive 2 minute blocks"

### Curation Steps
- Neurons: iscell > 0.5 + track2p matching (already applied in data)
- Trials: No explicit curation; missing video frames interpolated

### Decoders Trained in Paper
| Decoded variable | Method | Notes |
|---|---|---|
| Motion energy | Ridge regression | R² shown as function of age (Fig 7) |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neurons/mouse | N/A | 221-746 (mean 500) | 526 ± 190 | Consistent (within 1 std) |
| Sessions | N/A | 6-7/mouse, 41 total | ≥6 days | Consistent |
| Duration | N/A | 20 or 30 min | 20 min | Some mice have 30min sessions; use all data |
| Iscell | threshold 0.5 | All iscell=1 | 0.5 | Data pre-filtered; consistent |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| F.npy, Fneu.npy | neural | dF/F (neuropil correction + maximin baseline), bin by 10 | Matches Suite2p |
| Time from start | input[0] | Elapsed time in seconds from start of 2-min trial | [0, 119.7s] |
| motion_energy_glob.npy | output[0] | Interpolate missing frames, bin by 10, global quintile discretization | 5 bins |

### Key Decisions
1. **dF/F**: Baseline subtraction (not division), matching Suite2p's `dcnv.preprocess()`. sig_baseline=10 frames.
2. **Trials**: 2-minute blocks (360 bins each). 20-min → 10 trials, 30-min → 15 trials.
3. **Missing frames**: Interpolate ME to match neural frame count using interframe intervals.
4. **Discretization**: Global quintile bins across all sessions.
5. **Brain region**: Single "barrel_cortex" for all neurons.

---

## Step 6: Script Development
**Status**: COMPLETE

Script: `convert_data.py`
- Implements dF/F using Suite2p maximin baseline (Fc - F0)
- Handles missing video frames via interpolation
- Splits into 2-min trials, bins by 10 frames
- Discretizes ME into 5 global quintile bins

Bug found and fixed during development:
- Initial version had `sig_frames = sig_baseline * fs = 300` (incorrect)
- Fixed to `sig_baseline = 10.0` (frames, matching Suite2p convention)
- Also fixed from `(Fc-F0)/F0` to `Fc-F0` (baseline subtraction, matching Suite2p)
- Fix also dramatically improved runtime: 1420s → 60s

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (jm031, jm032) |
| Neurons | 221, 370 |
| Trials/session | 10, 10 |
| Total trials | 20 |
| Time bin | 333.3 ms |
| Input range | [0.0, 119.7] s |
| Output distribution | 20% per bin (balanced quintiles) |

### Processing Plots Review
- Processing plots saved for 2 sessions
- Neural rasters show expected calcium transients
- Motion energy alignment with neural data verified visually

### Run Time Estimates
| Step | Time/Session | Total |
|------|-------------|-------|
| dF/F + binning | ~1.5s | ~60s for 41 sessions |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- No errors, no warnings

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| motion_energy_bin | 0.6345 | 0.4020 |

Chance level: 0.2000. Validation is 2.0x chance. Loss decreases consistently.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: Full dataset (41 sessions, 545 trials)
- `verification_full_out.txt`: No errors, no warnings

### Consistency Check
| Statistic | Reference Paper | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Subjects | 6 | 6 | 6 | Yes |
| Sessions | ≥6/mouse | 41 total | 41 total | Yes |
| Sessions/subject | ≥6 | 7,7,7,7,6,7 | 7,7,7,7,6,7 | Yes |
| Neurons/mouse (mean) | 526 ± 190 | 500 | 498.7 | Yes (within 1 std) |
| Neuron counts | N/A | 221,370,685,746,541,435 | 221,370,685,746,541,435 | Yes |
| Time bin | 333 ms | N/A | 333.3 ms | Yes |
| Output distribution | N/A | N/A | 20% per bin | Yes (quintiles) |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

1. **Output log verification**: `verification_full_out.txt` — no errors, no warnings.

2. **Sanity checks on neural data**: Independently loaded F.npy and Fneu.npy, computed dF/F, binned, and compared with converted data. `np.allclose` = True, max diff = 0.0.

3. **Sanity checks on input data**: Verified time input matches expected values (np.arange * bin_size/fs). `np.allclose` = True.

4. **Sanity checks on output data**: Verified output values are in valid range [0,4]. Verified bin ordering (higher bins correspond to higher ME values when checked globally).

5. **Reference code comparison**:
   - (a) Data loading: F.npy, Fneu.npy loaded directly (matches load_data.ipynb)
   - (b) Neuron filtering: No additional filtering needed (data pre-filtered by track2p)
   - (c) Temporal alignment: Neural and video are synchronized at 30 Hz; missing frames interpolated
   - (d) Binning: 10-frame bins (matches paper: "bins of 10 consecutive timestamps")
   - (e) Input construction: Time elapsed in seconds (decoder input)
   - (f) Output construction: Global quintile discretization of ME

6. **Key statistics comparison**: All match (see Step 9 consistency table).

7. **Edge cases**: Missing video frames handled via interframe interval detection and interpolation. End-of-session partial bins discarded (correct behavior).

### Issues Found and Resolved
- **dF/F computation bug**: Initial implementation used `sig_baseline * fs = 300` instead of `sig_baseline = 10` (frames), and division `(Fc-F0)/F0` instead of subtraction `Fc-F0`. Fixed to match Suite2p's `dcnv.preprocess()`.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (74.9 → 1.27 over 200 epochs)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| motion_energy_bin | 0.5975 | 0.4681 | 2.3x chance (0.2000) |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Chance | Ratio | Paper Expectation |
|----------|------------------|--------|-------|-------------------|
| motion_energy_bin | 0.4681 | 0.2000 | 2.34x | R² varies 0-0.5+ by age |

### Check 1: Accuracy vs chance
- Validation accuracy 0.4681 is 2.34x above chance (0.2000). Well above the 1.5x threshold.

### Check 2: Accuracy comparison to paper
- Paper uses ridge regression with R² metric on continuous ME; we use neural network with 5-class categorical classification
- These are fundamentally different tasks, so R² comparison is not directly applicable
- Paper shows R² increasing with development (P8→P14); our decoder pools across all ages
- Our 46.8% balanced accuracy on 5-class classification is strong performance consistent with the neural-behavioral coupling reported in the paper
- The moderate (not extreme) accuracy is expected because: (1) early sessions (P8-P10) have weak neural-ME coupling (R²≈0 in paper), (2) we're pooling across ages

### Check 3: Train vs validation gap
- Training: 0.5975, Validation: 0.4681
- Ratio: 1.28x (< 1.5x threshold)
- No overfitting concern

### Issues Found and Resolved
- No issues found in this review

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] CONVERSION_NOTES.md complete
- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
