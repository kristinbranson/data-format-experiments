# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p - Longitudinal tracking of neuronal activity in developing mouse barrel cortex (Majnik et al. 2025)
- **Date started**: 2024
- **Goal**: Convert to decoder-compatible format for decoding motion energy from neural activity

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/paper.pdf` - Reference paper
- `/app/methods.txt` - Extracted methods text
- `/app/code/` - Track2p code repository
- `/app/data/` - Data directory with 6 subjects
- `/app/decoder.py` - Decoder library
- `/app/train_decoder.py` - Decoder training script

Python environment verified: numpy 2.4.4, torch 2.6.0+cu124, scipy 1.18.0

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_traces() | data/load_data.ipynb | LOADING | Load raw F traces from suite2p |
| load_fov() | data/load_data.ipynb | LOADING | Load mean FOV image from ops |
| zscore_rows() | data/load_data.ipynb | PROCESSING | Z-score for visualization |
| load_all_ds_stat_iscell() | code/track2p/io/s2p_loaders.py | LOADING | Load stat/iscell with threshold filtering |
| load_stat_ds_plane() | code/track2p/io/loaders.py | LOADING | Load stat with iscell filtering |
| preprocess() | suite2p/extraction/dcnv.py | PROCESSING | Baseline correction (maximin) for dF/F |
| baseline_maximin() | suite2p/extraction/dcnv.py | PROCESSING | GPU-accelerated maximin baseline estimation |

### Notes
- The provided data is already the output of Track2p in Suite2p format
- All cells in the data are tracked cells (iscell all = 1.0)
- Track2p filters cells using iscell_thr (probability threshold from Suite2p classifier)
- The demo notebook (demo_t2p_ouputs.ipynb) shows how to load matched cell traces
- For analysis, the paper uses dF/F (baseline corrected) not raw F
- Decoding uses binning of 10 frames for both dF/F and behavior
- Ridge regression decoder with nested 5-fold CV on 2-minute blocks

### Suite2p dF/F Parameters (from ops.npy)
- neucoeff: 0.7 (neuropil subtraction coefficient)
- baseline: "maximin" (baseline estimation method)
- win_baseline: 60.0 (window for baseline in seconds)
- sig_baseline: 10.0 (smoothing for baseline)
- prctile_baseline: 8.0 (percentile for baseline)
- tau: 0.3 (decay time constant for GCaMP)
- fs: 30 (frame rate in Hz)

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
/app/data/
  README.md
  load_data.ipynb
  jm031/ (mouse A) - 7 sessions (2023-10-18 to 2023-10-24)
  jm032/ (mouse B) - 7 sessions (2023-10-18 to 2023-10-24)
  jm038/ (mouse C) - 7 sessions (2023-04-30 to 2023-05-06) + ground_truth.csv
  jm039/ (mouse D) - 7 sessions (2024-04-30 to 2024-05-06) + ground_truth.csv
  jm040/ (mouse E) - 6 sessions (2024-05-01 to 2024-05-06)
  jm046/ (mouse F) - 7 sessions (2024-09-03 to 2024-09-09) + ground_truth.csv

Each session folder contains:
  suite2p/plane0/
    F.npy - Raw fluorescence traces (n_neurons x n_frames), float32
    Fneu.npy - Neuropil fluorescence (n_neurons x n_frames), float32
    spks.npy - Deconvolved spikes (n_neurons x n_frames), float32
    iscell.npy - Cell classification (n_neurons x 2), all 1.0 for tracked cells
    ops.npy - Suite2p options dict
    stat.npy - Cell statistics (ROI info)
  move_deve/
    motion_energy_glob.npy - Motion energy (n_frames,), uint64
    tstamps.npy - Camera timestamps (n_frames,), float64 (in kiloseconds)
    interframe_int.npy - Inter-frame intervals (n_frames-1,), float64
```

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 6 (jm031, jm032, jm038, jm039, jm040, jm046) |
| Sessions total | 41 (7+7+7+7+6+7) |
| Sessions / subject | 6-7 |
| Neurons per subject | 221, 370, 685, 746, 541, 435 |
| Neurons mean ± std | 499.7 ± 191.5 |
| Frames per session | 36000 (jm031/jm032) or 54000 (others) |
| Frame rate | 30 Hz |
| Session duration | 20 min (jm031/jm032) or 30 min (others) |
| Sessions with frame mismatch | 8 (missing camera frames) |

### Frame Mismatches (neural vs behavioral)
- jm031: 2023-10-20 (2 missing), 2023-10-21 (3 missing), 2023-10-22 (116 missing)
- jm032: 2023-10-20 (2 missing), 2023-10-21 (2 missing), 2023-10-22 (148 missing)
- jm039: 2024-05-04 (1 missing)
- jm040: 2024-05-04 (1 missing)
- jm046: 2024-09-09 (1 missing)

### Timestamp Analysis
- tstamps.npy contains cumulative camera timestamps in kiloseconds
- Verified: tstamps[-1] * 1000 ≈ expected session duration in seconds
- ifi (inter-frame interval) mean * 1000 ≈ 0.0336 seconds → ~29.76 Hz ≈ 30 Hz

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|---------------|
| Subjects | 6 mice | "full dataset of 6 mice" |
| Sessions / subject | ≥6 consecutive days | "imaged daily for a minimum of 6 consecutive days" |
| Age range | P7-P14 | "within the second postnatal week (P7 to P14)" |
| Neurons tracked/mouse | 526 ± 190 std | "On average 526 (± 190 std) neurons per mouse" |
| Fraction tracked | 33% ± 11% std | "corresponding to 33% (± 11% std) of neurons detected on first day" |
| Frame rate | 30 Hz | "Imaging rate was 30 Hz (resonant scanner)" |
| Session duration | 20 minutes | "each session lasted 20 minutes" |
| FOV size | 720×720 μm | "720×720 μm FOV" |
| Pixel resolution | 512×512 | "512 × 512 pixel resolution" |
| Neural data time bin | 10 frames (333 ms) | "averaging using a bin size of 10 frames" |
| Behavior data time bin | 10 frames (333 ms) | "averaging in bins of 10 consecutive timestamps" |
| Cell classification threshold | 0.5 (default) | "default threshold of 0.5 as true cells" |
| Neuropil coefficient | 0.7 | Suite2p default |
| Decoding CV | 5-fold nested | "5 fold splits for both inner and outer loops" |
| CV block size | 2 minutes | "splits were done on consecutive 2 minute blocks" |

### Processing Details
1. **Neural processing**: 
   - Suite2p preprocessing: motion correction, ROI detection, signal extraction, spike deconvolution
   - dF/F: baseline corrected fluorescence (Suite2p default parameters)
   - Binning: average in bins of 10 frames
2. **Behavior processing**:
   - Motion energy: pixel-wise difference of consecutive video frames, squared, summed
   - Binning: average in bins of 10 frames
3. **Decoding**: Ridge regression, nested 5-fold CV on 2-minute blocks

### Curation Steps

**Neuron curation rules**:
- Suite2p iscell classifier with threshold 0.5
- Track2p matching across all days (only cells present all days kept)
- In our data: all cells already pass these filters (iscell all 1.0)

**Trial curation rules**:
- No explicit trial curation mentioned (continuous recording, no discrete trials)
- Missing camera frames need interpolation or treatment as missing values

### Decoders Trained (in paper)
| Decoded variable | Method | Metric |
|------------------|--------|--------|
| Motion energy (continuous) | Ridge regression | R² |

### Note on Session Duration
The paper says "each session lasted 20 minutes" but jm038-jm046 have 54000 frames = 30 minutes at 30 Hz. The 20-minute statement may apply to the specific mouse discussed in the first part of the paper.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Neurons/mouse | N/A | 221,370,685,746,541,435 (mean=499.7±191.5) | 526±190 | Close match, within std |
| Session duration | N/A | 20 min (jm031/32) or 30 min (others) | 20 min | Some mice had 30 min sessions |
| iscell filtering | iscell[:,0]==1 or iscell[:,1]>thr | all iscell=1 | threshold 0.5 | Data already filtered by Track2p |
| Timestamp units | N/A | max ~1.21 for 36000 frames | N/A | Confirmed kiloseconds (×1000 for seconds) |
| Missing frames | README mentions | 8 sessions affected | README mentions | Interpolation used |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| F.npy, Fneu.npy | neural | dF/F = preprocess(F - 0.7*Fneu), bin 10 frames | Suite2p baseline correction |
| frame index | input[0] (time) | time_seconds = frame_idx / 3.0 + trial_start | Time elapsed from session start |
| motion_energy_glob.npy | output[0] | Bin 10 frames, discretize to 5 equal-percentile bins per session | Per-session percentile bins |
| subject folder name | subjects | Direct mapping | jm031-jm046 |
| session folder name | metadata | Session date info | |

### Key Decisions
1. **Neural data**: Use dF/F (F - 0.7*Fneu, baseline corrected via Suite2p maximin), NOT raw F or spks. Rationale: Paper explicitly says "baseline corrected fluorescence traces as our dF/F" for all analyses.
2. **Binning**: Average in bins of 10 frames for both neural and behavioral data. Rationale: Paper says "slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps".
3. **Time bin size**: After binning, effective rate = 3 Hz, bin size = 333.33 ms.
4. **Trial definition**: Split each session into 60-second trials (180 binned timepoints per trial). Rationale: Task spec says "Split sessions into 60-second trials".
5. **Missing camera frames**: Interpolate motion energy to match neural frame count using tstamps. Rationale: Data README says "indices of missing frames can be... interpolated over".
6. **Motion energy discretization**: 5 equal-percentile bins per session (quintiles). Rationale: Task spec says "discretized into five equal-percentile bins, selected per session".
7. **Brain region**: All recordings from barrel cortex. Single brain region.
8. **Input**: Time elapsed from beginning of session in seconds (time-varying).
9. **Incomplete last trial**: Drop the last trial if it has fewer timepoints than a full 60-second trial (180 bins).

### Planned Sanity Checks
- [x] Verify neuron counts match paper (526 ± 190 per mouse) → 499.7 ± 191.5 ✓
- [x] Verify session durations match expected (20 or 30 min) ✓
- [x] Verify dF/F values are reasonable (not all zeros, reasonable range) ✓
- [x] Verify motion energy discretization produces ~equal bin counts ✓
- [x] Verify trial count: 20 min session → 20 trials, 30 min session → 30 trials ✓
- [x] Verify binned timepoints per trial = 180 ✓
- [x] Cross-check specific neural values against manual computation ✓

---

## Step 6: Script Development
**Status**: COMPLETE

Implementation notes:
- Script at `/app/convert_data.py`
- Uses Suite2p's `preprocess` function for dF/F computation (baseline_maximin)
- Handles missing camera frames via interpolation
- Bins data in 10-frame bins (3 Hz effective rate)
- Splits into 60-second trials (180 timepoints each)
- Discretizes motion energy into 5 equal-percentile bins per session
- Supports --full, --sample, --show-processing modes

Code inefficiencies identified:
- None significant - each session processes in ~0.5-0.7s

Code speedups added:
- Uses GPU for Suite2p baseline computation when available
- Vectorized binning operation

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 906 (221 + 685) |
| Neurons / session | 221, 685 |
| Subjects | 2 (jm031, jm038) |
| Sessions / subject | 1 each |
| Trials (total) | 50 (20 + 30) |
| Trials / session | 20, 30 |
| Timepoints / trial | 180 |
| Time bin size | 333.33 ms |
| Input range (time_seconds) | [0.0, 1799.7] |
| Output distribution (motion_energy) | 0.200 each (perfectly uniform) |

### Processing Plots Review
- Processing plots saved for both sessions
- dF/F looks reasonable (not all zeros, reasonable range)
- Motion energy alignment looks correct
- Discretization produces uniform bin distribution

### Run Time Estimates
| Step | Time / Session | Notes |
|------|---------------|-------|
| Load | 0.05-0.26s | Varies with session size |
| dF/F | 0.38-0.49s | Suite2p baseline |
| Bin + split | <0.1s | Fast |
| Total | 0.53-0.73s | Per session |

Estimated full conversion time: 41 sessions × ~0.7s = ~29 seconds
Actual full conversion time: 22.9 seconds ✓

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| motion_energy | 0.6309 | 0.3430 | 0.2000 |

- Loss decreases consistently (89.86 → 1.16)
- Validation accuracy 1.7x chance level
- Training/validation gap expected with only 2 sessions

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 395.3 MB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Subjects | 6 | 6 | 6 | ✓ |
| Sessions total | ≥6/mouse | 41 (7+7+7+7+6+7) | 41 | ✓ |
| Sessions/subject | ≥6 | 7,7,7,7,6,7 | 7,7,7,7,6,7 | ✓ |
| Neurons/mouse | 526±190 | 221,370,685,746,541,435 | same | ✓ |
| Mean neurons | 526±190 | 499.7±191.5 | 498.7±182.5 | ✓ (close) |
| Total neurons (all sessions) | N/A | N/A | 20445 | N/A |
| Trials total | N/A | N/A | 1090 | N/A |
| Trials/session | N/A | N/A | 20 or 30 | ✓ |
| Timepoints/trial | N/A | N/A | 180 | ✓ |
| Time bin size | 333.33 ms | N/A | 333.33 ms | ✓ |
| Output bins | 5 (quintiles) | N/A | 5 (0.200 each) | ✓ |
| Frame rate | 30 Hz | 30 Hz | 30 Hz | ✓ |
| Bin size | 10 frames | N/A | 10 frames | ✓ |

Total conversion time: 22.9s

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Checks Performed

**Check 1: Output log verification**
- `/app/verification_full_out.txt`: No errors, no warnings
- Data format valid
- All statistics consistent

**Check 2: Sanity checks (loading original data independently)**

1. **Neural data sanity check**: Loaded raw F.npy and Fneu.npy for session 0, manually computed dF/F using Suite2p's preprocess function, binned, and compared with converted data at trial 5, neuron 10. Result: `np.allclose` PASSED (atol=1e-4).

2. **Input data sanity check**: Verified time values for trial 0 start at 0.0 and increment by 1/3 seconds. Verified trial 1 starts at 60.0 seconds. Result: PASSED.

3. **Output data sanity check**: Loaded raw motion energy, binned manually, computed percentile bin edges, discretized first 180 bins. Compared with converted output trial 0. Result: `np.array_equal` PASSED.

4. **Missing frames session check**: Verified session 4 (jm031/2023-10-22_a, 116 missing camera frames) has correct shape (221, 180). PASSED.

5. **Neuron count check**: Verified all 41 sessions have correct neuron counts matching raw data. PASSED.

6. **Brain region idx check**: Verified all sessions have correct brain_region_idx (all zeros, correct length). PASSED.

**Check 3: Reference code comparison**

| Processing Step | My Code | Reference Code/Paper | Match? |
|----------------|---------|---------------------|--------|
| (a) Data loading | Load F.npy, Fneu.npy from suite2p/plane0/ | Same (load_data.ipynb) | ✓ |
| (b) Neuron filtering | All neurons used (iscell all 1.0) | Track2p outputs only tracked cells | ✓ |
| (c) dF/F computation | F - 0.7*Fneu, then Suite2p preprocess (maximin baseline) | "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" | ✓ |
| (d) Binning | Average in bins of 10 frames | "averaging in bins of 10 consecutive timestamps" | ✓ |
| (e) Input construction | Time elapsed from session start in seconds | Task specification requirement | ✓ |
| (f) Output construction | Motion energy discretized into 5 equal-percentile bins per session | Task specification requirement | ✓ |
| (g) Missing frames | Interpolation using tstamps | README: "interpolated over" | ✓ |

**Check 4: Key statistics comparison**

| Statistic | Paper | Data | Converted | Match? |
|-----------|-------|------|-----------|--------|
| Subjects | 6 | 6 | 6 | ✓ |
| Sessions/subject | ≥6 | 6-7 | 6-7 | ✓ |
| Neurons/mouse (mean±std) | 526±190 | 499.7±191.5 | 498.7±182.5 | ✓ |
| Frame rate | 30 Hz | 30 Hz | 30 Hz | ✓ |
| Session duration | 20 min | 20 or 30 min | 20 or 30 min | ✓ |

Note: The mean neurons/mouse (499.7) is slightly lower than the paper's reported 526±190. This is within the standard deviation and likely due to the specific subset of mice in our dataset vs. the paper's full dataset (which may include the 7th mouse mentioned in the single-mouse analysis).

**Check 5: Edge cases**
- Missing camera frames: Handled by interpolation (8 sessions affected)
- Incomplete last trial: All sessions have exact multiples of 180 bins after binning, so no trials are dropped
- Off-by-one: Verified trial boundaries are correct (trial 0 = bins 0-179, trial 1 = bins 180-359, etc.)
- All neural values are float32, all output values are int64

### Issues Found and Resolved
- Initial issue: Output values were float32, causing TypeError in decoder.py. Fixed by changing to int64.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (83.23 → 1.08 over 200 epochs)
- Test loss: 3.30

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Ratio to Chance |
|--------|-------------|--------|-------|--------|
| motion_energy | 0.6337 | 0.2972 | 0.2000 | 1.49x |

Notes:
- Validation accuracy is 1.49x chance, indicating the decoder is learning meaningful patterns
- The training/validation gap suggests some overfitting (expected with high-dimensional data)
- The paper uses ridge regression for decoding (R² metric), not neural network classification
- Cross-session decoding with different neuron counts is inherently challenging

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Chance | Ratio | Notes |
|----------|------------------|--------|-------|-------|
| motion_energy | 0.2972 (validation) | 0.2000 | 1.49x | Above chance |

**Check 1: Accuracy vs chance analysis**
- Validation accuracy 0.2972 is 1.49x chance (0.2000)
- This is acceptable for the following reasons:
  1. The paper uses continuous R² with ridge regression (same-day, per-session), not 5-bin classification across sessions
  2. Early developmental sessions (≤P11) have poor decoding performance per the paper
  3. The neural network decoder must generalize across 6 different animals with different neuron counts (221-746)
  4. Training accuracy is 0.6337, showing the model can learn the relationship within training data

**Check 2: Accuracy comparison to paper**
- The paper reports R² values for continuous motion energy prediction using ridge regression
- R² values are shown in figures and vary from ~0 (early days) to moderate values (late days)
- Direct comparison not possible due to different metrics (5-bin classification vs continuous R²)
- The paper shows that decoding performance is poor for early developmental stages (≤P11) and improves for later stages
- Given that ~50% of our sessions are from early developmental stages, moderate accuracy is expected

**Check 3: Train vs validation gap**
- Training: 0.6337, Validation: 0.2972
- Gap ratio: 2.13x (training is 2.13x validation)
- This gap is expected given:
  1. High-dimensional data (up to 746 neurons) with relatively few trials per session
  2. Cross-session generalization is inherently harder than within-session
  3. The PCA reduction to 100 components helps but doesn't fully prevent overfitting

**Debugging investigation:**
1. Verified output values are correct by loading raw data and checking specific trials ✓
2. Verified temporal alignment - neural and ME are from the same frames ✓
3. Output has sufficient variation (perfectly uniform 5 bins, 20% each) ✓
4. Neural data has reasonable statistics (no zero-variance neurons, no NaN/Inf) ✓
5. Processing matches reference code and paper ✓

### Issues Found and Resolved
- No additional issues found. The accuracy is reasonable given the cross-session, cross-animal nature of the decoding task.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with README_CACHE.md
- [x] Processing plots moved to cache/
- [x] All 11 required files verified present
