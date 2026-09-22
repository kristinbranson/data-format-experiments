# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p - Longitudinal tracking of neuronal activity in developing mouse barrel cortex (Majnik et al. 2025)
- **Date started**: 2024
- **Goal**: Convert to decoder-compatible format for decoding motion energy from neural activity

## Step 0: Setup and Initialization
**Status**: COMPLETE

Directory contents:
- `/app/paper.pdf` - Reference paper
- `/app/methods.txt` - Methods text extracted from paper
- `/app/code/` - Track2p code repository
- `/app/data/` - Dataset with 6 mice (jm031-jm046)
- `/app/decoder.py` - Decoder implementation
- `/app/train_decoder.py` - Decoder training script

Python environment verified: numpy 2.4.4, torch 2.6.0+cu124

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|----------|
| load_traces | data/load_data.ipynb | LOADING | Loads raw F.npy fluorescence traces |
| F_processing | track2p/gui/data_management.py | PROCESSING | Computes baseline-corrected fluorescence: neuropil subtraction + maximin baseline |
| load_stat_ds_plane | track2p/io/loaders.py | LOADING | Loads stat and filters by iscell threshold |
| zscore_rows | data/load_data.ipynb | PROCESSING | Z-scores rows for visualization |
| npy_to_s2p | notebooks/utils/npy_to_s2p.ipynb | LOADING | Converts numpy format to Suite2p format |

### Notes
- **Baseline-corrected fluorescence** (from `data_management.py` `F_processing`):
  - Neuropil subtraction: `Fc = F - neucoeff * Fneu`
  - GUI code uses `neucoeff=0.0` but Suite2p ops has `neucoeff=0.7`
  - Paper says "using the default Suite2p parameters" → use neucoeff=0.7
  - Baseline: maximin method with `sig_baseline=10.0`, `win_baseline=60.0`
  - `Flow = gaussian_filter(Fc, [0., sig_baseline])` → `minimum_filter1d(Flow, win)` → `maximum_filter1d(Flow, win)`
  - Final: `dF = Fc - Flow` (baseline subtracted, NOT divided by baseline)
  - **Critical**: The code does NOT divide by Flow. Division would cause extreme values when Flow is near zero or negative.
- **Data already filtered by Track2p**: All iscell values are 1.0, neurons are tracked across all days
- **No decoding code** in the Track2p repository - decoding was done separately using PyTorch ridge regression
- Track2p saves matched neurons with same row index across sessions

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
data/
  jm031/  (Mouse A, 7 sessions, 2023-10-18 to 2023-10-24)
  jm032/  (Mouse B, 7 sessions, 2023-10-18 to 2023-10-24)
  jm038/  (Mouse C, 7 sessions, 2023-04-30 to 2023-05-06)
  jm039/  (Mouse D, 7 sessions, 2024-04-30 to 2024-05-06)
  jm040/  (Mouse E, 6 sessions, 2024-05-01 to 2024-05-06)
  jm046/  (Mouse F, 7 sessions, 2024-09-03 to 2024-09-09)

Each session:
  suite2p/plane0/
    F.npy       - Raw fluorescence (n_neurons x n_frames), float32
    Fneu.npy    - Neuropil fluorescence (n_neurons x n_frames), float32
    spks.npy    - Deconvolved spikes (n_neurons x n_frames), float32
    iscell.npy  - Cell classification (n_neurons x 2), all 1s (pre-filtered by Track2p)
    ops.npy     - Suite2p options dict (fs=30, baseline=maximin, neucoeff=0.7)
    stat.npy    - Cell statistics (ROI info)
  move_deve/
    motion_energy_glob.npy - Motion energy (n_camera_frames,), uint64
    tstamps.npy            - Camera timestamps (n_camera_frames,), float64
    interframe_int.npy     - Inter-frame intervals (n_camera_frames-1,), float64
```

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 6 |
| Sessions total | 41 |
| Sessions / subject | 6-7 (jm040 has 6, rest have 7) |
| Neurons (jm031) | 221 |
| Neurons (jm032) | 370 |
| Neurons (jm038) | 685 |
| Neurons (jm039) | 746 |
| Neurons (jm040) | 541 |
| Neurons (jm046) | 435 |
| Mean neurons/mouse | 499.7 |
| Frames (jm031/jm032) | 36000 (20 min at 30 Hz) |
| Frames (jm038-jm046) | 54000 (30 min at 30 Hz) |
| Frame rate | 30 Hz |

### Missing Camera Frames
Sessions with fewer camera frames than neural frames:
- jm031: 2023-10-20 (2 missing), 2023-10-21 (3), 2023-10-22 (116)
- jm032: 2023-10-20 (2), 2023-10-21 (2), 2023-10-22 (148)
- jm039: 2024-05-04 (1)
- jm040: 2024-05-04 (1)
- jm046: 2024-09-09 (1)

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 6 mice | "full dataset of 6 mice" |
| Sessions/subject | min 6 consecutive days | "imaged daily for a minimum of 6 consecutive days" |
| Age range | P7-P14 | "within the second postnatal week (P7 to P14)" |
| Neurons/mouse (mean±std) | 526 ± 190 | "On average 526 (± 190 std) neurons per mouse" |
| Fraction tracked | 33% ± 11% | "corresponding to 33% (± 11% std) of neurons detected on first day" |
| FOV | 720 x 720 μm | "720 × 720 µm field of view" |
| Resolution | 512 x 512 pixels | "512 × 512 pixel resolution" |
| Imaging rate | 30 Hz | "Imaging rate was 30 Hz (resonant scanner)" |
| Session duration | 20 minutes | "each session lasted 20 minutes" |
| Camera rate | 30 Hz | "Videos were recorded at 30 Hz" |
| Brain region | Barrel cortex L2/3 | "layer 2/3, depth 100-200 µm" |
| Calcium indicator | GCaMP8m | |
| Cell classification threshold | 0.5 | "default threshold of 0.5 as true cells" |
| Binning for decoding | 10 frames | "averaging in bins of 10 consecutive timestamps" |

### Processing Details

**Baseline-corrected fluorescence (from paper)**:
- "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)"
- Suite2p default: Fc = F - 0.7 * Fneu, then maximin baseline estimation
- Reference code implements: dF = Fc - Flow (no division by baseline)

**Decoding (from paper)**:
- Ridge regression with nested cross-validation
- 5-fold splits on consecutive 2-minute blocks
- Binning: 10 consecutive timestamps for both neural and behavioral data
- Decoded motion energy from neural activity

**Motion energy (from paper)**:
- "pixel-wise difference of consecutive frames, squared, summed across pixels"
- Yields scalar value per timepoint

### Curation Steps

**Neuron curation rules**:
- Suite2p cell classification: iscell probability > 0.5 (default threshold)
- Track2p tracking: only neurons tracked across ALL days are included
- Data already pre-filtered: all iscell values = 1.0

**Trial curation rules**:
- No explicit trial curation (spontaneous behavior, no task trials)

### Decoders Trained in Paper
| Decoded variable | Method | Notes |
|-----------------|--------|-------|
| Motion energy | Ridge regression | Nested 5-fold CV on 2-min blocks |

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| neucoeff | 0.0 (GUI) | ops: 0.7 | "default Suite2p parameters" | Use 0.7 (paper says default Suite2p) |
| Session duration | - | 36000 or 54000 frames | 20 minutes | jm031/jm032 = 20min, others = 30min. Paper says 20 min but data shows some are 30 min |
| Neurons/mouse | - | mean 499.7 | 526 ± 190 | Within 1 std, consistent |
| dF/F formula | Fc - Flow (no division) | - | "baseline corrected fluorescence traces as our dF/F" | Use Fc - Flow as in code. Division causes extreme values due to near-zero Flow |
| Missing frames | - | Some sessions have fewer ME frames | "missing frames can be interpolated" | Interpolate ME to match neural frame count |

### Resolution Notes
- The 30-min sessions for jm038-jm046 vs 20-min for jm031/jm032 is a real difference in the data. The paper says "20 minutes" but this may refer to a subset. All data is included.
- The dF/F terminology in the paper actually refers to baseline-subtracted fluorescence (Fc - Flow), not true dF/F (Fc - Flow)/Flow. The code confirms this.

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| F.npy, Fneu.npy | neural | Fc=F-0.7*Fneu, maximin baseline, Fc-Flow, bin by 10 | (n_neurons, n_timebins) per trial |
| time index | input[0] | Time in seconds from session start, at binned resolution | (1, n_timebins) per trial |
| motion_energy_glob.npy | output[0] | Interpolate missing frames, bin by 10, discretize into 5 equal-percentile bins per session | (1, n_timebins) per trial |

### Key Decisions
1. **dF/F = Fc - Flow**: Follow reference code (baseline subtraction without division). Division by Flow causes extreme values when Flow is near zero or negative.
2. **neucoeff = 0.7**: Follow paper ("default Suite2p parameters"). GUI code uses 0.0 but this is for visualization, not analysis.
3. **Binning = 10 frames**: Paper explicitly states this for decoding analysis.
4. **Trial duration = 60 seconds**: Task specification. At 3 Hz (after binning), each trial = 180 time bins.
5. **5 equal-percentile bins per session**: Task specification for motion energy discretization.
6. **Missing camera frames**: Interpolate using np.interp based on gap detection in interframe intervals.
7. **All neurons included**: Data is already filtered by Track2p (only tracked cells across all days).
8. **Brain region**: barrel_cortex (S1, layer 2/3).

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `/app/convert_data.py` with:
- `compute_dff()`: Baseline-corrected fluorescence using Suite2p default parameters
- `interpolate_missing_frames()`: Handles missing camera frames via interpolation
- `bin_data()`: Averages data in bins of 10 frames
- `discretize_motion_energy()`: 5 equal-percentile bins per session
- `process_session()`: Full processing pipeline for one session
- `plot_processing()`: Visualization of all processing steps

Initial bug found and fixed: dF/F computation used division by Flow, causing extreme values. Fixed to use subtraction only (Fc - Flow) matching reference code.

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (jm031 first 2 sessions) |
| Neurons / session | 221 |
| Trials / session | 20 |
| Total trials | 40 |
| Time bins / trial | 180 |
| Input range | [0.0, 1199.7] seconds |
| Output distribution | [0.2, 0.2, 0.2, 0.2, 0.2] (uniform) |

### Processing Plots Review
Plots saved for first 2 sessions showing all processing steps. No anomalies observed.

### Run Time Estimates
| Step | Time / Session | Notes |
|------|---------------|-------|
| Load | 0.02-0.15s | Depends on file size |
| dF/F | 0.25-0.95s | Depends on n_neurons |
| Binning | 0.01-0.06s | Fast |
| Total | 0.3-1.2s | Per session |
| Full dataset | ~41s | 41 sessions |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|----------------------|------------------------|--------|
| motion_energy | 0.3611 | 0.2660 | 0.2000 |

Validation accuracy is 1.33x chance with only 2 sessions - expected to improve with full data.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 395.2 MB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Subjects | 6 | 6 | 6 | ✓ |
| Sessions | ≥6/mouse | 41 total | 41 total | ✓ |
| Sessions/subject | ≥6 | 7,7,7,7,6,7 | 7,7,7,7,6,7 | ✓ |
| Neurons/mouse (mean) | 526 ± 190 | 499.7 | 498.66 | ✓ (within 1 std) |
| Total neurons | - | 2998 unique | 20445 across sessions | ✓ |
| Frame rate | 30 Hz | 30 Hz | 30 Hz (binned to 3 Hz) | ✓ |
| Trials/session | - | - | 20 or 30 | ✓ |
| Total trials | - | - | 1090 | ✓ |
| Output bins | 5 | - | 5 | ✓ |
| Output distribution | uniform | - | [0.2 each] | ✓ |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
`/app/verification_full_out.txt` shows:
- No errors
- No warnings
- All statistics consistent

### Check 2: Sanity checks (all passed)
1. **Neural data**: Spot-checked trials 0, 5, 10, 19 of session 0 (jm031 first session). Recomputed dF/F from raw F.npy and Fneu.npy, binned, and compared. Max difference = 0.000000 for all trials. ✓
2. **Input data**: Verified time values for trials 0, 5, 19. Trial 0 starts at 0.0s, trial 5 at 300.0s, trial 19 at 1140.0s. All correct. ✓
3. **Output data**: Recomputed motion energy discretization from raw data. Exact match for trials 0, 5, 19. ✓
4. **Missing frames**: Verified session jm031/2023-10-22_a (116 missing frames). Neural data matches exactly. ✓
5. **Cross-mouse**: Verified jm038 first session (685 neurons). Neural data matches exactly. ✓
6. **Neuron consistency**: Each mouse has consistent neuron count across all sessions (as expected from Track2p tracking). ✓
7. **Subject index**: Verified subject_idx array matches expected mapping. ✓

### Check 3: Reference code comparison
| Processing Step | My Code | Reference Code | Match? |
|----------------|---------|----------------|--------|
| Data loading | F.npy, Fneu.npy from suite2p/plane0 | Same (load_data.ipynb) | ✓ |
| Neuropil correction | Fc = F - 0.7 * Fneu | F_processing: Fc = F - neucoeff * Fneu (neucoeff=0.0 in GUI, 0.7 in ops) | Paper says "default Suite2p" = 0.7 |
| Baseline | maximin: gaussian_filter → min_filter → max_filter | Same in F_processing | ✓ |
| Baseline params | sig=10.0, win=60.0*30=1800 frames | Same in F_processing | ✓ |
| Final dF | Fc - Flow | F_processing: F = Fc - Flow | ✓ |
| Binning | Average 10 consecutive frames | Paper: "averaging in bins of 10 consecutive timestamps" | ✓ |
| Motion energy | Load motion_energy_glob.npy | From move_deve directory | ✓ |
| Missing frames | Interpolate based on ifi gaps | README: "interpolated over" | ✓ |

### Check 4: Key statistics comparison
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| N subjects | 6 | 6 | ✓ |
| N sessions/subject | ≥6 | 6-7 | ✓ |
| Mean neurons/mouse | 526 ± 190 | 499.7 | ✓ (within range) |
| Frame rate | 30 Hz | 30 Hz | ✓ |
| Binning | 10 frames | 10 frames | ✓ |

### Check 5: Edge cases
- **Missing camera frames**: Handled by interpolation. Verified for session with 116 missing frames.
- **Session length differences**: jm031/jm032 have 36000 frames (20 min), others have 54000 frames (30 min). Both handled correctly with different trial counts.
- **Off-by-one**: Last bin of each trial verified to be correct. No overlap between trials.

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (96.1 → 1.09 over 200 epochs)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|----------------------|------------------------|--------|-------|
| motion_energy | 0.6291 | 0.3120 | 0.2000 | 1.56x chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Accuracy Analysis

| Variable | Achieved Accuracy | Chance | Ratio | Notes |
|----------|------------------|--------|-------|-------|
| motion_energy | 0.3120 | 0.2000 | 1.56x | Above chance, reasonable for 5-class problem |

### Check 1: Accuracy vs chance
- Validation accuracy (0.3120) is 1.56x chance (0.2000) - clearly above chance ✓
- Training accuracy (0.6291) is 3.15x chance - model learns the relationship

### Check 2: Accuracy comparison to paper
- The paper uses ridge regression with nested 5-fold CV, not a neural network decoder
- The paper does not report specific decoding accuracy numbers for motion energy in the text
- The paper focuses on cross-day decoding stability rather than absolute accuracy
- Our decoder architecture (neural network with PCA) differs from the paper's (ridge regression)
- The 0.312 validation accuracy is reasonable given the 5-class discretization and the nature of the data

### Check 3: Train vs validation gap
- Training: 0.6291, Validation: 0.3120, ratio = 2.02x
- Some overfitting present, but expected with the model complexity and data size
- The decoder uses 100 PCs which may be more than optimal for some sessions

### Issues Found and Resolved
- **Initial dF/F bug**: Division by Flow caused extreme values (max > 60M). Fixed to use subtraction only (Fc - Flow) matching reference code. This improved validation accuracy from 0.2137 to 0.3120.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
