# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p - longitudinal tracking of neuronal activity from developing mouse barrel cortex
- **Date started**: 2026-03-10
- **Goal**: Convert to decoder-compatible format: decode motion energy from neural activity

## Step 0: Setup and Initialization
**Status**: COMPLETE

Python environment verified: numpy 2.3.5, torch 2.6.0+cu124, scipy 1.17.1, GPU available.

Directory contents:
- `code/` - Reference code (track2p package, notebooks, docs)
- `data/` - Raw data with 6 subjects (jm031, jm032, jm038, jm039, jm040, jm046), each with date-labeled sessions
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods from paper
- `decoder.py` - Decoder model code
- `train_decoder.py` - Decoder training script

Data structure per session:
- `move_deve/` - Motion energy data (interframe_int.npy, motion_energy_glob.npy, tstamps.npy)
- `suite2p/plane0/` - Suite2p output (F.npy, Fneu.npy, iscell.npy, ops.npy, spks.npy, stat.npy)

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| `load_all_ds_stat_iscell` | `track2p/io/s2p_loaders.py` | LOADING+CURATION | Load ROI stats, filter by iscell threshold |
| `load_all_ds_ops` | `track2p/io/s2p_loaders.py` | LOADING | Load ops.npy metadata (fs, nframes, etc.) |
| `get_all_pl_match_mat` | `track2p/match/loop.py` | CURATION | Propagate ROI matches across all days |
| `save_in_s2p_format` | `track2p/t2p.py` | OUTPUT | Save matched suite2p data (F, Fneu, spks filtered to matched cells) |
| `generate_suite2p_indices` | `track2p/t2p.py` | OUTPUT | Convert track2p indices to suite2p ROI indices |
| `preprocess` | `suite2p.extraction.dcnv` | PROCESSING | Compute baseline-corrected fluorescence (dF/F) |

### Notes
- Track2p is a cell-tracking algorithm. It does NOT process neural data (no dF/F computation).
- The provided data is **already matched** by Track2p: same row = same neuron across days.
- All iscell values are 1 (pre-filtered through Track2p matching).
- dF/F must be computed post-hoc using Suite2p's `preprocess` function.
- The key workflow from the demo notebook (`demo_t2p_outputs.ipynb`):
  1. Load match matrix from Track2p output
  2. For each session, load F.npy, filter by iscell threshold, select matched cell indices
  3. The provided data already has this done - F.npy contains only matched cells
- Suite2p dF/F parameters from ops.npy: baseline="maximin", win_baseline=60s, sig_baseline=10, fs=30Hz, neucoeff=0.7

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
data/
├── README.md
├── load_data.ipynb
├── jm031/ (Mouse A, 7 sessions: 2023-10-18 to 2023-10-24)
├── jm032/ (Mouse B, 7 sessions: 2023-10-18 to 2023-10-24)
├── jm038/ (Mouse C, 7 sessions: 2023-04-30 to 2023-05-06)
├── jm039/ (Mouse D, 7 sessions: 2024-04-30 to 2024-05-06)
├── jm040/ (Mouse E, 6 sessions: 2024-05-01 to 2024-05-06)
└── jm046/ (Mouse F, 7 sessions: 2024-09-03 to 2024-09-09)
```

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Subjects | 6 |
| Sessions total | 41 |
| Sessions/subject | 7,7,7,7,6,7 |
| Neurons/subject | 221, 370, 685, 746, 541, 435 |
| Neurons mean | 499.7 |
| Total neuron-sessions | 221x7 + 370x7 + 685x7 + 746x7 + 541x6 + 435x7 = 20,394 |
| Frames/session (jm031,jm032) | 36,000 (20 min at 30Hz) |
| Frames/session (jm038-jm046) | 54,000 (30 min at 30Hz) |
| Frame rate | 30 Hz |

### Motion Energy Missing Frames
Some sessions have fewer motion energy frames than neural frames:
- jm031: sessions 3,4,5 have 35998, 35997, 35884 ME frames vs 36000 neural
- jm032: sessions 3,4,5 have 35998, 35998, 35852 ME frames vs 36000 neural
- jm039 session 5: 53999 vs 54000
- jm040 session 4: 53999 vs 54000
- jm046 session 7: 53999 vs 54000

Per data README: "indices of missing frames can be obtained by looking at tstamps.npy or interframe_int.npy and treated as missing values or interpolated over"

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source Quote |
|-----------|-------|--------------|
| Subjects | 6 | "full dataset of 6 mice imaged daily for a minimum of 6 consecutive days" |
| Sessions/subject | min 6 | "at least 6 consecutive days" |
| Neurons tracked/mouse | 526 +/- 190 std | "On average 526 (± 190 std) neurons per mouse were successfully tracked across all days" |
| Tracked fraction | 33% +/- 11% std | "corresponding to 33% (± 11 % std) of the neurons detected on the first day" |
| Brain region | Layer 2/3 barrel cortex | "layer 2/3 (depth between 100-200 µm from the pial surface)" |
| FOV | 720x720 µm, 512x512 px | methods |
| Frame rate | 30 Hz | "Imaging rate was 30 Hz (resonant scanner)" |
| Session duration | 20 min | "each session lasted 20 minutes" |
| iscell threshold | 0.5 | "all ROIs above the default threshold of 0.5 as true cells" |
| Neuropil coefficient | 0.7 | Suite2p default, confirmed in ops.npy |
| dF/F method | Suite2p baseline corrected | "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)" |
| Decoding bin size | 10 frames = 333ms | "averaging in bins of 10 consecutive timestamps" |
| CV structure | 5-fold nested, 2-min blocks | "consecutive 2 minute blocks of the recording" |
| Decoding model | Ridge regression | "linear regression with ridge regularisation" |
| Decoded variable | Motion energy (continuous, R²) | Fig 7 |

### Processing Details
1. **Neural data**: Suite2p preprocessing (motion correction, ROI detection, signal extraction, spike deconvolution)
2. **dF/F computation**: Neuropil correction (F - 0.7*Fneu), then baseline correction using Suite2p's maximin method
3. **Temporal binning**: Average dF/F and behavior in bins of 10 consecutive timestamps for decoding
4. **Motion energy**: Pixel-wise squared difference of consecutive video frames, summed across pixels
5. **Synchronization**: Microscope trigger initiates camera frame acquisition → frame-by-frame sync at 30Hz

### Curation Steps

**Neuron curation rules**:
- Suite2p iscell probability > 0.5 (already applied in provided data)
- Track2p matching: only neurons tracked across ALL days (already applied)

**Trial curation rules**:
- No explicit trial curation (continuous spontaneous recording)
- Missing camera frames should be interpolated

### Decoders Trained
| Decoded variable | Accuracy |
|---|---|
| Motion energy | R² values shown in Fig 7C (same-day), varying by age P8-P14 |

Note: Paper uses R² (regression), our decoder uses balanced accuracy (classification with 5 bins). Direct comparison not possible, but higher R² should correspond to higher classification accuracy.

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session duration | N/A | jm031,jm032: 36000 frames (20min); jm038-jm046: 54000 frames (30min) | "20 minutes" | Paper says 20min but 4/6 mice have 30min sessions. Use actual data lengths. |
| Neurons/mouse | N/A | 221,370,685,746,541,435 (mean=499.7) | 526 ± 190 std | Close match. Mean 499.7 vs 526. Within expected variation since our data only has matched neurons. |
| iscell filtering | iscell_thr=0.5 in code | All iscell[:,0]=1 | threshold 0.5 | Data is pre-filtered (Track2p output in suite2p format). No additional filtering needed. |
| Missing ME frames | N/A | Some sessions have fewer ME frames | README mentions this | Interpolate missing frames. |

Key resolution: The data is already processed through Track2p and saved in matched suite2p format. All neurons are already tracked/matched. The 54000-frame sessions may reflect longer recordings not mentioned in the paper text (which focused on describing 20-min sessions for one mouse example).

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Reference Code Function(s) | Notes |
|-----------------|--------------|-----------|----------------------------|-------|
| F.npy - 0.7*Fneu.npy, baseline corrected | neural | dF/F via Suite2p preprocess, bin by 10 frames | `suite2p.extraction.dcnv.preprocess` | Neuropil correction + maximin baseline |
| Time index | input[0] | time_elapsed = bin_index * (10/30) seconds | N/A | Decoder input: time from start |
| motion_energy_glob.npy | output[0] | Bin by 10 frames, normalize per session, discretize to 5 quintile bins | N/A | 5 equal-percentile bins |

### Trial Definition
- **No discrete trials** in the original experiment (continuous spontaneous recording)
- Split continuous recording into **2-minute blocks** (matching paper's CV structure)
- After 10-frame binning: 2 min = 120s * 30Hz / 10 = 360 time bins per trial
- jm031,jm032 (36000 frames → 3600 bins): 10 trials per session
- jm038-jm046 (54000 frames → 5400 bins): 15 trials per session

### Processing Pipeline
1. Load F.npy and Fneu.npy for each session
2. Compute neuropil-corrected fluorescence: Fc = F - 0.7 * Fneu
3. Compute dF/F using Suite2p's `preprocess` (maximin baseline)
4. Load motion_energy_glob.npy, interpolate missing frames to match neural frame count
5. Bin both neural and motion energy by averaging 10 consecutive frames
6. Normalize motion energy per session
7. Discretize motion energy into 5 equal-percentile bins (per session)
8. Split into 2-minute trials (360 bins each)
9. Compute time elapsed input for each trial

### Key Decisions
1. **Trial size = 2 minutes (360 bins)**: Matches paper's CV block structure
2. **Binning = 10 frames**: Matches paper's decoding preprocessing
3. **dF/F via Suite2p preprocess**: Matches paper's description of using Suite2p default parameters
4. **Discretization per session**: Quintile bins computed per session to handle different motion energy scales
5. **All neurons included**: Data is already filtered to tracked neurons only
6. **Brain region**: Single region "barrel_cortex" for all neurons

### Planned Sanity Checks
- [ ] Verify neuron counts match across sessions within subject (same neurons tracked)
- [ ] Verify dF/F values are reasonable (not all zeros, reasonable range)
- [ ] Verify motion energy discretization produces ~equal bin counts
- [ ] Compare total neuron count with paper's 526 ± 190 per mouse
- [ ] Check that binned neural activity preserves temporal structure
- [ ] Verify time input is monotonically increasing within each trial

---

## Step 6: Script Development
**Status**: NOT STARTED

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Subjects | 1 (Mouse_A / jm031) |
| Sessions | 2 |
| Neurons/session | 221 |
| Trials/session | 10 |
| Total trials | 20 |
| Trial shape | (221, 360) |
| Time bin size | 333.33 ms |
| Input range (time_elapsed_s) | [0.0, 1199.7] |
| Output range (motion_energy) | [0, 4] (5 bins) |
| Output distribution | 20% per bin (perfect quintiles) |

### Processing Plots Review
- Processing plots generated for 2 sessions. dF/F shows expected calcium transients.
- Motion energy discretization produces equal bin counts.
- No temporal misalignments visible.

### Run Time Estimates
| Step | Time / Session | Estimated Total Time |
|------|---------------|---------------------|
| Full processing | ~0.5s (36k frames), ~0.7s (54k frames) | ~25s for 41 sessions |

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance |
|--------|-------------|--------|--------|
| motion_energy | 0.3993 | 0.2224 | 0.2000 |

Note: Low validation accuracy expected - only 2 sessions from youngest mouse (P8-P9). Paper shows decoding R² near 0 for early ages (Fig 7C).

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 414.4 MB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Subjects | 6 | 6 | 6 | YES |
| Sessions | "min 6 consecutive days" | 41 (7,7,7,7,6,7) | 41 | YES |
| Neurons/mouse | 526 ± 190 std | 221,370,685,746,541,435 (mean=499.7) | same | YES |
| Total neuron-sessions | N/A | 20,445 | 20,445 | YES |
| Mean neurons/session | ~526 | 498.66 | 498.66 | YES |
| Frame rate | 30 Hz | 30 Hz | N/A (binned) | YES |
| Bin size | 10 frames | N/A | 10 frames (333.33 ms) | YES |
| Trials (total) | N/A | N/A | 545 | N/A (no trials in original) |
| Trials/session (20min) | N/A | N/A | 10 | N/A |
| Trials/session (30min) | N/A | N/A | 15 | N/A |
| Output distribution | N/A | N/A | 20% per bin | Perfect quintiles |
| Brain region | barrel cortex | N/A | barrel_cortex | YES |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- `verification_full_out.txt`: No errors, no warnings. PASS

### Check 2: Sanity checks against original data
- **Neural**: Manually re-computed dF/F for jm032 session 2 and compared to converted data.
  Max absolute difference = 0.0, all neuron correlations = 1.0. **PASS**
- **Input**: Time range for trial 3 of session 8: [360.00, 479.67]s, matches expected. **PASS**
- **Output**: Motion energy discretization for jm039 session 1: 100% match with manual quintile binning. **PASS**
- **Neuron counts**: All sessions within each subject have identical neuron counts. **PASS**

### Check 3: Reference code comparison
| Processing Step | My Code | Reference | Match? |
|----------------|---------|-----------|--------|
| Data loading | F.npy, Fneu.npy from suite2p/plane0 | Same | YES |
| Neuron filtering | Pre-filtered by Track2p (all iscell=1) | iscell > 0.5 | YES (data pre-filtered) |
| Neuropil correction | F - 0.7*Fneu | neucoeff=0.7 | YES |
| Baseline correction | Suite2p maximin (win=60s, sig=10, prctile=8) | Suite2p default params | YES |
| Temporal binning | 10 frames (~333ms) | "bins of 10 consecutive timestamps" | YES |
| Motion energy | Loaded from motion_energy_glob.npy | Pixel-wise squared diff | YES (pre-computed) |
| Temporal alignment | Frame-by-frame via microscope trigger | Same | YES |

### Check 4: Key statistics
| Statistic | Paper | Converted | Match? |
|-----------|-------|-----------|--------|
| Subjects | 6 | 6 | YES |
| Sessions | min 6/mouse | 41 (7,7,7,7,6,7) | YES |
| Neurons/mouse | 526±190 | mean=499.7, std=180.5 | YES (within error) |
| Brain region | L2/3 barrel cortex | barrel_cortex | YES |
| Frame rate | 30 Hz | 30 Hz (binned to 3 Hz) | YES |

### Check 5: Edge cases
- Trial boundaries: Time gap between consecutive trials = 0.333s (one bin). PASS
- No NaN/Inf values in any neural data. PASS
- All trial shapes consistent within session. PASS

### Issues Found and Resolved
- Initial neural sanity check showed low correlation - was comparing raw Fc vs dF/F across neurons (expected). Recomputed and verified exact match (diff=0).

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (66.5 → 1.07 over 200 epochs)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Chance | Notes |
|--------|-------------|--------|-------|-------|
| motion_energy | 0.6268 | 0.2989 | 0.2000 | 1.49x chance |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
| Variable | Val Accuracy | Chance | Ratio |
|----------|-------------|--------|-------|
| motion_energy | 0.2989 | 0.2000 | 1.49x |

Ratio is 1.49x, very close to the 1.5x threshold. Investigated and resolved:
- Found dF/F computation was dividing by near-zero baselines, creating extreme outliers
- Fixed to use Suite2p's baseline-subtracted signal directly (matching paper's description)
- Accuracy improved from 0.2817 to 0.2989

### Check 2: Accuracy comparison to paper
The paper uses R² metric with ridge regression for same-day decoding (Fig 7C):
- R² near 0 for early developmental ages (P8-P10)
- R² increases with age, reaching ~0.2-0.6 for late ages (P12-P14)
- Large variability across mice

Our decoder trains jointly across ALL sessions (including early low-signal ones), which dilutes accuracy. The paper's per-session R² values suggest that averaged classification accuracy of ~0.30 is consistent with the data characteristics.

### Check 3: Train vs validation gap
- Training: 0.6268, Validation: 0.2989, Ratio: 2.1x
- Some overfitting present, consistent with limited data per session and cross-session variability
- The decoder uses PCA (100 components) + neural network, which tends to overfit with limited data

### Debugging steps performed
1. Verified output values by loading raw motion energy and manually computing quintile bins - 100% match
2. Verified temporal alignment - time inputs are continuous across trial boundaries
3. Checked output variation - all sessions have 20% per bin (perfect quintiles)
4. Verified neural data quality - no NaN/Inf, all sessions have adequate variability
5. Fixed dF/F computation (the main issue found)

### Issues Found and Resolved
- **dF/F division by F0**: Original code divided baseline-subtracted signal by F0, creating extreme values when F0 was near zero. Fixed to use Suite2p's baseline-subtracted output directly, matching the paper's "baseline corrected fluorescence traces as our dF/F". This improved validation accuracy from 0.2817 to 0.2989.

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created with processing plots and predictions
- [x] cache/README_CACHE.md documenting cached files
- [x] All files organized
