# Dataset Conversion Notes

## Overview
- **Dataset**: Track2p - longitudinal tracking of neuronal activity from barrel cortex (Majnik et al. 2025)
- **Date started**: 2026-03-10
- **Goal**: Convert to decoder-compatible format for decoding motion energy from neural activity

## Step 0: Setup and Initialization
**Status**: COMPLETE

Python environment verified:
- numpy 2.3.5, torch 2.6.0+cu124, scipy 1.17.1, suite2p installed

Directory contents:
- `code/` - Reference code from Track2p paper (README.md, notebooks/, track2p/, docs/)
- `data/` - Data directory with 6 mouse subjects (jm031-jm046), README.md, load_data.ipynb
- `paper.pdf` - Reference paper
- `methods.txt` - Extracted methods from paper
- `decoder.py` / `train_decoder.py` - Decoder code

---

## Step 1: Reference Code Exploration
**Status**: COMPLETE

### Key Functions Identified
| Function | File | Stage | Purpose |
|----------|------|-------|---------|
| load_traces() | data/load_data.ipynb | LOADING | Loads F.npy (raw fluorescence) from suite2p dir |
| load_fov() | data/load_data.ipynb | LOADING | Loads mean image from ops.npy |
| zscore_rows() | data/load_data.ipynb | PROCESSING | Z-scores each neuron's trace for visualization |
| preprocess() | suite2p.extraction.dcnv | PROCESSING | Baseline correction (maximin filter) |
| baseline_maximin() | suite2p.extraction.dcnv | PROCESSING | GPU-accelerated max/min pooling for baseline |

### Notes
- The data is already in Suite2p format with tracked neurons only (Track2p output in suite2p format)
- All neurons are already filtered to cells tracked across ALL days for each mouse
- iscell.npy has all values = 1.0 (all cells marked as cells since they're pre-filtered)
- The load_data.ipynb notebook demonstrates loading F.npy directly
- The demo notebook (demo_t2p_ouputs.ipynb) shows how to filter by iscell threshold and match indices
- Suite2p ops.npy contains processing parameters: fs=30, neucoeff=0.7, baseline='maximin', win_baseline=60, sig_baseline=10, prctile_baseline=8

---

## Step 2: Dataset Exploration
**Status**: COMPLETE

### Data Structure
```
data/
  jm031/  (Mouse A - 7 sessions, 221 neurons)
  jm032/  (Mouse B - 7 sessions, 370 neurons)
  jm038/  (Mouse C - 7 sessions, 685 neurons)
  jm039/  (Mouse D - 7 sessions, 746 neurons)
  jm040/  (Mouse E - 6 sessions, 541 neurons)
  jm046/  (Mouse F - 7 sessions, 435 neurons)
  Each session: YYYY-MM-DD_a/
    suite2p/plane0/
      F.npy      - Raw fluorescence (n_neurons, n_frames), float32
      Fneu.npy   - Neuropil fluorescence (n_neurons, n_frames), float32
      spks.npy   - Deconvolved spikes (n_neurons, n_frames), float32
      iscell.npy - Cell classification (n_neurons, 2), float64
      ops.npy    - Suite2p parameters (dict)
      stat.npy   - Cell statistics (array of dicts)
    move_deve/
      motion_energy_glob.npy - Motion energy (n_frames,), uint64
      tstamps.npy            - Timestamps (n_frames,), float64
      interframe_int.npy     - Inter-frame intervals (n_frames-1,), float64
```

### Dataset Size (from data files)
| Statistic | Value |
|-----------|-------|
| Neurons (total) | 2998 |
| Neurons / mouse | 221, 370, 685, 746, 541, 435 (mean=500) |
| Subjects | 6 |
| Sessions / subject | 7,7,7,7,6,7 |
| Sessions total | 41 |
| Frames per session | 36000 (jm031,jm032) or 54000 (others) |
| Frame mismatches (neural vs ME) | 0-148 frames in some sessions |

---

## Step 3: Reference Text Reading
**Status**: COMPLETE

### Expected Statistics (from paper/methods)
| Statistic | Value | Source |
|-----------|-------|--------|
| Subjects | 6 mice | "full dataset of 6 mice" |
| Sessions/mouse | min 6 consecutive days | "imaged daily for a minimum of 6 consecutive days" |
| Age range | P7-P14 | "within the second postnatal week (P7 to P14)" |
| Neurons/mouse | 526 +/- 190 std | "On average 526 (+-190 std) neurons per mouse" |
| % detected neurons tracked | 33% +/- 11% | "33% (+-11% std) of neurons detected on first day" |
| Imaging rate | 30 Hz | "Imaging rate was 30 Hz" |
| Session duration | 20 minutes | "each session lasted 20 minutes" |
| FOV | 720x720 um, 512x512 pixels | methods |
| iscell threshold | 0.5 (default) | "above the default threshold of 0.5" |
| Neuropil coefficient | 0.7 | ops.npy neucoeff=0.7 |
| Binning for decoding | 10 frames | "averaging in bins of 10 consecutive timestamps" |
| Videography rate | 30 Hz | "Videos were recorded at 30 Hz" |
| Decoding method (paper) | Ridge regression | "linear regression with ridge regularisation" |
| CV splits | 5-fold, 2-min blocks | "5 fold splits...consecutive 2 minute blocks" |

### Processing Details
1. **Neural preprocessing**: Suite2p pipeline (motion correction, ROI detection, signal extraction, spike deconvolution)
2. **dF/F computation**: "baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters)"
   - Neuropil subtraction: F_corr = F - 0.7 * Fneu
   - Baseline correction: maximin filter (win=60s, sig=10 frames)
3. **Binning**: Both dF/F and motion energy averaged in bins of 10 frames
4. **Motion energy**: Pixel-wise difference of consecutive video frames, squared and summed across pixels
5. **Camera sync**: Video at 30Hz triggered by microscope acquisition -> 1:1 frame correspondence
6. **Missing frames**: Some sessions have fewer ME frames than neural frames

### Curation Steps
**Neuron curation**: Data already contains only neurons tracked across all days (Track2p output)
**Trial curation**: No explicit trial curation mentioned - continuous recording

### Decoding Performance (from paper)
- Same-day R2 increases with development
- Early days (<=P11): low R2, Late days (>P11): higher R2
- R2 values shown in Fig. 7C (graphs, not exact numbers)
- Cross-day decoding: late-to-late is good, early-to-early and early-to-late are poor

---

## Step 4: Check for Consistency
**Status**: COMPLETE

### Discrepancies Found
| Topic | Code says | Data shows | Paper says | Resolution |
|-------|-----------|------------|------------|------------|
| Session duration | - | 36000 frames (20min) for jm031/jm032, 54000 (30min) for others | 20 minutes | jm038-jm046 have 30-min sessions; paper says 20min but some mice recorded longer. Use all available data. |
| Neurons/mouse | Data: 221-746 | mean=500, std~198 | 526+/-190 | Consistent within rounding |
| iscell threshold | demo: filter by iscell_thr | all iscell[:,0]=1 | 0.5 default | Provided data already pre-filtered to tracked cells; no additional filtering needed |

---

## Step 5: Mapping Planning
**Status**: COMPLETE

### Variable Mapping
| Source Variable | Target Field | Transform | Notes |
|-----------------|--------------|-----------|-------|
| F.npy, Fneu.npy | neural | Neuropil subtract, baseline correct (dF/F), bin by 10 frames | Suite2p default params |
| Time index | input[0] | Time in seconds from start of trial | (bin_index * 10 / 30) seconds within trial |
| motion_energy_glob.npy | output[0] | Interpolate missing frames, bin by 10, normalize, discretize into 5 equal-percentile bins | Global percentile bins across all data |

### Trial Structure
- Each session split into 2-minute blocks (matching paper's CV structure)
- 20-min sessions: 10 trials x 360 time bins
- 30-min sessions: 15 trials x 360 time bins
- Time bin size: 10 frames / 30 Hz = 333.33 ms

### Key Decisions
1. **dF/F method**: Use Suite2p's preprocess (baseline_maximin) with default params from ops.npy
2. **Trial splitting**: 2-minute blocks (paper uses "consecutive 2 minute blocks" for CV)
3. **Motion energy normalization**: Compute percentile bins globally across all sessions
4. **Missing ME frames**: Interpolate to match neural frame count using timestamps
5. **Brain region**: "barrel cortex" (single region for all neurons)
6. **Temporal alignment**: ME is already synced to neural (camera triggered by microscope)

### Planned Sanity Checks
- [x] Verify neuron count matches paper (526 +/- 190 mean/std)
- [x] Verify total sessions = 41
- [x] Verify binned time series lengths
- [x] Spot-check dF/F values against raw F
- [x] Verify ME discretization produces ~equal bins

---

## Step 6: Script Development
**Status**: COMPLETE

Implemented `convert_data.py` with:
- `--full` / `--sample` / `--show-processing` flags
- Suite2p's `preprocess()` for dF/F computation with GPU acceleration
- Vectorized binning (reshape + mean)
- Global percentile computation for ME discretization
- Processing time: ~1s per session

---

## Step 7: Sample Conversion and Validation
**Status**: COMPLETE

### Sample Statistics
| Statistic | Value |
|-----------|-------|
| Sessions | 2 (jm031 first 2 sessions) |
| Trials | 20 (10 per session) |
| Neurons | 221 per session |
| Time bins per trial | 360 |
| Time bin size | 333.33 ms |
| Input range | [0.0, 119.7] seconds |
| Output distribution | 20% per bin (global) |

### Processing Plots
- Saved `processing_2023-10-18_a.png` and `processing_2023-10-19_a.png`
- No anomalies detected

### Run Time
- Sample: 2.5s for 2 sessions (~1.25s/session)
- Estimated full: ~50s for 41 sessions

---

## Step 8: Sample Decoder Training
**Status**: COMPLETE

### Format Validation
- Errors: None
- Warnings: None

### Decoder Results (Sample)
| Output | Training Balanced Acc | Validation Balanced Acc |
|--------|-------------|--------|
| motion_energy_bin | 0.5251 | 0.3237 |

Loss decreases. Accuracy above chance (0.20). jm031 is an early developmental mouse (P8-P14), so moderate accuracy expected.

---

## Step 9: Full Conversion and Validation
**Status**: COMPLETE

### Output Files
- `converted_data.pkl`: 395.2 MB
- `verification_full_out.txt`: created, no errors or warnings

### Consistency Check
| Statistic | Reference Paper | Reference Data | Converted Data | Match? |
|-----------|-----------------|----------------|----------------|--------|
| Subjects | 6 | 6 | 6 | YES |
| Sessions | >=6/mouse | 7,7,7,7,6,7 (=41) | 41 | YES |
| Neurons/mouse (mean+/-std) | 526+/-190 | 500+/-198 | 500+/-198 | YES (within range) |
| Neurons total | ~3156 (from mean) | 2998 | 2998 | YES |
| Time bin size | 10 frames @ 30Hz | - | 333.33 ms | YES |
| Imaging rate | 30 Hz | 30 Hz (ops.npy) | 30 Hz | YES |
| Brain region | barrel cortex | - | barrel cortex | YES |
| Output bins | - | - | 5 equal-percentile | YES |
| Global output distribution | - | - | 20%/20%/20%/20%/20% | YES |

---

## Step 10: Critical Review 1
**Status**: COMPLETE

### Check 1: Output log verification
- `verification_full_out.txt`: No errors, no warnings. Data format valid.

### Check 2: Sanity checks (independent of conversion code)
All passed:
- **Neural spot check**: Session 0, trial 0, neuron 5, tp 10: converted matches independently computed dF/F (np.allclose, atol=1e-5)
- **Neural full trial check**: Full trial 0 and trial 9 match independently computed values (atol=1e-4)
- **Output spot check**: Session 0, trial 0 output matches independently discretized ME
- **Input check**: Time values match expected (arange * 10/30)

### Check 3: Reference code comparison
| Processing Step | My Code | Reference | Match? |
|----------------|---------|-----------|--------|
| Data loading | F.npy, Fneu.npy from suite2p/plane0 | load_data.ipynb: F.npy from suite2p/plane0 | YES |
| Neuron filtering | None needed (data pre-filtered) | iscell threshold 0.5, then Track2p matched | YES (data already filtered) |
| Neuropil correction | F - 0.7 * Fneu | Suite2p default neucoeff=0.7 | YES |
| Baseline correction | Suite2p preprocess(maximin, win=60, sig=10) | "default Suite2p parameters" | YES |
| dF/F | (F_corr - baseline) / baseline | "baseline corrected fluorescence traces as our dF/F" | YES |
| Binning | Average in 10-frame bins | "averaging in bins of 10 consecutive timestamps" | YES |
| ME processing | Load, interpolate missing frames, bin by 10 | "motion energy" from videography, binned same way | YES |

### Check 4: Key statistics
All match (see Step 9 Consistency Check table).

### Check 5: Edge cases
- Missing ME frames: handled via interpolation (affects 7 sessions)
- Partial bins at end of sessions: discarded (consistent with integer division)
- 30-min vs 20-min sessions: handled correctly (15 vs 10 trials)

---

## Step 11: Full Decoder Training
**Status**: COMPLETE

### Training Progress
- Loss decreasing: Yes (1.5M -> 11.5K over 200 epochs)

### Decoder Results (Full)
| Output | Training Balanced Acc | Validation Balanced Acc | Notes |
|--------|-------------|--------|-------|
| motion_energy_bin | 0.4144 | 0.3550 | 1.77x chance (0.20) |

---

## Step 12: Critical Review 2
**Status**: COMPLETE

### Check 1: Accuracy vs chance
- Validation accuracy: 0.3550, chance: 0.2000
- Ratio: 1.77x chance (above 1.5x threshold)
- Result: PASS

### Check 2: Accuracy comparison to paper
- Paper uses ridge regression with R2 metric (continuous), we use classification with balanced accuracy
- Paper shows R2 increases with development; early sessions (<=P11) have low R2, late sessions have higher R2
- Our accuracy averages across ALL sessions (early + late), so moderate overall accuracy (0.355) is expected
- Result: CONSISTENT with paper expectations

### Check 3: Train vs validation gap
- Train/Val ratio: 0.4144/0.3550 = 1.17
- Below 1.5x threshold
- Result: PASS (no evidence of overfitting)

### Accuracy Analysis
| Variable | Achieved Accuracy | Chance | Ratio | Paper Expectation |
|----------|------------------|--------|-------|-------------------|
| motion_energy_bin | 0.3550 | 0.2000 | 1.77x | Moderate (mixed early/late sessions) |

---

## Step 13: Documentation and Cleanup
**Status**: COMPLETE

- [x] README.md created
- [x] cache/ folder created
- [x] All files organized
- [x] CONVERSION_NOTES.md complete
